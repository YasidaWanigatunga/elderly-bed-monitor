"""
temporal.py
-----------
Stage 3: use TIME to fix single-frame mistakes.

The assignment asks the system to "understand transitions rather than
classifying every frame independently". Two steps do that:

1. Smoothing (confidence-weighted vote)
   Each frame's state is replaced by the state with the highest total
   confidence among its neighbours (a window of SMOOTH_WINDOW_SEC).
   One wrong frame surrounded by correct ones gets outvoted.
   The window is centred (uses a little of the future), which is fine for
   recorded video. For a live camera it would add ~1 s of delay.

2. State machine
   The person is always in exactly one state. A change is accepted only if:
     - the new state lasts at least MIN_DWELL_SEC      (removes flicker)
     - the jump is physically possible (ALLOWED below) (e.g. you cannot go
       from LYING to WALKING without sitting up and standing first).
       An "impossible" jump is only accepted with strong, long evidence
       (IMPLAUSIBLE_DWELL_SEC) and is flagged for the agent to review.
   When a change is accepted, it is back-dated to the frame where the new
   state really started, so durations are not delayed by the dwell time.

   NO_PERSON (nobody detected) is resolved using what happened before:
     - was in bed, bed still visible  -> still in bed (under the blanket),
                                         for up to HIDDEN_IN_BED_MAX_SEC
     - was standing/walking at the frame edge -> OUT_OF_BED (left the view)
     - otherwise                      -> UNKNOWN (e.g. camera blocked)
"""

from collections import defaultdict

from src import config as C
from src.classifier import (IN_BED, LYING, NO_PERSON, OUT, SIT_BED, SIT_OUT,
                            STAND, UNKNOWN, WALK)

# Which state can follow which (UNKNOWN can always come and go).
ALLOWED = {
    LYING:   {SIT_BED},
    SIT_BED: {LYING, STAND, WALK},        # WALK: standing can be shorter than 1 frame
    STAND:   {SIT_BED, WALK, SIT_OUT, OUT},
    WALK:    {STAND, SIT_BED, SIT_OUT, OUT},
    SIT_OUT: {STAND, WALK},
    OUT:     {WALK, STAND},
}


def average_motion(rows, half_window=1):
    """Replace each frame's motion by the mean over its neighbours.

    Frame-to-frame hip movement is jumpy (pose keypoints wobble), so one
    noisy frame could look like walking. Averaging over 3 frames (1.5 s)
    keeps real walking and removes most spikes. Measured on seq1:
    accuracy 90.8% -> 91.9%.
    """
    raw = [float(r["motion"]) if r.get("motion") not in ("", None) else None for r in rows]
    for i, r in enumerate(rows):
        window = [m for m in raw[max(0, i - half_window): i + half_window + 1] if m is not None]
        if window:
            r["motion"] = str(sum(window) / len(window))
    return rows


def allowed(a, b):
    return a == b or UNKNOWN in (a, b) or b in ALLOWED.get(a, set())


def _frames(seconds, dt):
    return max(1, int(round(seconds / dt)))


# ---------------------------------------------------------------------------
def smooth(obs, dt):
    """Confidence-weighted majority vote in a centred window."""
    half = _frames(C.SMOOTH_WINDOW_SEC, dt) // 2
    out = []
    for i in range(len(obs)):
        votes = defaultdict(float)
        for j in range(max(0, i - half), min(len(obs), i + half + 1)):
            votes[obs[j].state] += obs[j].confidence
        winner = max(votes, key=votes.get)
        # keep the frame's own guess if it ties with the winner
        if votes[obs[i].state] >= votes[winner]:
            winner = obs[i].state
        support = votes[winner] / sum(votes.values())
        out.append((winner, support))
    return out


def _at_edge(features):
    """Was the person's box touching the left/right/bottom frame edge?"""
    x1, x2, y2 = features.get("px1"), features.get("px2"), features.get("py2")
    if x1 is None:
        return False
    return x1 < C.EDGE_MARGIN or x2 > 1 - C.EDGE_MARGIN or y2 > 1 - C.EDGE_MARGIN


# ---------------------------------------------------------------------------
def run_state_machine(obs, dt):
    """Returns a list of dicts, one per frame: t, state, confidence, reason, flags."""
    smoothed = smooth(obs, dt)
    dwell = _frames(C.MIN_DWELL_SEC, dt)
    dwell_hard = _frames(C.IMPLAUSIBLE_DWELL_SEC, dt)
    hidden_max = _frames(C.HIDDEN_IN_BED_MAX_SEC, dt)

    result = []
    current = None
    hidden_count = 0                 # frames the person has been invisible in bed
    last_seen_at_edge = False
    cand, cand_start, cand_len = None, None, 0

    for i, (o, (s_state, support)) in enumerate(zip(obs, smoothed)):
        flags = list(o.flags)
        reason = o.reason

        # ---- resolve NO_PERSON using context ----
        proposed = s_state
        if proposed == NO_PERSON:
            if current in IN_BED and o.features.get("bed_seen") and hidden_count < hidden_max:
                proposed, reason = current, "not visible, bed visible: assumed still in bed (blanket)"
                hidden_count += 1
                flags.append("hidden_in_bed")
            elif current in (STAND, WALK, OUT) and last_seen_at_edge:
                proposed, reason = OUT, "walked out of the camera view"
            elif current == OUT:
                proposed, reason = OUT, "still out of the camera view"
            else:
                proposed, reason = UNKNOWN, "nobody visible and no context to explain it"
        else:
            hidden_count = 0
            last_seen_at_edge = _at_edge(o.features)

        # ---- first frame ----
        if current is None:
            current = proposed
            result.append(dict(t=o.t, state=current, confidence=round(o.confidence * support, 2),
                               reason=reason, flags=flags))
            continue

        # ---- same state: nothing to decide ----
        if proposed == current:
            cand, cand_len = None, 0
        else:
            # ---- a different state is proposed: is it a real change? ----
            if proposed != cand:
                cand, cand_start, cand_len = proposed, i, 0
            cand_len += 1
            needed = dwell if allowed(current, cand) else dwell_hard
            if cand_len >= needed:
                if not allowed(current, cand):
                    flags.append(f"implausible:{current}->{cand}")
                # accept, and back-date the change to where it began
                for k in range(cand_start, i):
                    result[k]["state"] = cand
                current = cand
                cand, cand_len = None, 0

        # a frame's own flags (e.g. possible_fall) only count if the final
        # state agrees with that frame's guess; frame-level facts like
        # low_light and flags added here (hidden_in_bed, implausible) stay.
        if o.state != current:
            flags = [x for x in flags if x not in o.flags or x == "low_light"]
        result.append(dict(t=o.t, state=current,
                           confidence=round(o.confidence * support, 2),
                           reason=reason if current == proposed else
                           f"kept {current} (brief '{proposed}' not trusted yet)",
                           flags=flags))
    return result