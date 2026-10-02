"""
build_sequences.py
------------------
Step 3 of the dataset pipeline.

Problem:
  The Pexels clips are 12-30 s long. The assignment is about CONTINUOUS video
  (long lying periods, bed exits, returns, time spent in each state).

Solution:
  Stitch the labelled clips into longer "scripted" sequences, and generate the
  ground truth automatically from the per-clip labels.

Three tricks make short clips into long, varied videos:
  1. Loop (ping-pong)   : play a piece forward, then backward, repeatedly.
                          Used to stretch "lying" or "walking" to minutes.
  2. Reverse            : play a piece backwards.
                          A reversed "stand up" is a realistic "sit back down",
                          which gives us the "stands briefly, sits back" case.
  3. Degradations       : dim lighting, an occluding box, a camera blackout,
                          applied on the output timeline (difficult cases).

How ground truth stays correct:
  Every output frame remembers (clip, source_time). Its label is looked up
  in that clip's CSV at source_time. So loops and reversals are labelled
  exactly, with no manual work.
  Degradations that remove the evidence (blackout, full occlusion) are
  labelled UNKNOWN, because the assignment says the system should answer
  UNKNOWN when there is not enough evidence.

Run:
  python build_sequences.py

Output (data/sequences/):
  seqX_name.mp4        the stitched video
  seqX_name.csv        ground-truth segments  (start_sec,end_sec,state)
  manifest.json        how each sequence was built (for the README)
"""

import csv
import json
from pathlib import Path

import cv2
import numpy as np

CLIP_DIR = Path("data") / "pexels" / "clips"
OUT_DIR = Path("data") / "sequences"
FPS = 25.0
SIZE = (1280, 720)

# Short names for the six labelled clips
CLIPS = {
    "c1": "c1_waking_stretching",     # lying -> sitting up (edge), stretching
    "c2": "c2_sitting_reading",       # walk in -> stand -> onto bed -> sit/read
    "c3": "c3_stretching",            # sitting -> stand -> walk away  (BED EXIT)
    "c4": "c4_making_bed",            # standing making bed -> kneel on bed
    "c5": "c5_making_bed_b",          # sitting on bed edge the whole time
    "c6": "c6_making_bed_relaxing",   # walk -> stand -> get in -> lie (RETURN)
}

# ---------------------------------------------------------------------------
# Sequence recipes
#   ("play", clip, start, end)            play a piece; end < start = reversed
#   ("loop", clip, start, end, seconds)   ping-pong the piece for `seconds`
#   degradations are listed separately on the OUTPUT timeline (seconds)
# ---------------------------------------------------------------------------
SEQUENCES = {
    "seq1_exit_and_return": {
        "story": "Sleeps, wakes, sits up, leaves the bed, moves around, "
                 "comes back and lies down again. One exit, one return.",
        "parts": [
            ("loop", "c6", 12.0, 17.8, 60),     # asleep
            ("play", "c1", 0.0, 17.08),         # wakes, sits up on edge
            ("play", "c3", 0.0, 12.48),         # stands and walks away (EXIT)
            ("loop", "c4", 0.0, 7.6, 30),       # standing by the bed
            ("play", "c6", 0.0, 17.84),         # walks back, gets in (RETURN)
            ("loop", "c6", 12.0, 17.8, 40),     # asleep again
        ],
        "degrade": [],
    },
    "seq2_no_exit_long_edge_sit": {
        "story": "Sits up and sits on the bed edge for a long time but never "
                 "leaves. Should give ZERO bed exits and a MONITOR decision.",
        "parts": [
            ("loop", "c6", 12.0, 17.8, 45),     # asleep
            ("play", "c1", 0.0, 17.08),         # sits up
            ("loop", "c5", 0.0, 23.9, 150),     # long edge sitting
            ("play", "c6", 8.5, 17.84),         # lies back down
            ("loop", "c6", 12.0, 17.8, 30),     # asleep
        ],
        "degrade": [],
    },
    "seq3_brief_stand_then_exit": {
        "story": "Stands up briefly and sits back down (NOT an exit), later "
                 "really leaves and stays away for a long time.",
        "parts": [
            ("loop", "c6", 12.0, 17.8, 40),     # asleep
            ("play", "c1", 0.0, 17.08),         # sits up
            ("play", "c3", 0.0, 11.3),          # starts to stand ...
            ("play", "c3", 11.3, 9.0),          # ... reversed: sits back down
            ("loop", "c5", 0.0, 23.9, 20),      # sits on edge
            ("play", "c3", 0.0, 12.48),         # real exit
            ("loop", "c2", 0.0, 3.9, 90),       # walking around, long absence
            ("loop", "c4", 0.0, 7.6, 60),       # standing
        ],
        "degrade": [],
    },
    "seq4_hard_conditions": {
        "story": "Same story as seq1 but with poor lighting, a temporary "
                 "occlusion and a short camera blackout.",
        "parts": [
            ("loop", "c6", 12.0, 17.8, 60),
            ("play", "c1", 0.0, 17.08),
            ("play", "c3", 0.0, 12.48),
            ("loop", "c4", 0.0, 7.6, 30),
            ("play", "c6", 0.0, 17.84),
            ("loop", "c6", 12.0, 17.8, 40),
        ],
        # (kind, from_sec, to_sec)
        "degrade": [
            ("dim", 0.0, 60.0),          # night: whole sleeping period is dark
            ("occlude", 70.0, 76.0),     # something blocks the person
            ("blackout", 100.0, 104.0),  # camera signal lost
            ("dim", 150.0, 190.0),
        ],
    },
}

# Degradations that destroy the evidence -> ground truth becomes UNKNOWN
EVIDENCE_REMOVING = {"occlude", "blackout"}


# ---------------------------------------------------------------------------
# Loading clips and labels
# ---------------------------------------------------------------------------
def load_clip(name):
    """Read every frame of a clip into memory, JPEG-compressed.

    Raw 720p frames are 2.7 MB each; all six clips raw would need ~8 GB RAM.
    Keeping them as JPEG bytes (~80 KB each) needs ~250 MB instead, and lets
    us jump to any frame instantly (needed for loops and reversals).
    """
    cap = cv2.VideoCapture(str(CLIP_DIR / f"{CLIPS[name]}.mp4"))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if (f.shape[1], f.shape[0]) != SIZE:
            f = cv2.resize(f, SIZE, interpolation=cv2.INTER_AREA)
        frames.append(cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 92])[1])
    cap.release()
    return frames


def decode(jpg_bytes):
    return cv2.imdecode(jpg_bytes, cv2.IMREAD_COLOR)


def load_labels(name):
    with open(CLIP_DIR / f"{CLIPS[name]}.csv") as f:
        return [(float(r["start_sec"]), float(r["end_sec"]), r["state"])
                for r in csv.DictReader(f)]


def label_at(labels, t):
    for s, e, st in labels:
        if s <= t < e:
            return st
    return labels[-1][2] if t >= labels[-1][1] else labels[0][2]


# ---------------------------------------------------------------------------
# Turning recipe parts into a list of (clip, source_time)
# ---------------------------------------------------------------------------
def times_play(start, end):
    """Source times for playing start->end (reversed if end < start)."""
    n = int(round(abs(end - start) * FPS))
    step = 1.0 / FPS if end >= start else -1.0 / FPS
    return [start + i * step for i in range(n)]


def times_loop(start, end, seconds):
    """Ping-pong start->end->start ... for `seconds` of output."""
    forward = times_play(start, end)
    cycle = forward + forward[::-1]
    n = int(round(seconds * FPS))
    return [cycle[i % len(cycle)] for i in range(n)]


def expand(parts):
    timeline = []  # list of (clip, source_time)
    for p in parts:
        if p[0] == "play":
            _, clip, a, b = p
            timeline += [(clip, t) for t in times_play(a, b)]
        elif p[0] == "loop":
            _, clip, a, b, secs = p
            timeline += [(clip, t) for t in times_loop(a, b, secs)]
        else:
            raise ValueError(f"unknown part {p}")
    return timeline


# ---------------------------------------------------------------------------
# Degradations
# ---------------------------------------------------------------------------
def active_degradations(degrade, t):
    return [kind for kind, a, b in degrade if a <= t < b]


_NOISE = None


def night_noise(shape, rng, i):
    """A few pre-made noise frames, cycled (generating fresh noise for every
    frame is slow and makes the video file very large)."""
    global _NOISE
    if _NOISE is None:
        _NOISE = [rng.normal(0, 5, shape).astype(np.float32) for _ in range(6)]
    return _NOISE[i % len(_NOISE)]


def apply_degradations(frame, kinds, rng, i=0):
    out = frame
    if "dim" in kinds:
        # darker + sensor noise, like a night-time camera
        out = (out.astype(np.float32) * 0.25
               + night_noise(out.shape, rng, i)).clip(0, 255).astype(np.uint8)
    if "occlude" in kinds:
        out = out.copy()
        h, w = out.shape[:2]
        cv2.rectangle(out, (int(w * 0.15), int(h * 0.05)),
                      (int(w * 0.85), int(h * 0.95)), (40, 40, 40), -1)
    if "blackout" in kinds:
        out = np.zeros_like(out)
    return out


# ---------------------------------------------------------------------------
# Ground truth: per-frame labels -> segments
# ---------------------------------------------------------------------------
def to_segments(frame_labels):
    segs = []
    start = 0
    for i in range(1, len(frame_labels) + 1):
        if i == len(frame_labels) or frame_labels[i] != frame_labels[start]:
            segs.append((round(start / FPS, 2), round(i / FPS, 2),
                         frame_labels[start]))
            start = i
    return segs


# ---------------------------------------------------------------------------
def build(name, spec, cache, labels, rng):
    timeline = expand(spec["parts"])
    out_path = OUT_DIR / f"{name}.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             FPS, SIZE)
    frame_labels = []
    for i, (clip, t_src) in enumerate(timeline):
        frames = cache[clip]
        idx = min(int(round(t_src * FPS)), len(frames) - 1)
        t_out = i / FPS
        kinds = active_degradations(spec["degrade"], t_out)

        writer.write(apply_degradations(decode(frames[idx]), kinds, rng, i))

        state = label_at(labels[clip], t_src)
        if EVIDENCE_REMOVING & set(kinds):
            state = "UNKNOWN"
        frame_labels.append(state)
    writer.release()

    segs = to_segments(frame_labels)
    with open(OUT_DIR / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start_sec", "end_sec", "state"])
        w.writerows(segs)

    dur = len(timeline) / FPS
    print(f"\n{name}: {dur:.1f}s ({dur / 60:.1f} min), {len(segs)} segments")
    for s, e, st in segs:
        print(f"  {s:7.2f} - {e:7.2f}  {st}")
    return dur


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    print("Loading clips ...")
    cache = {k: load_clip(k) for k in CLIPS}
    labels = {k: load_labels(k) for k in CLIPS}

    manifest = {}
    for name, spec in SEQUENCES.items():
        dur = build(name, spec, cache, labels, rng)
        manifest[name] = {"duration_sec": round(dur, 2), "story": spec["story"],
                          "parts": spec["parts"], "degrade": spec["degrade"]}

    with open(OUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nDone. Sequences saved in {OUT_DIR}")


if __name__ == "__main__":
    main()
