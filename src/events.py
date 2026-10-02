from src import config as C
from src.classifier import IN_BED, LYING, OUT, SIT_OUT, STAND, UNKNOWN, WALK

OUT_STATES = {STAND, WALK, SIT_OUT, OUT}
MOVING_AWAY = {WALK, OUT, SIT_OUT}


def status(state):
    if state in IN_BED:
        return "IN"
    if state in OUT_STATES:
        return "OUT"
    return None                              # UNKNOWN: no information


def hms(sec):
    sec = int(round(sec))
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def _confidence(segs):
    """Duration-weighted mean confidence, reduced if UNKNOWN was involved."""
    total = sum(s["end_sec"] - s["start_sec"] for s in segs) or 1
    conf = sum(s.get("confidence", 1.0) * (s["end_sec"] - s["start_sec"]) for s in segs) / total
    if any(s["state"] == UNKNOWN for s in segs):
        conf *= 0.8
    return round(conf, 2)


def detect_events(segments):
    """segments: list of dicts with state, start_sec, end_sec, confidence.

    Returns (events, notes, out_periods).
    out_periods = list of (start_sec, end_sec) the person spent out of bed.
    """
    events, notes, out_periods = [], [], []
    bed = None                 # current confirmed status: "IN" / "OUT"
    last_in_state = None       # the in-bed state before an exit (for previous_state)
    last_out_state = None
    pending = None             # a transition that is not confirmed yet

    for i, seg in enumerate(segments):
        st = status(seg["state"])
        if st == "IN":
            last_in_state = seg["state"] if bed == "IN" or bed is None else last_in_state
        if bed is None:
            bed = st
            if st == "IN":
                last_in_state = seg["state"]
            elif st == "OUT":
                last_out_state = seg["state"]
            continue
        if st is None:
            if pending:
                pending["segs"].append(seg)
            continue

        if pending is None and st != bed:
            pending = {"to": st, "start": seg["start_sec"], "segs": [seg],
                       "prev": last_in_state if bed == "IN" else last_out_state,
                       "prev_segs": [segments[i - 1]] if i else []}
        elif pending is not None:
            if st == pending["to"]:
                pending["segs"].append(seg)
            else:
                if pending["to"] == "OUT":
                    notes.append({"note": "brief_stand", "start_time": hms(pending["start"]),
                                  "end_time": hms(seg["start_sec"]),
                                  "explanation": "left the in-bed posture briefly but got "
                                                 "back on the bed before moving away: "
                                                 "not counted as a bed exit"})
                pending = None
                continue

        if pending is None:
            if st == "IN":
                last_in_state = seg["state"]
            else:
                last_out_state = seg["state"]
            continue

        known = [s for s in pending["segs"] if status(s["state"]) == pending["to"]]
        held = sum(s["end_sec"] - max(s["start_sec"], pending["start"]) for s in known)
        confirm_at = None
        if pending["to"] == "OUT":
            mover = next((s for s in known if s["state"] in MOVING_AWAY), None)
            if mover:
                confirm_at = mover["start_sec"]
            if held >= C.EXIT_CONFIRM_SEC:
                t = pending["start"] + C.EXIT_CONFIRM_SEC
                confirm_at = min(confirm_at, t) if confirm_at is not None else t
        else:
            lie = next((s for s in known if s["state"] == LYING), None)
            if lie:
                confirm_at = lie["start_sec"]
            if held >= C.RETURN_CONFIRM_SEC:
                t = pending["start"] + C.RETURN_CONFIRM_SEC
                confirm_at = min(confirm_at, t) if confirm_at is not None else t

        if confirm_at is not None:
            current = next((s["state"] for s in known
                            if s["start_sec"] <= confirm_at < s["end_sec"]), known[-1]["state"])
            events.append({
                "event": "bed_exit" if pending["to"] == "OUT" else "return_to_bed",
                "start_time": hms(pending["start"]),
                "confirmed_time": hms(confirm_at),
                "start_sec": round(pending["start"], 2),
                "confirmed_sec": round(confirm_at, 2),
                "previous_state": (pending["prev"] or "unknown").lower(),
                "current_state": current.lower(),
                "confidence": _confidence(pending["prev_segs"] + pending["segs"]),
            })
            if pending["to"] == "OUT":
                out_periods.append([pending["start"], None])
            elif out_periods and out_periods[-1][1] is None:
                out_periods[-1][1] = pending["start"]
            bed = pending["to"]
            if bed == "IN":
                last_in_state = known[-1]["state"]
            else:
                last_out_state = known[-1]["state"]
            pending = None

    end = segments[-1]["end_sec"] if segments else 0
    if out_periods and out_periods[-1][1] is None:
        out_periods[-1][1] = end
    return events, notes, [tuple(p) for p in out_periods]


def events_from_ground_truth(gt):
    """Apply exactly the same event definition to the ground-truth timeline."""
    segs = [{"state": st, "start_sec": s, "end_sec": e, "confidence": 1.0} for s, e, st in gt]
    return detect_events(segs)[0]


def match_events(pred, truth, tolerance=C.EVENT_MATCH_TOLERANCE_SEC):
    """Precision / recall per event type. A predicted event is correct if a
    true event of the same type starts within `tolerance` seconds."""
    report = {}
    for kind in ("bed_exit", "return_to_bed"):
        p = [e for e in pred if e["event"] == kind]
        t = [e for e in truth if e["event"] == kind]
        used, tp, errors = set(), 0, []
        for e in p:
            best = min((j for j in range(len(t)) if j not in used),
                       key=lambda j: abs(t[j]["start_sec"] - e["start_sec"]), default=None)
            if best is not None and abs(t[best]["start_sec"] - e["start_sec"]) <= tolerance:
                used.add(best)
                tp += 1
                errors.append(round(e["start_sec"] - t[best]["start_sec"], 1))
        fp, fn = len(p) - tp, len(t) - tp
        report[kind] = {"true": len(t), "predicted": len(p), "correct": tp,
                        "false_detections": fp, "missed": fn,
                        "precision": round(tp / len(p), 3) if p else None,
                        "recall": round(tp / len(t), 3) if t else None,
                        "timing_errors_sec": errors}
    return report