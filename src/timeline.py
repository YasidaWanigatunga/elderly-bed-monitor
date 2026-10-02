from collections import OrderedDict
from src.classifier import STATES


def fmt(sec):
    sec = int(round(sec))
    return f"{sec // 60:02d}:{sec % 60:02d}"


def build_segments(frames, video_end):
    """frames: list of dicts with t, state, confidence, reason, flags."""
    segs = []
    for i, fr in enumerate(frames):
        end = frames[i + 1]["t"] if i + 1 < len(frames) else video_end
        if segs and segs[-1]["state"] == fr["state"]:
            s = segs[-1]
            s["end_sec"] = end
            s["_conf"].append(fr["confidence"])
            s["_flags"].update(fr["flags"])
        else:
            segs.append({"state": fr["state"], "start_sec": fr["t"], "end_sec": end,
                         "reason": fr["reason"], "_conf": [fr["confidence"]],
                         "_flags": set(fr["flags"])})
    out = []
    for s in segs:
        out.append({
            "start": fmt(s["start_sec"]), "end": fmt(s["end_sec"]),
            "start_sec": round(s["start_sec"], 2), "end_sec": round(s["end_sec"], 2),
            "state": s["state"],
            "duration_sec": round(s["end_sec"] - s["start_sec"], 1),
            "confidence": round(sum(s["_conf"]) / len(s["_conf"]), 2),
            "reason": s["reason"],
            "flags": sorted(f for f in s["_flags"] if f not in ("low_light",)) +
                     (["low_light"] if "low_light" in s["_flags"] else []),
        })
    return out


def durations(segments):
    d = OrderedDict((st, 0.0) for st in STATES)
    for s in segments:
        d[s["state"]] = d.get(s["state"], 0.0) + s["duration_sec"]
    return OrderedDict((k, round(v, 1)) for k, v in d.items() if v > 0)


def print_timeline(segments):
    for s in segments:
        flag = f"  [{', '.join(s['flags'])}]" if s["flags"] else ""
        print(f"  {s['start']} - {s['end']}  {s['state']:<20} "
              f"conf {s['confidence']:.2f}{flag}")