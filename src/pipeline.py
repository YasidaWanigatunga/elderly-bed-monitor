"""
pipeline.py
-----------
Runs the analysis stages on a video's features and writes the results.

    features CSV (from perception.py)
      -> classifier   (one frame at a time)
      -> temporal     (smoothing + state machine)
      -> timeline     (segments + durations)
      -> agent        (reviews ambiguous segments: look back / forward / VLM)
      -> events       (bed exit / return to bed)
      -> alerts       (NORMAL / MONITOR / ALERT)
      -> evaluation   (if ground truth exists)

Run:
  python -m src.pipeline seq1_exit_and_return
  python -m src.pipeline --all          (every file in outputs/features)
  python -m src.pipeline --all --no-vlm (agent uses temporal context only)
  python -m src.pipeline --all --no-agent
Output:
  outputs/results/<name>.json           (timeline, events, alerts, summary, evaluation)
  outputs/results/<name>_agent_trace.md (the agent's reasoning, step by step)
  outputs/results/overall.json          (with --all: metrics over all videos)
"""

import argparse
import csv
import json
from pathlib import Path

from src import evaluate as E
from src.agent import Agent, frames_from_segments
from src.alerts import decision_for_event, evaluate_alerts
from src.classifier import IN_BED, NO_PERSON, UNKNOWN, classify
from src.events import (OUT_STATES, detect_events, events_from_ground_truth,
                        match_events)
from src.temporal import average_motion, run_state_machine
from src.timeline import build_segments, durations, print_timeline

FEATURE_DIR = Path("outputs") / "features"
RESULT_DIR = Path("outputs") / "results"
GT_DIRS = [Path("data") / "sequences", Path("data") / "pexels" / "clips"]


def find_gt(name):
    for d in GT_DIRS:
        if (d / f"{name}.csv").exists():
            return d / f"{name}.csv"
    return None


def find_video(name):
    for d in GT_DIRS:
        if (d / f"{name}.mp4").exists():
            return d / f"{name}.mp4"
    return None


def mmss(sec):
    sec = int(round(sec))
    return f"{sec // 60}m {sec % 60:02d}s"


def build_summary(video_end, segments, dur, events, out_periods):
    """The 'complete summary' in the format given in the assignment."""
    keys = ["lying_in_bed", "sitting_on_bed", "sitting_outside_bed",
            "standing", "walking", "out_of_bed", "unknown"]
    activity = {k: round(dur.get(k.upper(), 0)) for k in keys}
    in_bed = sum(v for k, v in dur.items() if k in IN_BED)
    out_bed = sum(v for k, v in dur.items() if k in OUT_STATES)
    return {
        "observation_duration_sec": round(video_end),
        "activity_duration_sec": activity,
        "activity_summary": {k: mmss(v) for k, v in activity.items() if v},
        "bed_exit_count": sum(e["event"] == "bed_exit" for e in events),
        "bed_return_count": sum(e["event"] == "return_to_bed" for e in events),
        "total_in_bed_sec": round(in_bed),
        "total_out_of_bed_sec": round(out_bed),
        "longest_out_of_bed_period_sec": round(max((b - a for a, b in out_periods), default=0)),
        "final_state": segments[-1]["state"].lower() if segments else "unknown",
    }


def analyse(name, verbose=True, use_agent=True, use_vlm=True, video_path=None):
    with open(FEATURE_DIR / f"{name}.csv") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{FEATURE_DIR / (name + '.csv')} is empty: run perception "
                         f"on a valid video first")
    times = [float(r["t"]) for r in rows]
    dt = (times[-1] - times[0]) / max(1, len(times) - 1)
    video_end = times[-1] + dt

    rows = average_motion(rows)
    observations = [classify(r) for r in rows]
    frames = run_state_machine(observations, dt)
    segments_before_agent = build_segments(frames, video_end)
    segments, agent = segments_before_agent, None
    if use_agent:
        agent = Agent(name, segments_before_agent, video_path or find_video(name),
                      use_vlm=use_vlm)
        segments = agent.run()
    dur = durations(segments)
    events, notes, out_periods = detect_events(segments)
    alerts, overall = evaluate_alerts(segments, events, out_periods)
    for e in events:
        e["decision"] = decision_for_event(e, alerts, events)

    result = {"video": name, "duration_sec": round(video_end, 1),
              "summary": build_summary(video_end, segments, dur, events, out_periods),
              "overall_decision": overall,
              "events": events, "notes": notes, "alerts": alerts,
              "timeline": segments,
              "agent": None if agent is None else {
                  "vlm_used": agent.use_vlm, "vlm_calls": agent.vlm_calls,
                  "reviews": agent.trace}}

    if verbose:
        print(f"\n=== {name} ({video_end / 60:.1f} min) ===")
        print_timeline(segments)
        if agent is not None:
            changed = sum(t["new_state"] not in t["segment"] for t in agent.trace)
            print(f"\n  agent: reviewed {len(agent.trace)} segments, changed {changed}, "
                  f"VLM {'on' if agent.use_vlm else 'off'} ({agent.vlm_calls} calls)")
        print("\n  events:")
        for e in events:
            print(f"    {e['event']:<14} start {e['start_time']}  confirmed {e['confirmed_time']}"
                  f"  {e['previous_state']} -> {e['current_state']}"
                  f"  conf {e['confidence']}  {e['decision']}")
        for n in notes:
            print(f"    (note) {n['note']} {n['start_time']}-{n['end_time']}: not an exit")
        if not events and not notes:
            print("    none")
        print(f"\n  alerts (overall decision: {overall}):")
        for a in alerts:
            print(f"    {a['time']}  {a['decision']:<8} {a['rule']:<24} {a['explanation']}")
        if not alerts:
            print("    none")

    gt_path = find_gt(name)
    if gt_path:
        gt = E.load_ground_truth(gt_path)
        raw = [UNKNOWN if o.state == NO_PERSON else o.state for o in observations]
        temporal = [fr["state"] for fr in frames]
        final = frames_from_segments(times, segments)
        true_events = events_from_ground_truth(gt)
        events_before = detect_events(segments_before_agent)[0]
        result["evaluation"] = {
            "frame_only": E.state_metrics(times, raw, gt),
            "temporal_no_agent": E.state_metrics(times, temporal, gt),
            "with_temporal": E.state_metrics(times, final, gt),     # final system
            "durations": E.duration_errors(dur, gt),
            "true_events": true_events,
            "events_no_agent": match_events(events_before, true_events),
            "events": match_events(events, true_events),
        }
        if verbose:
            ev = result["evaluation"]
            print(f"\n  accuracy: frame-only {ev['frame_only']['accuracy']:.1%}"
                  f"  ->  + temporal logic {ev['temporal_no_agent']['accuracy']:.1%}"
                  f"  ->  + agent {ev['with_temporal']['accuracy']:.1%}")
            print("  confusion (final system):")
            E.print_confusion(ev["with_temporal"]["confusion"])
            print("  durations (true vs predicted, seconds):")
            for s, d in ev["durations"].items():
                print(f"    {s:<20} {d['true_sec']:>7} {d['pred_sec']:>7}  "
                      f"error {d['error_sec']:+.1f}")
            print("  events (true / predicted / correct):")
            for k, m in ev["events"].items():
                print(f"    {k:<14} {m['true']} / {m['predicted']} / {m['correct']}"
                      f"   false detections {m['false_detections']}, missed {m['missed']}"
                      f"   timing error {m['timing_errors_sec']}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULT_DIR / f"{name}.json", "w") as f:
        json.dump(result, f, indent=2)
    if agent is not None:
        (RESULT_DIR / f"{name}_agent_trace.md").write_text(agent.trace_markdown(), encoding="utf-8")
    return result


def overall_report(results):
    """Metrics summed over all videos that have ground truth."""
    ev = [r for r in results if "evaluation" in r]
    if not ev:
        return
    frames = sum(r["evaluation"]["with_temporal"]["frames"] for r in ev)
    acc_t = sum(r["evaluation"]["with_temporal"]["accuracy"] *
                r["evaluation"]["with_temporal"]["frames"] for r in ev) / frames
    acc_f = sum(r["evaluation"]["frame_only"]["accuracy"] *
                r["evaluation"]["frame_only"]["frames"] for r in ev) / frames
    acc_n = sum(r["evaluation"]["temporal_no_agent"]["accuracy"] *
                r["evaluation"]["temporal_no_agent"]["frames"] for r in ev) / frames
    report = {"videos": len(ev), "frames": frames,
              "accuracy_frame_only": round(acc_f, 3),
              "accuracy_temporal_no_agent": round(acc_n, 3),
              "accuracy_final": round(acc_t, 3), "events": {}, "events_no_agent": {}}
    for key in ("events_no_agent", "events"):
        for kind in ("bed_exit", "return_to_bed"):
            t = sum(r["evaluation"][key][kind]["true"] for r in ev)
            p = sum(r["evaluation"][key][kind]["predicted"] for r in ev)
            c = sum(r["evaluation"][key][kind]["correct"] for r in ev)
            report[key][kind] = {
                "true": t, "predicted": p, "correct": c, "false_detections": p - c,
                "missed": t - c,
                "precision": round(c / p, 3) if p else None,
                "recall": round(c / t, 3) if t else None}
    abs_err = {}
    for r in ev:
        for s, d in r["evaluation"]["durations"].items():
            abs_err.setdefault(s, []).append(abs(d["error_sec"]))
    report["mean_abs_duration_error_sec"] = {s: round(sum(v) / len(v), 1)
                                             for s, v in abs_err.items()}

    print("\n================ OVERALL ================")
    print(f"  {len(ev)} videos, {frames} frames")
    print(f"  state accuracy: frame-only {acc_f:.1%} -> + temporal {acc_n:.1%}"
          f" -> + agent {acc_t:.1%}")
    for label, key in (("without agent", "events_no_agent"), ("with agent   ", "events")):
        for k, m in report[key].items():
            print(f"  {label} {k:<14} precision {m['precision']}  recall {m['recall']}"
                  f"  false {m['false_detections']}  missed {m['missed']}")
    print("  mean absolute duration error (s):", report["mean_abs_duration_error_sec"])
    with open(RESULT_DIR / "overall.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--no-agent", action="store_true", help="skip the agent stage")
    ap.add_argument("--no-vlm", action="store_true", help="agent uses context only")
    args = ap.parse_args()
    names = args.names or []
    if args.all:
        names = sorted(p.stem for p in FEATURE_DIR.glob("seq*.csv"))
    results = [analyse(n, use_agent=not args.no_agent, use_vlm=not args.no_vlm)
               for n in names]
    if len(results) > 1:
        overall_report(results)