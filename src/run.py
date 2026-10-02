"""
run.py
------
One command to analyse ANY video (no ground truth needed):

    python -m src.run path\to\video.mp4
    python -m src.run path\to\video.mp4 --viz      (also save a video with the skeleton drawn)
    python -m src.run path\to\video.mp4 --no-vlm   (agent uses temporal context only)

It runs perception (YOLO pose + bed detection) and then the full pipeline:
state machine -> timeline -> agent -> events -> alerts.

Output:
  outputs/features/<video>.csv
  outputs/results/<video>.json             timeline, durations, events, alerts, summary
  outputs/results/<video>_agent_trace.md   the agent's reasoning
"""

import argparse
import json
import sys
from pathlib import Path

from src import perception
from src.pipeline import analyse

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--viz", action="store_true")
    ap.add_argument("--no-vlm", action="store_true")
    args = ap.parse_args()

    video = Path(args.video)
    if not video.is_file():
        sys.exit(f"Video not found: {video}\n"
                 f"Give the path to a real video file, for example:\n"
                 f"  python -m src.run data\\pexels\\clips\\c2_sitting_reading.mp4")
    print(f"[1/2] perception on {video.name} ...")
    if perception.run(video, viz=args.viz) is None:
        sys.exit(f"Could not read any frames from {video} (unsupported or broken video file).")
    print("[2/2] analysis ...")
    result = analyse(video.stem, use_vlm=not args.no_vlm, video_path=video)
    print("\nSummary:")
    print(json.dumps(result["summary"], indent=2))
    print(f"Overall decision: {result['overall_decision']}")
    print(f"Saved: outputs/results/{video.stem}.json")