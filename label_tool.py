"""
label_tool.py
-------------
A tiny keyboard tool to create ground-truth labels for a video.

Idea:
  You watch the video. Whenever the person's state CHANGES, press the number
  key for the new state. The tool records (time, state). On save it turns
  those change-points into segments:  start, end, state

Run:
  python label_tool.py data\\pexels\\clips\\c1_waking_stretching.mp4

Keys:
  1 LYING_IN_BED        2 SITTING_ON_BED      3 SITTING_OUTSIDE_BED
  4 STANDING            5 WALKING             6 OUT_OF_BED
  7 UNKNOWN
  SPACE  play / pause
  d / a  forward / back 1 second
  . / ,  forward / back 1 frame   (for precise boundaries)
  u      undo last mark
  s      save
  q      save and quit

Output:
  same folder as the video, <video_name>.csv
"""

import csv
import sys
from pathlib import Path

import cv2

STATES = {
    ord("1"): "LYING_IN_BED",
    ord("2"): "SITTING_ON_BED",
    ord("3"): "SITTING_OUTSIDE_BED",
    ord("4"): "STANDING",
    ord("5"): "WALKING",
    ord("6"): "OUT_OF_BED",
    ord("7"): "UNKNOWN",
}
DISPLAY_W = 1280


def to_segments(marks, duration):
    """Turn change-points [(t, state), ...] into [(start, end, state), ...]."""
    marks = sorted(marks)
    segments = []
    for i, (t, state) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else duration
        if end > t:
            segments.append((round(t, 2), round(end, 2), state))
    return segments


def save(csv_path, marks, duration):
    segments = to_segments(marks, duration)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start_sec", "end_sec", "state"])
        w.writerows(segments)
    print(f"Saved {len(segments)} segments -> {csv_path}")
    for s, e, st in segments:
        print(f"  {s:6.2f} - {e:6.2f}  {st}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    video = Path(sys.argv[1])
    csv_path = video.with_suffix(".csv")
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps

    marks = []      # list of (time_sec, state)
    idx = 0         # current frame index
    playing = False

    while True:
        idx = max(0, min(idx, total - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break

        t = idx / fps
        # current state = last mark at or before time t
        current = next((s for mt, s in sorted(marks, reverse=True) if mt <= t), "-")

        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (DISPLAY_W, int(h * DISPLAY_W / w)))
        cv2.rectangle(frame, (0, 0), (DISPLAY_W, 70), (0, 0, 0), -1)
        cv2.putText(frame, f"t={t:6.2f}s / {duration:.2f}s   state: {current}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "1 lie 2 sitBed 3 sitOut 4 stand 5 walk 6 out 7 unk | "
                    "SPACE play  a/d 1s  ,/. frame  u undo  s save  q quit",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow("label_tool", frame)

        key = cv2.waitKey(int(1000 / fps) if playing else 0) & 0xFF

        if key == 255:              # no key pressed while playing
            idx += 1
            if idx >= total - 1:
                playing = False
        elif key == ord(" "):
            playing = not playing
        elif key == ord("d"):
            idx += int(fps)
        elif key == ord("a"):
            idx -= int(fps)
        elif key == ord("."):
            idx += 1
        elif key == ord(","):
            idx -= 1
        elif key in STATES:
            # replace any existing mark at this exact time
            marks = [m for m in marks if abs(m[0] - t) > 1e-6]
            marks.append((t, STATES[key]))
            print(f"mark {t:6.2f}s -> {STATES[key]}")
        elif key == ord("u") and marks:
            print(f"undo {marks.pop()}")
        elif key == ord("s"):
            save(csv_path, marks, duration)
        elif key == ord("q"):
            save(csv_path, marks, duration)
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()