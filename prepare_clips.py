"""
prepare_clips.py
----------------
Step 1 of the dataset pipeline.

What it does:
  - Finds the 6 Pexels clips (by their Pexels ID) in your Downloads folder
  - Gives them short, meaningful names
  - Downscales them from 4K (3840x2160) to 720p (1280x720)

Why downscale?
  4K frames are ~9x bigger than 720p. YOLO resizes to ~640px internally anyway,
  so 4K only makes everything slower on a CPU laptop with no accuracy gain.

Run (from the project folder):
  python prepare_clips.py
  python prepare_clips.py "C:\\Users\\you\\Downloads"   # if clips are elsewhere
"""

import sys
from pathlib import Path

import cv2

# Pexels ID -> short name (what happens in the clip)
CLIPS = {
    "9057924": "c1_waking_stretching",
    "9057932": "c2_sitting_reading",
    "9057929": "c3_stretching",
    "9057922": "c4_making_bed",
    "9057930": "c5_making_bed_b",
    "9057933": "c6_making_bed_relaxing",
}

SRC_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "Downloads"
DST_DIR = Path("data") / "pexels" / "clips"
TARGET_W, TARGET_H = 1280, 720


def convert(src: Path, dst: Path) -> None:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        print(f"  ! could not open {src.name}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    writer = cv2.VideoWriter(
        str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (TARGET_W, TARGET_H)
    )

    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # INTER_AREA is the best interpolation for shrinking images
        writer.write(cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA))
        n += 1

    cap.release()
    writer.release()
    print(f"  -> {dst.name}: {n} frames, {n / fps:.1f}s @ {fps:.0f}fps")


def main() -> None:
    DST_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Looking for clips in: {SRC_DIR}")

    for pexels_id, name in CLIPS.items():
        matches = sorted(SRC_DIR.glob(f"{pexels_id}*"))
        if not matches:
            print(f"  ! missing clip {pexels_id} ({name})")
            continue
        print(f"Converting {matches[0].name}")
        convert(matches[0], DST_DIR / f"{name}.mp4")

    print(f"\nDone. Clips saved in {DST_DIR}")


if __name__ == "__main__":
    main()
