"""
check_features.py
-----------------
Sanity check for the perception stage (not part of the final pipeline).

Question it answers:
  "Do the features actually separate the states?"

How:
  For every sampled frame, look up the TRUE state from the ground-truth CSV,
  then show the typical feature values for each true state.
  If LYING rows have a big torso angle and SITTING rows a small one, the
  feature is useful. If the numbers look the same for two states, the
  classifier will confuse them.

Uses only the Python standard library (no pandas), so it runs anywhere.

Run:
  python check_features.py seq1_exit_and_return
  python check_features.py seq4_hard_conditions
"""

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

name = sys.argv[1] if len(sys.argv) > 1 else "seq1_exit_and_return"

with open(Path("outputs") / "features" / f"{name}.csv") as f:
    feat = list(csv.DictReader(f))

# ground truth can be in data/sequences or data/pexels/clips
gt_path = Path("data") / "sequences" / f"{name}.csv"
if not gt_path.exists():
    gt_path = Path("data") / "pexels" / "clips" / f"{name}.csv"
with open(gt_path) as f:
    gt = [(float(r["start_sec"]), float(r["end_sec"]), r["state"])
          for r in csv.DictReader(f)]


def true_state(t):
    for s, e, st in gt:
        if s <= t < e:
            return st
    return gt[-1][2]


def num(x):
    """CSV value -> float, or None if empty."""
    return float(x) if x not in ("", None) else None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 2) if xs else "-"


def median(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 2) if xs else "-"


groups = defaultdict(list)
for row in feat:
    groups[true_state(float(row["t"]))].append(row)

cols = [("frames", None, None),
        ("person_found", "person_found", mean),
        ("kp_visible", "kp_visible", mean),
        ("torso_angle", "torso_angle", median),
        ("aspect", "aspect", median),
        ("hip_on_bed", "hip_on_bed", mean),
        ("motion", "motion", median),
        ("brightness", "brightness", mean)]

print(f"\nFeature summary per TRUE state: {name}\n")
print(f"{'true_state':<16}" + "".join(f"{c[0]:>13}" for c in cols))
for state in sorted(groups):
    rows = groups[state]
    cells = []
    for label, key, fn in cols:
        cells.append(len(rows) if key is None else fn([num(r[key]) for r in rows]))
    print(f"{state:<16}" + "".join(f"{str(c):>13}" for c in cells))

print("""
How to read it:
  torso_angle  median degrees from vertical  (~0 upright, ~90 lying flat)
  aspect       median box width/height       (>2 lying, <1 standing)
  hip_on_bed   fraction of frames with hips inside the bed box
  person_found fraction of frames where a person was detected
""")