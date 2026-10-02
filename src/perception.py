"""
perception.py
-------------
Stage 1 of the system: turn video into numbers.

For every sampled frame (SAMPLE_FPS per second) we:
  1. Detect people and their 17 body keypoints     (YOLO11n-pose)
  2. Detect the bed                                  (YOLO11n, COCO class "bed")
  3. Pick the patient if several people are visible  (caregiver handling)
  4. Compute simple, explainable features:

     person_found   is anyone visible?
     kp_visible     fraction of the 17 keypoints that are confidently visible
                    (low when covered by a blanket, occluded or in the dark)
     torso_angle    angle of the shoulder->hip line from vertical, degrees
                    0 = upright (sitting/standing), 90 = horizontal (lying)
     aspect         person box width / height  (lying people are wide)
     bed_found      is a bed box available (tracked, remembered for a while)
     bed_seen       was the bed detected in THIS frame (False when the camera
                    is blocked or dark; used to tell "under blanket" from
                    "camera blocked")
     bed_overlap    fraction of the person box that lies inside the bed box
     hip_on_bed     are the hips inside the bed box?
     knee_angle     hip-knee-ankle angle, degrees (180 = straight leg)
     motion         hip movement per second, in "body heights"
     brightness     mean image brightness (0-255), to detect poor lighting

Why keypoints + rules instead of a trained classifier?
  No training data of our own is needed, every decision is explainable,
  and it runs on a laptop CPU.

Run:
  python -m src.perception data/sequences/seq1_exit_and_return.mp4
  python -m src.perception data/sequences/seq1_exit_and_return.mp4 --viz
Output:
  outputs/features/<video>.csv      (one row per sampled frame)
  outputs/viz/<video>_viz.mp4       (with --viz: skeleton + bed drawn)
"""

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src import config as C

# COCO keypoint indices
L_SH, R_SH, L_HIP, R_HIP = 5, 6, 11, 12
L_KNEE, R_KNEE, L_ANK, R_ANK = 13, 14, 15, 16

FEATURES = ["t", "n_persons", "person_found", "person_conf", "kp_visible",
            "torso_angle", "aspect", "bed_found", "bed_seen", "bed_overlap", "hip_on_bed",
            "knee_angle", "motion", "brightness",
            "px1", "py1", "px2", "py2", "bx1", "by1", "bx2", "by2"]


# ---------------------------------------------------------------------------
# Small geometry helpers
# ---------------------------------------------------------------------------
def midpoint(kp, conf, a, b):
    """Midpoint of two keypoints; falls back to whichever one is visible."""
    pts = [kp[i] for i in (a, b) if conf[i] >= C.KP_CONF]
    return np.mean(pts, axis=0) if pts else None


def angle_from_vertical(top, bottom):
    dx, dy = bottom[0] - top[0], bottom[1] - top[1]
    return math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))


def joint_angle(a, b, c):
    """Angle at point b formed by a-b-c, in degrees."""
    v1, v2 = a - b, c - b
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return math.degrees(math.acos(np.clip(cos, -1, 1)))


def overlap_fraction(inner, outer):
    """Fraction of box `inner` covered by box `outer`."""
    x1, y1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    x2, y2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area = max(1e-6, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return inter / area


def inside(pt, box, margin=0.0):
    return (box[0] - margin <= pt[0] <= box[2] + margin
            and box[1] - margin <= pt[1] <= box[3] + margin)


# ---------------------------------------------------------------------------
# Bed tracking: detection is noisy frame to frame, so smooth it and
# remember the last good box for a while (the bed does not move).
# ---------------------------------------------------------------------------
class BedTracker:
    def __init__(self):
        self.box = None
        self.last_seen = -1e9

    def update(self, det_box, t):
        if det_box is not None:
            det_box = np.array(det_box, dtype=float)
            self.box = det_box if self.box is None else (
                C.BED_SMOOTHING * self.box + (1 - C.BED_SMOOTHING) * det_box)
            self.last_seen = t
        elif t - self.last_seen > C.BED_MEMORY_SEC:
            self.box = None
        return self.box


# ---------------------------------------------------------------------------
# Patient selection: if several people are visible (e.g. a caregiver),
# prefer the one closest to where the patient was, then the one on the bed.
# ---------------------------------------------------------------------------
def pick_patient(persons, bed_box, prev_center):
    def score(p):
        s = p["conf"]
        if bed_box is not None:
            s += overlap_fraction(p["box"], bed_box)
        if prev_center is not None:
            c = np.array([(p["box"][0] + p["box"][2]) / 2,
                          (p["box"][1] + p["box"][3]) / 2])
            s -= np.linalg.norm(c - prev_center) / 500.0   # pixels -> penalty
        return s
    return max(persons, key=score)


# ---------------------------------------------------------------------------
def extract_features(person, bed_box, prev_hip, dt):
    kp, kc = person["kp"], person["kp_conf"]
    x1, y1, x2, y2 = person["box"]
    w, h = x2 - x1, y2 - y1

    f = {"person_conf": round(person["conf"], 3),
         "kp_visible": round(float(np.mean(kc >= C.KP_CONF)), 3),
         "aspect": round(w / max(h, 1), 3)}

    sh = midpoint(kp, kc, L_SH, R_SH)
    hip = midpoint(kp, kc, L_HIP, R_HIP)
    f["torso_angle"] = round(angle_from_vertical(sh, hip), 1) if (
        sh is not None and hip is not None) else ""

    # knee angle: average of the legs we can see
    knees = []
    for hi, ki, ai in ((L_HIP, L_KNEE, L_ANK), (R_HIP, R_KNEE, R_ANK)):
        if min(kc[hi], kc[ki], kc[ai]) >= C.KP_CONF:
            knees.append(joint_angle(kp[hi], kp[ki], kp[ai]))
    f["knee_angle"] = round(float(np.mean(knees)), 1) if knees else ""

    if bed_box is not None:
        f["bed_overlap"] = round(overlap_fraction(person["box"], bed_box), 3)
        ref = hip if hip is not None else np.array([(x1 + x2) / 2, y2 - 0.3 * h])
        f["hip_on_bed"] = int(inside(ref, bed_box, margin=0.02 * (bed_box[2] - bed_box[0])))
    else:
        f["bed_overlap"], f["hip_on_bed"] = "", ""

    # motion: how far the hips moved per second, relative to body size
    centre = hip if hip is not None else np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    if prev_hip is not None and dt > 0:
        f["motion"] = round(float(np.linalg.norm(centre - prev_hip) / max(h, w, 1) / dt), 3)
    else:
        f["motion"] = ""
    return f, centre


def draw(frame, persons, patient, bed_box, row):
    out = frame.copy()
    if bed_box is not None:
        b = bed_box.astype(int)
        cv2.rectangle(out, (b[0], b[1]), (b[2], b[3]), (255, 160, 0), 2)
        cv2.putText(out, "bed", (b[0] + 4, b[1] + 22), 0, 0.7, (255, 160, 0), 2)
    for p in persons:
        colour = (0, 220, 0) if p is patient else (0, 0, 220)
        b = np.array(p["box"]).astype(int)
        cv2.rectangle(out, (b[0], b[1]), (b[2], b[3]), colour, 2)
        for (x, y), c in zip(p["kp"], p["kp_conf"]):
            if c >= C.KP_CONF:
                cv2.circle(out, (int(x), int(y)), 4, colour, -1)
    txt = (f"t={row['t']:.1f}s torso={row['torso_angle']} hip_on_bed={row['hip_on_bed']} "
           f"kp={row['kp_visible']} motion={row['motion']}")
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(out, txt, (8, 21), 0, 0.6, (255, 255, 255), 1)
    return out


# ---------------------------------------------------------------------------
def run(video_path, out_dir="outputs", viz=False):
    video_path = Path(video_path)
    pose_model, det_model = YOLO(C.POSE_MODEL), YOLO(C.DET_MODEL)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ! cannot open video: {video_path}")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(fps / C.SAMPLE_FPS)))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    feat_dir = Path(out_dir) / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    if viz:
        (Path(out_dir) / "viz").mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(Path(out_dir) / "viz" / f"{video_path.stem}_viz.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), C.SAMPLE_FPS, (W, H))

    bed = BedTracker()
    prev_center, prev_t = None, None
    rows = []

    for idx in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps

        # --- bed ---
        det = det_model(frame, classes=[C.BED_CLASS_ID], conf=C.BED_CONF, verbose=False)[0]
        det_box = None
        if len(det.boxes):
            i = int(det.boxes.conf.argmax())
            det_box = det.boxes.xyxy[i].cpu().numpy()
        bed_box = bed.update(det_box, t)

        # --- people + keypoints ---
        res = pose_model(frame, conf=C.PERSON_CONF, verbose=False)[0]
        persons = []
        if res.keypoints is not None and len(res.boxes):
            kps = res.keypoints.xy.cpu().numpy()
            kcs = res.keypoints.conf.cpu().numpy()
            for b, c, kp, kc in zip(res.boxes.xyxy.cpu().numpy(),
                                    res.boxes.conf.cpu().numpy(), kps, kcs):
                persons.append({"box": b, "conf": float(c), "kp": kp, "kp_conf": kc})

        row = {k: "" for k in FEATURES}
        row.update(t=round(t, 2), n_persons=len(persons), person_found=int(bool(persons)),
                   bed_found=int(bed_box is not None),   # tracked (with memory)
                   bed_seen=int(det_box is not None),    # detected in THIS frame
                   brightness=round(float(frame.mean()), 1))
        if bed_box is not None:
            row.update(bx1=round(bed_box[0] / W, 3), by1=round(bed_box[1] / H, 3),
                       bx2=round(bed_box[2] / W, 3), by2=round(bed_box[3] / H, 3))

        patient = None
        if persons:
            patient = pick_patient(persons, bed_box, prev_center)
            dt = (t - prev_t) if prev_t is not None else 0
            f, prev_center = extract_features(patient, bed_box, prev_center, dt)
            prev_t = t
            row.update(f)
            b = patient["box"]
            row.update(px1=round(b[0] / W, 3), py1=round(b[1] / H, 3),
                       px2=round(b[2] / W, 3), py2=round(b[3] / H, 3))
        rows.append(row)

        if writer is not None:
            writer.write(draw(frame, persons, patient, bed_box, row))
        if len(rows) % 20 == 0:
            print(f"  {t:6.1f}s / {total / fps:.1f}s")

    cap.release()
    if writer is not None:
        writer.release()

    if not rows:
        print(f"  ! no frames could be read from {video_path}")
        return None
    out_csv = feat_dir / f"{video_path.stem}.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FEATURES)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {len(rows)} rows -> {out_csv}")
    return out_csv


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--viz", action="store_true", help="also save an annotated video")
    args = ap.parse_args()
    for v in args.videos:
        print(f"Processing {v}")
        run(v, viz=args.viz)