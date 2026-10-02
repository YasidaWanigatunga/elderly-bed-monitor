import csv
from collections import Counter, defaultdict

from src.classifier import STATES


def load_ground_truth(path):
    with open(path) as f:
        return [(float(r["start_sec"]), float(r["end_sec"]), r["state"])
                for r in csv.DictReader(f)]


def gt_state_at(gt, t):
    for s, e, st in gt:
        if s <= t < e:
            return st
    return gt[-1][2]


def gt_durations(gt):
    d = defaultdict(float)
    for s, e, st in gt:
        d[st] += e - s
    return d


def state_metrics(times, predicted, gt):
    truth = [gt_state_at(gt, t) for t in times]
    correct = sum(p == g for p, g in zip(predicted, truth))
    confusion = defaultdict(Counter)
    for p, g in zip(predicted, truth):
        confusion[g][p] += 1
    recall = {g: round(confusion[g][g] / sum(confusion[g].values()), 3) for g in confusion}
    return {"accuracy": round(correct / len(truth), 3), "frames": len(truth),
            "per_state_recall": recall,
            "confusion": {g: dict(c) for g, c in confusion.items()}}


def duration_errors(pred_durations, gt):
    true_d = gt_durations(gt)
    states = [s for s in STATES if s in true_d or s in pred_durations]
    return {s: {"true_sec": round(true_d.get(s, 0), 1),
                "pred_sec": round(pred_durations.get(s, 0), 1),
                "error_sec": round(pred_durations.get(s, 0) - true_d.get(s, 0), 1)}
            for s in states}


def print_confusion(conf):
    labels = [s for s in STATES if s in conf or any(s in r for r in conf.values())]
    short = {s: s.replace("_IN_BED", "").replace("_ON_BED", "_BED")
             .replace("_OUTSIDE_BED", "_OUT").replace("OUT_OF_BED", "OUT")[:10]
             for s in labels}
    print("  true \\ pred   " + "".join(f"{short[s]:>11}" for s in labels))
    for g in labels:
        if g not in conf:
            continue
        print(f"  {short[g]:<13}" + "".join(f"{conf[g].get(p, 0):>11}" for p in labels))