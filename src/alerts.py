"""
alerts.py
---------
Stage 6: decide NORMAL / MONITOR / ALERT, with an explanation.

Rules (each one has a reason a caregiver would agree with):

  ALERT    possible fall
           A horizontal body OUTSIDE the bed. Lying on the floor is the most
           dangerous situation for an elderly person living alone.

  ALERT    prolonged absence from bed
           Out of bed longer than PROLONGED_ABSENCE_SEC. At night a long
           absence may mean a fall out of view, confusion or wandering.

  MONITOR  bed exit
           Every bed exit is worth a look (most night-time falls happen
           shortly after getting up), but it is not an emergency by itself.
           This matches the assignment's example (bed_exit -> MONITOR).

  MONITOR  long sitting on the bed
           Sitting on the bed longer than LONG_SIT_SEC. Long sitting on the
           bed edge often comes before an exit, or means dizziness or
           difficulty standing up.

  MONITOR  cannot see the person
           UNKNOWN for longer than UNKNOWN_MONITOR_SEC. The system should
           say "I don't know" instead of guessing, and a human should check.

  MONITOR  implausible transition
           The state machine accepted a physically unlikely jump
           (e.g. lying -> walking). Something unusual happened, or the
           system made an error. Either way a human should check.

  NORMAL   otherwise: lying, sitting, standing or walking normally.

Thresholds come from config.ALERT_PROFILES: "realistic" values for a real
night, and "demo" values scaled down for our 3-4 minute test videos.
"""

from src import config as C
from src.classifier import SIT_BED, UNKNOWN
from src.events import hms

SEVERITY = {"NORMAL": 0, "MONITOR": 1, "ALERT": 2}


def thresholds():
    return C.ALERT_PROFILES[C.ALERT_PROFILE]


def evaluate_alerts(segments, events, out_periods):
    th = thresholds()
    alerts = []

    def add(decision, rule, t, explanation):
        alerts.append({"decision": decision, "rule": rule, "time": hms(t),
                       "time_sec": round(t, 2), "explanation": explanation})

    # possible fall
    for s in segments:
        if "possible_fall" in s["flags"]:
            add("ALERT", "possible_fall", s["start_sec"],
                "horizontal body detected outside the bed region")

    # prolonged absence
    for a, b in out_periods:
        if b - a > th["PROLONGED_ABSENCE_SEC"]:
            add("ALERT", "prolonged_absence", a + th["PROLONGED_ABSENCE_SEC"],
                f"out of bed for {b - a:.0f}s (limit {th['PROLONGED_ABSENCE_SEC']}s)")

    # every bed exit
    for e in events:
        if e["event"] == "bed_exit":
            add("MONITOR", "bed_exit", e["confirmed_sec"],
                f"left the bed ({e['previous_state']} -> {e['current_state']})")

    # long sitting on the bed
    for s in segments:
        if s["state"] == SIT_BED and s["duration_sec"] > th["LONG_SIT_SEC"]:
            add("MONITOR", "long_sitting_on_bed", s["start_sec"] + th["LONG_SIT_SEC"],
                f"sitting on the bed for {s['duration_sec']:.0f}s "
                f"(limit {th['LONG_SIT_SEC']}s)")

    # cannot see the person
    for s in segments:
        if s["state"] == UNKNOWN and s["duration_sec"] > th["UNKNOWN_MONITOR_SEC"]:
            add("MONITOR", "state_unknown", s["start_sec"] + th["UNKNOWN_MONITOR_SEC"],
                f"state could not be determined for {s['duration_sec']:.0f}s "
                f"({', '.join(s['flags']) or 'no evidence'})")

    # implausible transitions
    for s in segments:
        for fl in s["flags"]:
            if fl.startswith("implausible:"):
                add("MONITOR", "implausible_transition", s["start_sec"],
                    f"physically unlikely change {fl.split(':', 1)[1]}")

    alerts.sort(key=lambda a: a["time_sec"])
    overall = max((a["decision"] for a in alerts), key=SEVERITY.get, default="NORMAL")
    return alerts, overall


def decision_for_event(event, alerts, events):
    """Decision attached to a bed event.

    bed_exit:      ALERT if a fall / prolonged absence happens before the
                   person returns to bed, otherwise MONITOR.
    return_to_bed: NORMAL (the person is safely back).
    """
    if event["event"] != "bed_exit":
        return "NORMAL"
    back = min((e["start_sec"] for e in events
                if e["event"] == "return_to_bed" and e["start_sec"] > event["start_sec"]),
               default=float("inf"))
    serious = [a for a in alerts if event["start_sec"] <= a["time_sec"] <= back
               and a["rule"] in ("prolonged_absence", "possible_fall")]
    return "ALERT" if serious else "MONITOR"