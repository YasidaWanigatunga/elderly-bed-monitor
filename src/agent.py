import base64
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import cv2

from src import config as C
from src.classifier import (IN_BED, LYING, OUT, SIT_BED, SIT_OUT, STAND,
                            UNKNOWN, WALK)
from src.events import OUT_STATES, status
from src.timeline import fmt

CACHE_DIR = Path("outputs") / "agent_cache"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

VLM_OPTIONS = {
    "A": LYING,
    "B": SIT_BED,
    "C": STAND,
    "D": SIT_OUT,
    "E": None,               # cannot tell
    "F": "FLOOR",            # lying on the floor: possible fall
}
VLM_QUESTION = (
    "These {n} frames come from a fixed bedroom camera that monitors an elderly "
    "person. They are {gap:.1f} seconds apart. What is the person doing?\n"
    "A) lying on the bed\n"
    "B) sitting on the bed (hips resting on the mattress)\n"
    "C) standing or walking with feet on the floor, including standing beside "
    "the bed or bending over it\n"
    "D) sitting on a chair or anywhere outside the bed\n"
    "E) cannot tell (person not visible, image too dark, or view blocked)\n"
    "F) lying on the floor\n"
    'Reply with JSON only: {{"answer": "A|B|C|D|E|F", "confidence": 0.0-1.0, '
    '"reason": "one short sentence"}}'
)


def load_api_key():
    """GROQ_API_KEY from the environment or from a .env file in the project root."""
    if os.environ.get("GROQ_API_KEY"):
        return os.environ["GROQ_API_KEY"]
    env = Path(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            if line.strip().startswith("GROQ_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def merge_segments(segments):
    """Join neighbouring segments that now have the same state."""
    out = []
    for s in segments:
        if out and out[-1]["state"] == s["state"]:
            p = out[-1]
            d1, d2 = p["duration_sec"], s["duration_sec"]
            p["confidence"] = round((p["confidence"] * d1 + s["confidence"] * d2) / max(d1 + d2, 1e-6), 2)
            p["end_sec"], p["end"] = s["end_sec"], s["end"]
            p["duration_sec"] = round(p["end_sec"] - p["start_sec"], 1)
            p["flags"] = sorted(set(p["flags"]) | set(s["flags"]))
        else:
            out.append(dict(s))
    return out

class Agent:
    def __init__(self, name, segments, video_path=None, use_vlm=True):
        self.name = name
        self.segments = [dict(s) for s in segments]
        self.video_path = video_path
        self.api_key = load_api_key() if use_vlm else None
        self.use_vlm = bool(self.api_key and video_path and Path(video_path).exists())
        self.vlm_calls = 0
        self.trace = []
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.cache_file = CACHE_DIR / f"{name}.json"
        self.cache = json.loads(self.cache_file.read_text()) if self.cache_file.exists() else {}

    def look_back(self, i, seconds=C.AGENT_CONTEXT_SEC):
        start = self.segments[i]["start_sec"] - seconds
        return self._summarise(start, self.segments[i]["start_sec"])

    def look_forward(self, i, seconds=C.AGENT_CONTEXT_SEC):
        end = self.segments[i]["end_sec"]
        return self._summarise(end, end + seconds)

    def neighbour(self, i, direction):
        """State of the nearest segment before (-1) or after (+1), skipping UNKNOWN."""
        j = i + direction
        while 0 <= j < len(self.segments):
            if self.segments[j]["state"] != UNKNOWN:
                return self.segments[j]["state"]
            j += direction
        return None

    def _summarise(self, a, b):
        """Seconds spent in each state between a and b, and the dominant one."""
        time_in = defaultdict(float)
        for s in self.segments:
            overlap = min(s["end_sec"], b) - max(s["start_sec"], a)
            if overlap > 0:
                time_in[s["state"]] += overlap
        if not time_in:
            return {"states": {}, "dominant": None, "bed": None}
        dominant = max(time_in, key=time_in.get)
        bed_time = defaultdict(float)
        for st, sec in time_in.items():
            if status(st):
                bed_time[status(st)] += sec
        return {"states": {k: round(v, 1) for k, v in time_in.items()},
                "dominant": dominant,
                "bed": max(bed_time, key=bed_time.get) if bed_time else None}

    def ask_vlm(self, times):
        """Show frames at `times` to the VLM. Returns dict or None."""
        if not self.use_vlm:
            return None
        key = ",".join(f"{t:.1f}" for t in times)
        if key in self.cache:
            return self.cache[key]
        images = [self._frame_b64(t) for t in times]
        images = [im for im in images if im]
        if not images:
            return None
        gap = (times[-1] - times[0]) / max(1, len(times) - 1)
        content = [{"type": "text", "text": VLM_QUESTION.format(n=len(images), gap=gap)}]
        content += [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{im}"}}
                    for im in images]
        body = {"model": C.VLM_MODEL, "temperature": 0, "max_tokens": 200,
                "messages": [{"role": "user", "content": content}],
                "response_format": {"type": "json_object"},
                "reasoning_effort": "none"}
        answer = self._post(body)
        if answer is not None:
            self.cache[key] = answer
            self.cache_file.write_text(json.dumps(self.cache, indent=2))
        return answer

    def _post(self, body):
        import requests                          # installed with ultralytics
        headers = {"Authorization": f"Bearer {self.api_key}"}
        for attempt in range(3):
            try:
                time.sleep(C.VLM_MIN_INTERVAL_SEC)       # stay under the free rate limit
                r = requests.post(GROQ_URL, headers=headers, json=body, timeout=60)
                self.vlm_calls += 1
                if r.status_code == 400 and "reasoning_effort" in body:
                    body = {k: v for k, v in body.items() if k != "reasoning_effort"}
                    continue
                if r.status_code == 429:
                    time.sleep(10 * (attempt + 1))
                    continue
                r.raise_for_status()
                text = r.json()["choices"][0]["message"]["content"]
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
                m = re.search(r"\{.*\}", text, flags=re.S)
                data = json.loads(m.group(0)) if m else {}
                letter = str(data.get("answer", "E")).strip()[:1].upper()
                return {"answer": letter, "state": VLM_OPTIONS.get(letter),
                        "confidence": float(data.get("confidence", 0.5)),
                        "reason": str(data.get("reason", ""))[:200]}
            except Exception as e:                       # network, JSON, ...
                print(f"    VLM call failed ({e.__class__.__name__}: {e})")
        return None

    def _frame_b64(self, t):
        cap = cv2.VideoCapture(str(self.video_path))
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return None
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (C.VLM_IMAGE_WIDTH, int(h * C.VLM_IMAGE_WIDTH / w)))
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf).decode() if ok else None

    def _frame_times(self, seg):
        """Up to 3 evenly spaced moments inside a segment."""
        a, b = seg["start_sec"], seg["end_sec"]
        if b - a < 1.0:
            return [round((a + b) / 2, 1)]
        return [round(a + (b - a) * f, 1) for f in (0.2, 0.5, 0.8)]

    def triggers(self):
        found = []
        for i, s in enumerate(self.segments):
            if (s["state"] in IN_BED and s["duration_sec"] <= C.AGENT_SHORT_SEGMENT_SEC
                    and ("ambiguous_sit_or_stand" in s["flags"]
                         or s["confidence"] < C.AGENT_LOW_CONFIDENCE)):
                found.append((i, "short_ambiguous_in_bed"))
            elif s["state"] == UNKNOWN and "possible_fall" in s["flags"]:
                found.append((i, "possible_fall"))
            elif s["state"] == UNKNOWN:
                found.append((i, "unknown"))
            elif any(f.startswith("implausible:") for f in s["flags"]):
                found.append((i, "implausible_transition"))
        return found

    # reviews 
    def _describe(self, ctx):
        if not ctx["states"]:
            return "nothing (start/end of video)"
        return ", ".join(f"{k} {v:.0f}s" for k, v in sorted(ctx["states"].items(),
                                                          key=lambda kv: -kv[1]))

    def review(self, i, trigger):
        seg = self.segments[i]
        steps = []
        label = f"{seg['start']}-{seg['end']} {seg['state']} (conf {seg['confidence']})"
        new_state, decided_by = seg["state"], "unchanged"

        back = self.look_back(i)
        steps.append(("Analyze previous segment", f"previous {C.AGENT_CONTEXT_SEC:.0f}s: "
                      f"{self._describe(back)}"))
        fwd = self.look_forward(i)
        steps.append(("Analyze following segment", f"next {C.AGENT_CONTEXT_SEC:.0f}s: "
                      f"{self._describe(fwd)}"))

        if trigger == "short_ambiguous_in_bed":
            observation = (f"{seg['state']} for only {seg['duration_sec']:.0f}s with low "
                           f"confidence; hips overlap the bed box in 2-D")
            thought = "One frame cannot tell sitting on the bed from standing beside it."
            prev_st, next_st = self.neighbour(i, -1), self.neighbour(i, +1)
            steps.append(("Check neighbouring segments",
                          f"directly before: {prev_st}, directly after: {next_st}"))
            if status(prev_st) == "OUT" and status(next_st) == "OUT":
                hypothesis = STAND
                steps.append(("Compare context", "sandwiched between out-of-bed segments: a real "
                              f"sit-down of only {seg['duration_sec']:.0f}s is unlikely; "
                              "probably standing beside the bed"))
            elif status(prev_st) == "IN" and status(next_st) == "IN":
                hypothesis = seg["state"]
                steps.append(("Compare context", "in bed before and after: consistent, keep"))
            else:
                hypothesis = None
                steps.append(("Compare context", "this is a boundary between in and out of "
                              "bed: context alone cannot decide"))
            new_state, decided_by = self._decide_with_vlm(seg, hypothesis, steps)

        elif trigger == "unknown":
            observation = f"state UNKNOWN for {seg['duration_sec']:.0f}s ({', '.join(seg['flags']) or 'no evidence'})"
            thought = "Check whether the person can actually be seen before guessing."
            if "blackout" in seg["flags"]:
                steps.append(("Inspect frames", "camera shows a black image: no evidence"))
                new_state, decided_by = UNKNOWN, "evidence_check"
            else:
                new_state, decided_by = self._decide_with_vlm(seg, None, steps,
                                                              min_conf=C.AGENT_VLM_STRONG)

        elif trigger == "possible_fall":
            observation = "horizontal body detected outside the bed region"
            thought = "Determine whether the person is on the bed or on the floor."
            if back["dominant"] == LYING and back["bed"] == "IN":
                steps.append(("Compare context", "was lying in bed just before: likely the "
                              "bed box was too small, not a fall"))
                hypothesis = LYING
            else:
                hypothesis = None
                steps.append(("Compare context", "was not lying in bed before: cannot rule out "
                              "a fall from context; safety first, keep the alert unless the "
                              "frames clearly show otherwise"))
            new_state, decided_by = self._decide_with_vlm(seg, hypothesis, steps,
                                                          min_conf=C.AGENT_VLM_STRONG)

        else:  # implausible_transition
            observation = f"physically unlikely change into {seg['state']}"
            thought = "Verify the new state before accepting the jump."
            new_state, decided_by = self._decide_with_vlm(seg, seg["state"], steps)

        conclusion = (f"{seg['state']} confirmed" if new_state == seg["state"]
                      else f"{seg['state']} -> {new_state}")
        self.trace.append({"segment": label, "trigger": trigger, "observation": observation,
                           "agent": thought, "steps": [{"action": a, "finding": f} for a, f in steps],
                           "conclusion": conclusion, "decided_by": decided_by,
                           "new_state": new_state})
        return new_state, decided_by

    def _decide_with_vlm(self, seg, hypothesis, steps, min_conf=None):
        """Use the VLM if available; otherwise fall back to the context hypothesis."""
        min_conf = min_conf or C.AGENT_VLM_MIN_CONF
        ans = self.ask_vlm(self._frame_times(seg)) if self.use_vlm else None
        if ans is not None:
            steps.append(("Ask vision-language model",
                          f"answer {ans['answer']} ({ans['state'] or 'cannot tell'}), "
                          f"confidence {ans['confidence']:.2f}: {ans['reason']}"))
            if ans["state"] == "FLOOR":
                return UNKNOWN, "vlm_floor"              # keep as possible fall -> ALERT
                return (hypothesis or UNKNOWN,
                        "vlm_cannot_tell" + ("+context" if hypothesis else ""))
            if ans["confidence"] >= min_conf:
                if hypothesis and status(ans["state"]) == status(hypothesis):
                    return hypothesis, "context+vlm"
                return ans["state"], "vlm"
        elif not self.use_vlm:
            steps.append(("Ask vision-language model", "not available (no API key): "
                          "using temporal context only"))
        if hypothesis:
            return hypothesis, "context"
        return seg["state"], "unchanged"

    def run(self):
        for i, trigger in self.triggers():
            new_state, by = self.review(i, trigger)
            if by == "vlm_floor":
                self.segments[i]["flags"] = sorted(set(self.segments[i]["flags"]) |
                                                   {"possible_fall", "agent_reviewed"})
                continue
            if new_state != self.segments[i]["state"]:
                s = self.segments[i]
                s["state"] = new_state
                s["reason"] = f"agent ({by}): {self.trace[-1]['conclusion']}"
                s["flags"] = sorted(set(s["flags"]) - {"ambiguous_sit_or_stand", "possible_fall"}
                                    | {"agent_corrected"})
            else:
                self.segments[i]["flags"] = sorted(set(self.segments[i]["flags"]) | {"agent_reviewed"})
        return merge_segments(self.segments)

    def trace_markdown(self):
        lines = [f"# Agent trace: {self.name}",
                 f"VLM used: {'yes (' + C.VLM_MODEL + ')' if self.use_vlm else 'no (context only)'}"
                 f", VLM calls this run: {self.vlm_calls}", ""]
        for k, t in enumerate(self.trace, 1):
            lines += [f"## {k}. {t['segment']}  [{t['trigger']}]",
                      f"**Observation:** {t['observation']}  ",
                      f"**Agent:** {t['agent']}  "]
            for st in t["steps"]:
                lines += [f"**Action:** {st['action']}  ", f"**Finding:** {st['finding']}  "]
            lines += [f"**Conclusion:** {t['conclusion']} (decided by: {t['decided_by']})", ""]
        return "\n".join(lines)


def frames_from_segments(times, segments):
    """State at each sampled time, from a list of segments (for evaluation)."""
    out, j = [], 0
    for t in times:
        while j < len(segments) - 1 and t >= segments[j]["end_sec"]:
            j += 1
        out.append(segments[j]["state"])
    return out