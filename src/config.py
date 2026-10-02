# Perception
SAMPLE_FPS = 2.0            # frames analysed per second of video
POSE_MODEL = "yolo11n-pose.pt"   # person + 17 body keypoints (COCO format)
DET_MODEL = "yolo11n.pt"         # general detector, used for the "bed" class
BED_CLASS_ID = 59                # "bed" in the COCO class list
PERSON_CONF = 0.35          # ignore person detections below this
BED_CONF = 0.25             # ignore bed detections below this
KP_CONF = 0.5               # a keypoint counts as "visible" above this
BED_SMOOTHING = 0.7         # EMA weight on the previous bed box (0 = no smoothing)
BED_MEMORY_SEC = 30.0       # keep using the last bed box this long if bed not seen

# Frame classifier (one frame at a time)
# Measured medians: lying torso ~78 deg, sitting ~10, standing ~22.
LYING_TORSO_DEG = 50        # torso tilted more than this = body is horizontal
UPRIGHT_TORSO_DEG = 35      # torso tilted less than this = body is upright
# Measured medians: lying aspect ~2.9, sitting ~1.1, standing ~0.9, walking ~0.6
LYING_MIN_ASPECT = 1.3      # a horizontal torso also needs a wide box to be "lying"
WIDE_ASPECT = 2.2           # box this wide = lying, even if torso not measurable
STANDING_MAX_ASPECT = 0.9   # tall, narrow box with straight legs = standing
STRAIGHT_KNEE_DEG = 160     # knee angle above this = straight leg
SIT_KNEE_DEG = 115          # knee angle below this = bent leg (sitting)
WALK_MOTION = 0.30          # hip movement (body heights / second) above this = walking
HIDDEN_TORSO_LYING_ASPECT = 1.2  # on bed, shoulders hidden, box wider than this = lying
AMBIGUOUS_OVERLAP = 0.35    # "sitting" but less than this much of the body inside
                            # the bed box: may be standing beside it (agent checks)
MIN_KP_VISIBLE = 0.25       # fewer visible keypoints than this = pose not reliable
DARK_BRIGHTNESS = 40        # darker than this = low light, reduce confidence
BLACKOUT_BRIGHTNESS = 8     # darker than this = camera shows nothing

# Temporal layer (across frames) 
SMOOTH_WINDOW_SEC = 2.5     # confidence-weighted vote over this window
MIN_DWELL_SEC = 1.5         # a new state must last this long to be accepted
IMPLAUSIBLE_DWELL_SEC = 3.0 # an "impossible" jump needs this much evidence
HIDDEN_IN_BED_MAX_SEC = 60  # person invisible but bed visible: assume still
                            # in bed (under blanket) for at most this long
EDGE_MARGIN = 0.05          # person box this close to the frame edge = "at the edge"

# Agent
AGENT_CONTEXT_SEC = 10.0          # how far look_back / look_forward look
AGENT_SHORT_SEGMENT_SEC = 5.0     # in-bed segments this short are suspicious
AGENT_LOW_CONFIDENCE = 0.5        # segments below this confidence get reviewed
AGENT_VLM_MIN_CONF = 0.6          # VLM answer needed to agree/override
AGENT_VLM_STRONG = 0.75           # VLM answer needed to REPLACE an UNKNOWN
VLM_MODEL = "qwen/qwen3.8-27b"    # Groq vision model (check console.groq.com/docs/vision)
VLM_IMAGE_WIDTH = 640             # frames are resized to this width before sending
VLM_MIN_INTERVAL_SEC = 2.5        # wait between calls (free tier: 30 requests/min)

# Bed events
EXIT_CONFIRM_SEC = 5.0      # out of bed this long (or starts walking) = confirmed exit
RETURN_CONFIRM_SEC = 10.0   # back on bed this long (or lies down) = confirmed return
EVENT_MATCH_TOLERANCE_SEC = 10.0  # evaluation: predicted vs true event start

# Alerts
ALERT_PROFILE = "demo"
ALERT_PROFILES = {
    "realistic": {"PROLONGED_ABSENCE_SEC": 900,   # 15 min out of bed at night
                  "LONG_SIT_SEC": 180,            # 3 min sitting on the bed
                  "UNKNOWN_MONITOR_SEC": 60},     # 1 min without seeing the person
    "demo":      {"PROLONGED_ABSENCE_SEC": 120,
                  "LONG_SIT_SEC": 90,
                  "UNKNOWN_MONITOR_SEC": 3},
}