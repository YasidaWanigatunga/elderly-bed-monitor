from dataclasses import dataclass, field

from src import config as C

LYING = "LYING_IN_BED"
SIT_BED = "SITTING_ON_BED"
SIT_OUT = "SITTING_OUTSIDE_BED"
STAND = "STANDING"
WALK = "WALKING"
OUT = "OUT_OF_BED"
UNKNOWN = "UNKNOWN"
NO_PERSON = "NO_PERSON"

STATES = [LYING, SIT_BED, SIT_OUT, STAND, WALK, OUT, UNKNOWN]
IN_BED = {LYING, SIT_BED}


@dataclass
class Observation:
    t: float
    state: str
    confidence: float
    reason: str
    flags: list = field(default_factory=list)
    features: dict = field(default_factory=dict)


def _num(row, key):
    v = row.get(key, "")
    return float(v) if v not in ("", None) else None


def _clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def classify(row):
    """row: one line of the features CSV (dict of strings)."""
    t = float(row["t"])
    f = {k: _num(row, k) for k in row}          # everything as numbers
    torso, aspect, knee = f.get("torso_angle"), f.get("aspect"), f.get("knee_angle")
    kp, motion = f.get("kp_visible") or 0.0, f.get("motion")
    bright = f.get("brightness") or 0.0

    #1. can we see anything at all? 
    if bright < C.BLACKOUT_BRIGHTNESS:
        return Observation(t, UNKNOWN, 0.9, "camera shows nothing (blackout)",
                           ["blackout"], f)
    if not f.get("person_found"):
        where = "bed visible" if f.get("bed_seen") else "bed not visible either"
        return Observation(t, NO_PERSON, 0.5, f"no person detected, {where}", [], f)
    if kp < C.MIN_KP_VISIBLE:
        return Observation(t, UNKNOWN, 0.6,
                           f"person detected but only {kp:.0%} of body visible",
                           ["occluded"], f)

    #2. body shape
    if torso is not None:
        horizontal = torso > C.LYING_TORSO_DEG and (aspect or 0) > C.LYING_MIN_ASPECT
        upright = torso < C.UPRIGHT_TORSO_DEG
        shape_margin = abs(torso - 45) / 45          # 0 = ambiguous, 1 = very clear
    else:
        horizontal = (aspect or 0) > C.WIDE_ASPECT
        upright = (aspect or 99) < 1.2
        shape_margin = 0.3                            # guessed from box only
    if (aspect or 0) > C.WIDE_ASPECT:
        horizontal = True
    straight_legs = knee is not None and knee > C.STRAIGHT_KNEE_DEG
    bent_legs = knee is not None and knee < C.SIT_KNEE_DEG
    moving = motion is not None and motion > C.WALK_MOTION

    # 3. where is the body relative to the bed? 
    if f.get("hip_on_bed") is not None:
        on_bed = f["hip_on_bed"] == 1
    elif f.get("bed_overlap") is not None:
        on_bed = f["bed_overlap"] > 0.5
    else:
        on_bed = None                                  # no bed detected

    # 4. decide
    flags = []
    if on_bed is None:
        flags.append("no_bed")
        if horizontal:
            state, why = LYING, "horizontal body (no bed detected, assumed in bed)"
        elif moving:
            state, why = WALK, "moving (no bed detected)"
        else:
            state, why = STAND, "upright (no bed detected)"
        base = 0.35
    elif on_bed:
        if torso is not None and torso > C.LYING_TORSO_DEG:
            # on the bed a tilted torso is enough (no box-shape check needed:
            # bending over the bed usually puts the hips OFF the bed box)
            state, why = LYING, f"hips on bed, torso {torso:.0f} deg from vertical"
        elif torso is None and (aspect or 0) > C.HIDDEN_TORSO_LYING_ASPECT:
            # shoulders hidden (usually by the blanket) and a wide box
            state, why = LYING, "hips on bed, shoulders hidden, wide box (under blanket?)"
            shape_margin = 0.0
        elif straight_legs and (aspect or 1) < C.STANDING_MAX_ASPECT:
            state, why = STAND, "hips overlap bed box but legs straight and box tall"
        else:
            state, why = SIT_BED, f"hips on bed, torso upright ({torso or 0:.0f} deg)"
            if (f.get("bed_overlap") or 1) < C.AMBIGUOUS_OVERLAP:
                # hips inside the bed box in 2-D, but most of the body is not:
                # could be standing BESIDE the bed. Let the agent check.
                flags.append("ambiguous_sit_or_stand")
                shape_margin = 0.0
        base = 0.55
    else:
        if horizontal:
            state, why = UNKNOWN, "horizontal body OUTSIDE the bed"
            flags.append("possible_fall")
        elif bent_legs and not moving:
            state, why = SIT_OUT, f"hips off bed, knees bent ({knee:.0f} deg)"
        elif moving:
            state, why = WALK, f"hips off bed, moving {motion:.2f} body-heights/s"
        else:
            state, why = STAND, "hips off bed, upright, not moving"
        base = 0.55

    # 5. confidence 
    conf = base + 0.25 * kp + 0.2 * _clip(shape_margin)
    if not upright and not horizontal:
        conf -= 0.1                                    # in-between posture
    if bright < C.DARK_BRIGHTNESS:
        conf *= 0.75
        flags.append("low_light")
    return Observation(t, state, round(_clip(conf), 2), why, flags, f)