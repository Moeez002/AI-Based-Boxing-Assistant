"""

Metrics:
  1. Peak Extension (Elbow Angle)        - Jab, Cross only
  2. Guard Hand Drop (DTW frame-wise)    - All punches
  3. Punch Duration                      - All punches
  4. Retraction Speed                    - Jab, Cross only
  5. Shoulder Rotation (Total Arc)       - All punches
  6. Hip Rotation (Total Arc)            - All punches
  7. Arm Compactness (Wrist-Shoulder)    - Lead Hook, Rear Hook, Lead Uppercut, Rear Uppercut
  8. Hook Elbow Height                   - Lead Hook, Rear Hook only
  9. Foot Stance Width (Pre-punch)       - All punches
"""

import numpy as np
from dtaidistance import dtw as dtw1d


# ─────────────────────────────────────────────
# FEATURE INDEX CONSTANTS
# ─────────────────────────────────────────────

# Normalized coordinate indices (0-31)
LW_X, LW_Y = 8,  9    # Left wrist
RW_X, RW_Y = 10, 11   # Right wrist
LS_X, LS_Y = 0,  1    # Left shoulder
RS_X, RS_Y = 2,  3    # Right shoulder
LE_X, LE_Y = 4,  5    # Left elbow
RE_X, RE_Y = 6,  7    # Right elbow
LH_X, LH_Y = 12, 13   # Left hip
RH_X, RH_Y = 14, 15   # Right hip

# Angles (44-49)
L_ELBOW_ANGLE = 44
R_ELBOW_ANGLE = 45

# Wrist speeds (50-51)
LW_SPEED = 50
RW_SPEED = 51

# Wrist-shoulder distances (52-53)
LW_SHLDR_DIST = 52
RW_SHLDR_DIST = 53

# Punch type groupings
STRAIGHT_PUNCHES = {"jab", "cross"}
HOOK_PUNCHES     = {"lead_hook", "rear_hook"}
UPPERCUT_PUNCHES = {"lead_uppercut", "rear_uppercut"}
CURVED_PUNCHES   = HOOK_PUNCHES | UPPERCUT_PUNCHES   # hooks + uppercuts

THRESHOLDS = {
    # Metric 1 - Peak Extension (straight punches only)
    "elbow_diff":              8.0,   # degrees - ref minus user peak elbow
    "jab_min_elbow":           165.0, # minimum acceptable peak elbow for jab
    "cross_min_elbow":         165.0, # minimum acceptable peak elbow for cross

    # Metric 2 - Guard Hand
    "guard_avg_diff":          0.06,
    "guard_max_diff":          0.10,

    # Metric 3 - Duration
    "duration_ratio_slow":     1.25,
    "duration_ratio_fast":     0.75,

    # Metric 4 - Retraction (straight punches only)
    "retraction_ratio":        0.80,

    # Metric 5 - Shoulder Rotation
    "shoulder_arc_ratio":      0.80,
    "shoulder_arc_min_signal": 0.05,

    # Metric 6 - Hip Rotation
    "hip_arc_ratio":           0.80,
    "hip_arc_min_signal":      0.04,

    # Metric 7 - Arm Compactness (curved punches only)
    "compactness_window":            10,    # frames each side of peak wrist speed
    "compactness_mae_threshold":     0.05,  # normalized units - below this, profiles are "close enough"
    "compactness_signed_threshold":  0.03,  # normalized units - minimum consistent bias to call a direction

    # Metric 8 - Hook Elbow Height (hooks only)
    "elbow_height_diff":       0.08,  # normalized units - user elbow lower than ref
    "elbow_height_high_diff":  0.10,  # normalized units - user elbow much higher than ref

    # Metric 9 - Foot Stance Width
    "stance_ratio_wide":       1.30,  # user stance > 1.30x ref = too wide
    "stance_ratio_narrow":     0.70,  # user stance < 0.70x ref = too narrow
    "stance_frames":           10,    # number of pre-punch frames to average over
}

RETRACTION_SPAN = 8


# ─────────────────────────────────────────────
# METRIC HELPERS
# ─────────────────────────────────────────────

def get_shoulder_width(frame):
    ls = np.array([frame[LS_X], frame[LS_Y]])
    rs = np.array([frame[RS_X], frame[RS_Y]])
    return float(np.linalg.norm(rs - ls))


def get_hip_width(frame):
    lh = np.array([frame[LH_X], frame[LH_Y]])
    rh = np.array([frame[RH_X], frame[RH_Y]])
    return float(np.linalg.norm(rh - lh))


def compute_total_arc(seq, width_fn):
    """
    Sum of absolute frame-to-frame changes in width.
    No cancellation between positive/negative movements, so it works for
    hooks/uppercuts where rotation reverses direction.
    """
    total = 0.0
    for i in range(1, len(seq)):
        total += abs(width_fn(seq[i]) - width_fn(seq[i - 1]))
    return total


def compute_stance_width(ankle_data, n_frames=10):
    """
    Averages ankle distance over the first n_frames pre-punch frames.
    ankle_data: list of (lax, lay, rax, ray) coords.
    """
    n = min(n_frames, len(ankle_data))
    if n == 0:
        return 0.0
    widths = []
    for i in range(n):
        left_ankle = np.array([ankle_data[i][0], ankle_data[i][1]])
        right_ankle = np.array([ankle_data[i][2], ankle_data[i][3]])
        widths.append(float(np.linalg.norm(right_ankle - left_ankle)))
    return float(np.mean(widths))


def get_delivery_phase_window(arr, speed_idx, half_window):
    """
    Returns (start, end, peak_idx) - a window of frames centered on peak
    wrist speed (the delivery/impact moment), clipped to sequence bounds.
    """
    peak_idx = int(np.argmax(arr[:, speed_idx]))
    start = max(0, peak_idx - half_window)
    end = min(len(arr), peak_idx + half_window + 1)
    return start, end, peak_idx


# ─────────────────────────────────────────────
# FEEDBACK ENGINE
# ─────────────────────────────────────────────

def compute_feedback(user_seq, ref_seq, path, punch_type,
                     user_ankle_data, ref_ankle_data, log_lines):
    punch_type = punch_type.lower()
    is_lead = punch_type in {"jab", "lead_hook", "lead_uppercut"}

    elbow_idx   = L_ELBOW_ANGLE if is_lead else R_ELBOW_ANGLE
    speed_idx   = LW_SPEED      if is_lead else RW_SPEED
    guard_y_idx = RW_Y          if is_lead else LW_Y

    # Elbow and shoulder y indices for the hook elbow height metric
    elbow_y_idx    = LE_Y if is_lead else RE_Y
    shoulder_y_idx = LS_Y if is_lead else RS_Y

    user_arr = np.array(user_seq)
    ref_arr  = np.array(ref_seq)

    feedback = []
    log_lines.append("=" * 60)
    log_lines.append(f"PUNCH TYPE : {punch_type.upper()}")
    log_lines.append(f"USER FRAMES: {len(user_arr)}  |  REF FRAMES: {len(ref_arr)}")
    log_lines.append("=" * 60)

    # ── METRIC 1: Peak Extension (Straight punches only) ──────────
    log_lines.append("\n[METRIC 1] Peak Extension (Elbow Angle)")
    if punch_type not in STRAIGHT_PUNCHES:
        log_lines.append(f"  SKIPPED - not applicable for {punch_type}")
    else:
        elbow_threshold = THRESHOLDS.get(f"{punch_type}_min_elbow", 165.0)

        user_peak_elbow = float(np.max(user_arr[:, elbow_idx]))
        ref_peak_elbow  = float(np.max(ref_arr[:,  elbow_idx]))
        elbow_diff      = ref_peak_elbow - user_peak_elbow

        log_lines.append(f"  User peak elbow : {user_peak_elbow:.1f} deg")
        log_lines.append(f"  Ref  peak elbow : {ref_peak_elbow:.1f} deg")
        log_lines.append(f"  Difference      : {elbow_diff:.1f} deg")
        log_lines.append(f"  Threshold diff  : {THRESHOLDS['elbow_diff']} deg")
        log_lines.append(f"  Min elbow needed: {elbow_threshold} deg")

        if elbow_diff > THRESHOLDS["elbow_diff"] and user_peak_elbow < elbow_threshold:
            feedback.append("Extend your elbow more during the punch")
            log_lines.append("  -> FEEDBACK FIRED")
        else:
            log_lines.append("  -> OK")

    # ── METRIC 2: Guard Hand (DTW frame-wise) ─────────────────────
    guard_diffs = []
    for (u_idx, r_idx) in path:
        if u_idx < len(user_arr) and r_idx < len(ref_arr):
            guard_diffs.append(
                float(user_arr[u_idx, guard_y_idx]) - float(ref_arr[r_idx, guard_y_idx])
            )

    guard_avg = float(np.mean(guard_diffs)) if guard_diffs else 0.0
    guard_max = float(np.max(guard_diffs))  if guard_diffs else 0.0

    log_lines.append("\n[METRIC 2] Guard Hand (DTW frame-wise)")
    log_lines.append(f"  Avg guard drop  : {guard_avg:.4f}")
    log_lines.append(f"  Max guard drop  : {guard_max:.4f}")
    log_lines.append(f"  Threshold avg   : {THRESHOLDS['guard_avg_diff']}")
    log_lines.append(f"  Threshold max   : {THRESHOLDS['guard_max_diff']}")

    if guard_avg > THRESHOLDS["guard_avg_diff"] or guard_max > THRESHOLDS["guard_max_diff"]:
        feedback.append("Keep your guard hand up while punching")
        log_lines.append("  -> FEEDBACK FIRED")
    else:
        log_lines.append("  -> OK")

    # ── METRIC 3: Punch Duration ───────────────────────────────────
    dur_ratio = len(user_arr) / len(ref_arr) if len(ref_arr) > 0 else 1.0

    log_lines.append("\n[METRIC 3] Punch Duration")
    log_lines.append(f"  User frames    : {len(user_arr)}")
    log_lines.append(f"  Ref  frames    : {len(ref_arr)}")
    log_lines.append(f"  Duration ratio : {dur_ratio:.3f}")
    log_lines.append(f"  Slow threshold : {THRESHOLDS['duration_ratio_slow']}")
    log_lines.append(f"  Fast threshold : {THRESHOLDS['duration_ratio_fast']}")

    if dur_ratio > THRESHOLDS["duration_ratio_slow"]:
        feedback.append("Increase your punch speed")
        log_lines.append("  -> FEEDBACK FIRED (too slow)")
    elif dur_ratio < THRESHOLDS["duration_ratio_fast"]:
        feedback.append("Slow down slightly - focus on full extension")
        log_lines.append("  -> FEEDBACK FIRED (too fast)")
    else:
        log_lines.append("  -> OK")

    # ── METRIC 4: Retraction Speed (Straight punches only) ────────
    log_lines.append("\n[METRIC 4] Retraction Speed")
    if punch_type not in STRAIGHT_PUNCHES:
        log_lines.append(f"  SKIPPED - not applicable for {punch_type}")
    else:
        u_peak_idx = int(np.argmax(user_arr[:, elbow_idx]))
        r_peak_idx = int(np.argmax(ref_arr[:,  elbow_idx]))

        u_post = user_arr[u_peak_idx:u_peak_idx + RETRACTION_SPAN, speed_idx]
        r_post = ref_arr[r_peak_idx:r_peak_idx + RETRACTION_SPAN,  speed_idx]

        if len(u_post) > 0 and len(r_post) > 0:
            u_ret = float(np.mean(u_post))
            r_ret = float(np.mean(r_post))
            ret_ratio = u_ret / r_ret if r_ret > 1e-6 else 1.0

            log_lines.append(f"  User peak frame       : {u_peak_idx}")
            log_lines.append(f"  Ref  peak frame       : {r_peak_idx}")
            log_lines.append(f"  User retraction speed : {u_ret:.4f}")
            log_lines.append(f"  Ref  retraction speed : {r_ret:.4f}")
            log_lines.append(f"  Ratio (user/ref)      : {ret_ratio:.3f}")
            log_lines.append(f"  Threshold             : {THRESHOLDS['retraction_ratio']}")

            if r_ret > 1e-6 and ret_ratio < THRESHOLDS["retraction_ratio"]:
                feedback.append("Retract your hand faster after impact")
                log_lines.append("  -> FEEDBACK FIRED")
            else:
                log_lines.append("  -> OK")
        else:
            log_lines.append("  Not enough post-peak frames to measure")

    # ── METRIC 5: Shoulder Rotation (Total Arc) ───────────────────
    user_shoulder_arc  = compute_total_arc(user_arr, get_shoulder_width)
    ref_shoulder_arc   = compute_total_arc(ref_arr,  get_shoulder_width)
    shoulder_arc_ratio = user_shoulder_arc / ref_shoulder_arc if ref_shoulder_arc > 1e-6 else 1.0

    log_lines.append("\n[METRIC 5] Shoulder Rotation (Total Arc Method)")
    log_lines.append(f"  User shoulder arc    : {user_shoulder_arc:.4f}")
    log_lines.append(f"  Ref  shoulder arc    : {ref_shoulder_arc:.4f}")
    log_lines.append(f"  Arc ratio (user/ref) : {shoulder_arc_ratio:.3f}")
    log_lines.append(f"  Min ref arc signal   : {THRESHOLDS['shoulder_arc_min_signal']}")
    log_lines.append(f"  Fire threshold ratio : {THRESHOLDS['shoulder_arc_ratio']}")

    if ref_shoulder_arc < THRESHOLDS["shoulder_arc_min_signal"]:
        log_lines.append("  SKIPPED - reference shoulder arc too small")
    elif shoulder_arc_ratio < THRESHOLDS["shoulder_arc_ratio"]:
        feedback.append("Rotate your shoulders more to generate power")
        log_lines.append("  -> FEEDBACK FIRED")
    else:
        log_lines.append("  -> OK")

    # ── METRIC 6: Hip Rotation (Total Arc) ────────────────────────
    user_hip_arc  = compute_total_arc(user_arr, get_hip_width)
    ref_hip_arc   = compute_total_arc(ref_arr,  get_hip_width)
    hip_arc_ratio = user_hip_arc / ref_hip_arc if ref_hip_arc > 1e-6 else 1.0

    log_lines.append("\n[METRIC 6] Hip Rotation (Total Arc Method)")
    log_lines.append(f"  User hip arc         : {user_hip_arc:.4f}")
    log_lines.append(f"  Ref  hip arc         : {ref_hip_arc:.4f}")
    log_lines.append(f"  Arc ratio (user/ref) : {hip_arc_ratio:.3f}")
    log_lines.append(f"  Min ref arc signal   : {THRESHOLDS['hip_arc_min_signal']}")
    log_lines.append(f"  Fire threshold ratio : {THRESHOLDS['hip_arc_ratio']}")

    if ref_hip_arc < THRESHOLDS["hip_arc_min_signal"]:
        log_lines.append("  SKIPPED - reference hip arc too small")
    elif hip_arc_ratio < THRESHOLDS["hip_arc_ratio"]:
        feedback.append("Rotate your hips more to generate greater power")
        log_lines.append("  -> FEEDBACK FIRED")
    else:
        log_lines.append("  -> OK")

    # ── METRIC 7: Arm Compactness (Curved punches only) ────────────
    # Delivery-phase, DTW-aligned wrist-to-shoulder distance profile:
    #  - window both clips around peak wrist speed (the delivery moment)
    #  - DTW-align the two 1-D distance profiles
    #  - mean absolute error says IF they differ, mean signed error says WHICH WAY
    log_lines.append("\n[METRIC 7] Arm Compactness (Wrist-Shoulder Distance)")
    if punch_type not in CURVED_PUNCHES:
        log_lines.append(f"  SKIPPED - not applicable for {punch_type}")
    else:
        dist_idx = LW_SHLDR_DIST if is_lead else RW_SHLDR_DIST
        half_win = int(THRESHOLDS["compactness_window"])

        u_start, u_end, u_peak = get_delivery_phase_window(user_arr, speed_idx, half_win)
        r_start, r_end, r_peak = get_delivery_phase_window(ref_arr,  speed_idx, half_win)

        u_series = user_arr[u_start:u_end, dist_idx].astype(np.double)
        r_series = ref_arr[r_start:r_end,  dist_idx].astype(np.double)

        log_lines.append(f"  User peak-speed frame : {u_peak}  (window {u_start}-{u_end - 1})")
        log_lines.append(f"  Ref  peak-speed frame : {r_peak}  (window {r_start}-{r_end - 1})")

        if len(u_series) < 2 or len(r_series) < 2:
            log_lines.append("  SKIPPED - not enough frames in delivery-phase window")
        else:
            local_path = dtw1d.warping_path(u_series, r_series)
            errors = np.array([u_series[ui] - r_series[ri] for (ui, ri) in local_path])
            mae = float(np.mean(np.abs(errors)))
            mean_signed = float(np.mean(errors))

            log_lines.append(f"  Mean absolute error   : {mae:.4f}")
            log_lines.append(f"  Mean signed error     : {mean_signed:.4f}  (+ = user more open, - = user more compact)")
            log_lines.append(f"  MAE threshold         : {THRESHOLDS['compactness_mae_threshold']}")
            log_lines.append(f"  Signed threshold      : {THRESHOLDS['compactness_signed_threshold']}")

            if mae < THRESHOLDS["compactness_mae_threshold"]:
                log_lines.append("  -> OK (delivery-phase profiles closely matched)")
            elif mean_signed > THRESHOLDS["compactness_signed_threshold"]:
                feedback.append("Keep your punch compact - avoid letting your arm flare outward")
                log_lines.append("  -> FEEDBACK FIRED (user consistently more open than reference)")
            elif mean_signed < -THRESHOLDS["compactness_signed_threshold"]:
                feedback.append("Allow your arm to open more and extend through the punch")
                log_lines.append("  -> FEEDBACK FIRED (user consistently more compact than reference)")
            else:
                log_lines.append("  -> OK (difference not directionally consistent - likely timing noise)")

    # ── METRIC 8: Hook Elbow Height (Hooks only) ──────────────────
    # At the peak wrist speed frame (impact), compare elbow y relative to shoulder y.
    log_lines.append("\n[METRIC 8] Hook Elbow Height")
    if punch_type not in HOOK_PUNCHES:
        log_lines.append(f"  SKIPPED - not applicable for {punch_type}")
    else:
        u_impact_idx = int(np.argmax(user_arr[:, speed_idx]))
        r_impact_idx = int(np.argmax(ref_arr[:,  speed_idx]))

        u_elbow_rel = float(user_arr[u_impact_idx, elbow_y_idx]) - \
                      float(user_arr[u_impact_idx, shoulder_y_idx])
        r_elbow_rel = float(ref_arr[r_impact_idx, elbow_y_idx]) - \
                      float(ref_arr[r_impact_idx, shoulder_y_idx])

        # Positive = user elbow LOWER than reference, negative = HIGHER
        height_diff = u_elbow_rel - r_elbow_rel

        log_lines.append(f"  User impact frame         : {u_impact_idx}")
        log_lines.append(f"  Ref  impact frame         : {r_impact_idx}")
        log_lines.append(f"  User elbow rel to shoulder: {u_elbow_rel:.4f}")
        log_lines.append(f"  Ref  elbow rel to shoulder: {r_elbow_rel:.4f}")
        log_lines.append(f"  Difference (user - ref)   : {height_diff:.4f}")
        log_lines.append(f"  Low threshold             : {THRESHOLDS['elbow_height_diff']}")
        log_lines.append(f"  High threshold            : {THRESHOLDS['elbow_height_high_diff']}")

        if height_diff > THRESHOLDS["elbow_height_diff"]:
            feedback.append("Keep your elbow higher during the hook")
            log_lines.append("  -> FEEDBACK FIRED (elbow too low)")
        elif height_diff < -THRESHOLDS["elbow_height_high_diff"]:
            feedback.append("Maintain a natural elbow position - avoid exaggerating the hook")
            log_lines.append("  -> FEEDBACK FIRED (elbow too high)")
        else:
            log_lines.append("  -> OK")

    # ── METRIC 9: Foot Stance Width (Pre-punch) ───────────────────
    # Averages ankle distance over the first N frames, normalized by hip width
    # to remove body size variation.
    log_lines.append("\n[METRIC 9] Foot Stance Width (Pre-punch)")
    if len(user_ankle_data) == 0 or len(ref_ankle_data) == 0:
        log_lines.append("  SKIPPED - ankle data unavailable")
    else:
        n_frames = THRESHOLDS["stance_frames"]

        user_raw_stance = compute_stance_width(user_ankle_data, n_frames)
        ref_raw_stance  = compute_stance_width(ref_ankle_data,  n_frames)

        user_hip_w = get_hip_width(user_arr[0])
        ref_hip_w  = get_hip_width(ref_arr[0])

        user_norm_stance = user_raw_stance / user_hip_w if user_hip_w > 1e-6 else user_raw_stance
        ref_norm_stance  = ref_raw_stance  / ref_hip_w  if ref_hip_w  > 1e-6 else ref_raw_stance

        stance_ratio = user_norm_stance / ref_norm_stance if ref_norm_stance > 1e-6 else 1.0

        log_lines.append(f"  User raw ankle dist      : {user_raw_stance:.4f}")
        log_lines.append(f"  Ref  raw ankle dist      : {ref_raw_stance:.4f}")
        log_lines.append(f"  User hip width           : {user_hip_w:.4f}")
        log_lines.append(f"  Ref  hip width           : {ref_hip_w:.4f}")
        log_lines.append(f"  User normalized stance   : {user_norm_stance:.4f}")
        log_lines.append(f"  Ref  normalized stance   : {ref_norm_stance:.4f}")
        log_lines.append(f"  Stance ratio (user/ref)  : {stance_ratio:.3f}")
        log_lines.append(f"  Wide threshold           : {THRESHOLDS['stance_ratio_wide']}")
        log_lines.append(f"  Narrow threshold         : {THRESHOLDS['stance_ratio_narrow']}")

        if stance_ratio > THRESHOLDS["stance_ratio_wide"]:
            feedback.append("Reduce your stance width slightly to maintain mobility")
            log_lines.append("  -> FEEDBACK FIRED (stance too wide)")
        elif stance_ratio < THRESHOLDS["stance_ratio_narrow"]:
            feedback.append("Widen your stance slightly for a stronger base")
            log_lines.append("  -> FEEDBACK FIRED (stance too narrow)")
        else:
            log_lines.append("  -> OK")

    # ── Summary ───────────────────────────────────────────────────
    log_lines.append("\n" + "=" * 60)
    if not feedback:
        feedback.append("Excellent form! Great execution.")
        log_lines.append("RESULT: No issues detected - Excellent form.")
    else:
        log_lines.append(f"RESULT: {len(feedback)} feedback point(s) generated:")
        for msg in feedback:
            log_lines.append(f"  >> {msg}")
    log_lines.append("=" * 60)

    return feedback
