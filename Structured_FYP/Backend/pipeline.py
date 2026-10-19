"""

Punch detection pipeline used by the FastAPI backend:
  video -> MediaPipe pose -> punch trigger -> clip crop -> 3-model hierarchical
  classifier -> DTW feedback against reference punches -> exported clip.

Public functions:
  get_models()                              load the 3 models once (cached)
  run_punch_detection(video_path, out_dir)  process one video, return list of punches
"""

import glob
import os
import pickle

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Masking
from dtaidistance import dtw_ndim

from feedback_analyzer import compute_feedback


# ==========================================
# 1. CONFIGURATION
# ==========================================
REFERENCE_ROOT = r"C:\Users\abdul\OneDrive\Desktop\AI BOXING ASSISTANT\goldenStandardPunches"

MODEL1_PATH = r"C:\Users\abdul\Downloads\model_fixed.h5"    # jab | leadhook | cross_rearhook | uppercuts_nopunch
MODEL2_PATH = r"C:\Users\abdul\Downloads\model2_fixed.h5"   # leaduppercut | rearuppercut | nopunch
MODEL3_PATH = r"C:\Users\abdul\Downloads\model3_fixed.h5"   # cross | rearhook

# Set False to skip reference lookup + DTW + feedback text (clips are still exported)
ENABLE_FEEDBACK = True

# Trimming constants
VELOCITY_THRESHOLD = 0.008
PRE_PAD_LEAD, POST_PAD_LEAD = 12, 12
PRE_PAD_REAR, POST_PAD_REAR = 18, 20

MAX_FRAMES = 110            # time steps the models were trained with

# Trigger thresholds
LEAD_THRESHOLD = 0.15
REAR_THRESHOLD = 0.10
UPPERCUT_Y_THRESHOLD = 0.30

WINDOW_BEFORE, WINDOW_AFTER = 60, 60
TOTAL_WINDOW = WINDOW_BEFORE + WINDOW_AFTER + 1
MAX_HISTORY = 200         

# MediaPipe indexes removed after Z-removal (same as the training pipeline)
REMOVE_INDEXES = set([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45
])

# Model class mappings (index order MUST match training)
M1_CLASSES = {0: "Jab", 1: "Lead Hook", 2: "Cross/Rear Hook", 3: "Uppercut/NoPunch"}
M2_CLASSES = {0: "Lead Uppercut", 1: "Rear Uppercut", 2: "NoPunch"}
M3_CLASSES = {0: "Cross", 1: "Rear Hook"}

FEATURE_DIM = 67   # per-frame feature size all three models were trained with

# When Model 2 says NoPunch we revert to Model 1's answer. Model 1's merged
# class "Uppercut/NoPunch" can't be output as-is, so it becomes NoPunch.
M1_FALLBACK = {"Jab": "Jab", "Lead Hook": "Lead Hook", "Uppercut/NoPunch": "NoPunch"}
M1_TO_M3 = {"Cross/Rear Hook"}

LABEL_TO_FOLDER = {
    "Jab": "jab", "Cross": "cross", "Lead Hook": "leadhook", "Rear Hook": "rearhook",
    "Lead Uppercut": "leaduppercut", "Rear Uppercut": "rearuppercut",
}
LABEL_TO_PUNCH_TYPE = {
    "Jab": "jab", "Cross": "cross", "Lead Hook": "lead_hook", "Rear Hook": "rear_hook",
    "Lead Uppercut": "lead_uppercut", "Rear Uppercut": "rear_uppercut",
}

# Must match the string compute_feedback() appends when nothing fires
EXCELLENT_MSG = "Excellent form! Great execution."

_MODELS = None            # filled by get_models()
_REFERENCE_CACHE = {}     # punch label -> list of reference entries (or None)


# ==========================================
# 2. FEATURE ENGINEERING  (67-dim per-frame vector)
# ==========================================
LANDMARK_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

# Joint positions inside the 16-landmark list
LS, RS = 0, 1
LE, RE = 2, 3
LW, RW = 4, 5
LHIP, RHIP = 6, 7
LKNEE, RKNEE = 8, 9


def get_hip_center(frame):
    return np.array([(frame[LHIP * 2] + frame[RHIP * 2]) / 2.0,
                     (frame[LHIP * 2 + 1] + frame[RHIP * 2 + 1]) / 2.0])


def get_shoulder_center(frame):
    return np.array([(frame[LS * 2] + frame[RS * 2]) / 2.0,
                     (frame[LS * 2 + 1] + frame[RS * 2 + 1]) / 2.0])


def translation_normalization(frame):
    hip_center = get_hip_center(frame)
    normalized = []
    for i in range(0, len(frame), 2):
        joint_xy = np.array([frame[i], frame[i + 1]]) - hip_center
        normalized.extend(joint_xy.tolist())
    return normalized


def scale_normalization(frame):
    torso = np.linalg.norm(get_shoulder_center(frame) - get_hip_center(frame))
    if torso < 1e-6:
        torso = 1e-6
    return [v / torso for v in frame]


def get_xy(frame, joint):
    return [frame[joint * 2], frame[joint * 2 + 1]]


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = abs((180.0 * radians) / np.pi)
    if angle > 180.0:
        return 360.0 - angle
    return angle


def extract_angles(frame):
    return [
        calculate_angle(get_xy(frame, LS), get_xy(frame, LE), get_xy(frame, LW)),
        calculate_angle(get_xy(frame, RS), get_xy(frame, RE), get_xy(frame, RW)),
        calculate_angle(get_xy(frame, LHIP), get_xy(frame, LS), get_xy(frame, LE)),
        calculate_angle(get_xy(frame, RHIP), get_xy(frame, RS), get_xy(frame, RE)),
        calculate_angle(get_xy(frame, LS), get_xy(frame, LHIP), get_xy(frame, LKNEE)),
        calculate_angle(get_xy(frame, RS), get_xy(frame, RHIP), get_xy(frame, RKNEE)),
    ]


def extract_velocities(curr_frame, prev_frame):
    velocities = []
    for joint in [LS, RS, LE, RE, LW, RW]:
        velocities.append(curr_frame[joint * 2] - prev_frame[joint * 2])
        velocities.append(curr_frame[joint * 2 + 1] - prev_frame[joint * 2 + 1])
    return velocities


def extract_wrist_speeds(velocities):
    left = np.sqrt(velocities[8] ** 2 + velocities[9] ** 2)
    right = np.sqrt(velocities[10] ** 2 + velocities[11] ** 2)
    return [left, right]


def wrist_to_shoulder_distance(frame):
    left = np.linalg.norm(np.array(get_xy(frame, LW)) - np.array(get_xy(frame, LS)))
    right = np.linalg.norm(np.array(get_xy(frame, RW)) - np.array(get_xy(frame, RS)))
    return [left, right]


def lead_rear_asymmetry(frame):
    return [frame[LW * 2] - frame[RW * 2]]


def wrist_motion_angle(velocities):
    return [np.arctan2(velocities[9], velocities[8]), np.arctan2(velocities[11], velocities[10])]


def cumulative_wrist_displacement(curr_frame, frame0):
    lw_x0, lw_y0 = get_xy(frame0, LW)
    rw_x0, rw_y0 = get_xy(frame0, RW)
    lw_x, lw_y = get_xy(curr_frame, LW)
    rw_x, rw_y = get_xy(curr_frame, RW)
    vertical = [lw_y - lw_y0, rw_y - rw_y0]
    horizontal = [lw_x - lw_x0, rw_x - rw_x0]
    return vertical, horizontal


def vertical_horizontal_ratio(vert_disp, horiz_disp):
    return [abs(vert_disp[0]) / (abs(horiz_disp[0]) + 1e-6),
            abs(vert_disp[1]) / (abs(horiz_disp[1]) + 1e-6)]


def elbow_shoulder_height(frame):
    return [frame[LE * 2 + 1] - frame[LS * 2 + 1], frame[RE * 2 + 1] - frame[RS * 2 + 1]]


def elbow_torso_offset(frame):
    hip_center_x = (frame[LHIP * 2] + frame[RHIP * 2]) / 2.0
    return [frame[LE * 2] - hip_center_x, frame[RE * 2] - hip_center_x]


def build_feature_vector(curr_frame, prev_frame, frame0):
    angles = extract_angles(curr_frame)
    velocities = extract_velocities(curr_frame, prev_frame)
    speeds = extract_wrist_speeds(velocities)
    wrist_dist = wrist_to_shoulder_distance(curr_frame)
    asymmetry = lead_rear_asymmetry(curr_frame)
    motion_angle = wrist_motion_angle(velocities)
    vert_disp, horiz_disp = cumulative_wrist_displacement(curr_frame, frame0)
    vh_ratio = vertical_horizontal_ratio(vert_disp, horiz_disp)
    return np.concatenate([
        curr_frame, velocities, angles, speeds, wrist_dist, asymmetry,
        motion_angle, vert_disp, horiz_disp, vh_ratio,
        elbow_shoulder_height(curr_frame), elbow_torso_offset(curr_frame)
    ])


def process_punch_sequence(punch):
    """punch: list of 32-dim normalized frames for ONE isolated clip.
    Returns a list of 67-dim feature vectors, anchored to the first frame (matches training)."""
    if len(punch) == 0:
        return []

    frame0 = punch[0]
    zero_vel = [0.0] * 12
    zero_speed = [0.0, 0.0]
    zero_asym = [0.0]

    # First frame is special: no previous frame, so velocity / speed / asymmetry are zero
    vert0, horiz0 = cumulative_wrist_displacement(frame0, frame0)
    features = [np.concatenate([
        frame0, zero_vel, extract_angles(frame0), zero_speed,
        wrist_to_shoulder_distance(frame0), zero_asym,
        wrist_motion_angle(zero_vel), vert0, horiz0, vertical_horizontal_ratio(vert0, horiz0),
        elbow_shoulder_height(frame0), elbow_torso_offset(frame0)
    ])]

    for i in range(1, len(punch)):
        features.append(build_feature_vector(punch[i], punch[i - 1], frame0))
    return features


# ==========================================
# 3. MODEL LOADING
# ==========================================
def build_model(num_classes, feature_dim, max_frames=MAX_FRAMES):
    """Exact architecture used in training (same for all three models)."""
    return Sequential([
        Input(shape=(max_frames, feature_dim)),
        Masking(mask_value=0.0),
        LSTM(64, return_sequences=True),
        Dropout(0.4),
        LSTM(32, return_sequences=False),
        Dropout(0.4),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax'),
    ])


def load_punch_model(path, num_classes):
    """Try a normal load_model first; if that fails, rebuild the architecture and load the weights."""
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception as e:
        print(f"  load_model failed ({type(e).__name__}); rebuilding architecture + loading weights")
        model = build_model(num_classes, FEATURE_DIM)
        model.load_weights(path)
        return model


def run_model(model, x):
    return np.asarray(model(x, training=False))[0]


def get_models():
    global _MODELS
    if _MODELS is not None:
        return _MODELS

    specs = [
        ("M1", MODEL1_PATH, M1_CLASSES),
        ("M2", MODEL2_PATH, M2_CLASSES),
        ("M3", MODEL3_PATH, M3_CLASSES),
    ]
    loaded = []
    for name, path, classes in specs:
        print(f"Loading {name}: {path}")
        model = load_punch_model(path, len(classes))
        if model.output_shape[-1] != len(classes):
            raise ValueError(f"{name}: model has {model.output_shape[-1]} outputs "
                             f"but {len(classes)} classes are mapped")
        time_steps = model.input_shape[1] or MAX_FRAMES
        run_model(model, np.zeros((1, time_steps, model.input_shape[-1]), np.float32))
        loaded.append(model)

    _MODELS = tuple(loaded)
    print("All models loaded and warmed up.")
    return _MODELS


def prepare_model_input(clip_features, model):
    """Take the leading F features (F = model's input size), then pad/trim to the
    model's time dimension. Returns shape (1, T, F) float32."""
    t_dim = model.input_shape[1] or MAX_FRAMES
    f_dim = model.input_shape[-1]
    seq = np.array(clip_features)[:, :f_dim]
    frames = seq.shape[0]
    if frames > t_dim:
        seq = seq[:t_dim]
    elif frames < t_dim:
        seq = np.vstack([seq, np.zeros((t_dim - frames, f_dim))])
    return np.expand_dims(seq, axis=0).astype(np.float32)


# ==========================================
# 4. HIERARCHICAL CLASSIFICATION
#    M1 == cross_rearhook -> M3 decides (Cross / Rear Hook)
#    M1 == anything else  -> M2 decides; M2 == NoPunch -> revert to M1
# ==========================================
def predict_hierarchical(clip_features, model1, model2, model3):
    """Returns (final_label, model1_label, route, confidence)."""
    p1 = run_model(model1, prepare_model_input(clip_features, model1))
    m1_label = M1_CLASSES[int(np.argmax(p1))]

    if m1_label in M1_TO_M3:
        p3 = run_model(model3, prepare_model_input(clip_features, model3))
        return M3_CLASSES[int(np.argmax(p3))], m1_label, "M3", float(np.max(p3))

    p2 = run_model(model2, prepare_model_input(clip_features, model2))
    m2_label = M2_CLASSES[int(np.argmax(p2))]

    if m2_label == "NoPunch":
        return M1_FALLBACK.get(m1_label, m1_label), m1_label, "M1-fallback", float(np.max(p1))
    return m2_label, m1_label, "M2", float(np.max(p2))


# ==========================================
# 5. REFERENCE PUNCHES FROM .pkl
#
#   Each .pkl holds sequences in the same raw format as the training dataset
#   (frames of 99 MediaPipe values: x,y,z for 33 landmarks). Every sequence goes
#   through: drop z -> drop noisy joints -> translation/scale normalisation ->
#   process_punch_sequence -> first 55 features. That is the same recipe the
#   user's clip goes through, so features line up for DTW + compute_feedback.
#   Raw ankle coordinates (before normalisation) are kept for the stance metric.
# ==========================================
def clean_raw_sequence(seq):
    """99 values/frame (x,y,z), 66 (x,y) or already-cleaned 32 -> list of 32-value frames."""
    cleaned = []
    for frame in seq:
        vals = np.asarray(frame, dtype=np.float64).ravel().tolist()
        if len(vals) == 99:
            vals = [v for i, v in enumerate(vals) if i % 3 != 2]     # drop z
        if len(vals) == 66:
            vals = [v for i, v in enumerate(vals) if i not in REMOVE_INDEXES]
        if len(vals) != 32:
            raise ValueError(f"unexpected frame length {len(vals)} (expected 99, 66 or 32)")
        cleaned.append(vals)
    return cleaned


def load_reference_sequences(pkl_path):
    """Returns a list of raw sequences from one .pkl file."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        data = list(data.values())

    if isinstance(data, np.ndarray):
        if data.ndim == 3:
            return list(data)
        return [data]

    if isinstance(data, (list, tuple)):
        # a list of frames (single sequence) vs. a list of sequences
        if len(data) > 0 and np.asarray(data[0], dtype=np.float64).ndim == 1:
            return [data]
        return data

    raise ValueError(f"unsupported pickle content type: {type(data)}")


def build_reference_entry(raw_seq, source):
    cleaned = clean_raw_sequence(raw_seq)

    # raw (un-normalised) ankle coords: joints 10/11 = left/right ankle -> values 20..23
    ankle = [(f[20], f[21], f[22], f[23]) for f in cleaned]

    norm = [scale_normalization(translation_normalization(f)) for f in cleaned]
    features = process_punch_sequence(norm)
    return {
        "seq": [np.asarray(v)[:55] for v in features],
        "ankle": ankle,
        "source": source,
    }


def get_reference_data(predicted_label):
    """Returns a list of reference entries for this punch type, or None. Cached per label."""
    if predicted_label in _REFERENCE_CACHE:
        return _REFERENCE_CACHE[predicted_label]

    refs = []
    folder_name = LABEL_TO_FOLDER.get(predicted_label)

    if folder_name is not None:
        folder_path = os.path.join(REFERENCE_ROOT, folder_name)
        if not os.path.isdir(folder_path):
            print(f"[WARN] Reference folder not found: {folder_path}")
        else:
            pkl_files = sorted(glob.glob(os.path.join(folder_path, "**", "*.pkl"), recursive=True))
            if not pkl_files:
                print(f"[WARN] No .pkl reference files found in: {folder_path}")

            for pkl_path in pkl_files:
                try:
                    sequences = load_reference_sequences(pkl_path)
                except Exception as e:
                    print(f"[WARN] Could not read {pkl_path}: {e}")
                    continue

                for k, seq in enumerate(sequences):
                    if len(seq) < 2:
                        continue
                    source = f"{os.path.basename(pkl_path)}#{k}"
                    try:
                        refs.append(build_reference_entry(seq, source))
                    except Exception as e:
                        print(f"[WARN] Skipped {source}: {e}")

            if refs:
                print(f"[ref] {predicted_label}: {len(refs)} reference sequence(s) "
                      f"from {len(pkl_files)} file(s)")

    result = refs if refs else None
    _REFERENCE_CACHE[predicted_label] = result
    return result


# ==========================================
# 6. FEEDBACK ENSEMBLE (majority vote across ALL references)
#
#   compute_feedback() is run once per reference sequence. Every message that a
#   strict majority of references agree on is kept. Opposing messages (e.g.
#   "too slow" vs "too fast") are different strings, so they are voted on
#   separately and can't cancel each other out.
# ==========================================
def run_feedback_ensemble(user_seq_55, refs, punch_type, clip_ankle, window=20):
    """Returns (final_feedback, votes, n_refs_used, per_ref_logs)."""
    user_arr = np.array(user_seq_55, dtype=np.double)
    votes = {}                  # message -> number of references that fired it (insertion-ordered)
    per_ref_logs = []
    n_used = 0

    for ref in refs:
        ref_arr = np.array(ref["seq"], dtype=np.double)
        try:
            path = dtw_ndim.warping_path(user_arr, ref_arr, window=window)
        except Exception as e:
            print(f"[WARN] DTW failed against {ref['source']}: {e}")
            continue

        log = []
        feedback = compute_feedback(user_seq_55, ref["seq"], path, punch_type,
                                    clip_ankle, ref["ankle"], log)
        n_used += 1
        per_ref_logs.append((ref["source"], log))

        for msg in feedback:
            if msg == EXCELLENT_MSG:
                continue
            votes[msg] = votes.get(msg, 0) + 1

    if n_used == 0:
        return ["Could not compare against any reference."], {}, 0, per_ref_logs

    needed = n_used // 2 + 1    # strict majority (3 of 5)
    final = [msg for msg in votes if votes[msg] >= needed]
    if not final:
        final = [EXCELLENT_MSG]
    return final, votes, n_used, per_ref_logs


def write_feedback_log(log_path, votes, n_refs, per_ref_logs):
    needed = n_refs // 2 + 1
    lines = [f"Ensemble feedback: {n_refs} references, majority = {needed}", "Votes:"]
    for msg, count in sorted(votes.items(), key=lambda item: -item[1]):
        lines.append(f"  {count}/{n_refs}  {msg}")
    if not votes:
        lines.append("  (no reference fired any feedback)")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n\n")
        for source, log in per_ref_logs:
            f.write(f"##### REFERENCE: {source} #####\n")
            f.write("\n".join(log) + "\n\n")


def get_feedback(clip_features, predicted_punch, clip_ankle, log_path):
    """Returns the list of feedback strings for one classified punch."""
    user_seq_55 = [np.asarray(v)[:55] for v in clip_features]

    refs = get_reference_data(predicted_punch)
    if refs is None:
        return [f"No reference .pkl data available for '{predicted_punch}'."]
    if len(user_seq_55) < 2:
        return ["Clip too short to analyze."]

    punch_type = LABEL_TO_PUNCH_TYPE.get(predicted_punch, predicted_punch.lower())
    feedback, votes, n_refs, per_ref_logs = run_feedback_ensemble(
        user_seq_55, refs, punch_type, clip_ankle
    )
    write_feedback_log(log_path, votes, n_refs, per_ref_logs)
    return feedback


# ==========================================
# 7. PER-FRAME HELPERS + TRIGGER DETECTION
# ==========================================
def get_wrist_distances(landmarks):
    nose = np.array([landmarks[0].x, landmarks[0].y])
    l_wrist = np.array([landmarks[15].x, landmarks[15].y])
    r_wrist = np.array([landmarks[16].x, landmarks[16].y])
    return np.linalg.norm(l_wrist - nose), np.linalg.norm(r_wrist - nose)


def compute_streaming_metrics(norm, prev_norm):
    """Frame-to-frame values used ONLY for live trigger / boundary detection."""
    if prev_norm is None:
        velocities = [0.0] * 12
    else:
        velocities = extract_velocities(norm, prev_norm)
    lw_speed, rw_speed = extract_wrist_speeds(velocities)
    angles = extract_angles(norm)
    return lw_speed, rw_speed, angles[0], angles[1]


def build_frame_entry(frame, landmarks, prev_norm):
    """Collects everything we need to remember about one frame.
    Returns (entry_dict, norm) - norm is needed as prev_norm for the next frame."""
    dist_l, dist_r = get_wrist_distances(landmarks)

    raw_coords = []
    for i in LANDMARK_INDICES:
        raw_coords.extend([landmarks[i].x, landmarks[i].y])

    norm = scale_normalization(translation_normalization(raw_coords))
    lw_speed, rw_speed, l_elbow, r_elbow = compute_streaming_metrics(norm, prev_norm)

    entry = {
        "frame": frame.copy(),
        "norm": norm,
        "ankle": (landmarks[27].x, landmarks[27].y, landmarks[28].x, landmarks[28].y),
        "dist_l": dist_l,
        "dist_r": dist_r,
        "wrist_y": landmarks[15].y,
        "lw_speed": lw_speed,
        "rw_speed": rw_speed,
        "l_elbow": l_elbow,
        "r_elbow": r_elbow,
    }
    return entry, norm


def column(entries, key):
    """Pulls one field out of a list of frame entries."""
    return [e[key] for e in entries]


def detect_trigger(history):
    """Checks the frame in the middle of the window (WINDOW_AFTER frames ago).
    Returns "LEAD_HORIZONTAL", "REAR_HORIZONTAL", "LEAD_UPPERCUT_Y" or None."""
    idx = -(WINDOW_AFTER + 1)
    vals_l = column(history, "dist_l")
    vals_r = column(history, "dist_r")
    vals_y = column(history, "wrist_y")
    vals_lw_speed = column(history, "lw_speed")

    is_peak_l = vals_l[idx] >= max(vals_l[idx - 5:idx + 6]) * 0.97
    is_peak_r = vals_r[idx] >= max(vals_r[idx - 5:idx + 6]) * 0.97
    is_lead_peak = is_peak_l and vals_l[idx] > LEAD_THRESHOLD
    is_rear_peak = is_peak_r and vals_r[idx] > REAR_THRESHOLD

    neighbours_y = vals_y[idx - 7:idx + 8]
    val_y = vals_y[idx]
    is_y_minimum = len(neighbours_y) == 15 and val_y == min(neighbours_y)
    is_moving_up = len(neighbours_y) == 15 and val_y < vals_y[idx - 10]
    is_high_enough = val_y < UPPERCUT_Y_THRESHOLD
    has_major_rise = len(vals_y) >= abs(idx - 20) and (vals_y[idx - 20] - val_y) >= 0.08
    has_explosive_speed = vals_lw_speed[idx] > 0.04
    is_lead_uppercut = (is_y_minimum and is_moving_up and is_high_enough
                        and has_major_rise and has_explosive_speed)

    if is_lead_peak:
        return "LEAD_HORIZONTAL"
    if is_rear_peak:
        return "REAR_HORIZONTAL"
    if is_lead_uppercut:
        return "LEAD_UPPERCUT_Y"
    return None


# ==========================================
# 8. ADAPTIVE MOTION BOUNDARY SCANNER
# ==========================================
def find_adaptive_motion_boundaries(speeds, wrist_dists, elbow_angles, peak_pos, base_threshold=0.008):
    peak_speed = speeds[peak_pos]
    adaptive_threshold = max(0.003, min(base_threshold, peak_speed * 0.15))

    start = peak_pos
    while start > 0:
        window = speeds[max(0, start - 3):start]
        if len(window) > 0 and max(window) < adaptive_threshold:
            break
        start -= 1

    end = peak_pos
    while end < len(speeds) - 1:
        window = speeds[end + 1:min(len(speeds), end + 4)]
        if len(window) > 0 and max(window) < adaptive_threshold:
            if end + 3 < len(wrist_dists):
                dist_delta = abs(wrist_dists[end + 3] - wrist_dists[end])
                angle_delta = abs(elbow_angles[end + 3] - elbow_angles[end])
                if dist_delta < 0.01 and angle_delta < 3.0:
                    break
            else:
                break
        end += 1

    is_slow_punch = peak_speed < 0.035
    extra_post_pad = 12 if is_slow_punch else 0
    return start, end, extra_post_pad


def crop_punch_clip(history, trigger):
    """Returns the frame entries that make up the punch clip."""
    window = history[-TOTAL_WINDOW:]
    is_lead = "LEAD" in trigger
    is_rear = "REAR" in trigger

    if is_lead:
        speeds = column(window, "lw_speed")
        wrist_dists = column(window, "dist_l")
        elbow_angles = column(window, "l_elbow")
    else:
        speeds = column(window, "rw_speed")
        wrist_dists = column(window, "dist_r")
        elbow_angles = column(window, "r_elbow")

    motion_start, motion_end, extra_post_pad = find_adaptive_motion_boundaries(
        speeds, wrist_dists, elbow_angles, WINDOW_BEFORE, VELOCITY_THRESHOLD
    )

    pre_pad = PRE_PAD_REAR if is_rear else PRE_PAD_LEAD
    post_pad = (POST_PAD_REAR if is_rear else POST_PAD_LEAD) + extra_post_pad

    crop_start = max(0, motion_start - pre_pad)
    crop_end = min(TOTAL_WINDOW - 1, motion_end + post_pad)
    return window[crop_start:crop_end + 1]


def save_clip(frames, out_path, fps, width, height):
    """Plain cut clip: no drawing, no text."""
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for f in frames:
        writer.write(f)
    writer.release()


# ==========================================
# 9. MAIN ENTRY POINT (called by FastAPI)
# ==========================================
def run_punch_detection(video_path, output_dir):
    """Processes one video. Returns a list of dicts, one per detected clip:
       {punch_id, classification, confidence, clip_filename, feedback}
    'NoPunch' clips are included (main.py filters them out)."""
    model1, model2, model3 = get_models()
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    pose = mp.solutions.pose.Pose(min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5,
                                  model_complexity=1)

    history = []          # most recent frames (see MAX_HISTORY)
    results = []
    prev_norm = None
    cooldown = 0
    punch_counter = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            pose_result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if pose_result.pose_landmarks:
                entry, prev_norm = build_frame_entry(frame, pose_result.pose_landmarks.landmark, prev_norm)
                history.append(entry)
                if len(history) > MAX_HISTORY:
                    history.pop(0)

                if len(history) >= TOTAL_WINDOW and cooldown == 0:
                    trigger = detect_trigger(history)

                    if trigger is not None:
                        punch_counter += 1

                        # ---- crop clip + build features ----
                        clip = crop_punch_clip(history, trigger)
                        clip_norm = column(clip, "norm")
                        clip_ankle = column(clip, "ankle")
                        clip_features = process_punch_sequence(clip_norm)

                        # ---- classify ----
                        predicted_punch, m1_label, route, confidence = predict_hierarchical(
                            clip_features, model1, model2, model3
                        )

                        # ---- feedback (text only) ----
                        feedback = []
                        if ENABLE_FEEDBACK and predicted_punch != "NoPunch":
                            log_path = os.path.join(
                                output_dir, f"punch_{punch_counter}_{predicted_punch}_feedback.txt"
                            )
                            feedback = get_feedback(clip_features, predicted_punch, clip_ankle, log_path)

                        # ---- export clip ----
                        clip_filename = f"punch_{punch_counter}_{predicted_punch}.mp4"
                        save_clip(column(clip, "frame"),
                                  os.path.join(output_dir, clip_filename), fps, width, height)

                        print(f"[Punch {punch_counter}] M1: {m1_label} -> via {route} -> "
                              f"'{predicted_punch}' ({confidence:.2f}) | {clip_filename}")

                        results.append({
                            "punch_id": punch_counter,
                            "classification": predicted_punch,
                            "confidence": confidence,
                            "clip_filename": clip_filename,
                            "feedback": feedback,
                        })

                        cooldown = WINDOW_AFTER + 20

            if cooldown > 0:
                cooldown -= 1
    finally:
        cap.release()
        pose.close()

    print(f"Inference complete. Extracted & classified {punch_counter} punches.")
    return results
