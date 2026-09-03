"""
Full joint tracking via MediaPipe Pose Landmarker (Tasks API).

Requires pose_landmarker.task to be present alongside the app (downloaded
at Docker build time -- see Dockerfile). Runs in VIDEO mode so MediaPipe
can use its own internal frame-to-frame tracking, which keeps it locked
onto one person instead of flickering between people in the frame.
"""
import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pose_landmarker.task")

IDX = {
    "nose": 0,
    "l_shoulder": 11, "r_shoulder": 12,
    "l_elbow": 13, "r_elbow": 14,
    "l_wrist": 15, "r_wrist": 16,
    "l_hip": 23, "r_hip": 24,
    "l_knee": 25, "r_knee": 26,
    "l_ankle": 27, "r_ankle": 28,
    "l_heel": 29, "r_heel": 30,
    "l_foot": 31, "r_foot": 32,
}

SKELETON_EDGES = [
    ("l_shoulder", "r_shoulder"), ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
    ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"),
    ("l_shoulder", "l_hip"), ("r_shoulder", "r_hip"), ("l_hip", "r_hip"),
    ("l_hip", "l_knee"), ("l_knee", "l_ankle"), ("l_ankle", "l_heel"), ("l_ankle", "l_foot"),
    ("r_hip", "r_knee"), ("r_knee", "r_ankle"), ("r_ankle", "r_heel"), ("r_ankle", "r_foot"),
]


def model_available():
    return os.path.exists(MODEL_PATH)


def extract_landmarks(frames, fps):
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.4,
        min_pose_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    H, W = frames[0].shape[:2]
    out = []
    last_ts = -1
    with mp_vision.PoseLandmarker.create_from_options(options) as detector:
        for i, frame in enumerate(frames):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int(i * (1000.0 / fps))
            if ts_ms <= last_ts:
                ts_ms = last_ts + 1
            last_ts = ts_ms
            result = detector.detect_for_video(mp_image, ts_ms)
            if result.pose_landmarks:
                pts = result.pose_landmarks[0]
                joints = {}
                for name, idx in IDX.items():
                    p = pts[idx]
                    joints[name] = (p.x * W, p.y * H, p.visibility)
                out.append(joints)
            else:
                out.append(None)
    return out


def interpolate_joints(landmarks_per_frame, n):
    joints = {}
    for name in IDX:
        x = np.full(n, np.nan)
        y = np.full(n, np.nan)
        for i, lm in enumerate(landmarks_per_frame):
            if lm is not None and name in lm and lm[name][2] > 0.3:
                x[i], y[i] = lm[name][0], lm[name][1]
        idx = np.arange(n)
        good = ~np.isnan(x)
        if good.sum() >= 2:
            x[~good] = np.interp(idx[~good], idx[good], x[good])
            y[~good] = np.interp(idx[~good], idx[good], y[good])
        else:
            x = np.nan_to_num(x)
            y = np.nan_to_num(y)
        joints[name] = (x, y)
    return joints


def angle_deg(a, b, c):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    v1 = a - b
    v2 = c - b
    dot = (v1 * v2).sum(axis=-1)
    n1 = np.linalg.norm(v1, axis=-1)
    n2 = np.linalg.norm(v2, axis=-1)
    cos = np.clip(dot / (n1 * n2 + 1e-9), -1, 1)
    return np.degrees(np.arccos(cos))
