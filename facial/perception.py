"""Per-frame perception with MediaPipe Tasks: hands + gestures, face + blendshapes, pose.

Each sampled frame becomes one JSON-serialisable dict ("sample"). Coordinates are
normalised to [0, 1] (x by frame width, y by frame height).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from .media import VideoInfo
from .models import DEFAULT_MODEL_DIR, ensure_models

# (mcp, pip, tip) landmark indices for the four fingers.
FINGERS = {"index": (5, 6, 8), "middle": (9, 10, 12), "ring": (13, 14, 16), "pinky": (17, 18, 20)}
PALM = [0, 5, 9, 13, 17]

# MediaPipe's canned gesture classes mapped onto our shape vocabulary.
MP_GESTURES = {
    "Open_Palm": "open_palm",
    "Closed_Fist": "fist",
    "Pointing_Up": "pointing",
    "Thumb_Up": "thumb",
    "Thumb_Down": "thumb",
    "Victory": "two_fingers",
}


def describe_hand(pts: np.ndarray, world: np.ndarray | None, appearance: str) -> dict:
    """Classify hand shape, palm facing and axis from 21 landmarks.

    pts: (21, 2) pixel coordinates. world: (21, 3) metric coordinates (x right,
    y down, z away from camera) or None. appearance: "left"/"right", the
    chirality of the hand as it looks in the image.
    """
    wrist = pts[0]
    size = float(np.linalg.norm(pts[9] - wrist)) + 1e-6  # wrist -> middle MCP: a stable hand-scale unit

    extended = {
        name: np.linalg.norm(pts[tip] - wrist) > 1.15 * np.linalg.norm(pts[pip] - wrist)
        for name, (_, pip, tip) in FINGERS.items()
    }
    n_ext = sum(extended.values())
    thumb_out = (
        np.linalg.norm(pts[4] - pts[17]) > 1.1 * np.linalg.norm(pts[3] - pts[17])
        and np.linalg.norm(pts[4] - pts[5]) > 0.5 * size
    )
    pinch = np.linalg.norm(pts[4] - pts[8]) < 0.3 * size

    if pinch and not extended["index"]:
        shape = "pinch"
    elif n_ext >= 4:
        shape = "open_palm"
    elif n_ext == 0:
        shape = "thumb" if thumb_out else "fist"
    elif extended["index"] and n_ext == 1:
        shape = "pointing"
    elif extended["index"] and extended["middle"] and n_ext == 2:
        shape = "two_fingers"
    else:
        shape = "relaxed"

    axis_vec = pts[9] - wrist
    if abs(axis_vec[0]) > abs(axis_vec[1]):
        axis = "horizontal"
    else:
        axis = "up" if axis_vec[1] < 0 else "down"

    # Palm normal: (index MCP - wrist) x (pinky MCP - wrist) points out of the palm
    # toward the camera (-z) for a right-looking hand; mirror chirality flips it.
    sign = 1.0 if appearance == "right" else -1.0
    v1, v2 = pts[5] - wrist, pts[17] - wrist
    sin_2d = (v1[0] * v2[1] - v1[1] * v2[0]) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9) * sign
    if world is not None:
        normal = np.cross(world[5] - world[0], world[17] - world[0]) * sign
    else:
        normal = np.array([0.0, 0.0, sin_2d])
    normal = normal / (np.linalg.norm(normal) + 1e-9)
    # The 2D winding is reliable when the palm is clearly turned toward or away from
    # the camera; trust it over the (noisier) 3D estimate when they disagree.
    if abs(sin_2d) > 0.25 and np.sign(sin_2d) != np.sign(normal[2]):
        normal = np.array([0.0, 0.0, np.sign(sin_2d)])
    if normal[2] < -0.6:
        facing = "palm_out"
    elif normal[2] > 0.6:
        facing = "back_out"
    elif abs(normal[1]) >= abs(normal[0]):
        facing = "palm_up" if normal[1] < 0 else "palm_down"
    else:
        facing = "palm_side"

    return {"shape": shape, "facing": facing, "axis": axis, "size": size}


def _face_metrics(bs: dict[str, float]) -> dict[str, float]:
    def avg(*names: str) -> float:
        return float(np.mean([bs.get(n, 0.0) for n in names]))

    metrics = {
        "smile": avg("mouthSmileLeft", "mouthSmileRight"),
        "frown": avg("mouthFrownLeft", "mouthFrownRight"),
        "brow_furrow": avg("browDownLeft", "browDownRight"),
        "brow_raise": avg("browInnerUp", "browOuterUpLeft", "browOuterUpRight"),
        "lip_press": avg("mouthPressLeft", "mouthPressRight"),
        "blink": avg("eyeBlinkLeft", "eyeBlinkRight"),
        "eye_wide": avg("eyeWideLeft", "eyeWideRight"),
        "squint": avg("eyeSquintLeft", "eyeSquintRight"),
        "jaw": bs.get("jawOpen", 0.0),
        "gaze_side": max(avg("eyeLookOutLeft", "eyeLookInRight"), avg("eyeLookInLeft", "eyeLookOutRight")),
        "gaze_down": avg("eyeLookDownLeft", "eyeLookDownRight"),
    }
    return {k: round(v, 3) for k, v in metrics.items()}


def _face(res) -> dict | None:
    if not res.face_landmarks:
        return None

    def bbox(lms):
        xs = [p.x for p in lms]
        ys = [p.y for p in lms]
        return min(xs), min(ys), max(xs), max(ys)

    boxes = [bbox(lms) for lms in res.face_landmarks]
    best = max(range(len(boxes)), key=lambda i: (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]))
    x0, y0, x1, y1 = boxes[best]
    out = {"bbox": [round(v, 4) for v in (x0, y0, x1, y1)], "size": round(y1 - y0, 4)}
    if res.face_blendshapes:
        out.update(_face_metrics({c.category_name: c.score for c in res.face_blendshapes[best]}))
    if res.facial_transformation_matrixes:
        rot = np.asarray(res.facial_transformation_matrixes[best], dtype=np.float64)[:3, :3]
        pitch, yaw, roll = cv2.RQDecomp3x3(rot)[0]
        out.update(pitch=round(float(pitch), 1), yaw=round(float(yaw), 1), roll=round(float(roll), 1))
    return out


def _hands(res, width: int, height: int, mirrored: bool) -> list[dict]:
    hands = []
    for i, lms in enumerate(res.hand_landmarks):
        handed = res.handedness[i][0]
        # With the Tasks API the label matches how the hand looks in the image
        # (verified on MediaPipe's own right_hands.jpg test asset). In a mirrored
        # selfie video that is the opposite of the speaker's real hand.
        appearance = handed.category_name.lower()
        side = appearance if not mirrored else ("left" if appearance == "right" else "right")

        pts = np.array([[p.x * width, p.y * height] for p in lms])
        world = None
        if res.hand_world_landmarks and i < len(res.hand_world_landmarks):
            world = np.array([[p.x, p.y, p.z] for p in res.hand_world_landmarks[i]])
        geo = describe_hand(pts, world, appearance)

        shape, score, src = geo["shape"], 0.8 * handed.score, "geometry"
        if res.gestures and i < len(res.gestures) and res.gestures[i]:
            top = res.gestures[i][0]
            if top.category_name in MP_GESTURES and top.score >= 0.5:
                shape, score, src = MP_GESTURES[top.category_name], top.score, "classifier"

        anchor = pts[PALM].mean(axis=0)
        hands.append({
            "side": side,
            "shape": shape,
            "facing": geo["facing"],
            "axis": geo["axis"],
            "score": round(float(score), 3),
            "src": src,
            "anchor": [round(anchor[0] / width, 4), round(anchor[1] / height, 4)],
            "size": round(geo["size"] / height, 4),
            "pts": [[round(x / width, 4), round(y / height, 4)] for x, y in pts],
        })
    return hands


def _pose(res, width: int, height: int) -> dict | None:
    if not res.pose_landmarks:
        return None
    lm = res.pose_landmarks[0]

    def point(i):
        return np.array([lm[i].x * width, lm[i].y * height]), (lm[i].visibility or 0.0)

    ls, lv = point(11)
    rs, rv = point(12)
    if min(lv, rv) < 0.5:
        return None
    shoulder_w = float(np.linalg.norm(ls - rs)) + 1e-6
    out = {
        "shoulder_w": round(shoulder_w / height, 4),
        "tilt": round(math.degrees(math.atan2(ls[1] - rs[1], abs(ls[0] - rs[0]))), 1),
    }
    lw, lwv = point(15)
    rw, rwv = point(16)
    if min(lwv, rwv) >= 0.5:
        out["arms_open"] = round(float(np.linalg.norm(lw - rw)) / shoulder_w, 3)
    return out


def _histogram(frame_bgr) -> np.ndarray:
    hsv = cv2.cvtColor(cv2.resize(frame_bgr, (64, 36)), cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    return cv2.normalize(hist, hist)


def run_perception(
    info: VideoInfo,
    analysis_fps: float = 15.0,
    mirrored: bool = False,
    model_dir: Path = DEFAULT_MODEL_DIR,
) -> list[dict]:
    """Run face, gesture and pose models over the video at ~analysis_fps."""
    paths = ensure_models(model_dir)
    video_mode = vision.RunningMode.VIDEO

    def base(name):
        return mp_python.BaseOptions(model_asset_path=str(paths[name]))

    face_task = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=base("face"), running_mode=video_mode, num_faces=3,
        output_face_blendshapes=True, output_facial_transformation_matrixes=True,
    ))
    hand_task = vision.GestureRecognizer.create_from_options(vision.GestureRecognizerOptions(
        base_options=base("gesture"), running_mode=video_mode, num_hands=2,
    ))
    pose_task = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=base("pose"), running_mode=video_mode, num_poses=1,
    ))

    step = max(1, round(info.fps / analysis_fps))
    cap = cv2.VideoCapture(str(info.path))
    samples: list[dict] = []
    prev_hist = None
    index = 0
    next_report = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % step == 0:
                t = index / info.fps
                hist = _histogram(frame)
                cut = prev_hist is not None and cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA) > 0.45
                prev_hist = hist

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts_ms = int(round(t * 1000))
                face_res = face_task.detect_for_video(image, ts_ms)
                hand_res = hand_task.recognize_for_video(image, ts_ms)
                pose_res = pose_task.detect_for_video(image, ts_ms)

                h, w = frame.shape[:2]
                samples.append({
                    "t": round(t, 3),
                    "cut": bool(cut),
                    "faces": len(face_res.face_landmarks),
                    "face": _face(face_res),
                    "hands": _hands(hand_res, w, h, mirrored),
                    "pose": _pose(pose_res, w, h),
                })
                if info.frames and index / info.frames >= next_report:
                    print(f"  perception {100 * index / info.frames:5.1f}%  (t={t:.1f}s)", file=sys.stderr)
                    next_report += 0.1
            index += 1
    finally:
        cap.release()
        face_task.close()
        hand_task.close()
        pose_task.close()
    return samples
