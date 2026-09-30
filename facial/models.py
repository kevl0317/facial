"""Download and cache the MediaPipe Tasks model bundles."""

from __future__ import annotations

import shutil
import sys
import urllib.request
from pathlib import Path

_BASE = "https://storage.googleapis.com/mediapipe-models"

MODEL_URLS = {
    "face": f"{_BASE}/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "gesture": f"{_BASE}/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
    "pose": f"{_BASE}/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
}

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def ensure_models(model_dir: Path = DEFAULT_MODEL_DIR) -> dict[str, Path]:
    """Return local paths to the three .task files, downloading any that are missing."""
    model_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, url in MODEL_URLS.items():
        path = model_dir / url.rsplit("/", 1)[-1]
        if not path.exists():
            print(f"Downloading MediaPipe {name} model -> {path}", file=sys.stderr)
            tmp = path.with_suffix(".part")
            with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as fh:
                shutil.copyfileobj(resp, fh)
            tmp.rename(path)
        paths[name] = path
    return paths
