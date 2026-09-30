"""Download and cache the MediaPipe Tasks model bundles and the MediaPipe web runtime."""

from __future__ import annotations

import io
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

_BASE = "https://storage.googleapis.com/mediapipe-models"

MODEL_URLS = {
    "face": f"{_BASE}/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "gesture": f"{_BASE}/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
    "pose": f"{_BASE}/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
}

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

# MediaPipe Tasks for the web (runs the same .task models in the browser via WASM/WebGL).
TASKS_VISION_VERSION = "1.0.1"
_TASKS_VISION_URL = ("https://registry.npmjs.org/@mediapipe/tasks-vision/-/"
                     f"tasks-vision-{TASKS_VISION_VERSION}.tgz")


def _download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as resp:
        return resp.read()


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


def ensure_web_vendor(model_dir: Path = DEFAULT_MODEL_DIR) -> Path:
    """Vendor @mediapipe/tasks-vision (JS bundle + WASM) so phones need no CDN access."""
    dest = model_dir / "web" / "tasks-vision"
    marker = dest / ".version"
    if marker.exists() and marker.read_text() == TASKS_VISION_VERSION:
        return dest
    print(f"Downloading MediaPipe web runtime {TASKS_VISION_VERSION} -> {dest}", file=sys.stderr)
    archive = tarfile.open(fileobj=io.BytesIO(_download(_TASKS_VISION_URL)), mode="r:gz")
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "wasm").mkdir(parents=True)
    for member in archive.getmembers():
        rel = member.name.removeprefix("package/")
        wanted = rel == "vision_bundle.mjs" or (rel.startswith("wasm/") and rel.count("/") == 1)
        if member.isfile() and wanted:
            (dest / rel).write_bytes(archive.extractfile(member).read())
    marker.write_text(TASKS_VISION_VERSION)
    return dest
