"""Video/audio I/O: probing, clipping, audio extraction and H.264 writing via ffmpeg."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2


def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover - depends on the machine
        raise RuntimeError("ffmpeg not found: install ffmpeg or `pip install imageio-ffmpeg`") from exc


@dataclass
class VideoInfo:
    path: Path
    fps: float
    frames: int
    width: int
    height: int

    @property
    def duration(self) -> float:
        return self.frames / self.fps


def probe(path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    info = VideoInfo(
        path=Path(path),
        fps=fps,
        frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cap.release()
    return info


def cut_clip(src: Path, dst: Path, start: float, end: float | None) -> Path:
    """Re-encode [start, end) of src into dst so every later stage sees a clip starting at 0."""
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-ss", f"{start:.3f}", "-i", str(src)]
    if end is not None:
        cmd += ["-t", f"{end - start:.3f}"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-c:a", "aac", str(dst)]
    subprocess.run(cmd, check=True)
    return dst


def extract_audio(video: Path, wav: Path, sample_rate: int = 16000) -> Path | None:
    """Extract mono PCM audio; returns None when the video has no audio stream."""
    cmd = [
        ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(video),
        "-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "wav", str(wav),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0 or not wav.exists() or wav.stat().st_size < 1024:
        return None
    return wav


class VideoWriter:
    """Pipe BGR frames into ffmpeg (H.264, yuv420p) and copy audio over from the source."""

    def __init__(self, path: Path, width: int, height: int, fps: float, audio_from: Path | None = None):
        cmd = [
            ffmpeg_exe(), "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", f"{fps}", "-i", "-",
        ]
        if audio_from is not None:
            cmd += ["-i", str(audio_from), "-map", "0:v:0", "-map", "1:a:0?",
                    "-c:a", "aac", "-b:a", "160k", "-shortest"]
        cmd += [
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(path),
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame_bgr) -> None:
        self._proc.stdin.write(frame_bgr.tobytes())

    def close(self) -> None:
        self._proc.stdin.close()
        if self._proc.wait() != 0:
            raise RuntimeError("ffmpeg failed while writing the output video")
