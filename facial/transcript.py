"""Subtitles: load SRT/VTT files, or transcribe with faster-whisper when it is installed."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Segment:
    start: float
    end: float
    text: str


_TS = r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})"
_CUE = re.compile(_TS + r"\s*-->\s*" + _TS)
_TAGS = re.compile(r"<[^>]+>|\{[^}]*\}")


def _seconds(h, m, s, frac) -> float:
    return int(h or 0) * 3600 + int(m) * 60 + int(s) + int(frac.ljust(3, "0")) / 1000


def parse_subtitles(text: str) -> list[Segment]:
    """Parse SRT or WebVTT text into time-ordered segments."""
    segments = []
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n")):
        lines = [ln.strip() for ln in block.strip().split("\n")]
        for i, line in enumerate(lines):
            match = _CUE.search(line)
            if match:
                g = match.groups()
                body = " ".join(_TAGS.sub("", ln) for ln in lines[i + 1:] if ln).strip()
                if body:
                    segments.append(Segment(_seconds(*g[:4]), _seconds(*g[4:]), body))
                break
    return sorted(segments, key=lambda s: s.start)


def load_subtitles(path: Path) -> list[Segment]:
    return parse_subtitles(Path(path).read_text(encoding="utf-8-sig"))


def shift(segments: list[Segment], offset: float, duration: float) -> list[Segment]:
    """Move segments by -offset and keep those overlapping [0, duration)."""
    out = []
    for s in segments:
        start, end = s.start - offset, s.end - offset
        if end > 0 and start < duration:
            out.append(Segment(max(0.0, start), min(duration, end), s.text))
    return out


def transcribe(wav: Path, model_size: str = "small", language: str | None = None) -> list[Segment]:
    """Transcribe with faster-whisper (optional dependency). Returns [] if unavailable."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("faster-whisper is not installed; continuing without subtitles "
              "(pass --srt, or `pip install faster-whisper`).", file=sys.stderr)
        return []
    print(f"Transcribing with faster-whisper ({model_size})...", file=sys.stderr)
    model = WhisperModel(model_size, device="auto", compute_type="default")
    result, _ = model.transcribe(str(wav), language=language, vad_filter=True)
    return [Segment(s.start, s.end, s.text.strip()) for s in result if s.text.strip()]


def to_srt(segments: list[Segment]) -> str:
    def ts(t: float) -> str:
        ms = int(round(t * 1000))
        return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"

    return "\n".join(f"{i}\n{ts(s.start)} --> {ts(s.end)}\n{s.text}\n" for i, s in enumerate(segments, 1))
