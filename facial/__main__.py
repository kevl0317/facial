"""Command-line entry point.

    python -m facial VIDEO [options]     annotate a video file
    python -m facial serve [options]     web GUI: live camera (phone/PC) + video files
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .judge import DEFAULT_MODEL
from .pipeline import Options, run

EFFORTS = ("low", "medium", "high", "xhigh", "max")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m facial",
        description="Annotate a speaker's gestures, face and voice with MediaPipe, "
                    "then let Claude read intent and demeanor window by window. "
                    "Run `python -m facial serve` for the web GUI with live camera mode.",
    )
    p.add_argument("video", type=Path, help="input video file")
    p.add_argument("-o", "--output", type=Path, help="output video (default: <video>_annotated.mp4)")
    p.add_argument("--srt", type=Path, help="subtitles (.srt/.vtt) for the input video")
    p.add_argument("--whisper", metavar="MODEL", help="transcribe with faster-whisper (e.g. small, medium) "
                                                      "when no --srt is given")
    p.add_argument("--start", type=float, default=0.0, help="clip start, seconds")
    p.add_argument("--end", type=float, help="clip end, seconds")
    p.add_argument("--lang", choices=("en", "zh"), default="en", help="overlay and commentary language")
    p.add_argument("--judge", choices=("claude", "heuristic"), default="claude",
                   help="judgment layer (default: claude; falls back to heuristic without credentials)")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Claude model (default: {DEFAULT_MODEL})")
    p.add_argument("--effort", choices=EFFORTS, default="medium", help="Claude effort level (default: medium)")
    p.add_argument("--context", default="", help='who/what the clip is, e.g. "CEO keynote Q&A about export rules"')
    p.add_argument("--window", type=float, default=5.0, help="target window length in seconds (default 5)")
    p.add_argument("--analysis-fps", type=float, default=15.0, help="frames/s to run MediaPipe on (default 15)")
    p.add_argument("--mirrored", action="store_true", help="input is a mirrored selfie video (fixes left/right)")
    p.add_argument("--skeleton", action="store_true", help="also draw hand landmarks and face brackets")
    p.add_argument("--font", help="path to a .ttf/.ttc font (needed for --lang zh if no CJK font is found)")
    p.add_argument("--snapshot", type=float, metavar="T", help="render only the frame at T seconds to a PNG")
    p.add_argument("--workdir", type=Path, help="cache directory (default: <output stem>_work)")
    p.add_argument("--fresh", action="store_true", help="ignore cached stages and recompute everything")
    p.add_argument("--version", action="version", version=f"facial {__version__}")
    return p.parse_args(argv)


def parse_serve_args(argv) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m facial serve",
        description="Web GUI: real-time analysis from a phone or PC camera, and video-file processing.",
    )
    p.add_argument("--host", default="0.0.0.0", help="interface to listen on (default: all, so phones can connect)")
    p.add_argument("--port", type=int, help="port (default 8443 with HTTPS, 8000 without)")
    p.add_argument("--no-https", action="store_true",
                   help="plain HTTP (phones then only get camera access through an HTTPS tunnel)")
    p.add_argument("--token", help="access key required by the API (default: random per run); '' disables it")
    p.add_argument("--judge", choices=("claude", "heuristic"), default="claude")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Claude model (default: {DEFAULT_MODEL})")
    p.add_argument("--effort", choices=EFFORTS, default="medium", help="effort for video-file jobs")
    p.add_argument("--live-effort", choices=EFFORTS, default="low",
                   help="effort for live windows (default: low, for fast verdicts)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "serve":
        from .server import serve

        a = parse_serve_args(argv[1:])
        serve(host=a.host, port=a.port, https=not a.no_https, token=a.token, judge=a.judge, model=a.model,
              effort=a.effort, live_effort=a.live_effort)
        return 0

    args = parse_args(argv)
    opts = Options(**{k: v for k, v in vars(args).items() if k in Options.__dataclass_fields__})
    try:
        run(opts)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
