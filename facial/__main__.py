"""Command-line entry point.

    python -m facial VIDEO [options]     annotate a video file
    python -m facial serve [options]     web GUI: live camera (phone/PC) + video files
    python -m facial build-site          the same GUI as a static site (runs fully in the browser)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .jev import JEV_ROUTES, jev_settings
from .providers import PROVIDERS, RULES
from .pipeline import Options, run

EFFORTS = ("low", "medium", "high", "xhigh", "max")
JUDGES = (*PROVIDERS, RULES)
JUDGE_HELP = ("who judges each window: an AI provider (" + ", ".join(PROVIDERS) + ") or heuristic "
              "(rules, no key). Default: claude. Keys come from the environment, e.g. ANTHROPIC_API_KEY, "
              "OPENAI_API_KEY, DEEPSEEK_API_KEY, GEMINI_API_KEY; without one it falls back to heuristic")
MODEL_HELP = "model name (default per provider: " + ", ".join(
    f"{k} {v['model']}" for k, v in PROVIDERS.items() if v["model"]) + ")"


def add_jev_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--jev", action="store_true",
                   help="let Jev (TypeSafe AI's fast decision model) decide each window's scores and intent; "
                        "the --judge model then only writes the words, and decides when Jev is unsure")
    p.add_argument("--jev-via", choices=tuple(JEV_ROUTES),
                   help="reach Jev through TypeSafe (TYPESAFE_API_KEY) or OpenRouter (OPENROUTER_API_KEY); "
                        "default: whichever key is set")
    p.add_argument("--jev-model", help="Jev model (default: jev-latest)")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m facial",
        description="Annotate a speaker's gestures, face and voice with MediaPipe, "
                    "then let an AI model (Claude, GPT, DeepSeek, Gemini...) read intent and demeanor "
                    "window by window. "
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
    p.add_argument("--judge", choices=JUDGES, default="claude", metavar="PROVIDER", help=JUDGE_HELP)
    p.add_argument("--model", help=MODEL_HELP)
    p.add_argument("--base-url", help="API base URL (needed for --judge custom; overrides the provider's)")
    p.add_argument("--effort", choices=EFFORTS, default="medium", help="reasoning effort (default: medium)")
    add_jev_args(p)
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
    p.add_argument("--judge", choices=JUDGES, default="claude", metavar="PROVIDER", help=JUDGE_HELP)
    p.add_argument("--model", help=MODEL_HELP)
    p.add_argument("--base-url", help="API base URL (needed for --judge custom; overrides the provider's)")
    p.add_argument("--effort", choices=EFFORTS, default="medium", help="effort for video-file jobs")
    p.add_argument("--live-effort", choices=EFFORTS, default="low",
                   help="effort for live windows (default: low, for fast verdicts)")
    add_jev_args(p)
    return p.parse_args(argv)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "build-site":
        from .webbuild import build_site

        p = argparse.ArgumentParser(prog="python -m facial build-site",
                                    description="Build the server-free web app as a static site (e.g. for GitHub Pages).")
        p.add_argument("--out", type=Path, default=Path("site"), help="output folder (default: site)")
        out = build_site(p.parse_args(argv[1:]).out)
        print(f"Static site written to {out}", file=sys.stderr)
        return 0
    if argv and argv[0] == "serve":
        from .server import serve

        a = parse_serve_args(argv[1:])
        serve(host=a.host, port=a.port, https=not a.no_https, token=a.token, judge=a.judge, model=a.model,
              base_url=a.base_url, effort=a.effort, live_effort=a.live_effort,
              jev=jev_settings(a.jev, a.jev_via, a.jev_model))
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
