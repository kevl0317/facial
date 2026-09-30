"""Command-line entry point: python -m facial VIDEO [options]."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from . import __version__
from .audio import analyze_audio
from .features import add_hand_motion, assign_segments, build_windows, clip_baseline, window_features
from .judge import DEFAULT_MODEL, judge_all
from .media import cut_clip, extract_audio, probe
from .perception import run_perception
from .render import Overlay, hand_label, render_snapshot, render_video
from .transcript import load_subtitles, shift, to_srt, transcribe


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m facial",
        description="Annotate a speaker's gestures, face and voice with MediaPipe, "
                    "then let Claude read intent and demeanor window by window.",
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
    p.add_argument("--effort", choices=("low", "medium", "high", "xhigh", "max"), default="medium",
                   help="Claude effort level (default: medium)")
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


def _cached(path: Path, key: dict, fresh: bool, compute):
    """Reuse a JSON stage result when it was produced with the same key."""
    if path.exists() and not fresh:
        blob = json.loads(path.read_text(encoding="utf-8"))
        if blob.get("key") == key:
            print(f"Using cached {path.name}", file=sys.stderr)
            return blob["data"]
    data = compute()
    path.write_text(json.dumps({"key": key, "data": data}, ensure_ascii=False), encoding="utf-8")
    return data


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def main(argv=None) -> int:
    args = parse_args(argv)
    src = args.video.resolve()
    if not src.exists():
        print(f"No such file: {src}", file=sys.stderr)
        return 1
    output = args.output or src.with_name(f"{src.stem}_annotated.mp4")
    workdir = args.workdir or output.with_name(f"{output.stem}_work")
    workdir.mkdir(parents=True, exist_ok=True)

    # 0. Optional clip ---------------------------------------------------------
    video = src
    if args.start > 0 or args.end is not None:
        clip = workdir / f"clip_{args.start:g}_{args.end if args.end is not None else 'end'}.mp4"
        if not clip.exists() or args.fresh:
            print(f"Cutting clip {args.start}s -> {args.end}s", file=sys.stderr)
            cut_clip(src, clip, args.start, args.end)
        video = clip
    info = probe(video)
    print(f"Video: {info.width}x{info.height} @ {info.fps:.2f} fps, {info.duration:.1f}s", file=sys.stderr)

    # 1. Audio: voice measurements ---------------------------------------------
    wav = extract_audio(video, workdir / "audio.wav")
    audio = analyze_audio(wav) if wav else None
    if audio is None:
        print("No audio track: voice fields will be empty.", file=sys.stderr)

    # 2. Subtitles --------------------------------------------------------------
    segments = []
    if args.srt:
        segments = shift(load_subtitles(args.srt), args.start, info.duration)
    elif args.whisper and wav:
        segments = transcribe(wav, args.whisper, language="zh" if args.lang == "zh" else None)
        (workdir / "transcript.srt").write_text(to_srt(segments), encoding="utf-8")
    print(f"Subtitles: {len(segments)} segments", file=sys.stderr)

    # 3. Perception: MediaPipe on every sampled frame ----------------------------
    stat = video.stat()
    perception_key = {"video": str(video), "size": stat.st_size, "mtime": int(stat.st_mtime),
                      "fps": args.analysis_fps, "mirrored": args.mirrored, "v": 1}
    print("Running MediaPipe (face, hands/gestures, pose)...", file=sys.stderr)
    samples = _cached(workdir / "samples.json", perception_key, args.fresh,
                      lambda: run_perception(info, args.analysis_fps, args.mirrored))
    add_hand_motion(samples, info.width / info.height)

    # 4. Windows with five fields each -----------------------------------------
    spans = build_windows(info.duration, segments, target=args.window,
                          min_len=args.window / 2, max_len=args.window * 1.8)
    per_window = assign_segments(spans, segments)
    baseline = clip_baseline(samples)
    feats = [window_features(i, len(spans), span, samples, audio, per_window[i], baseline)
             for i, span in enumerate(spans)]

    # 5. Judgment layer ---------------------------------------------------------
    judge_key = {"feats": _digest(feats), "judge": args.judge, "model": args.model, "effort": args.effort,
                 "lang": args.lang, "context": args.context}
    print(f"Judging {len(feats)} windows with {args.judge}...", file=sys.stderr)
    clip_info = {"duration": info.duration, "baseline": baseline}
    judgments_path = workdir / "judgments.json"
    judgments = _cached(judgments_path, judge_key, args.fresh, lambda: judge_all(
        feats, clip_info, args.judge, args.lang, args.model, args.effort, args.context,
        lambda h: hand_label(h, args.lang)))
    if args.judge == "claude" and any(j["source"] != "claude" for j in judgments):
        judgments_path.unlink()  # don't cache fallbacks: the next run retries Claude

    analysis = output.with_name(f"{output.stem}_analysis.json")
    analysis.write_text(json.dumps(
        {"video": str(src), "start": args.start, "duration": info.duration, "baseline": baseline,
         "windows": [{**f, "judgment": j} for f, j in zip(feats, judgments)]},
        ensure_ascii=False, indent=2), encoding="utf-8")

    # 6. Render -------------------------------------------------------------------
    overlay = Overlay(info.width, info.height, samples, feats, judgments, segments,
                      lang=args.lang, skeleton=args.skeleton, font=args.font)
    if args.lang == "zh" and not overlay.fonts.has_cjk:
        print("Warning: no CJK font found; pass --font /path/to/NotoSansCJK.ttc for Chinese text.",
              file=sys.stderr)
    if args.snapshot is not None:
        png = output.with_name(f"{output.stem}_t{args.snapshot:g}.png")
        render_snapshot(info, png, overlay, args.snapshot)
        print(f"Snapshot: {png}", file=sys.stderr)
        return 0
    print("Rendering overlay...", file=sys.stderr)
    render_video(info, output, overlay, audio_from=video)
    print(f"Done: {output}\nAnalysis: {analysis}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
