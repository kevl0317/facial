"""The offline video pipeline, shared by the CLI and the web GUI's video-file mode."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .audio import analyze_audio
from .features import add_hand_motion, assign_segments, build_windows, clip_baseline, window_features
from .jev import jev_settings
from .judge import judge_all
from .media import cut_clip, extract_audio, probe
from .perception import run_perception
from .providers import PROVIDERS
from .render import Overlay, hand_label, render_snapshot, render_video
from .transcript import load_subtitles, shift, to_srt, transcribe

# report(stage, fraction of that stage done or None, message or "")
Report = Callable[[str, "float | None", str], None]

# Share of the whole job each stage takes, for an overall progress bar.
STAGES = {"prepare": (0.0, 0.05), "perception": (0.05, 0.6), "judge": (0.6, 0.72), "render": (0.72, 1.0)}


class StderrReporter:
    """Print messages, and each stage's progress in 10% steps."""

    def __init__(self):
        self._last: dict[str, int] = {}

    def __call__(self, stage: str, frac: float | None, msg: str) -> None:
        if msg:
            print(msg, file=sys.stderr)
        elif frac is not None:
            step = int(frac * 10)
            if self._last.get(stage) != step:
                self._last[stage] = step
                print(f"  {stage} {step * 10:3d}%", file=sys.stderr)


@dataclass
class Options:
    video: Path
    output: Path | None = None
    srt: Path | None = None
    whisper: str | None = None
    start: float = 0.0
    end: float | None = None
    lang: str = "en"
    judge: str = "claude"  # a provider id from providers.py, or "heuristic"
    model: str | None = None  # default: the provider's default model
    base_url: str | None = None
    effort: str = "medium"
    jev: bool = False  # Jev decides each window; the judge above only writes the words
    jev_via: str | None = None  # "typesafe" or "openrouter" (default: whichever key is set)
    jev_model: str | None = None
    context: str = ""
    window: float = 5.0
    analysis_fps: float = 15.0
    mirrored: bool = False
    skeleton: bool = False
    font: str | None = None
    snapshot: float | None = None
    workdir: Path | None = None
    fresh: bool = False


@dataclass
class Result:
    analysis: Path
    output: Path | None = None  # annotated video (None in snapshot mode)
    snapshot: Path | None = None


def _cached(path: Path, key: dict, fresh: bool, compute, report: Report):
    """Reuse a JSON stage result when it was produced with the same key."""
    if path.exists() and not fresh:
        blob = json.loads(path.read_text(encoding="utf-8"))
        if blob.get("key") == key:
            report("prepare", None, f"Using cached {path.name}")
            return blob["data"]
    data = compute()
    path.write_text(json.dumps({"key": key, "data": data}, ensure_ascii=False), encoding="utf-8")
    return data


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def run(opts: Options, report: Report | None = None) -> Result:
    report = report or StderrReporter()
    src = Path(opts.video).resolve()
    if not src.exists():
        raise FileNotFoundError(f"No such file: {src}")
    output = Path(opts.output) if opts.output else src.with_name(f"{src.stem}_annotated.mp4")
    workdir = Path(opts.workdir) if opts.workdir else output.with_name(f"{output.stem}_work")
    workdir.mkdir(parents=True, exist_ok=True)

    def progress(stage: str):
        return lambda frac: report(stage, frac, "")

    # 0. Optional clip ---------------------------------------------------------
    video = src
    if opts.start > 0 or opts.end is not None:
        clip = workdir / f"clip_{opts.start:g}_{opts.end if opts.end is not None else 'end'}.mp4"
        if not clip.exists() or opts.fresh:
            report("prepare", 0.1, f"Cutting clip {opts.start}s -> {opts.end}s")
            cut_clip(src, clip, opts.start, opts.end)
        video = clip
    info = probe(video)
    report("prepare", 0.3, f"Video: {info.width}x{info.height} @ {info.fps:.2f} fps, {info.duration:.1f}s")

    # 1. Audio: voice measurements ---------------------------------------------
    wav = extract_audio(video, workdir / "audio.wav")
    audio = analyze_audio(wav) if wav else None
    if audio is None:
        report("prepare", 0.6, "No audio track: voice fields will be empty.")

    # 2. Subtitles --------------------------------------------------------------
    segments = []
    if opts.srt:
        segments = shift(load_subtitles(opts.srt), opts.start, info.duration)
    elif opts.whisper and wav:
        segments = transcribe(wav, opts.whisper, language="zh" if opts.lang == "zh" else None)
        (workdir / "transcript.srt").write_text(to_srt(segments), encoding="utf-8")
    report("prepare", 1.0, f"Subtitles: {len(segments)} segments")

    # 3. Perception: MediaPipe on every sampled frame ----------------------------
    stat = video.stat()
    perception_key = {"video": str(video), "size": stat.st_size, "mtime": int(stat.st_mtime),
                      "fps": opts.analysis_fps, "mirrored": opts.mirrored, "v": 3}
    report("perception", 0.0, "Running MediaPipe (face, hands/gestures, pose)...")
    samples = _cached(workdir / "samples.json", perception_key, opts.fresh,
                      lambda: run_perception(info, opts.analysis_fps, opts.mirrored,
                                             progress=progress("perception")), report)
    add_hand_motion(samples, info.width / info.height)

    # 4. Windows with five fields each -----------------------------------------
    spans = build_windows(info.duration, segments, target=opts.window,
                          min_len=opts.window / 2, max_len=opts.window * 1.8)
    per_window = assign_segments(spans, segments)
    baseline = clip_baseline(samples)
    feats = [window_features(i, len(spans), span, samples, audio, per_window[i], baseline)
             for i, span in enumerate(spans)]

    # 5. Judgment layer ---------------------------------------------------------
    model = opts.model or PROVIDERS.get(opts.judge, {}).get("model")
    jev = jev_settings(opts.jev, opts.jev_via, opts.jev_model)
    judge_key = {"feats": _digest(feats), "judge": opts.judge, "model": model, "effort": opts.effort,
                 "lang": opts.lang, "context": opts.context, **({"base_url": opts.base_url} if opts.base_url else {}),
                 **({"jev": jev} if jev else {})}
    report("judge", 0.0, f"Judging {len(feats)} windows with {'Jev + ' if jev else ''}{opts.judge}...")
    clip_info = {"duration": info.duration, "baseline": baseline}
    judgments_path = workdir / "judgments.json"
    judgments = _cached(judgments_path, judge_key, opts.fresh, lambda: judge_all(
        feats, clip_info, opts.judge, opts.lang, model, opts.effort, opts.context,
        lambda h: hand_label(h, opts.lang), log=lambda m: report("judge", None, m),
        progress=progress("judge"), base_url=opts.base_url, jev=jev), report)
    if (opts.judge != "heuristic" or jev) and any(j["source"] == "heuristic" for j in judgments):
        judgments_path.unlink()  # don't cache fallbacks: the next run retries the AI judge

    analysis = output.with_name(f"{output.stem}_analysis.json")
    analysis.write_text(json.dumps(
        {"video": str(src), "start": opts.start, "duration": info.duration, "baseline": baseline,
         "windows": [{**f, "judgment": j} for f, j in zip(feats, judgments)]},
        ensure_ascii=False, indent=2), encoding="utf-8")

    # 6. Render -------------------------------------------------------------------
    overlay = Overlay(info.width, info.height, samples, feats, judgments, segments,
                      lang=opts.lang, skeleton=opts.skeleton, font=opts.font)
    if opts.lang == "zh" and not overlay.fonts.has_cjk:
        report("render", None, "Warning: no CJK font found; pass --font /path/to/NotoSansCJK.ttc "
                               "for Chinese text.")
    if opts.snapshot is not None:
        png = output.with_name(f"{output.stem}_t{opts.snapshot:g}.png")
        render_snapshot(info, png, overlay, opts.snapshot)
        report("render", 1.0, f"Snapshot: {png}")
        return Result(analysis=analysis, snapshot=png)
    report("render", 0.0, "Rendering overlay...")
    render_video(info, output, overlay, audio_from=video, progress=progress("render"))
    report("render", 1.0, f"Done: {output}\nAnalysis: {analysis}")
    return Result(analysis=analysis, output=output)
