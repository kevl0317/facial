"""Live sessions: measurement windows streamed from a browser (phone or PC camera).

The browser runs MediaPipe on-device and sends one window (~5 s) of per-frame
samples, microphone audio and speech-recognition text at a time. The server
turns each window into the same five fields as the video pipeline and judges it.
"""

from __future__ import annotations

import base64
import threading
import time
import uuid

import librosa
import numpy as np

from .audio import analyze_samples
from .features import add_hand_motion, clip_baseline, window_features
from .judge import judge_or_fallback, make_judge
from .render import hand_label
from .transcript import Segment

AUDIO_SR = 16000
AUDIO_KEEP_S = 120.0  # audio kept for the loudness/pitch baseline
SAMPLES_KEEP_S = 600.0  # frames kept for the face baseline
HAND_KEYS = {"side", "shape", "facing", "axis", "score", "anchor", "size"}


def _clean_sample(s: dict) -> dict | None:
    try:
        t = float(s["t"])
    except (KeyError, TypeError, ValueError):
        return None
    face = s.get("face") or None
    if face is not None and not isinstance(face, dict):
        face = None
    hands = [{k: v for k, v in h.items() if k != "pts"} for h in (s.get("hands") or [])
             if isinstance(h, dict) and HAND_KEYS <= h.keys()]
    return {"t": t, "cut": False, "faces": int(s.get("faces") or (1 if face else 0)), "face": face,
            "hands": hands, "pose": s.get("pose") or None}


class LiveSession:
    def __init__(self, lang: str = "en", context: str = "", judge: str = "claude",
                 model: str | None = None, effort: str = "low", log=None, base_url: str | None = None):
        self.id = uuid.uuid4().hex[:12]
        self.lang = lang
        self.log = log
        self.lock = threading.Lock()
        self.samples: list[dict] = []
        self.audio = np.zeros(0, np.float32)
        self.audio_start: float | None = None  # session time of self.audio[0]
        self.windows: list[dict] = []
        self.judgments: list[dict] = []
        self.judge = make_judge(judge, model=model, effort=effort, lang=lang, context=context, max_turns=24,
                                log=log, base_url=base_url)
        self.last_seen = time.time()

    def _add_audio(self, audio_b64: str, sr: int, t0: float) -> None:
        pcm = np.frombuffer(base64.b64decode(audio_b64), dtype="<i2").astype(np.float32) / 32768.0
        if sr != AUDIO_SR and len(pcm):
            pcm = librosa.resample(pcm, orig_sr=sr, target_sr=AUDIO_SR)
        if self.audio_start is None:
            self.audio_start = t0
        self.audio = np.concatenate([self.audio, pcm])
        excess = len(self.audio) - int(AUDIO_KEEP_S * AUDIO_SR)
        if excess > 0:
            self.audio = self.audio[excess:]
            self.audio_start += excess / AUDIO_SR

    def add_window(self, start: float, end: float, samples: list[dict], aspect: float = 16 / 9,
                   audio: str | None = None, audio_sr: int = AUDIO_SR, audio_t0: float | None = None,
                   transcript: list[dict] | None = None) -> dict:
        with self.lock:
            self.last_seen = time.time()
            clean = [c for c in (_clean_sample(s) for s in samples) if c is not None and start <= c["t"] < end]
            clean.sort(key=lambda s: s["t"])
            self.samples = [s for s in self.samples if s["t"] >= end - SAMPLES_KEEP_S] + clean

            track = None
            if audio:
                self._add_audio(audio, audio_sr, start if audio_t0 is None else audio_t0)
            if len(self.audio) >= AUDIO_SR:
                track = analyze_samples(self.audio, AUDIO_SR, offset=self.audio_start)

            window_samples = [dict(s, hands=[dict(h) for h in s["hands"]]) for s in clean]
            # Phones may deliver only a few frames per second: widen the tracking gap to match.
            dt = float(np.median(np.diff([s["t"] for s in clean]))) if len(clean) > 2 else 0.1
            add_hand_motion(window_samples, aspect, max_gap=max(0.35, 2.5 * dt))
            segments = [Segment(float(x.get("start", start)), float(x.get("end", end)), str(x["text"]).strip())
                        for x in (transcript or []) if str(x.get("text", "")).strip()]
            baseline = clip_baseline(self.samples)
            feats = window_features(len(self.windows), None, (start, end), window_samples, track, segments,
                                    baseline)

            clip = {"live": True, "baseline": baseline} if not self.windows else None
            judgment, self.judge = judge_or_fallback(self.judge, feats, clip, self.lang,
                                                     lambda h: hand_label(h, self.lang), self.log)
            self.windows.append(feats)
            self.judgments.append(judgment)
            return {"window": feats, "judgment": judgment}


class SessionStore:
    """Thread-safe registry of live sessions that expires idle ones."""

    def __init__(self, idle_timeout: float = 1800.0):
        self._sessions: dict[str, LiveSession] = {}
        self._lock = threading.Lock()
        self.idle_timeout = idle_timeout

    def create(self, **kwargs) -> LiveSession:
        session = LiveSession(**kwargs)
        with self._lock:
            now = time.time()
            for sid in [k for k, s in self._sessions.items() if now - s.last_seen > self.idle_timeout]:
                del self._sessions[sid]
            self._sessions[session.id] = session
        return session

    def get(self, sid: str) -> LiveSession | None:
        with self._lock:
            return self._sessions.get(sid)

    def delete(self, sid: str) -> None:
        with self._lock:
            self._sessions.pop(sid, None)
