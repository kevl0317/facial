"""Voice measurements: loudness, pitch, voicing and pauses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
from scipy.ndimage import median_filter

HOP_S = 0.01
FMIN, FMAX = 65.0, 450.0


@dataclass
class AudioTrack:
    times: np.ndarray  # frame centres, seconds
    db: np.ndarray  # RMS loudness, dBFS
    f0: np.ndarray  # pitch in Hz, NaN where unvoiced / unreliable
    speech: np.ndarray  # bool voice-activity mask
    base_db: float  # clip median loudness while speaking
    base_f0: float  # clip median pitch

    def db_at(self, t: np.ndarray) -> np.ndarray:
        return np.interp(t, self.times, self.db)


def analyze_audio(wav: Path, sample_rate: int = 16000) -> AudioTrack:
    y, sr = librosa.load(str(wav), sr=sample_rate, mono=True)
    hop = int(sr * HOP_S)
    frame = 1024
    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
    db = 20 * np.log10(rms + 1e-10)
    times = librosa.times_like(rms, sr=sr, hop_length=hop)

    # Voice activity: energy gate relative to the loud end of the clip, smoothed so
    # short consonant dips don't count as pauses.
    ref = np.percentile(db, 95)
    speech = db > max(ref - 28.0, -55.0)
    speech = median_filter(speech.astype(np.uint8), size=15).astype(bool)

    f0 = librosa.yin(y, fmin=FMIN, fmax=FMAX, sr=sr, frame_length=frame, hop_length=hop)[: len(db)]
    f0 = median_filter(f0, size=5)
    valid = speech & (f0 > FMIN * 1.05) & (f0 < FMAX * 0.95)
    # Drop octave jumps: frames far from their local median are unreliable.
    semis = 12 * np.log2(np.maximum(f0, 1.0))
    local = median_filter(semis, size=25)
    valid &= np.abs(semis - local) < 3.0
    f0 = np.where(valid, f0, np.nan)

    base_db = float(np.median(db[speech])) if speech.any() else float(np.median(db))
    base_f0 = float(np.nanmedian(f0)) if np.isfinite(f0).any() else float("nan")
    return AudioTrack(times=times, db=db, f0=f0, speech=speech, base_db=base_db, base_f0=base_f0)


def voice_features(track: AudioTrack, start: float, end: float) -> dict:
    sel = (track.times >= start) & (track.times < end)
    if not sel.any():
        return {"voiced_frac": 0.0}
    speech = track.speech[sel]
    db = track.db[sel]
    f0 = track.f0[sel]
    out: dict = {"voiced_frac": round(float(speech.mean()), 2)}

    if speech.any():
        out["loudness_rel_db"] = round(float(np.median(db[speech]) - track.base_db), 1)
    f0_ok = f0[np.isfinite(f0)]
    if len(f0_ok) >= 10 and np.isfinite(track.base_f0):
        semis = 12 * np.log2(f0_ok / track.base_f0)
        out["pitch_rel_st"] = round(float(np.median(semis)), 1)
        out["pitch_var_st"] = round(float(np.std(semis)), 1)

    # Pauses: silent runs >= 0.3 s bounded by speech on both sides.
    pauses = []
    run = 0
    seen_speech = False
    for s in speech:
        if s:
            if seen_speech and run * HOP_S >= 0.3:
                pauses.append(run * HOP_S)
            seen_speech, run = True, 0
        else:
            run += 1
    out["pauses"] = len(pauses)
    out["longest_pause_s"] = round(max(pauses), 2) if pauses else 0.0
    return out
