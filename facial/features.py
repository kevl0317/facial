"""Cut the clip into windows and summarise each one as five fields:
scene, speaker, subtitle, voice and gesture.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict

import numpy as np

from .audio import AudioTrack, voice_features
from .transcript import Segment

FILLERS = {"um", "uh", "erm", "hmm", "like", "basically", "actually", "嗯", "啊", "呃", "那个", "就是"}
RELATIVE_FACE = ("smile", "frown", "brow_furrow", "brow_raise", "lip_press")
_LATIN_WORD = re.compile(r"[A-Za-z0-9']+")
_CJK_CHAR = re.compile(r"[㐀-鿿]")
JITTER_FLOOR = 0.3  # hand-lengths/s of apparent motion from landmark noise on a still hand
BEAT_SPEED = 1.5  # hand-lengths/s peak that counts as a beat gesture


def build_windows(duration: float, segments: list[Segment], target: float = 5.0,
                  min_len: float = 2.5, max_len: float = 9.0) -> list[tuple[float, float]]:
    """Contiguous windows of ~target seconds whose boundaries snap to subtitle ends."""
    if duration <= max_len:
        return [(0.0, duration)]
    cands = sorted({round(s.end, 3) for s in segments if 0 < s.end < duration})
    bounds = [0.0]

    def pick(start: float, lo: float, hi: float) -> float | None:
        options = [c for c in cands if lo <= c - start <= hi and duration - c >= min_len]
        return min(options, key=lambda c: abs(c - start - target)) if options else None

    while duration - bounds[-1] > max_len:
        start = bounds[-1]
        nxt = pick(start, min_len, max_len)
        bounds.append(nxt if nxt is not None else start + target)
    tail = duration - bounds[-1]
    if tail > 1.4 * target:
        nxt = pick(bounds[-1], min_len, tail)
        bounds.append(nxt if nxt is not None else bounds[-1] + tail / 2)
    bounds.append(duration)
    return list(zip(bounds[:-1], bounds[1:]))


def assign_segments(windows: list[tuple[float, float]], segments: list[Segment]) -> list[list[Segment]]:
    """Each subtitle segment belongs to the window containing its midpoint."""
    out: list[list[Segment]] = [[] for _ in windows]
    starts = [w[0] for w in windows]
    for seg in segments:
        mid = (seg.start + seg.end) / 2
        idx = max(0, int(np.searchsorted(starts, mid, side="right")) - 1)
        out[idx].append(seg)
    return out


def add_hand_motion(samples: list[dict], aspect: float, max_gap: float = 0.35) -> None:
    """Track hands across samples (nearest neighbour) and annotate `track` and `speed`.

    Speed is in hand-lengths per second. Tracking by position rather than by the
    left/right label matters: MediaPipe sometimes gives two hands the same label.
    """
    next_id = 0
    live: list[dict] = []  # recent hands: {"t", "anchor", "size", "id"}
    for s in samples:
        t = s["t"]
        live = [p for p in live if t - p["t"] <= max_gap]
        used: set[int] = set()
        for h in sorted(s["hands"], key=lambda h: -h["score"]):
            best, best_d = None, 0.0
            for p in live:
                if p["id"] in used:
                    continue
                d = math.hypot((h["anchor"][0] - p["anchor"][0]) * aspect, h["anchor"][1] - p["anchor"][1])
                if d < 2.5 * max(h["size"], p["size"]) and (best is None or d < best_d):
                    best, best_d = p, d
            if best is None:
                h["track"], h["speed"] = next_id, 0.0
                next_id += 1
            else:
                h["track"] = best["id"]
                h["speed"] = round(best_d / max(h["size"], 1e-3) / max(t - best["t"], 1e-3), 3)
            used.add(h["track"])
        live = [p for p in live if p["id"] not in used] + [
            {"t": t, "anchor": h["anchor"], "size": h["size"], "id": h["track"]} for h in s["hands"]]

    # Smooth each track over 3 samples and remove the landmark-jitter floor, so a
    # resting hand reads as ~0 rather than as small constant motion.
    by_track: dict[int, list[dict]] = defaultdict(list)
    for s in samples:
        for h in s["hands"]:
            by_track[h["track"]].append(h)
    for hands in by_track.values():
        raw = [h["speed"] for h in hands]
        for i, h in enumerate(hands):
            window = raw[max(0, i - 1):i + 2]
            h["speed"] = round(max(0.0, sum(window) / len(window) - JITTER_FLOOR), 3)


def primary_hand(sample: dict) -> dict | None:
    """The hand that is doing the most: confident and moving."""
    if not sample["hands"]:
        return None
    return max(sample["hands"], key=lambda h: h["score"] * (1.0 + min(h.get("speed", 0.0), 4.0)))


def hand_key(h: dict) -> str:
    """Hand state label used for grouping; `side` is the speaker's left/right."""
    return f'{h["side"]} {h["shape"]} {h["facing"]} {h["axis"]}'


def shot_type(face_size: float | None) -> str:
    if not face_size:
        return "cutaway"
    if face_size >= 0.33:
        return "close-up"
    if face_size >= 0.12:
        return "medium"
    return "wide"


def clip_baseline(samples: list[dict]) -> dict:
    faces = [s["face"] for s in samples if s["face"]]
    if not faces:
        return {}

    def med(key):
        vals = [f[key] for f in faces if key in f]
        return round(float(np.median(vals)), 3) if vals else None

    return {"yaw": med("yaw"), "pitch": med("pitch"), "face_size": med("size"),
            **{k: med(k) for k in RELATIVE_FACE}}


def _hand_to_face(sample: dict) -> bool:
    face = sample["face"]
    if not face:
        return False
    x0, y0, x1, y1 = face["bbox"]
    pad_x, pad_y = 0.25 * (x1 - x0), 0.25 * (y1 - y0)
    return any(x0 - pad_x <= h["anchor"][0] <= x1 + pad_x and y0 - pad_y <= h["anchor"][1] <= y1 + pad_y
               for h in sample["hands"])


def _gesture(S: list[dict], start: float) -> dict:
    out: dict = {}
    n = len(S)
    with_hands = [s for s in S if s["hands"]]
    out["hands_visible"] = round(len(with_hands) / n, 2)

    # Dominant hand state, weighted by confidence and motion.
    weights: dict[str, float] = defaultdict(float)
    best: dict[str, tuple[float, float, dict]] = {}
    for s in with_hands:
        for h in s["hands"]:
            k = hand_key(h)
            weights[k] += h["score"] * (1.0 + min(h.get("speed", 0.0), 4.0))
            if k not in best or h["score"] > best[k][0]:
                best[k] = (h["score"], s["t"], h)
    if weights:
        k = max(weights, key=weights.get)
        score, t, h = best[k]
        frac = sum(1 for s in S if any(hand_key(x) == k for x in s["hands"])) / n
        out["dominant"] = {"hand": h["side"], "shape": h["shape"], "facing": h["facing"], "axis": h["axis"],
                           "score": round(score, 2), "share": round(frac, 2), "at_s": round(t - start, 1)}

        # The other hand (a different track), when it is up for at least a fifth of the window.
        main = h.get("track")
        others: dict[str, float] = defaultdict(float)
        example: dict[str, dict] = {}
        for s in with_hands:
            for x in s["hands"]:
                if x.get("track") != main:
                    k2 = hand_key(x)
                    others[k2] += x["score"] * (1.0 + min(x.get("speed", 0.0), 4.0))
                    example.setdefault(k2, x)
        if others:
            k2 = max(others, key=others.get)
            share2 = sum(1 for s in S if any(hand_key(x) == k2 and x.get("track") != main for x in s["hands"])) / n
            if share2 >= 0.2:
                x = example[k2]
                out["second"] = {"hand": x["side"], "shape": x["shape"], "facing": x["facing"], "axis": x["axis"],
                                 "share": round(share2, 2)}
    out["two_hands"] = round(sum(1 for s in S if len(s["hands"]) >= 2) / n, 2)

    # Short sequence of what the primary hand did, as runs >= 0.4 s.
    runs: list[list] = []
    for s in S:
        h = primary_hand(s)
        label = hand_key(h) if h else "no hands"
        if runs and runs[-1][0] == label:
            runs[-1][2] = s["t"]
        else:
            runs.append([label, s["t"], s["t"]])
    seq = [{"from_s": round(a - start, 1), "to_s": round(b - start, 1), "state": lab}
           for lab, a, b in runs if b - a >= 0.4]
    out["sequence"] = seq[:5]

    speeds = [h.get("speed", 0.0) for s in with_hands for h in s["hands"]]
    out["energy"] = round(float(np.mean(speeds)), 2) if speeds else 0.0
    beats = 0
    tracks = {h.get("track") for s in S for h in s["hands"]}
    by_track: dict[int, list[float]] = defaultdict(list)
    for s in S:
        present = {h.get("track"): h.get("speed", 0.0) for h in s["hands"]}
        for track in tracks:
            by_track[track].append(present.get(track, 0.0))
    dt = float(np.median(np.diff([s["t"] for s in S]))) if n > 1 else 0.1
    min_gap = max(1, math.ceil(0.35 / max(dt, 1e-3)))
    for series in by_track.values():
        last_peak = -min_gap
        for i in range(1, len(series) - 1):
            if (series[i] > BEAT_SPEED and series[i] >= series[i - 1] and series[i] > series[i + 1]
                    and i - last_peak >= min_gap):
                beats += 1
                last_peak = i
    out["beats"] = beats
    out["hand_to_face"] = round(sum(_hand_to_face(s) for s in S) / n, 2)
    return out


def _face_and_head(S: list[dict], base: dict) -> tuple[dict, dict]:
    faces = [(s["t"], s["face"]) for s in S if s["face"]]
    if len(faces) < 3:
        return {}, {}
    ts = np.array([t for t, _ in faces])
    F = [f for _, f in faces]

    def arr(key):
        return np.array([f.get(key, 0.0) for f in F], dtype=float)

    face = {k: round(float(arr(k).mean()), 2)
            for k in ("smile", "frown", "brow_furrow", "brow_raise", "lip_press", "eye_wide", "squint")}
    # Blendshapes carry per-face bias (a broad smile can read as a furrowed brow), so
    # changes against the speaker's own clip median are the more trustworthy signal.
    deltas = {k: round(face[k] - base[k], 2) for k in RELATIVE_FACE if base.get(k) is not None}
    if deltas:
        face["vs_baseline"] = deltas

    blink = arr("blink")
    blinks, closed = 0, False
    for v in blink:  # hysteresis so a half-closed eye doesn't double count
        if not closed and v > 0.5:
            blinks, closed = blinks + 1, True
        elif closed and v < 0.35:
            closed = False
    span = max(ts[-1] - ts[0], 0.5)
    face["blink_per_min"] = round(blinks / span * 60, 0)

    yaw, pitch = arr("yaw"), arr("pitch")
    base_yaw = base.get("yaw") or 0.0
    gaze_away = (np.abs(yaw - base_yaw) > 15) | (arr("gaze_side") > 0.5)
    centred = pitch - pitch.mean()
    crossings = int(np.sum(np.abs(np.diff(np.sign(centred[np.abs(centred) > 2.5]))) > 0))
    head = {"yaw_std": round(float(yaw.std()), 1), "pitch_std": round(float(pitch.std()), 1),
            "nods": crossings // 2, "gaze_away": round(float(gaze_away.mean()), 2),
            "yaw_vs_baseline": round(float(yaw.mean() - base_yaw), 1)}
    return face, head


def _posture(S: list[dict]) -> dict:
    poses = [s["pose"] for s in S if s["pose"]]
    if not poses:
        return {}
    out = {"shoulder_tilt": round(float(np.mean([p["tilt"] for p in poses])), 1)}
    arms = [p["arms_open"] for p in poses if "arms_open" in p]
    if arms:
        out["arms_open"] = round(float(np.mean(arms)), 2)
    return out


def _speaker(S: list[dict], audio: AudioTrack | None, voiced: float) -> dict:
    faces = [s for s in S if s["face"]]
    face_vis = len(faces) / len(S)
    out: dict = {"face_visible": round(face_vis, 2)}
    jaw = np.array([s["face"].get("jaw", 0.0) for s in faces])
    mouth = float(jaw.std()) if len(jaw) > 2 else 0.0
    out["mouth_activity"] = round(mouth, 3)
    sync = None
    if audio is not None and len(faces) >= 8 and jaw.std() > 1e-4:
        db = audio.db_at(np.array([s["t"] for s in faces]))
        if db.std() > 1e-3:
            sync = float(np.corrcoef(jaw, db)[0, 1])
            out["lip_sync"] = round(sync, 2)

    if voiced < 0.15:
        state = "silent"
    elif face_vis < 0.3:
        state = "offscreen"
    elif (sync is not None and sync > 0.25) or mouth > 0.035:
        state = "target"
    else:
        state = "listening"
    out["state"] = state
    return out


def window_features(index: int, total: int, window: tuple[float, float], samples: list[dict],
                    audio: AudioTrack | None, segments: list[Segment], base: dict) -> dict:
    start, end = window
    S = [s for s in samples if start <= s["t"] < end] or [{"t": start, "cut": False, "faces": 0,
                                                              "face": None, "hands": [], "pose": None}]
    text = " ".join(s.text for s in segments).strip()

    sizes = [s["face"]["size"] for s in S if s["face"]]
    scene = {"shot": shot_type(float(np.median(sizes)) if sizes else None),
             "faces": max(s["faces"] for s in S), "cuts": sum(s["cut"] for s in S)}

    voice = voice_features(audio, start, end) if audio is not None else {"voiced_frac": 0.0}
    words = text.split()
    speech_time = voice.get("voiced_frac", 0.0) * (end - start)
    # CJK text has no spaces: count ~1.6 characters per word.
    n_words = len(_LATIN_WORD.findall(text)) + len(_CJK_CHAR.findall(text)) / 1.6
    if n_words and speech_time > 0.5:
        voice["speech_rate_wps"] = round(n_words / speech_time, 1)
    lowered = [w.strip(",.!?;:").lower() for w in words]
    voice["fillers"] = sum(w in FILLERS for w in lowered) + sum(text.count(f) for f in FILLERS if not f.isascii())

    gesture = _gesture(S, start)
    face, head = _face_and_head(S, base)
    if face:
        gesture["face"] = face
    if head:
        gesture["head"] = head
    posture = _posture(S)
    if posture:
        gesture["posture"] = posture

    return {
        "window": index + 1,
        "of": total,
        "start": round(start, 2),
        "end": round(end, 2),
        "scene": scene,
        "speaker": _speaker(S, audio, voice.get("voiced_frac", 0.0)),
        "subtitle": text,
        "voice": voice,
        "gesture": gesture,
    }
