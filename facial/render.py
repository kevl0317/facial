"""Draw the analysis HUD onto video frames with Pillow."""

from __future__ import annotations

import bisect
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .features import shot_type
from .media import VideoInfo, VideoWriter

STRINGS = {
    "en": {
        "left": "Left hand", "right": "Right hand", "verdict": "Verdict", "fusion": "subs + voice + gesture",
        "confidence": "Confident", "focus": "Focused", "tension": "Tense", "intent": "Intent",
        "arc": "Emotion arc", "quote": "QUOTE", "vision": "VISION  gesture model",
        "footer": "Gesture: MediaPipe per-frame  ·  Reading & scores: {judge}  ·  Not calibrated, demo only",
        "rules": "rules",
    },
    "zh": {
        "left": "左手", "right": "右手", "verdict": "综合判定", "fusion": "字幕+声音+动作",
        "confidence": "自信", "focus": "专注", "tension": "紧张", "intent": "意图",
        "arc": "情绪弧", "quote": "引语", "vision": "VISION  手势模型",
        "footer": "动作 MediaPipe 逐帧检测  ·  解读和评分 {judge}  ·  未经人工校准 仅供演示",
        "rules": "规则",
    },
}
SHAPES = {
    "en": {"open_palm": "open palm", "fist": "fist", "pointing": "pointing", "two_fingers": "two fingers",
           "thumb": "thumb out", "pinch": "pinch", "relaxed": "relaxed"},
    "zh": {"open_palm": "张开手掌", "fist": "握拳", "pointing": "指点", "two_fingers": "两指",
           "thumb": "竖拇指", "pinch": "捏合", "relaxed": "放松"},
}
FACINGS = {
    "en": {"palm_up": "palm up", "palm_down": "palm down", "palm_out": "palm out", "back_out": "back out",
           "palm_side": "sideways"},
    "zh": {"palm_up": "掌心向上", "palm_down": "掌心向下", "palm_out": "掌心朝外", "back_out": "手背朝外",
           "palm_side": "侧掌"},
}
AXES = {
    "en": {"horizontal": "horizontal", "up": "raised", "down": "lowered"},
    "zh": {"horizontal": "横", "up": "竖起", "down": "下垂"},
}

TEXT = (236, 239, 246, 255)
MUTED = (165, 175, 196, 255)
BLUE = (74, 144, 255, 255)
ORANGE = (255, 159, 67, 255)
PANEL_BG = (12, 18, 34, 190)
PANEL_EDGE = (84, 132, 255, 220)
SHADOW = (0, 0, 0, 170)

HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
              (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]


def hand_label(h: dict, lang: str = "en") -> str:
    side = STRINGS[lang][h.get("side") or h.get("hand")]
    shape = SHAPES[lang].get(h["shape"], h["shape"])
    detail = FACINGS[lang][h["facing"]] if h["shape"] in ("open_palm", "relaxed") else AXES[lang][h["axis"]]
    return f"{side} · {shape}（{detail}）" if lang == "zh" else f"{side} · {shape} ({detail})"


# --------------------------------------------------------------------------- fonts

_FONT_PATHS = {
    "sans": ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf",
             "/Library/Fonts/Arial.ttf", "C:/Windows/Fonts/arial.ttf", "DejaVuSans.ttf"],
    "bold": ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
             "/System/Library/Fonts/Supplemental/Arial Bold.ttf", "C:/Windows/Fonts/arialbd.ttf",
             "DejaVuSans-Bold.ttf"],
    "mono": ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "/System/Library/Fonts/Menlo.ttc",
             "C:/Windows/Fonts/consola.ttf", "DejaVuSansMono.ttf"],
    "cjk": ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Medium.ttc", "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"],
}
_CJK = re.compile(r"[\u3000-\u303f\u3400-\u9fff\uff00-\uffef]")


def _find_font(role: str) -> str | None:
    for candidate in _FONT_PATHS[role]:
        try:
            ImageFont.truetype(candidate, 12)
            return candidate
        except OSError:
            continue
    return None


@lru_cache(maxsize=None)
def _load(path: str | None, size: int):
    if path is None:
        return ImageFont.load_default(size=size)
    return ImageFont.truetype(path, size)


class Fonts:
    def __init__(self, custom: str | None = None):
        self.paths = {role: _find_font(role) for role in _FONT_PATHS}
        if custom:
            for role in ("sans", "bold", "cjk"):
                self.paths[role] = custom
        self.has_cjk = self.paths["cjk"] is not None

    def get(self, role: str, size: float, text: str = ""):
        path = self.paths[role]
        if self.has_cjk and (role != "mono") and _CJK.search(text):
            path = self.paths["cjk"]
        return _load(path, max(8, int(round(size))))


def wrap(draw: ImageDraw.ImageDraw, text: str, font, max_w: float, max_lines: int) -> list[str]:
    tokens = re.findall(r"[\u3000-\u9fff\uff00-\uffef]|[^\s\u3000-\u9fff\uff00-\uffef]+\s*|\s+", text)
    lines, cur = [], ""
    for tok in tokens:
        if not cur or draw.textlength(cur + tok, font=font) <= max_w:
            cur += tok
        else:
            lines.append(cur.rstrip())
            cur = tok.lstrip()
    if cur.strip():
        lines.append(cur.rstrip())
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        last = lines[-1]
        while last and draw.textlength(last + "…", font=font) > max_w:
            last = last[:-1]
        lines[-1] = last.rstrip() + "…"
    return lines


def _smoothstep(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def _mix(c0, c1, a):
    return tuple(int(round(c0[i] + (c1[i] - c0[i]) * a)) for i in range(len(c0)))


def valence_colour(v: float):
    neutral = (110, 130, 175, 255)
    return _mix(neutral, (70, 165, 255, 255), v) if v >= 0 else _mix(neutral, (236, 92, 80, 255), -v)


# --------------------------------------------------------------------------- overlay

class Overlay:
    def __init__(self, width: int, height: int, samples: list[dict], windows: list[dict],
                 judgments: list[dict], segments: list, lang: str = "en", skeleton: bool = False,
                 font: str | None = None):
        self.W, self.H = width, height
        self.s = height / 720
        self.samples = samples
        self.sample_t = [x["t"] for x in samples]
        self.windows = windows
        self.win_starts = [w["start"] for w in windows]
        self.judgments = judgments
        self.segments = segments
        self.seg_starts = [seg.start for seg in segments]
        self.lang = lang
        self.skeleton = skeleton
        self.fonts = Fonts(font)
        self.str = STRINGS[lang]
        sources = {j["source"] for j in judgments}
        self.judge_name = "Claude" if "claude" in sources else self.str["rules"]

    # -- lookups -------------------------------------------------------------
    def _window(self, t: float) -> int:
        return max(0, bisect.bisect_right(self.win_starts, t) - 1)

    def _subtitle(self, t: float) -> str:
        i = bisect.bisect_right(self.seg_starts, t) - 1
        if i >= 0 and self.segments[i].start <= t < self.segments[i].end:
            return self.segments[i].text
        return ""

    def _nearest_sample(self, t: float) -> dict | None:
        if not self.samples:
            return None
        i = bisect.bisect_left(self.sample_t, t)
        cands = [j for j in (i - 1, i) if 0 <= j < len(self.samples)]
        return self.samples[min(cands, key=lambda j: abs(self.sample_t[j] - t))]

    def _anchor(self, track: int, t: float):
        """Anchor of hand `track`, interpolated between the samples around t."""
        i = bisect.bisect_right(self.sample_t, t)

        def find(indices):
            for j in indices:
                s = self.samples[j]
                if abs(s["t"] - t) > 0.3:
                    return None
                for h in s["hands"]:
                    if h.get("track") == track:
                        return s["t"], h["anchor"]
            return None

        before = find(range(i - 1, max(-1, i - 6), -1))
        after = find(range(i, min(len(self.samples), i + 5)))
        if before and after and after[0] > before[0]:
            a = (t - before[0]) / (after[0] - before[0])
            return [before[1][k] + (after[1][k] - before[1][k]) * a for k in (0, 1)]
        found = before or after
        return found[1] if found else None

    def _live_hand(self, t: float) -> dict | None:
        """The most active hand over the last 0.8 s, with its most common state over the
        last 0.4 s, so the label neither flickers nor jumps between hands."""
        i0 = bisect.bisect_left(self.sample_t, t - 0.8)
        i1 = bisect.bisect_right(self.sample_t, t)
        weights: dict[int, float] = {}
        for s in self.samples[i0:i1]:
            for h in s["hands"]:
                weights[h["track"]] = weights.get(h["track"], 0.0) + h["score"] * (1.0 + min(h["speed"], 4.0))
        if not weights:
            return None
        track = max(weights, key=weights.get)
        same = [h for s in self.samples[i0:i1] if s["t"] >= t - 0.4 for h in s["hands"] if h["track"] == track]
        if not same:
            return None
        side, shape, facing, axis = Counter(
            (h["side"], h["shape"], h["facing"], h["axis"]) for h in same).most_common(1)[0][0]
        return {"side": side, "shape": shape, "facing": facing, "axis": axis,
                "score": float(np.mean([h["score"] for h in same])), "anchor": self._anchor(track, t)}

    def _value(self, k: int, key: str, t: float) -> float:
        cur = self.judgments[k][key]
        if k == 0:
            return cur
        prev = self.judgments[k - 1][key]
        return prev + (cur - prev) * _smoothstep((t - self.windows[k]["start"]) / 0.6)

    # -- drawing -------------------------------------------------------------
    def draw(self, frame_bgr: np.ndarray, t: float) -> np.ndarray:
        base = Image.fromarray(np.ascontiguousarray(frame_bgr[:, :, ::-1])).convert("RGBA")
        layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        k = self._window(t)
        sample = self._nearest_sample(t)

        if self.skeleton and sample:
            self._draw_skeleton(d, sample)
        self._draw_hud(d, t, k, sample)
        self._draw_gesture(d, t)
        if self.judgments:
            self._draw_panel(d, t, k)
            self._draw_reading(d, k)
        self._draw_subtitle(d, t)
        self._draw_footer(d)

        out = Image.alpha_composite(base, layer).convert("RGB")
        return np.ascontiguousarray(np.asarray(out)[:, :, ::-1])

    def _text(self, d, xy, text, font, fill=TEXT, stroke=0):
        if stroke:
            d.text(xy, text, font=font, fill=fill, stroke_width=stroke, stroke_fill=SHADOW)
        else:
            d.text((xy[0] + max(1, self.s), xy[1] + max(1, self.s)), text, font=font, fill=SHADOW)
            d.text(xy, text, font=font, fill=fill)

    def _draw_hud(self, d, t, k, sample):
        s = self.s
        d.rectangle([0, 0, self.W, 24 * s], fill=(0, 0, 0, 90))
        shot = shot_type(sample["face"]["size"] if sample and sample["face"] else None)
        hud = f"t={t:05.1f}   WIN {k + 1}/{max(1, len(self.windows))}   shot={shot}"
        self._text(d, (12 * s, 5 * s), hud, self.fonts.get("mono", 12 * s), fill=(220, 225, 235, 255))

    def _draw_gesture(self, d, t):
        live = self._live_hand(t)
        if not live:
            return
        s = self.s
        x, y = 22 * s, 42 * s
        label = hand_label(live, self.lang)
        font = self.fonts.get("bold", 22 * s, label)
        self._text(d, (x, y), label, font, stroke=max(1, round(2 * s)))
        tw = d.textlength(label, font=font)

        y2 = y + 32 * s
        d.line([(x, y2 + 7 * s), (x + 16 * s, y2 + 7 * s)], fill=BLUE, width=max(1, round(2 * s)))
        vis = f"{self.str['vision']}  {live['score']:.2f}"
        self._text(d, (x + 22 * s, y2), vis, self.fonts.get("sans", 11 * s, vis), fill=(190, 210, 255, 255))

        anchor = live["anchor"]
        if anchor is not None:
            ax, ay = anchor[0] * self.W, anchor[1] * self.H
            sx, sy = x + tw + 10 * s, y + 13 * s
            d.line([(sx, sy), (ax, ay)], fill=(255, 255, 255, 185), width=max(1, round(1.2 * s)))
            r = 6 * s
            d.ellipse([ax - r, ay - r, ax + r, ay + r], outline=(255, 255, 255, 230), width=max(1, round(2 * s)))
            r2 = 2 * s
            d.ellipse([ax - r2, ay - r2, ax + r2, ay + r2], fill=(255, 255, 255, 230))

    def _panel_geometry(self):
        s = self.s
        pw, ph = 252 * s, 164 * s
        x0 = self.W - pw - 20 * s
        y0 = self.H - 100 * s - ph
        return x0, y0, pw, ph

    def _draw_panel(self, d, t, k):
        s = self.s
        x0, y0, pw, ph = self._panel_geometry()
        j = self.judgments[k]
        d.rounded_rectangle([x0, y0, x0 + pw, y0 + ph], radius=8 * s, fill=PANEL_BG, outline=PANEL_EDGE,
                            width=max(1, round(1.5 * s)))
        title = self.str["verdict"]
        tfont = self.fonts.get("bold", 15 * s, title)
        d.text((x0 + 12 * s, y0 + 9 * s), title, font=tfont, fill=TEXT)
        sub = f"{self.judge_name} · {self.str['fusion']}"
        d.text((x0 + 18 * s + d.textlength(title, font=tfont), y0 + 13 * s), sub,
               font=self.fonts.get("sans", 10 * s, sub), fill=(140, 175, 255, 255))

        bars = (("confidence", BLUE), ("focus", BLUE), ("tension", ORANGE))
        label_fonts = {key: self.fonts.get("sans", 12 * s, self.str[key]) for key, _ in bars}
        label_w = max(d.textlength(self.str[key], font=f) for key, f in label_fonts.items())
        bx0 = x0 + 12 * s + label_w + 10 * s
        bx1 = max(x0 + pw - 52 * s, bx0 + 4 * s)  # fonts have a minimum size, so tiny frames can squeeze this
        for i, (key, colour) in enumerate(bars):
            yy = y0 + 40 * s + i * 21 * s
            label = self.str[key]
            d.text((x0 + 12 * s, yy), label, font=label_fonts[key], fill=MUTED)
            v = self._value(k, key, t)
            cy = yy + 8 * s
            d.rounded_rectangle([bx0, cy - 2.5 * s, bx1, cy + 2.5 * s], radius=2.5 * s, fill=(255, 255, 255, 40))
            if v > 0.005:
                d.rounded_rectangle([bx0, cy - 2.5 * s, bx0 + (bx1 - bx0) * v, cy + 2.5 * s],
                                    radius=2.5 * s, fill=colour)
            d.text((bx1 + 8 * s, yy + 1 * s), f"{v:.2f}", font=self.fonts.get("mono", 11 * s), fill=colour)

        yy = y0 + 40 * s + 3 * 21 * s + 4 * s
        cert = f"{j['intent_certainty']:.2f}"
        mono = self.fonts.get("mono", 11 * s)
        room = pw - 24 * s - d.textlength(cert, font=mono) - 8 * s
        intent_text = f"{self.str['intent']} → {j['intent']}"
        ifont = self.fonts.get("sans", 13 * s, intent_text)
        intent_line = wrap(d, intent_text, ifont, room, 1)[0]
        d.text((x0 + 12 * s, yy), intent_line, font=ifont, fill=TEXT)
        d.text((x0 + pw - 12 * s - d.textlength(cert, font=mono), yy + 2 * s), cert, font=mono, fill=BLUE)

        yy += 26 * s
        arc = self.str["arc"]
        arc_font = self.fonts.get("sans", 10 * s, arc)
        d.text((x0 + 12 * s, yy), arc, font=arc_font, fill=MUTED)
        size, gap = 10 * s, 4 * s
        ax0 = max(bx0, x0 + 12 * s + d.textlength(arc, font=arc_font) + 10 * s)
        n_fit = max(1, int((x0 + pw - 12 * s - ax0) // (size + gap)))
        first = max(0, k + 1 - n_fit)
        for n, idx in enumerate(range(first, k + 1)):
            qx = ax0 + n * (size + gap)
            colour = valence_colour(self.judgments[idx]["valence"])
            outline = (255, 255, 255, 255) if idx == k else None
            d.rounded_rectangle([qx, yy + 1 * s, qx + size, yy + 1 * s + size], radius=2 * s, fill=colour,
                                outline=outline, width=max(1, round(1.2 * s)))

    def _draw_reading(self, d, k):
        s = self.s
        j = self.judgments[k]
        if not j["reading"] and not j["quote"]:
            return
        x0, _, _, _ = self._panel_geometry()
        left = 20 * s
        max_w = max(60 * s, min(self.W * 0.56, x0 - left - 24 * s))
        text = f"W{k + 1} > {j['reading']}"
        font = self.fonts.get("sans", 14 * s, text)
        lines = wrap(d, text, font, max_w - 20 * s, 2)
        line_h = 20 * s

        quote = j["quote"]
        chip = self.str["quote"]
        chip_font = self.fonts.get("bold", 10 * s, chip)
        q_text = f"“{quote}”"
        q_font = self.fonts.get("sans", 13 * s, q_text)
        chip_w = d.textlength(chip, font=chip_font) + 10 * s
        q_lines = wrap(d, q_text, q_font, max_w - chip_w - 28 * s, 1) if quote else []

        height = 10 * s + len(lines) * line_h + (22 * s if q_lines else 0) + 6 * s
        bottom = self.H - 100 * s
        top = bottom - height
        width = max([d.textlength(ln, font=font) for ln in lines] +
                    ([chip_w + 8 * s + d.textlength(q_lines[0], font=q_font)] if q_lines else [0])) + 20 * s
        d.rounded_rectangle([left, top, left + width, bottom], radius=6 * s, fill=(0, 0, 0, 125))
        y = top + 8 * s
        for ln in lines:
            d.text((left + 10 * s, y), ln, font=font, fill=TEXT)
            y += line_h
        if q_lines:
            y += 2 * s
            d.rounded_rectangle([left + 10 * s, y, left + 10 * s + chip_w, y + 16 * s], radius=3 * s, fill=ORANGE)
            d.text((left + 15 * s, y + 2 * s), chip, font=chip_font, fill=(40, 24, 8, 255))
            d.text((left + 18 * s + chip_w, y), q_lines[0], font=q_font, fill=TEXT)

    def _draw_subtitle(self, d, t):
        text = self._subtitle(t)
        if not text:
            return
        s = self.s
        font = self.fonts.get("sans", 18 * s, text)
        lines = wrap(d, text, font, self.W * 0.62, 2)
        y = self.H - 36 * s - len(lines) * 24 * s
        for ln in lines:
            w = d.textlength(ln, font=font)
            d.text(((self.W - w) / 2, y), ln, font=font, fill=(255, 255, 255, 255),
                   stroke_width=max(1, round(2 * s)), stroke_fill=(0, 0, 0, 220))
            y += 24 * s

    def _draw_footer(self, d):
        s = self.s
        text = self.str["footer"].format(judge=self.judge_name)
        font = self.fonts.get("sans", 9 * s, text)
        self._text(d, (12 * s, self.H - 16 * s), text, font, fill=(185, 192, 208, 230))

    def _draw_skeleton(self, d, sample):
        W, H, s = self.W, self.H, self.s
        for h in sample["hands"]:
            pts = [(x * W, y * H) for x, y in h["pts"]]
            for a, b in HAND_EDGES:
                d.line([pts[a], pts[b]], fill=(120, 200, 255, 150), width=max(1, round(1.5 * s)))
            for x, y in pts:
                r = 2 * s
                d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255, 200))
        if sample["face"]:
            x0, y0, x1, y1 = sample["face"]["bbox"]
            x0, x1, y0, y1 = x0 * W, x1 * W, y0 * H, y1 * H
            c = 0.18 * (x1 - x0)
            for (px, py, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
                d.line([(px, py), (px + dx * c, py)], fill=(255, 255, 255, 170), width=max(1, round(1.5 * s)))
                d.line([(px, py), (px, py + dy * c)], fill=(255, 255, 255, 170), width=max(1, round(1.5 * s)))


def render_video(info: VideoInfo, out_path: Path, overlay: Overlay, audio_from: Path | None,
                 progress: Callable[[float], None] | None = None) -> None:
    cap = cv2.VideoCapture(str(info.path))
    writer = VideoWriter(out_path, info.width, info.height, info.fps, audio_from)
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(overlay.draw(frame, index / info.fps))
            if progress and info.frames and index % 10 == 0:
                progress(min(1.0, index / info.frames))
            index += 1
    finally:
        cap.release()
        writer.close()


def render_snapshot(info: VideoInfo, png_path: Path, overlay: Overlay, t: float) -> None:
    cap = cv2.VideoCapture(str(info.path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * info.fps))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read a frame at t={t}s")
    cv2.imwrite(str(png_path), overlay.draw(frame, t))
