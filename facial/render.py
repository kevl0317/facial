"""Draw the analysis HUD onto video frames with Pillow.

Same cartoon design as the live HUD (facial/web/hud.js): white stickers with ink
outlines and hard shadows, the bundled Fredoka font, and the shared palette.
"""

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
from .providers import judge_label, judge_tag

STRINGS = {
    "en": {
        "left": "Left hand", "right": "Right hand", "verdict": "Verdict", "confidence": "Confident",
        "focus": "Focused", "tension": "Tense", "intent": "Intent", "arc": "Mood", "vision": "VISION",
        "rules": "Rules", "footer": "MediaPipe · {judge} · demo only",
    },
    "zh": {
        "left": "左手", "right": "右手", "verdict": "综合判定", "confidence": "自信",
        "focus": "专注", "tension": "紧张", "intent": "意图", "arc": "情绪弧", "vision": "VISION",
        "rules": "规则", "footer": "MediaPipe · {judge} · 仅供演示",
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

# Theme (keep in sync with facial/web/style.css and hud.js).
INK = (30, 27, 46)
PAPER = (255, 246, 230)
WHITE = (255, 255, 255)
MUTED = (107, 102, 128)
BLUE = (91, 108, 255)
YELLOW = (255, 200, 61)
CORAL = (255, 107, 107)
MINT = (51, 209, 160)
HAND_COLOURS = (YELLOW, MINT)  # pointer + dot colour per hand label

HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
              (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]


def hand_label(h: dict, lang: str = "en") -> str:
    side = STRINGS[lang][h.get("side") or h.get("hand")]
    shape = SHAPES[lang].get(h["shape"], h["shape"])
    detail = FACINGS[lang][h["facing"]] if h["shape"] in ("open_palm", "relaxed") else AXES[lang][h["axis"]]
    return f"{side} · {shape}（{detail}）" if lang == "zh" else f"{side} · {shape} ({detail})"


# --------------------------------------------------------------------------- fonts

FREDOKA = Path(__file__).resolve().parent / "web" / "fonts" / "Fredoka.ttf"
_CJK_PATHS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc", "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf",
]
_CJK = re.compile(r"[\u3000-\u303f\u3400-\u9fff\uff00-\uffef]")


def _find(paths: list[str]) -> str | None:
    for candidate in paths:
        try:
            ImageFont.truetype(candidate, 12)
            return candidate
        except OSError:
            continue
    return None


@lru_cache(maxsize=None)
def _load(path: str | None, size: int, weight: int):
    if path is None:
        return ImageFont.load_default(size=size)
    font = ImageFont.truetype(path, size)
    if path == str(FREDOKA):
        font.set_variation_by_axes([weight, 100])  # axes: weight, width
    return font


class Fonts:
    def __init__(self, custom: str | None = None):
        self.main = custom or (str(FREDOKA) if FREDOKA.exists() else _find(["DejaVuSans.ttf"]))
        self.cjk = custom or _find(_CJK_PATHS)
        self.has_cjk = self.cjk is not None

    def get(self, weight: int, size: float, text: str = ""):
        path = self.cjk if (self.has_cjk and _CJK.search(text)) else self.main
        return _load(path, max(8, int(round(size))), weight)


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
    """Mood colour: coral (negative) -> yellow (neutral) -> mint (positive)."""
    return _mix(YELLOW, MINT, v) if v >= 0 else _mix(YELLOW, CORAL, -v)


# --------------------------------------------------------------------------- overlay

class Overlay:
    def __init__(self, width: int, height: int, samples: list[dict], windows: list[dict],
                 judgments: list[dict], segments: list, lang: str = "en", skeleton: bool = False,
                 font: str | None = None):
        self.W, self.H = width, height
        self.portrait = width < height
        self.u = max(0.4, width / 360 if self.portrait else height / 720)
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
        ai = next((j for j in judgments if j["source"] != "heuristic"), {"source": "heuristic"})
        self.judge_name = judge_label(ai, self.str["rules"])

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

    def _live_hands(self, t: float) -> list[dict]:
        """Up to two hands: the most active tracks over the last 0.8 s, each with its most common
        state over the last 0.4 s. A second hand needs to show in 40% of recent frames (no flicker)."""
        i0 = bisect.bisect_left(self.sample_t, t - 0.8)
        i1 = bisect.bisect_right(self.sample_t, t)
        weights: dict[int, float] = {}
        for s in self.samples[i0:i1]:
            for h in s["hands"]:
                weights[h["track"]] = weights.get(h["track"], 0.0) + h["score"] * (1.0 + min(h["speed"], 4.0))
        recent = [s for s in self.samples[i0:i1] if s["t"] >= t - 0.4]
        out: list[dict] = []
        for track in sorted(weights, key=weights.get, reverse=True):
            if len(out) == 2:
                break
            same = [h for s in recent for h in s["hands"] if h["track"] == track]
            if not same or (out and len(same) < 0.4 * len(recent)):
                continue
            side, shape, facing, axis = Counter(
                (h["side"], h["shape"], h["facing"], h["axis"]) for h in same).most_common(1)[0][0]
            out.append({"side": side, "shape": shape, "facing": facing, "axis": axis,
                        "score": float(np.mean([h["score"] for h in same])), "anchor": self._anchor(track, t)})
        # Top label = leftmost hand on screen, so the two pointers don't cross.
        return sorted(out, key=lambda h: h["anchor"][0] if h["anchor"] else 2.0)

    def _value(self, k: int, key: str, t: float) -> float:
        cur = self.judgments[k][key]
        if k == 0:
            return cur
        prev = self.judgments[k - 1][key]
        return prev + (cur - prev) * _smoothstep((t - self.windows[k]["start"]) / 0.6)

    # -- primitives ----------------------------------------------------------
    def _px(self, v: float) -> int:
        return max(1, int(round(v * self.u)))

    def _rr(self, d, x0, y0, x1, y1, r, **kw):
        r = max(0.0, min(r, (y1 - y0) / 2, (x1 - x0) / 2))
        d.rounded_rectangle([x0, y0, x1, y1], radius=r, **kw)

    def _sticker(self, d, x, y, w, h, r, fill=WHITE, shadow=4.0, line=3.0):
        u = self.u
        if shadow:
            self._rr(d, x + shadow * u, y + shadow * u, x + w + shadow * u, y + h + shadow * u, r, fill=INK)
        self._rr(d, x, y, x + w, y + h, r, fill=fill, outline=INK, width=self._px(line))

    def _pill(self, d, x, cy, text, size=12, fill=WHITE, color=INK, weight=700, pad=8, shadow=0.0, line=2.0):
        u = self.u
        font = self.fonts.get(weight, size * u, text)
        w = d.textlength(text, font=font) + 2 * pad * u
        h = size * u + 10 * u
        self._sticker(d, x, cy - h / 2, w, h, h / 2, fill=fill, shadow=shadow, line=line)
        d.text((x + pad * u, cy + 0.5 * u), text, font=font, fill=color, anchor="lm")
        return w

    # -- drawing -------------------------------------------------------------
    def draw(self, frame_bgr: np.ndarray, t: float) -> np.ndarray:
        img = Image.fromarray(np.ascontiguousarray(frame_bgr[:, :, ::-1]))
        d = ImageDraw.Draw(img)
        k = self._window(t)
        sample = self._nearest_sample(t)
        u, W, H = self.u, self.W, self.H

        if self.skeleton and sample:
            self._draw_skeleton(d, sample)
        self._draw_chips(d, t, k, sample)
        self._draw_gesture(d, t)
        foot_top = self._draw_footer(d)
        sub_top = self._draw_subtitle(d, t, foot_top - 8 * u)
        if self.judgments:
            x0, y0, pw, ph = self._panel_geometry()
            if self.portrait:
                top = self._draw_reading(d, k, 12 * u, min(H - 90 * u, sub_top - 20 * u), W - 28 * u)
                y0 = top - 14 * u - ph
            else:
                self._draw_reading(d, k, 18 * u, H - 104 * u, min(W * 0.56, x0 - 48 * u))
            self._draw_panel(d, t, k, x0, y0, pw, ph)
        return np.ascontiguousarray(np.asarray(img)[:, :, ::-1])

    def _draw_chips(self, d, t, k, sample):
        u = self.u
        cy = 24 * u
        x = 14 * u
        x += self._pill(d, x, cy, f"{t:.1f}s", weight=600) + 6 * u
        x += self._pill(d, x, cy, f"W{k + 1}/{max(1, len(self.windows))}", fill=BLUE, color=WHITE) + 6 * u
        self._pill(d, x, cy, shot_type(sample["face"]["size"] if sample and sample["face"] else None), weight=600)

    def _draw_gesture(self, d, t):
        """One sticker per hand (up to two), each with a colour-matched pointer to its hand."""
        hands = self._live_hands(t)
        u, W, H = self.u, self.W, self.H
        x, y = 14 * u, 46 * u
        items = []
        for hand, colour in zip(hands, HAND_COLOURS):
            label = hand_label(hand, self.lang)
            size = 19 * u
            font = self.fonts.get(700, size, label)
            while d.textlength(label, font=font) > W - 2 * x - 40 * u and size > 11:
                size -= 1
                font = self.fonts.get(700, size, label)
            w, h = d.textlength(label, font=font) + 38 * u, size + 18 * u
            items.append((hand, label, font, w, h, y, colour))
            y += h + 40 * u

        for hand, _, _, w, h, top, colour in items:  # pointers first, under the stickers
            if hand["anchor"] is None:
                continue
            ax, ay = hand["anchor"][0] * W, hand["anchor"][1] * H
            sx, sy = min(max(ax, x), x + w), min(max(ay, top), top + h)
            d.line([(sx, sy), (ax, ay)], fill=WHITE, width=self._px(9))
            d.line([(sx, sy), (ax, ay)], fill=INK, width=self._px(4))
            r = 8 * u
            d.ellipse([ax - r, ay - r, ax + r, ay + r], fill=colour, outline=INK, width=self._px(3))
        for hand, label, font, w, h, top, colour in items:
            self._sticker(d, x, top, w, h, 14 * u, shadow=3)
            cx, cy, r = x + 16 * u, top + h / 2, 6 * u
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=colour, outline=INK, width=self._px(2))
            d.text((x + 28 * u, cy + 0.5 * u), label, font=font, fill=INK, anchor="lm")
            self._pill(d, x + 8 * u, top + h + 16 * u, f"{self.str['vision']} {hand['score']:.2f}", size=11,
                       fill=BLUE, color=WHITE)

    def _panel_geometry(self):
        u = self.u
        ph = 178 * u
        if self.portrait:
            pw = min(self.W - 28 * u, 300 * u)
            return self.W - pw - 14 * u, 0.0, pw, ph
        pw = 262 * u
        return self.W - pw - 22 * u, self.H - 104 * u - ph, pw, ph

    def _draw_panel(self, d, t, k, x0, y0, pw, ph):
        u, s = self.u, self.str
        j = self.judgments[k]
        self._sticker(d, x0, y0, pw, ph, 18 * u)

        title_font = self.fonts.get(700, 18 * u, s["verdict"])
        d.text((x0 + 14 * u, y0 + 22 * u), s["verdict"], font=title_font, fill=INK, anchor="lm")
        tag = judge_tag(j["source"], s["rules"])
        self._pill(d, x0 + 22 * u + d.textlength(s["verdict"], font=title_font), y0 + 22 * u, tag, size=11,
                   fill=YELLOW)

        bars = (("confidence", BLUE), ("focus", MINT), ("tension", CORAL))
        label_fonts = {key: self.fonts.get(600, 13 * u, s[key]) for key, _ in bars}
        label_w = max(d.textlength(s[key], font=f) for key, f in label_fonts.items())
        bx0 = x0 + 24 * u + label_w
        bx1 = max(x0 + pw - 50 * u, bx0 + 4 * u)  # fonts have a minimum size, so tiny frames can squeeze this
        value_font = self.fonts.get(700, 13 * u)
        for i, (key, colour) in enumerate(bars):
            cy = y0 + 52 * u + i * 25 * u
            d.text((x0 + 14 * u, cy), s[key], font=label_fonts[key], fill=MUTED, anchor="lm")
            v = self._value(k, key, t)
            self._rr(d, bx0, cy - 6 * u, bx1, cy + 6 * u, 6 * u, fill=PAPER)
            fill_w = (bx1 - bx0) * v
            if fill_w >= 2:
                self._rr(d, bx0, cy - 6 * u, bx0 + fill_w, cy + 6 * u, 6 * u, fill=colour)
            self._rr(d, bx0, cy - 6 * u, bx1, cy + 6 * u, 6 * u, outline=INK, width=self._px(2))
            d.text((bx1 + 8 * u, cy), f"{v:.2f}", font=value_font, fill=INK, anchor="lm")

        cy = y0 + 52 * u + 3 * 25 * u + 2 * u
        d.text((x0 + 14 * u, cy), s["intent"], font=self.fonts.get(600, 12 * u, s["intent"]), fill=MUTED,
               anchor="lm")
        cert = f"{j['intent_certainty']:.2f}"
        cert_w = d.textlength(cert, font=self.fonts.get(700, 11 * u)) + 16 * u
        self._pill(d, x0 + pw - 14 * u - cert_w, cy, cert, size=11, fill=BLUE, color=WHITE)
        ix = x0 + 24 * u + label_w
        intent_font = self.fonts.get(700, 15 * u, j["intent"])
        line = wrap(d, j["intent"], intent_font, x0 + pw - 24 * u - cert_w - ix, 1)
        d.text((ix, cy), line[0] if line else "", font=intent_font, fill=INK, anchor="lm")

        cy += 27 * u
        d.text((x0 + 14 * u, cy), s["arc"], font=self.fonts.get(600, 12 * u, s["arc"]), fill=MUTED, anchor="lm")
        r, gap = 6 * u, 5 * u
        fit = max(1, int((x0 + pw - 14 * u - ix) // (2 * r + gap)))
        first = max(0, k + 1 - fit)
        for n, idx in enumerate(range(first, k + 1)):
            rr = r * 1.25 if idx == k else r
            cx = ix + r + n * (2 * r + gap)
            d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=valence_colour(self.judgments[idx]["valence"]),
                      outline=INK, width=self._px(2.5 if idx == k else 2))

    def _draw_reading(self, d, k, left, bottom, max_w) -> float:
        """Speech bubble with the reading, bottom edge at `bottom`; returns its top."""
        u = self.u
        j = self.judgments[k]
        if not j["reading"] and not j["quote"]:
            return bottom
        tag = f"W{k + 1}"
        tag_w = d.textlength(tag, font=self.fonts.get(700, 12 * u)) + 16 * u
        font = self.fonts.get(500, 15 * u, j["reading"])
        lines = wrap(d, j["reading"], font, max(40 * u, max_w - 36 * u - tag_w), 2)
        line_h = 21 * u
        q_text = f"“{j['quote']}”" if j["quote"] else ""
        q_font = self.fonts.get(600, 13 * u, q_text)
        q_lines = wrap(d, q_text, q_font, max(40 * u, max_w - 44 * u), 1) if q_text else []
        height = 16 * u + len(lines) * line_h + (28 * u if q_lines else 0)
        top = bottom - height
        width = tag_w + 8 * u + max([d.textlength(ln, font=font) for ln in lines] or [0])
        if q_lines:
            width = max(width, d.textlength(q_lines[0], font=q_font) + 16 * u)
        width += 28 * u

        self._rr(d, left + 4 * u, top + 4 * u, left + width + 4 * u, bottom + 4 * u, 16 * u, fill=INK)
        tail = [(left + 22 * u, bottom - 2 * u), (left + 14 * u, bottom + 14 * u), (left + 42 * u, bottom - 2 * u)]
        d.polygon(tail, fill=WHITE, outline=INK, width=self._px(3))
        self._rr(d, left, top, left + width, bottom, 16 * u, fill=WHITE, outline=INK, width=self._px(3))

        y = top + 8 * u + line_h / 2
        self._pill(d, left + 12 * u, y, tag, fill=BLUE, color=WHITE)
        for i, ln in enumerate(lines):
            d.text((left + 20 * u + tag_w, y + i * line_h), ln, font=font, fill=INK, anchor="lm")
        if q_lines:
            y += len(lines) * line_h + 4 * u
            self._pill(d, left + 12 * u, y, q_lines[0], size=13, weight=600, fill=YELLOW)
        return top

    def _draw_subtitle(self, d, t, bottom) -> float:
        text = self._subtitle(t)
        if not text:
            return bottom
        u = self.u
        font = self.fonts.get(600, (18 if self.portrait else 20) * u, text)
        lines = wrap(d, text, font, self.W * (0.9 if self.portrait else 0.62), 2)
        lh = 26 * u
        top = bottom - len(lines) * lh
        y = top + lh / 2
        for ln in lines:
            w = d.textlength(ln, font=font)
            d.text(((self.W - w) / 2, y), ln, font=font, fill=WHITE, anchor="lm",
                   stroke_width=self._px(3), stroke_fill=INK)
            y += lh
        return top

    def _draw_footer(self, d) -> float:
        u = self.u
        cy = self.H - 18 * u
        self._pill(d, 12 * u, cy, self.str["footer"].format(judge=self.judge_name), size=10, weight=600,
                   fill=PAPER, line=1.5, pad=7)
        return cy - 11 * u

    def _draw_skeleton(self, d, sample):
        W, H, u = self.W, self.H, self.u
        for h in sample["hands"]:
            pts = [(x * W, y * H) for x, y in h["pts"]]
            for colour, width in ((WHITE, 5), (INK, 2.2)):
                for a, b in HAND_EDGES:
                    d.line([pts[a], pts[b]], fill=colour, width=self._px(width))
            for x, y in pts:
                r = 3 * u
                d.ellipse([x - r, y - r, x + r, y + r], fill=YELLOW, outline=INK, width=self._px(1.5))
        if sample["face"]:
            x0, y0, x1, y1 = sample["face"]["bbox"]
            x0, x1, y0, y1 = x0 * W, x1 * W, y0 * H, y1 * H
            c = 0.18 * (x1 - x0)
            for colour, width in ((WHITE, 6), (INK, 3)):
                for (px, py, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
                    d.line([(px + dx * c, py), (px, py), (px, py + dy * c)], fill=colour, width=self._px(width),
                           joint="curve")


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
