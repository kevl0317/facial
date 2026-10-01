"""Emotions and traits: what the overlay names, and a face-expression reading from blendshapes.

Three things live here, shared by the Python pipeline and (through webbuild's prompt.js)
the browser:

- EXPRESSIONS: what the face alone shows, frame by frame, read from MediaPipe's
  blendshapes with FACS-style combinations of facial action units (e.g. a genuine smile
  raises the cheeks; disgust wrinkles the nose; fear stretches the mouth). This is a
  rough, uncalibrated reading of *expressions*, not of feelings.
- EMOTIONS: the named emotion a judge gives each window, from face, voice and words.
- TRAITS: eight specific 0-1 scores per window (0.5 = this speaker's usual self).

facial/web/expressions.js is a line-for-line port of expression_scores() and
summarize_expressions(); tests/test_web.py checks they agree.
"""

from __future__ import annotations

# key: (English, Chinese, face glyph). Glyphs are drawn by hud.js / render.py:
# mouth smile|grin|frown|flat|o|wavy|smirk, brows none|sad|angry|up|one, fill colour name.
EXPRESSIONS = {
    "neutral": ("Neutral", "平静", ("flat", "none", "paper")),
    "happy": ("Happy", "开心", ("smile", "none", "yellow")),
    "sad": ("Sad", "难过", ("frown", "sad", "sky")),
    "surprised": ("Surprised", "惊讶", ("o", "up", "yellow")),
    "angry": ("Angry", "生气", ("frown", "angry", "coral")),
    "disgusted": ("Disgusted", "厌恶", ("wavy", "angry", "mint")),
    "fearful": ("Fearful", "害怕", ("o", "sad", "sky")),
    "contempt": ("Contempt", "不屑", ("smirk", "one", "orange")),
}

# key: (English, Chinese, what it looks like, glyph, face expression it usually shows as)
EMOTIONS = {
    "calm": ("Calm", "平静", "composed, nothing strong showing", ("flat", "none", "paper")),
    "happy": ("Happy", "开心", "pleased, glad, smiling warmly", ("smile", "none", "yellow")),
    "amused": ("Amused", "被逗乐", "finds something funny, laughing or grinning", ("grin", "none", "yellow")),
    "excited": ("Excited", "兴奋", "eager and energised, animated", ("grin", "up", "orange")),
    "proud": ("Proud", "自豪", "pleased with an achievement, chin up, beaming", ("smile", "up", "yellow")),
    "interested": ("Interested", "好奇", "curious, attentive, leaning in", ("flat", "up", "mint")),
    "surprised": ("Surprised", "惊讶", "caught off guard, brows up, eyes wide", ("o", "up", "yellow")),
    "confused": ("Confused", "困惑", "puzzled, unsure what is meant", ("wavy", "one", "sky")),
    "skeptical": ("Skeptical", "怀疑", "doubtful, not convinced", ("smirk", "one", "paper")),
    "dismissive": ("Dismissive", "不屑", "brushing something off, a lopsided smirk", ("smirk", "none", "orange")),
    "frustrated": ("Frustrated", "烦躁", "annoyed at being blocked, tight-lipped", ("flat", "angry", "orange")),
    "angry": ("Angry", "生气", "hostile or indignant, brows down, forceful", ("frown", "angry", "coral")),
    "sad": ("Sad", "难过", "down, disappointed, inner brows raised", ("frown", "sad", "sky")),
    "anxious": ("Anxious", "担忧", "worried or nervous, fidgety, wide-eyed", ("wavy", "sad", "sky")),
    "embarrassed": ("Embarrassed", "尴尬", "self-conscious, awkward smile, looks away", ("wavy", "sad", "coral")),
    "disgusted": ("Disgusted", "厌恶", "repelled, nose wrinkled", ("wavy", "angry", "mint")),
}

# The emotion a face expression usually means, when nothing else says more.
EXPRESSION_EMOTION = {"neutral": "calm", "happy": "happy", "sad": "sad", "surprised": "surprised",
                      "angry": "angry", "disgusted": "disgusted", "fearful": "anxious", "contempt": "dismissive"}

# key: (English, Chinese, adjective for Jev's scale, what it means)
TRAITS = {
    "confident": ("Confident", "自信", "confident", "sure of themselves, steady, not hedging"),
    "nervous": ("Nervous", "紧张", "nervous", "anxious or on edge: fidgeting, self-touching, tight lips"),
    "enthusiastic": ("Enthusiastic", "热情", "enthusiastic", "energetic and eager: animated hands, lively voice"),
    "warm": ("Warm", "亲和", "warm", "friendly and approachable: smiles, soft tone, nods"),
    "assertive": ("Assertive", "强势", "assertive", "forceful, pushing a point: firm voice, fists or pointing"),
    "defensive": ("Defensive", "防御", "defensive", "guarded or closed off: arms in, looks away, justifies"),
    "engaged": ("Engaged", "投入", "engaged", "focused on the exchange: eye contact, steady attention"),
    "hesitant": ("Hesitant", "犹豫", "hesitant", "unsure or holding back: pauses, fillers, trailing off"),
}

# Blendshapes where a person's resting face often carries a bias (a "resting frown"):
# expressions use them relative to that person's median.
BIAS_KEYS = ("brow_furrow", "lip_press", "squint", "frown", "brow_inner_up")


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def expression_scores(face: dict, base: dict | None = None) -> dict:
    """Probabilities (summing to 1) of each EXPRESSIONS key for one frame's face metrics."""
    base = base or {}

    def g(k: str) -> float:
        v = float(face.get(k) or 0.0)
        if k in BIAS_KEYS and base.get(k) is not None:
            v = max(0.0, v - float(base[k]))
        return v

    smile = g("smile")
    raw = {
        "happy": 0.75 * smile + 0.45 * g("cheek_squint"),
        "sad": 0.8 * g("frown") + 0.6 * g("brow_inner_up") + 0.2 * g("brow_furrow") - 0.6 * smile,
        "surprised": (0.5 * g("eye_wide") + 0.45 * g("brow_outer_up") + 0.25 * g("brow_inner_up")
                      + 0.4 * g("jaw") - 0.4 * g("brow_furrow") - 0.3 * smile),
        "angry": (0.9 * g("brow_furrow") + 0.4 * g("lip_press") + 0.25 * g("squint") + 0.2 * g("nose_sneer")
                  - 0.5 * smile - 0.3 * g("brow_inner_up")),
        "disgusted": 1.0 * g("nose_sneer") + 0.4 * g("upper_lip_up") + 0.2 * g("frown") - 0.3 * smile,
        "fearful": (0.35 * g("eye_wide") + 0.45 * g("brow_inner_up") + 0.7 * g("mouth_stretch")
                    + 0.15 * g("brow_furrow") - 0.5 * smile),
        "contempt": 1.5 * g("smile_asym") - 0.3 * g("frown") - 0.4 * max(0.0, smile - 0.5),
    }
    # Below 0.1 is noise (a resting face); from 0.5 up it's a clear expression.
    scores = {k: _clamp((v - 0.1) / 0.4) for k, v in raw.items()}
    scores["neutral"] = _clamp(1.0 - 1.6 * max(scores.values()))
    total = sum(scores.values()) or 1.0
    return {k: scores[k] / total for k in EXPRESSIONS}


def summarize_expressions(faces: list[dict], base: dict | None = None) -> dict | None:
    """A window's expression: mean probabilities over its frames, the top one and its share."""
    if not faces:
        return None
    totals = dict.fromkeys(EXPRESSIONS, 0.0)
    for f in faces:
        for k, p in expression_scores(f, base).items():
            totals[k] += p
    mean = {k: v / len(faces) for k, v in totals.items()}
    top = max(EXPRESSIONS, key=lambda k: mean[k])
    return {"top": top, "share": round(mean[top], 2),
            "scores": {k: round(v, 2) for k, v in sorted(mean.items(), key=lambda kv: -kv[1])[:3]}}
