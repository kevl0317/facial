"""Jev, TypeSafe AI's "System One" decision model, as the judge's decision layer.

Jev doesn't write text. It answers typed questions about a state (a Score on an
ordered scale, a Choice between named options) with calibrated probabilities, in
well under a second. Here it decides each window's scores, intent and tone; the LLM,
if one is set up, only puts the reading into words at low effort, and gets the whole
window back when Jev is unsure (accept when confident, escalate when unsure).

Jev is reported to read words better than numbers, so each window's measurements are
described in plain sentences first (describe()). facial/web/jev.js is a line-for-line
port; tests/test_web.py checks they agree.
"""

from __future__ import annotations

import http.client
import os
import time

from . import __version__
from .expressions import EMOTIONS, EXPRESSIONS, TRAITS
from .net import JudgeUnavailable, _error_message, _post_json
from .render import hand_label

# How to reach Jev: TypeSafe's own API, or OpenRouter (one key for Jev and most LLMs).
# TypeSafe's API sends no CORS headers, so web pages can only try OpenRouter ("browser").
# OpenRouter also serves the same request at /api/alpha/decisions ("alt_urls").
JEV_ROUTES = {
    "typesafe": {"name": "TypeSafe", "base_url": "https://api.typesafe.ai/v1", "model": "jev-latest",
                 "env": ["TYPESAFE_API_KEY"], "key_url": "https://typesafe.ai", "browser": False, "alt_urls": []},
    "openrouter": {"name": "OpenRouter", "base_url": "https://openrouter.ai/api/v1", "model": "typesafe/jev-1.13",
                   "env": ["OPENROUTER_API_KEY"], "key_url": "https://openrouter.ai/keys", "browser": True,
                   "alt_urls": ["https://openrouter.ai/api/alpha/decisions"]},
}

# Ordered levels, lowest first; for traits the middle level is the speaker's usual self (0.5).
JEV_SCALES = {
    **{k: (f"How {adj} does the speaker come across right now ({meaning}), compared with how they usually "
           "come across?",
           [f"not {adj} at all", f"less {adj} than usual", f"as {adj} as usual", f"more {adj} than usual",
            f"very {adj}"])
       for k, (_, _, adj, meaning) in TRAITS.items()},
    "valence": ("What is the emotional tone of this moment?",
                ["very negative", "negative", "neutral", "positive", "very positive"]),
    "emotion_intensity": ("How strongly is the speaker's main emotion showing?",
                          ["barely", "faintly", "clearly", "strongly", "very strongly"]),
}

# key: (English label, Chinese label, what Jev should look for)
JEV_INTENTS = {
    "stating": ("Stating a position", "陈述立场", "states a view, claim or fact plainly"),
    "explaining": ("Explaining", "解释说明", "lays out reasons, steps or details"),
    "emphasizing": ("Emphasizing a point", "强调要点", "drives a point home with force or repetition"),
    "enumerating": ("Enumerating", "逐条列举", "lists several items or options one by one"),
    "persuading": ("Selling a vision", "描绘愿景", "tries to persuade or inspire, paints a picture of the future"),
    "questioning": ("Asking a question", "提出问题", "asks something or raises a question"),
    "conceding": ("Conceding a point", "承认让步", "admits a limit, a mistake or the other side's point"),
    "deflecting": ("Deflecting", "回避转移", "avoids the question, changes the subject or stays vague"),
    "reassuring": ("Reassuring", "安抚对方", "calms or comforts, says things will be fine"),
    "joking": ("Lightening the mood", "调侃缓和", "jokes, teases or keeps things light"),
    "listening": ("Listening", "倾听", "is not talking and listens to someone else"),
    "pausing": ("Pausing", "停顿", "is silent or gathering their thoughts"),
}

# Below this probability for its top intent, Jev hands the window to the LLM.
JEV_ESCALATE = 0.4

JEV_QUESTIONS = {
    **{k: {"type": "score", "instructions": q, "criteria": levels} for k, (q, levels) in JEV_SCALES.items()},
    "emotion": {"type": "choice", "instructions": "Which emotion is the speaker showing most?",
                "criteria": {k: v[2] for k, v in EMOTIONS.items()}},
    "intent": {"type": "choice", "instructions": "What is the speaker mainly doing in this moment?",
               "criteria": {k: v[2] for k, v in JEV_INTENTS.items()}},
}


# --------------------------------------------------------------------------- describing a window

def _band(x: float, steps: list[tuple[float, str]], rest: str) -> str:
    """The word for the first threshold x reaches (thresholds high to low), else `rest`."""
    return next((word for threshold, word in steps if x >= threshold), rest)


def describe(f: dict, previous: dict | None = None) -> str:
    """A window's five fields in plain sentences, for Jev."""
    scene, spk, v, g = f["scene"], f["speaker"], f["voice"], f["gesture"]
    lines = []

    shot = {"close-up": "a close-up", "medium": "a medium shot", "wide": "a wide shot",
            "cutaway": "a cutaway with no face"}.get(scene.get("shot"), "a shot")
    extra = (", several faces" if scene.get("faces", 0) > 1 else "") + (", a camera cut" if scene.get("cuts") else "")
    lines.append(f"Scene: {shot}{extra}.")
    lines.append({"target": "The person on camera is the one talking.",
                  "listening": "The person on camera is listening while someone else talks.",
                  "offscreen": "The speaker is off camera.",
                  "silent": "Nobody is talking."}.get(spk.get("state"), "It is unclear who is talking."))
    text = f.get("subtitle", "").strip()
    lines.append(f'Said: "{text}"' if text else "Nothing was said.")

    if v.get("voiced_frac", 0) < 0.15:
        lines.append("Voice: barely any speech.")
    else:
        parts = []
        if v.get("loudness_rel_db") is not None:
            parts.append(_band(v["loudness_rel_db"], [(4, "much louder than usual"), (1.5, "a bit louder than usual"),
                                                      (-1.5, "at their usual loudness"),
                                                      (-4, "a bit quieter than usual")], "much quieter than usual"))
        if v.get("pitch_rel_st") is not None:
            parts.append(_band(v["pitch_rel_st"], [(2, "higher-pitched than usual"), (-2, "at their usual pitch")],
                               "lower-pitched than usual"))
        if v.get("pitch_var_st") is not None:
            parts.append(_band(v["pitch_var_st"], [(4, "very animated intonation"), (2.5, "lively intonation"),
                                                   (1.2, "normal intonation")], "flat, monotone intonation"))
        if v.get("speech_rate_wps") is not None:
            parts.append(_band(v["speech_rate_wps"], [(3.5, "speaking fast"), (1.8, "speaking at a normal pace")],
                               "speaking slowly"))
        pauses = int(v.get("pauses", 0))
        parts.append(["no pauses", "one pause", "two pauses"][pauses] if pauses < 3 else "several pauses")
        fillers = v.get("fillers", 0)
        if fillers:
            parts.append("many filler words" if fillers >= 3 else "a filler word or two")
        lines.append("Voice: " + ", ".join(parts) + ".")

    dom = g.get("dominant")
    if g.get("hands_visible", 0) < 0.15 or not dom:
        lines.append("Hands: not visible.")
    else:
        parts = [hand_label(dom, "en") + " " + _band(dom.get("share", 0), [(0.6, "for most of the window"),
                                                                         (0.3, "for part of the window")],
                                                     "briefly")]
        if g.get("second"):
            parts.append("the other hand: " + hand_label(g["second"], "en"))
        if g.get("two_hands", 0) >= 0.4:
            parts.append("both hands up much of the time")
        parts.append(_band(g.get("energy", 0), [(2.0, "animated movement"), (0.7, "some movement")],
                           "hands mostly still"))
        if g.get("beats", 0) >= 2:
            parts.append("beat gestures on the words")
        touch = g.get("hand_to_face", 0)
        if touch >= 0.1:
            parts.append("touches their face often" if touch >= 0.3 else "touches their face briefly")
        lines.append("Hands: " + "; ".join(parts) + ".")

    face = g.get("face")
    if face:
        rel = face.get("vs_baseline", {})
        parts = []
        expr = face.get("expression") or {}
        if expr.get("top", "neutral") != "neutral" and expr.get("share", 0) >= 0.25:
            parts.append(f"looks {EXPRESSIONS[expr['top']][0].lower()}"
                         + (" most of the time" if expr["share"] >= 0.5 else " at times"))
        elif expr:
            parts.append("neutral expression")
        if rel.get("smile", 0) >= 0.15:
            parts.append("smiling more than usual")
        elif rel.get("smile", 0) <= -0.15:
            parts.append("smiling less than usual")
        if rel.get("frown", 0) >= 0.08:
            parts.append("frowning")
        if rel.get("brow_furrow", 0) >= 0.1:
            parts.append("brow furrowed more than usual")
        if rel.get("brow_raise", 0) >= 0.1:
            parts.append("eyebrows raised")
        if rel.get("lip_press", 0) >= 0.1:
            parts.append("lips pressed together")
        blink = face.get("blink_per_min", 17)
        if blink >= 30:
            parts.append("blinking a lot")
        elif blink <= 6:
            parts.append("hardly blinking")
        lines.append("Face: " + (", ".join(parts) if parts else "their usual expression") + ".")

    head = g.get("head")
    if head:
        parts = [_band(head.get("yaw_std", 0) + head.get("pitch_std", 0), [(10, "head moving a lot"),
                                                                          (4, "some head movement")], "head steady"),
                 _band(head.get("gaze_away", 0), [(0.4, "often looking away"), (0.1, "sometimes looking away")],
                       "looking toward the camera")]
        if head.get("nods", 0) >= 2:
            parts.append("nodding")
        lines.append("Head: " + ", ".join(parts) + ".")

    posture = g.get("posture")
    if posture:
        parts = []
        arms = posture.get("arms_open")
        if arms is not None and arms >= 0.6:
            parts.append("arms open wide")
        elif arms is not None and arms <= 0.25:
            parts.append("arms held close")
        if abs(posture.get("shoulder_tilt", 0)) >= 6:
            parts.append("shoulders tilted")
        if parts:
            lines.append("Posture: " + ", ".join(parts) + ".")

    if previous:
        lines.append(f"Just before: {JEV_INTENTS[previous['intent_key']][0].lower()}, looking "
                     f"{EMOTIONS[previous['emotion']][0].lower()}.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- reading Jev's answer

def _scale_position(answer: dict, levels: int) -> float | None:
    """Jev's score as 0..1 along the scale (probability-weighted position)."""
    score = answer.get("score")
    if score is None:
        probs = answer.get("probabilities") or {}
        total = sum(float(p) for p in probs.values())
        if not total:
            return None
        score = sum(int(k) * float(p) for k, p in probs.items()) / total
    return max(0.0, min(1.0, float(score) / (levels - 1)))


def parse_decision(data: dict, lang: str = "en") -> dict | None:
    """Traits (0-1), emotion, valence (-1..1), intent and certainty from a /systemone response."""
    answers = data.get("answers") if isinstance(data.get("answers"), dict) else data
    out: dict = {"traits": {}, "words": {}}
    for key, (_, levels) in JEV_SCALES.items():
        pos = _scale_position(answers.get(key) or {}, len(levels))
        if pos is None:
            return None
        if key in TRAITS:
            out["traits"][key] = round(pos, 2)
        else:
            out[key] = round(pos * 2 - 1, 2) if key == "valence" else round(pos, 2)
        out["words"][key] = levels[int(pos * (len(levels) - 1) + 0.5)]  # half up, as in jev.js
    emotion = _choice(answers.get("emotion") or {})[0]
    out["emotion"] = emotion if emotion in EMOTIONS else "calm"
    key, sure = _choice(answers.get("intent") or {})
    if key not in JEV_INTENTS:
        return None
    out.update(intent_key=key, intent=JEV_INTENTS[key][1 if lang == "zh" else 0],
               intent_certainty=round(sure, 2), certain=sure >= JEV_ESCALATE)
    return out


def _choice(a: dict) -> tuple[str | None, float]:
    """A Choice answer's pick and its probability."""
    probs = {k: float(p) for k, p in (a.get("probabilities") or {}).items()}
    key = a.get("choice") or (max(probs, key=probs.get) if probs else None)
    return key, float(probs.get(key, a.get("confidence", 0.0)))


def decision_note(d: dict) -> str:
    """Jev's decision, as the LLM is told it."""
    w = d["words"]
    top = sorted(d["traits"].items(), key=lambda kv: -kv[1])[:3]
    return (f"\nJev's decision for this window (settled): intent \"{d['intent']}\"; emotion "
            f"{EMOTIONS[d['emotion']][0].lower()} ({w['emotion_intensity']}); stands out: "
            + ", ".join(w[k] for k, _ in top) + f"; tone {w['valence']}. "
            "Reply with only reading, quote and evidence, written to fit this decision.")


# --------------------------------------------------------------------------- client

def jev_settings(enabled: bool, via: str | None = None, model: str | None = None) -> dict | None:
    """Options for make_judge(jev=...): the route defaults to whichever key is set."""
    if not enabled:
        return None
    if via is None:
        via = next((r for r, v in JEV_ROUTES.items() if any(os.environ.get(e) for e in v["env"])), "typesafe")
    return {"route": via, "model": model}


class JevClient:
    RETRIES = 2

    def __init__(self, route: str = "typesafe", model: str | None = None, api_key: str | None = None,
                 base_url: str | None = None, log=None):
        if route not in JEV_ROUTES:
            raise ValueError(f"Unknown Jev route {route!r}; choose one of: {', '.join(JEV_ROUTES)}")
        r = JEV_ROUTES[route]
        self.route = route
        self.model = model or r["model"]
        self.base_url = (base_url or r["base_url"]).rstrip("/")
        self.api_key = api_key or next((os.environ[e] for e in r["env"] if os.environ.get(e)), "")
        self.log = log or (lambda m: None)
        self.post = _post_json
        self.sleep = time.sleep
        self.usage = {"input": 0}

    def decide(self, feats: dict, lang: str = "en", previous: dict | None = None) -> dict | None:
        """Jev's decision for one window; None when it's busy or unreachable just now.

        Raises JudgeUnavailable when it can't be used at all (no key, key rejected, bad model).
        """
        if not self.api_key:
            env = " or ".join(JEV_ROUTES[self.route]["env"])
            raise JudgeUnavailable(f"No Jev API key found. Set {env} to use Jev")
        headers = {"Content-Type": "application/json", "User-Agent": f"facial/{__version__}",
                   "Authorization": f"Bearer {self.api_key}"}
        body = {"model": self.model, "state": describe(feats, previous), "questions": JEV_QUESTIONS}
        url = self.base_url + "/systemone"
        for attempt in range(self.RETRIES + 1):
            try:
                status, data = self.post(url, headers, body, 30)
            except ValueError as exc:
                raise JudgeUnavailable(f"Bad Jev URL {url!r} ({exc})") from exc
            except (OSError, http.client.HTTPException) as exc:  # network error or timeout
                status, data = 0, {"error": {"message": str(exc) or exc.__class__.__name__}}
            if status == 200:
                self.usage["input"] += (data.get("usage") or {}).get("input_tokens") or 0
                decision = parse_decision(data, lang)
                if decision is None:
                    self.log(f"  window {feats['window']}: Jev's answer was incomplete")
                return decision
            if status and status not in (408, 409, 429) and status < 500:
                raise JudgeUnavailable(f"Jev request rejected ({status}: {_error_message(data)})")
            if attempt < self.RETRIES:
                self.sleep(0.5 * 2 ** attempt)
        self.log(f"  window {feats['window']}: Jev unavailable ({status or _error_message(data)})")
        return None
