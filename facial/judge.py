"""The judgment layer: fuse the five fields of each window into a reading.

`ClaudeJudge` runs one conversation per clip, one window per turn, so the
emotion arc stays coherent and every turn reuses the cached history.
`heuristic_judgment` is a rule-based fallback for running without an API key.
"""

from __future__ import annotations

import json
import re
import sys

import anthropic

DEFAULT_MODEL = "claude-opus-5-5"
# Models that accept server-side refusal fallbacks (`fallbacks: "default"`).
FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"}

SCHEMA = {
    "type": "object",
    "properties": {
        "reading": {"type": "string"},
        "quote": {"type": "string"},
        "confidence": {"type": "number"},
        "focus": {"type": "number"},
        "tension": {"type": "number"},
        "intent": {"type": "string"},
        "intent_certainty": {"type": "number"},
        "valence": {"type": "number"},
        "evidence": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["reading", "quote", "confidence", "focus", "tension", "intent",
                 "intent_certainty", "valence", "evidence"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """\
You are the judgment layer of a video-analysis pipeline that annotates how a person speaks: what their hands, face and voice are doing, and what communicative intent that suggests. The annotations are overlaid on the video for viewers who want to study body language in interviews and keynotes.

Upstream, computer vision (Google MediaPipe) and audio analysis have measured each time window of the clip. You receive one window per turn as JSON with five fields:
- scene: shot type (close-up / medium / wide / cutaway), number of faces, hard cuts.
- speaker: whether the tracked on-screen person is the one talking ("target"), visibly listening while someone else talks ("listening"), off-screen, or silent. lip_sync is the correlation between their jaw movement and the audio loudness.
- subtitle: what was said in this window.
- voice: loudness and pitch relative to this speaker's own median in the clip (dB, semitones), pitch variability, share of time voiced, pauses, speech rate (words/s), filler words.
- gesture: the dominant hand state (hand, shape, palm facing, axis) and a short sequence of hand states with times relative to the window start, gesture energy (hand-lengths per second), beat gestures, hand-to-face contact, head movement and gaze, facial blendshape averages (0-1), blink rate and posture.

Fuse the fields into one reading of the window:
- reading: one vivid line of at most ~90 characters that ties the body language to what is being said, e.g. "Left hand spreads open as if laying out facts; the claim about culture comes out calm and firm." Lead with the gesture when there is one.
- quote: the most telling short phrase from the subtitle, copied verbatim; an empty string if there is no subtitle.
- confidence, focus, tension: 0-1 scores for what the person expresses in this window. 0.5 is this speaker's own baseline; move away from it only as far as the evidence supports.
- intent: a 2-4 word label for the communicative act, e.g. "Stating a position", "Deflecting", "Building suspense", "Conceding a point", "Selling a vision".
- intent_certainty: 0-1, how clearly the evidence supports that intent.
- valence: -1 (negative) to +1 (positive) emotional tone, used for the emotion arc.
- evidence: 2-4 short cues you relied on, citing the measurements.

Ground rules:
- Use only the measurements and the subtitle. They are noisy: one blink spike or one odd frame is not a signal, so look for cues that agree across fields.
- Describe expressed demeanor and communicative intent. Do not claim to know hidden thoughts or whether someone is lying, do not comment on health, and do not guess personal attributes.
- Refer to the person as "the speaker", or by the name the user gives in the clip context.
- When the target is not speaking or not visible, say so in the reading, keep the scores near the previous window's, and lower intent_certainty.
- Keep continuity with earlier windows in this conversation: the arc should read as one story, and a change in a score should reflect a change in the data.
"""

LANGUAGE_RULES = {
    "en": "Write reading, intent and evidence in English. Keep quote in the subtitle's original language.",
    "zh": ("Write reading, intent and evidence in Simplified Chinese, in the style of "
           "\"左手摊开如摆事实——他将文化归因说得笃定而坦然\" and intents like \"陈述立场\". "
           "Keep quote in the subtitle's original language."),
}


def _clamp(v, lo=0.0, hi=1.0) -> float:
    try:
        return round(max(lo, min(hi, float(v))), 2)
    except (TypeError, ValueError):
        return round((lo + hi) / 2, 2)


def normalise(j: dict, source: str) -> dict:
    return {
        "reading": str(j.get("reading", "")).strip(),
        "quote": str(j.get("quote", "")).strip().strip('"“”'),
        "confidence": _clamp(j.get("confidence")),
        "focus": _clamp(j.get("focus")),
        "tension": _clamp(j.get("tension")),
        "intent": str(j.get("intent", "")).strip(),
        "intent_certainty": _clamp(j.get("intent_certainty")),
        "valence": _clamp(j.get("valence"), -1.0, 1.0),
        "evidence": [str(e) for e in j.get("evidence", [])][:4],
        "source": source,
    }


class ClaudeJudge:
    def __init__(self, model: str = DEFAULT_MODEL, effort: str = "medium", lang: str = "en",
                 context: str = ""):
        self.client = anthropic.Anthropic(max_retries=4)
        self.model = model
        self.effort = effort
        self.context = context
        self.system = SYSTEM_PROMPT + "- " + LANGUAGE_RULES[lang] + "\n"
        self.messages: list[dict] = []
        self.usage = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}

    def _prompt(self, feats: dict, clip: dict | None) -> str:
        body = json.dumps(feats, ensure_ascii=False, separators=(",", ":"))
        head = f'Window {feats["window"]}/{feats["of"]}:\n'
        if clip is None:
            return head + body
        intro = (f"Clip: {clip['duration']:.1f}s in {feats['of']} windows. "
                 f"Context from the user: {self.context or 'none given'}.\n"
                 f"Clip baselines (medians): {json.dumps(clip['baseline'], separators=(',', ':'))}\n\n")
        return intro + head + body

    def judge(self, feats: dict, clip: dict | None = None) -> dict | None:
        """Judge one window. Returns None when this window should fall back to heuristics.

        Raises anthropic.AuthenticationError / PermissionDeniedError / BadRequestError and
        TypeError (no credentials) so the caller can stop using Claude for the rest of the clip.
        """
        self.messages.append({"role": "user", "content": self._prompt(feats, clip)})
        extra = {}
        if self.model in FALLBACK_MODELS:
            extra = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                system=self.system,
                messages=self.messages,
                thinking={"type": "adaptive"},
                output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": SCHEMA}},
                cache_control={"type": "ephemeral"},
                **extra,
            )
        except Exception as exc:
            self.messages.pop()
            transient = isinstance(exc, (anthropic.RateLimitError, anthropic.APIConnectionError)) or (
                isinstance(exc, anthropic.APIStatusError) and exc.status_code >= 500)
            if not transient:
                raise
            print(f"  window {feats['window']}: Claude unavailable ({exc.__class__.__name__}); "
                  "using heuristics for this window", file=sys.stderr)
            return None

        u = response.usage
        self.usage["input"] += u.input_tokens
        self.usage["output"] += u.output_tokens
        self.usage["cache_read"] += u.cache_read_input_tokens or 0
        self.usage["cache_write"] += u.cache_creation_input_tokens or 0

        if response.stop_reason != "end_turn":
            print(f"  window {feats['window']}: Claude stopped with {response.stop_reason}; "
                  "using heuristics for this window", file=sys.stderr)
            self.messages.pop()
            return None
        text = next((b.text for b in reversed(response.content) if b.type == "text"), "")
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            self.messages.pop()
            return None
        # Append the full content (thinking blocks included) so history stays append-only.
        self.messages.append({"role": "assistant", "content": response.content})
        return normalise(data, "claude")


# --------------------------------------------------------------------------- heuristics

INTENTS = {
    "en": {
        "listening": "Listening", "silent": "Pausing", "offscreen": "Off camera", "question": "Asking a question",
        "pointing": "Emphasizing a point", "fist": "Stressing conviction", "pinch": "Making a precise point",
        "two_fingers": "Enumerating", "palm_up": "Presenting / explaining", "palm_down": "Asserting / settling",
        "palm_out": "Holding back / clarifying", "self_touch": "Self-soothing", "default": "Stating a position",
    },
    "zh": {
        "listening": "倾听", "silent": "停顿", "offscreen": "不在画面", "question": "提出问题",
        "pointing": "强调要点", "fist": "表达决心", "pinch": "精确论证",
        "two_fingers": "逐条列举", "palm_up": "陈述解释", "palm_down": "断言定调",
        "palm_out": "保留澄清", "self_touch": "自我安抚", "default": "陈述立场",
    },
}


def _quote(text: str) -> str:
    clauses = [c.strip() for c in re.split(r"[,.;!?，。；！？]", text) if len(c.strip().split()) >= 3 or
               (c.strip() and not c.strip().isascii() and len(c.strip()) >= 4)]
    return max(clauses, key=len)[:70] if clauses else text[:70]


def heuristic_judgment(f: dict, lang: str = "en", hand_label=None) -> dict:
    """Transparent rule-based scores; used without an API key or when a Claude call fails."""
    voice, g, spk = f["voice"], f["gesture"], f["speaker"]
    face, head = g.get("face", {}), g.get("head", {})
    rel = face.get("vs_baseline", {})
    dom = g.get("dominant")
    blink = face.get("blink_per_min", 17)

    tension = (0.15 + 1.0 * max(0, rel.get("lip_press", 0)) + 0.8 * max(0, rel.get("brow_furrow", 0))
               + 0.5 * g.get("hand_to_face", 0) + 0.006 * max(0, blink - 22)
               + 0.05 * voice.get("pauses", 0) + 0.05 * voice.get("fillers", 0)
               + 0.04 * max(0, voice.get("pitch_var_st", 2) - 3))
    open_gesture = 1.0 if dom and dom["shape"] in ("open_palm", "pointing", "fist") else 0.0
    steadiness = 1.0 - min(1.0, (head.get("yaw_std", 5) + head.get("pitch_std", 5)) / 20)
    confidence = (0.45 + 0.03 * voice.get("loudness_rel_db", 0) + 0.12 * open_gesture
                  + 0.1 * steadiness - 0.25 * head.get("gaze_away", 0) - 0.3 * (tension - 0.15)
                  + 0.05 * min(g.get("beats", 0), 3))
    focus = (0.45 + 0.3 * (1 - head.get("gaze_away", 0.3)) + 0.1 * steadiness
             + 0.1 * voice.get("voiced_frac", 0) - 0.2 * g.get("hand_to_face", 0))
    valence = (0.6 * face.get("smile", 0) - 0.6 * face.get("frown", 0)
               + 1.5 * rel.get("smile", 0) - 1.5 * rel.get("frown", 0) - 0.5 * rel.get("brow_furrow", 0))

    names = INTENTS[lang]
    text = f["subtitle"]
    if spk["state"] in ("listening", "silent", "offscreen"):
        key = spk["state"]
    elif g.get("hand_to_face", 0) > 0.3:
        key = "self_touch"
    elif text.rstrip().endswith(("?", "？")):
        key = "question"
    elif dom and dom["shape"] in ("pointing", "fist", "pinch", "two_fingers"):
        key = dom["shape"]
    elif dom and dom["shape"] == "open_palm" and dom["facing"] in ("palm_up", "palm_down", "palm_out"):
        key = dom["facing"]
    else:
        key = "default"
    intent = names[key]

    gesture_text = hand_label(dom) if (dom and hand_label) else ""
    sep = "，" if lang == "zh" else " — "
    reading = f"{gesture_text}{sep}{intent}" if gesture_text else intent
    return normalise({
        "reading": reading, "quote": _quote(text) if text else "",
        "confidence": confidence, "focus": focus, "tension": tension, "intent": intent,
        "intent_certainty": 0.45 if key != "default" else 0.3, "valence": valence,
        "evidence": ["rule-based: blendshapes, head motion, voice, hand shape"],
    }, "heuristic")


def judge_all(windows: list[dict], clip: dict, method: str, lang: str, model: str, effort: str,
              context: str, hand_label) -> list[dict]:
    judge = None
    if method == "claude":
        judge = ClaudeJudge(model=model, effort=effort, lang=lang, context=context)

    results = []
    for i, feats in enumerate(windows):
        result = None
        if judge is not None:
            try:
                result = judge.judge(feats, clip if i == 0 else None)
            except TypeError as exc:
                print(f"No Anthropic credentials found ({exc}). Set ANTHROPIC_API_KEY to use Claude; "
                      "falling back to heuristic scoring.", file=sys.stderr)
                judge = None
            except (anthropic.AuthenticationError, anthropic.PermissionDeniedError,
                    anthropic.BadRequestError, anthropic.NotFoundError) as exc:
                print(f"Claude request rejected ({exc.__class__.__name__}: {exc}); "
                      "falling back to heuristic scoring.", file=sys.stderr)
                judge = None
        if result is None:
            result = heuristic_judgment(feats, lang, hand_label)
        results.append(result)
        print(f"  window {i + 1}/{len(windows)} [{result['source']}] {result['intent']}: {result['reading']}",
              file=sys.stderr)

    if judge is not None and judge.usage["input"]:
        u = judge.usage
        print(f"Claude usage: {u['input']} input + {u['cache_read']} cache-read + {u['cache_write']} "
              f"cache-write input tokens, {u['output']} output tokens", file=sys.stderr)
    return results
