"""The judgment layer: fuse the five fields of each window into a reading.

A judge runs one conversation per clip, one window per turn, so the emotion arc
stays coherent. `ClaudeJudge` talks to Claude through the Anthropic SDK (with
prompt caching); `ChatJudge` talks to any OpenAI-compatible chat API (OpenAI,
DeepSeek, Gemini, Grok, Mistral, Qwen, Kimi, GLM, Groq, OpenRouter, Ollama...).
`heuristic_judgment` is a rule-based fallback for running without an API key.
"""

from __future__ import annotations

import http.client
import json
import re
import sys
import time

import anthropic

from . import __version__
from .expressions import EMOTIONS, EXPRESSION_EMOTION, TRAITS
from .jev import JEV_INTENTS, JevClient, decision_note
from .net import JudgeUnavailable, _error_message, _post_json
from .providers import OPENAI_REASONING, PROVIDERS, RULES, THINKING_MODELS, env_key, provider

DEFAULT_MODEL = PROVIDERS["claude"]["model"]
# Models that accept server-side refusal fallbacks (`fallbacks: "default"`).
FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"}

SCHEMA = {
    "type": "object",
    "properties": {
        "reading": {"type": "string"},
        "quote": {"type": "string"},
        "emotion": {"type": "string", "enum": list(EMOTIONS)},
        "emotion_intensity": {"type": "number"},
        "traits": {"type": "object", "properties": {k: {"type": "number"} for k in TRAITS},
                   "required": list(TRAITS), "additionalProperties": False},
        "intent": {"type": "string"},
        "intent_certainty": {"type": "number"},
        "valence": {"type": "number"},
        "evidence": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["reading", "quote", "emotion", "emotion_intensity", "traits", "intent", "intent_certainty",
                 "valence", "evidence"],
    "additionalProperties": False,
}

# The parts of a judgment a decision (Jev's, or the LLM's) settles; the rest is words.
DECISION_KEYS = ("emotion", "emotion_intensity", "traits", "valence", "intent", "intent_certainty")

SYSTEM_PROMPT = """\
You are the judgment layer of a video-analysis pipeline that annotates how a person speaks: what their hands, face and voice are doing, and what emotion and communicative intent that suggests. The annotations are overlaid on the video for viewers who want to study body language in interviews and keynotes.

Upstream, computer vision (Google MediaPipe) and audio analysis have measured each time window of the clip. You receive one window per turn as JSON with five fields:
- scene: shot type (close-up / medium / wide / cutaway), number of faces, hard cuts.
- speaker: whether the tracked on-screen person is the one talking ("target"), visibly listening while someone else talks ("listening"), off-screen, or silent. lip_sync is the correlation between their jaw movement and the audio loudness.
- subtitle: what was said in this window.
- voice: loudness and pitch relative to this speaker's own median in the clip (dB, semitones), pitch variability, share of time voiced, pauses, speech rate (words/s), filler words.
- gesture: the dominant hand state (hand, shape, palm facing, axis), the other hand's state when both hands are up (second) and the share of the window with both hands visible (two_hands), a short sequence of hand states with times relative to the window start, gesture energy (hand-lengths per second), beat gestures, hand-to-face contact, head movement and gaze, facial blendshape averages (0-1), the facial expression read from the blendshapes (face.expression: the top expression, its share of the window and the runners-up; a rough reading of the face alone), blink rate and posture.

Fuse the fields into one reading of the window:
- reading: one vivid line of at most ~90 characters that ties the body language to what is being said, e.g. "Left hand spreads open as if laying out facts; the claim about culture comes out calm and firm." Lead with the gesture when there is one.
- quote: the most telling short phrase from the subtitle, copied verbatim; an empty string if there is no subtitle.
- emotion: the emotion the speaker is showing, read from the face, the voice and the words together. One of:
""" + "".join(f"  - {k}: {v[2]}\n" for k, v in EMOTIONS.items()) + """\
- emotion_intensity: 0-1, how strongly that emotion shows.
- traits: a 0-1 score for each of these eight traits. 0.5 is this speaker's usual self; move away from it only as far as the evidence supports:
""" + "".join(f"  - {k}: {v[3]}\n" for k, v in TRAITS.items()) + """\
- intent: a 2-4 word label for the communicative act, e.g. "Stating a position", "Deflecting", "Building suspense", "Conceding a point", "Selling a vision".
- intent_certainty: 0-1, how clearly the evidence supports that intent.
- valence: -1 (negative) to +1 (positive) emotional tone, used for the emotion arc.
- evidence: 2-4 short cues you relied on, citing the measurements.

Ground rules:
- Use only the measurements and the subtitle. They are noisy: one blink spike or one odd frame is not a signal, so look for cues that agree across fields.
- Describe expressed emotion, demeanor and communicative intent. Do not claim to know hidden thoughts or whether someone is lying, do not comment on health, and do not guess personal attributes.
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


# When Jev decides a window, the LLM only writes these.
WRITE_SCHEMA = {
    "type": "object",
    "properties": {"reading": {"type": "string"}, "quote": {"type": "string"},
                   "evidence": {"type": "array", "items": {"type": "string"}}},
    "required": ["reading", "quote", "evidence"],
    "additionalProperties": False,
}

WRITER_RULE = ("- Jev, a fast decision model, may have already decided a window's scores, intent and tone. The turn "
               "then gives Jev's decision: take it as settled and reply with only reading, quote and evidence, "
               "written to fit it. When a turn gives no decision, judge the window fully as above.\n")

# Added to the system prompt for chat-completions providers, which can't all be held to the schema.
JSON_RULE = ("- Reply with only the JSON object for the window: no prose, no code fences. "
             f"It must match this JSON Schema: {json.dumps(SCHEMA, separators=(',', ':'))}\n")


def top_traits(traits: dict, n: int = 3) -> list[tuple[str, float]]:
    """The n traits that show most (ties keep TRAITS order)."""
    return sorted(traits.items(), key=lambda kv: -kv[1])[:n]


def _clamp(v, lo=0.0, hi=1.0) -> float:
    try:
        return round(max(lo, min(hi, float(v))), 2)
    except (TypeError, ValueError):
        return round((lo + hi) / 2, 2)


def normalise(j: dict, source: str, writer: str | None = None) -> dict:
    emotion = str(j.get("emotion", "")).strip().lower()
    traits = j.get("traits") if isinstance(j.get("traits"), dict) else {}
    out = {
        "reading": str(j.get("reading", "")).strip(),
        "quote": str(j.get("quote", "")).strip().strip('"“”'),
        "emotion": emotion if emotion in EMOTIONS else "calm",
        "emotion_intensity": _clamp(j.get("emotion_intensity")),
        "traits": {k: _clamp(traits.get(k)) for k in TRAITS},
        "intent": str(j.get("intent", "")).strip(),
        "intent_certainty": _clamp(j.get("intent_certainty")),
        "valence": _clamp(j.get("valence"), -1.0, 1.0),
        "evidence": [str(e) for e in j.get("evidence", [])][:4],
        "source": source,
    }
    if writer:
        out["writer"] = writer  # the LLM that put a Jev decision into words
    return out


class BaseJudge:
    """One conversation per clip or live session; subclasses make the API call."""

    def __init__(self, pid: str, model: str | None = None, effort: str = "medium", lang: str = "en",
                 context: str = "", max_turns: int | None = None, log=None):
        self.provider = provider(pid)
        self.id = pid
        self.name = self.provider["name"]
        self.model = model or self.provider["model"]
        self.effort = effort
        self.context = context
        self.max_turns = max_turns
        self.log = log or _stderr
        self.system = SYSTEM_PROMPT + "- " + LANGUAGE_RULES[lang] + "\n"
        self.messages: list[dict] = []
        self.clip: dict | None = None
        self.history: list[dict] = []  # (window, judgment) summaries, for conversation rollover
        self.usage = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}

    def _intro(self, feats: dict) -> str:
        clip = self.clip or {}
        ctx = self.context or "none given"
        if clip.get("live"):
            text = f"Live camera session, judged window by window as it happens. Context from the user: {ctx}.\n"
        else:
            text = (f"Clip: {clip.get('duration', 0):.1f}s in {feats.get('of')} windows. "
                    f"Context from the user: {ctx}.\n")
        if clip.get("baseline"):
            text += f"Clip baselines (medians): {json.dumps(clip['baseline'], separators=(',', ':'))}\n"
        if self.history:
            text += "Earlier windows, summarised (most recent last):\n" + "\n".join(
                f"- W{h['window']}: {h['intent']} | {h['emotion']} {h['emotion_intensity']:.2f} | "
                + ", ".join(f"{k} {v:.2f}" for k, v in top_traits(h["traits"]))
                + f", valence {h['valence']:+.2f} | {h['reading']}"
                for h in self.history[-6:]) + "\n"
        return text + "\n"

    def _prompt(self, feats: dict, first: bool) -> str:
        body = json.dumps(feats, ensure_ascii=False, separators=(",", ":"))
        total = f"/{feats['of']}" if feats.get("of") else " (live)"
        head = f"Window {feats['window']}{total}:\n"
        return (self._intro(feats) if first else "") + head + body

    def _call(self, window: int, schema: dict) -> tuple[dict, dict] | None:
        """Send self.messages; return (parsed JSON, assistant message) or None for a transient failure."""
        raise NotImplementedError

    def judge(self, feats: dict, clip: dict | None = None, decision: dict | None = None) -> dict | None:
        """Judge one window. Returns None when this window should fall back to heuristics.

        With a Jev `decision`, the scores and intent are settled and the model only writes
        the reading, quote and evidence. Raises JudgeUnavailable when the judge can't be
        used at all (no credentials, a rejected key or request).
        """
        if clip is not None:
            self.clip = clip
        turns = sum(1 for m in self.messages if m["role"] == "assistant")
        if self.max_turns and turns >= self.max_turns:
            # Long sessions: start a fresh conversation (history stays append-only
            # within each one) and carry the recent arc forward as a summary.
            self.messages = []
        content = self._prompt(feats, first=not self.messages) + (decision_note(decision) if decision else "")
        self.messages.append({"role": "user", "content": content})
        try:
            out = self._call(feats["window"], WRITE_SCHEMA if decision else SCHEMA)
        except BaseException:
            self.messages.pop()
            raise
        if out is None:
            self.messages.pop()
            return None
        data, assistant = out
        self.messages.append(assistant)
        if decision:
            data = {**data, **{k: decision[k] for k in DECISION_KEYS}}
        result = normalise(data, self.id)
        self.history.append({"window": feats["window"], **result})
        return result


class ClaudeJudge(BaseJudge):
    """Claude via the Anthropic SDK: structured outputs, adaptive thinking, prompt caching."""

    def __init__(self, model: str = DEFAULT_MODEL, effort: str = "medium", lang: str = "en",
                 context: str = "", max_turns: int | None = None, log=None, api_key: str | None = None,
                 base_url: str | None = None):
        super().__init__("claude", model, effort, lang, context, max_turns, log)
        self.client = anthropic.Anthropic(api_key=api_key or None, base_url=base_url or None, max_retries=4)

    def _call(self, window: int, schema: dict) -> tuple[dict, dict] | None:
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
                output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
                cache_control={"type": "ephemeral"},
                **extra,
            )
        except TypeError as exc:
            raise JudgeUnavailable(f"No Anthropic credentials found ({exc}). Set ANTHROPIC_API_KEY to use "
                                   "Claude") from exc
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError, anthropic.BadRequestError,
                anthropic.NotFoundError) as exc:
            raise JudgeUnavailable(f"Claude request rejected ({exc.__class__.__name__}: {exc})") from exc
        except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.APIStatusError) as exc:
            if isinstance(exc, anthropic.APIStatusError) and exc.status_code < 500 and \
                    not isinstance(exc, anthropic.RateLimitError):
                raise JudgeUnavailable(f"Claude request rejected ({exc.__class__.__name__}: {exc})") from exc
            self.log(f"  window {window}: Claude unavailable ({exc.__class__.__name__}); "
                     "using heuristics for this window")
            return None

        u = response.usage
        self.usage["input"] += u.input_tokens
        self.usage["output"] += u.output_tokens
        self.usage["cache_read"] += u.cache_read_input_tokens or 0
        self.usage["cache_write"] += u.cache_creation_input_tokens or 0

        if response.stop_reason != "end_turn":
            self.log(f"  window {window}: Claude stopped with {response.stop_reason}; "
                     "using heuristics for this window")
            return None
        text = next((b.text for b in reversed(response.content) if b.type == "text"), "")
        data = extract_json(text)
        if data is None:
            return None
        # Keep the full content (thinking blocks included) so history stays append-only.
        return data, {"role": "assistant", "content": response.content}


def extract_json(text: str | None) -> dict | None:
    """The JSON object in a model's reply, tolerating code fences, <think> blocks and chatter."""
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        return None
    try:
        data = json.loads(text[a:b + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


class ChatJudge(BaseJudge):
    """Any OpenAI-compatible chat-completions API (OpenAI, DeepSeek, Gemini, Grok, Ollama...)."""

    RETRIES = 3

    def __init__(self, pid: str, model: str | None = None, effort: str = "medium", lang: str = "en",
                 context: str = "", max_turns: int | None = None, log=None, api_key: str | None = None,
                 base_url: str | None = None):
        # Without prompt caching the whole history is re-sent each turn: keep conversations short.
        super().__init__(pid, model, effort, lang, context, min(max_turns or 12, 12), log)
        self.system += JSON_RULE
        self.base_url = (base_url or self.provider["base_url"]).rstrip("/")
        self.api_key = api_key or env_key(pid)
        self.plain = False  # the provider rejected our optional parameters: send a bare request
        self.post = _post_json
        self.sleep = time.sleep

    def _body(self, schema: dict) -> dict:
        body = {"model": self.model, "messages": [{"role": "system", "content": self.system}, *self.messages]}
        if self.plain:
            return body
        p = self.provider
        thinking = re.search(THINKING_MODELS, self.model, re.I) is not None
        body["max_completion_tokens" if self.id == "openai" else "max_tokens"] = 16000 if thinking else 4000
        if self.id == "openai" and re.search(OPENAI_REASONING, self.model):
            body["reasoning_effort"] = {"low": "low", "medium": "medium"}.get(self.effort, "high")
        if p["json"] == "schema":
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "window_judgment", "strict": True, "schema": schema}}
        elif p["json"] == "object":
            body["response_format"] = {"type": "json_object"}
        return body

    def _call(self, window: int, schema: dict) -> tuple[dict, dict] | None:
        if not self.base_url:
            raise JudgeUnavailable("No base URL for the custom provider: pass --base-url")
        if self.provider["key"] == "required" and not self.api_key:
            env = " or ".join(self.provider["env"])
            raise JudgeUnavailable(f"No {self.provider['maker']} API key found. Set {env} to use {self.name}")
        headers = {"Content-Type": "application/json", "User-Agent": f"facial/{__version__}"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        url = self.base_url + "/chat/completions"

        attempt = 0
        while True:
            try:
                status, data = self.post(url, headers, self._body(schema), 180)
            except ValueError as exc:  # e.g. a base URL without https://
                raise JudgeUnavailable(f"Bad {self.name} URL {url!r} ({exc})") from exc
            except (OSError, http.client.HTTPException) as exc:  # network error or timeout
                status, data = 0, {"error": {"message": str(exc) or exc.__class__.__name__}}
            if status == 200:
                break
            if status == 400 and not self.plain:
                # Some providers or models reject JSON mode, reasoning effort or the token cap.
                self.log(f"  {self.name} rejected optional parameters ({_error_message(data)}); retrying without")
                self.plain = True
                continue
            if status and status not in (408, 409, 429) and status < 500:
                raise JudgeUnavailable(f"{self.name} request rejected ({status}: {_error_message(data)})")
            attempt += 1
            if attempt > self.RETRIES:
                self.log(f"  window {window}: {self.name} unavailable ({status or _error_message(data)}); "
                         "using heuristics for this window")
                return None
            self.sleep(min(20.0, 1.5 * 2 ** attempt))

        u = data.get("usage") or {}
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or u.get("prompt_cache_hit_tokens") or 0
        self.usage["input"] += (u.get("prompt_tokens") or 0) - cached
        self.usage["cache_read"] += cached
        self.usage["output"] += u.get("completion_tokens") or 0

        choice = (data.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content")
        if isinstance(text, list):  # content parts
            text = "".join(part.get("text", "") for part in text if isinstance(part, dict))
        parsed = extract_json(text)
        if parsed is None:
            self.log(f"  window {window}: {self.name} gave no usable JSON "
                     f"(finish_reason {choice.get('finish_reason')}); using heuristics for this window")
            return None
        return parsed, {"role": "assistant", "content": text}


class JevJudge:
    """Jev decides each window; the LLM, if any, only puts it into words (at low effort).

    When Jev is unsure of the intent, or can't answer, the LLM judges the whole window
    instead; without an LLM the words come from the same templates the rules use.
    """

    id = "jev"

    def __init__(self, jev: JevClient, llm: BaseJudge | None, lang: str = "en", hand_label=None, log=None):
        self.jev = jev
        self.llm = llm
        self.lang = lang
        self.hand_label = hand_label
        self.log = log or _stderr
        self.previous: dict | None = None
        if llm is not None:
            llm.system += WRITER_RULE
            llm.effort = "low"  # the decision is made: the words don't need deep thought

    @property
    def name(self) -> str:
        return "Jev" + (f" + {self.llm.name}" if self.llm else "")

    @property
    def usage(self) -> dict:
        return self.llm.usage if self.llm else {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}

    def _ask_llm(self, feats: dict, clip: dict | None, decision: dict | None = None) -> dict | None:
        if self.llm is None:
            return None
        try:
            return self.llm.judge(feats, clip, decision)
        except JudgeUnavailable as exc:
            self.log(f"{exc}; Jev carries on without {self.llm.name}.")
            self.llm = None
            return None

    def _template(self, feats: dict, d: dict) -> dict:
        """Words for a Jev decision when no LLM writes them."""
        g = feats["gesture"]
        dom, second = g.get("dominant"), g.get("second")
        gesture = self.hand_label(dom) if (dom and self.hand_label) else ""
        if gesture and second:
            gesture += " + " + self.hand_label(second)
        sep = "，" if self.lang == "zh" else " — "
        w = d["words"]
        return {"reading": f"{gesture}{sep}{d['intent']}" if gesture else d["intent"],
                "quote": _quote(feats["subtitle"]) if feats["subtitle"] else "",
                "evidence": [f"Jev: {JEV_INTENTS[d['intent_key']][0]} ({d['intent_certainty']:.2f})",
                             f"Jev: {d['emotion']} ({w['emotion_intensity']}); "
                             + ", ".join(w[k] for k, _ in top_traits(d["traits"]))]}

    def judge(self, feats: dict, clip: dict | None = None) -> dict | None:
        if self.jev is None and self.llm is None:
            raise JudgeUnavailable("Neither Jev nor an LLM is available")
        decision = None
        if self.jev is not None:
            try:
                decision = self.jev.decide(feats, self.lang, self.previous)
            except JudgeUnavailable as exc:
                self.jev = None
                if self.llm is None:
                    raise
                self.log(f"{exc}; carrying on without Jev.")

        if decision is None or not decision["certain"]:
            if decision is not None and self.llm is not None:
                self.log(f"  window {feats['window']}: Jev unsure ({decision['intent_certainty']:.2f}); "
                         f"{self.llm.name} decides")
            full = self._ask_llm(feats, clip)
            if full is not None:
                return full
            if decision is None:
                return None  # neither could judge this window: the caller uses the rules

        self.previous = decision
        scores = {k: decision[k] for k in DECISION_KEYS}
        written = self._ask_llm(feats, clip, decision)
        if written is not None:
            return normalise({**written, **scores}, "jev", writer=written["source"])
        return normalise({**self._template(feats, decision), **scores}, "jev")


def make_judge(pid: str, model: str | None = None, effort: str = "medium", lang: str = "en", context: str = "",
               max_turns: int | None = None, log=None, api_key: str | None = None,
               base_url: str | None = None, jev: dict | None = None, hand_label=None):
    """The judge for a provider id, or None for the rule-based scoring.

    jev: {"route", "model", "api_key", "base_url"} to let Jev make the decisions, with the
    provider (or the rule templates, for pid "heuristic") only writing the words.
    """
    if jev is not None:
        llm = make_judge(pid, model, effort, lang, context, max_turns, log, api_key, base_url)
        return JevJudge(JevClient(log=log or _stderr, **jev), llm, lang, hand_label, log)
    if pid in (RULES, "rules", "none"):
        return None
    if provider(pid)["kind"] == "anthropic":
        return ClaudeJudge(model=model or DEFAULT_MODEL, effort=effort, lang=lang, context=context,
                           max_turns=max_turns, log=log, api_key=api_key, base_url=base_url)
    return ChatJudge(pid, model=model, effort=effort, lang=lang, context=context, max_turns=max_turns, log=log,
                     api_key=api_key, base_url=base_url)


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
    """Transparent rule-based scores; used without an API key or when an AI judge call fails."""
    voice, g, spk = f["voice"], f["gesture"], f["speaker"]
    face, head, posture = g.get("face", {}), g.get("head", {}), g.get("posture", {})
    rel = face.get("vs_baseline", {})
    dom = g.get("dominant")
    blink = face.get("blink_per_min", 17)
    gaze_away = head.get("gaze_away", 0.2)
    loud = voice.get("loudness_rel_db", 0)
    beats = min(g.get("beats", 0), 3)
    pauses, fillers = voice.get("pauses", 0), voice.get("fillers", 0)
    touch = g.get("hand_to_face", 0)
    arms = posture.get("arms_open")

    open_gesture = 1.0 if dom and dom["shape"] in ("open_palm", "pointing", "fist") else 0.0
    forceful = 1.0 if dom and dom["shape"] in ("fist", "pointing") else 0.0
    steadiness = 1.0 - min(1.0, (head.get("yaw_std", 5) + head.get("pitch_std", 5)) / 20)
    nervous = (0.35 + 1.0 * max(0, rel.get("lip_press", 0)) + 0.8 * max(0, rel.get("brow_furrow", 0)) + 0.5 * touch
               + 0.006 * max(0, blink - 22) + 0.05 * pauses + 0.05 * fillers
               + 0.04 * max(0, voice.get("pitch_var_st", 2) - 3))
    traits = {
        "confident": (0.45 + 0.03 * loud + 0.12 * open_gesture + 0.1 * steadiness - 0.25 * gaze_away
                      - 0.3 * (nervous - 0.35) + 0.05 * beats),
        "nervous": nervous,
        "enthusiastic": (0.35 + 0.1 * min(g.get("energy", 0), 3) + 0.04 * max(0, voice.get("pitch_var_st", 2) - 2)
                         + 0.6 * max(0, rel.get("smile", 0)) + 0.04 * beats + 0.02 * max(0, loud)
                         + (0.05 if voice.get("speech_rate_wps", 0) > 3 else 0)),
        "warm": (0.4 + 0.3 * face.get("smile", 0) + 0.8 * rel.get("smile", 0) + 0.03 * min(head.get("nods", 0), 3)
                 - 0.8 * max(0, rel.get("frown", 0)) - 0.4 * max(0, rel.get("brow_furrow", 0))),
        "assertive": (0.4 + 0.03 * loud + 0.15 * forceful + 0.04 * beats + 0.08 * steadiness - 0.2 * gaze_away
                      - 0.04 * pauses),
        "defensive": (0.3 + (0.2 if arms is not None and arms <= 0.25 else 0) + 0.4 * touch
                      + 0.6 * max(0, rel.get("lip_press", 0)) + 0.25 * gaze_away + 0.3 * max(0, rel.get("brow_furrow", 0))),
        "engaged": (0.45 + 0.3 * (1 - gaze_away) + 0.1 * steadiness + 0.1 * voice.get("voiced_frac", 0)
                    - 0.2 * touch),
        "hesitant": (0.3 + 0.07 * pauses + 0.08 * fillers + 0.25 * gaze_away
                     + (0.15 if spk["state"] == "target" and voice.get("voiced_frac", 0) < 0.4 else 0)),
    }
    valence = (0.6 * face.get("smile", 0) - 0.6 * face.get("frown", 0)
               + 1.5 * rel.get("smile", 0) - 1.5 * rel.get("frown", 0) - 0.5 * rel.get("brow_furrow", 0))

    # The emotion is what the face shows, when it clearly shows something.
    expr = face.get("expression") or {}
    shown = expr.get("top", "neutral") != "neutral" and expr.get("share", 0) >= 0.25
    emotion = EXPRESSION_EMOTION[expr["top"]] if shown else "calm"
    emotion_intensity = expr.get("share", 0) if shown else 0.3

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
    second = g.get("second")
    if gesture_text and second:
        gesture_text += " + " + hand_label(second)
    sep = "，" if lang == "zh" else " — "
    reading = f"{gesture_text}{sep}{intent}" if gesture_text else intent
    return normalise({
        "reading": reading, "quote": _quote(text) if text else "",
        "emotion": emotion, "emotion_intensity": emotion_intensity, "traits": traits, "intent": intent,
        "intent_certainty": 0.45 if key != "default" else 0.3, "valence": valence,
        "evidence": ["rule-based: face expression, blendshapes, head motion, voice, hand shape"],
    }, "heuristic")


def judge_or_fallback(judge: BaseJudge | JevJudge | None, feats: dict, clip: dict | None, lang: str, hand_label,
                      log=None) -> tuple[dict, BaseJudge | JevJudge | None]:
    """Judge one window with the AI judge when possible, else heuristics.

    Returns the judgment and the judge to keep using (None once the judge is unusable,
    e.g. missing credentials or a rejected request).
    """
    log = log or _stderr
    result = None
    if judge is not None:
        try:
            result = judge.judge(feats, clip)
        except JudgeUnavailable as exc:
            log(f"{exc}; falling back to heuristic scoring.")
            judge = None
    if result is None:
        result = heuristic_judgment(feats, lang, hand_label)
    return result, judge


def judge_all(windows: list[dict], clip: dict, method: str, lang: str, model: str | None, effort: str,
              context: str, hand_label, log=None, progress=None, base_url: str | None = None,
              jev: dict | None = None) -> list[dict]:
    log = log or _stderr
    judge = make_judge(method, model=model, effort=effort, lang=lang, context=context, log=log, base_url=base_url,
                       jev=jev, hand_label=hand_label)

    results = []
    for i, feats in enumerate(windows):
        result, judge = judge_or_fallback(judge, feats, clip if i == 0 else None, lang, hand_label, log)
        results.append(result)
        log(f"  window {i + 1}/{len(windows)} [{result['source']}] {result['intent']}: {result['reading']}")
        if progress:
            progress((i + 1) / len(windows))

    if judge is not None and judge.usage["input"]:
        log(usage_line(judge.usage, getattr(judge, "llm", judge).name))
    if isinstance(judge, JevJudge) and judge.jev is not None and judge.jev.usage["input"]:
        log(f"Jev usage: {judge.jev.usage['input']} input tokens")
    return results


def usage_line(u: dict, name: str = "Claude") -> str:
    return (f"{name} usage: {u['input']} input + {u['cache_read']} cache-read + {u['cache_write']} "
            f"cache-write input tokens, {u['output']} output tokens")


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr)
