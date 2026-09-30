// The judgment layer in the browser: Claude (with the viewer's own API key) or the
// transparent rules. Ports ClaudeJudge / heuristic_judgment from facial/judge.py.

import { Anthropic } from "./vendor/anthropic-sdk.mjs";
import { DEFAULT_MODEL, FALLBACK_MODELS, LANGUAGE_RULES, SCHEMA, SYSTEM_PROMPT } from "./prompt.js";

const clamp = (v, lo = 0, hi = 1) => {
  const n = Number(v);
  return Math.round((Number.isFinite(n) ? Math.max(lo, Math.min(hi, n)) : (lo + hi) / 2) * 100) / 100;
};

export function normalise(j, source) {
  return {
    reading: String(j.reading ?? "").trim(),
    quote: String(j.quote ?? "").trim().replace(/^["“”]+|["“”]+$/g, ""),
    confidence: clamp(j.confidence), focus: clamp(j.focus), tension: clamp(j.tension),
    intent: String(j.intent ?? "").trim(), intent_certainty: clamp(j.intent_certainty),
    valence: clamp(j.valence, -1, 1),
    evidence: (j.evidence || []).slice(0, 4).map(String),
    source,
  };
}

const INTENTS = {
  en: {
    listening: "Listening", silent: "Pausing", offscreen: "Off camera", question: "Asking a question",
    pointing: "Emphasizing a point", fist: "Stressing conviction", pinch: "Making a precise point",
    two_fingers: "Enumerating", palm_up: "Presenting / explaining", palm_down: "Asserting / settling",
    palm_out: "Holding back / clarifying", self_touch: "Self-soothing", default: "Stating a position",
  },
  zh: {
    listening: "倾听", silent: "停顿", offscreen: "不在画面", question: "提出问题",
    pointing: "强调要点", fist: "表达决心", pinch: "精确论证", two_fingers: "逐条列举",
    palm_up: "陈述解释", palm_down: "断言定调", palm_out: "保留澄清", self_touch: "自我安抚", default: "陈述立场",
  },
};

function quoteOf(text) {
  const clauses = text.split(/[,.;!?，。；！？]/).map((c) => c.trim())
    .filter((c) => c.split(/\s+/).length >= 3 || (c && /[^\x00-\x7f]/.test(c) && c.length >= 4));
  return clauses.length ? clauses.reduce((a, b) => (b.length > a.length ? b : a)).slice(0, 70) : text.slice(0, 70);
}

/** Transparent rule-based scores (heuristic_judgment in judge.py). */
export function heuristicJudgment(f, lang = "en", handLabel = null) {
  const voice = f.voice, g = f.gesture, spk = f.speaker;
  const face = g.face || {}, head = g.head || {}, rel = face.vs_baseline || {};
  const dom = g.dominant;
  const blink = face.blink_per_min ?? 17;
  const get = (o, k, d = 0) => (o[k] ?? d);

  const tension = 0.15 + Math.max(0, get(rel, "lip_press")) + 0.8 * Math.max(0, get(rel, "brow_furrow"))
    + 0.5 * get(g, "hand_to_face") + 0.006 * Math.max(0, blink - 22)
    + 0.05 * get(voice, "pauses") + 0.05 * get(voice, "fillers") + 0.04 * Math.max(0, get(voice, "pitch_var_st", 2) - 3);
  const open = dom && ["open_palm", "pointing", "fist"].includes(dom.shape) ? 1 : 0;
  const steadiness = 1 - Math.min(1, (get(head, "yaw_std", 5) + get(head, "pitch_std", 5)) / 20);
  const confidence = 0.45 + 0.03 * get(voice, "loudness_rel_db") + 0.12 * open + 0.1 * steadiness
    - 0.25 * get(head, "gaze_away") - 0.3 * (tension - 0.15) + 0.05 * Math.min(get(g, "beats"), 3);
  const focus = 0.45 + 0.3 * (1 - get(head, "gaze_away", 0.3)) + 0.1 * steadiness + 0.1 * get(voice, "voiced_frac")
    - 0.2 * get(g, "hand_to_face");
  const valence = 0.6 * get(face, "smile") - 0.6 * get(face, "frown") + 1.5 * get(rel, "smile") - 1.5 * get(rel, "frown")
    - 0.5 * get(rel, "brow_furrow");

  const text = f.subtitle;
  let key;
  if (["listening", "silent", "offscreen"].includes(spk.state)) key = spk.state;
  else if (get(g, "hand_to_face") > 0.3) key = "self_touch";
  else if (/[?？]\s*$/.test(text)) key = "question";
  else if (dom && ["pointing", "fist", "pinch", "two_fingers"].includes(dom.shape)) key = dom.shape;
  else if (dom && dom.shape === "open_palm" && ["palm_up", "palm_down", "palm_out"].includes(dom.facing)) key = dom.facing;
  else key = "default";
  const intent = INTENTS[lang][key];
  const gestureText = dom && handLabel ? handLabel(dom) : "";
  const reading = gestureText ? `${gestureText}${lang === "zh" ? "，" : " — "}${intent}` : intent;
  return normalise({
    reading, quote: text ? quoteOf(text) : "", confidence, focus, tension, intent,
    intent_certainty: key !== "default" ? 0.45 : 0.3, valence,
    evidence: ["rule-based: blendshapes, head motion, voice, hand shape"],
  }, "heuristic");
}

/** One Claude conversation per session, one window per turn (ClaudeJudge in judge.py). */
export class ClaudeJudge {
  constructor({ apiKey, model = DEFAULT_MODEL, effort = "low", lang = "en", context = "", maxTurns = 24 }) {
    this.client = new Anthropic({ apiKey, dangerouslyAllowBrowser: true, maxRetries: 3 });
    this.model = model;
    this.effort = effort;
    this.context = context;
    this.maxTurns = maxTurns;
    this.system = `${SYSTEM_PROMPT}- ${LANGUAGE_RULES[lang]}\n`;
    this.messages = [];
    this.clip = null;
    this.history = [];
  }

  intro(feats) {
    const clip = this.clip || {};
    const ctx = this.context || "none given";
    let text = clip.live
      ? `Live camera session, judged window by window as it happens. Context from the user: ${ctx}.\n`
      : `Clip: ${(clip.duration || 0).toFixed(1)}s in ${feats.of} windows. Context from the user: ${ctx}.\n`;
    if (clip.baseline && Object.keys(clip.baseline).length) text += `Clip baselines (medians): ${JSON.stringify(clip.baseline)}\n`;
    if (this.history.length) {
      text += "Earlier windows, summarised (most recent last):\n" + this.history.slice(-6).map((h) =>
        `- W${h.window}: ${h.intent} | confidence ${h.confidence.toFixed(2)}, focus ${h.focus.toFixed(2)}, `
        + `tension ${h.tension.toFixed(2)}, valence ${h.valence >= 0 ? "+" : ""}${h.valence.toFixed(2)} | ${h.reading}`).join("\n") + "\n";
    }
    return `${text}\n`;
  }

  /** Returns a judgment, or null for a transient failure; throws when Claude can't be used at all. */
  async judge(feats, clip = null) {
    if (clip) this.clip = clip;
    const turns = this.messages.filter((m) => m.role === "assistant").length;
    if (this.maxTurns && turns >= this.maxTurns) this.messages = [];
    const head = `Window ${feats.window}${feats.of ? `/${feats.of}` : " (live)"}:\n`;
    const first = !this.messages.length;
    this.messages.push({ role: "user", content: (first ? this.intro(feats) : "") + head + JSON.stringify(feats) });
    const params = {
      model: this.model,
      max_tokens: 16000,
      system: this.system,
      messages: this.messages,
      thinking: { type: "adaptive" },
      output_config: { effort: this.effort, format: { type: "json_schema", schema: SCHEMA } },
      cache_control: { type: "ephemeral" },
    };
    if (FALLBACK_MODELS.includes(this.model)) {
      params.betas = ["server-side-fallback-2026-07-01"];
      params.fallbacks = "default";
    }
    let response;
    try {
      response = await this.client.beta.messages.create(params);
    } catch (err) {
      this.messages.pop();
      const transient = err instanceof Anthropic.RateLimitError || err instanceof Anthropic.APIConnectionError
        || (err instanceof Anthropic.APIError && err.status >= 500);
      if (transient) return null;
      throw err;
    }
    if (response.stop_reason !== "end_turn") { this.messages.pop(); return null; }
    const text = [...response.content].reverse().find((b) => b.type === "text")?.text || "";
    let data;
    try { data = JSON.parse(text); } catch { this.messages.pop(); return null; }
    this.messages.push({ role: "assistant", content: response.content }); // append-only history
    const result = normalise(data, "claude");
    this.history.push({ window: feats.window, ...result });
    return result;
  }
}

/** Which judge a browser session uses: Claude when a key is set (falls back per window), else rules. */
export class BrowserJudge {
  constructor({ apiKey, lang, context, handLabel, onNotice }) {
    this.lang = lang;
    this.handLabel = handLabel;
    this.onNotice = onNotice || (() => {});
    this.claude = apiKey ? new ClaudeJudge({ apiKey, lang, context }) : null;
  }

  get name() { return this.claude ? "claude" : "heuristic"; }

  async judge(feats, clip) {
    if (this.claude) {
      try {
        const res = await this.claude.judge(feats, clip);
        if (res) return res;
      } catch (err) {
        this.onNotice(err instanceof Anthropic.AuthenticationError ? "Claude key rejected, using rules"
          : `Claude unavailable (${err.status || err.name}), using rules`);
        this.claude = null;
      }
    }
    return heuristicJudgment(feats, this.lang, this.handLabel);
  }
}
