// The judgment layer in the browser: an AI model with the viewer's own API key (Claude,
// GPT, DeepSeek, Gemini, ...) or the transparent rules. Ports facial/judge.py.

import { Anthropic } from "./vendor/anthropic-sdk.mjs";
import { JevClient, decisionNote } from "./jev.js";
import {
  DEFAULT_MODEL, FALLBACK_MODELS, JEV_INTENTS, JSON_RULE, LANGUAGE_RULES, OPENAI_REASONING, PROVIDERS, SCHEMA,
  SYSTEM_PROMPT, THINKING_MODELS, WRITE_SCHEMA, WRITER_RULE,
} from "./prompt.js";

export { PROVIDERS };

/** Short label for a judgment's source ("Claude", "GPT", "Jev", ... or the rules). */
export const judgeTag = (source, rules = "Rules") => (source === "jev" ? "Jev" : PROVIDERS[source]?.name ?? rules);

const clamp = (v, lo = 0, hi = 1) => {
  const n = Number(v);
  return Math.round((Number.isFinite(n) ? Math.max(lo, Math.min(hi, n)) : (lo + hi) / 2) * 100) / 100;
};

export function normalise(j, source, writer = null) {
  const out = {
    reading: String(j.reading ?? "").trim(),
    quote: String(j.quote ?? "").trim().replace(/^["“”]+|["“”]+$/g, ""),
    confidence: clamp(j.confidence), focus: clamp(j.focus), tension: clamp(j.tension),
    intent: String(j.intent ?? "").trim(), intent_certainty: clamp(j.intent_certainty),
    valence: clamp(j.valence, -1, 1),
    evidence: (j.evidence || []).slice(0, 4).map(String),
    source,
  };
  if (writer) out.writer = writer; // the LLM that put a Jev decision into words
  return out;
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
  let gestureText = dom && handLabel ? handLabel(dom) : "";
  if (gestureText && g.second) gestureText += ` + ${handLabel(g.second)}`;
  const reading = gestureText ? `${gestureText}${lang === "zh" ? "，" : " — "}${intent}` : intent;
  return normalise({
    reading, quote: text ? quoteOf(text) : "", confidence, focus, tension, intent,
    intent_certainty: key !== "default" ? 0.45 : 0.3, valence,
    evidence: ["rule-based: blendshapes, head motion, voice, hand shape"],
  }, "heuristic");
}

/** The judge can't be used this session. reason: "key" | "blocked" | "setup" | "rejected". */
export class JudgeUnavailable extends Error {
  constructor(message, reason = "rejected") {
    super(message);
    this.reason = reason;
  }
}

/** The JSON object in a model's reply, tolerating code fences, <think> blocks and chatter. */
export function extractJson(text) {
  if (typeof text !== "string") return null;
  const t = text.replace(/<think>[\s\S]*?<\/think>/g, "");
  const a = t.indexOf("{"), b = t.lastIndexOf("}");
  if (a < 0 || b <= a) return null;
  try {
    const data = JSON.parse(t.slice(a, b + 1));
    return data && typeof data === "object" && !Array.isArray(data) ? data : null;
  } catch { return null; }
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const SCORE_KEYS = ["confidence", "focus", "tension", "valence", "intent", "intent_certainty"];
const pick = (o, keys) => Object.fromEntries(keys.map((k) => [k, o[k]]));

/** One conversation per session, one window per turn (BaseJudge in judge.py). */
class ConversationJudge {
  constructor({ provider, model, effort = "low", lang = "en", context = "", maxTurns = 24 }) {
    this.id = provider;
    this.provider = PROVIDERS[provider];
    this.name = this.provider.name;
    this.model = model || this.provider.model;
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

  /**
   * Returns a judgment, or null for a transient failure; throws JudgeUnavailable when it can't be used.
   * With a Jev `decision`, the scores and intent are settled and the model only writes the words.
   */
  async judge(feats, clip = null, decision = null) {
    if (clip) this.clip = clip;
    const turns = this.messages.filter((m) => m.role === "assistant").length;
    if (this.maxTurns && turns >= this.maxTurns) this.messages = [];
    const head = `Window ${feats.window}${feats.of ? `/${feats.of}` : " (live)"}:\n`;
    const first = !this.messages.length;
    const note = decision ? decisionNote(decision) : "";
    this.messages.push({ role: "user", content: (first ? this.intro(feats) : "") + head + JSON.stringify(feats) + note });
    let out;
    try {
      out = await this.call(decision ? WRITE_SCHEMA : SCHEMA);
    } catch (err) {
      this.messages.pop();
      throw err;
    }
    if (!out) { this.messages.pop(); return null; }
    this.messages.push(out.assistant); // append-only history
    const data = decision ? { ...out.data, ...pick(decision, SCORE_KEYS) } : out.data;
    const result = normalise(data, this.id);
    this.history.push({ window: feats.window, ...result });
    return result;
  }
}

/** Claude through the Anthropic SDK (ClaudeJudge in judge.py). */
export class ClaudeJudge extends ConversationJudge {
  constructor({ apiKey, model = DEFAULT_MODEL, ...rest }) {
    super({ provider: "claude", model, ...rest });
    this.client = new Anthropic({ apiKey, dangerouslyAllowBrowser: true, maxRetries: 3 });
  }

  async call(schema) {
    const params = {
      model: this.model,
      max_tokens: 16000,
      system: this.system,
      messages: this.messages,
      thinking: { type: "adaptive" },
      output_config: { effort: this.effort, format: { type: "json_schema", schema } },
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
      const transient = err instanceof Anthropic.RateLimitError || err instanceof Anthropic.APIConnectionError
        || (err instanceof Anthropic.APIError && err.status >= 500);
      if (transient) return null;
      const key = err instanceof Anthropic.AuthenticationError || err instanceof Anthropic.PermissionDeniedError;
      throw new JudgeUnavailable(err.message, key ? "key" : "rejected");
    }
    if (response.stop_reason !== "end_turn") return null;
    const text = [...response.content].reverse().find((b) => b.type === "text")?.text || "";
    const data = extractJson(text);
    return data && { data, assistant: { role: "assistant", content: response.content } };
  }
}

/** Any OpenAI-compatible chat-completions API: OpenAI, DeepSeek, Gemini, Grok, Ollama... (ChatJudge). */
export class ChatJudge extends ConversationJudge {
  // Without prompt caching the whole history is re-sent each turn: keep conversations short.
  constructor({ apiKey, baseUrl, fetchImpl, maxTurns = 12, ...rest }) {
    super({ maxTurns, ...rest });
    this.system += JSON_RULE;
    this.apiKey = apiKey || "";
    this.baseUrl = (baseUrl || this.provider.base_url || "").replace(/\/+$/, "");
    this.fetch = fetchImpl || ((...a) => fetch(...a));
    this.plain = false; // the provider rejected our optional parameters: send a bare request
    this.reached = false;
    this.retryMs = 1500;
  }

  body(schema) {
    const body = { model: this.model, messages: [{ role: "system", content: this.system }, ...this.messages] };
    if (this.plain) return body;
    const p = this.provider;
    const thinking = new RegExp(THINKING_MODELS, "i").test(this.model);
    body[this.id === "openai" ? "max_completion_tokens" : "max_tokens"] = thinking ? 16000 : 4000;
    if (this.id === "openai" && new RegExp(OPENAI_REASONING).test(this.model)) {
      body.reasoning_effort = { low: "low", medium: "medium" }[this.effort] || "high";
    }
    if (p.json === "schema") {
      body.response_format = { type: "json_schema", json_schema: { name: "window_judgment", strict: true, schema } };
    } else if (p.json === "object") body.response_format = { type: "json_object" };
    return body;
  }

  async call(schema) {
    if (!this.baseUrl) throw new JudgeUnavailable("No base URL", "setup");
    if (this.provider.key === "required" && !this.apiKey) throw new JudgeUnavailable("No API key", "key");
    const headers = { "Content-Type": "application/json" };
    if (this.apiKey) headers.Authorization = `Bearer ${this.apiKey}`;
    for (let attempt = 0; ;) {
      let res = null;
      let data = {};
      try {
        res = await this.fetch(`${this.baseUrl}/chat/completions`, { method: "POST", headers, body: JSON.stringify(this.body(schema)) });
        data = await res.json().catch(() => ({}));
        this.reached = true;
      } catch (err) {
        // A page can't tell a dropped connection from a provider that refuses browsers (CORS).
        if (!this.reached) throw new JudgeUnavailable(err.message || String(err), "blocked");
      }
      const status = res ? res.status : 0;
      if (res && res.ok) {
        const choice = (data.choices || [])[0] || {};
        let text = choice.message?.content;
        if (Array.isArray(text)) text = text.map((part) => part?.text || "").join("");
        const parsed = extractJson(text);
        return parsed && { data: parsed, assistant: { role: "assistant", content: text } };
      }
      const detail = errorMessage(data) || `HTTP ${status}`;
      if (status === 400 && !this.plain) { this.plain = true; continue; }
      if (status === 401 || status === 403) throw new JudgeUnavailable(detail, "key");
      if (status && ![408, 409, 429].includes(status) && status < 500) throw new JudgeUnavailable(detail, "rejected");
      attempt += 1;
      if (attempt > 2) return null;
      await sleep(this.retryMs * 2 ** attempt);
    }
  }
}

function errorMessage(data) {
  let err = data && (data.error ?? data);
  if (Array.isArray(err)) err = err[0];
  if (err && typeof err === "object") err = err.message || err.detail || "";
  return String(err || "").slice(0, 200);
}

/** Is there enough in the settings to call this provider? */
export function judgeReady({ provider, apiKey, baseUrl }) {
  const p = PROVIDERS[provider];
  return Boolean(p && (p.key !== "required" || apiKey) && (p.base_url || baseUrl));
}

/** The AI judge for these settings, or null for the rules. */
export function makeJudge({ provider = "claude", apiKey, model, baseUrl, effort = "low", lang = "en", context = "", fetchImpl }) {
  if (!judgeReady({ provider, apiKey, baseUrl })) return null;
  const common = { model: model || undefined, effort, lang, context };
  return PROVIDERS[provider].kind === "anthropic"
    ? new ClaudeJudge({ apiKey, ...common })
    : new ChatJudge({ provider, apiKey, baseUrl, fetchImpl, ...common });
}

const NOT_CHAT = /embed|tts|whisper|dall-e|image|audio|realtime|moderation|transcribe|search|computer-use|babbage|davinci|rerank|ocr|guard|veo|imagen|aqa|learnlm|video|speech/i;

/** Model names the provider offers to this key (also a quick key check: throws with .status). */
export async function listModels({ provider, apiKey, baseUrl, fetchImpl = (...a) => fetch(...a) }) {
  const p = PROVIDERS[provider];
  if (p.kind === "anthropic") {
    const client = new Anthropic({ apiKey, dangerouslyAllowBrowser: true, maxRetries: 0 });
    const page = await client.models.list({ limit: 100 });
    return page.data.map((m) => m.id);
  }
  const base = (baseUrl || p.base_url).replace(/\/+$/, "");
  const res = await fetchImpl(`${base}/models`, { headers: apiKey ? { Authorization: `Bearer ${apiKey}` } : {} });
  if (!res.ok) throw Object.assign(new Error(`HTTP ${res.status}`), { status: res.status });
  const data = await res.json();
  const ids = (data.data || data.models || []).map((m) => String(m.id || m.name || "").replace(/^models\//, ""));
  return [...new Set(ids.filter((id) => id && !NOT_CHAT.test(id)))].sort();
}

/**
 * Jev decides each window; the LLM, if any, only puts it into words at low effort (JevJudge in
 * judge.py). When Jev is unsure of the intent, or can't answer, the LLM judges the whole window.
 */
class JevJudge {
  constructor({ jev, llm, lang, handLabel, onNotice }) {
    this.jev = jev;
    this.llm = llm;
    this.lang = lang;
    this.handLabel = handLabel;
    this.onNotice = onNotice;
    this.previous = null;
    if (llm) {
      llm.system += WRITER_RULE;
      llm.effort = "low"; // the decision is made: the words don't need deep thought
    }
  }

  get id() { return "jev"; }

  async askLlm(feats, clip, decision = null) {
    if (!this.llm) return null;
    try {
      return await this.llm.judge(feats, clip, decision);
    } catch (err) {
      console.warn(err);
      this.onNotice(notice(this.llm.name, err, this.jev ? "Jev" : "rules"));
      this.llm = null;
      return null;
    }
  }

  template(feats, d) {
    const g = feats.gesture;
    let gesture = g.dominant && this.handLabel ? this.handLabel(g.dominant) : "";
    if (gesture && g.second) gesture += ` + ${this.handLabel(g.second)}`;
    const w = d.words;
    return {
      reading: gesture ? `${gesture}${this.lang === "zh" ? "，" : " — "}${d.intent}` : d.intent,
      quote: feats.subtitle ? quoteOf(feats.subtitle) : "",
      evidence: [`Jev: ${JEV_INTENTS[d.intent_key][0]} (${d.intent_certainty.toFixed(2)})`,
        `Jev: ${w.confidence}, ${w.focus}, ${w.tension}, ${w.valence}`],
    };
  }

  /** A judgment, or null when neither Jev nor the LLM could judge this window. */
  async judge(feats, clip) {
    let decision = null;
    if (this.jev) {
      try {
        decision = await this.jev.decide(feats, this.lang, this.previous);
      } catch (err) {
        console.warn(err);
        this.onNotice(notice("Jev", err, this.llm ? this.llm.name : "rules"));
        this.jev = null;
      }
    }
    if (!decision || !decision.certain) {
      const full = await this.askLlm(feats, clip);
      if (full) return full;
      if (!decision) return null;
    }
    this.previous = decision;
    const scores = pick(decision, SCORE_KEYS);
    const written = await this.askLlm(feats, clip, decision);
    if (written) return normalise({ ...written, ...scores }, "jev", written.source);
    return normalise({ ...this.template(feats, decision), ...scores }, "jev");
  }
}

/** What went wrong with `name`, and what takes over. */
function notice(name, err, fallback = "rules") {
  if (err.reason === "key") return `${name} key rejected, using ${fallback}`;
  if (err.reason === "blocked") return `${name} can't be reached from this page, using ${fallback}`;
  return `${name} unavailable (${String(err.message).slice(0, 80)}), using ${fallback}`;
}

/**
 * Which judge a browser session uses: the chosen AI when it's set up, Jev first when it's on,
 * falling back per window to the rules.
 */
export class BrowserJudge {
  constructor({ provider = "claude", apiKey, model, baseUrl, lang, context, handLabel, onNotice, fetchImpl, jev = null }) {
    this.lang = lang;
    this.handLabel = handLabel;
    this.onNotice = onNotice || (() => {});
    this.llm = makeJudge({ provider, apiKey, model, baseUrl, lang, context, fetchImpl });
    if (jev && jev.apiKey) {
      this.judgeImpl = new JevJudge({
        jev: new JevClient({ ...jev, fetchImpl }), llm: this.llm, lang, handLabel, onNotice: this.onNotice,
      });
    } else this.judgeImpl = this.llm;
  }

  get name() { return this.judgeImpl ? this.judgeImpl.id : "heuristic"; }

  /** Who decides (and writes): "Jev · Claude", "GPT", "Rules"... */
  get label() {
    if (!this.judgeImpl) return judgeTag("heuristic");
    if (this.judgeImpl instanceof JevJudge) return this.judgeImpl.llm ? `Jev · ${this.judgeImpl.llm.name}` : "Jev";
    return this.judgeImpl.name;
  }

  async judge(feats, clip) {
    if (this.judgeImpl) {
      try {
        const res = await this.judgeImpl.judge(feats, clip);
        if (res) return res;
        if (this.judgeImpl instanceof JevJudge && !this.judgeImpl.jev && !this.judgeImpl.llm) this.judgeImpl = null;
      } catch (err) {
        console.warn(err);
        this.onNotice(notice(this.judgeImpl.name || "AI", err));
        this.judgeImpl = null;
      }
    }
    return heuristicJudgment(feats, this.lang, this.handLabel);
  }
}
