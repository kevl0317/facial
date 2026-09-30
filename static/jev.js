// Jev, TypeSafe AI's "System One" decision model, as the judge's decision layer in the
// browser. Port of facial/jev.py: Jev decides each window's scores, intent and tone from a
// plain-words description of the measurements; the LLM only writes the words.

import { handLabel } from "./hud.js";
import { JEV_ESCALATE, JEV_INTENTS, JEV_QUESTIONS, JEV_ROUTES, JEV_SCALES } from "./prompt.js";

export { JEV_ROUTES };

const band = (x, steps, rest) => (steps.find(([threshold]) => x >= threshold) || [null, rest])[1];

/** A window's five fields in plain sentences, for Jev (describe() in jev.py). */
export function describe(f, previous = null) {
  const scene = f.scene, spk = f.speaker, v = f.voice, g = f.gesture;
  const lines = [];

  const shot = { "close-up": "a close-up", medium: "a medium shot", wide: "a wide shot", cutaway: "a cutaway with no face" }[scene.shot] || "a shot";
  const extra = ((scene.faces || 0) > 1 ? ", several faces" : "") + (scene.cuts ? ", a camera cut" : "");
  lines.push(`Scene: ${shot}${extra}.`);
  lines.push({
    target: "The person on camera is the one talking.",
    listening: "The person on camera is listening while someone else talks.",
    offscreen: "The speaker is off camera.",
    silent: "Nobody is talking.",
  }[spk.state] || "It is unclear who is talking.");
  const text = (f.subtitle || "").trim();
  lines.push(text ? `Said: "${text}"` : "Nothing was said.");

  if ((v.voiced_frac || 0) < 0.15) {
    lines.push("Voice: barely any speech.");
  } else {
    const parts = [];
    if (v.loudness_rel_db != null) {
      parts.push(band(v.loudness_rel_db, [[4, "much louder than usual"], [1.5, "a bit louder than usual"],
        [-1.5, "at their usual loudness"], [-4, "a bit quieter than usual"]], "much quieter than usual"));
    }
    if (v.pitch_rel_st != null) {
      parts.push(band(v.pitch_rel_st, [[2, "higher-pitched than usual"], [-2, "at their usual pitch"]], "lower-pitched than usual"));
    }
    if (v.pitch_var_st != null) {
      parts.push(band(v.pitch_var_st, [[4, "very animated intonation"], [2.5, "lively intonation"], [1.2, "normal intonation"]],
        "flat, monotone intonation"));
    }
    if (v.speech_rate_wps != null) {
      parts.push(band(v.speech_rate_wps, [[3.5, "speaking fast"], [1.8, "speaking at a normal pace"]], "speaking slowly"));
    }
    const pauses = Math.trunc(v.pauses || 0);
    parts.push(pauses < 3 ? ["no pauses", "one pause", "two pauses"][pauses] : "several pauses");
    const fillers = v.fillers || 0;
    if (fillers) parts.push(fillers >= 3 ? "many filler words" : "a filler word or two");
    lines.push(`Voice: ${parts.join(", ")}.`);
  }

  const dom = g.dominant;
  if ((g.hands_visible || 0) < 0.15 || !dom) {
    lines.push("Hands: not visible.");
  } else {
    const parts = [`${handLabel(dom, "en")} ${band(dom.share || 0, [[0.6, "for most of the window"], [0.3, "for part of the window"]], "briefly")}`];
    if (g.second) parts.push(`the other hand: ${handLabel(g.second, "en")}`);
    if ((g.two_hands || 0) >= 0.4) parts.push("both hands up much of the time");
    parts.push(band(g.energy || 0, [[2.0, "animated movement"], [0.7, "some movement"]], "hands mostly still"));
    if ((g.beats || 0) >= 2) parts.push("beat gestures on the words");
    const touch = g.hand_to_face || 0;
    if (touch >= 0.1) parts.push(touch >= 0.3 ? "touches their face often" : "touches their face briefly");
    lines.push(`Hands: ${parts.join("; ")}.`);
  }

  const face = g.face;
  if (face) {
    const rel = face.vs_baseline || {};
    const parts = [];
    if ((rel.smile || 0) >= 0.15) parts.push("smiling more than usual");
    else if ((rel.smile || 0) <= -0.15) parts.push("smiling less than usual");
    if ((rel.frown || 0) >= 0.08) parts.push("frowning");
    if ((rel.brow_furrow || 0) >= 0.1) parts.push("brow furrowed more than usual");
    if ((rel.brow_raise || 0) >= 0.1) parts.push("eyebrows raised");
    if ((rel.lip_press || 0) >= 0.1) parts.push("lips pressed together");
    const blink = face.blink_per_min ?? 17;
    if (blink >= 30) parts.push("blinking a lot");
    else if (blink <= 6) parts.push("hardly blinking");
    lines.push(`Face: ${parts.length ? parts.join(", ") : "their usual expression"}.`);
  }

  const head = g.head;
  if (head) {
    const parts = [
      band((head.yaw_std || 0) + (head.pitch_std || 0), [[10, "head moving a lot"], [4, "some head movement"]], "head steady"),
      band(head.gaze_away || 0, [[0.4, "often looking away"], [0.1, "sometimes looking away"]], "looking toward the camera"),
    ];
    if ((head.nods || 0) >= 2) parts.push("nodding");
    lines.push(`Head: ${parts.join(", ")}.`);
  }

  const posture = g.posture;
  if (posture) {
    const parts = [];
    const arms = posture.arms_open;
    if (arms != null && arms >= 0.6) parts.push("arms open wide");
    else if (arms != null && arms <= 0.25) parts.push("arms held close");
    if (Math.abs(posture.shoulder_tilt || 0) >= 6) parts.push("shoulders tilted");
    if (parts.length) lines.push(`Posture: ${parts.join(", ")}.`);
  }

  if (previous) {
    lines.push(`Just before: ${JEV_INTENTS[previous.intent_key][0].toLowerCase()}, came across `
      + `${previous.words.confidence} and ${previous.words.tension}.`);
  }
  return lines.join("\n");
}

const round2 = (x) => Math.round(x * 100) / 100;

function scalePosition(answer, levels) {
  let score = answer.score;
  if (score == null) {
    const probs = answer.probabilities || {};
    const total = Object.values(probs).reduce((a, p) => a + Number(p), 0);
    if (!total) return null;
    score = Object.entries(probs).reduce((a, [k, p]) => a + Number(k) * Number(p), 0) / total;
  }
  return Math.max(0, Math.min(1, Number(score) / (levels - 1)));
}

/** Scores (0-1, valence -1..1), intent and certainty from a /systemone response (parse_decision). */
export function parseDecision(data, lang = "en") {
  const answers = data && typeof data.answers === "object" && data.answers ? data.answers : (data || {});
  const out = { words: {} };
  for (const [key, [, levels]] of Object.entries(JEV_SCALES)) {
    const pos = scalePosition(answers[key] || {}, levels.length);
    if (pos == null) return null;
    out[key] = key === "valence" ? round2(pos * 2 - 1) : round2(pos);
    out.words[key] = levels[Math.floor(pos * (levels.length - 1) + 0.5)];
  }
  const a = answers.intent || {};
  const probs = Object.fromEntries(Object.entries(a.probabilities || {}).map(([k, p]) => [k, Number(p)]));
  const keys = Object.keys(probs);
  const key = a.choice || (keys.length ? keys.reduce((x, y) => (probs[y] > probs[x] ? y : x)) : null);
  if (!(key in JEV_INTENTS)) return null;
  const sure = Number(probs[key] ?? a.confidence ?? 0);
  return Object.assign(out, {
    intent_key: key, intent: JEV_INTENTS[key][lang === "zh" ? 1 : 0],
    intent_certainty: round2(sure), certain: sure >= JEV_ESCALATE,
  });
}

/** Jev's decision, as the LLM is told it (decision_note). */
export function decisionNote(d) {
  const w = d.words;
  return `\nJev's decision for this window (settled): intent "${d.intent}"; comes across ${w.confidence}; `
    + `${w.focus}; ${w.tension}; tone ${w.valence}. Reply with only reading, quote and evidence, written to fit this decision.`;
}

export class JevUnavailable extends Error {
  constructor(message, reason = "rejected") {
    super(message);
    this.reason = reason;
  }
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** Talks to Jev's /systemone endpoint (JevClient in jev.py). */
export class JevClient {
  constructor({ route = "openrouter", apiKey = "", model = "", fetchImpl } = {}) {
    const r = JEV_ROUTES[route] || JEV_ROUTES.openrouter;
    this.route = route;
    this.name = r.name;
    this.apiKey = apiKey;
    this.model = model || r.model;
    this.url = `${r.base_url}/systemone`;
    this.fetch = fetchImpl || ((...a) => fetch(...a));
    this.reached = false;
    this.retryMs = 500;
  }

  /** Jev's decision, or null when it's busy just now; throws JevUnavailable when it can't be used. */
  async decide(feats, lang = "en", previous = null) {
    if (!this.apiKey) throw new JevUnavailable("No Jev API key", "key");
    const body = JSON.stringify({ model: this.model, state: describe(feats, previous), questions: JEV_QUESTIONS });
    const headers = { "Content-Type": "application/json", Authorization: `Bearer ${this.apiKey}` };
    for (let attempt = 0; attempt <= 2; attempt += 1) {
      let res = null;
      let data = {};
      try {
        res = await this.fetch(this.url, { method: "POST", headers, body });
        data = await res.json().catch(() => ({}));
        this.reached = true;
      } catch (err) {
        // A page can't tell a dropped connection from an API that refuses browsers (CORS).
        if (!this.reached) throw new JevUnavailable(err.message || String(err), "blocked");
      }
      if (res && res.ok) return parseDecision(data, lang);
      const status = res ? res.status : 0;
      if (status === 401 || status === 403) throw new JevUnavailable(`HTTP ${status}`, "key");
      if (status && ![408, 409, 429].includes(status) && status < 500) {
        const msg = data?.error?.message || data?.detail || `HTTP ${status}`;
        throw new JevUnavailable(String(typeof msg === "string" ? msg : JSON.stringify(msg)).slice(0, 120), "rejected");
      }
      if (attempt < 2) await sleep(this.retryMs * 2 ** attempt);
    }
    return null;
  }
}
