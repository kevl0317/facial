// Cartoon HUD for live mode: white "stickers" with ink outlines and hard shadows.
// facial/render.py draws the same design onto rendered videos.

import { PROVIDERS } from "./prompt.js";

const tag = (source, rules) => (source === "jev" ? "Jev" : PROVIDERS[source]?.name ?? rules);

/** Who decided and who wrote a judgment, e.g. "Jev · Claude" (judge_label in providers.py). */
export const judgeLabel = (j, rules = "Rules") => (j.writer ? `${tag(j.source, rules)} · ${tag(j.writer, rules)}` : tag(j.source, rules));

export const STRINGS = {
  en: {
    left: "Left hand", right: "Right hand", verdict: "Verdict", confidence: "Confident", focus: "Focused",
    tension: "Tense", intent: "Intent", arc: "Mood", vision: "VISION", rules: "Rules",
    footer: "MediaPipe · {judge} · demo only", collecting: "Warming up", judging: "Thinking", live: "LIVE", playing: "PLAYING",
  },
  zh: {
    left: "左手", right: "右手", verdict: "综合判定", confidence: "自信", focus: "专注",
    tension: "紧张", intent: "意图", arc: "情绪弧", vision: "VISION", rules: "规则",
    footer: "MediaPipe · {judge} · 仅供演示", collecting: "准备中", judging: "判定中", live: "直播", playing: "播放中",
  },
};
const SHAPES = {
  en: { open_palm: "open palm", fist: "fist", pointing: "pointing", two_fingers: "two fingers", thumb: "thumb out", pinch: "pinch", relaxed: "relaxed" },
  zh: { open_palm: "张开手掌", fist: "握拳", pointing: "指点", two_fingers: "两指", thumb: "竖拇指", pinch: "捏合", relaxed: "放松" },
};
const FACINGS = {
  en: { palm_up: "palm up", palm_down: "palm down", palm_out: "palm out", back_out: "back out", palm_side: "sideways" },
  zh: { palm_up: "掌心向上", palm_down: "掌心向下", palm_out: "掌心朝外", back_out: "手背朝外", palm_side: "侧掌" },
};
const AXES = {
  en: { horizontal: "horizontal", up: "raised", down: "lowered" },
  zh: { horizontal: "横", up: "竖起", down: "下垂" },
};

export function handLabel(h, lang) {
  const side = STRINGS[lang][h.side || h.hand];
  const shape = SHAPES[lang][h.shape] || h.shape;
  const detail = ["open_palm", "relaxed"].includes(h.shape) ? FACINGS[lang][h.facing] : AXES[lang][h.axis];
  return lang === "zh" ? `${side} · ${shape}（${detail}）` : `${side} · ${shape} (${detail})`;
}

const FONT = 'Fredoka, "PingFang SC", "Hiragino Sans GB", "Noto Sans CJK SC", "Microsoft YaHei", ui-rounded, system-ui, sans-serif';
export const THEME = {
  ink: "#1e1b2e", paper: "#fff6e6", white: "#ffffff", muted: "#6b6680",
  blue: "#5b6cff", yellow: "#ffc83d", coral: "#ff6b6b", mint: "#33d1a0", orange: "#ff9f43",
};
const T = THEME;
const HAND_COLOURS = [THEME.yellow, THEME.mint]; // pointer + dot colour per hand label
const HAND_EDGES = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20]];

const smoothstep = (x) => { x = Math.min(1, Math.max(0, x)); return x * x * (3 - 2 * x); };
const hex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
const mix = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t));

/** Mood colour: coral (negative) -> yellow (neutral) -> mint (positive). */
export function valenceColour(v) {
  const c = v >= 0 ? mix(hex(T.yellow), hex(T.mint), v) : mix(hex(T.yellow), hex(T.coral), -v);
  return `rgb(${c.join(",")})`;
}

const font = (weight, size) => `${weight} ${size}px ${FONT}`;

export function wrap(ctx, text, maxW, maxLines) {
  const tokens = text.match(/[　-鿿＀-￯]|[^\s　-鿿＀-￯]+\s*|\s+/g) || [];
  const lines = [];
  let cur = "";
  for (const tok of tokens) {
    if (!cur || ctx.measureText(cur + tok).width <= maxW) cur += tok;
    else { lines.push(cur.trimEnd()); cur = tok.trimStart(); }
  }
  if (cur.trim()) lines.push(cur.trimEnd());
  if (lines.length > maxLines) {
    lines.length = maxLines;
    let last = lines[maxLines - 1];
    while (last && ctx.measureText(last + "…").width > maxW) last = last.slice(0, -1);
    lines[maxLines - 1] = last.trimEnd() + "…";
  }
  return lines;
}

function rr(ctx, x, y, w, h, r) { ctx.beginPath(); ctx.roundRect(x, y, w, h, Math.min(r, h / 2, w / 2)); }

/** White card with an ink outline and a hard offset shadow. */
function sticker(ctx, x, y, w, h, r, u, { fill = T.white, shadow = 4, line = 3 } = {}) {
  if (shadow) { rr(ctx, x + shadow * u, y + shadow * u, w, h, r); ctx.fillStyle = T.ink; ctx.fill(); }
  rr(ctx, x, y, w, h, r);
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.lineWidth = line * u;
  ctx.strokeStyle = T.ink;
  ctx.stroke();
}

/** Pill with centred text; returns its width. `cy` is the vertical centre. */
function pill(ctx, x, cy, text, u, { size = 12, fill = T.white, color = T.ink, weight = 700, pad = 8, shadow = 0, line = 2 } = {}) {
  ctx.font = font(weight, size * u);
  const w = ctx.measureText(text).width + 2 * pad * u;
  const h = size * u + 10 * u;
  sticker(ctx, x, cy - h / 2, w, h, h / 2, u, { fill, shadow, line });
  ctx.fillStyle = color;
  ctx.textBaseline = "middle";
  ctx.fillText(text, x + pad * u, cy + 0.5 * u);
  return w;
}

export class Hud {
  constructor(lang = "en") {
    this.lang = lang;
    this.s = STRINGS[lang];
    this.judgments = [];
    this.arrivals = [];
  }

  addJudgment(j) {
    this.judgments.push(j);
    this.arrivals.push(performance.now());
  }

  value(key) {
    const k = this.judgments.length - 1;
    const cur = this.judgments[k][key];
    if (k === 0) return cur;
    const prev = this.judgments[k - 1][key];
    return prev + (cur - prev) * smoothstep((performance.now() - this.arrivals[k]) / 600);
  }

  /**
   * state: {t, shot, hands: [{label, score, anchor:[x,y]}] (up to two), subtitle, judge,
   *         status: {kind: "live"|"judging"|"collecting", text}, skeleton: {hands, face} | null}
   * Coordinates are CSS pixels; the caller sets the device-pixel transform.
   */
  draw(ctx, W, H, state, uOverride) {
    const portrait = W < H;
    const u = uOverride ?? (portrait ? Math.min(1.3, W / 360) : Math.max(0.75, Math.min(1.6, H / 720)));
    ctx.textBaseline = "middle";
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    // Tall screens (phones held upright) get a compact layout that keeps the face clear.
    const compact = portrait;
    if (state.skeleton) this.drawSkeleton(ctx, u, state.skeleton);
    this.drawChips(ctx, W, u, state);
    if (state.hands && state.hands.length) this.drawGestures(ctx, W, H, u, state.hands, compact);

    const footTop = this.drawFooter(ctx, H, u, state.judge);
    const subTop = this.drawSubtitle(ctx, W, footTop - 8 * u, u, portrait, state.subtitle);
    if (!this.judgments.length) return;
    const panel = this.panelGeometry(W, H, u, portrait);
    if (portrait) {
      const top = this.drawReading(ctx, 12 * u, compact ? subTop - 20 * u : Math.min(H - 90 * u, subTop - 20 * u), W - 28 * u, u, compact);
      panel.y0 = top - 14 * u - panel.ph;
      this.drawCompactPanel(ctx, u, panel);
      return;
    } else {
      this.drawReading(ctx, 18 * u, H - 104 * u, Math.min(W * 0.56, panel.x0 - 48 * u), u);
    }
    this.drawPanel(ctx, u, panel);
  }

  drawChips(ctx, W, u, state) {
    const cy = 24 * u;
    let x = 14 * u;
    x += pill(ctx, x, cy, `${state.t.toFixed(1)}s`, u, { size: 12, weight: 600 }) + 6 * u;
    const win = Math.min(this.judgments.length + 1, state.total || Infinity);
    x += pill(ctx, x, cy, state.total ? `W${win}/${state.total}` : `W${win}`, u, { size: 12, fill: T.blue, color: T.white }) + 6 * u;
    pill(ctx, x, cy, state.shot, u, { size: 12, weight: 600 });

    const st = state.status;
    if (!st || !st.text) return;
    const live = st.kind === "live";
    const dot = live ? 13 * u : 0; // room for the blinking recording dot
    ctx.font = font(700, 12 * u);
    const w = ctx.measureText(st.text).width + 16 * u + dot;
    const h = 22 * u;
    const x0 = W - 14 * u - w;
    sticker(ctx, x0, cy - h / 2, w, h, h / 2, u, { fill: live ? T.white : T.yellow, shadow: 0, line: 2 });
    ctx.fillStyle = T.ink;
    ctx.fillText(st.text, x0 + 8 * u + dot, cy + 0.5 * u);
    if (live) {
      ctx.beginPath();
      ctx.arc(x0 + 13 * u, cy, 4.5 * u, 0, 2 * Math.PI);
      ctx.fillStyle = Math.floor(performance.now() / 500) % 2 ? T.coral : "#ffb3b3"; // stepped blink
      ctx.fill();
    }
  }

  /** One sticker per hand (up to two), each with a colour-matched pointer to its hand. */
  drawGestures(ctx, W, H, u, hands, compact = false) {
    const x = 14 * u;
    ctx.font = font(700, 10 * u);
    const scoreW = compact ? ctx.measureText(`${this.s.vision} 0.00`).width + 22 * u : 0; // score inside the sticker
    const items = hands.map((hand, i) => {
      let size = (compact ? 15 : 19) * u;
      ctx.font = font(700, size);
      while (ctx.measureText(hand.label).width > W - 2 * x - 40 * u - scoreW && size > 11) { size -= 1; ctx.font = font(700, size); }
      const w = ctx.measureText(hand.label).width + 38 * u + scoreW, h = size + (compact ? 12 : 18) * u;
      return { ...hand, size, w, h, colour: HAND_COLOURS[i] };
    });
    let y = (compact ? 42 : 46) * u;
    for (const it of items) { it.y = y; y += it.h + (compact ? 8 : 40) * u; }

    for (const it of items) { // pointers first, under the stickers
      const a = it.anchor;
      if (!a || a[0] < 0 || a[0] > W || a[1] < 0 || a[1] > H) continue;
      const sx = Math.min(Math.max(a[0], x), x + it.w), sy = Math.min(Math.max(a[1], it.y), it.y + it.h);
      for (const [colour, width] of [[T.white, 9], [T.ink, 4]]) {
        ctx.strokeStyle = colour;
        ctx.lineWidth = width * u;
        ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(a[0], a[1]); ctx.stroke();
      }
      ctx.beginPath(); ctx.arc(a[0], a[1], 8 * u, 0, 2 * Math.PI);
      ctx.fillStyle = it.colour; ctx.fill();
      ctx.lineWidth = 3 * u; ctx.strokeStyle = T.ink; ctx.stroke();
    }
    for (const it of items) {
      sticker(ctx, x, it.y, it.w, it.h, 14 * u, u, { shadow: 3 });
      ctx.beginPath(); ctx.arc(x + 16 * u, it.y + it.h / 2, 6 * u, 0, 2 * Math.PI);
      ctx.fillStyle = it.colour; ctx.fill();
      ctx.lineWidth = 2 * u; ctx.strokeStyle = T.ink; ctx.stroke();
      ctx.font = font(700, it.size);
      ctx.fillStyle = T.ink;
      ctx.fillText(it.label, x + 28 * u, it.y + it.h / 2 + 0.5 * u);
      const score = `${this.s.vision} ${it.score.toFixed(2)}`;
      if (compact) {
        pill(ctx, x + it.w - scoreW + 2 * u, it.y + it.h / 2, score, u, { size: 10, fill: T.blue, color: T.white, shadow: 0, line: 1.5, pad: 6 });
      } else {
        pill(ctx, x + 8 * u, it.y + it.h + 16 * u, score, u, { size: 11, fill: T.blue, color: T.white });
      }
    }
  }

  panelGeometry(W, H, u, portrait) {
    if (portrait) {
      const pw = W - 28 * u;
      return { x0: 14 * u, y0: 0, pw, ph: 100 * u };
    }
    const ph = 178 * u;
    const pw = 262 * u;
    return { x0: W - pw - 22 * u, y0: H - 104 * u - ph, pw, ph };
  }

  drawPanel(ctx, u, { x0, y0, pw, ph }) {
    const s = this.s;
    const j = this.judgments[this.judgments.length - 1];
    sticker(ctx, x0, y0, pw, ph, 18 * u, u);

    ctx.font = font(700, 18 * u);
    ctx.fillStyle = T.ink;
    ctx.fillText(s.verdict, x0 + 14 * u, y0 + 22 * u);
    const titleW = ctx.measureText(s.verdict).width;
    pill(ctx, x0 + 22 * u + titleW, y0 + 22 * u, tag(j.source, s.rules), u, { size: 11, fill: T.yellow });

    const bars = [["confidence", T.blue], ["focus", T.mint], ["tension", T.coral]];
    ctx.font = font(600, 13 * u);
    const labelW = Math.max(...bars.map(([k]) => ctx.measureText(s[k]).width));
    const bx0 = x0 + 24 * u + labelW, bx1 = Math.max(x0 + pw - 50 * u, bx0 + 4 * u);
    bars.forEach(([key, colour], i) => {
      const cy = y0 + 52 * u + i * 25 * u;
      ctx.font = font(600, 13 * u);
      ctx.fillStyle = T.muted;
      ctx.fillText(s[key], x0 + 14 * u, cy);
      const v = this.value(key);
      rr(ctx, bx0, cy - 6 * u, bx1 - bx0, 12 * u, 6 * u);
      ctx.fillStyle = T.paper; ctx.fill();
      if (v > 0.01) {
        ctx.save();
        rr(ctx, bx0, cy - 6 * u, bx1 - bx0, 12 * u, 6 * u);
        ctx.clip();
        ctx.fillStyle = colour;
        ctx.fillRect(bx0, cy - 6 * u, (bx1 - bx0) * v, 12 * u);
        ctx.restore();
      }
      rr(ctx, bx0, cy - 6 * u, bx1 - bx0, 12 * u, 6 * u);
      ctx.lineWidth = 2 * u; ctx.strokeStyle = T.ink; ctx.stroke();
      ctx.font = font(700, 13 * u);
      ctx.fillStyle = T.ink;
      ctx.fillText(v.toFixed(2), bx1 + 8 * u, cy);
    });

    let cy = y0 + 52 * u + 3 * 25 * u + 2 * u;
    ctx.font = font(600, 12 * u);
    ctx.fillStyle = T.muted;
    ctx.fillText(s.intent, x0 + 14 * u, cy);
    const cert = j.intent_certainty.toFixed(2);
    ctx.font = font(700, 11 * u);
    const certW = ctx.measureText(cert).width + 16 * u;
    pill(ctx, x0 + pw - 14 * u - certW, cy, cert, u, { size: 11, fill: T.blue, color: T.white });
    ctx.font = font(700, 15 * u);
    ctx.fillStyle = T.ink;
    const ix = x0 + 24 * u + labelW;
    ctx.fillText(wrap(ctx, j.intent, x0 + pw - 24 * u - certW - ix, 1)[0] || "", ix, cy);

    cy += 27 * u;
    ctx.font = font(600, 12 * u);
    ctx.fillStyle = T.muted;
    ctx.fillText(s.arc, x0 + 14 * u, cy);
    const r = 6 * u, gap = 5 * u;
    const fit = Math.max(1, Math.floor((x0 + pw - 14 * u - ix) / (2 * r + gap)));
    const first = Math.max(0, this.judgments.length - fit);
    for (let i = first; i < this.judgments.length; i++) {
      const current = i === this.judgments.length - 1;
      const cx = ix + r + (i - first) * (2 * r + gap);
      ctx.beginPath(); ctx.arc(cx, cy, current ? r * 1.25 : r, 0, 2 * Math.PI);
      ctx.fillStyle = valenceColour(this.judgments[i].valence);
      ctx.fill();
      ctx.lineWidth = (current ? 2.5 : 2) * u; ctx.strokeStyle = T.ink; ctx.stroke();
    }
  }

  /** The verdict for tall screens: title and mood, three bars side by side, then the intent. */
  drawCompactPanel(ctx, u, { x0, y0, pw, ph }) {
    const s = this.s;
    const j = this.judgments[this.judgments.length - 1];
    sticker(ctx, x0, y0, pw, ph, 16 * u, u, { shadow: 3 });

    let cy = y0 + 19 * u;
    ctx.font = font(700, 16 * u);
    ctx.fillStyle = T.ink;
    ctx.fillText(s.verdict, x0 + 12 * u, cy);
    pill(ctx, x0 + 20 * u + ctx.measureText(s.verdict).width, cy, tag(j.source, s.rules), u, { size: 10, fill: T.yellow, shadow: 0, line: 1.5 });
    const r = 5 * u, gap = 4 * u;
    const n = Math.min(this.judgments.length, 8);
    for (let i = 0; i < n; i++) {
      const k = this.judgments.length - n + i, current = k === this.judgments.length - 1;
      const cx = x0 + pw - 12 * u - r - (n - 1 - i) * (2 * r + gap);
      ctx.beginPath(); ctx.arc(cx, cy, current ? r * 1.2 : r, 0, 2 * Math.PI);
      ctx.fillStyle = valenceColour(this.judgments[k].valence);
      ctx.fill();
      ctx.lineWidth = 1.8 * u; ctx.strokeStyle = T.ink; ctx.stroke();
    }

    const bars = [["confidence", T.blue], ["focus", T.mint], ["tension", T.coral]];
    const colGap = 10 * u, colW = (pw - 24 * u - 2 * colGap) / 3;
    bars.forEach(([key, colour], i) => {
      const bx = x0 + 12 * u + i * (colW + colGap);
      const v = this.value(key);
      ctx.font = font(600, 11 * u);
      ctx.fillStyle = T.muted;
      ctx.fillText(wrap(ctx, s[key], colW - 30 * u, 1)[0] || "", bx, y0 + 42 * u);
      ctx.font = font(700, 11 * u);
      ctx.fillStyle = T.ink;
      const val = v.toFixed(2);
      ctx.fillText(val, bx + colW - ctx.measureText(val).width, y0 + 42 * u);
      const by = y0 + 51 * u, bh = 9 * u;
      rr(ctx, bx, by, colW, bh, bh / 2);
      ctx.fillStyle = T.paper; ctx.fill();
      if (v > 0.01) {
        ctx.save();
        rr(ctx, bx, by, colW, bh, bh / 2);
        ctx.clip();
        ctx.fillStyle = colour;
        ctx.fillRect(bx, by, colW * v, bh);
        ctx.restore();
      }
      rr(ctx, bx, by, colW, bh, bh / 2);
      ctx.lineWidth = 1.8 * u; ctx.strokeStyle = T.ink; ctx.stroke();
    });

    cy = y0 + ph - 19 * u;
    ctx.font = font(600, 12 * u);
    ctx.fillStyle = T.muted;
    ctx.fillText(s.intent, x0 + 12 * u, cy);
    const ix = x0 + 20 * u + ctx.measureText(s.intent).width;
    const cert = j.intent_certainty.toFixed(2);
    ctx.font = font(700, 10 * u);
    const certW = ctx.measureText(cert).width + 14 * u;
    pill(ctx, x0 + pw - 12 * u - certW, cy, cert, u, { size: 10, fill: T.blue, color: T.white, shadow: 0, line: 1.5 });
    ctx.font = font(700, 15 * u);
    ctx.fillStyle = T.ink;
    ctx.fillText(wrap(ctx, j.intent, x0 + pw - 22 * u - certW - ix, 1)[0] || "", ix, cy);
  }

  /** Speech bubble with the reading, bottom edge at `bottom`; returns its top. */
  drawReading(ctx, left, bottom, maxW, u, compact = false) {
    const k = this.judgments.length - 1;
    const j = this.judgments[k];
    if (!j.reading && !j.quote) return bottom;
    const tag = `W${k + 1}`;
    ctx.font = font(700, 12 * u);
    const tagW = ctx.measureText(tag).width + 16 * u;
    const size = compact ? 14 : 15, qSize = compact ? 12 : 13;
    ctx.font = font(500, size * u);
    const lines = wrap(ctx, j.reading, maxW - 36 * u - tagW, 2);
    const lineH = (compact ? 19 : 21) * u;
    ctx.font = font(600, qSize * u);
    // On tall screens the quote only shows when the reading fits on one line.
    const qLines = j.quote && !(compact && lines.length > 1) ? wrap(ctx, `“${j.quote}”`, maxW - 44 * u, 1) : [];
    const height = (compact ? 14 : 16) * u + lines.length * lineH + (qLines.length ? (compact ? 25 : 28) * u : 0);
    const top = bottom - height;
    ctx.font = font(500, size * u);
    let width = tagW + 8 * u + Math.max(...lines.map((l) => ctx.measureText(l).width));
    ctx.font = font(600, qSize * u);
    if (qLines.length) width = Math.max(width, ctx.measureText(qLines[0]).width + 16 * u);
    width += 28 * u;

    // Bubble tail, then the bubble over its base.
    rr(ctx, left + 4 * u, top + 4 * u, width, height, 16 * u);
    ctx.fillStyle = T.ink; ctx.fill();
    ctx.beginPath();
    ctx.moveTo(left + 22 * u, bottom - 2 * u); ctx.lineTo(left + 14 * u, bottom + 14 * u); ctx.lineTo(left + 42 * u, bottom - 2 * u);
    ctx.closePath();
    ctx.fillStyle = T.white; ctx.fill();
    ctx.lineWidth = 3 * u; ctx.strokeStyle = T.ink; ctx.stroke();
    sticker(ctx, left, top, width, height, 16 * u, u, { shadow: 0 });

    let y = top + 8 * u + lineH / 2;
    pill(ctx, left + 12 * u, y, tag, u, { size: 12, fill: T.blue, color: T.white });
    ctx.font = font(500, size * u);
    ctx.fillStyle = T.ink;
    lines.forEach((l, i) => { ctx.fillText(l, left + 20 * u + tagW, y + i * lineH); });
    if (qLines.length) {
      y += lines.length * lineH + 4 * u;
      pill(ctx, left + 12 * u, y, qLines[0], u, { size: qSize, weight: 600, fill: T.yellow });
    }
    return top;
  }

  /** Comic-caption subtitles ending at `bottom`; returns the top of the block. */
  drawSubtitle(ctx, W, bottom, u, portrait, text) {
    if (!text) return bottom;
    ctx.font = font(600, (portrait ? 18 : 20) * u);
    const lines = wrap(ctx, text, W * (portrait ? 0.9 : 0.62), 2);
    const lh = 26 * u;
    let y = bottom - lines.length * lh + lh / 2;
    const top = bottom - lines.length * lh;
    ctx.lineWidth = 6 * u;
    ctx.strokeStyle = T.ink;
    for (const l of lines) {
      const w = ctx.measureText(l).width;
      ctx.strokeText(l, (W - w) / 2, y);
      ctx.fillStyle = T.white;
      ctx.fillText(l, (W - w) / 2, y);
      y += lh;
    }
    return top;
  }

  drawFooter(ctx, H, u, label) {
    const name = label || this.s.rules;
    const cy = H - 18 * u;
    pill(ctx, 12 * u, cy, this.s.footer.replace("{judge}", name), u, { size: 10, weight: 600, fill: T.paper, line: 1.5, pad: 7 });
    return cy - 11 * u;
  }

  drawSkeleton(ctx, u, sk) {
    for (const pts of sk.hands) {
      for (const [colour, width] of [[T.white, 5], [T.ink, 2.2]]) {
        ctx.strokeStyle = colour;
        ctx.lineWidth = width * u;
        ctx.beginPath();
        for (const [a, b] of HAND_EDGES) { ctx.moveTo(...pts[a]); ctx.lineTo(...pts[b]); }
        ctx.stroke();
      }
      for (const [x, y] of pts) {
        ctx.beginPath(); ctx.arc(x, y, 3 * u, 0, 2 * Math.PI);
        ctx.fillStyle = T.yellow; ctx.fill();
        ctx.lineWidth = 1.5 * u; ctx.strokeStyle = T.ink; ctx.stroke();
      }
    }
    if (sk.face) {
      const [x0, y0, x1, y1] = sk.face;
      const c = 0.18 * Math.abs(x1 - x0);
      for (const [colour, width] of [[T.white, 6], [T.ink, 3]]) {
        ctx.strokeStyle = colour;
        ctx.lineWidth = width * u;
        ctx.beginPath();
        for (const [px, py, dx, dy] of [[x0, y0, 1, 1], [x1, y0, -1, 1], [x0, y1, 1, -1], [x1, y1, -1, -1]]) {
          ctx.moveTo(px + dx * c, py); ctx.lineTo(px, py); ctx.lineTo(px, py + dy * c);
        }
        ctx.stroke();
      }
    }
  }
}
