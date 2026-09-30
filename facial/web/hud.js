// Canvas HUD for live mode. Same layout and wording as facial/render.py, with an
// extra portrait layout for phones.

export const STRINGS = {
  en: {
    left: "Left hand", right: "Right hand", verdict: "Verdict", fusion: "subs + voice + gesture",
    confidence: "Confident", focus: "Focused", tension: "Tense", intent: "Intent", arc: "Emotion arc",
    quote: "QUOTE", vision: "VISION  gesture model", rules: "rules",
    footer: "Gesture: MediaPipe per-frame  ·  Reading & scores: {judge}  ·  Not calibrated, demo only",
    collecting: "collecting first window", judging: "judging", live: "LIVE",
  },
  zh: {
    left: "左手", right: "右手", verdict: "综合判定", fusion: "字幕+声音+动作",
    confidence: "自信", focus: "专注", tension: "紧张", intent: "意图", arc: "情绪弧",
    quote: "引语", vision: "VISION  手势模型", rules: "规则",
    footer: "动作 MediaPipe 逐帧检测  ·  解读和评分 {judge}  ·  未经人工校准 仅供演示",
    collecting: "正在收集第一个窗口", judging: "判定中", live: "直播",
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

const SANS = 'system-ui, -apple-system, "Segoe UI", Roboto, "PingFang SC", "Hiragino Sans GB", "Noto Sans CJK SC", "Microsoft YaHei", sans-serif';
const MONO = 'ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace';
const TEXT = "rgb(236,239,246)";
const MUTED = "rgb(165,175,196)";
const BLUE = "rgb(74,144,255)";
const ORANGE = "rgb(255,159,67)";
const HAND_EDGES = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20]];

const smoothstep = (x) => { x = Math.min(1, Math.max(0, x)); return x * x * (3 - 2 * x); };
const mix = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t));

export function valenceColour(v) {
  const neutral = [110, 130, 175];
  const c = v >= 0 ? mix(neutral, [70, 165, 255], v) : mix(neutral, [236, 92, 80], -v);
  return `rgb(${c.join(",")})`;
}

function font(weight, size, family = SANS) { return `${weight} ${size}px ${family}`; }

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

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, r);
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
   * state: {t, shot, live: {label, score, anchor:[x,y]} | null, subtitle, judge, status,
   *         skeleton: {hands: [[[x,y]...]], face: [x0,y0,x1,y1]} | null}
   * Coordinates are CSS pixels; the caller sets the device-pixel transform.
   */
  draw(ctx, W, H, state) {
    const portrait = W < H;
    const u = portrait ? Math.min(1.3, W / 360) : Math.max(0.75, Math.min(1.6, H / 720));
    ctx.textBaseline = "top";
    ctx.lineJoin = "round";
    if (state.skeleton) this.drawSkeleton(ctx, u, state.skeleton);
    this.drawStrip(ctx, W, u, state);
    if (state.live) this.drawGesture(ctx, W, H, u, portrait, state.live);

    // Bottom stack: footer, subtitles, reading, then the panel above/beside it.
    const footTop = this.drawFooter(ctx, W, H, u, state.judge);
    const subTop = this.drawSubtitle(ctx, W, Math.min(H, footTop + 20 * u), u, portrait, state.subtitle);
    if (!this.judgments.length) return;
    const panel = this.panelGeometry(W, H, u, portrait);
    let readingTop;
    if (portrait) {
      readingTop = this.drawReading(ctx, 12 * u, Math.min(H - 86 * u, subTop - 8 * u), W - 24 * u, u);
      panel.y0 = readingTop - 10 * u - panel.ph;
    } else {
      this.drawReading(ctx, 20 * u, H - 100 * u, Math.min(W * 0.56, panel.x0 - 44 * u), u);
    }
    this.drawPanel(ctx, u, panel);
  }

  drawStrip(ctx, W, u, state) {
    ctx.fillStyle = "rgba(0,0,0,0.35)";
    ctx.fillRect(0, 0, W, 24 * u);
    const win = this.judgments.length + 1;
    const hud = `t=${state.t.toFixed(1).padStart(5, "0")}   WIN ${win}   shot=${state.shot}`;
    ctx.font = font(400, 12 * u, MONO);
    ctx.fillStyle = "rgb(220,225,235)";
    ctx.fillText(hud, 12 * u, 6 * u);
    if (state.status) {
      const w = ctx.measureText(state.status).width;
      ctx.fillStyle = state.status.startsWith("●") ? "rgb(255,120,110)" : "rgb(190,210,255)";
      ctx.fillText(state.status, W - 12 * u - w, 6 * u);
    }
  }

  drawGesture(ctx, W, H, u, portrait, live) {
    const x = (portrait ? 14 : 22) * u;
    const y = 40 * u;
    let size = (portrait ? 20 : 22) * u;
    ctx.font = font(700, size);
    while (ctx.measureText(live.label).width > W - 2 * x && size > 12) {
      size -= 1;
      ctx.font = font(700, size);
    }
    const tw = ctx.measureText(live.label).width;
    ctx.lineWidth = 3 * u;
    ctx.strokeStyle = "rgba(0,0,0,0.65)";
    ctx.strokeText(live.label, x, y);
    ctx.fillStyle = TEXT;
    ctx.fillText(live.label, x, y);

    const y2 = y + size + 10 * u;
    ctx.strokeStyle = BLUE;
    ctx.lineWidth = Math.max(1, 2 * u);
    ctx.beginPath(); ctx.moveTo(x, y2 + 7 * u); ctx.lineTo(x + 16 * u, y2 + 7 * u); ctx.stroke();
    const vis = `${this.s.vision}  ${live.score.toFixed(2)}`;
    ctx.font = font(400, 11 * u);
    ctx.fillStyle = "rgb(190,210,255)";
    ctx.fillText(vis, x + 22 * u, y2 + 1 * u);

    // Skip the callout when the hand sits in the part of the frame cropped away on screen.
    if (live.anchor && live.anchor[0] >= 0 && live.anchor[0] <= W && live.anchor[1] >= 0 && live.anchor[1] <= H) {
      const [ax, ay] = live.anchor;
      ctx.strokeStyle = "rgba(255,255,255,0.72)";
      ctx.lineWidth = Math.max(1, 1.2 * u);
      ctx.beginPath(); ctx.moveTo(x + tw + 10 * u, y + size * 0.6); ctx.lineTo(ax, ay); ctx.stroke();
      ctx.strokeStyle = "rgba(255,255,255,0.9)";
      ctx.lineWidth = Math.max(1, 2 * u);
      ctx.beginPath(); ctx.arc(ax, ay, 6 * u, 0, 2 * Math.PI); ctx.stroke();
      ctx.fillStyle = "rgba(255,255,255,0.9)";
      ctx.beginPath(); ctx.arc(ax, ay, 2 * u, 0, 2 * Math.PI); ctx.fill();
    }
  }

  panelGeometry(W, H, u, portrait) {
    const ph = 164 * u;
    if (portrait) {
      const pw = Math.min(W - 24 * u, 300 * u);
      return { x0: W - pw - 12 * u, y0: 0, pw, ph };
    }
    const pw = 252 * u;
    return { x0: W - pw - 20 * u, y0: H - 100 * u - ph, pw, ph };
  }

  drawPanel(ctx, u, { x0, y0, pw, ph }) {
    const s = this.s;
    const j = this.judgments[this.judgments.length - 1];
    roundRect(ctx, x0, y0, pw, ph, 8 * u);
    ctx.fillStyle = "rgba(12,18,34,0.75)";
    ctx.fill();
    ctx.strokeStyle = "rgba(84,132,255,0.86)";
    ctx.lineWidth = Math.max(1, 1.5 * u);
    ctx.stroke();

    ctx.font = font(700, 15 * u);
    ctx.fillStyle = TEXT;
    ctx.fillText(s.verdict, x0 + 12 * u, y0 + 10 * u);
    const titleW = ctx.measureText(s.verdict).width;
    ctx.font = font(400, 10 * u);
    ctx.fillStyle = "rgb(140,175,255)";
    ctx.fillText(`${j.source === "claude" ? "Claude" : s.rules} · ${s.fusion}`, x0 + 18 * u + titleW, y0 + 14 * u);

    const bars = [["confidence", BLUE], ["focus", BLUE], ["tension", ORANGE]];
    ctx.font = font(400, 12 * u);
    const labelW = Math.max(...bars.map(([k]) => ctx.measureText(s[k]).width));
    const bx0 = x0 + 22 * u + labelW, bx1 = Math.max(x0 + pw - 52 * u, bx0 + 4 * u);
    bars.forEach(([key, colour], i) => {
      const yy = y0 + 40 * u + i * 21 * u;
      ctx.font = font(400, 12 * u);
      ctx.fillStyle = MUTED;
      ctx.fillText(s[key], x0 + 12 * u, yy);
      const v = this.value(key);
      const cy = yy + 8 * u;
      roundRect(ctx, bx0, cy - 2.5 * u, bx1 - bx0, 5 * u, 2.5 * u);
      ctx.fillStyle = "rgba(255,255,255,0.16)";
      ctx.fill();
      if (v > 0.005) {
        roundRect(ctx, bx0, cy - 2.5 * u, (bx1 - bx0) * v, 5 * u, 2.5 * u);
        ctx.fillStyle = colour;
        ctx.fill();
      }
      ctx.font = font(400, 11 * u, MONO);
      ctx.fillStyle = colour;
      ctx.fillText(v.toFixed(2), bx1 + 8 * u, yy + 1 * u);
    });

    let yy = y0 + 40 * u + 3 * 21 * u + 4 * u;
    const cert = j.intent_certainty.toFixed(2);
    ctx.font = font(400, 11 * u, MONO);
    const certW = ctx.measureText(cert).width;
    ctx.fillStyle = BLUE;
    ctx.fillText(cert, x0 + pw - 12 * u - certW, yy + 2 * u);
    ctx.font = font(400, 13 * u);
    ctx.fillStyle = TEXT;
    ctx.fillText(wrap(ctx, `${s.intent} → ${j.intent}`, pw - 32 * u - certW, 1)[0] || "", x0 + 12 * u, yy);

    yy += 26 * u;
    ctx.font = font(400, 10 * u);
    ctx.fillStyle = MUTED;
    ctx.fillText(s.arc, x0 + 12 * u, yy);
    const size = 10 * u, gap = 4 * u;
    const ax0 = Math.max(bx0, x0 + 22 * u + ctx.measureText(s.arc).width);
    const fit = Math.max(1, Math.floor((x0 + pw - 12 * u - ax0) / (size + gap)));
    const first = Math.max(0, this.judgments.length - fit);
    for (let i = first; i < this.judgments.length; i++) {
      const qx = ax0 + (i - first) * (size + gap);
      roundRect(ctx, qx, yy + 1 * u, size, size, 2 * u);
      ctx.fillStyle = valenceColour(this.judgments[i].valence);
      ctx.fill();
      if (i === this.judgments.length - 1) {
        ctx.strokeStyle = "white";
        ctx.lineWidth = Math.max(1, 1.2 * u);
        ctx.stroke();
      }
    }
  }

  /** Draws the reading block with its bottom edge at `bottom`; returns its top. */
  drawReading(ctx, left, bottom, maxW, u) {
    const k = this.judgments.length - 1;
    const j = this.judgments[k];
    if (!j.reading && !j.quote) return bottom;
    ctx.font = font(400, 14 * u);
    const lines = wrap(ctx, `W${k + 1} > ${j.reading}`, maxW - 20 * u, 2);
    const lineH = 20 * u;
    ctx.font = font(700, 10 * u);
    const chipW = ctx.measureText(this.s.quote).width + 10 * u;
    ctx.font = font(400, 13 * u);
    const qLines = j.quote ? wrap(ctx, `“${j.quote}”`, maxW - chipW - 28 * u, 1) : [];
    const height = 10 * u + lines.length * lineH + (qLines.length ? 22 * u : 0) + 6 * u;
    const top = bottom - height;

    ctx.font = font(400, 14 * u);
    let width = Math.max(...lines.map((l) => ctx.measureText(l).width));
    ctx.font = font(400, 13 * u);
    if (qLines.length) width = Math.max(width, chipW + 8 * u + ctx.measureText(qLines[0]).width);
    roundRect(ctx, left, top, width + 20 * u, height, 6 * u);
    ctx.fillStyle = "rgba(0,0,0,0.5)";
    ctx.fill();

    let y = top + 8 * u;
    ctx.font = font(400, 14 * u);
    ctx.fillStyle = TEXT;
    for (const l of lines) { ctx.fillText(l, left + 10 * u, y); y += lineH; }
    if (qLines.length) {
      y += 2 * u;
      roundRect(ctx, left + 10 * u, y, chipW, 16 * u, 3 * u);
      ctx.fillStyle = ORANGE;
      ctx.fill();
      ctx.font = font(700, 10 * u);
      ctx.fillStyle = "rgb(40,24,8)";
      ctx.fillText(this.s.quote, left + 15 * u, y + 3 * u);
      ctx.font = font(400, 13 * u);
      ctx.fillStyle = TEXT;
      ctx.fillText(qLines[0], left + 18 * u + chipW, y + 1 * u);
    }
    return top;
  }

  /** Returns the top of the subtitle block (or where it would be). */
  drawSubtitle(ctx, W, H, u, portrait, text) {
    const bottom = H - 36 * u;
    if (!text) return bottom;
    const size = (portrait ? 17 : 18) * u;
    ctx.font = font(500, size);
    const lines = wrap(ctx, text, W * (portrait ? 0.9 : 0.62), 2);
    let y = bottom - lines.length * 24 * u;
    const top = y;
    ctx.lineWidth = 3 * u;
    ctx.strokeStyle = "rgba(0,0,0,0.85)";
    for (const l of lines) {
      const w = ctx.measureText(l).width;
      ctx.strokeText(l, (W - w) / 2, y);
      ctx.fillStyle = "white";
      ctx.fillText(l, (W - w) / 2, y);
      y += 24 * u;
    }
    return top;
  }

  /** Draws the footer (wrapped to the width) at the bottom; returns its top. */
  drawFooter(ctx, W, H, u, judge) {
    const name = judge === "claude" ? "Claude" : this.s.rules;
    ctx.font = font(400, 9 * u);
    ctx.fillStyle = "rgba(185,192,208,0.9)";
    const lines = wrap(ctx, this.s.footer.replace("{judge}", name), W - 24 * u, 2);
    let y = H - 6 * u - lines.length * 12 * u;
    const top = y;
    for (const l of lines) { ctx.fillText(l, 12 * u, y); y += 12 * u; }
    return top;
  }

  drawSkeleton(ctx, u, sk) {
    ctx.strokeStyle = "rgba(120,200,255,0.6)";
    ctx.lineWidth = Math.max(1, 1.5 * u);
    for (const pts of sk.hands) {
      ctx.beginPath();
      for (const [a, b] of HAND_EDGES) { ctx.moveTo(...pts[a]); ctx.lineTo(...pts[b]); }
      ctx.stroke();
      ctx.fillStyle = "rgba(255,255,255,0.8)";
      for (const [x, y] of pts) { ctx.beginPath(); ctx.arc(x, y, 2 * u, 0, 2 * Math.PI); ctx.fill(); }
    }
    if (sk.face) {
      const [x0, y0, x1, y1] = sk.face;
      const c = 0.18 * Math.abs(x1 - x0);
      ctx.strokeStyle = "rgba(255,255,255,0.67)";
      ctx.beginPath();
      for (const [px, py, dx, dy] of [[x0, y0, 1, 1], [x1, y0, -1, 1], [x0, y1, 1, -1], [x1, y1, -1, -1]]) {
        ctx.moveTo(px + dx * c, py); ctx.lineTo(px, py); ctx.lineTo(px, py + dy * c);
      }
      ctx.stroke();
    }
  }
}
