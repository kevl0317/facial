// Opening animation, drawn frame by frame at 12 fps like a pencil flipbook:
// the glove sketches itself in (with boiling lines), gets coloured, waves while
// the name pops in letter by letter, then an iris wipe opens onto the app.

import { CENTER, COLORS, INK, PAPER, PIVOT, SHAPES, outline } from "./hand.js";

const FPS = 12;
const DRAW = [0, 12]; // frames: outline sketched in
const FILL = 12; // colour pops in
const WAVE = [14, 26];
const WAVE_ANGLES = [-12, -20, -12, 0, 12, 20, 12, 0, -12, -20, -12, 0];
const WORD_START = 15;
const IRIS = [28, 35];
const WORD = "facial";
const WORD_COLOURS = [COLORS.blue, COLORS.coral, COLORS.yellow, COLORS.mint, COLORS.sky, COLORS.orange];

// Deterministic noise so each "drawing" in the boil cycle is stable.
function noise(a, b, c) {
  let h = (a * 374761393 + b * 668265263 + c * 2147483647) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return (((h ^ (h >>> 16)) >>> 0) / 4294967295) * 2 - 1;
}

const OUTLINES = SHAPES.map(outline);
const LENGTHS = OUTLINES.map((pts) => pts.slice(1).reduce((s, p, i) => s + Math.hypot(p[0] - pts[i][0], p[1] - pts[i][1]), 0));
const TOTAL = LENGTHS.reduce((a, b) => a + b, 0);

function jittered(pts, shapeIdx, boil, amp) {
  return pts.map(([x, y], i) => [x + noise(shapeIdx, i, boil) * amp, y + noise(shapeIdx, i + 999, boil) * amp]);
}

function tracePath(ctx, pts, upTo, closed) {
  ctx.beginPath();
  let left = upTo;
  ctx.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length && left > 0; i++) {
    const [x0, y0] = pts[i - 1], [x1, y1] = pts[i];
    const seg = Math.hypot(x1 - x0, y1 - y0);
    if (seg <= left) ctx.lineTo(x1, y1);
    else ctx.lineTo(x0 + ((x1 - x0) * left) / seg, y0 + ((y1 - y0) * left) / seg);
    left -= seg;
  }
  if (closed && left > 0) ctx.closePath();
}

function drawHand(ctx, frame, boil) {
  const drawn = frame >= DRAW[1] ? TOTAL : (TOTAL * (frame - DRAW[0] + 1)) / (DRAW[1] - DRAW[0]);
  const coloured = frame >= FILL;
  let budget = drawn;
  SHAPES.forEach((s, idx) => {
    if (budget <= 0) return;
    const len = Math.min(budget, LENGTHS[idx]);
    budget -= LENGTHS[idx];
    const pts = jittered(OUTLINES[idx], idx, boil, 1.4);
    const closed = s.kind !== "line" && len >= LENGTHS[idx];
    if (coloured && s.fill) {
      tracePath(ctx, pts, LENGTHS[idx], true);
      ctx.fillStyle = s.fill;
      ctx.fill();
    }
    // Faint second pencil pass, then the ink line.
    ctx.strokeStyle = "rgba(30,27,46,0.28)";
    ctx.lineWidth = 2.2;
    tracePath(ctx, jittered(OUTLINES[idx], idx, boil + 7, 2.2), len, closed);
    ctx.stroke();
    ctx.strokeStyle = INK;
    ctx.lineWidth = 5;
    tracePath(ctx, pts, len, closed);
    ctx.stroke();
  });
}

function drawSparkles(ctx, boil) {
  ctx.strokeStyle = INK;
  ctx.lineWidth = 4.5;
  const [cx, cy] = CENTER;
  for (let i = 0; i < 6; i++) {
    const a = (-Math.PI * 5) / 6 + (i * Math.PI) / 5 + noise(i, 3, boil) * 0.08;
    const r0 = 118 + noise(i, 4, boil) * 4;
    ctx.beginPath();
    ctx.moveTo(cx + Math.cos(a) * r0, cy + Math.sin(a) * r0);
    ctx.lineTo(cx + Math.cos(a) * (r0 + 16), cy + Math.sin(a) * (r0 + 16));
    ctx.stroke();
  }
}

function drawSwoosh(ctx, dir, boil) {
  // Motion arcs on the trailing side of the wave.
  ctx.strokeStyle = INK;
  ctx.lineWidth = 4.5;
  for (let i = 0; i < 2; i++) {
    const r = 128 + i * 18;
    const mid = -Math.PI / 2 - dir * 0.62;
    ctx.beginPath();
    ctx.arc(PIVOT[0], PIVOT[1], r + noise(i, 8, boil) * 2, mid - 0.14, mid + 0.14);
    ctx.stroke();
  }
}

function drawWord(ctx, frame, cx, y, k) {
  const shown = frame - WORD_START + 1;
  if (shown <= 0) return;
  ctx.font = `700 ${Math.round(58 * k)}px Fredoka, "Baloo 2", ui-rounded, system-ui, sans-serif`;
  ctx.textBaseline = "middle";
  ctx.lineJoin = "round";
  const widths = [...WORD].map((ch) => ctx.measureText(ch).width + 2 * k);
  let x = cx - widths.reduce((a, b) => a + b, 0) / 2;
  [...WORD].forEach((ch, i) => {
    if (i < shown) {
      const pop = i === shown - 1 && shown <= WORD.length ? 1.3 : 1; // squash on the frame it lands
      const boil = frame % 3;
      ctx.save();
      ctx.translate(x + widths[i] / 2 + noise(i, 1, boil) * 1.2 * k, y + noise(i, 2, boil) * 1.2 * k);
      ctx.rotate(noise(i, 5, 0) * 0.08);
      ctx.scale(pop, 1 / pop);
      ctx.lineWidth = 7 * k;
      ctx.strokeStyle = INK;
      ctx.strokeText(ch, -widths[i] / 2 + k, 0);
      ctx.fillStyle = WORD_COLOURS[i % WORD_COLOURS.length];
      ctx.fillText(ch, -widths[i] / 2 + k, 0);
      ctx.restore();
    }
    x += widths[i];
  });
}

function paper(ctx, W, H) {
  ctx.fillStyle = PAPER;
  ctx.fillRect(0, 0, W, H);
  ctx.fillStyle = "rgba(30,27,46,0.07)";
  for (let y = 10; y < H; y += 22) for (let x = (y / 22) % 2 ? 21 : 10; x < W; x += 22) ctx.fillRect(x, y, 2.2, 2.2);
}

/** Plays the opening on `canvas`; resolves when done or skipped (tap / key). */
export async function playIntro(canvas) {
  const reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reduce) { canvas.remove(); return; }
  try {
    await Promise.race([document.fonts.load("700 58px Fredoka"), new Promise((r) => setTimeout(r, 900))]);
  } catch { /* fall back to system fonts */ }

  const ctx = canvas.getContext("2d");
  let skip = false;
  const onSkip = () => { skip = true; };
  canvas.addEventListener("pointerdown", onSkip);
  window.addEventListener("keydown", onSkip);

  for (let frame = 0; frame < IRIS[1] && !skip; frame++) {
    const dpr = window.devicePixelRatio || 1;
    const W = canvas.clientWidth, H = canvas.clientHeight;
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    paper(ctx, W, H);

    const k = Math.min(W, H) / 430; // scene scale: the 200 x 240 hand box -> screen
    const hx = W / 2, hy = H / 2 - 70 * k;
    const boil = frame % 3;
    const wi = frame - WAVE[0];
    const angle = wi >= 0 && wi < WAVE_ANGLES.length ? WAVE_ANGLES[wi] : 0;

    ctx.save();
    ctx.translate(hx - CENTER[0] * k, hy - CENTER[1] * k);
    ctx.scale(k, k);
    if (frame === FILL || frame === FILL + 1) drawSparkles(ctx, boil);
    if (Math.abs(angle) >= 12) drawSwoosh(ctx, Math.sign(angle), boil);
    ctx.translate(PIVOT[0], PIVOT[1]);
    ctx.rotate((angle * Math.PI) / 180);
    ctx.translate(-PIVOT[0], -PIVOT[1]);
    drawHand(ctx, frame, boil);
    ctx.restore();
    drawWord(ctx, frame, W / 2, hy + 150 * k, k);

    if (frame >= IRIS[0]) {
      // Iris wipe: a growing hole with an inked rim reveals the app underneath.
      const step = (frame - IRIS[0] + 1) / (IRIS[1] - IRIS[0]);
      const r = Math.hypot(W, H) * step ** 1.6;
      ctx.globalCompositeOperation = "destination-out";
      ctx.fillStyle = "#000"; // erase fully: destination-out uses the fill's alpha
      ctx.beginPath(); ctx.arc(hx, hy + 40 * k, r, 0, 2 * Math.PI); ctx.fill();
      ctx.globalCompositeOperation = "source-over";
      ctx.strokeStyle = INK;
      ctx.lineWidth = 10;
      ctx.beginPath(); ctx.arc(hx, hy + 40 * k, r, 0, 2 * Math.PI); ctx.stroke();
    }
    await new Promise((r) => setTimeout(r, 1000 / FPS));
  }
  canvas.removeEventListener("pointerdown", onSkip);
  window.removeEventListener("keydown", onSkip);
  canvas.remove();
}
