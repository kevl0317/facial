// Face expression per frame from MediaPipe blendshapes (FACS-style action-unit combinations).
// A line-for-line port of facial/expressions.py; tests/test_web.py checks they agree.

import { EXPRESSIONS } from "./prompt.js";

const KEYS = Object.keys(EXPRESSIONS);
// Blendshapes where a resting face often carries a bias: used relative to the person's median.
const BIAS_KEYS = ["brow_furrow", "lip_press", "squint", "frown", "brow_inner_up"];
const clamp = (x) => Math.max(0, Math.min(1, x));

/** Probabilities (summing to 1) of each expression for one frame's face metrics. */
export function expressionScores(face, base = {}) {
  const g = (k) => {
    let v = Number(face[k] || 0);
    if (BIAS_KEYS.includes(k) && base && base[k] != null) v = Math.max(0, v - Number(base[k]));
    return v;
  };
  const smile = g("smile");
  const raw = {
    happy: 0.75 * smile + 0.45 * g("cheek_squint"),
    sad: 0.8 * g("frown") + 0.6 * g("brow_inner_up") + 0.2 * g("brow_furrow") - 0.6 * smile,
    surprised: 0.5 * g("eye_wide") + 0.45 * g("brow_outer_up") + 0.25 * g("brow_inner_up")
      + 0.4 * g("jaw") - 0.4 * g("brow_furrow") - 0.3 * smile,
    angry: 0.9 * g("brow_furrow") + 0.4 * g("lip_press") + 0.25 * g("squint") + 0.2 * g("nose_sneer")
      - 0.5 * smile - 0.3 * g("brow_inner_up"),
    disgusted: 1.0 * g("nose_sneer") + 0.4 * g("upper_lip_up") + 0.2 * g("frown") - 0.3 * smile,
    fearful: 0.35 * g("eye_wide") + 0.45 * g("brow_inner_up") + 0.7 * g("mouth_stretch")
      + 0.15 * g("brow_furrow") - 0.5 * smile,
    contempt: 1.5 * g("smile_asym") - 0.3 * g("frown") - 0.4 * Math.max(0, smile - 0.5),
  };
  const scores = {};
  for (const [k, v] of Object.entries(raw)) scores[k] = clamp((v - 0.1) / 0.4); // < 0.1 noise, >= 0.5 clear
  scores.neutral = clamp(1 - 1.6 * Math.max(...Object.values(scores)));
  const total = Object.values(scores).reduce((a, b) => a + b, 0) || 1;
  return Object.fromEntries(KEYS.map((k) => [k, scores[k] / total]));
}

const round2 = (x) => Math.round(x * 100) / 100;

/** A window's expression: mean probabilities over its frames, the top one and its share. */
export function summarizeExpressions(faces, base = {}) {
  if (!faces.length) return null;
  const totals = Object.fromEntries(KEYS.map((k) => [k, 0]));
  for (const f of faces) for (const [k, p] of Object.entries(expressionScores(f, base))) totals[k] += p;
  const mean = Object.fromEntries(KEYS.map((k) => [k, totals[k] / faces.length]));
  let top = KEYS[0];
  for (const k of KEYS) if (mean[k] > mean[top]) top = k;
  const ranked = [...KEYS].sort((a, b) => mean[b] - mean[a]).slice(0, 3);
  return { top, share: round2(mean[top]), scores: Object.fromEntries(ranked.map((k) => [k, round2(mean[k])])) };
}
