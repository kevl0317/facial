// The five fields per window (scene, speaker, subtitle, voice, gesture), computed in
// the browser. A port of facial/features.py; keep the two in step.

import { summarizeExpressions } from "./expressions.js";

const FILLERS = ["um", "uh", "erm", "hmm", "like", "basically", "actually", "嗯", "啊", "呃", "那个", "就是"];
const RELATIVE_FACE = ["smile", "frown", "brow_furrow", "brow_raise", "lip_press"];
const BIAS_KEYS = ["brow_furrow", "lip_press", "squint", "frown", "brow_inner_up"]; // expressions.py
const BEAT_SPEED = 1.5;

const r = (v, d) => Math.round(v * 10 ** d) / 10 ** d;
const mean = (a) => (a.length ? a.reduce((s, v) => s + v, 0) / a.length : 0);
const std = (a) => { const m = mean(a); return Math.sqrt(mean(a.map((v) => (v - m) ** 2))); };
export const median = (a) => {
  if (!a.length) return NaN;
  const s = [...a].sort((x, y) => x - y);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};
function corr(a, b) {
  const ma = mean(a), mb = mean(b);
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < a.length; i++) { num += (a[i] - ma) * (b[i] - mb); da += (a[i] - ma) ** 2; db += (b[i] - mb) ** 2; }
  return num / Math.sqrt(da * db || 1e-12);
}

export function shotType(size) {
  if (!size) return "cutaway";
  if (size >= 0.33) return "close-up";
  if (size >= 0.12) return "medium";
  return "wide";
}

const handKey = (h) => `${h.side} ${h.shape} ${h.facing} ${h.axis}`;
const primaryHand = (s) => (s.hands.length
  ? s.hands.reduce((best, h) => (h.score * (1 + Math.min(h.speed || 0, 4)) > best.score * (1 + Math.min(best.speed || 0, 4)) ? h : best))
  : null);

export function clipBaseline(samples) {
  const faces = samples.filter((s) => s.face).map((s) => s.face);
  if (!faces.length) return {};
  const med = (k) => { const v = faces.filter((f) => k in f).map((f) => f[k]); return v.length ? r(median(v), 3) : null; };
  const out = { yaw: med("yaw"), pitch: med("pitch"), face_size: med("size") };
  for (const k of RELATIVE_FACE) out[k] = med(k);
  for (const k of BIAS_KEYS) if (!RELATIVE_FACE.includes(k)) out[k] = med(k);
  return out;
}

function handToFace(s) {
  if (!s.face) return false;
  const [x0, y0, x1, y1] = s.face.bbox;
  const px = 0.25 * (x1 - x0), py = 0.25 * (y1 - y0);
  return s.hands.some((h) => x0 - px <= h.anchor[0] && h.anchor[0] <= x1 + px && y0 - py <= h.anchor[1] && h.anchor[1] <= y1 + py);
}

function gesture(S, start) {
  const n = S.length;
  const withHands = S.filter((s) => s.hands.length);
  const out = { hands_visible: r(withHands.length / n, 2) };

  const weights = new Map(), best = new Map();
  for (const s of withHands) {
    for (const h of s.hands) {
      const k = handKey(h);
      weights.set(k, (weights.get(k) || 0) + h.score * (1 + Math.min(h.speed || 0, 4)));
      if (!best.has(k) || h.score > best.get(k)[0]) best.set(k, [h.score, s.t, h]);
    }
  }
  if (weights.size) {
    const k = [...weights.entries()].sort((a, b) => b[1] - a[1])[0][0];
    const [score, t, h] = best.get(k);
    const share = S.filter((s) => s.hands.some((x) => handKey(x) === k)).length / n;
    out.dominant = { hand: h.side, shape: h.shape, facing: h.facing, axis: h.axis, score: r(score, 2), share: r(share, 2), at_s: r(t - start, 1) };

    // The other hand (a different track), when it is up for at least a fifth of the window.
    const main = h.track;
    const others = new Map(), example = new Map();
    for (const s of withHands) {
      for (const x of s.hands) {
        if (x.track === main) continue;
        const k2 = handKey(x);
        others.set(k2, (others.get(k2) || 0) + x.score * (1 + Math.min(x.speed || 0, 4)));
        if (!example.has(k2)) example.set(k2, x);
      }
    }
    if (others.size) {
      const k2 = [...others.entries()].sort((a, b) => b[1] - a[1])[0][0];
      const share2 = S.filter((s) => s.hands.some((x) => handKey(x) === k2 && x.track !== main)).length / n;
      if (share2 >= 0.2) {
        const x = example.get(k2);
        out.second = { hand: x.side, shape: x.shape, facing: x.facing, axis: x.axis, share: r(share2, 2) };
      }
    }
  }
  out.two_hands = r(S.filter((s) => s.hands.length >= 2).length / n, 2);

  const runs = [];
  for (const s of S) {
    const h = primaryHand(s);
    const label = h ? handKey(h) : "no hands";
    if (runs.length && runs[runs.length - 1][0] === label) runs[runs.length - 1][2] = s.t;
    else runs.push([label, s.t, s.t]);
  }
  out.sequence = runs.filter(([, a, b]) => b - a >= 0.4).slice(0, 5)
    .map(([state, a, b]) => ({ from_s: r(a - start, 1), to_s: r(b - start, 1), state }));

  const speeds = withHands.flatMap((s) => s.hands.map((h) => h.speed || 0));
  out.energy = speeds.length ? r(mean(speeds), 2) : 0;
  const tracks = new Set(S.flatMap((s) => s.hands.map((h) => h.track)));
  const dt = n > 1 ? median(S.slice(1).map((s, i) => s.t - S[i].t)) : 0.1;
  const minGap = Math.max(1, Math.ceil(0.35 / Math.max(dt, 1e-3)));
  let beats = 0;
  for (const track of tracks) {
    const series = S.map((s) => { const h = s.hands.find((x) => x.track === track); return h ? h.speed || 0 : 0; });
    let last = -minGap;
    for (let i = 1; i < series.length - 1; i++) {
      if (series[i] > BEAT_SPEED && series[i] >= series[i - 1] && series[i] > series[i + 1] && i - last >= minGap) { beats++; last = i; }
    }
  }
  out.beats = beats;
  out.hand_to_face = r(S.filter(handToFace).length / n, 2);
  return out;
}

function faceAndHead(S, base) {
  const faces = S.filter((s) => s.face);
  if (faces.length < 3) return [null, null];
  const arr = (k) => faces.map((s) => s.face[k] ?? 0);
  const face = {};
  for (const k of ["smile", "frown", "brow_furrow", "brow_raise", "lip_press", "eye_wide", "squint"]) face[k] = r(mean(arr(k)), 2);
  const deltas = {};
  for (const k of RELATIVE_FACE) if (base[k] !== null && base[k] !== undefined) deltas[k] = r(face[k] - base[k], 2);
  if (Object.keys(deltas).length) face.vs_baseline = deltas;
  face.expression = summarizeExpressions(faces.map((s) => s.face), base);

  let blinks = 0, closed = false;
  for (const v of arr("blink")) {
    if (!closed && v > 0.5) { blinks++; closed = true; } else if (closed && v < 0.35) closed = false;
  }
  const span = Math.max(faces[faces.length - 1].t - faces[0].t, 0.5);
  face.blink_per_min = Math.round((blinks / span) * 60);

  const yaw = arr("yaw"), pitch = arr("pitch"), gazeSide = arr("gaze_side");
  const baseYaw = base.yaw || 0;
  const away = yaw.filter((y, i) => Math.abs(y - baseYaw) > 15 || gazeSide[i] > 0.5).length / yaw.length;
  const pm = mean(pitch);
  const signs = pitch.map((p) => p - pm).filter((c) => Math.abs(c) > 2.5).map(Math.sign);
  let crossings = 0;
  for (let i = 1; i < signs.length; i++) if (signs[i] !== signs[i - 1]) crossings++;
  const head = { yaw_std: r(std(yaw), 1), pitch_std: r(std(pitch), 1), nods: crossings >> 1, gaze_away: r(away, 2), yaw_vs_baseline: r(mean(yaw) - baseYaw, 1) };
  return [face, head];
}

function posture(S) {
  const poses = S.filter((s) => s.pose).map((s) => s.pose);
  if (!poses.length) return null;
  const out = { shoulder_tilt: r(mean(poses.map((p) => p.tilt)), 1) };
  const arms = poses.filter((p) => "arms_open" in p).map((p) => p.arms_open);
  if (arms.length) out.arms_open = r(mean(arms), 2);
  return out;
}

function speaker(S, voice, voiced) {
  const faces = S.filter((s) => s.face);
  const faceVis = faces.length / S.length;
  const out = { face_visible: r(faceVis, 2) };
  const jaw = faces.map((s) => s.face.jaw ?? 0);
  const mouth = jaw.length > 2 ? std(jaw) : 0;
  out.mouth_activity = r(mouth, 3);
  let sync = null;
  if (voice && faces.length >= 8 && std(jaw) > 1e-4) {
    const db = voice.dbAt(faces.map((s) => s.t));
    if (std(db) > 1e-3) { sync = corr(jaw, db); out.lip_sync = r(sync, 2); }
  }
  let state;
  if (voiced < 0.15) state = "silent";
  else if (faceVis < 0.3) state = "offscreen";
  else if ((sync !== null && sync > 0.25) || mouth > 0.035) state = "target";
  else state = "listening";
  out.state = state;
  return out;
}

/** window_features() from features.py. `voice` is a VoiceAnalyzer (voice.js) or null. */
export function windowFeatures(index, total, [start, end], samples, voice, segments, base) {
  let S = samples.filter((s) => s.t >= start && s.t < end);
  if (!S.length) S = [{ t: start, cut: false, faces: 0, face: null, hands: [], pose: null }];
  const text = segments.map((s) => s.text).join(" ").trim();
  const sizes = S.filter((s) => s.face).map((s) => s.face.size);
  const scene = { shot: shotType(sizes.length ? median(sizes) : null), faces: Math.max(...S.map((s) => s.faces)), cuts: S.filter((s) => s.cut).length };

  const v = voice ? voice.features(start, end) : { voiced_frac: 0 };
  const speechTime = (v.voiced_frac || 0) * (end - start);
  const nWords = (text.match(/[A-Za-z0-9']+/g) || []).length + (text.match(/[㐀-鿿]/g) || []).length / 1.6;
  if (nWords && speechTime > 0.5) v.speech_rate_wps = r(nWords / speechTime, 1);
  const lowered = text.split(/\s+/).map((w) => w.replace(/^[,.!?;:]+|[,.!?;:]+$/g, "").toLowerCase());
  v.fillers = lowered.filter((w) => FILLERS.includes(w)).length
    + FILLERS.filter((f) => /[^\x00-\x7f]/.test(f)).reduce((s, f) => s + text.split(f).length - 1, 0);

  const g = gesture(S, start);
  const [face, head] = faceAndHead(S, base);
  if (face) g.face = face;
  if (head) g.head = head;
  const p = posture(S);
  if (p) g.posture = p;

  return {
    window: index + 1, of: total, start: r(start, 2), end: r(end, 2), scene,
    speaker: speaker(S, voice, v.voiced_frac || 0), subtitle: text, voice: v, gesture: g,
  };
}
