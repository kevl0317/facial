// Voice measurements in the browser: loudness, pitch (YIN), voice activity, pauses.
// A port of facial/audio.py that analyses audio incrementally (each frame once).

import { median } from "./features.js";

export const SR = 16000;
const HOP = 160; // 10 ms
const FRAME = 1024;
const WIN = 512;
const FMIN = 65, FMAX = 450;
const TAU_MIN = Math.floor(SR / FMAX), TAU_MAX = Math.ceil(SR / FMIN);
const KEEP_S = 120;

function yin(x, start) {
  const d = new Float64Array(TAU_MAX + 1);
  for (let tau = 1; tau <= TAU_MAX; tau++) {
    let s = 0;
    for (let j = 0; j < WIN; j++) { const diff = x[start + j] - x[start + j + tau]; s += diff * diff; }
    d[tau] = s;
  }
  const cmnd = new Float64Array(TAU_MAX + 1);
  let run = 0;
  for (let tau = 1; tau <= TAU_MAX; tau++) { run += d[tau]; cmnd[tau] = run > 0 ? (d[tau] * tau) / run : 1; }
  let tau = -1;
  for (let t = TAU_MIN; t < TAU_MAX; t++) {
    if (cmnd[t] < 0.1) { while (t + 1 < TAU_MAX && cmnd[t + 1] < cmnd[t]) t++; tau = t; break; }
  }
  if (tau < 0) { tau = TAU_MIN; for (let t = TAU_MIN; t <= TAU_MAX; t++) if (cmnd[t] < cmnd[tau]) tau = t; }
  if (tau > TAU_MIN && tau < TAU_MAX) { // parabolic refinement
    const a = cmnd[tau - 1], b = cmnd[tau], c = cmnd[tau + 1];
    const den = a - 2 * b + c;
    if (Math.abs(den) > 1e-12) tau += (0.5 * (a - c)) / den;
  }
  return SR / tau;
}

function medianFilter(values, size) {
  const h = size >> 1, out = new Array(values.length);
  for (let i = 0; i < values.length; i++) out[i] = median(values.slice(Math.max(0, i - h), i + h + 1));
  return out;
}

const semis = (f) => 12 * Math.log2(Math.max(f, 1));

export class VoiceAnalyzer {
  constructor() {
    this.pcm = new Float32Array(0);
    this.t0 = null; // session time of pcm[0]
    this.db = [];
    this.f0 = []; // raw YIN pitch per frame (0 for quiet frames)
  }

  /** Append mono 16 kHz samples; `t0` is the session time of the first sample ever pushed. */
  push(samples, t0) {
    if (this.t0 === null) this.t0 = t0;
    const merged = new Float32Array(this.pcm.length + samples.length);
    merged.set(this.pcm);
    merged.set(samples, this.pcm.length);
    this.pcm = merged;
    // Frames are centred at k * HOP; analyse every frame whose window is now complete.
    for (let k = this.db.length; k * HOP + FRAME / 2 <= this.pcm.length; k++) {
      const c = k * HOP, s0 = c - FRAME / 2;
      let e = 0;
      for (let i = s0; i < s0 + FRAME; i++) { const v = i >= 0 ? this.pcm[i] : 0; e += v * v; }
      const db = 20 * Math.log10(Math.sqrt(e / FRAME) + 1e-10);
      this.db.push(db);
      this.f0.push(db > -50 && s0 >= 0 ? yin(this.pcm, s0) : 0);
    }
    const excessFrames = this.db.length - KEEP_S * (SR / HOP);
    if (excessFrames > 0) {
      this.db.splice(0, excessFrames);
      this.f0.splice(0, excessFrames);
      this.pcm = this.pcm.slice(excessFrames * HOP);
      this.t0 += (excessFrames * HOP) / SR;
    }
  }

  time(k) { return this.t0 + (k * HOP) / SR; }

  /** Speech mask, validated pitch and clip baselines over everything kept. */
  analysis() {
    const n = this.db.length;
    const sorted = [...this.db].sort((a, b) => a - b);
    const ref = sorted[Math.floor(0.95 * (n - 1))] ?? -100;
    const gate = Math.max(ref - 28, -55);
    const speech = medianFilter(this.db.map((d) => (d > gate ? 1 : 0)), 15).map(Boolean);
    const f0 = medianFilter(this.f0, 5);
    const local = medianFilter(f0.map(semis), 25);
    const valid = f0.map((f, i) => speech[i] && f > FMIN * 1.05 && f < FMAX * 0.95 && Math.abs(semis(f) - local[i]) < 3);
    const speechDb = this.db.filter((_, i) => speech[i]);
    const baseDb = speechDb.length ? median(speechDb) : median(this.db);
    const good = f0.filter((_, i) => valid[i]);
    return { speech, f0, valid, baseDb, baseF0: good.length ? median(good) : NaN };
  }

  /** voice_features() from audio.py for [start, end) in session time. */
  features(start, end) {
    if (this.t0 === null || !this.db.length) return { voiced_frac: 0 };
    const a = this.analysis();
    const idx = [];
    for (let k = 0; k < this.db.length; k++) { const t = this.time(k); if (t >= start && t < end) idx.push(k); }
    if (!idx.length) return { voiced_frac: 0 };
    const speech = idx.map((k) => a.speech[k]);
    const out = { voiced_frac: Math.round((speech.filter(Boolean).length / idx.length) * 100) / 100 };
    const sdb = idx.filter((k) => a.speech[k]).map((k) => this.db[k]);
    if (sdb.length) out.loudness_rel_db = Math.round((median(sdb) - a.baseDb) * 10) / 10;
    const f = idx.filter((k) => a.valid[k]).map((k) => a.f0[k]);
    if (f.length >= 10 && Number.isFinite(a.baseF0)) {
      const st = f.map((x) => 12 * Math.log2(x / a.baseF0));
      const m = st.reduce((s, v) => s + v, 0) / st.length;
      out.pitch_rel_st = Math.round(median(st) * 10) / 10;
      out.pitch_var_st = Math.round(Math.sqrt(st.reduce((s, v) => s + (v - m) ** 2, 0) / st.length) * 10) / 10;
    }
    const pauses = [];
    let run = 0, seen = false;
    for (const s of speech) {
      if (s) { if (seen && run * 0.01 >= 0.3) pauses.push(run * 0.01); seen = true; run = 0; } else run++;
    }
    out.pauses = pauses.length;
    out.longest_pause_s = pauses.length ? Math.round(Math.max(...pauses) * 100) / 100 : 0;
    return out;
  }

  /** Loudness (dB) at the given session times, for lip-sync correlation. */
  dbAt(times) {
    return times.map((t) => {
      const k = Math.round(((t - this.t0) * SR) / HOP);
      return this.db[Math.min(this.db.length - 1, Math.max(0, k))] ?? -100;
    });
  }
}

/** Linear resample to 16 kHz (for browsers that capture at 44.1/48 kHz). */
export function resampleTo16k(data, sr) {
  if (sr === SR) return data;
  const n = Math.floor((data.length * SR) / sr);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const x = (i * sr) / SR, i0 = Math.floor(x), f = x - i0;
    out[i] = (data[i0] || 0) * (1 - f) + (data[i0 + 1] || 0) * f;
  }
  return out;
}
