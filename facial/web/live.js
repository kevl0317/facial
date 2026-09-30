// Live mode: camera -> MediaPipe (in the browser) -> HUD, plus one window of
// measurements every few seconds to the server, which answers with a judgment.

import { buildSample, HandTracker } from "./perception.js";
import { Hud, STRINGS, handLabel } from "./hud.js";

const VENDOR = "/models/web/tasks-vision";
const MODELS = {
  face: "/models/face_landmarker.task",
  gesture: "/models/gesture_recognizer.task",
  pose: "/models/pose_landmarker_lite.task",
};
const SAMPLE_FPS = 15; // samples sent to the server (MediaPipe itself runs every frame)

let tasksPromise = null;

/** Create the three MediaPipe tasks once (GPU if possible, else CPU) and reuse them. */
export function loadTasks() {
  tasksPromise ??= (async () => {
    const { FilesetResolver, FaceLandmarker, GestureRecognizer, PoseLandmarker } = await import(`${VENDOR}/vision_bundle.mjs`);
    const fileset = await FilesetResolver.forVisionTasks(`${location.origin}${VENDOR}/wasm`);
    const make = (delegate) => Promise.all([
      FaceLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: MODELS.face, delegate }, runningMode: "VIDEO", numFaces: 2,
        outputFaceBlendshapes: true, outputFacialTransformationMatrixes: true,
      }),
      GestureRecognizer.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: MODELS.gesture, delegate }, runningMode: "VIDEO", numHands: 2,
      }),
      PoseLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: MODELS.pose, delegate }, runningMode: "VIDEO", numPoses: 1,
      }),
    ]);
    let delegate = "GPU";
    let tasks;
    try {
      tasks = await make("GPU");
    } catch (err) {
      console.warn("MediaPipe GPU delegate unavailable, using CPU:", err);
      delegate = "CPU";
      tasks = await make("CPU");
    }
    const [face, hands, pose] = tasks;
    return { face, hands, pose, delegate };
  })();
  tasksPromise.catch(() => { tasksPromise = null; });
  return tasksPromise;
}

function shotType(size) {
  if (!size) return "cutaway";
  if (size >= 0.33) return "close-up";
  if (size >= 0.12) return "medium";
  return "wide";
}

async function startAudio(stream) {
  const open = async (options) => {
    const ctx = new AudioContext(options);
    await ctx.audioWorklet.addModule("/static/pcm-worklet.js");
    const src = ctx.createMediaStreamSource(stream);
    const node = new AudioWorkletNode(ctx, "pcm-tap");
    src.connect(node);
    node.connect(ctx.destination); // the tap outputs silence; connecting keeps it running
    await ctx.resume();
    return { ctx, node, chunks: [], sr: ctx.sampleRate };
  };
  try {
    return await open({ sampleRate: 16000 });
  } catch {
    return open(); // some browsers can't resample a mic stream: use the native rate
  }
}

/** Drain captured audio as base64 int16 PCM (decimated to 16 kHz when the rate allows). */
function takePcm(audio) {
  const chunks = audio.chunks.splice(0);
  const n = chunks.reduce((s, c) => s + c.length, 0);
  let data = new Float32Array(n);
  let o = 0;
  for (const c of chunks) { data.set(c, o); o += c.length; }
  let sr = audio.sr;
  const factor = sr % 16000 === 0 ? sr / 16000 : 1;
  if (factor > 1) {
    const m = Math.floor(n / factor);
    const d = new Float32Array(m);
    for (let i = 0; i < m; i++) {
      let s = 0;
      for (let k = 0; k < factor; k++) s += data[i * factor + k];
      d[i] = s / factor;
    }
    data = d;
    sr = 16000;
  }
  const pcm = new Int16Array(data.length);
  for (let i = 0; i < data.length; i++) pcm[i] = Math.max(-1, Math.min(1, data[i])) * 32767;
  const bytes = new Uint8Array(pcm.buffer);
  let bin = "";
  for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  return { audio: btoa(bin), audio_sr: sr };
}

/** Browser speech recognition (Chrome, Edge, Safari) for live subtitles. */
class Speech {
  constructor(lang, now) {
    this.lang = lang;
    this.now = now;
    this.pending = [];
    this.last = null;
    this.interim = "";
    this.gen = 0;
    this.starts = new Map();
  }

  start() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) return false;
    const rec = new SR();
    rec.continuous = true;
    rec.interimResults = true;
    rec.lang = this.lang === "zh" ? "zh-CN" : "en-US";
    rec.onresult = (e) => {
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const r = e.results[i];
        const key = `${this.gen}:${i}`;
        if (!this.starts.has(key)) this.starts.set(key, Math.max(0, this.now() - 0.8));
        if (r.isFinal) {
          const text = r[0].transcript.trim();
          if (text) {
            const seg = { start: this.starts.get(key), end: this.now(), text };
            this.pending.push(seg);
            this.last = seg;
          }
          this.interim = "";
        } else {
          this.interim = r[0].transcript;
        }
      }
    };
    rec.onerror = (e) => {
      if (["not-allowed", "service-not-allowed", "audio-capture"].includes(e.error)) this.disabled = e.error;
    };
    rec.onend = () => {
      this.gen += 1;
      this.starts.clear();
      if (this.running && !this.disabled) {
        try { rec.start(); } catch { /* already starting */ }
      }
    };
    this.rec = rec;
    this.running = true;
    try { rec.start(); } catch { return false; }
    return true;
  }

  stop() {
    this.running = false;
    try { this.rec && this.rec.stop(); } catch { /* not started */ }
  }

  /** Final segments since the last call, clamped into [start, end]. */
  take(start, end) {
    return this.pending.splice(0).map((s) => ({
      start: Math.min(Math.max(s.start, start), end), end: Math.min(Math.max(s.end, start), end), text: s.text,
    }));
  }

  current(t) {
    if (this.interim) return this.interim;
    return this.last && t - this.last.end < 3 ? this.last.text : "";
  }
}

export class LiveEngine {
  constructor({ canvas, api, onStatus, onWindow }) {
    this.canvas = canvas;
    this.api = api;
    this.onStatus = onStatus || (() => {});
    this.onWindow = onWindow || (() => {});
    this.running = false;
  }

  status(msg) { this.onStatus(msg); }

  async start(opts) {
    this.opts = { lang: "en", context: "", facing: "user", mic: true, speech: true, skeleton: false, windowSec: 5, ...opts };
    this.s = STRINGS[this.opts.lang];
    this.hud = new Hud(this.opts.lang);
    this.status("Loading…");
    this.tasks = await loadTasks();
    try { await document.fonts.load("700 20px Fredoka"); } catch { /* system font fallback */ }

    this.status("Opening camera…");
    this.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: this.opts.facing, width: { ideal: 1280 }, height: { ideal: 720 } },
      // Raw audio: echo cancellation / noise suppression / AGC would distort loudness and pitch.
      audio: this.opts.mic ? { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false } : false,
    });
    this.video = document.createElement("video");
    this.video.playsInline = true;
    this.video.muted = true;
    this.video.srcObject = this.stream;
    await this.video.play();
    this.mirror = this.opts.facing === "user";

    const session = await this.api("/api/live/sessions", { method: "POST", body: { lang: this.opts.lang, context: this.opts.context } });
    this.sessionId = session.id;
    this.judge = session.judge;

    this.t0 = performance.now();
    this.windowStart = 0;
    this.windowSamples = [];
    this.recent = [];
    this.tracker = new HandTracker();
    this.lastSampleT = -1;
    this.lastVideoTime = -1;
    this.frameNo = 0;
    this.lastPose = null;
    this.fps = 0;
    this.fpsFrames = [];
    this.inFlight = false;
    this.windowCount = 0;

    this.audio = null;
    this.audioT0 = null;
    if (this.opts.mic && this.stream.getAudioTracks().length) {
      try {
        this.audio = await startAudio(this.stream);
        this.audio.node.port.onmessage = (e) => {
          if (this.audioT0 === null) this.audioT0 = Math.max(0, this.now() - e.data.length / this.audio.sr);
          this.audio.chunks.push(e.data);
        };
      } catch (err) {
        console.warn("Audio capture unavailable:", err);
      }
    }
    this.speech = null;
    if (this.opts.speech) {
      const sp = new Speech(this.opts.lang, () => this.now());
      if (sp.start()) this.speech = sp;
    }
    try { this.wakeLock = await navigator.wakeLock?.request("screen"); } catch { /* optional */ }

    this.running = true;
    this.status(this.opts.speech && !this.speech ? "No subtitles in this browser" : "");
    this.loop();
  }

  now() { return (performance.now() - this.t0) / 1000; }

  async stop() {
    this.running = false;
    cancelAnimationFrame(this.raf);
    this.stream?.getTracks().forEach((tr) => tr.stop());
    this.speech?.stop();
    try { await this.audio?.ctx.close(); } catch { /* closed */ }
    try { await this.wakeLock?.release(); } catch { /* released */ }
    if (this.sessionId) this.api(`/api/live/sessions/${this.sessionId}`, { method: "DELETE" }).catch(() => {});
    this.sessionId = null;
  }

  loop() {
    if (!this.running) return;
    try {
      this.step();
    } catch (err) {
      console.error(err);
      this.status(err.message);
    }
    this.raf = requestAnimationFrame(() => this.loop());
  }

  step() {
    const t = this.now();
    const v = this.video;
    if (v.readyState >= 2 && v.videoWidth && v.currentTime !== this.lastVideoTime) {
      this.lastVideoTime = v.currentTime;
      const ts = performance.now();
      const face = this.tasks.face.detectForVideo(v, ts);
      const hands = this.tasks.hands.recognizeForVideo(v, ts);
      if (this.frameNo % 3 === 0) this.lastPose = this.tasks.pose.detectForVideo(v, ts);
      this.frameNo += 1;
      // Camera frames are not mirrored (only the preview is), so no handedness swap.
      const sample = buildSample(t, face, hands, this.lastPose, v.videoWidth, v.videoHeight, false);
      this.tracker.update(sample, v.videoWidth / v.videoHeight);
      this.recent.push(sample);
      while (this.recent.length && this.recent[0].t < t - 1.0) this.recent.shift();
      if (t - this.lastSampleT >= 1 / SAMPLE_FPS - 0.005) {
        this.lastSampleT = t;
        this.windowSamples.push({ ...sample, hands: sample.hands.map(({ pts, ...h }) => h) });
      }
      this.fpsFrames.push(ts);
      while (this.fpsFrames.length && this.fpsFrames[0] < ts - 1000) this.fpsFrames.shift();
      this.fps = this.fpsFrames.length;
    }
    this.render(t);
    if (!this.inFlight && t - this.windowStart >= this.opts.windowSec) this.sendWindow(t);
  }

  liveHand(t) {
    const weights = new Map();
    for (const s of this.recent) {
      if (s.t < t - 0.8) continue;
      for (const h of s.hands) weights.set(h.track, (weights.get(h.track) || 0) + h.score * (1 + Math.min(h.speed, 4)));
    }
    if (!weights.size) return null;
    const track = [...weights.entries()].sort((a, b) => b[1] - a[1])[0][0];
    const same = [];
    for (const s of this.recent) if (s.t >= t - 0.4) for (const h of s.hands) if (h.track === track) same.push([s.t, h]);
    if (!same.length) return null;
    const counts = new Map();
    for (const [, h] of same) {
      const k = [h.side, h.shape, h.facing, h.axis].join("|");
      counts.set(k, (counts.get(k) || 0) + 1);
    }
    const [side, shape, facing, axis] = [...counts.entries()].sort((a, b) => b[1] - a[1])[0][0].split("|");
    const [lastT, last] = same[same.length - 1];
    return {
      side, shape, facing, axis,
      score: same.reduce((a, [, h]) => a + h.score, 0) / same.length,
      anchor: t - lastT <= Math.max(0.3, 2.5 * this.tracker.dt) ? last.anchor : null,
    };
  }

  render(t) {
    const c = this.canvas;
    const dpr = window.devicePixelRatio || 1;
    const W = c.clientWidth, H = c.clientHeight;
    if (!W || !H) return;
    if (c.width !== Math.round(W * dpr) || c.height !== Math.round(H * dpr)) {
      c.width = Math.round(W * dpr);
      c.height = Math.round(H * dpr);
    }
    const ctx = c.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#2a2640";
    ctx.fillRect(0, 0, W, H);

    const v = this.video;
    const vw = v.videoWidth || 16, vh = v.videoHeight || 9;
    const scale = Math.max(W / vw, H / vh); // "cover" crop
    const dw = vw * scale, dh = vh * scale, ox = (W - dw) / 2, oy = (H - dh) / 2;
    if (v.videoWidth) {
      ctx.save();
      if (this.mirror) { ctx.translate(W, 0); ctx.scale(-1, 1); }
      ctx.drawImage(v, ox, oy, dw, dh);
      ctx.restore();
    }
    const toPx = ([nx, ny]) => [ox + (this.mirror ? 1 - nx : nx) * dw, oy + ny * dh];

    const latest = this.recent[this.recent.length - 1];
    const live = this.liveHand(t);
    let skeleton = null;
    if (this.opts.skeleton && latest) {
      let faceBox = null;
      if (latest.face) {
        const [a, b] = [toPx(latest.face.bbox.slice(0, 2)), toPx(latest.face.bbox.slice(2))];
        faceBox = [Math.min(a[0], b[0]), a[1], Math.max(a[0], b[0]), b[1]];
      }
      skeleton = { hands: latest.hands.map((h) => h.pts.map(toPx)), face: faceBox };
    }
    let status;
    if (this.inFlight) status = { kind: "judging", text: `${this.s.judging}…` };
    else if (!this.hud.judgments.length) {
      status = { kind: "collecting", text: `${this.s.collecting} ${Math.max(0, this.opts.windowSec - t).toFixed(0)}s` };
    } else status = { kind: "live", text: `${this.s.live} · ${this.fps} fps` };

    this.hud.draw(ctx, W, H, {
      t,
      shot: shotType(latest && latest.face ? latest.face.size : null),
      live: live ? { label: handLabel(live, this.opts.lang), score: live.score, anchor: live.anchor && toPx(live.anchor) } : null,
      subtitle: this.speech ? this.speech.current(t) : "",
      judge: this.hud.judgments.length ? this.hud.judgments[this.hud.judgments.length - 1].source : this.judge,
      status,
      skeleton,
    });
  }

  async sendWindow(end) {
    this.inFlight = true;
    const start = Math.max(this.windowStart, end - 110);
    this.windowStart = end;
    const samples = this.windowSamples.splice(0).filter((s) => s.t >= start && s.t < end);
    const v = this.video;
    const body = {
      start, end, samples, aspect: v.videoWidth / v.videoHeight || 16 / 9,
      transcript: this.speech ? this.speech.take(start, end) : [],
    };
    if (this.audio && this.audio.chunks.length) Object.assign(body, takePcm(this.audio), { audio_t0: this.audioT0 });
    try {
      const res = await this.api(`/api/live/sessions/${this.sessionId}/windows`, { method: "POST", body });
      if (!this.running) return;
      this.windowCount += 1;
      this.hud.addJudgment(res.judgment);
      this.onWindow(res);
    } catch (err) {
      this.status(`Verdict failed: ${err.message}`);
    } finally {
      this.inFlight = false;
    }
  }
}
