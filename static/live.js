// The analysis engine: a source (camera, or an uploaded video played in real time) ->
// MediaPipe in the browser -> HUD, plus one window of measurements every few seconds
// to a judge. The judge is the facial server (self-hosted) or runs in the browser.

import { buildSample, HandTracker } from "./perception.js";
import { Hud, STRINGS, handLabel, judgeLabel } from "./hud.js";
import { clipBaseline, shotType, windowFeatures } from "./features.js";
import { VoiceAnalyzer, resampleTo16k } from "./voice.js";
import { BrowserJudge } from "./judge.js";
import { expressionScores } from "./expressions.js";
import { EXPRESSIONS } from "./prompt.js";

const asset = (path) => new URL(path, document.baseURI).href;
const VENDOR = "models/web/tasks-vision";
const MODELS = {
  face: "models/face_landmarker.task",
  gesture: "models/gesture_recognizer.task",
  pose: "models/pose_landmarker_lite.task",
};
const SAMPLE_FPS = 15; // samples kept for judging (MediaPipe itself runs every frame)

// ---------------------------------------------------------------------------------- MediaPipe

async function fetchAll(urls, onProgress) {
  const sizes = new Array(urls.length).fill(0), done = new Array(urls.length).fill(0);
  const report = () => {
    const total = sizes.reduce((a, b) => a + b, 0);
    if (total) onProgress(Math.min(1, done.reduce((a, b) => a + b, 0) / total));
  };
  return Promise.all(urls.map(async (url, i) => {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${res.status} loading ${url}`);
    sizes[i] = Number(res.headers.get("content-length")) || 0;
    const reader = res.body.getReader();
    const chunks = [];
    for (;;) {
      const { done: end, value } = await reader.read();
      if (end) break;
      chunks.push(value);
      done[i] += value.length;
      report();
    }
    const out = new Uint8Array(done[i]);
    let o = 0;
    for (const c of chunks) { out.set(c, o); o += c.length; }
    return out;
  }));
}

let tasksPromise = null;
let progressListener = () => {};

/** Create the three MediaPipe tasks once (GPU if possible, else CPU); reports download progress. */
export function loadTasks(onProgress) {
  if (onProgress) progressListener = onProgress;
  tasksPromise ??= (async () => {
    const { FilesetResolver, FaceLandmarker, GestureRecognizer, PoseLandmarker } = await import(asset(`${VENDOR}/vision_bundle.mjs`));
    const fileset = await FilesetResolver.forVisionTasks(asset(`${VENDOR}/wasm`));
    // Download the WASM runtime (warms the cache MediaPipe then reads) and the models, with progress.
    const [, face, gesture, pose] = await fetchAll(
      [fileset.wasmBinaryPath, asset(MODELS.face), asset(MODELS.gesture), asset(MODELS.pose)],
      (f) => progressListener(f),
    );
    const make = (delegate) => Promise.all([
      FaceLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetBuffer: face, delegate }, runningMode: "VIDEO", numFaces: 2,
        outputFaceBlendshapes: true, outputFacialTransformationMatrixes: true,
      }),
      GestureRecognizer.createFromOptions(fileset, {
        baseOptions: { modelAssetBuffer: gesture, delegate }, runningMode: "VIDEO", numHands: 2,
      }),
      PoseLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetBuffer: pose, delegate }, runningMode: "VIDEO", numPoses: 1,
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
    progressListener(1);
    const [faceTask, handTask, poseTask] = tasks;
    return { face: faceTask, hands: handTask, pose: poseTask, delegate };
  })();
  tasksPromise.catch(() => { tasksPromise = null; });
  return tasksPromise;
}

// ---------------------------------------------------------------------------------- audio

/**
 * Wait for `promise`, but no longer than `ms`. On iPhones AudioContext.resume() can stay
 * pending for good once the microphone or speech recognition takes the audio session,
 * and that must never stall the app.
 */
const settle = (promise, ms) => Promise.race([promise, new Promise((r) => setTimeout(r, ms))]);

async function openAudio(connect, preferred16k, unlocked) {
  const open = async (options) => {
    const ctx = unlocked || new AudioContext(options);
    await ctx.audioWorklet.addModule(asset("static/pcm-worklet.js"));
    const node = new AudioWorkletNode(ctx, "pcm-tap");
    const source = connect(ctx);
    source.connect(node);
    node.connect(ctx.destination); // the tap outputs silence; connecting keeps it running
    await settle(ctx.resume(), 1500);
    return { ctx, node, source, chunks: [], sr: ctx.sampleRate };
  };
  if (preferred16k && !unlocked) {
    try { return await open({ sampleRate: 16000 }); } catch { /* fall through to the native rate */ }
  }
  return open();
}

function drainPcm(audio) {
  const chunks = audio.chunks.splice(0);
  const out = new Float32Array(chunks.reduce((s, c) => s + c.length, 0));
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}

function toBase64Pcm16(data, sr) {
  const factor = sr % 16000 === 0 ? sr / 16000 : 1;
  if (factor > 1) {
    const m = Math.floor(data.length / factor), d = new Float32Array(m);
    for (let i = 0; i < m; i++) { let s = 0; for (let k = 0; k < factor; k++) s += data[i * factor + k]; d[i] = s / factor; }
    data = d; sr = 16000;
  }
  const pcm = new Int16Array(data.length);
  for (let i = 0; i < data.length; i++) pcm[i] = Math.max(-1, Math.min(1, data[i])) * 32767;
  const bytes = new Uint8Array(pcm.buffer);
  let bin = "";
  for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  return { audio: btoa(bin), audio_sr: sr };
}

// ---------------------------------------------------------------------------------- judges

/** Windows go to the facial server, which measures and judges them (API key stays there). */
class ServerBackend {
  constructor(api) { this.api = api; }
  async open(opts) {
    const s = await this.api("api/live/sessions", { method: "POST", body: { lang: opts.lang, context: opts.context } });
    this.id = s.id;
    this.name = s.judge;
    this.label = s.label || s.judge;
  }
  async judge(win) {
    const body = { start: win.start, end: win.end, samples: win.samples, aspect: win.aspect, transcript: win.transcript };
    if (win.pcm && win.pcm.length) Object.assign(body, toBase64Pcm16(win.pcm, win.sr), { audio_t0: win.audioT0 });
    return this.api(`api/live/sessions/${this.id}/windows`, { method: "POST", body });
  }
  close() { if (this.id) this.api(`api/live/sessions/${this.id}`, { method: "DELETE" }).catch(() => {}); }
}

/** Everything in the browser: the same five fields, then an AI model (own key) or the rules. */
class LocalBackend {
  constructor(opts, onNotice) {
    this.opts = opts;
    this.voice = new VoiceAnalyzer();
    this.samples = [];
    this.index = 0;
    this.judgeImpl = new BrowserJudge({
      provider: opts.provider, apiKey: opts.apiKey, model: opts.model, baseUrl: opts.baseUrl,
      lang: opts.lang, context: opts.context, handLabel: (h) => handLabel(h, opts.lang), onNotice,
      jev: opts.jev ? { route: opts.jevVia, apiKey: opts.jevKey, model: opts.jevModel } : null,
    });
  }
  get name() { return this.judgeImpl.name; }
  get label() { return this.judgeImpl.label; }
  async open() {}
  async judge(win) {
    if (win.pcm && win.pcm.length) this.voice.push(resampleTo16k(win.pcm, win.sr), win.audioT0 ?? win.start);
    this.samples = this.samples.filter((s) => s.t >= win.end - 600).concat(win.samples);
    const baseline = clipBaseline(this.samples);
    const feats = windowFeatures(this.index, win.total ?? null, [win.start, win.end], win.samples,
      this.voice.t0 === null ? null : this.voice, win.transcript, baseline);
    const clip = this.index === 0 ? { live: !win.total, duration: win.duration, baseline } : null;
    this.index += 1;
    const judgment = await this.judgeImpl.judge(feats, clip);
    return { window: feats, judgment };
  }
  close() {}
}

// ---------------------------------------------------------------------------------- speech

/**
 * Browser speech recognition (Chrome, Edge, Safari) for live subtitles.
 *
 * Phone browsers end recognition after every sentence even in "continuous" mode, and
 * may refuse a restart that doesn't come from a tap. So each sentence gets a fresh
 * recognizer, failed restarts retry with backoff, and a refusal after it has already
 * worked asks for a tap instead of switching subtitles off for good.
 */
export class Speech {
  constructor(lang, now, { onNeedsTap, onGiveUp } = {}) {
    this.lang = lang;
    this.now = now;
    this.onNeedsTap = onNeedsTap || (() => {});
    this.onGiveUp = onGiveUp || (() => {});
    this.pending = [];
    this.last = null;
    this.interim = "";
    this.interimStart = null;
    this.gen = 0;
    this.starts = new Map();
    this.heard = false; // true once any speech has been recognised
    this.failures = 0;
    this.captureFailures = 0;
  }

  start() {
    this.SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!this.SR) return false;
    this.running = true;
    this.spawn();
    return true;
  }

  spawn() {
    if (!this.running) return;
    const rec = new this.SR();
    const gen = ++this.gen;
    rec.continuous = true;
    rec.interimResults = true;
    rec.lang = this.lang === "zh" ? "zh-CN" : "en-US";
    rec.onresult = (e) => {
      this.heard = true;
      this.failures = 0;
      this.captureFailures = 0;
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const res = e.results[i];
        const key = `${gen}:${i}`;
        if (!this.starts.has(key)) this.starts.set(key, Math.max(0, this.now() - 0.8));
        if (res.isFinal) {
          this.commit(res[0].transcript, this.starts.get(key));
          this.starts.delete(key);
        } else {
          this.interim = res[0].transcript;
          this.interimStart = this.starts.get(key);
        }
      }
    };
    rec.onerror = (e) => { this.lastError = e.error; };
    rec.onend = () => {
      if (rec !== this.rec) return;
      // Keep a sentence the recognizer never finalised (phones often stop mid-sentence).
      if (this.interim) this.commit(this.interim, this.interimStart);
      if (!this.running) return;
      const err = this.lastError;
      this.lastError = null;
      if (err === "not-allowed" || err === "service-not-allowed") {
        if (!this.heard) { this.giveUp("Subtitles need microphone permission"); return; }
        this.needsTap = true; // worked before: the browser wants a tap to listen again
        this.onNeedsTap();
        return;
      }
      if (err === "audio-capture") this.captureFailures += 1;
      if (this.captureFailures >= 3 && !this.heard) {
        this.giveUp("Subtitles can't share the microphone with Voice on this device. Turn Voice off to use them.");
        return;
      }
      if (err && err !== "no-speech" && err !== "aborted") this.failures += 1;
      const delay = err ? Math.min(3000, 200 * 2 ** Math.min(this.failures, 4)) : 50;
      this.timer = setTimeout(() => this.spawn(), delay);
    };
    this.rec = rec;
    try {
      rec.start();
    } catch {
      this.failures += 1;
      this.timer = setTimeout(() => this.spawn(), 1000);
    }
  }

  commit(text, start) {
    text = (text || "").trim();
    this.interim = "";
    this.interimStart = null;
    if (!text) return;
    const seg = { start: start ?? Math.max(0, this.now() - 2), end: this.now(), text };
    this.pending.push(seg);
    this.last = seg;
  }

  giveUp(message) {
    this.running = false;
    this.disabled = true;
    this.onGiveUp(message);
  }

  /** Call from a tap: restarts listening when the browser asked for a gesture. */
  resume() {
    if (!this.needsTap || this.disabled) return;
    this.needsTap = false;
    this.running = true;
    this.spawn();
  }

  stop() {
    this.running = false;
    clearTimeout(this.timer);
    try { this.rec && this.rec.abort(); } catch { /* not started */ }
  }

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

// ---------------------------------------------------------------------------------- engine

export class LiveEngine {
  /** mode: "server" (facial server judges) or "static" (everything in the browser). */
  constructor({ canvas, api, mode, onStatus, onWindow, onEnd }) {
    this.canvas = canvas;
    this.api = api;
    this.mode = mode;
    this.onStatus = onStatus || (() => {});
    this.onWindow = onWindow || (() => {});
    this.onEnd = onEnd || (() => {});
    this.running = false;
  }

  status(msg) { this.onStatus(msg); }

  /**
   * opts: {file?: File, segments?: [{start,end,text}], lang, context, facing, mic, speech, skeleton,
   *        windowSec, provider, apiKey, model, baseUrl, jev, jevVia, jevKey, jevModel, record}
   */
  async start(opts) {
    this.opts = { lang: "en", context: "", facing: "user", mic: true, speech: true, skeleton: false, windowSec: 5, ...opts };
    this.isFile = Boolean(this.opts.file);
    this.s = STRINGS[this.opts.lang];
    this.hud = new Hud(this.opts.lang);
    this.status("Loading…");
    this.tasks = await loadTasks();
    try { await document.fonts.load("700 20px Fredoka"); } catch { /* system font fallback */ }

    this.video = document.createElement("video");
    this.video.playsInline = true;
    if (this.isFile) {
      this.video.src = URL.createObjectURL(this.opts.file);
      this.video.preload = "auto";
      await new Promise((ok, fail) => {
        const unplayable = () => fail(new Error("This browser can't play that video. Try an MP4 or WebM file."));
        const timer = setTimeout(unplayable, 15000);
        this.video.onloadedmetadata = () => { clearTimeout(timer); ok(); };
        this.video.onerror = () => { clearTimeout(timer); unplayable(); };
      });
      if (!this.video.videoWidth) throw new Error("This browser can't show that video's picture. Try an MP4 or WebM file.");
      this.mirror = false;
    } else {
      this.status("Opening camera…");
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: this.opts.facing, width: { ideal: 1280 }, height: { ideal: 720 } },
        // Raw audio: echo cancellation / noise suppression / AGC would distort loudness and pitch.
        audio: this.opts.mic ? { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false } : false,
      });
      this.video.muted = true;
      this.video.srcObject = this.stream;
      this.mirror = this.opts.facing === "user";
    }

    this.backend = this.mode === "server" ? new ServerBackend(this.api) : new LocalBackend(this.opts, (m) => this.status(m));
    await this.backend.open(this.opts);
    this.judgeName = this.backend.name;
    this.judgeLabel = this.backend.name === "heuristic" ? this.s.rules : this.backend.label;

    this.t0 = performance.now();
    this.windowStart = 0;
    this.windowSamples = [];
    this.recent = [];
    this.tracker = new HandTracker();
    this.lastSampleT = -1;
    this.lastVideoTime = -1;
    this.frameNo = 0;
    this.lastPose = null;
    this.fpsFrames = [];
    this.fps = 0;
    this.inFlight = false;
    this.windowCount = 0;
    this.results = [];
    this.audio = null;
    this.audioT0 = null;

    if (this.isFile) {
      try {
        this.audio = await openAudio((ctx) => {
          const src = ctx.createMediaElementSource(this.video);
          src.connect(ctx.destination); // keep the soundtrack audible
          return src;
        }, false, this.opts.audioCtx);
      } catch (err) { console.warn("No audio analysis for this file:", err); }
    } else if (this.opts.mic && this.stream.getAudioTracks().length) {
      try { this.audio = await openAudio((ctx) => ctx.createMediaStreamSource(this.stream), true, this.opts.audioCtx); } catch (err) {
        console.warn("Audio capture unavailable:", err);
      }
    }
    if (this.audio) {
      this.audio.node.port.onmessage = (e) => {
        this.lastPcm = performance.now();
        if (this.isFile && this.video.paused) return; // keep the audio timeline in step with playback
        if (this.audioT0 === null) this.audioT0 = Math.max(0, this.now() - e.data.length / this.audio.sr);
        this.audio.chunks.push(e.data);
      };
    }
    this.speech = null;
    if (!this.isFile && this.opts.speech) {
      const sp = new Speech(this.opts.lang, () => this.now(), {
        onNeedsTap: () => {
          this.status("Tap the picture to resume subtitles");
          this.canvas.addEventListener("pointerdown", () => { sp.resume(); this.status(""); }, { once: true });
        },
        onGiveUp: (msg) => this.status(msg),
      });
      if (sp.start()) this.speech = sp;
    }
    if (this.isFile) this.setupFile();
    try { this.wakeLock = await navigator.wakeLock?.request("screen"); } catch { /* optional */ }

    this.running = true;
    this.status(!this.isFile && this.opts.speech && !this.speech ? "No subtitles in this browser" : "");
    this.lastPcm = performance.now();
    if (!this.isFile && this.audio) this.audioWatch = setInterval(() => this.checkAudio(), 2000);
    this.loop();
    await this.play();
  }

  setupFile() {
    const v = this.video;
    const scale = Math.min(1, 1280 / Math.max(v.videoWidth, v.videoHeight));
    this.canvas.width = Math.round(v.videoWidth * scale);
    this.canvas.height = Math.round(v.videoHeight * scale);
    this.canvas.classList.add("contain");
    // HUD scale: the offline renderer's (H/720), enlarged when the video is shown small
    // (e.g. a landscape clip on a phone) so it stays readable on screen.
    const base = this.canvas.width < this.canvas.height ? this.canvas.width / 360 : this.canvas.height / 720;
    const shown = this.canvas.clientWidth ? this.canvas.width / this.canvas.clientWidth : 1;
    this.hudScale = Math.min(2 * base, Math.max(base, 0.55 * shown));
    this.total = Math.max(1, Math.ceil(v.duration / this.opts.windowSec));
    v.addEventListener("ended", () => this.finish());
    this.chunks = [];
    this.recorder = null;
    if (this.opts.record !== false && window.MediaRecorder && this.canvas.captureStream) {
      const tracks = this.canvas.captureStream(30).getVideoTracks();
      if (this.audio) {
        const dest = this.audio.ctx.createMediaStreamDestination();
        this.audio.source.connect(dest);
        tracks.push(...dest.stream.getAudioTracks());
      }
      const type = ["video/mp4;codecs=avc1.42E01E,mp4a.40.2", "video/mp4", "video/webm;codecs=vp9,opus",
        "video/webm;codecs=vp8,opus", "video/webm"].find((t) => MediaRecorder.isTypeSupported(t));
      try {
        this.recorder = new MediaRecorder(new MediaStream(tracks), type ? { mimeType: type, videoBitsPerSecond: 5e6 } : {});
        this.recorder.ondataavailable = (e) => { if (e.data.size) this.chunks.push(e.data); };
        this.recorder.start(1000);
        // Only record while the video plays, so waiting for a verdict doesn't stretch the result.
        v.addEventListener("pause", () => { if (this.recorder?.state === "recording") this.recorder.pause(); });
        v.addEventListener("playing", () => { if (this.recorder?.state === "paused") this.recorder.resume(); });
      } catch (err) { console.warn("Recording unavailable:", err); this.recorder = null; }
    }
  }

  now() { return this.isFile ? this.video.currentTime : (performance.now() - this.t0) / 1000; }

  /** Start playback; some browsers (iOS Safari) need a fresh tap for sound, so report that. */
  async play() {
    try {
      if (this.audio) await settle(this.audio.ctx.resume(), 1500);
      await settle(this.video.play(), 4000);
      this.needsTap = false;
    } catch (err) {
      if (err.name !== "NotAllowedError") throw err;
      this.needsTap = true;
    }
    this.onStatus(this.needsTap ? "tap-to-play" : "");
  }

  togglePause() {
    if (!this.isFile || !this.running) return;
    if (this.video.paused) { this.userPaused = false; if (!this.waiting) this.play(); } else { this.userPaused = true; this.video.pause(); }
  }

  /** Live mode: notice when Voice stops getting sound (another app or Subtitles took the mic). */
  checkAudio() {
    if (!this.running || !this.audio || this.audioHint || performance.now() - this.lastPcm < 4000) return;
    this.audioHint = true;
    settle(this.audio.ctx.resume(), 1500).catch(() => {});
    this.status(this.speech ? "Voice has no sound. Tap the picture, or turn Subtitles off." : "Voice has no sound. Tap the picture.");
    this.canvas.addEventListener("pointerdown", () => {
      this.audio?.ctx.resume().catch(() => {});
      this.audioHint = false;
      this.lastPcm = performance.now();
      this.status("");
    }, { once: true });
  }

  async stop() {
    // Live: leave a clean last frame on screen (no "Thinking…") to look at after stopping.
    if (!this.isFile && this.running && this.hud) {
      this.stopped = true;
      try { this.render(this.now()); } catch { /* nothing drawn yet */ }
    }
    this.running = false;
    clearInterval(this.audioWatch);
    cancelAnimationFrame(this.raf);
    this.stream?.getTracks().forEach((tr) => tr.stop());
    this.speech?.stop();
    if (this.video) { this.video.pause(); if (this.isFile) URL.revokeObjectURL(this.video.src); }
    if (this.recorder && this.recorder.state !== "inactive") this.recorder.stop();
    if (!this.opts.audioCtx) { try { await this.audio?.ctx.close(); } catch { /* closed */ } }
    else { try { this.audio?.node.disconnect(); this.audio?.source.disconnect(); } catch { /* disconnected */ } }
    try { await this.wakeLock?.release(); } catch { /* released */ }
    this.backend?.close();
    this.canvas.classList.remove("contain");
    this.status("");
  }

  async finish() {
    if (this.finishing) return;
    this.finishing = true;
    const t = this.video.duration;
    while (this.inFlight) await new Promise((r) => setTimeout(r, 100));
    if (t - this.windowStart >= 0.5) await this.sendWindow(t);
    this.render(t);
    let blob = null;
    if (this.recorder && this.recorder.state !== "inactive") {
      await new Promise((r) => { this.recorder.onstop = r; this.recorder.stop(); });
      blob = new Blob(this.chunks, { type: this.recorder.mimeType || "video/webm" });
    }
    const results = this.results;
    await this.stop();
    this.onEnd({ blob, results });
  }

  loop() {
    if (!this.running) return;
    try { this.step(); } catch (err) { console.error(err); this.status(err.message); }
    this.raf = requestAnimationFrame(() => this.loop());
  }

  step() {
    const v = this.video;
    const t = this.now();
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
      if (sample.face) this.trackFaceBaseline(sample);
      while (this.recent.length && this.recent[0].t < t - 1.6) this.recent.shift();
      if (t - this.lastSampleT >= 1 / SAMPLE_FPS - 0.005) {
        this.lastSampleT = t;
        this.windowSamples.push({ ...sample, hands: sample.hands.map(({ pts, ...h }) => h) });
      }
      this.fpsFrames.push(ts);
      while (this.fpsFrames.length && this.fpsFrames[0] < ts - 1000) this.fpsFrames.shift();
      this.fps = this.fpsFrames.length;
    }
    this.render(t);
    const due = this.windowStart + this.opts.windowSec;
    if (this.finishing || t < due) return;
    // Files are cut on an exact grid; live windows end at the current frame.
    if (!this.inFlight) this.sendWindow(this.isFile ? due : t);
    else if (this.isFile && !v.paused) { this.waiting = true; v.pause(); } // wait for the verdict
  }

  /** Up to two hands: the most active tracks over the last 0.8 s, each with its most common
   * state over the last 0.4 s. A second hand needs to show in 40% of recent frames (no flicker). */
  /** The person's resting face (medians of recent frames), so a resting frown isn't read as anger. */
  trackFaceBaseline(sample) {
    this.faceFrames = this.faceFrames || [];
    this.faceFrames.push({ face: sample.face });
    if (this.faceFrames.length > 400) this.faceFrames.shift();
    this.faceCount = (this.faceCount || 0) + 1;
    if (this.faceCount % 30 === 1) this.faceBase = clipBaseline(this.faceFrames);
  }

  /** The face expression over the last 1.5 s: its name and share. */
  liveFace(t) {
    const faces = this.recent.filter((s) => s.t >= t - 1.5 && s.face).map((s) => s.face);
    if (!faces.length || !faces[faces.length - 1].bbox) return null;
    const mean = {};
    for (const f of faces) for (const [k, p] of Object.entries(expressionScores(f, this.faceBase || {}))) mean[k] = (mean[k] || 0) + p / faces.length;
    let top = "neutral";
    for (const k of Object.keys(mean)) if (mean[k] > mean[top]) top = k;
    return { top, score: mean[top], bbox: faces[faces.length - 1].bbox };
  }

  liveHands(t) {
    const weights = new Map();
    for (const s of this.recent) {
      if (s.t < t - 0.8) continue;
      for (const h of s.hands) weights.set(h.track, (weights.get(h.track) || 0) + h.score * (1 + Math.min(h.speed, 4)));
    }
    const recent = this.recent.filter((s) => s.t >= t - 0.4);
    const out = [];
    for (const [track] of [...weights.entries()].sort((a, b) => b[1] - a[1])) {
      if (out.length === 2) break;
      const same = [];
      for (const s of recent) for (const h of s.hands) if (h.track === track) same.push([s.t, h]);
      if (!same.length || (out.length === 1 && same.length < 0.4 * recent.length)) continue;
      const counts = new Map();
      for (const [, h] of same) {
        const k = [h.side, h.shape, h.facing, h.axis].join("|");
        counts.set(k, (counts.get(k) || 0) + 1);
      }
      const [side, shape, facing, axis] = [...counts.entries()].sort((a, b) => b[1] - a[1])[0][0].split("|");
      const [lastT, last] = same[same.length - 1];
      out.push({
        side, shape, facing, axis,
        score: same.reduce((a, [, h]) => a + h.score, 0) / same.length,
        anchor: t - lastT <= Math.max(0.3, 2.5 * this.tracker.dt) ? last.anchor : null,
      });
    }
    // Top label = leftmost hand on screen, so the two pointers don't cross.
    const screenX = (h) => (h.anchor ? (this.mirror ? 1 - h.anchor[0] : h.anchor[0]) : 2);
    return out.sort((a, b) => screenX(a) - screenX(b));
  }

  render(t) {
    const c = this.canvas;
    const v = this.video;
    const vw = v.videoWidth || 16, vh = v.videoHeight || 9;
    let W, H, u, ox, oy, dw, dh;
    const ctx = c.getContext("2d");
    if (this.isFile) {
      // Draw at the video's own resolution (shown with object-fit: contain, and recorded as is).
      W = c.width; H = c.height;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ox = 0; oy = 0; dw = W; dh = H;
      u = this.hudScale;
    } else {
      const dpr = window.devicePixelRatio || 1;
      W = c.clientWidth; H = c.clientHeight;
      if (!W || !H) return;
      if (c.width !== Math.round(W * dpr) || c.height !== Math.round(H * dpr)) { c.width = Math.round(W * dpr); c.height = Math.round(H * dpr); }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const scale = Math.max(W / vw, H / vh); // "cover" crop
      dw = vw * scale; dh = vh * scale; ox = (W - dw) / 2; oy = (H - dh) / 2;
    }
    ctx.fillStyle = "#2a2640";
    ctx.fillRect(0, 0, W, H);
    if (v.videoWidth) {
      ctx.save();
      if (this.mirror) { ctx.translate(W, 0); ctx.scale(-1, 1); }
      ctx.drawImage(v, ox, oy, dw, dh);
      ctx.restore();
    }
    const toPx = ([nx, ny]) => [ox + (this.mirror ? 1 - nx : nx) * dw, oy + ny * dh];

    const latest = this.recent[this.recent.length - 1];
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
    if (this.stopped) status = { kind: "stopped", text: this.s.stopped };
    else if (this.inFlight) status = { kind: "judging", text: `${this.s.judging}…` };
    else if (!this.hud.judgments.length) {
      status = { kind: "collecting", text: `${this.s.collecting} ${Math.max(0, this.opts.windowSec - (t - this.windowStart)).toFixed(0)}s` };
    } else status = { kind: "live", text: this.isFile ? this.s.playing : `${this.s.live} · ${this.fps} fps` };

    this.hud.draw(ctx, W, H, {
      t,
      total: this.isFile ? this.total : null,
      shot: shotType(latest && latest.face ? latest.face.size : null),
      hands: this.liveHands(t).map((h) => ({ label: handLabel(h, this.opts.lang), score: h.score, anchor: h.anchor && toPx(h.anchor) })),
      face: this.faceSticker(t),
      subtitle: this.subtitleAt(t),
      judge: this.hud.judgments.length ? judgeLabel(this.hud.judgments[this.hud.judgments.length - 1], this.s.rules)
        : this.judgeLabel,
      status,
      skeleton,
    }, this.isFile ? u : undefined);
  }

  faceSticker(t) {
    const f = this.liveFace(t);
    if (!f) return null;
    const e = EXPRESSIONS[f.top];
    // No pointer: there's one face, and a line across the picture would only add clutter.
    return { label: `${this.s.face} · ${e[this.opts.lang === "zh" ? 1 : 0]}`, score: f.score, glyph: e[2], anchor: null };
  }

  subtitleAt(t) {
    if (this.speech) return this.speech.current(t);
    const seg = (this.opts.segments || []).find((s) => s.start <= t && t < s.end);
    return seg ? seg.text : "";
  }

  transcriptFor(start, end) {
    if (this.speech) return this.speech.take(start, end);
    return (this.opts.segments || []).filter((s) => { const m = (s.start + s.end) / 2; return m >= start && m < end; });
  }

  async sendWindow(end) {
    this.inFlight = true;
    const start = Math.max(this.windowStart, end - 110);
    this.windowStart = end;
    const pending = this.windowSamples.splice(0);
    const samples = pending.filter((s) => s.t >= start && s.t < end);
    this.windowSamples = pending.filter((s) => s.t >= end); // already belongs to the next window
    const v = this.video;
    const win = {
      start, end, samples, aspect: v.videoWidth / v.videoHeight || 16 / 9,
      transcript: this.transcriptFor(start, end), total: this.isFile ? this.total : null,
      duration: this.isFile ? v.duration : null,
    };
    if (this.audio && this.audio.chunks.length) Object.assign(win, { pcm: drainPcm(this.audio), sr: this.audio.sr, audioT0: this.audioT0 });
    try {
      const res = await this.backend.judge(win);
      if (!this.running) return;
      this.windowCount += 1;
      this.hud.addJudgment(res.judgment);
      this.results.push(res);
      this.onWindow(res);
    } catch (err) {
      this.status(`Verdict failed: ${err.message}`);
    } finally {
      this.inFlight = false;
      if (this.waiting) { this.waiting = false; if (!this.userPaused && this.running && !this.finishing) this.play(); }
    }
  }
}
