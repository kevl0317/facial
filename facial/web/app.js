import { CONFIG } from "./config.js";
import { handSvg } from "./hand.js";
import { playIntro } from "./intro.js";
import { LiveEngine, loadTasks } from "./live.js";
import { parseSubtitles } from "./srt.js";
import qrcode from "./vendor/qrcode.mjs";

const $ = (sel) => document.querySelector(sel);
const STATIC = CONFIG.mode === "static";
const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { if (v === null) localStorage.removeItem(k); else localStorage.setItem(k, v); } catch { /* private mode */ } },
};

playIntro($("#intro"));
for (const el of document.querySelectorAll("[data-mascot]")) {
  el.innerHTML = handSvg({ size: Number(el.dataset.mascot), cls: el.dataset.mascotClass || "" });
}
if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register(new URL("sw.js", document.baseURI)).catch(() => { /* optional */ });
}

// Access key for the self-hosted server (?k=...); the static site has no server.
const params = new URLSearchParams(location.search);
let key = params.get("k") ?? store.get("facial-key") ?? "";
if (params.has("k")) store.set("facial-key", key);

async function api(path, { method = "GET", body } = {}) {
  const headers = { "X-Facial-Key": key };
  let payload;
  if (body !== undefined) { headers["Content-Type"] = "application/json"; payload = JSON.stringify(body); }
  const res = await fetch(new URL(path, document.baseURI), { method, headers, body: payload });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch { /* not JSON */ }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return res.json();
}

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (t) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;

function windowItem(win, j) {
  const li = document.createElement("li");
  li.innerHTML = `
    <div class="head"><span class="w">W${win.window}</span><span class="intent">${esc(j.intent)}</span>
      <span class="time">${fmt(win.start)}–${fmt(win.end)}</span></div>
    <div class="reading">${esc(j.reading)}</div>
    ${j.quote ? `<div class="quote">“${esc(j.quote)}”</div>` : ""}
    <div class="scores"><span class="s-c" title="Confident">C ${j.confidence.toFixed(2)}</span>
      <span class="s-f" title="Focused">F ${j.focus.toFixed(2)}</span>
      <span class="s-t" title="Tense">T ${j.tension.toFixed(2)}</span></div>`;
  return li;
}

function qrSvg(text) {
  const qr = qrcode(0, "M");
  qr.addData(text);
  qr.make();
  return qr.createSvgTag({ cellSize: 4, margin: 2, scalable: true });
}

function initSeg(seg, value, onChange) {
  const set = (v) => {
    for (const b of seg.querySelectorAll("button")) b.setAttribute("aria-checked", String(b.dataset.value === v));
    onChange(v);
  };
  seg.addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) set(b.dataset.value); });
  set(value);
}

/** Created synchronously inside a tap, so iOS lets it run (audio analysis + sound). */
function unlockedAudio() {
  const Ctx = window.AudioContext || window.webkitAudioContext;
  if (!Ctx) return undefined;
  const ctx = new Ctx();
  ctx.resume().catch(() => {});
  return ctx;
}

// ---------------------------------------------------------------------------------- tabs
let engine = null;
let fileEngine = null;
for (const btn of document.querySelectorAll(".tabs button")) {
  btn.addEventListener("click", async () => {
    const tab = btn.dataset.tab;
    for (const b of document.querySelectorAll(".tabs button")) b.setAttribute("aria-selected", String(b === btn));
    $("#tab-live").hidden = tab !== "live";
    $("#tab-file").hidden = tab !== "file";
    if (tab !== "live" && engine && engine.running) await stopLive();
    if (tab !== "file" && fileEngine && fileEngine.running) await stopFile();
  });
}

// ---------------------------------------------------------------------------------- judge + info
const live = {
  lang: store.get("facial-lang") || "en",
  facing: store.get("facial-facing") || "user",
  mic: store.get("facial-mic") !== "0",
  speech: store.get("facial-speech") !== "0",
  skeleton: store.get("facial-skeleton") === "1",
  windowSec: Number(store.get("facial-windowSec") || 5),
  context: store.get("facial-context") || "",
  apiKey: store.get("facial-apikey") || "",
};
const save = (k, v) => { live[k] = v; store.set(`facial-${k}`, typeof v === "boolean" ? (v ? "1" : "0") : String(v)); };

function showJudge(name, title) {
  $("#judgeChip").hidden = false;
  $("#judgeChip span").textContent = name;
  $("#judgeChip").title = title;
}

function showPhoneQr(url) {
  const onThisComputer = ["localhost", "127.0.0.1", "[::1]"].includes(location.hostname) || !STATIC;
  if (!url || !(onThisComputer || window.matchMedia("(min-width: 821px)").matches)) return;
  $("#phoneQr").innerHTML = qrSvg(url);
  $("#phoneCard").title = url;
  $("#phoneCard").hidden = false;
}

if (STATIC) {
  const refresh = () => showJudge(live.apiKey ? "Claude" : "Rules", live.apiKey ? "Claude judges each window" : "Rule-based scoring; add a Claude key in settings");
  refresh();
  $("#keyRow").hidden = false;
  $("#apiKey").value = live.apiKey;
  $("#apiKey").addEventListener("change", () => {
    const v = $("#apiKey").value.trim();
    live.apiKey = v;
    store.set("facial-apikey", v || null);
    refresh();
  });
  $("#whisperBox").closest("label").hidden = true;
  showPhoneQr(location.href.split("#")[0]);
} else {
  api("api/info").then((info) => {
    showJudge(info.judge === "claude" ? "Claude" : "Rules", info.judge === "claude" ? info.model : "Rule-based scoring (no Claude)");
    $("#whisperBox").disabled = !info.whisper;
    if (["localhost", "127.0.0.1", "[::1]"].includes(location.hostname)) showPhoneQr(info.phone_url);
  }).catch((err) => toast(err.message));
}

// ---------------------------------------------------------------------------------- live settings
initSeg($('.seg[data-setting="lang"]'), live.lang, (v) => save("lang", v));
for (const t of document.querySelectorAll(".toggle")) {
  const k = t.dataset.setting;
  t.setAttribute("aria-pressed", String(live[k]));
  t.addEventListener("click", () => {
    const on = t.getAttribute("aria-pressed") !== "true";
    t.setAttribute("aria-pressed", String(on));
    save(k, on);
  });
}
$("#windowSec").value = live.windowSec;
const syncWindow = () => { $("#windowOut").textContent = `${$("#windowSec").value}s`; save("windowSec", Number($("#windowSec").value)); };
$("#windowSec").addEventListener("input", syncWindow);
syncWindow();
$("#context").value = live.context;
$("#context").addEventListener("change", () => save("context", $("#context").value.trim()));
$("#settingsBtn").addEventListener("click", () => {
  const open = $("#settings").classList.toggle("open");
  $("#settingsBtn").setAttribute("aria-expanded", String(open));
});

function toast(msg) { $("#liveStatus").textContent = msg === "tap-to-play" ? "" : msg || ""; }

// ---------------------------------------------------------------------------------- model loading
function setLoad(frac) {
  $("#loadBar").hidden = frac >= 1;
  $("#loadBar span").style.width = `${Math.round(frac * 100)}%`;
  $("#readyText").textContent = frac >= 1 ? "Ready when you are" : `Getting ready ${Math.round(frac * 100)}%`;
}
const tasksReady = loadTasks(setLoad).catch((err) => { toast(`Couldn't load the vision models: ${err.message}`); throw err; });

// ---------------------------------------------------------------------------------- live run
async function startLive() {
  const btn = $("#startBtn");
  const audioCtx = live.mic ? unlockedAudio() : undefined;
  btn.disabled = true;
  if (!window.isSecureContext || !navigator.mediaDevices) {
    toast("The camera needs an https:// link");
    btn.disabled = false;
    return;
  }
  $("#windowLog").innerHTML = "";
  engine = new LiveEngine({
    canvas: $("#stage"), api, mode: CONFIG.mode, onStatus: toast,
    onWindow: ({ window: win, judgment }) => $("#windowLog").prepend(windowItem(win, judgment)),
  });
  try {
    await engine.start({ ...live, audioCtx });
    document.body.classList.add("running");
    $("#settings").classList.remove("open");
    $("#settingsBtn").setAttribute("aria-expanded", "false");
    btn.setAttribute("aria-label", "Stop");
    $(".rec-label").textContent = "Stop";
  } catch (err) {
    console.error(err);
    toast(err.name === "NotAllowedError" ? "Camera permission denied" : `Couldn't start: ${err.message}`);
    await engine.stop().catch(() => {});
    audioCtx?.close().catch(() => {});
  } finally {
    btn.disabled = false;
  }
}

async function stopLive() {
  await engine.stop();
  engine.opts.audioCtx?.close().catch(() => {});
  document.body.classList.remove("running");
  $("#startBtn").setAttribute("aria-label", "Start");
  $(".rec-label").textContent = "Start";
  toast("");
}

$("#startBtn").addEventListener("click", () => (engine && engine.running ? stopLive() : startLive()));
$("#flipBtn").addEventListener("click", async () => {
  save("facing", live.facing === "user" ? "environment" : "user");
  if (engine && engine.running) { await stopLive(); await startLive(); }
});

// ---------------------------------------------------------------------------------- video file
const STAGES = { prepare: "Preparing", perception: "Tracking", judge: "Judging", render: "Drawing" };
let fileLang = live.lang;
initSeg($('.seg[data-field="lang"]'), fileLang, (v) => { fileLang = v; $('input[name="lang"]').value = v; });

const drop = $("#drop");
function showVideoName() {
  const f = $("#videoInput").files[0];
  drop.classList.toggle("has-file", Boolean(f));
  $("#dropText").textContent = f ? f.name : "Drop a video";
  $("#dropSub").textContent = f ? `${(f.size / 1e6).toFixed(1)} MB` : "or tap to choose";
}
$("#videoInput").addEventListener("change", showVideoName);
for (const ev of ["dragenter", "dragover"]) drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("drag"); });
for (const ev of ["dragleave", "drop"]) drop.addEventListener(ev, () => drop.classList.remove("drag"));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  if (e.dataTransfer.files.length) { $("#videoInput").files = e.dataTransfer.files; showVideoName(); }
});
$("#srtInput").addEventListener("change", () => {
  const f = $("#srtInput").files[0];
  $("#srtChip").classList.toggle("has-file", Boolean(f));
  $("#srtText").textContent = f ? f.name : "Subtitles";
});

function setBar(sel, frac) {
  const pct = Math.round(frac * 100);
  $(`${sel} span`).style.width = `${pct}%`;
  $(sel).setAttribute("aria-valuenow", String(pct));
}

function show(id) {
  for (const s of ["#jobForm", "#jobProgress", "#player", "#jobResult"]) $(s).hidden = s !== id;
}

$("#jobForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!$("#videoInput").files.length) {
    drop.classList.remove("shake");
    void drop.offsetWidth; // restart the animation
    drop.classList.add("shake");
    return;
  }
  if (STATIC) return analyzeInBrowser(e.target, unlockedAudio());
  const form = new FormData(e.target);
  for (const k of ["start", "end"]) if (!form.get(k)) form.delete(k);
  if (!form.get("srt") || !form.get("srt").size) form.delete("srt");
  show("#jobProgress");
  $("#jobLog").textContent = "";
  setBar("#jobBar", 0);
  try {
    const job = await upload(form, (f) => { setBar("#jobBar", f * 0.05); $("#jobStage").textContent = `Uploading ${Math.round(f * 100)}%`; });
    await pollJob(job.id);
  } catch (err) {
    show("#jobForm");
    $("#dropSub").textContent = `Failed: ${err.message}`;
  }
});

// Static site: play the file through the same engine, judged window by window, and record it.
async function analyzeInBrowser(form, audioCtx) {
  const data = new FormData(form);
  const file = $("#videoInput").files[0];
  const srt = $("#srtInput").files[0];
  const segments = srt ? parseSubtitles(await srt.text()) : [];
  show("#player");
  $("#playerLog").innerHTML = "";
  $("#playerStatus").textContent = "Loading…";
  await tasksReady.catch(() => {});
  fileEngine = new LiveEngine({
    canvas: $("#playerCanvas"), api, mode: CONFIG.mode,
    onStatus: (m) => {
      $("#tapPlay").hidden = m !== "tap-to-play";
      $("#playerStatus").textContent = m === "tap-to-play" ? "" : m || "";
    },
    onWindow: ({ window: win, judgment }) => $("#playerLog").prepend(windowItem(win, judgment)),
    onEnd: ({ blob, results }) => showLocalResult(blob, results, file.name),
  });
  try {
    await fileEngine.start({
      file, segments, lang: fileLang, context: String(data.get("context") || "").trim(),
      windowSec: Number(data.get("window") || 5), skeleton: data.get("skeleton") === "true",
      apiKey: live.apiKey, audioCtx, speech: false,
    });
    trackPlayback();
  } catch (err) {
    console.error(err);
    await fileEngine.stop().catch(() => {});
    show("#jobForm");
    $("#dropSub").textContent = err.message;
  }
}

function trackPlayback() {
  const tick = () => {
    if (!fileEngine || !fileEngine.running) return;
    const v = fileEngine.video;
    if (v.duration) setBar("#playBar", v.currentTime / v.duration);
    $("#pauseBtn use").setAttribute("href", v.paused && !fileEngine.waiting ? "#i-play" : "#i-pause");
    requestAnimationFrame(tick);
  };
  tick();
}

async function stopFile() {
  await fileEngine.stop();
  fileEngine.opts.audioCtx?.close().catch(() => {});
  show("#jobForm");
}

$("#pauseBtn").addEventListener("click", () => fileEngine && fileEngine.togglePause());
$("#stopBtn").addEventListener("click", () => fileEngine && fileEngine.running && fileEngine.finish());
$("#tapPlay").addEventListener("click", () => fileEngine && fileEngine.play());

let lastUrls = [];
function showLocalResult(blob, results, name) {
  fileEngine?.opts.audioCtx?.close().catch(() => {});
  for (const u of lastUrls) URL.revokeObjectURL(u);
  lastUrls = [];
  show("#jobResult");
  const base = name.replace(/\.[^.]+$/, "") || "video";
  const video = $("#resultVideo");
  if (blob) {
    const url = URL.createObjectURL(blob);
    lastUrls.push(url);
    video.src = url;
    $("#downloadVideo").href = url;
    $("#downloadVideo").download = `${base}_facial.${blob.type.includes("mp4") ? "mp4" : "webm"}`;
  }
  $("#resultVideo").closest(".screen").hidden = !blob;
  $("#downloadVideo").hidden = !blob;
  const json = new Blob([JSON.stringify({ video: name, windows: results.map((r) => ({ ...r.window, judgment: r.judgment })) }, null, 2)],
    { type: "application/json" });
  const jurl = URL.createObjectURL(json);
  lastUrls.push(jurl);
  $("#downloadJson").href = jurl;
  $("#downloadJson").download = `${base}_facial.json`;
  fillWindows(results.map((r) => ({ ...r.window, judgment: r.judgment })), video);
}

function fillWindows(windows, video) {
  const list = $("#resultWindows");
  list.innerHTML = "";
  for (const w of windows) {
    const li = windowItem(w, w.judgment);
    li.tabIndex = 0;
    const seek = () => { video.currentTime = w.start; video.play().catch(() => {}); };
    li.addEventListener("click", seek);
    li.addEventListener("keydown", (e) => { if (e.key === "Enter") seek(); });
    list.append(li);
  }
}

// Self-hosted server: upload, process with the offline pipeline, show the rendered video.
function upload(form, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", new URL("api/jobs", document.baseURI));
    xhr.setRequestHeader("X-Facial-Key", key);
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) onProgress(e.loaded / e.total); };
    xhr.onload = () => {
      let data = {};
      try { data = JSON.parse(xhr.responseText); } catch { /* not JSON */ }
      if (xhr.status < 300) resolve(data);
      else reject(new Error(data.detail ? JSON.stringify(data.detail) : xhr.statusText));
    };
    xhr.onerror = () => reject(new Error("Upload failed"));
    xhr.send(form);
  });
}

async function pollJob(id) {
  for (;;) {
    const job = await api(`api/jobs/${id}`);
    setBar("#jobBar", job.progress);
    $("#jobStage").textContent = job.status === "queued" ? "Waiting…" : `${STAGES[job.stage] || "Working"} ${Math.round(job.progress * 100)}%`;
    $("#jobLog").textContent = job.log.join("\n");
    if (job.status === "done") return showServerResult(job);
    if (job.status === "error") throw new Error(job.error);
    await new Promise((r) => setTimeout(r, 1000));
  }
}

async function showServerResult(job) {
  const q = `?k=${encodeURIComponent(key)}`;
  show("#jobResult");
  const video = $("#resultVideo");
  video.closest(".screen").hidden = false;
  $("#downloadVideo").hidden = false;
  video.src = job.video.replace(/^\//, "") + q;
  $("#downloadVideo").href = video.src;
  $("#downloadJson").href = job.analysis.replace(/^\//, "") + q;
  const analysis = await api(job.analysis.replace(/^\//, ""));
  fillWindows(analysis.windows, video);
}

$("#newJob").addEventListener("click", () => {
  $("#resultVideo").removeAttribute("src");
  $("#jobForm").reset();
  $('input[name="lang"]').value = fileLang;
  showVideoName();
  $("#srtInput").dispatchEvent(new Event("change"));
  show("#jobForm");
});
