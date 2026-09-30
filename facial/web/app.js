import { handSvg } from "./hand.js";
import { playIntro } from "./intro.js";
import { LiveEngine, loadTasks } from "./live.js";

const $ = (sel) => document.querySelector(sel);
const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
};

playIntro($("#intro"));
for (const el of document.querySelectorAll("[data-mascot]")) {
  el.innerHTML = handSvg({ size: Number(el.dataset.mascot), cls: el.dataset.mascotClass || "" });
}

// Access key from the URL the server printed (?k=...), remembered in this browser.
const params = new URLSearchParams(location.search);
let key = params.get("k") ?? store.get("facial-key") ?? "";
if (params.has("k")) store.set("facial-key", key);

function askKey() {
  const entered = prompt("Access key (the k=... part of the server's link):", key);
  if (entered !== null) { key = entered.trim(); store.set("facial-key", key); }
}

async function api(path, { method = "GET", body } = {}) {
  const headers = { "X-Facial-Key": key };
  let payload;
  if (body !== undefined) { headers["Content-Type"] = "application/json"; payload = JSON.stringify(body); }
  const res = await fetch(path, { method, headers, body: payload });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch { /* not JSON */ }
    if (res.status === 401) askKey();
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

// Segmented controls: `data-setting` (live settings) or `data-field` (form field).
function initSeg(seg, value, onChange) {
  const set = (v) => {
    for (const b of seg.querySelectorAll("button")) b.setAttribute("aria-checked", String(b.dataset.value === v));
    onChange(v);
  };
  seg.addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) set(b.dataset.value); });
  set(value);
}

// ---------------------------------------------------------------------------------- tabs
let engine = null;
for (const btn of document.querySelectorAll(".tabs button")) {
  btn.addEventListener("click", async () => {
    const tab = btn.dataset.tab;
    for (const b of document.querySelectorAll(".tabs button")) b.setAttribute("aria-selected", String(b === btn));
    $("#tab-live").hidden = tab !== "live";
    $("#tab-file").hidden = tab !== "file";
    if (tab !== "live" && engine && engine.running) await stopLive();
  });
}

// ---------------------------------------------------------------------------------- info
api("/api/info").then((info) => {
  $("#judgeChip").hidden = false;
  $("#judgeChip span").textContent = info.judge === "claude" ? "Claude" : "Rules";
  $("#judgeChip").title = info.judge === "claude" ? info.model : "Rule-based scoring (no Claude)";
  $("#whisperBox").disabled = !info.whisper;
  if (!info.whisper) $("#whisperBox").parentElement.title = "Install faster-whisper on the server";
  const onThisComputer = ["localhost", "127.0.0.1", "[::1]"].includes(location.hostname);
  if (info.phone_url && onThisComputer) {
    fetch(`/api/qr.svg?data=${encodeURIComponent(info.phone_url)}`, { headers: { "X-Facial-Key": key } })
      .then((r) => (r.ok ? r.blob() : null))
      .then((b) => {
        if (!b) return;
        $("#phoneQr").src = URL.createObjectURL(b);
        $("#phoneCard").title = info.phone_url;
        $("#phoneCard").hidden = false;
      });
  }
}).catch((err) => { toast(err.message); });

// ---------------------------------------------------------------------------------- live settings
const live = {
  lang: store.get("facial-lang") || "en",
  facing: store.get("facial-facing") || "user",
  mic: store.get("facial-mic") !== "0",
  speech: store.get("facial-speech") !== "0",
  skeleton: store.get("facial-skeleton") === "1",
  windowSec: Number(store.get("facial-windowSec") || 5),
  context: store.get("facial-context") || "",
};
const save = (k, v) => { live[k] = v; store.set(`facial-${k}`, typeof v === "boolean" ? (v ? "1" : "0") : String(v)); };

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

function toast(msg) { $("#liveStatus").textContent = msg || ""; }

// ---------------------------------------------------------------------------------- live run
async function startLive() {
  const btn = $("#startBtn");
  btn.disabled = true;
  if (!window.isSecureContext || !navigator.mediaDevices) {
    toast("Open the https:// link to use the camera");
    btn.disabled = false;
    return;
  }
  $("#windowLog").innerHTML = "";
  engine = new LiveEngine({
    canvas: $("#stage"),
    api,
    onStatus: toast,
    onWindow: ({ window: win, judgment }) => $("#windowLog").prepend(windowItem(win, judgment)),
  });
  try {
    await engine.start({ ...live });
    document.body.classList.add("running");
    $("#settings").classList.remove("open");
    $("#settingsBtn").setAttribute("aria-expanded", "false");
    btn.setAttribute("aria-label", "Stop");
    $(".rec-label").textContent = "Stop";
  } catch (err) {
    console.error(err);
    toast(err.name === "NotAllowedError" ? "Camera permission denied" : `Couldn't start: ${err.message}`);
    await engine.stop().catch(() => {});
  } finally {
    btn.disabled = false;
  }
}

async function stopLive() {
  await engine.stop();
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

// Warm MediaPipe up in the background so Start is quick.
const idle = window.requestIdleCallback || ((fn) => setTimeout(fn, 800));
idle(() => loadTasks().catch((err) => toast(`MediaPipe failed to load: ${err.message}`)));

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

function upload(form, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/jobs");
    xhr.setRequestHeader("X-Facial-Key", key);
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) onProgress(e.loaded / e.total); };
    xhr.onload = () => {
      let data = {};
      try { data = JSON.parse(xhr.responseText); } catch { /* not JSON */ }
      if (xhr.status < 300) resolve(data);
      else {
        if (xhr.status === 401) askKey();
        reject(new Error(data.detail ? JSON.stringify(data.detail) : xhr.statusText));
      }
    };
    xhr.onerror = () => reject(new Error("Upload failed"));
    xhr.send(form);
  });
}

function setBar(frac) {
  const pct = Math.round(frac * 100);
  $("#jobBar span").style.width = `${pct}%`;
  $("#jobBar").setAttribute("aria-valuenow", String(pct));
}

$("#jobForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!$("#videoInput").files.length) {
    drop.classList.remove("shake");
    void drop.offsetWidth; // restart the animation
    drop.classList.add("shake");
    return;
  }
  const form = new FormData(e.target);
  for (const k of ["start", "end"]) if (!form.get(k)) form.delete(k);
  if (!form.get("srt") || !form.get("srt").size) form.delete("srt");
  $("#jobForm").hidden = true;
  $("#jobProgress").hidden = false;
  $("#jobLog").textContent = "";
  setBar(0);
  try {
    const job = await upload(form, (f) => { setBar(f * 0.05); $("#jobStage").textContent = `Uploading ${Math.round(f * 100)}%`; });
    await pollJob(job.id);
  } catch (err) {
    $("#jobProgress").hidden = true;
    $("#jobForm").hidden = false;
    $("#dropSub").textContent = `Failed: ${err.message}`;
  }
});

async function pollJob(id) {
  for (;;) {
    const job = await api(`/api/jobs/${id}`);
    setBar(job.progress);
    $("#jobStage").textContent = job.status === "queued" ? "Waiting…" : `${STAGES[job.stage] || "Working"} ${Math.round(job.progress * 100)}%`;
    $("#jobLog").textContent = job.log.join("\n");
    if (job.status === "done") return showResult(job);
    if (job.status === "error") throw new Error(job.error);
    await new Promise((r) => setTimeout(r, 1000));
  }
}

async function showResult(job) {
  const q = `?k=${encodeURIComponent(key)}`;
  $("#jobProgress").hidden = true;
  $("#jobResult").hidden = false;
  const video = $("#resultVideo");
  video.src = job.video + q;
  $("#downloadVideo").href = job.video + q;
  $("#downloadJson").href = job.analysis + q;
  const analysis = await api(job.analysis);
  const list = $("#resultWindows");
  list.innerHTML = "";
  for (const w of analysis.windows) {
    const li = windowItem(w, w.judgment);
    li.tabIndex = 0;
    const seek = () => { video.currentTime = w.start; video.play().catch(() => {}); };
    li.addEventListener("click", seek);
    li.addEventListener("keydown", (e) => { if (e.key === "Enter") seek(); });
    list.append(li);
  }
}

$("#newJob").addEventListener("click", () => {
  $("#jobResult").hidden = true;
  $("#resultVideo").removeAttribute("src");
  $("#jobForm").reset();
  $('input[name="lang"]').value = fileLang;
  showVideoName();
  $("#srtInput").dispatchEvent(new Event("change"));
  $("#jobForm").hidden = false;
});
