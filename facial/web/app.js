import { LiveEngine, loadTasks } from "./live.js";

const $ = (sel) => document.querySelector(sel);
const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
};

// Access key from the URL the server printed (?k=...), remembered in this browser.
const params = new URLSearchParams(location.search);
let key = params.get("k") ?? store.get("facial-key") ?? "";
if (params.has("k")) store.set("facial-key", key);

function askKey() {
  const entered = prompt("Access key (the k=... part of the URL the server printed):", key);
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
    <div class="head"><span>W${win.window} · ${fmt(win.start)}–${fmt(win.end)}</span>
      <span>conf ${j.confidence.toFixed(2)} · focus ${j.focus.toFixed(2)} · tense ${j.tension.toFixed(2)}</span></div>
    <div><span class="intent">${esc(j.intent)}</span> — ${esc(j.reading)}</div>
    ${j.quote ? `<div class="quote">“${esc(j.quote)}”</div>` : ""}`;
  return li;
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
  $("#judgeBadge").textContent = info.judge === "claude" ? `Judge: Claude · ${info.model}` : "Judge: rules (no Claude)";
  $("#whisperBox").disabled = !info.whisper;
  if (!info.whisper) $("#whisperBox").parentElement.title = "Install faster-whisper on the server to enable";
  const onThisComputer = ["localhost", "127.0.0.1", "[::1]"].includes(location.hostname);
  if (info.phone_url && onThisComputer) {
    $("#phoneCard").hidden = false;
    $("#phoneUrl").textContent = info.phone_url;
    fetch(`/api/qr.svg?data=${encodeURIComponent(info.phone_url)}`, { headers: { "X-Facial-Key": key } })
      .then((r) => (r.ok ? r.blob() : null))
      .then((b) => { if (b) $("#phoneQr").src = URL.createObjectURL(b); else $("#phoneQr").hidden = true; });
  }
}).catch((err) => { $("#liveStatus").textContent = `Server: ${err.message}`; });

// ---------------------------------------------------------------------------------- live
const settings = ["facing", "lang", "context", "windowSec", "mic", "speech", "skeleton"];
for (const id of settings) {
  const el = $(`#${id}`);
  const saved = store.get(`facial-${id}`);
  if (saved !== null) { if (el.type === "checkbox") el.checked = saved === "1"; else el.value = saved; }
  el.addEventListener("change", () => store.set(`facial-${id}`, el.type === "checkbox" ? (el.checked ? "1" : "0") : el.value));
}
const syncWindowOut = () => { $("#windowOut").textContent = `${$("#windowSec").value} s`; };
$("#windowSec").addEventListener("input", syncWindowOut);
syncWindowOut();

$("#settingsBtn").addEventListener("click", () => {
  const open = $("#settings").classList.toggle("open");
  $("#settingsBtn").setAttribute("aria-expanded", String(open));
});

const liveStatus = (msg) => { $("#liveStatus").textContent = msg; };

async function startLive() {
  const btn = $("#startBtn");
  btn.disabled = true;
  if (!window.isSecureContext || !navigator.mediaDevices) {
    liveStatus("Camera access needs HTTPS (or localhost). Open the https:// URL the server printed.");
    btn.disabled = false;
    return;
  }
  $("#windowLog").innerHTML = "";
  engine = new LiveEngine({
    canvas: $("#stage"),
    api,
    onStatus: liveStatus,
    onWindow: ({ window: win, judgment }) => $("#windowLog").prepend(windowItem(win, judgment)),
  });
  try {
    await engine.start({
      facing: $("#facing").value, lang: $("#lang").value, context: $("#context").value.trim(),
      windowSec: Number($("#windowSec").value), mic: $("#mic").checked, speech: $("#speech").checked,
      skeleton: $("#skeleton").checked,
    });
    document.body.classList.add("running");
    $("#settings").classList.remove("open");
    btn.textContent = "Stop";
    btn.classList.add("stop");
  } catch (err) {
    console.error(err);
    liveStatus(err.name === "NotAllowedError" ? "Camera/microphone permission was denied." : `Could not start: ${err.message}`);
    await engine.stop().catch(() => {});
  } finally {
    btn.disabled = false;
  }
}

async function stopLive() {
  await engine.stop();
  document.body.classList.remove("running");
  $("#startBtn").textContent = "Start";
  $("#startBtn").classList.remove("stop");
}

$("#startBtn").addEventListener("click", () => (engine && engine.running ? stopLive() : startLive()));

// Warm MediaPipe up in the background so Start is quick.
const idle = window.requestIdleCallback || ((fn) => setTimeout(fn, 800));
idle(() => loadTasks().catch((err) => liveStatus(`MediaPipe failed to load: ${err.message}`)));

// ---------------------------------------------------------------------------------- video file
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
  const form = new FormData(e.target);
  for (const k of ["start", "end"]) if (!form.get(k)) form.delete(k);
  if (!form.get("srt") || !form.get("srt").size) form.delete("srt");
  $("#jobForm").hidden = true;
  $("#jobProgress").hidden = false;
  $("#jobTitle").textContent = "Uploading…";
  $("#jobLog").textContent = "";
  try {
    const job = await upload(form, (f) => { setBar(f * 0.05); $("#jobStage").textContent = `Uploading ${Math.round(f * 100)}%`; });
    $("#jobTitle").textContent = "Processing…";
    await pollJob(job.id);
  } catch (err) {
    $("#jobTitle").textContent = "Failed";
    $("#jobStage").textContent = err.message;
    $("#jobForm").hidden = false;
  }
});

async function pollJob(id) {
  for (;;) {
    const job = await api(`/api/jobs/${id}`);
    setBar(job.progress);
    $("#jobStage").textContent = job.status === "queued" ? "Waiting for the previous job…" : `${job.stage} · ${Math.round(job.progress * 100)}%`;
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
  $("#jobForm").hidden = false;
});
