"""The browser port (facial/web/*.js) must agree with the Python pipeline."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from facial.audio import analyze_samples, voice_features
from facial.features import add_hand_motion, clip_baseline, window_features
from facial.judge import heuristic_judgment
from facial.render import hand_label
from facial.transcript import Segment
from facial.webbuild import PROMPT_JS, prompt_js

from .test_core import _samples

WEB = Path(__file__).resolve().parent.parent / "facial" / "web"
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")


def _node(script: str, payload) -> dict:
    out = subprocess.run([NODE, "--input-type=module", "-e", script], input=json.dumps(payload),
                         capture_output=True, text=True, cwd=WEB, check=True)
    return json.loads(out.stdout)


def _close(a, b, path="", tol=0.011):
    if isinstance(a, dict):
        assert set(a) == set(b), f"{path}: keys {sorted(a)} != {sorted(b)}"
        for k in a:
            _close(a[k], b[k], f"{path}.{k}", tol)
    elif isinstance(a, list):
        assert len(a) == len(b), f"{path}: {a} != {b}"
        for i, (x, y) in enumerate(zip(a, b)):
            _close(x, y, f"{path}[{i}]", tol)
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        assert abs(a - b) <= tol * max(1, abs(a)), f"{path}: {a} != {b}"
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def test_prompt_js_is_generated_from_judge_py():
    assert PROMPT_JS.read_text(encoding="utf-8") == prompt_js(), "run: python -c 'from facial.webbuild import write_prompt_js; write_prompt_js()'"


@needs_node
def test_window_features_and_rules_match_python():
    samples = _samples(n=60, fps=10.0)
    add_hand_motion(samples, aspect=16 / 9)
    base = clip_baseline(samples)
    segments = [Segment(0.5, 3.5, "Is this the right call?"), Segment(3.6, 5.5, "We think so.")]
    py = window_features(0, 2, (0.0, 6.0), samples, None, segments, base)
    py_judgment = heuristic_judgment(py, "en", lambda h: hand_label(h, "en"))

    js = _node("""
      import { readFileSync } from "node:fs";
      import { windowFeatures, clipBaseline } from "./features.js";
      import { heuristicJudgment } from "./judge.js";
      import { handLabel } from "./hud.js";
      const { samples, segments } = JSON.parse(readFileSync(0, "utf8"));
      const feats = windowFeatures(0, 2, [0, 6], samples, null, segments, clipBaseline(samples));
      const judgment = heuristicJudgment(feats, "en", (h) => handLabel(h, "en"));
      console.log(JSON.stringify({ feats, judgment }));
    """, {"samples": samples, "segments": [s.__dict__ for s in segments]})

    _close(py, js["feats"])
    _close(py_judgment, js["judgment"], tol=0.02)


@needs_node
def test_voice_measurements_match_python():
    sr = 16000
    t = np.arange(8 * sr) / sr
    f0 = 140 + 20 * np.sin(2 * np.pi * 0.4 * t)
    tone = sum(np.sin(k * 2 * np.pi * np.cumsum(f0) / sr) / k for k in range(1, 6))
    gate = ((t % 2.0) < 1.5).astype(float)  # 1.5 s of voice, 0.5 s pauses
    y = (0.2 * tone * gate + 0.001 * np.random.default_rng(0).standard_normal(len(t))).astype(np.float32)

    py = voice_features(analyze_samples(y, sr), 2.0, 7.0)
    js = _node("""
      import { readFileSync } from "node:fs";
      import { VoiceAnalyzer } from "./voice.js";
      const pcm = Float32Array.from(JSON.parse(readFileSync(0, "utf8")));
      const v = new VoiceAnalyzer();
      for (let i = 0; i < pcm.length; i += 4096) v.push(pcm.subarray(i, i + 4096), 0);
      console.log(JSON.stringify(v.features(2, 7)));
    """, y.tolist())

    assert abs(py["voiced_frac"] - js["voiced_frac"]) <= 0.05
    assert py["pauses"] == js["pauses"]
    assert abs(py["loudness_rel_db"] - js["loudness_rel_db"]) <= 1.0
    assert abs(py["pitch_rel_st"] - js["pitch_rel_st"]) <= 0.5
    assert abs(py["pitch_var_st"] - js["pitch_var_st"]) <= 0.5


SPEECH_HARNESS = """
  import { Speech } from "./live.js";
  const MODE = process.argv[1];
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const result = (text, isFinal) => ({ resultIndex: 0, results: [Object.assign([{ transcript: text }], { isFinal })] });
  class FakeRecognition {  // behaves like a phone browser: ends after every sentence
    static count = 0;
    constructor() { this.n = ++FakeRecognition.count; }
    start() {
      const fail = (error) => setTimeout(() => { this.onerror?.({ error }); this.onend?.(); }, 5);
      if (MODE === "refuse" && this.n === 2) return fail("not-allowed");
      if (MODE === "throw" && this.n === 2) throw new Error("InvalidStateError");
      if (MODE === "busy") return fail("audio-capture");
      setTimeout(() => this.onresult?.(result(`sentence ${this.n} so`, false)), 10);
      if (MODE === "cutoff" && this.n % 2 === 0) { setTimeout(() => this.onend?.(), 20); return; }
      setTimeout(() => { this.onresult?.(result(`sentence ${this.n}`, true)); this.onend?.(); }, 20);
    }
    abort() {}
  }
  globalThis.window = { SpeechRecognition: FakeRecognition };
  let t = 0;
  const events = [];
  const sp = new Speech("en", () => (t += 0.05), { onNeedsTap: () => events.push("tap"), onGiveUp: (m) => events.push(m) });
  sp.start();
  await sleep(1500);
  if (MODE === "refuse") { sp.resume(); await sleep(500); }
  sp.stop();
  console.log(JSON.stringify({ texts: sp.take(0, 1e9).map((s) => s.text), events }));
"""


def _speech(mode: str) -> dict:
    out = subprocess.run([NODE, "--input-type=module", "-e", SPEECH_HARNESS, mode], capture_output=True, text=True,
                         cwd=WEB, check=True, timeout=30)
    return json.loads(out.stdout)


@needs_node
def test_subtitles_keep_going_after_each_sentence():
    res = _speech("phone")
    assert len(res["texts"]) >= 5 and res["texts"][:2] == ["sentence 1", "sentence 2"]


@needs_node
def test_subtitles_keep_unfinished_sentences():
    texts = _speech("cutoff")["texts"]
    assert "sentence 2 so" in texts and "sentence 3" in texts  # a sentence cut off mid-way is still kept


@needs_node
def test_subtitles_retry_after_a_failed_restart():
    assert len(_speech("throw")["texts"]) >= 2


@needs_node
def test_subtitles_ask_for_a_tap_when_the_browser_refuses_a_restart():
    res = _speech("refuse")
    assert res["events"] == ["tap"]
    assert res["texts"][0] == "sentence 1" and len(res["texts"]) >= 2  # resumes after the tap


@needs_node
def test_subtitles_explain_a_microphone_clash():
    res = _speech("busy")
    assert res["texts"] == [] and len(res["events"]) == 1 and "Voice" in res["events"][0]


BROWSER_JUDGE = """
  import { BrowserJudge, listModels } from "./judge.js";
  const reply = { reading: "Palm up", quote: "", confidence: 0.7, focus: 0.6, tension: 0.2, intent: "Explaining",
                  intent_certainty: 0.6, valence: 0.3, evidence: ["palm"] };
  const feats = (i) => ({ window: i, of: null, start: 0, end: 5, scene: {}, speaker: { state: "target" },
                          subtitle: "", voice: {}, gesture: {} });
  const calls = [];
  const fakeFetch = (mode) => async (url, init) => {
    calls.push({ url, headers: init?.headers || {}, body: init?.body ? JSON.parse(init.body) : null });
    if (mode === "cors") throw new TypeError("Failed to fetch");
    if (mode === "badkey") return new Response(JSON.stringify({ error: { message: "bad key" } }), { status: 401 });
    if (url.endsWith("/models")) {
      return Response.json({ data: [{ id: "deepseek-chat" }, { id: "text-embedding-3" }, { id: "deepseek-reasoner" }] });
    }
    return Response.json({ choices: [{ message: { content: "```json\\n" + JSON.stringify(reply) + "\\n```" } }] });
  };
  const out = {};
  for (const mode of ["ok", "cors", "badkey"]) {
    const notices = [];
    const j = new BrowserJudge({ provider: "deepseek", apiKey: "sk-1", lang: "en", context: "", fetchImpl: fakeFetch(mode),
                                 onNotice: (m) => notices.push(m) });
    const name = j.name;
    const sources = [(await j.judge(feats(1), { live: true })).source, (await j.judge(feats(2))).source];
    out[mode] = { name, sources, notices, after: j.name };
  }
  out.calls = calls.slice(0, 2);
  out.noKey = new BrowserJudge({ provider: "openai", apiKey: "", lang: "en" }).name;
  out.models = await listModels({ provider: "deepseek", apiKey: "sk-1", fetchImpl: fakeFetch("ok") });
  console.log(JSON.stringify(out));
"""


@needs_node
def test_browser_judge_talks_to_other_providers_and_falls_back_to_rules():
    out = _node(BROWSER_JUDGE, None)
    assert out["ok"] == {"name": "deepseek", "sources": ["deepseek", "deepseek"], "notices": [], "after": "deepseek"}
    first, second = out["calls"]
    assert first["url"] == "https://api.deepseek.com/chat/completions"
    assert first["headers"]["Authorization"] == "Bearer sk-1"
    assert first["body"]["response_format"] == {"type": "json_object"}
    assert [m["role"] for m in second["body"]["messages"]] == ["system", "user", "assistant", "user"]
    # A provider that refuses browsers (CORS) or a wrong key: say so once, then use the rules.
    assert out["cors"]["sources"] == ["heuristic", "heuristic"] and out["cors"]["after"] == "heuristic"
    assert out["cors"]["notices"] == ["DeepSeek can't be reached from this page, using rules"]
    assert out["badkey"]["notices"] == ["DeepSeek key rejected, using rules"]
    assert out["noKey"] == "heuristic"
    assert out["models"] == ["deepseek-chat", "deepseek-reasoner"]
