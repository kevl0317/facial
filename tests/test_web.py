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
