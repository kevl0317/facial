import json

from facial.jev import JEV_QUESTIONS, JevClient, describe, parse_decision
from facial.judge import WRITE_SCHEMA, ClaudeJudge, JevJudge, judge_or_fallback, make_judge
from facial.render import hand_label

from .test_judge import FakeMessages, _feats


def _window():
    f = _feats(2)
    f.update(subtitle="We made a decision a long time ago.", scene={"shot": "medium", "faces": 1, "cuts": 0},
             voice={"voiced_frac": 0.8, "loudness_rel_db": 3.1, "pitch_rel_st": -0.4, "pitch_var_st": 2.9,
                    "pauses": 1, "fillers": 0, "speech_rate_wps": 2.6},
             gesture={"hands_visible": 0.9, "two_hands": 0.5, "energy": 1.1, "beats": 2, "hand_to_face": 0.0,
                      "dominant": {"hand": "right", "side": "right", "shape": "open_palm", "facing": "palm_up",
                                   "axis": "horizontal", "score": 0.8, "share": 0.7},
                      "face": {"smile": 0.4, "blink_per_min": 12, "vs_baseline": {"smile": 0.2, "brow_furrow": 0.0}},
                      "head": {"yaw_std": 2.0, "pitch_std": 1.5, "nods": 0, "gaze_away": 0.05}})
    return f


def jev_reply(intent="explaining", p=0.82, levels=(3, 2.4, 1, 3)):
    scales = dict(zip(("confidence", "focus", "tension", "valence"), levels))
    answers = {k: {"type": "score", "score": v, "confidence": 0.6} for k, v in scales.items()}
    answers["intent"] = {"type": "choice", "choice": intent, "confidence": p,
                         "probabilities": {intent: p, "stating": round(1 - p, 2)}}
    return 200, {"model": "jev-1.13.0", "answers": answers, "usage": {"input_tokens": 300}}


def _jev(*replies):
    calls = []

    def post(url, headers, body, timeout):
        calls.append({"url": url, "headers": headers, "body": json.loads(json.dumps(body))})
        return replies[min(len(calls), len(replies)) - 1]

    client = JevClient("typesafe", api_key="ts-key", log=lambda m: None)
    client.post, client.sleep = post, (lambda s: None)
    return client, calls


def _claude():
    judge = ClaudeJudge(api_key="k", log=lambda m: None)
    judge.client = type("C", (), {"beta": type("B", (), {"messages": FakeMessages()})()})()
    return judge


def test_jev_reads_the_window_in_words():
    text = describe(_window())
    assert "Right hand · open palm (palm up) for most of the window" in text
    assert "a bit louder than usual" in text and "lively intonation" in text and "one pause" in text
    assert "smiling more than usual" in text and "head steady" in text
    assert 'Said: "We made a decision a long time ago."' in text
    assert "3.1" not in text and "0.7" not in text  # words, not numbers


def test_jev_decides_and_the_llm_only_writes():
    jev, calls = _jev(jev_reply())
    claude = _claude()
    judge = JevJudge(jev, claude, "en", lambda h: hand_label(h, "en"), log=lambda m: None)
    result = judge.judge(_window(), {"live": True})

    body = calls[0]["body"]
    assert calls[0]["url"] == "https://api.typesafe.ai/v1/systemone"
    assert calls[0]["headers"]["Authorization"] == "Bearer ts-key"
    assert body["model"] == "jev-latest" and body["questions"] == JEV_QUESTIONS and "Voice:" in body["state"]

    assert result["source"] == "jev" and result["writer"] == "claude"
    assert (result["confidence"], result["focus"], result["tension"], result["valence"]) == (0.75, 0.6, 0.25, 0.5)
    assert result["intent"] == "Explaining" and result["intent_certainty"] == 0.82
    assert result["reading"] == "reading 1"  # the words come from the LLM

    call = claude.client.beta.messages.calls[0]
    assert call["output_config"] == {"effort": "low", "format": {"type": "json_schema", "schema": WRITE_SCHEMA}}
    assert "Jev's decision for this window (settled): intent \"Explaining\"" in call["messages"][0]["content"]
    assert "Jev, a fast decision model" in call["system"]


def test_an_unsure_jev_hands_the_window_to_the_llm():
    jev, _ = _jev(jev_reply(intent="deflecting", p=0.3))
    claude = _claude()
    result = JevJudge(jev, claude, "en", None, log=lambda m: None).judge(_window())
    assert result["source"] == "claude" and "writer" not in result
    assert claude.client.beta.messages.calls[0]["output_config"]["format"]["schema"]["required"][-1] == "evidence"
    assert "Jev's decision" not in claude.client.beta.messages.calls[0]["messages"][0]["content"]


def test_jev_without_an_llm_uses_the_rule_templates_for_words():
    judge = make_judge("heuristic", jev={"route": "openrouter", "api_key": "or-key"},
                       hand_label=lambda h: hand_label(h, "en"), log=lambda m: None)
    judge.jev.post = lambda url, headers, body, timeout: jev_reply("stating", 0.9)
    result = judge.judge(_window())
    assert judge.jev.base_url == "https://openrouter.ai/api/v1" and judge.jev.model == "typesafe/jev-1.13"
    assert result["source"] == "jev" and "writer" not in result
    assert result["reading"] == "Right hand · open palm (palm up) — Stating a position"
    assert result["quote"] == "We made a decision a long time ago"


def test_no_jev_key_leaves_the_llm_in_charge(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    logs = []
    claude = _claude()
    judge = JevJudge(JevClient("typesafe", log=logs.append), claude, "en", None, log=logs.append)
    result, still = judge_or_fallback(judge, _window(), None, "en", None, logs.append)
    assert result["source"] == "claude" and still is judge and judge.jev is None
    assert any("TYPESAFE_API_KEY" in m for m in logs)


def test_a_rejected_jev_and_no_llm_falls_back_to_rules():
    jev, _ = _jev((401, {"error": {"message": "bad key"}}))
    judge = JevJudge(jev, None, "en", None, log=lambda m: None)
    result, still = judge_or_fallback(judge, _window(), None, "en", None, lambda m: None)
    assert result["source"] == "heuristic" and still is None


def test_decisions_can_come_from_probabilities_alone():
    data = {"answers": {k: {"type": "score", "probabilities": {"0": 0, "1": 0, "2": 0.5, "3": 0.5, "4": 0}}
                        for k in ("confidence", "focus", "tension", "valence")}}
    data["answers"]["intent"] = {"type": "choice", "probabilities": {"questioning": 0.7, "stating": 0.3}}
    d = parse_decision(data, "zh")
    assert d["confidence"] == 0.62 and d["valence"] == 0.25 and d["words"]["tension"] == "tense"
    assert d["intent"] == "提出问题" and d["certain"]
