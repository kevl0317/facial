import http.client
import json
from types import SimpleNamespace

import pytest

from facial.judge import JSON_RULE, ChatJudge, ClaudeJudge, JudgeUnavailable, judge_or_fallback, make_judge
from facial.providers import judge_tag


class FakeMessages:
    """Stands in for client.beta.messages and records each request's messages."""

    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        n = len(self.calls)
        body = {"reading": f"reading {n}", "quote": "", "confidence": 0.5 + n / 100, "focus": 0.6,
                "tension": 0.2, "intent": "Stating a position", "intent_certainty": 0.7, "valence": 0.1,
                "evidence": ["x"]}
        return SimpleNamespace(
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=0,
                                  cache_creation_input_tokens=0),
            content=[SimpleNamespace(type="text", text=json.dumps(body))],
        )


def _feats(i):
    return {"window": i, "of": None, "start": 5.0 * (i - 1), "end": 5.0 * i, "scene": {}, "speaker": {"state": "target"},
            "subtitle": "", "voice": {}, "gesture": {}}


def test_live_conversation_rolls_over_with_summary():
    judge = ClaudeJudge(max_turns=2, log=lambda m: None)
    fake = FakeMessages()
    judge.client = SimpleNamespace(beta=SimpleNamespace(messages=fake))

    results = [judge.judge(_feats(i), {"live": True, "baseline": {"yaw": 1.0}} if i == 1 else None)
               for i in range(1, 4)]
    assert [r["source"] for r in results] == ["claude"] * 3

    first, second, third = (c["messages"] for c in fake.calls)
    assert "Live camera session" in first[0]["content"] and "Window 1 (live)" in first[0]["content"]
    assert [m["role"] for m in second] == ["user", "assistant", "user"]  # history replayed append-only
    # After max_turns the conversation restarts with the intro and a summary of earlier windows.
    assert len(third) == 1
    assert "Earlier windows, summarised" in third[0]["content"]
    assert "W1:" in third[0]["content"] and "W2:" in third[0]["content"]
    assert fake.calls[0]["output_config"]["format"]["type"] == "json_schema"


# --------------------------------------------------------------------------- other providers
REPLY = {"reading": "Open palm lays out the facts", "quote": "", "confidence": 0.7, "focus": 0.6, "tension": 0.2,
         "intent": "Explaining", "intent_certainty": 0.6, "valence": 0.3, "evidence": ["open palm"]}


class FakeChat:
    """Stands in for the HTTP POST of ChatJudge; replies with queued (status, body) pairs."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, url, headers, body, timeout):
        self.calls.append({"url": url, "headers": headers, "body": json.loads(json.dumps(body))})
        status, content = self.replies.pop(0) if self.replies else (200, json.dumps(REPLY))
        if status != 200:
            return status, {"error": {"message": content}}
        return 200, {"choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": 100, "completion_tokens": 20}}


def _chat(pid, *replies, **kw):
    judge = ChatJudge(pid, log=kw.pop("log", lambda m: None), **kw)
    judge.post = FakeChat(*replies)
    judge.sleep = lambda s: None
    return judge


def test_openai_gets_a_strict_schema_and_the_conversation_so_far():
    judge = _chat("openai", api_key="sk-test", effort="low")
    first = judge.judge(_feats(1), {"live": True, "baseline": {}})
    second = judge.judge(_feats(2))
    assert first["source"] == second["source"] == "openai" and first["reading"] == REPLY["reading"]

    a, b = judge.post.calls
    assert a["url"] == "https://api.openai.com/v1/chat/completions"
    assert a["headers"]["Authorization"] == "Bearer sk-test"
    body = a["body"]
    assert body["model"] == "gpt-5-mini" and body["reasoning_effort"] == "low" and "max_completion_tokens" in body
    assert body["response_format"]["type"] == "json_schema" and body["response_format"]["json_schema"]["strict"]
    assert body["messages"][0]["role"] == "system" and body["messages"][0]["content"].endswith(JSON_RULE)
    assert [m["role"] for m in b["body"]["messages"]] == ["system", "user", "assistant", "user"]
    assert judge.usage["input"] == 200 and judge.usage["output"] == 40


def test_deepseek_uses_json_mode_and_tolerates_code_fences():
    fenced = "Sure!\n```json\n" + json.dumps(REPLY) + "\n```"
    judge = _chat("deepseek", (200, fenced), api_key="k", base_url="https://proxy.example/v1/")
    result = judge.judge(_feats(1), {"duration": 10.0})
    call = judge.post.calls[0]
    assert call["url"] == "https://proxy.example/v1/chat/completions"
    assert call["body"]["response_format"] == {"type": "json_object"} and "max_tokens" in call["body"]
    assert "reasoning_effort" not in call["body"] and result["intent"] == "Explaining"


def test_a_provider_that_rejects_json_mode_gets_a_plain_request():
    judge = _chat("groq", (400, "response_format is not supported"), api_key="k")
    assert judge.judge(_feats(1))["source"] == "groq"
    retry = judge.post.calls[1]["body"]
    assert set(retry) == {"model", "messages"}


def test_a_rejected_key_switches_to_rules_for_the_rest():
    logs = []
    judge = _chat("gemini", (401, "API key not valid"), api_key="bad", log=logs.append)
    result, judge_after = judge_or_fallback(judge, _feats(1), None, "en", None, logs.append)
    assert result["source"] == "heuristic" and judge_after is None
    assert any("Gemini request rejected (401: API key not valid)" in m for m in logs)


def test_a_missing_key_names_the_environment_variable(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    logs = []
    judge = _chat("deepseek", log=logs.append)
    result, judge_after = judge_or_fallback(judge, _feats(1), None, "en", None, logs.append)
    assert result["source"] == "heuristic" and judge_after is None and not judge.post.calls
    assert any("DEEPSEEK_API_KEY" in m for m in logs)


def test_busy_provider_falls_back_for_one_window_only():
    judge = _chat("mistral", *[(429, "slow down")] * 4, api_key="k")
    result, judge_after = judge_or_fallback(judge, _feats(1), None, "en", None, lambda m: None)
    assert result["source"] == "heuristic" and judge_after is judge and not judge.messages
    result, _ = judge_or_fallback(judge, _feats(2), None, "en", None, lambda m: None)
    assert result["source"] == "mistral"


def test_make_judge_covers_every_provider():
    assert make_judge("heuristic") is None
    assert isinstance(make_judge("claude", api_key="k"), ClaudeJudge)
    ollama = make_judge("ollama")
    assert isinstance(ollama, ChatJudge) and ollama.base_url == "http://localhost:11434/v1" and not ollama.api_key
    with pytest.raises(JudgeUnavailable, match="base URL"):
        make_judge("custom", log=lambda m: None).judge(_feats(1))
    with pytest.raises(ValueError):
        make_judge("nope")
    assert judge_tag("openai") == "GPT" and judge_tag("claude") == "Claude" and judge_tag("heuristic") == "Rules"


def test_network_trouble_never_stops_the_run():
    judge = ChatJudge("custom", base_url="api.example.com/v1", log=lambda m: None)
    result, judge_after = judge_or_fallback(judge, _feats(1), None, "en", None, lambda m: None)
    assert result["source"] == "heuristic" and judge_after is None  # bad URL: rules from now on

    def flaky(url, headers, body, timeout):
        raise http.client.IncompleteRead(b"")

    judge = _chat("openai", api_key="k")
    judge.post = flaky
    result, judge_after = judge_or_fallback(judge, _feats(1), None, "en", None, lambda m: None)
    assert result["source"] == "heuristic" and judge_after is judge  # dropped connection: just this window
