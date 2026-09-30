import json
from types import SimpleNamespace

from facial.judge import ClaudeJudge


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
    return {"window": i, "of": None, "start": 5.0 * (i - 1), "end": 5.0 * i, "scene": {}, "speaker": {},
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
