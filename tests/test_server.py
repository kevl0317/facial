import base64
import time

import cv2
import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from facial.server import create_app  # noqa: E402

from .test_core import _samples  # noqa: E402

KEY = "test-key"


@pytest.fixture()
def client(tmp_path):
    app = create_app(token=KEY, judge="heuristic", state_dir=tmp_path)
    return TestClient(app, headers={"X-Facial-Key": KEY})


def test_api_requires_key(client):
    assert client.get("/api/info").status_code == 200
    assert client.get("/api/info", headers={"X-Facial-Key": "nope"}).status_code == 401
    assert client.get(f"/api/info?k={KEY}", headers={"X-Facial-Key": ""}).status_code == 200
    # The page itself is public; the key rides along in the URL.
    assert client.get("/", headers={"X-Facial-Key": ""}).status_code == 200


def test_live_window_roundtrip(client):
    sid = client.post("/api/live/sessions", json={"lang": "en", "context": "test"}).json()["id"]
    sr = 16000
    t = np.arange(5 * sr) / sr
    tone = (0.3 * np.sin(2 * np.pi * 140 * t) * (np.sin(2 * np.pi * 3 * t) > 0)).astype(np.float32)
    pcm = base64.b64encode((tone * 32767).astype("<i2").tobytes()).decode()
    samples = _samples(n=50, fps=10.0)  # t = 0.0 .. 4.9
    body = {"start": 0.0, "end": 5.0, "aspect": 16 / 9, "samples": samples, "audio": pcm, "audio_sr": sr,
            "audio_t0": 0.0, "transcript": [{"start": 0.5, "end": 3.0, "text": "We stuck with it."}]}
    res = client.post(f"/api/live/sessions/{sid}/windows", json=body)
    assert res.status_code == 200, res.text
    out = res.json()
    win, judgment = out["window"], out["judgment"]
    assert win["window"] == 1 and win["of"] is None
    assert win["subtitle"] == "We stuck with it."
    assert win["voice"]["voiced_frac"] > 0.3 and "pitch_rel_st" in win["voice"]
    assert win["gesture"]["dominant"]["shape"] == "open_palm"
    assert judgment["source"] == "heuristic" and 0 <= judgment["traits"]["confident"] <= 1
    assert judgment["emotion"] in ("calm", "happy") and "expression" in win["gesture"]["face"]

    # Second window continues the session.
    body2 = dict(body, start=5.0, end=10.0, samples=[dict(s, t=s["t"] + 5.0) for s in samples], transcript=[])
    assert client.post(f"/api/live/sessions/{sid}/windows", json=body2).json()["window"]["window"] == 2
    assert client.delete(f"/api/live/sessions/{sid}").status_code == 200
    assert client.post(f"/api/live/sessions/{sid}/windows", json=body2).status_code == 404


def test_video_job(client, tmp_path):
    video = tmp_path / "tiny.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    for i in range(30):
        frame = np.full((120, 160, 3), 40 + i * 3, np.uint8)
        writer.write(frame)
    writer.release()

    with open(video, "rb") as fh:
        res = client.post("/api/jobs", files={"video": ("tiny.mp4", fh, "video/mp4")},
                          data={"lang": "en", "window": "2"})
    assert res.status_code == 200, res.text
    job_id = res.json()["id"]
    deadline = time.time() + 120
    while True:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error") or time.time() > deadline:
            break
        time.sleep(0.5)
    assert job["status"] == "done", job
    assert job["progress"] == 1.0

    full = client.get(job["video"])
    assert full.status_code == 200 and full.headers["content-type"] == "video/mp4" and len(full.content) > 1000
    part = client.get(job["video"], headers={"Range": "bytes=0-99"})
    assert part.status_code == 206 and len(part.content) == 100  # seeking works in Safari/iOS
    analysis = client.get(job["analysis"]).json()
    assert len(analysis["windows"]) >= 1


def test_info_names_the_judge(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    app = create_app(token="", judge="deepseek", state_dir=tmp_path)
    c = TestClient(app)
    info = c.get("/api/info").json()
    assert (info["judge"], info["tag"], info["model"]) == ("deepseek", "DeepSeek", "deepseek-chat")
    assert c.post("/api/live/sessions", json={}).json()["judge"] == "deepseek"
    rules = TestClient(create_app(token="", judge="heuristic", state_dir=tmp_path)).get("/api/info").json()
    assert (rules["tag"], rules["model"]) == ("Rules", "")


def test_info_credits_jev(tmp_path):
    app = create_app(token="", judge="deepseek", state_dir=tmp_path, jev={"route": "openrouter", "model": None})
    info = TestClient(app).get("/api/info").json()
    assert info["jev"] is True and info["label"] == "Jev · DeepSeek"
