import numpy as np
import pytest

from facial.features import add_hand_motion, build_windows, window_features
from facial.judge import heuristic_judgment
from facial.perception import describe_hand
from facial.render import Overlay, hand_label
from facial.transcript import Segment, parse_subtitles, shift


def test_parse_srt_and_vtt():
    srt = "1\n00:00:01,500 --> 00:00:03,000\n<i>Hello</i> there\n\n2\n00:01:02,000 --> 00:01:04,250\nSecond\n"
    segs = parse_subtitles(srt)
    assert [(s.start, s.end, s.text) for s in segs] == [(1.5, 3.0, "Hello there"), (62.0, 64.25, "Second")]

    vtt = "WEBVTT\n\n00:05.000 --> 00:06.500\n字幕\n"
    assert [(s.start, s.end, s.text) for s in parse_subtitles(vtt)] == [(5.0, 6.5, "字幕")]


def test_shift_clips_to_range():
    segs = [Segment(1, 2, "a"), Segment(9, 12, "b"), Segment(30, 31, "c")]
    out = shift(segs, offset=10, duration=15)
    assert [(s.start, s.end, s.text) for s in out] == [(0.0, 2.0, "b")]


def test_windows_are_contiguous_and_snap_to_sentences():
    segs = [Segment(0.2, 4.8, "a"), Segment(5.0, 10.1, "b"), Segment(10.5, 14.9, "c"), Segment(15.2, 19.0, "d")]
    wins = build_windows(22.0, segs, target=5.0, min_len=2.5, max_len=9.0)
    assert wins[0][0] == 0.0 and wins[-1][1] == 22.0
    assert all(a[1] == b[0] for a, b in zip(wins, wins[1:]))
    assert all(2.5 <= b - a <= 9.0 for a, b in wins)
    assert 4.8 in [w[1] for w in wins]


def _open_right_hand_palm_to_camera():
    """Synthetic right hand, fingers up, palm facing the camera (thumb on image right)."""
    pts = np.zeros((21, 2))
    pts[0] = (100, 200)
    for base, x in ((5, 130), (9, 110), (13, 90), (17, 70)):  # index..pinky MCP -> tip
        pts[base] = (x, 110)
        pts[base + 1] = (x, 80)
        pts[base + 2] = (x, 55)
        pts[base + 3] = (x, 30)
    pts[1:5] = [(125, 175), (145, 160), (160, 145), (172, 130)]
    return pts


def test_palm_facing_follows_chirality():
    pts = _open_right_hand_palm_to_camera()
    geo = describe_hand(pts, None, "right")
    assert (geo["shape"], geo["facing"], geo["axis"]) == ("open_palm", "palm_out", "up")

    mirrored = pts.copy()
    mirrored[:, 0] = 300 - mirrored[:, 0]
    assert describe_hand(mirrored, None, "left")["facing"] == "palm_out"
    assert describe_hand(mirrored, None, "right")["facing"] == "back_out"


def _hand(side, x, y, score=0.8):
    return {"side": side, "shape": "open_palm", "facing": "palm_out", "axis": "up", "score": score,
            "src": "geometry", "anchor": [x, y], "size": 0.1, "pts": [[x, y]] * 21}


def _samples(n=40, fps=10.0):
    samples = []
    for i in range(n):
        t = i / fps
        samples.append({
            "t": t, "cut": False, "faces": 1,
            "face": {"bbox": [0.4, 0.2, 0.6, 0.5], "size": 0.3, "smile": 0.3, "frown": 0.0, "brow_furrow": 0.1,
                     "brow_raise": 0.1, "lip_press": 0.1, "blink": 0.1, "eye_wide": 0.0, "squint": 0.1,
                     "jaw": 0.1 + 0.1 * np.sin(i), "gaze_side": 0.1, "gaze_down": 0.1,
                     "yaw": 0.0, "pitch": 0.0, "roll": 0.0},
            # Two hands carry the same label: one still, one sweeping right.
            "hands": [_hand("right", 0.2, 0.8), _hand("right", 0.5 + 0.01 * i, 0.6)],
            "pose": {"shoulder_w": 0.3, "tilt": 0.0, "arms_open": 1.2},
        })
    return samples


def test_tracking_separates_hands_with_the_same_label():
    samples = _samples()
    add_hand_motion(samples, aspect=16 / 9)
    still, moving = samples[20]["hands"]
    assert still["track"] != moving["track"]
    assert still["speed"] == 0.0
    # 0.01 * 16/9 per 0.1 s over a 0.1 hand length ~ 1.78 hand-lengths/s, minus the jitter floor
    assert moving["speed"] == pytest.approx(1.78 - 0.3, abs=0.05)


def test_window_features_and_heuristic_judgment():
    samples = _samples()
    add_hand_motion(samples, aspect=16 / 9)
    feats = window_features(0, 1, (0.0, 4.0), samples, None, [Segment(0.5, 3.5, "Is this right?")], {})
    assert set(feats) >= {"scene", "speaker", "subtitle", "voice", "gesture"}
    assert feats["scene"]["shot"] == "medium"
    j = heuristic_judgment(feats, "en", lambda h: hand_label(h, "en"))
    for key in ("confidence", "focus", "tension", "intent_certainty"):
        assert 0.0 <= j[key] <= 1.0
    assert -1.0 <= j["valence"] <= 1.0
    assert j["source"] == "heuristic"


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_overlay_draws(lang):
    samples = _samples()
    add_hand_motion(samples, aspect=16 / 9)
    feats = [window_features(0, 1, (0.0, 4.0), samples, None, [], {})]
    judgments = [heuristic_judgment(feats[0], lang, lambda h: hand_label(h, lang))]
    overlay = Overlay(640, 360, samples, feats, judgments, [Segment(0.0, 4.0, "hello")], lang=lang)
    frame = np.zeros((360, 640, 3), np.uint8)
    out = overlay.draw(frame, 2.0)
    assert out.shape == frame.shape and out.any()
