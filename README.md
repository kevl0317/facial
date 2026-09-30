# facial

Annotate how someone speaks in a video: what their hands, face and voice are doing, and what that suggests about their intent. It works like the "per-frame action + intent" breakdowns of interview clips:

- **Google MediaPipe** detects hands, gestures, face blendshapes and pose on each frame.
- **Audio analysis** measures loudness, pitch, pauses and speech rate.
- **Claude** acts as the *judgment layer*. It fuses five fields per time window (scene, speaker, subtitle, voice, gesture) into a reading, confidence/focus/tension scores, an intent label and an emotion arc.
- Everything is drawn back onto the video as a HUD.

```
video ─┬─ MediaPipe (face · hands/gestures · pose) ─┐
       ├─ audio (loudness · pitch · pauses) ─────────┤→ windows of ~5 s → Claude → HUD overlay → annotated.mp4
       └─ subtitles (.srt / faster-whisper) ─────────┘      5 fields each     judgment         + analysis.json
```

What the overlay shows:

| Where | What |
|---|---|
| top strip | `t=027.8  WIN 10/10  shot=close-up`: time, current window, shot type |
| top left | live hand label, e.g. **Left hand · open palm (palm up)**, the MediaPipe confidence, and a callout line to the hand |
| right panel | **Verdict**: Confident / Focused / Tense bars, Intent → label + certainty, and the emotion arc (one square per window) |
| bottom left | `W10 > reading…` from the judgment layer, plus the most telling quote |
| bottom | subtitles and a "not calibrated, demo only" footer |

`--lang zh` switches the overlay and Claude's commentary to Chinese (综合判定 / 自信 / 专注 / 紧张 / 意图 / 情绪弧).

## Setup

Python 3.10+.

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...        # for the Claude judgment layer
```

- **Linux only:** MediaPipe needs EGL/GLES: `sudo apt install libegl1 libgles2`.
- **ffmpeg:** used from `PATH` if present, otherwise the `imageio-ffmpeg` bundled binary.
- **Chinese overlay:** needs a CJK font. PingFang (macOS), Microsoft YaHei (Windows), and Noto CJK or WenQuanYi (Linux) are found automatically; otherwise pass `--font /path/to/font.ttc`.
- **MediaPipe models:** the three `.task` models (~18 MB) download into `models/` on first run.

## Usage

```bash
# Whole video, with subtitles you already have
python -m facial interview.mp4 --srt interview.srt --context "CEO interview about company culture"

# Just the key segment (00:01:40 - 00:02:10), Chinese overlay
python -m facial keynote.mp4 --start 100 --end 130 --srt keynote.srt --lang zh

# No subtitles? Transcribe locally (pip install faster-whisper)
python -m facial clip.mp4 --whisper small

# Preview one frame as a PNG before rendering everything
python -m facial clip.mp4 --srt clip.srt --snapshot 27.8

# No API key: transparent rule-based scoring instead of Claude
python -m facial clip.mp4 --srt clip.srt --judge heuristic
```

Outputs, next to the output video:

- `clip_annotated.mp4`: the video with the HUD (H.264, original audio).
- `clip_annotated_analysis.json`: every window's five fields plus Claude's judgment (reading, quote, scores, intent, evidence).
- `clip_annotated_work/`: cached stages (`samples.json` holds the per-frame MediaPipe results, `judgments.json` the Claude answers). Re-running with a different `--lang`, `--skeleton` or font only redoes what changed. `--fresh` recomputes everything.

Useful options: `--window 5` (target window length in seconds), `--analysis-fps 15` (MediaPipe sampling rate; keep ≥ 15 so blinks are caught), `--skeleton` (draw hand landmarks and face brackets), `--mirrored` (selfie videos, which fixes left/right), `--model` / `--effort` (Claude model, default `claude-opus-5-5` at `medium` effort).

## How the five fields are measured

| Field | Source | Contents |
|---|---|---|
| **scene** | face size, colour-histogram cuts | shot type (close-up / medium / wide / cutaway), faces, hard cuts |
| **speaker** | jaw blendshape vs. audio loudness | `target` (on camera and talking), `listening`, `offscreen` or `silent`, plus lip-sync correlation |
| **subtitle** | `.srt`/`.vtt` or faster-whisper | the words said in the window |
| **voice** | librosa | loudness and pitch *relative to the speaker's own median*, pitch variability, voiced share, pauses, speech rate, fillers |
| **gesture** | MediaPipe GestureRecognizer, FaceLandmarker, PoseLandmarker | dominant hand state (hand, shape, palm facing, axis), a short timeline of hand states, gesture energy, beat gestures, hand-to-face contact, head movement and gaze, 52 blendshapes condensed into smile / frown / brow / lip press (with deltas against the clip baseline), blink rate, posture |

Hand shape comes from MediaPipe's gesture classifier when it is confident, otherwise from finger geometry. Palm facing (up / down / out / back / sideways) comes from the palm normal of the 3D hand landmarks, checked against the 2D winding. Hands are tracked by position, not by their left/right label, because MediaPipe sometimes gives two hands the same label.

The judgment layer runs as **one Claude conversation per clip, one window per turn**. That way the emotion arc reads as a single story and each turn reuses the cached history, so cost grows slowly with clip length. Every answer is constrained to a JSON schema (structured outputs). If a call fails, that window falls back to the heuristic scorer, and the footer says which judge produced the scores.

## Caveats

This is a demo for studying body language, not a lie detector. Reading intent or hidden feelings from gestures and micro-expressions is not scientifically reliable, and the scores are uncalibrated. They describe what a person *expresses* relative to their own baseline in the clip, and the overlay says so. MediaPipe landmarks are noisy on small, fast-moving or partly hidden hands. Check the `evidence` field in the analysis JSON before trusting a reading.

## Project layout

```
facial/
  __main__.py    CLI and pipeline orchestration (with stage caching)
  perception.py  MediaPipe per-frame: hands/gestures, face blendshapes + head pose, pose
  audio.py       loudness, pitch (YIN), voice activity, pauses
  transcript.py  SRT/VTT parsing, optional faster-whisper transcription
  features.py    hand tracking, windowing, the five fields per window
  judge.py       Claude judgment layer + heuristic fallback
  render.py      Pillow HUD overlay and video writing
  media.py       ffmpeg helpers (clip, audio extraction, H.264 writer)
  models.py      MediaPipe model download
tests/           unit tests (python -m pytest)
```
