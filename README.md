# facial

Annotate how someone speaks: what their hands, face and voice are doing, and what that suggests about their intent. It works like the "per-frame action + intent" breakdowns of interview clips, and it runs in two ways:

- **Live camera** on a phone or PC. A web GUI tracks you (or whoever the camera points at) in real time and shows a verdict every few seconds.
- **Video files.** Upload in the GUI, or use the command line, to get an annotated MP4 plus a JSON analysis.

In both modes:

- **Google MediaPipe** detects hands, gestures, face blendshapes and pose on each frame.
- **Audio analysis** measures loudness, pitch, pauses and speech rate.
- **Claude** acts as the *judgment layer*. It fuses five fields per time window (scene, speaker, subtitle, voice, gesture) into a reading, confidence/focus/tension scores, an intent label and an emotion arc.
- A HUD is drawn over the picture.

```
                     ┌──────────── phone / PC browser (live) ────────────┐
camera + mic ──────▶ │ MediaPipe (WASM/WebGL) · mic PCM · speech-to-text │──┐ every ~5 s: samples,
                     │ HUD drawn live on the camera image                │  │ audio, transcript
                     └───────────────────────────────────────────────────┘  ▼
video file ──▶ MediaPipe (Python) · librosa · subtitles ──▶ 5 fields per window ──▶ Claude ──▶ verdict / annotated.mp4
                                        (python -m facial serve  or  python -m facial VIDEO)
```

What the overlay shows:

| Where | What |
|---|---|
| top strip | `t=027.8  WIN 10/10  shot=close-up`: time, current window, shot type; live mode adds status and fps |
| top left | live hand label, e.g. **Left hand · open palm (palm up)**, the MediaPipe confidence, and a callout line to the hand |
| right panel | **Verdict**: Confident / Focused / Tense bars, Intent → label + certainty, and the emotion arc (one square per window) |
| bottom left | `W10 > reading…` from the judgment layer, plus the most telling quote |
| bottom | subtitles and a "not calibrated, demo only" footer |

Language `zh` switches the overlay and Claude's commentary to Chinese (综合判定 / 自信 / 专注 / 紧张 / 意图 / 情绪弧).

## Setup

Python 3.10+ on the computer that runs the server.

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...        # for the Claude judgment layer
```

- **Linux only:** MediaPipe needs EGL/GLES: `sudo apt install libegl1 libgles2`.
- **ffmpeg:** used from `PATH` if present, otherwise the `imageio-ffmpeg` bundled binary.
- **Chinese in rendered videos:** needs a CJK font. PingFang (macOS), Microsoft YaHei (Windows), and Noto CJK or WenQuanYi (Linux) are found automatically; otherwise pass `--font /path/to/font.ttc`. The live GUI uses the browser's fonts.
- **Downloads on first run:** the MediaPipe models (~18 MB) and the MediaPipe web runtime (~35 MB) go into `models/`. The phone loads them from your computer, so it needs no internet access beyond the local network.

## Web GUI: live camera on your phone, and video files

```bash
python -m facial serve
```

This prints two URLs and a QR code:

```
  On this computer:  https://localhost:8443/?k=Ab3dE...
  On your phone:     https://192.168.1.23:8443/?k=Ab3dE...   (same Wi-Fi)
```

1. Scan the QR code with the phone (or open the URL). The PC page shows the same QR code under *Settings*.
2. Accept the certificate warning once. Phones only allow camera access over HTTPS, so the server uses a self-signed certificate. On iOS tap *Show Details → visit this website*; on Android Chrome tap *Advanced → Proceed*.
3. Allow camera and microphone, prop the phone up so face and hands are in frame, and tap **Start**.

**Live camera tab:**

- MediaPipe runs on the phone itself: GPU (WebGL) when available, otherwise CPU.
- The hand label and callout update on every frame.
- Every *window length* seconds (default 5), the phone sends that window's measurements, microphone audio and speech-to-text transcript to the server. The server builds the same five fields as the video pipeline and asks Claude for a verdict, which appears in the panel and the list below it.
- Video frames never leave the device. Only the numbers derived from them, the audio and the transcript go to your computer, and only the five-field summary goes to Claude.
- Front or back camera, English or 中文, optional context ("practising a product pitch") and an optional hand skeleton can be set in *Settings*.
- Live subtitles use the browser's built-in speech recognition (Chrome, Edge, Safari; not Firefox). On Chrome that service runs on Google's servers.

**Video file tab:** upload a video (from the PC, or from the phone's gallery), optionally with `.srt`/`.vtt` subtitles, a start/end range and context. The server runs the full offline pipeline and shows a progress bar. When it finishes you can play the annotated video in the page, download it or the analysis JSON, and click any window to jump to it.

Server options:

| Option | Default | |
|---|---|---|
| `--port` | 8443 (HTTPS), 8000 with `--no-https` | |
| `--token` | random each run | the access key in the URL (`k=`), which stops others on your network from using your API key; `--token ""` disables it |
| `--live-effort` | `low` | Claude effort for live windows (fast verdicts) |
| `--effort` | `medium` | Claude effort for video-file jobs |
| `--model` | `claude-opus-5-5` | |
| `--judge heuristic` | | no Claude, just transparent rule-based scores |

If a phone can't reach the server (guest Wi-Fi often isolates devices) or refuses the certificate, use an HTTPS tunnel instead: `python -m facial serve --no-https`, then `cloudflared tunnel --url http://localhost:8000`. Open the tunnel URL with `/?k=<token>` appended.

## Command line: video files

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
| **subtitle** | `.srt`/`.vtt`, faster-whisper, or live browser speech recognition | the words said in the window |
| **voice** | librosa | loudness and pitch *relative to the speaker's own median*, pitch variability, voiced share, pauses, speech rate, fillers |
| **gesture** | MediaPipe GestureRecognizer, FaceLandmarker, PoseLandmarker | dominant hand state (hand, shape, palm facing, axis), a short timeline of hand states, gesture energy, beat gestures, hand-to-face contact, head movement and gaze, 52 blendshapes condensed into smile / frown / brow / lip press (with deltas against the baseline), blink rate, posture |

- **Hand shape** comes from MediaPipe's gesture classifier when it is confident, otherwise from finger geometry.
- **Palm facing** (up / down / out / back / sideways) comes from the palm normal of the 3D hand landmarks, checked against the 2D winding.
- **Hand tracking** is by position, not by the left/right label, because MediaPipe sometimes gives two hands the same label.
- **Browser and Python share this logic.** `web/perception.js` is a line-for-line port of the same geometry, and the head-pose angles were checked to match between the two. Live samples therefore go through exactly the same feature code on the server.

The judgment layer runs as **one Claude conversation per clip or live session, one window per turn**. That way the emotion arc reads as a single story and each turn reuses the cached history. Long live sessions start a fresh conversation every 24 windows and carry a summary of the recent arc forward. Every answer is constrained to a JSON schema (structured outputs). If a call fails, that window falls back to the heuristic scorer, and the footer says which judge produced the scores.

## Caveats

This is a demo for studying body language, not a lie detector. Reading intent or hidden feelings from gestures and micro-expressions is not scientifically reliable, and the scores are uncalibrated. They describe what a person *expresses* relative to their own baseline, and the overlay says so. MediaPipe landmarks are noisy on small, fast-moving or partly hidden hands. Check the `evidence` field before trusting a reading.

## Project layout

```
facial/
  __main__.py    CLI: `python -m facial VIDEO`, `python -m facial serve`
  pipeline.py    offline video pipeline (stage caching, progress reporting)
  server.py      FastAPI web GUI: live sessions, video jobs, HTTPS + QR code
  live.py        live sessions: windows streamed from the browser -> five fields -> judgment
  perception.py  MediaPipe per-frame: hands/gestures, face blendshapes + head pose, pose
  audio.py       loudness, pitch (YIN), voice activity, pauses
  transcript.py  SRT/VTT parsing, optional faster-whisper transcription
  features.py    hand tracking, windowing, the five fields per window
  judge.py       Claude judgment layer + heuristic fallback
  render.py      Pillow HUD overlay and video writing
  media.py       ffmpeg helpers (clip, audio extraction, H.264 writer)
  models.py      MediaPipe model + web runtime download
  web/           the GUI: index.html, app.js, live.js (camera engine), perception.js, hud.js
tests/           unit + API tests (python -m pytest)
```
