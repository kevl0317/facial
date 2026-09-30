# facial

Annotate how someone speaks: what their hands, face and voice are doing, and what that suggests about their intent. It works like the "per-frame action + intent" breakdowns of interview clips, and it runs in two ways:

- **Live camera** on a phone or PC. A web GUI tracks you (or whoever the camera points at) in real time and shows a verdict every few seconds.
- **Video files.** Upload in the GUI, or use the command line, to get an annotated MP4 plus a JSON analysis.

In both modes:

- **Google MediaPipe** detects hands, gestures, face blendshapes and pose on each frame.
- **Audio analysis** measures loudness, pitch, pauses and speech rate.
- **An AI model** acts as the *judgment layer*: Claude by default, or GPT, DeepSeek, Gemini, Grok, Mistral, Qwen, Kimi, GLM, Groq, OpenRouter, a local Ollama model or any OpenAI-compatible API. It fuses five fields per time window (scene, speaker, subtitle, voice, gesture) into a reading, confidence/focus/tension scores, an intent label and an emotion arc.
- A HUD is drawn over the picture.

```
                     ┌──────────── phone / PC browser (live) ────────────┐
camera + mic ──────▶ │ MediaPipe (WASM/WebGL) · mic PCM · speech-to-text │──┐ every ~5 s: samples,
                     │ HUD drawn live on the camera image                │  │ audio, transcript
                     └───────────────────────────────────────────────────┘  ▼
video file ──▶ MediaPipe (Python) · librosa · subtitles ──▶ 5 fields per window ──▶ AI judge ──▶ verdict / annotated.mp4
                                        (python -m facial serve  or  python -m facial VIDEO)
```

What the overlay shows (same cartoon look live and in rendered videos):

| Where | What |
|---|---|
| top chips | time, current window, shot type; live mode adds a LIVE / Thinking status chip |
| top left | the hand label sticker, e.g. **Left hand · open palm (palm up)**, its MediaPipe confidence, and a pointer to the hand |
| verdict card | Confident / Focused / Tense bars, the intent with its certainty, and Mood dots (one per window, coral → yellow → mint) |
| speech bubble | the judgment layer's reading for the window, with the most telling quote highlighted |
| bottom | comic-style subtitles and a small "demo only" tag |

Language `zh` switches the overlay and the judge's commentary to Chinese (综合判定 / 自信 / 专注 / 紧张 / 意图 / 情绪弧).

The GUI opens with a short hand-drawn animation (12 fps, frame by frame: the glove sketches itself in, waves, then an iris wipe opens the app; tap to skip). The theme uses the bundled [Fredoka](https://github.com/hafontia/Fredoka-One) font (SIL Open Font License, `facial/web/fonts/OFL.txt`).

## Setup

Python 3.10+ on the computer that runs the server.

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...        # for the Claude judgment layer (the default)
```

Other AI providers take their usual key variable and `--judge <provider>`:

| `--judge` | Key variable | Default model |
|---|---|---|
| `claude` | `ANTHROPIC_API_KEY` | `claude-opus-5-5` |
| `openai` | `OPENAI_API_KEY` | `gpt-5-mini` |
| `deepseek` | `DEEPSEEK_API_KEY` | `deepseek-chat` |
| `gemini` | `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | `gemini-2.5-flash` |
| `grok` | `XAI_API_KEY` | `grok-3-mini` |
| `mistral` | `MISTRAL_API_KEY` | `mistral-small-latest` |
| `qwen` | `DASHSCOPE_API_KEY` | `qwen-plus` |
| `kimi` | `MOONSHOT_API_KEY` | `kimi-k2-0905-preview` |
| `glm` | `ZHIPUAI_API_KEY` (or `ZAI_API_KEY`) | `glm-4.5-air` |
| `groq` | `GROQ_API_KEY` | `llama-3.3-70b-versatile` |
| `openrouter` | `OPENROUTER_API_KEY` | `google/gemini-2.5-flash` |
| `ollama` | none (runs locally) | `llama3.2` |
| `custom` | `OPENAI_COMPATIBLE_API_KEY` (optional) | pass `--base-url` and `--model` |
| `heuristic` | none | rule-based scores, no AI |

### Jev: fast decisions, less thinking

[Jev](https://typesafe.ai) (TypeSafe AI, September 2026) is a "System One" decision model: instead of writing text it answers typed questions (a score on a scale, a choice between options) with calibrated probabilities, in well under a second. With Jev switched on:

1. Each window's measurements are described in plain words (`facial/jev.py`, `describe()`), because Jev handles words better than numbers.
2. Jev decides the Confident / Focused / Tense scores, the mood and the intent.
3. The AI model only writes the one-line reading, the quote and the evidence, at low effort, so it thinks far less.
4. When Jev is unsure of the intent (below 40%), that window goes back to the AI model to judge in full. Without an AI model, the words come from the rule templates.

The overlay credits it the way the original breakdown videos do: `MediaPipe · Jev · Claude`.

```bash
export TYPESAFE_API_KEY=...          # or OPENROUTER_API_KEY, which reaches Jev through OpenRouter
python -m facial clip.mp4 --srt clip.srt --jev                    # Jev decides, Claude writes
python -m facial clip.mp4 --srt clip.srt --jev --judge heuristic  # Jev alone, rule templates write
python -m facial serve --jev --judge deepseek
```

On the linked site, turn on **Jev** in Settings and paste an OpenRouter key (the same key can also serve as the AI model through OpenRouter); the model is `typesafe/jev-1.13`. TypeSafe's own API sends no CORS headers, so web pages can't call it at all and the site only offers OpenRouter (trying both of its Jev addresses, `/api/v1/systemone` and `/api/alpha/decisions`). If the browser still can't reach Jev, the app says so and the AI model decides instead; the self-hosted server has no such limit and works with both routes.

Pick another model with `--model`, and point any provider at a different endpoint (a proxy, a regional endpoint such as `https://dashscope.aliyuncs.com/compatible-mode/v1` or `https://api.moonshot.cn/v1`) with `--base-url`. The providers and their defaults live in `facial/providers.py`.

- **Linux only:** MediaPipe needs EGL/GLES: `sudo apt install libegl1 libgles2`.
- **ffmpeg:** used from `PATH` if present, otherwise the `imageio-ffmpeg` bundled binary.
- **Chinese in rendered videos:** needs a CJK font. PingFang (macOS), Microsoft YaHei (Windows), and Noto CJK or WenQuanYi (Linux) are found automatically; otherwise pass `--font /path/to/font.ttc`. The live GUI uses the browser's fonts.
- **Downloads on first run:** the MediaPipe models (~18 MB) and the MediaPipe web runtime (~35 MB) go into `models/`. The phone loads them from your computer, so it needs no internet access beyond the local network.

## Open it from a link (no install)

The whole app also runs **entirely in the browser**, with no server:

- MediaPipe runs on the device.
- The five fields, the voice analysis and the judge are ported to JavaScript. `tests/test_web.py` checks the port against the Python pipeline.
- By default the judge is the transparent rule set. In Settings, pick a provider (Claude, GPT, DeepSeek, Gemini, and the rest) and paste your own API key to get AI verdicts. The model field suggests the models your key can use, and the key field turns green once the provider accepts it. Keys are stored only in that browser and sent only to the provider you picked.
- Some providers don't accept requests straight from a web page (CORS). If one of them can't be reached, the app says so and falls back to the rules. Then try OpenRouter, which offers most models under one key, or use the self-hosted server below, which works with every provider.
- **Live** uses the phone or laptop camera.
- **Video** plays an uploaded file through the same engine, pausing for verdicts when needed, and saves an annotated recording plus the JSON.
- The site is installable (add to home screen). A service worker caches the ~30 MB of models, so later visits start instantly and work offline.

```bash
python -m facial build-site --out site     # static site: index.html, static/, models/
```

It is deployed from the `gh-pages` branch. To publish it, open **Settings → Pages**, choose **Deploy from a branch → gh-pages / (root)**, and it appears at `https://<user>.github.io/<repo>/`. Any static host that serves HTTPS works, because phones only allow the camera on https:// pages.

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
- Every *window length* seconds (default 5), the phone sends that window's measurements, microphone audio and speech-to-text transcript to the server. The server builds the same five fields as the video pipeline and asks the AI judge for a verdict, which appears in the panel and the list below it.
- Video frames never leave the device. Only the numbers derived from them, the audio and the transcript go to your computer, and only the five-field summary goes to the AI provider.
- Front or back camera, English or 中文, optional context ("practising a product pitch") and an optional hand skeleton can be set in *Settings*.
- Live subtitles use the browser's built-in speech recognition (Chrome, Edge, Safari; not Firefox). On Chrome that service runs on Google's servers.

**Video file tab:** upload a video (from the PC, or from the phone's gallery), optionally with `.srt`/`.vtt` subtitles, a start/end range and context. The server runs the full offline pipeline and shows a progress bar. When it finishes you can play the annotated video in the page, download it or the analysis JSON, and click any window to jump to it.

Server options:

| Option | Default | |
|---|---|---|
| `--port` | 8443 (HTTPS), 8000 with `--no-https` | |
| `--token` | random each run | the access key in the URL (`k=`), which stops others on your network from using your API key; `--token ""` disables it |
| `--judge` | `claude` | the AI provider (see the table under Setup), or `heuristic` for rule-based scores only |
| `--model` | the provider's default | |
| `--base-url` | the provider's | required for `--judge custom` |
| `--jev` | off | Jev decides, the `--judge` model only writes (see Jev above); `--jev-via typesafe\|openrouter`, `--jev-model` |
| `--live-effort` | `low` | reasoning effort for live windows (fast verdicts) |
| `--effort` | `medium` | reasoning effort for video-file jobs |

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

# Another provider (key from DEEPSEEK_API_KEY), or a local model through Ollama
python -m facial clip.mp4 --srt clip.srt --judge deepseek
python -m facial clip.mp4 --srt clip.srt --judge ollama --model qwen3

# No API key: transparent rule-based scoring instead of an AI judge
python -m facial clip.mp4 --srt clip.srt --judge heuristic
```

Outputs, next to the output video:

- `clip_annotated.mp4`: the video with the HUD (H.264, original audio).
- `clip_annotated_analysis.json`: every window's five fields plus the AI judgment (reading, quote, scores, intent, evidence).
- `clip_annotated_work/`: cached stages (`samples.json` holds the per-frame MediaPipe results, `judgments.json` the AI answers). Re-running with a different `--lang`, `--skeleton` or font only redoes what changed. `--fresh` recomputes everything.

Useful options: `--window 5` (target window length in seconds), `--analysis-fps 15` (MediaPipe sampling rate; keep ≥ 15 so blinks are caught), `--skeleton` (draw hand landmarks and face brackets), `--mirrored` (selfie videos, which fixes left/right), `--judge` / `--model` / `--effort` (AI provider and model, default Claude `claude-opus-5-5` at `medium` effort).

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

The judgment layer runs as **one AI conversation per clip or live session, one window per turn**. That way the emotion arc reads as a single story. Long live sessions start a fresh conversation every 24 windows and carry a summary of the recent arc forward; with providers other than Claude, which re-send the whole history each turn instead of caching it, that happens every 12 windows in any session. Claude's answers are constrained to a JSON schema (structured outputs); OpenAI and Grok get the same strict schema, the other providers get JSON mode plus the schema in the prompt, and a provider that rejects those options gets a plain request. If a call fails, that window falls back to the heuristic scorer, and the verdict card and footer say which judge produced the scores.

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
  judge.py       judgment layer: Claude (Anthropic SDK), OpenAI-compatible providers, heuristic fallback
  providers.py   the AI providers: base URLs, default models, key variables
  jev.py         Jev decision layer: window described in words, typed questions, /systemone client
  net.py         small JSON-over-HTTP helpers
  render.py      Pillow HUD overlay and video writing
  media.py       ffmpeg helpers (clip, audio extraction, H.264 writer)
  models.py      MediaPipe model + web runtime download
  webbuild.py    static site build (python -m facial build-site) + prompt.js generation
  web/           the GUI: index.html, style.css, app.js, live.js (engine: camera or file, server or in-browser),
                 perception.js, features.js, voice.js, judge.js, srt.js (browser ports), hud.js (overlay),
                 intro.js (opening animation), hand.js (the cartoon glove), sw.js + manifest (PWA), vendor/, fonts/
tests/           unit + API tests (python -m pytest)
```
