# Kei.AI V2

Kei.AI is a local-first conversational productivity assistant that combines speech recognition, LLM chat, text-to-speech, and a real-time 3D VRM avatar in the browser whose purpose is to help the user focus on tasks and avoid procrastination or veering off their work responsibilities. 

The project is built as a full-stack prototype with a Python API backend and a JavaScript frontend.

## Core Features

- Speech-to-text transcription with OpenAI Whisper
- Text chat endpoint backed by OpenRouter-compatible models
- Async audio generation jobs with status polling and file caching
- Japanese TTS support via Qwen3-TTS, with Windows SAPI fallback
- Browser 3D avatar rendering (VRM) with emotion states and lip-sync
- Responsive text-chat UI with live assistant output

## Tech Stack

### Backend

- Python
- Django (lightweight API routing)
- OpenAI Whisper (`openai-whisper`)
- Requests (HTTP client for model calls)
- Threading + file-based audio caching

### Frontend

- JavaScript (ES modules)
- Three.js
- `@pixiv/three-vrm`
- Web Audio API (playback analysis for mouth movement)
- HTML/CSS

### AI / Model Integrations

- OpenRouter Chat Completions API
- Whisper STT model
- Qwen3-TTS voice generation (optional)
- Windows SAPI TTS fallback

## Architecture Overview

1. User submits text (or voice input pipeline transcribes speech).
2. Frontend sends message to backend `/chat` endpoint.
3. Backend calls OpenRouter model and returns assistant text.
4. TTS engine generates audio (async job + cached reuse).
5. Frontend plays audio and drives avatar lip-sync/emotion updates.

## Project Structure

```text
kei-ai-v2/
	whisper_server.py      # API endpoints: /transcribe, /chat, /audio-jobs
	ai_engine.py           # LLM prompt + OpenRouter integration
	tts.py                 # Qwen/Windows TTS implementations
	src/
		main.js              # App bootstrap
		speech.js            # Chat form + backend requests
		audio.js             # Audio player + analyser for lip-sync
		avatar.js            # Three.js VRM loader and animation helpers
	data/memory/           # Character/user memory files
	generated_audio/       # Runtime-generated/cached audio outputs
	model/                 # Avatar assets
```

## Local Setup

### Prerequisites

- Python 3.x
- Node.js
- ffmpeg (recommended for audio tooling)

Windows install example:

```powershell
winget install ffmpeg
```

### 1. Install frontend dependencies

```bash
npm install
```

### 2. Create Python environment and install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment variables

Use a local `.env` file (not committed) with your own keys.

```powershell
$env:OPENROUTER_API_KEY='YOUR_KEY_HERE'
$env:OPENROUTER_MODEL='google/gemma-3-4b-it:free'
$env:OPENROUTER_ENDPOINT='https://api.openrouter.ai/v1/chat/completions'
```
### 4. Start backend (Terminal 1)

```bash
.venv\Scripts\activate
python whisper_server.py
```

On first run, Whisper downloads its model locally.

### 5. Start frontend (Terminal 2)

```bash
npx serve .
```

### 6. Open app

Visit `http://localhost:3000` (or the port shown in terminal output).

## Optional: Qwen3 Japanese TTS Voice Clone

To enable Qwen-based Japanese TTS in the backend:

```bash
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts
pip install -U qwen-tts soundfile
```

## Current Status

This is an active prototype focused on local development and rapid iteration. The codebase is structured to support future production hardening (auth, deployment config, monitoring, and test coverage).
