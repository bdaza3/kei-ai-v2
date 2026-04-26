Kei.AI V2

## Setup & Running

### Prerequisites
- Discrete GPU prefered, but as it is ran locally with the base model, your laptop should run it fine.
- VsCode if available
- Python 3.x, Vscode python extension recommended as well
- Node.js
- ffmpeg — install with `winget install ffmpeg`, then add it to your PATH windows environment variables, you can also install ffmpeg online.
- Optional to have python virtual environment, but used in this to do list.
### 1. Install Node dependencies
```bash
npm install
```

### 2. Set up Python environment
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
#^---This will install django backend utilities and OpenAi's whisper api
Then run:
pip install flask
pip install flask_cors
```

### 3. Start the Whisper server (Terminal 1)
```bash
.venv\Scripts\activate # Or download python extension on vscode and run python create environment
python whisper_server.py
```
First run will download the Whisper `base` model (~145 MB). Wait for:
`Whisper model ready.`

### 4. Start the frontend (Terminal 2)
```bash
npx serve .
```

### 5. Open the app
Go to `http://localhost:3000` in Chrome or Edge. Chromium Browser prefered. Sometimes, if port is taken already, it will use a different port which you will see in the terminal.

### Voice Commands

## Qwen3 Japanese TTS (Voice Clone)

This project now supports returning Japanese audio from the `/chat` endpoint using Qwen3-TTS, while showing English text in the UI.

### Install (recommended in a dedicated env)

```bash
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts
pip install -U qwen-tts
pip install soundfile
```

Optional GPU acceleration:

```bash
pip install flash-attn --no-build-isolation
```

### Required environment variables

Set these before starting `python whisper_server.py`:

Important:
- These must be set in the terminal that launches the Python backend, or saved in `.env`.
- Setting them only in the `npx serve .` terminal will not enable Qwen TTS, because that terminal only serves the frontend.

```powershell
$env:YUUKA_ENABLE_QWEN_TTS='1'
$env:YUUKA_QWEN_TTS_MODEL='Qwen/Qwen3-TTS-12Hz-1.7B-Base'
$env:YUUKA_QWEN_TTS_DEVICE_MAP='cpu'   # use 'cuda:0' only if your torch build has CUDA
$env:YUUKA_QWEN_TTS_DTYPE='float32'    # use bfloat16 only on supported GPU setups
$env:YUUKA_QWEN_TTS_ATTN_IMPL='eager'  # flash_attention_2 is optional, not required

$env:YUUKA_QWEN_TTS_REF_AUDIO='data/voice/your_reference_voice.ogg'
$env:YUUKA_QWEN_TTS_REF_TEXT='This is my voice speaking naturally for cloning.'
$env:YUUKA_QWEN_TTS_LANGUAGE='Japanese'
$env:YUUKA_QWEN_TTS_INSTRUCT='Speak softly and naturally.'
```

Notes:
- The voice clone prompt is created once and reused automatically for low latency.
- Generated audio files are cached under `generated_audio/`.
- If your PyTorch build does not have CUDA, keep `YUUKA_QWEN_TTS_DEVICE_MAP='cpu'`.
