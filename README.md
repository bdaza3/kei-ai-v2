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
