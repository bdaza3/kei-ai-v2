"""VoiceVox text-to-speech integration."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
import threading


class VoiceVoxTTS:
    def __init__(self, host: str = "127.0.0.1", port: int = 50021, speaker: int = 1) -> None:
        self.host = host
        self.port = port
        self.speaker = speaker

    @property
    def _base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _post(self, path: str, params: dict, body: bytes = b"") -> bytes:
        query = urllib.parse.urlencode(params)
        request = urllib.request.Request(
            f"{self._base_url}{path}?{query}",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.read()

    def synthesize(self, text: str) -> bytes:
        audio_query = self._post("/audio_query", {"text": text, "speaker": self.speaker})
        return self._post("/synthesis", {"speaker": self.speaker}, body=audio_query)

    def _play_wav_file(self, wav_path: Path) -> None:
        if os.name == "nt":
            try:
                import winsound

                winsound.PlaySound(str(wav_path), winsound.SND_FILENAME)
                return
            except Exception:
                pass
        player = shutil.which("aplay") or shutil.which("afplay")
        if player:
            subprocess.run([player, str(wav_path)], check=False)

    def speak(self, text: str) -> None:
        if not text.strip():
            return

        temp_path: Path | None = None
        try:
            wav_bytes = self.synthesize(text)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
                temp_wav.write(wav_bytes)
                temp_path = Path(temp_wav.name)
            self._play_wav_file(temp_path)
        except Exception:
            # Graceful fallback keeps local loop running even if VoiceVox is down.
            print(f"[TTS fallback] {text}")
        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except Exception:
                    pass

    def speak_async(self, text: str) -> None:
        """Non-blocking wrapper around `speak` that runs synthesis/playback on a daemon thread."""
        if not text.strip():
            return
        threading.Thread(target=self.speak, args=(text,), daemon=True).start()
