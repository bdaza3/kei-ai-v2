"""Simple voice input client that sends short audio segments to a Whisper
HTTP server and prints transcripts to the terminal.

This implementation intentionally does NOT load any local Whisper model.
It records short segments with `sounddevice`, encodes them as WAV in-memory,
POSTs them to `YUUKA_WHISPER_REMOTE_URL` (or the env must be set by
`main.py` when autostarting the local server), and prints the returned
transcription. Transcripts are also enqueued on `transcript_queue` so the
rest of the assistant can consume them as before.
"""

from typing import Optional
import io
import os
import queue
import logging

import numpy as np
import sounddevice as sd
import soundfile as sf
import requests


class VoiceInput:
    def __init__(self, samplerate: int = 16000, channels: int = 1):
        self.samplerate = samplerate
        self.channels = channels
        self._stream: Optional[sd.InputStream] = None
        self._buffer: list = []
        self._recording = False
        self.transcript_queue: "queue.Queue[str]" = queue.Queue()
        self.remote_url: Optional[str] = os.environ.get("YUUKA_WHISPER_REMOTE_URL")
        if not self.remote_url:
            logging.warning("VoiceInput: no remote whisper server configured (YUUKA_WHISPER_REMOTE_URL)")

    def wait_loaded(self, timeout: Optional[float] = None) -> bool:
        """Compatibility shim -> this client has no local model to load."""
        return True

    def start_recording(self) -> None:
        if self._recording:
            return
        self._buffer = []
        self._stream = sd.InputStream(samplerate=self.samplerate, channels=self.channels, callback=self._callback)
        self._stream.start()
        self._recording = True
        logging.info("VoiceInput: started recording")

    def _callback(self, indata, frames, time_info, status):
        if status:
            logging.debug("InputStream status: %s", status)
        # copy frame buffer
        self._buffer.append(indata.copy())

    def stop_and_transcribe(self) -> None:
        if not self._recording:
            return
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass
        self._recording = False
        logging.info("VoiceInput: stopped recording, scheduling transcription")

        if not self._buffer:
            logging.info("VoiceInput: no audio frames captured; skipping transcription")
            return

        data = np.concatenate(self._buffer, axis=0)
        # Keep a single channel (do NOT average channels) — browser/Opus is already well-processed.
        if data.ndim > 1:
            audio = data[:, 0].copy()
        else:
            audio = data.copy()

        # skip very short audio
        min_samples = int(self.samplerate * 0.05)
        if getattr(audio, "size", 0) < min_samples:
            logging.info("VoiceInput: captured audio too short (%d samples), skipping", getattr(audio, "size", 0))
            self._buffer = []
            return

        # cast to float32 (do NOT perform gain-normalization here — keep raw waveform)
        try:
            audio = audio.astype(np.float32)
        except Exception:
            logging.exception("VoiceInput: error casting audio to float32")
            self._buffer = []
            return

        # debug info to help compare with browser-captured audio
        try:
            print("AUDIO DEBUG:", audio.shape, audio.dtype, float(np.min(audio)), float(np.max(audio)))
        except Exception:
            logging.exception("VoiceInput: failed to print audio debug info")

        # optional: save a WAV for offline inspection if env enabled
        try:
            if os.environ.get("YUUKA_WHISPER_SAVE_TEST_WAV", "0").lower() in ("1", "true", "yes"):
                debug_path = os.environ.get("YUUKA_WHISPER_TEST_WAV_PATH", "debug_audio.wav")
                sf.write(debug_path, audio, self.samplerate, format="WAV", subtype="PCM_16")
                logging.info("VoiceInput: saved debug WAV to %s", debug_path)
        except Exception:
            logging.exception("VoiceInput: failed to write debug WAV")

        remote = self.remote_url or os.environ.get("YUUKA_WHISPER_REMOTE_URL")
        if not remote:
            logging.error("VoiceInput: no remote whisper URL configured; set YUUKA_WHISPER_REMOTE_URL")
            self._buffer = []
            return

        try:
            buf = io.BytesIO()
            # Use 16-bit PCM WAV (closer to browser output quality)
            sf.write(buf, audio, self.samplerate, format="WAV", subtype="PCM_16")
            buf.seek(0)
            files = {"file": ("audio.wav", buf, "audio/wav")}
            resp = requests.post(remote.rstrip("/") + "/transcribe", files=files, timeout=15)
            resp.raise_for_status()
            j = resp.json()
            text = j.get("text") or j.get("result") or ""
            if text:
                text = text.strip()
                # print to terminal for immediate visibility
                print(f"Transcript: {text}")
                try:
                    self.transcript_queue.put(text)
                except Exception:
                    logging.exception("VoiceInput: failed to queue transcript")
            else:
                logging.info("VoiceInput: remote returned empty transcript")
        except Exception:
            logging.exception("VoiceInput: remote transcription failed; skipping (no local fallback)")
        finally:
            self._buffer = []

    def get_nowait(self) -> Optional[str]:
        try:
            return self.transcript_queue.get_nowait()
        except queue.Empty:
            return None
