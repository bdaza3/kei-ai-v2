"""VoiceVox text-to-speech integration."""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
import threading
from typing import Any, Optional

import numpy as np


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


class QwenJapaneseTTS:
    """Qwen3-TTS voice cloning for low-latency Japanese speech generation.

    Key performance behavior:
    - model is loaded once
    - voice clone prompt is built once and reused
    - generated clips are cached by text hash
    """

    def __init__(self) -> None:
        self.enabled = os.environ.get("YUUKA_ENABLE_QWEN_TTS", "0").lower() in ("1", "true", "yes")
        self.model_name = os.environ.get("YUUKA_QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
        self.ref_audio = os.environ.get("YUUKA_QWEN_TTS_REF_AUDIO", "")
        self.ref_text = os.environ.get("YUUKA_QWEN_TTS_REF_TEXT", "")
        self.instruct = os.environ.get("YUUKA_QWEN_TTS_INSTRUCT", "Speak softly and naturally.")
        self.language = os.environ.get("YUUKA_QWEN_TTS_LANGUAGE", "Japanese")

        self._model: Optional[Any] = None
        self._voice_prompt: Optional[Any] = None
        self._soundfile: Optional[Any] = None
        self._lock = threading.Lock()
        self._cache: dict[str, Path] = {}

    def _ensure_loaded(self) -> bool:
        if not self.enabled:
            return False

        with self._lock:
            if self._model is not None and self._voice_prompt is not None:
                return True

            try:
                import torch
                from qwen_tts import Qwen3TTSModel
                import soundfile as sf
            except Exception:
                logging.exception("Qwen3-TTS dependencies are missing. Install qwen-tts and soundfile.")
                return False

            attn_impl = os.environ.get("YUUKA_QWEN_TTS_ATTN_IMPL", "flash_attention_2")
            dtype_name = os.environ.get("YUUKA_QWEN_TTS_DTYPE", "bfloat16").lower()
            device_map = os.environ.get("YUUKA_QWEN_TTS_DEVICE_MAP", "auto")

            cuda_available = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())

            dtype = torch.bfloat16
            if dtype_name == "float16":
                dtype = torch.float16
            elif dtype_name == "float32":
                dtype = torch.float32

            if not cuda_available:
                # Torch without CUDA cannot use flash attention or CUDA device maps.
                # Force a CPU-safe load path instead of failing the whole TTS backend.
                if device_map != "cpu":
                    logging.info("Qwen3-TTS: CUDA is unavailable; forcing device_map=cpu")
                device_map = "cpu"
                attn_impl = "eager"
                dtype = torch.float32

            load_attempts = []
            if cuda_available and attn_impl:
                load_attempts.append({"attn_implementation": attn_impl})
            # FlashAttention is great when available, but Windows installs often lack it.
            # Fall back to the safer PyTorch attention path automatically.
            load_attempts.append({"attn_implementation": "eager"})
            load_attempts.append({})

            model_loaded = False
            last_error: Exception | None = None
            for extra_kwargs in load_attempts:
                try:
                    kwargs = {
                        "device_map": device_map,
                        "dtype": dtype,
                    }
                    kwargs.update(extra_kwargs)
                    self._model = Qwen3TTSModel.from_pretrained(self.model_name, **kwargs)
                    self._soundfile = sf
                    model_loaded = True
                    break
                except Exception as exc:
                    last_error = exc
                    logging.warning(
                        "Qwen3-TTS load attempt failed for %s with kwargs=%s",
                        self.model_name,
                        extra_kwargs,
                        exc_info=True,
                    )

            if not model_loaded:
                logging.error("Failed to load Qwen3-TTS model: %s", self.model_name, exc_info=last_error)
                self._model = None
                return False

            if not self.ref_audio or not self.ref_text:
                logging.warning(
                    "Qwen3-TTS ref audio/text not set. Configure YUUKA_QWEN_TTS_REF_AUDIO and YUUKA_QWEN_TTS_REF_TEXT."
                )
                return False

            ref_path = Path(self.ref_audio)
            if not ref_path.is_absolute():
                ref_path = Path(os.getcwd()) / ref_path
            if not ref_path.exists():
                logging.warning("Qwen3-TTS ref audio file not found: %s", ref_path)
                return False

            try:
                self._voice_prompt = self._model.create_voice_clone_prompt(
                    ref_audio=str(ref_path),
                    ref_text=self.ref_text,
                )
            except Exception:
                logging.exception("Failed to create Qwen3-TTS voice clone prompt")
                self._voice_prompt = None
                return False

            logging.info("Qwen3-TTS initialized successfully with reusable clone prompt")
            return True

    def _text_key(self, text: str) -> str:
        return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:16]

    def _expand_short_text(self, text: str) -> str:
        cleaned = (text or "").strip()
        if not cleaned:
            return cleaned

        # If the model returns a tiny line, add a short natural follow-up so the clip is not just 0-1s.
        if len(cleaned) < 18:
            if cleaned.endswith("。"):
                return cleaned + " もう少し落ち着いて話すから、ちゃんと聞いてください。"
            return cleaned + "。もう少し落ち着いて話すから、ちゃんと聞いてください。"
        return cleaned

    def generate_to_file(self, text: str, output_dir: Path) -> Optional[Path]:
        text = (text or "").strip()
        if not text:
            return None
        if not self._ensure_loaded():
            return None

        text = self._expand_short_text(text)

        key = self._text_key(text)
        with self._lock:
            cached = self._cache.get(key)
            if cached and cached.exists():
                return cached

        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"qwen_jp_{key}.wav"
        if out_path.exists():
            with self._lock:
                self._cache[key] = out_path
            return out_path

        try:
            wavs, sample_rate = self._model.generate_voice_clone(
                text=text,
                language=self.language,
                voice_clone_prompt=self._voice_prompt,
                instruct=self.instruct,
            )

            if isinstance(wavs, (list, tuple)):
                chunks = []
                for chunk in wavs:
                    if chunk is None:
                        continue
                    arr = np.asarray(chunk)
                    if arr.size:
                        chunks.append(arr.reshape(-1))
                wav = np.concatenate(chunks) if chunks else np.asarray([], dtype=np.float32)
            else:
                wav = np.asarray(wavs).reshape(-1)

            if wav.size == 0:
                logging.warning("Qwen3-TTS produced empty audio for text: %s", text)
                return None

            wav = wav.astype(np.float32, copy=False)
            self._soundfile.write(str(out_path), wav, sample_rate, format="WAV", subtype="PCM_16")
            with self._lock:
                self._cache[key] = out_path
            return out_path
        except Exception:
            logging.exception("Qwen3-TTS generation failed")
            return None


_QWEN_TTS_SINGLETON: Optional[QwenJapaneseTTS] = None
_QWEN_TTS_LOCK = threading.Lock()


def get_qwen_japanese_tts() -> QwenJapaneseTTS:
    global _QWEN_TTS_SINGLETON
    with _QWEN_TTS_LOCK:
        if _QWEN_TTS_SINGLETON is None:
            _QWEN_TTS_SINGLETON = QwenJapaneseTTS()
        return _QWEN_TTS_SINGLETON
