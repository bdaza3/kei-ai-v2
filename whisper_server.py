import django
from django.conf import settings
from django.urls import path, re_path
from django.http import JsonResponse, HttpResponse, HttpResponseForbidden, HttpResponseNotFound
from django.views.decorators.csrf import csrf_exempt
from pathlib import Path
import mimetypes
import whisper, tempfile, os, sys
import json
import logging
import hashlib
import threading
import time

from ai_engine import generate_openrouter_bilingual_reply
from tts import get_qwen_japanese_tts, get_windows_sapi_tts

settings.configure(
    DEBUG=True,
    SECRET_KEY="local-whisper-server-secret",
    ALLOWED_HOSTS=["*"],
    ROOT_URLCONF=__name__,
)

print("Loading Whisper model...")
model = whisper.load_model("small") # oginally base
print("Whisper model ready.")

PROJECT_ROOT = Path(__file__).resolve().parent
GENERATED_AUDIO_DIR = PROJECT_ROOT / "generated_audio"
_AUDIO_JOBS: dict[str, dict[str, object]] = {}
_AUDIO_JOBS_LOCK = threading.Lock()


def _strip_wrapped_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and ((value[0] == "'" and value[-1] == "'") or (value[0] == '"' and value[-1] == '"')):
        return value[1:-1]
    return value


def _load_env_file(env_path: Path) -> None:
    """Load environment variables from .env if present.

    Supports both standard dotenv `KEY=VALUE` and PowerShell style
    `$env:KEY='VALUE'` lines.
    """
    if not env_path.exists() or not env_path.is_file():
        return

    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("$env:"):
                # Example: $env:OPENROUTER_API_KEY='sk-...'
                tail = line[5:]
                if "=" not in tail:
                    continue
                key, value = tail.split("=", 1)
                key = key.strip()
                value = _strip_wrapped_quotes(value.split("#", 1)[0].strip())
                if key and key not in os.environ:
                    os.environ[key] = value
                continue

            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = _strip_wrapped_quotes(value.split("#", 1)[0].strip())
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception:
        print("Warning: failed to load .env file")


_load_env_file(PROJECT_ROOT / ".env")


def _audio_job_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _job_audio_url(path: Path) -> str:
    rel = path.relative_to(PROJECT_ROOT).as_posix()
    return f"/{rel}"


def _upsert_audio_job(job_id: str, **values: object) -> dict[str, object]:
    with _AUDIO_JOBS_LOCK:
        job = _AUDIO_JOBS.get(job_id, {"job_id": job_id, "status": "pending", "updated_at": time.time()})
        job.update(values)
        job["updated_at"] = time.time()
        _AUDIO_JOBS[job_id] = job
        return dict(job)


def _get_audio_job(job_id: str) -> dict[str, object] | None:
    with _AUDIO_JOBS_LOCK:
        job = _AUDIO_JOBS.get(job_id)
        return dict(job) if job else None


def _generate_audio_job(job_id: str, text: str) -> None:
    _upsert_audio_job(job_id, status="running", audio_url=None, tts_error=None)
    try:
        wav_path = None
        tts_status = {}

        qwen_engine = get_qwen_japanese_tts()
        wav_path = qwen_engine.generate_to_file(text, GENERATED_AUDIO_DIR)
        tts_status = qwen_engine.status()

        if wav_path is None:
            fallback_engine = get_windows_sapi_tts()
            fallback_path = fallback_engine.generate_to_file(text, GENERATED_AUDIO_DIR)
            if fallback_path is not None:
                wav_path = fallback_path
                tts_status = fallback_engine.status()
                tts_status["fallback"] = "windows_sapi"

        if wav_path is not None:
            _upsert_audio_job(
                job_id,
                status="ready",
                audio_url=_job_audio_url(wav_path),
                tts_status=tts_status,
                tts_error=None,
            )
            return

        _upsert_audio_job(
            job_id,
            status="error",
            audio_url=None,
            tts_status=tts_status,
            tts_error=(
                "Japanese TTS did not generate usable audio. Check Qwen TTS env vars, reference voice settings, "
                "or verify Windows SAPI is available on this machine."
            ),
        )
    except Exception:
        logging.exception("Failed generating async TTS audio")
        _upsert_audio_job(
            job_id,
            status="error",
            audio_url=None,
            tts_error="Japanese TTS failed inside the Python backend. Check backend logs for details.",
        )


def _get_or_start_audio_job(text: str) -> dict[str, object]:
    job_id = _audio_job_id(text)
    existing = _get_audio_job(job_id)
    if existing and existing.get("status") in {"pending", "running", "ready"}:
        return existing

    qwen_engine = get_qwen_japanese_tts()
    cached = qwen_engine.get_cached_file(text, GENERATED_AUDIO_DIR)
    if cached is not None:
        return _upsert_audio_job(
            job_id,
            status="ready",
            audio_url=_job_audio_url(cached),
            tts_status=qwen_engine.status(),
            tts_error=None,
        )

    fallback_engine = get_windows_sapi_tts()
    fallback_cached = fallback_engine.get_cached_file(text, GENERATED_AUDIO_DIR)
    if fallback_cached is not None:
        status = fallback_engine.status()
        status["fallback"] = "windows_sapi"
        return _upsert_audio_job(
            job_id,
            status="ready",
            audio_url=_job_audio_url(fallback_cached),
            tts_status=status,
            tts_error=None,
        )

    job = _upsert_audio_job(job_id, status="pending", audio_url=None, tts_error=None, tts_status={})
    threading.Thread(target=_generate_audio_job, args=(job_id, text), daemon=True).start()
    return job

@csrf_exempt
def transcribe(request):
    if request.method == "OPTIONS":
        response = JsonResponse({})
    elif request.method == "POST":
        audio = request.FILES.get("file")
        if not audio:
            response = JsonResponse({"error": "No file provided"}, status=400)
        else:
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
                    for chunk in audio.chunks():
                        tmp.write(chunk)
                    tmp_path = tmp.name
                
                file_size = os.path.getsize(tmp_path)
                print(f"\n=== Transcription Request ===")
                print(f"Audio file: {tmp_path}")
                print(f"File size: {file_size} bytes")
                
                result = model.transcribe(tmp_path, fp16=False)
                
                print(f"✓ Transcription: '{result['text']}'")
                response = JsonResponse({"text": result["text"]})
                
            except Exception as e:
                print(f"\n✗ TRANSCRIPTION ERROR:")
                print(f"  Type: {type(e).__name__}")
                print(f"  Message: {str(e)}")
                import traceback
                traceback.print_exc()
                response = JsonResponse({"error": str(e)}, status=500)
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
    else:
        response = JsonResponse({"error": "Method not allowed"}, status=405)

    response["Access-Control-Allow-Origin"] = "*"
    response["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    response["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@csrf_exempt
def chat(request):
    if request.method == "OPTIONS":
        response = JsonResponse({})
    elif request.method == "POST":
        text = ""
        try:
            if request.body:
                payload = json.loads(request.body.decode("utf-8"))
                if isinstance(payload, dict):
                    text = str(payload.get("text") or "").strip()
        except Exception:
            text = ""

        if not text:
            response = JsonResponse({"error": "No text provided"}, status=400)
        else:
            model_name = os.environ.get("OPENROUTER_MODEL", "google/gemma-3-4b-it:free")
            try:
                timeout = float(os.environ.get("OPENROUTER_TIMEOUT", "12"))
            except Exception:
                timeout = 12.0

            reply = generate_openrouter_bilingual_reply(
                text,
                context={"source": "web_voice_input"},
                model=model_name,
                timeout=timeout,
            )

            if reply:
                japanese_text = str(reply.get("japanese", "") or "").strip()
                english_text = str(reply.get("english", "") or "").strip()

                audio_url = None
                audio_job_id = None
                audio_pending = False
                tts_error = None
                tts_status = {}
                if japanese_text:
                    try:
                        audio_job = _get_or_start_audio_job(japanese_text)
                        audio_job_id = str(audio_job.get("job_id") or "")
                        audio_url = str(audio_job.get("audio_url") or "") or None
                        tts_status = audio_job.get("tts_status") if isinstance(audio_job.get("tts_status"), dict) else {}
                        tts_error = str(audio_job.get("tts_error") or "") or None
                        audio_pending = audio_url is None and audio_job.get("status") in {"pending", "running"}
                    except Exception:
                        logging.exception("Failed generating Qwen3-TTS audio")
                        tts_error = "Japanese TTS failed inside the Python backend. Check backend logs for details."

                response = JsonResponse({
                    "japanese": japanese_text,
                    "english": english_text,
                    "audio_url": audio_url,
                    "audio_job_id": audio_job_id,
                    "audio_pending": audio_pending,
                    "tts_error": tts_error,
                    "tts_status": tts_status,
                    "model": model_name,
                })
            else:
                response = JsonResponse(
                    {
                        "error": "No response from OpenRouter. Verify OPENROUTER_API_KEY/OPENROUTER_MODEL and restart whisper_server.py after env changes.",
                        "model": model_name,
                    },
                    status=503,
                )
    else:
        response = JsonResponse({"error": "Method not allowed"}, status=405)

    response["Access-Control-Allow-Origin"] = "*"
    response["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    response["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@csrf_exempt
def audio_job_status(request, job_id: str):
    if request.method == "OPTIONS":
        response = JsonResponse({})
    elif request.method == "GET":
        job = _get_audio_job(job_id)
        if not job:
            response = JsonResponse({"error": "Audio job not found"}, status=404)
        else:
            response = JsonResponse(job)
    else:
        response = JsonResponse({"error": "Method not allowed"}, status=405)

    response["Access-Control-Allow-Origin"] = "*"
    response["Access-Control-Allow-Methods"] = "GET, OPTIONS"
    response["Access-Control-Allow-Headers"] = "Content-Type"
    return response

def _resolve_path(path_fragment):
    target = (PROJECT_ROOT / path_fragment.lstrip("/")).resolve()
    try:
        target.relative_to(PROJECT_ROOT)
    except ValueError:
        return None
    return target


def serve_frontend(request, path=""):
    target = _resolve_path(path)
    if target is None:
        return HttpResponseForbidden("Invalid path")

    if path in ("", "/") or target.is_dir():
        target = PROJECT_ROOT / "index.html"

    if not target.exists() or not target.is_file():
        return HttpResponseNotFound("Not found")

    content_type, _ = mimetypes.guess_type(str(target))
    with open(target, "rb") as f:
        response = HttpResponse(
            f.read(),
            content_type=content_type or "application/octet-stream"
        )
    if target.suffix in {".js", ".mjs"}:
        response["Content-Type"] = "application/javascript"

    return response


urlpatterns = [
    path("transcribe", transcribe),
    path("chat", chat),
    path("audio-jobs/<str:job_id>", audio_job_status),
    re_path(r"^(?P<path>.*)$", serve_frontend),
]

if __name__ == "__main__":
    threading.Thread(target=lambda: get_qwen_japanese_tts().warm_up(), daemon=True).start()
    sys.argv = ["whisper_server.py", "runserver", "5000", "--noreload"]
    from django.core.management import execute_from_command_line
    execute_from_command_line(sys.argv)

# Credits:
# OpenAI Whisper - Speech recognition model
# https://github.com/openai/whisper
# Radford, A., Kim, J.W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2022).
# "Robust Speech Recognition via Large-Scale Weak Supervision."
# Licensed under MIT License
