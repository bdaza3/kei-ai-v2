"""Entry point for the local focus assistant."""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import subprocess, sys

from ai_engine import generate_conversation_reply, generate_with_ollama, get_template_response
from brain import DecisionConfig, DecisionEngine
from event_logger import AssistantEventLogger
from monitor import ActivityMonitor, ActivitySnapshot
from timers import PomodoroTimer
from tts import VoiceVoxTTS
from ui import PopupOverlay
from voice_input import VoiceInput
from command_handler import normalize_text


def configure_logging(logs_dir: str = "logs") -> None:
    os.makedirs(logs_dir, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(os.path.join(logs_dir, "assistant.log"), encoding="utf-8")
    file_handler.setFormatter(formatter)

    root.addHandler(stream_handler)
    root.addHandler(file_handler)


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _build_persistent_status(activity: ActivitySnapshot, pomodoro: PomodoroTimer) -> str:
    timer = pomodoro.snapshot()
    base = (
        f"Tracking {activity.active_app} | {activity.current_category} | "
        f"Window {_format_duration(activity.current_window_seconds)}"
    )
    if timer["state"] != "idle":
        base += f" | Pomodoro {timer['state']} {_format_duration(timer['remaining_seconds'])} left"
    return base


def run_assistant(poll_interval: float = 1.0) -> None:
    configure_logging()
    # Optionally autostart a local Whisper server (provides /transcribe endpoint)
    proc_whisper_server = None
    autostart = os.environ.get("YUUKA_WHISPER_AUTOSTART_SERVER", "0").lower() in ("1", "true", "yes")
    if autostart and not os.environ.get("YUUKA_WHISPER_REMOTE_URL"):
        server_script = os.path.join(os.path.dirname(__file__), "whisper_server.py")
        def _start_local_whisper():
            if not os.path.exists(server_script):
                logging.warning("Local whisper server script not found: %s", server_script)
                return None
            try:
                proc = subprocess.Popen([sys.executable, server_script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                logging.info("Started local whisper server (PID %s)", proc.pid)
                return proc
            except Exception:
                logging.exception("Failed to start local whisper server")
                return None
        proc_whisper_server = _start_local_whisper()
        if proc_whisper_server:
            port = int(os.environ.get('YUUKA_WHISPER_REMOTE_PORT','5000'))
            remote_base = f"http://127.0.0.1:{port}"
            os.environ["YUUKA_WHISPER_REMOTE_URL"] = remote_base
            # wait for the server to become ready (poll /transcribe OPTIONS)
            import requests as _requests
            ready = False
            start_wait = time.time()
            startup_timeout = float(os.environ.get("YUUKA_WHISPER_STARTUP_TIMEOUT", "30"))
            while time.time() - start_wait < startup_timeout:
                # if the process died, stop waiting
                if proc_whisper_server.poll() is not None:
                    logging.error("Local whisper server exited with code %s", proc_whisper_server.returncode)
                    break
                try:
                    resp = _requests.options(remote_base + "/transcribe", timeout=2)
                    if resp.status_code < 500:
                        ready = True
                        logging.info("Local whisper server is ready at %s", remote_base)
                        break
                except Exception:
                    pass
                time.sleep(0.5)
            if not ready:
                logging.warning("Local whisper server did not respond within %.0f seconds", startup_timeout)
    event_logger = AssistantEventLogger()
    event_logger.log_event("assistant_started", {"poll_interval": poll_interval})

    monitor = ActivityMonitor(
        on_entry_logged=lambda entry: event_logger.log_event("activity_entry", {"entry": entry}),
    )
    voice = None
    try:
        voice = VoiceInput()
    except Exception as exc:
        logging.exception("Failed to initialize microphone input: %s", exc)

    def start_transcript_receiver(host: str = "127.0.0.1", port: int = 8765) -> None:
        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):  # noqa: A003
                return

            def do_OPTIONS(self):
                self.send_response(200)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.end_headers()

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length) if length else b""
                try:
                    payload = json.loads(body.decode("utf-8")) if body else {}
                except Exception:
                    payload = {}

                text = payload.get("text") if isinstance(payload, dict) else None
                if text:
                    try:
                        if voice is not None and hasattr(voice, "transcript_queue"):
                            voice.transcript_queue.put(str(text))
                            event_logger.log_event("external_transcript", {"text": str(text)})
                        else:
                            logging.warning("No transcript sink available; discarding text")
                    except Exception:
                        logging.exception("Error injecting transcript into voice input")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(b"{\"ok\": true}")
                    return

                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b"{\"error\": \"no text provided\"}")

        server = ThreadingHTTPServer((host, port), _Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        logging.info("Transcript receiver listening on http://%s:%d", host, port)

    start_transcript_receiver()

    stop_event = threading.Event()
    # segment_seconds controls how long the recorder records before sending
    # audio to transcription. Shorter values reduce perceived latency but
    # increase request frequency. Configure via env `YUUKA_WHISPER_SEGMENT_SECONDS`.
    try:
        segment_seconds = float(os.environ.get("YUUKA_WHISPER_SEGMENT_SECONDS", "1.0"))
    except Exception:
        segment_seconds = 1.0
    last_transcript: str | None = None
    last_transcript_time = 0.0
    dedup_seconds = 1.5

    wakeword_enabled = True
    wake_words = ["hey k", "hey", "kei", "key", "kay"]
    silence_timeout = 1.0
    is_awake = False
    speech_buffer: list[str] = []
    last_speech_time = 0.0

    def _auto_segmenting_transcribe(stop_evt: threading.Event, segment_sec: float = 2.0) -> None:
        if voice is None:
            return
        try:
            try:
                voice.start_recording()
            except Exception:
                logging.exception("Failed to start recording")

            while not stop_evt.is_set():
                stop_evt.wait(segment_sec)
                if stop_evt.is_set():
                    break
                try:
                    voice.stop_and_transcribe()
                except Exception:
                    logging.exception("Error during stop_and_transcribe")
                try:
                    voice.start_recording()
                except Exception:
                    logging.exception("Failed to restart recording")
        finally:
            try:
                if voice is not None:
                    voice.stop_and_transcribe()
            except Exception:
                pass

    if voice is not None:
        threading.Thread(target=_auto_segmenting_transcribe, args=(stop_event, segment_seconds), daemon=True).start()

    decision_engine = DecisionEngine(
        DecisionConfig(
            idle_trigger_seconds=120, #trigger idle if no active window or app for 2 minutes
            procrastination_trigger_seconds=300, #trigger procrastination if using non-work app for 5 minutes
            break_trigger_seconds=3000, #trigger break if using work app for 50 minutes
            cooldown_seconds=45, #minimum seconds between triggers to avoid spamming
            focused_ping_seconds=300, #periodically ping focused status every 5 minutes
        )
    )
    pomodoro = PomodoroTimer()
    tts = VoiceVoxTTS()
    overlay = PopupOverlay()
    overlay.show_persistent("Kei assistant started.")

    llm_result_queue: "queue.Queue[str]" = queue.Queue()
    last_hour_bucket = datetime.now().replace(minute=0, second=0, microsecond=0)
    last_status_refresh_at = 0.0
    avatar_path = os.path.join(os.path.dirname(__file__), "public", "images", "Kei_(Unarmed)_00.png")

    def speak_trigger(trigger: str, context: dict, *, allow_async_ollama: bool = True) -> None:
        event_logger.log_event("trigger", {"trigger": trigger, "context": context})
        message = get_template_response(trigger, context)
        overlay.show_dialogue(message, image_path=avatar_path if os.path.exists(avatar_path) else None)
        tts.speak_async(message)

        enabled = os.environ.get("YUUKA_ENABLE_OLLAMA", "0").lower() in ("1", "true", "yes")
        if allow_async_ollama and enabled and trigger not in {"conversation", "confirm", "pomodoro_started", "pomodoro_stopped"}:
            def _llm_worker(trig: str, ctx: dict, out_queue: "queue.Queue[str]") -> None:
                try:
                    candidate = generate_with_ollama(trig, ctx, model=os.environ.get("YUUKA_OLLAMA_MODEL"), timeout=6.0)
                    if candidate:
                        out_queue.put(candidate)
                except Exception:
                    logging.exception("Ollama background update failed")

            threading.Thread(target=_llm_worker, args=(trigger, context, llm_result_queue), daemon=True).start()

    try:
        while True:
            user_text = None
            if voice is not None:
                try:
                    user_text = voice.get_nowait()
                except Exception:
                    logging.exception("Error reading from voice input queue")
                    user_text = None

            if not user_text and voice is not None and hasattr(voice, "transcript_queue"):
                try:
                    user_text = voice.transcript_queue.get_nowait()
                except queue.Empty:
                    user_text = None

            if user_text:
                now_ts = time.time()
                text_stripped = user_text.strip()
                if last_transcript is not None and text_stripped == last_transcript and (now_ts - last_transcript_time) < dedup_seconds:
                    logging.info("Ignoring duplicate transcript: %s", text_stripped)
                    user_text = None
                else:
                    last_transcript = text_stripped
                    last_transcript_time = now_ts
                    event_logger.log_event("transcript", {"text": text_stripped})

                    if wakeword_enabled:
                        normalized = normalize_text(text_stripped)
                        if not is_awake:
                            matched = next((word for word in wake_words if word in normalized), None)
                            if matched:
                                logging.info("Wakeword detected: %s", matched)
                                is_awake = True
                                speech_buffer = []
                                cleaned = re.sub(rf"\b{re.escape(matched)}\b", "", normalized, count=1).strip()
                                if cleaned:
                                    speech_buffer.append(cleaned)
                                last_speech_time = now_ts
                                overlay.show_persistent("Listening...")
                                user_text = None
                            else:
                                user_text = None
                        else:
                            speech_buffer.append(normalized)
                            last_speech_time = now_ts
                            overlay.show_persistent("Listening...")
                            user_text = None

            if is_awake:
                now_check = time.time()
                if (now_check - last_speech_time) > silence_timeout and speech_buffer:
                    final_text = " ".join(speech_buffer).strip()
                    speech_buffer = []
                    is_awake = False
                    event_logger.log_event("utterance_finalized", {"text": final_text})
                    overlay.show_dialogue(final_text, duration_ms=4500, image_path=avatar_path if os.path.exists(avatar_path) else None)
                    user_text = final_text

            activity = monitor.sample()

            if time.time() - last_status_refresh_at >= 5.0:
                overlay.show_persistent(_build_persistent_status(activity, pomodoro))
                last_status_refresh_at = time.time()

            trigger, context = decision_engine.evaluate(activity, user_text=user_text)
            if trigger == "conversation": #if it's a conversation trigger, log the event before generating the reply to capture the context accurately
                event_logger.log_event("conversation", context)
                reply = generate_conversation_reply(context.get("user_text", ""), context)
                overlay.show_dialogue(reply, image_path=avatar_path if os.path.exists(avatar_path) else None)
                tts.speak_async(reply)
            elif trigger == "user_command": #if it's a user command, handle it and log the event with the command details
                command_name = context.get("command")
                if command_name == "start_pomodoro": #if it's a pomodoro command, handle it and log the event with the timer details
                    context.update(pomodoro.start(context.get("minutes"), context.get("break_minutes")))
                    speak_trigger("pomodoro_started", context, allow_async_ollama=False)
                elif command_name == "stop_pomodoro": #if it's a pomodoro stop command, handle it and log the event with the timer details
                    context.update(pomodoro.stop())
                    speak_trigger("pomodoro_stopped", context, allow_async_ollama=False)
                elif command_name == "status": #if it's a status command, log the event with the current status details
                    context.update(pomodoro.snapshot())
                    speak_trigger("user_command", context, allow_async_ollama=False)
                else:
                    speak_trigger("user_command", context)
            elif trigger:
                speak_trigger(trigger, context)

            for timer_event in pomodoro.poll():
                timer_context = dict(timer_event.context)
                timer_context.update(activity.to_dict())
                speak_trigger(timer_event.trigger, timer_context, allow_async_ollama=False)

            now_dt = datetime.now()
            current_hour_bucket = now_dt.replace(minute=0, second=0, microsecond=0)
            if current_hour_bucket != last_hour_bucket:
                last_hour_bucket = current_hour_bucket
                speak_trigger(
                    "hourly_chime",
                    {
                        "hour_label": now_dt.strftime("%I:%M %p").lstrip("0"),
                        **activity.to_dict(),
                    },
                    allow_async_ollama=False,
                )

            try: #process any pending LLM results
                while True:
                    candidate = llm_result_queue.get_nowait()
                    overlay.show_dialogue(candidate, image_path=avatar_path if os.path.exists(avatar_path) else None)
                    tts.speak_async(candidate)
                    event_logger.log_event("ollama_reply", {"text": candidate})
            except queue.Empty:
                pass

            overlay.process_events()
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        logging.info("Shutting down assistant")
    finally:
        stop_event.set()
        # terminate local whisper server if we started one
        try:
            if proc_whisper_server:
                try:
                    proc_whisper_server.terminate()
                    proc_whisper_server.wait(timeout=3)
                except Exception:
                    try:
                        proc_whisper_server.kill()
                    except Exception:
                        pass
        except Exception:
            pass
        event_logger.log_event("assistant_stopped", {})
        overlay.close()


if __name__ == "__main__":
    run_assistant()
