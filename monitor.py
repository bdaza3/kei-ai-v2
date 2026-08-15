"""Desktop activity monitoring."""
#Low level desktop activity monitoring for Windows

from __future__ import annotations

import ctypes
import ctypes.wintypes
import os
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Callable, Deque, Iterable, Optional

DEFAULT_DISTRACTING_APPS = (
    "youtube",
    "netflix",
    "steam",
    "reddit",
    "twitter",
    "x",
    "twitch",
    "discord",
    "game",
)
DEFAULT_PRODUCTIVE_APPS = (
    "code",
    "cursor",
    "pycharm",
    "sublime",
    "terminal",
    "powershell",
    "cmd",
    "python",
    "notion",
    "obsidian",
    "figma",
)


@dataclass
class ActivityEntry:
    app_name: str
    process_name: str
    window_title: str
    category: str
    started_at: float
    ended_at: float
    duration_seconds: float


@dataclass
class ActivitySnapshot:
    active_app: str
    active_window_title: str
    active_process_name: str
    current_category: str
    idle_seconds: float
    distraction_seconds: float
    productive_seconds: float
    productive_streak_seconds: float
    distracting_streak_seconds: float
    current_window_seconds: float
    recent_entries: tuple[ActivityEntry, ...]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["recent_entries"] = [asdict(entry) for entry in self.recent_entries]
        return data


class ActivityMonitor:
    def __init__(
        self,
        distracting_apps: Optional[Iterable[str]] = None,
        productive_apps: Optional[Iterable[str]] = None,
        idle_threshold_seconds: int = 120,
        history_limit: int = 100,
        on_entry_logged: Optional[Callable[[ActivityEntry], None]] = None,
    ) -> None:
        self.distracting_apps = {app.lower() for app in (distracting_apps or DEFAULT_DISTRACTING_APPS)}
        self.productive_apps = {app.lower() for app in (productive_apps or DEFAULT_PRODUCTIVE_APPS)}
        self.idle_threshold_seconds = idle_threshold_seconds
        now = time.time()
        self._last_input_time = now
        self._last_sample_time = now
        self._distraction_seconds = 0.0
        self._productive_seconds = 0.0
        self._productive_streak_seconds = 0.0
        self._distracting_streak_seconds = 0.0
        self._keyboard_listener = None
        self._mouse_listener = None
        self._recent_entries: Deque[ActivityEntry] = deque(maxlen=history_limit)
        self._current_started_at = now
        self._current_title = "Unknown"
        self._current_process_name = "unknown"
        self._current_app = "Unknown"
        self._current_category = "neutral"
        self._on_entry_logged = on_entry_logged
        self._setup_input_hooks()

    def _setup_input_hooks(self) -> None:
        """Best effort input tracking via pynput when available."""
        try:
            from pynput import keyboard, mouse  # type: ignore
        except Exception:
            return

        def on_activity(*_args, **_kwargs) -> None:
            self._last_input_time = time.time()

        try:
            self._keyboard_listener = keyboard.Listener(on_press=on_activity)
            self._keyboard_listener.daemon = True
            self._keyboard_listener.start()
            self._mouse_listener = mouse.Listener(on_move=on_activity, on_click=on_activity, on_scroll=on_activity)
            self._mouse_listener.daemon = True
            self._mouse_listener.start()
        except Exception:
            self._keyboard_listener = None
            self._mouse_listener = None

    def _get_idle_seconds_windows(self) -> Optional[float]:
        try:
            class LASTINPUTINFO(ctypes.Structure):
                _fields_ = [("cbSize", ctypes.wintypes.UINT), ("dwTime", ctypes.wintypes.DWORD)]

            info = LASTINPUTINFO()
            info.cbSize = ctypes.sizeof(LASTINPUTINFO)
            if ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)) == 0:
                return None
            elapsed_ms = ctypes.windll.kernel32.GetTickCount64() - info.dwTime
            return max(0.0, float(elapsed_ms) / 1000.0)
        except Exception:
            return None

    def _get_foreground_window_details_windows(self) -> tuple[str, str]:
        try:
            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32
            hwnd = user32.GetForegroundWindow()
            if not hwnd:
                return "Unknown", "unknown"

            length = user32.GetWindowTextLengthW(hwnd)
            title_buffer = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, title_buffer, length + 1)
            title = title_buffer.value.strip() or "Unknown"

            pid = ctypes.wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            process_name = "unknown"
            if pid.value:
                process = kernel32.OpenProcess(0x1000, False, pid.value)
                if process:
                    try:
                        image_buffer = ctypes.create_unicode_buffer(1024)
                        size = ctypes.wintypes.DWORD(len(image_buffer))
                        if kernel32.QueryFullProcessImageNameW(process, 0, image_buffer, ctypes.byref(size)):
                            process_name = os.path.splitext(os.path.basename(image_buffer.value))[0].lower()
                    finally:
                        kernel32.CloseHandle(process)

            return title, process_name
        except Exception:
            return "Unknown", "unknown"

    def get_active_window_details(self) -> tuple[str, str]:
        """Return active window title and process name when available."""
        title, process_name = self._get_foreground_window_details_windows()
        if title != "Unknown" or process_name != "unknown":
            return title, process_name

        try:
            import pygetwindow as gw  # type: ignore

            fallback_title = gw.getActiveWindowTitle()
            if fallback_title:
                return fallback_title, "unknown"
        except Exception:
            pass
        return "Unknown", "unknown"

    def get_idle_seconds(self) -> float:
        system_idle = self._get_idle_seconds_windows()
        if system_idle is not None:
            return system_idle
        return max(0.0, time.time() - self._last_input_time)

    def _classify(self, title: str, process_name: str, idle_seconds: float) -> str:
        if idle_seconds >= self.idle_threshold_seconds:
            return "idle"

        haystack = f"{process_name} {title}".lower()
        if any(name in haystack for name in self.distracting_apps):
            return "distracting"
        if any(name in haystack for name in self.productive_apps):
            return "productive"
        return "neutral"

    def _commit_current_entry(self, ended_at: float) -> None:
        duration = max(0.0, ended_at - self._current_started_at)
        if duration <= 0.0:
            return

        entry = ActivityEntry(
            app_name=self._current_app,
            process_name=self._current_process_name,
            window_title=self._current_title,
            category=self._current_category,
            started_at=self._current_started_at,
            ended_at=ended_at,
            duration_seconds=duration,
        )
        self._recent_entries.append(entry)
        if self._on_entry_logged is not None:
            self._on_entry_logged(entry)

    def sample(self) -> ActivitySnapshot:
        now = time.time()
        delta = max(0.0, now - self._last_sample_time)
        self._last_sample_time = now

        title, process_name = self.get_active_window_details()
        app_name = process_name if process_name != "unknown" else title
        idle_seconds = self.get_idle_seconds()
        category = self._classify(title, process_name, idle_seconds)

        title_changed = title != self._current_title or process_name != self._current_process_name
        if title_changed:
            self._commit_current_entry(now)
            self._current_started_at = now

        self._current_title = title
        self._current_process_name = process_name
        self._current_app = app_name
        self._current_category = category

        if category == "distracting":
            self._distraction_seconds += delta
            self._distracting_streak_seconds += delta
        else:
            self._distracting_streak_seconds = 0.0

        if category == "productive":
            self._productive_seconds += delta
            self._productive_streak_seconds += delta
        else:
            self._productive_streak_seconds = 0.0

        return ActivitySnapshot(
            active_app=app_name,
            active_window_title=title,
            active_process_name=process_name,
            current_category=category,
            idle_seconds=idle_seconds,
            distraction_seconds=self._distraction_seconds,
            productive_seconds=self._productive_seconds,
            productive_streak_seconds=self._productive_streak_seconds,
            distracting_streak_seconds=self._distracting_streak_seconds,
            current_window_seconds=max(0.0, now - self._current_started_at),
            recent_entries=tuple(self._recent_entries)[-5:],
        )
