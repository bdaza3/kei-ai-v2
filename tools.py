"""Allowlisted runtime tools shared by the legacy loop and LangGraph."""

from datetime import datetime
import json
import os
import threading
from typing import Any, Callable

from monitor import ActivityMonitor, ActivitySnapshot, ActivityEntry
from timers import PomodoroTimer


_FOCUS_TIMER = PomodoroTimer()
_FOCUS_TIMER_LOCK = threading.Lock()
_TASKS_LOCK = threading.Lock()
_TASKS_PATH = os.path.join(os.path.dirname(__file__), "data", "focus_tasks.json")


def get_active_window() -> dict[str, object]:
    """Return the current foreground application and window details."""
    monitor = ActivityMonitor()
    snapshot = monitor.sample()
    return {
        "active_app": snapshot.active_app,
        "active_window_title": snapshot.active_window_title,
        "active_process_name": snapshot.active_process_name,
        "current_category": snapshot.current_category,
        "idle_seconds": round(snapshot.idle_seconds, 1),
    }


def get_activity_snapshot() -> dict[str, object]:
    """Return the current activity state and accumulated monitor metrics."""
    monitor = ActivityMonitor()
    return monitor.sample().to_dict()


def get_current_time() -> dict[str, str]:
    """Return the local time and timezone used by the assistant host."""
    now = datetime.now().astimezone()
    return {
        "local_time": now.isoformat(timespec="seconds"),
        "timezone": now.tzname() or "unknown",
    }


def get_focus_status() -> dict[str, object]:
    """Return desktop activity and the current focus-session state."""
    with _FOCUS_TIMER_LOCK:
        timer = _FOCUS_TIMER.snapshot()
    return {
        "activity": get_active_window(),
        "focus_session": timer,
        "tasks": _load_tasks(include_completed=False),
    }


def start_focus_session(minutes: int = 25, break_minutes: int = 5) -> dict[str, object]:
    """Start a Pomodoro work session with validated durations."""
    _validate_minutes(minutes, "minutes")
    _validate_minutes(break_minutes, "break_minutes")
    with _FOCUS_TIMER_LOCK:
        return _FOCUS_TIMER.start(minutes, break_minutes)


def stop_focus_session() -> dict[str, object]:
    """Stop the current focus session and return its final snapshot."""
    with _FOCUS_TIMER_LOCK:
        return _FOCUS_TIMER.stop()


def get_focus_session() -> dict[str, object]:
    """Return the current Pomodoro state without changing it."""
    with _FOCUS_TIMER_LOCK:
        return _FOCUS_TIMER.snapshot()


def add_focus_task(title: str, priority: str = "normal") -> dict[str, object]:
    """Persist a small actionable task for later focus follow-up."""
    clean_title = (title or "").strip()
    if not clean_title or len(clean_title) > 240:
        raise ValueError("title must contain 1-240 characters")
    clean_priority = (priority or "normal").strip().lower()
    if clean_priority not in {"low", "normal", "high"}:
        raise ValueError("priority must be low, normal, or high")
    tasks = _load_tasks(include_completed=True)
    task = {
        "id": max((int(item.get("id", 0)) for item in tasks), default=0) + 1,
        "title": clean_title,
        "priority": clean_priority,
        "completed": False,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    tasks.append(task)
    _save_tasks(tasks)
    return task


def list_focus_tasks(include_completed: bool = False) -> list[dict[str, object]]:
    """List persisted focus tasks, hiding completed tasks by default."""
    return _load_tasks(include_completed=include_completed)


def complete_focus_task(task_id: int) -> dict[str, object]:
    """Mark one persisted focus task complete."""
    try:
        requested_id = int(task_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("task_id must be an integer") from exc
    tasks = _load_tasks(include_completed=True)
    for task in tasks:
        if task.get("id") == requested_id:
            task["completed"] = True
            task["completed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
            _save_tasks(tasks)
            return task
    raise ValueError(f"focus task {requested_id} was not found")


def _validate_minutes(value: int, name: str) -> None:
    try:
        amount = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not 1 <= amount <= 240:
        raise ValueError(f"{name} must be between 1 and 240")


def _load_tasks(*, include_completed: bool) -> list[dict[str, object]]:
    with _TASKS_LOCK:
        try:
            with open(_TASKS_PATH, "r", encoding="utf-8") as handle:
                tasks = json.load(handle)
        except (FileNotFoundError, json.JSONDecodeError):
            tasks = []
    if not isinstance(tasks, list):
        return []
    return [task for task in tasks if include_completed or not task.get("completed")]


def _save_tasks(tasks: list[dict[str, object]]) -> None:
    directory = os.path.dirname(_TASKS_PATH)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with _TASKS_LOCK:
        temporary_path = f"{_TASKS_PATH}.tmp"
        with open(temporary_path, "w", encoding="utf-8") as handle:
            json.dump(tasks, handle, ensure_ascii=False, indent=2)
        os.replace(temporary_path, _TASKS_PATH)


TOOL_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "get_active_window": get_active_window,
    "get_activity_snapshot": get_activity_snapshot,
    "get_current_time": get_current_time,
    "get_focus_status": get_focus_status,
    "start_focus_session": start_focus_session,
    "stop_focus_session": stop_focus_session,
    "get_focus_session": get_focus_session,
    "add_focus_task": add_focus_task,
    "list_focus_tasks": list_focus_tasks,
    "complete_focus_task": complete_focus_task,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_active_window",
            "description": "Get the current active app and window title.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_activity_snapshot",
            "description": "Get current activity classification, idle time, streaks, and recent window history.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the assistant host's current local time and timezone.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_focus_status",
            "description": "Read current desktop activity, focus-session state, and open focus tasks.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "start_focus_session",
            "description": "Start a Pomodoro work session only when the user asks to start focusing or a timer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "minutes": {"type": "integer", "minimum": 1, "maximum": 240},
                    "break_minutes": {"type": "integer", "minimum": 1, "maximum": 240},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stop_focus_session",
            "description": "Stop the current Pomodoro work session when the user asks to stop it.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_focus_session",
            "description": "Read the current Pomodoro state without changing it.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_focus_task",
            "description": "Save an actionable task when the user asks Kei to remember or track it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "minLength": 1, "maxLength": 240},
                    "priority": {"type": "string", "enum": ["low", "normal", "high"]},
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_focus_tasks",
            "description": "List open focus tasks, or all tasks if include_completed is true.",
            "parameters": {
                "type": "object",
                "properties": {"include_completed": {"type": "boolean"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "complete_focus_task",
            "description": "Mark a tracked focus task complete after the user confirms it is done.",
            "parameters": {
                "type": "object",
                "properties": {"task_id": {"type": "integer"}},
                "required": ["task_id"],
            },
        },
    },
]


def execute_tool(name: str, arguments: dict[str, Any] | None = None) -> Any:
    """Execute one allowlisted tool after the model has selected its name."""
    try:
        function = TOOL_FUNCTIONS[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported tool: {name}") from exc
    return function(**(arguments or {}))


