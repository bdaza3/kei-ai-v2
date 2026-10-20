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
_FOCUS_SESSION_CONFIGURED = False
_TASKS_LOCK = threading.RLock()
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
        _FOCUS_TIMER.poll()
        timer = _FOCUS_TIMER.snapshot()
    tasks = _load_tasks(include_completed=False)
    return {
        "activity": get_active_window(),
        "focus_session": timer,
        "active_task": _active_task(tasks),
        "next_task": _select_next_task(tasks),
        "tasks": tasks,
    }


def start_focus_session(minutes: int = 25, break_minutes: int = 5) -> dict[str, object]:
    """Start a Pomodoro work session with validated durations."""
    global _FOCUS_SESSION_CONFIGURED
    _validate_minutes(minutes, "minutes")
    _validate_minutes(break_minutes, "break_minutes")
    with _FOCUS_TIMER_LOCK:
        _FOCUS_TIMER.poll()
        if _FOCUS_TIMER.state in {"work", "break"}:
            raise ValueError("A focus session is already running; check its status or resume it after the break")
        _FOCUS_SESSION_CONFIGURED = True
        return _FOCUS_TIMER.start(minutes, break_minutes)


def stop_focus_session() -> dict[str, object]:
    """Stop the current focus session and return its final snapshot."""
    with _FOCUS_TIMER_LOCK:
        _FOCUS_TIMER.poll()
        return _FOCUS_TIMER.stop()


def get_focus_session() -> dict[str, object]:
    """Return the current Pomodoro state without changing it."""
    with _FOCUS_TIMER_LOCK:
        _FOCUS_TIMER.poll()
        return _FOCUS_TIMER.snapshot()


def resume_focus_session() -> dict[str, object]:
    """Resume work after a break using the previous session durations."""
    with _FOCUS_TIMER_LOCK:
        _FOCUS_TIMER.poll()
        if _FOCUS_TIMER.state == "work":
            return _FOCUS_TIMER.snapshot()
        if not _FOCUS_SESSION_CONFIGURED:
            raise ValueError("There is no previous focus session to resume; start a new session first")
        return _FOCUS_TIMER.start(_FOCUS_TIMER.work_minutes, _FOCUS_TIMER.break_minutes)


def add_focus_task(title: str, priority: str = "normal") -> dict[str, object]:
    """Persist a small actionable task for later focus follow-up."""
    clean_title = (title or "").strip()
    if not clean_title or len(clean_title) > 240:
        raise ValueError("title must contain 1-240 characters")
    clean_priority = (priority or "normal").strip().lower()
    if clean_priority not in {"low", "normal", "high"}:
        raise ValueError("priority must be low, normal, or high")
    with _TASKS_LOCK:
        tasks = _load_tasks(include_completed=True)
        task = {
            "id": max((int(item.get("id", 0)) for item in tasks), default=0) + 1,
            "title": clean_title,
            "priority": clean_priority,
            "status": "open",
            "completed": False,
            "progress_updates": [],
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        tasks.append(task)
        _save_tasks(tasks)
    return task


def list_focus_tasks(include_completed: bool = False) -> list[dict[str, object]]:
    """List persisted focus tasks, hiding completed tasks by default."""
    return _load_tasks(include_completed=include_completed)


def get_next_focus_task() -> dict[str, object]:
    """Recommend the current task or the highest-priority open task."""
    tasks = _load_tasks(include_completed=False)
    task = _select_next_task(tasks)
    if task is None:
        return {"task": None, "message": "There are no open focus tasks. Add a task to choose what to work on next."}
    return {"task": task, "message": "Continue the active task." if task.get("status") == "in_progress" else "Recommended next task."}


def start_focus_task(task_id: int) -> dict[str, object]:
    """Set one open task as active, pausing any other active task."""
    requested_id = _parse_task_id(task_id)
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    with _TASKS_LOCK:
        tasks = _load_tasks(include_completed=True)
        selected = next((item for item in tasks if item.get("id") == requested_id), None)
        if selected is None:
            raise ValueError(f"focus task {requested_id} was not found")
        if selected.get("completed") or selected.get("status") == "completed":
            raise ValueError(f"focus task {requested_id} is already complete")
        for task in tasks:
            if task is selected:
                continue
            if task.get("status") == "in_progress":
                task["status"] = "paused"
                task["paused_at"] = now
        selected["status"] = "in_progress"
        selected["started_at"] = selected.get("started_at") or now
        selected.pop("paused_at", None)
        selected["updated_at"] = now
        _save_tasks(tasks)
    return selected


def update_focus_task_progress(task_id: int, note: str) -> dict[str, object]:
    """Record a short user-reported progress update on an open task."""
    requested_id = _parse_task_id(task_id)
    clean_note = (note or "").strip()
    if not clean_note or len(clean_note) > 500:
        raise ValueError("note must contain 1-500 characters")
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    with _TASKS_LOCK:
        tasks = _load_tasks(include_completed=True)
        selected = next((item for item in tasks if item.get("id") == requested_id), None)
        if selected is None:
            raise ValueError(f"focus task {requested_id} was not found")
        if selected.get("completed") or selected.get("status") == "completed":
            raise ValueError(f"focus task {requested_id} is already complete")
        for task in tasks:
            if task is not selected and task.get("status") == "in_progress":
                task["status"] = "paused"
                task["paused_at"] = now
        selected["status"] = "in_progress"
        selected["started_at"] = selected.get("started_at") or now
        progress_updates = selected.get("progress_updates")
        if not isinstance(progress_updates, list):
            progress_updates = []
            selected["progress_updates"] = progress_updates
        progress_updates.append({"note": clean_note, "at": now})
        selected["updated_at"] = now
        _save_tasks(tasks)
    return selected


def complete_focus_task(task_id: int) -> dict[str, object]:
    """Mark one persisted focus task complete."""
    requested_id = _parse_task_id(task_id)
    with _TASKS_LOCK:
        tasks = _load_tasks(include_completed=True)
        for task in tasks:
            if task.get("id") == requested_id:
                if not task.get("completed") or task.get("status") != "completed":
                    now = datetime.now().astimezone().isoformat(timespec="seconds")
                    task["completed"] = True
                    task["status"] = "completed"
                    task["completed_at"] = now
                    task["updated_at"] = now
                    _save_tasks(tasks)
                return task
    raise ValueError(f"focus task {requested_id} was not found")


def _parse_task_id(task_id: int) -> int:
    try:
        return int(task_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("task_id must be an integer") from exc


def _active_task(tasks: list[dict[str, object]]) -> dict[str, object] | None:
    return next((task for task in tasks if task.get("status") == "in_progress"), None)


def _select_next_task(tasks: list[dict[str, object]]) -> dict[str, object] | None:
    active = _active_task(tasks)
    if active is not None:
        return active
    priorities = {"high": 0, "normal": 1, "low": 2}
    candidates = [task for task in tasks if not task.get("completed") and task.get("status") != "completed"]
    candidates.sort(
        key=lambda task: (
            priorities.get(str(task.get("priority", "normal")).lower(), 1),
            0 if task.get("status") == "paused" else 1,
            str(task.get("created_at", "")),
            int(task.get("id", 0)),
        )
    )
    return candidates[0] if candidates else None


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
    clean_tasks = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        if "status" not in task:
            task["status"] = "completed" if task.get("completed") else "open"
        if include_completed or (not task.get("completed") and task.get("status") != "completed"):
            clean_tasks.append(task)
    return clean_tasks


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
    "resume_focus_session": resume_focus_session,
    "stop_focus_session": stop_focus_session,
    "get_focus_session": get_focus_session,
    "add_focus_task": add_focus_task,
    "list_focus_tasks": list_focus_tasks,
    "get_next_focus_task": get_next_focus_task,
    "start_focus_task": start_focus_task,
    "update_focus_task_progress": update_focus_task_progress,
    "complete_focus_task": complete_focus_task,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "resume_focus_session",
            "description": "Resume a work block after a break, using the previous work and break durations. Use when the user says they are ready to continue.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
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
            "name": "get_next_focus_task",
            "description": "Recommend the active task or next open focus task, preferring high priority and then older tasks. Read-only.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "start_focus_task",
            "description": "Mark a specific open focus task as the active task when the user chooses to begin or resume it. Pauses any other active task.",
            "parameters": {
                "type": "object",
                "properties": {"task_id": {"type": "integer"}},
                "required": ["task_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_focus_task_progress",
            "description": "Save a brief progress note the user has reported for an open focus task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "integer"},
                    "note": {"type": "string", "minLength": 1, "maxLength": 500},
                },
                "required": ["task_id", "note"],
            },
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


