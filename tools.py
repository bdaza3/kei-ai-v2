"""Allowlisted runtime tools shared by the legacy loop and LangGraph."""

from datetime import datetime
from typing import Any, Callable

from monitor import ActivityMonitor, ActivitySnapshot, ActivityEntry


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


TOOL_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "get_active_window": get_active_window,
    "get_activity_snapshot": get_activity_snapshot,
    "get_current_time": get_current_time,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_active_window",
            "description": "Get the current active app and window title.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
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
]


def execute_tool(name: str, arguments: dict[str, Any] | None = None) -> Any:
    """Execute one allowlisted tool after the model has selected its name."""
    try:
        function = TOOL_FUNCTIONS[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported tool: {name}") from exc
    return function(**(arguments or {}))


