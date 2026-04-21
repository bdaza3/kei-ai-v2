"""Voice command parsing for the assistant."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import re


@dataclass(frozen=True)
class CommandResult:
    name: str
    args: dict[str, Any] = field(default_factory=dict)


_PHRASE_COMMANDS = {
    "take a break": "take_break",
    "start working": "start_working",
    "status": "status",
    "what is my status": "status",
    "what's my status": "status",
    "what am i doing": "status",
    "stop pomodoro": "stop_pomodoro",
    "cancel pomodoro": "stop_pomodoro",
}

_REQUIRES_CONFIRM = {"take_break"}

_POMODORO_PATTERN = re.compile(
    r"\b(?:start\s+)?pomodoro(?:\s+for\s+|\s+)?(?P<minutes>\d{1,3})?(?:\s*(?:minutes|min))?"
    r"(?:\s+with\s+(?P<break_minutes>\d{1,2})\s*(?:minute|min)\s+break)?\b"
)


def normalize_text(text: str) -> str:
    if not text:
        return ""
    normalized = text.lower()
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    normalized = " ".join(normalized.split())
    return normalized


def requires_confirmation(command_name: str) -> bool:
    return command_name in _REQUIRES_CONFIRM


def parse_command(text: str) -> Optional[CommandResult]:
    normalized = normalize_text(text)
    if not normalized:
        return None

    for phrase, internal in _PHRASE_COMMANDS.items():
        if phrase in normalized:
            return CommandResult(name=internal)

    pomodoro_match = _POMODORO_PATTERN.search(normalized)
    if pomodoro_match:
        minutes = int(pomodoro_match.group("minutes") or 25)
        break_minutes = int(pomodoro_match.group("break_minutes") or 5)
        return CommandResult(
            name="start_pomodoro",
            args={"minutes": minutes, "break_minutes": break_minutes},
        )
    return None
