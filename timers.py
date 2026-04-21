"""Timer utilities for the assistant."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class TimerEvent:
    trigger: str
    context: dict


class PomodoroTimer:
    def __init__(self, default_minutes: int = 25, default_break_minutes: int = 5) -> None:
        self.default_minutes = default_minutes
        self.default_break_minutes = default_break_minutes
        self.state = "idle"
        self.started_at: Optional[float] = None
        self.ends_at: Optional[float] = None
        self.work_minutes = default_minutes
        self.break_minutes = default_break_minutes
        self.completed_work_sessions = 0

    def start(self, minutes: Optional[int] = None, break_minutes: Optional[int] = None, *, now: Optional[float] = None) -> dict:
        now = now or time.time()
        self.work_minutes = max(1, int(minutes or self.default_minutes))
        self.break_minutes = max(1, int(break_minutes or self.default_break_minutes))
        self.state = "work"
        self.started_at = now
        self.ends_at = now + (self.work_minutes * 60)
        return self.snapshot(now)

    def stop(self, *, now: Optional[float] = None) -> dict:
        snapshot = self.snapshot(now or time.time())
        self.state = "idle"
        self.started_at = None
        self.ends_at = None
        return snapshot

    def snapshot(self, now: Optional[float] = None) -> dict:
        now = now or time.time()
        remaining_seconds = max(0, int((self.ends_at or now) - now)) if self.ends_at else 0
        return {
            "state": self.state,
            "work_minutes": self.work_minutes,
            "break_minutes": self.break_minutes,
            "remaining_seconds": remaining_seconds,
            "completed_work_sessions": self.completed_work_sessions,
        }

    def poll(self, *, now: Optional[float] = None) -> list[TimerEvent]:
        now = now or time.time()
        if self.state == "idle" or self.ends_at is None:
            return []
        if now < self.ends_at:
            return []

        if self.state == "work":
            self.completed_work_sessions += 1
            self.state = "break"
            self.started_at = now
            self.ends_at = now + (self.break_minutes * 60)
            return [
                TimerEvent(
                    trigger="pomodoro_break",
                    context=self.snapshot(now),
                )
            ]

        self.state = "idle"
        self.started_at = None
        self.ends_at = None
        return [
            TimerEvent(
                trigger="pomodoro_completed",
                context=self.snapshot(now),
            )
        ]
