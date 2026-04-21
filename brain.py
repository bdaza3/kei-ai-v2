"""Rule-based decision engine."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from command_handler import parse_command, normalize_text, requires_confirmation
from monitor import ActivitySnapshot


@dataclass
class DecisionConfig:
    idle_trigger_seconds: int = 120 # 2 minutes of inactivity
    procrastination_trigger_seconds: int = 300 # 5 minutes of distraction
    break_trigger_seconds: int = 3000 # 50 minutes of productivity
    cooldown_seconds: int = 45 # Minimum seconds between same trigger firing
    focused_ping_seconds: int = 300 # Ping every 5 minutes to check if user is focused or distracted


class DecisionEngine:
    def __init__(self, config: Optional[DecisionConfig] = None) -> None:
        self.config = config or DecisionConfig()
        self._last_trigger_at: Dict[str, float] = {}
        self.state: Dict[str, object] = {
            "mode": "focused",
            "last_activity_time": time.time(),
            "current_app": "",
            "current_title": "",
            "focus_score": 0.0,
        }
        self.pending_confirmation: Optional[str] = None

    def _cooled_down(self, trigger: str, now: float) -> bool:
        last = self._last_trigger_at.get(trigger)
        return last is None or (now - last) >= self.config.cooldown_seconds

    def _mark_triggered(self, trigger: str, now: float) -> None:
        self._last_trigger_at[trigger] = now

    #function to update the focus score based on the current activity snapshot, which influences the "mode" of the assistant (focused vs distracted)
    def _update_focus_score(self, activity: ActivitySnapshot) -> None:
        score = float(self.state.get("focus_score", 0.0))
        score *= 0.9

        if activity.current_category == "productive":
            score += 2.0
        elif activity.current_category == "distracting": 
            score -= 3.0
        elif activity.current_category == "idle":
            score -= 2.0

        if activity.productive_streak_seconds >= 900:
            score += 1.5
        if activity.distracting_streak_seconds >= self.config.procrastination_trigger_seconds:
            score -= 2.0

        self.state["focus_score"] = max(-10.0, min(10.0, score))

    #function to build context dictionary from activity snapshot for use in dialogue generation
    def _build_context(self, activity: ActivitySnapshot) -> Dict:
        return {
            "active_app": activity.active_app,
            "active_window_title": activity.active_window_title,
            "active_process_name": activity.active_process_name,
            "current_category": activity.current_category,
            "idle_seconds": activity.idle_seconds,
            "distraction_seconds": activity.distraction_seconds,
            "productive_seconds": activity.productive_seconds,
            "productive_streak_seconds": activity.productive_streak_seconds,
            "distracting_streak_seconds": activity.distracting_streak_seconds,
            "current_window_seconds": activity.current_window_seconds,
            "recent_entries": [
                {
                    "app_name": entry.app_name,
                    "process_name": entry.process_name,
                    "window_title": entry.window_title,
                    "category": entry.category,
                    "duration_seconds": entry.duration_seconds,
                }
                for entry in activity.recent_entries
            ],
        }
    
    #function to evaluate the current activity and user input, returning a trigger and context for dialogue generation
    def evaluate(self, activity: ActivitySnapshot, user_text: Optional[str] = None) -> Tuple[Optional[str], Dict]:
        now = time.time()
        context = self._build_context(activity)

        self.state["last_activity_time"] = now
        self.state["current_app"] = activity.active_app
        self.state["current_title"] = activity.active_window_title
        self._update_focus_score(activity)

        if user_text:
            normalized = normalize_text(user_text)

            if self.pending_confirmation:
                confirm_words = ["confirm", "yes", "yeah", "yep", "sure", "ok", "okay", "affirmative", "proceed", "go ahead"]
                cancel_words = ["no", "cancel", "stop", "never", "nope", "don t", "do not"]
                if any(word in normalized for word in confirm_words):
                    command_name = self.pending_confirmation
                    self.pending_confirmation = None
                    if self._cooled_down("user_command", now):
                        self._mark_triggered("user_command", now)
                        response_context = dict(context)
                        response_context["command"] = command_name
                        response_context["confirmed"] = True
                        return "user_command", response_context
                if any(word in normalized for word in cancel_words):
                    self.pending_confirmation = None
                    return None, {}

            command = parse_command(user_text)
            if command:
                if requires_confirmation(command.name) and not self.pending_confirmation:
                    self.pending_confirmation = command.name
                    return "confirm", {"command": command.name}

                if self._cooled_down("user_command", now):
                    self._mark_triggered("user_command", now)
                    response_context = dict(context)
                    response_context["command"] = command.name
                    response_context.update(command.args)
                    return "user_command", response_context

            if self._cooled_down("conversation", now):
                self._mark_triggered("conversation", now)
                response_context = dict(context)
                response_context["user_text"] = user_text
                return "conversation", response_context

        if activity.idle_seconds >= self.config.idle_trigger_seconds and self._cooled_down("idle", now):
            self._mark_triggered("idle", now)
            return "idle", context

        if (
            activity.current_category == "distracting"
            and activity.distracting_streak_seconds >= self.config.procrastination_trigger_seconds
            and self._cooled_down("procrastination", now)
        ):
            self._mark_triggered("procrastination", now)
            return "procrastination", context

        if (
            activity.current_category == "productive"
            and activity.productive_streak_seconds >= self.config.break_trigger_seconds
            and self._cooled_down("break_reminder", now)
        ):
            self._mark_triggered("break_reminder", now)
            return "break_reminder", context

        last_focused = self._last_trigger_at.get("focused")
        if last_focused is None or (now - last_focused) >= self.config.focused_ping_seconds:
            score = float(self.state.get("focus_score", 0.0))
            self.state["mode"] = "distracted" if score < -2.0 else "focused"
            self._mark_triggered("focused", now)
            return "focused", context

        return None, {}
