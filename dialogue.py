"""Dialogue template module.

Adds a small `DialogueManager` to support scene/sequence playback (visual-novel style)
and an 80/20 rule-based vs LLM selection for replies.
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, Optional, Any, List, Union


def _format_minutes(seconds: float) -> str:
    total_minutes = max(0, int(round(seconds / 60.0)))
    if total_minutes <= 1:
        return "1 minute"
    return f"{total_minutes} minutes"


# Default template responses for various triggers (rule-based)
RESPONSES: Dict[str, List[str]] = {
    "procrastination": [
        "You have drifted long enough. Close the distraction and return to work, Sensei.",
        "That tab is not helping you. Switch back to your actual task now.",
    ],
    "idle": [
        "Sensei... are you working or just staring at the screen?",
        "If you need a break, say it clearly. If not, get back to work.",
    ],
    "focused": [
        "Good. Keep this pace and finish the next concrete step.",
        "You are doing fine. Stay focused and close this task block.",
    ],
    "break_reminder": [
        "You have been working for a long stretch. Take a short break before you burn out.",
        "It's time for a quick break. Step away from your desk and stretch.",
    ],
    "hourly_chime": [
        "Another hour has passed. Are you still on track with your priorities?",
        "It's a new hour. Just wanted to let you know.",
    ],
    "pomodoro_started": [
        "I started your Pomodoro timer. Stay focused until the timer ends.",
    ],
    "pomodoro_break": [
        "Work session complete. You can take a short break now.",
    ],
    "pomodoro_completed": [
        "Break over. If you are ready, start the next pomodoro.",
    ],
    "pomodoro_stopped": [
        "I stopped your Pomodoro timer.",
    ],
    "conversation": [
        "Sensei? What is it this time?",
    ],
    "user_command": [
        "Understood. I will keep you on schedule.",
    ],
}


# Follow-up lines (can be used as chained lines after the main line)
FOLLOW_UP_RESPONSES: Dict[str, List[str]] = {
    "procrastination": [
        "Do not make me repeat myself. Focus on your work.",
        "I know it is tempting, but wasting time now will only stress you later. Back to task.",
    ],
    "idle": [
        "What have you been doing? You're late!",
        "Do not drift. Choose one next action and do it now.",
    ],
    "focused": [
        "Keep it up. Small steps add up.",
        "Good. Stay with it and make measurable progress.",
    ],
    "break_reminder": [
        "Take a short walk, then come back refreshed.",
        "Work in moderation. Do not burn out trying to prove a point.",
    ],
    "conversation": [
        "What is it? Please don't call me if you don't have a reason.",
        "If you have something to say, say it directly. I am listening.",
    ],
}


class DialogueManager:
    """Manage scene-based dialogue sequences and rule/LLM selection.

    Scenes format (JSON):
    {
      "scenes": {
        "intro": {
          "lines": [
            {"speaker": "Kei", "variants": [{"text":"Hello","weight":0.8}, {"text":"Hi","weight":0.2}]},
            {"speaker": "Kei", "text": "Let's start."}
          ]
        }
      }
    }
    """

    def __init__(self, scenes: Optional[Dict[str, Any]] = None, rule_fraction: float = 0.8, rng: Optional[random.Random] = None):
        self.rule_fraction = float(rule_fraction)
        self.rng = rng or random.Random()
        self.scenes: Dict[str, Any] = scenes or {}
        self.current_scene: Optional[str] = None
        self.index: int = 0
        self.follow_up_queue: List[Dict[str, Any]] = []

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self.scenes = data.get("scenes", {})

    #Start a scene by ID, resetting the line index and follow-up queue
    def start_scene(self, scene_id: str) -> None:
        if scene_id in self.scenes:
            self.current_scene = scene_id
            self.index = 0
            self.follow_up_queue.clear()

    #Enqueue a follow-up line to be said after the current line or scene finishes
    def enqueue_follow_up(self, line: Dict[str, Any]) -> None: 
        self.follow_up_queue.append(line)

    #Randomly choose a dialogue variant from a list of options, weighted if specified.
    def _choose_variant(self, variants: Union[List[Any], Dict[str, Any]]) -> str:
        if isinstance(variants, dict):
            variants = [variants]
        texts = []
        weights = []
        for v in variants:
            if isinstance(v, str):
                texts.append(v)
                weights.append(1)
            else:
                texts.append(v.get("text", ""))
                weights.append(float(v.get("weight", 1)))
        if not texts:
            return ""
        return self.rng.choices(texts, weights=weights, k=1)[0]

    #Get the next line from the current scene, if any, and advance the index. Returns None if no more lines.
    def _get_next_from_scene(self) -> Optional[str]:
        if not self.current_scene:
            return None
        scene = self.scenes.get(self.current_scene, {})
        lines = scene.get("lines", [])
        if self.index >= len(lines):
            return None
        line = lines[self.index]
        self.index += 1
        # handle variants
        if "variants" in line:
            return self._choose_variant(line["variants"])
        return line.get("text") or None

    # Get the next response for a trigger, using follow-up queue, scene lines, follow-up mapping, or fallback responses.
    def _rule_response(self, trigger: str, context: Optional[Dict[str, Any]] = None) -> str:
        # follow-up queue first
        if self.follow_up_queue:
            item = self.follow_up_queue.pop(0)
            if "variants" in item:
                return self._choose_variant(item["variants"])
            return item.get("text", "")

        # next scene line
        scene_line = self._get_next_from_scene()
        if scene_line:
            return scene_line

        # follow-up mapping
        options = FOLLOW_UP_RESPONSES.get(trigger)
        if options:
            return self.rng.choice(options)

        # fallback to RESPONSES
        options = RESPONSES.get(trigger)
        if options:
            return self.rng.choice(options)

        return ""

    def _llm_response(self, trigger: str, context: Optional[Dict[str, Any]] = None) -> str:
        # Placeholder LLM behavior: choose a variant from RESPONSES if available
        opts = RESPONSES.get(trigger) or ["I don't have a response right now."]
        return f"(LLM Response) {self.rng.choice(opts)}"

    def next(self, trigger: str, context: Optional[Dict[str, Any]] = None, rng: Optional[random.Random] = None) -> str:
        context = context or {}
        rng = rng or self.rng
        # Decide rule-based vs LLM
        if rng.random() < self.rule_fraction:
            resp = self._rule_response(trigger, context)
            if resp:
                return resp
            # fallback to llm if rule produced nothing
            return self._llm_response(trigger, context)
        else:
            return self._llm_response(trigger, context)


# Module-level default manager: attempts to load `data/dialogues.json` if present
_DEFAULT_MANAGER = DialogueManager()
_DEFAULT_MANAGER.load(os.path.join(os.path.dirname(__file__), "data", "dialogues.json"))


def get_response(trigger: str, context: Optional[Dict] = None, rng: Optional[random.Random] = None) -> str:
    """Main compatibility wrapper used by other modules.

    Keeps existing special-case behavior for commands and short-circuits,
    then defers to the `DialogueManager` for scene-based or rule responses.
    """
    context = context or {}
    command = context.get("command")

    # Preserve explicit command responses from prior implementation
    if trigger == "user_command" and command == "take_break":
        return "Command: Take a timed break. Return in 10 minutes and continue your priority task."

    if trigger == "user_command" and command == "start_working":
        return "Command: Work mode enabled. Begin with your highest-priority item now."

    if trigger == "user_command" and command == "start_pomodoro":
        minutes = int(context.get("minutes", 25))
        break_minutes = int(context.get("break_minutes", 5))
        return f"Command: Pomodoro started for {minutes} minutes with a {break_minutes} minute break."

    if trigger == "user_command" and command == "stop_pomodoro":
        return "Command: Pomodoro cancelled."

    if trigger == "user_command" and command == "status":
        app = context.get("active_app") or "unknown app"
        title = context.get("active_window_title") or "unknown window"
        category = context.get("current_category") or "unknown"
        idle = int(context.get("idle_seconds", 0))
        window_minutes = _format_minutes(float(context.get("current_window_seconds", 0.0)))
        productive_streak = _format_minutes(float(context.get("productive_streak_seconds", 0.0)))
        distracting_streak = _format_minutes(float(context.get("distracting_streak_seconds", 0.0)))
        return f"Command: Status: {app} on {title}. " \
               f"Category is {category}. " \
               f"You have been here for {window_minutes}, idle for {idle} seconds, " \
               f"productive streak {productive_streak}, distraction streak {distracting_streak}."

    if trigger == "confirm":
        cmd = context.get("command", "that action")
        human = str(cmd).replace("_", " ")
        return f"Command: Are you sure you want to {human}? Say 'confirm' to proceed or 'cancel' to abort."

    if trigger == "procrastination":
        app = context.get("active_app") or "that app"
        title = context.get("active_window_title") or "that window"
        streak = _format_minutes(float(context.get("distracting_streak_seconds", 0.0)))
        return f"Command: You have been on {app} - {title} for {streak}. Return to your work now."

    if trigger == "break_reminder":
        streak = _format_minutes(float(context.get("productive_streak_seconds", 0.0)))
        return f"Command: You have been working for {streak}. Stand up, stretch, and take a short break."

    if trigger == "hourly_chime":
        current_hour = context.get("hour_label")
        if current_hour:
            return f"Command: It is now {current_hour}. Pause for a moment and check your priorities."

    # Defer to the DialogueManager for sequences and rule-based replies
    return _DEFAULT_MANAGER.next(trigger, context, rng=rng)
