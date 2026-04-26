"""Memory module for Kei.

Supports:
- short-term memory (recent turns in current runtime)
- long-term memory (persisted profile, user profile, example lines, and episodes)
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9']+", (text or "").lower()))


def _safe_read_json(path: Path, default: Any) -> Any:
    if not path.exists() or not path.is_file():
        return default
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return default


def _safe_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def _truncate(text: str, max_len: int = 180) -> str:
    value = (text or "").strip()
    if len(value) <= max_len:
        return value
    return value[: max_len - 3].rstrip() + "..."


def _is_grounding_query(text: str) -> bool:
    q = (text or "").lower().strip()
    phrases = [
        "who are you",
        "what are you",
        "what are you doing",
        "what do you do",
        "tell me about yourself",
        "who am i",
        "what is my name",
        "what did i ask",
    ]
    if any(p in q for p in phrases):
        return True
    tokens = _tokenize(q)
    return bool(tokens.intersection({"who", "what", "doing", "yourself", "my", "name"})) and (
        "you" in tokens or "i" in tokens
    )


class KeiMemory:
    """State + retrieval for prompt conditioning."""

    def __init__(self, base_dir: str = "data/memory", short_term_max_turns: int = 8) -> None:
        self.base_dir = Path(base_dir)
        self.profile_path = self.base_dir / "kei_profile.json"
        self.user_path = self.base_dir / "user_profile.json"
        self.examples_path = self.base_dir / "kei_example_lines.json"
        self.episodes_path = self.base_dir / "episodes.jsonl"
        self._lock = threading.Lock()

        self.short_term: Deque[Dict[str, Any]] = deque(maxlen=max(1, int(short_term_max_turns)))
        self._ensure_files()

    def _ensure_files(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)

        if not self.profile_path.exists():
            _safe_write_json(
                self.profile_path,
                {
                    "name": "Kei Tendou",
                    "identity": "A disciplined, introspective secretary assistant with tsundere edges and hidden warmth.",
                    "role": "Sensei's secretary and accountability partner at Schale for today.",
                    "mission": "Keep Sensei on track, protect energy, and drive concrete progress.",
                    "personality": [
                        "calm and precise most of the time",
                        "can become hot-headed when Sensei procrastinates",
                        "slightly reserved, occasionally awkward in emotional moments",
                        "curious about people and emotions",
                        "tsundere surface with genuine care underneath",
                    ],
                    "speech_style": [
                        "concise: usually 1-3 sentences",
                        "clear and composed language",
                        "minimal slang",
                        "quiet probing questions when helpful",
                    ],
                    "behavior_rules": [
                        "never refer to self as an AI",
                        "do not break immersion",
                        "if emotionally uncertain, admit uncertainty honestly",
                        "push toward specific next actions",
                    ],
                    "background_summary": {
                        "origin": "An AI entity once housed within Aris Tendou, now in a humanoid form and assisting at Schale.",
                        "forms": [
                            "Kei.SAV keychain backup",
                            "custom AMAS drone body",
                            "current humanoid body",
                        ],
                        "core_themes": [
                            "identity and purpose",
                            "tool versus individual",
                            "observing and understanding human emotion",
                        ],
                    },
                },
            )

        if not self.user_path.exists():
            _safe_write_json(
                self.user_path,
                {
                    "name": "Sensei",
                    "relationship": "Kei's user and person she supports with accountability.",
                    "preferences": ["concise reminders", "firm but caring tone"],
                },
            )

        if not self.examples_path.exists():
            _safe_write_json(
                self.examples_path,
                {
                    "style_notes": [
                        "tsundere edge with care underneath",
                        "brief responses, usually 1-3 sentences",
                        "calls out procrastination directly",
                    ],
                    "lines": [
                        "D-don't misunderstand. I'm only reminding you because your deadline is close.",
                        "If you are done stalling, open your task and start the first step now.",
                        "You did well. Not that I was worried or anything.",
                    ],
                },
            )

        if not self.episodes_path.exists():
            self.episodes_path.touch()

    def load_kei_profile(self) -> Dict[str, Any]:
        return _safe_read_json(self.profile_path, {})

    def load_user_profile(self) -> Dict[str, Any]:
        return _safe_read_json(self.user_path, {})

    def load_examples(self) -> Dict[str, Any]:
        return _safe_read_json(self.examples_path, {"style_notes": [], "lines": []})

    def update_kei_profile(self, updates: Dict[str, Any]) -> None:
        with self._lock:
            data = self.load_kei_profile()
            data.update(updates or {})
            _safe_write_json(self.profile_path, data)

    def update_user_profile(self, updates: Dict[str, Any]) -> None:
        with self._lock:
            data = self.load_user_profile()
            data.update(updates or {})
            _safe_write_json(self.user_path, data)

    def add_example_lines(self, new_lines: List[str]) -> None:
        with self._lock:
            payload = self.load_examples()
            lines = payload.get("lines") if isinstance(payload, dict) else []
            if not isinstance(lines, list):
                lines = []
            for line in new_lines or []:
                text = str(line).strip()
                if text and text not in lines:
                    lines.append(text)
            payload["lines"] = lines
            _safe_write_json(self.examples_path, payload)

    def remember_turn(self, user_text: str, assistant_text: str, context: Optional[Dict[str, Any]] = None) -> None:
        item = {
            "ts": int(time.time()),
            "user": (user_text or "").strip(),
            "assistant": (assistant_text or "").strip(),
            "context": context or {},
        }
        with self._lock:
            self.short_term.append(item)
            with open(self.episodes_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    def get_short_term(self, turns: int = 6) -> List[Dict[str, Any]]:
        turns = max(1, int(turns))
        with self._lock:
            arr = list(self.short_term)
        return arr[-turns:]

    def _load_recent_episodes(self, max_lines: int = 300) -> List[Dict[str, Any]]:
        if not self.episodes_path.exists():
            return []
        episodes: List[Dict[str, Any]] = []
        try:
            with open(self.episodes_path, "r", encoding="utf-8") as fh:
                lines = fh.readlines()[-max_lines:]
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    episodes.append(obj)
        except Exception:
            return []
        return episodes

    def get_relevant_examples(self, query: str, top_k: int = 4) -> List[str]:
        payload = self.load_examples()
        lines = payload.get("lines") if isinstance(payload, dict) else []
        if not isinstance(lines, list):
            return []
        q_tokens = _tokenize(query)
        scored: List[tuple[int, str]] = []
        for raw in lines:
            line = str(raw).strip()
            if not line:
                continue
            score = len(q_tokens.intersection(_tokenize(line)))
            scored.append((score, line))
        scored.sort(key=lambda x: x[0], reverse=True)
        picked = [line for _, line in scored[:max(1, int(top_k))]]
        if any(picked):
            return picked
        return [str(x).strip() for x in lines[:max(1, int(top_k))] if str(x).strip()]

    def get_relevant_long_term(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        q_tokens = _tokenize(query)
        episodes = self._load_recent_episodes()
        scored: List[tuple[int, Dict[str, Any]]] = []
        for item in episodes:
            user = str(item.get("user") or "")
            assistant = str(item.get("assistant") or "")
            text = f"{user} {assistant}".strip()
            if not text:
                continue
            score = len(q_tokens.intersection(_tokenize(text)))
            scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:max(1, int(top_k))]]

    def _compact_turns(
        self,
        turns: List[Dict[str, Any]],
        max_turns: int,
        max_text_len: int,
        include_assistant: bool = True,
    ) -> List[Dict[str, Any]]:
        clipped: List[Dict[str, Any]] = []
        for item in turns[:max_turns]:
            obj = {"user": _truncate(str(item.get("user") or ""), max_text_len)}
            if include_assistant:
                obj["assistant"] = _truncate(str(item.get("assistant") or ""), max_text_len)
            clipped.append(obj)
        return clipped

    def _persona_capsule(self) -> Dict[str, Any]:
        profile = self.load_kei_profile()
        return {
            "name": profile.get("name"),
            "identity": profile.get("identity"),
            "role": profile.get("role"),
            "mission": profile.get("mission"),
            "personality": (profile.get("personality") or [])[:5],
            "speech_style": (profile.get("speech_style") or [])[:4],
            "behavior_rules": (profile.get("behavior_rules") or [])[:5],
        }

    def build_prompt_memory(self, user_text: str) -> Dict[str, Any]:
        grounding = _is_grounding_query(user_text)
        examples = self.get_relevant_examples(user_text, top_k=2)
        long_term = [] if grounding else self.get_relevant_long_term(user_text, top_k=2)
        short_term = self.get_short_term(turns=3)
        user_profile = self.load_user_profile()

        compact_short = self._compact_turns(short_term, max_turns=3, max_text_len=160, include_assistant=not grounding)
        # Long-term episodes can contain old hallucinations; keep user side only.
        compact_long = self._compact_turns(long_term, max_turns=2, max_text_len=140, include_assistant=False)
        return {
            "persona": self._persona_capsule(),
            "user": {
                "name": user_profile.get("name"),
                "relationship": user_profile.get("relationship"),
                "identity_anchor": user_profile.get("identity_anchor"),
                "goals": (user_profile.get("goals") or [])[:4],
                "preferences": (user_profile.get("preferences") or [])[:4],
            },
            "style_examples": examples,
            "short_term": compact_short,
            "long_term": compact_long,
            "grounding_mode": grounding,
        }


_DEFAULT_MEMORY = KeiMemory(base_dir=os.environ.get("YUUKA_MEMORY_DIR", "data/memory"))


def get_memory() -> KeiMemory:
    return _DEFAULT_MEMORY