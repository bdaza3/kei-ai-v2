"""Structured event logging helpers."""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, is_dataclass
from typing import Any


class AssistantEventLogger:
    def __init__(self, logs_dir: str = "logs", filename: str = "events.jsonl") -> None:
        self.logs_dir = logs_dir
        self.path = os.path.join(logs_dir, filename)
        self._lock = threading.Lock()
        os.makedirs(self.logs_dir, exist_ok=True)

    def _normalize(self, value: Any) -> Any:
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, dict):
            return {key: self._normalize(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._normalize(item) for item in value]
        return value

    def log_event(self, kind: str, payload: dict[str, Any]) -> None:
        event = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "kind": kind,
            "payload": self._normalize(payload),
        }
        line = json.dumps(event, ensure_ascii=False)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
