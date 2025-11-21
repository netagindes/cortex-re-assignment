"""
Helpers for consistent logging across the API, agents, and UI.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict

_LOGGING_CONFIGURED = False


def setup_logging(level: str | None = None) -> None:
    """
    Configure the root logger once.
    """

    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    env_level = level or os.getenv("LOG_LEVEL", "INFO")
    try:
        parsed_level = getattr(logging, env_level.upper())
    except AttributeError:
        parsed_level = logging.INFO

    logging.basicConfig(
        level=parsed_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    _LOGGING_CONFIGURED = True


class PipelineLogEntry(BaseModel):
    """
    Structured representation of a log line that can be rendered for the UI.
    """

    model_config = ConfigDict(frozen=True)

    timestamp: str
    level: str
    message: str
    metadata: Dict[str, Any] | None = None

    def as_text(self) -> str:
        suffix = f" | {self.metadata}" if self.metadata else ""
        return f"{self.timestamp} [{self.level}] {self.message}{suffix}"


class PipelineLogger:
    """
    Lightweight collector that mirrors log lines to the stdlib logger and stores them for the UI.
    """

    def __init__(
        self,
        name: str,
        *,
        level: str | None = None,
        context: Dict[str, Any] | None = None,
    ) -> None:
        setup_logging(level)
        self._logger = logging.getLogger(name)
        self._context = dict(context or {})
        self._entries: List[PipelineLogEntry] = []

    def info(self, message: str, **metadata: Any) -> None:
        self._log(logging.INFO, message, metadata)

    def warning(self, message: str, **metadata: Any) -> None:
        self._log(logging.WARNING, message, metadata)

    def error(self, message: str, **metadata: Any) -> None:
        self._log(logging.ERROR, message, metadata)

    def debug(self, message: str, **metadata: Any) -> None:
        self._log(logging.DEBUG, message, metadata)

    def _log(self, level: int, message: str, metadata: Optional[Dict[str, Any]]) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        merged_metadata = {**self._context, **(metadata or {})} or None
        entry = PipelineLogEntry(
            timestamp=timestamp,
            level=logging.getLevelName(level),
            message=message,
            metadata=merged_metadata,
        )

        if merged_metadata:
            self._logger.log(level, "%s | %s", message, merged_metadata)
        else:
            self._logger.log(level, "%s", message)
        self._entries.append(entry)

    def as_entries(self) -> List[PipelineLogEntry]:
        return list(self._entries)

    def as_text_lines(self) -> List[str]:
        return [entry.as_text() for entry in self._entries]


__all__ = ["PipelineLogEntry", "PipelineLogger", "setup_logging"]



