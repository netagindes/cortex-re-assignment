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
    requirement_section: Optional[str] = None
    agent: Optional[str] = None
    metadata: Dict[str, Any] | None = None

    def as_text(self) -> str:
        parts = [f"{self.timestamp} [{self.level}] {self.message}"]
        if self.agent:
            parts.append(f"agent={self.agent}")
        if self.requirement_section:
            parts.append(f"req={self.requirement_section}")
        if self.metadata:
            parts.append(str(self.metadata))
        return " | ".join(parts)


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
        meta = dict(metadata or {})
        agent = meta.pop("agent", None)
        requirement_section = meta.pop("requirement_section", None)
        merged_metadata = {**self._context, **meta} or None
        entry = PipelineLogEntry(
            timestamp=timestamp,
            level=logging.getLevelName(level),
            message=message,
            agent=agent or self._context.get("agent"),
            requirement_section=requirement_section,
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

    def as_markdown(self) -> str:
        if not self._entries:
            return ""
        lines: List[str] = []
        for entry in self._entries:
            meta_segments: List[str] = []
            if entry.agent:
                meta_segments.append(f"agent `{entry.agent}`")
            if entry.requirement_section:
                meta_segments.append(f"req `{entry.requirement_section}`")
            if entry.metadata:
                meta_segments.append(f"meta {entry.metadata}")
            meta_text = f" ({' • '.join(meta_segments)})" if meta_segments else ""
            lines.append(f"- `{entry.timestamp}` – **{entry.message}**{meta_text}")
        return "\n".join(lines)


__all__ = ["PipelineLogEntry", "PipelineLogger", "setup_logging"]



