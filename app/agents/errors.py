"""
Shared error primitives so specialist agents can signal clarification/fallback needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class ErrorType(str, Enum):
    UNKNOWN_PROPERTY = "unknown_property"
    MISSING_PROPERTY = "missing_property"
    MISSING_TIMEFRAME = "missing_timeframe"
    DATA_UNAVAILABLE = "data_unavailable"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class AgentError(Exception):
    error_type: ErrorType
    message: str
    details: Optional[Dict[str, Any]] = None

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"message": self.message, "error_type": self.error_type.value}
        if self.details:
            payload["details"] = self.details
        return payload


__all__ = ["AgentError", "ErrorType"]

