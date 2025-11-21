"""
Shared utility functions and tool abstractions used across agents.
"""

from __future__ import annotations

from typing import Any, Dict


def format_currency(value: float, currency: str = "USD") -> str:
    """
    Basic formatting helper for monetary values.
    """

    return f"{currency} {value:,.2f}"


def build_response_payload(result: Any, metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Standardize responses sent back to the UI/API.
    """

    payload = {"result": result}
    if metadata:
        payload["metadata"] = metadata
    return payload


__all__ = ["format_currency", "build_response_payload"]

