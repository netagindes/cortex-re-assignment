"""
Shared state definitions flowing through the LangGraph workflow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class QueryContext:
    user_input: str
    request_type: str
    addresses: List[str] = field(default_factory=list)
    period: Optional[str] = None


@dataclass
class GraphState:
    context: QueryContext
    result: Optional[Dict[str, Any]] = None
    diagnostics: List[str] = field(default_factory=list)


__all__ = ["QueryContext", "GraphState"]

