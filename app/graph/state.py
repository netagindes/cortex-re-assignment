"""
Shared state definitions flowing through the LangGraph workflow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.logging_utils import PipelineLogEntry, PipelineLogger


@dataclass
class QueryContext:
    user_input: str
    request_type: str
    addresses: List[str] = field(default_factory=list)
    suggested_addresses: List[str] = field(default_factory=list)
    address_matches: List[Dict[str, Any]] = field(default_factory=list)
    candidate_terms: List[str] = field(default_factory=list)
    unresolved_terms: List[str] = field(default_factory=list)
    missing_addresses: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    period: Optional[str] = None
    period_level: Optional[str] = None
    entity_name: Optional[str] = None
    property_name: Optional[str] = None
    tenant_name: Optional[str] = None
    year: Optional[int] = None
    quarter: Optional[str] = None
    month: Optional[str] = None
    needs_clarification: bool = False
    clarification_reasons: List[str] = field(default_factory=list)


@dataclass
class GraphState:
    context: QueryContext
    result: Optional[Dict[str, Any]] = None
    diagnostics: List[PipelineLogEntry] = field(default_factory=list)
    logger: PipelineLogger | None = field(default=None, repr=False)
    pnl_result: Optional[Dict[str, Any]] = None

    def log(self, message: str, *, level: str = "info", **metadata: Any) -> None:
        entry = PipelineLogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            level=level.upper(),
            message=message,
            metadata=metadata or None,
        )
        self.diagnostics.append(entry)
        if self.logger:
            log_method = getattr(self.logger, level, self.logger.info)
            log_method(message, **metadata)

    def diagnostics_as_text(self) -> List[str]:
        return [entry.as_text() for entry in self.diagnostics]


__all__ = ["QueryContext", "GraphState"]

