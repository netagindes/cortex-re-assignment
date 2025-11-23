"""
Agent prompting users for additional details when the query is ambiguous.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence

from app.agents.request_types import RequestType, normalize_request_type
from app.graph.state import ClarificationItem, QueryContext
from app.prompts.clarification_prompt import CLARIFICATION_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class ClarificationAgent:
    """
    Generates clarifying follow-up prompts.

    The `SYSTEM_PROMPT` attribute exposes the canonical clarification rules so the agent can
    eventually be run as an LLM node without duplicating prompt text elsewhere.
    """

    SYSTEM_PROMPT: str = CLARIFICATION_SYSTEM_PROMPT

    def run(
        self,
        context: QueryContext,
        request_type: Optional[RequestType | str] = None,
        reasons: Optional[List[str]] = None,
        suggestions: Optional[List[str]] = None,
    ) -> ClarificationItem:
        req_type = normalize_request_type(request_type or context.request_type, default=RequestType.CLARIFICATION)
        reasons = reasons or list(context.clarification_reasons or [])
        reasons = reasons or ["property"]
        field = self._select_field(req_type, reasons, context)
        question, kind, options = self._build_question(field, context, suggestions)
        logger.info(
            "ClarificationAgent generated question field=%s kind=%s options=%s",
            field,
            kind,
            options,
        )
        return ClarificationItem(field=field, question=question, kind=kind, options=options)

    @staticmethod
    def _select_field(
        request_type: RequestType,
        reasons: List[str],
        context: QueryContext,
    ) -> str:
        ordered = list(reasons)
        priority = ["property", "period", "tenant_name", "aggregation_level", "comparison_periods"]
        for candidate in priority:
            if candidate in ordered:
                return "property_name" if candidate == "property" else candidate
        if request_type == RequestType.PNL and not context.property_name:
            return "property_name"
        if request_type == RequestType.PNL and not context.period and not context.year and not context.quarter and not context.month:
            return "period"
        if request_type == RequestType.PNL and not context.tenant_name and not context.aggregation_level:
            return "aggregation_level"
        return "property_name"

    @staticmethod
    def _build_question(
        field: str,
        context: QueryContext,
        suggestions: Optional[List[str]],
    ) -> tuple[str, Optional[str], List[str]]:
        if field == "property_name":
            prompt = "Which property are you referring to?"
            if suggestions:
                preview = ClarificationAgent._format_suggestion_block(suggestions)
                prompt = f"{prompt} Try {preview}."
            return prompt, "value", list(suggestions or [])
        if field == "period":
            if not context.period_level:
                return (
                    "Which period should I use? Month, quarter, or year?",
                    "granularity",
                    ["month", "quarter", "year"],
                )
            return (
                f"Which {context.period_level} would you like? For example 2025 or 2025-Q1.",
                "value",
                [],
            )
        if field == "aggregation_level":
            return (
                "Do you want tenant-level, property-level, or combined totals?",
                "choice",
                ["tenant", "property", "combined"],
            )
        if field == "comparison_periods":
            return ("Which two periods would you like to compare?", "value", [])
        if field == "tenant_name":
            return ("Which tenant should I focus on?", "value", [])
        return ("Could you share the missing details so I can continue?", "value", [])

    @staticmethod
    def _format_suggestion_block(suggestions: Sequence[str]) -> str:
        unique: List[str] = []
        for suggestion in suggestions:
            if suggestion not in unique:
                unique.append(suggestion)
        if not unique:
            return ""
        return " or ".join(unique[:3])


__all__ = ["ClarificationAgent"]

