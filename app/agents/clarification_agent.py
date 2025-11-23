"""
Agent prompting users for additional details when the query is ambiguous.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from app.agents.request_types import RequestType, normalize_request_type
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
        user_input: str,
        request_type: Optional[RequestType | str] = None,
        reasons: Optional[List[str]] = None,
        suggestions: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        req_type = normalize_request_type(request_type, default=RequestType.CLARIFICATION)
        logger.info(
            "ClarificationAgent invoked for query '%s' (type=%s, reasons=%s)",
            user_input,
            req_type.value,
            reasons,
        )
        message = self._build_message(req_type, reasons, suggestions)
        return {
            "message": message,
            "original_query": user_input,
        }

    @staticmethod
    def _build_message(
        request_type: RequestType,
        reasons: Optional[List[str]],
        suggestions: Optional[List[str]] = None,
    ) -> str:
        reasons = reasons or []
        targeted_reasons = [reason for reason in reasons if reason not in {"period", "granularity", "comparison_periods", "property_selection"}]

        reason_text = ""
        if targeted_reasons:
            friendly = ", ".join(reason.replace("_", " ") for reason in targeted_reasons)
            reason_text = f"I still need: {friendly}. "

        suggestion_text = ""
        suggestion_block = ""
        if suggestions:
            preview = " or ".join(suggestions[:2]) if len(suggestions) > 1 else suggestions[0]
            suggestion_text = f" Try {preview}."
            suggestion_block = ClarificationAgent._format_suggestion_block(suggestions)

        period_prompt = None
        granularity_prompt = None
        comparison_prompt = None
        property_selection_prompt = None
        if "period" in reasons:
            period_prompt = "Which period would you like (month, quarter, or year)?"
        if "granularity" in reasons:
            granularity_prompt = "Do you want tenant-level, property-level, or combined totals?"
        if "comparison_periods" in reasons:
            comparison_prompt = "Which two periods would you like to compare?"
        if "property_selection" in reasons:
            property_selection_prompt = "Comparison requires a single property. Which property should I analyze?"

        if request_type == RequestType.PRICE_COMPARISON:
            return (
                f"{reason_text}"
                "To compare property values, please mention two property names "
                "(e.g., Building 120 and Building 160)."
                f"{suggestion_text}"
                f"{suggestion_block}"
            )
        if request_type == RequestType.ASSET_DETAILS:
            return (
                f"{reason_text}"
                "Let me know which property you want details for (for example, Building 140)."
                f"{suggestion_text}"
                f"{suggestion_block}"
            )
        if request_type == RequestType.PNL:
            prompts = [prompt for prompt in (property_selection_prompt, period_prompt, granularity_prompt, comparison_prompt) if prompt]
            if not prompts:
                prompts.append("Specify the time frame (year, quarter, or month) and optionally a property or tenant.")
            return (
                f"{reason_text}{' '.join(prompts)}"
                f"{suggestion_text}"
                f"{suggestion_block}"
            )

        capabilities_hint = (
            "I can compare two properties, compute P&L for any period, explain ledger codes, "
            "or describe any property from the dataset."
        )
        return (
            f"{reason_text}{capabilities_hint} Let me know which asset or metric you'd like me to focus on."
            f"{suggestion_text}"
            f"{suggestion_block}"
        )

    @staticmethod
    def _format_suggestion_block(suggestions: Sequence[str]) -> str:
        unique: List[str] = []
        for suggestion in suggestions:
            if suggestion not in unique:
                unique.append(suggestion)
        if not unique:
            return ""
        lines = "\n".join(f"- {label}" for label in unique[:3])
        return f"\nHere are a few portfolio addresses you can reference:\n{lines}"


__all__ = ["ClarificationAgent"]

