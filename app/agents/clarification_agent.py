"""
Agent prompting users for additional details when the query is ambiguous.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ClarificationAgent:
    """
    Generates clarifying follow-up prompts.
    """

    def run(
        self,
        user_input: str,
        request_type: Optional[str] = None,
        reasons: Optional[List[str]] = None,
        suggestions: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        logger.info(
            "ClarificationAgent invoked for query '%s' (type=%s, reasons=%s)",
            user_input,
            request_type,
            reasons,
        )
        message = self._build_message(request_type, reasons, suggestions)
        return {
            "message": message,
            "original_query": user_input,
        }

    @staticmethod
    def _build_message(
        request_type: Optional[str],
        reasons: Optional[List[str]],
        suggestions: Optional[List[str]] = None,
    ) -> str:
        reasons = reasons or []
        targeted_reasons = [reason for reason in reasons if reason not in {"period", "granularity"}]

        reason_text = ""
        if targeted_reasons:
            friendly = ", ".join(reason.replace("_", " ") for reason in targeted_reasons)
            reason_text = f"I still need: {friendly}. "

        suggestion_text = ""
        if suggestions:
            preview = " or ".join(suggestions[:2]) if len(suggestions) > 1 else suggestions[0]
            suggestion_text = f" Try {preview}."

        period_prompt = None
        granularity_prompt = None
        if "period" in reasons:
            period_prompt = "Which period would you like (month, quarter, or year)?"
        if "granularity" in reasons:
            granularity_prompt = "Do you want tenant-level, property-level, or combined totals?"

        if request_type == "price_comparison":
            return (
                f"{reason_text}"
                "To compare property values, please mention two property names "
                "(e.g., Building 120 and Building 160)."
                f"{suggestion_text}"
            )
        if request_type == "asset_details":
            return (
                f"{reason_text}"
                "Let me know which property you want details for (for example, Building 140)."
                f"{suggestion_text}"
            )
        if request_type == "pnl":
            prompts = [prompt for prompt in (period_prompt, granularity_prompt) if prompt]
            if not prompts:
                prompts.append("Specify the time frame (year, quarter, or month) and optionally a property or tenant.")
            return (
                f"{reason_text}{' '.join(prompts)}"
                f"{suggestion_text}"
            )

        return (
            f"{reason_text}"
            "Please share a bit more detail about the assets or metrics you want me to analyze."
            f"{suggestion_text}"
        )


__all__ = ["ClarificationAgent"]

