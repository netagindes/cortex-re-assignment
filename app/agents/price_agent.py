"""
Agent informing the user that price comparisons are unsupported.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.prompts.price_prompt import PRICE_COMPARISON_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class PriceComparisonAgent:
    """
    Communicate dataset limitations whenever a price comparison is requested.
    """

    SYSTEM_PROMPT: str = PRICE_COMPARISON_SYSTEM_PROMPT

    def run(self, property_a: Optional[str] = None, property_b: Optional[str] = None) -> Dict[str, Any]:
        logger.info("PriceComparisonAgent invoked for %s vs %s", property_a, property_b)
        properties = [value for value in (property_a, property_b) if value]
        message = self._build_message(properties)
        suggestions = self._build_suggestions(properties)
        payload: Dict[str, Any] = {
            "supported": False,
            "reason": "no_price_data",
            "message": message,
        }
        if suggestions:
            payload["suggestions"] = suggestions
        logger.info("PriceComparisonAgent returning dataset limitation response for %s", properties or "price request")
        return payload

    def _build_message(self, properties: List[str]) -> str:
        if not properties:
            subject = "those assets"
        elif len(properties) == 1:
            subject = properties[0]
        else:
            subject = f"{properties[0]} and {properties[1]}"
        return (
            "Price or valuation data is not part of the ledger dataset, "
            f"so I can't compare the value of {subject}. "
            "The available rows only include entity_name, property_name, tenant_name, "
            "ledger_type/group/category/code/description, month, quarter, year, and the signed profit column. "
            "Please ask for a P&L summary or property details instead."
        )

    def _build_suggestions(self, properties: List[str]) -> List[str]:
        if properties:
            target = properties[0]
        else:
            target = "Building 180"
        return [
            f"Ask: 'Show me the P&L for {target} for 2025.'",
            f"Ask: 'Tell me about {target}.'",
        ]


__all__ = ["PriceComparisonAgent"]

