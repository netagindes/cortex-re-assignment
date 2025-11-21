"""
Supervisor agent responsible for classifying and routing user queries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

RequestType = Literal["price_comparison", "pnl", "asset_details", "general", "clarification"]


@dataclass
class SupervisorAgent:
    """
    Naive classifier that will later be replaced with an LLM-based router.
    """

    def classify(self, user_input: str) -> RequestType:
        text = user_input.lower()
        if "compare" in text or "price" in text:
            classification = "price_comparison"
        elif "p&l" in text or "profit" in text:
            classification = "pnl"
        elif "detail" in text or "tell me about" in text:
            classification = "asset_details"
        elif len(user_input.strip()) < 5:
            classification = "clarification"
        else:
            classification = "general"
        logger.info("Supervisor classified '%s' as %s", user_input, classification)
        return classification


__all__ = ["SupervisorAgent", "RequestType"]

