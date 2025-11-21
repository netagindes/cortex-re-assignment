"""
Supervisor agent responsible for classifying and routing user queries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


RequestType = Literal["price_comparison", "pnl", "asset_details", "general", "clarification"]


@dataclass
class SupervisorAgent:
    """
    Naive classifier that will later be replaced with an LLM-based router.
    """

    def classify(self, user_input: str) -> RequestType:
        text = user_input.lower()
        if "compare" in text or "price" in text:
            return "price_comparison"
        if "p&l" in text or "profit" in text:
            return "pnl"
        if "detail" in text or "tell me about" in text:
            return "asset_details"
        if len(user_input.strip()) < 5:
            return "clarification"
        return "general"


__all__ = ["SupervisorAgent", "RequestType"]

