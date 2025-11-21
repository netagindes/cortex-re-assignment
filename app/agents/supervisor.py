"""
Supervisor agent responsible for classifying and routing user queries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Literal, Optional

from app import tools

logger = logging.getLogger(__name__)

RequestType = Literal["price_comparison", "pnl", "asset_details", "general", "clarification"]


@dataclass
class SupervisorDecision:
    request_type: RequestType
    addresses: List[str]
    period: Optional[str] = None


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

    def analyze(self, user_input: str) -> SupervisorDecision:
        request_type = self.classify(user_input)
        addresses = tools.extract_addresses(user_input)
        period = tools.extract_period_hint(user_input)

        if request_type == "price_comparison" and len(addresses) < 2:
            request_type = "clarification"
        if request_type == "asset_details" and not addresses:
            request_type = "clarification"

        logger.info(
            "Supervisor analysis: request_type=%s addresses=%s period=%s",
            request_type,
            addresses,
            period,
        )
        return SupervisorDecision(request_type=request_type, addresses=addresses, period=period)


__all__ = ["SupervisorAgent", "SupervisorDecision", "RequestType"]

