"""
Supervisor agent responsible for classifying and routing user queries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Literal, Optional

from app import tools

logger = logging.getLogger(__name__)

RequestType = Literal["price_comparison", "pnl", "asset_details", "general", "clarification"]


@dataclass
class SupervisorDecision:
    request_type: RequestType
    addresses: List[str] = field(default_factory=list)
    period: Optional[str] = None
    period_level: Optional[str] = None
    entity_name: Optional[str] = None
    property_name: Optional[str] = None
    tenant_name: Optional[str] = None
    missing_requirements: List[str] = field(default_factory=list)
    needs_clarification: bool = False


@dataclass
class SupervisorAgent:
    """
    Naive classifier that will later be replaced with an LLM-based router.
    """

    def classify(self, user_input: str) -> RequestType:
        text = user_input.lower()
        price_markers = ("compare", "price", "value", "worth", "valuation")
        pnl_markers = ("p&l", "pnl", "profit", "loss", "income", "statement", "ledger")
        detail_markers = ("detail", "tell me about", "tenant", "info", "information")

        if any(marker in text for marker in price_markers):
            classification = "price_comparison"
        elif any(marker in text for marker in pnl_markers):
            classification = "pnl"
        elif any(marker in text for marker in detail_markers):
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
        property_name = addresses[0] if addresses else None
        missing: List[str] = []

        if request_type == "price_comparison" and len(addresses) < 2:
            missing.append("second_property")
        if request_type == "asset_details" and not addresses:
            missing.append("property")
        if request_type == "pnl" and period is None:
            missing.append("period")

        needs_clarification = bool(missing)

        logger.info(
            "Supervisor analysis: request_type=%s addresses=%s period=%s",
            request_type,
            addresses,
            period,
        )
        return SupervisorDecision(
            request_type=request_type,
            addresses=addresses,
            period=period,
            property_name=property_name,
            missing_requirements=missing,
            needs_clarification=needs_clarification,
        )


__all__ = ["SupervisorAgent", "SupervisorDecision", "RequestType"]

