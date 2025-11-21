"""
Agent comparing asset prices between two properties.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict

from app import tools

logger = logging.getLogger(__name__)


@dataclass
class PriceComparisonAgent:
    """
    Compare the valuation of two properties identified by address.
    """

    def run(self, property_a: str, property_b: str) -> Dict[str, Any]:
        logger.info("PriceComparisonAgent invoked for %s vs %s", property_a, property_b)
        result = tools.compare_asset_values(property_a, property_b)
        logger.info(
            "PriceComparisonAgent complete: %s vs %s delta=%s",
            result["property_a"]["address"],
            result["property_b"]["address"],
            result["difference"],
        )
        return result


__all__ = ["PriceComparisonAgent"]

