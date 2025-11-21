"""
Agent responsible for profit & loss aggregation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict

from app import tools

logger = logging.getLogger(__name__)


@dataclass
class PnLAgent:
    """
    Performs simple aggregations over the assets dataset.
    """

    def run(self, period: str | None = None) -> Dict[str, Any]:
        logger.info("PnLAgent invoked for period=%s", period or "all")
        response = tools.compute_portfolio_pnl(period)
        logger.info("PnLAgent total pnl=%s", response["value"])
        return response


__all__ = ["PnLAgent"]

