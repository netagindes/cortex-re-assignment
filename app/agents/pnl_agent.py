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

    def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("PnLAgent invoked with filters %s", {k: v for k, v in task.items() if v})
        response = tools.compute_portfolio_pnl(**task)
        logger.info("PnLAgent total pnl=%s", response["value"])
        return response


__all__ = ["PnLAgent"]

