"""
Agent responsible for profit & loss aggregation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd

from app.data_layer import load_assets
from app.tools import format_currency

logger = logging.getLogger(__name__)


@dataclass
class PnLAgent:
    """
    Performs simple aggregations over the assets dataset.
    """

    def run(self, period: str | None = None) -> Dict[str, Any]:
        logger.info("PnLAgent invoked for period=%s", period or "all")
        df = load_assets()
        if "pnl" not in df.columns:
            raise ValueError("Dataset missing 'pnl' column required for this agent.")

        total_pnl = float(df["pnl"].sum())
        response = {
            "label": f"Total P&L{f' for {period}' if period else ''}",
            "value": total_pnl,
            "formatted": format_currency(total_pnl),
            "record_count": len(df),
        }
        logger.info("PnLAgent total pnl=%s over %s assets", total_pnl, len(df))
        return response


__all__ = ["PnLAgent"]

