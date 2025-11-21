"""
Agent comparing asset prices between two properties.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd

from app.data_layer import load_assets

logger = logging.getLogger(__name__)


@dataclass
class PriceComparisonAgent:
    """
    Compare the valuation of two properties identified by address.
    """

    def run(self, property_a: str, property_b: str) -> Dict[str, Any]:
        logger.info("PriceComparisonAgent invoked for %s vs %s", property_a, property_b)
        df = load_assets()
        for column in ("address", "price"):
            if column not in df.columns:
                raise ValueError(f"Dataset missing '{column}' column required for price comparisons.")

        def lookup(address: str) -> pd.Series:
            match = df[df["address"].str.contains(address, case=False, na=False)]
            if match.empty:
                raise ValueError(f"Address '{address}' not found in dataset.")
            return match.iloc[0]

        asset_a = lookup(property_a)
        asset_b = lookup(property_b)
        delta = float(asset_a["price"] - asset_b["price"])
        result = {
            "property_a": {"address": asset_a["address"], "price": float(asset_a["price"])},
            "property_b": {"address": asset_b["address"], "price": float(asset_b["price"])},
            "difference": delta,
        }
        logger.info(
            "PriceComparisonAgent complete: %s (%s) - %s (%s) = %.2f",
            asset_a["address"],
            asset_a["price"],
            asset_b["address"],
            asset_b["price"],
            delta,
        )
        return result


__all__ = ["PriceComparisonAgent"]

