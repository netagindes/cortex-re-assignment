"""
Agent returning detailed information about a single asset.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict

from app.data_layer import load_assets

logger = logging.getLogger(__name__)


@dataclass
class AssetDetailsAgent:
    """
    Retrieves a single row and serializes it into a dictionary.
    """

    def run(self, address: str) -> Dict[str, Any]:
        logger.info("AssetDetailsAgent invoked for address contains '%s'", address)
        df = load_assets()
        if "address" not in df.columns:
            raise ValueError("Dataset missing 'address' column required for lookups.")

        match = df[df["address"].str.contains(address, case=False, na=False)]
        if match.empty:
            raise ValueError(f"Address '{address}' not found.")

        record = match.iloc[0].to_dict()
        logger.info("AssetDetailsAgent returning record for %s", record.get("address", "unknown"))
        return record


__all__ = ["AssetDetailsAgent"]

