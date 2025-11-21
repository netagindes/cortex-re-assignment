"""
Agent returning detailed information about a single asset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

from app.data_layer import load_assets


@dataclass
class AssetDetailsAgent:
    """
    Retrieves a single row and serializes it into a dictionary.
    """

    def run(self, address: str) -> Dict[str, Any]:
        df = load_assets()
        if "address" not in df.columns:
            raise ValueError("Dataset missing 'address' column required for lookups.")

        match = df[df["address"].str.contains(address, case=False, na=False)]
        if match.empty:
            raise ValueError(f"Address '{address}' not found.")

        return match.iloc[0].to_dict()


__all__ = ["AssetDetailsAgent"]

