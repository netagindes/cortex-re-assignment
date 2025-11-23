"""
Shared request-type literals used by supervisor-adjacent components.
"""

from typing import Literal

RequestType = Literal[
    "price_comparison",
    "pnl",
    "asset_details",
    "general",
    "clarification",
]

__all__ = ["RequestType"]


