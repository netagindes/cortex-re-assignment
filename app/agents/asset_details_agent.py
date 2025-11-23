"""
Agent returning detailed information about a single asset.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict

from app import tools
from app.prompts.asset_prompt import ASSET_DETAILS_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class AssetDetailsAgent:
    """
    Retrieves a single row and serializes it into a dictionary.

    `SYSTEM_PROMPT` captures the descriptive-only behavior contract.
    """

    SYSTEM_PROMPT: str = ASSET_DETAILS_SYSTEM_PROMPT

    def run(self, address: str) -> Dict[str, Any]:
        logger.info("AssetDetailsAgent invoked for address contains '%s'", address)
        record = tools.get_asset_snapshot(address)
        logger.info("AssetDetailsAgent returning record for %s", record.get("address", "unknown"))
        return record


__all__ = ["AssetDetailsAgent"]

