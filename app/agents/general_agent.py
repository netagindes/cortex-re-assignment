"""
General-knowledge agent that answers portfolio, ledger, and concept questions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

from app import tools
from app.data_layer import load_assets
from app.data_layer import summarize_assets


_P_AND_L_KEYWORDS = ("p&l", "pnl", "profit and loss", "profit & loss", "noi")
_LEDGER_KEYWORDS = ("ledger", "code", "ledger code", "ledger group")
_DATASET_KEYWORDS = ("dataset", "portfolio", "data", "assets")


@dataclass
class GeneralKnowledgeAgent:
    """
    Provides concise explanations that don't require a numeric computation.
    """

    def run(self, user_input: str) -> Dict[str, object]:
        lowered = user_input.lower()
        if any(keyword in lowered for keyword in _P_AND_L_KEYWORDS):
            return self._explain_pnl()
        if any(keyword in lowered for keyword in _LEDGER_KEYWORDS):
            return self._explain_ledger(user_input)
        if any(keyword in lowered for keyword in _DATASET_KEYWORDS):
            return self._summarize_dataset()
        return self._capabilities_overview()

    def _explain_pnl(self) -> Dict[str, object]:
        steps = [
            "Gather ledger rows (revenue and expenses) for the requested period.",
            "Group by the requested granularity (month, quarter, year, property, or tenant).",
            "Aggregate totals and report revenue, expenses, and net operating income.",
        ]
        message = (
            "Profit & Loss (P&L) reports total revenue, expenses, and net operating income "
            "for the period you specify. Ask for a year, quarter, or month—and optionally a property "
            "or tenant—to trigger the P&L agent."
        )
        return {
            "topic": "pnl_overview",
            "message": message,
            "steps": steps,
            "next_actions": "Example: 'Show tenant 14 P&L for 2025-Q1.'",
        }

    def _explain_ledger(self, user_input: str) -> Dict[str, object]:
        code = self._extract_ledger_code(user_input)
        details = tools.explain_ledger_code(code) if code else "Ledger codes group expenses/revenue into categories like OPEX or Parking."
        message = (
            "Ledger descriptions come straight from the dataset. "
            "Provide a specific ledger code to retrieve its category or group."
        )
        return {
            "topic": "ledger_explanation",
            "message": message,
            "details": details,
            "next_actions": "Example: 'Explain ledger code 5100.'",
        }

    def _summarize_dataset(self) -> Dict[str, object]:
        df = load_assets()
        summary = summarize_assets(df)
        row_count = summary.get("rows", 0)
        columns = summary.get("columns", [])
        message = (
            f"The working dataset currently contains {row_count} ledger rows "
            f"across columns {', '.join(columns[:8])}."
        )
        return {
            "topic": "dataset_overview",
            "message": message,
            "columns": columns,
            "rows": row_count,
            "next_actions": "Ask for a P&L, price comparison, or property detail to drill into the data.",
        }

    def _capabilities_overview(self) -> Dict[str, object]:
        bullets: List[str] = [
            "Compare two properties if valuation data is available.",
            "Compute P&L at the portfolio, property, or tenant level.",
            "Retrieve the latest snapshot for any property in the dataset.",
            "Explain ledger codes or summarize the dataset structure.",
        ]
        return {
            "topic": "capabilities_overview",
            "message": "Here's what I can help with inside the Cortex asset manager.",
            "capabilities": bullets,
            "next_actions": "Try: 'What is the total P&L for 2025?' or 'Tell me about Building 180.'",
        }

    @staticmethod
    def _extract_ledger_code(user_input: str) -> str | None:
        match = re.search(r"\b(\d{3,5})\b", user_input)
        if match:
            return match.group(1)
        return None


__all__ = ["GeneralKnowledgeAgent"]

