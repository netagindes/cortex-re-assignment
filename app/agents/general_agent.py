"""
General-knowledge agent that answers conceptual questions about the dataset and metrics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

from app import tools
from app.prompts.general_prompt import GENERAL_KNOWLEDGE_SYSTEM_PROMPT
_LEDGER_KEYWORDS = ("ledger", "code", "ledger code", "ledger group", "ledger_category")
_DATASET_KEYWORDS = ("dataset", "portfolio", "data", "assets", "profit column")
_PERIOD_KEYWORDS = ("month", "quarter", "year", "period", "2025-", "-m0", "-q")


@dataclass
class GeneralKnowledgeAgent:
    """
    Provides concise explanations that don't require a numeric computation.
    """

    SYSTEM_PROMPT: str = GENERAL_KNOWLEDGE_SYSTEM_PROMPT

    def run(self, user_input: str) -> Dict[str, object]:
        lowered = user_input.lower()
        if any(keyword in lowered for keyword in _LEDGER_KEYWORDS):
            return self._response(topic="ledger_explanation", message=self._explain_ledger(user_input))
        if any(keyword in lowered for keyword in _PERIOD_KEYWORDS):
            return self._response(topic="period_filtering", message=self._explain_periods())
        if any(keyword in lowered for keyword in _DATASET_KEYWORDS):
            return self._response(topic="dataset_overview", message=self._summarize_dataset())
        return self._response(topic="capabilities_overview", message=self._capabilities_overview())

    def _response(self, *, topic: str, message: str, details: Dict[str, object] | None = None) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "type": "general_knowledge",
            "topic": topic,
            "message": message,
        }
        if details:
            payload.update(details)
        return payload

    def _explain_ledger(self, user_input: str) -> str:
        code = self._extract_ledger_code(user_input)
        if code:
            explanation = tools.explain_ledger_code(code)
            return explanation or "Ledger codes roll up into ledger_group and ledger_category to describe account purpose."
        return (
            "Ledgers form a hierarchy: ledger_type → ledger_group → ledger_category. "
            "They describe whether a row represents revenue, expenses, parking income, discounts, etc."
        )

    @staticmethod
    def _explain_periods() -> str:
        return (
            "Periods are normalized strings: months like '2025-M03', quarters like '2025-Q1', and years like '2025'. "
            "Users mention a time range, and the system filters rows whose month/quarter/year fields match that string."
        )

    @staticmethod
    def _summarize_dataset() -> str:
        return (
            "The dataset is a ledger table where each row is a signed transaction with entity, property, tenant, "
            "ledger hierarchy columns, and period labels (month, quarter, year). It powers every specialist agent."
        )

    @staticmethod
    def _capabilities_overview() -> str:
        return (
            "I can clarify ledger hierarchies, describe the dataset columns, and explain how period labels like "
            "'2025-M03' or '2025-Q1' control filtering—always conceptually and without numeric results."
        )

    @staticmethod
    def _extract_ledger_code(user_input: str) -> str | None:
        match = re.search(r"\b(\d{3,5})\b", user_input)
        if match:
            return match.group(1)
        return None


__all__ = ["GeneralKnowledgeAgent"]

