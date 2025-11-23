"""
Hybrid intent parser used by the supervisor to understand user requests.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage

try:
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover - optional dependency
    ChatOpenAI = None  # type: ignore

from app.agents.request_types import RequestType

logger = logging.getLogger(__name__)
_INTENT_MODEL_ENV = "SUPERVISOR_INTENT_MODEL"
_DEFAULT_INTENT_MODEL = "gpt-4o-mini"

_PRICE_MARKERS = ("compare", "comparison", "price", "value", "worth", "valuation", "versus", "vs", "delta")
_PNL_MARKERS = ("p&l", "pnl", "profit", "loss", "income", "statement", "ledger")
_DETAIL_MARKERS = ("detail", "describe", "tell me about", "info", "information", "tenant", "occupancy")
_COMPARISON_PATTERNS = [r"\bversus\b", r"\bvs\.?\b", r"compared to", r"against", r"between"]


@dataclass
class IntentParseResult:
    request_type: RequestType
    address_terms: List[str] = field(default_factory=list)
    comparison_markers: List[str] = field(default_factory=list)
    entity_name: Optional[str] = None
    tenant_name: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    needs_clarification: bool = False
    missing_fields: List[str] = field(default_factory=list)
    llm_used: bool = False

    def merge(self, payload: Dict[str, Any]) -> None:
        request_type = payload.get("request_type")
        normalized_type = _normalize_request_type(request_type)
        if normalized_type:
            self.request_type = normalized_type

        def _extend(field: str, existing: List[str]) -> List[str]:
            incoming = payload.get(field) or []
            merged = existing + [value for value in incoming if value and value not in existing]
            return merged

        self.address_terms = _extend("address_terms", self.address_terms)
        self.comparison_markers = _extend("comparison_markers", self.comparison_markers)
        self.notes = _extend("notes", self.notes)
        self.missing_fields = _extend("missing_fields", self.missing_fields)

        self.entity_name = payload.get("entity_name") or self.entity_name
        self.tenant_name = payload.get("tenant_name") or self.tenant_name
        if isinstance(payload.get("needs_clarification"), bool):
            self.needs_clarification = self.needs_clarification or bool(payload["needs_clarification"])
        if payload:
            self.llm_used = True


class IntentParser:
    """
    Applies a deterministic rule set and optionally enhances the result via an LLM call.
    """

    def __init__(self, model_name: Optional[str] = None, enable_llm: Optional[bool] = None) -> None:
        self._llm = None
        use_llm = enable_llm if enable_llm is not None else bool(os.getenv("OPENAI_API_KEY"))
        if use_llm and ChatOpenAI is not None:
            model = model_name or os.getenv(_INTENT_MODEL_ENV, _DEFAULT_INTENT_MODEL)
            try:
                self._llm = ChatOpenAI(model=model, temperature=0, timeout=10)
                logger.info("IntentParser initialized LLM model %s", model)
            except Exception as exc:  # pragma: no cover - optional network path
                logger.warning("IntentParser failed to initialize ChatOpenAI (%s). Continuing with rules.", exc)
                self._llm = None

    def parse(self, user_input: str) -> IntentParseResult:
        base = self._rules_only_parse(user_input)
        if not user_input or not self._llm:
            return base

        payload = self._llm_enhancement(user_input)
        if payload:
            base.merge(payload)
        return base

    def _rules_only_parse(self, user_input: str) -> IntentParseResult:
        lowered = user_input.lower()
        tokens = lowered.split()
        request_type: RequestType = "general"

        if any(marker in lowered for marker in _PRICE_MARKERS):
            request_type = "price_comparison"
        elif any(marker in lowered for marker in _PNL_MARKERS):
            request_type = "pnl"
        elif any(marker in lowered for marker in _DETAIL_MARKERS):
            request_type = "asset_details"
        elif len(tokens) < 4:
            request_type = "clarification"

        address_terms = self._extract_address_terms(user_input)
        comparison_markers = [pat for pat in _COMPARISON_PATTERNS if re.search(pat, lowered)]
        needs_clarification = request_type == "price_comparison" and len(address_terms) < 2
        missing = ["second_property"] if needs_clarification else []

        return IntentParseResult(
            request_type=request_type,
            address_terms=address_terms,
            comparison_markers=comparison_markers,
            needs_clarification=needs_clarification,
            missing_fields=missing,
        )

    def _llm_enhancement(self, user_input: str) -> Optional[Dict[str, Any]]:
        if not self._llm:
            return None
        instruction = (
            "You are an intent classifier for a real-estate assistant. "
            "Respond ONLY with JSON using keys: "
            "request_type (one of price_comparison, pnl, asset_details, clarification, general), "
            "address_terms (array of key property phrases), "
            "comparison_markers (array), "
            "entity_name (string or null), "
            "tenant_name (string or null), "
            "needs_clarification (boolean), "
            "missing_fields (array), "
            "notes (array of short insights)."
        )
        message = HumanMessage(content=f"{instruction}\nUser query: ```{user_input}```")

        try:
            response = self._llm.invoke([message])
        except Exception as exc:  # pragma: no cover - optional network path
            logger.warning("IntentParser LLM call failed (%s). Using rule output.", exc)
            return None

        content = getattr(response, "content", "")
        if isinstance(content, list):
            content = " ".join(part.get("text", "") for part in content if isinstance(part, dict))
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            logger.debug("IntentParser received malformed JSON: %s", content)
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    @staticmethod
    def _extract_address_terms(user_input: str) -> List[str]:
        if not user_input:
            return []
        building_matches = re.findall(r"(building\s+\d+)", user_input, flags=re.IGNORECASE)
        separators = re.compile(r"\b(?:versus|vs\.?|and|between|compared to|with|against|to)\b", flags=re.IGNORECASE)
        pieces = [piece.strip(" ,.;:") for piece in separators.split(user_input) if piece.strip()]
        candidates: List[str] = []
        for collection in (building_matches, pieces):
            for value in collection:
                normalized = value.strip()
                if normalized and normalized not in candidates:
                    candidates.append(normalized)
        return candidates[:4]


def _normalize_request_type(value: Any) -> Optional[RequestType]:
    mapping = {
        "price_comparison": "price_comparison",
        "price": "price_comparison",
        "compare": "price_comparison",
        "pnl": "pnl",
        "p&l": "pnl",
        "asset_details": "asset_details",
        "details": "asset_details",
        "clarification": "clarification",
        "question": "clarification",
        "general": "general",
    }
    if not value:
        return None
    lowered = str(value).strip().lower()
    normalized = mapping.get(lowered)
    if normalized in mapping.values():
        return normalized  # type: ignore[return-value]
    return None


__all__ = ["IntentParser", "IntentParseResult"]


