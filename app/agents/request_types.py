"""
Centralized request-type definitions + metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple


class RequestType(str, Enum):
    PRICE_COMPARISON = "price_comparison"
    PNL = "pnl"
    ASSET_DETAILS = "asset_details"
    GENERAL = "general"
    CLARIFICATION = "clarification"


CoverageLevel = Literal["full", "partial", "missing"]


@dataclass(frozen=True)
class RequestDefinition:
    """
    Registry entry describing how each request type behaves.
    """

    type: RequestType
    title: str
    description: str
    triggers: Tuple[str, ...]
    required_inputs: Tuple[str, ...] = ()
    fallback_hint: str = ""
    measurement_id: str = ""
    coverage: CoverageLevel = "full"

    def matches_text(self, text: str) -> bool:
        lowered = text.lower()
        return any(marker in lowered for marker in self.triggers)


REQUEST_REGISTRY: Dict[RequestType, RequestDefinition] = {
    RequestType.PRICE_COMPARISON: RequestDefinition(
        type=RequestType.PRICE_COMPARISON,
        title="Price Comparison",
        description="Compare valuations between two properties.",
        triggers=(
            "compare",
            "comparison",
            "price",
            "value",
            "worth",
            "valuation",
            "versus",
            "vs",
            "delta",
        ),
        required_inputs=("address_a", "address_b"),
        fallback_hint="Please mention two portfolio properties so I can compare them.",
        measurement_id="req_price_comparison",
        coverage="partial",
    ),
    RequestType.PNL: RequestDefinition(
        type=RequestType.PNL,
        title="Profit & Loss",
        description="Aggregate P&L (revenue, expenses, NOI) for a time period.",
        triggers=(
            "p&l",
            "pnl",
            "profit",
            "loss",
            "income",
            "statement",
            "ledger",
            "noi",
        ),
        required_inputs=("timeframe",),
        fallback_hint="Specify a month, quarter, or year so I can compute P&L.",
        measurement_id="req_pnl",
        coverage="full",
    ),
    RequestType.ASSET_DETAILS: RequestDefinition(
        type=RequestType.ASSET_DETAILS,
        title="Asset Details",
        description="Return a snapshot for a single property.",
        triggers=(
            "detail",
            "details",
            "describe",
            "tell me about",
            "info",
            "information",
            "tenant",
            "occupancy",
        ),
        required_inputs=("address",),
        fallback_hint="Let me know which property you want details for.",
        measurement_id="req_asset_detail",
        coverage="full",
    ),
    RequestType.GENERAL: RequestDefinition(
        type=RequestType.GENERAL,
        title="General Knowledge",
        description="High-level questions (definitions, ledger explanations, how-to).",
        triggers=(
            "what is",
            "explain",
            "meaning of",
            "definition",
            "how does",
            "why is",
            "general",
            "help me understand",
            "ledger group",
            "ledger code",
        ),
        required_inputs=(),
        fallback_hint="Ask me something about the portfolio, metrics, or ledger codes.",
        measurement_id="req_general",
        coverage="missing",
    ),
    RequestType.CLARIFICATION: RequestDefinition(
        type=RequestType.CLARIFICATION,
        title="Clarification / Fallback",
        description="Prompt user for more information or explain unsupported requests.",
        triggers=(),
        required_inputs=(),
        fallback_hint="Please share more context so I can help.",
        measurement_id="req_clarification",
        coverage="partial",
    ),
}


REQUEST_ALIASES: Dict[str, RequestType] = {
    "price": RequestType.PRICE_COMPARISON,
    "compare": RequestType.PRICE_COMPARISON,
    "comparison": RequestType.PRICE_COMPARISON,
    "p&l": RequestType.PNL,
    "pnl": RequestType.PNL,
    "profit": RequestType.PNL,
    "loss": RequestType.PNL,
    "asset_details": RequestType.ASSET_DETAILS,
    "details": RequestType.ASSET_DETAILS,
    "asset": RequestType.ASSET_DETAILS,
    "question": RequestType.CLARIFICATION,
    "clarification": RequestType.CLARIFICATION,
    "general": RequestType.GENERAL,
}


PRIORITIZED_TYPES: Tuple[RequestType, ...] = (
    RequestType.PRICE_COMPARISON,
    RequestType.PNL,
    RequestType.ASSET_DETAILS,
    RequestType.GENERAL,
)


def normalize_request_type(value: Any, *, default: RequestType = RequestType.GENERAL) -> RequestType:
    if isinstance(value, RequestType):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if not text:
        return default
    if text in RequestType._value2member_map_:
        return RequestType(text)
    return REQUEST_ALIASES.get(text, default)


def request_definition_for(value: RequestType | str | None) -> RequestDefinition:
    req_type = normalize_request_type(value)
    return REQUEST_REGISTRY[req_type]


def all_request_definitions() -> List[RequestDefinition]:
    return list(REQUEST_REGISTRY.values())


__all__ = [
    "RequestType",
    "RequestDefinition",
    "REQUEST_REGISTRY",
    "REQUEST_ALIASES",
    "PRIORITIZED_TYPES",
    "normalize_request_type",
    "request_definition_for",
    "all_request_definitions",
]
