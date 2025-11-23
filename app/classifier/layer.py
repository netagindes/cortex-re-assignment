"""
LLM-backed classification layer that produces Supervisor-ready routing hints.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

try:  # pragma: no cover - optional dependency
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover - optional dependency
    ChatOpenAI = None  # type: ignore

logger = logging.getLogger(__name__)

CLASSIFIER_MODEL_ENV = "CLASSIFIER_MODEL"
DEFAULT_CLASSIFIER_MODEL = "gpt-4o-mini"

CLASSIFIER_SYSTEM_PROMPT = """
You are the CLASSIFICATION LAYER for a LangGraph-based real estate assistant.

Your ONLY job is to read a single user query and return a structured
classification called SupervisorDecision with:

- request_type: one of
    "pnl"              – numeric Profit & Loss / NOI questions
    "asset_details"    – descriptive questions about properties/tenants/entities
    "price_comparison" – questions about market price or value
    "general"          – conceptual / definitional questions
    "unsupported"      – clearly out-of-scope requests

- addresses: list of property/address strings mentioned in the query
  (e.g. ["Building 180"], ["123 Main St", "456 Oak Ave"])

- period: a normalized period string or null, where:
    "March 2025" / "Mar 2025"  → "2025-M03"
    "Q1 2025"                  → "2025-Q1"
    "2025"                     → "2025"

- comparison_periods: optional list of exactly two normalized periods for
  P&L period comparison (otherwise empty or null)

- missing_fields: list of high-level missing items, e.g. ["property", "period"]

- clarifications_needed: true/false (based ONLY on whether the user’s request
  clearly requires more info before any specialist could answer).

You NEVER:
- compute numbers,
- access the dataset,
- answer the user,
- fabricate properties or periods.

INTENT RULES
------------
1) request_type = "pnl"
   The user wants numeric financial results: P&L, NOI, revenue, expenses,
   or comparisons of those values.

   Cues:
   - Uses words like "P&L", "profit and loss", "NOI", "income",
     "revenue", "expenses" in a numeric sense.
   - Verbs like "show", "give", "compute", "calculate", "compare".
   - May include property, tenant, and period.

   HARD RULE:
   If the query mentions "P&L" or "NOI" AND refers to a specific property (or tenant)
   AND to a time period (month/quarter/year), then request_type MUST be "pnl".

   Examples:
   - "Show me the P&L for Building 180 for March 2025."
   - "What was Tenant 14’s P&L for Q1 2025 in Building 180?"
   - "Compare January and February P&L for Building 17."

2) request_type = "asset_details"
   Descriptive info, no numeric P&L requested:
   - "Which properties do we have?"
   - "List tenants in Building 180."
   - "Tell me about Building 220."

3) request_type = "price_comparison"
   Price/value/worth questions:
   - "What is the price of my asset at 123 Main St compared to 456 Oak Ave?"
   - "Which building is worth more?"

4) request_type = "general"
   Conceptual / definitional questions with:
   - NO property, and
   - NO period, and
   - NO request to compute/compare numeric P&L/NOI.

   Examples:
   - "What is NOI?"
   - "How is P&L calculated?"
   - "What does the profit column mean?"

5) request_type = "unsupported"
   Clearly outside the dataset’s scope.

ADDRESS AND PERIOD EXTRACTION
-----------------------------
- Extract property/address phrases as-is (e.g. "Building 180"). Do not invent them.
- Normalize periods as:
    "March 2025" → "2025-M03"
    "Q1 2025" → "2025-Q1"
    "2025" → "2025"
- If the user wants a comparison of P&L across two periods and explicitly names
  two periods, normalize both and put them into comparison_periods.

MISSING FIELDS
--------------
If the user clearly wants P&L but does not specify:
- a property → add "property" to missing_fields.
- a period   → add "period" to missing_fields.

Set clarifications_needed = true if at least one required field is missing.

SUMMARY
-------
Your job is to:
- choose the correct request_type,
- extract addresses and periods,
- identify obvious missing pieces.

You do NOT answer, compute, or route; you only classify and extract.
""".strip()

_REQUEST_TYPE_VALUES = {"pnl", "asset_details", "price_comparison", "general", "unsupported"}
_P_AND_L_KEYWORDS = ("p&l", "pnl", "profit and loss", "net operating income", "noi", "revenue", "expense", "expenses", "income")
_PRICE_KEYWORDS = ("price", "worth", "value", "valuation")
_ASSET_DETAIL_KEYWORDS = ("list", "show", "tell me about", "which properties", "tenants", "tenant", "property")
_COMPARISON_WORDS = ("compare", "versus", "vs", "difference", "between")
_GENERAL_QUESTION_PATTERN = re.compile(r"\b(what is|what does|meaning of|definition|explain)\b", flags=re.IGNORECASE)
_BUILDING_PATTERN = re.compile(r"(building\s+\d+)", flags=re.IGNORECASE)
_TENANT_PATTERN = re.compile(r"(tenant\s+\d+)", flags=re.IGNORECASE)
_ADDRESS_SEPARATORS = re.compile(r"\b(?:versus|vs\.?|and|between|compared to|with|against|to)\b", flags=re.IGNORECASE)
_STREET_PATTERN = re.compile(
    r"\b\d{2,5}\s+[A-Za-z0-9]+(?:\s+[A-Za-z0-9]+)*\s+(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Dr|Drive|Ln|Lane|Ct|Court)\b",
    flags=re.IGNORECASE,
)

_MONTH_MAP = {
    "january": "M01",
    "february": "M02",
    "march": "M03",
    "april": "M04",
    "may": "M05",
    "june": "M06",
    "july": "M07",
    "august": "M08",
    "september": "M09",
    "october": "M10",
    "november": "M11",
    "december": "M12",
}
_MONTH_NAMES = "|".join(_MONTH_MAP.keys())
_MONTH_YEAR_REGEX = re.compile(rf"\b(?P<month>{_MONTH_NAMES})\s+(?P<year>20\d{{2}})\b", flags=re.IGNORECASE)
_YEAR_MONTH_REGEX = re.compile(rf"\b(?P<year>20\d{{2}})\s+(?P<month>{_MONTH_NAMES})\b", flags=re.IGNORECASE)
_QUARTER_REGEX = re.compile(
    r"\b(?:(?P<year>20\d{2})[-\s]*(?P<quarter>q[1-4])|(?P<quarter_alt>q[1-4])[-\s]*(?P<year_alt>20\d{2}))\b",
    flags=re.IGNORECASE,
)
_YEAR_REGEX = re.compile(r"\b(20\d{2})\b")


@dataclass
class ClassificationResult:
    """
    Minimal routing payload returned by the classifier.
    """

    request_type: str = "general"
    addresses: List[str] = field(default_factory=list)
    period: Optional[str] = None
    comparison_periods: List[str] = field(default_factory=list)
    missing_fields: List[str] = field(default_factory=list)
    clarifications_needed: bool = False


class ClassificationLayer:
    """
    Stateless helper that calls the LLM (when available) and enforces hard routing rules.
    """

    def __init__(self, model_name: Optional[str] = None, enable_llm: Optional[bool] = None) -> None:
        self._llm = None
        use_llm = enable_llm if enable_llm is not None else bool(os.getenv("OPENAI_API_KEY"))
        if use_llm and ChatOpenAI is not None:
            model = model_name or os.getenv(CLASSIFIER_MODEL_ENV, DEFAULT_CLASSIFIER_MODEL)
            try:  # pragma: no cover - network interaction
                self._llm = ChatOpenAI(model=model, temperature=0, timeout=10)
                logger.info("ClassificationLayer initialized LLM model %s", model)
            except Exception as exc:  # pragma: no cover - optional path
                logger.warning("ClassificationLayer failed to initialize ChatOpenAI (%s). Falling back to heuristics.", exc)
                self._llm = None

    def classify(self, user_input: str) -> ClassificationResult:
        """
        Return a fully-populated ClassificationResult for the given user input.
        """

        baseline_result, hints = _baseline_classification(user_input)
        llm_result: Optional[ClassificationResult] = None
        if self._llm and user_input.strip():
            llm_result = self._invoke_llm(user_input)

        result = llm_result or baseline_result
        final = _apply_overrides(user_input, result, baseline_result, hints)
        _compute_missing_fields(final)
        return final

    def _invoke_llm(self, user_input: str) -> Optional[ClassificationResult]:
        """
        Call the configured LLM and coerce the JSON payload into a ClassificationResult.
        """

        if not self._llm:
            return None

        messages = [
            SystemMessage(content=CLASSIFIER_SYSTEM_PROMPT),
            HumanMessage(content=f"User query: ```{user_input.strip()}```\nRespond ONLY with JSON."),
        ]
        try:  # pragma: no cover - network interaction
            response = self._llm.invoke(messages)
        except Exception as exc:
            logger.warning("ClassificationLayer LLM call failed (%s). Falling back to heuristics.", exc)
            return None

        content = getattr(response, "content", "")
        if isinstance(content, list):
            content = " ".join(part.get("text", "") for part in content if isinstance(part, dict))
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            logger.debug("ClassificationLayer received malformed JSON: %s", content)
            return None

        return _result_from_payload(payload)


def _result_from_payload(payload: Any) -> Optional[ClassificationResult]:
    if not isinstance(payload, dict):
        return None

    request_type = str(payload.get("request_type", "general")).strip().lower() or "general"
    if request_type not in _REQUEST_TYPE_VALUES:
        request_type = "general"

    raw_addresses = payload.get("addresses") or payload.get("address_terms") or []
    addresses = [str(value).strip() for value in raw_addresses if isinstance(value, (str, int, float))]
    addresses = [value for value in addresses if value]

    period_value = payload.get("period")
    period = _normalize_period_string(str(period_value)) if isinstance(period_value, str) else None

    comparison_periods: List[str] = []
    raw_comparison = payload.get("comparison_periods") or []
    if isinstance(raw_comparison, list):
        for value in raw_comparison:
            normalized = _normalize_period_string(str(value)) if isinstance(value, str) else None
            if normalized and normalized not in comparison_periods:
                comparison_periods.append(normalized)
            if len(comparison_periods) == 2:
                break

    missing_fields = payload.get("missing_fields") or []
    missing = [str(field).strip().lower() for field in missing_fields if isinstance(field, (str, int, float))]
    missing = [field for field in missing if field]

    clarifications_needed = bool(payload.get("clarifications_needed"))

    return ClassificationResult(
        request_type=request_type,
        addresses=addresses,
        period=period,
        comparison_periods=comparison_periods,
        missing_fields=missing,
        clarifications_needed=clarifications_needed,
    )


def _baseline_classification(user_input: str) -> Tuple[ClassificationResult, Dict[str, bool]]:
    lowered = user_input.lower()
    addresses = _extract_addresses(user_input)
    periods = _extract_periods(user_input)

    has_compare_word = any(marker in lowered for marker in _COMPARISON_WORDS)
    comparison_periods: List[str] = []
    if has_compare_word and len(periods) >= 2:
        comparison_periods = periods[:2]

    period = None if comparison_periods else (periods[0] if periods else None)

    has_pnl = any(keyword in lowered for keyword in _P_AND_L_KEYWORDS)
    has_price = any(keyword in lowered for keyword in _PRICE_KEYWORDS)
    has_property = bool(addresses) or bool(_TENANT_PATTERN.search(lowered))
    has_period = bool(period) or len(comparison_periods) == 2
    general_question = bool(_GENERAL_QUESTION_PATTERN.search(lowered))
    has_asset_detail_term = any(keyword in lowered for keyword in _ASSET_DETAIL_KEYWORDS)

    request_type = "general"
    general_only = general_question and not has_property and not has_period
    if general_only:
        request_type = "general"
    elif has_pnl:
        request_type = "pnl"
    elif has_compare_word and len(addresses) >= 2:
        request_type = "price_comparison"
    elif has_price:
        request_type = "price_comparison"
    elif has_asset_detail_term and has_property:
        request_type = "asset_details"

    baseline = ClassificationResult(
        request_type=request_type,
        addresses=addresses,
        period=period,
        comparison_periods=comparison_periods,
    )
    hints = {
        "has_pnl": has_pnl,
        "has_property": has_property,
        "has_period": has_period,
        "has_compare_word": has_compare_word,
    }
    return baseline, hints


def _apply_overrides(
    user_input: str,
    primary: ClassificationResult,
    baseline: ClassificationResult,
    hints: Dict[str, bool],
) -> ClassificationResult:
    result = ClassificationResult(
        request_type=primary.request_type or baseline.request_type,
        addresses=list(primary.addresses or baseline.addresses),
        period=primary.period or baseline.period,
        comparison_periods=list(primary.comparison_periods or baseline.comparison_periods),
        missing_fields=list(primary.missing_fields),
        clarifications_needed=primary.clarifications_needed,
    )

    if result.request_type not in _REQUEST_TYPE_VALUES:
        result.request_type = baseline.request_type

    if not result.addresses and baseline.addresses:
        result.addresses = baseline.addresses

    if (not result.comparison_periods or len(result.comparison_periods) != 2) and len(baseline.comparison_periods) == 2:
        result.comparison_periods = baseline.comparison_periods

    if result.comparison_periods and len(result.comparison_periods) == 2:
        result.period = None
    elif not result.period and baseline.period:
        result.period = baseline.period

    hard_rule_applies = hints["has_pnl"] and hints["has_property"] and (hints["has_period"] or len(result.comparison_periods) == 2)
    if hard_rule_applies:
        result.request_type = "pnl"

    if not result.request_type:
        result.request_type = baseline.request_type

    if result.request_type not in _REQUEST_TYPE_VALUES:
        result.request_type = "general"

    missing = list(dict.fromkeys(result.missing_fields + baseline.missing_fields))
    result.missing_fields = missing

    result.clarifications_needed = result.clarifications_needed or bool(result.missing_fields)
    return result


def _compute_missing_fields(result: ClassificationResult) -> None:
    missing: List[str] = []
    if result.request_type == "pnl":
        if not result.addresses:
            missing.append("property")
        if not result.period and len(result.comparison_periods) != 2:
            missing.append("period")
    result.missing_fields = list(dict.fromkeys(missing or result.missing_fields))
    result.clarifications_needed = bool(result.missing_fields)


def _extract_addresses(user_input: str) -> List[str]:
    if not user_input:
        return []
    candidates: List[str] = []
    for match in _BUILDING_PATTERN.findall(user_input):
        normalized = match.strip()
        if normalized and normalized not in candidates:
            candidates.append(normalized)

    for match in _TENANT_PATTERN.findall(user_input):
        normalized = match.strip()
        if normalized and normalized not in candidates:
            candidates.append(normalized)

    for match in _STREET_PATTERN.findall(user_input):
        normalized = match.strip()
        if normalized and normalized not in candidates:
            candidates.append(normalized)

    pieces = [
        piece.strip(" ,.;:")
        for piece in _ADDRESS_SEPARATORS.split(user_input)
        if piece and piece.strip()
    ]
    for piece in pieces:
        trimmed = piece.strip(" ,.;:?!")
        if not trimmed:
            continue
        street_match = _STREET_PATTERN.search(trimmed)
        value = street_match.group(0).strip() if street_match else trimmed
        has_digits = bool(re.search(r"\d", value))
        has_keywords = bool(re.search(r"\b(building|tenant|suite|unit|property)\b", value, flags=re.IGNORECASE))
        if (has_digits or has_keywords) and value not in candidates:
            candidates.append(value)

    return candidates[:4]


def _extract_periods(user_input: str) -> List[str]:
    if not user_input:
        return []
    periods: List[str] = []

    def _append(value: Optional[str]) -> None:
        if value and value not in periods:
            periods.append(value)

    for match in _MONTH_YEAR_REGEX.finditer(user_input):
        month = match.group("month").lower()
        year = int(match.group("year"))
        _append(f"{year}-{_MONTH_MAP[month]}")

    for match in _YEAR_MONTH_REGEX.finditer(user_input):
        month = match.group("month").lower()
        year = int(match.group("year"))
        _append(f"{year}-{_MONTH_MAP[month]}")

    for match in _QUARTER_REGEX.finditer(user_input):
        quarter = (match.group("quarter") or match.group("quarter_alt") or "").upper()
        year_str = match.group("year") or match.group("year_alt")
        if quarter and year_str:
            _append(f"{int(year_str)}-{quarter.upper()}")

    for match in _YEAR_REGEX.finditer(user_input):
        year = match.group(1)
        _append(year)

    return periods


def _normalize_period_string(value: str) -> Optional[str]:
    value = value.strip()
    if not value:
        return None
    upper = value.upper()
    if re.fullmatch(r"20\d\d", upper):
        return upper
    if re.fullmatch(r"20\d\d-M\d\d", upper):
        return upper
    if re.fullmatch(r"20\d\d-Q[1-4]", upper):
        return upper

    month_match = _MONTH_YEAR_REGEX.search(value)
    if month_match:
        month = month_match.group("month").lower()
        year = int(month_match.group("year"))
        return f"{year}-{_MONTH_MAP[month]}"

    month_match_rev = _YEAR_MONTH_REGEX.search(value)
    if month_match_rev:
        month = month_match_rev.group("month").lower()
        year = int(month_match_rev.group("year"))
        return f"{year}-{_MONTH_MAP[month]}"

    quarter_match = _QUARTER_REGEX.search(value)
    if quarter_match:
        quarter = (quarter_match.group("quarter") or quarter_match.group("quarter_alt") or "").upper()
        year_str = quarter_match.group("year") or quarter_match.group("year_alt")
        if quarter and year_str:
            return f"{int(year_str)}-{quarter}"

    year_match = _YEAR_REGEX.search(value)
    if year_match:
        return year_match.group(1)
    return None

