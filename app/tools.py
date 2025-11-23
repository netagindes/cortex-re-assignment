"""
Shared utility functions and tool abstractions used across agents.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from datetime import datetime
from difflib import get_close_matches
from functools import lru_cache
from typing import Any, Dict, List, Literal, Optional, Sequence

import pandas as pd

from app.config import load_address_aliases
from app.data_layer import load_assets
from app.knowledge import PropertyMatch, PropertyMemoryResult, get_property_memory


# --------------------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------------------


def format_currency(value: float, currency: str = "USD") -> str:
    return f"{currency} {value:,.2f}"


def build_response_payload(result: Any, metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
    payload = {"result": result}
    if metadata:
        payload["metadata"] = metadata
    return payload


# --------------------------------------------------------------------------------------
# Supervisor helper utilities
# --------------------------------------------------------------------------------------


def search_properties(text: str, top_k: int = 5) -> List[PropertyMatch]:
    """
    Wrapper around PropertyMemory.search for callers that only need the ranked matches.
    """

    memory = get_property_memory()
    return memory.search(text, top_k=top_k)


@dataclass
class PropertyResolution:
    matches: List[PropertyMatch] = field(default_factory=list)
    candidate_terms: List[str] = field(default_factory=list)
    unresolved_terms: List[str] = field(default_factory=list)
    missing_assets: List[str] = field(default_factory=list)


def resolve_property_mentions(text: str, max_matches: int = 2) -> PropertyMemoryResult:
    """
    Resolve natural-language property mentions into dataset-backed matches.
    """

    memory = get_property_memory()
    return memory.resolve_mentions(text, expected=max_matches)


def resolve_properties(text: str, max_matches: int = 2) -> PropertyResolution:
    """
    Hybrid resolver combining alias/literal extraction with PropertyMemory lookups
    and validating each candidate against the dataset.
    """

    memory = get_property_memory()
    memory_result = memory.resolve_mentions(text, expected=max_matches * 2)
    candidate_terms = list(memory_result.candidate_terms)
    unresolved_terms = list(memory_result.unresolved_terms)
    missing_assets: List[str] = []
    matches: List[PropertyMatch] = []
    seen: set[str] = set()

    def _append(match: PropertyMatch, reason: Optional[str] = None) -> None:
        address = (match.address or "").strip()
        if not address:
            return
        key = address.lower()
        if key in seen:
            return
        seen.add(key)
        updated = match if reason is None else replace(match, reason=reason)
        matches.append(replace(updated, rank=len(matches) + 1))

    alias_hits = extract_addresses(text, max_matches=max_matches * 2)
    for alias in alias_hits:
        if alias not in candidate_terms:
            candidate_terms.append(alias)
        try:
            record = _lookup_asset(alias)
        except ValueError:
            missing_assets.append(alias)
            placeholder = PropertyMatch(
                address=alias,
                property_name=alias,
                confidence=0.0,
                rank=0,
                reason="User provided address (not in dataset)",
                metadata={},
            )
            _append(placeholder)
            continue
        metadata = _record_metadata(record)
        match = PropertyMatch(
            address=str(record.get("address") or alias),
            property_name=str(record.get("property_name") or record.get("address") or alias),
            confidence=1.0,
            rank=0,
            reason="Alias or direct match",
            metadata=metadata,
        )
        _append(match)

    for match in memory_result.matches:
        _append(match)

    candidate_terms = list(dict.fromkeys(candidate_terms))
    unresolved_terms.extend(missing_assets)
    return PropertyResolution(
        matches=matches,
        candidate_terms=candidate_terms,
        unresolved_terms=unresolved_terms,
        missing_assets=missing_assets,
    )


def suggest_alternative_properties(
    *,
    exclude: Sequence[str] | None = None,
    limit: int = 3,
) -> List[str]:
    """
    Provide user-friendly property names for clarification prompts.
    """

    memory = get_property_memory()
    excluded = {item.lower() for item in (exclude or []) if item}
    suggestions: List[str] = []

    for record in memory.records:
        label = record.property_name or record.address
        if not label:
            continue
        if label.lower() in excluded:
            continue
        suggestions.append(label)
        if len(suggestions) >= limit:
            break
    return suggestions


@lru_cache(maxsize=1)
def _address_aliases() -> Dict[str, str]:
    return load_address_aliases()


@lru_cache(maxsize=1)
def _known_properties() -> List[str]:
    df = load_assets(columns=["address", "property_name"])
    names: List[str] = []
    for column in ("address", "property_name"):
        if column in df.columns:
            names.extend(df[column].dropna().astype(str).tolist())
    seen = set()
    unique_names: List[str] = []
    for name in names:
        normalized = name.strip()
        if normalized and normalized not in seen:
            unique_names.append(normalized)
            seen.add(normalized)
    return unique_names


@lru_cache(maxsize=1)
def _known_tenants() -> List[str]:
    df = load_assets(columns=["tenant_name"])
    if "tenant_name" not in df.columns:
        return []
    tenants = df["tenant_name"].dropna().astype(str).unique().tolist()
    return [tenant for tenant in tenants if tenant.strip()]


def _building_pattern_matches(text: str) -> List[str]:
    pattern = re.findall(r"(building\s+\d+)", text, flags=re.IGNORECASE)
    return [match.title() for match in pattern]


def _fuzzy_property_match(text: str) -> Optional[str]:
    candidates = _known_properties()
    if not candidates:
        return None
    matches = get_close_matches(text, candidates, n=1, cutoff=0.75)
    return matches[0] if matches else None


def extract_addresses(text: str, max_matches: int = 2) -> List[str]:
    lowered = text.lower()
    matches: List[str] = []
    alias_map = _address_aliases()

    def _record(address: str) -> bool:
        if address and address not in matches:
            matches.append(address)
        return len(matches) >= max_matches

    for alias, canonical in alias_map.items():
        if alias in lowered and _record(canonical):
            return matches

    for address in _known_properties():
        if address.lower() in lowered and _record(address):
            return matches

    for pattern_match in _building_pattern_matches(text):
        if _record(pattern_match):
            return matches

    if len(matches) < max_matches:
        candidate = _fuzzy_property_match(text)
        if candidate:
            _record(candidate)
    return matches


def extract_tenant_names(text: str, max_matches: int = 2) -> List[str]:
    lowered = text.lower()
    matches: List[str] = []
    tenants = sorted(_known_tenants(), key=lambda value: (-len(value), value))
    for tenant in tenants:
        if tenant.lower() in lowered and tenant not in matches:
            matches.append(tenant)
            if len(matches) >= max_matches:
                break
    return matches


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


def extract_period_hint(text: str) -> Dict[str, Optional[str] | Optional[int]]:
    """
    Parse a natural-language period hint into dataset-ready filters.

    Returns a dict with keys: label, level, year, quarter, month.
    """

    lowered = text.lower()
    now = datetime.utcnow()

    def build_response(
        *,
        label: Optional[str] = None,
        level: Optional[str] = None,
        year: Optional[int] = None,
        quarter: Optional[str] = None,
        month: Optional[str] = None,
    ) -> Dict[str, Optional[str] | Optional[int]]:
        return {
            "label": label,
            "level": level,
            "year": year,
            "quarter": quarter,
            "month": month,
        }

    if "this year" in lowered or "ytd" in lowered:
        year = now.year
        return build_response(label=str(year), level="year", year=year)
    if "last year" in lowered:
        year = now.year - 1
        return build_response(label=str(year), level="year", year=year)
    if "this quarter" in lowered:
        quarter = f"{now.year}-Q{((now.month - 1) // 3) + 1}"
        return build_response(label=quarter, level="quarter", year=now.year, quarter=quarter)
    if "last quarter" in lowered:
        q = ((now.month - 1) // 3) + 1
        year = now.year
        q -= 1
        if q == 0:
            q = 4
            year -= 1
        quarter = f"{year}-Q{q}"
        return build_response(label=quarter, level="quarter", year=year, quarter=quarter)

    quarter_match = re.search(r"(20\d{2})[-\s]?(q[1-4])", lowered)
    if quarter_match:
        year = int(quarter_match.group(1))
        quarter_suffix = quarter_match.group(2).upper()
        quarter = f"{year}-{quarter_suffix}"
        return build_response(label=quarter, level="quarter", year=year, quarter=quarter)

    quarter_match_alt = re.search(r"(q[1-4])\s*(20\d{2})", lowered)
    if quarter_match_alt:
        quarter_suffix = quarter_match_alt.group(1).upper()
        year = int(quarter_match_alt.group(2))
        quarter = f"{year}-{quarter_suffix}"
        return build_response(label=quarter, level="quarter", year=year, quarter=quarter)

    for name, code in _MONTH_MAP.items():
        if name in lowered:
            year_match = re.search(r"(20\d{2})", lowered)
            year = int(year_match.group(1)) if year_match else now.year
            month_label = f"{year}-{code}"
            return build_response(label=month_label, level="month", year=year, month=month_label)

    year_match = re.search(r"\b(20\d{2})\b", lowered)
    if year_match:
        year = int(year_match.group(1))
        return build_response(label=str(year), level="year", year=year)

    return build_response()


# --------------------------------------------------------------------------------------
# Data access helpers
# --------------------------------------------------------------------------------------


def _require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")


def _lookup_asset(address_query: str, df: Optional[pd.DataFrame] = None) -> pd.Series:
    df = df if df is not None else load_assets()
    _require_columns(df, ["address"])
    match = df[df["address"].str.contains(address_query, case=False, na=False)]
    if match.empty:
        raise ValueError(f"Address '{address_query}' not found.")
    return match.iloc[0]


def _record_metadata(record: pd.Series) -> Dict[str, Optional[str]]:
    metadata: Dict[str, Optional[str]] = {}
    for field in ("entity", "city", "state", "tenant_name"):
        if field in record.index:
            value = record[field]
            if pd.notna(value):
                metadata[field] = str(value)
    return metadata


@lru_cache(maxsize=1)
def has_price_data() -> bool:
    df = load_assets(columns=["price"])
    return "price" in df.columns


# --------------------------------------------------------------------------------------
# PnL toolkit
# --------------------------------------------------------------------------------------

PeriodLevel = Literal["month", "quarter", "year"]


def filter_ledger(
    *,
    entity_name: Optional[str] = None,
    property_name: Optional[str] = None,
    tenant_name: Optional[str] = None,
    year: Optional[int] = None,
    quarter: Optional[str] = None,
    month: Optional[str] = None,
) -> pd.DataFrame:
    """
    Filter the ledger rows according to the provided dimensions.
    """

    df = load_assets()
    mask = pd.Series(True, index=df.index)

    def _apply(column: str, value: Optional[str]) -> None:
        nonlocal mask
        if value is None or column not in df.columns:
            return
        mask &= df[column].astype(str).str.lower() == str(value).lower()

    if entity_name:
        col = "entity" if "entity" in df.columns else "entity_name"
        _apply(col, entity_name)
    if property_name:
        _apply("property_name", property_name)
    if tenant_name:
        _apply("tenant_name", tenant_name)
    if year is not None and "year" in df.columns:
        mask &= df["year"].astype("Int64") == int(year)
    if quarter and "quarter" in df.columns:
        _apply("quarter", quarter)
    if month and "month" in df.columns:
        _apply("month", month)

    return df[mask].copy()


def aggregate_pnl(
    df: pd.DataFrame,
    *,
    period_level: PeriodLevel = "month",
    include_breakdown: bool = True,
) -> Dict[str, Any]:
    """
    Aggregate a filtered ledger DataFrame into P&L metrics.
    """

    if df.empty:
        return {"totals": pd.DataFrame(), "breakdown": None}

    working = df.copy()
    if "type" not in working.columns:
        working["type"] = working["pnl"].apply(lambda value: "revenue" if float(value) >= 0 else "expenses")

    period_column_map = {
        "month": "month",
        "quarter": "quarter",
        "year": "year",
    }
    period_col = period_column_map[period_level]
    base_group = []
    for column in ("entity", "property_name", period_col):
        if column in working.columns:
            base_group.append(column)

    breakdown_group = list(base_group)
    for column in ("type", "group", "category"):
        if column in working.columns:
            breakdown_group.append(column)

    breakdown_df = None
    if include_breakdown and breakdown_group:
        breakdown_df = (
            working.groupby(breakdown_group, dropna=False)["pnl"]
            .sum()
            .reset_index()
            .rename(columns={"pnl": "amount"})
        )

    group_keys = list(base_group)
    if "type" in working.columns:
        group_keys.append("type")

    ledger_totals = (
        working.groupby(group_keys, dropna=False)["pnl"]
        .sum()
        .reset_index()
        .rename(columns={"pnl": "amount"})
    )

    pivot = (
        ledger_totals.pivot_table(
            index=base_group,
            columns="type",
            values="amount",
            fill_value=0.0,
        )
        .reset_index()
    )
    pivot["total_revenue"] = pivot.get("revenue", 0.0)
    expenses = pivot.get("expenses", 0.0)
    if hasattr(expenses, "abs"):
        expenses = expenses.abs()
    else:
        expenses = abs(expenses)
    pivot["total_expenses"] = expenses
    pivot["net_operating_income"] = pivot["total_revenue"] - pivot["total_expenses"]

    return {
        "totals": pivot,
        "breakdown": breakdown_df,
    }


def pnl_by_property(rows: pd.DataFrame, limit: int = 5) -> List[Dict[str, Any]]:
    if rows.empty or "pnl" not in rows.columns:
        return []
    column = "address" if "address" in rows.columns else "property_name"
    if column not in rows.columns:
        return []
    ranked = rows.groupby(column)["pnl"].sum().sort_values(ascending=False).head(limit)
    return [{"address": address or "Portfolio", "pnl": float(value)} for address, value in ranked.items()]


def _infer_period_level(year: Optional[int], quarter: Optional[str], month: Optional[str]) -> PeriodLevel:
    if month:
        return "month"
    if quarter:
        return "quarter"
    return "year"


def _resolve_period_label(
    *,
    label: Optional[str],
    year: Optional[int],
    quarter: Optional[str],
    month: Optional[str],
) -> str:
    if label:
        return label
    if month:
        return month
    if quarter:
        return quarter
    if year:
        return str(year)
    return "all data"


def _describe_subject(property_name: Optional[str], tenant_name: Optional[str]) -> str:
    parts = []
    if property_name:
        parts.append(property_name)
    if tenant_name:
        parts.append(f"Tenant {tenant_name}")
    if not parts:
        return "portfolio"
    return " / ".join(parts)


def compute_portfolio_pnl(
    *,
    level: Optional[PeriodLevel] = None,
    label: Optional[str] = None,
    entity_name: Optional[str] = None,
    property_name: Optional[str] = None,
    tenant_name: Optional[str] = None,
    year: Optional[int] = None,
    quarter: Optional[str] = None,
    month: Optional[str] = None,
    include_breakdown: bool = True,
) -> Dict[str, Any]:
    """
    Compute P&L totals for the requested filters.
    """

    filters = {
        "entity_name": entity_name,
        "property_name": property_name,
        "tenant_name": tenant_name,
        "year": year,
        "quarter": quarter,
        "month": month,
    }
    rows = filter_ledger(**filters)
    if rows.empty:
        period_label = _resolve_period_label(label=label, year=year, quarter=quarter, month=month)
        subject = _describe_subject(property_name, tenant_name)
        no_data_message = (
            f"I couldn't find any financial data for {subject} during {period_label}. "
            "Please double-check the property or timeframe, or specify whether you need "
            "tenant-level, property-level, or combined totals."
        )
        return {
            "status": "no_data",
            "label": f"Total P&L ({period_label})",
            "value": 0.0,
            "formatted": format_currency(0.0),
            "breakdown": [],
            "record_count": 0,
            "message": no_data_message,
            "filters": {k: v for k, v in filters.items() if v is not None},
        }

    level = level or _infer_period_level(year, quarter, month)
    agg = aggregate_pnl(rows, period_level=level, include_breakdown=include_breakdown)
    totals_df: pd.DataFrame = agg["totals"]

    totals_summary = {
        "total_revenue": float(totals_df["total_revenue"].sum()),
        "total_expenses": float(totals_df["total_expenses"].sum()),
        "net_operating_income": float(totals_df["net_operating_income"].sum()),
    }
    net_value = totals_summary["net_operating_income"]

    period_label = _resolve_period_label(label=label, year=year, quarter=quarter, month=month)
    human_subject = _describe_subject(property_name, tenant_name)
    message = (
        f"P&L for {human_subject} ({period_label}): "
        f"revenue {format_currency(totals_summary['total_revenue'])}, "
        f"expenses {format_currency(totals_summary['total_expenses'])}, "
        f"net {format_currency(net_value)}."
    )

    ledger_breakdown = None
    if agg["breakdown"] is not None:
        ledger_breakdown = agg["breakdown"].to_dict(orient="records")  # type: ignore[union-attr]

    result = {
        "status": "ok",
        "label": f"Total P&L ({period_label})",
        "value": net_value,
        "formatted": format_currency(net_value),
        "breakdown": pnl_by_property(rows),
        "record_count": len(rows),
        "totals": totals_df.to_dict(orient="records"),
        "totals_summary": totals_summary,
        "ledger_breakdown": ledger_breakdown,
        "filters": {k: v for k, v in filters.items() if v is not None},
        "level": level,
        "message": message,
    }
    return result


# --------------------------------------------------------------------------------------
# Price / asset valuation toolkit
# --------------------------------------------------------------------------------------


def get_asset_value(address: str) -> float:
    record = _lookup_asset(address)
    if "price" not in record.index:
        raise ValueError("Price data is not available in the current dataset.")
    return float(record["price"])


def percent_difference(value_a: float, value_b: float) -> Optional[float]:
    if value_b == 0:
        return None
    return (value_a - value_b) / value_b * 100.0


def compare_asset_values(address_a: str, address_b: str) -> Dict[str, Any]:
    value_a = get_asset_value(address_a)
    value_b = get_asset_value(address_b)
    diff = value_a - value_b
    pct = percent_difference(value_a, value_b)
    return {
        "property_a": {"address": address_a, "price": value_a},
        "property_b": {"address": address_b, "price": value_b},
        "difference": diff,
        "difference_formatted": format_currency(diff),
        "percent_delta": pct,
    }


def get_asset_value_history(address: str, limit: int = 4) -> List[Dict[str, Any]]:
    df = load_assets()
    if not {"address", "year"}.issubset(df.columns):
        return []
    rows = df[df["address"].str.contains(address, case=False, na=False)]
    if rows.empty:
        return []
    cols = [col for col in ("year", "quarter", "month", "price", "pnl") if col in rows.columns]
    return rows.sort_values(by=[col for col in ("year", "quarter", "month") if col in cols]).tail(limit)[
        cols
    ].to_dict(orient="records")


# --------------------------------------------------------------------------------------
# Property / asset metadata toolkit
# --------------------------------------------------------------------------------------


def list_assets(limit: int = 5) -> List[Dict[str, Any]]:
    df = load_assets()
    _require_columns(df, ["address"])
    columns = [col for col in ("address", "city", "state", "price", "pnl") if col in df.columns]
    return df.head(limit)[columns].to_dict(orient="records")


def check_asset_exists(address: str) -> bool:
    df = load_assets(columns=["address"])
    if "address" not in df.columns:
        return False
    return df["address"].str.contains(address, case=False, na=False).any()


def get_asset_snapshot(address: str) -> Dict[str, Any]:
    return _lookup_asset(address).to_dict()


def explain_ledger_code(code: str) -> str:
    df = load_assets(columns=["code", "category", "group"])
    if "code" not in df.columns:
        return "Ledger code information is unavailable."
    match = df[df["code"].astype(str) == str(code)]
    if match.empty:
        return f"No ledger entries found for code {code}."
    row = match.iloc[0]
    details = []
    for field in ("category", "group"):
        if field in row and pd.notna(row[field]):
            details.append(f"{field.title()}: {row[field]}")
    return " | ".join(details) if details else f"Ledger code {code} exists in the dataset."


__all__ = [
    "format_currency",
    "build_response_payload",
    "search_properties",
    "resolve_property_mentions",
    "resolve_properties",
    "suggest_alternative_properties",
    "has_price_data",
    "extract_addresses",
    "extract_tenant_names",
    "extract_period_hint",
    # pnl toolkit
    "filter_ledger",
    "aggregate_pnl",
    "pnl_by_property",
    "compute_portfolio_pnl",
    # price toolkit
    "get_asset_value",
    "percent_difference",
    "compare_asset_values",
    "get_asset_value_history",
    # property toolkit
    "list_assets",
    "check_asset_exists",
    "get_asset_snapshot",
    "explain_ledger_code",
]


