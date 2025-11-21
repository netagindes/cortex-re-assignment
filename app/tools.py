"""
Shared utility functions and tool abstractions used across agents.
"""

from __future__ import annotations

import re
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from app.data_layer import load_assets


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


@lru_cache(maxsize=1)
def _known_addresses() -> List[str]:
    df = load_assets(columns=["address"])
    if "address" not in df.columns:
        return []
    return [addr for addr in df["address"].dropna().unique().tolist()]


def extract_addresses(text: str, max_matches: int = 2) -> List[str]:
    lowered = text.lower()
    matches: List[str] = []
    for address in _known_addresses():
        if address.lower() in lowered:
            matches.append(address)
            if len(matches) >= max_matches:
                break
    return matches


def extract_period_hint(text: str) -> Optional[str]:
    lowered = text.lower()
    current_year = datetime.utcnow().year
    if "this year" in lowered or "ytd" in lowered:
        return str(current_year)
    if "last year" in lowered:
        return str(current_year - 1)
    match = re.search(r"\b(20\d{2})\b", text)
    if match:
        return match.group(1)
    return None


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


# --------------------------------------------------------------------------------------
# PnL toolkit
# --------------------------------------------------------------------------------------


def get_ledger_rows(filters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    df = load_assets()
    if not filters:
        return df
    mask = pd.Series([True] * len(df), index=df.index)
    for column, value in (filters or {}).items():
        if not value or column not in df.columns:
            continue
        mask &= df[column].astype(str).str.contains(str(value), case=False, na=False)
    return df[mask]


def aggregate_by_category(rows: pd.DataFrame, category: str = "category") -> List[Dict[str, Any]]:
    if category not in rows.columns or "pnl" not in rows.columns:
        return []
    grouped = rows.groupby(category)["pnl"].sum().sort_values(ascending=False)
    return [
        {"category": name, "pnl": float(value), "formatted": format_currency(float(value))}
        for name, value in grouped.items()
    ]


def calculate_pnl(rows: pd.DataFrame) -> float:
    _require_columns(rows, ["pnl"])
    return float(rows["pnl"].sum())


def pnl_by_property(rows: pd.DataFrame, limit: int = 5) -> List[Dict[str, Any]]:
    if "address" not in rows.columns or "pnl" not in rows.columns:
        return []
    ranked = rows.groupby("address")["pnl"].sum().sort_values(ascending=False).head(limit)
    return [{"address": address, "pnl": float(value)} for address, value in ranked.items()]


def compute_portfolio_pnl(period: Optional[str] = None) -> Dict[str, Any]:
    filters = {"period": period} if period else None
    rows = get_ledger_rows(filters)
    total = calculate_pnl(rows)
    breakdown = pnl_by_property(rows)
    categories = aggregate_by_category(rows)

    result = {
        "label": f"Total P&L ({period or 'all data'})",
        "value": total,
        "formatted": format_currency(total),
        "breakdown": breakdown,
        "categories": categories,
        "record_count": len(rows),
    }
    if period and rows.empty:
        result["note"] = f"No rows matched period '{period}'. Showing overall totals."
    return result


# --------------------------------------------------------------------------------------
# Price / asset valuation toolkit
# --------------------------------------------------------------------------------------


def get_asset_value(address: str) -> float:
    record = _lookup_asset(address)
    if "price" not in record:
        raise ValueError("Dataset missing 'price' column.")
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
    cols = [col for col in ("year", "quarter", "month", "price") if col in rows.columns]
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
    "extract_addresses",
    "extract_period_hint",
    # pnl toolkit
    "get_ledger_rows",
    "aggregate_by_category",
    "calculate_pnl",
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


