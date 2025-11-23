"""
LangGraph workflow construction.
"""

from __future__ import annotations

import re

from typing import Any, Dict, List, Optional

from langgraph.graph import END, StateGraph

from app import tools
from app.agents import (
    AssetDetailsAgent,
    ClarificationAgent,
    PnLAgent,
    PriceComparisonAgent,
    SupervisorAgent,
)
from app.graph.state import GraphState, QueryContext

supervisor = SupervisorAgent()
pnl_agent = PnLAgent()
price_agent = PriceComparisonAgent()
asset_agent = AssetDetailsAgent()
clarify_agent = ClarificationAgent()

_YEAR_PATTERN = re.compile(r"^\d{4}$")
_QUARTER_PATTERN = re.compile(r"^\d{4}-Q[1-4]$", re.IGNORECASE)
_MONTH_PATTERN = re.compile(r"^\d{4}-M\d{2}$", re.IGNORECASE)


def build_workflow() -> StateGraph:
    graph = StateGraph(GraphState)

    graph.add_node("supervisor", _classify_node)
    graph.add_node("price_comparison", _price_node)
    graph.add_node("pnl", _pnl_node)
    graph.add_node("asset_details", _asset_node)
    graph.add_node("clarification", _clarification_node)

    graph.add_conditional_edges(
        "supervisor",
        _route_from_supervisor,
        {
            "price_comparison": "price_comparison",
            "pnl": "pnl",
            "asset_details": "asset_details",
            "clarification": "clarification",
            "general": "clarification",
        },
    )
    graph.add_edge("price_comparison", END)
    graph.add_edge("pnl", END)
    graph.add_edge("asset_details", END)
    graph.add_edge("clarification", END)

    graph.set_entry_point("supervisor")
    return graph


def _classify_node(state: GraphState) -> GraphState:
    state.log("Supervisor classification started")
    decision = supervisor.analyze(state.context.user_input)
    state.context.request_type = decision.request_type
    state.context.addresses = decision.addresses
    state.context.suggested_addresses = decision.suggested_addresses
    state.context.address_matches = decision.address_matches
    state.context.candidate_terms = decision.candidate_terms
    state.context.unresolved_terms = decision.unresolved_terms
    state.context.missing_addresses = decision.missing_addresses
    state.context.notes = decision.notes
    state.context.period = decision.period
    state.context.period_level = decision.period_level
    state.context.entity_name = decision.entity_name
    state.context.property_name = decision.property_name
    state.context.tenant_name = decision.tenant_name
    state.context.year = decision.year
    state.context.quarter = decision.quarter
    state.context.month = decision.month
    state.context.needs_clarification = decision.needs_clarification
    state.context.clarification_reasons = decision.missing_requirements
    state.log(
        "Supervisor classification complete",
        request_type=decision.request_type,
        addresses=decision.addresses,
        period=decision.period,
    )
    return state


def _route_from_supervisor(state: GraphState) -> str:
    destination = state.context.request_type or "clarification"
    state.log("Routing to specialist", destination=destination)
    return destination


def _price_node(state: GraphState) -> GraphState:
    addresses = state.context.addresses or []
    suggestions = state.context.suggested_addresses or []
    missing_addresses = state.context.missing_addresses or []
    state.log("Price comparison node entered", addresses=addresses)
    if not tools.has_price_data():
        message = _build_price_capability_message()
        state.result = {"message": message}
        state.log("Price comparison skipped - missing price data")
        return state
    if len(addresses) < 2 and missing_addresses:
        state.result = {"message": _build_missing_property_message(missing_addresses, suggestions)}
        state.log("Price comparison skipped - properties not in dataset", missing=missing_addresses)
        return state
    if len(addresses) < 2:
        state.result = {"message": _build_price_prompt(suggestions)}
        state.log("Price comparison skipped due to insufficient addresses")
        return state
    try:
        state.result = price_agent.run(addresses[0], addresses[1])
    except ValueError as exc:
        fallback = _attempt_price_fallback(state, str(exc))
        if fallback:
            state.result = fallback
            state.log(
                "Price comparison fallback succeeded",
                property_a=fallback["property_a"]["address"],
                property_b=fallback["property_b"]["address"],
            )
            return state
        state.result = {
            "message": _build_price_error_message(str(exc), suggestions, missing_addresses=missing_addresses)
        }
        state.log("Price comparison failed", error=str(exc))
        return state
    state.log(
        "Price comparison completed",
        property_a=state.result["property_a"]["address"],
        property_b=state.result["property_b"]["address"],
        difference=state.result["difference"],
    )
    return state


def _pnl_node(state: GraphState) -> GraphState:
    state.log("PnL node entered", period=state.context.period)
    if state.context.request_type == "pnl" and state.context.needs_clarification:
        state.result = {
            "message": (
                "I need the time period (year, quarter, or month) to calculate P&L. "
                "For example: 'What is the total P&L for 2025?'"
            )
        }
        state.log("PnL aggregation skipped - clarification required")
        return state
    try:
        task = _build_pnl_task(state.context)
        state.result = pnl_agent.run(task)
        state.pnl_result = state.result
    except ValueError as exc:
        state.result = {"message": str(exc)}
        state.log("PnL aggregation failed", error=str(exc))
        return state
    state.log("PnL aggregation completed", total=state.result["value"])
    return state


def _build_pnl_task(context: QueryContext) -> Dict[str, Any]:
    year = context.year
    quarter = context.quarter
    month = context.month
    label = context.period

    if label:
        if year is None and _YEAR_PATTERN.match(label):
            year = int(label)
        elif _QUARTER_PATTERN.match(label):
            normalized = label.upper()
            quarter = quarter or normalized
            year = year or int(normalized[:4])
        elif _MONTH_PATTERN.match(label):
            normalized = label.upper()
            month = month or normalized
            year = year or int(normalized[:4])

    level = context.period_level
    if level is None:
        if month:
            level = "month"
        elif quarter:
            level = "quarter"
        else:
            level = "year"

    return {
        "entity_name": context.entity_name,
        "property_name": context.property_name,
        "tenant_name": context.tenant_name,
        "year": year,
        "quarter": quarter.upper() if isinstance(quarter, str) else quarter,
        "month": month.upper() if isinstance(month, str) else month,
        "level": level,
        "label": label,
    }


def _asset_node(state: GraphState) -> GraphState:
    state.log("Asset detail node entered", addresses=state.context.addresses)
    if not state.context.addresses:
        state.result = {"message": "Please include the property address you want details about."}
        state.log("Asset detail lookup skipped - no address provided")
        return state
    try:
        state.result = asset_agent.run(state.context.addresses[0])
    except ValueError as exc:
        state.result = {"message": str(exc)}
        state.log("Asset detail lookup failed", error=str(exc))
        return state
    state.log("Asset detail lookup completed", address=state.context.addresses[0])
    return state


def _clarification_node(state: GraphState) -> GraphState:
    state.log("Clarification node entered")
    state.result = clarify_agent.run(
        state.context.user_input,
        request_type=state.context.request_type,
        reasons=state.context.clarification_reasons,
        suggestions=state.context.suggested_addresses,
    )
    state.log("Clarification prompt generated")
    return state


__all__ = ["build_workflow"]


def _build_price_prompt(suggestions: List[str]) -> str:
    base = (
        "Please mention two properties (e.g., Building 120 and Building 160) "
        "so I can compare their values."
    )
    if suggestions:
        preview = " or ".join(suggestions[:2]) if len(suggestions) > 1 else suggestions[0]
        return f"{base} Try {preview}."
    return base


def _build_price_error_message(error: str, suggestions: List[str], *, missing_addresses: List[str]) -> str:
    message = f"I couldn't compare those properties: {error}"
    if missing_addresses:
        message += f" These locations are not in the dataset: {', '.join(missing_addresses)}."
    if suggestions:
        message += f" You can reference {', '.join(suggestions[:2])} instead."
    return message


def _attempt_price_fallback(state: GraphState, error_message: str) -> Optional[Dict[str, Any]]:
    addresses = list(state.context.addresses or [])
    if len(addresses) < 1:
        return None
    matches = state.context.address_matches or []
    if not matches:
        return None

    missing = _extract_missing_address(error_message)
    candidate_addresses: List[str] = []
    for entry in matches:
        addr = entry.get("address")
        if addr and addr not in candidate_addresses:
            candidate_addresses.append(addr)

    fallback_pool = [addr for addr in candidate_addresses if addr not in addresses]
    for alternative in fallback_pool:
        new_pair = list(addresses)
        if len(new_pair) < 2:
            new_pair.append(alternative)
        else:
            replace_idx = 0
            if missing and missing in new_pair:
                replace_idx = new_pair.index(missing)
            elif len(new_pair) > 1:
                replace_idx = 1
            new_pair[replace_idx] = alternative
        if len(new_pair) < 2:
            continue
        try:
            result = price_agent.run(new_pair[0], new_pair[1])
        except ValueError:
            continue
        swapped = missing or "an unresolved property"
        result["note"] = (
            f"Used {alternative} instead of {swapped} because the original address "
            "didn't match the dataset."
        )
        return result
    return None


def _extract_missing_address(error_message: str) -> Optional[str]:
    match = re.search(r"Address '(.+?)' not found", error_message)
    if match:
        return match.group(1)
    return None


def _build_missing_property_message(missing: List[str], suggestions: List[str]) -> str:
    base = f"I couldn't find these properties in the dataset: {', '.join(missing)}."
    if suggestions:
        base += f" Try {', '.join(suggestions[:2])} instead."
    return base


def _build_price_capability_message() -> str:
    return (
        "Property valuation data isn't included in the current dataset, so I can't compare prices. "
        "I can still summarize P&L or other metrics if that helps."
    )

