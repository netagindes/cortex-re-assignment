"""
LangGraph workflow construction.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

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
    state.context.period = decision.period
    state.context.period_level = decision.period_level
    state.context.entity_name = decision.entity_name
    state.context.property_name = decision.property_name
    state.context.tenant_name = decision.tenant_name
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
    state.log("Price comparison node entered", addresses=addresses)
    if len(addresses) < 2:
        state.result = {
            "message": (
                "Please mention two properties (e.g., Building 120 and Building 160) "
                "so I can compare their values."
            )
        }
        state.log("Price comparison skipped due to insufficient addresses")
        return state
    try:
        state.result = price_agent.run(addresses[0], addresses[1])
    except ValueError as exc:
        state.result = {"message": str(exc)}
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
        state.result = pnl_agent.run(state.context.period)
    except ValueError as exc:
        state.result = {"message": str(exc)}
        state.log("PnL aggregation failed", error=str(exc))
        return state
    state.log("PnL aggregation completed", total=state.result["value"])
    return state


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
    )
    state.log("Clarification prompt generated")
    return state


__all__ = ["build_workflow"]

