"""
LangGraph workflow construction.
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END

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

    graph.add_node("classify", _classify_node)
    graph.add_node("price_comparison", _price_node)
    graph.add_node("pnl", _pnl_node)
    graph.add_node("asset_details", _asset_node)
    graph.add_node("clarification", _clarification_node)

    graph.add_edge("classify", "price_comparison")
    graph.add_edge("classify", "pnl")
    graph.add_edge("classify", "asset_details")
    graph.add_edge("classify", "clarification")
    graph.add_edge("price_comparison", END)
    graph.add_edge("pnl", END)
    graph.add_edge("asset_details", END)
    graph.add_edge("clarification", END)

    graph.set_entry_point("classify")
    return graph


def _classify_node(state: GraphState) -> GraphState:
    req_type = supervisor.classify(state.context.user_input)
    state.context.request_type = req_type
    return state


def _price_node(state: GraphState) -> GraphState:
    addresses = state.context.addresses or []
    if len(addresses) < 2:
        state.result = {
            "message": "Please mention two property addresses so I can compare their prices."
        }
        return state
    state.result = price_agent.run(addresses[0], addresses[1])
    return state


def _pnl_node(state: GraphState) -> GraphState:
    state.result = pnl_agent.run(state.context.period)
    return state


def _asset_node(state: GraphState) -> GraphState:
    if not state.context.addresses:
        state.result = {"message": "Please include the property address you want details about."}
        return state
    state.result = asset_agent.run(state.context.addresses[0])
    return state


def _clarification_node(state: GraphState) -> GraphState:
    state.result = clarify_agent.run(state.context.user_input)
    return state


__all__ = ["build_workflow"]

