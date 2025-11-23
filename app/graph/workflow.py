"""
LangGraph workflow construction.
"""

from __future__ import annotations

import re

from typing import Any, Dict, List, Optional

from langgraph.graph import END, StateGraph

from app.agents import (
    AssetDetailsAgent,
    ClarificationAgent,
    GeneralKnowledgeAgent,
    PnLAgent,
    PriceComparisonAgent,
    SupervisorAgent,
)
from app.agents.errors import AgentError, ErrorType
from app.agents.request_types import RequestType
from app.graph.state import ClarificationItem, GraphState, QueryContext

supervisor = SupervisorAgent()
pnl_agent = PnLAgent()
price_agent = PriceComparisonAgent()
asset_agent = AssetDetailsAgent()
clarify_agent = ClarificationAgent()
general_agent = GeneralKnowledgeAgent()

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
    graph.add_node("general_knowledge", _general_node)

    graph.add_conditional_edges(
        "supervisor",
        _route_from_supervisor,
        {
            RequestType.PRICE_COMPARISON.value: "price_comparison",
            RequestType.PNL.value: "pnl",
            RequestType.ASSET_DETAILS.value: "asset_details",
            RequestType.CLARIFICATION.value: "clarification",
            RequestType.GENERAL.value: "general_knowledge",
        },
    )
    graph.add_edge("price_comparison", END)
    graph.add_edge("pnl", END)
    graph.add_edge("asset_details", END)
    graph.add_edge("clarification", END)
    graph.add_edge("general_knowledge", END)

    graph.set_entry_point("supervisor")
    return graph


def _classify_node(state: GraphState) -> GraphState:
    state.log("Supervisor classification started", agent="supervisor", requirement_section="req1-routing")
    decision = supervisor.analyze(state.context)
    state.context.request_type = decision.request_type
    state.context.request_measurement = decision.measurement_id
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
    state.context.comparison_periods = decision.comparison_periods
    state.context.needs_clarification = decision.needs_clarification
    state.context.clarification_needed = decision.needs_clarification
    state.context.aggregation_level = decision.aggregation_level
    state.context.clarifications = decision.clarifications
    state.context.awaiting_user_reply = decision.awaiting_user_reply
    state.context.clarification_reasons = decision.missing_requirements
    state.log(
        "Supervisor classification complete",
        agent="supervisor",
        requirement_section="req1-routing",
        request_type=decision.request_type.value,
        addresses=decision.addresses,
        period=decision.period,
    )
    return state


def _route_from_supervisor(state: GraphState) -> str:
    if state.context.awaiting_user_reply or state.context.needs_clarification:
        destination = RequestType.CLARIFICATION.value
    else:
        request_type = state.context.request_type or RequestType.CLARIFICATION
        if isinstance(request_type, RequestType):
            destination = request_type.value
        else:
            destination = str(request_type)
    state.log(
        "Routing to specialist",
        agent="supervisor",
        requirement_section="req1-routing",
        destination=destination,
    )
    return destination


def _price_node(state: GraphState) -> GraphState:
    addresses = state.context.addresses or []
    property_a = addresses[0] if len(addresses) > 0 else None
    property_b = addresses[1] if len(addresses) > 1 else None
    state.log(
        "Price comparison node entered",
        agent="price_agent",
        requirement_section="req3-processing",
        addresses=addresses,
    )
    response = price_agent.run(property_a, property_b)
    state.result = response
    if isinstance(response, dict) and response.get("message"):
        state.explanation = response["message"]
    state.log(
        "Price comparison unsupported due to missing valuation data",
        agent="price_agent",
        requirement_section="req3-processing",
        supported=response.get("supported"),
        reason=response.get("reason"),
    )
    return state


def _pnl_node(state: GraphState) -> GraphState:
    state.log(
        "PnL node entered",
        agent="pnl_agent",
        requirement_section="req3-processing",
        period=state.context.period,
    )
    if state.context.request_type == RequestType.PNL and state.context.needs_clarification:
        reasons = set(state.context.clarification_reasons or [])
        prompts: List[str] = []
        if "period" in reasons:
            prompts.append(
                "Which period would you like (month, quarter, or year)? For example: 'What is the total P&L for 2025?'"
            )
        if "aggregation_level" in reasons:
            prompts.append("Do you want tenant-level, property-level, or combined totals?")
        if not prompts:
            prompts.append("Please share the missing details so I can calculate the P&L.")
        state.result = {
            "message": " ".join(prompts)
        }
        state.log(
            "PnL aggregation skipped - clarification required",
            agent="pnl_agent",
            requirement_section="req5-error",
        )
        return state
    try:
        task = _build_pnl_task(state.context)
        state.result = pnl_agent.run(task)
        state.pnl_result = state.result
        if isinstance(state.result, dict) and state.result.get("message"):
            state.explanation = state.result["message"]
    except ValueError as exc:
        state.result = {"message": str(exc)}
        state.log(
            "PnL aggregation failed",
            agent="pnl_agent",
            requirement_section="req5-error",
            error=str(exc),
        )
        return state
    log_data: Dict[str, Any] = {
        "agent": "pnl_agent",
        "requirement_section": "req3-processing",
    }
    if isinstance(state.result, dict):
        if "value" in state.result:
            log_data["total"] = state.result["value"]
        elif state.result.get("mode") == "comparison":
            log_data["mode"] = "comparison"
    state.log("PnL aggregation completed", **log_data)
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

    comparison_periods = context.comparison_periods or []
    level = context.period_level
    if comparison_periods:
        label = None
        level = None
        year = None
        quarter = None
        month = None
    else:
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
        "comparison_periods": comparison_periods,
    }


def _asset_node(state: GraphState) -> GraphState:
    state.log(
        "Asset detail node entered",
        agent="asset_agent",
        requirement_section="req3-processing",
        addresses=state.context.addresses,
    )
    if not state.context.addresses:
        error = AgentError(
            ErrorType.MISSING_PROPERTY,
            "Please include the property address you want details about.",
        )
        return _handle_agent_error(state, error, agent_name="asset_agent")
    try:
        state.result = asset_agent.run(state.context.addresses[0])
    except ValueError as exc:
        error = AgentError(ErrorType.UNKNOWN_PROPERTY, str(exc))
        return _handle_agent_error(state, error, agent_name="asset_agent")
    state.log(
        "Asset detail lookup completed",
        agent="asset_agent",
        requirement_section="req3-processing",
        address=state.context.addresses[0],
    )
    return state


def _clarification_node(state: GraphState) -> GraphState:
    state.log("Clarification node entered", agent="clarification_agent", requirement_section="req5-error")
    item = clarify_agent.run(
        state.context,
        request_type=state.context.request_type,
        reasons=state.context.clarification_reasons,
        suggestions=state.context.suggested_addresses,
    )
    _apply_clarification_item(state, item)
    state.log(
        "Clarification prompt generated",
        agent="clarification_agent",
        requirement_section="req5-error",
        field=item.field,
    )
    return state


def _general_node(state: GraphState) -> GraphState:
    if state.context.awaiting_user_reply and state.context.clarifications:
        pending = state.context.clarifications[-1].question
        state.result = {
            "message": f"I'm still waiting for your answer to: {pending}",
            "note": "A clarification is pending; please answer the question above.",
        }
        state.log(
            "General knowledge blocked due to pending clarification",
            agent="general_agent",
            requirement_section="req5-error",
        )
        return state
    state.log("General knowledge node entered", agent="general_agent", requirement_section="req3-processing")
    state.result = general_agent.run(state.context.user_input)
    if state.result:
        state.explanation = state.result.get("message")
    state.log(
        "General knowledge response generated",
        agent="general_agent",
        requirement_section="req3-processing",
        topic=state.result.get("topic"),
    )
    return state


__all__ = ["build_workflow"]


def _handle_agent_error(state: GraphState, error: AgentError, *, agent_name: str) -> GraphState:
    state.log(
        "Agent error encountered",
        level="warning",
        agent=agent_name,
        requirement_section="req5-error",
        error_type=error.error_type.value,
        error_message=error.message,
    )
    clarification_needed = error.error_type in {
        ErrorType.UNKNOWN_PROPERTY,
        ErrorType.MISSING_PROPERTY,
        ErrorType.MISSING_TIMEFRAME,
    }
    if clarification_needed:
        reasons = list(state.context.clarification_reasons or [])
        reason = _clarification_reason_for_error(state.context.request_type, error.error_type)
        if reason and reason not in reasons:
            reasons.append(reason)
        state.context.clarification_reasons = reasons
        item = clarify_agent.run(
            state.context,
            request_type=state.context.request_type,
            reasons=reasons,
            suggestions=state.context.suggested_addresses,
        )
        _apply_clarification_item(state, item)
        state.log(
            "Clarification prompt issued after agent error",
            agent="clarification_agent",
            requirement_section="req5-error",
            reasons=reasons,
            field=item.field,
        )
        return state
    payload = error.to_payload()
    state.result = payload
    return state


def _clarification_reason_for_error(request_type: RequestType | None, error_type: ErrorType) -> str:
    if request_type == RequestType.PNL and error_type == ErrorType.MISSING_TIMEFRAME:
        return "period"
    if request_type == RequestType.PRICE_COMPARISON:
        return "second_property"
    if request_type == RequestType.ASSET_DETAILS:
        return "property"
    if request_type == RequestType.PNL:
        return "property"
    return "details"


def _apply_clarification_item(state: GraphState, item: ClarificationItem) -> None:
    state.context.clarifications = [item]
    state.context.awaiting_user_reply = True
    state.context.needs_clarification = True
    state.context.clarification_needed = True
    reasons = list(state.context.clarification_reasons or [])
    if item.field not in reasons:
        reasons.append(item.field)
    state.context.clarification_reasons = reasons
    state.result = {
        "message": item.question,
        "clarification": _serialize_clarification_item(item),
    }


def _serialize_clarification_item(item: ClarificationItem) -> Dict[str, Any]:
    return {
        "field": item.field,
        "question": item.question,
        "kind": item.kind,
        "options": list(item.options),
        "value": item.value,
    }

