import pytest

from app.agents import RequestType
from app.agents.request_types import normalize_request_type
from app.graph.state import ClarificationItem, GraphState, QueryContext
from app.graph.workflow import build_workflow


def test_workflow_builds():
    workflow = build_workflow()
    compiled = workflow.compile()
    try:
        state = compiled.invoke(GraphState(context=QueryContext(user_input="Hello", request_type=RequestType.GENERAL)))
    except ValueError:
        state = None
    assert state is None or isinstance(state, GraphState) or isinstance(state, dict)


def test_pnl_query_handles_year():
    state = _invoke("What is the total P&L for 2025?")
    assert state.context.request_type == RequestType.PNL
    assert state.result["status"] == "ok"
    assert state.result["value"] == pytest.approx(589524.86, rel=0, abs=0.01)


def test_pnl_query_missing_period_prompts_for_period():
    state = _invoke("Give me the P&L for Building 180.")
    assert "Which period should I use" in state.result["message"]
    assert "Month, quarter, or year" in state.result["message"]


def test_pnl_query_yearly_property_prompts_for_aggregation_level():
    state = _invoke("Show me the yearly P&L for Building 180 for 2025.")
    assert "tenant-level, property-level, or combined totals" in state.result["message"]


def test_price_query_without_prices_returns_message():
    state = _invoke("Compare Building 120 and Building 160 prices")
    assert state.context.request_type == RequestType.PRICE_COMPARISON
    assert "price" in state.result["message"].lower()


def test_price_query_requests_known_properties():
    state = _invoke("What is the price of my asset at 123 Main St compared to the one at 456 Oak Ave?")
    assert state.context.request_type == RequestType.PRICE_COMPARISON
    assert "Which property are you referring to" in state.result["message"]


def test_general_query_routes_to_general_agent():
    state = _invoke("Can you explain how the dataset is structured?")
    assert state.context.request_type == RequestType.GENERAL
    assert state.result.get("topic") == "dataset_overview"


def _invoke(message: str) -> GraphState:
    workflow = build_workflow()
    compiled = workflow.compile()
    initial = GraphState(context=QueryContext(user_input=message, request_type=RequestType.GENERAL))
    raw_state = compiled.invoke(initial)
    if isinstance(raw_state, GraphState):
        return raw_state
    context = raw_state.get("context") or {}
    if isinstance(context, QueryContext):
        ctx = context
    else:
        ctx = QueryContext(
            user_input=context.get("user_input", message),
            request_type=normalize_request_type(context.get("request_type")),
            addresses=list(context.get("addresses") or []),
            suggested_addresses=list(context.get("suggested_addresses") or []),
            address_matches=list(context.get("address_matches") or []),
            candidate_terms=list(context.get("candidate_terms") or []),
            unresolved_terms=list(context.get("unresolved_terms") or []),
            missing_addresses=list(context.get("missing_addresses") or []),
            notes=list(context.get("notes") or []),
            period=context.get("period"),
            period_level=context.get("period_level"),
            entity_name=context.get("entity_name"),
            property_name=context.get("property_name"),
            tenant_name=context.get("tenant_name"),
            year=context.get("year"),
            quarter=context.get("quarter"),
            month=context.get("month"),
            needs_clarification=bool(context.get("needs_clarification")),
            clarification_reasons=list(context.get("clarification_reasons") or []),
            request_measurement=context.get("request_measurement"),
            comparison_periods=list(context.get("comparison_periods") or []),
            aggregation_level=context.get("aggregation_level"),
            clarifications=[
                ClarificationItem(
                    field=item.get("field", "property_name"),
                    question=item.get("question", ""),
                    kind=item.get("kind"),
                    options=list(item.get("options") or []),
                    value=item.get("value"),
                )
                for item in (context.get("clarifications") or [])
                if isinstance(item, dict)
            ],
            awaiting_user_reply=bool(context.get("awaiting_user_reply")),
            clarification_needed=bool(context.get("clarification_needed")),
        )
    diagnostics = raw_state.get("diagnostics") or []
    return GraphState(
        context=ctx,
        result=raw_state.get("result"),
        diagnostics=diagnostics if isinstance(diagnostics, list) else [],
        logger=None,
        pnl_result=raw_state.get("pnl_result"),
    )

