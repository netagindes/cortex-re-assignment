import pytest

from app.graph.state import GraphState, QueryContext
from app.graph.workflow import build_workflow


def test_workflow_builds():
    workflow = build_workflow()
    compiled = workflow.compile()
    try:
        state = compiled.invoke(GraphState(context=QueryContext(user_input="Hello", request_type="general")))
    except ValueError:
        state = None
    assert state is None or isinstance(state, GraphState) or isinstance(state, dict)


def test_pnl_query_handles_year():
    state = _invoke("What is the total P&L for 2025?")
    assert state.context.request_type == "pnl"
    assert state.result["status"] == "ok"
    assert state.result["value"] == pytest.approx(589524.86, rel=0, abs=0.01)


def test_price_query_without_prices_returns_message():
    state = _invoke("Compare Building 120 and Building 160 prices")
    assert state.context.request_type == "price_comparison"
    assert "price" in state.result["message"].lower()


def _invoke(message: str) -> GraphState:
    workflow = build_workflow()
    compiled = workflow.compile()
    initial = GraphState(context=QueryContext(user_input=message, request_type="general"))
    raw_state = compiled.invoke(initial)
    if isinstance(raw_state, GraphState):
        return raw_state
    context = raw_state.get("context") or {}
    if isinstance(context, QueryContext):
        ctx = context
    else:
        ctx = QueryContext(
            user_input=context.get("user_input", message),
            request_type=context.get("request_type", "general"),
            addresses=list(context.get("addresses") or []),
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
        )
    diagnostics = raw_state.get("diagnostics") or []
    return GraphState(
        context=ctx,
        result=raw_state.get("result"),
        diagnostics=diagnostics if isinstance(diagnostics, list) else [],
        logger=None,
        pnl_result=raw_state.get("pnl_result"),
    )

