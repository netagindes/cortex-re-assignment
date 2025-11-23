from app.agents.request_types import RequestType
from app.api.main import _format_response
from app.graph.state import GraphState, QueryContext
from app.tools import format_currency


def _state(result, request_type=RequestType.GENERAL) -> GraphState:
    context = QueryContext(user_input="hello", request_type=request_type)
    return GraphState(context=context, result=result)


def test_format_response_returns_agent_error_message():
    error_payload = {
        "message": "Data is temporarily unavailable.",
        "error_type": "data_unavailable",
        "details": "Ledger service offline",
    }
    text, note = _format_response(_state(error_payload, RequestType.PNL))
    assert text == "Data is temporarily unavailable."
    assert note == "Ledger service offline"


def test_format_response_prioritizes_pnl_layout_over_embedded_message():
    formatted_value = format_currency(10.0)
    pnl_payload = {
        "label": "Total P&L (2025)",
        "formatted": formatted_value,
        "totals_summary": {"total_revenue": 100.0, "total_expenses": 90.0},
        "message": "Should not short circuit specialized formatting.",
    }
    text, _ = _format_response(_state(pnl_payload, RequestType.PNL))
    assert "Total P&L (2025)" in text
    assert formatted_value in text

