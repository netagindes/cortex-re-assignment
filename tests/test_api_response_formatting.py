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


def test_format_response_handles_comparison_results():
    comparison_payload = {
        "mode": "comparison",
        "comparison": {
            "periods": [
                {"label": "2025-M01", "formatted": format_currency(100.0), "totals_summary": {"total_revenue": 100.0, "total_expenses": -20.0}},
                {"label": "2025-M02", "formatted": format_currency(120.0), "totals_summary": {"total_revenue": 120.0, "total_expenses": -30.0}},
            ],
            "delta": {
                "total_revenue": 20.0,
                "total_expenses": -10.0,
                "net_operating_income": 10.0,
                "noi_percent_change": 10.0,
            },
        },
        "message": "Comparison summary.",
    }
    text, _ = _format_response(_state(comparison_payload, RequestType.PNL))
    assert "2025-M01" in text
    assert "2025-M02" in text
    assert "Difference" in text

