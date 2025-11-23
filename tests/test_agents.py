import pytest

from app.agents import ClarificationAgent, GeneralKnowledgeAgent, PnLAgent, RequestType, SupervisorAgent
from app.data_layer import load_assets
from app.graph.state import ClarificationItem, QueryContext


def test_supervisor_classification():
    agent = SupervisorAgent()
    assert agent.classify("Compare price") == RequestType.PRICE_COMPARISON


def test_supervisor_resolves_price_addresses():
    agent = SupervisorAgent()
    df = load_assets()
    addresses = df["address"].dropna().unique().tolist()
    if len(addresses) < 2:
        pytest.skip("Dataset does not contain two addresses")
    query = f"Compare {addresses[0]} versus {addresses[1]}"
    decision = agent.analyze(query)
    assert decision.request_type == RequestType.PRICE_COMPARISON
    assert len(decision.addresses) == 2
    assert set(decision.addresses) == {addresses[0], addresses[1]}
    assert not decision.needs_clarification


def test_supervisor_suggests_when_missing_matches():
    agent = SupervisorAgent()
    decision = agent.analyze("Compare the asset at 123 Main St to one on 456 Oak Ave")
    assert decision.request_type == RequestType.PRICE_COMPARISON
    assert decision.needs_clarification
    assert decision.suggested_addresses


def test_supervisor_pnl_leaves_property_blank():
    agent = SupervisorAgent()
    decision = agent.analyze("What is the total P&L for 2025?")
    assert decision.request_type == RequestType.PNL
    assert decision.property_name is None


def test_supervisor_requires_aggregation_level_for_yearly_property():
    agent = SupervisorAgent()
    decision = agent.analyze("Show me the yearly P&L for Building 180 for 2025.")
    assert decision.request_type == RequestType.PNL
    assert "aggregation_level" in decision.missing_requirements
    assert decision.needs_clarification


def test_supervisor_marks_missing_buildings():
    agent = SupervisorAgent()
    decision = agent.analyze("Compare Building 999 with Building 998")
    assert decision.request_type == RequestType.PRICE_COMPARISON
    assert decision.missing_addresses
    assert any("Building 999" in note for note in decision.notes)


def test_supervisor_routes_timebound_pnl_comparisons():
    agent = SupervisorAgent()
    decision = agent.analyze("Compare January and February 2025 P&L for Building 180.")
    assert decision.request_type == RequestType.PNL
    assert not decision.needs_clarification
    assert decision.addresses
    assert decision.property_name == "Building 180"
    assert len(decision.comparison_periods) == 2
    labels = [period["label"] for period in decision.comparison_periods]
    assert labels == ["2025-M01", "2025-M02"]


def test_supervisor_requires_two_periods_for_comparison():
    agent = SupervisorAgent()
    decision = agent.analyze("Compare January 2025 P&L for Building 180.")
    assert decision.request_type == RequestType.PNL
    assert decision.needs_clarification
    assert "comparison_periods" in decision.missing_requirements


def test_supervisor_requires_single_property_for_pnl_comparison():
    agent = SupervisorAgent()
    decision = agent.analyze("Compare January and February 2025 P&L for Building 180 and Building 140.")
    assert decision.request_type == RequestType.PNL
    assert decision.needs_clarification
    assert "property_selection" in decision.missing_requirements


def test_clarification_message_includes_suggestions():
    agent = ClarificationAgent()
    context = QueryContext(user_input="Need comparison help")
    item = agent.run(
        context,
        request_type=RequestType.PRICE_COMPARISON,
        reasons=["second_property"],
        suggestions=["Building 120", "Building 160"],
    )
    assert "Building 120" in item.question
    assert "Building 160" in item.question


def test_clarification_pnl_aggregation_prompt():
    agent = ClarificationAgent()
    context = QueryContext(user_input="Need more info")
    item = agent.run(
        context,
        request_type=RequestType.PNL,
        reasons=["aggregation_level"],
    )
    assert "tenant-level, property-level, or combined totals" in item.question


def test_supervisor_follow_up_sets_aggregation_level():
    agent = SupervisorAgent()
    context = QueryContext(
        user_input="property",
        request_type=RequestType.PNL,
        property_name="Building 180",
        period="2025",
        period_level="year",
        clarifications=[
            ClarificationItem(
                field="aggregation_level",
                question="Do you want tenant-level, property-level, or combined totals?",
                kind="choice",
                options=["tenant", "property", "combined"],
            )
        ],
        awaiting_user_reply=True,
        needs_clarification=True,
        clarification_reasons=["aggregation_level"],
    )
    decision = agent.analyze(context)
    assert decision.aggregation_level == "property"
    assert not decision.needs_clarification
    assert context.aggregation_level == "property"
    assert not context.awaiting_user_reply


def test_supervisor_follow_up_period_granularity_requests_specific_value():
    agent = SupervisorAgent()
    context = QueryContext(
        user_input="year",
        request_type=RequestType.PNL,
        property_name="Building 180",
        clarifications=[
            ClarificationItem(
                field="period",
                question="Which period should I use? Month, quarter, or year?",
                kind="granularity",
                options=["month", "quarter", "year"],
            )
        ],
        awaiting_user_reply=True,
        needs_clarification=True,
        clarification_reasons=["period"],
    )
    decision = agent.analyze(context)
    assert context.period_level == "year"
    assert decision.needs_clarification
    assert "period" in decision.missing_requirements


def test_pnl_agent_year_totals():
    agent = PnLAgent()
    result = agent.run({"year": 2025})
    assert result["status"] == "ok"
    summary = result["totals_summary"]
    assert summary["total_revenue"] == pytest.approx(592124.15, rel=0, abs=0.01)
    assert summary["total_expenses"] == pytest.approx(2599.29, rel=0, abs=0.01)
    assert summary["net_operating_income"] == pytest.approx(589524.86, rel=0, abs=0.01)


def test_pnl_agent_no_data_message_is_friendly():
    agent = PnLAgent()
    result = agent.run({"property_name": "Building 999", "year": 2025})
    assert result["status"] == "no_data"
    assert "couldn't find any financial data" in result["message"].lower()
    assert "tenant-level, property-level, or combined totals" in result["message"]


def test_pnl_agent_comparison_mode_handles_two_periods():
    agent = PnLAgent()
    task = {
        "property_name": "Building 17",
        "comparison_periods": [
            {"label": "2025-M01", "level": "month", "year": 2025, "month": "2025-M01"},
            {"label": "2025-M02", "level": "month", "year": 2025, "month": "2025-M02"},
        ],
    }
    result = agent.run(task)
    assert result["status"] == "ok"
    assert result["mode"] == "comparison"
    comparison = result["comparison"]
    assert len(comparison["periods"]) == 2
    delta = comparison["delta"]
    assert delta["net_operating_income"] == pytest.approx(0.0, abs=0.01)


def test_pnl_agent_comparison_requires_two_periods():
    agent = PnLAgent()
    result = agent.run(
        {
            "property_name": "Building 17",
            "comparison_periods": [
                {"label": "2025-M01", "level": "month", "year": 2025, "month": "2025-M01"},
            ],
        }
    )
    assert result["status"] == "error"
    assert "comparison_period_count_invalid" in result.get("errors", [])


def test_general_agent_handles_ledger_questions():
    agent = GeneralKnowledgeAgent()
    response = agent.run("Explain ledger code 120.")
    assert response["topic"] == "ledger_explanation"
    assert "ledger" in response["message"].lower()

