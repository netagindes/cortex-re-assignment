import pytest

from app.agents import (
    ClarificationAgent,
    GeneralKnowledgeAgent,
    PnLAgent,
    RequestType,
    SupervisorAgent,
)
from app.data_layer import load_assets


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


def test_supervisor_requires_granularity_for_yearly_property():
    agent = SupervisorAgent()
    decision = agent.analyze("Show me the yearly P&L for Building 180 for 2025.")
    assert decision.request_type == RequestType.PNL
    assert "granularity" in decision.missing_requirements
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
    assert "Building 180" in decision.addresses


def test_clarification_message_includes_suggestions():
    agent = ClarificationAgent()
    response = agent.run(
        "Need comparison help",
        request_type=RequestType.PRICE_COMPARISON,
        reasons=["second_property"],
        suggestions=["Building 120", "Building 160"],
    )
    assert "Building 120" in response["message"]
    assert "Building 160" in response["message"]


def test_clarification_pnl_granularity_prompt():
    agent = ClarificationAgent()
    response = agent.run(
        "Need more info",
        request_type=RequestType.PNL,
        reasons=["granularity"],
    )
    assert "tenant-level, property-level, or combined totals" in response["message"]


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


def test_general_agent_pnl_overview():
    agent = GeneralKnowledgeAgent()
    response = agent.run("What is P&L and how do you calculate it?")
    assert response["topic"] == "pnl_overview"
    assert "Profit & Loss" in response["message"]

