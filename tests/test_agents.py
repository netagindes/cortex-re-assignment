import pytest

from app.agents import ClarificationAgent, PnLAgent, SupervisorAgent
from app.data_layer import load_assets


def test_supervisor_classification():
    agent = SupervisorAgent()
    assert agent.classify("Compare price") == "price_comparison"


def test_supervisor_resolves_price_addresses():
    agent = SupervisorAgent()
    df = load_assets()
    addresses = df["address"].dropna().unique().tolist()
    if len(addresses) < 2:
        pytest.skip("Dataset does not contain two addresses")
    query = f"Compare {addresses[0]} versus {addresses[1]}"
    decision = agent.analyze(query)
    assert decision.request_type == "price_comparison"
    assert len(decision.addresses) == 2
    assert set(decision.addresses) == {addresses[0], addresses[1]}
    assert not decision.needs_clarification


def test_supervisor_suggests_when_missing_matches():
    agent = SupervisorAgent()
    decision = agent.analyze("Compare the asset at 123 Main St to one on 456 Oak Ave")
    assert decision.request_type == "price_comparison"
    assert decision.needs_clarification
    assert decision.suggested_addresses


def test_supervisor_pnl_leaves_property_blank():
    agent = SupervisorAgent()
    decision = agent.analyze("What is the total P&L for 2025?")
    assert decision.request_type == "pnl"
    assert decision.property_name is None


def test_supervisor_marks_missing_buildings():
    agent = SupervisorAgent()
    decision = agent.analyze("Compare Building 999 with Building 998")
    assert decision.request_type == "price_comparison"
    assert decision.missing_addresses
    assert any("Building 999" in note for note in decision.notes)


def test_clarification_message_includes_suggestions():
    agent = ClarificationAgent()
    response = agent.run(
        "Need comparison help",
        request_type="price_comparison",
        reasons=["second_property"],
        suggestions=["Building 120", "Building 160"],
    )
    assert "Building 120" in response["message"]
    assert "Building 160" in response["message"]


def test_pnl_agent_year_totals():
    agent = PnLAgent()
    result = agent.run({"year": 2025})
    assert result["status"] == "ok"
    summary = result["totals_summary"]
    assert summary["total_revenue"] == pytest.approx(592124.15, rel=0, abs=0.01)
    assert summary["total_expenses"] == pytest.approx(2599.29, rel=0, abs=0.01)
    assert summary["net_operating_income"] == pytest.approx(589524.86, rel=0, abs=0.01)

