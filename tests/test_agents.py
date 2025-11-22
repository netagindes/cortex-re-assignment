import pytest

from app.agents import PnLAgent, SupervisorAgent


def test_supervisor_classification():
    agent = SupervisorAgent()
    assert agent.classify("Compare price") == "price_comparison"


def test_pnl_agent_year_totals():
    agent = PnLAgent()
    result = agent.run({"year": 2025})
    assert result["status"] == "ok"
    summary = result["totals_summary"]
    assert summary["total_revenue"] == pytest.approx(592124.15, rel=0, abs=0.01)
    assert summary["total_expenses"] == pytest.approx(2599.29, rel=0, abs=0.01)
    assert summary["net_operating_income"] == pytest.approx(589524.86, rel=0, abs=0.01)

