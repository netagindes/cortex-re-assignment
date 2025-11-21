from app.agents import SupervisorAgent


def test_supervisor_classification():
    agent = SupervisorAgent()
    assert agent.classify("Compare price") == "price_comparison"

