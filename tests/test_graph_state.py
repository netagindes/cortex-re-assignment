from app.agents.request_types import RequestType
from app.graph.state import GraphState, QueryContext
from app.logging_utils import PipelineLogger


def test_graph_state_log_promotes_agent_and_requirement_fields():
    context = QueryContext(user_input="Route this", request_type=RequestType.PNL)
    logger = PipelineLogger("tests.graph_state", context={"pipeline": "unit"})
    state = GraphState(context=context, logger=logger)

    state.log(
        "Supervisor routed request",
        agent="supervisor",
        requirement_section="req1-routing",
        correlation_id="abc123",
    )

    assert state.diagnostics, "Log entry should be captured in diagnostics"
    entry = state.diagnostics[-1]
    assert entry.agent == "supervisor"
    assert entry.requirement_section == "req1-routing"
    assert entry.metadata is not None
    assert entry.metadata.get("correlation_id") == "abc123"
    assert entry.metadata.get("request_type") == RequestType.PNL.value
    assert "agent" not in entry.metadata
    assert "requirement_section" not in entry.metadata

    logger_entries = logger.as_entries()
    assert logger_entries, "PipelineLogger should also capture entries"
    logger_entry = logger_entries[-1]
    assert logger_entry.agent == "supervisor"
    assert logger_entry.requirement_section == "req1-routing"

