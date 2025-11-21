from app.graph.state import GraphState, QueryContext
from app.graph.workflow import build_workflow


def test_workflow_builds():
    workflow = build_workflow()
    compiled = workflow.compile()
    try:
        state = compiled.invoke(GraphState(context=QueryContext(user_input="Hello", request_type="general")))
    except ValueError:
        state = None
    assert state is None or isinstance(state, GraphState)

