import pytest

from ragx.api import public_api
from ragx.control.planner import Planner
from ragx.models import QueryResult


@pytest.fixture(autouse=True)
def reset_state():
    public_api._reset_state_for_tests()
    yield
    public_api._reset_state_for_tests()


def test_planner_includes_graph_step():
    planner = Planner()
    plan = planner.plan("Compare solar and wind adoption globally")
    assert any(step.name == "graph_retrieve" for step in plan)


def test_agentic_pipeline_triggers_critic_retry():
    result = public_api.answer_agentic("Compare fusion and fission energy outputs")
    assert isinstance(result, QueryResult)
    assert result.meta.get("critic_retries") == "1"
    assert result.meta.get("critic_status") == "LOW_CONFIDENCE"
