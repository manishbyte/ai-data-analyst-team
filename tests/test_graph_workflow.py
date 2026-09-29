
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import HumanMessage

from graph.nodes import create_nodes
from graph.workflow import build_analyst_graph


@pytest.fixture
def mock_service():
    return SimpleNamespace(
        analyze=AsyncMock(),
        engine=object(),
    )


@pytest.fixture
def mock_schema_inspection(monkeypatch):
    schema = {
        "schema": "public",
        "table_count": 1,
        "tables": [
            {
                "table_name": "documents",
                "columns": [
                    {"name": "id", "type": "integer"},
                    {"name": "title", "type": "text"},
                ],
            }
        ],
    }

    monkeypatch.setattr(
        "graph.nodes.inspect_schema",
        AsyncMock(return_value=schema),
    )

    return schema


@pytest.fixture
def mock_planning_and_execution(monkeypatch):
    """Mock the planner, SQL generator, validator, and executor."""

    plan_data = {
        "objective": "Count documents in the database.",
        "metrics": ["document count"],
        "dimensions": [],
        "required_tables": ["documents"],
        "steps": [
            "Read the documents table.",
            "Count the rows.",
        ],
        "expected_result": (
            "A single row containing the document count."
        ),
    }

    draft_data = {
        "explanation": "Count all permitted documents.",
        "sql": (
            "SELECT COUNT(*) AS document_count "
            "FROM public.documents"
        ),
        "parameters": {},
    }

    class FakePlanner:
        def __init__(self, *args, **kwargs):
            pass

        async def create_plan(self, *, question, schema_context):
            return SimpleNamespace(
                **plan_data,
                model_dump=lambda: dict(plan_data),
            )

    class FakeSQLGenerator:
        def __init__(self, *args, **kwargs):
            pass

        async def generate_sql(
            self,
            *,
            question,
            schema_context,
            plan,
            revision_feedback=None,
            previous_sql=None,
        ):
            return SimpleNamespace(
                **draft_data,
                model_dump=lambda: dict(draft_data),
            )

    monkeypatch.setattr(
        "graph.nodes.AnalysisPlanner",
        FakePlanner,
    )
    monkeypatch.setattr(
        "graph.nodes.SQLGenerator",
        FakeSQLGenerator,
    )
    monkeypatch.setattr(
        "graph.nodes.validate_readonly_sql",
        lambda sql, **kwargs: SimpleNamespace(sql=sql),
    )

    query_result = SimpleNamespace(
        columns=["document_count"],
        rows=[{"document_count": 99}],
        row_count=1,
        truncated=False,
        referenced_tables=["documents"],
    )

    execute_mock = AsyncMock(return_value=query_result)

    monkeypatch.setattr(
        "graph.nodes.execute_readonly_query",
        execute_mock,
    )

    return execute_mock


def make_state(**overrides):
    """Create the minimum state required by individual graph nodes."""

    state = {
        "question": "Count documents",
        "schema_context": (
            '{"tables": [{"table_name": "documents"}]}'
        ),
        "analysis_plan": {
            "objective": "Count documents",
            "required_tables": ["documents"],
        },
        "allowed_tables": {"documents"},
        "schema_name": "public",
        "max_rows": 10,
        "sql_draft": {
            "explanation": "Count documents",
            "sql": (
                "SELECT COUNT(*) AS document_count "
                "FROM public.documents"
            ),
            "parameters": {},
        },
        "sql": (
            "SELECT COUNT(*) AS document_count "
            "FROM public.documents"
        ),
        "parameters": {},
        "query_result": {
            "columns": ["document_count"],
            "rows": [{"document_count": 99}],
            "row_count": 1,
            "truncated": False,
            "referenced_tables": ["documents"],
        },
        "result_analysis": (
            '{"summary": "99 documents", '
            '"key_findings": ["99 documents"], '
            '"caveats": [], '
            '"answer_sufficient": true}'
        ),
        "critic_review": {},
        "critic_feedback": "",
        "revision_count": 0,
        "max_revisions": 2,
        "revision_reason": "",
        "status": "running",
    }

    state.update(overrides)
    return state


def make_review(
    *,
    passed,
    needs_revision,
    feedback="Review feedback",
    issues=None,
):
    """Build a mock CriticReview-like object."""

    data = {
        "passed": passed,
        "needs_revision": needs_revision,
        "feedback": feedback,
        "issues": issues or [],
    }

    return SimpleNamespace(
        **data,
        model_dump=lambda: dict(data),
    )


@pytest.mark.asyncio
async def test_graph_completes_successful_request(
    mock_service,
    mock_schema_inspection,
    mock_planning_and_execution,
):
    graph = build_analyst_graph(mock_service)

    result = await graph.ainvoke(
        {
            "messages": [
                HumanMessage(content="Count documents")
            ],
            "allowed_tables": {"documents"},
            "schema_name": "public",
            "max_rows": 10,
            "status": "pending",
        }
    )

    assert result["status"] == "completed"
    assert result["next_step"] == "end"
    assert result["query_result"]["rows"] == [
        {"document_count": 99}
    ]
    assert result["query_result"]["row_count"] == 1
    assert result["final_answer"]
    assert result["sql"] == (
        "SELECT COUNT(*) AS document_count "
        "FROM public.documents"
    )

    mock_planning_and_execution.assert_awaited_once()
    mock_service.analyze.assert_not_awaited()


@pytest.mark.asyncio
async def test_graph_stops_when_analysis_fails(
    mock_service,
    mock_schema_inspection,
    mock_planning_and_execution,
):
    mock_planning_and_execution.side_effect = RuntimeError(
        "Simulated SQL execution failure"
    )

    graph = build_analyst_graph(mock_service)

    result = await graph.ainvoke(
        {
            "messages": [
                HumanMessage(content="Count documents")
            ],
            "allowed_tables": {"documents"},
            "schema_name": "public",
            "max_rows": 10,
            "status": "pending",
        }
    )

    assert result["status"] == "failed"
    assert result["next_step"] == "end"
    assert result["error"] == (
        "SQL validation or execution failed."
    )
    assert "final_answer" not in result

    mock_planning_and_execution.assert_awaited_once()
    mock_service.analyze.assert_not_awaited()


@pytest.mark.asyncio
async def test_critic_approval_routes_to_final_response(monkeypatch):
    class FakeCritic:
        def __init__(self, *args, **kwargs):
            pass

        async def review(self, **kwargs):
            return make_review(
                passed=True,
                needs_revision=False,
                feedback="Analysis approved",
            )

    monkeypatch.setattr(
        "graph.nodes.CriticAgent",
        FakeCritic,
    )

    nodes = create_nodes(SimpleNamespace(engine=object()))
    result = await nodes["critic"](make_state())

    assert result["next_step"] == "final_response"
    assert result["critic_review"]["passed"] is True
    assert result["critic_review"]["needs_revision"] is False


@pytest.mark.asyncio
async def test_critic_requests_revision(monkeypatch):
    class FakeCritic:
        def __init__(self, *args, **kwargs):
            pass

        async def review(self, **kwargs):
            return make_review(
                passed=False,
                needs_revision=True,
                feedback="Include document IDs in the results.",
                issues=["Document IDs are missing"],
            )

    monkeypatch.setattr(
        "graph.nodes.CriticAgent",
        FakeCritic,
    )

    nodes = create_nodes(SimpleNamespace(engine=object()))
    result = await nodes["critic"](make_state())

    assert result["next_step"] == "sql_generator"
    assert result["revision_count"] == 1
    assert "document IDs" in result["critic_feedback"]


@pytest.mark.asyncio
async def test_critic_stops_at_revision_limit(monkeypatch):
    class FakeCritic:
        def __init__(self, *args, **kwargs):
            pass

        async def review(self, **kwargs):
            return make_review(
                passed=False,
                needs_revision=True,
                feedback="The analysis remains incomplete.",
                issues=["Missing required information"],
            )

    monkeypatch.setattr(
        "graph.nodes.CriticAgent",
        FakeCritic,
    )

    nodes = create_nodes(SimpleNamespace(engine=object()))

    result = await nodes["critic"](
        make_state(
            revision_count=2,
            max_revisions=2,
        )
    )

    assert result["next_step"] == "final_response"
    assert result["critic_review"]["passed"] is False
    assert result["critic_feedback"] == (
        "The analysis remains incomplete."
    )


@pytest.mark.asyncio
async def test_sql_generator_receives_critic_feedback(monkeypatch):
    calls = []

    draft_data = {
        "explanation": "Count documents",
        "sql": (
            "SELECT id, COUNT(*) AS document_count "
            "FROM public.documents GROUP BY id"
        ),
        "parameters": {},
    }

    class FakeSQLGenerator:
        def __init__(self, *args, **kwargs):
            pass

        async def generate_sql(self, **kwargs):
            calls.append(kwargs)

            return SimpleNamespace(
                **draft_data,
                model_dump=lambda: dict(draft_data),
            )

    monkeypatch.setattr(
        "graph.nodes.SQLGenerator",
        FakeSQLGenerator,
    )

    nodes = create_nodes(SimpleNamespace(engine=object()))

    state = make_state(
        revision_count=1,
        critic_feedback="Include document IDs.",
    )

    result = await nodes["sql_generator"](state)

    assert result["next_step"] == "analysis"
    assert len(calls) == 1
    assert calls[0]["revision_feedback"] == (
        "Include document IDs."
    )
    assert calls[0]["previous_sql"] == state["sql"]
    assert result["sql"] == draft_data["sql"]


@pytest.mark.asyncio
async def test_unsafe_revised_sql_is_not_executed(monkeypatch):
    """Rejected SQL must never reach the database executor."""

    def reject_unsafe_sql(sql, **kwargs):
        raise ValueError("Unsafe SQL rejected")

    monkeypatch.setattr(
        "graph.nodes.validate_readonly_sql",
        reject_unsafe_sql,
    )

    execute_mock = AsyncMock()

    monkeypatch.setattr(
        "graph.nodes.execute_readonly_query",
        execute_mock,
    )

    nodes = create_nodes(SimpleNamespace(engine=object()))

    state = make_state(
        revision_count=1,
        sql_draft={
            "sql": "DELETE FROM public.documents",
            "parameters": {},
        },
    )

    result = await nodes["analysis"](state)

    assert result["status"] == "failed"
    assert result["next_step"] == "end"
    assert result["error"] == (
        "SQL validation or execution failed."
    )

    execute_mock.assert_not_awaited()