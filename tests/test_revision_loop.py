
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from graph.nodes import create_nodes


def make_state(**overrides):
    """Create a default graph state for revision-loop tests."""
    state = {
        "question": "Count documents",
        "schema_context": '{"tables": [{"table_name": "documents"}]}',
        "analysis_plan": {
            "objective": "Count documents",
            "required_tables": ["documents"],
        },
        "allowed_tables": {"documents"},
        "max_rows": 10,
        "sql_draft": {
            "sql": "SELECT COUNT(*) FROM public.documents",
            "parameters": {},
        },
        "sql": "SELECT COUNT(*) FROM public.documents",
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
    """Create a fake critic review."""
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
async def test_critic_approval_routes_to_final_response(monkeypatch):
    """An approved analysis should proceed to the final response."""

    class FakeCritic:
        def __init__(self, *args, **kwargs):
            pass

        async def review(self, **kwargs):
            return make_review(
                passed=True,
                needs_revision=False,
                feedback="Analysis approved",
            )

    monkeypatch.setattr("graph.nodes.CriticAgent", FakeCritic)

    nodes = create_nodes(SimpleNamespace(engine=object()))
    result = await nodes["critic"](make_state())

    assert result["next_step"] == "final_response"
    assert result["critic_review"]["passed"] is True
    assert result["critic_review"]["needs_revision"] is False


@pytest.mark.asyncio
async def test_critic_requests_revision(monkeypatch):
    """A failed review should route back to SQL generation."""

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

    monkeypatch.setattr("graph.nodes.CriticAgent", FakeCritic)

    nodes = create_nodes(SimpleNamespace(engine=object()))
    result = await nodes["critic"](make_state())

    assert result["next_step"] == "sql_generator"
    assert result["revision_count"] == 1
    assert "document IDs" in result["critic_feedback"]
    assert result["critic_review"]["passed"] is False


@pytest.mark.asyncio
async def test_critic_stops_at_revision_limit(monkeypatch):
    """The graph should stop revising at the configured limit."""

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

    monkeypatch.setattr("graph.nodes.CriticAgent", FakeCritic)

    nodes = create_nodes(SimpleNamespace(engine=object()))

    result = await nodes["critic"](
        make_state(
            revision_count=2,
            max_revisions=2,
        )
    )

    assert result["next_step"] == "final_response"
    assert result.get("revision_count", 2) == 2
    assert result["critic_review"]["passed"] is False
    assert result["critic_review"]["needs_revision"] is True


@pytest.mark.asyncio
async def test_sql_generator_receives_critic_feedback(monkeypatch):
    """The SQL generator should receive feedback and previous SQL."""

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
    assert calls[0]["revision_feedback"] == "Include document IDs."
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

    service = SimpleNamespace(engine=object())
    nodes = create_nodes(service)

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

    execute_mock.assert_not_awaited()