import pytest


@pytest.mark.asyncio
async def test_analysis_node_executes_generated_sql(
    mock_service,
    monkeypatch,
):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from graph.nodes import create_nodes

    sql = "SELECT COUNT(*) FROM public.documents LIMIT 11"

    monkeypatch.setattr(
        "graph.nodes.validate_readonly_sql",
        lambda *args, **kwargs: SimpleNamespace(sql=sql),
    )

    query_result = SimpleNamespace(
        columns=["count"],
        rows=[{"count": 99}],
        row_count=1,
        truncated=False,
        referenced_tables=["public.documents"],
    )

    execute_mock = AsyncMock(return_value=query_result)

    monkeypatch.setattr(
        "graph.nodes.execute_readonly_query",
        execute_mock,
    )

    nodes = create_nodes(mock_service)

    result = await nodes["analysis"](
        {
            "question": "How many documents?",
            "schema_name": "public",
            "allowed_tables": {"documents"},
            "max_rows": 10,
            "sql_draft": {
                "explanation": "Count documents.",
                "sql": sql,
                "parameters": {},
            },
            "analysis_plan": {
                "objective": "Count documents.",
                "metrics": ["document count"],
                "dimensions": [],
                "required_tables": ["documents"],
                "steps": ["Count document records."],
                "expected_result": "A document count.",
            },
        }
    )

    assert result["query_result"]["rows"] == [
        {"count": 99}
    ]
    assert result["sql"] == sql
    assert result["next_step"] == "result_analyst"

    execute_mock.assert_awaited_once()
    mock_service.analyze.assert_not_awaited()