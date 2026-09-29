
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.analyst import AnalysisPlan
from services.analyst_service import AnalystService
from sql.validator import SQLValidationError


@pytest.fixture
def service():
    analyst = MagicMock()
    analyst.create_plan = AsyncMock()

    with patch(
        "services.analyst_service.create_database_engine"
    ) as create_engine:
        mock_engine = MagicMock()
        mock_engine.dispose = AsyncMock()
        create_engine.return_value = mock_engine

        instance = AnalystService(
            database_url=(
                "postgresql+psycopg://test:test@localhost:5000/test"
            ),
            analyst=analyst,
        )

    return instance, analyst


@pytest.mark.asyncio
async def test_analyze_executes_validated_plan(service):
    instance, analyst = service

    analyst.create_plan.return_value = AnalysisPlan(
        explanation="Count the sales records.",
        sql="SELECT COUNT(*) AS total FROM public.sales",
        parameters={},
    )

    schema = {
        "schema": "public",
        "table_count": 1,
        "tables": [{"table_name": "sales"}],
    }

    result_mock = MagicMock()
    result_mock.columns = ["total"]
    result_mock.rows = [{"total": 12}]
    result_mock.row_count = 1
    result_mock.truncated = False
    result_mock.referenced_tables = ("public.sales",)

    with (
        patch(
            "services.analyst_service.inspect_schema",
            new_callable=AsyncMock,
            return_value=schema,
        ),
        patch(
            "services.analyst_service.execute_readonly_query",
            new_callable=AsyncMock,
            return_value=result_mock,
        ) as execute,
    ):
        result = await instance.analyze(
            question="How many documents are there for each file extension? Order by document count descending. ?",
            allowed_tables={"sales"},
        )

    assert result["rows"] == [{"total": 12}]
    assert result["row_count"] == 1
    execute.assert_awaited_once()

    await instance.close()


@pytest.mark.asyncio
async def test_invalid_sql_is_not_executed(service):
    instance, analyst = service

    analyst.create_plan.return_value = AnalysisPlan(
        explanation="Attempt an unsafe operation.",
        sql="DELETE FROM public.sales",
        parameters={},
    )

    schema = {
        "schema": "public",
        "table_count": 1,
        "tables": [{"table_name": "sales"}],
    }

    with (
        patch(
            "services.analyst_service.inspect_schema",
            new_callable=AsyncMock,
            return_value=schema,
        ),
        patch(
            "services.analyst_service.execute_readonly_query",
            new_callable=AsyncMock,
        ) as execute,
    ):
        with pytest.raises(SQLValidationError):
            await instance.analyze(
                question="Delete the sales records.",
                allowed_tables={"sales"},
            )

    execute.assert_not_awaited()
    await instance.close()


@pytest.mark.asyncio
async def test_empty_question_is_rejected_before_llm_call(service):
    instance, analyst = service

    with pytest.raises(ValueError, match="Question cannot be empty"):
        await instance.analyze(question=" ")

    analyst.create_plan.assert_not_awaited()
    await instance.close()