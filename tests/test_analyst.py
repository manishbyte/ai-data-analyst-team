import json
from unittest.mock import AsyncMock

import pytest

from agents.analyst import DataAnalyst, AnalysisPlan


@pytest.mark.asyncio
async def test_create_plan_returns_structured_plan():
    llm = AsyncMock()
    llm.generate.return_value = json.dumps({
        "explanation": "Calculate total sales by product.",
        "sql": (
            "SELECT product_name, SUM(amount) AS total_sales "
            "FROM public.sales "
            "GROUP BY product_name "
            "ORDER BY total_sales DESC "
            "LIMIT 100"
        ),
        "parameters": {},
    })

    analyst = DataAnalyst(llm=llm)

    plan = await analyst.create_plan(
        question="Show total sales by product.",
        schema_context=(
            "Table public.sales: "
            "product_name TEXT, amount NUMERIC"
        ),
    )

    assert isinstance(plan, AnalysisPlan)
    assert plan.sql.startswith("SELECT")
    assert plan.parameters == {}
    llm.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_empty_question_is_rejected():
    llm = AsyncMock()
    analyst = DataAnalyst(llm=llm)

    with pytest.raises(ValueError, match="Question cannot be empty"):
        await analyst.create_plan(
            question="  ",
            schema_context="Table public.sales",
        )

    llm.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_invalid_json_is_rejected():
    llm = AsyncMock()
    llm.generate.return_value = "This is not JSON"

    analyst = DataAnalyst(llm=llm)

    with pytest.raises(
        ValueError,
        match="invalid analysis plan",
    ):
        await analyst.create_plan(
            question="Show total sales.",
            schema_context="Table public.sales",
        )


@pytest.mark.asyncio
async def test_empty_schema_is_rejected():
    llm = AsyncMock()
    analyst = DataAnalyst(llm=llm)

    with pytest.raises(ValueError, match="Schema context cannot be empty"):
        await analyst.create_plan(
            question="Show total sales.",
            schema_context=" ",
        )

    llm.generate.assert_not_awaited()