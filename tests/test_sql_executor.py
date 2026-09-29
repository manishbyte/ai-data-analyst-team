
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from database.connection import create_database_engine
from sql.executor import execute_readonly_query
from sql.validator import SQLValidationError


load_dotenv(Path(__file__).resolve().parents[1] / ".env")


@pytest.fixture
async def engine():
    database_url = os.getenv("TEST_DATABASE_URL")

    if not database_url:
        pytest.skip("TEST_DATABASE_URL is not configured.")

    db_engine = create_database_engine(database_url)

    try:
        yield db_engine
    finally:
        await db_engine.dispose()


@pytest.mark.asyncio
async def test_executes_select_and_returns_structured_result(engine):
    result = await execute_readonly_query(
        engine,
        "SELECT 42 AS answer",
    )

    assert result.columns == ["answer"]
    assert result.rows == [{"answer": 42}]
    assert result.row_count == 1
    assert result.truncated is False


@pytest.mark.asyncio
async def test_supports_bound_parameters(engine):
    result = await execute_readonly_query(
        engine,
        "SELECT CAST(:value AS INTEGER) AS answer",
        params={"value": 123},
    )

    assert result.rows == [{"answer": 123}]


@pytest.mark.asyncio
async def test_transaction_is_read_only(engine):
    result = await execute_readonly_query(
        engine,
        "SELECT current_setting('transaction_read_only') AS mode",
    )

    assert result.rows[0]["mode"] == "on"


@pytest.mark.asyncio
async def test_rejects_write_query(engine):
    with pytest.raises(SQLValidationError):
        await execute_readonly_query(
            engine,
            "DELETE FROM customers",
        )


@pytest.mark.asyncio
async def test_rejects_invalid_timeout(engine):
    with pytest.raises(ValueError):
        await execute_readonly_query(
            engine,
            "SELECT 1",
            statement_timeout_ms=0,
        )


@pytest.mark.asyncio
async def test_reports_truncation(engine):
    query = """
        SELECT 1 AS n
        UNION ALL SELECT 2
        UNION ALL SELECT 3
        UNION ALL SELECT 4
        UNION ALL SELECT 5
        UNION ALL SELECT 6
        UNION ALL SELECT 7
        UNION ALL SELECT 8
        UNION ALL SELECT 9
        UNION ALL SELECT 10
    """

    result = await execute_readonly_query(
        engine,
        query,
        max_rows=3,
    )

    assert result.row_count == 3
    assert result.truncated is True
    assert [row["n"] for row in result.rows] == [1, 2, 3]