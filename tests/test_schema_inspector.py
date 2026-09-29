
from pathlib import Path

import pytest
import pytest_asyncio
from dotenv import dotenv_values

from database.connection import create_database_engine
from database.schema_inspector import inspect_schema


@pytest_asyncio.fixture
async def engine():
    """Create and dispose of the async test database engine."""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    env = dotenv_values(env_path)
    database_url = env.get("TEST_DATABASE_URL")

    if not database_url:
        pytest.skip("TEST_DATABASE_URL is not configured.")

    db_engine = create_database_engine(database_url)

    try:
        yield db_engine
    finally:
        await db_engine.dispose()


@pytest.mark.asyncio
async def test_inspector_returns_schema_structure(engine):
    result = await inspect_schema(engine, schema="public")

    assert result["schema"] == "public"
    assert isinstance(result["table_count"], int)
    assert result["table_count"] >= 0
    assert isinstance(result["tables"], list)


@pytest.mark.asyncio
async def test_inspector_rejects_empty_schema(engine):
    with pytest.raises(ValueError, match="Schema name cannot be empty"):
        await inspect_schema(engine, schema="")