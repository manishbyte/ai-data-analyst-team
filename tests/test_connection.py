
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from database.connection import (
    DatabaseConfigurationError,
    check_database_connection,
    validate_database_url,
)
from database.connection import configure_windows_asyncio

configure_windows_asyncio()

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def test_empty_url_is_rejected():
    with pytest.raises(DatabaseConfigurationError):
        validate_database_url("")


def test_non_postgres_url_is_rejected():
    with pytest.raises(DatabaseConfigurationError):
        validate_database_url(
            "mysql+pymysql://user:password@localhost/example"
        )


def test_missing_host_is_rejected():
    with pytest.raises(DatabaseConfigurationError):
        validate_database_url(
            "postgresql://user:password@/example"
        )


def test_missing_database_is_rejected():
    with pytest.raises(DatabaseConfigurationError):
        validate_database_url(
            "postgresql://user:password@localhost"
        )


def test_missing_credentials_are_rejected():
    with pytest.raises(DatabaseConfigurationError):
        validate_database_url(
            "postgresql://localhost/example"
        )


def test_postgres_url_uses_psycopg3():
    url = validate_database_url(
        "postgresql://user:password@localhost:5000/example"
    )

    assert url.drivername == "postgresql+psycopg"
    assert url.host == "localhost"
    assert url.port == 5000
    assert url.database == "example"


@pytest.mark.asyncio
async def test_real_postgres_connection():
    database_url = os.getenv("TEST_DATABASE_URL")

    if not database_url:
        pytest.skip("TEST_DATABASE_URL is not configured.")

    result = await check_database_connection(database_url)

    assert result.success is True
    assert result.server_version
    assert "PostgreSQL" in result.server_version