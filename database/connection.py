
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    create_async_engine,
)
import asyncio
import sys

def configure_windows_asyncio() -> None:
    """Use the Selector event loop required by Psycopg async on Windows."""
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(
            asyncio.WindowsSelectorEventLoopPolicy()
        )


class DatabaseConfigurationError(ValueError):
    """Raised when the database configuration is invalid."""


class DatabaseConnectionError(RuntimeError):
    """Raised when the database connection fails."""


@dataclass(frozen=True)
class ConnectionTestResult:
    success: bool
    server_version: str | None = None


def validate_database_url(database_url: str) -> URL:
    """Validate a PostgreSQL URL without opening a connection."""
    if not isinstance(database_url, str) or not database_url.strip():
        raise DatabaseConfigurationError(
            "A database connection URL is required."
        )

    try:
        url = make_url(database_url.strip())
    except Exception as exc:
        raise DatabaseConfigurationError(
            "The database connection URL is malformed."
        ) from exc

    if url.drivername not in {
        "postgresql",
        "postgresql+psycopg",
    }:
        raise DatabaseConfigurationError(
            "Only PostgreSQL with Psycopg 3 is supported."
        )

    if not url.host:
        raise DatabaseConfigurationError(
            "The database URL must include a host."
        )

    if not url.database:
        raise DatabaseConfigurationError(
            "The database URL must include a database name."
        )

    if not url.username or url.password is None:
        raise DatabaseConfigurationError(
            "The database URL must include credentials."
        )

    if url.port is not None and not 1 <= url.port <= 65535:
        raise DatabaseConfigurationError(
            "The database port is invalid."
        )

    return url.set(drivername="postgresql+psycopg")


def create_database_engine(
    database_url: str,
    *,
    connect_timeout: int = 5,
    pool_size: int = 2,
    max_overflow: int = 0,
) -> AsyncEngine:
    """Create an async SQLAlchemy engine using Psycopg 3."""
    if not 1 <= connect_timeout <= 30:
        raise ValueError("connect_timeout must be between 1 and 30.")

    if pool_size < 1 or max_overflow < 0:
        raise ValueError("Invalid connection pool configuration.")

    url = validate_database_url(database_url)

    return create_async_engine(
        url,
        pool_pre_ping=True,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_timeout=10,
        connect_args={
            "connect_timeout": connect_timeout,
            "application_name": "ai_data_analyst",
        },
    )


async def check_database_connection(
    database_url: str,
    *,
    connect_timeout: int = 5,
) -> ConnectionTestResult:
    """Asynchronously connect and read the PostgreSQL version."""
    engine = create_database_engine(
        database_url,
        connect_timeout=connect_timeout,
        pool_size=1,
        max_overflow=0,
    )

    try:
        async with engine.connect() as connection:
            result = await connection.execute(text("SELECT version()"))
            version = result.scalar_one()

        return ConnectionTestResult(
            success=True,
            server_version=str(version),
        )

    except SQLAlchemyError:
        # Do not expose the URL or driver error to the caller.
        raise DatabaseConnectionError(
            "Could not connect to the PostgreSQL database. "
            "Check the connection settings and database availability."
        ) from None

    finally:
        await engine.dispose()