
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncEngine

from app.config import get_settings
from database.connection import (
    check_database_connection,
    create_database_engine,
)
from database.schema_inspector import inspect_schema


@dataclass(frozen=True)
class DatabaseSessionInfo:
    server_version: str
    schema: str
    table_count: int
    tables: list[str]


class DatabaseSession:
    """Manage a database engine for one analysis session."""

    def __init__(self, engine: AsyncEngine) -> None:
        self.engine = engine
        self._closed = False

    @classmethod
    async def connect(
        cls,
        database_url: str,
        *,
        schema: str = "public",
    ) -> tuple["DatabaseSession", DatabaseSessionInfo]:
        settings = get_settings()

        # Test the supplied URL before creating the session engine.
        connection_result = await check_database_connection(
            database_url,
            connect_timeout=settings.db_connect_timeout,
        )

        if not connection_result.success:
            raise RuntimeError("Database connection failed.")

        engine = create_database_engine(
            database_url,
            connect_timeout=settings.db_connect_timeout,
        )

        try:
            schema_info = await inspect_schema(
                engine,
                schema=schema,
            )

            info = DatabaseSessionInfo(
                server_version=connection_result.server_version or "Unknown",
                schema=schema_info["schema"],
                table_count=schema_info["table_count"],
                tables=[
                    table["table_name"]
                    for table in schema_info["tables"]
                ],
            )

            return cls(engine), info

        except Exception:
            await engine.dispose()
            raise

    async def close(self) -> None:
        """Dispose of the session engine safely."""
        if not self._closed:
            await self.engine.dispose()
            self._closed = True

    async def __aenter__(self) -> "DatabaseSession":
        if self._closed:
            raise RuntimeError("Database session is already closed.")
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.close()