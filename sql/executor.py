
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine

from sql.validator import validate_readonly_sql
from sqlalchemy.sql import text

@dataclass(frozen=True)
class QueryResult:
    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int
    truncated: bool
    referenced_tables: tuple[str, ...]


class SQLExecutionError(RuntimeError):
    """Raised when a query cannot be executed safely."""


async def execute_readonly_query(
    engine: AsyncEngine,
    sql: str,
    *,
    params: dict[str, Any] | None = None,
    allowed_tables: set[str] | None = None,
    max_rows: int = 500,
    statement_timeout_ms: int = 5000,
) -> QueryResult:
    """Validate and asynchronously execute a read-only query."""
    if not 100 <= statement_timeout_ms <= 60_000:
        raise ValueError(
            "statement_timeout_ms must be between 100 and 60000."
        )

    validated = validate_readonly_sql(
        sql,
        allowed_tables=allowed_tables,
        max_rows=max_rows,
    )

    try:
        async with engine.connect() as connection:
            async with connection.begin():
                await connection.exec_driver_sql(
                    "SET TRANSACTION READ ONLY"
                )

                await connection.execute(
                    text(
                        "SELECT set_config("
                        "'statement_timeout', :timeout, true)"
                    ),
                    {"timeout": f"{statement_timeout_ms}ms"},
                )

                statement = text(validated.sql)

                result = await connection.execute(
                    statement,
                    params or {},
                )

                columns = list(result.keys())
                fetched = result.mappings().fetchmany(max_rows + 1)

                truncated = len(fetched) > max_rows
                visible_rows = fetched[:max_rows]

                rows = [dict(row) for row in visible_rows]

                return QueryResult(
                    columns=columns,
                    rows=rows,
                    row_count=len(rows),
                    truncated=truncated,
                    referenced_tables=validated.referenced_tables,
                )

    except SQLAlchemyError as exc:
      original_error = exc.orig
      print("SQL execution error type:", type(original_error).__name__)
      print(
          "SQL execution error:",
          str(original_error).splitlines()[0][:300],
      )
      raise SQLExecutionError(
          "The read-only query could not be executed."
      ) from None