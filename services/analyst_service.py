
from typing import Any

from agents.analyst import DataAnalyst
from app.config import get_settings
from database.connection import create_database_engine
from database.schema_inspector import inspect_schema
from sql.executor import execute_readonly_query
from sql.validator import validate_readonly_sql


class AnalystService:
    """Coordinate schema inspection, SQL planning and safe execution."""

    def __init__(
        self,
        *,
        database_url: str,
        analyst: DataAnalyst | None = None,
    ) -> None:
        settings = get_settings()

        self.engine = create_database_engine(
            database_url,
            connect_timeout=settings.db_connect_timeout,
        )
        self.analyst = analyst or DataAnalyst()

    async def analyze(
        self,
        *,
        question: str,
        schema_name: str = "public",
        allowed_tables: set[str] | None = None,
        max_rows: int = 100,
    ) -> dict[str, Any]:
        """Generate, validate and execute a read-only analytical query."""

        # 1. Validate the user's request.
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        if schema_name != "public":
            raise ValueError("Only the public schema is supported.")

        if max_rows < 1:
            raise ValueError("max_rows must be at least 1.")

        # 2. Inspect the database schema.
        schema = await inspect_schema(
            self.engine,
            schema=schema_name,
        )

        if schema["table_count"] == 0:
            raise ValueError(
                f"No tables found in schema '{schema_name}'."
            )

        # 3. Establish an explicit allowlist.
        available_tables = {
            table["table_name"]
            for table in schema["tables"]
        }

        if allowed_tables is None:
            effective_allowlist = available_tables
        else:
            if not allowed_tables:
                raise ValueError(
                    "The table allowlist cannot be empty."
                )

            unknown_tables = allowed_tables - available_tables

            if unknown_tables:
                raise ValueError(
                    "Unknown or unavailable tables: "
                    + ", ".join(sorted(unknown_tables))
                )

            effective_allowlist = set(allowed_tables)

        # 4. Expose only permitted tables to the analyst.
        visible_tables = [
            table
            for table in schema["tables"]
            if table["table_name"] in effective_allowlist
        ]

        if not visible_tables:
            raise ValueError(
                "No permitted tables are available for analysis."
            )

        filtered_schema = {
            "schema": schema["schema"],
            "table_count": len(visible_tables),
            "tables": visible_tables,
        }

        # 5. Ask the analyst to generate a query.
        plan = await self.analyst.create_plan(
            question=question.strip(),
            schema_context=str(filtered_schema),
        )

        # 6. Validate generated SQL before execution.
        validated = validate_readonly_sql(
            plan.sql,
            allowed_tables=effective_allowlist,
            max_rows=max_rows,
        )

        # 7. Execute the validated query.
        result = await execute_readonly_query(
            self.engine,
            validated.sql,
            params=plan.parameters,
            allowed_tables=effective_allowlist,
            max_rows=max_rows,
        )

        # 8. Return structured results.
        return {
            "question": question.strip(),
            "explanation": plan.explanation,
            "sql": validated.sql,
            "parameters": plan.parameters,
            "columns": result.columns,
            "rows": result.rows,
            "row_count": result.row_count,
            "truncated": result.truncated,
            "referenced_tables": list(result.referenced_tables),
        }

    async def close(self) -> None:
        """Release database connections and the connection pool."""
        await self.engine.dispose()