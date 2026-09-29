
from typing import Any

from graph.state import AnalystContext
from graph.workflow import build_analyst_graph
from services.database_session import DatabaseSession


class GraphAnalysisService:
    """Run a LangGraph analysis against a user-provided database."""

    async def analyze(
        self,
        *,
        database_url: str,
        question: str,
        schema_name: str = "public",
        allowed_tables: set[str] | None = None,
        max_rows: int = 100,
    ) -> dict[str, Any]:
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        if schema_name != "public":
            raise ValueError("Only the public schema is supported.")

        if max_rows < 1:
            raise ValueError("max_rows must be at least 1.")

        session, database_info = await DatabaseSession.connect(
            database_url,
            schema=schema_name,
        )

        try:
            # Discover tables available through the database connection.
            available_tables = set(database_info.tables)

            if not available_tables:
                raise ValueError(
                    f"No tables found in schema '{database_info.schema}'."
                )

            # Use the discovered tables when no explicit allowlist is given.
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

            graph = build_analyst_graph()

            initial_state = {
                "messages": [
                    {
                        "role": "user",
                        "content": question.strip(),
                    }
                ],
                "schema_name": database_info.schema,
                "allowed_tables": effective_allowlist,
                "max_rows": max_rows,
                "status": "pending",
            }

            result = await graph.ainvoke(
                initial_state,
                context=AnalystContext(engine=session.engine),
            )

            if result.get("status") != "completed":
                raise RuntimeError(
                    result.get("error")
                    or "The analysis workflow did not complete."
                )

            return {
                "status": result.get("status"),
                "question": question.strip(),
                "database": {
                    "schema": database_info.schema,
                    "table_count": database_info.table_count,
                    "tables": database_info.tables,
                },
                "answer": result.get("final_answer"),
                "sql": result.get("sql"),
                "query_result": result.get("query_result"),
                "result_analysis": result.get("result_analysis"),
            }

        finally:
            await session.close()