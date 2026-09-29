
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from sqlalchemy.engine import make_url

from app.config import get_settings
from database.connection import (
    configure_windows_asyncio,
    create_database_engine,
)
from graph.state import AnalystContext
from graph.workflow import build_analyst_graph


async def main() -> None:
    load_dotenv(Path(__file__).resolve().parent / ".env")
    settings = get_settings()

    if not settings.database_url:
        raise RuntimeError("DATABASE_URL is not configured.")

    # Reuse local connection details, targeting nse_rag.
    configured_url = make_url(
        settings.database_url.get_secret_value()
    )
    test_url = configured_url.set(database="nse_rag")

    # Create a request-scoped engine.
    engine = create_database_engine(
        test_url.render_as_string(hide_password=False),
        connect_timeout=settings.db_connect_timeout,
    )

    try:
        # The graph no longer needs a service-bound engine.
        graph = build_analyst_graph()

        context = AnalystContext(engine=engine)

        initial_state = {
            "messages": [
                HumanMessage(
                    content=(
                        "Show the 10 documents with the highest "
                        "number of chunks. Include each document's "
                        "title and chunk count, ordered by chunk "
                        "count descending."
                    )
                )
            ],
            "schema_name": "public",
            "allowed_tables": {
                "documents",
                "document_chunks",
            },
            "max_rows": 10,
            "status": "pending",
        }

        result = await graph.ainvoke(
            initial_state,
            context=context,
        )

        print("\n===== GRAPH RESULT =====")
        print("Status:", result.get("status"))
        print("Error:", result.get("error"))

        print("\n===== FINAL ANSWER =====")
        print(
            result.get(
                "final_answer",
                "No final answer generated.",
            )
        )

        print("\n===== STRUCTURED QUERY RESULT =====")
        query_result = result.get("query_result")

        if query_result is not None:
            print(
                json.dumps(
                    query_result,
                    indent=2,
                    default=str,
                )
            )

        if result.get("status") != "completed":
            raise RuntimeError(
                "The graph did not complete successfully."
            )

    finally:
        await engine.dispose()


if __name__ == "__main__":
    configure_windows_asyncio()
    asyncio.run(main())