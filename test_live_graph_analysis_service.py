
import asyncio
import json
from database.connection import configure_windows_asyncio

from sqlalchemy.engine import make_url

from app.config import get_settings
from services.graph_analysis_service import GraphAnalysisService


async def main():
    settings = get_settings()

    if settings.database_url is None:
        raise RuntimeError(
            "DATABASE_URL is missing from your local .env configuration."
        )

    # Read the URL locally without displaying credentials.
    database_url = settings.database_url.get_secret_value()

    # Target the existing NSE database.
    parsed_url = make_url(database_url)
    nse_url = parsed_url.set(database="nse_rag").render_as_string(
        hide_password=False
    )

    service = GraphAnalysisService()

    result = await service.analyze(
        database_url=nse_url,
        question=(
            "Which 10 documents have the most chunks?"
           
        ),
        schema_name="public",
        # allowed_tables={"documents", "document_chunks"},
        max_rows=10,
    )

    print("\n" + "=" * 60)
    print("LIVE GRAPH ANALYSIS SERVICE TEST")
    print("=" * 60)

    print("\nStatus:")
    print(result["status"])

    print("\nDatabase schema:")
    print(result["database"]["schema"])

    print("\nTables discovered:")
    print(json.dumps(result["database"]["tables"], indent=2))

    print("\nGenerated SQL:")
    print(result.get("sql"))

    print("\nQuery results:")
    print(json.dumps(result.get("query_result"), indent=2, default=str))

    print("\nFinal answer:")
    print(result.get("answer"))

    print("\nResult analysis:")
    print(result.get("result_analysis"))

    assert result["status"] == "completed"
    assert result.get("query_result") is not None
    assert isinstance(result["query_result"].get("rows"), list)

    print("\nLIVE INTEGRATION TEST PASSED")


if __name__ == "__main__":
    configure_windows_asyncio()
    asyncio.run(main())