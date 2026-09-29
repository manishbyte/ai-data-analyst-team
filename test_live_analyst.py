
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from app.config import get_settings
from services.analyst_service import AnalystService


async def main() -> None:
    # Load this project's environment configuration.
    load_dotenv(Path(__file__).resolve().parent / ".env")

    settings = get_settings()

    # Use nse_rag only for this test; do not change the main DATABASE_URL.
    database_url = settings.database_url.get_secret_value()

    # Replace the database name in the URL while preserving credentials.
    from sqlalchemy.engine import make_url

    url = make_url(database_url)
    test_url = url.set(database="nse_rag")

    service = AnalystService(
        database_url=test_url.render_as_string(hide_password=False),
    )

    try:
        result = await service.analyze(
            question=(
                "Count the documents and show the number of chunks "
                "for each document. Return the document title and "
                "chunk count, ordered by chunk count descending."
            ),
            schema_name="public",
            allowed_tables={"documents", "document_chunks"},
            max_rows=10,
        )

        # Avoid printing parameters or document contents unnecessarily.
        print(json.dumps(result, indent=2, default=str))

    finally:
        await service.close()


if __name__ == "__main__":
    from database.connection import configure_windows_asyncio

    configure_windows_asyncio()
    asyncio.run(main())