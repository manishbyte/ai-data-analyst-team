
import asyncio
import json
from pathlib import Path

from dotenv import dotenv_values

from database.connection import create_database_engine
from database.schema_inspector import inspect_schema


async def main() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    env = dotenv_values(env_path)

    database_url = env.get("TEST_DATABASE_URL")

    if not database_url:
        raise RuntimeError(
            "TEST_DATABASE_URL is not configured in .env"
        )

    engine = create_database_engine(database_url)

    try:
        schema_info = await inspect_schema(
            engine,
            schema="public",
        )
        print(json.dumps(schema_info, indent=2))
    finally:
        await engine.dispose()


if __name__ == "__main__":
    from database.connection import configure_windows_asyncio

    configure_windows_asyncio()
    asyncio.run(main())