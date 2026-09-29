import asyncio
from pathlib import Path

from dotenv import dotenv_values
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from database.connection import validate_database_url
from database.connection import configure_windows_asyncio

async def main():
    env_path = Path(__file__).resolve().parent / ".env"
    env = dotenv_values(env_path)
    database_url = env.get("TEST_DATABASE_URL")

    if not database_url:
        print("ERROR: TEST_DATABASE_URL is missing from .env")
        return

    try:
        url = validate_database_url(database_url)
        print("URL validation: OK")
        print("Host:", url.host)
        print("Port:", url.port or 5432)
        print("Database:", url.database)

        engine = create_async_engine(
            url,
            connect_args={"connect_timeout": 5},
        )

        try:
            async with engine.connect() as connection:
                result = await connection.execute(text("SELECT 1"))
                print("Connection: OK")
                print("Test query result:", result.scalar_one())
        finally:
            await engine.dispose()

    except Exception as exc:
        print("Connection: FAILED")
        print("Error type:", type(exc).__name__)
        print("Error:", str(exc).splitlines()[0][:400])

if __name__ == "__main__":
    configure_windows_asyncio()
    asyncio.run(main())
