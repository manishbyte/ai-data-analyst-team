
import asyncio

from sqlalchemy import text

from app.config import get_settings
from database.connection import (
    create_database_engine,
    configure_windows_asyncio,
)


async def main():
    settings = get_settings()

    if not settings.database_url:
        raise RuntimeError("DATABASE_URL is not configured.")

    engine = create_database_engine(
        settings.database_url.get_secret_value(),
        connect_timeout=settings.db_connect_timeout,
    )

    try:
        async with engine.connect() as conn:
            result = await conn.execute(
                text("""
                    SELECT table_schema, table_name
                    FROM information_schema.tables
                    WHERE table_type = 'BASE TABLE'
                      AND table_schema NOT IN (
                          'pg_catalog',
                          'information_schema'
                      )
                    ORDER BY table_schema, table_name
                """)
            )

            tables = result.fetchall()

            if not tables:
                print("No user tables found in this database.")
            else:
                print("Tables found:")
                for schema, table in tables:
                    print(f"{schema}.{table}")

    finally:
        await engine.dispose()


if __name__ == "__main__":
    configure_windows_asyncio()
    asyncio.run(main())