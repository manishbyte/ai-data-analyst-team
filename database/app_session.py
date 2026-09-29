
from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.config import get_settings
from database.connection import create_database_engine


_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_app_engine() -> AsyncEngine:
    """Return the lazily initialized application database engine."""
    global _engine

    if _engine is None:
        settings = get_settings()

        if settings.database_url is None:
            raise RuntimeError(
                "The application database is not configured."
            )

        _engine = create_database_engine(
            settings.database_url.get_secret_value(),
            connect_timeout=settings.db_connect_timeout,
            pool_size=5,
            max_overflow=5,
        )

    return _engine


def get_app_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the shared async session factory."""
    global _session_factory

    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_app_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )

    return _session_factory


async def get_db_session() -> AsyncIterator[AsyncSession]:
    """Yield a session and close it after the request or operation."""
    session_factory = get_app_session_factory()

    async with session_factory() as session:
        yield session


async def dispose_app_engine() -> None:
    """Close the application database connection pool."""
    global _engine, _session_factory

    if _engine is not None:
        await _engine.dispose()

    _engine = None
    _session_factory = None