
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator
from api.conversations import router as conversations_router
from fastapi import FastAPI
from sqlalchemy import text

from app.config import get_settings
from auth.router import router as auth_router
from database.app_session import (
    dispose_app_engine,
    get_app_engine,
)
from database.base import Base
from database.connection import configure_windows_asyncio
from database.models import (
    Conversation,
    Message,
    RevokedToken,
    User,
)  # noqa: F401


configure_windows_asyncio()
logger = logging.getLogger(__name__)



@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize application tables and release resources."""
    settings = get_settings()

    if settings.database_url is None:
        raise RuntimeError(
            "DATABASE_URL is not configured. "
            "Configure the application-owned PostgreSQL database."
        )

    try:
        engine = get_app_engine()

        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))

        logger.info("Application database initialized successfully.")
        yield

    except Exception:
        logger.exception("Application startup or runtime failed.")
        raise

    finally:
        await dispose_app_engine()
        logger.info("Application database connections closed.")


app = FastAPI(
    title=get_settings().app_name,
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(auth_router)
app.include_router(conversations_router)

@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}