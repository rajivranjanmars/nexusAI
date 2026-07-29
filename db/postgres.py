"""
PostgreSQL connection layer using SQLAlchemy + asyncpg/psycopg2.
"""
from __future__ import annotations
import contextlib
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from shared.config import settings
from shared.logger import get_logger

logger = get_logger(__name__)

def _build_connection_url(driver: str = "asyncpg") -> str:
    url = settings.database_url
    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            return f"postgresql+{driver}://{url[len(prefix):]}"
    return url

_engine = create_async_engine(
    _build_connection_url("asyncpg"),
    echo=settings.log_level.lower() == "debug",
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)

_async_session_factory = async_sessionmaker(
    bind=_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    session = _async_session_factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        logger.exception("Database session rolled back due to error")
        raise
    finally:
        await session.close()

_sync_engine = create_engine(
    _build_connection_url("psycopg2"),
    echo=settings.log_level.lower() == "debug",
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)

_sync_session_factory = sessionmaker(
    bind=_sync_engine,
    class_=Session,
    expire_on_commit=False,
)

@contextlib.contextmanager
def get_sync_session() -> Generator[Session, None, None]:
    session = _sync_session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        logger.exception("Database session rolled back due to error")
        raise
    finally:
        session.close()

async def dispose_engines() -> None:
    await _engine.dispose()
    _sync_engine.dispose()
    logger.info("PostgreSQL engines disposed")
