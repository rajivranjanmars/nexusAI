"""
Database Seeding Script �?" LPUAI Central.

Initialises the PostgreSQL database schema (if not exists) and seeds a sample
student profile for testing the "student://STU123/profile" resource.
"""

from __future__ import annotations

import asyncio
import datetime
import json
from urllib.parse import urlsplit, urlunsplit

from db.models.student import Student
from db.models.app import App
from db.models.token_usage import TokenUsage
from cache_module.models.response_cache import ResponseCache
from db.models.base import Base
import db.models.knowledge  # noqa: F401
from db.postgres import _build_connection_url, _engine, get_session
from shared.logger import get_logger
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import create_async_engine

logger = get_logger(__name__)


async def ensure_db_exists() -> None:
    """Connect to the 'postgres' maintenance database and create the target database if it doesn't exist."""
    parts = urlsplit(_build_connection_url("asyncpg"))
    db_name = parts.path.lstrip("/")
    admin_url = urlunsplit(parts._replace(path="/postgres"))

    admin_engine = create_async_engine(admin_url, isolation_level="AUTOCOMMIT")

    async with admin_engine.connect() as conn:
        logger.info("Checking for database: %s", db_name)
        exists = await conn.scalar(
            text("SELECT 1 FROM pg_database WHERE datname = :name"), {"name": db_name}
        )
        if not exists:
            await conn.execute(text(f'CREATE DATABASE "{db_name}"'))

    await admin_engine.dispose()
    logger.info("Database '%s' assured.", db_name)


async def seed_db() -> None:
    """Create tables and seed sample data."""
    await ensure_db_exists()
    
    logger.info("Initializing database schema...")
    
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    logger.info("Schema synced. Seeding STU123...")
    
    async with get_session() as session:
        import os
        
        json_path = os.path.join(os.path.dirname(__file__), "students.json")
        with open(json_path, "r", encoding="utf-8") as f:
            students_data = json.load(f)

        for s_data in students_data:
            # Convert date strings to datetime.date objects
            if "next_exam" in s_data and s_data["next_exam"]:
                s_data["next_exam"] = datetime.datetime.strptime(s_data["next_exam"], "%Y-%m-%d").date()

            res = await session.execute(select(Student).where(Student.student_id == s_data["student_id"]))
            existing = res.scalar_one_or_none()

            if existing:
                for k, v in s_data.items():
                    setattr(existing, k, v)
                logger.info("Updated existing student: %s", s_data["student_id"])
            else:
                student = Student(**s_data)
                session.add(student)
                logger.info("Seeded new student: %s", s_data["student_id"])

        await session.commit()

async def _main() -> None:
    from db.postgres import dispose_engines
    try:
        await seed_db()
    finally:
        await dispose_engines()


if __name__ == "__main__":
    asyncio.run(_main())
