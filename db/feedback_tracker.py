"""Chat feedback tracker using SQLAlchemy ORM."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import func, select

from db.models.feedback import ChatFeedback
from db.postgres import get_sync_session


def _serialize_feedback(feedback: ChatFeedback) -> dict[str, str | bool | None]:
    """Serialize a feedback ORM row into an API-safe dictionary."""

    return feedback.to_dict()


def _record_feedback_sync(
    *,
    app_id: str,
    actor_id: str,
    session_id: str,
    feedback_flag: bool,
    feedback_text: Optional[str],
    user_message: str,
    response: str,
) -> str:
    """Persist one feedback record using the synchronous SQLAlchemy session."""

    feedback_id = uuid.uuid4()
    with get_sync_session() as session:
        feedback = ChatFeedback(
            id=feedback_id,
            app_id=uuid.UUID(app_id),
            actor_id=actor_id,
            session_id=session_id,
            feedback_flag=feedback_flag,
            feedback_text=feedback_text,
            user_message=user_message,
            response=response,
        )
        session.add(feedback)
        session.flush()
    return str(feedback_id)


def _apply_feedback_date_filters(
    statement: Any,
    start_date: Optional[datetime],
    end_date: Optional[datetime],
    actor_id: Optional[str],
) -> Any:
    """Apply actor and created_at filters to a SQLAlchemy statement."""

    if start_date is not None:
        statement = statement.where(ChatFeedback.created_at >= start_date)
    if end_date is not None:
        statement = statement.where(ChatFeedback.created_at <= end_date)
    if actor_id is not None:
        statement = statement.where(ChatFeedback.actor_id == actor_id)
    return statement


def _get_feedback_by_id_sync(feedback_id: str) -> Optional[dict[str, str | bool | None]]:
    """Fetch a single feedback row by ID using the synchronous session."""

    try:
        parsed_feedback_id = uuid.UUID(feedback_id)
    except ValueError:
        return None

    with get_sync_session() as session:
        feedback = session.get(ChatFeedback, parsed_feedback_id)
        if feedback is None:
            return None
        return _serialize_feedback(feedback)


def _list_feedback_sync(
    *,
    page: int,
    page_size: int,
    last: Optional[int],
    start_date: Optional[datetime],
    end_date: Optional[datetime],
    actor_id: Optional[str],
) -> dict[str, object]:
    """Fetch feedback rows in either paginated or recent-N mode."""

    with get_sync_session() as session:
        base_statement = select(ChatFeedback)
        base_statement = _apply_feedback_date_filters(base_statement, start_date, end_date, actor_id)
        base_statement = base_statement.order_by(ChatFeedback.created_at.desc())

        if last is not None:
            rows = session.execute(base_statement.limit(last)).scalars().all()
            return {
                'items': [_serialize_feedback(row) for row in rows],
                'page': 1,
                'page_size': last,
                'total': len(rows),
            }

        count_statement = select(func.count()).select_from(ChatFeedback)
        count_statement = _apply_feedback_date_filters(count_statement, start_date, end_date, actor_id)
        total = session.execute(count_statement).scalar_one()
        offset = (page - 1) * page_size
        rows = session.execute(base_statement.offset(offset).limit(page_size)).scalars().all()
        return {
            'items': [_serialize_feedback(row) for row in rows],
            'page': page,
            'page_size': page_size,
            'total': int(total),
        }


async def record_feedback(
    *,
    app_id: str,
    actor_id: str,
    session_id: str,
    feedback_flag: bool,
    feedback_text: Optional[str],
    user_message: str,
    response: str,
) -> str:
    """Persist one chat feedback record and return its generated UUID."""

    return await asyncio.to_thread(
        _record_feedback_sync,
        app_id=app_id,
        actor_id=actor_id,
        session_id=session_id,
        feedback_flag=feedback_flag,
        feedback_text=feedback_text,
        user_message=user_message,
        response=response,
    )


async def get_feedback_by_id(feedback_id: str) -> Optional[dict[str, str | bool | None]]:
    """Return one stored feedback record by ID, if present."""

    return await asyncio.to_thread(_get_feedback_by_id_sync, feedback_id)


async def list_feedback(
    *,
    page: int,
    page_size: int,
    last: Optional[int],
    start_date: Optional[datetime],
    end_date: Optional[datetime],
    actor_id: Optional[str],
) -> dict[str, object]:
    """Return feedback rows for either paginated or recent-N admin queries."""

    return await asyncio.to_thread(
        _list_feedback_sync,
        page=page,
        page_size=page_size,
        last=last,
        start_date=start_date,
        end_date=end_date,
        actor_id=actor_id,
    )
