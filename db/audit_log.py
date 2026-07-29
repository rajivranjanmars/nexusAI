"""App configuration audit trail using SQLAlchemy ORM."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, Dict, Optional

from sqlalchemy import select

from db.models.audit_log import AppAuditLog
from db.postgres import get_sync_session


def _record_app_change_sync(
    *,
    app_id: str,
    changed_by: str,
    action: str,
    changes: Dict[str, Any],
) -> str:
    """Persist one audit entry using the synchronous SQLAlchemy session."""

    entry_id = uuid.uuid4()
    with get_sync_session() as session:
        entry = AppAuditLog(
            id=entry_id,
            app_id=uuid.UUID(app_id),
            changed_by=changed_by,
            action=action,
            changes=changes,
        )
        session.add(entry)
        session.flush()
    return str(entry_id)


def _list_app_audit_log_sync(*, app_id: str, limit: int) -> list[dict]:
    """Fetch the most recent audit entries for one app."""

    with get_sync_session() as session:
        statement = (
            select(AppAuditLog)
            .where(AppAuditLog.app_id == uuid.UUID(app_id))
            .order_by(AppAuditLog.created_at.desc())
            .limit(limit)
        )
        rows = session.execute(statement).scalars().all()
        return [row.to_dict() for row in rows]


async def record_app_change(
    *,
    app_id: str,
    changed_by: str,
    action: str,
    changes: Dict[str, Any],
) -> Optional[str]:
    """Persist one audit entry for an app config change.

    No-ops (does not raise) when ``changes`` is empty, since a PATCH that
    touches no field should not create a log entry.
    """
    if not changes:
        return None
    return await asyncio.to_thread(
        _record_app_change_sync,
        app_id=app_id,
        changed_by=changed_by,
        action=action,
        changes=changes,
    )


async def list_app_audit_log(*, app_id: str, limit: int = 100) -> list[dict]:
    """Return the most recent audit entries for ``app_id``, newest first."""

    return await asyncio.to_thread(_list_app_audit_log_sync, app_id=app_id, limit=limit)
