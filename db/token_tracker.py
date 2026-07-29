"""
Token usage tracker using SQLAlchemy ORM.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import case, cast, func, select, Date
from db.models.token_usage import TokenUsage
from db.postgres import get_sync_session
from shared.logger import get_logger

logger = get_logger(__name__)

def _record_usage_sync(
    app_id: Optional[str],
    actor_id: Optional[str],
    actor_type: Optional[str],
    action: Optional[str],
    model: Optional[str],
    model_tier: Optional[str],
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    estimated_cost_usd: Optional[float] = None,
    cache_hit: bool = False,
    latency_ms: Optional[int] = None,
    correlation_id: Optional[str] = None,
) -> None:
    with get_sync_session() as session:
        usage = TokenUsage(
            app_id=app_id,
            actor_id=actor_id,
            actor_type=actor_type,
            action=action,
            model=model,
            model_tier=model_tier,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=estimated_cost_usd,
            cache_hit=cache_hit,
            latency_ms=latency_ms,
            correlation_id=correlation_id,
        )
        session.add(usage)
        session.commit()

async def record_usage(
    app_id: Optional[str] = None,
    actor_id: Optional[str] = None,
    actor_type: Optional[str] = None,
    action: Optional[str] = None,
    model: Optional[str] = None,
    model_tier: Optional[str] = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    estimated_cost_usd: Optional[float] = None,
    cache_hit: bool = False,
    latency_ms: Optional[int] = None,
    correlation_id: Optional[str] = None,
) -> None:
    await asyncio.to_thread(
        _record_usage_sync,
        app_id=app_id,
        actor_id=actor_id,
        actor_type=actor_type,
        action=action,
        model=model,
        model_tier=model_tier,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        estimated_cost_usd=estimated_cost_usd,
        cache_hit=cache_hit,
        latency_ms=latency_ms,
        correlation_id=correlation_id,
    )

async def get_monthly_tokens(app_id: str) -> int:
    def _query() -> int:
        with get_sync_session() as session:
            result = session.query(func.sum(TokenUsage.total_tokens)).filter(
                TokenUsage.app_id == app_id,
                TokenUsage.timestamp >= func.date_trunc('month', func.now())
            ).scalar()
            return int(result) if result else 0
            
    return await asyncio.to_thread(_query)


# ── Admin observability query helpers ───────────────────────────────────────

def _period_filter(period: str) -> datetime:
    """Return the start datetime for a given period string."""
    from datetime import timedelta
    now = datetime.now(timezone.utc)
    if period == "today":
        return now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == "7d":
        return now - timedelta(days=7)
    elif period == "30d":
        return now - timedelta(days=30)
    else:
        # "all" — return epoch
        return datetime(2000, 1, 1, tzinfo=timezone.utc)


def _get_usage_summary_sync(
    period: str = "30d",
    app_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Aggregated token usage summary with per-model and per-app breakdowns."""
    cutoff = _period_filter(period)
    with get_sync_session() as session:
        base = session.query(TokenUsage).filter(TokenUsage.timestamp >= cutoff)
        if app_id:
            base = base.filter(TokenUsage.app_id == app_id)

        # Totals
        totals = session.query(
            func.count(TokenUsage.id).label("total_calls"),
            func.coalesce(func.sum(TokenUsage.prompt_tokens), 0).label("total_prompt_tokens"),
            func.coalesce(func.sum(TokenUsage.completion_tokens), 0).label("total_completion_tokens"),
            func.coalesce(func.sum(TokenUsage.total_tokens), 0).label("total_tokens"),
            func.coalesce(func.sum(TokenUsage.estimated_cost_usd), 0).label("total_cost_usd"),
            func.coalesce(func.avg(TokenUsage.latency_ms), 0).label("avg_latency_ms"),
        ).filter(TokenUsage.timestamp >= cutoff)
        if app_id:
            totals = totals.filter(TokenUsage.app_id == app_id)
        row = totals.one()

        # By model
        by_model_rows = session.query(
            TokenUsage.model,
            func.count(TokenUsage.id).label("calls"),
            func.coalesce(func.sum(TokenUsage.total_tokens), 0).label("tokens"),
            func.coalesce(func.sum(TokenUsage.estimated_cost_usd), 0).label("cost_usd"),
        ).filter(
            TokenUsage.timestamp >= cutoff,
        )
        if app_id:
            by_model_rows = by_model_rows.filter(TokenUsage.app_id == app_id)
        by_model_rows = by_model_rows.group_by(TokenUsage.model).all()

        # By action
        by_action_rows = session.query(
            TokenUsage.action,
            func.count(TokenUsage.id).label("calls"),
            func.coalesce(func.sum(TokenUsage.total_tokens), 0).label("tokens"),
        ).filter(
            TokenUsage.timestamp >= cutoff,
        )
        if app_id:
            by_action_rows = by_action_rows.filter(TokenUsage.app_id == app_id)
        by_action_rows = by_action_rows.group_by(TokenUsage.action).all()

        return {
            "period": period,
            "total_calls": row.total_calls,
            "total_prompt_tokens": int(row.total_prompt_tokens),
            "total_completion_tokens": int(row.total_completion_tokens),
            "total_tokens": int(row.total_tokens),
            "total_cost_usd": round(float(row.total_cost_usd), 6),
            "avg_latency_ms": round(float(row.avg_latency_ms), 1),
            "by_model": [
                {
                    "model": r.model or "unknown",
                    "calls": r.calls,
                    "tokens": int(r.tokens),
                    "cost_usd": round(float(r.cost_usd), 6),
                }
                for r in by_model_rows
            ],
            "by_action": [
                {
                    "action": r.action or "unknown",
                    "calls": r.calls,
                    "tokens": int(r.tokens),
                }
                for r in by_action_rows
            ],
        }


async def get_usage_summary(
    period: str = "30d",
    app_id: Optional[str] = None,
) -> Dict[str, Any]:
    return await asyncio.to_thread(_get_usage_summary_sync, period, app_id)


def _get_usage_by_app_sync(
    app_id: str,
    period: str = "30d",
) -> Dict[str, Any]:
    """Daily token usage breakdown for a specific app."""
    cutoff = _period_filter(period)
    with get_sync_session() as session:
        daily_rows = session.query(
            cast(TokenUsage.timestamp, Date).label("date"),
            func.count(TokenUsage.id).label("calls"),
            func.coalesce(func.sum(TokenUsage.total_tokens), 0).label("tokens"),
            func.coalesce(func.sum(TokenUsage.estimated_cost_usd), 0).label("cost_usd"),
        ).filter(
            TokenUsage.app_id == app_id,
            TokenUsage.timestamp >= cutoff,
        ).group_by(
            cast(TokenUsage.timestamp, Date),
        ).order_by(
            cast(TokenUsage.timestamp, Date).asc(),
        ).all()

        return {
            "app_id": app_id,
            "period": period,
            "daily": [
                {
                    "date": str(r.date),
                    "calls": r.calls,
                    "tokens": int(r.tokens),
                    "cost_usd": round(float(r.cost_usd), 6),
                }
                for r in daily_rows
            ],
        }


async def get_usage_by_app(
    app_id: str,
    period: str = "30d",
) -> Dict[str, Any]:
    return await asyncio.to_thread(_get_usage_by_app_sync, app_id, period)


def _get_recent_usage_sync(
    limit: int = 50,
    offset: int = 0,
    app_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Paginated list of recent individual LLM invocations."""
    with get_sync_session() as session:
        query = session.query(TokenUsage).order_by(TokenUsage.timestamp.desc())
        if app_id:
            query = query.filter(TokenUsage.app_id == app_id)

        total = query.count()
        rows = query.offset(offset).limit(limit).all()

        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "items": [row.to_dict() for row in rows],
        }


async def get_recent_usage(
    limit: int = 50,
    offset: int = 0,
    app_id: Optional[str] = None,
) -> Dict[str, Any]:
    return await asyncio.to_thread(_get_recent_usage_sync, limit, offset, app_id)
