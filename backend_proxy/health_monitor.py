"""Background health monitor — polls external dependencies every 30 seconds."""
from __future__ import annotations

import asyncio
from typing import Callable, Optional
from urllib.parse import urlparse

import httpx

from shared.config import settings
from shared.logger import get_logger

logger = get_logger(__name__)

_POLL_INTERVAL = 30
_state: dict[str, Optional[bool]] = {}  # None=unknown, True=up, False=down


def _get_alerter():
    if not settings.slack_webhook_url:
        return None
    from shared.alerting import SlackAlerter
    return SlackAlerter(
        webhook_url=settings.slack_webhook_url,
        rate_limit_seconds=settings.alert_rate_limit_seconds,
    )


async def _check(name: str, fn: Callable, alerter) -> None:
    try:
        await fn()
        was_down = _state.get(name) is False
        _state[name] = True
        if was_down:
            logger.info("Health monitor: %s recovered", name)
            if alerter:
                await alerter.send_alert(
                    level="INFO",
                    logger_name="health_monitor",
                    message=f"{name} recovered",
                )
    except Exception as exc:
        was_up = _state.get(name) is not False  # True or None → first alert
        _state[name] = False
        if was_up:
            logger.error("Health monitor: %s is DOWN — %s", name, exc)
            # logger.error fires SlackAlertHandler automatically when configured


async def _check_llm() -> None:
    import openai
    client = openai.AsyncOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        timeout=10,
    )
    await client.chat.completions.create(
        model=settings.llm_fast_model,
        messages=[{"role": "user", "content": "ping"}],
        max_tokens=1,
    )


async def _check_redis() -> None:
    from shared.redis_client import get_client
    client = await get_client()
    await client.ping()


async def _check_sql() -> None:
    from sqlalchemy import text
    from db.postgres import get_session
    async with get_session() as session:
        await session.execute(text("SELECT 1"))


async def _check_mcp() -> None:
    parsed = urlparse(settings.proxy_mcp_sse_url)
    url = f"{parsed.scheme}://{parsed.netloc}/health/version"
    async with httpx.AsyncClient(timeout=8) as client:
        resp = await client.get(url)
        resp.raise_for_status()


async def _poll_loop() -> None:
    alerter = _get_alerter()
    checks = [
        ("LLM API", _check_llm),
        ("Redis", _check_redis),
        ("PostgreSQL", _check_sql),
        ("MCP Server", _check_mcp),
    ]
    while True:
        for name, fn in checks:
            await _check(name, fn, alerter)
        await asyncio.sleep(_POLL_INTERVAL)


def start_health_monitor(app) -> None:
    @app.on_event("startup")
    async def _start() -> None:
        app.state.health_monitor_task = asyncio.create_task(_poll_loop())

    @app.on_event("shutdown")
    async def _stop() -> None:
        task = getattr(app.state, "health_monitor_task", None)
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
