"""Slack alerting handler for the structured logger."""
from __future__ import annotations

import asyncio
import logging
import sys
import threading
import time

import httpx


class SlackAlerter:
    def __init__(self, webhook_url: str, rate_limit_seconds: int = 60) -> None:
        self._url = webhook_url
        self._rate_limit_seconds = rate_limit_seconds
        self._rate_limits: dict[str, float] = {}
        self._lock = threading.Lock()

    def _should_send(self, fingerprint: str) -> bool:
        with self._lock:
            now = time.monotonic()
            if now - self._rate_limits.get(fingerprint, 0) < self._rate_limit_seconds:
                return False
            self._rate_limits[fingerprint] = now
            return True

    async def send_alert(
        self,
        level: str,
        logger_name: str,
        message: str,
        correlation_id: str = "",
    ) -> None:
        service = logger_name.split(".")[0] if logger_name else "nexus"
        fingerprint = f"{service}:{level}:{message[:120]}"
        if not self._should_send(fingerprint):
            return
        emoji = "🚨" if level == "CRITICAL" else ("✅" if level == "INFO" else "⚠️")
        payload = {
            "text": f"{emoji} [{level}] {service}: {message[:200]}",
            "blocks": [
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": f"{emoji} [{level}] {service}"},
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Service:*\n{service}"},
                        {"type": "mrkdwn", "text": f"*Level:*\n{level}"},
                        {"type": "mrkdwn", "text": f"*Correlation ID:*\n{correlation_id or 'N/A'}"},
                        {"type": "mrkdwn", "text": f"*Logger:*\n{logger_name}"},
                    ],
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"```{message[:500]}```"},
                },
            ],
        }
        try:
            async with httpx.AsyncClient() as client:
                await client.post(self._url, json=payload, timeout=5.0)
        except Exception as exc:
            sys.stderr.write(f"[alerting] Slack POST failed: {exc}\n")


class SlackAlertHandler(logging.Handler):
    def __init__(self, alerter: SlackAlerter) -> None:
        super().__init__(level=logging.ERROR)
        self._alerter = alerter

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            cid = getattr(record, "correlation_id", "") or ""
            coro = self._alerter.send_alert(
                level=record.levelname,
                logger_name=record.name,
                message=msg,
                correlation_id=cid,
            )
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(coro)
            except RuntimeError:
                threading.Thread(target=asyncio.run, args=(coro,), daemon=True).start()
        except Exception:
            pass


def init_slack_alerting(webhook_url: str, rate_limit_seconds: int = 60) -> SlackAlertHandler:
    alerter = SlackAlerter(webhook_url=webhook_url, rate_limit_seconds=rate_limit_seconds)
    return SlackAlertHandler(alerter)
