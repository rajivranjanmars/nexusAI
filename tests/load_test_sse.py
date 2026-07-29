"""
SSE load test for the backend proxy.

Authenticates N concurrent virtual users via a configurable token endpoint,
then exercises a configurable chat-stream SSE endpoint with real chat
messages. Each virtual user gets its own JWT access token and a unique
``session_id``, simulating independent browser sessions.

The test also periodically polls a configurable health endpoint and can write
a JSON report to disk for later analysis.

Usage:
    # Defaults come from env vars when available.
    python -m tests.load_test_sse --app scripts/app_keypairs/lpude/app_credentials.json

    # Run against any externally reachable proxy URL.
    python -m tests.load_test_sse \
        --base-url https://proxy.example.com \
        --app scripts/app_keypairs/lpude/app_credentials.json \
        --clients 10 --duration 300 \
        --report-file reports/load_test.json

Requirements:
    pip install httpx
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urljoin
from typing import Any, Dict, List, Optional

import httpx

# ── Configuration ──────────────────────────────────────────────────────────

DEFAULT_BASE_URL = "http://127.0.0.1:5123"
DEFAULT_CLIENTS = 5
DEFAULT_DURATION_S = 120  # 2 minutes
DEFAULT_JWT_TTL_S = 600  # 10 minutes (long enough to cover the test)
DEFAULT_AUTH_ENDPOINT = "/api/auth/dev/app-user-token"
DEFAULT_CHAT_ENDPOINT = "/api/chat/message/stream"
DEFAULT_HEALTH_ENDPOINT = "/health/version"
DEFAULT_REPORT_FILE = "load_test_report.json"
DEFAULT_LOG_FILE = "load_test.log"
DEFAULT_CASES_FILE = ""

DEFAULT_MESSAGES = [
    "What programs does LPU offer?",
    "Tell me about admission requirements",
    "What is the fee structure MBA?",
    "Do you offer scholarships?",
    "How do I apply online?",
]


# ── Helpers ────────────────────────────────────────────────────────────────


def _load_manifest(path: str) -> Dict[str, Any]:
    """Load app credentials manifest from disk."""
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    required_keys = {"app_id", "private_key_path"}
    missing = sorted(required_keys - set(manifest))
    if missing:
        raise ValueError(f"Manifest {path} is missing keys: {', '.join(missing)}")
    return manifest


def _load_private_key(manifest: Dict[str, Any], manifest_path: str) -> str:
    """Resolve and read the private key PEM from the manifest."""
    key_path = Path(manifest["private_key_path"])
    if not key_path.is_absolute():
        key_path = (Path(manifest_path).parent / key_path).resolve()
    return key_path.read_text(encoding="utf-8")


@dataclass
class Stats:
    """Mutable stats container shared across tasks."""

    tokens_issued: int = 0
    token_failures: int = 0
    chat_requests_sent: int = 0
    chat_requests_ok: int = 0
    chat_requests_fail: int = 0
    sse_events_received: int = 0
    health_checks_ok: int = 0
    health_checks_fail: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    response_times_ms: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class BenchmarkCase:
    """One benchmark turn and its expected behavior."""

    case_id: str
    prompt: str
    expected_rag: Optional[bool] = None
    expected_keywords: List[str] = field(default_factory=list)
    forbidden_keywords: List[str] = field(default_factory=list)
    min_sources: int = 0
    description: str = ""


@dataclass
class ChatTurnResult:
    """One observed response captured during the load test."""

    client_id: int
    session_id: str
    case_id: str
    prompt: str
    response_text: str
    workflow: str = "general"
    cache_hit: bool = False
    rag_triggered: bool = False
    source_urls: List[str] = field(default_factory=list)
    source_count: int = 0
    response_time_ms: float = 0.0
    benchmark_score: float = 0.0
    benchmark_passed: bool = False
    notes: List[str] = field(default_factory=list)


@dataclass
class LoadTestConfig:
    """Resolved configuration for one load-test run."""

    base_url: str
    auth_endpoint: str
    chat_endpoint: str
    health_endpoint: str
    app_id: str
    clients: int
    duration_s: int
    actor_type: str
    force_workflow: Optional[str]
    jwt_ttl_seconds: int
    report_file: Optional[str]
    log_file: Optional[str]
    origin: str
    messages: List[str]
    benchmark_cases: List[BenchmarkCase]


@dataclass
class LoadTestReport:
    """Structured JSON report for one load-test run."""

    generated_at: str
    passed: bool
    verdict: str
    elapsed_seconds: float
    requests_per_second: float
    config: LoadTestConfig
    stats: Stats
    turns: List[ChatTurnResult]
    benchmark_summary: Dict[str, Any]


def _env_value(name: str, fallback: str) -> str:
    """Read an environment variable with a string fallback."""

    value = os.getenv(name)
    if value is None or not value.strip():
        return fallback
    return value.strip()


def _resolve_base_url(cli_value: Optional[str]) -> str:
    """Resolve the proxy base URL from CLI args or environment variables."""

    if cli_value:
        return cli_value.rstrip("/")

    env_url = os.getenv("LOAD_TEST_BASE_URL") or os.getenv("PROXY_URL")
    if env_url and env_url.strip():
        return env_url.strip().rstrip("/")

    host = _env_value("PROXY_HOST", "127.0.0.1")
    port = _env_value("PROXY_HOST_PORT", _env_value("PROXY_PORT", "5123"))
    return f"http://{host}:{port}"


def _resolve_endpoint(cli_value: Optional[str], env_name: str, default_value: str) -> str:
    """Resolve an endpoint path or full URL from CLI args or environment variables."""

    value = cli_value or os.getenv(env_name) or default_value
    return value.strip()


def _build_url(base_url: str, endpoint: str) -> str:
    """Join a base URL with an endpoint path or return the absolute URL unchanged."""

    if endpoint.startswith("http://") or endpoint.startswith("https://"):
        return endpoint
    return urljoin(f"{base_url.rstrip('/')}/", endpoint.lstrip("/"))


def _default_benchmark_cases(messages: List[str]) -> List[BenchmarkCase]:
    """Create a lightweight benchmark suite from the default message pool."""

    cases: List[BenchmarkCase] = []
    for index, prompt in enumerate(messages, start=1):
        lowered = prompt.lower()
        expected_rag = any(
            keyword in lowered
            for keyword in ("fee", "scholarship", "program", "admission", "apply")
        )
        cases.append(
            BenchmarkCase(
                case_id=f"case-{index:02d}",
                prompt=prompt,
                expected_rag=expected_rag,
                expected_keywords=[
                    keyword
                    for keyword in ("fee", "scholarship", "admission", "program", "apply")
                    if keyword in lowered
                ],
                min_sources=1 if expected_rag else 0,
                description="Auto-generated from the default message pool.",
            )
        )
    return cases


def _load_benchmark_cases(path: Optional[str], fallback_messages: List[str]) -> List[BenchmarkCase]:
    """Load benchmark cases from JSON, or fall back to generated cases."""

    if not path:
        return _default_benchmark_cases(fallback_messages)

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Benchmark cases file must contain a JSON list.")

    cases: List[BenchmarkCase] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError("Each benchmark case must be a JSON object.")
        prompt = str(item.get("prompt") or item.get("message") or "").strip()
        if not prompt:
            raise ValueError(f"Benchmark case {index} is missing prompt/message.")
        cases.append(
            BenchmarkCase(
                case_id=str(item.get("case_id") or item.get("id") or f"case-{index:02d}"),
                prompt=prompt,
                expected_rag=item.get("expected_rag"),
                expected_keywords=[str(value) for value in item.get("expected_keywords", []) if str(value).strip()],
                forbidden_keywords=[str(value) for value in item.get("forbidden_keywords", []) if str(value).strip()],
                min_sources=int(item.get("min_sources", 0)),
                description=str(item.get("description") or ""),
            )
        )
    return cases


def _extract_source_urls(response_text: str) -> List[str]:
    """Pull source URLs from an appended Sources block."""

    source_urls: List[str] = []
    in_sources = False
    for line in response_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower() == "sources:":
            in_sources = True
            continue
        if not in_sources:
            continue
        if stripped.startswith("[") and "]" in stripped:
            remainder = stripped.split("]", 1)[1].strip()
            if " - " in remainder:
                source_urls.append(remainder.rsplit(" - ", 1)[1].strip())
            else:
                source_urls.append(remainder)
    return [url for url in source_urls if url]


def _score_benchmark_case(case: BenchmarkCase, response_text: str, workflow: str, source_urls: List[str]) -> tuple[float, bool, List[str]]:
    """Score a single benchmark case against the observed response."""

    notes: List[str] = []
    score_parts: List[float] = []
    response_lower = response_text.lower()
    rag_triggered = bool(source_urls) or "sources:" in response_lower

    if case.expected_rag is not None:
        passed = rag_triggered == case.expected_rag
        score_parts.append(1.0 if passed else 0.0)
        if not passed:
            notes.append(f"rag_expected={case.expected_rag} observed={rag_triggered}")
    elif rag_triggered:
        score_parts.append(1.0)

    if case.min_sources:
        passed = len(source_urls) >= case.min_sources
        score_parts.append(1.0 if passed else 0.0)
        if not passed:
            notes.append(f"min_sources={case.min_sources} observed={len(source_urls)}")

    if case.expected_keywords:
        matches = sum(1 for keyword in case.expected_keywords if keyword.lower() in response_lower)
        keyword_score = matches / len(case.expected_keywords)
        score_parts.append(keyword_score)
        if matches != len(case.expected_keywords):
            missing = [keyword for keyword in case.expected_keywords if keyword.lower() not in response_lower]
            notes.append(f"missing_keywords={missing}")

    if case.forbidden_keywords:
        forbidden_hits = [keyword for keyword in case.forbidden_keywords if keyword.lower() in response_lower]
        score_parts.append(1.0 if not forbidden_hits else 0.0)
        if forbidden_hits:
            notes.append(f"forbidden_keywords={forbidden_hits}")

    if not score_parts:
        score_parts.append(1.0 if response_text.strip() else 0.0)

    score = round(sum(score_parts) / len(score_parts), 3)
    passed = score >= 0.75 and bool(response_text.strip())
    if not response_text.strip():
        notes.append("empty_response")
    if workflow == "lead_capture":
        notes.append("workflow=lead_capture")
    return score, passed, notes


# ── JWT token acquisition ──────────────────────────────────────────────────


async def _acquire_token(
    http: httpx.AsyncClient,
    base_url: str,
    auth_endpoint: str,
    app_id: str,
    private_key: str,
    actor_id: str,
    actor_type: str,
    origin: str,
    jwt_ttl_seconds: int,
    force_workflow: Optional[str],
) -> Optional[str]:
    """Call the dev token-mint endpoint and return an access_token, or None on failure."""
    payload: Dict[str, Any] = {
        "app_id": app_id,
        "private_key": private_key,
        "actor_id": actor_id,
        "actor_type": actor_type,
        "origin": origin,
        "jwt_ttl_seconds": jwt_ttl_seconds,
    }
    if force_workflow:
        payload["force_workflow"] = force_workflow

    headers: Dict[str, str] = {}
    if origin:
        headers["Origin"] = origin

    resp = await http.post(_build_url(base_url, auth_endpoint), json=payload, headers=headers)
    if resp.status_code != 200:
        return None

    data = resp.json()
    return data.get("access_token")


# ── SSE chat client task ──────────────────────────────────────────────────


async def chat_client_task(
    client_id: int,
    base_url: str,
    chat_endpoint: str,
    access_token: str,
    origin: str,
    session_id: str,
    cases: List[BenchmarkCase],
    duration: float,
    stats: Stats,
) -> List[ChatTurnResult]:
    """Send chat messages in a loop for *duration* seconds, consuming SSE events."""
    deadline = time.monotonic() + duration
    msg_index = 0
    turns: List[ChatTurnResult] = []

    headers: Dict[str, str] = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "text/event-stream",
    }
    if origin:
        headers["Origin"] = origin

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as http:
            while time.monotonic() < deadline:
                case = cases[msg_index % len(cases)]
                msg_index += 1
                stats.chat_requests_sent += 1

                try:
                    response_text = ""
                    workflow = "general"
                    cache_hit = False
                    event_name = ""
                    current_data_lines: List[str] = []
                    turn_start = time.monotonic()

                    async with http.stream(
                        "POST",
                        _build_url(base_url, chat_endpoint),
                        json={"message": case.prompt, "session_id": session_id},
                        headers=headers,
                    ) as resp:
                        if resp.status_code != 200:
                            stats.chat_requests_fail += 1
                            body = await resp.aread()
                            stats.errors.append(
                                f"client-{client_id}: HTTP {resp.status_code} — {body[:200]}"
                            )
                            continue

                        async for line in resp.aiter_lines():
                            stats.sse_events_received += 1
                            if line == "":
                                if current_data_lines:
                                    payload_text = "\n".join(current_data_lines)
                                    if event_name == "chat":
                                        try:
                                            payload = json.loads(payload_text)
                                            response_text += str(payload.get("text", ""))
                                        except Exception:
                                            response_text += payload_text
                                    elif event_name == "metadata":
                                        try:
                                            payload = json.loads(payload_text)
                                            workflow = str(payload.get("workflow") or workflow)
                                            cache_hit = bool(payload.get("cache_hit", False))
                                        except Exception:
                                            pass
                                    current_data_lines = []
                                    event_name = ""
                                continue

                            if line.startswith("event:"):
                                event_name = line.split(":", 1)[1].strip()
                                continue
                            if line.startswith("data:"):
                                current_data_lines.append(line.split(":", 1)[1].lstrip())
                                continue

                            if time.monotonic() >= deadline:
                                break

                    stats.chat_requests_ok += 1
                    turn_elapsed_ms = round((time.monotonic() - turn_start) * 1000, 2)
                    stats.response_times_ms.append(turn_elapsed_ms)
                    if cache_hit:
                        stats.cache_hits += 1
                    else:
                        stats.cache_misses += 1
                    source_urls = _extract_source_urls(response_text)
                    benchmark_score, benchmark_passed, notes = _score_benchmark_case(
                        case,
                        response_text,
                        workflow,
                        source_urls,
                    )
                    turns.append(
                        ChatTurnResult(
                            client_id=client_id,
                            session_id=session_id,
                            case_id=case.case_id,
                            prompt=case.prompt,
                            response_text=response_text,
                            workflow=workflow,
                            cache_hit=cache_hit,
                            rag_triggered=bool(source_urls) or "sources:" in response_text.lower(),
                            source_urls=source_urls,
                            source_count=len(source_urls),
                            response_time_ms=turn_elapsed_ms,
                            benchmark_score=benchmark_score,
                            benchmark_passed=benchmark_passed,
                            notes=notes,
                        )
                    )

                except httpx.ReadTimeout:
                    stats.chat_requests_fail += 1
                    stats.errors.append(f"client-{client_id}: read timeout on chat stream")
                except (httpx.ReadError, httpx.RemoteProtocolError) as exc:
                    stats.chat_requests_fail += 1
                    stats.errors.append(f"client-{client_id}: stream error — {exc}")

                # Small pause between messages to simulate realistic user behavior
                await asyncio.sleep(0.5)

    except Exception as exc:
        stats.errors.append(f"client-{client_id}: {type(exc).__name__}: {exc}")
        print(f"  [client-{client_id}] ✗ Fatal error: {exc}")

    return turns


# ── Periodic health checks ─────────────────────────────────────────────────


async def health_poller(
    base_url: str,
    health_endpoint: str,
    interval: float,
    duration: float,
    stats: Stats,
) -> None:
    """Hit ``/health/version`` every *interval* seconds for *duration*."""
    deadline = time.monotonic() + duration
    async with httpx.AsyncClient(timeout=httpx.Timeout(10)) as http:
        while time.monotonic() < deadline:
            try:
                r = await http.get(_build_url(base_url, health_endpoint))
                if r.status_code == 200:
                    stats.health_checks_ok += 1
                else:
                    stats.health_checks_fail += 1
            except Exception:
                stats.health_checks_fail += 1
            await asyncio.sleep(interval)


# ── Runner ─────────────────────────────────────────────────────────────────


async def run_load_test(
    config: LoadTestConfig,
    private_key: str,
    log_file: Optional[str] = None,
) -> tuple[Stats, float, float, List[ChatTurnResult]]:
    """Orchestrate the full load test: mint tokens → run chat clients → report."""
    stats = Stats()
    started_at = time.time()

    def log_print(msg: str = "") -> None:
        print(msg)
        if log_file:
            try:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(msg + "\n")
            except Exception as e:
                print(f"Failed to write to log file {log_file}: {e}")

    if log_file:
        try:
            with open(log_file, "w", encoding="utf-8") as f:
                f.write(f"--- Load Test Run Started at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        except Exception as e:
            print(f"Failed to initialize log file {log_file}: {e}")

    log_print(f"\n{'=' * 60}")
    log_print("  SSE Chat Load Test (JWT Auth)")
    log_print(f"  Server:         {config.base_url}")
    log_print(f"  Auth endpoint:  {config.auth_endpoint}")
    log_print(f"  Chat endpoint:   {config.chat_endpoint}")
    log_print(f"  Health endpoint: {config.health_endpoint}")
    log_print(f"  App ID:         {config.app_id}")
    log_print(f"  Clients:        {config.clients}")
    log_print(f"  Duration:       {config.duration_s}s")
    log_print(f"  Messages/pool:  {len(config.messages)}")
    log_print(f"  Benchmark cases: {len(config.benchmark_cases)}")
    log_print(f"  Force workflow: {config.force_workflow or '(auto)'}")
    log_print(f"{'=' * 60}\n")

    # ── Phase 1: Mint one JWT per virtual user ──────────────────────────
    log_print("  [setup] Minting JWT tokens for virtual users...")
    tokens: List[Dict[str, str]] = []

    async with httpx.AsyncClient(timeout=httpx.Timeout(30)) as http:
        for i in range(config.clients):
            actor_id = f"load-test-user-{i}-{uuid.uuid4().hex[:8]}"
            session_id = f"load-test-session-{i}-{uuid.uuid4().hex[:8]}"
            token = await _acquire_token(
                http=http,
                base_url=config.base_url,
                auth_endpoint=config.auth_endpoint,
                app_id=config.app_id,
                private_key=private_key,
                actor_id=actor_id,
                actor_type=config.actor_type,
                origin=config.origin,
                jwt_ttl_seconds=config.jwt_ttl_seconds,
                force_workflow=config.force_workflow,
            )
            if token:
                stats.tokens_issued += 1
                tokens.append({
                    "client_id": str(i),
                    "access_token": token,
                    "session_id": session_id,
                })
                log_print(f"    [OK] client-{i} token issued (actor={actor_id})")
            else:
                stats.token_failures += 1
                stats.errors.append(f"client-{i}: failed to acquire JWT")
                log_print(f"    [FAIL] client-{i} token FAILED")

    if not tokens:
        log_print("\n  [FAIL] No tokens acquired — aborting load test.")
        elapsed_seconds = max(time.time() - started_at, 0.001)
        return stats, elapsed_seconds, 0.0

    log_print(f"\n  [setup] {len(tokens)}/{config.clients} tokens acquired.\n")

    # ── Phase 2: Cache warm-up (seed cache with one pass of each prompt) ──
    log_print("  [warmup] Seeding cache with one pass of each unique prompt...")
    warmup_token = tokens[0]
    warmup_headers: Dict[str, str] = {
        "Authorization": f"Bearer {warmup_token['access_token']}",
        "Accept": "text/event-stream",
    }
    if config.origin:
        warmup_headers["Origin"] = config.origin

    seen_prompts: set[str] = set()
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as http:
        for case in config.benchmark_cases:
            if case.prompt in seen_prompts:
                continue
            seen_prompts.add(case.prompt)
            try:
                async with http.stream(
                    "POST",
                    _build_url(config.base_url, config.chat_endpoint),
                    json={
                        "message": case.prompt,
                        "session_id": f"warmup-{uuid.uuid4().hex[:8]}",
                    },
                    headers=warmup_headers,
                ) as resp:
                    async for _ in resp.aiter_lines():
                        pass
                log_print(f"    [OK] warmed: {case.prompt[:50]}")
            except Exception as exc:
                log_print(f"    [SKIP] warmup failed for '{case.prompt[:40]}': {exc}")
            # Pause to let server complete cache store (embed + write)
            await asyncio.sleep(2.0)

    log_print(f"  [warmup] Seeded {len(seen_prompts)} unique prompts.\n")

    # ── Phase 3: Run concurrent chat clients + health poller ────────────
    log_print("  [test] Starting concurrent chat clients...\n")

    tasks = [
        chat_client_task(
            client_id=int(t["client_id"]),
            base_url=config.base_url,
            chat_endpoint=config.chat_endpoint,
            access_token=t["access_token"],
            origin=config.origin,
            session_id=t["session_id"],
            cases=config.benchmark_cases,
            duration=config.duration_s,
            stats=stats,
        )
        for t in tokens
    ]
    tasks.append(health_poller(config.base_url, config.health_endpoint, 5.0, config.duration_s, stats))

    client_results = await asyncio.gather(*tasks, return_exceptions=True)
    turns: List[ChatTurnResult] = []
    for result in client_results:
        if isinstance(result, list):
            turns.extend(result)
    elapsed_seconds = max(time.time() - started_at, 0.001)
    requests_per_second = round(stats.chat_requests_ok / elapsed_seconds, 3)
    benchmark_passed = sum(1 for turn in turns if turn.benchmark_passed)
    benchmark_total = len(turns)
    benchmark_pass_rate = round(benchmark_passed / benchmark_total, 3) if benchmark_total else 0.0
    benchmark_score_avg = round(
        sum(turn.benchmark_score for turn in turns) / benchmark_total, 3
    ) if benchmark_total else 0.0
    rag_triggered_cases = [turn.case_id for turn in turns if turn.rag_triggered]
    cache_hit_turns = [turn for turn in turns if turn.cache_hit]
    cache_miss_turns = [turn for turn in turns if not turn.cache_hit]
    cache_hit_rate = round(len(cache_hit_turns) / benchmark_total, 3) if benchmark_total else 0.0
    latencies = [turn.response_time_ms for turn in turns if turn.response_time_ms > 0]
    cached_latencies = [turn.response_time_ms for turn in cache_hit_turns if turn.response_time_ms > 0]
    uncached_latencies = [turn.response_time_ms for turn in cache_miss_turns if turn.response_time_ms > 0]
    benchmark_summary = {
        "benchmark_total": benchmark_total,
        "benchmark_passed": benchmark_passed,
        "benchmark_failed": benchmark_total - benchmark_passed,
        "benchmark_pass_rate": benchmark_pass_rate,
        "benchmark_score_avg": benchmark_score_avg,
        "rag_triggered_total": len(rag_triggered_cases),
        "rag_triggered_case_ids": rag_triggered_cases,
        "unique_workflows": sorted({turn.workflow for turn in turns if turn.workflow}),
        "caching_summary": {
            "cache_hits": len(cache_hit_turns),
            "cache_misses": len(cache_miss_turns),
            "cache_hit_rate": cache_hit_rate,
            "cache_hit_case_ids": [turn.case_id for turn in cache_hit_turns],
        },
        "latency_summary": {
            "avg_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
            "min_ms": round(min(latencies), 2) if latencies else 0.0,
            "max_ms": round(max(latencies), 2) if latencies else 0.0,
            "p50_ms": round(sorted(latencies)[len(latencies) // 2], 2) if latencies else 0.0,
            "p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2) if latencies else 0.0,
            "p99_ms": round(sorted(latencies)[int(len(latencies) * 0.99)], 2) if latencies else 0.0,
            "cached_avg_ms": round(sum(cached_latencies) / len(cached_latencies), 2) if cached_latencies else 0.0,
            "uncached_avg_ms": round(sum(uncached_latencies) / len(uncached_latencies), 2) if uncached_latencies else 0.0,
        },
    }

    # ── Report ──────────────────────────────────────────────────────────
    caching = benchmark_summary["caching_summary"]
    latency = benchmark_summary["latency_summary"]

    log_print(f"\n{'=' * 60}")
    log_print("  RESULTS")
    log_print(f"{'=' * 60}")
    log_print(f"  Tokens issued ............... {stats.tokens_issued}")
    log_print(f"  Token failures .............. {stats.token_failures}")
    log_print(f"  Chat requests sent .......... {stats.chat_requests_sent}")
    log_print(f"  Chat requests OK ............ {stats.chat_requests_ok}")
    log_print(f"  Chat requests FAIL .......... {stats.chat_requests_fail}")
    log_print(f"  SSE events received ......... {stats.sse_events_received}")
    log_print(f"  Health checks OK ............ {stats.health_checks_ok}")
    log_print(f"  Health checks FAIL .......... {stats.health_checks_fail}")
    log_print(f"  Elapsed seconds ............. {elapsed_seconds:.2f}")
    log_print(f"  Chat RPS .................... {requests_per_second:.3f}")

    log_print(f"\n  -- Benchmarking --")
    log_print(f"  Benchmark turns ............. {benchmark_total}")
    log_print(f"  Benchmark pass rate ......... {benchmark_summary['benchmark_pass_rate']:.3f}")
    log_print(f"  Benchmark score avg ......... {benchmark_summary['benchmark_score_avg']:.3f}")
    log_print(f"  RAG-triggered turns ......... {benchmark_summary['rag_triggered_total']}")
    log_print(f"  Unique workflows ............ {', '.join(benchmark_summary['unique_workflows']) or '(none)'}")

    log_print(f"\n  -- Caching --")
    log_print(f"  Cache hits .................. {caching['cache_hits']}")
    log_print(f"  Cache misses ................ {caching['cache_misses']}")
    log_print(f"  Cache hit rate .............. {caching['cache_hit_rate']:.3f}")
    if caching['cache_hit_case_ids']:
        unique_cached = sorted(set(caching['cache_hit_case_ids']))
        log_print(f"  Cached case IDs ............. {', '.join(unique_cached)}")

    log_print(f"\n  -- Latency (ms) --")
    log_print(f"  Avg ......................... {latency['avg_ms']:.2f}")
    log_print(f"  Min ......................... {latency['min_ms']:.2f}")
    log_print(f"  Max ......................... {latency['max_ms']:.2f}")
    log_print(f"  P50 ......................... {latency['p50_ms']:.2f}")
    log_print(f"  P95 ......................... {latency['p95_ms']:.2f}")
    log_print(f"  P99 ......................... {latency['p99_ms']:.2f}")
    log_print(f"  Cached avg .................. {latency['cached_avg_ms']:.2f}")
    log_print(f"  Uncached avg ................ {latency['uncached_avg_ms']:.2f}")
    if latency['cached_avg_ms'] > 0 and latency['uncached_avg_ms'] > 0:
        speedup = round(latency['uncached_avg_ms'] / latency['cached_avg_ms'], 2)
        log_print(f"  Cache speedup ............... {speedup}x")

    log_print(f"\n  -- Benchmarking Parameters --")
    log_print(f"  Concurrent clients .......... {config.clients}")
    log_print(f"  Duration (s) ................ {config.duration_s}")
    log_print(f"  JWT TTL (s) ................. {config.jwt_ttl_seconds}")
    log_print(f"  Actor type .................. {config.actor_type}")
    log_print(f"  Force workflow .............. {config.force_workflow or '(auto)'}")
    log_print(f"  Message pool size ........... {len(config.messages)}")
    log_print(f"  Benchmark cases ............. {len(config.benchmark_cases)}")
    log_print(f"  Base URL .................... {config.base_url}")
    log_print(f"  Chat endpoint ............... {config.chat_endpoint}")

    if stats.errors:
        log_print(f"\n  ERRORS ({len(stats.errors)}):")
        for e in stats.errors[:20]:
            log_print(f"    - {e}")

    passed = (
        stats.token_failures == 0
        and stats.chat_requests_fail == 0
        and stats.health_checks_fail == 0
        and stats.tokens_issued == config.clients
        and (benchmark_pass_rate >= 0.75 if benchmark_total else True)
    )
    verdict = "PASS" if passed else "FAIL"
    log_print(f"\n  Verdict: {verdict}")
    log_print(f"{'=' * 60}\n")

    return stats, elapsed_seconds, requests_per_second, turns


def _build_report(
    config: LoadTestConfig,
    stats: Stats,
    elapsed_seconds: float,
    requests_per_second: float,
    turns: List[ChatTurnResult],
) -> LoadTestReport:
    """Build a structured report payload from config and runtime stats."""

    passed = (
        stats.token_failures == 0
        and stats.chat_requests_fail == 0
        and stats.health_checks_fail == 0
        and stats.tokens_issued == config.clients
    )
    benchmark_passed = sum(1 for turn in turns if turn.benchmark_passed)
    rag_triggered_turns = [turn for turn in turns if turn.rag_triggered]
    cache_hit_turns = [turn for turn in turns if turn.cache_hit]
    cache_miss_turns = [turn for turn in turns if not turn.cache_hit]
    benchmark_pass_rate = round(benchmark_passed / len(turns), 3) if turns else 0.0
    benchmark_score_avg = round(sum(turn.benchmark_score for turn in turns) / len(turns), 3) if turns else 0.0
    passed = passed and (benchmark_pass_rate >= 0.75 if turns else True)

    latencies = [turn.response_time_ms for turn in turns if turn.response_time_ms > 0]
    cached_latencies = [turn.response_time_ms for turn in cache_hit_turns if turn.response_time_ms > 0]
    uncached_latencies = [turn.response_time_ms for turn in cache_miss_turns if turn.response_time_ms > 0]
    cache_hit_rate = round(len(cache_hit_turns) / len(turns), 3) if turns else 0.0

    benchmark_summary = {
        "total_turns": len(turns),
        "passed_turns": benchmark_passed,
        "failed_turns": len(turns) - benchmark_passed,
        "pass_rate": benchmark_pass_rate,
        "score_avg": benchmark_score_avg,
        "rag_triggered_turns": len(rag_triggered_turns),
        "rag_triggered_cases": [turn.case_id for turn in rag_triggered_turns],
        "queries_with_rag": [
            {
                "case_id": turn.case_id,
                "prompt": turn.prompt,
                "workflow": turn.workflow,
                "source_count": turn.source_count,
            }
            for turn in rag_triggered_turns
        ],
        "caching_summary": {
            "cache_hits": len(cache_hit_turns),
            "cache_misses": len(cache_miss_turns),
            "cache_hit_rate": cache_hit_rate,
            "cache_hit_case_ids": sorted(set(turn.case_id for turn in cache_hit_turns)),
            "cache_triggered": len(cache_hit_turns) > 0,
            "queries_served_from_cache": [
                {
                    "case_id": turn.case_id,
                    "prompt": turn.prompt,
                    "client_id": turn.client_id,
                    "response_time_ms": turn.response_time_ms,
                }
                for turn in cache_hit_turns
            ],
        },
        "latency_summary": {
            "avg_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
            "min_ms": round(min(latencies), 2) if latencies else 0.0,
            "max_ms": round(max(latencies), 2) if latencies else 0.0,
            "p50_ms": round(sorted(latencies)[len(latencies) // 2], 2) if latencies else 0.0,
            "p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2) if latencies else 0.0,
            "p99_ms": round(sorted(latencies)[int(len(latencies) * 0.99)], 2) if latencies else 0.0,
            "cached_avg_ms": round(sum(cached_latencies) / len(cached_latencies), 2) if cached_latencies else 0.0,
            "uncached_avg_ms": round(sum(uncached_latencies) / len(uncached_latencies), 2) if uncached_latencies else 0.0,
            "cache_speedup_factor": round(
                (sum(uncached_latencies) / len(uncached_latencies))
                / (sum(cached_latencies) / len(cached_latencies)),
                2,
            ) if cached_latencies and uncached_latencies else None,
        },
        "benchmarking_parameters": {
            "concurrent_clients": config.clients,
            "duration_seconds": config.duration_s,
            "jwt_ttl_seconds": config.jwt_ttl_seconds,
            "actor_type": config.actor_type,
            "force_workflow": config.force_workflow,
            "message_pool_size": len(config.messages),
            "benchmark_case_count": len(config.benchmark_cases),
            "base_url": config.base_url,
            "chat_endpoint": config.chat_endpoint,
            "auth_endpoint": config.auth_endpoint,
            "health_endpoint": config.health_endpoint,
        },
    }
    return LoadTestReport(
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        passed=passed,
        verdict="PASS" if passed else "FAIL",
        elapsed_seconds=round(elapsed_seconds, 3),
        requests_per_second=round(requests_per_second, 3),
        config=config,
        stats=stats,
        turns=turns,
        benchmark_summary=benchmark_summary,
    )


def _write_report(report_path: str, report: LoadTestReport) -> None:
    """Write the structured report as JSON to disk."""

    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(report)
    payload["stats"]["errors"] = list(report.stats.errors)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


# ── CLI ────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SSE chat load test for the backend proxy (JWT auth)"
    )
    parser.add_argument(
        "--base-url",
        "--url",
        dest="base_url",
        default=None,
        help="Backend proxy base URL. Defaults to LOAD_TEST_BASE_URL, PROXY_URL, or localhost.",
    )
    parser.add_argument(
        "--auth-endpoint",
        default=None,
        help=f"Token-mint endpoint path or URL (default env LOAD_TEST_AUTH_ENDPOINT, else {DEFAULT_AUTH_ENDPOINT})",
    )
    parser.add_argument(
        "--chat-endpoint",
        default=None,
        help=f"Chat stream endpoint path or URL (default env LOAD_TEST_CHAT_ENDPOINT, else {DEFAULT_CHAT_ENDPOINT})",
    )
    parser.add_argument(
        "--health-endpoint",
        default=None,
        help=f"Health endpoint path or URL (default env LOAD_TEST_HEALTH_ENDPOINT, else {DEFAULT_HEALTH_ENDPOINT})",
    )
    parser.add_argument(
        "--app",
        default=os.getenv("LOAD_TEST_APP_CREDENTIALS", ""),
        help="Path to app_credentials.json (defaults to LOAD_TEST_APP_CREDENTIALS)",
    )
    parser.add_argument(
        "--clients",
        type=int,
        default=int(os.getenv("LOAD_TEST_CLIENTS", DEFAULT_CLIENTS)),
        help=f"Number of concurrent virtual users (default: {DEFAULT_CLIENTS})",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=int(os.getenv("LOAD_TEST_DURATION_S", DEFAULT_DURATION_S)),
        help=f"Test duration in seconds (default: {DEFAULT_DURATION_S})",
    )
    parser.add_argument(
        "--actor-type",
        default=os.getenv("LOAD_TEST_ACTOR_TYPE", "student"),
        choices=["student", "staff", "customer", "system"],
        help="Actor type for minted tokens (default: student)",
    )
    parser.add_argument(
        "--origin",
        default=os.getenv("LOAD_TEST_ORIGIN", ""),
        help="Origin header for domain verification. Defaults to manifest domain or env var.",
    )
    parser.add_argument(
        "--force-workflow",
        default=os.getenv("LOAD_TEST_FORCE_WORKFLOW"),
        help="Optional workflow lock for all sessions (e.g. 'lead_capture')",
    )
    parser.add_argument(
        "--jwt-ttl",
        type=int,
        default=int(os.getenv("LOAD_TEST_JWT_TTL_S", DEFAULT_JWT_TTL_S)),
        help=f"JWT TTL in seconds for the dev token endpoint (default: {DEFAULT_JWT_TTL_S})",
    )
    parser.add_argument(
        "--messages",
        nargs="+",
        default=None,
        help="Custom chat messages to cycle through. Defaults to built-in set.",
    )
    parser.add_argument(
        "--benchmark-cases-file",
        default=os.getenv("LOAD_TEST_BENCHMARK_CASES_FILE", DEFAULT_CASES_FILE),
        help="Optional JSON file containing benchmark cases. Defaults to auto-generated cases.",
    )
    parser.add_argument(
        "--report-file",
        default=os.getenv("LOAD_TEST_REPORT_FILE", DEFAULT_REPORT_FILE),
        help=f"Path to save the JSON report (default: {DEFAULT_REPORT_FILE})",
    )
    parser.add_argument(
        "--output-log",
        default=os.getenv("LOAD_TEST_LOG_FILE", DEFAULT_LOG_FILE),
        help=f"Path to save the human-readable log (default: {DEFAULT_LOG_FILE})",
    )
    args = parser.parse_args()

    if not args.app:
        raise SystemExit("Missing app credentials path. Pass --app or set LOAD_TEST_APP_CREDENTIALS.")

    # ── Load app credentials ────────────────────────────────────────────
    manifest = _load_manifest(args.app)
    private_key = _load_private_key(manifest, args.app)
    app_id = manifest["app_id"]

    base_url = _resolve_base_url(args.base_url)
    auth_endpoint = _resolve_endpoint(args.auth_endpoint, "LOAD_TEST_AUTH_ENDPOINT", DEFAULT_AUTH_ENDPOINT)
    chat_endpoint = _resolve_endpoint(args.chat_endpoint, "LOAD_TEST_CHAT_ENDPOINT", DEFAULT_CHAT_ENDPOINT)
    health_endpoint = _resolve_endpoint(
        args.health_endpoint,
        "LOAD_TEST_HEALTH_ENDPOINT",
        DEFAULT_HEALTH_ENDPOINT,
    )
    origin = args.origin or manifest.get("domain", "")
    messages = args.messages or DEFAULT_MESSAGES
    benchmark_cases = _load_benchmark_cases(args.benchmark_cases_file or None, messages)
    config = LoadTestConfig(
        base_url=base_url,
        auth_endpoint=auth_endpoint,
        chat_endpoint=chat_endpoint,
        health_endpoint=health_endpoint,
        app_id=app_id,
        clients=args.clients,
        duration_s=args.duration,
        actor_type=args.actor_type,
        force_workflow=args.force_workflow,
        jwt_ttl_seconds=args.jwt_ttl,
        report_file=args.report_file,
        log_file=args.output_log,
        origin=origin,
        messages=messages,
        benchmark_cases=benchmark_cases,
    )

    stats, elapsed_seconds, requests_per_second, turns = asyncio.run(
        run_load_test(config=config, private_key=private_key, log_file=args.output_log)
    )

    report = _build_report(config, stats, elapsed_seconds, requests_per_second, turns)
    if config.report_file:
        _write_report(config.report_file, report)

    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
