"""
Async BFS Web Crawler with quality filtering and observability.

Crawls a seed domain up to a maximum depth, respects a per-page fetch
timeout, filters low-signal pages, and emits structured progress logs
so the caller can detect stuck jobs.
"""
import asyncio
import hashlib
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, List, Optional, Set
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup, Comment, Tag
import trafilatura

from shared.logger import get_logger

logger = get_logger(__name__)

# ── Quality thresholds ──────────────────────────────────────────────────────
_MIN_CONTENT_CHARS = 150        # pages shorter than this are noise
_MAX_CONTENT_CHARS = 50_000     # cap runaway pages to avoid huge chunks
_FETCH_TIMEOUT_S   = 15.0       # per-page fetch timeout
_PROGRAM_DETAIL_CONTEXT_LINES = 2

# URL path segments that almost never contain useful knowledge
_BLOCKLIST_PATH_SEGMENTS = {
    "login", "logout", "signin", "signup", "register",
    "cart", "checkout", "wishlist",
    "search", "query",
    "404", "403", "500", "error",
    "print", "rss", "feed", "sitemap",
    "tag", "tags", "author",
    "wp-admin", "wp-login", "wp-json",
    "cdn-cgi", "static", "assets",
}

# Query-string patterns that indicate dynamic / low-value URLs
_BLOCKLIST_QUERY_PATTERNS = re.compile(
    r"(session|token|sid|PHPSESSID|jsessionid|utm_|ref=|redirect)", re.IGNORECASE
)
_SITEWIDE_NOISE_PATTERNS = (
    re.compile(r"public notice recognitions online education deb-id contact us", re.IGNORECASE),
    re.compile(r"admission lpu e-connect login contact us faqs blog study material", re.IGNORECASE),
    re.compile(r"call back whatsapp announcements view previous announcements", re.IGNORECASE),
    re.compile(r"download mobile app public notice", re.IGNORECASE),
    re.compile(r"the university does not possess a study center/examination center", re.IGNORECASE),
    re.compile(r"online admission process\s+offline admission process", re.IGNORECASE),
)
_PROGRAM_PAGE_SIGNAL_PATTERNS = (
    re.compile(r"\beligibility\b", re.IGNORECASE),
    re.compile(r"\bfee details?\b", re.IGNORECASE),
    re.compile(r"\bprogramme duration\b|\bprogram duration\b|\bduration\b", re.IGNORECASE),
    re.compile(r"\bspeciali[sz]ation", re.IGNORECASE),
    re.compile(r"\bobjectives?\b", re.IGNORECASE),
    re.compile(r"\bhow to apply\b", re.IGNORECASE),
    re.compile(r"\bpayment mode\b", re.IGNORECASE),
)


def _normalize_table_cell_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _extract_table_text(table: Tag) -> str:
    """Render HTML tables as readable row-oriented text for chunking."""
    headers: list[str] = []
    header_row = table.find("tr")
    if header_row:
        headers = [
            _normalize_table_cell_text(cell.get_text(" ", strip=True))
            for cell in header_row.find_all(["th", "td"])
        ]

    lines: list[str] = []
    if headers and any(headers):
        lines.append("Table columns: " + " | ".join(h or f"column_{idx + 1}" for idx, h in enumerate(headers)))

    row_index = 0
    for row in table.find_all("tr"):
        cells = [
            _normalize_table_cell_text(cell.get_text(" ", strip=True))
            for cell in row.find_all(["th", "td"])
        ]
        if not cells or not any(cells):
            continue

        row_index += 1
        if headers and len(cells) == len(headers) and cells == headers:
            continue

        if headers and len(headers) == len(cells):
            rendered = " | ".join(
                f"{header or f'column_{idx + 1}'}: {value}"
                for idx, (header, value) in enumerate(zip(headers, cells))
                if value
            )
        else:
            rendered = " | ".join(value for value in cells if value)

        if rendered:
            lines.append(f"Row {row_index}: {rendered}")

    return "\n".join(lines).strip()


def _extract_comment_table_text(soup: BeautifulSoup) -> str:
    """Parse HTML comments that contain table markup used by some CMS pages."""
    extracted_tables: list[str] = []
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        raw = str(comment)
        if "<table" not in raw.lower():
            continue
        comment_soup = BeautifulSoup(raw, "html.parser")
        for table in comment_soup.find_all("table"):
            table_text = _extract_table_text(table)
            if table_text:
                extracted_tables.append(table_text)
    return "\n\n".join(extracted_tables).strip()


def _extract_fallback_content(soup: BeautifulSoup) -> str:
    """Fallback extractor that preserves table semantics better than get_text()."""
    working_soup = BeautifulSoup(str(soup), "html.parser")
    for tag in working_soup(["nav", "header", "footer", "script", "style", "aside",
                             "form", "button", "noscript", "iframe"]):
        tag.decompose()

    table_blocks: list[str] = []
    for table in working_soup.find_all("table"):
        table_text = _extract_table_text(table)
        if table_text:
            table_blocks.append(table_text)
        table.decompose()

    body_text = working_soup.get_text(separator="\n", strip=True)
    comment_table_text = _extract_comment_table_text(soup)

    content_parts = [part for part in [body_text, "\n\n".join(table_blocks), comment_table_text] if part]
    return "\n\n".join(content_parts).strip()


def _is_noise_line(line: str) -> bool:
    normalized = re.sub(r"\s+", " ", line or "").strip()
    if not normalized:
        return True
    return any(pattern.search(normalized) for pattern in _SITEWIDE_NOISE_PATTERNS)


def _dedupe_preserve_order(lines: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for line in lines:
        normalized = re.sub(r"\s+", " ", line).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _extract_programme_focus(lines: list[str]) -> list[str]:
    """Pull detail-heavy windows from programme pages to improve RAG quality."""
    selected_indices: set[int] = set()
    for index, line in enumerate(lines):
        if not any(pattern.search(line) for pattern in _PROGRAM_PAGE_SIGNAL_PATTERNS):
            continue
        start = max(0, index - _PROGRAM_DETAIL_CONTEXT_LINES)
        end = min(len(lines), index + _PROGRAM_DETAIL_CONTEXT_LINES + 1)
        selected_indices.update(range(start, end))

    if not selected_indices:
        return []

    return [lines[index] for index in sorted(selected_indices)]


def _clean_extracted_content(url: str, content: str) -> str:
    """Remove sitewide boilerplate and prioritize detail-rich programme sections."""
    raw_lines = [
        re.sub(r"\s+", " ", line).strip()
        for line in re.split(r"\n+", content or "")
    ]
    lines = [line for line in raw_lines if line and not _is_noise_line(line)]
    lines = _dedupe_preserve_order(lines)

    if "/programmes/" in (url or "").lower():
        focused_lines = _extract_programme_focus(lines)
        if focused_lines:
            merged = _dedupe_preserve_order(focused_lines + lines)
            return "\n".join(merged).strip()

    return "\n".join(lines).strip()


@dataclass
class CrawledPage:
    url: str
    title: str
    content: str
    content_hash: str
    depth: int
    crawled_at: datetime
    fetch_ms: int = 0           # how long the HTTP fetch took


@dataclass
class CrawlAuditPage:
    """Per-URL crawl audit snapshot used for admin observability."""

    url: str
    section: str
    discovered_via: str
    depth: Optional[int] = None
    crawl_status: str = "discovered"
    ingestion_status: str = "pending"
    http_status: Optional[int] = None
    fetch_ms: Optional[int] = None
    content_chars: Optional[int] = None
    title: Optional[str] = None
    content_hash: Optional[str] = None
    error: Optional[str] = None
    crawled_at: Optional[datetime] = None
    ingested_at: Optional[datetime] = None
    chunk_count: int = 0

    def to_dict(self) -> dict:
        """Convert CrawlAuditPage to dictionary representation."""
        from datetime import datetime
        payload = {}
        for field_name in [
            "url", "section", "discovered_via", "depth", "crawl_status",
            "ingestion_status", "http_status", "fetch_ms", "content_chars",
            "title", "content_hash", "error", "crawled_at", "ingested_at",
            "chunk_count"
        ]:
            value = getattr(self, field_name)
            if isinstance(value, datetime):
                value = value.isoformat() if value else None
            payload[field_name] = value
        return payload


@dataclass
class CrawlStats:
    """Live counters updated throughout the crawl."""
    urls_queued: int = 0
    urls_processed: int = 0
    urls_skipped_quality: int = 0
    urls_skipped_duplicate: int = 0
    urls_failed: int = 0
    pages_extracted: int = 0
    expected_urls: int = 0
    expected_urls_from_sitemap: int = 0
    started_at: float = field(default_factory=time.monotonic)

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def pages_per_minute(self) -> float:
        elapsed = self.elapsed_s
        if elapsed < 1:
            return 0.0
        return round(self.pages_extracted / elapsed * 60, 1)

    def to_dict(self) -> dict:
        coverage_pct = None
        if self.expected_urls > 0:
            coverage_pct = round(min(self.urls_processed / self.expected_urls * 100, 100), 1)
        return {
            "urls_queued": self.urls_queued,
            "urls_processed": self.urls_processed,
            "urls_skipped_quality": self.urls_skipped_quality,
            "urls_skipped_duplicate": self.urls_skipped_duplicate,
            "urls_failed": self.urls_failed,
            "pages_extracted": self.pages_extracted,
            "expected_urls": self.expected_urls,
            "expected_urls_from_sitemap": self.expected_urls_from_sitemap,
            "coverage_pct": coverage_pct,
            "elapsed_s": round(self.elapsed_s, 1),
            "pages_per_minute": self.pages_per_minute,
        }


class AsyncCrawler:
    """BFS Web Crawler with quality filtering and progress observability."""

    def __init__(
        self,
        seed_url: str,
        max_depth: int = 2,
        allowed_paths: Optional[List[str]] = None,
        auto_discover_sitemap: bool = True,
        crawl_delay_ms: int = 500,
        max_concurrency: int = 5,
        correlation_id: Optional[str] = None,
        on_progress: Optional[Callable[[CrawlStats], None]] = None,
    ):
        self.seed_url = seed_url
        self.max_depth = max_depth
        self.allowed_paths = allowed_paths or []
        self.auto_discover_sitemap = auto_discover_sitemap
        self.crawl_delay_ms = crawl_delay_ms
        self.max_concurrency = max_concurrency
        self.correlation_id = correlation_id or "local"
        self.on_progress = on_progress  # optional async callback

        parsed_seed = urlparse(seed_url)
        self.base_domain = parsed_seed.netloc
        self.scheme = parsed_seed.scheme

        self.visited_urls: Set[str] = set()
        self.crawled_pages: List[CrawledPage] = []
        self.page_audits: dict[str, CrawlAuditPage] = {}
        self.expected_urls: Set[str] = set()
        self.stats = CrawlStats()

        self._semaphore = asyncio.Semaphore(self.max_concurrency)
        self._client: Optional[httpx.AsyncClient] = None

    # ── URL filtering ─────────────────────────────────────────────────────────

    def _is_allowed_url(self, url: str) -> bool:
        """Check if a URL passes all crawl rules."""
        parsed = urlparse(url)

        # 1. Same domain only
        if parsed.netloc != self.base_domain:
            return False

        path_lower = parsed.path.lower()

        # 2. Skip binary / static assets
        if any(path_lower.endswith(ext) for ext in [
            ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".css", ".js",
            ".zip", ".rar", ".exe", ".mp4", ".mp3", ".svg", ".ico",
            ".woff", ".woff2", ".ttf", ".eot", ".webp", ".xml",
        ]):
            return False

        # 3. Skip low-value path segments
        path_segments = set(path_lower.strip("/").split("/"))
        if path_segments & _BLOCKLIST_PATH_SEGMENTS:
            return False

        # 4. Skip URLs with session/tracking query params
        if parsed.query and _BLOCKLIST_QUERY_PATTERNS.search(parsed.query):
            return False

        # 5. Respect allowed_paths prefix whitelist
        if self.allowed_paths:
            if not any(parsed.path.startswith(p) for p in self.allowed_paths):
                return False

        return True

    def _clean_url(self, url: str) -> str:
        """Strip fragments and trailing slashes; normalise scheme."""
        parsed = urlparse(url)
        path = parsed.path.rstrip("/") or "/"
        return f"{parsed.scheme}://{parsed.netloc}{path}"

    def _section_for_url(self, url: str) -> str:
        parsed = urlparse(url)
        segments = [segment for segment in parsed.path.split("/") if segment]
        return segments[0] if segments else "/"

    def _ensure_audit_page(
        self,
        url: str,
        *,
        discovered_via: str,
        depth: Optional[int] = None,
    ) -> CrawlAuditPage:
        page = self.page_audits.get(url)
        if page is None:
            page = CrawlAuditPage(
                url=url,
                section=self._section_for_url(url),
                discovered_via=discovered_via,
                depth=depth,
            )
            self.page_audits[url] = page
        else:
            if page.discovered_via != discovered_via and page.discovered_via != "crawl+sitemap":
                page.discovered_via = "crawl+sitemap"
            if depth is not None and (page.depth is None or depth < page.depth):
                page.depth = depth
        return page

    async def _discover_sitemap_urls(self) -> list[str]:
        """Best-effort sitemap discovery used to estimate coverage automatically."""
        if not self._client:
            raise RuntimeError("HTTP client not initialised")

        seeds = [
            f"{self.scheme}://{self.base_domain}/robots.txt",
            f"{self.scheme}://{self.base_domain}/sitemap.xml",
            f"{self.scheme}://{self.base_domain}/sitemap_index.xml",
        ]
        sitemap_urls: Set[str] = set()
        seen_sitemaps: Set[str] = set()

        async def fetch_sitemap(target: str) -> None:
            if target in seen_sitemaps:
                return
            seen_sitemaps.add(target)
            try:
                response = await self._client.get(target, follow_redirects=True)
                response.raise_for_status()
            except Exception:
                return

            body = response.text
            if target.endswith("robots.txt"):
                for line in body.splitlines():
                    if line.lower().startswith("sitemap:"):
                        await fetch_sitemap(line.split(":", 1)[1].strip())
                return

            try:
                root = ET.fromstring(body)
            except ET.ParseError:
                return

            for loc in root.findall(".//{*}loc"):
                raw_url = (loc.text or "").strip()
                if not raw_url:
                    continue
                clean = self._clean_url(raw_url)
                if clean.endswith(".xml"):
                    await fetch_sitemap(clean)
                    continue
                if self._is_allowed_url(clean):
                    sitemap_urls.add(clean)

        for seed in seeds:
            await fetch_sitemap(seed)

        return sorted(sitemap_urls)

    # ── Fetch + extract ───────────────────────────────────────────────────────

    async def _fetch_and_extract(
        self, url: str, depth: int
    ) -> tuple[Optional[CrawledPage], List[str], Optional[int]]:
        """Fetch a page, extract text, collect outlinks. Returns (page, links)."""
        if not self._client:
            raise RuntimeError("HTTP client not initialised")

        audit = self._ensure_audit_page(url, discovered_via="crawl", depth=depth)
        t0 = time.monotonic()
        try:
            response = await asyncio.wait_for(
                self._client.get(url, follow_redirects=True),
                timeout=_FETCH_TIMEOUT_S,
            )
            response.raise_for_status()
            html = response.text
        except asyncio.TimeoutError:
            logger.warning(
                "Fetch timeout after %.0fs: %s", _FETCH_TIMEOUT_S, url,
                extra={"correlation_id": self.correlation_id},
            )
            self.stats.urls_failed += 1
            audit.crawl_status = "failed"
            audit.error = f"timeout after {_FETCH_TIMEOUT_S:.0f}s"
            return None, [], None
        except Exception as exc:
            logger.warning(
                "Fetch failed: %s — %s", url, exc,
                extra={"correlation_id": self.correlation_id},
            )
            self.stats.urls_failed += 1
            audit.crawl_status = "failed"
            audit.error = str(exc)
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            return None, [], status_code

        fetch_ms = int((time.monotonic() - t0) * 1000)
        audit.http_status = response.status_code
        audit.fetch_ms = fetch_ms
        audit.crawled_at = datetime.utcnow()

        # ── Collect outlinks ──────────────────────────────────────────────────
        soup = BeautifulSoup(html, "html.parser")
        outlinks: List[str] = []
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            if href.startswith(("mailto:", "tel:", "#", "javascript:")):
                continue
            absolute = urljoin(url, href)
            clean = self._clean_url(absolute)
            if not self._is_allowed_url(clean):
                continue
            if clean in self.visited_urls:
                self.stats.urls_skipped_duplicate += 1
                self._ensure_audit_page(clean, discovered_via="crawl")
                continue
            outlinks.append(clean)

        # ── Extract main content ──────────────────────────────────────────────
        content = await asyncio.to_thread(
            trafilatura.extract,
            html,
            include_links=False,
            include_images=False,
            favor_recall=True,
        )
        fallback_content = _extract_fallback_content(soup)

        if not content:
            content = fallback_content
        elif fallback_content:
            content = f"{content.strip()}\n\n{fallback_content}"

        if not content:
            content = ""

        content = _clean_extracted_content(url, content)

        # ── Quality gate ──────────────────────────────────────────────────────
        if len(content) < _MIN_CONTENT_CHARS:
            logger.debug(
                "Skipped (too short, %d chars): %s", len(content), url,
                extra={"correlation_id": self.correlation_id},
            )
            self.stats.urls_skipped_quality += 1
            audit.crawl_status = "skipped_quality"
            audit.content_chars = len(content)
            return None, outlinks, response.status_code

        # Cap runaway pages
        if len(content) > _MAX_CONTENT_CHARS:
            logger.debug(
                "Truncated content at %d chars: %s", _MAX_CONTENT_CHARS, url,
                extra={"correlation_id": self.correlation_id},
            )
            content = content[:_MAX_CONTENT_CHARS]

        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        title = (
            soup.title.string.strip()
            if soup.title and soup.title.string
            else url
        )

        page = CrawledPage(
            url=url,
            title=title,
            content=content,
            content_hash=content_hash,
            depth=depth,
            crawled_at=datetime.utcnow(),
            fetch_ms=fetch_ms,
        )
        audit.crawl_status = "extracted"
        audit.title = title[:512]
        audit.content_hash = content_hash
        audit.content_chars = len(content)
        return page, outlinks, response.status_code

    # ── Worker ────────────────────────────────────────────────────────────────

    async def _crawl_worker(self, queue: asyncio.Queue) -> None:
        """BFS worker: pop URLs, fetch, enqueue outlinks."""
        while True:
            try:
                url, depth = queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            if depth > self.max_depth:
                queue.task_done()
                self.stats.urls_processed += 1
                continue

            async with self._semaphore:
                await asyncio.sleep(self.crawl_delay_ms / 1000.0)

                logger.info(
                    "Crawling [depth=%d] %s", depth, url,
                    extra={"correlation_id": self.correlation_id},
                )
                page, outlinks, http_status = await self._fetch_and_extract(url, depth)
                audit = self._ensure_audit_page(url, discovered_via="crawl", depth=depth)
                if http_status is not None:
                    audit.http_status = http_status

                if page:
                    self.crawled_pages.append(page)
                    self.stats.pages_extracted += 1
                    logger.info(
                        "Extracted %d chars in %dms: %s",
                        len(page.content), page.fetch_ms, url,
                        extra={"correlation_id": self.correlation_id},
                    )

                for link in outlinks:
                    if link not in self.visited_urls:
                        self.visited_urls.add(link)
                        self._ensure_audit_page(link, discovered_via="crawl", depth=depth + 1)
                        queue.put_nowait((link, depth + 1))
                        self.stats.urls_queued += 1

                self.stats.urls_processed += 1

                # Emit a progress snapshot every 10 pages
                if self.on_progress and self.stats.urls_processed % 10 == 0:
                    await asyncio.to_thread(self.on_progress, self.stats)

            queue.task_done()

    # ── Public entry point ────────────────────────────────────────────────────

    async def run(self) -> List[CrawledPage]:
        """Execute the BFS crawl and return all extracted pages."""
        clean_seed = self._clean_url(self.seed_url)
        self.visited_urls.add(clean_seed)
        self.stats.urls_queued = 1

        queue: asyncio.Queue = asyncio.Queue()
        queue.put_nowait((clean_seed, 0))

        headers = {"User-Agent": "LPUAI-Bot/1.0 (+https://lpu.in)"}

        async with httpx.AsyncClient(headers=headers, timeout=_FETCH_TIMEOUT_S) as client:
            self._client = client

            sitemap_urls = await self._discover_sitemap_urls() if self.auto_discover_sitemap else []
            self.expected_urls = set(sitemap_urls)
            self.stats.expected_urls = len(self.expected_urls)
            self.stats.expected_urls_from_sitemap = len(sitemap_urls)

            for sitemap_url in sitemap_urls:
                self._ensure_audit_page(sitemap_url, discovered_via="sitemap", depth=0)
                if sitemap_url not in self.visited_urls:
                    self.visited_urls.add(sitemap_url)
                    queue.put_nowait((sitemap_url, 0))
                    self.stats.urls_queued += 1
                else:
                    self.stats.urls_skipped_duplicate += 1

            workers = [
                asyncio.create_task(self._crawl_worker(queue))
                for _ in range(self.max_concurrency)
            ]

            await queue.join()
            if self.stats.expected_urls == 0:
                self.stats.expected_urls = self.stats.urls_queued

            for w in workers:
                w.cancel()

        logger.info(
            "Crawl finished — %s",
            self.stats.to_dict(),
            extra={"correlation_id": self.correlation_id},
        )
        return self.crawled_pages
