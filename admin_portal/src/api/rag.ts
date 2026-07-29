import { apiFetch } from "./client"

export interface CrawlRequest {
  app_id: string
  seed_url: string
  max_depth?: number
  allowed_paths?: string[]
  auto_discover_sitemap?: boolean
}

export interface IngestRequest {
  app_id: string
  source_type: "file" | "url" | "structured"
  content: string
  source_ref: string
}

export interface CrawlJobStatus {
  job_id: string
  app_id: string
  seed_url: string
  status: string
  progress_pct: number | null
  elapsed_seconds: number
  pages_per_minute: number | null
  is_stuck: boolean
  coverage_pct: number | null
  urls_queued: number
  urls_processed: number
  urls_skipped_quality: number
  urls_skipped_duplicate: number
  urls_failed: number
  expected_urls: number | null
  expected_urls_from_sitemap: number | null
  pages_found: number
  pages_skipped: number
  pages_failed: number
  chunks_upserted: number
  page_status_counts: Record<string, Record<string, number>>
  section_summary: { section: string; total_urls: number; extracted_urls: number; ingested_urls: number; total_chunks: number }[]
  started_at: string | null
  finished_at: string | null
  last_activity_at: string | null
  error: string | null
}

export function startCrawl(data: CrawlRequest) {
  return apiFetch<{ job_id: string; status: string; auto_discover_sitemap: boolean }>("/rag/crawl", {
    method: "POST",
    body: JSON.stringify(data),
  })
}

export function ingestKnowledge(data: IngestRequest) {
  return apiFetch<{ status: string; chunks_upserted: number; shared_cache_entries_invalidated: number }>("/rag/ingest", {
    method: "POST",
    body: JSON.stringify(data),
  })
}

export function deleteAppKnowledge(appId: string) {
  return apiFetch<{ status: string; app_id: string; chunks_deleted: number; parent_chunks_deleted: number }>(`/rag/app/${appId}`, {
    method: "DELETE",
  })
}

export function getCrawlStatus(jobId: string) {
  return apiFetch<CrawlJobStatus>(`/rag/status/${jobId}`)
}

export function getLatestCrawlStatus(appId: string) {
  return apiFetch<CrawlJobStatus>(`/rag/status/app/${appId}/latest`)
}