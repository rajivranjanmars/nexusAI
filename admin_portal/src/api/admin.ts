import { apiFetch } from "./client"

// ── Types ────────────────────────────────────────────────────────────────

export interface AppSummary {
  app_id: string
  name: string
  domain: string
  allowed_workflows: string[]
  allowed_tools: string[]
  rate_limit_rpm: number
  cache_ttl_seconds: number
  is_active: boolean
  created_at: string | null
  updated_at: string | null
}

export interface AppDetail extends AppSummary {
  public_key: string
  llm_model_override: string | null
  token_quota_monthly: number | null
  app_config: Record<string, unknown> | null
  metadata: Record<string, unknown> | null
}

export interface AppUpdate {
  name?: string
  domain?: string
  allowed_workflows?: string[]
  allowed_tools?: string[]
  rate_limit_rpm?: number
  cache_ttl_seconds?: number
  token_quota_monthly?: number | null
  llm_model_override?: string | null
  app_config?: Record<string, unknown> | null
  metadata?: Record<string, unknown> | null
  is_active?: boolean
}

export interface AdminUserSummary {
  id: string
  email: string
  display_name: string
  role: string
  app_id: string | null
  is_active: boolean
  last_login_at: string | null
  created_at: string | null
}

export interface AdminUserCreate {
  email: string
  password: string
  display_name: string
  role: string
  app_id?: string | null
}

export interface AdminUserUpdate {
  display_name?: string
  role?: string
  app_id?: string | null
  is_active?: boolean
}

export interface AuditLogEntry {
  id: number
  admin_user_id: string | null
  action: string
  target_type: string
  target_id: string | null
  details: Record<string, unknown> | null
  ip_address: string | null
  created_at: string | null
}

export interface AuditLogResponse {
  total: number
  limit: number
  offset: number
  entries: AuditLogEntry[]
}

export interface TokenSummary {
  period: string
  total_calls: number
  total_prompt_tokens: number
  total_completion_tokens: number
  total_tokens: number
  total_cost_usd: number
  avg_latency_ms: number
  by_model: { model: string; calls: number; tokens: number; cost_usd: number }[]
  by_action: { action: string; calls: number; tokens: number }[]
}

export interface TokenDailyItem {
  date: string
  calls: number
  tokens: number
  cost_usd: number
}

export interface TokenDetail {
  id: string
  timestamp: string | null
  app_id: string | null
  actor_id: string | null
  actor_type: string | null
  action: string | null
  model: string | null
  model_tier: string | null
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
  estimated_cost_usd: number | null
  cache_hit: boolean
  latency_ms: number | null
  correlation_id: string | null
}

export interface TokenRecentResponse {
  total: number
  limit: number
  offset: number
  items: TokenDetail[]
}

export interface RagStats {
  total_chunks: number
  total_apps: number
  apps: { app_id: string; chunk_count: number; source_types: Record<string, number>; source_count: number }[]
}

export interface CacheStats {
  total_entries: number
  active_entries: number
  total_hits: number
  top_cached: Record<string, unknown>[]
}

export interface LlmHealth {
  status: string
  model: string
  latency_ms: number
  error: string | null
}

// ── Apps ──────────────────────────────────────────────────────────────────

export interface AppBootstrapRequest {
  name: string
  domain?: string
  allowed_workflows?: string[]
  allowed_tools?: string[]
}

export interface AppBootstrapResponse {
  app_id: string
  name: string
  domain: string
  public_key: string
  private_key: string
  allowed_workflows: string[]
  allowed_tools: string[]
  proxy_url: string
}

export function listApps() {
  return apiFetch<AppSummary[]>("/admin/apps")
}

export function bootstrapApp(data: AppBootstrapRequest) {
  return apiFetch<AppBootstrapResponse>("/apps/bootstrap", {
    method: "POST",
    body: JSON.stringify(data),
  })
}

export function getApp(appId: string) {
  return apiFetch<AppDetail>(`/admin/apps/${appId}`)
}

export function updateApp(appId: string, data: AppUpdate) {
  return apiFetch<AppDetail>(`/admin/apps/${appId}`, {
    method: "PATCH",
    body: JSON.stringify(data),
  })
}

export function activateApp(appId: string) {
  return apiFetch<AppDetail>(`/admin/apps/${appId}/activate`, { method: "POST" })
}

export function deactivateApp(appId: string) {
  return apiFetch<AppDetail>(`/admin/apps/${appId}/deactivate`, { method: "POST" })
}

// ── Admin Users ───────────────────────────────────────────────────────────

export function listAdminUsers() {
  return apiFetch<AdminUserSummary[]>("/admin/users")
}

export function createAdminUser(data: AdminUserCreate) {
  return apiFetch<AdminUserSummary>("/admin/users", {
    method: "POST",
    body: JSON.stringify(data),
  })
}

export function getAdminUser(userId: string) {
  return apiFetch<AdminUserSummary>(`/admin/users/${userId}`)
}

export function updateAdminUser(userId: string, data: AdminUserUpdate) {
  return apiFetch<AdminUserSummary>(`/admin/users/${userId}`, {
    method: "PATCH",
    body: JSON.stringify(data),
  })
}

export function deleteAdminUser(userId: string) {
  return apiFetch(`/admin/users/${userId}`, { method: "DELETE" })
}

// ── Audit Log ─────────────────────────────────────────────────────────────

export function getAuditLog(params: {
  limit?: number
  offset?: number
  action?: string
  admin_user_id?: string
} = {}) {
  const searchParams = new URLSearchParams()
  if (params.limit) searchParams.set("limit", String(params.limit))
  if (params.offset) searchParams.set("offset", String(params.offset))
  if (params.action) searchParams.set("action", params.action)
  if (params.admin_user_id) searchParams.set("admin_user_id", params.admin_user_id)
  const qs = searchParams.toString()
  return apiFetch<AuditLogResponse>(`/admin/audit-log${qs ? `?${qs}` : ""}`)
}

// ── Observability ─────────────────────────────────────────────────────────

export function getTokenSummary(appId?: string) {
  const qs = appId ? `?app_id=${appId}` : ""
  return apiFetch<TokenSummary>(`/admin/observability/tokens/summary${qs}`)
}

export function getTokensRecent(params: {
  limit?: number
  offset?: number
  app_id?: string
} = {}) {
  const searchParams = new URLSearchParams()
  if (params.limit) searchParams.set("limit", String(params.limit))
  if (params.offset) searchParams.set("offset", String(params.offset))
  if (params.app_id) searchParams.set("app_id", params.app_id)
  const qs = searchParams.toString()
  return apiFetch<TokenRecentResponse>(`/admin/observability/tokens/recent${qs ? `?${qs}` : ""}`)
}

export function getRagStats(appId?: string) {
  const qs = appId ? `?app_id=${appId}` : ""
  return apiFetch<RagStats>(`/admin/observability/rag/stats${qs}`)
}

export function getCacheStats(appId?: string) {
  const qs = appId ? `?app_id=${appId}` : ""
  return apiFetch<CacheStats>(`/admin/observability/cache/stats${qs}`)
}

export function getLlmHealth() {
  return apiFetch<LlmHealth>("/admin/observability/llm/health")
}