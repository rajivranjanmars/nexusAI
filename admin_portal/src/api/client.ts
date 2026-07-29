const BASE_URL = "/api"

interface Tokens {
  access_token: string
  refresh_token: string
  expires_in: number
}

export interface AdminUser {
  id: string
  email: string
  display_name: string
  admin_role: "super_admin" | "app_admin"
  app_id: string | null
}

let accessToken: string | null = localStorage.getItem("access_token")
let refreshToken: string | null = localStorage.getItem("refresh_token")
let refreshTimer: ReturnType<typeof setTimeout> | null = null

function setTokens(tokens: Tokens) {
  accessToken = tokens.access_token
  refreshToken = tokens.refresh_token
  localStorage.setItem("access_token", tokens.access_token)
  localStorage.setItem("refresh_token", tokens.refresh_token)
  scheduleRefresh(tokens.expires_in)
}

function clearTokens() {
  accessToken = null
  refreshToken = null
  localStorage.removeItem("access_token")
  localStorage.removeItem("refresh_token")
  if (refreshTimer) {
    clearTimeout(refreshTimer)
    refreshTimer = null
  }
}

function scheduleRefresh(expiresIn: number) {
  if (refreshTimer) clearTimeout(refreshTimer)
  // Refresh 60s before expiry
  const delay = Math.max((expiresIn - 60) * 1000, 0)
  refreshTimer = setTimeout(() => {
    doRefresh().catch(() => clearTokens())
  }, delay)
}

async function doRefresh(): Promise<void> {
  if (!refreshToken) throw new Error("No refresh token")
  const res = await fetch(`${BASE_URL}/auth/admin/refresh`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ refresh_token: refreshToken }),
  })
  if (!res.ok) throw new Error("Refresh failed")
  const data = await res.json()
  setTokens(data)
}

async function apiFetch<T = unknown>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(options.headers as Record<string, string>),
  }
  if (accessToken) {
    headers["Authorization"] = `Bearer ${accessToken}`
  }
  const res = await fetch(`${BASE_URL}${path}`, { ...options, headers })
  if (res.status === 401 && refreshToken) {
    try {
      await doRefresh()
      headers["Authorization"] = `Bearer ${accessToken}`
      const retry = await fetch(`${BASE_URL}${path}`, { ...options, headers })
      if (!retry.ok) {
        const err = await retry.json().catch(() => ({}))
        throw new ApiError(retry.status, err.detail || err.message || "Request failed")
      }
      return retry.json()
    } catch (e) {
      if (e instanceof ApiError) throw e
      clearTokens()
      throw new ApiError(401, "Session expired")
    }
  }
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new ApiError(res.status, err.detail || err.message || "Request failed")
  }
  if (res.status === 204) return undefined as T
  return res.json()
}

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

// Auth endpoints
export async function login(email: string, password: string) {
  const data = await apiFetch<{
    access_token: string
    refresh_token: string
    expires_in: number
    admin_user: AdminUser
  }>("/auth/admin/login", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  })
  setTokens(data)
  return data.admin_user
}

export async function getMe(): Promise<AdminUser> {
  return apiFetch<AdminUser>("/auth/admin/me")
}

export function logout() {
  clearTokens()
}

export function getAccessToken(): string | null {
  return accessToken
}

export { apiFetch, clearTokens, setTokens }