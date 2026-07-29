import { useState, useEffect } from "react"
import { useAuth } from "@/contexts/AuthContext"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { getApp, updateApp, type AppDetail } from "@/api/admin"
import { ApiError } from "@/api/client"

export default function AppSettings() {
  const { user } = useAuth()
  const appId = user?.app_id ?? ""

  const [app, setApp] = useState<AppDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState("")

  useEffect(() => {
    if (!appId) { setLoading(false); return }
    getApp(appId)
      .then(setApp)
      .catch(() => setError("Failed to load app settings"))
      .finally(() => setLoading(false))
  }, [appId])

  if (!appId) return <p className="text-destructive">No app assigned to this account.</p>
  if (loading) return <p className="text-muted-foreground">Loading...</p>
  if (!app) return <p className="text-destructive">{error || "App not found"}</p>

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold tracking-tight">App Settings</h1>
      {error && <p className="text-sm text-destructive">{error}</p>}

      <Card>
        <CardHeader><CardTitle>{app.name}</CardTitle></CardHeader>
        <CardContent>
          <SettingsForm app={app} onSaved={setApp} onError={setError} />
        </CardContent>
      </Card>
    </div>
  )
}

function SettingsForm({ app, onSaved, onError }: { app: AppDetail; onSaved: (a: AppDetail) => void; onError: (m: string) => void }) {
  const [domain, setDomain] = useState(app.domain)
  const [rateLimit, setRateLimit] = useState(app.rate_limit_rpm)
  const [tokenQuota, setTokenQuota] = useState(app.token_quota_monthly ?? 0)
  const [modelOverride, setModelOverride] = useState(app.llm_model_override ?? "")
  const [allowedWorkflows, setAllowedWorkflows] = useState(app.allowed_workflows.join(", "))
  const [allowedTools, setAllowedTools] = useState(app.allowed_tools.join(", "))
  const [saving, setSaving] = useState(false)

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setSaving(true)
    onError("")
    try {
      const updated = await updateApp(app.app_id, {
        domain,
        rate_limit_rpm: rateLimit,
        token_quota_monthly: tokenQuota || null,
        llm_model_override: modelOverride || null,
        allowed_workflows: allowedWorkflows.split(",").map((s) => s.trim()).filter(Boolean),
        allowed_tools: allowedTools.split(",").map((s) => s.trim()).filter(Boolean),
      })
      onSaved(updated)
    } catch (e) {
      onError(e instanceof ApiError ? e.message : "Save failed")
    } finally {
      setSaving(false)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="grid gap-4 md:grid-cols-2">
        <div className="space-y-2">
          <Label htmlFor="domain">Domain</Label>
          <Input id="domain" value={domain} onChange={(e) => setDomain(e.target.value)} />
        </div>
        <div className="space-y-2">
          <Label htmlFor="rateLimit">Rate Limit (rpm)</Label>
          <Input id="rateLimit" type="number" value={rateLimit} onChange={(e) => setRateLimit(Number(e.target.value))} />
        </div>
        <div className="space-y-2">
          <Label htmlFor="tokenQuota">Monthly Token Quota</Label>
          <Input id="tokenQuota" type="number" value={tokenQuota} onChange={(e) => setTokenQuota(Number(e.target.value))} />
        </div>
        <div className="space-y-2">
          <Label htmlFor="modelOverride">Model Override</Label>
          <Input id="modelOverride" value={modelOverride} onChange={(e) => setModelOverride(e.target.value)} placeholder="(default)" />
        </div>
        <div className="space-y-2 md:col-span-2">
          <Label htmlFor="workflows">Allowed Workflows (comma-separated)</Label>
          <Input id="workflows" value={allowedWorkflows} onChange={(e) => setAllowedWorkflows(e.target.value)} />
        </div>
        <div className="space-y-2 md:col-span-2">
          <Label htmlFor="tools">Allowed Tools (comma-separated)</Label>
          <Input id="tools" value={allowedTools} onChange={(e) => setAllowedTools(e.target.value)} />
        </div>
      </div>
      <Button type="submit" disabled={saving}>{saving ? "Saving..." : "Save"}</Button>
    </form>
  )
}