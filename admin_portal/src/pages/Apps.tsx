import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { listApps, getApp, updateApp, activateApp, deactivateApp, type AppSummary, type AppDetail, type AppUpdate } from "@/api/admin"
import { ApiError } from "@/api/client"

export default function Apps() {
  const [apps, setApps] = useState<AppSummary[]>([])
  const [selected, setSelected] = useState<AppDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [search, setSearch] = useState("")
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState("")

  useEffect(() => {
    listApps()
      .then(setApps)
      .catch(() => setError("Failed to load apps"))
      .finally(() => setLoading(false))
  }, [])

  async function selectApp(appId: string) {
    setError("")
    try {
      const detail = await getApp(appId)
      setSelected(detail)
    } catch {
      setError("Failed to load app detail")
    }
  }

  async function handleSave(updates: AppUpdate) {
    if (!selected) return
    setSaving(true)
    setError("")
    try {
      const updated = await updateApp(selected.app_id, updates)
      setSelected(updated)
      setApps((prev) => prev.map((a) => (a.app_id === updated.app_id ? updated : a)))
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Save failed")
    } finally {
      setSaving(false)
    }
  }

  async function handleToggleActive() {
    if (!selected) return
    setSaving(true)
    setError("")
    try {
      const updated = selected.is_active
        ? await deactivateApp(selected.app_id)
        : await activateApp(selected.app_id)
      setSelected(updated)
      setApps((prev) => prev.map((a) => (a.app_id === updated.app_id ? updated : a)))
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Toggle failed")
    } finally {
      setSaving(false)
    }
  }

  const filtered = apps.filter(
    (a) =>
      a.name.toLowerCase().includes(search.toLowerCase()) ||
      a.domain.toLowerCase().includes(search.toLowerCase()),
  )

  if (loading) return <p className="text-muted-foreground">Loading...</p>

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold tracking-tight">App Management</h1>
      </div>

      <div className="flex gap-4">
        <Input
          placeholder="Search apps..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="max-w-sm"
        />
      </div>

      {error && <p className="text-sm text-destructive">{error}</p>}

      <div className="rounded-md border">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b bg-muted/50">
              <th className="px-4 py-3 text-left font-medium">Name</th>
              <th className="px-4 py-3 text-left font-medium">Domain</th>
              <th className="px-4 py-3 text-left font-medium">Status</th>
              <th className="px-4 py-3 text-left font-medium">Rate Limit</th>
              <th className="px-4 py-3 text-right font-medium">Actions</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((app) => (
              <tr key={app.app_id} className="border-b hover:bg-muted/30">
                <td className="px-4 py-3 font-medium">{app.name}</td>
                <td className="px-4 py-3 text-muted-foreground">{app.domain}</td>
                <td className="px-4 py-3">
                  <span className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${app.is_active ? "bg-green-100 text-green-700" : "bg-red-100 text-red-700"}`}>
                    {app.is_active ? "Active" : "Inactive"}
                  </span>
                </td>
                <td className="px-4 py-3 text-muted-foreground">{app.rate_limit_rpm} rpm</td>
                <td className="px-4 py-3 text-right">
                  <Button variant="outline" size="sm" onClick={() => selectApp(app.app_id)}>
                    View
                  </Button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {selected && (
        <Card>
          <CardHeader>
            <CardTitle>Edit: {selected.name}</CardTitle>
          </CardHeader>
          <CardContent>
            <AppEditForm
              app={selected}
              onSave={handleSave}
              onToggleActive={handleToggleActive}
              saving={saving}
            />
          </CardContent>
        </Card>
      )}
    </div>
  )
}

function AppEditForm({
  app,
  onSave,
  onToggleActive,
  saving,
}: {
  app: AppDetail
  onSave: (updates: AppUpdate) => Promise<void>
  onToggleActive: () => Promise<void>
  saving: boolean
}) {
  const [name, setName] = useState(app.name)
  const [domain, setDomain] = useState(app.domain)
  const [rateLimit, setRateLimit] = useState(app.rate_limit_rpm)
  const [cacheTtl, setCacheTtl] = useState(app.cache_ttl_seconds)
  const [tokenQuota, setTokenQuota] = useState(app.token_quota_monthly ?? 0)
  const [modelOverride, setModelOverride] = useState(app.llm_model_override ?? "")

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    onSave({
      name,
      domain,
      rate_limit_rpm: rateLimit,
      cache_ttl_seconds: cacheTtl,
      token_quota_monthly: tokenQuota || null,
      llm_model_override: modelOverride || null,
    })
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="grid gap-4 md:grid-cols-2">
        <div className="space-y-2">
          <Label htmlFor="name">Name</Label>
          <Input id="name" value={name} onChange={(e) => setName(e.target.value)} />
        </div>
        <div className="space-y-2">
          <Label htmlFor="domain">Domain</Label>
          <Input id="domain" value={domain} onChange={(e) => setDomain(e.target.value)} />
        </div>
        <div className="space-y-2">
          <Label htmlFor="rateLimit">Rate Limit (rpm)</Label>
          <Input id="rateLimit" type="number" value={rateLimit} onChange={(e) => setRateLimit(Number(e.target.value))} />
        </div>
        <div className="space-y-2">
          <Label htmlFor="cacheTtl">Cache TTL (seconds)</Label>
          <Input id="cacheTtl" type="number" value={cacheTtl} onChange={(e) => setCacheTtl(Number(e.target.value))} />
        </div>
        <div className="space-y-2">
          <Label htmlFor="tokenQuota">Monthly Token Quota</Label>
          <Input id="tokenQuota" type="number" value={tokenQuota} onChange={(e) => setTokenQuota(Number(e.target.value))} />
        </div>
        <div className="space-y-2">
          <Label htmlFor="modelOverride">Model Override</Label>
          <Input id="modelOverride" value={modelOverride} onChange={(e) => setModelOverride(e.target.value)} placeholder="(default)" />
        </div>
      </div>
      <div className="flex gap-3">
        <Button type="submit" disabled={saving}>
          {saving ? "Saving..." : "Save"}
        </Button>
        <Button type="button" variant="outline" onClick={onToggleActive} disabled={saving}>
          {app.is_active ? "Deactivate" : "Activate"}
        </Button>
      </div>
    </form>
  )
}