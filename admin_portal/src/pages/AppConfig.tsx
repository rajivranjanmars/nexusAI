import { useState, useEffect } from "react"
import { useAuth } from "@/contexts/AuthContext"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { getApp, updateApp } from "@/api/admin"
import { ApiError } from "@/api/client"

export default function AppConfig() {
  const { user } = useAuth()
  const appId = user?.app_id ?? ""

  const [appName, setAppName] = useState("")
  const [raw, setRaw] = useState("")
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState("")
  const [success, setSuccess] = useState("")

  useEffect(() => {
    if (!appId) { setLoading(false); return }
    getApp(appId)
      .then((app) => {
        setAppName(app.name)
        setRaw(JSON.stringify(app.app_config ?? {}, null, 2))
      })
      .catch(() => setError("Failed to load app config"))
      .finally(() => setLoading(false))
  }, [appId])

  async function handleSave() {
    setError("")
    setSuccess("")
    let parsed: Record<string, unknown>
    try {
      parsed = JSON.parse(raw)
    } catch {
      setError("Invalid JSON")
      return
    }
    setSaving(true)
    try {
      await updateApp(appId, { app_config: parsed })
      setSuccess("Saved")
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Save failed")
    } finally {
      setSaving(false)
    }
  }

  if (!appId) return <p className="text-destructive">No app assigned to this account.</p>
  if (loading) return <p className="text-muted-foreground">Loading...</p>

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold tracking-tight">App Configuration</h1>

      <Card>
        <CardHeader>
          <CardTitle>{appName} — system prompt, persona, workflows, and theme</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-sm text-muted-foreground">
            Edit the raw <code className="text-xs bg-muted px-1 py-0.5 rounded">app_config</code> JSON.
            Supports <code className="text-xs bg-muted px-1 py-0.5 rounded">system_prompt</code>,{" "}
            <code className="text-xs bg-muted px-1 py-0.5 rounded">workflow_config</code>,{" "}
            <code className="text-xs bg-muted px-1 py-0.5 rounded">response_styles</code>, and visual theme keys.
          </p>
          <textarea
            value={raw}
            onChange={(e) => setRaw(e.target.value)}
            rows={24}
            spellCheck={false}
            className="w-full rounded-md border border-input bg-transparent px-3 py-2 font-mono text-xs shadow-sm"
          />
          {error && <p className="text-sm text-destructive">{error}</p>}
          {success && <p className="text-sm text-green-600">{success}</p>}
          <Button onClick={handleSave} disabled={saving}>{saving ? "Saving..." : "Save Config"}</Button>
        </CardContent>
      </Card>
    </div>
  )
}