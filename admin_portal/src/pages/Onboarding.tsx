import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { bootstrapApp, type AppBootstrapResponse } from "@/api/admin"
import { ApiError } from "@/api/client"

export default function Onboarding() {
  const [step, setStep] = useState<"form" | "success">("form")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")
  const [result, setResult] = useState<AppBootstrapResponse | null>(null)

  const [form, setForm] = useState({
    name: "",
    domain: "",
    workflows: "general",
    tools: "detect_workflow, get_student, chat_complete",
  })

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!form.name.trim()) {
      setError("App name is required")
      return
    }

    setLoading(true)
    setError("")

    try {
      const res = await bootstrapApp({
        name: form.name,
        domain: form.domain || undefined,
        allowed_workflows: form.workflows.split(",").map((s) => s.trim()).filter(Boolean),
        allowed_tools: form.tools.split(",").map((s) => s.trim()).filter(Boolean),
      })
      setResult(res)
      setStep("success")
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Onboarding failed")
    } finally {
      setLoading(false)
    }
  }

  if (step === "success" && result) {
    return (
      <div className="space-y-6 max-w-2xl">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">App Onboarded Successfully</h1>
          <p className="text-muted-foreground mt-2">
            Save these credentials in a secure location. You won't see the private key again.
          </p>
        </div>

        <div className="space-y-4">
          <KeyDisplay label="App ID" value={result.app_id} />
          <KeyDisplay label="Proxy URL" value={result.proxy_url} />
          <KeyDisplay label="Domain" value={result.domain || "(empty)" } />
          <KeyDisplay label="Private Key" value={result.private_key} />
          <KeyDisplay label="Public Key" value={result.public_key} multiline />
        </div>

        <div className="border-t pt-4">
          <p className="text-sm font-medium mb-2">Configuration</p>
          <div className="space-y-1 text-sm font-mono text-muted-foreground">
            <p>Workflows: {result.allowed_workflows.join(", ")}</p>
            <p>Tools: {result.allowed_tools.join(", ")}</p>
          </div>
        </div>

        <Button onClick={() => window.location.href = "/apps"}>
          View All Apps
        </Button>
      </div>
    )
  }

  return (
    <div className="space-y-6 max-w-2xl">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Onboard New App</h1>
        <p className="text-muted-foreground mt-2">Register a new application and generate its API credentials.</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>App Details</CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            {error && <p className="text-sm text-destructive">{error}</p>}

            <div className="space-y-2">
              <Label htmlFor="name">App Name *</Label>
              <Input
                id="name"
                value={form.name}
                onChange={(e) => setForm({ ...form, name: e.target.value })}
                placeholder="e.g., Student Portal"
                required
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="domain">Domain (optional)</Label>
              <Input
                id="domain"
                value={form.domain}
                onChange={(e) => setForm({ ...form, domain: e.target.value })}
                placeholder="e.g., app.example.com or comma-separated list"
              />
              <p className="text-xs text-muted-foreground">
                Allowed origin(s) for requests. Leave empty for internal use.
              </p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="workflows">Workflows (comma-separated)</Label>
              <Input
                id="workflows"
                value={form.workflows}
                onChange={(e) => setForm({ ...form, workflows: e.target.value })}
                placeholder="general"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="tools">Tools (comma-separated)</Label>
              <Input
                id="tools"
                value={form.tools}
                onChange={(e) => setForm({ ...form, tools: e.target.value })}
                placeholder="detect_workflow, get_student, chat_complete"
              />
            </div>

            <Button type="submit" disabled={loading} className="w-full">
              {loading ? "Onboarding..." : "Onboard App"}
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  )
}

function KeyDisplay({
  label,
  value,
  multiline = false,
}: {
  label: string
  value: string
  multiline?: boolean
}) {
  const [copied, setCopied] = useState(false)

  function handleCopy() {
    navigator.clipboard.writeText(value)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between">
        <Label className="text-sm font-medium">{label}</Label>
        <Button size="sm" variant="ghost" onClick={handleCopy}>
          {copied ? "Copied!" : "Copy"}
        </Button>
      </div>
      <div
        className={`p-3 rounded-md bg-muted font-mono text-xs overflow-auto break-all ${multiline ? "max-h-32" : ""}`}
      >
        {value}
      </div>
    </div>
  )
}
