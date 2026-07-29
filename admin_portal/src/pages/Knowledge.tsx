import { useState, useEffect, useCallback } from "react"
import { useAuth } from "@/contexts/AuthContext"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { startCrawl, ingestKnowledge, deleteAppKnowledge, getLatestCrawlStatus, type CrawlJobStatus } from "@/api/rag"
import { ApiError } from "@/api/client"

export default function Knowledge() {
  const { user } = useAuth()
  const appId = user?.app_id ?? ""

  const [job, setJob] = useState<CrawlJobStatus | null>(null)
  const [error, setError] = useState("")
  const [success, setSuccess] = useState("")

  const refreshStatus = useCallback(() => {
    if (!appId) return
    getLatestCrawlStatus(appId)
      .then(setJob)
      .catch(() => setJob(null))
  }, [appId])

  useEffect(() => { refreshStatus() }, [refreshStatus])

  // Poll while a job is actively running
  useEffect(() => {
    if (!job || !["queued", "processing"].includes(job.status)) return
    const t = setInterval(refreshStatus, 3000)
    return () => clearInterval(t)
  }, [job, refreshStatus])

  if (!appId) {
    return <p className="text-destructive">No app assigned to this account.</p>
  }

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold tracking-tight">Knowledge Base</h1>

      {error && <p className="text-sm text-destructive">{error}</p>}
      {success && <p className="text-sm text-green-600">{success}</p>}

      <div className="grid gap-6 md:grid-cols-2">
        <CrawlForm
          appId={appId}
          onStarted={() => { setSuccess("Crawl started"); setError(""); refreshStatus() }}
          onError={(m) => { setError(m); setSuccess("") }}
        />
        <ManualIngestForm
          appId={appId}
          onIngested={(chunks) => { setSuccess(`Ingested ${chunks} chunks`); setError("") }}
          onError={(m) => { setError(m); setSuccess("") }}
        />
      </div>

      {job && <CrawlStatusCard job={job} onRefresh={refreshStatus} />}

      <DangerZone
        appId={appId}
        onDeleted={(n) => { setSuccess(`Deleted ${n} chunks`); setError("") }}
        onError={(m) => { setError(m); setSuccess("") }}
      />
    </div>
  )
}

function CrawlForm({ appId, onStarted, onError }: { appId: string; onStarted: () => void; onError: (m: string) => void }) {
  const [seedUrl, setSeedUrl] = useState("")
  const [maxDepth, setMaxDepth] = useState(2)
  const [submitting, setSubmitting] = useState(false)

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setSubmitting(true)
    try {
      await startCrawl({ app_id: appId, seed_url: seedUrl, max_depth: maxDepth })
      onStarted()
      setSeedUrl("")
    } catch (e) {
      onError(e instanceof ApiError ? e.message : "Failed to start crawl")
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <Card>
      <CardHeader><CardTitle>Start Crawl</CardTitle></CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="seedUrl">Seed URL</Label>
            <Input id="seedUrl" type="url" value={seedUrl} onChange={(e) => setSeedUrl(e.target.value)} required placeholder="https://example.com" />
          </div>
          <div className="space-y-2">
            <Label htmlFor="maxDepth">Max Depth</Label>
            <Input id="maxDepth" type="number" min={0} max={5} value={maxDepth} onChange={(e) => setMaxDepth(Number(e.target.value))} />
          </div>
          <Button type="submit" disabled={submitting}>{submitting ? "Starting..." : "Start Crawl"}</Button>
        </form>
      </CardContent>
    </Card>
  )
}

function ManualIngestForm({ appId, onIngested, onError }: { appId: string; onIngested: (chunks: number) => void; onError: (m: string) => void }) {
  const [content, setContent] = useState("")
  const [sourceRef, setSourceRef] = useState("")
  const [submitting, setSubmitting] = useState(false)

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setSubmitting(true)
    try {
      const res = await ingestKnowledge({ app_id: appId, source_type: "structured", content, source_ref: sourceRef })
      onIngested(res.chunks_upserted)
      setContent("")
      setSourceRef("")
    } catch (e) {
      onError(e instanceof ApiError ? e.message : "Failed to ingest")
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <Card>
      <CardHeader><CardTitle>Manual Ingest</CardTitle></CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="sourceRef">Source Reference</Label>
            <Input id="sourceRef" value={sourceRef} onChange={(e) => setSourceRef(e.target.value)} required placeholder="e.g. faq-2026" />
          </div>
          <div className="space-y-2">
            <Label htmlFor="content">Content (text or JSON)</Label>
            <textarea
              id="content"
              value={content}
              onChange={(e) => setContent(e.target.value)}
              required
              rows={5}
              className="flex w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm shadow-sm"
              placeholder="Paste content to ingest..."
            />
          </div>
          <Button type="submit" disabled={submitting}>{submitting ? "Ingesting..." : "Ingest"}</Button>
        </form>
      </CardContent>
    </Card>
  )
}

function CrawlStatusCard({ job, onRefresh }: { job: CrawlJobStatus; onRefresh: () => void }) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle>Latest Crawl Job</CardTitle>
        <Button variant="outline" size="sm" onClick={onRefresh}>Refresh</Button>
      </CardHeader>
      <CardContent>
        <div className="grid gap-3 md:grid-cols-3 text-sm">
          <div><span className="text-muted-foreground">Status: </span><span className="font-medium">{job.status}</span></div>
          <div><span className="text-muted-foreground">Progress: </span><span className="font-medium">{job.progress_pct != null ? `${job.progress_pct}%` : "—"}</span></div>
          <div><span className="text-muted-foreground">Pages Found: </span><span className="font-medium">{job.pages_found}</span></div>
          <div><span className="text-muted-foreground">Chunks: </span><span className="font-medium">{job.chunks_upserted}</span></div>
          <div><span className="text-muted-foreground">Failed: </span><span className="font-medium">{job.urls_failed}</span></div>
          <div><span className="text-muted-foreground">Elapsed: </span><span className="font-medium">{job.elapsed_seconds}s</span></div>
        </div>
        {job.error && <p className="mt-3 text-sm text-destructive">{job.error}</p>}
      </CardContent>
    </Card>
  )
}

function DangerZone({ appId, onDeleted, onError }: { appId: string; onDeleted: (n: number) => void; onError: (m: string) => void }) {
  const [deleting, setDeleting] = useState(false)

  async function handleDelete() {
    if (!confirm("Delete ALL knowledge chunks for this app? This cannot be undone.")) return
    setDeleting(true)
    try {
      const res = await deleteAppKnowledge(appId)
      onDeleted(res.chunks_deleted)
    } catch (e) {
      onError(e instanceof ApiError ? e.message : "Failed to delete")
    } finally {
      setDeleting(false)
    }
  }

  return (
    <Card className="border-destructive/50">
      <CardHeader><CardTitle className="text-destructive">Danger Zone</CardTitle></CardHeader>
      <CardContent>
        <p className="text-sm text-muted-foreground mb-4">Permanently delete all ingested knowledge chunks for this app.</p>
        <Button variant="destructive" onClick={handleDelete} disabled={deleting}>
          {deleting ? "Deleting..." : "Delete All Knowledge"}
        </Button>
      </CardContent>
    </Card>
  )
}