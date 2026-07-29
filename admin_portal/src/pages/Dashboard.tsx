import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { getTokenSummary, getRagStats, getCacheStats, getLlmHealth, listApps, type TokenSummary, type RagStats, type CacheStats, type LlmHealth } from "@/api/admin"
import { Activity, Zap, Layers, HardDrive } from "lucide-react"

function StatCard({ title, value, subtitle, icon: Icon }: { title: string; value: string; subtitle?: string; icon: React.ComponentType<{ className?: string }> }) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <CardTitle className="text-sm font-medium text-muted-foreground">{title}</CardTitle>
        <Icon className="h-4 w-4 text-muted-foreground" />
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">{value}</div>
        {subtitle && <p className="text-xs text-muted-foreground mt-1">{subtitle}</p>}
      </CardContent>
    </Card>
  )
}

export default function Dashboard() {
  const [tokens, setTokens] = useState<TokenSummary | null>(null)
  const [rag, setRag] = useState<RagStats | null>(null)
  const [cache, setCache] = useState<CacheStats | null>(null)
  const [llm, setLlm] = useState<LlmHealth | null>(null)
  const [appCount, setAppCount] = useState(0)
  const [activeAppCount, setActiveAppCount] = useState(0)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([
      getTokenSummary().catch(() => null),
      getRagStats().catch(() => null),
      getCacheStats().catch(() => null),
      getLlmHealth().catch(() => null),
      listApps().catch(() => []),
    ]).then(([t, r, c, l, apps]) => {
      setTokens(t)
      setRag(r)
      setCache(c)
      setLlm(l)
      setAppCount(apps.length)
      setActiveAppCount(apps.filter((a) => a.is_active).length)
      setLoading(false)
    })
  }, [])

  if (loading) {
    return (
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Dashboard</h1>
        <p className="mt-2 text-muted-foreground">Loading...</p>
      </div>
    )
  }

  const fmt = (n: number) => n.toLocaleString()
  const fmtCost = (n: number) => `$${n.toFixed(2)}`

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold tracking-tight">Dashboard</h1>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <StatCard
          title="Apps"
          value={`${activeAppCount} / ${appCount}`}
          subtitle="Active / Total"
          icon={Layers}
        />
        <StatCard
          title="Tokens (30d)"
          value={tokens ? fmt(tokens.total_tokens) : "—"}
          subtitle={tokens ? `${fmt(tokens.total_calls)} calls` : undefined}
          icon={Zap}
        />
        <StatCard
          title="Cost (30d)"
          value={tokens ? fmtCost(tokens.total_cost_usd) : "—"}
          subtitle={tokens ? `${tokens.avg_latency_ms.toFixed(0)}ms avg latency` : undefined}
          icon={Activity}
        />
        <StatCard
          title="Cache Hits"
          value={cache ? fmt(cache.total_hits) : "—"}
          subtitle={cache ? `${cache.active_entries} active entries` : undefined}
          icon={HardDrive}
        />
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-medium">RAG Knowledge</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Total Chunks</span>
                <span className="font-medium">{rag ? fmt(rag.total_chunks) : "—"}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Apps with Data</span>
                <span className="font-medium">{rag ? rag.total_apps : "—"}</span>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-medium">LLM Health</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Status</span>
                <span className={`font-medium ${llm?.status === "ok" ? "text-green-600" : "text-red-600"}`}>
                  {llm?.status ?? "—"}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Model</span>
                <span className="font-medium">{llm?.model ?? "—"}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Latency</span>
                <span className="font-medium">{llm ? `${llm.latency_ms}ms` : "—"}</span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}