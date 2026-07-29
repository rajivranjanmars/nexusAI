import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { getTokenSummary, getRagStats, getCacheStats, getLlmHealth, getTokensRecent, type TokenSummary, type RagStats, type CacheStats, type LlmHealth, type TokenDetail } from "@/api/admin"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from "recharts"

const COLORS = ["#8884d8", "#82ca9d", "#ffc658", "#ff8042", "#0088fe", "#00c49f"]

export default function Analytics() {
  const [tokens, setTokens] = useState<TokenSummary | null>(null)
  const [rag, setRag] = useState<RagStats | null>(null)
  const [cache, setCache] = useState<CacheStats | null>(null)
  const [llm, setLlm] = useState<LlmHealth | null>(null)
  const [recent, setRecent] = useState<TokenDetail[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([
      getTokenSummary().catch(() => null),
      getRagStats().catch(() => null),
      getCacheStats().catch(() => null),
      getLlmHealth().catch(() => null),
      getTokensRecent({ limit: 100 }).catch(() => ({ total: 0, limit: 0, offset: 0, items: [] })),
    ]).then(([t, r, c, l, rec]) => {
      setTokens(t)
      setRag(r)
      setCache(c)
      setLlm(l)
      setRecent(rec.items)
      setLoading(false)
    })
  }, [])

  if (loading) {
    return (
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Analytics</h1>
        <p className="mt-2 text-muted-foreground">Loading...</p>
      </div>
    )
  }

  const fmt = (n: number) => n.toLocaleString()
  const fmtCost = (n: number) => `$${n.toFixed(4)}`

  const modelData = tokens?.by_model.map((m) => ({ name: m.model, tokens: m.tokens, cost: m.cost_usd })) ?? []
  const actionData = tokens?.by_action.map((a) => ({ name: a.action, calls: a.calls })) ?? []

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold tracking-tight">Analytics</h1>

      {/* Summary cards */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardHeader className="pb-2"><CardTitle className="text-sm font-medium text-muted-foreground">Total Calls</CardTitle></CardHeader>
          <CardContent><div className="text-2xl font-bold">{tokens ? fmt(tokens.total_calls) : "—"}</div></CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2"><CardTitle className="text-sm font-medium text-muted-foreground">Total Tokens</CardTitle></CardHeader>
          <CardContent><div className="text-2xl font-bold">{tokens ? fmt(tokens.total_tokens) : "—"}</div></CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2"><CardTitle className="text-sm font-medium text-muted-foreground">Total Cost</CardTitle></CardHeader>
          <CardContent><div className="text-2xl font-bold">{tokens ? fmtCost(tokens.total_cost_usd) : "—"}</div></CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2"><CardTitle className="text-sm font-medium text-muted-foreground">Avg Latency</CardTitle></CardHeader>
          <CardContent><div className="text-2xl font-bold">{tokens ? `${tokens.avg_latency_ms.toFixed(0)}ms` : "—"}</div></CardContent>
        </Card>
      </div>

      {/* Charts */}
      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader><CardTitle>Tokens by Model</CardTitle></CardHeader>
          <CardContent>
            {modelData.length > 0 ? (
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={modelData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" fontSize={12} />
                  <YAxis fontSize={12} />
                  <Tooltip />
                  <Bar dataKey="tokens" fill="#8884d8" name="Tokens" />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <p className="text-muted-foreground text-sm">No data</p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader><CardTitle>Cost by Model</CardTitle></CardHeader>
          <CardContent>
            {modelData.length > 0 ? (
              <ResponsiveContainer width="100%" height={250}>
                <PieChart>
                  <Pie data={modelData} dataKey="cost" nameKey="name" cx="50%" cy="50%" outerRadius={100} label={(props) => `${props.name}: $${(props.value as number).toFixed(2)}`}>
                    {modelData.map((_, i) => (<Cell key={i} fill={COLORS[i % COLORS.length]} />))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <p className="text-muted-foreground text-sm">No data</p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader><CardTitle>Calls by Action</CardTitle></CardHeader>
          <CardContent>
            {actionData.length > 0 ? (
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={actionData} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" fontSize={12} />
                  <YAxis dataKey="name" type="category" fontSize={12} width={100} />
                  <Tooltip />
                  <Bar dataKey="calls" fill="#82ca9d" name="Calls" />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <p className="text-muted-foreground text-sm">No data</p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader><CardTitle>Cache & RAG</CardTitle></CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Cache Hits</span>
                <span className="font-medium">{cache ? fmt(cache.total_hits) : "—"}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Active Cache Entries</span>
                <span className="font-medium">{cache ? cache.active_entries : "—"}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">RAG Chunks</span>
                <span className="font-medium">{rag ? fmt(rag.total_chunks) : "—"}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">LLM Status</span>
                <span className={`font-medium ${llm?.status === "ok" ? "text-green-600" : "text-red-600"}`}>
                  {llm?.status ?? "—"}
                </span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Recent activity */}
      <Card>
        <CardHeader><CardTitle>Recent Activity</CardTitle></CardHeader>
        <CardContent>
          {recent.length > 0 ? (
            <div className="rounded-md border max-h-80 overflow-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b bg-muted/50 sticky top-0">
                    <th className="px-4 py-2 text-left font-medium">Time</th>
                    <th className="px-4 py-2 text-left font-medium">App</th>
                    <th className="px-4 py-2 text-left font-medium">Action</th>
                    <th className="px-4 py-2 text-left font-medium">Model</th>
                    <th className="px-4 py-2 text-right font-medium">Tokens</th>
                    <th className="px-4 py-2 text-right font-medium">Cost</th>
                  </tr>
                </thead>
                <tbody>
                  {recent.slice(0, 50).map((r) => (
                    <tr key={r.id} className="border-b hover:bg-muted/30">
                      <td className="px-4 py-2 text-xs text-muted-foreground">{r.timestamp ? new Date(r.timestamp).toLocaleString() : "—"}</td>
                      <td className="px-4 py-2 text-xs font-mono">{r.app_id?.slice(0, 8) ?? "—"}</td>
                      <td className="px-4 py-2">{r.action ?? "—"}</td>
                      <td className="px-4 py-2 text-muted-foreground">{r.model ?? "—"}</td>
                      <td className="px-4 py-2 text-right">{fmt(r.total_tokens)}</td>
                      <td className="px-4 py-2 text-right">{r.estimated_cost_usd != null ? `$${r.estimated_cost_usd.toFixed(4)}` : "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="text-muted-foreground text-sm">No recent activity</p>
          )}
        </CardContent>
      </Card>
    </div>
  )
}