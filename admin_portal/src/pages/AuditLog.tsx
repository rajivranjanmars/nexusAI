import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { getAuditLog, type AuditLogEntry } from "@/api/admin"

export default function AuditLog() {
  const [entries, setEntries] = useState<AuditLogEntry[]>([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState("")
  const [actionFilter, setActionFilter] = useState("")
  const [offset, setOffset] = useState(0)
  const limit = 50

  function load() {
    setLoading(true)
    setError("")
    getAuditLog({ limit, offset, ...(actionFilter ? { action: actionFilter } : {}) })
      .then((res) => {
        setEntries(res.entries)
        setTotal(res.total)
      })
      .catch(() => setError("Failed to load audit log"))
      .finally(() => setLoading(false))
  }

  useEffect(() => { load() }, [offset])

  function handleFilter() {
    setOffset(0)
    load()
  }

  const totalPages = Math.ceil(total / limit)
  const currentPage = Math.floor(offset / limit) + 1

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold tracking-tight">Audit Log</h1>

      <div className="flex gap-4">
        <Input
          placeholder="Filter by action (e.g. user.create)"
          value={actionFilter}
          onChange={(e) => setActionFilter(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleFilter()}
          className="max-w-xs"
        />
        <Button variant="outline" onClick={handleFilter}>Filter</Button>
      </div>

      {error && <p className="text-sm text-destructive">{error}</p>}

      {loading ? (
        <p className="text-muted-foreground">Loading...</p>
      ) : (
        <>
          <div className="rounded-md border">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b bg-muted/50">
                  <th className="px-4 py-3 text-left font-medium">Action</th>
                  <th className="px-4 py-3 text-left font-medium">Target</th>
                  <th className="px-4 py-3 text-left font-medium">Admin</th>
                  <th className="px-4 py-3 text-left font-medium">IP</th>
                  <th className="px-4 py-3 text-right font-medium">Time</th>
                </tr>
              </thead>
              <tbody>
                {entries.length === 0 ? (
                  <tr><td colSpan={5} className="px-4 py-8 text-center text-muted-foreground">No entries found</td></tr>
                ) : (
                  entries.map((e) => (
                    <tr key={e.id} className="border-b hover:bg-muted/30">
                      <td className="px-4 py-3 font-medium">{e.action}</td>
                      <td className="px-4 py-3 text-muted-foreground">
                        {e.target_type}{e.target_id ? ` / ${e.target_id.slice(0, 8)}...` : ""}
                      </td>
                      <td className="px-4 py-3 text-xs font-mono text-muted-foreground">
                        {e.admin_user_id?.slice(0, 8) ?? "—"}
                      </td>
                      <td className="px-4 py-3 text-xs text-muted-foreground">{e.ip_address ?? "—"}</td>
                      <td className="px-4 py-3 text-right text-xs text-muted-foreground">
                        {e.created_at ? new Date(e.created_at).toLocaleString() : "—"}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>

          <div className="flex items-center justify-between">
            <p className="text-sm text-muted-foreground">
              {total} entries · Page {currentPage} of {totalPages}
            </p>
            <div className="flex gap-2">
              <Button variant="outline" size="sm" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - limit))}>
                Previous
              </Button>
              <Button variant="outline" size="sm" disabled={offset + limit >= total} onClick={() => setOffset(offset + limit)}>
                Next
              </Button>
            </div>
          </div>
        </>
      )}
    </div>
  )
}