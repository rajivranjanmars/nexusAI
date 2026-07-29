import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { listAdminUsers, createAdminUser, updateAdminUser, deleteAdminUser, type AdminUserSummary, type AdminUserCreate, type AdminUserUpdate } from "@/api/admin"
import { ApiError } from "@/api/client"
import { Plus, X } from "lucide-react"

export default function Users() {
  const [users, setUsers] = useState<AdminUserSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState("")
  const [showCreate, setShowCreate] = useState(false)
  const [editing, setEditing] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)

  function loadUsers() {
    setLoading(true)
    listAdminUsers()
      .then(setUsers)
      .catch(() => setError("Failed to load users"))
      .finally(() => setLoading(false))
  }

  useEffect(() => { loadUsers() }, [])

  async function handleCreate(data: AdminUserCreate) {
    setSaving(true)
    setError("")
    try {
      await createAdminUser(data)
      setShowCreate(false)
      loadUsers()
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Create failed")
    } finally {
      setSaving(false)
    }
  }

  async function handleUpdate(userId: string, data: AdminUserUpdate) {
    setSaving(true)
    setError("")
    try {
      await updateAdminUser(userId, data)
      setEditing(null)
      loadUsers()
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Update failed")
    } finally {
      setSaving(false)
    }
  }

  async function handleDelete(userId: string) {
    if (!confirm("Deactivate this admin user?")) return
    setError("")
    try {
      await deleteAdminUser(userId)
      loadUsers()
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Delete failed")
    }
  }

  if (loading) return <p className="text-muted-foreground">Loading...</p>

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold tracking-tight">Admin Users</h1>
        <Button onClick={() => setShowCreate(true)}>
          <Plus className="h-4 w-4" /> Add User
        </Button>
      </div>

      {error && <p className="text-sm text-destructive">{error}</p>}

      <div className="rounded-md border">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b bg-muted/50">
              <th className="px-4 py-3 text-left font-medium">Name</th>
              <th className="px-4 py-3 text-left font-medium">Email</th>
              <th className="px-4 py-3 text-left font-medium">Role</th>
              <th className="px-4 py-3 text-left font-medium">Status</th>
              <th className="px-4 py-3 text-left font-medium">Last Login</th>
              <th className="px-4 py-3 text-right font-medium">Actions</th>
            </tr>
          </thead>
          <tbody>
            {users.map((u) => (
              <tr key={u.id} className="border-b hover:bg-muted/30">
                <td className="px-4 py-3 font-medium">{u.display_name}</td>
                <td className="px-4 py-3 text-muted-foreground">{u.email}</td>
                <td className="px-4 py-3">
                  <span className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${u.role === "super_admin" ? "bg-purple-100 text-purple-700" : "bg-blue-100 text-blue-700"}`}>
                    {u.role}
                  </span>
                </td>
                <td className="px-4 py-3">
                  <span className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${u.is_active ? "bg-green-100 text-green-700" : "bg-red-100 text-red-700"}`}>
                    {u.is_active ? "Active" : "Inactive"}
                  </span>
                </td>
                <td className="px-4 py-3 text-muted-foreground text-xs">
                  {u.last_login_at ? new Date(u.last_login_at).toLocaleDateString() : "Never"}
                </td>
                <td className="px-4 py-3 text-right">
                  <Button variant="outline" size="sm" onClick={() => setEditing(u.id)}>
                    Edit
                  </Button>{" "}
                  <Button variant="outline" size="sm" onClick={() => handleDelete(u.id)}>
                    Deactivate
                  </Button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {showCreate && (
        <UserForm
          title="Create Admin User"
          onSave={(data) => handleCreate(data as AdminUserCreate)}
          onClose={() => setShowCreate(false)}
          saving={saving}
        />
      )}

      {editing && (
        <UserForm
          title="Edit Admin User"
          initial={users.find((u) => u.id === editing)}
          onSave={(data) => handleUpdate(editing, data as AdminUserUpdate)}
          onClose={() => setEditing(null)}
          saving={saving}
          editing
        />
      )}
    </div>
  )
}

function UserForm({
  title,
  initial,
  onSave,
  onClose,
  saving,
  editing,
}: {
  title: string
  initial?: AdminUserSummary
  onSave: (data: AdminUserCreate | AdminUserUpdate) => Promise<void>
  onClose: () => void
  saving: boolean
  editing?: boolean
}) {
  const [email, setEmail] = useState(initial?.email ?? "")
  const [password, setPassword] = useState("")
  const [displayName, setDisplayName] = useState(initial?.display_name ?? "")
  const [role, setRole] = useState(initial?.role ?? "app_admin")
  const [appId, setAppId] = useState(initial?.app_id ?? "")
  const [isActive, setIsActive] = useState(initial?.is_active ?? true)

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (editing) {
      onSave({ display_name: displayName, role, app_id: appId || null, is_active: isActive })
    } else {
      onSave({ email, password, display_name: displayName, role, app_id: appId || null })
    }
  }

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle>{title}</CardTitle>
        <Button variant="ghost" size="icon" onClick={onClose}><X className="h-4 w-4" /></Button>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            {!editing && (
              <div className="space-y-2">
                <Label htmlFor="email">Email</Label>
                <Input id="email" type="email" value={email} onChange={(e) => setEmail(e.target.value)} required />
              </div>
            )}
            {!editing && (
              <div className="space-y-2">
                <Label htmlFor="password">Password</Label>
                <Input id="password" type="password" value={password} onChange={(e) => setPassword(e.target.value)} required={!editing} minLength={8} />
              </div>
            )}
            <div className="space-y-2">
              <Label htmlFor="displayName">Display Name</Label>
              <Input id="displayName" value={displayName} onChange={(e) => setDisplayName(e.target.value)} required />
            </div>
            <div className="space-y-2">
              <Label htmlFor="role">Role</Label>
              <select
                id="role"
                value={role}
                onChange={(e) => setRole(e.target.value)}
                className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm"
              >
                <option value="app_admin">App Admin</option>
                <option value="super_admin">Super Admin</option>
              </select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="appId">App ID (optional)</Label>
              <Input id="appId" value={appId} onChange={(e) => setAppId(e.target.value)} placeholder="Required for app_admin" />
            </div>
            {editing && (
              <div className="space-y-2 flex items-end pb-1">
                <label className="flex items-center gap-2 text-sm">
                  <input type="checkbox" checked={isActive} onChange={(e) => setIsActive(e.target.checked)} className="rounded" />
                  Active
                </label>
              </div>
            )}
          </div>
          <Button type="submit" disabled={saving}>
            {saving ? "Saving..." : "Save"}
          </Button>
        </form>
      </CardContent>
    </Card>
  )
}