import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom"
import { AuthProvider, useAuth } from "@/contexts/AuthContext"
import AdminLayout from "@/components/layout/AdminLayout"
import LoginPage from "@/pages/Login"
import Dashboard from "@/pages/Dashboard"
import Apps from "@/pages/Apps"
import Onboarding from "@/pages/Onboarding"
import Users from "@/pages/Users"
import Analytics from "@/pages/Analytics"
import AuditLog from "@/pages/AuditLog"
import Knowledge from "@/pages/Knowledge"
import AppConfig from "@/pages/AppConfig"
import AppSettings from "@/pages/AppSettings"

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { user, loading } = useAuth()
  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <p className="text-muted-foreground">Loading...</p>
      </div>
    )
  }
  if (!user) return <Navigate to="/login" replace />
  return <>{children}</>
}

function RoleRoute({ role, children }: { role: "super_admin" | "app_admin"; children: React.ReactNode }) {
  const { user } = useAuth()
  if (user?.admin_role !== role) return <Navigate to="/" replace />
  return <>{children}</>
}

function AppRoutes() {
  const { user } = useAuth()

  return (
    <Routes>
      <Route path="/login" element={user ? <Navigate to="/" replace /> : <LoginPage />} />
      <Route
        element={
          <ProtectedRoute>
            <AdminLayout />
          </ProtectedRoute>
        }
      >
        {/* Shared */}
        <Route index element={<Dashboard />} />
        <Route path="analytics" element={<Analytics />} />

        {/* Super Admin only */}
        <Route
          path="apps"
          element={
            <RoleRoute role="super_admin">
              <Apps />
            </RoleRoute>
          }
        />
        <Route
          path="onboarding"
          element={
            <RoleRoute role="super_admin">
              <Onboarding />
            </RoleRoute>
          }
        />
        <Route
          path="users"
          element={
            <RoleRoute role="super_admin">
              <Users />
            </RoleRoute>
          }
        />
        <Route
          path="audit-log"
          element={
            <RoleRoute role="super_admin">
              <AuditLog />
            </RoleRoute>
          }
        />

        {/* App Admin only */}
        <Route
          path="config"
          element={
            <RoleRoute role="app_admin">
              <AppConfig />
            </RoleRoute>
          }
        />
        <Route
          path="knowledge"
          element={
            <RoleRoute role="app_admin">
              <Knowledge />
            </RoleRoute>
          }
        />
        <Route
          path="settings"
          element={
            <RoleRoute role="app_admin">
              <AppSettings />
            </RoleRoute>
          }
        />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <AppRoutes />
      </AuthProvider>
    </BrowserRouter>
  )
}