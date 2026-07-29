import { NavLink } from "react-router-dom"
import { useAuth } from "@/contexts/AuthContext"
import { cn } from "@/lib/utils"
import { Separator } from "@/components/ui/separator"
import {
  LayoutDashboard,
  AppWindow,
  Plus,
  Users,
  BarChart3,
  ScrollText,
  Settings,
  Database,
  SlidersHorizontal,
  type LucideIcon,
} from "lucide-react"

interface NavItem {
  label: string
  href: string
  icon: LucideIcon
  roles: Array<"super_admin" | "app_admin">
}

const navItems: NavItem[] = [
  { label: "Dashboard", href: "/", icon: LayoutDashboard, roles: ["super_admin", "app_admin"] },
  { label: "Apps", href: "/apps", icon: AppWindow, roles: ["super_admin"] },
  { label: "Onboard App", href: "/onboarding", icon: Plus, roles: ["super_admin"] },
  { label: "Admin Users", href: "/users", icon: Users, roles: ["super_admin"] },
  { label: "Analytics", href: "/analytics", icon: BarChart3, roles: ["super_admin", "app_admin"] },
  { label: "Config", href: "/config", icon: SlidersHorizontal, roles: ["app_admin"] },
  { label: "Knowledge Base", href: "/knowledge", icon: Database, roles: ["app_admin"] },
  { label: "Audit Log", href: "/audit-log", icon: ScrollText, roles: ["super_admin"] },
  { label: "Settings", href: "/settings", icon: Settings, roles: ["app_admin"] },
]

export default function Sidebar() {
  const { user } = useAuth()

  const visibleItems = navItems.filter(
    (item) => user && item.roles.includes(user.admin_role),
  )

  return (
    <aside className="fixed left-0 top-0 z-40 h-screen w-56 border-r bg-background">
      <div className="flex h-14 items-center border-b px-4">
        <span className="font-semibold text-lg">LPU Nexus</span>
        {user && (
          <span className="ml-2 rounded bg-primary/10 px-1.5 py-0.5 text-[10px] font-medium text-primary">
            {user.admin_role === "super_admin" ? "Super Admin" : "App Admin"}
          </span>
        )}
      </div>
      <nav className="flex flex-col gap-1 p-3">
        {visibleItems.map((item) => (
          <NavLink
            key={item.href}
            to={item.href}
            end={item.href === "/"}
            className={({ isActive }) =>
              cn(
                "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                isActive
                  ? "bg-primary/10 text-primary"
                  : "text-muted-foreground hover:bg-accent hover:text-accent-foreground",
              )
            }
          >
            <item.icon className="h-4 w-4" />
            {item.label}
          </NavLink>
        ))}
      </nav>
      <Separator />
    </aside>
  )
}