import { Outlet } from "react-router-dom"
import Sidebar from "./Sidebar"
import Header from "./Header"

export default function AdminLayout() {
  return (
    <div className="min-h-screen bg-background">
      <Sidebar />
      <div className="ml-56">
        <Header />
        <main className="p-6">
          <Outlet />
        </main>
      </div>
    </div>
  )
}