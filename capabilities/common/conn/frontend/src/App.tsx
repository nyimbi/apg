import React, { useState } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { AnimatePresence } from 'framer-motion'
import { Sidebar } from '@/components/layout/Sidebar'
import { Header } from '@/components/layout/Header'
import { DashboardPage } from '@/pages/DashboardPage'
import { ConnectionsPage } from '@/pages/ConnectionsPage'
import { FlowsPage } from '@/pages/FlowsPage'
import { DesignerPage } from '@/pages/DesignerPage'
import { LineagePage } from '@/pages/LineagePage'
import { TapsPage } from '@/pages/TapsPage'
import { AnalyticsPage } from '@/pages/AnalyticsPage'
import { SearchPage } from '@/pages/SearchPage'
import { SettingsPage } from '@/pages/SettingsPage'
import { LoginPage } from '@/pages/LoginPage'
import { useAuth, ProtectedRoute } from '@/providers/AuthProvider'

// Main App Layout Component
function AppLayout() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)

  const toggleSidebar = () => {
    setSidebarCollapsed(!sidebarCollapsed)
  }

  return (
    <div className="flex h-screen bg-gray-50 dark:bg-gray-900">
      {/* Sidebar */}
      <Sidebar isCollapsed={sidebarCollapsed} onToggle={toggleSidebar} />

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Routes with page-specific headers and content */}
        <AnimatePresence mode="wait">
          <Routes>
            <Route path="/" element={<DashboardPage />} />
            <Route path="/connections" element={<ConnectionsPage />} />
            <Route path="/flows" element={<FlowsPage />} />
            <Route path="/designer" element={<DesignerPage />} />
            <Route path="/lineage" element={<LineagePage />} />
            <Route path="/taps" element={<TapsPage />} />
            <Route path="/analytics" element={<AnalyticsPage />} />
            <Route path="/search" element={<SearchPage />} />
            <Route path="/settings" element={<SettingsPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </AnimatePresence>
      </div>
    </div>
  )
}

export default function App() {
  const { isAuthenticated, isLoading } = useAuth()

  // Show loading spinner while checking auth
  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-50 dark:bg-gray-900">
        <div className="flex flex-col items-center space-y-4">
          <div className="loading-spinner h-8 w-8" />
          <p className="text-gray-600 dark:text-gray-400">Loading...</p>
        </div>
      </div>
    )
  }

  return (
    <Routes>
      {/* Public Routes */}
      <Route
        path="/login"
        element={
          isAuthenticated ? <Navigate to="/" replace /> : <LoginPage />
        }
      />

      {/* Protected Routes */}
      <Route
        path="/*"
        element={
          <ProtectedRoute>
            <AppLayout />
          </ProtectedRoute>
        }
      />
    </Routes>
  )
}