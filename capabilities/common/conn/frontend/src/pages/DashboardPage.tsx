import React from 'react'
import { motion } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import { Header } from '@/components/layout/Header'
import { DashboardOverview } from '@/components/dashboard/DashboardOverview'
import { Button } from '@/components/ui/Button'
import { PlusIcon, RefreshCwIcon } from 'lucide-react'
import { useConnectionStats } from '@/hooks/useConnections'
import { useFlowStats } from '@/hooks/useFlows'
import { useLineageStats } from '@/hooks/useLineage'
import { useQueryClient } from '@tanstack/react-query'

export function DashboardPage() {
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  // Fetch dashboard statistics
  const { data: connectionStats, isLoading: loadingConnections } = useConnectionStats()
  const { data: flowStats, isLoading: loadingFlows } = useFlowStats()
  const { data: lineageStats, isLoading: loadingLineage } = useLineageStats()

  const handleRefresh = () => {
    // Invalidate all dashboard-related queries to force refresh
    queryClient.invalidateQueries({ queryKey: ['connections', 'stats'] })
    queryClient.invalidateQueries({ queryKey: ['flows', 'stats'] })
    queryClient.invalidateQueries({ queryKey: ['lineage', 'stats'] })
  }

  const handleNewConnection = () => {
    navigate('/connections?action=create')
  }

  const isLoading = loadingConnections || loadingFlows || loadingLineage

  const headerActions = (
    <div className="flex items-center space-x-3">
      <Button
        variant="outline"
        size="sm"
        icon={<RefreshCwIcon className="h-4 w-4" />}
        onClick={handleRefresh}
        loading={isLoading}
        disabled={isLoading}
      >
        Refresh
      </Button>
      <Button
        variant="default"
        size="sm"
        icon={<PlusIcon className="h-4 w-4" />}
        onClick={handleNewConnection}
      >
        New Connection
      </Button>
    </div>
  )

  return (
    <>
      <Header
        title="Dashboard"
        subtitle="Overview of your data integration platform"
        actions={headerActions}
      />

      <motion.main
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className="flex-1 p-6 overflow-auto"
      >
        <DashboardOverview
          connectionStats={connectionStats}
          flowStats={flowStats}
          lineageStats={lineageStats}
          isLoading={isLoading}
        />
      </motion.main>
    </>
  )
}