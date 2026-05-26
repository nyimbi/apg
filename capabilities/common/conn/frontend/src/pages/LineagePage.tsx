import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Header } from '@/components/layout/Header'
import { LineageGraph } from '@/components/lineage/LineageGraph'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import {
  GitBranchIcon,
  DownloadIcon,
  FilterIcon,
  RefreshCwIcon,
  EyeIcon,
  MapIcon,
  AlertCircleIcon
} from 'lucide-react'
import {
  useLineageVisualization,
  useLineageStats,
  useExportLineage,
  useRefreshLineage
} from '@/hooks/useLineage'
import { useQueryClient } from '@tanstack/react-query'

// Mock lineage data - in production, this would come from API
const mockLineageData = {
  nodes: [
    {
      id: 'postgres-prod',
      label: 'Production PostgreSQL',
      type: 'connection',
      source_type: 'database',
      metadata: {
        description: 'Main production database',
        sensitive: false,
        pii: false,
      }
    },
    {
      id: 'users-table',
      label: 'users',
      type: 'entity',
      source_type: 'table',
      metadata: {
        description: 'User accounts table',
        sensitive: true,
        pii: true,
      }
    },
    {
      id: 'user-id-field',
      label: 'users.id',
      type: 'field',
      source_type: 'integer',
      metadata: {
        description: 'User ID primary key',
        sensitive: false,
        pii: false,
      }
    },
    {
      id: 'user-email-field',
      label: 'users.email',
      type: 'field',
      source_type: 'string',
      metadata: {
        description: 'User email address',
        sensitive: true,
        pii: true,
      }
    },
    {
      id: 'etl-flow',
      label: 'User ETL Pipeline',
      type: 'flow',
      source_type: 'flow',
      metadata: {
        description: 'Daily user data processing',
        sensitive: false,
        pii: false,
      }
    },
    {
      id: 'snowflake-dw',
      label: 'Snowflake DW',
      type: 'connection',
      source_type: 'warehouse',
      metadata: {
        description: 'Data warehouse',
        sensitive: false,
        pii: false,
      }
    },
    {
      id: 'dim-users',
      label: 'dim_users',
      type: 'entity',
      source_type: 'table',
      metadata: {
        description: 'User dimension table',
        sensitive: true,
        pii: true,
      }
    }
  ],
  edges: [
    {
      id: 'edge-1',
      source: 'postgres-prod',
      target: 'users-table',
      label: 'contains',
      type: 'contains'
    },
    {
      id: 'edge-2',
      source: 'users-table',
      target: 'user-id-field',
      label: 'contains',
      type: 'contains'
    },
    {
      id: 'edge-3',
      source: 'users-table',
      target: 'user-email-field',
      label: 'contains',
      type: 'contains'
    },
    {
      id: 'edge-4',
      source: 'users-table',
      target: 'etl-flow',
      label: 'feeds into',
      type: 'derives_from'
    },
    {
      id: 'edge-5',
      source: 'etl-flow',
      target: 'dim-users',
      label: 'creates',
      type: 'maps_to'
    },
    {
      id: 'edge-6',
      source: 'dim-users',
      target: 'snowflake-dw',
      label: 'stored in',
      type: 'contains'
    }
  ],
  summary: {
    total_nodes: 7,
    total_edges: 6,
    node_types: {
      connection: 2,
      entity: 2,
      field: 2,
      flow: 1
    },
    sensitive_data_nodes: 3
  }
}

type VisualizationType = 'full' | 'upstream' | 'downstream' | 'impact'

export function LineagePage() {
  const queryClient = useQueryClient()
  const [selectedView, setSelectedView] = useState<VisualizationType>('full')
  const [selectedNode, setSelectedNode] = useState<string | null>(null)

  // Fetch lineage data using real API
  const {
    data: lineageData,
    isLoading,
    error,
    refetch
  } = useLineageVisualization({
    node_id: selectedNode || undefined,
    visualization_type: selectedView
  })

  const { data: stats } = useLineageStats()
  const exportLineage = useExportLineage()
  const refreshLineage = useRefreshLineage()

  const handleRefresh = () => {
    refreshLineage()
    refetch()
  }

  const handleExport = async (format: 'json' | 'csv' | 'graphml') => {
    try {
      await exportLineage.mutateAsync({
        format,
        node_id: selectedNode || undefined,
        visualization_type: selectedView,
        include_metadata: true
      })
    } catch (error) {
      console.error('Export failed:', error)
    }
  }

  const handleNodeClick = (nodeId: string, node: any) => {
    setSelectedNode(nodeId === selectedNode ? null : nodeId)
    console.log('Selected node:', nodeId, node)
  }

  const handleEdgeClick = (edgeId: string, edge: any) => {
    console.log('Selected edge:', edgeId, edge)
  }

  const headerActions = (
    <div className="flex items-center space-x-3">
      {/* View Options */}
      <div className="flex items-center space-x-1 bg-gray-100 dark:bg-gray-800 rounded-lg p-1">
        {[
          { key: 'full', label: 'Full', icon: <MapIcon className="h-3 w-3" /> },
          { key: 'upstream', label: 'Upstream', icon: <GitBranchIcon className="h-3 w-3" /> },
          { key: 'downstream', label: 'Downstream', icon: <GitBranchIcon className="h-3 w-3 rotate-180" /> },
          { key: 'impact', label: 'Impact', icon: <EyeIcon className="h-3 w-3" /> },
        ].map((view) => (
          <Button
            key={view.key}
            variant={selectedView === view.key ? 'default' : 'ghost'}
            size="sm"
            icon={view.icon}
            onClick={() => setSelectedView(view.key as VisualizationType)}
          >
            {view.label}
          </Button>
        ))}
      </div>

      <div className="w-px h-6 bg-gray-300 dark:bg-gray-600" />

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
        variant="outline"
        size="sm"
        icon={<DownloadIcon className="h-4 w-4" />}
        onClick={() => handleExport('json')}
        loading={exportLineage.isPending}
        disabled={exportLineage.isPending || !lineageData}
      >
        Export
      </Button>
    </div>
  )

  return (
    <>
      <Header
        title="Data Lineage"
        subtitle="Visualize and analyze data flow relationships across your organization"
        actions={headerActions}
      />

      <motion.main
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className="flex-1 flex flex-col min-h-0"
      >
        {/* Statistics Bar */}
        <div className="px-6 py-4 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-6">
              <div className="flex items-center space-x-2">
                <span className="text-sm text-gray-600 dark:text-gray-400">Total Nodes:</span>
                <Badge variant="secondary">
                  {lineageData?.summary?.total_nodes || stats?.total_nodes || 0}
                </Badge>
              </div>
              <div className="flex items-center space-x-2">
                <span className="text-sm text-gray-600 dark:text-gray-400">Relationships:</span>
                <Badge variant="secondary">
                  {lineageData?.summary?.total_edges || stats?.total_relationships || 0}
                </Badge>
              </div>
              <div className="flex items-center space-x-2">
                <span className="text-sm text-gray-600 dark:text-gray-400">Sensitive Data:</span>
                <Badge variant="destructive">
                  {lineageData?.summary?.sensitive_data_nodes || stats?.sensitive_entities || 0}
                </Badge>
              </div>
            </div>

            <div className="flex items-center space-x-4">
              {Object.entries(mockLineageData.summary.node_types).map(([type, count]) => (
                <div key={type} className="flex items-center space-x-2">
                  <div className={`w-3 h-3 rounded-full ${
                    type === 'connection' ? 'bg-primary-500' :
                    type === 'entity' ? 'bg-success-500' :
                    type === 'field' ? 'bg-warning-500' :
                    'bg-secondary-500'
                  }`} />
                  <span className="text-sm text-gray-600 dark:text-gray-400 capitalize">
                    {type}: {count}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Lineage Visualization */}
        <div className="flex-1 p-6">
          <div className="h-full">
            {/* Loading State */}
            {isLoading && (
              <div className="flex items-center justify-center h-full">
                <div className="flex flex-col items-center space-y-4">
                  <div className="loading-spinner h-8 w-8" />
                  <p className="text-gray-600 dark:text-gray-400">Loading lineage data...</p>
                </div>
              </div>
            )}

            {/* Error State */}
            {error && !isLoading && (
              <div className="flex items-center justify-center h-full">
                <div className="text-center">
                  <AlertCircleIcon className="h-12 w-12 text-danger-500 mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                    Failed to load lineage data
                  </h3>
                  <p className="text-gray-600 dark:text-gray-400 mb-4">
                    {error.message || 'An unexpected error occurred'}
                  </p>
                  <Button variant="outline" onClick={handleRefresh}>
                    Try Again
                  </Button>
                </div>
              </div>
            )}

            {/* Empty State */}
            {!isLoading && !error && (!lineageData || lineageData.nodes.length === 0) && (
              <div className="flex items-center justify-center h-full">
                <div className="text-center">
                  <GitBranchIcon className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                    No lineage data available
                  </h3>
                  <p className="text-gray-600 dark:text-gray-400 mb-4">
                    Start by creating connections and flows to visualize data lineage.
                  </p>
                </div>
              </div>
            )}

            {/* Lineage Graph */}
            {!isLoading && !error && lineageData && lineageData.nodes.length > 0 && (
              <LineageGraph
                lineageData={lineageData}
                onNodeClick={handleNodeClick}
                onEdgeClick={handleEdgeClick}
                className="w-full h-full"
              />
            )}
          </div>
        </div>
      </motion.main>
    </>
  )
}