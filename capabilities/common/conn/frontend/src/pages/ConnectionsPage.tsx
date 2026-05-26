import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { useSearchParams } from 'react-router-dom'
import { Header } from '@/components/layout/Header'
import { ConnectionCard } from '@/components/connections/ConnectionCard'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import {
  PlusIcon,
  FilterIcon,
  SearchIcon,
  RefreshCwIcon,
  GridIcon,
  ListIcon,
  AlertCircleIcon
} from 'lucide-react'
import { useConnections, useConnectionStats } from '@/hooks/useConnections'
import { useQueryClient } from '@tanstack/react-query'
import { Connection } from '@/api/connections'

type FilterType = 'all' | 'active' | 'inactive' | 'error' | 'testing'
type ViewMode = 'grid' | 'list'

export function ConnectionsPage() {
  const [searchParams] = useSearchParams()
  const queryClient = useQueryClient()
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedFilter, setSelectedFilter] = useState<FilterType>('all')
  const [viewMode, setViewMode] = useState<ViewMode>('grid')

  // Build query parameters for API call
  const queryParams = {
    search: searchTerm || undefined,
    status: selectedFilter !== 'all' ? selectedFilter as Connection['status'] : undefined,
    limit: 50,
  }

  // Fetch connections using real API
  const {
    data: connectionsData,
    isLoading,
    error,
    refetch
  } = useConnections(queryParams)

  const { data: stats } = useConnectionStats()

  const connections = connectionsData?.items || []
  const totalCount = connectionsData?.total || 0

  // Event handlers
  const handleNewConnection = () => {
    console.log('Creating new connection...')
    // Navigate to create connection modal/page
  }

  const handleRefresh = () => {
    refetch()
  }

  const handleEdit = (connection: Connection) => {
    console.log('Editing connection:', connection)
    // Navigate to edit connection modal/page
  }

  const handleDelete = (connection: Connection) => {
    console.log('Deleting connection:', connection)
    // Show delete confirmation modal
  }

  const handleTest = (connection: Connection) => {
    console.log('Testing connection:', connection)
    // Trigger connection test
  }

  const handleToggle = (connection: Connection) => {
    console.log('Toggling connection:', connection)
    // Toggle connection status
  }

  const handleView = (connection: Connection) => {
    console.log('Viewing connection:', connection)
    // Navigate to connection details page
  }

  const headerActions = (
    <div className="flex items-center space-x-3">
      {/* Search */}
      <div className="relative">
        <SearchIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
        <input
          type="text"
          placeholder="Search connections..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="pl-10 pr-4 py-2 w-64 text-sm input"
        />
      </div>

      {/* View Mode Toggle */}
      <div className="flex items-center space-x-1 bg-gray-100 dark:bg-gray-800 rounded-lg p-1">
        <Button
          variant={viewMode === 'grid' ? 'default' : 'ghost'}
          size="sm"
          icon={<GridIcon className="h-4 w-4" />}
          onClick={() => setViewMode('grid')}
        />
        <Button
          variant={viewMode === 'list' ? 'default' : 'ghost'}
          size="sm"
          icon={<ListIcon className="h-4 w-4" />}
          onClick={() => setViewMode('list')}
        />
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
        title="Connections"
        subtitle="Manage your data source and destination connections"
        actions={headerActions}
      />

      <motion.main
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className="flex-1 p-6 overflow-auto"
      >
        {/* Filters */}
        <div className="mb-6 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <span className="text-sm text-gray-600 dark:text-gray-400">Filter by status:</span>
            {(['all', 'active', 'inactive', 'error', 'testing'] as FilterType[]).map((filter) => (
              <Button
                key={filter}
                variant={selectedFilter === filter ? 'default' : 'outline'}
                size="sm"
                onClick={() => setSelectedFilter(filter)}
                className="capitalize"
              >
                {filter}
                {filter !== 'all' && stats && (
                  <Badge variant="secondary" size="sm" className="ml-2">
                    {stats.by_status[filter as keyof typeof stats.by_status] || 0}
                  </Badge>
                )}
              </Button>
            ))}
          </div>

          <div className="flex items-center space-x-4 text-sm text-gray-600 dark:text-gray-400">
            <span>
              Showing {connections.length} of {totalCount} connections
            </span>
          </div>
        </div>

        {/* Loading State */}
        {isLoading && (
          <div className="flex items-center justify-center py-12">
            <div className="loading-spinner h-8 w-8" />
            <span className="ml-3 text-gray-600 dark:text-gray-400">Loading connections...</span>
          </div>
        )}

        {/* Error State */}
        {error && (
          <div className="flex items-center justify-center py-12">
            <div className="text-center">
              <AlertCircleIcon className="h-12 w-12 text-danger-500 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                Failed to load connections
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

        {/* Connections Grid */}
        {!isLoading && !error && (
          <div className={
            viewMode === 'grid'
              ? 'grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6'
              : 'space-y-4'
          }>
            {connections.map((connection) => (
              <ConnectionCard
                key={connection.id}
                connection={connection}
                onEdit={handleEdit}
                onDelete={handleDelete}
                onTest={handleTest}
                onToggle={handleToggle}
                onView={handleView}
              />
            ))}
          </div>
        )}

        {/* Empty State */}
        {!isLoading && !error && connections.length === 0 && (
          <div className="text-center py-12">
            <div className="w-16 h-16 bg-gray-100 dark:bg-gray-800 rounded-full flex items-center justify-center mx-auto mb-4">
              <FilterIcon className="h-8 w-8 text-gray-400" />
            </div>
            <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
              No connections found
            </h3>
            <p className="text-gray-600 dark:text-gray-400 mb-6">
              {searchTerm || selectedFilter !== 'all'
                ? 'Try adjusting your search or filter criteria.'
                : 'Get started by creating your first connection.'
              }
            </p>
            <Button
              variant="default"
              icon={<PlusIcon className="h-4 w-4" />}
              onClick={handleNewConnection}
            >
              Create Connection
            </Button>
          </div>
        )}
      </motion.main>
    </>
  )
}