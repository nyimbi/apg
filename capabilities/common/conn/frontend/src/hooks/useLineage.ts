/**
 * APG Connection Management - Lineage Hooks
 * React Query hooks for data lineage management
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useState, useEffect } from 'react'
import { lineageApi } from '@/api/lineage'
import toast from 'react-hot-toast'

// Query Keys
export const lineageKeys = {
  all: ['lineage'] as const,
  visualization: (params?: any) => [...lineageKeys.all, 'visualization', params] as const,
  upstream: (nodeId: string, maxDepth?: number) => [...lineageKeys.all, 'upstream', nodeId, maxDepth] as const,
  downstream: (nodeId: string, maxDepth?: number) => [...lineageKeys.all, 'downstream', nodeId, maxDepth] as const,
  impact: (nodeId: string) => [...lineageKeys.all, 'impact', nodeId] as const,
  catalog: () => [...lineageKeys.all, 'catalog'] as const,
  search: (query: string, params?: any) => [...lineageKeys.all, 'search', query, params] as const,
  cycles: () => [...lineageKeys.all, 'cycles'] as const,
  rootSources: () => [...lineageKeys.all, 'root-sources'] as const,
  leafDestinations: () => [...lineageKeys.all, 'leaf-destinations'] as const,
  stats: () => [...lineageKeys.all, 'stats'] as const,
  nodeDetails: (nodeId: string) => [...lineageKeys.all, 'node-details', nodeId] as const,
}

// Hooks
export function useLineageVisualization(params: {
  node_id?: string
  visualization_type?: 'full' | 'upstream' | 'downstream' | 'impact'
} = {}) {
  return useQuery({
    queryKey: lineageKeys.visualization(params),
    queryFn: () => lineageApi.generateVisualization(params),
    staleTime: 300000, // 5 minutes
    cacheTime: 600000, // 10 minutes
  })
}

export function useUpstreamLineage(nodeId: string, maxDepth = 10) {
  return useQuery({
    queryKey: lineageKeys.upstream(nodeId, maxDepth),
    queryFn: () => lineageApi.getUpstream(nodeId, maxDepth),
    enabled: !!nodeId,
    staleTime: 300000, // 5 minutes
  })
}

export function useDownstreamLineage(nodeId: string, maxDepth = 10) {
  return useQuery({
    queryKey: lineageKeys.downstream(nodeId, maxDepth),
    queryFn: () => lineageApi.getDownstream(nodeId, maxDepth),
    enabled: !!nodeId,
    staleTime: 300000, // 5 minutes
  })
}

export function useImpactAnalysis(nodeId: string) {
  return useQuery({
    queryKey: lineageKeys.impact(nodeId),
    queryFn: () => lineageApi.analyzeImpact(nodeId),
    enabled: !!nodeId,
    staleTime: 300000, // 5 minutes
  })
}

export function useDataCatalog() {
  return useQuery({
    queryKey: lineageKeys.catalog(),
    queryFn: () => lineageApi.getDataCatalog(),
    staleTime: 600000, // 10 minutes
    cacheTime: 1200000, // 20 minutes
  })
}

export function useLineageSearch(
  query: string,
  params?: {
    search_type?: 'all' | 'entities' | 'fields' | 'flows'
    filters?: {
      types?: string[]
      connections?: string[]
      tags?: string[]
    }
    limit?: number
    offset?: number
  }
) {
  return useQuery({
    queryKey: lineageKeys.search(query, params),
    queryFn: () => lineageApi.search({ query, ...params }),
    enabled: !!query && query.length > 2, // Only search when query is at least 3 characters
    staleTime: 300000, // 5 minutes
  })
}

export function useLineageCycles() {
  return useQuery({
    queryKey: lineageKeys.cycles(),
    queryFn: () => lineageApi.detectCycles(),
    staleTime: 600000, // 10 minutes
  })
}

export function useRootSources() {
  return useQuery({
    queryKey: lineageKeys.rootSources(),
    queryFn: () => lineageApi.getRootSources(),
    staleTime: 600000, // 10 minutes
  })
}

export function useLeafDestinations() {
  return useQuery({
    queryKey: lineageKeys.leafDestinations(),
    queryFn: () => lineageApi.getLeafDestinations(),
    staleTime: 600000, // 10 minutes
  })
}

export function useLineageStats() {
  return useQuery({
    queryKey: lineageKeys.stats(),
    queryFn: () => lineageApi.getStats(),
    refetchInterval: 300000, // 5 minutes
    staleTime: 180000, // 3 minutes
  })
}

export function useNodeDetails(nodeId: string) {
  return useQuery({
    queryKey: lineageKeys.nodeDetails(nodeId),
    queryFn: () => lineageApi.getNodeDetails(nodeId),
    enabled: !!nodeId,
    staleTime: 300000, // 5 minutes
  })
}

// Mutations
export function useTrackConnection() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (data: {
      connection_id: string
      connection_name: string
      connection_type: string
      schema_info: Record<string, any>
    }) => lineageApi.trackConnection(data),
    onSuccess: () => {
      // Invalidate lineage data to refresh visualization
      queryClient.invalidateQueries({ queryKey: lineageKeys.all })
      toast.success('Connection tracked in lineage successfully!')
    },
    onError: (error: any) => {
      toast.error(error.message || 'Failed to track connection in lineage')
    }
  })
}

export function useTrackFlow() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (data: {
      flow_id: string
      flow_name: string
      source_connection_id: string
      target_connection_id: string
      transformations: Array<Record<string, any>>
      field_mappings: Record<string, string>
    }) => lineageApi.trackFlow(data),
    onSuccess: () => {
      // Invalidate lineage data to refresh visualization
      queryClient.invalidateQueries({ queryKey: lineageKeys.all })
      toast.success('Flow tracked in lineage successfully!')
    },
    onError: (error: any) => {
      toast.error(error.message || 'Failed to track flow in lineage')
    }
  })
}

export function useExportLineage() {
  return useMutation({
    mutationFn: (params: {
      format: 'json' | 'csv' | 'graphml'
      node_id?: string
      visualization_type?: 'full' | 'upstream' | 'downstream'
      include_metadata?: boolean
    }) => lineageApi.exportLineage(params),
    onSuccess: (blob, variables) => {
      // Create download link
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `lineage.${variables.format}`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      URL.revokeObjectURL(url)

      toast.success('Lineage exported successfully!')
    },
    onError: (error: any) => {
      toast.error(error.message || 'Failed to export lineage')
    }
  })
}

// Custom hooks for common lineage operations
export function useRefreshLineage() {
  const queryClient = useQueryClient()

  return () => {
    queryClient.invalidateQueries({ queryKey: lineageKeys.all })
    toast.success('Lineage data refreshed!')
  }
}

export function useLineageNodeSelection() {
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null)

  const nodeDetails = useNodeDetails(selectedNodeId || '')
  const upstreamLineage = useUpstreamLineage(selectedNodeId || '', 5)
  const downstreamLineage = useDownstreamLineage(selectedNodeId || '', 5)
  const impactAnalysis = useImpactAnalysis(selectedNodeId || '')

  const selectNode = (nodeId: string | null) => {
    setSelectedNodeId(nodeId)
  }

  return {
    selectedNodeId,
    selectNode,
    nodeDetails: nodeDetails.data,
    upstreamLineage: upstreamLineage.data,
    downstreamLineage: downstreamLineage.data,
    impactAnalysis: impactAnalysis.data,
    isLoading: nodeDetails.isLoading || upstreamLineage.isLoading ||
               downstreamLineage.isLoading || impactAnalysis.isLoading,
  }
}

// Hook for lineage search with debouncing
export function useDebouncedLineageSearch(
  query: string,
  delay = 500,
  params?: {
    search_type?: 'all' | 'entities' | 'fields' | 'flows'
    filters?: {
      types?: string[]
      connections?: string[]
      tags?: string[]
    }
  }
) {
  const [debouncedQuery, setDebouncedQuery] = useState('')

  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedQuery(query)
    }, delay)

    return () => {
      clearTimeout(handler)
    }
  }, [query, delay])

  return useLineageSearch(debouncedQuery, params)
}