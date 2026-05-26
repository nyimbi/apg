/**
 * APG Connection Management - Flows Hooks
 * React Query hooks for flow management
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { flowsApi, DataFlow, CreateFlowRequest } from '@/api/flows'
import toast from 'react-hot-toast'

// Query Keys
export const flowKeys = {
  all: ['flows'] as const,
  lists: () => [...flowKeys.all, 'list'] as const,
  list: (params?: any) => [...flowKeys.lists(), params] as const,
  details: () => [...flowKeys.all, 'detail'] as const,
  detail: (id: string) => [...flowKeys.details(), id] as const,
  executions: (id: string) => [...flowKeys.detail(id), 'executions'] as const,
  execution: (flowId: string, executionId: string) => [...flowKeys.executions(flowId), executionId] as const,
  metrics: (id: string) => [...flowKeys.detail(id), 'metrics'] as const,
  logs: (id: string) => [...flowKeys.detail(id), 'logs'] as const,
  stats: () => [...flowKeys.all, 'stats'] as const,
}

// Hooks
export function useFlows(params?: {
  status?: DataFlow['status']
  source_connection_id?: string
  target_connection_id?: string
  search?: string
  tags?: string[]
  enabled?: boolean
  limit?: number
  offset?: number
}) {
  return useQuery({
    queryKey: flowKeys.list(params),
    queryFn: () => flowsApi.list(params),
    keepPreviousData: true,
    staleTime: 30000, // 30 seconds
  })
}

export function useFlow(id: string) {
  return useQuery({
    queryKey: flowKeys.detail(id),
    queryFn: () => flowsApi.get(id),
    enabled: !!id,
    staleTime: 60000, // 1 minute
  })
}

export function useFlowExecutions(
  id: string,
  params?: {
    status?: string
    limit?: number
    offset?: number
  }
) {
  return useQuery({
    queryKey: [...flowKeys.executions(id), params],
    queryFn: () => flowsApi.getExecutions(id, params),
    enabled: !!id,
    refetchInterval: (data) => {
      // Refetch more frequently if there are running executions
      const hasRunning = data?.items.some(exec => exec.status === 'running')
      return hasRunning ? 5000 : 30000 // 5s if running, 30s otherwise
    },
    staleTime: 15000, // 15 seconds
  })
}

export function useFlowExecution(flowId: string, executionId: string) {
  return useQuery({
    queryKey: flowKeys.execution(flowId, executionId),
    queryFn: () => flowsApi.getExecution(flowId, executionId),
    enabled: !!flowId && !!executionId,
    refetchInterval: (data) => {
      // Refetch while execution is running
      return data?.status === 'running' ? 2000 : false
    },
    staleTime: 10000, // 10 seconds
  })
}

export function useFlowMetrics(id: string, period: '24h' | '7d' | '30d' = '24h') {
  return useQuery({
    queryKey: [...flowKeys.metrics(id), period],
    queryFn: () => flowsApi.getMetrics(id, period),
    enabled: !!id,
    staleTime: 300000, // 5 minutes
  })
}

export function useFlowLogs(
  id: string,
  params?: {
    execution_id?: string
    level?: 'info' | 'warning' | 'error'
    limit?: number
    offset?: number
  }
) {
  return useQuery({
    queryKey: [...flowKeys.logs(id), params],
    queryFn: () => flowsApi.getLogs(id, params),
    enabled: !!id,
    refetchInterval: 30000, // 30 seconds
    staleTime: 15000, // 15 seconds
  })
}

export function useFlowStats() {
  return useQuery({
    queryKey: flowKeys.stats(),
    queryFn: () => flowsApi.getStats(),
    refetchInterval: 60000, // Refetch every minute
    staleTime: 30000, // 30 seconds
  })
}

export function useAvailableStreams(sourceConnectionId: string) {
  return useQuery({
    queryKey: ['flows', 'streams', sourceConnectionId],
    queryFn: () => flowsApi.getAvailableStreams(sourceConnectionId),
    enabled: !!sourceConnectionId,
    staleTime: 300000, // 5 minutes
  })
}

// Mutations
export function useCreateFlow() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (data: CreateFlowRequest) => flowsApi.create(data),
    onSuccess: (newFlow) => {
      // Invalidate and refetch flows list
      queryClient.invalidateQueries({ queryKey: flowKeys.lists() })
      queryClient.invalidateQueries({ queryKey: flowKeys.stats() })

      // Add the new flow to the cache
      queryClient.setQueryData(flowKeys.detail(newFlow.id), newFlow)

      toast.success(`Flow "${newFlow.name}" created successfully!`)
    },
    onError: (error: any) => {
      toast.error(error.message || 'Failed to create flow')
    }
  })
}

export function useUpdateFlow() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({ id, data }: { id: string; data: Partial<CreateFlowRequest> }) =>
      flowsApi.update(id, data),
    onSuccess: (updatedFlow) => {
      // Update the flow in cache
      queryClient.setQueryData(flowKeys.detail(updatedFlow.id), updatedFlow)

      // Invalidate lists to ensure consistency
      queryClient.invalidateQueries({ queryKey: flowKeys.lists() })

      toast.success(`Flow "${updatedFlow.name}" updated successfully!`)
    },
    onError: (error: any) => {
      toast.error(error.message || 'Failed to update flow')
    }
  })
}

export function useDeleteFlow() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (id: string) => flowsApi.delete(id),
    onSuccess: (_, deletedId) => {
      // Remove from cache
      queryClient.removeQueries({ queryKey: flowKeys.detail(deletedId) })
      queryClient.removeQueries({ queryKey: flowKeys.executions(deletedId) })
      queryClient.removeQueries({ queryKey: flowKeys.metrics(deletedId) })
      queryClient.removeQueries({ queryKey: flowKeys.logs(deletedId) })

      // Invalidate lists
      queryClient.invalidateQueries({ queryKey: flowKeys.lists() })
      queryClient.invalidateQueries({ queryKey: flowKeys.stats() })

      toast.success('Flow deleted successfully!')
    },
    onError: (error: any) => {
      toast.error(error.message || 'Failed to delete flow')
    }
  })
}

export function useStartFlow() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (id: string) => flowsApi.start(id),
    onSuccess: (execution, id) => {
      // Invalidate flow to update status
      queryClient.invalidateQueries({ queryKey: flowKeys.detail(id) })
      queryClient.invalidateQueries({ queryKey: flowKeys.executions(id) })
      queryClient.invalidateQueries({ queryKey: flowKeys.lists() })

      toast.success('Flow started successfully!')
    },
    onError: (error: any) => {
      toast.error(error.message || 'Failed to start flow')
    }
  })
}

export function useStopFlow() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (id: string) => flowsApi.stop(id),
    onSuccess: (_, id) => {
      // Invalidate flow to update status
      queryClient.invalidateQueries({ queryKey: flowKeys.detail(id) })
      queryClient.invalidateQueries({ queryKey: flowKeys.executions(id) })
      queryClient.invalidateQueries({ queryKey: flowKeys.lists() })

      toast.success('Flow stopped successfully!')
    },
    onError: (error: any) => {
      toast.error(error.message || 'Failed to stop flow')
    }
  })
}

export function useExecuteFlowOnce() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (id: string) => flowsApi.executeOnce(id),
    onSuccess: (execution, id) => {
      // Invalidate executions to show new execution
      queryClient.invalidateQueries({ queryKey: flowKeys.executions(id) })
      queryClient.invalidateQueries({ queryKey: flowKeys.detail(id) })

      toast.success('Flow execution started!')
    },
    onError: (error: any) => {
      toast.error(error.message || 'Failed to execute flow')
    }
  })
}

export function useValidateFlow() {
  return useMutation({
    mutationFn: (flowConfig: CreateFlowRequest) => flowsApi.validate(flowConfig),
    onError: (error: any) => {
      toast.error(error.message || 'Flow validation failed')
    }
  })
}

// Bulk operations
export function useBulkDeleteFlows() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (ids: string[]) => flowsApi.bulkDelete(ids),
    onSuccess: (result) => {
      queryClient.invalidateQueries({ queryKey: flowKeys.lists() })
      queryClient.invalidateQueries({ queryKey: flowKeys.stats() })

      if (result.errors.length > 0) {
        toast.error(`${result.deleted} flows deleted, ${result.errors.length} failed`)
      } else {
        toast.success(`${result.deleted} flows deleted successfully!`)
      }
    },
    onError: (error: any) => {
      toast.error(error.message || 'Bulk delete failed')
    }
  })
}

export function useBulkToggleFlows() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (ids: string[]) => flowsApi.bulkToggle(ids),
    onSuccess: (result) => {
      queryClient.invalidateQueries({ queryKey: flowKeys.lists() })
      queryClient.invalidateQueries({ queryKey: flowKeys.stats() })

      if (result.errors.length > 0) {
        toast.error(`${result.updated} flows updated, ${result.errors.length} failed`)
      } else {
        toast.success(`${result.updated} flows updated successfully!`)
      }
    },
    onError: (error: any) => {
      toast.error(error.message || 'Bulk toggle failed')
    }
  })
}

export function useBulkStartFlows() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (ids: string[]) => flowsApi.bulkStart(ids),
    onSuccess: (result) => {
      queryClient.invalidateQueries({ queryKey: flowKeys.lists() })

      if (result.errors.length > 0) {
        toast.error(`${result.started} flows started, ${result.errors.length} failed`)
      } else {
        toast.success(`${result.started} flows started successfully!`)
      }
    },
    onError: (error: any) => {
      toast.error(error.message || 'Bulk start failed')
    }
  })
}

export function useBulkStopFlows() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (ids: string[]) => flowsApi.bulkStop(ids),
    onSuccess: (result) => {
      queryClient.invalidateQueries({ queryKey: flowKeys.lists() })

      if (result.errors.length > 0) {
        toast.error(`${result.stopped} flows stopped, ${result.errors.length} failed`)
      } else {
        toast.success(`${result.stopped} flows stopped successfully!`)
      }
    },
    onError: (error: any) => {
      toast.error(error.message || 'Bulk stop failed')
    }
  })
}