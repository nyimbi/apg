/**
 * APG Connection Management - Connections Hooks
 * React Query hooks for connection management
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { connectionsApi, Connection, CreateConnectionRequest } from '@/api/connections'
import toast from 'react-hot-toast'

// Query Keys
export const connectionKeys = {
  all: ['connections'] as const,
  lists: () => [...connectionKeys.all, 'list'] as const,
  list: (params?: any) => [...connectionKeys.lists(), params] as const,
  details: () => [...connectionKeys.all, 'detail'] as const,
  detail: (id: string) => [...connectionKeys.details(), id] as const,
  health: (id: string) => [...connectionKeys.detail(id), 'health'] as const,
  schema: (id: string) => [...connectionKeys.detail(id), 'schema'] as const,
  stats: () => [...connectionKeys.all, 'stats'] as const,
}

// Hooks
export function useConnections(params?: {
  status?: Connection['status']
  connection_type?: Connection['connection_type']
  search?: string
  tags?: string[]
  limit?: number
  offset?: number
}) {
  return useQuery({
    queryKey: connectionKeys.list(params),
    queryFn: () => connectionsApi.list(params),
    keepPreviousData: true,
    staleTime: 30000, // 30 seconds
  })
}

export function useConnection(id: string) {
  return useQuery({
    queryKey: connectionKeys.detail(id),
    queryFn: () => connectionsApi.get(id),
    enabled: !!id,
    staleTime: 60000, // 1 minute
  })
}

export function useConnectionHealth(id: string) {
  return useQuery({
    queryKey: connectionKeys.health(id),
    queryFn: () => connectionsApi.getHealth(id),
    enabled: !!id,
    refetchInterval: 30000, // Refetch every 30 seconds
    staleTime: 15000, // 15 seconds
  })
}

export function useConnectionSchema(id: string) {
  return useQuery({
    queryKey: connectionKeys.schema(id),
    queryFn: () => connectionsApi.getSchema(id),
    enabled: !!id,
    staleTime: 300000, // 5 minutes (schema doesn't change often)
  })
}

export function useConnectionStats() {
  return useQuery({
    queryKey: connectionKeys.stats(),
    queryFn: () => connectionsApi.getStats(),
    refetchInterval: 60000, // Refetch every minute
    staleTime: 30000, // 30 seconds
  })
}

// Mutations
export function useCreateConnection() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (data: CreateConnectionRequest) => connectionsApi.create(data),
    onSuccess: (newConnection) => {
      // Invalidate and refetch connections list
      queryClient.invalidateQueries({ queryKey: connectionKeys.lists() })
      queryClient.invalidateQueries({ queryKey: connectionKeys.stats() })

      // Add the new connection to the cache
      queryClient.setQueryData(
        connectionKeys.detail(newConnection.id),
        newConnection
      )

      toast.success(`Connection "${newConnection.name}" created successfully!`)
    },
    onError: (error: any) => {
      toast.error(error.message || 'Failed to create connection')
    }
  })
}

export function useUpdateConnection() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({ id, data }: { id: string; data: Partial<CreateConnectionRequest> }) =>
      connectionsApi.update(id, data),
    onSuccess: (updatedConnection) => {
      // Update the connection in cache
      queryClient.setQueryData(
        connectionKeys.detail(updatedConnection.id),
        updatedConnection
      )

      // Invalidate lists to ensure consistency
      queryClient.invalidateQueries({ queryKey: connectionKeys.lists() })

      toast.success(`Connection "${updatedConnection.name}" updated successfully!`)
    },
    onError: (error: any) => {
      toast.error(error.message || 'Failed to update connection')
    }
  })
}

export function useDeleteConnection() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (id: string) => connectionsApi.delete(id),
    onSuccess: (_, deletedId) => {
      // Remove from cache
      queryClient.removeQueries({ queryKey: connectionKeys.detail(deletedId) })
      queryClient.removeQueries({ queryKey: connectionKeys.health(deletedId) })
      queryClient.removeQueries({ queryKey: connectionKeys.schema(deletedId) })

      // Invalidate lists
      queryClient.invalidateQueries({ queryKey: connectionKeys.lists() })
      queryClient.invalidateQueries({ queryKey: connectionKeys.stats() })

      toast.success('Connection deleted successfully!')
    },
    onError: (error: any) => {
      toast.error(error.message || 'Failed to delete connection')
    }
  })
}

export function useTestConnection() {
  return useMutation({
    mutationFn: (id: string) => connectionsApi.test(id),
    onSuccess: (result) => {
      if (result.success) {
        toast.success(`Connection test successful! (${result.duration_ms}ms)`)
      } else {
        toast.error(`Connection test failed: ${result.message}`)
      }
    },
    onError: (error: any) => {
      toast.error(error.message || 'Connection test failed')
    }
  })
}

export function useToggleConnection() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (id: string) => connectionsApi.toggle(id),
    onSuccess: (updatedConnection) => {
      // Update the connection in cache
      queryClient.setQueryData(
        connectionKeys.detail(updatedConnection.id),
        updatedConnection
      )

      // Invalidate lists to update status counts
      queryClient.invalidateQueries({ queryKey: connectionKeys.lists() })
      queryClient.invalidateQueries({ queryKey: connectionKeys.stats() })

      const action = updatedConnection.status === 'active' ? 'activated' : 'deactivated'
      toast.success(`Connection "${updatedConnection.name}" ${action}!`)
    },
    onError: (error: any) => {
      toast.error(error.message || 'Failed to toggle connection')
    }
  })
}

export function useBulkDeleteConnections() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (ids: string[]) => connectionsApi.bulkDelete(ids),
    onSuccess: (result) => {
      // Remove deleted connections from cache
      result.deleted && queryClient.invalidateQueries({ queryKey: connectionKeys.lists() })
      queryClient.invalidateQueries({ queryKey: connectionKeys.stats() })

      if (result.errors.length > 0) {
        toast.error(`${result.deleted} connections deleted, ${result.errors.length} failed`)
      } else {
        toast.success(`${result.deleted} connections deleted successfully!`)
      }
    },
    onError: (error: any) => {
      toast.error(error.message || 'Bulk delete failed')
    }
  })
}

export function useBulkToggleConnections() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (ids: string[]) => connectionsApi.bulkToggle(ids),
    onSuccess: (result) => {
      // Invalidate lists to refresh status
      queryClient.invalidateQueries({ queryKey: connectionKeys.lists() })
      queryClient.invalidateQueries({ queryKey: connectionKeys.stats() })

      if (result.errors.length > 0) {
        toast.error(`${result.updated} connections updated, ${result.errors.length} failed`)
      } else {
        toast.success(`${result.updated} connections updated successfully!`)
      }
    },
    onError: (error: any) => {
      toast.error(error.message || 'Bulk toggle failed')
    }
  })
}

export function useSyncConnectionSchema() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (id: string) => connectionsApi.syncSchema(id),
    onSuccess: (result, id) => {
      // Invalidate schema cache to refetch latest
      queryClient.invalidateQueries({ queryKey: connectionKeys.schema(id) })

      if (result.success) {
        const changeCount = result.changes.length
        if (changeCount > 0) {
          toast.success(`Schema synced! ${changeCount} changes detected.`)
        } else {
          toast.success('Schema synced! No changes detected.')
        }
      } else {
        toast.error('Schema sync failed')
      }
    },
    onError: (error: any) => {
      toast.error(error.message || 'Failed to sync schema')
    }
  })
}