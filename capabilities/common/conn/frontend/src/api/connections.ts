/**
 * APG Connection Management - Connections API
 */

import apiClient from './client'

// Types
export interface Connection {
  id: string
  name: string
  description?: string
  connection_type: 'database' | 'api' | 'file' | 'stream'
  status: 'active' | 'inactive' | 'error' | 'testing'
  singer_tap?: string
  singer_target?: string
  tap_config: Record<string, any>
  target_config: Record<string, any>
  sync_mode: 'full_refresh' | 'incremental'
  sync_frequency?: string
  batch_size: number
  last_sync?: string
  last_success?: string
  last_error?: string
  error_count: number
  tags: string[]
  created_at: string
  updated_at: string
  created_by: string
  health_score?: number
  records_processed?: number
}

export interface CreateConnectionRequest {
  name: string
  description?: string
  connection_type: Connection['connection_type']
  singer_tap?: string
  singer_target?: string
  tap_config?: Record<string, any>
  target_config?: Record<string, any>
  sync_mode?: Connection['sync_mode']
  sync_frequency?: string
  batch_size?: number
  tags?: string[]
}

export interface UpdateConnectionRequest extends Partial<CreateConnectionRequest> {
  id: string
}

export interface ConnectionTestResult {
  success: boolean
  message: string
  details?: Record<string, any>
  duration_ms: number
}

export interface ConnectionHealth {
  connection_id: string
  status: string
  health_score: number
  last_check: string
  checks: Array<{
    name: string
    status: 'healthy' | 'warning' | 'error'
    message: string
    details?: any
  }>
}

// API Functions
export const connectionsApi = {
  // List connections with optional filtering
  async list(params?: {
    status?: Connection['status']
    connection_type?: Connection['connection_type']
    search?: string
    tags?: string[]
    limit?: number
    offset?: number
  }): Promise<{ items: Connection[]; total: number }> {
    const searchParams = new URLSearchParams()

    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) {
          if (Array.isArray(value)) {
            value.forEach(v => searchParams.append(key, v))
          } else {
            searchParams.append(key, value.toString())
          }
        }
      })
    }

    return apiClient.get(`/connections?${searchParams.toString()}`)
  },

  // Get single connection
  async get(id: string): Promise<Connection> {
    return apiClient.get(`/connections/${id}`)
  },

  // Create new connection
  async create(data: CreateConnectionRequest): Promise<Connection> {
    return apiClient.post('/connections', data)
  },

  // Update existing connection
  async update(id: string, data: Partial<CreateConnectionRequest>): Promise<Connection> {
    return apiClient.put(`/connections/${id}`, data)
  },

  // Delete connection
  async delete(id: string): Promise<void> {
    return apiClient.delete(`/connections/${id}`)
  },

  // Test connection
  async test(id: string): Promise<ConnectionTestResult> {
    return apiClient.post(`/connections/${id}/test`)
  },

  // Get connection health
  async getHealth(id: string): Promise<ConnectionHealth> {
    return apiClient.get(`/connections/${id}/health`)
  },

  // Toggle connection status
  async toggle(id: string): Promise<Connection> {
    return apiClient.post(`/connections/${id}/toggle`)
  },

  // Bulk operations
  async bulkDelete(ids: string[]): Promise<{ deleted: number; errors: any[] }> {
    return apiClient.post('/connections/bulk/delete', { ids })
  },

  async bulkToggle(ids: string[]): Promise<{ updated: number; errors: any[] }> {
    return apiClient.post('/connections/bulk/toggle', { ids })
  },

  // Get connection statistics
  async getStats(): Promise<{
    total: number
    by_status: Record<Connection['status'], number>
    by_type: Record<Connection['connection_type'], number>
    health_average: number
    recent_errors: number
  }> {
    return apiClient.get('/connections/stats')
  },

  // Get connection schema info
  async getSchema(id: string): Promise<{
    schema: Record<string, any>
    last_updated: string
    field_count: number
    table_count: number
  }> {
    return apiClient.get(`/connections/${id}/schema`)
  },

  // Sync connection schema
  async syncSchema(id: string): Promise<{
    success: boolean
    changes: Array<{
      type: 'added' | 'removed' | 'modified'
      path: string
      old_value?: any
      new_value?: any
    }>
  }> {
    return apiClient.post(`/connections/${id}/sync-schema`)
  }
}

export default connectionsApi