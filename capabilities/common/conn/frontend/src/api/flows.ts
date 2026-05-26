/**
 * APG Connection Management - Flows API
 */

import apiClient from './client'

// Types
export interface DataFlow {
  id: string
  name: string
  description?: string
  source_connection_id: string
  target_connection_id: string
  status: 'active' | 'inactive' | 'running' | 'error' | 'paused'
  selected_streams: string[]
  transformation_rules: string[]
  schedule_expression?: string
  enabled: boolean
  tags: string[]
  created_at: string
  updated_at: string
  created_by: string
  last_run?: string
  last_success?: string
  last_error?: string
  run_count: number
  success_rate: number
  avg_runtime: number
}

export interface CreateFlowRequest {
  name: string
  description?: string
  source_connection_id: string
  target_connection_id: string
  selected_streams?: string[]
  transformation_rules?: string[]
  schedule_expression?: string
  enabled?: boolean
  tags?: string[]
}

export interface FlowExecution {
  id: string
  flow_id: string
  status: 'running' | 'completed' | 'failed' | 'cancelled'
  started_at: string
  completed_at?: string
  duration_seconds?: number
  records_processed: number
  records_failed: number
  error_message?: string
  logs: Array<{
    timestamp: string
    level: 'info' | 'warning' | 'error'
    message: string
    details?: any
  }>
}

export interface FlowMetrics {
  flow_id: string
  total_runs: number
  successful_runs: number
  failed_runs: number
  success_rate: number
  avg_runtime: number
  total_records: number
  last_24h: {
    runs: number
    records: number
    success_rate: number
  }
  performance_trend: Array<{
    date: string
    runtime: number
    records: number
    success: boolean
  }>
}

// API Functions
export const flowsApi = {
  // List flows
  async list(params?: {
    status?: DataFlow['status']
    source_connection_id?: string
    target_connection_id?: string
    search?: string
    tags?: string[]
    enabled?: boolean
    limit?: number
    offset?: number
  }): Promise<{ items: DataFlow[]; total: number }> {
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

    return apiClient.get(`/flows?${searchParams.toString()}`)
  },

  // Get single flow
  async get(id: string): Promise<DataFlow> {
    return apiClient.get(`/flows/${id}`)
  },

  // Create new flow
  async create(data: CreateFlowRequest): Promise<DataFlow> {
    return apiClient.post('/flows', data)
  },

  // Update existing flow
  async update(id: string, data: Partial<CreateFlowRequest>): Promise<DataFlow> {
    return apiClient.put(`/flows/${id}`, data)
  },

  // Delete flow
  async delete(id: string): Promise<void> {
    return apiClient.delete(`/flows/${id}`)
  },

  // Start flow execution
  async start(id: string): Promise<FlowExecution> {
    return apiClient.post(`/flows/${id}/start`)
  },

  // Stop flow execution
  async stop(id: string): Promise<void> {
    return apiClient.post(`/flows/${id}/stop`)
  },

  // Execute flow once
  async executeOnce(id: string): Promise<FlowExecution> {
    return apiClient.post(`/flows/${id}/execute`)
  },

  // Get flow executions
  async getExecutions(
    id: string,
    params?: {
      status?: FlowExecution['status']
      limit?: number
      offset?: number
    }
  ): Promise<{ items: FlowExecution[]; total: number }> {
    const searchParams = new URLSearchParams()

    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) {
          searchParams.append(key, value.toString())
        }
      })
    }

    return apiClient.get(`/flows/${id}/executions?${searchParams.toString()}`)
  },

  // Get single execution
  async getExecution(flowId: string, executionId: string): Promise<FlowExecution> {
    return apiClient.get(`/flows/${flowId}/executions/${executionId}`)
  },

  // Get flow metrics
  async getMetrics(id: string, period: '24h' | '7d' | '30d' = '24h'): Promise<FlowMetrics> {
    return apiClient.get(`/flows/${id}/metrics?period=${period}`)
  },

  // Get flow logs
  async getLogs(
    id: string,
    params?: {
      execution_id?: string
      level?: 'info' | 'warning' | 'error'
      limit?: number
      offset?: number
    }
  ): Promise<{
    items: Array<{
      timestamp: string
      level: string
      message: string
      execution_id?: string
      details?: any
    }>
    total: number
  }> {
    const searchParams = new URLSearchParams()

    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) {
          searchParams.append(key, value.toString())
        }
      })
    }

    return apiClient.get(`/flows/${id}/logs?${searchParams.toString()}`)
  },

  // Bulk operations
  async bulkDelete(ids: string[]): Promise<{ deleted: number; errors: any[] }> {
    return apiClient.post('/flows/bulk/delete', { ids })
  },

  async bulkToggle(ids: string[]): Promise<{ updated: number; errors: any[] }> {
    return apiClient.post('/flows/bulk/toggle', { ids })
  },

  async bulkStart(ids: string[]): Promise<{ started: number; errors: any[] }> {
    return apiClient.post('/flows/bulk/start', { ids })
  },

  async bulkStop(ids: string[]): Promise<{ stopped: number; errors: any[] }> {
    return apiClient.post('/flows/bulk/stop', { ids })
  },

  // Get flow statistics
  async getStats(): Promise<{
    total: number
    by_status: Record<DataFlow['status'], number>
    total_executions_today: number
    success_rate_today: number
    avg_runtime: number
    active_flows: number
  }> {
    return apiClient.get('/flows/stats')
  },

  // Validate flow configuration
  async validate(flowConfig: CreateFlowRequest): Promise<{
    valid: boolean
    errors: Array<{
      field: string
      message: string
      code: string
    }>
    warnings: Array<{
      field: string
      message: string
      code: string
    }>
  }> {
    return apiClient.post('/flows/validate', flowConfig)
  },

  // Get available streams from source connection
  async getAvailableStreams(sourceConnectionId: string): Promise<Array<{
    name: string
    schema?: Record<string, any>
    record_count?: number
    last_updated?: string
  }>> {
    return apiClient.get(`/flows/streams/${sourceConnectionId}`)
  }
}

export default flowsApi