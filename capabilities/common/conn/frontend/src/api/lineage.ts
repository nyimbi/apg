/**
 * APG Connection Management - Data Lineage API
 */

import apiClient from './client'

// Types
export interface LineageNode {
  id: string
  label: string
  type: 'connection' | 'entity' | 'field' | 'flow' | 'transformation'
  source_type?: string
  metadata?: {
    description?: string
    owner?: string
    tags?: string[]
    sensitive?: boolean
    pii?: boolean
    data_quality_score?: number
    freshness_score?: number
    last_updated?: string
  }
}

export interface LineageEdge {
  id: string
  source: string
  target: string
  label?: string
  type?: string
  metadata?: {
    transformation_type?: string
    transformation_logic?: string
    volume_estimate?: number
    frequency?: string
    latency?: number
    quality_impact?: number
  }
}

export interface LineageVisualization {
  nodes: LineageNode[]
  edges: LineageEdge[]
  layout?: {
    algorithm: string
    parameters: Record<string, any>
  }
  summary: {
    total_nodes: number
    total_edges: number
    node_types: Record<string, number>
    sensitive_data_nodes: number
    data_quality_avg: number
    last_updated: string
  }
  visualization_type: 'full' | 'upstream' | 'downstream' | 'impact'
  generated_at: string
}

export interface DataCatalog {
  entities: Array<{
    id: string
    name: string
    schema?: string
    table?: string
    description?: string
    owner?: string
    tags: string[]
    connection_id: string
    data_quality_score: number
    freshness_score: number
    last_updated: string
    fields: Array<{
      name: string
      data_type: string
      sensitive: boolean
      pii: boolean
      description?: string
    }>
  }>
  connections: Array<{
    id: string
    name: string
    source_type: string
    description?: string
    connection_id: string
    last_updated: string
  }>
  flows: Array<{
    id: string
    name: string
    description?: string
    flow_id: string
    last_updated: string
  }>
  summary: {
    total_entities: number
    total_fields: number
    total_connections: number
    total_flows: number
    sensitive_fields: number
    pii_fields: number
  }
}

export interface ImpactAnalysis {
  affected_nodes: number
  affected_flows: number
  affected_connections: number
  risk_level: 'low' | 'medium' | 'high'
  recommendations: string[]
  sensitive_data_affected?: number
  estimated_impact_score: number
  affected_entities: Array<{
    id: string
    name: string
    type: string
    impact_level: 'low' | 'medium' | 'high'
    relationship_path: string[]
  }>
}

export interface LineageSearchResult {
  results: Array<{
    id: string
    name: string
    type: string
    description?: string
    tags: string[]
    relevance_score: number
    context: {
      connection_name?: string
      schema_name?: string
      table_name?: string
    }
  }>
  total: number
  facets: {
    types: Record<string, number>
    connections: Record<string, number>
    tags: Record<string, number>
  }
}

// API Functions
export const lineageApi = {
  // Track connection in lineage
  async trackConnection(data: {
    connection_id: string
    connection_name: string
    connection_type: string
    schema_info: Record<string, any>
  }): Promise<{ message: string; node_ids: string[] }> {
    return apiClient.post('/lineage/track-connection', data)
  },

  // Track flow execution
  async trackFlow(data: {
    flow_id: string
    flow_name: string
    source_connection_id: string
    target_connection_id: string
    transformations: Array<Record<string, any>>
    field_mappings: Record<string, string>
  }): Promise<{ message: string }> {
    return apiClient.post('/lineage/track-flow', data)
  },

  // Generate lineage visualization
  async generateVisualization(params: {
    node_id?: string
    visualization_type?: 'full' | 'upstream' | 'downstream' | 'impact'
  } = {}): Promise<LineageVisualization> {
    return apiClient.post('/lineage/visualization', params)
  },

  // Get upstream lineage
  async getUpstream(nodeId: string, maxDepth = 10): Promise<{
    nodes: LineageNode[]
    edges: LineageEdge[]
    depth: number
  }> {
    return apiClient.get(`/lineage/upstream/${nodeId}?max_depth=${maxDepth}`)
  },

  // Get downstream lineage
  async getDownstream(nodeId: string, maxDepth = 10): Promise<{
    nodes: LineageNode[]
    edges: LineageEdge[]
    depth: number
  }> {
    return apiClient.get(`/lineage/downstream/${nodeId}?max_depth=${maxDepth}`)
  },

  // Analyze impact
  async analyzeImpact(nodeId: string): Promise<ImpactAnalysis> {
    return apiClient.get(`/lineage/impact/${nodeId}`)
  },

  // Get data catalog
  async getDataCatalog(): Promise<DataCatalog> {
    return apiClient.get('/lineage/catalog')
  },

  // Search lineage
  async search(params: {
    query: string
    search_type?: 'all' | 'entities' | 'fields' | 'flows'
    filters?: {
      types?: string[]
      connections?: string[]
      tags?: string[]
    }
    limit?: number
    offset?: number
  }): Promise<LineageSearchResult> {
    return apiClient.post('/lineage/search', params)
  },

  // Detect cycles
  async detectCycles(): Promise<{
    cycles: string[][]
    cycle_count: number
  }> {
    return apiClient.get('/lineage/cycles')
  },

  // Get root sources
  async getRootSources(): Promise<{
    root_sources: Array<{
      id: string
      name: string
      type: string
      source_type: string
      description?: string
    }>
  }> {
    return apiClient.get('/lineage/root-sources')
  },

  // Get leaf destinations
  async getLeafDestinations(): Promise<{
    leaf_destinations: Array<{
      id: string
      name: string
      type: string
      source_type: string
      description?: string
    }>
  }> {
    return apiClient.get('/lineage/leaf-destinations')
  },

  // Get lineage statistics
  async getStats(): Promise<{
    total_nodes: number
    total_edges: number
    node_types: Record<string, number>
    edge_types: Record<string, number>
    sensitive_nodes: number
    pii_nodes: number
    avg_data_quality: number
    coverage_percentage: number
    last_updated: string
  }> {
    return apiClient.get('/lineage/stats')
  },

  // Export lineage
  async exportLineage(params: {
    format: 'json' | 'csv' | 'graphml'
    node_id?: string
    visualization_type?: 'full' | 'upstream' | 'downstream'
    include_metadata?: boolean
  }): Promise<Blob> {
    const response = await apiClient.client.post('/lineage/export', params, {
      responseType: 'blob'
    })
    return response.data
  },

  // Get node details
  async getNodeDetails(nodeId: string): Promise<{
    node: LineageNode
    upstream_count: number
    downstream_count: number
    related_flows: Array<{
      id: string
      name: string
      status: string
    }>
    data_quality_history: Array<{
      date: string
      score: number
    }>
    recent_changes: Array<{
      timestamp: string
      type: 'schema_change' | 'config_change' | 'status_change'
      description: string
      user: string
    }>
  }> {
    return apiClient.get(`/lineage/nodes/${nodeId}`)
  }
}

export default lineageApi