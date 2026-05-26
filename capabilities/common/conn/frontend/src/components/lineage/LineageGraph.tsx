import React, { useCallback, useEffect, useState, useMemo } from 'react'
import ReactFlow, {
  Node,
  Edge,
  addEdge,
  useNodesState,
  useEdgesState,
  Controls,
  Background,
  MiniMap,
  ReactFlowProvider,
  Panel,
  useReactFlow,
  MarkerType,
  Position,
} from '@reactflow/core'
import { motion, AnimatePresence } from 'framer-motion'
import {
  DatabaseIcon,
  WorkflowIcon,
  BrainIcon,
  LinkIcon,
  FilterIcon,
  LayersIcon,
  ZoomInIcon,
  ZoomOutIcon,
  FullscreenIcon,
  DownloadIcon,
  SearchIcon,
  EyeIcon,
  EyeOffIcon,
  ShieldCheckIcon,
  AlertTriangleIcon,
} from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { cn } from '@/utils/cn'

// Custom node components
interface CustomNodeData {
  label: string
  type: 'connection' | 'entity' | 'field' | 'flow' | 'transformation'
  source_type?: string
  sensitive?: boolean
  pii?: boolean
  description?: string
  metadata?: Record<string, any>
}

const nodeColors = {
  connection: { bg: '#3B82F6', border: '#2563EB' },
  entity: { bg: '#10B981', border: '#059669' },
  field: { bg: '#F59E0B', border: '#D97706' },
  flow: { bg: '#8B5CF6', border: '#7C3AED' },
  transformation: { bg: '#EF4444', border: '#DC2626' },
}

function CustomNode({ data, selected }: { data: CustomNodeData; selected: boolean }) {
  const colors = nodeColors[data.type] || nodeColors.entity

  return (
    <motion.div
      initial={{ scale: 0.8, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      whileHover={{ scale: 1.05 }}
      transition={{ duration: 0.2 }}
      className={cn(
        'px-4 py-3 rounded-lg border-2 shadow-elegant min-w-[140px] bg-white dark:bg-gray-800',
        selected && 'ring-2 ring-primary-500 ring-offset-2 dark:ring-offset-gray-900',
        (data.sensitive || data.pii) && 'ring-2 ring-danger-400 ring-offset-1'
      )}
      style={{ borderColor: colors.border }}
    >
      <div className="flex items-center space-x-2">
        <div
          className="p-1.5 rounded-md text-white"
          style={{ backgroundColor: colors.bg }}
        >
          {data.type === 'connection' && <DatabaseIcon className="h-4 w-4" />}
          {data.type === 'entity' && <LayersIcon className="h-4 w-4" />}
          {data.type === 'field' && <FilterIcon className="h-4 w-4" />}
          {data.type === 'flow' && <WorkflowIcon className="h-4 w-4" />}
          {data.type === 'transformation' && <BrainIcon className="h-4 w-4" />}
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-center space-x-1">
            <p className="text-sm font-semibold text-gray-900 dark:text-white truncate">
              {data.label}
            </p>
            {(data.sensitive || data.pii) && (
              <ShieldCheckIcon className="h-3 w-3 text-danger-500 flex-shrink-0" />
            )}
          </div>
          <p className="text-xs text-gray-600 dark:text-gray-400 capitalize">
            {data.source_type || data.type}
          </p>
        </div>
      </div>

      {data.description && (
        <p className="text-xs text-gray-500 dark:text-gray-500 mt-1 line-clamp-2">
          {data.description}
        </p>
      )}
    </motion.div>
  )
}

interface LineageGraphProps {
  lineageData: {
    nodes: Array<{
      id: string
      label: string
      type: string
      source_type?: string
      metadata?: Record<string, any>
    }>
    edges: Array<{
      id: string
      source: string
      target: string
      label?: string
      type?: string
      metadata?: Record<string, any>
    }>
    summary?: {
      total_nodes: number
      total_edges: number
      node_types: Record<string, number>
      sensitive_data_nodes: number
    }
  }
  onNodeClick?: (nodeId: string, node: any) => void
  onEdgeClick?: (edgeId: string, edge: any) => void
  className?: string
}

const nodeTypes = {
  custom: CustomNode,
}

export function LineageGraph({
  lineageData,
  onNodeClick,
  onEdgeClick,
  className
}: LineageGraphProps) {
  const [nodes, setNodes, onNodesChange] = useNodesState([])
  const [edges, setEdges, onEdgesChange] = useEdgesState([])
  const [selectedNode, setSelectedNode] = useState<string | null>(null)
  const [showSensitiveOnly, setShowSensitiveOnly] = useState(false)
  const [searchTerm, setSearchTerm] = useState('')
  const [layoutType, setLayoutType] = useState<'force' | 'hierarchy' | 'radial'>('force')

  // Convert lineage data to React Flow format
  const reactFlowNodes = useMemo(() => {
    if (!lineageData?.nodes) return []

    return lineageData.nodes
      .filter(node => {
        if (showSensitiveOnly) {
          return node.metadata?.sensitive || node.metadata?.pii
        }
        if (searchTerm) {
          return node.label.toLowerCase().includes(searchTerm.toLowerCase()) ||
                 node.type.toLowerCase().includes(searchTerm.toLowerCase())
        }
        return true
      })
      .map((node, index) => {
        // Simple layout algorithm - in production, use a proper layout library
        const angle = (index / lineageData.nodes.length) * 2 * Math.PI
        const radius = Math.min(300, 50 + index * 20)

        return {
          id: node.id,
          type: 'custom',
          position: {
            x: 400 + radius * Math.cos(angle),
            y: 300 + radius * Math.sin(angle),
          },
          data: {
            ...node,
            sensitive: node.metadata?.sensitive,
            pii: node.metadata?.pii,
            description: node.metadata?.description,
          } as CustomNodeData,
          sourcePosition: Position.Right,
          targetPosition: Position.Left,
        }
      })
  }, [lineageData, showSensitiveOnly, searchTerm])

  const reactFlowEdges = useMemo(() => {
    if (!lineageData?.edges) return []

    const visibleNodeIds = new Set(reactFlowNodes.map(n => n.id))

    return lineageData.edges
      .filter(edge => visibleNodeIds.has(edge.source) && visibleNodeIds.has(edge.target))
      .map(edge => ({
        id: edge.id,
        source: edge.source,
        target: edge.target,
        label: edge.label || edge.type,
        type: 'smoothstep',
        animated: edge.type === 'real-time',
        markerEnd: {
          type: MarkerType.ArrowClosed,
          width: 20,
          height: 20,
        },
        style: {
          stroke: edge.type === 'maps_to' ? '#3B82F6' :
                  edge.type === 'derives_from' ? '#10B981' : '#6B7280',
          strokeWidth: 2,
        },
        labelStyle: {
          fontSize: 12,
          fontWeight: 500,
        },
        labelBgStyle: {
          fill: 'white',
          fillOpacity: 0.8,
        },
      }))
  }, [lineageData, reactFlowNodes])

  // Update nodes and edges when data changes
  useEffect(() => {
    setNodes(reactFlowNodes)
    setEdges(reactFlowEdges)
  }, [reactFlowNodes, reactFlowEdges, setNodes, setEdges])

  const onConnect = useCallback(
    (params: any) => setEdges((eds) => addEdge(params, eds)),
    [setEdges]
  )

  const handleNodeClick = useCallback((event: React.MouseEvent, node: Node) => {
    setSelectedNode(node.id)
    onNodeClick?.(node.id, node)
  }, [onNodeClick])

  const handleEdgeClick = useCallback((event: React.MouseEvent, edge: Edge) => {
    onEdgeClick?.(edge.id, edge)
  }, [onEdgeClick])

  return (
    <div className={cn('w-full h-full bg-gray-50 dark:bg-gray-900 rounded-lg overflow-hidden', className)}>
      <ReactFlowProvider>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onNodeClick={handleNodeClick}
          onEdgeClick={handleEdgeClick}
          nodeTypes={nodeTypes}
          fitView
          attributionPosition="bottom-right"
          className="bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800"
        >
          <Background
            color="#94a3b8"
            gap={20}
            className="opacity-30 dark:opacity-10"
          />

          <MiniMap
            nodeColor={(node) => {
              const data = node.data as CustomNodeData
              return nodeColors[data.type]?.bg || '#6B7280'
            }}
            nodeStrokeWidth={3}
            className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg"
          />

          <Controls
            className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-elegant"
            showInteractive={false}
          />

          {/* Custom Toolbar */}
          <Panel position="top-left">
            <div className="flex items-center space-x-3">
              {/* Search */}
              <div className="relative">
                <SearchIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                <input
                  type="text"
                  placeholder="Search nodes..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-10 pr-4 py-2 w-64 text-sm bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                />
              </div>

              {/* Filters */}
              <Button
                variant={showSensitiveOnly ? 'default' : 'outline'}
                size="sm"
                onClick={() => setShowSensitiveOnly(!showSensitiveOnly)}
                icon={showSensitiveOnly ? <EyeIcon className="h-4 w-4" /> : <EyeOffIcon className="h-4 w-4" />}
              >
                Sensitive Data
              </Button>

              {/* Layout Options */}
              <select
                value={layoutType}
                onChange={(e) => setLayoutType(e.target.value as any)}
                className="px-3 py-2 text-sm bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500"
              >
                <option value="force">Force Layout</option>
                <option value="hierarchy">Hierarchy</option>
                <option value="radial">Radial</option>
              </select>
            </div>
          </Panel>

          {/* Statistics Panel */}
          <Panel position="top-right">
            <Card className="w-64">
              <CardHeader>
                <CardTitle className="text-sm">Lineage Statistics</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600 dark:text-gray-400">Nodes:</span>
                  <span className="font-medium">{lineageData?.summary?.total_nodes || 0}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600 dark:text-gray-400">Edges:</span>
                  <span className="font-medium">{lineageData?.summary?.total_edges || 0}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600 dark:text-gray-400">Sensitive:</span>
                  <span className="font-medium text-danger-600">
                    {lineageData?.summary?.sensitive_data_nodes || 0}
                  </span>
                </div>

                {lineageData?.summary?.node_types && (
                  <div className="pt-2 border-t border-gray-200 dark:border-gray-700">
                    <p className="text-xs text-gray-500 mb-2">Node Types:</p>
                    <div className="space-y-1">
                      {Object.entries(lineageData.summary.node_types).map(([type, count]) => (
                        <div key={type} className="flex items-center justify-between text-xs">
                          <Badge variant="secondary" size="sm" className="capitalize">
                            {type}
                          </Badge>
                          <span className="font-medium">{count}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </Panel>

          {/* Export Actions */}
          <Panel position="bottom-right">
            <div className="flex items-center space-x-2">
              <Button variant="outline" size="sm" icon={<DownloadIcon className="h-4 w-4" />}>
                Export PNG
              </Button>
              <Button variant="outline" size="sm" icon={<DownloadIcon className="h-4 w-4" />}>
                Export SVG
              </Button>
            </div>
          </Panel>
        </ReactFlow>
      </ReactFlowProvider>

      {/* Node Details Sidebar */}
      <AnimatePresence>
        {selectedNode && (
          <motion.div
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ type: 'spring', damping: 20 }}
            className="absolute top-0 right-0 w-80 h-full bg-white dark:bg-gray-800 border-l border-gray-200 dark:border-gray-700 shadow-elegant z-10"
          >
            <div className="p-6 space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                  Node Details
                </h3>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setSelectedNode(null)}
                >
                  ×
                </Button>
              </div>

              {/* Node details would be rendered here */}
              <div className="space-y-3">
                <div>
                  <p className="text-sm font-medium text-gray-700 dark:text-gray-300">ID</p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">{selectedNode}</p>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}