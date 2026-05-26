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
  Connection,
  NodeTypes,
} from '@reactflow/core'
import { motion, AnimatePresence } from 'framer-motion'
import {
  DatabaseIcon,
  WorkflowIcon,
  BrainIcon,
  FilterIcon,
  SettingsIcon,
  PlayIcon,
  SaveIcon,
  ShareIcon,
  UndoIcon,
  RedoIcon,
  ZoomInIcon,
  ZoomOutIcon,
  LayersIcon,
  PlusIcon,
  TrashIcon,
  CopyIcon,
} from 'lucide-react'
import { DndProvider, useDrag, useDrop } from 'react-dnd'
import { HTML5Backend } from 'react-dnd-html5-backend'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { cn } from '@/utils/cn'

// Node types for the designer
interface FlowNodeData {
  label: string
  type: 'source' | 'transform' | 'destination' | 'filter' | 'aggregate' | 'join'
  config?: Record<string, any>
  description?: string
  icon?: React.ReactNode
  isConfigured?: boolean
  errors?: string[]
  isSelected?: boolean
}

// Custom node components
function SourceNode({ data, selected }: { data: FlowNodeData; selected: boolean }) {
  return (
    <motion.div
      initial={{ scale: 0.8, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      whileHover={{ scale: 1.02 }}
      className={cn(
        'px-4 py-3 bg-white dark:bg-gray-800 border-2 border-primary-300 dark:border-primary-600 rounded-xl shadow-elegant min-w-[160px]',
        selected && 'ring-2 ring-primary-500 ring-offset-2 dark:ring-offset-gray-900',
        !data.isConfigured && 'border-dashed border-warning-400'
      )}
    >
      <div className="flex items-center space-x-3">
        <div className="p-2 bg-primary-100 dark:bg-primary-900 rounded-lg">
          <DatabaseIcon className="h-5 w-5 text-primary-600 dark:text-primary-400" />
        </div>
        <div className="flex-1">
          <p className="font-semibold text-sm text-gray-900 dark:text-white">
            {data.label}
          </p>
          <p className="text-xs text-gray-600 dark:text-gray-400">
            {data.description || 'Data source'}
          </p>
        </div>
        {!data.isConfigured && (
          <div className="w-2 h-2 bg-warning-400 rounded-full animate-pulse" />
        )}
      </div>

      {data.errors && data.errors.length > 0 && (
        <div className="mt-2 p-2 bg-danger-50 dark:bg-danger-900/20 rounded border border-danger-200 dark:border-danger-800">
          <p className="text-xs text-danger-600 dark:text-danger-400">
            {data.errors[0]}
          </p>
        </div>
      )}
    </motion.div>
  )
}

function TransformNode({ data, selected }: { data: FlowNodeData; selected: boolean }) {
  return (
    <motion.div
      initial={{ scale: 0.8, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      whileHover={{ scale: 1.02 }}
      className={cn(
        'px-4 py-3 bg-white dark:bg-gray-800 border-2 border-success-300 dark:border-success-600 rounded-xl shadow-elegant min-w-[160px]',
        selected && 'ring-2 ring-success-500 ring-offset-2 dark:ring-offset-gray-900',
        !data.isConfigured && 'border-dashed border-warning-400'
      )}
    >
      <div className="flex items-center space-x-3">
        <div className="p-2 bg-success-100 dark:bg-success-900 rounded-lg">
          <BrainIcon className="h-5 w-5 text-success-600 dark:text-success-400" />
        </div>
        <div className="flex-1">
          <p className="font-semibold text-sm text-gray-900 dark:text-white">
            {data.label}
          </p>
          <p className="text-xs text-gray-600 dark:text-gray-400 capitalize">
            {data.type} operation
          </p>
        </div>
        {data.config && Object.keys(data.config).length > 0 && (
          <Badge variant="success" size="sm">
            Configured
          </Badge>
        )}
      </div>
    </motion.div>
  )
}

function DestinationNode({ data, selected }: { data: FlowNodeData; selected: boolean }) {
  return (
    <motion.div
      initial={{ scale: 0.8, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      whileHover={{ scale: 1.02 }}
      className={cn(
        'px-4 py-3 bg-white dark:bg-gray-800 border-2 border-secondary-300 dark:border-secondary-600 rounded-xl shadow-elegant min-w-[160px]',
        selected && 'ring-2 ring-secondary-500 ring-offset-2 dark:ring-offset-gray-900',
        !data.isConfigured && 'border-dashed border-warning-400'
      )}
    >
      <div className="flex items-center space-x-3">
        <div className="p-2 bg-secondary-100 dark:bg-secondary-800 rounded-lg">
          <LayersIcon className="h-5 w-5 text-secondary-600 dark:text-secondary-400" />
        </div>
        <div className="flex-1">
          <p className="font-semibold text-sm text-gray-900 dark:text-white">
            {data.label}
          </p>
          <p className="text-xs text-gray-600 dark:text-gray-400">
            Data destination
          </p>
        </div>
      </div>
    </motion.div>
  )
}

const nodeTypes: NodeTypes = {
  source: SourceNode,
  transform: TransformNode,
  destination: DestinationNode,
  filter: TransformNode,
  aggregate: TransformNode,
  join: TransformNode,
}

// Draggable node palette item
interface PaletteItemProps {
  type: string
  label: string
  icon: React.ReactNode
  description: string
}

function PaletteItem({ type, label, icon, description }: PaletteItemProps) {
  const [{ isDragging }, drag] = useDrag({
    type: 'flowNode',
    item: { type, label, description },
    collect: (monitor) => ({
      isDragging: monitor.isDragging(),
    }),
  })

  return (
    <motion.div
      ref={drag}
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      className={cn(
        'p-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-sm cursor-grab active:cursor-grabbing transition-all duration-200',
        isDragging && 'opacity-50 scale-95'
      )}
    >
      <div className="flex items-center space-x-3">
        <div className="p-2 bg-gray-100 dark:bg-gray-700 rounded">
          {icon}
        </div>
        <div>
          <p className="font-medium text-sm text-gray-900 dark:text-white">
            {label}
          </p>
          <p className="text-xs text-gray-600 dark:text-gray-400">
            {description}
          </p>
        </div>
      </div>
    </motion.div>
  )
}

// Main flow designer component
interface FlowDesignerProps {
  onSave?: (flowDefinition: any) => void
  onExecute?: (flowDefinition: any) => void
  initialFlow?: { nodes: Node[]; edges: Edge[] }
  readonly?: boolean
}

export function FlowDesigner({
  onSave,
  onExecute,
  initialFlow,
  readonly = false
}: FlowDesignerProps) {
  const [nodes, setNodes, onNodesChange] = useNodesState(initialFlow?.nodes || [])
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialFlow?.edges || [])
  const [selectedNode, setSelectedNode] = useState<string | null>(null)
  const [isPaletteVisible, setIsPaletteVisible] = useState(true)
  const [history, setHistory] = useState<{ nodes: Node[]; edges: Edge[] }[]>([])
  const [historyIndex, setHistoryIndex] = useState(-1)

  const { fitView, getViewport, setViewport } = useReactFlow()

  // Node palette items
  const paletteItems = [
    {
      type: 'source',
      label: 'Data Source',
      icon: <DatabaseIcon className="h-4 w-4 text-primary-600" />,
      description: 'Connect to data source'
    },
    {
      type: 'filter',
      label: 'Filter',
      icon: <FilterIcon className="h-4 w-4 text-success-600" />,
      description: 'Filter records'
    },
    {
      type: 'transform',
      label: 'Transform',
      icon: <BrainIcon className="h-4 w-4 text-success-600" />,
      description: 'Transform data'
    },
    {
      type: 'aggregate',
      label: 'Aggregate',
      icon: <LayersIcon className="h-4 w-4 text-success-600" />,
      description: 'Group and aggregate'
    },
    {
      type: 'join',
      label: 'Join',
      icon: <WorkflowIcon className="h-4 w-4 text-success-600" />,
      description: 'Join data sources'
    },
    {
      type: 'destination',
      label: 'Destination',
      icon: <LayersIcon className="h-4 w-4 text-secondary-600" />,
      description: 'Send data to destination'
    },
  ]

  // Handle dropping nodes from palette
  const [{ isOver }, drop] = useDrop({
    accept: 'flowNode',
    drop: (item: any, monitor) => {
      const dropPosition = monitor.getClientOffset()
      if (!dropPosition) return

      const reactFlowBounds = document.querySelector('.react-flow')?.getBoundingClientRect()
      if (!reactFlowBounds) return

      const position = {
        x: dropPosition.x - reactFlowBounds.left,
        y: dropPosition.y - reactFlowBounds.top,
      }

      const newNode: Node<FlowNodeData> = {
        id: `${item.type}-${Date.now()}`,
        type: item.type,
        position,
        data: {
          label: item.label,
          type: item.type,
          description: item.description,
          isConfigured: false,
        },
      }

      setNodes((nds) => [...nds, newNode])
    },
    collect: (monitor) => ({
      isOver: monitor.isOver(),
    }),
  })

  // Handle connecting nodes
  const onConnect = useCallback(
    (params: Connection) => {
      const edge: Edge = {
        ...params,
        type: 'smoothstep',
        markerEnd: {
          type: MarkerType.ArrowClosed,
          width: 20,
          height: 20,
        },
        style: {
          strokeWidth: 2,
        },
      }
      setEdges((eds) => addEdge(edge, eds))
    },
    [setEdges]
  )

  // Handle node selection
  const onNodeClick = useCallback((event: React.MouseEvent, node: Node) => {
    setSelectedNode(node.id)
  }, [])

  // Save flow definition
  const handleSave = useCallback(() => {
    const flowDefinition = {
      nodes: nodes.map(node => ({
        id: node.id,
        type: node.type,
        position: node.position,
        data: node.data,
      })),
      edges: edges.map(edge => ({
        id: edge.id,
        source: edge.source,
        target: edge.target,
        type: edge.type,
      })),
      viewport: getViewport(),
    }
    onSave?.(flowDefinition)
  }, [nodes, edges, getViewport, onSave])

  // Execute flow
  const handleExecute = useCallback(() => {
    const flowDefinition = {
      nodes: nodes.filter(node => node.data.isConfigured),
      edges,
    }
    onExecute?.(flowDefinition)
  }, [nodes, edges, onExecute])

  // Validation
  const validationErrors = useMemo(() => {
    const errors: string[] = []
    const unconfiguredNodes = nodes.filter(node => !node.data.isConfigured)

    if (unconfiguredNodes.length > 0) {
      errors.push(`${unconfiguredNodes.length} node(s) need configuration`)
    }

    const sourceNodes = nodes.filter(node => node.type === 'source')
    const destinationNodes = nodes.filter(node => node.type === 'destination')

    if (sourceNodes.length === 0) {
      errors.push('At least one data source is required')
    }

    if (destinationNodes.length === 0) {
      errors.push('At least one destination is required')
    }

    return errors
  }, [nodes])

  const canExecute = validationErrors.length === 0

  return (
    <div className="flex h-full">
      {/* Node Palette */}
      <AnimatePresence>
        {isPaletteVisible && (
          <motion.div
            initial={{ x: -300, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: -300, opacity: 0 }}
            transition={{ type: 'spring', damping: 20 }}
            className="w-80 bg-gray-50 dark:bg-gray-900 border-r border-gray-200 dark:border-gray-800 p-4 overflow-y-auto"
          >
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="font-semibold text-gray-900 dark:text-white">
                  Node Library
                </h3>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setIsPaletteVisible(false)}
                >
                  ×
                </Button>
              </div>

              <div className="space-y-3">
                {paletteItems.map((item) => (
                  <PaletteItem
                    key={item.type}
                    type={item.type}
                    label={item.label}
                    icon={item.icon}
                    description={item.description}
                  />
                ))}
              </div>

              {/* Templates Section */}
              <div className="pt-6 border-t border-gray-200 dark:border-gray-700">
                <h4 className="font-medium text-sm text-gray-900 dark:text-white mb-3">
                  Quick Templates
                </h4>
                <div className="space-y-2">
                  <Button variant="outline" size="sm" className="w-full justify-start">
                    <DatabaseIcon className="h-4 w-4 mr-2" />
                    Database ETL
                  </Button>
                  <Button variant="outline" size="sm" className="w-full justify-start">
                    <WorkflowIcon className="h-4 w-4 mr-2" />
                    API Integration
                  </Button>
                  <Button variant="outline" size="sm" className="w-full justify-start">
                    <FilterIcon className="h-4 w-4 mr-2" />
                    Data Cleaning
                  </Button>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Designer Area */}
      <div className="flex-1 relative">
        <div ref={drop} className={cn('w-full h-full', isOver && 'bg-blue-50 dark:bg-blue-900/10')}>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onNodeClick={onNodeClick}
            nodeTypes={nodeTypes}
            fitView
            className="bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800"
            deleteKeyCode={readonly ? null : ['Delete', 'Backspace']}
          >
            <Background
              color="#94a3b8"
              gap={20}
              className="opacity-30 dark:opacity-10"
            />

            <MiniMap
              nodeColor={(node) => {
                const colors = {
                  source: '#3B82F6',
                  transform: '#10B981',
                  destination: '#6B7280',
                  filter: '#10B981',
                  aggregate: '#10B981',
                  join: '#10B981',
                }
                return colors[node.type as keyof typeof colors] || '#6B7280'
              }}
              className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg"
            />

            <Controls
              className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-elegant"
            />

            {/* Toolbar */}
            <Panel position="top-left">
              <div className="flex items-center space-x-2 bg-white dark:bg-gray-800 rounded-lg shadow-elegant border border-gray-200 dark:border-gray-700 p-2">
                {!isPaletteVisible && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setIsPaletteVisible(true)}
                    icon={<LayersIcon className="h-4 w-4" />}
                  >
                    Nodes
                  </Button>
                )}

                <Button
                  variant="outline"
                  size="sm"
                  icon={<UndoIcon className="h-4 w-4" />}
                  disabled={historyIndex <= 0}
                >
                  Undo
                </Button>

                <Button
                  variant="outline"
                  size="sm"
                  icon={<RedoIcon className="h-4 w-4" />}
                  disabled={historyIndex >= history.length - 1}
                >
                  Redo
                </Button>

                <div className="w-px h-6 bg-gray-300 dark:bg-gray-600" />

                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleSave}
                  icon={<SaveIcon className="h-4 w-4" />}
                  disabled={readonly}
                >
                  Save
                </Button>

                <Button
                  variant="default"
                  size="sm"
                  onClick={handleExecute}
                  icon={<PlayIcon className="h-4 w-4" />}
                  disabled={!canExecute || readonly}
                >
                  Execute
                </Button>
              </div>
            </Panel>

            {/* Validation Panel */}
            {validationErrors.length > 0 && (
              <Panel position="top-right">
                <Card className="w-80">
                  <CardHeader>
                    <CardTitle className="text-sm text-warning-700 dark:text-warning-400">
                      Flow Validation
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ul className="space-y-1 text-sm">
                      {validationErrors.map((error, index) => (
                        <li key={index} className="flex items-start space-x-2 text-warning-600 dark:text-warning-400">
                          <div className="w-1 h-1 bg-warning-500 rounded-full mt-2 flex-shrink-0" />
                          <span>{error}</span>
                        </li>
                      ))}
                    </ul>
                  </CardContent>
                </Card>
              </Panel>
            )}

            {/* Node Statistics */}
            <Panel position="bottom-right">
              <div className="bg-white dark:bg-gray-800 rounded-lg shadow-elegant border border-gray-200 dark:border-gray-700 p-3">
                <div className="flex items-center space-x-4 text-sm">
                  <div>
                    <span className="text-gray-600 dark:text-gray-400">Nodes:</span>
                    <span className="font-medium ml-1">{nodes.length}</span>
                  </div>
                  <div>
                    <span className="text-gray-600 dark:text-gray-400">Edges:</span>
                    <span className="font-medium ml-1">{edges.length}</span>
                  </div>
                  {canExecute && (
                    <Badge variant="success" size="sm">
                      Ready
                    </Badge>
                  )}
                </div>
              </div>
            </Panel>
          </ReactFlow>
        </div>
      </div>
    </div>
  )
}

// Wrapper component with DnD context
export function FlowDesignerWrapper(props: FlowDesignerProps) {
  return (
    <DndProvider backend={HTML5Backend}>
      <ReactFlowProvider>
        <FlowDesigner {...props} />
      </ReactFlowProvider>
    </DndProvider>
  )
}