/**
 * APG Workflow Orchestration - React Drag-Drop Canvas Interface
 * 
 * Advanced workflow designer with drag-drop canvas, component palette, 
 * real-time validation, and collaborative editing.
 * 
 * © 2025 Datacraft. All rights reserved.
 * Author: Nyimbi Odero <nyimbi@gmail.com>
 */

// React and core dependencies
import React, { useState, useRef, useCallback, useEffect, useMemo } from 'react';
import { DndProvider, useDrag, useDrop } from 'react-dnd';
import { HTML5Backend } from 'react-dnd-html5-backend';
import { v4 as uuidv4 } from 'uuid';

// Canvas and visualization
import { TransformWrapper, TransformComponent } from 'react-zoom-pan-pinch';
import { Stage, Layer, Group, Rect, Circle, Line, Text } from 'react-konva';

// State management
import { useImmer } from 'use-immer';
import { useDebounce } from 'use-debounce';

// UI Components
import {
  Box,
  Paper,
  Drawer,
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Tooltip,
  Fab,
  Snackbar,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  Chip,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Card,
  CardContent,
  CardActions,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Grid,
  Avatar,
  Badge,
  Menu,
  MenuItem
} from '@mui/material';

// Icons
import {
  PlayArrow,
  Pause,
  Stop,
  Save,
  Undo,
  Redo,
  ZoomIn,
  ZoomOut,
  FitScreen,
  GridOn,
  Share,
  Comment,
  Validation,
  Settings,
  BugReport,
  Timeline,
  Dashboard,
  Storage,
  Api,
  CloudQueue,
  Schedule,
  Functions,
  Transform,
  Decision,
  Loop,
  Branch,
  Join,
  Error,
  CheckCircle,
  Warning,
  Info,
  ExpandMore,
  People,
  Visibility,
  Edit
} from '@mui/icons-material';

// Canvas constants
const CANVAS_WIDTH = 5000;
const CANVAS_HEIGHT = 3000;
const GRID_SIZE = 20;
const COMPONENT_WIDTH = 120;
const COMPONENT_HEIGHT = 80;
const CONNECTION_SNAP_DISTANCE = 15;

// Component types for drag and drop
const COMPONENT_TYPES = {
  TASK: 'task',
  DECISION: 'decision', 
  LOOP: 'loop',
  PARALLEL: 'parallel',
  JOIN: 'join',
  START: 'start',
  END: 'end',
  CONNECTOR: 'connector',
  TIMER: 'timer',
  HUMAN_TASK: 'human_task',
  SCRIPT: 'script',
  API_CALL: 'api_call',
  DATABASE: 'database',
  EMAIL: 'email',
  TRANSFORM: 'transform'
};

// Component categories for palette organization
const COMPONENT_CATEGORIES = {
  BASIC: {
    name: 'Basic Components',
    components: [
      { type: COMPONENT_TYPES.START, label: 'Start', icon: PlayArrow, color: '#4CAF50' },
      { type: COMPONENT_TYPES.END, label: 'End', icon: Stop, color: '#F44336' },
      { type: COMPONENT_TYPES.TASK, label: 'Task', icon: Functions, color: '#2196F3' },
      { type: COMPONENT_TYPES.DECISION, label: 'Decision', icon: Decision, color: '#FF9800' }
    ]
  },
  FLOW_CONTROL: {
    name: 'Flow Control',
    components: [
      { type: COMPONENT_TYPES.LOOP, label: 'Loop', icon: Loop, color: '#9C27B0' },
      { type: COMPONENT_TYPES.PARALLEL, label: 'Parallel', icon: Branch, color: '#607D8B' },
      { type: COMPONENT_TYPES.JOIN, label: 'Join', icon: Join, color: '#795548' },
      { type: COMPONENT_TYPES.TIMER, label: 'Timer', icon: Schedule, color: '#009688' }
    ]
  },
  INTEGRATIONS: {
    name: 'Integrations',
    components: [
      { type: COMPONENT_TYPES.API_CALL, label: 'API Call', icon: Api, color: '#3F51B5' },
      { type: COMPONENT_TYPES.DATABASE, label: 'Database', icon: Storage, color: '#FF5722' },
      { type: COMPONENT_TYPES.EMAIL, label: 'Email', icon: CloudQueue, color: '#E91E63' },
      { type: COMPONENT_TYPES.CONNECTOR, label: 'Connector', icon: Transform, color: '#FFC107' }
    ]
  },
  ADVANCED: {
    name: 'Advanced',
    components: [
      { type: COMPONENT_TYPES.SCRIPT, label: 'Script', icon: Functions, color: '#8BC34A' },
      { type: COMPONENT_TYPES.TRANSFORM, label: 'Transform', icon: Transform, color: '#CDDC39' },
      { type: COMPONENT_TYPES.HUMAN_TASK, label: 'Human Task', icon: People, color: '#00BCD4' }
    ]
  }
};

// Workflow validation rules
const VALIDATION_RULES = {
  MUST_HAVE_START: 'Workflow must have exactly one Start component',
  MUST_HAVE_END: 'Workflow must have at least one End component',
  NO_ORPHANED_COMPONENTS: 'All components must be connected to the workflow',
  DECISION_MUST_HAVE_BRANCHES: 'Decision components must have at least two outgoing connections',
  PARALLEL_MUST_HAVE_JOIN: 'Parallel components should have corresponding Join components',
  NO_CIRCULAR_DEPENDENCIES: 'Workflow must not contain circular dependencies',
  VALID_CONNECTIONS: 'All connections must be between compatible component types'
};

/**
 * Draggable Component Palette Item
 */
const PaletteItem = ({ component, onDragStart }) => {
  const [{ isDragging }, drag] = useDrag({
    type: 'component',
    item: { 
      type: component.type,
      label: component.label,
      icon: component.icon,
      color: component.color
    },
    collect: (monitor) => ({
      isDragging: monitor.isDragging()
    }),
    begin: onDragStart
  });

  const IconComponent = component.icon;

  return (
    <Card
      ref={drag}
      sx={{
        opacity: isDragging ? 0.5 : 1,
        cursor: 'grab',
        '&:hover': {
          transform: 'scale(1.02)',
          boxShadow: 3
        },
        transition: 'all 0.2s ease',
        mb: 1,
        backgroundColor: component.color + '10',
        border: `2px solid ${component.color}30`
      }}
    >
      <CardContent sx={{ p: 1.5, '&:last-child': { pb: 1.5 } }}>
        <Box display="flex" alignItems="center" gap={1}>
          <IconComponent sx={{ color: component.color, fontSize: 20 }} />
          <Typography variant="caption" fontWeight="medium">
            {component.label}
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
};

/**
 * Component Palette Panel
 */
const ComponentPalette = ({ isOpen, onToggle, onComponentDragStart }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [expandedCategory, setExpandedCategory] = useState('BASIC');

  const filteredCategories = useMemo(() => {
    if (!searchTerm) return COMPONENT_CATEGORIES;
    
    const filtered = {};
    Object.entries(COMPONENT_CATEGORIES).forEach(([key, category]) => {
      const matchingComponents = category.components.filter(comp =>
        comp.label.toLowerCase().includes(searchTerm.toLowerCase()) ||
        comp.type.toLowerCase().includes(searchTerm.toLowerCase())
      );
      
      if (matchingComponents.length > 0) {
        filtered[key] = { ...category, components: matchingComponents };
      }
    });
    
    return filtered;
  }, [searchTerm]);

  return (
    <Drawer
      variant="persistent"
      anchor="left"
      open={isOpen}
      sx={{
        width: 280,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: 280,
          boxSizing: 'border-box',
          top: 64, // Below app bar
          height: 'calc(100vh - 64px)'
        }
      }}
    >
      <Box sx={{ p: 2 }}>
        <Typography variant="h6" gutterBottom>
          Component Palette
        </Typography>
        
        <TextField
          fullWidth
          size="small"
          placeholder="Search components..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          sx={{ mb: 2 }}
        />

        {Object.entries(filteredCategories).map(([key, category]) => (
          <Accordion
            key={key}
            expanded={expandedCategory === key}
            onChange={() => setExpandedCategory(expandedCategory === key ? null : key)}
          >
            <AccordionSummary expandIcon={<ExpandMore />}>
              <Typography variant="subtitle2">{category.name}</Typography>
              <Chip 
                label={category.components.length} 
                size="small" 
                sx={{ ml: 'auto', mr: 1 }} 
              />
            </AccordionSummary>
            <AccordionDetails sx={{ pt: 0 }}>
              <Box>
                {category.components.map((component) => (
                  <PaletteItem
                    key={component.type}
                    component={component}
                    onDragStart={onComponentDragStart}
                  />
                ))}
              </Box>
            </AccordionDetails>
          </Accordion>
        ))}
      </Box>
    </Drawer>
  );
};

/**
 * Workflow Canvas Component
 */
const WorkflowComponent = ({ 
  component, 
  isSelected, 
  onSelect, 
  onUpdate, 
  onDelete,
  onConnectionStart,
  onConnectionEnd,
  collaborators = []
}) => {
  const [isEditing, setIsEditing] = useState(false);
  const [localLabel, setLocalLabel] = useState(component.label);

  const handleDoubleClick = useCallback(() => {
    setIsEditing(true);
  }, []);

  const handleLabelSave = useCallback(() => {
    onUpdate(component.id, { label: localLabel });
    setIsEditing(false);
  }, [component.id, localLabel, onUpdate]);

  const handleKeyPress = useCallback((e) => {
    if (e.key === 'Enter') {
      handleLabelSave();
    } else if (e.key === 'Escape') {
      setLocalLabel(component.label);
      setIsEditing(false);
    }
  }, [component.label, handleLabelSave]);

  const getStatusColor = useCallback(() => {
    switch (component.status) {
      case 'running': return '#4CAF50';
      case 'error': return '#F44336';
      case 'warning': return '#FF9800';
      case 'completed': return '#2196F3';
      default: return component.color || '#9E9E9E';
    }
  }, [component.status, component.color]);

  const IconComponent = component.icon || Functions;
  const statusColor = getStatusColor();

  // Show collaborator indicators for components being edited
  const editingCollaborators = collaborators.filter(c => c.editingComponent === component.id);

  return (
    <Group
      x={component.x}
      y={component.y}
      draggable
      onDragEnd={(e) => {
        onUpdate(component.id, {
          x: e.target.x(),
          y: e.target.y()
        });
      }}
      onDblClick={handleDoubleClick}
      onClick={(e) => {
        e.cancelBubble = true;
        onSelect(component.id);
      }}
    >
      {/* Main component rectangle */}
      <Rect
        width={COMPONENT_WIDTH}
        height={COMPONENT_HEIGHT}
        fill={statusColor + '20'}
        stroke={isSelected ? '#1976D2' : statusColor}
        strokeWidth={isSelected ? 3 : 2}
        cornerRadius={8}
        shadowEnabled={true}
        shadowBlur={isSelected ? 10 : 5}
        shadowColor={statusColor}
        shadowOpacity={0.3}
      />

      {/* Component icon area */}
      <Rect
        x={5}
        y={5}
        width={30}
        height={30}
        fill={statusColor}
        cornerRadius={4}
      />

      {/* Component label */}
      <Text
        x={40}
        y={15}
        text={isEditing ? localLabel : component.label}
        fontSize={12}
        fontFamily="Arial"
        fill="#333"
        width={COMPONENT_WIDTH - 45}
        ellipsis={true}
      />

      {/* Component type */}
      <Text
        x={40}
        y={30}
        text={component.type}
        fontSize={9}
        fontFamily="Arial"
        fill="#666"
        width={COMPONENT_WIDTH - 45}
        ellipsis={true}
      />

      {/* Status indicator */}
      {component.status && (
        <Circle
          x={COMPONENT_WIDTH - 15}
          y={15}
          radius={6}
          fill={statusColor}
          stroke="#fff"
          strokeWidth={2}
        />
      )}

      {/* Collaboration indicators */}
      {editingCollaborators.map((collaborator, index) => (
        <Circle
          key={collaborator.id}
          x={COMPONENT_WIDTH - 30 - (index * 15)}
          y={COMPONENT_HEIGHT - 15}
          radius={8}
          fill={collaborator.color}
          stroke="#fff"
          strokeWidth={2}
        />
      ))}

      {/* Connection points */}
      <Circle
        x={COMPONENT_WIDTH / 2}
        y={0}
        radius={4}
        fill="#fff"
        stroke={statusColor}
        strokeWidth={2}
        onMouseEnter={(e) => {
          e.target.getStage().container().style.cursor = 'crosshair';
        }}
        onMouseLeave={(e) => {
          e.target.getStage().container().style.cursor = 'default';
        }}
        onClick={() => onConnectionStart(component.id, 'input')}
      />
      
      <Circle
        x={COMPONENT_WIDTH / 2}
        y={COMPONENT_HEIGHT}
        radius={4}
        fill="#fff"
        stroke={statusColor}
        strokeWidth={2}
        onMouseEnter={(e) => {
          e.target.getStage().container().style.cursor = 'crosshair';
        }}
        onMouseLeave={(e) => {
          e.target.getStage().container().style.cursor = 'default';
        }}
        onClick={() => onConnectionStart(component.id, 'output')}
      />
    </Group>
  );
};

/**
 * Workflow Connection Component
 */
const WorkflowConnection = ({ connection, components, isSelected, onSelect, onDelete }) => {
  const sourceComponent = components.find(c => c.id === connection.sourceId);
  const targetComponent = components.find(c => c.id === connection.targetId);

  if (!sourceComponent || !targetComponent) return null;

  const sourceX = sourceComponent.x + COMPONENT_WIDTH / 2;
  const sourceY = sourceComponent.y + COMPONENT_HEIGHT;
  const targetX = targetComponent.x + COMPONENT_WIDTH / 2;
  const targetY = targetComponent.y;

  // Calculate control points for curved line
  const controlPointOffset = Math.abs(targetY - sourceY) / 2;
  const controlPoint1X = sourceX;
  const controlPoint1Y = sourceY + controlPointOffset;
  const controlPoint2X = targetX;
  const controlPoint2Y = targetY - controlPointOffset;

  return (
    <Group onClick={() => onSelect(connection.id)}>
      {/* Connection line */}
      <Line
        points={[
          sourceX, sourceY,
          controlPoint1X, controlPoint1Y,
          controlPoint2X, controlPoint2Y,
          targetX, targetY
        ]}
        stroke={isSelected ? '#1976D2' : '#666'}
        strokeWidth={isSelected ? 3 : 2}
        lineCap="round"
        tension={0.3}
        bezier={true}
      />

      {/* Arrow head */}
      <Line
        points={[
          targetX - 5, targetY - 10,
          targetX, targetY,
          targetX + 5, targetY - 10
        ]}
        stroke={isSelected ? '#1976D2' : '#666'}
        strokeWidth={isSelected ? 3 : 2}
        lineCap="round"
        closed={false}
      />

      {/* Connection label */}
      {connection.label && (
        <Text
          x={(sourceX + targetX) / 2 - 20}
          y={(sourceY + targetY) / 2 - 5}
          text={connection.label}
          fontSize={10}
          fontFamily="Arial"
          fill="#666"
          align="center"
        />
      )}
    </Group>
  );
};

/**
 * Main Workflow Canvas
 */
const WorkflowCanvas = ({ 
  workflow, 
  onWorkflowUpdate, 
  collaborators = [],
  isReadOnly = false 
}) => {
  // Canvas state
  const [components, setComponents] = useImmer(workflow?.components || []);
  const [connections, setConnections] = useImmer(workflow?.connections || []);
  const [selectedItems, setSelectedItems] = useState(new Set());
  const [isConnecting, setIsConnecting] = useState(false);
  const [connectionStart, setConnectionStart] = useState(null);
  const [canvasOffset, setCanvasOffset] = useState({ x: 0, y: 0 });
  const [zoomLevel, setZoomLevel] = useState(1);
  const [showGrid, setShowGrid] = useState(true);

  // UI state
  const [paletteOpen, setPaletteOpen] = useState(true);
  const [validationResults, setValidationResults] = useState([]);
  const [showValidation, setShowValidation] = useState(false);
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'info' });

  // Refs
  const stageRef = useRef();
  const canvasRef = useRef();

  // Debounced workflow updates
  const [debouncedComponents] = useDebounce(components, 500);
  const [debouncedConnections] = useDebounce(connections, 500);

  // Auto-save workflow changes
  useEffect(() => {
    if (onWorkflowUpdate && !isReadOnly) {
      onWorkflowUpdate({
        ...workflow,
        components: debouncedComponents,
        connections: debouncedConnections,
        updatedAt: new Date().toISOString()
      });
    }
  }, [debouncedComponents, debouncedConnections, workflow, onWorkflowUpdate, isReadOnly]);

  // Validation
  useEffect(() => {
    const results = validateWorkflow(components, connections);
    setValidationResults(results);
  }, [components, connections]);

  /**
   * Canvas drop target
   */
  const [{ isOver, canDrop }, drop] = useDrop({
    accept: 'component',
    drop: (item, monitor) => {
      if (isReadOnly) return;
      
      const clientOffset = monitor.getClientOffset();
      const canvasRect = canvasRef.current.getBoundingClientRect();
      
      const x = (clientOffset.x - canvasRect.left - canvasOffset.x) / zoomLevel;
      const y = (clientOffset.y - canvasRect.top - canvasOffset.y) / zoomLevel;
      
      addComponent(item, x, y);
    },
    collect: (monitor) => ({
      isOver: monitor.isOver(),
      canDrop: monitor.canDrop()
    })
  });

  /**
   * Add new component to canvas
   */
  const addComponent = useCallback((componentTemplate, x, y) => {
    const newComponent = {
      id: uuidv4(),
      type: componentTemplate.type,
      label: componentTemplate.label,
      icon: componentTemplate.icon,
      color: componentTemplate.color,
      x: Math.round(x / GRID_SIZE) * GRID_SIZE,
      y: Math.round(y / GRID_SIZE) * GRID_SIZE,
      status: 'pending',
      config: {},
      createdAt: new Date().toISOString(),
      createdBy: 'current_user' // Would be actual user ID
    };

    setComponents(draft => {
      draft.push(newComponent);
    });

    setSelectedItems(new Set([newComponent.id]));
    
    setSnackbar({
      open: true,
      message: `Added ${componentTemplate.label} component`,
      severity: 'success'
    });
  }, [setComponents]);

  /**
   * Update component properties
   */
  const updateComponent = useCallback((componentId, updates) => {
    setComponents(draft => {
      const component = draft.find(c => c.id === componentId);
      if (component) {
        Object.assign(component, updates, {
          updatedAt: new Date().toISOString(),
          updatedBy: 'current_user'
        });
      }
    });
  }, [setComponents]);

  /**
   * Delete selected components
   */
  const deleteSelectedComponents = useCallback(() => {
    if (isReadOnly || selectedItems.size === 0) return;

    setComponents(draft => {
      return draft.filter(c => !selectedItems.has(c.id));
    });

    setConnections(draft => {
      return draft.filter(conn => 
        !selectedItems.has(conn.id) &&
        !selectedItems.has(conn.sourceId) &&
        !selectedItems.has(conn.targetId)
      );
    });

    setSelectedItems(new Set());
    
    setSnackbar({
      open: true,
      message: `Deleted ${selectedItems.size} component(s)`,
      severity: 'info'
    });
  }, [selectedItems, setComponents, setConnections, isReadOnly]);

  /**
   * Start connection creation
   */
  const startConnection = useCallback((componentId, connectionType) => {
    if (isReadOnly) return;
    
    setIsConnecting(true);
    setConnectionStart({ componentId, connectionType });
  }, [isReadOnly]);

  /**
   * Complete connection creation
   */
  const completeConnection = useCallback((targetComponentId, connectionType) => {
    if (!isConnecting || !connectionStart || isReadOnly) return;

    // Prevent self-connections
    if (connectionStart.componentId === targetComponentId) {
      setIsConnecting(false);
      setConnectionStart(null);
      return;
    }

    // Check for existing connection
    const existingConnection = connections.find(conn =>
      conn.sourceId === connectionStart.componentId &&
      conn.targetId === targetComponentId
    );

    if (existingConnection) {
      setSnackbar({
        open: true,
        message: 'Connection already exists',
        severity: 'warning'
      });
      setIsConnecting(false);
      setConnectionStart(null);
      return;
    }

    const newConnection = {
      id: uuidv4(),
      sourceId: connectionStart.componentId,
      targetId: targetComponentId,
      sourceType: connectionStart.connectionType,
      targetType: connectionType,
      label: '',
      condition: null,
      createdAt: new Date().toISOString(),
      createdBy: 'current_user'
    };

    setConnections(draft => {
      draft.push(newConnection);
    });

    setIsConnecting(false);
    setConnectionStart(null);
    
    setSnackbar({
      open: true,
      message: 'Connection created',
      severity: 'success'
    });
  }, [isConnecting, connectionStart, connections, setConnections, isReadOnly]);

  /**
   * Canvas click handler
   */
  const handleCanvasClick = useCallback((e) => {
    // If clicking on empty canvas, clear selection
    if (e.target === e.target.getStage()) {
      setSelectedItems(new Set());
    }
  }, []);

  /**
   * Keyboard shortcuts
   */
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (isReadOnly) return;
      
      // Delete selected components
      if ((e.key === 'Delete' || e.key === 'Backspace') && selectedItems.size > 0) {
        e.preventDefault();
        deleteSelectedComponents();
      }
      
      // Select all
      if (e.ctrlKey && e.key === 'a') {
        e.preventDefault();
        setSelectedItems(new Set(components.map(c => c.id)));
      }
      
      // Escape to cancel connection
      if (e.key === 'Escape' && isConnecting) {
        setIsConnecting(false);
        setConnectionStart(null);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [selectedItems, components, deleteSelectedComponents, isConnecting, isReadOnly]);

  /**
   * Render grid
   */
  const renderGrid = useCallback(() => {
    if (!showGrid) return null;

    const gridLines = [];
    const gridColor = '#E0E0E0';
    
    // Vertical lines
    for (let i = 0; i <= CANVAS_WIDTH; i += GRID_SIZE) {
      gridLines.push(
        <Line
          key={`v-${i}`}
          points={[i, 0, i, CANVAS_HEIGHT]}
          stroke={gridColor}
          strokeWidth={0.5}
        />
      );
    }
    
    // Horizontal lines
    for (let i = 0; i <= CANVAS_HEIGHT; i += GRID_SIZE) {
      gridLines.push(
        <Line
          key={`h-${i}`}
          points={[0, i, CANVAS_WIDTH, i]}
          stroke={gridColor}
          strokeWidth={0.5}
        />
      );
    }
    
    return gridLines;
  }, [showGrid]);

  return (
    <DndProvider backend={HTML5Backend}>
      <Box sx={{ display: 'flex', height: '100vh' }}>
        {/* Component Palette */}
        <ComponentPalette
          isOpen={paletteOpen}
          onToggle={() => setPaletteOpen(!paletteOpen)}
          onComponentDragStart={() => {}}
        />

        {/* Main Canvas Area */}
        <Box
          sx={{
            flexGrow: 1,
            ml: paletteOpen ? '280px' : 0,
            transition: 'margin-left 0.3s ease',
            position: 'relative',
            overflow: 'hidden'
          }}
        >
          {/* Canvas Toolbar */}
          <AppBar position="static" color="default" elevation={1}>
            <Toolbar variant="dense">
              <IconButton onClick={() => setPaletteOpen(!paletteOpen)}>
                <GridOn />
              </IconButton>
              
              <Typography variant="h6" sx={{ flexGrow: 1, ml: 2 }}>
                {workflow?.name || 'Untitled Workflow'}
              </Typography>

              {/* Validation Status */}
              <Tooltip title={`${validationResults.length} validation issues`}>
                <IconButton 
                  color={validationResults.length > 0 ? 'error' : 'success'}
                  onClick={() => setShowValidation(true)}
                >
                  <Badge badgeContent={validationResults.length} color="error">
                    <Validation />
                  </Badge>
                </IconButton>
              </Tooltip>

              {/* Collaborators */}
              {collaborators.length > 0 && (
                <Box sx={{ display: 'flex', ml: 2 }}>
                  {collaborators.slice(0, 3).map((collaborator) => (
                    <Tooltip key={collaborator.id} title={collaborator.name}>
                      <Avatar
                        sx={{
                          width: 32,
                          height: 32,
                          ml: -1,
                          border: '2px solid white',
                          backgroundColor: collaborator.color
                        }}
                      >
                        {collaborator.name[0]}
                      </Avatar>
                    </Tooltip>
                  ))}
                  {collaborators.length > 3 && (
                    <Tooltip title={`+${collaborators.length - 3} more`}>
                      <Avatar sx={{ width: 32, height: 32, ml: -1, fontSize: 12 }}>
                        +{collaborators.length - 3}
                      </Avatar>
                    </Tooltip>
                  )}
                </Box>
              )}

              {/* Canvas Controls */}
              <Box sx={{ ml: 2 }}>
                <IconButton onClick={() => setZoomLevel(z => Math.min(z * 1.2, 3))}>
                  <ZoomIn />
                </IconButton>
                <IconButton onClick={() => setZoomLevel(z => Math.max(z * 0.8, 0.1))}>
                  <ZoomOut />
                </IconButton>
                <IconButton onClick={() => setZoomLevel(1)}>
                  <FitScreen />
                </IconButton>
                <IconButton onClick={() => setShowGrid(!showGrid)}>
                  <GridOn color={showGrid ? 'primary' : 'inherit'} />
                </IconButton>
              </Box>
            </Toolbar>
          </AppBar>

          {/* Canvas */}
          <Box
            ref={(node) => {
              canvasRef.current = node;
              drop(node);
            }}
            sx={{
              width: '100%',
              height: 'calc(100vh - 48px)',
              backgroundColor: isOver && canDrop ? '#E3F2FD' : '#FAFAFA',
              position: 'relative',
              cursor: isConnecting ? 'crosshair' : 'default'
            }}
          >
            <TransformWrapper
              initialScale={zoomLevel}
              minScale={0.1}
              maxScale={3}
              centerOnInit={true}
              onTransformed={(ref, state) => {
                setCanvasOffset({ x: state.positionX, y: state.positionY });
                setZoomLevel(state.scale);
              }}
            >
              <TransformComponent>
                <Stage
                  ref={stageRef}
                  width={CANVAS_WIDTH}
                  height={CANVAS_HEIGHT}
                  onClick={handleCanvasClick}
                >
                  <Layer>
                    {/* Grid */}
                    {renderGrid()}

                    {/* Connections */}
                    {connections.map((connection) => (
                      <WorkflowConnection
                        key={connection.id}
                        connection={connection}
                        components={components}
                        isSelected={selectedItems.has(connection.id)}
                        onSelect={(id) => setSelectedItems(new Set([id]))}
                        onDelete={() => {
                          setConnections(draft => 
                            draft.filter(c => c.id !== connection.id)
                          );
                        }}
                      />
                    ))}

                    {/* Components */}
                    {components.map((component) => (
                      <WorkflowComponent
                        key={component.id}
                        component={component}
                        isSelected={selectedItems.has(component.id)}
                        onSelect={(id) => setSelectedItems(new Set([id]))}
                        onUpdate={updateComponent}
                        onDelete={deleteSelectedComponents}
                        onConnectionStart={startConnection}
                        onConnectionEnd={completeConnection}
                        collaborators={collaborators}
                      />
                    ))}
                  </Layer>
                </Stage>
              </TransformComponent>
            </TransformWrapper>
          </Box>

          {/* Floating Action Buttons */}
          {!isReadOnly && (
            <Box sx={{ position: 'fixed', bottom: 16, right: 16 }}>
              <Fab
                color="primary"
                sx={{ mb: 1 }}
                onClick={() => setShowValidation(true)}
              >
                <Badge badgeContent={validationResults.length} color="error">
                  <Validation />
                </Badge>
              </Fab>
            </Box>
          )}
        </Box>

        {/* Validation Dialog */}
        <Dialog
          open={showValidation}
          onClose={() => setShowValidation(false)}
          maxWidth="md"
          fullWidth
        >
          <DialogTitle>Workflow Validation</DialogTitle>
          <DialogContent>
            <WorkflowValidationPanel
              validationResults={validationResults}
              components={components}
              connections={connections}
            />
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setShowValidation(false)}>Close</Button>
          </DialogActions>
        </Dialog>

        {/* Snackbar for notifications */}
        <Snackbar
          open={snackbar.open}
          autoHideDuration={4000}
          onClose={() => setSnackbar({ ...snackbar, open: false })}
        >
          <Alert severity={snackbar.severity} onClose={() => setSnackbar({ ...snackbar, open: false })}>
            {snackbar.message}
          </Alert>
        </Snackbar>
      </Box>
    </DndProvider>
  );
};

/**
 * Workflow Validation Panel
 */
const WorkflowValidationPanel = ({ validationResults, components, connections }) => {
  return (
    <Box>
      {validationResults.length === 0 ? (
        <Box display="flex" alignItems="center" gap={2} p={2}>
          <CheckCircle color="success" />
          <Typography>Workflow validation passed! No issues found.</Typography>
        </Box>
      ) : (
        <List>
          {validationResults.map((result, index) => (
            <ListItem key={index}>
              <ListItemIcon>
                {result.severity === 'error' ? (
                  <Error color="error" />
                ) : result.severity === 'warning' ? (
                  <Warning color="warning" />
                ) : (
                  <Info color="info" />
                )}
              </ListItemIcon>
              <ListItemText
                primary={result.message}
                secondary={result.suggestion}
              />
            </ListItem>
          ))}
        </List>
      )}
      
      <Box mt={2} p={2} bgcolor="grey.50" borderRadius={1}>
        <Typography variant="body2" color="textSecondary">
          Components: {components.length} | Connections: {connections.length}
        </Typography>
      </Box>
    </Box>
  );
};

/**
 * Workflow validation logic
 */
const validateWorkflow = (components, connections) => {
  const results = [];

  // Check for start component
  const startComponents = components.filter(c => c.type === COMPONENT_TYPES.START);
  if (startComponents.length === 0) {
    results.push({
      severity: 'error',
      message: 'Workflow must have a Start component',
      suggestion: 'Add a Start component from the Basic Components palette'
    });
  } else if (startComponents.length > 1) {
    results.push({
      severity: 'warning',
      message: 'Workflow has multiple Start components',
      suggestion: 'Consider using only one Start component'
    });
  }

  // Check for end component
  const endComponents = components.filter(c => c.type === COMPONENT_TYPES.END);
  if (endComponents.length === 0) {
    results.push({
      severity: 'error',
      message: 'Workflow must have at least one End component',
      suggestion: 'Add an End component from the Basic Components palette'
    });
  }

  // Check for orphaned components
  const connectedComponentIds = new Set([
    ...connections.map(c => c.sourceId),
    ...connections.map(c => c.targetId)
  ]);
  
  const orphanedComponents = components.filter(c => 
    !connectedComponentIds.has(c.id) && 
    c.type !== COMPONENT_TYPES.START &&
    components.length > 1
  );

  orphanedComponents.forEach(component => {
    results.push({
      severity: 'warning',
      message: `Component "${component.label}" is not connected`,
      suggestion: 'Connect this component to the workflow or remove it'
    });
  });

  // Check decision components
  components.filter(c => c.type === COMPONENT_TYPES.DECISION).forEach(decision => {
    const outgoingConnections = connections.filter(c => c.sourceId === decision.id);
    if (outgoingConnections.length < 2) {
      results.push({
        severity: 'warning',
        message: `Decision component "${decision.label}" should have at least 2 outgoing connections`,
        suggestion: 'Add decision branches to represent different outcomes'
      });
    }
  });

  // Check for circular dependencies (simplified)
  const hasCircularDependency = checkCircularDependencies(components, connections);
  if (hasCircularDependency) {
    results.push({
      severity: 'error',
      message: 'Workflow contains circular dependencies',
      suggestion: 'Review connections to ensure workflow has a clear flow direction'
    });
  }

  return results;
};

/**
 * Check for circular dependencies in workflow
 */
const checkCircularDependencies = (components, connections) => {
  // Simple DFS-based cycle detection
  const visited = new Set();
  const recursionStack = new Set();

  const hasCycle = (nodeId) => {
    if (recursionStack.has(nodeId)) return true;
    if (visited.has(nodeId)) return false;

    visited.add(nodeId);
    recursionStack.add(nodeId);

    const outgoingConnections = connections.filter(c => c.sourceId === nodeId);
    for (const connection of outgoingConnections) {
      if (hasCycle(connection.targetId)) return true;
    }

    recursionStack.delete(nodeId);
    return false;
  };

  for (const component of components) {
    if (!visited.has(component.id)) {
      if (hasCycle(component.id)) return true;
    }
  }

  return false;
};

export default WorkflowCanvas;