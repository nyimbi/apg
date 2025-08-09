"""
APG Workflow Canvas Engine

High-performance canvas engine for workflow visualization and manipulation.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from uuid import uuid4
import json

from pydantic import BaseModel, Field, ConfigDict
from pydantic.functional_validators import AfterValidator
from typing_extensions import Annotated

logger = logging.getLogger(__name__)

class CanvasNode(BaseModel):
	"""Represents a node on the workflow canvas."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(..., description="Unique node identifier")
	component_type: str = Field(..., description="Component type")
	position: Dict[str, float] = Field(..., description="Node position (x, y)")
	size: Dict[str, float] = Field(default_factory=lambda: {"width": 200, "height": 100}, description="Node size")
	config: Dict[str, Any] = Field(default_factory=dict, description="Node configuration")
	
	# Visual properties
	label: str = Field(default="", description="Display label")
	icon: Optional[str] = Field(default=None, description="Node icon")
	color: str = Field(default="#3498db", description="Node color")
	border_color: str = Field(default="#2980b9", description="Border color")
	
	# State
	selected: bool = Field(default=False, description="Selection state")
	locked: bool = Field(default=False, description="Lock state")
	visible: bool = Field(default=True, description="Visibility state")
	
	# Ports
	input_ports: List[Dict[str, Any]] = Field(default_factory=list, description="Input connection ports")
	output_ports: List[Dict[str, Any]] = Field(default_factory=list, description="Output connection ports")
	
	# Metadata
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class CanvasConnection(BaseModel):
	"""Represents a connection between nodes."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(..., description="Unique connection identifier")
	source_node_id: str = Field(..., description="Source node ID")
	target_node_id: str = Field(..., description="Target node ID")
	source_port: str = Field(default="output", description="Source port name")
	target_port: str = Field(default="input", description="Target port name")
	
	# Visual properties
	color: str = Field(default="#7f8c8d", description="Connection color")
	width: float = Field(default=2.0, ge=1.0, le=10.0, description="Connection width")
	style: str = Field(default="solid", regex="^(solid|dashed|dotted)$", description="Connection style")
	
	# Path data for custom routing
	path_points: List[Dict[str, float]] = Field(default_factory=list, description="Custom path points")
	bezier_control: Optional[Dict[str, float]] = Field(default=None, description="Bezier curve control point")
	
	# State
	selected: bool = Field(default=False, description="Selection state")
	animated: bool = Field(default=False, description="Animation state")
	
	# Metadata
	data: Dict[str, Any] = Field(default_factory=dict, description="Connection data/configuration")
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class CanvasState(BaseModel):
	"""Represents the complete state of the canvas."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	# Viewport
	zoom: float = Field(default=1.0, ge=0.1, le=5.0, description="Zoom level")
	pan_x: float = Field(default=0.0, description="Horizontal pan offset")
	pan_y: float = Field(default=0.0, description="Vertical pan offset")
	
	# Canvas properties
	width: int = Field(default=5000, ge=1000, le=20000, description="Canvas width")
	height: int = Field(default=3000, ge=1000, le=15000, description="Canvas height")
	grid_size: int = Field(default=20, ge=5, le=50, description="Grid size")
	
	# Nodes and connections
	nodes: List[CanvasNode] = Field(default_factory=list, description="All nodes on canvas")
	connections: List[CanvasConnection] = Field(default_factory=list, description="All connections")
	
	# Selection state
	selected_nodes: List[str] = Field(default_factory=list, description="Selected node IDs")
	selected_connections: List[str] = Field(default_factory=list, description="Selected connection IDs")
	
	# UI state
	show_grid: bool = Field(default=True, description="Show grid")
	show_minimap: bool = Field(default=True, description="Show minimap")
	snap_to_grid: bool = Field(default=True, description="Enable grid snapping")

class CanvasEngine:
	"""
	High-performance canvas engine for workflow visualization.
	
	Features:
	- Infinite scrolling and zooming
	- Smart node positioning and routing
	- Real-time collaboration
	- Performance optimization with virtualization
	- Advanced selection and manipulation tools
	"""
	
	def __init__(self, config):
		self.config = config
		self.canvas_sessions: Dict[str, CanvasState] = {}
		self.spatial_index: Dict[str, Dict] = {}  # For efficient spatial queries
		self.is_initialized = False
		
		logger.info("Canvas engine initialized")
	
	async def initialize(self) -> None:
		"""Initialize the canvas engine."""
		try:
			self.is_initialized = True
			logger.info("Canvas engine initialization completed")
		except Exception as e:
			logger.error(f"Failed to initialize canvas engine: {e}")
			raise
	
	async def shutdown(self) -> None:
		"""Shutdown the canvas engine."""
		try:
			self.canvas_sessions.clear()
			self.spatial_index.clear()
			self.is_initialized = False
			logger.info("Canvas engine shutdown completed")
		except Exception as e:
			logger.error(f"Error during canvas engine shutdown: {e}")
	
	async def get_initial_state(self) -> Dict[str, Any]:
		"""Get initial canvas state for new session."""
		return CanvasState(
			zoom=1.0,
			pan_x=0.0,
			pan_y=0.0,
			width=self.config.canvas_width,
			height=self.config.canvas_height,
			grid_size=self.config.grid_size,
			snap_to_grid=self.config.snap_to_grid
		).model_dump()
	
	async def get_canvas_state(self, session_id: str) -> Dict[str, Any]:
		"""Get current canvas state for session."""
		try:
			if session_id not in self.canvas_sessions:
				self.canvas_sessions[session_id] = CanvasState()
			
			return self.canvas_sessions[session_id].model_dump()
		except Exception as e:
			logger.error(f"Failed to get canvas state: {e}")
			raise
	
	async def restore_state(self, session_id: str, state: Dict[str, Any]) -> None:
		"""Restore canvas to a specific state."""
		try:
			self.canvas_sessions[session_id] = CanvasState(**state)
			await self._update_spatial_index(session_id)
			logger.debug(f"Restored canvas state for session {session_id}")
		except Exception as e:
			logger.error(f"Failed to restore canvas state: {e}")
			raise
	
	async def add_node(self, session_id: str, component_type: str, position: Dict[str, float], config: Optional[Dict[str, Any]] = None) -> CanvasNode:
		"""Add a new node to the canvas."""
		try:
			if session_id not in self.canvas_sessions:
				self.canvas_sessions[session_id] = CanvasState()
			
			canvas = self.canvas_sessions[session_id]
			
			# Check node limit
			if len(canvas.nodes) >= self.config.max_nodes:
				raise ValueError(f"Maximum number of nodes ({self.config.max_nodes}) reached")
			
			# Snap position to grid if enabled
			if canvas.snap_to_grid:
				position = self._snap_to_grid(position, canvas.grid_size)
			
			# Create node
			node = CanvasNode(
				id=str(uuid4()),
				component_type=component_type,
				position=position,
				config=config or {},
				label=self._generate_node_label(component_type),
				input_ports=self._get_default_input_ports(component_type),
				output_ports=self._get_default_output_ports(component_type)
			)
			
			# Add to canvas
			canvas.nodes.append(node)
			
			# Update spatial index
			await self._update_spatial_index(session_id)
			
			logger.debug(f"Added node {node.id} to canvas {session_id}")
			return node
			
		except Exception as e:
			logger.error(f"Failed to add node: {e}")
			raise
	
	async def remove_node(self, session_id: str, node_id: str) -> None:
		"""Remove a node from the canvas."""
		try:
			if session_id not in self.canvas_sessions:
				return
			
			canvas = self.canvas_sessions[session_id]
			
			# Remove node
			canvas.nodes = [n for n in canvas.nodes if n.id != node_id]
			
			# Remove connections to/from this node
			canvas.connections = [
				c for c in canvas.connections 
				if c.source_node_id != node_id and c.target_node_id != node_id
			]
			
			# Update selection
			if node_id in canvas.selected_nodes:
				canvas.selected_nodes.remove(node_id)
			
			# Update spatial index
			await self._update_spatial_index(session_id)
			
			logger.debug(f"Removed node {node_id} from canvas {session_id}")
			
		except Exception as e:
			logger.error(f"Failed to remove node: {e}")
			raise
	
	async def move_node(self, session_id: str, node_id: str, position: Dict[str, float]) -> None:
		"""Move a node to a new position."""
		try:
			if session_id not in self.canvas_sessions:
				return
			
			canvas = self.canvas_sessions[session_id]
			
			# Find and update node
			for node in canvas.nodes:
				if node.id == node_id:
					if canvas.snap_to_grid:
						position = self._snap_to_grid(position, canvas.grid_size)
					
					node.position = position
					node.updated_at = datetime.now(timezone.utc)
					break
			
			# Update spatial index
			await self._update_spatial_index(session_id)
			
		except Exception as e:
			logger.error(f"Failed to move node: {e}")
			raise
	
	async def add_connection(self, session_id: str, source_id: str, target_id: str, source_port: str = "output", target_port: str = "input") -> CanvasConnection:
		"""Add a connection between two nodes."""
		try:
			if session_id not in self.canvas_sessions:
				self.canvas_sessions[session_id] = CanvasState()
			
			canvas = self.canvas_sessions[session_id]
			
			# Validate nodes exist
			source_node = next((n for n in canvas.nodes if n.id == source_id), None)
			target_node = next((n for n in canvas.nodes if n.id == target_id), None)
			
			if not source_node or not target_node:
				raise ValueError("Source or target node not found")
			
			# Check for existing connection
			existing = next((
				c for c in canvas.connections 
				if c.source_node_id == source_id and c.target_node_id == target_id 
				and c.source_port == source_port and c.target_port == target_port
			), None)
			
			if existing:
				raise ValueError("Connection already exists")
			
			# Create connection
			connection = CanvasConnection(
				id=str(uuid4()),
				source_node_id=source_id,
				target_node_id=target_id,
				source_port=source_port,
				target_port=target_port,
				path_points=self._calculate_connection_path(source_node, target_node)
			)
			
			# Add to canvas
			canvas.connections.append(connection)
			
			logger.debug(f"Added connection {connection.id} to canvas {session_id}")
			return connection
			
		except Exception as e:
			logger.error(f"Failed to add connection: {e}")
			raise
	
	async def remove_connection(self, session_id: str, connection_id: str) -> None:
		"""Remove a connection from the canvas."""
		try:
			if session_id not in self.canvas_sessions:
				return
			
			canvas = self.canvas_sessions[session_id]
			
			# Remove connection
			canvas.connections = [c for c in canvas.connections if c.id != connection_id]
			
			# Update selection
			if connection_id in canvas.selected_connections:
				canvas.selected_connections.remove(connection_id)
			
			logger.debug(f"Removed connection {connection_id} from canvas {session_id}")
			
		except Exception as e:
			logger.error(f"Failed to remove connection: {e}")
			raise
	
	async def update_node_config(self, session_id: str, node_id: str, config: Dict[str, Any]) -> None:
		"""Update node configuration."""
		try:
			if session_id not in self.canvas_sessions:
				return
			
			canvas = self.canvas_sessions[session_id]
			
			# Find and update node
			for node in canvas.nodes:
				if node.id == node_id:
					node.config.update(config)
					node.updated_at = datetime.now(timezone.utc)
					break
			
		except Exception as e:
			logger.error(f"Failed to update node config: {e}")
			raise
	
	async def select_nodes(self, session_id: str, node_ids: List[str], replace: bool = True) -> None:
		"""Select nodes on the canvas."""
		try:
			if session_id not in self.canvas_sessions:
				return
			
			canvas = self.canvas_sessions[session_id]
			
			if replace:
				canvas.selected_nodes.clear()
			
			# Update selection
			for node_id in node_ids:
				if node_id not in canvas.selected_nodes:
					canvas.selected_nodes.append(node_id)
			
			# Update node selection state
			for node in canvas.nodes:
				node.selected = node.id in canvas.selected_nodes
			
		except Exception as e:
			logger.error(f"Failed to select nodes: {e}")
			raise
	
	async def update_viewport(self, session_id: str, zoom: Optional[float] = None, pan_x: Optional[float] = None, pan_y: Optional[float] = None) -> None:
		"""Update canvas viewport."""
		try:
			if session_id not in self.canvas_sessions:
				return
			
			canvas = self.canvas_sessions[session_id]
			
			if zoom is not None:
				canvas.zoom = max(0.1, min(5.0, zoom))
			if pan_x is not None:
				canvas.pan_x = pan_x
			if pan_y is not None:
				canvas.pan_y = pan_y
			
		except Exception as e:
			logger.error(f"Failed to update viewport: {e}")
			raise
	
	async def get_nodes_in_region(self, session_id: str, x1: float, y1: float, x2: float, y2: float) -> List[CanvasNode]:
		"""Get nodes within a rectangular region."""
		try:
			if session_id not in self.canvas_sessions:
				return []
			
			canvas = self.canvas_sessions[session_id]
			nodes_in_region = []
			
			for node in canvas.nodes:
				node_x = node.position['x']
				node_y = node.position['y']
				node_w = node.size.get('width', 200)
				node_h = node.size.get('height', 100)
				
				# Check if node intersects with region
				if (node_x < x2 and node_x + node_w > x1 and 
				    node_y < y2 and node_y + node_h > y1):
					nodes_in_region.append(node)
			
			return nodes_in_region
			
		except Exception as e:
			logger.error(f"Failed to get nodes in region: {e}")
			return []
	
	async def auto_layout(self, session_id: str, algorithm: str = "hierarchical") -> None:
		"""Automatically layout nodes on the canvas."""
		try:
			if session_id not in self.canvas_sessions:
				return
			
			canvas = self.canvas_sessions[session_id]
			
			if algorithm == "hierarchical":
				await self._hierarchical_layout(canvas)
			elif algorithm == "force":
				await self._force_directed_layout(canvas)
			elif algorithm == "grid":
				await self._grid_layout(canvas)
			
			# Update spatial index
			await self._update_spatial_index(session_id)
			
		except Exception as e:
			logger.error(f"Failed to auto-layout: {e}")
			raise
	
	# Private methods
	
	def _snap_to_grid(self, position: Dict[str, float], grid_size: int) -> Dict[str, float]:
		"""Snap position to grid."""
		return {
			'x': round(position['x'] / grid_size) * grid_size,
			'y': round(position['y'] / grid_size) * grid_size
		}
	
	def _generate_node_label(self, component_type: str) -> str:
		"""Generate a default label for a node."""
		return component_type.replace('_', ' ').title()
	
	def _get_default_input_ports(self, component_type: str) -> List[Dict[str, Any]]:
		"""Get default input ports for component type."""
		return [{'name': 'input', 'type': 'any', 'required': False}]
	
	def _get_default_output_ports(self, component_type: str) -> List[Dict[str, Any]]:
		"""Get default output ports for component type."""
		return [{'name': 'output', 'type': 'any'}]
	
	def _calculate_connection_path(self, source_node: CanvasNode, target_node: CanvasNode) -> List[Dict[str, float]]:
		"""Calculate connection path between nodes."""
		source_x = source_node.position['x'] + source_node.size.get('width', 200)
		source_y = source_node.position['y'] + source_node.size.get('height', 100) / 2
		
		target_x = target_node.position['x']
		target_y = target_node.position['y'] + target_node.size.get('height', 100) / 2
		
		# Simple straight line for now
		return [
			{'x': source_x, 'y': source_y},
			{'x': target_x, 'y': target_y}
		]
	
	async def _update_spatial_index(self, session_id: str) -> None:
		"""Update spatial index for efficient queries."""
		try:
			if session_id not in self.canvas_sessions:
				return
			
			canvas = self.canvas_sessions[session_id]
			self.spatial_index[session_id] = {
				'nodes': {node.id: node.position for node in canvas.nodes},
				'bounds': self._calculate_bounds(canvas.nodes)
			}
			
		except Exception as e:
			logger.error(f"Failed to update spatial index: {e}")
	
	def _calculate_bounds(self, nodes: List[CanvasNode]) -> Dict[str, float]:
		"""Calculate bounding box of all nodes."""
		if not nodes:
			return {'min_x': 0, 'min_y': 0, 'max_x': 0, 'max_y': 0}
		
		positions = [node.position for node in nodes]
		return {
			'min_x': min(pos['x'] for pos in positions),
			'min_y': min(pos['y'] for pos in positions),
			'max_x': max(pos['x'] + 200 for pos in positions),  # Assuming default width
			'max_y': max(pos['y'] + 100 for pos in positions)   # Assuming default height
		}
	
	async def _hierarchical_layout(self, canvas: CanvasState) -> None:
		"""Apply hierarchical layout algorithm."""
		try:
			# Simple hierarchical layout - arrange nodes in layers
			layers: Dict[int, List[CanvasNode]] = {}
			visited = set()
			
			# Find root nodes (no incoming connections)
			root_nodes = []
			for node in canvas.nodes:
				has_incoming = any(
					conn.target_node_id == node.id 
					for conn in canvas.connections
				)
				if not has_incoming:
					root_nodes.append(node)
			
			# BFS to assign layers
			def assign_layer(node: CanvasNode, layer: int):
				if node.id in visited:
					return
				visited.add(node.id)
				
				if layer not in layers:
					layers[layer] = []
				layers[layer].append(node)
				
				# Process children
				for conn in canvas.connections:
					if conn.source_node_id == node.id:
						child_node = next((n for n in canvas.nodes if n.id == conn.target_node_id), None)
						if child_node:
							assign_layer(child_node, layer + 1)
			
			# Start with root nodes
			for root in root_nodes:
				assign_layer(root, 0)
			
			# Handle unvisited nodes (cycles or disconnected)
			for node in canvas.nodes:
				if node.id not in visited:
					assign_layer(node, len(layers))
			
			# Position nodes
			layer_height = 200
			node_spacing = 250
			
			for layer_idx, layer_nodes in layers.items():
				start_x = -(len(layer_nodes) - 1) * node_spacing / 2
				for i, node in enumerate(layer_nodes):
					node.position = {
						'x': start_x + i * node_spacing,
						'y': layer_idx * layer_height
					}
					node.updated_at = datetime.now(timezone.utc)
			
		except Exception as e:
			logger.error(f"Failed to apply hierarchical layout: {e}")
	
	async def _force_directed_layout(self, canvas: CanvasState) -> None:
		"""Apply force-directed layout algorithm."""
		try:
			# Simplified force-directed layout
			iterations = 50
			repulsion_force = 10000
			attraction_force = 0.1
			damping = 0.9
			
			for _ in range(iterations):
				forces = {node.id: {'x': 0, 'y': 0} for node in canvas.nodes}
				
				# Repulsion forces between all nodes
				for i, node1 in enumerate(canvas.nodes):
					for node2 in canvas.nodes[i+1:]:
						dx = node2.position['x'] - node1.position['x']
						dy = node2.position['y'] - node1.position['y']
						distance = max(1, (dx**2 + dy**2)**0.5)
						
						force = repulsion_force / (distance**2)
						fx = force * (dx / distance)
						fy = force * (dy / distance)
						
						forces[node1.id]['x'] -= fx
						forces[node1.id]['y'] -= fy
						forces[node2.id]['x'] += fx
						forces[node2.id]['y'] += fy
				
				# Attraction forces for connected nodes
				for conn in canvas.connections:
					source = next((n for n in canvas.nodes if n.id == conn.source_node_id), None)
					target = next((n for n in canvas.nodes if n.id == conn.target_node_id), None)
					
					if source and target:
						dx = target.position['x'] - source.position['x']
						dy = target.position['y'] - source.position['y']
						
						force = attraction_force
						forces[source.id]['x'] += force * dx
						forces[source.id]['y'] += force * dy
						forces[target.id]['x'] -= force * dx
						forces[target.id]['y'] -= force * dy
				
				# Apply forces
				for node in canvas.nodes:
					force = forces[node.id]
					node.position['x'] += force['x'] * damping
					node.position['y'] += force['y'] * damping
			
			# Update timestamps
			for node in canvas.nodes:
				node.updated_at = datetime.now(timezone.utc)
			
		except Exception as e:
			logger.error(f"Failed to apply force-directed layout: {e}")
	
	async def _grid_layout(self, canvas: CanvasState) -> None:
		"""Apply grid layout algorithm."""
		try:
			import math
			
			node_count = len(canvas.nodes)
			if node_count == 0:
				return
			
			# Calculate grid dimensions
			cols = math.ceil(math.sqrt(node_count))
			rows = math.ceil(node_count / cols)
			
			# Grid spacing
			spacing_x = 300
			spacing_y = 200
			
			# Position nodes
			for i, node in enumerate(canvas.nodes):
				row = i // cols
				col = i % cols
				
				node.position = {
					'x': col * spacing_x,
					'y': row * spacing_y
				}
				node.updated_at = datetime.now(timezone.utc)
			
		except Exception as e:
			logger.error(f"Failed to apply grid layout: {e}")