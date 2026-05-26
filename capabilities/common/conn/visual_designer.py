"""
APG Connection Management - Visual Flow Designer

Drag-and-drop visual flow designer with real-time collaboration,
template gallery, and intuitive flow creation capabilities.

Author: APG Platform Team
Version: 1.0.0
License: Proprietary - Datacraft © 2025
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

@dataclass
class FlowNode:
	"""Visual flow node representing connections, transformations, or actions."""

	id: str = field(default_factory=lambda: str(uuid4()))
	type: str = "generic"  # source, target, transform, filter, etc.
	name: str = "Untitled Node"
	position: Tuple[int, int] = (0, 0)
	size: Tuple[int, int] = (120, 80)

	# Node Configuration
	config: Dict[str, Any] = field(default_factory=dict)
	properties: Dict[str, Any] = field(default_factory=dict)

	# Visual Properties
	color: str = "#3498db"
	icon: str = "default"
	selected: bool = False

	# Connection Points
	input_ports: List[str] = field(default_factory=list)
	output_ports: List[str] = field(default_factory=list)

	# Metadata
	created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
	updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class FlowConnection:
	"""Connection between flow nodes."""

	id: str = field(default_factory=lambda: str(uuid4()))
	source_node_id: str = ""
	source_port: str = "output"
	target_node_id: str = ""
	target_port: str = "input"

	# Visual Properties
	color: str = "#34495e"
	width: int = 2
	style: str = "solid"  # solid, dashed, dotted
	selected: bool = False

	# Connection Metadata
	data_flow: bool = True
	validated: bool = False
	last_data_transfer: Optional[datetime] = None

@dataclass
class FlowCanvas:
	"""Visual canvas for flow design with collaboration support."""

	id: str = field(default_factory=lambda: str(uuid4()))
	name: str = "Untitled Flow"
	description: str = ""

	# Canvas Elements
	nodes: Dict[str, FlowNode] = field(default_factory=dict)
	connections: Dict[str, FlowConnection] = field(default_factory=dict)

	# Canvas Properties
	zoom: float = 1.0
	pan: Tuple[int, int] = (0, 0)
	grid_enabled: bool = True
	snap_to_grid: bool = True

	# Collaboration
	active_users: Set[str] = field(default_factory=set)
	user_cursors: Dict[str, Tuple[int, int]] = field(default_factory=dict)

	# Versioning
	version: int = 1
	created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
	updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
	created_by: str = "system"

@dataclass
class VisualFlowDesigner:
	"""
	Main visual flow designer with drag-and-drop capabilities,
	real-time collaboration, and comprehensive template support.
	"""

	# Active Canvases
	canvases: Dict[str, FlowCanvas] = field(default_factory=dict)

	# Templates and Presets
	templates: Dict[str, Dict[str, Any]] = field(default_factory=dict)
	node_library: Dict[str, Dict[str, Any]] = field(default_factory=dict)

	# Collaboration State
	active_sessions: Dict[str, Set[str]] = field(default_factory=dict)  # canvas_id -> user_ids

	def __post_init__(self):
		"""Initialize designer with default templates and nodes."""
		async def initialize() -> None:
			await self._initialize_templates()
			await self._initialize_node_library()

		try:
			loop = asyncio.get_running_loop()
		except RuntimeError:
			asyncio.run(initialize())
		else:
			loop.create_task(initialize())

	def _log_designer_operation(self, operation: str) -> None:
		"""Log designer operations following APG patterns."""
		print(f"Visual flow designer: {operation}")

	async def ensure_initialized(self) -> None:
		"""Ensure template and node libraries are available before use."""
		if not self.templates:
			await self._initialize_templates()
		if not self.node_library:
			await self._initialize_node_library()

	async def _initialize_templates(self) -> None:
		"""Initialize flow templates gallery."""
		self.templates = {
			"database_sync": {
				"name": "Database Synchronization",
				"description": "Sync data between two databases",
				"category": "integration",
				"nodes": [
					{"type": "source", "name": "Source DB", "position": (50, 100)},
					{"type": "transform", "name": "Data Transform", "position": (250, 100)},
					{"type": "target", "name": "Target DB", "position": (450, 100)}
				],
				"connections": [
					{"from": 0, "to": 1}, {"from": 1, "to": 2}
				]
			},
			"api_to_warehouse": {
				"name": "API to Data Warehouse",
				"description": "Extract API data to warehouse",
				"category": "etl",
				"nodes": [
					{"type": "source", "name": "REST API", "position": (50, 100)},
					{"type": "filter", "name": "Filter Data", "position": (200, 100)},
					{"type": "transform", "name": "Transform", "position": (350, 100)},
					{"type": "target", "name": "Warehouse", "position": (500, 100)}
				],
				"connections": [
					{"from": 0, "to": 1}, {"from": 1, "to": 2}, {"from": 2, "to": 3}
				]
			},
			"real_time_streaming": {
				"name": "Real-time Streaming",
				"description": "Stream data processing pipeline",
				"category": "streaming",
				"nodes": [
					{"type": "stream_source", "name": "Bytewax Stream", "position": (50, 100)},
					{"type": "processor", "name": "Stream Processor", "position": (250, 100)},
					{"type": "stream_target", "name": "Analytics DB", "position": (450, 100)}
				],
				"connections": [
					{"from": 0, "to": 1}, {"from": 1, "to": 2}
				]
			}
		}

	async def _initialize_node_library(self) -> None:
		"""Initialize node library with available components."""
		self.node_library = {
			# Source Nodes
			"postgres_source": {
				"name": "PostgreSQL Source",
				"type": "source",
				"icon": "database",
				"color": "#336791",
				"ports": {"output": ["data"]},
				"config_schema": {
					"host": {"type": "string", "required": True},
					"database": {"type": "string", "required": True}
				}
			},
			"rest_api_source": {
				"name": "REST API Source",
				"type": "source",
				"icon": "api",
				"color": "#27ae60",
				"ports": {"output": ["data"]},
				"config_schema": {
					"url": {"type": "string", "required": True},
					"method": {"type": "string", "default": "GET"}
				}
			},
			"bytewax_source": {
				"name": "Bytewax Source",
				"type": "stream_source",
				"icon": "stream",
				"color": "#e74c3c",
				"ports": {"output": ["stream"]},
				"config_schema": {
					"stream": {"type": "string", "required": True},
					"flow_id": {"type": "string", "required": True}
				}
			},

			# Transform Nodes
			"field_mapper": {
				"name": "Field Mapper",
				"type": "transform",
				"icon": "transform",
				"color": "#f39c12",
				"ports": {"input": ["data"], "output": ["data"]},
				"config_schema": {
					"mappings": {"type": "object", "required": True}
				}
			},
			"data_filter": {
				"name": "Data Filter",
				"type": "filter",
				"icon": "filter",
				"color": "#9b59b6",
				"ports": {"input": ["data"], "output": ["data"]},
				"config_schema": {
					"conditions": {"type": "array", "required": True}
				}
			},

			# Target Nodes
			"postgres_target": {
				"name": "PostgreSQL Target",
				"type": "target",
				"icon": "database",
				"color": "#336791",
				"ports": {"input": ["data"]},
				"config_schema": {
					"host": {"type": "string", "required": True},
					"table": {"type": "string", "required": True}
				}
			},
			"s3_target": {
				"name": "S3 Target",
				"type": "target",
				"icon": "cloud",
				"color": "#ff9900",
				"ports": {"input": ["data"]},
				"config_schema": {
					"bucket": {"type": "string", "required": True},
					"key": {"type": "string", "required": True}
				}
			}
		}

	async def create_canvas(
		self,
		name: str,
		created_by: str,
		description: str = ""
	) -> str:
		"""Create a new flow canvas."""
		canvas = FlowCanvas(
			name=name,
			description=description,
			created_by=created_by
		)

		self.canvases[canvas.id] = canvas
		self._log_designer_operation(f"Created canvas: {name}")

		return canvas.id

	async def add_node_to_canvas(
		self,
		canvas_id: str,
		node_type: str,
		position: Tuple[int, int],
		name: str = None
	) -> str:
		"""Add a node to the canvas."""
		assert canvas_id in self.canvases, f"Canvas {canvas_id} not found"

		canvas = self.canvases[canvas_id]

		# Get node template
		node_template = self.node_library.get(node_type, {})

		node = FlowNode(
			type=node_template.get("type", "generic"),
			name=name or node_template.get("name", "Untitled Node"),
			position=position,
			color=node_template.get("color", "#3498db"),
			icon=node_template.get("icon", "default"),
			input_ports=node_template.get("ports", {}).get("input", []),
			output_ports=node_template.get("ports", {}).get("output", []),
			config={"schema": node_template.get("config_schema", {})}
		)

		canvas.nodes[node.id] = node
		canvas.updated_at = datetime.now(timezone.utc)
		canvas.version += 1

		self._log_designer_operation(f"Added node {node_type} to canvas {canvas_id}")

		# Notify other users
		await self._broadcast_canvas_update(canvas_id, {
			"action": "node_added",
			"node_id": node.id,
			"node": self._serialize_node(node)
		})

		return node.id

	async def connect_nodes(
		self,
		canvas_id: str,
		source_node_id: str,
		target_node_id: str,
		source_port: str = "output",
		target_port: str = "input"
	) -> str:
		"""Connect two nodes on the canvas."""
		assert canvas_id in self.canvases, f"Canvas {canvas_id} not found"

		canvas = self.canvases[canvas_id]

		assert source_node_id in canvas.nodes, f"Source node {source_node_id} not found"
		assert target_node_id in canvas.nodes, f"Target node {target_node_id} not found"

		connection = FlowConnection(
			source_node_id=source_node_id,
			source_port=source_port,
			target_node_id=target_node_id,
			target_port=target_port
		)

		canvas.connections[connection.id] = connection
		canvas.updated_at = datetime.now(timezone.utc)
		canvas.version += 1

		self._log_designer_operation(f"Connected nodes {source_node_id} -> {target_node_id}")

		# Validate connection
		await self._validate_connection(canvas_id, connection.id)

		# Notify other users
		await self._broadcast_canvas_update(canvas_id, {
			"action": "connection_added",
			"connection_id": connection.id,
			"connection": self._serialize_connection(connection)
		})

		return connection.id

	async def create_flow_from_template(
		self,
		template_name: str,
		canvas_name: str,
		created_by: str
	) -> str:
		"""Create a new flow from a template."""
		await self.ensure_initialized()
		assert template_name in self.templates, f"Template {template_name} not found"

		template = self.templates[template_name]
		canvas_id = await self.create_canvas(canvas_name, created_by, template["description"])

		canvas = self.canvases[canvas_id]
		node_mapping = {}

		# Add nodes from template
		for i, node_template in enumerate(template["nodes"]):
			node = FlowNode(
				type=node_template["type"],
				name=node_template["name"],
				position=node_template["position"],
				color=self.node_library.get(node_template["type"], {}).get("color", "#3498db")
			)

			canvas.nodes[node.id] = node
			node_mapping[i] = node.id

		# Add connections from template
		for conn_template in template.get("connections", []):
			source_id = node_mapping[conn_template["from"]]
			target_id = node_mapping[conn_template["to"]]

			connection = FlowConnection(
				source_node_id=source_id,
				target_node_id=target_id
			)

			canvas.connections[connection.id] = connection

		canvas.updated_at = datetime.now(timezone.utc)
		self._log_designer_operation(f"Created flow from template: {template_name}")

		return canvas_id

	async def validate_flow(self, canvas_id: str) -> Dict[str, Any]:
		"""Validate complete flow for correctness and completeness."""
		assert canvas_id in self.canvases, f"Canvas {canvas_id} not found"

		canvas = self.canvases[canvas_id]
		validation_result = {
			"valid": True,
			"errors": [],
			"warnings": [],
			"node_count": len(canvas.nodes),
			"connection_count": len(canvas.connections)
		}

		# Check for source nodes
		source_nodes = [n for n in canvas.nodes.values() if n.type in ["source", "stream_source"]]
		if not source_nodes:
			validation_result["errors"].append("Flow must have at least one source node")
			validation_result["valid"] = False

		# Check for target nodes
		target_nodes = [n for n in canvas.nodes.values() if n.type in ["target", "stream_target"]]
		if not target_nodes:
			validation_result["errors"].append("Flow must have at least one target node")
			validation_result["valid"] = False

		# Check for disconnected nodes
		connected_nodes = set()
		for conn in canvas.connections.values():
			connected_nodes.add(conn.source_node_id)
			connected_nodes.add(conn.target_node_id)

		disconnected = set(canvas.nodes.keys()) - connected_nodes
		if disconnected:
			validation_result["warnings"].append(f"Disconnected nodes: {len(disconnected)}")

		# Validate individual connections
		for conn in canvas.connections.values():
			conn_validation = await self._validate_connection(canvas_id, conn.id)
			if not conn_validation["valid"]:
				validation_result["errors"].extend(conn_validation["errors"])
				validation_result["valid"] = False

		return validation_result

	async def _validate_connection(
		self,
		canvas_id: str,
		connection_id: str
	) -> Dict[str, Any]:
		"""Validate a specific connection."""
		canvas = self.canvases[canvas_id]
		connection = canvas.connections[connection_id]

		validation = {"valid": True, "errors": []}

		# Check if nodes exist
		if connection.source_node_id not in canvas.nodes:
			validation["errors"].append("Source node not found")
			validation["valid"] = False

		if connection.target_node_id not in canvas.nodes:
			validation["errors"].append("Target node not found")
			validation["valid"] = False

		# Check port compatibility (simplified)
		if validation["valid"]:
			source_node = canvas.nodes[connection.source_node_id]
			target_node = canvas.nodes[connection.target_node_id]

			if connection.source_port not in source_node.output_ports and source_node.output_ports:
				validation["warnings"] = validation.get("warnings", [])
				validation["warnings"].append("Source port may not exist")

		connection.validated = validation["valid"]
		return validation

	async def join_collaborative_session(
		self,
		canvas_id: str,
		user_id: str
	) -> Dict[str, Any]:
		"""Join collaborative editing session."""
		assert canvas_id in self.canvases, f"Canvas {canvas_id} not found"

		if canvas_id not in self.active_sessions:
			self.active_sessions[canvas_id] = set()

		self.active_sessions[canvas_id].add(user_id)

		canvas = self.canvases[canvas_id]
		canvas.active_users.add(user_id)

		self._log_designer_operation(f"User {user_id} joined canvas {canvas_id}")

		# Notify other users
		await self._broadcast_canvas_update(canvas_id, {
			"action": "user_joined",
			"user_id": user_id,
			"active_users": list(canvas.active_users)
		})

		return {
			"canvas": self._serialize_canvas(canvas),
			"active_users": list(canvas.active_users)
		}

	async def update_user_cursor(
		self,
		canvas_id: str,
		user_id: str,
		position: Tuple[int, int]
	) -> None:
		"""Update user cursor position for real-time collaboration."""
		if canvas_id in self.canvases:
			canvas = self.canvases[canvas_id]
			canvas.user_cursors[user_id] = position

			# Broadcast cursor update
			await self._broadcast_canvas_update(canvas_id, {
				"action": "cursor_moved",
				"user_id": user_id,
				"position": position
			})

	async def _broadcast_canvas_update(
		self,
		canvas_id: str,
		update_data: Dict[str, Any]
	) -> None:
		"""Broadcast canvas updates to all active users."""
		# In production, this would use WebSocket connections
		active_users = self.active_sessions.get(canvas_id, set())

		if active_users:
			self._log_designer_operation(
				f"Broadcasting update to {len(active_users)} users on canvas {canvas_id}"
			)

	def _serialize_node(self, node: FlowNode) -> Dict[str, Any]:
		"""Serialize node for transmission."""
		return {
			"id": node.id,
			"type": node.type,
			"name": node.name,
			"position": node.position,
			"size": node.size,
			"color": node.color,
			"icon": node.icon,
			"selected": node.selected,
			"config": node.config,
			"input_ports": node.input_ports,
			"output_ports": node.output_ports
		}

	def _serialize_connection(self, connection: FlowConnection) -> Dict[str, Any]:
		"""Serialize connection for transmission."""
		return {
			"id": connection.id,
			"source_node_id": connection.source_node_id,
			"source_port": connection.source_port,
			"target_node_id": connection.target_node_id,
			"target_port": connection.target_port,
			"color": connection.color,
			"width": connection.width,
			"style": connection.style,
			"validated": connection.validated
		}

	def _serialize_canvas(self, canvas: FlowCanvas) -> Dict[str, Any]:
		"""Serialize complete canvas for transmission."""
		return {
			"id": canvas.id,
			"name": canvas.name,
			"description": canvas.description,
			"nodes": {nid: self._serialize_node(node) for nid, node in canvas.nodes.items()},
			"connections": {cid: self._serialize_connection(conn) for cid, conn in canvas.connections.items()},
			"zoom": canvas.zoom,
			"pan": canvas.pan,
			"version": canvas.version,
			"active_users": list(canvas.active_users),
			"user_cursors": canvas.user_cursors
		}

	async def export_flow_definition(
		self,
		canvas_id: str,
		format: str = "apg"
	) -> Dict[str, Any]:
		"""Export flow as executable definition."""
		assert canvas_id in self.canvases, f"Canvas {canvas_id} not found"

		canvas = self.canvases[canvas_id]

		if format == "apg":
			# APG-specific flow definition
			flow_def = {
				"version": "1.0",
				"name": canvas.name,
				"description": canvas.description,
				"steps": [],
				"connections": []
			}

			# Convert nodes to steps
			for node in canvas.nodes.values():
				step = {
					"id": node.id,
					"name": node.name,
					"type": node.type,
					"config": node.config.get("user_config", {})
				}
				flow_def["steps"].append(step)

			# Convert connections to flow
			for conn in canvas.connections.values():
				connection = {
					"from": conn.source_node_id,
					"to": conn.target_node_id,
					"port": conn.source_port
				}
				flow_def["connections"].append(connection)

			return flow_def

		# Default format
		return self._serialize_canvas(canvas)
