"""
© 2025 Datacraft
Advanced Visualization & AR System for Workflow Orchestration

This module provides advanced 3D visualization and augmented reality capabilities
for workflow orchestration, including holographic workflow representation,
AR debugging interfaces, and spatial manipulation tools.
"""

import asyncio
import json
import math
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, Field, ConfigDict, AfterValidator
from annotated_types import Annotated
from uuid_extensions import uuid7str

from apg.core.base_service import APGBaseService
from apg.core.database import DatabaseManager
from apg.core.websocket import WebSocketManager
from apg.core.metrics import MetricsCollector
from apg.common.logging import get_logger
from apg.common.exceptions import APGException

logger = get_logger(__name__)

class VisualizationMode(str, Enum):
	"""Visualization display modes"""
	FLAT_2D = "flat_2d"
	LAYERED_2D = "layered_2d"
	ISOMETRIC_3D = "isometric_3d"
	HOLOGRAPHIC_3D = "holographic_3d"
	AUGMENTED_REALITY = "augmented_reality"
	VIRTUAL_REALITY = "virtual_reality"

class SpatialDimension(str, Enum):
	"""Spatial dimensions for 3D positioning"""
	X_AXIS = "x"
	Y_AXIS = "y"
	Z_AXIS = "z"
	TIME_AXIS = "t"

class InteractionMode(str, Enum):
	"""User interaction modes"""
	MOUSE_KEYBOARD = "mouse_keyboard"
	TOUCH_GESTURE = "touch_gesture"
	HAND_TRACKING = "hand_tracking"
	EYE_TRACKING = "eye_tracking"
	VOICE_CONTROL = "voice_control"
	BRAIN_INTERFACE = "brain_interface"

class RenderingEngine(str, Enum):
	"""3D rendering engines"""
	WEBGL = "webgl"
	WEBGPU = "webgpu"
	THREE_JS = "three_js"
	BABYLON_JS = "babylon_js"
	UNITY_WEBGL = "unity_webgl"
	CUSTOM_SHADER = "custom_shader"

class Vector3D(BaseModel):
	"""3D vector representation"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	x: float = 0.0
	y: float = 0.0
	z: float = 0.0
	
	def magnitude(self) -> float:
		return math.sqrt(self.x**2 + self.y**2 + self.z**2)
	
	def normalize(self) -> 'Vector3D':
		mag = self.magnitude()
		if mag == 0:
			return Vector3D()
		return Vector3D(x=self.x/mag, y=self.y/mag, z=self.z/mag)
	
	def dot(self, other: 'Vector3D') -> float:
		return self.x * other.x + self.y * other.y + self.z * other.z
	
	def cross(self, other: 'Vector3D') -> 'Vector3D':
		return Vector3D(
			x=self.y * other.z - self.z * other.y,
			y=self.z * other.x - self.x * other.z,
			z=self.x * other.y - self.y * other.x
		)

class Quaternion(BaseModel):
	"""Quaternion for 3D rotations"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	w: float = 1.0
	x: float = 0.0
	y: float = 0.0
	z: float = 0.0
	
	def normalize(self) -> 'Quaternion':
		mag = math.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
		if mag == 0:
			return Quaternion()
		return Quaternion(w=self.w/mag, x=self.x/mag, y=self.y/mag, z=self.z/mag)
	
	def multiply(self, other: 'Quaternion') -> 'Quaternion':
		return Quaternion(
			w=self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
			x=self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
			y=self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
			z=self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
		)

class Transform3D(BaseModel):
	"""3D transformation matrix"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	position: Vector3D = Field(default_factory=Vector3D)
	rotation: Quaternion = Field(default_factory=Quaternion)
	scale: Vector3D = Field(default_factory=lambda: Vector3D(x=1.0, y=1.0, z=1.0))
	
	def to_matrix(self) -> List[List[float]]:
		"""Convert to 4x4 transformation matrix"""
		# Rotation matrix from quaternion
		q = self.rotation.normalize()
		r = [
			[1-2*(q.y**2 + q.z**2), 2*(q.x*q.y - q.w*q.z), 2*(q.x*q.z + q.w*q.y)],
			[2*(q.x*q.y + q.w*q.z), 1-2*(q.x**2 + q.z**2), 2*(q.y*q.z - q.w*q.x)],
			[2*(q.x*q.z - q.w*q.y), 2*(q.y*q.z + q.w*q.x), 1-2*(q.x**2 + q.y**2)]
		]
		
		# Apply scale and translation
		matrix = [
			[r[0][0]*self.scale.x, r[0][1]*self.scale.y, r[0][2]*self.scale.z, self.position.x],
			[r[1][0]*self.scale.x, r[1][1]*self.scale.y, r[1][2]*self.scale.z, self.position.y],
			[r[2][0]*self.scale.x, r[2][1]*self.scale.y, r[2][2]*self.scale.z, self.position.z],
			[0, 0, 0, 1]
		]
		
		return matrix

class Material3D(BaseModel):
	"""3D material properties"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	name: str
	color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)  # RGBA
	metallic: float = Field(default=0.0, ge=0.0, le=1.0)
	roughness: float = Field(default=0.5, ge=0.0, le=1.0)
	emission: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # RGB
	texture_diffuse: Optional[str] = None
	texture_normal: Optional[str] = None
	texture_metallic: Optional[str] = None
	texture_roughness: Optional[str] = None
	shader_program: Optional[str] = None

class Geometry3D(BaseModel):
	"""3D geometry definition"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	geometry_type: str  # box, sphere, cylinder, plane, mesh
	vertices: List[Vector3D] = Field(default_factory=list)
	indices: List[int] = Field(default_factory=list)
	normals: List[Vector3D] = Field(default_factory=list)
	uvs: List[Tuple[float, float]] = Field(default_factory=list)
	parameters: Dict[str, Any] = Field(default_factory=dict)  # size, radius, etc.

class SceneObject3D(BaseModel):
	"""3D scene object"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	name: str
	object_type: str  # workflow_node, connection, data_flow, etc.
	transform: Transform3D = Field(default_factory=Transform3D)
	geometry: Geometry3D
	material: Material3D
	parent_id: Optional[str] = None
	children_ids: List[str] = Field(default_factory=list)
	metadata: Dict[str, Any] = Field(default_factory=dict)
	visible: bool = True
	interactive: bool = True
	animation_state: Dict[str, Any] = Field(default_factory=dict)

class Camera3D(BaseModel):
	"""3D camera configuration"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	name: str
	camera_type: str = "perspective"  # perspective, orthographic, panoramic
	transform: Transform3D = Field(default_factory=Transform3D)
	fov: float = Field(default=75.0, ge=1.0, le=179.0)  # Field of view in degrees
	aspect_ratio: float = Field(default=16/9, gt=0)
	near_plane: float = Field(default=0.1, gt=0)
	far_plane: float = Field(default=1000.0, gt=0)
	target: Vector3D = Field(default_factory=Vector3D)
	up_vector: Vector3D = Field(default_factory=lambda: Vector3D(x=0, y=1, z=0))

class Light3D(BaseModel):
	"""3D lighting configuration"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	name: str
	light_type: str = "directional"  # directional, point, spot, ambient, area
	transform: Transform3D = Field(default_factory=Transform3D)
	color: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # RGB
	intensity: float = Field(default=1.0, ge=0.0)
	range: Optional[float] = Field(default=None, ge=0.0)
	spot_angle: Optional[float] = Field(default=None, ge=0.0, le=180.0)
	cast_shadows: bool = True
	shadow_resolution: int = Field(default=1024, ge=256, le=4096)

class Scene3D(BaseModel):
	"""Complete 3D scene definition"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	name: str
	workflow_id: str
	objects: Dict[str, SceneObject3D] = Field(default_factory=dict)
	cameras: Dict[str, Camera3D] = Field(default_factory=dict)
	lights: Dict[str, Light3D] = Field(default_factory=dict)
	active_camera_id: Optional[str] = None
	background_color: Tuple[float, float, float, float] = (0.1, 0.1, 0.1, 1.0)
	environment_map: Optional[str] = None
	fog_enabled: bool = False
	fog_color: Tuple[float, float, float] = (0.5, 0.5, 0.5)
	fog_density: float = 0.01
	physics_enabled: bool = False
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)

class ARMarker(BaseModel):
	"""Augmented reality marker"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	marker_type: str  # qr_code, image, object, location
	pattern_data: str  # QR code data, image URL, etc.
	transform: Transform3D = Field(default_factory=Transform3D)
	scale_factor: float = Field(default=1.0, gt=0)
	tracking_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
	is_tracked: bool = False
	content_object_ids: List[str] = Field(default_factory=list)

class HolographicDisplay(BaseModel):
	"""Holographic display configuration"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	display_type: str  # looking_glass, hololens, magic_leap, custom
	resolution: Tuple[int, int] = (1920, 1080)
	viewport_count: int = Field(default=45, ge=1, le=100)  # For light field displays
	quilt_settings: Dict[str, Any] = Field(default_factory=dict)
	calibration_data: Dict[str, Any] = Field(default_factory=dict)
	depth_range: Tuple[float, float] = (-1.0, 1.0)
	brightness: float = Field(default=1.0, ge=0.0, le=2.0)
	contrast: float = Field(default=1.0, ge=0.0, le=2.0)

class SpatialInteraction(BaseModel):
	"""Spatial interaction event"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	user_id: str
	interaction_type: str  # select, move, rotate, scale, gesture
	target_object_id: str
	start_position: Vector3D
	current_position: Vector3D
	start_rotation: Quaternion
	current_rotation: Quaternion
	gesture_data: Dict[str, Any] = Field(default_factory=dict)
	timestamp: datetime = Field(default_factory=datetime.utcnow)
	is_active: bool = True

class VisualizationConfig(BaseModel):
	"""Configuration for advanced visualization system"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	default_visualization_mode: VisualizationMode = VisualizationMode.ISOMETRIC_3D
	rendering_engine: RenderingEngine = RenderingEngine.THREE_JS
	max_scene_objects: int = 10000
	auto_level_of_detail: bool = True
	enable_physics: bool = True
	enable_shadows: bool = True
	shadow_quality: str = "medium"  # low, medium, high, ultra
	anti_aliasing: bool = True
	frame_rate_target: int = 60
	adaptive_quality: bool = True
	vr_enabled: bool = False
	ar_enabled: bool = False
	holographic_enabled: bool = False
	hand_tracking_enabled: bool = False
	eye_tracking_enabled: bool = False

class WorkflowSpatialMapper:
	"""Maps workflow components to 3D spatial representations"""
	
	def __init__(self, config: VisualizationConfig):
		self.config = config
		self.node_spacing = 5.0
		self.layer_height = 3.0
		self.connection_height = 0.5
	
	async def map_workflow_to_3d(self, workflow_data: Dict[str, Any]) -> Scene3D:
		"""Convert workflow to 3D scene"""
		try:
			scene = Scene3D(
				name=f"Workflow_{workflow_data.get('id', 'Unknown')}",
				workflow_id=workflow_data.get('id', '')
			)
			
			# Create workflow nodes as 3D objects
			nodes = workflow_data.get('nodes', [])
			await self._create_workflow_nodes(scene, nodes)
			
			# Create connections as 3D flow lines
			connections = workflow_data.get('connections', [])
			await self._create_workflow_connections(scene, connections, nodes)
			
			# Create data flow visualizations
			data_flows = workflow_data.get('data_flows', [])
			await self._create_data_flow_visualizations(scene, data_flows)
			
			# Setup cameras and lighting
			await self._setup_scene_cameras(scene)
			await self._setup_scene_lighting(scene)
			
			return scene
		
		except Exception as e:
			logger.error(f"3D workflow mapping failed: {str(e)}")
			# Return minimal scene as fallback
			return Scene3D(
				name="Error_Scene",
				workflow_id=workflow_data.get('id', 'error')
			)
	
	async def _create_workflow_nodes(self, scene: Scene3D, nodes: List[Dict[str, Any]]) -> None:
		"""Create 3D representations of workflow nodes"""
		node_positions = self._calculate_node_positions(nodes)
		
		for i, node in enumerate(nodes):
			node_id = node.get('id', f'node_{i}')
			node_type = node.get('type', 'generic')
			
			# Determine geometry based on node type
			geometry = self._create_node_geometry(node_type)
			
			# Determine material based on node state
			material = self._create_node_material(node)
			
			# Position node in 3D space
			position = node_positions.get(node_id, Vector3D())
			
			scene_object = SceneObject3D(
				id=node_id,
				name=node.get('name', f'Node {i}'),
				object_type='workflow_node',
				transform=Transform3D(position=position),
				geometry=geometry,
				material=material,
				metadata={
					'node_data': node,
					'node_type': node_type,
					'node_state': node.get('state', 'idle')
				}
			)
			
			scene.objects[node_id] = scene_object
	
	async def _create_workflow_connections(self, scene: Scene3D, connections: List[Dict[str, Any]], nodes: List[Dict[str, Any]]) -> None:
		"""Create 3D representations of workflow connections"""
		node_positions = {node.get('id'): scene.objects[node.get('id')].transform.position 
						  for node in nodes if node.get('id') in scene.objects}
		
		for i, connection in enumerate(connections):
			connection_id = connection.get('id', f'connection_{i}')
			source_id = connection.get('source_id')
			target_id = connection.get('target_id')
			
			if source_id not in node_positions or target_id not in node_positions:
				continue
			
			# Create curved connection line
			geometry = self._create_connection_geometry(
				node_positions[source_id],
				node_positions[target_id]
			)
			
			# Connection material with flow animation
			material = self._create_connection_material(connection)
			
			# Position at midpoint
			midpoint = Vector3D(
				x=(node_positions[source_id].x + node_positions[target_id].x) / 2,
				y=(node_positions[source_id].y + node_positions[target_id].y) / 2 + self.connection_height,
				z=(node_positions[source_id].z + node_positions[target_id].z) / 2
			)
			
			scene_object = SceneObject3D(
				id=connection_id,
				name=f'Connection {i}',
				object_type='workflow_connection',
				transform=Transform3D(position=midpoint),
				geometry=geometry,
				material=material,
				metadata={
					'connection_data': connection,
					'source_id': source_id,
					'target_id': target_id
				}
			)
			
			scene.objects[connection_id] = scene_object
	
	async def _create_data_flow_visualizations(self, scene: Scene3D, data_flows: List[Dict[str, Any]]) -> None:
		"""Create 3D visualizations for data flows"""
		for i, data_flow in enumerate(data_flows):
			flow_id = data_flow.get('id', f'flow_{i}')
			
			# Create particle system for data flow
			geometry = self._create_particle_geometry(data_flow)
			material = self._create_particle_material(data_flow)
			
			scene_object = SceneObject3D(
				id=flow_id,
				name=f'Data Flow {i}',
				object_type='data_flow',
				transform=Transform3D(),
				geometry=geometry,
				material=material,
				metadata={
					'flow_data': data_flow,
					'particle_count': data_flow.get('volume', 100)
				}
			)
			
			scene.objects[flow_id] = scene_object
	
	def _calculate_node_positions(self, nodes: List[Dict[str, Any]]) -> Dict[str, Vector3D]:
		"""Calculate optimal 3D positions for workflow nodes"""
		positions = {}
		
		# Simple grid layout for now - can be enhanced with graph algorithms
		grid_size = math.ceil(math.sqrt(len(nodes)))
		
		for i, node in enumerate(nodes):
			node_id = node.get('id', f'node_{i}')
			
			# Calculate grid position
			row = i // grid_size
			col = i % grid_size
			
			# Add some variation based on node type/priority
			y_offset = node.get('priority', 0) * 0.5
			
			positions[node_id] = Vector3D(
				x=col * self.node_spacing - (grid_size * self.node_spacing) / 2,
				y=y_offset,
				z=row * self.node_spacing - (grid_size * self.node_spacing) / 2
			)
		
		return positions
	
	def _create_node_geometry(self, node_type: str) -> Geometry3D:
		"""Create geometry based on node type"""
		geometry_configs = {
			'start': {'geometry_type': 'sphere', 'parameters': {'radius': 1.0}},
			'end': {'geometry_type': 'sphere', 'parameters': {'radius': 1.0}},
			'process': {'geometry_type': 'box', 'parameters': {'width': 2.0, 'height': 1.0, 'depth': 1.0}},
			'decision': {'geometry_type': 'diamond', 'parameters': {'size': 1.5}},
			'parallel': {'geometry_type': 'cylinder', 'parameters': {'radius': 1.0, 'height': 2.0}},
			'data': {'geometry_type': 'octahedron', 'parameters': {'size': 1.2}},
			'service': {'geometry_type': 'hexagon', 'parameters': {'size': 1.3}},
			'generic': {'geometry_type': 'box', 'parameters': {'width': 1.5, 'height': 1.0, 'depth': 1.0}}
		}
		
		config = geometry_configs.get(node_type, geometry_configs['generic'])
		return Geometry3D(**config)
	
	def _create_node_material(self, node: Dict[str, Any]) -> Material3D:
		"""Create material based on node state and properties"""
		state = node.get('state', 'idle')
		node_type = node.get('type', 'generic')
		
		# State-based colors
		state_colors = {
			'idle': (0.7, 0.7, 0.7, 1.0),
			'running': (0.2, 0.8, 0.2, 1.0),
			'completed': (0.2, 0.2, 0.8, 1.0),
			'failed': (0.8, 0.2, 0.2, 1.0),
			'waiting': (0.8, 0.8, 0.2, 1.0),
			'paused': (0.8, 0.5, 0.2, 1.0)
		}
		
		# Type-based emission for special nodes
		type_emissions = {
			'start': (0.2, 0.8, 0.2),
			'end': (0.8, 0.2, 0.2),
			'service': (0.2, 0.2, 0.8)
		}
		
		return Material3D(
			name=f"{node_type}_{state}",
			color=state_colors.get(state, (0.5, 0.5, 0.5, 1.0)),
			metallic=0.3,
			roughness=0.4,
			emission=type_emissions.get(node_type, (0.0, 0.0, 0.0))
		)
	
	def _create_connection_geometry(self, start: Vector3D, end: Vector3D) -> Geometry3D:
		"""Create curved connection geometry between two points"""
		# Create bezier curve for smooth connection
		control_height = self.connection_height * 2
		midpoint = Vector3D(
			x=(start.x + end.x) / 2,
			y=max(start.y, end.y) + control_height,
			z=(start.z + end.z) / 2
		)
		
		return Geometry3D(
			geometry_type='bezier_curve',
			parameters={
				'start': [start.x, start.y, start.z],
				'control': [midpoint.x, midpoint.y, midpoint.z],
				'end': [end.x, end.y, end.z],
				'segments': 32,
				'radius': 0.1
			}
		)
	
	def _create_connection_material(self, connection: Dict[str, Any]) -> Material3D:
		"""Create animated material for connections"""
		state = connection.get('state', 'idle')
		
		# Animated flow colors
		flow_colors = {
			'idle': (0.4, 0.4, 0.4, 0.8),
			'active': (0.2, 0.8, 0.8, 1.0),
			'transmitting': (0.8, 0.8, 0.2, 1.0),
			'error': (0.8, 0.2, 0.2, 1.0)
		}
		
		return Material3D(
			name=f"connection_{state}",
			color=flow_colors.get(state, (0.4, 0.4, 0.4, 0.8)),
			metallic=0.1,
			roughness=0.2,
			emission=(0.2, 0.2, 0.2) if state == 'active' else (0.0, 0.0, 0.0),
			shader_program='flow_animation'
		)
	
	def _create_particle_geometry(self, data_flow: Dict[str, Any]) -> Geometry3D:
		"""Create particle system geometry for data flows"""
		particle_count = min(data_flow.get('volume', 100), 1000)  # Limit for performance
		
		return Geometry3D(
			geometry_type='particle_system',
			parameters={
				'particle_count': particle_count,
				'particle_size': 0.1,
				'spawn_rate': data_flow.get('rate', 10),
				'lifetime': 5.0,
				'velocity': data_flow.get('velocity', 1.0)
			}
		)
	
	def _create_particle_material(self, data_flow: Dict[str, Any]) -> Material3D:
		"""Create particle material for data flows"""
		data_type = data_flow.get('data_type', 'generic')
		
		# Data type colors
		type_colors = {
			'json': (0.2, 0.8, 0.2, 0.7),
			'xml': (0.8, 0.5, 0.2, 0.7),
			'binary': (0.8, 0.2, 0.8, 0.7),
			'text': (0.2, 0.2, 0.8, 0.7),
			'image': (0.8, 0.8, 0.2, 0.7),
			'generic': (0.5, 0.5, 0.5, 0.7)
		}
		
		return Material3D(
			name=f"particles_{data_type}",
			color=type_colors.get(data_type, (0.5, 0.5, 0.5, 0.7)),
			metallic=0.0,
			roughness=1.0,
			shader_program='particle_system'
		)
	
	async def _setup_scene_cameras(self, scene: Scene3D) -> None:
		"""Setup default cameras for the scene"""
		# Main perspective camera
		main_camera = Camera3D(
			name="Main Camera",
			camera_type="perspective",
			transform=Transform3D(
				position=Vector3D(x=10, y=10, z=10),
				rotation=Quaternion()  # Looking at origin
			),
			fov=60.0,
			target=Vector3D(x=0, y=0, z=0)
		)
		
		# Orthographic camera for technical views
		ortho_camera = Camera3D(
			name="Orthographic Camera",
			camera_type="orthographic",
			transform=Transform3D(
				position=Vector3D(x=0, y=20, z=0),
				rotation=Quaternion()  # Top-down view
			),
			target=Vector3D(x=0, y=0, z=0)
		)
		
		scene.cameras[main_camera.id] = main_camera
		scene.cameras[ortho_camera.id] = ortho_camera
		scene.active_camera_id = main_camera.id
	
	async def _setup_scene_lighting(self, scene: Scene3D) -> None:
		"""Setup lighting for the scene"""
		# Main directional light (sun)
		main_light = Light3D(
			name="Main Light",
			light_type="directional",
			transform=Transform3D(
				position=Vector3D(x=10, y=20, z=10),
				rotation=Quaternion()  # Pointing down and towards origin
			),
			color=(1.0, 1.0, 0.9),
			intensity=1.5,
			cast_shadows=True
		)
		
		# Ambient light for general illumination
		ambient_light = Light3D(
			name="Ambient Light",
			light_type="ambient",
			color=(0.3, 0.3, 0.4),
			intensity=0.3,
			cast_shadows=False
		)
		
		# Fill light to reduce harsh shadows
		fill_light = Light3D(
			name="Fill Light",
			light_type="directional",
			transform=Transform3D(
				position=Vector3D(x=-5, y=10, z=-5),
				rotation=Quaternion()
			),
			color=(0.8, 0.8, 1.0),
			intensity=0.5,
			cast_shadows=False
		)
		
		scene.lights[main_light.id] = main_light
		scene.lights[ambient_light.id] = ambient_light
		scene.lights[fill_light.id] = fill_light

class ARDebugInterface:
	"""Augmented reality debugging interface"""
	
	def __init__(self, config: VisualizationConfig):
		self.config = config
		self.active_markers: Dict[str, ARMarker] = {}
		self.debug_objects: Dict[str, SceneObject3D] = {}
	
	async def create_debug_overlay(self, workflow_id: str, debug_data: Dict[str, Any]) -> Dict[str, ARMarker]:
		"""Create AR overlay for workflow debugging"""
		markers = {}
		
		# Create markers for each debug point
		debug_points = debug_data.get('debug_points', [])
		
		for i, point in enumerate(debug_points):
			marker_id = f"debug_{workflow_id}_{i}"
			
			# Create QR code marker for each debug point
			marker = ARMarker(
				id=marker_id,
				marker_type="qr_code",
				pattern_data=f"DEBUG:{workflow_id}:{point.get('node_id', '')}",
				transform=Transform3D(
					position=Vector3D(
						x=point.get('x', 0),
						y=point.get('y', 0),
						z=point.get('z', 0)
					)
				),
				content_object_ids=await self._create_debug_content(point)
			)
			
			markers[marker_id] = marker
			self.active_markers[marker_id] = marker
		
		return markers
	
	async def _create_debug_content(self, debug_point: Dict[str, Any]) -> List[str]:
		"""Create 3D debug content for AR marker"""
		content_ids = []
		
		# Variable inspection panel
		if debug_point.get('variables'):
			panel_id = f"var_panel_{debug_point.get('node_id', 'unknown')}"
			
			panel_object = SceneObject3D(
				id=panel_id,
				name="Variable Panel",
				object_type="debug_panel",
				transform=Transform3D(
					position=Vector3D(x=0, y=1, z=0),
					scale=Vector3D(x=2, y=1.5, z=0.1)
				),
				geometry=Geometry3D(
					geometry_type="plane",
					parameters={"width": 1, "height": 1}
				),
				material=Material3D(
					name="debug_panel",
					color=(0.1, 0.1, 0.1, 0.9),
					emission=(0.2, 0.2, 0.8)
				),
				metadata={"variables": debug_point.get('variables', {})}
			)
			
			self.debug_objects[panel_id] = panel_object
			content_ids.append(panel_id)
		
		# Execution flow indicator
		if debug_point.get('execution_state'):
			flow_id = f"flow_indicator_{debug_point.get('node_id', 'unknown')}"
			
			flow_object = SceneObject3D(
				id=flow_id,
				name="Flow Indicator",
				object_type="flow_indicator",
				transform=Transform3D(
					position=Vector3D(x=0, y=2, z=0),
					scale=Vector3D(x=0.5, y=0.5, z=0.5)
				),
				geometry=Geometry3D(
					geometry_type="sphere",
					parameters={"radius": 1}
				),
				material=Material3D(
					name="flow_indicator",
					color=(0.8, 0.2, 0.2, 0.8),
					emission=(0.8, 0.4, 0.4),
					shader_program='pulsing_glow'
				),
				metadata={"execution_state": debug_point.get('execution_state')}
			)
			
			self.debug_objects[flow_id] = flow_object
			content_ids.append(flow_id)
		
		return content_ids
	
	async def update_debug_info(self, marker_id: str, debug_data: Dict[str, Any]) -> None:
		"""Update debug information for AR marker"""
		if marker_id not in self.active_markers:
			return
		
		marker = self.active_markers[marker_id]
		
		# Update content objects with new debug data
		for content_id in marker.content_object_ids:
			if content_id in self.debug_objects:
				debug_object = self.debug_objects[content_id]
				debug_object.metadata.update(debug_data)
				debug_object.updated_at = datetime.utcnow()
	
	async def remove_debug_marker(self, marker_id: str) -> None:
		"""Remove debug marker and associated content"""
		if marker_id not in self.active_markers:
			return
		
		marker = self.active_markers[marker_id]
		
		# Remove content objects
		for content_id in marker.content_object_ids:
			if content_id in self.debug_objects:
				del self.debug_objects[content_id]
		
		# Remove marker
		del self.active_markers[marker_id]

class SpatialManipulationEngine:
	"""Engine for spatial manipulation of workflow objects"""
	
	def __init__(self, config: VisualizationConfig):
		self.config = config
		self.active_interactions: Dict[str, SpatialInteraction] = {}
		self.manipulation_history: List[Dict[str, Any]] = []
	
	async def start_spatial_interaction(self, user_id: str, object_id: str, 
									   interaction_type: str, position: Vector3D, 
									   rotation: Quaternion) -> str:
		"""Start a spatial interaction with an object"""
		interaction_id = uuid7str()
		
		interaction = SpatialInteraction(
			id=interaction_id,
			user_id=user_id,
			interaction_type=interaction_type,
			target_object_id=object_id,
			start_position=position,
			current_position=position,
			start_rotation=rotation,
			current_rotation=rotation
		)
		
		self.active_interactions[interaction_id] = interaction
		
		return interaction_id
	
	async def update_spatial_interaction(self, interaction_id: str, 
										position: Vector3D, rotation: Quaternion,
										gesture_data: Optional[Dict[str, Any]] = None) -> bool:
		"""Update ongoing spatial interaction"""
		if interaction_id not in self.active_interactions:
			return False
		
		interaction = self.active_interactions[interaction_id]
		interaction.current_position = position
		interaction.current_rotation = rotation
		
		if gesture_data:
			interaction.gesture_data.update(gesture_data)
		
		# Apply transformation to target object
		await self._apply_spatial_transformation(interaction)
		
		return True
	
	async def end_spatial_interaction(self, interaction_id: str) -> None:
		"""End spatial interaction and record in history"""
		if interaction_id not in self.active_interactions:
			return
		
		interaction = self.active_interactions[interaction_id]
		interaction.is_active = False
		
		# Record in history for undo/redo
		self.manipulation_history.append({
			'interaction': interaction.dict(),
			'timestamp': datetime.utcnow(),
			'action': 'spatial_manipulation'
		})
		
		# Clean up
		del self.active_interactions[interaction_id]
	
	async def _apply_spatial_transformation(self, interaction: SpatialInteraction) -> None:
		"""Apply spatial transformation based on interaction"""
		try:
			# Calculate transformation delta
			position_delta = Vector3D(
				x=interaction.current_position.x - interaction.start_position.x,
				y=interaction.current_position.y - interaction.start_position.y,
				z=interaction.current_position.z - interaction.start_position.z
			)
			
			rotation_delta = interaction.current_rotation.multiply(
				Quaternion(
					w=interaction.start_rotation.w,
					x=-interaction.start_rotation.x,
					y=-interaction.start_rotation.y,
					z=-interaction.start_rotation.z
				)
			)
			
			# Apply transformation based on interaction type
			if interaction.interaction_type == "move":
				await self._apply_translation(interaction.target_object_id, position_delta)
			elif interaction.interaction_type == "rotate":
				await self._apply_rotation(interaction.target_object_id, rotation_delta)
			elif interaction.interaction_type == "scale":
				scale_factor = interaction.current_position.magnitude() / interaction.start_position.magnitude()
				await self._apply_scaling(interaction.target_object_id, scale_factor)
			
		except Exception as e:
			logger.error(f"Spatial transformation failed: {str(e)}")
	
	async def _apply_translation(self, object_id: str, delta: Vector3D) -> None:
		"""Apply translation to object"""
		# Implementation would update object position in scene
		logger.info(f"Translating object {object_id} by ({delta.x}, {delta.y}, {delta.z})")
	
	async def _apply_rotation(self, object_id: str, delta: Quaternion) -> None:
		"""Apply rotation to object"""
		# Implementation would update object rotation in scene
		logger.info(f"Rotating object {object_id}")
	
	async def _apply_scaling(self, object_id: str, factor: float) -> None:
		"""Apply scaling to object"""
		# Implementation would update object scale in scene
		logger.info(f"Scaling object {object_id} by factor {factor}")

class HolographicRenderer:
	"""Holographic display renderer"""
	
	def __init__(self, display_config: HolographicDisplay):
		self.display_config = display_config
		self.viewport_cameras: List[Camera3D] = []
		self.quilt_texture = None
	
	async def initialize_holographic_display(self) -> None:
		"""Initialize holographic display system"""
		try:
			# Setup viewport cameras for light field rendering
			await self._setup_viewport_cameras()
			
			# Initialize quilt rendering system
			await self._initialize_quilt_rendering()
			
			# Setup holographic projection parameters
			await self._setup_projection_parameters()
			
			logger.info(f"Holographic display initialized: {self.display_config.display_type}")
			
		except Exception as e:
			logger.error(f"Holographic display initialization failed: {str(e)}")
			raise APGException(f"Failed to initialize holographic display: {str(e)}")
	
	async def render_holographic_scene(self, scene: Scene3D) -> Dict[str, Any]:
		"""Render scene for holographic display"""
		try:
			rendered_views = []
			
			# Render from each viewport camera
			for i, camera in enumerate(self.viewport_cameras):
				view_data = await self._render_viewport(scene, camera, i)
				rendered_views.append(view_data)
			
			# Compose quilt texture
			quilt_data = await self._compose_quilt(rendered_views)
			
			# Apply holographic post-processing
			holographic_output = await self._apply_holographic_processing(quilt_data)
			
			return {
				'holographic_data': holographic_output,
				'viewport_count': len(rendered_views),
				'display_config': self.display_config.dict(),
				'render_time': time.time()
			}
			
		except Exception as e:
			logger.error(f"Holographic rendering failed: {str(e)}")
			return {'error': str(e)}
	
	async def _setup_viewport_cameras(self) -> None:
		"""Setup cameras for light field capture"""
		viewport_count = self.display_config.viewport_count
		
		# Calculate camera positions for light field
		for i in range(viewport_count):
			angle = (i / viewport_count) * 2 * math.pi
			radius = 2.0  # Distance from center
			
			camera_position = Vector3D(
				x=math.cos(angle) * radius,
				y=0,
				z=math.sin(angle) * radius
			)
			
			camera = Camera3D(
				id=f"viewport_camera_{i}",
				name=f"Viewport Camera {i}",
				transform=Transform3D(position=camera_position),
				fov=45.0,
				target=Vector3D(x=0, y=0, z=0)
			)
			
			self.viewport_cameras.append(camera)
	
	async def _initialize_quilt_rendering(self) -> None:
		"""Initialize quilt texture rendering system"""
		# Setup quilt parameters based on display type
		if self.display_config.display_type == "looking_glass":
			self.display_config.quilt_settings = {
				'columns': 8,
				'rows': 6,
				'total_views': 48,
				'aspect_ratio': 0.75
			}
		
		logger.info("Quilt rendering system initialized")
	
	async def _setup_projection_parameters(self) -> None:
		"""Setup holographic projection parameters"""
		# Display-specific calibration
		if self.display_config.display_type == "looking_glass":
			self.display_config.calibration_data = {
				'pitch': 47.56,
				'slope': -7.5,
				'center': 0.5,
				'dpi': 338,
				'invView': False
			}
		
		logger.info("Projection parameters configured")
	
	async def _render_viewport(self, scene: Scene3D, camera: Camera3D, viewport_index: int) -> Dict[str, Any]:
		"""Render scene from single viewport"""
		# Simulate rendering process
		render_data = {
			'viewport_index': viewport_index,
			'camera_position': [camera.transform.position.x, camera.transform.position.y, camera.transform.position.z],
			'rendered_objects': len(scene.objects),
			'render_resolution': [512, 512]  # Per-viewport resolution
		}
		
		return render_data
	
	async def _compose_quilt(self, rendered_views: List[Dict[str, Any]]) -> Dict[str, Any]:
		"""Compose individual views into quilt texture"""
		quilt_settings = self.display_config.quilt_settings
		
		quilt_data = {
			'width': self.display_config.resolution[0],
			'height': self.display_config.resolution[1],
			'views': len(rendered_views),
			'columns': quilt_settings.get('columns', 8),
			'rows': quilt_settings.get('rows', 6),
			'format': 'rgba'
		}
		
		return quilt_data
	
	async def _apply_holographic_processing(self, quilt_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Apply holographic-specific post-processing"""
		processed_data = quilt_data.copy()
		
		# Apply brightness and contrast adjustments
		processed_data['brightness'] = self.display_config.brightness
		processed_data['contrast'] = self.display_config.contrast
		
		# Apply depth range mapping
		processed_data['depth_range'] = self.display_config.depth_range
		
		# Apply calibration corrections
		processed_data['calibration'] = self.display_config.calibration_data
		
		return processed_data

class AdvancedVisualizationService(APGBaseService):
	"""Main advanced visualization service"""
	
	def __init__(self, config: VisualizationConfig, 
				 websocket_manager: WebSocketManager,
				 db_manager: DatabaseManager):
		super().__init__()
		self.config = config
		self.websocket_manager = websocket_manager
		self.db_manager = db_manager
		
		# Components
		self.spatial_mapper = WorkflowSpatialMapper(config)
		self.ar_interface = ARDebugInterface(config)
		self.spatial_engine = SpatialManipulationEngine(config)
		
		# Scene management
		self.active_scenes: Dict[str, Scene3D] = {}
		self.holographic_renderers: Dict[str, HolographicRenderer] = {}
		
		# Background tasks
		self.rendering_task: Optional[asyncio.Task] = None
		self.running = False
	
	async def start(self) -> None:
		"""Start visualization service"""
		if self.running:
			return
		
		self.running = True
		self.rendering_task = asyncio.create_task(self._rendering_loop())
		logger.info("Advanced visualization service started")
	
	async def stop(self) -> None:
		"""Stop visualization service"""
		self.running = False
		if self.rendering_task:
			self.rendering_task.cancel()
			try:
				await self.rendering_task
			except asyncio.CancelledError:
				pass
		logger.info("Advanced visualization service stopped")
	
	async def create_3d_visualization(self, workflow_id: str, workflow_data: Dict[str, Any], 
									  visualization_mode: VisualizationMode = VisualizationMode.ISOMETRIC_3D) -> Scene3D:
		"""Create 3D visualization of workflow"""
		try:
			# Map workflow to 3D scene
			scene = await self.spatial_mapper.map_workflow_to_3d(workflow_data)
			
			# Store scene
			self.active_scenes[workflow_id] = scene
			
			# Setup holographic rendering if enabled
			if visualization_mode == VisualizationMode.HOLOGRAPHIC_3D and self.config.holographic_enabled:
				await self._setup_holographic_rendering(workflow_id, scene)
			
			# Broadcast scene update
			await self._broadcast_scene_update(workflow_id, scene)
			
			return scene
		
		except Exception as e:
			logger.error(f"3D visualization creation failed: {str(e)}")
			raise APGException(f"Failed to create 3D visualization: {str(e)}")
	
	async def create_ar_debug_session(self, workflow_id: str, debug_data: Dict[str, Any]) -> Dict[str, ARMarker]:
		"""Create AR debugging session"""
		try:
			if not self.config.ar_enabled:
				raise APGException("AR debugging not enabled")
			
			# Create AR markers
			markers = await self.ar_interface.create_debug_overlay(workflow_id, debug_data)
			
			# Broadcast AR session info
			await self._broadcast_ar_session(workflow_id, markers)
			
			return markers
		
		except Exception as e:
			logger.error(f"AR debug session creation failed: {str(e)}")
			raise APGException(f"Failed to create AR debug session: {str(e)}")
	
	async def handle_spatial_interaction(self, user_id: str, interaction_data: Dict[str, Any]) -> str:
		"""Handle spatial manipulation interaction"""
		try:
			object_id = interaction_data.get('object_id')
			interaction_type = interaction_data.get('type', 'select')
			position = Vector3D(**interaction_data.get('position', {}))
			rotation = Quaternion(**interaction_data.get('rotation', {}))
			
			# Start spatial interaction
			interaction_id = await self.spatial_engine.start_spatial_interaction(
				user_id, object_id, interaction_type, position, rotation
			)
			
			# Broadcast interaction start
			await self._broadcast_spatial_interaction(user_id, interaction_id, 'started')
			
			return interaction_id
		
		except Exception as e:
			logger.error(f"Spatial interaction handling failed: {str(e)}")
			raise APGException(f"Failed to handle spatial interaction: {str(e)}")
	
	async def update_spatial_interaction(self, interaction_id: str, update_data: Dict[str, Any]) -> bool:
		"""Update ongoing spatial interaction"""
		try:
			position = Vector3D(**update_data.get('position', {}))
			rotation = Quaternion(**update_data.get('rotation', {}))
			gesture_data = update_data.get('gesture_data')
			
			success = await self.spatial_engine.update_spatial_interaction(
				interaction_id, position, rotation, gesture_data
			)
			
			if success:
				await self._broadcast_spatial_interaction('', interaction_id, 'updated')
			
			return success
		
		except Exception as e:
			logger.error(f"Spatial interaction update failed: {str(e)}")
			return False
	
	async def get_scene_data(self, workflow_id: str) -> Optional[Scene3D]:
		"""Get 3D scene data for workflow"""
		return self.active_scenes.get(workflow_id)
	
	async def get_holographic_output(self, workflow_id: str) -> Optional[Dict[str, Any]]:
		"""Get holographic rendering output"""
		if workflow_id not in self.holographic_renderers:
			return None
		
		renderer = self.holographic_renderers[workflow_id]
		scene = self.active_scenes.get(workflow_id)
		
		if not scene:
			return None
		
		return await renderer.render_holographic_scene(scene)
	
	async def _setup_holographic_rendering(self, workflow_id: str, scene: Scene3D) -> None:
		"""Setup holographic rendering for scene"""
		display_config = HolographicDisplay(
			display_type=self.config.rendering_engine,
			resolution=(2048, 2048),
			viewport_count=45
		)
		
		renderer = HolographicRenderer(display_config)
		await renderer.initialize_holographic_display()
		
		self.holographic_renderers[workflow_id] = renderer
	
	async def _rendering_loop(self) -> None:
		"""Background rendering loop"""
		while self.running:
			try:
				await asyncio.sleep(1.0 / self.config.frame_rate_target)
				
				# Update active scenes
				for workflow_id, scene in self.active_scenes.items():
					await self._update_scene_animations(scene)
					
					# Render holographic output if enabled
					if workflow_id in self.holographic_renderers:
						holographic_data = await self.get_holographic_output(workflow_id)
						if holographic_data:
							await self._broadcast_holographic_frame(workflow_id, holographic_data)
				
			except asyncio.CancelledError:
				break
			except Exception as e:
				logger.error(f"Rendering loop error: {str(e)}")
				await asyncio.sleep(1.0)  # Wait before retrying
	
	async def _update_scene_animations(self, scene: Scene3D) -> None:
		"""Update scene object animations"""
		current_time = time.time()
		
		for obj in scene.objects.values():
			if obj.animation_state:
				# Update object animations based on state
				if obj.object_type == 'workflow_connection' and obj.metadata.get('connection_data', {}).get('state') == 'active':
					# Animate data flow
					obj.animation_state['flow_progress'] = (current_time % 2.0) / 2.0
				
				elif obj.object_type == 'data_flow':
					# Update particle positions
					obj.animation_state['particle_time'] = current_time
	
	async def _broadcast_scene_update(self, workflow_id: str, scene: Scene3D) -> None:
		"""Broadcast scene update to connected clients"""
		try:
			message = {
				'type': 'scene_update',
				'workflow_id': workflow_id,
				'scene_data': {
					'id': scene.id,
					'name': scene.name,
					'object_count': len(scene.objects),
					'camera_count': len(scene.cameras),
					'light_count': len(scene.lights)
				}
			}
			
			await self.websocket_manager.broadcast_to_room(f"workflow_{workflow_id}", message)
		
		except Exception as e:
			logger.error(f"Scene update broadcast failed: {str(e)}")
	
	async def _broadcast_ar_session(self, workflow_id: str, markers: Dict[str, ARMarker]) -> None:
		"""Broadcast AR session information"""
		try:
			message = {
				'type': 'ar_session_created',
				'workflow_id': workflow_id,
				'marker_count': len(markers),
				'markers': {marker_id: marker.dict() for marker_id, marker in markers.items()}
			}
			
			await self.websocket_manager.broadcast_to_room(f"workflow_{workflow_id}", message)
		
		except Exception as e:
			logger.error(f"AR session broadcast failed: {str(e)}")
	
	async def _broadcast_spatial_interaction(self, user_id: str, interaction_id: str, event_type: str) -> None:
		"""Broadcast spatial interaction event"""
		try:
			message = {
				'type': 'spatial_interaction',
				'user_id': user_id,
				'interaction_id': interaction_id,
				'event': event_type,
				'timestamp': datetime.utcnow().isoformat()
			}
			
			await self.websocket_manager.broadcast_to_all(message)
		
		except Exception as e:
			logger.error(f"Spatial interaction broadcast failed: {str(e)}")
	
	async def _broadcast_holographic_frame(self, workflow_id: str, holographic_data: Dict[str, Any]) -> None:
		"""Broadcast holographic frame data"""
		try:
			message = {
				'type': 'holographic_frame',
				'workflow_id': workflow_id,
				'frame_data': holographic_data,
				'timestamp': time.time()
			}
			
			await self.websocket_manager.broadcast_to_room(f"holographic_{workflow_id}", message)
		
		except Exception as e:
			logger.error(f"Holographic frame broadcast failed: {str(e)}")

# Factory function for creating advanced visualization service
async def create_advanced_visualization_service(
	config: Optional[VisualizationConfig] = None,
	websocket_manager: Optional[WebSocketManager] = None,
	db_manager: Optional[DatabaseManager] = None
) -> AdvancedVisualizationService:
	"""Create and configure advanced visualization service"""
	
	if config is None:
		config = VisualizationConfig()
	
	# Initialize required services if not provided
	if websocket_manager is None:
		from apg.core.websocket import create_websocket_manager
		websocket_manager = await create_websocket_manager()
	
	if db_manager is None:
		from apg.core.database import create_database_manager
		db_manager = await create_database_manager()
	
	return AdvancedVisualizationService(
		config=config,
		websocket_manager=websocket_manager,
		db_manager=db_manager
	)