"""
3D Visualization Models

Database models for advanced 3D WebGL visualization engine including
scenes, objects, materials, animations, and rendering configurations.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from sqlalchemy import Column, String, Text, Integer, Float, Boolean, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship
from uuid_extensions import uuid7str
import json

from ..auth_rbac.models import BaseMixin, AuditMixin, Model


class VIS3DScene(Model, AuditMixin, BaseMixin):
	"""
	3D scene configuration and management.
	
	Represents complete 3D scenes with objects, lighting, cameras,
	and rendering settings for digital twin visualization.
	"""
	__tablename__ = 'vis3d_scene'
	
	# Identity
	scene_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Scene Information
	scene_name = Column(String(200), nullable=False, index=True)
	description = Column(Text, nullable=True)
	scene_type = Column(String(50), nullable=False, index=True)  # industrial, architectural, molecular, abstract
	category = Column(String(100), nullable=True)
	
	# Scene Configuration
	scene_data = Column(JSON, default=dict)  # Complete scene definition
	background_color = Column(JSON, default=[0.2, 0.2, 0.2, 1.0])  # RGBA
	environment_map = Column(String(500), nullable=True)  # Environment texture URL
	fog_settings = Column(JSON, default=dict)  # Fog configuration
	
	# Rendering Settings
	render_mode = Column(String(50), default='solid')  # wireframe, solid, textured, transparent
	quality_level = Column(String(20), default='medium')  # low, medium, high, ultra
	anti_aliasing = Column(Boolean, default=True)
	shadows_enabled = Column(Boolean, default=True)
	post_processing = Column(JSON, default=list)  # Post-processing effects
	
	# Performance Settings
	max_objects = Column(Integer, default=1000)
	max_vertices = Column(Integer, default=100000)
	level_of_detail_enabled = Column(Boolean, default=True)
	culling_enabled = Column(Boolean, default=True)
	instancing_enabled = Column(Boolean, default=True)
	
	# Interaction Settings
	controls_enabled = Column(Boolean, default=True)
	zoom_enabled = Column(Boolean, default=True)
	pan_enabled = Column(Boolean, default=True)
	rotate_enabled = Column(Boolean, default=True)
	selection_enabled = Column(Boolean, default=True)
	
	# Animation Settings
	animations_enabled = Column(Boolean, default=True)
	auto_play = Column(Boolean, default=False)
	loop_animations = Column(Boolean, default=True)
	animation_speed = Column(Float, default=1.0)
	
	# Data Integration
	data_sources = Column(JSON, default=list)  # Connected data sources
	real_time_updates = Column(Boolean, default=False)
	update_interval_ms = Column(Integer, default=1000)
	data_filters = Column(JSON, default=dict)
	
	# Access Control
	visibility = Column(String(20), default='private')  # private, shared, public
	shared_with = Column(JSON, default=list)
	created_by = Column(String(36), nullable=False, index=True)
	
	# Performance Metrics
	render_time_ms = Column(Float, nullable=True)
	frame_rate = Column(Float, nullable=True)
	polygon_count = Column(Integer, default=0)
	texture_memory_mb = Column(Float, default=0.0)
	
	# Usage Statistics
	view_count = Column(Integer, default=0)
	interaction_count = Column(Integer, default=0)
	last_viewed = Column(DateTime, nullable=True)
	average_session_duration = Column(Float, nullable=True)
	
	# Version Control
	version = Column(String(20), default='1.0.0')
	parent_scene_id = Column(String(36), nullable=True)
	is_template = Column(Boolean, default=False)
	
	# Relationships
	objects = relationship("VIS3DObject", back_populates="scene")
	cameras = relationship("VIS3DCamera", back_populates="scene")
	lights = relationship("VIS3DLight", back_populates="scene")
	materials = relationship("VIS3DMaterial", back_populates="scene")
	
	def __repr__(self):
		return f"<VIS3DScene {self.scene_name}>"
	
	def calculate_complexity_score(self) -> float:
		"""Calculate scene complexity score"""
		complexity_factors = {
			'object_count': len(self.objects) / 100.0,  # Normalize to 100 objects
			'polygon_density': self.polygon_count / 10000.0,  # Normalize to 10k polygons
			'texture_usage': self.texture_memory_mb / 100.0,  # Normalize to 100MB
			'lighting_complexity': len(self.lights) / 10.0,  # Normalize to 10 lights
			'animation_complexity': len([obj for obj in self.objects if obj.animations_enabled]) / 50.0
		}
		
		return min(1.0, sum(complexity_factors.values()) / len(complexity_factors))
	
	def estimate_render_performance(self) -> Dict[str, Any]:
		"""Estimate rendering performance"""
		complexity = self.calculate_complexity_score()
		
		# Estimate frame rate based on complexity
		if complexity < 0.3:
			estimated_fps = 60
			performance_tier = 'excellent'
		elif complexity < 0.6:
			estimated_fps = 30
			performance_tier = 'good'
		elif complexity < 0.8:
			estimated_fps = 15
			performance_tier = 'fair'
		else:
			estimated_fps = 10
			performance_tier = 'poor'
		
		return {
			'complexity_score': complexity,
			'estimated_fps': estimated_fps,
			'performance_tier': performance_tier,
			'optimization_suggestions': self._get_optimization_suggestions(complexity)
		}
	
	def _get_optimization_suggestions(self, complexity: float) -> List[str]:
		"""Provide optimization suggestions"""
		suggestions = []
		
		if complexity > 0.7:
			suggestions.append("Enable level-of-detail (LOD) for distant objects")
			suggestions.append("Use texture atlasing to reduce draw calls")
			suggestions.append("Enable frustum culling")
		
		if self.polygon_count > 50000:
			suggestions.append("Consider mesh decimation for high-poly objects")
		
		if len(self.lights) > 8:
			suggestions.append("Reduce number of dynamic lights")
		
		if not self.culling_enabled:
			suggestions.append("Enable backface culling")
		
		return suggestions
	
	def update_usage_metrics(self, session_duration: float):
		"""Update scene usage metrics"""
		self.view_count += 1
		self.last_viewed = datetime.utcnow()
		
		if self.average_session_duration:
			# Running average
			self.average_session_duration = (
				(self.average_session_duration * (self.view_count - 1) + session_duration) / 
				self.view_count
			)
		else:
			self.average_session_duration = session_duration


class VIS3DObject(Model, AuditMixin, BaseMixin):
	"""
	3D scene object with geometry, materials, and transform data.
	
	Represents individual 3D objects within scenes including meshes,
	primitives, point clouds, and procedural geometry.
	"""
	__tablename__ = 'vis3d_object'
	
	# Identity
	object_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	scene_id = Column(String(36), ForeignKey('vis3d_scene.scene_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Object Information
	object_name = Column(String(200), nullable=False, index=True)
	description = Column(Text, nullable=True)
	object_type = Column(String(50), nullable=False, index=True)  # mesh, primitive, point_cloud, cad, voxel
	category = Column(String(100), nullable=True)
	
	# Geometry Data
	geometry_type = Column(String(50), nullable=False)  # mesh, box, sphere, cylinder, custom
	vertices = Column(JSON, default=list)  # 3D vertex positions
	indices = Column(JSON, default=list)  # Face indices
	normals = Column(JSON, default=list)  # Vertex normals
	uvs = Column(JSON, default=list)  # Texture coordinates
	vertex_colors = Column(JSON, default=list)  # Per-vertex colors
	
	# Transform Properties
	position = Column(JSON, default=[0.0, 0.0, 0.0])  # XYZ position
	rotation = Column(JSON, default=[0.0, 0.0, 0.0])  # Euler angles
	scale = Column(JSON, default=[1.0, 1.0, 1.0])  # XYZ scale
	transform_matrix = Column(JSON, default=list)  # 4x4 transform matrix
	
	# Material Properties
	material_id = Column(String(36), ForeignKey('vis3d_material.material_id'), nullable=True, index=True)
	base_color = Column(JSON, default=[1.0, 1.0, 1.0, 1.0])  # RGBA
	metallic_factor = Column(Float, default=0.0)
	roughness_factor = Column(Float, default=0.5)
	emissive_factor = Column(JSON, default=[0.0, 0.0, 0.0])  # RGB
	
	# Visibility and Interaction
	visible = Column(Boolean, default=True)
	selectable = Column(Boolean, default=True)
	pickable = Column(Boolean, default=True)
	cast_shadows = Column(Boolean, default=True)
	receive_shadows = Column(Boolean, default=True)
	
	# Animation Properties
	animations_enabled = Column(Boolean, default=False)
	animation_data = Column(JSON, default=list)  # Animation keyframes
	current_animation = Column(String(100), nullable=True)
	animation_speed = Column(Float, default=1.0)
	animation_loop = Column(Boolean, default=True)
	
	# Data Binding
	data_bindings = Column(JSON, default=dict)  # Bind properties to data sources
	data_source_id = Column(String(36), nullable=True)
	data_mapping = Column(JSON, default=dict)  # Map data fields to object properties
	
	# Performance Properties
	level_of_detail = Column(JSON, default=dict)  # LOD configurations
	bounding_box = Column(JSON, default=dict)  # Object bounding box
	polygon_count = Column(Integer, default=0)
	vertex_count = Column(Integer, default=0)
	
	# Metadata
	custom_properties = Column(JSON, default=dict)  # Custom object properties
	tags = Column(JSON, default=list)  # Object tags
	parent_object_id = Column(String(36), nullable=True)  # Parent for hierarchies
	
	# Relationships
	scene = relationship("VIS3DScene", back_populates="objects")
	material = relationship("VIS3DMaterial", back_populates="objects")
	
	def __repr__(self):
		return f"<VIS3DObject {self.object_name}>"
	
	def calculate_bounding_box(self) -> Dict[str, List[float]]:
		"""Calculate object bounding box from vertices"""
		if not self.vertices:
			return {'min': [0, 0, 0], 'max': [0, 0, 0]}
		
		vertices = self.vertices
		min_coords = [min(v[i] for v in vertices) for i in range(3)]
		max_coords = [max(v[i] for v in vertices) for i in range(3)]
		
		self.bounding_box = {'min': min_coords, 'max': max_coords}
		return self.bounding_box
	
	def update_geometry_stats(self):
		"""Update geometry statistics"""
		if self.vertices:
			self.vertex_count = len(self.vertices)
		
		if self.indices:
			# Assuming triangular faces
			self.polygon_count = len(self.indices) // 3
		elif self.vertices:
			# Assume triangles if no indices
			self.polygon_count = len(self.vertices) // 3
	
	def get_transform_matrix(self) -> List[List[float]]:
		"""Get 4x4 transformation matrix"""
		if self.transform_matrix:
			return self.transform_matrix
		
		# Generate from position, rotation, scale
		# Simplified implementation - would use proper matrix math
		return [
			[self.scale[0], 0, 0, self.position[0]],
			[0, self.scale[1], 0, self.position[1]],
			[0, 0, self.scale[2], self.position[2]],
			[0, 0, 0, 1]
		]
	
	def animate_property(self, property_name: str, target_value: Any, duration_ms: float):
		"""Add animation for object property"""
		animation = {
			'property': property_name,
			'target_value': target_value,
			'duration_ms': duration_ms,
			'created_at': datetime.utcnow().isoformat()
		}
		
		if not self.animation_data:
			self.animation_data = []
		
		self.animation_data.append(animation)
		self.animations_enabled = True


class VIS3DMaterial(Model, AuditMixin, BaseMixin):
	"""
	3D material definition with PBR properties.
	
	Represents physically-based rendering materials with textures,
	lighting properties, and shader configurations.
	"""
	__tablename__ = 'vis3d_material'
	
	# Identity
	material_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	scene_id = Column(String(36), ForeignKey('vis3d_scene.scene_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Material Information
	material_name = Column(String(200), nullable=False, index=True)
	description = Column(Text, nullable=True)
	material_type = Column(String(50), default='pbr')  # pbr, phong, lambert, unlit
	category = Column(String(100), nullable=True)
	
	# Base Properties
	base_color = Column(JSON, default=[1.0, 1.0, 1.0, 1.0])  # RGBA
	metallic_factor = Column(Float, default=0.0)
	roughness_factor = Column(Float, default=0.5)
	normal_scale = Column(Float, default=1.0)
	occlusion_strength = Column(Float, default=1.0)
	emissive_factor = Column(JSON, default=[0.0, 0.0, 0.0])  # RGB
	
	# Texture Maps
	base_color_texture = Column(String(500), nullable=True)
	metallic_roughness_texture = Column(String(500), nullable=True)
	normal_texture = Column(String(500), nullable=True)
	occlusion_texture = Column(String(500), nullable=True)
	emissive_texture = Column(String(500), nullable=True)
	environment_map = Column(String(500), nullable=True)
	
	# Transparency
	alpha_mode = Column(String(20), default='opaque')  # opaque, mask, blend
	alpha_cutoff = Column(Float, default=0.5)
	transparency = Column(Float, default=1.0)
	
	# Rendering Properties
	double_sided = Column(Boolean, default=False)
	unlit = Column(Boolean, default=False)
	cast_shadows = Column(Boolean, default=True)
	receive_shadows = Column(Boolean, default=True)
	
	# Animation Properties
	animated_properties = Column(JSON, default=dict)  # Properties that can animate
	animation_speed = Column(Float, default=1.0)
	
	# Shader Configuration
	shader_type = Column(String(50), default='standard')  # standard, custom, toon, etc.
	shader_uniforms = Column(JSON, default=dict)  # Custom shader uniforms
	vertex_shader = Column(Text, nullable=True)  # Custom vertex shader
	fragment_shader = Column(Text, nullable=True)  # Custom fragment shader
	
	# Texture Settings
	texture_repeat = Column(JSON, default=[1.0, 1.0])  # UV repeat
	texture_offset = Column(JSON, default=[0.0, 0.0])  # UV offset
	texture_rotation = Column(Float, default=0.0)  # UV rotation
	
	# Performance Settings
	texture_resolution = Column(String(20), default='medium')  # low, medium, high
	mipmaps_enabled = Column(Boolean, default=True)
	anisotropic_filtering = Column(Integer, default=4)
	
	# Usage Statistics
	usage_count = Column(Integer, default=0)
	last_used = Column(DateTime, nullable=True)
	
	# Access Control
	visibility = Column(String(20), default='private')  # private, shared, public
	created_by = Column(String(36), nullable=False, index=True)
	
	# Relationships
	scene = relationship("VIS3DScene", back_populates="materials")
	objects = relationship("VIS3DObject", back_populates="material")
	
	def __repr__(self):
		return f"<VIS3DMaterial {self.material_name}>"
	
	def get_pbr_properties(self) -> Dict[str, Any]:
		"""Get PBR material properties"""
		return {
			'base_color': self.base_color,
			'metallic': self.metallic_factor,
			'roughness': self.roughness_factor,
			'normal_scale': self.normal_scale,
			'occlusion_strength': self.occlusion_strength,
			'emissive': self.emissive_factor,
			'alpha_mode': self.alpha_mode,
			'alpha_cutoff': self.alpha_cutoff,
			'double_sided': self.double_sided
		}
	
	def get_texture_configuration(self) -> Dict[str, Any]:
		"""Get texture configuration"""
		return {
			'base_color_texture': self.base_color_texture,
			'metallic_roughness_texture': self.metallic_roughness_texture,
			'normal_texture': self.normal_texture,
			'occlusion_texture': self.occlusion_texture,
			'emissive_texture': self.emissive_texture,
			'environment_map': self.environment_map,
			'repeat': self.texture_repeat,
			'offset': self.texture_offset,
			'rotation': self.texture_rotation
		}
	
	def estimate_memory_usage(self) -> float:
		"""Estimate texture memory usage in MB"""
		# Simplified estimation based on texture count and assumed resolution
		texture_count = sum(1 for texture in [
			self.base_color_texture, self.metallic_roughness_texture,
			self.normal_texture, self.occlusion_texture, self.emissive_texture
		] if texture)
		
		resolution_multiplier = {
			'low': 0.25,
			'medium': 1.0,
			'high': 4.0
		}.get(self.texture_resolution, 1.0)
		
		# Assume 2MB per texture at medium resolution
		return texture_count * 2.0 * resolution_multiplier
	
	def record_usage(self):
		"""Record material usage"""
		self.usage_count += 1
		self.last_used = datetime.utcnow()


class VIS3DCamera(Model, AuditMixin, BaseMixin):
	"""
	3D camera configuration and view settings.
	
	Represents camera objects with position, orientation,
	projection settings, and animation capabilities.
	"""
	__tablename__ = 'vis3d_camera'
	
	# Identity
	camera_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	scene_id = Column(String(36), ForeignKey('vis3d_scene.scene_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Camera Information
	camera_name = Column(String(200), nullable=False, index=True)
	description = Column(Text, nullable=True)
	camera_type = Column(String(50), default='perspective')  # perspective, orthographic
	
	# Position and Orientation
	position = Column(JSON, default=[0.0, 0.0, 5.0])  # XYZ position
	target = Column(JSON, default=[0.0, 0.0, 0.0])  # Look-at target
	up_vector = Column(JSON, default=[0.0, 1.0, 0.0])  # Up direction
	
	# Projection Settings (Perspective)
	field_of_view = Column(Float, default=45.0)  # FOV in degrees
	aspect_ratio = Column(Float, default=1.777)  # 16:9 aspect ratio
	near_plane = Column(Float, default=0.1)
	far_plane = Column(Float, default=1000.0)
	
	# Projection Settings (Orthographic)
	ortho_left = Column(Float, default=-5.0)
	ortho_right = Column(Float, default=5.0)
	ortho_top = Column(Float, default=5.0)
	ortho_bottom = Column(Float, default=-5.0)
	
	# Camera Controls
	controls_enabled = Column(Boolean, default=True)
	zoom_enabled = Column(Boolean, default=True)
	pan_enabled = Column(Boolean, default=True)
	rotate_enabled = Column(Boolean, default=True)
	auto_rotate = Column(Boolean, default=False)
	auto_rotate_speed = Column(Float, default=1.0)
	
	# Movement Constraints
	min_distance = Column(Float, default=1.0)
	max_distance = Column(Float, default=100.0)
	min_polar_angle = Column(Float, default=0.0)
	max_polar_angle = Column(Float, default=3.14159)  # π radians
	min_azimuth_angle = Column(Float, nullable=True)
	max_azimuth_angle = Column(Float, nullable=True)
	
	# Animation
	animation_enabled = Column(Boolean, default=False)
	animation_path = Column(JSON, default=list)  # Keyframe positions
	animation_duration = Column(Float, default=5.0)  # Seconds
	animation_loop = Column(Boolean, default=False)
	
	# Rendering Settings
	is_active = Column(Boolean, default=True)
	render_order = Column(Integer, default=0)
	
	# Relationships
	scene = relationship("VIS3DScene", back_populates="cameras")
	
	def __repr__(self):
		return f"<VIS3DCamera {self.camera_name}>"
	
	def get_view_matrix(self) -> List[List[float]]:
		"""Get camera view matrix"""
		# Simplified view matrix calculation
		# In real implementation, would use proper 3D math library
		return [
			[1, 0, 0, -self.position[0]],
			[0, 1, 0, -self.position[1]],
			[0, 0, 1, -self.position[2]],
			[0, 0, 0, 1]
		]
	
	def get_projection_matrix(self) -> List[List[float]]:
		"""Get camera projection matrix"""
		if self.camera_type == 'perspective':
			# Simplified perspective projection
			f = 1.0 / (self.field_of_view * 0.5)
			return [
				[f / self.aspect_ratio, 0, 0, 0],
				[0, f, 0, 0],
				[0, 0, (self.far_plane + self.near_plane) / (self.near_plane - self.far_plane), -1],
				[0, 0, (2 * self.far_plane * self.near_plane) / (self.near_plane - self.far_plane), 0]
			]
		else:
			# Orthographic projection
			return [
				[2 / (self.ortho_right - self.ortho_left), 0, 0, -(self.ortho_right + self.ortho_left) / (self.ortho_right - self.ortho_left)],
				[0, 2 / (self.ortho_top - self.ortho_bottom), 0, -(self.ortho_top + self.ortho_bottom) / (self.ortho_top - self.ortho_bottom)],
				[0, 0, -2 / (self.far_plane - self.near_plane), -(self.far_plane + self.near_plane) / (self.far_plane - self.near_plane)],
				[0, 0, 0, 1]
			]
	
	def calculate_distance_to_target(self) -> float:
		"""Calculate distance from camera to target"""
		dx = self.position[0] - self.target[0]
		dy = self.position[1] - self.target[1]
		dz = self.position[2] - self.target[2]
		return (dx*dx + dy*dy + dz*dz) ** 0.5


class VIS3DLight(Model, AuditMixin, BaseMixin):
	"""
	3D light source configuration.
	
	Represents various types of lights including directional,
	point, spot, and area lights with shadow settings.
	"""
	__tablename__ = 'vis3d_light'
	
	# Identity
	light_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	scene_id = Column(String(36), ForeignKey('vis3d_scene.scene_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Light Information
	light_name = Column(String(200), nullable=False, index=True)
	description = Column(Text, nullable=True)
	light_type = Column(String(50), nullable=False)  # directional, point, spot, area, ambient
	
	# Position and Orientation
	position = Column(JSON, default=[0.0, 10.0, 0.0])  # XYZ position
	direction = Column(JSON, default=[0.0, -1.0, 0.0])  # Light direction
	target = Column(JSON, default=[0.0, 0.0, 0.0])  # Target position for spot lights
	
	# Light Properties
	color = Column(JSON, default=[1.0, 1.0, 1.0])  # RGB color
	intensity = Column(Float, default=1.0)
	power = Column(Float, nullable=True)  # Light power in lumens
	temperature = Column(Float, nullable=True)  # Color temperature in Kelvin
	
	# Attenuation (for point and spot lights)
	range = Column(Float, default=10.0)  # Light range
	decay = Column(Float, default=2.0)  # Decay factor
	
	# Spot Light Properties
	inner_cone_angle = Column(Float, default=0.0)  # Inner cone angle in radians
	outer_cone_angle = Column(Float, default=0.785)  # Outer cone angle (45 degrees)
	
	# Area Light Properties
	width = Column(Float, default=1.0)  # Area light width
	height = Column(Float, default=1.0)  # Area light height
	
	# Shadow Settings
	cast_shadows = Column(Boolean, default=True)
	shadow_map_size = Column(Integer, default=1024)  # Shadow map resolution
	shadow_bias = Column(Float, default=0.0001)
	shadow_normal_bias = Column(Float, default=0.0)
	shadow_radius = Column(Float, default=1.0)
	
	# Shadow Camera Settings (for directional and spot lights)
	shadow_camera_near = Column(Float, default=0.1)
	shadow_camera_far = Column(Float, default=100.0)
	shadow_camera_fov = Column(Float, default=50.0)
	shadow_camera_left = Column(Float, default=-5.0)
	shadow_camera_right = Column(Float, default=5.0)
	shadow_camera_top = Column(Float, default=5.0)
	shadow_camera_bottom = Column(Float, default=-5.0)
	
	# Rendering Properties
	visible = Column(Boolean, default=True)
	enabled = Column(Boolean, default=True)
	helper_visible = Column(Boolean, default=False)  # Show light helper
	
	# Animation
	animated = Column(Boolean, default=False)
	animation_data = Column(JSON, default=list)  # Animation keyframes
	
	# Relationships
	scene = relationship("VIS3DScene", back_populates="lights")
	
	def __repr__(self):
		return f"<VIS3DLight {self.light_name}>"
	
	def get_light_configuration(self) -> Dict[str, Any]:
		"""Get complete light configuration"""
		config = {
			'type': self.light_type,
			'position': self.position,
			'color': self.color,
			'intensity': self.intensity,
			'cast_shadows': self.cast_shadows,
			'visible': self.visible,
			'enabled': self.enabled
		}
		
		if self.light_type in ['directional', 'spot']:
			config['direction'] = self.direction
			config['target'] = self.target
		
		if self.light_type in ['point', 'spot']:
			config['range'] = self.range
			config['decay'] = self.decay
		
		if self.light_type == 'spot':
			config['inner_cone_angle'] = self.inner_cone_angle
			config['outer_cone_angle'] = self.outer_cone_angle
		
		if self.light_type == 'area':
			config['width'] = self.width
			config['height'] = self.height
		
		if self.cast_shadows:
			config['shadow_settings'] = {
				'map_size': self.shadow_map_size,
				'bias': self.shadow_bias,
				'normal_bias': self.shadow_normal_bias,
				'radius': self.shadow_radius,
				'camera_near': self.shadow_camera_near,
				'camera_far': self.shadow_camera_far
			}
		
		return config
	
	def calculate_light_influence(self, point: List[float]) -> float:
		"""Calculate light influence at a given point"""
		if self.light_type == 'ambient':
			return self.intensity
		
		# Calculate distance from light to point
		dx = point[0] - self.position[0]
		dy = point[1] - self.position[1]
		dz = point[2] - self.position[2]
		distance = (dx*dx + dy*dy + dz*dz) ** 0.5
		
		if self.light_type == 'directional':
			return self.intensity  # Directional lights have constant intensity
		
		if self.light_type in ['point', 'spot']:
			if distance > self.range:
				return 0.0
			
			# Inverse square law with decay factor
			attenuation = 1.0 / (1.0 + self.decay * distance * distance)
			return self.intensity * attenuation
		
		return 0.0


class VIS3DRenderSession(Model, AuditMixin, BaseMixin):
	"""
	3D rendering session tracking and analytics.
	
	Tracks user sessions, performance metrics, and interaction
	data for 3D visualization analytics.
	"""
	__tablename__ = 'vis3d_render_session'
	
	# Identity
	session_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	scene_id = Column(String(36), ForeignKey('vis3d_scene.scene_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Session Information
	user_id = Column(String(36), nullable=True, index=True)
	session_name = Column(String(200), nullable=True)
	device_type = Column(String(50), nullable=True)  # desktop, mobile, tablet, vr
	browser_type = Column(String(100), nullable=True)
	user_agent = Column(Text, nullable=True)
	
	# Session Timing
	started_at = Column(DateTime, default=datetime.utcnow, index=True)
	ended_at = Column(DateTime, nullable=True)
	duration_seconds = Column(Float, nullable=True)
	
	# Performance Metrics
	average_fps = Column(Float, nullable=True)
	min_fps = Column(Float, nullable=True)
	max_fps = Column(Float, nullable=True)
	average_render_time_ms = Column(Float, nullable=True)
	total_frames_rendered = Column(Integer, default=0)
	
	# Quality Metrics
	render_quality = Column(String(20), nullable=True)  # low, medium, high, ultra
	resolution = Column(String(20), nullable=True)  # e.g., "1920x1080"
	anti_aliasing_used = Column(Boolean, default=False)
	shadows_enabled = Column(Boolean, default=False)
	
	# Resource Usage
	memory_usage_mb = Column(Float, nullable=True)
	gpu_memory_usage_mb = Column(Float, nullable=True)
	cpu_usage_percentage = Column(Float, nullable=True)
	gpu_usage_percentage = Column(Float, nullable=True)
	
	# Interaction Metrics
	total_interactions = Column(Integer, default=0)
	camera_movements = Column(Integer, default=0)
	object_selections = Column(Integer, default=0)
	zoom_actions = Column(Integer, default=0)
	pan_actions = Column(Integer, default=0)
	rotation_actions = Column(Integer, default=0)
	
	# Error Tracking
	webgl_errors = Column(Integer, default=0)
	shader_compilation_errors = Column(Integer, default=0)
	texture_loading_errors = Column(Integer, default=0)
	performance_warnings = Column(Integer, default=0)
	
	# Feature Usage
	features_used = Column(JSON, default=list)  # List of features used during session
	effects_applied = Column(JSON, default=list)  # Post-processing effects used
	
	# Quality of Experience
	user_satisfaction_score = Column(Float, nullable=True)  # 1-5 scale
	perceived_performance = Column(String(20), nullable=True)  # poor, fair, good, excellent
	
	# Network Metrics
	total_bytes_downloaded = Column(Float, nullable=True)
	texture_download_time_ms = Column(Float, nullable=True)
	model_download_time_ms = Column(Float, nullable=True)
	
	def __repr__(self):
		return f"<VIS3DRenderSession {self.session_id}>"
	
	def calculate_duration(self) -> Optional[float]:
		"""Calculate session duration in seconds"""
		if self.started_at and self.ended_at:
			duration = self.ended_at - self.started_at
			self.duration_seconds = duration.total_seconds()
			return self.duration_seconds
		return None
	
	def calculate_performance_score(self) -> float:
		"""Calculate overall performance score (0-1)"""
		scores = []
		
		# FPS score
		if self.average_fps:
			fps_score = min(1.0, self.average_fps / 60.0)  # Normalize to 60 FPS
			scores.append(fps_score)
		
		# Render time score
		if self.average_render_time_ms:
			# Lower render time is better, normalize to 16.67ms (60 FPS)
			render_score = min(1.0, 16.67 / self.average_render_time_ms)
			scores.append(render_score)
		
		# Error score (fewer errors is better)
		total_errors = (self.webgl_errors + self.shader_compilation_errors + 
						self.texture_loading_errors + self.performance_warnings)
		error_score = max(0.0, 1.0 - (total_errors / 10.0))  # Normalize to 10 errors
		scores.append(error_score)
		
		return sum(scores) / len(scores) if scores else 0.0
	
	def get_interaction_summary(self) -> Dict[str, Any]:
		"""Get interaction summary"""
		return {
			'total_interactions': self.total_interactions,
			'interaction_breakdown': {
				'camera_movements': self.camera_movements,
				'object_selections': self.object_selections,
				'zoom_actions': self.zoom_actions,
				'pan_actions': self.pan_actions,
				'rotation_actions': self.rotation_actions
			},
			'interactions_per_minute': (
				self.total_interactions / (self.duration_seconds / 60) 
				if self.duration_seconds and self.duration_seconds > 0 else 0
			)
		}
	
	def end_session(self):
		"""End the rendering session"""
		self.ended_at = datetime.utcnow()
		self.calculate_duration()