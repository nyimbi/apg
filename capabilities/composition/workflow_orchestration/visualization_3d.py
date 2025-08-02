"""
Advanced 3D Visualization Engine

Provides comprehensive 3D visualization capabilities:
- 3D workflow graph rendering with physics simulation
- Interactive 3D canvas with spatial navigation
- Node clustering and hierarchical layout
- Real-time animation and transitions
- Performance optimized rendering
- VR/AR-ready output

© 2025 Datacraft
Author: Nyimbi Odero
"""

import asyncio
import json
import math
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from pydantic import BaseModel, Field, ConfigDict
from uuid_extensions import uuid7str

from .database import get_async_db_session


class RenderingEngine(str, Enum):
	"""3D rendering engine types"""
	THREE_JS = "three_js"
	BABYLON_JS = "babylon_js"
	WEBGL = "webgl"
	WEBGPU = "webgpu"
	A_FRAME = "a_frame"  # For VR/AR


class VisualizationMode(str, Enum):
	"""Visualization modes"""
	STANDARD_3D = "standard_3d"
	VR_IMMERSIVE = "vr_immersive"
	AR_OVERLAY = "ar_overlay"
	HOLOGRAPHIC = "holographic"
	SPATIAL_COMPUTING = "spatial_computing"


class LayoutAlgorithm(str, Enum):
	"""3D layout algorithms"""
	FORCE_DIRECTED = "force_directed"
	HIERARCHICAL = "hierarchical"
	CIRCULAR = "circular"
	SPHERE = "sphere"
	HELIX = "helix"
	CLUSTERED = "clustered"
	ORGANIC = "organic"
	PHYSICS_BASED = "physics_based"


class AnimationType(str, Enum):
	"""Animation types"""
	NONE = "none"
	FADE = "fade"
	SLIDE = "slide"
	SCALE = "scale"
	MORPH = "morph"
	PARTICLE = "particle"
	FLUID = "fluid"
	PHYSICS = "physics"


@dataclass
class Vector3D:
	"""3D vector representation"""
	x: float = 0.0
	y: float = 0.0
	z: float = 0.0
	
	def __add__(self, other):
		return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)
	
	def __sub__(self, other):
		return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)
	
	def __mul__(self, scalar):
		return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)
	
	def magnitude(self):
		return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)
	
	def normalize(self):
		mag = self.magnitude()
		if mag > 0:
			return Vector3D(self.x / mag, self.y / mag, self.z / mag)
		return Vector3D()
	
	def distance_to(self, other):
		return (self - other).magnitude()


@dataclass
class Node3D:
	"""3D workflow node representation"""
	id: str
	position: Vector3D = field(default_factory=Vector3D)
	velocity: Vector3D = field(default_factory=Vector3D)
	force: Vector3D = field(default_factory=Vector3D)
	size: float = 1.0
	mass: float = 1.0
	color: str = "#4CAF50"
	opacity: float = 1.0
	label: str = ""
	node_type: str = "task"
	metadata: Dict[str, Any] = field(default_factory=dict)
	fixed: bool = False
	visible: bool = True
	selected: bool = False
	highlighted: bool = False
	cluster_id: Optional[str] = None


@dataclass
class Edge3D:
	"""3D workflow edge representation"""
	id: str
	source_id: str
	target_id: str
	weight: float = 1.0
	color: str = "#2196F3"
	opacity: float = 0.8
	width: float = 2.0
	edge_type: str = "flow"
	animated: bool = False
	bidirectional: bool = False
	metadata: Dict[str, Any] = field(default_factory=dict)
	visible: bool = True


@dataclass
class Cluster3D:
	"""3D node cluster representation"""
	id: str
	center: Vector3D = field(default_factory=Vector3D)
	radius: float = 5.0
	color: str = "#FF9800"
	opacity: float = 0.3
	label: str = ""
	node_ids: List[str] = field(default_factory=list)
	metadata: Dict[str, Any] = field(default_factory=dict)


class PhysicsEngine:
	"""3D physics simulation engine"""
	
	def __init__(self):
		self.gravity = Vector3D(0, -0.1, 0)
		self.damping = 0.95
		self.repulsion_force = 100.0
		self.attraction_force = 0.1
		self.spring_length = 50.0
		self.spring_strength = 0.01
		self.time_step = 0.016  # ~60 FPS
		self.max_velocity = 10.0
		self.collision_enabled = True
		self.collision_radius = 2.0
	
	def apply_forces(self, nodes: List[Node3D], edges: List[Edge3D]):
		"""Apply physics forces to nodes"""
		# Reset forces
		for node in nodes:
			if not node.fixed:
				node.force = Vector3D()
		
		# Apply repulsion forces between nodes
		self._apply_repulsion_forces(nodes)
		
		# Apply attraction forces along edges
		self._apply_attraction_forces(nodes, edges)
		
		# Apply gravity
		self._apply_gravity(nodes)
		
		# Apply clustering forces
		self._apply_clustering_forces(nodes)
	
	def _apply_repulsion_forces(self, nodes: List[Node3D]):
		"""Apply repulsion forces between nodes"""
		for i, node1 in enumerate(nodes):
			if node1.fixed:
				continue
				
			for j, node2 in enumerate(nodes):
				if i == j or not node2.visible:
					continue
				
				distance = node1.position.distance_to(node2.position)
				if distance < 0.1:
					distance = 0.1
				
				# Coulomb-like repulsion
				force_magnitude = self.repulsion_force / (distance ** 2)
				direction = (node1.position - node2.position).normalize()
				
				force = direction * force_magnitude
				node1.force = node1.force + force
	
	def _apply_attraction_forces(self, nodes: List[Node3D], edges: List[Edge3D]):
		"""Apply attraction forces along edges (springs)"""
		node_map = {node.id: node for node in nodes}
		
		for edge in edges:
			if not edge.visible:
				continue
				
			source = node_map.get(edge.source_id)
			target = node_map.get(edge.target_id)
			
			if not source or not target:
				continue
			
			# Spring force
			distance = source.position.distance_to(target.position)
			spring_force = self.spring_strength * (distance - self.spring_length)
			direction = (target.position - source.position).normalize()
			
			force = direction * spring_force * edge.weight
			
			if not source.fixed:
				source.force = source.force + force
			if not target.fixed:
				target.force = target.force - force
	
	def _apply_gravity(self, nodes: List[Node3D]):
		"""Apply gravity force"""
		for node in nodes:
			if not node.fixed:
				node.force = node.force + (self.gravity * node.mass)
	
	def _apply_clustering_forces(self, nodes: List[Node3D]):
		"""Apply forces to maintain clusters"""
		clusters = {}
		
		# Group nodes by cluster
		for node in nodes:
			if node.cluster_id:
				if node.cluster_id not in clusters:
					clusters[node.cluster_id] = []
				clusters[node.cluster_id].append(node)
		
		# Apply clustering forces
		for cluster_nodes in clusters.values():
			if len(cluster_nodes) < 2:
				continue
			
			# Calculate cluster center
			center = Vector3D()
			for node in cluster_nodes:
				center = center + node.position
			center = center * (1.0 / len(cluster_nodes))
			
			# Apply forces toward cluster center
			for node in cluster_nodes:
				if not node.fixed:
					direction = (center - node.position).normalize()
					force = direction * self.attraction_force
					node.force = node.force + force
	
	def update_positions(self, nodes: List[Node3D]):
		"""Update node positions based on forces"""
		for node in nodes:
			if node.fixed or not node.visible:
				continue
			
			# Update velocity
			acceleration = node.force * (1.0 / node.mass)
			node.velocity = node.velocity + (acceleration * self.time_step)
			
			# Apply damping
			node.velocity = node.velocity * self.damping
			
			# Limit velocity
			if node.velocity.magnitude() > self.max_velocity:
				node.velocity = node.velocity.normalize() * self.max_velocity
			
			# Update position
			node.position = node.position + (node.velocity * self.time_step)
			
			# Handle collisions
			if self.collision_enabled:
				self._handle_collisions(node, nodes)
	
	def _handle_collisions(self, node: Node3D, all_nodes: List[Node3D]):
		"""Handle node collisions"""
		for other in all_nodes:
			if node.id == other.id or not other.visible:
				continue
			
			distance = node.position.distance_to(other.position)
			min_distance = self.collision_radius * (node.size + other.size)
			
			if distance < min_distance:
				# Separate nodes
				direction = (node.position - other.position).normalize()
				separation = (min_distance - distance) * 0.5
				
				if not node.fixed:
					node.position = node.position + (direction * separation)
				if not other.fixed:
					other.position = other.position - (direction * separation)


class LayoutEngine3D:
	"""3D layout algorithms for workflow visualization"""
	
	def __init__(self):
		self.physics = PhysicsEngine()
	
	def apply_layout(self, nodes: List[Node3D], edges: List[Edge3D], algorithm: LayoutAlgorithm, **kwargs) -> Tuple[List[Node3D], List[Edge3D]]:
		"""Apply layout algorithm to nodes and edges"""
		if algorithm == LayoutAlgorithm.FORCE_DIRECTED:
			return self._force_directed_layout(nodes, edges, **kwargs)
		elif algorithm == LayoutAlgorithm.HIERARCHICAL:
			return self._hierarchical_layout(nodes, edges, **kwargs)
		elif algorithm == LayoutAlgorithm.CIRCULAR:
			return self._circular_layout(nodes, edges, **kwargs)
		elif algorithm == LayoutAlgorithm.SPHERE:
			return self._sphere_layout(nodes, edges, **kwargs)
		elif algorithm == LayoutAlgorithm.HELIX:
			return self._helix_layout(nodes, edges, **kwargs)
		elif algorithm == LayoutAlgorithm.CLUSTERED:
			return self._clustered_layout(nodes, edges, **kwargs)
		elif algorithm == LayoutAlgorithm.ORGANIC:
			return self._organic_layout(nodes, edges, **kwargs)
		elif algorithm == LayoutAlgorithm.PHYSICS_BASED:
			return self._physics_based_layout(nodes, edges, **kwargs)
		else:
			return nodes, edges
	
	def _force_directed_layout(self, nodes: List[Node3D], edges: List[Edge3D], iterations: int = 100, **kwargs) -> Tuple[List[Node3D], List[Edge3D]]:
		"""Force-directed 3D layout"""
		# Initialize random positions if not set
		for node in nodes:
			if node.position.magnitude() == 0:
				node.position = Vector3D(
					(np.random.random() - 0.5) * 100,
					(np.random.random() - 0.5) * 100,
					(np.random.random() - 0.5) * 100
				)
		
		# Run physics simulation
		for _ in range(iterations):
			self.physics.apply_forces(nodes, edges)
			self.physics.update_positions(nodes)
		
		return nodes, edges
	
	def _hierarchical_layout(self, nodes: List[Node3D], edges: List[Edge3D], **kwargs) -> Tuple[List[Node3D], List[Edge3D]]:
		"""Hierarchical 3D layout"""
		# Build hierarchy from edges
		hierarchy = self._build_hierarchy(nodes, edges)
		
		# Position nodes by level
		level_height = kwargs.get('level_height', 50.0)
		level_spacing = kwargs.get('level_spacing', 30.0)
		
		for level, level_nodes in hierarchy.items():
			y = level * level_height
			
			# Arrange nodes in level in a circle
			if len(level_nodes) == 1:
				level_nodes[0].position = Vector3D(0, y, 0)
			else:
				radius = max(20.0, len(level_nodes) * 5.0)
				for i, node in enumerate(level_nodes):
					angle = (2 * math.pi * i) / len(level_nodes)
					x = radius * math.cos(angle)
					z = radius * math.sin(angle)
					node.position = Vector3D(x, y, z)
		
		return nodes, edges
	
	def _circular_layout(self, nodes: List[Node3D], edges: List[Edge3D], **kwargs) -> Tuple[List[Node3D], List[Edge3D]]:
		"""Circular 3D layout"""
		radius = kwargs.get('radius', 50.0)
		height_variation = kwargs.get('height_variation', 20.0)
		
		for i, node in enumerate(nodes):
			angle = (2 * math.pi * i) / len(nodes)
			x = radius * math.cos(angle)
			z = radius * math.sin(angle)
			y = (np.random.random() - 0.5) * height_variation
			node.position = Vector3D(x, y, z)
		
		return nodes, edges
	
	def _sphere_layout(self, nodes: List[Node3D], edges: List[Edge3D], **kwargs) -> Tuple[List[Node3D], List[Edge3D]]:
		"""Spherical 3D layout"""
		radius = kwargs.get('radius', 50.0)
		
		for i, node in enumerate(nodes):
			# Fibonacci sphere distribution
			phi = math.acos(1 - 2 * (i + 0.5) / len(nodes))
			theta = math.pi * (1 + 5**0.5) * (i + 0.5)
			
			x = radius * math.sin(phi) * math.cos(theta)
			y = radius * math.sin(phi) * math.sin(theta)
			z = radius * math.cos(phi)
			
			node.position = Vector3D(x, y, z)
		
		return nodes, edges
	
	def _helix_layout(self, nodes: List[Node3D], edges: List[Edge3D], **kwargs) -> Tuple[List[Node3D], List[Edge3D]]:
		"""Helical 3D layout"""
		radius = kwargs.get('radius', 30.0)
		pitch = kwargs.get('pitch', 10.0)
		turns = kwargs.get('turns', 3.0)
		
		total_angle = turns * 2 * math.pi
		
		for i, node in enumerate(nodes):
			t = i / len(nodes)
			angle = total_angle * t
			
			x = radius * math.cos(angle)
			z = radius * math.sin(angle)
			y = pitch * turns * t
			
			node.position = Vector3D(x, y, z)
		
		return nodes, edges
	
	def _clustered_layout(self, nodes: List[Node3D], edges: List[Edge3D], **kwargs) -> Tuple[List[Node3D], List[Edge3D]]:
		"""Clustered 3D layout"""
		# Group nodes by cluster_id
		clusters = {}
		unclustered = []
		
		for node in nodes:
			if node.cluster_id:
				if node.cluster_id not in clusters:
					clusters[node.cluster_id] = []
				clusters[node.cluster_id].append(node)
			else:
				unclustered.append(node)
		
		cluster_radius = kwargs.get('cluster_radius', 20.0)
		cluster_separation = kwargs.get('cluster_separation', 100.0)
		
		# Position cluster centers
		cluster_centers = []
		for i, (cluster_id, cluster_nodes) in enumerate(clusters.items()):
			angle = (2 * math.pi * i) / len(clusters)
			center_x = cluster_separation * math.cos(angle)
			center_z = cluster_separation * math.sin(angle)
			center_y = (np.random.random() - 0.5) * 20.0
			
			cluster_center = Vector3D(center_x, center_y, center_z)
			cluster_centers.append(cluster_center)
			
			# Position nodes within cluster
			for j, node in enumerate(cluster_nodes):
				if len(cluster_nodes) == 1:
					node.position = cluster_center
				else:
					node_angle = (2 * math.pi * j) / len(cluster_nodes)
					offset_x = cluster_radius * math.cos(node_angle)
					offset_z = cluster_radius * math.sin(node_angle)
					offset_y = (np.random.random() - 0.5) * 10.0
					
					node.position = cluster_center + Vector3D(offset_x, offset_y, offset_z)
		
		# Position unclustered nodes
		for i, node in enumerate(unclustered):
			angle = (2 * math.pi * i) / max(1, len(unclustered))
			x = 150 * math.cos(angle)
			z = 150 * math.sin(angle)
			y = (np.random.random() - 0.5) * 30.0
			node.position = Vector3D(x, y, z)
		
		return nodes, edges
	
	def _organic_layout(self, nodes: List[Node3D], edges: List[Edge3D], **kwargs) -> Tuple[List[Node3D], List[Edge3D]]:
		"""Organic 3D layout with natural clustering"""
		# Start with force-directed layout
		nodes, edges = self._force_directed_layout(nodes, edges, iterations=50)
		
		# Apply organic forces for natural clustering
		organic_strength = kwargs.get('organic_strength', 0.5)
		
		for _ in range(50):
			for node in nodes:
				if node.fixed:
					continue
				
				# Apply organic forces based on node type similarity
				organic_force = Vector3D()
				
				for other in nodes:
					if node.id == other.id:
						continue
					
					distance = node.position.distance_to(other.position)
					if distance < 0.1:
						continue
					
					# Attract similar node types
					type_similarity = 1.0 if node.node_type == other.node_type else 0.3
					force_magnitude = organic_strength * type_similarity / distance
					direction = (other.position - node.position).normalize()
					
					organic_force = organic_force + (direction * force_magnitude)
				
				# Apply force
				node.position = node.position + (organic_force * 0.1)
		
		return nodes, edges
	
	def _physics_based_layout(self, nodes: List[Node3D], edges: List[Edge3D], **kwargs) -> Tuple[List[Node3D], List[Edge3D]]:
		"""Advanced physics-based layout with realistic dynamics"""
		iterations = kwargs.get('iterations', 200)
		
		# Configure physics parameters
		self.physics.repulsion_force = kwargs.get('repulsion_force', 150.0)
		self.physics.attraction_force = kwargs.get('attraction_force', 0.05)
		self.physics.spring_length = kwargs.get('spring_length', 60.0)
		self.physics.damping = kwargs.get('damping', 0.9)
		
		# Initialize with some energy
		for node in nodes:
			if node.position.magnitude() == 0:
				node.position = Vector3D(
					(np.random.random() - 0.5) * 200,
					(np.random.random() - 0.5) * 200,
					(np.random.random() - 0.5) * 200
				)
				node.velocity = Vector3D(
					(np.random.random() - 0.5) * 5,
					(np.random.random() - 0.5) * 5,
					(np.random.random() - 0.5) * 5
				)
		
		# Run advanced physics simulation
		for i in range(iterations):
			self.physics.apply_forces(nodes, edges)
			self.physics.update_positions(nodes)
			
			# Gradually reduce forces for stability
			if i > iterations * 0.7:
				self.physics.damping = min(0.98, self.physics.damping + 0.001)
		
		return nodes, edges
	
	def _build_hierarchy(self, nodes: List[Node3D], edges: List[Edge3D]) -> Dict[int, List[Node3D]]:
		"""Build hierarchy levels from workflow edges"""
		node_map = {node.id: node for node in nodes}
		
		# Find root nodes (no incoming edges)
		incoming = set()
		for edge in edges:
			incoming.add(edge.target_id)
		
		roots = [node for node in nodes if node.id not in incoming]
		if not roots:
			# If no clear roots, pick nodes with minimal incoming edges
			roots = nodes[:1]
		
		# Build levels using BFS
		hierarchy = {0: roots}
		visited = set(node.id for node in roots)
		current_level = 0
		
		while True:
			next_level_nodes = []
			
			for edge in edges:
				if edge.source_id in visited and edge.target_id not in visited:
					target_node = node_map.get(edge.target_id)
					if target_node:
						next_level_nodes.append(target_node)
						visited.add(edge.target_id)
			
			if not next_level_nodes:
				break
			
			current_level += 1
			hierarchy[current_level] = next_level_nodes
		
		return hierarchy


class AnimationEngine:
	"""3D animation and transition engine"""
	
	def __init__(self):
		self.active_animations = []
		self.animation_speed = 1.0
		self.easing_function = "ease_in_out"
	
	def animate_layout_transition(self, nodes: List[Node3D], target_positions: Dict[str, Vector3D], duration: float = 2.0, animation_type: AnimationType = AnimationType.SLIDE) -> str:
		"""Animate transition between layouts"""
		animation_id = uuid7str()
		
		animation = {
			"id": animation_id,
			"type": "layout_transition",
			"duration": duration,
			"animation_type": animation_type,
			"start_time": time.time(),
			"nodes": [],
			"progress": 0.0
		}
		
		# Store initial and target positions
		for node in nodes:
			node_animation = {
				"node_id": node.id,
				"start_position": Vector3D(node.position.x, node.position.y, node.position.z),
				"target_position": target_positions.get(node.id, node.position),
				"current_position": Vector3D(node.position.x, node.position.y, node.position.z)
			}
			animation["nodes"].append(node_animation)
		
		self.active_animations.append(animation)
		return animation_id
	
	def animate_node_selection(self, node: Node3D, duration: float = 0.5) -> str:
		"""Animate node selection effect"""
		animation_id = uuid7str()
		
		animation = {
			"id": animation_id,
			"type": "node_selection",
			"node_id": node.id,
			"duration": duration,
			"start_time": time.time(),
			"original_size": node.size,
			"target_size": node.size * 1.3,
			"progress": 0.0
		}
		
		self.active_animations.append(animation)
		return animation_id
	
	def animate_workflow_execution(self, execution_path: List[str], nodes: List[Node3D], edges: List[Edge3D], duration: float = 5.0) -> str:
		"""Animate workflow execution flow"""
		animation_id = uuid7str()
		
		animation = {
			"id": animation_id,
			"type": "workflow_execution",
			"execution_path": execution_path,
			"duration": duration,
			"start_time": time.time(),
			"current_step": 0,
			"progress": 0.0,
			"particle_effects": []
		}
		
		self.active_animations.append(animation)
		return animation_id
	
	def update_animations(self, nodes: List[Node3D], edges: List[Edge3D], delta_time: float):
		"""Update all active animations"""
		current_time = time.time()
		completed_animations = []
		
		for animation in self.active_animations:
			elapsed = current_time - animation["start_time"]
			progress = min(1.0, elapsed / animation["duration"])
			animation["progress"] = progress
			
			if animation["type"] == "layout_transition":
				self._update_layout_transition(animation, nodes, progress)
			elif animation["type"] == "node_selection":
				self._update_node_selection(animation, nodes, progress)
			elif animation["type"] == "workflow_execution":
				self._update_workflow_execution(animation, nodes, edges, progress)
			
			if progress >= 1.0:
				completed_animations.append(animation)
		
		# Remove completed animations
		for animation in completed_animations:
			self.active_animations.remove(animation)
	
	def _update_layout_transition(self, animation: Dict[str, Any], nodes: List[Node3D], progress: float):
		"""Update layout transition animation"""
		# Apply easing
		eased_progress = self._apply_easing(progress)
		
		node_map = {node.id: node for node in nodes}
		
		for node_anim in animation["nodes"]:
			node = node_map.get(node_anim["node_id"])
			if not node:
				continue
			
			start_pos = node_anim["start_position"]
			target_pos = node_anim["target_position"]
			
			# Interpolate position
			node.position = Vector3D(
				start_pos.x + (target_pos.x - start_pos.x) * eased_progress,
				start_pos.y + (target_pos.y - start_pos.y) * eased_progress,
				start_pos.z + (target_pos.z - start_pos.z) * eased_progress
			)
	
	def _update_node_selection(self, animation: Dict[str, Any], nodes: List[Node3D], progress: float):
		"""Update node selection animation"""
		node_map = {node.id: node for node in nodes}
		node = node_map.get(animation["node_id"])
		
		if node:
			# Pulse effect
			pulse = math.sin(progress * math.pi * 4) * 0.2 + 1.0
			node.size = animation["original_size"] * pulse
			
			if progress >= 1.0:
				node.size = animation["original_size"]
	
	def _update_workflow_execution(self, animation: Dict[str, Any], nodes: List[Node3D], edges: List[Edge3D], progress: float):
		"""Update workflow execution animation"""
		path = animation["execution_path"]
		if not path:
			return
		
		# Calculate current step
		step_progress = progress * len(path)
		current_step = int(step_progress)
		step_fraction = step_progress - current_step
		
		animation["current_step"] = current_step
		
		# Highlight current nodes
		node_map = {node.id: node for node in nodes}
		
		# Reset all nodes
		for node in nodes:
			node.highlighted = False
		
		# Highlight active nodes
		if current_step < len(path):
			current_node = node_map.get(path[current_step])
			if current_node:
				current_node.highlighted = True
		
		# Add particle effects along edges
		if current_step > 0 and current_step < len(path):
			prev_node_id = path[current_step - 1]
			curr_node_id = path[current_step]
			
			# Find edge between nodes
			for edge in edges:
				if (edge.source_id == prev_node_id and edge.target_id == curr_node_id) or \
				   (edge.source_id == curr_node_id and edge.target_id == prev_node_id):
					edge.animated = True
	
	def _apply_easing(self, t: float) -> float:
		"""Apply easing function to animation progress"""
		if self.easing_function == "linear":
			return t
		elif self.easing_function == "ease_in":
			return t * t
		elif self.easing_function == "ease_out":
			return 1 - (1 - t) * (1 - t)
		elif self.easing_function == "ease_in_out":
			if t < 0.5:
				return 2 * t * t
			else:
				return 1 - 2 * (1 - t) * (1 - t)
		elif self.easing_function == "bounce":
			if t < 0.5:
				return 2 * t * t
			else:
				return 1 - 2 * (1 - t) * (1 - t) * abs(math.sin(t * math.pi * 8))
		else:
			return t


class Visualization3DEngine:
	"""Main 3D visualization engine"""
	
	def __init__(self, rendering_engine: RenderingEngine = RenderingEngine.THREE_JS):
		self.rendering_engine = rendering_engine
		self.layout_engine = LayoutEngine3D()
		self.animation_engine = AnimationEngine()
		self.physics_enabled = True
		self.auto_layout = True
		self.performance_mode = "balanced"  # low, balanced, high
		
		# Rendering settings
		self.camera_position = Vector3D(0, 0, 100)
		self.camera_target = Vector3D(0, 0, 0)
		self.field_of_view = 75
		self.near_clip = 0.1
		self.far_clip = 2000
		
		# Lighting
		self.ambient_light = {"color": "#404040", "intensity": 0.4}
		self.directional_light = {"color": "#ffffff", "intensity": 0.8, "position": Vector3D(10, 10, 5)}
		
		# Effects
		self.shadows_enabled = True
		self.fog_enabled = True
		self.post_processing = True
		
		# Interaction
		self.controls_enabled = True
		self.zoom_enabled = True
		self.rotation_enabled = True
		self.selection_enabled = True
	
	async def render_workflow_3d(self, workflow_id: str, mode: VisualizationMode = VisualizationMode.STANDARD_3D, layout: LayoutAlgorithm = LayoutAlgorithm.FORCE_DIRECTED) -> Dict[str, Any]:
		"""Render workflow in 3D"""
		try:
			# Load workflow data
			workflow_data = await self._load_workflow_data(workflow_id)
			
			# Convert to 3D representation
			nodes_3d, edges_3d, clusters_3d = await self._convert_to_3d(workflow_data)
			
			# Apply layout algorithm
			nodes_3d, edges_3d = self.layout_engine.apply_layout(nodes_3d, edges_3d, layout)
			
			# Generate rendering data
			render_data = self._generate_render_data(nodes_3d, edges_3d, clusters_3d, mode)
			
			# Add metadata
			render_data.update({
				"workflow_id": workflow_id,
				"mode": mode.value,
				"layout": layout.value,
				"rendering_engine": self.rendering_engine.value,
				"timestamp": datetime.utcnow().isoformat(),
				"camera": {
					"position": {"x": self.camera_position.x, "y": self.camera_position.y, "z": self.camera_position.z},
					"target": {"x": self.camera_target.x, "y": self.camera_target.y, "z": self.camera_target.z},
					"fov": self.field_of_view
				},
				"lighting": {
					"ambient": self.ambient_light,
					"directional": {
						"color": self.directional_light["color"],
						"intensity": self.directional_light["intensity"],
						"position": {
							"x": self.directional_light["position"].x,
							"y": self.directional_light["position"].y,
							"z": self.directional_light["position"].z
						}
					}
				},
				"effects": {
					"shadows": self.shadows_enabled,
					"fog": self.fog_enabled,
					"post_processing": self.post_processing
				}
			})
			
			return render_data
			
		except Exception as e:
			print(f"3D rendering error: {e}")
			raise
	
	async def _load_workflow_data(self, workflow_id: str) -> Dict[str, Any]:
		"""Load workflow data from database"""
		try:
			async with get_async_db_session() as session:
				from sqlalchemy import text
				
				# Get workflow
				workflow_result = await session.execute(
					text("""
					SELECT workflow_id, name, description, definition, metadata
					FROM cr_workflows 
					WHERE workflow_id = :workflow_id
					"""),
					{"workflow_id": workflow_id}
				)
				
				workflow_row = workflow_result.fetchone()
				if not workflow_row:
					raise ValueError(f"Workflow {workflow_id} not found")
				
				workflow_definition = json.loads(workflow_row[3]) if workflow_row[3] else {}
				
				# Get execution data for enhancement
				execution_result = await session.execute(
					text("""
					SELECT status, current_step, execution_path, metrics
					FROM cr_workflow_executions 
					WHERE workflow_id = :workflow_id
					ORDER BY created_at DESC
					LIMIT 5
					"""),
					{"workflow_id": workflow_id}
				)
				
				executions = execution_result.fetchall()
				
				return {
					"workflow_id": workflow_row[0],
					"name": workflow_row[1],
					"description": workflow_row[2],
					"definition": workflow_definition,
					"metadata": json.loads(workflow_row[4]) if workflow_row[4] else {},
					"recent_executions": [
						{
							"status": row[0],
							"current_step": row[1],
							"execution_path": json.loads(row[2]) if row[2] else [],
							"metrics": json.loads(row[3]) if row[3] else {}
						}
						for row in executions
					]
				}
				
		except Exception as e:
			print(f"Workflow data loading error: {e}")
			raise
	
	async def _convert_to_3d(self, workflow_data: Dict[str, Any]) -> Tuple[List[Node3D], List[Edge3D], List[Cluster3D]]:
		"""Convert workflow data to 3D representation"""
		definition = workflow_data.get("definition", {})
		nodes = definition.get("nodes", [])
		edges = definition.get("edges", [])
		
		# Convert nodes
		nodes_3d = []
		for node in nodes:
			node_3d = Node3D(
				id=node.get("id", ""),
				label=node.get("name", node.get("label", "")),
				node_type=node.get("type", "task"),
				size=self._calculate_node_size(node),
				color=self._get_node_color(node),
				metadata=node.get("metadata", {})
			)
			
			# Set cluster if specified
			if "cluster" in node:
				node_3d.cluster_id = node["cluster"]
			
			nodes_3d.append(node_3d)
		
		# Convert edges
		edges_3d = []
		for edge in edges:
			edge_3d = Edge3D(
				id=edge.get("id", f"{edge.get('source', '')}-{edge.get('target', '')}"),
				source_id=edge.get("source", ""),
				target_id=edge.get("target", ""),
				weight=edge.get("weight", 1.0),
				color=self._get_edge_color(edge),
				edge_type=edge.get("type", "flow"),
				metadata=edge.get("metadata", {})
			)
			edges_3d.append(edge_3d)
		
		# Create clusters
		clusters_3d = self._create_clusters(nodes_3d)
		
		return nodes_3d, edges_3d, clusters_3d
	
	def _calculate_node_size(self, node: Dict[str, Any]) -> float:
		"""Calculate node size based on properties"""
		base_size = 2.0
		
		# Size based on node type
		type_sizes = {
			"start": 3.0,
			"end": 3.0,
			"decision": 2.5,
			"process": 2.0,
			"task": 2.0,
			"connector": 1.5
		}
		
		node_type = node.get("type", "task")
		size = type_sizes.get(node_type, base_size)
		
		# Adjust based on complexity
		complexity = node.get("complexity", 1)
		size *= (1.0 + complexity * 0.2)
		
		# Adjust based on execution frequency
		exec_count = node.get("execution_count", 0)
		if exec_count > 0:
			size *= (1.0 + min(exec_count / 100.0, 0.5))
		
		return size
	
	def _get_node_color(self, node: Dict[str, Any]) -> str:
		"""Get node color based on type and status"""
		node_type = node.get("type", "task")
		status = node.get("status", "inactive")
		
		# Base colors by type
		type_colors = {
			"start": "#4CAF50",      # Green
			"end": "#F44336",        # Red
			"decision": "#FF9800",   # Orange
			"process": "#2196F3",    # Blue
			"task": "#9C27B0",       # Purple
			"connector": "#607D8B"   # Blue Grey
		}
		
		# Status modifiers
		if status == "running":
			return "#FFD700"  # Gold
		elif status == "completed":
			return "#4CAF50"  # Green
		elif status == "error":
			return "#F44336"  # Red
		elif status == "waiting":
			return "#FF9800"  # Orange
		
		return type_colors.get(node_type, "#9E9E9E")
	
	def _get_edge_color(self, edge: Dict[str, Any]) -> str:
		"""Get edge color based on type and status"""
		edge_type = edge.get("type", "flow")
		status = edge.get("status", "inactive")
		
		# Base colors by type
		type_colors = {
			"flow": "#2196F3",       # Blue
			"data": "#4CAF50",       # Green
			"error": "#F44336",      # Red
			"condition": "#FF9800"   # Orange
		}
		
		# Status modifiers
		if status == "active":
			return "#FFD700"  # Gold
		elif status == "completed":
			return "#4CAF50"  # Green
		
		return type_colors.get(edge_type, "#9E9E9E")
	
	def _create_clusters(self, nodes_3d: List[Node3D]) -> List[Cluster3D]:
		"""Create clusters from nodes"""
		cluster_groups = {}
		
		for node in nodes_3d:
			if node.cluster_id:
				if node.cluster_id not in cluster_groups:
					cluster_groups[node.cluster_id] = []
				cluster_groups[node.cluster_id].append(node.id)
		
		clusters_3d = []
		for cluster_id, node_ids in cluster_groups.items():
			cluster = Cluster3D(
				id=cluster_id,
				label=f"Cluster {cluster_id}",
				node_ids=node_ids,
				radius=max(10.0, len(node_ids) * 2.5),
				color="#E0E0E0",
				opacity=0.2
			)
			clusters_3d.append(cluster)
		
		return clusters_3d
	
	def _generate_render_data(self, nodes_3d: List[Node3D], edges_3d: List[Edge3D], clusters_3d: List[Cluster3D], mode: VisualizationMode) -> Dict[str, Any]:
		"""Generate rendering data for 3D engine"""
		render_data = {
			"nodes": [
				{
					"id": node.id,
					"position": {"x": node.position.x, "y": node.position.y, "z": node.position.z},
					"size": node.size,
					"color": node.color,
					"opacity": node.opacity,
					"label": node.label,
					"type": node.node_type,
					"metadata": node.metadata,
					"visible": node.visible,
					"selected": node.selected,
					"highlighted": node.highlighted
				}
				for node in nodes_3d
			],
			"edges": [
				{
					"id": edge.id,
					"source": edge.source_id,
					"target": edge.target_id,
					"weight": edge.weight,
					"color": edge.color,
					"opacity": edge.opacity,
					"width": edge.width,
					"type": edge.edge_type,
					"animated": edge.animated,
					"bidirectional": edge.bidirectional,
					"metadata": edge.metadata,
					"visible": edge.visible
				}
				for edge in edges_3d
			],
			"clusters": [
				{
					"id": cluster.id,
					"center": {"x": cluster.center.x, "y": cluster.center.y, "z": cluster.center.z},
					"radius": cluster.radius,
					"color": cluster.color,
					"opacity": cluster.opacity,
					"label": cluster.label,
					"nodes": cluster.node_ids,
					"metadata": cluster.metadata
				}
				for cluster in clusters_3d
			]
		}
		
		# Add mode-specific enhancements
		if mode == VisualizationMode.VR_IMMERSIVE:
			render_data["vr_settings"] = {
				"room_scale": True,
				"hand_tracking": True,
				"teleportation": True,
				"haptic_feedback": True
			}
		elif mode == VisualizationMode.AR_OVERLAY:
			render_data["ar_settings"] = {
				"surface_detection": True,
				"occlusion": True,
				"lighting_estimation": True,
				"anchor_tracking": True
			}
		elif mode == VisualizationMode.HOLOGRAPHIC:
			render_data["hologram_settings"] = {
				"projection_type": "volumetric",
				"viewing_angle": 360,
				"depth_layers": 32,
				"transparency": True
			}
		
		return render_data
	
	async def animate_workflow_execution(self, workflow_id: str, execution_id: str) -> str:
		"""Animate workflow execution in 3D"""
		try:
			# Load execution data
			async with get_async_db_session() as session:
				from sqlalchemy import text
				
				result = await session.execute(
					text("""
					SELECT execution_path, current_step, status
					FROM cr_workflow_executions 
					WHERE execution_id = :execution_id
					"""),
					{"execution_id": execution_id}
				)
				
				row = result.fetchone()
				if not row:
					raise ValueError(f"Execution {execution_id} not found")
				
				execution_path = json.loads(row[0]) if row[0] else []
				
				# Load workflow for node/edge data
				workflow_data = await self._load_workflow_data(workflow_id)
				nodes_3d, edges_3d, _ = await self._convert_to_3d(workflow_data)
				
				# Start animation
				animation_id = self.animation_engine.animate_workflow_execution(
					execution_path, nodes_3d, edges_3d, duration=10.0
				)
				
				return animation_id
				
		except Exception as e:
			print(f"Workflow execution animation error: {e}")
			raise
	
	async def update_visualization(self, delta_time: float, nodes_3d: List[Node3D], edges_3d: List[Edge3D]):
		"""Update visualization (called on each frame)"""
		# Update animations
		self.animation_engine.update_animations(nodes_3d, edges_3d, delta_time)
		
		# Update physics if enabled
		if self.physics_enabled:
			self.layout_engine.physics.apply_forces(nodes_3d, edges_3d)
			self.layout_engine.physics.update_positions(nodes_3d)
	
	def optimize_performance(self, node_count: int, edge_count: int):
		"""Optimize rendering performance based on complexity"""
		total_elements = node_count + edge_count
		
		if total_elements > 1000:
			self.performance_mode = "low"
			self.shadows_enabled = False
			self.post_processing = False
			self.layout_engine.physics.collision_enabled = False
		elif total_elements > 500:
			self.performance_mode = "balanced"
			self.shadows_enabled = True
			self.post_processing = True
		else:
			self.performance_mode = "high"
			self.shadows_enabled = True
			self.post_processing = True
			self.fog_enabled = True


# Global 3D visualization engine instance
visualization_3d = Visualization3DEngine()