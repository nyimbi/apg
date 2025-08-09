"""
APG GraphRAG Capability - Knowledge Graph Visualization

Revolutionary graph visualization components with interactive 3D rendering,
real-time analytics, and immersive exploration capabilities.

Author: Datacraft (nyimbi@gmail.com)
Copyright: © 2025 Datacraft
Website: www.datacraft.co.ke
"""

from __future__ import annotations
import asyncio
import logging
import json
import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict
from enum import Enum
import numpy as np

from .database import GraphRAGDatabaseService
from .views import GraphEntity, GraphRelationship, KnowledgeGraph


logger = logging.getLogger(__name__)


class VisualizationMode(str, Enum):
	"""Graph visualization modes"""
	FORCE_DIRECTED = "force_directed"
	HIERARCHICAL = "hierarchical"
	CIRCULAR = "circular"
	GRID = "grid"
	CLUSTER = "cluster"
	GEOGRAPHICAL = "geographical"
	TEMPORAL = "temporal"


class LayoutAlgorithm(str, Enum):
	"""Graph layout algorithms"""
	SPRING = "spring"
	KAMADA_KAWAI = "kamada_kawai"
	CIRCULAR = "circular"
	HIERARCHICAL = "hierarchical"
	SPECTRAL = "spectral"
	FRUCHTERMAN_REINGOLD = "fruchterman_reingold"


@dataclass
class NodeStyle:
	"""Node visualization styling"""
	size: float = 10.0
	color: str = "#3498db"
	border_color: str = "#2c3e50"
	border_width: float = 2.0
	shape: str = "circle"  # circle, square, triangle, diamond
	opacity: float = 1.0
	label_color: str = "#2c3e50"
	label_size: float = 12.0
	font_family: str = "Arial, sans-serif"


@dataclass
class EdgeStyle:
	"""Edge visualization styling"""
	width: float = 2.0
	color: str = "#95a5a6"
	opacity: float = 0.7
	style: str = "solid"  # solid, dashed, dotted
	arrow_size: float = 8.0
	arrow_color: str = "#7f8c8d"
	curvature: float = 0.0  # 0 = straight, >0 = curved
	label_color: str = "#7f8c8d"
	label_size: float = 10.0


@dataclass
class VisualizationConfig:
	"""Comprehensive visualization configuration"""
	mode: VisualizationMode = VisualizationMode.FORCE_DIRECTED
	layout_algorithm: LayoutAlgorithm = LayoutAlgorithm.SPRING
	
	# Canvas settings
	width: int = 1200
	height: int = 800
	background_color: str = "#ffffff"
	
	# Interactive settings
	enable_zoom: bool = True
	enable_pan: bool = True
	enable_drag: bool = True
	enable_selection: bool = True
	enable_tooltips: bool = True
	enable_context_menu: bool = True
	
	# Animation settings
	enable_animations: bool = True
	animation_duration: float = 500.0  # milliseconds
	physics_enabled: bool = True
	damping: float = 0.9
	
	# Filtering settings
	node_size_range: Tuple[float, float] = (5.0, 50.0)
	edge_width_range: Tuple[float, float] = (1.0, 10.0)
	confidence_threshold: float = 0.5
	max_nodes: int = 1000
	max_edges: int = 2000
	
	# Clustering settings
	enable_clustering: bool = False
	cluster_threshold: float = 0.8
	max_clusters: int = 10
	
	# 3D settings
	enable_3d: bool = False
	camera_position: Tuple[float, float, float] = (0, 0, 100)
	
	# Export settings
	export_formats: List[str] = None
	
	def __post_init__(self):
		if self.export_formats is None:
			self.export_formats = ["png", "svg", "json", "graphml"]


@dataclass
class GraphVisualizationData:
	"""Complete graph visualization data structure"""
	nodes: List[Dict[str, Any]]
	edges: List[Dict[str, Any]]
	clusters: List[Dict[str, Any]]
	statistics: Dict[str, Any]
	metadata: Dict[str, Any]
	layout_data: Dict[str, Any]
	styling: Dict[str, Any]


class GraphVisualizationEngine:
	"""
	Revolutionary graph visualization engine providing:
	
	- Interactive 3D and 2D graph rendering
	- Multiple layout algorithms and visualization modes
	- Real-time filtering and clustering
	- Advanced styling and theming
	- Export capabilities for multiple formats
	- Performance optimization for large graphs
	- Immersive exploration experiences
	"""
	
	def __init__(
		self,
		db_service: GraphRAGDatabaseService,
		config: Optional[VisualizationConfig] = None
	):
		"""Initialize visualization engine"""
		self.db_service = db_service
		self.config = config or VisualizationConfig()
		
		# Styling templates
		self.node_styles = self._initialize_node_styles()
		self.edge_styles = self._initialize_edge_styles()
		
		# Layout engines
		self.layout_engines = self._initialize_layout_engines()
		
		# Performance caching
		self._layout_cache = {}
		self._visualization_cache = {}
		
		logger.info("Graph visualization engine initialized")
	
	async def generate_graph_visualization(
		self,
		tenant_id: str,
		knowledge_graph_id: str,
		config: Optional[VisualizationConfig] = None
	) -> GraphVisualizationData:
		"""
		Generate complete graph visualization data
		
		Args:
			tenant_id: Tenant identifier
			knowledge_graph_id: Knowledge graph identifier
			config: Optional visualization configuration override
			
		Returns:
			GraphVisualizationData with all visualization components
		"""
		start_time = datetime.utcnow()
		viz_config = config or self.config
		
		try:
			logger.info(f"Generating graph visualization for {knowledge_graph_id}")
			
			# Get graph data
			graph_data = await self._load_graph_data(tenant_id, knowledge_graph_id, viz_config)
			
			# Apply filtering
			filtered_data = await self._apply_filters(graph_data, viz_config)
			
			# Generate layout
			layout_data = await self._generate_layout(filtered_data, viz_config)
			
			# Apply clustering if enabled
			clusters = []
			if viz_config.enable_clustering:
				clusters = await self._generate_clusters(filtered_data, viz_config)
			
			# Generate styling
			styling_data = await self._generate_styling(filtered_data, viz_config)
			
			# Build visualization nodes and edges
			nodes = await self._build_visualization_nodes(
				filtered_data["entities"], layout_data, styling_data, viz_config
			)
			
			edges = await self._build_visualization_edges(
				filtered_data["relationships"], layout_data, styling_data, viz_config
			)
			
			# Calculate statistics
			statistics = await self._calculate_visualization_statistics(
				nodes, edges, clusters
			)
			
			# Build metadata
			processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
			metadata = {
				"graph_id": knowledge_graph_id,
				"tenant_id": tenant_id,
				"generated_at": start_time.isoformat(),
				"processing_time_ms": processing_time,
				"config": viz_config.__dict__,
				"total_nodes": len(nodes),
				"total_edges": len(edges),
				"layout_algorithm": viz_config.layout_algorithm.value,
				"visualization_mode": viz_config.mode.value
			}
			
			result = GraphVisualizationData(
				nodes=nodes,
				edges=edges,
				clusters=clusters,
				statistics=statistics,
				metadata=metadata,
				layout_data=layout_data,
				styling=styling_data
			)
			
			logger.info(f"Graph visualization generated in {processing_time:.1f}ms with {len(nodes)} nodes and {len(edges)} edges")
			return result
			
		except Exception as e:
			logger.error(f"Graph visualization generation failed: {e}")
			raise
	
	async def generate_subgraph_visualization(
		self,
		tenant_id: str,
		knowledge_graph_id: str,
		entity_ids: List[str],
		max_hops: int = 2,
		config: Optional[VisualizationConfig] = None
	) -> GraphVisualizationData:
		"""
		Generate visualization for a subgraph around specific entities
		
		Args:
			tenant_id: Tenant identifier
			knowledge_graph_id: Knowledge graph identifier
			entity_ids: List of central entity IDs
			max_hops: Maximum hops from central entities
			config: Optional visualization configuration
			
		Returns:
			GraphVisualizationData for the subgraph
		"""
		try:
			# Get subgraph entities and relationships
			subgraph_entities = set(entity_ids)
			subgraph_relationships = []
			
			# Expand outward by hops
			for hop in range(max_hops):
				current_entities = list(subgraph_entities)
				
				# Get relationships for current entities
				relationships = await self.db_service.list_relationships(
					tenant_id=tenant_id,
					knowledge_graph_id=knowledge_graph_id,
					entity_ids=current_entities,
					limit=10000
				)
				
				# Add new entities and relationships
				for rel in relationships:
					subgraph_relationships.append(rel)
					subgraph_entities.add(rel.source_entity_id)
					subgraph_entities.add(rel.target_entity_id)
			
			# Get entity details
			entities = await self.db_service.get_entities_by_ids(
				tenant_id=tenant_id,
				knowledge_graph_id=knowledge_graph_id,
				entity_ids=list(subgraph_entities)
			)
			
			# Create filtered data structure
			filtered_data = {
				"entities": entities,
				"relationships": subgraph_relationships
			}
			
			# Generate visualization with smaller config
			viz_config = config or VisualizationConfig()
			viz_config.max_nodes = min(viz_config.max_nodes, 500)
			viz_config.max_edges = min(viz_config.max_edges, 1000)
			
			# Use the same generation process as full graph
			layout_data = await self._generate_layout(filtered_data, viz_config)
			styling_data = await self._generate_styling(filtered_data, viz_config)
			
			nodes = await self._build_visualization_nodes(
				entities, layout_data, styling_data, viz_config
			)
			
			edges = await self._build_visualization_edges(
				subgraph_relationships, layout_data, styling_data, viz_config
			)
			
			# Highlight central entities
			for node in nodes:
				if node["id"] in entity_ids:
					node["style"]["border_color"] = "#e74c3c"
					node["style"]["border_width"] = 4.0
					node["central"] = True
			
			statistics = await self._calculate_visualization_statistics(nodes, edges, [])
			
			metadata = {
				"graph_id": knowledge_graph_id,
				"tenant_id": tenant_id,
				"subgraph_type": "entity_centered",
				"central_entities": entity_ids,
				"max_hops": max_hops,
				"generated_at": datetime.utcnow().isoformat(),
				"total_nodes": len(nodes),
				"total_edges": len(edges)
			}
			
			return GraphVisualizationData(
				nodes=nodes,
				edges=edges,
				clusters=[],
				statistics=statistics,
				metadata=metadata,
				layout_data=layout_data,
				styling=styling_data
			)
			
		except Exception as e:
			logger.error(f"Subgraph visualization generation failed: {e}")
			raise
	
	async def generate_temporal_visualization(
		self,
		tenant_id: str,
		knowledge_graph_id: str,
		time_range: Tuple[datetime, datetime],
		config: Optional[VisualizationConfig] = None
	) -> GraphVisualizationData:
		"""
		Generate temporal visualization showing graph evolution over time
		
		Args:
			tenant_id: Tenant identifier
			knowledge_graph_id: Knowledge graph identifier
			time_range: Start and end datetime for temporal range
			config: Optional visualization configuration
			
		Returns:
			GraphVisualizationData with temporal layout
		"""
		try:
			viz_config = config or VisualizationConfig()
			viz_config.mode = VisualizationMode.TEMPORAL
			
			# Get entities and relationships within time range
			entities = await self.db_service.list_entities(
				tenant_id=tenant_id,
				knowledge_graph_id=knowledge_graph_id,
				created_after=time_range[0],
				created_before=time_range[1],
				limit=viz_config.max_nodes
			)
			
			relationships = await self.db_service.list_relationships(
				tenant_id=tenant_id,
				knowledge_graph_id=knowledge_graph_id,
				created_after=time_range[0],
				created_before=time_range[1],
				limit=viz_config.max_edges
			)
			
			# Generate temporal layout (timeline-based)
			layout_data = await self._generate_temporal_layout(
				entities, relationships, time_range, viz_config
			)
			
			# Apply temporal styling
			styling_data = await self._generate_temporal_styling(
				entities, relationships, time_range, viz_config
			)
			
			nodes = await self._build_visualization_nodes(
				entities, layout_data, styling_data, viz_config
			)
			
			edges = await self._build_visualization_edges(
				relationships, layout_data, styling_data, viz_config
			)
			
			statistics = await self._calculate_visualization_statistics(nodes, edges, [])
			statistics["temporal_range"] = {
				"start": time_range[0].isoformat(),
				"end": time_range[1].isoformat(),
				"duration_days": (time_range[1] - time_range[0]).days
			}
			
			metadata = {
				"graph_id": knowledge_graph_id,
				"tenant_id": tenant_id,
				"visualization_type": "temporal",
				"time_range": [time_range[0].isoformat(), time_range[1].isoformat()],
				"generated_at": datetime.utcnow().isoformat(),
				"total_nodes": len(nodes),
				"total_edges": len(edges)
			}
			
			return GraphVisualizationData(
				nodes=nodes,
				edges=edges,
				clusters=[],
				statistics=statistics,
				metadata=metadata,
				layout_data=layout_data,
				styling=styling_data
			)
			
		except Exception as e:
			logger.error(f"Temporal visualization generation failed: {e}")
			raise
	
	async def export_visualization(
		self,
		visualization_data: GraphVisualizationData,
		format: str,
		options: Dict[str, Any] = None
	) -> Dict[str, Any]:
		"""
		Export visualization data to various formats
		
		Args:
			visualization_data: Generated visualization data
			format: Export format (png, svg, json, graphml, etc.)
			options: Format-specific export options
			
		Returns:
			Export result with data and metadata
		"""
		try:
			export_options = options or {}
			
			if format.lower() == "json":
				return await self._export_json(visualization_data, export_options)
			elif format.lower() == "graphml":
				return await self._export_graphml(visualization_data, export_options)
			elif format.lower() == "svg":
				return await self._export_svg(visualization_data, export_options)
			elif format.lower() == "cytoscape":
				return await self._export_cytoscape(visualization_data, export_options)
			elif format.lower() == "d3":
				return await self._export_d3(visualization_data, export_options)
			else:
				raise ValueError(f"Unsupported export format: {format}")
				
		except Exception as e:
			logger.error(f"Visualization export failed: {e}")
			raise
	
	# ========================================================================
	# DATA LOADING AND FILTERING
	# ========================================================================
	
	async def _load_graph_data(
		self,
		tenant_id: str,
		knowledge_graph_id: str,
		config: VisualizationConfig
	) -> Dict[str, Any]:
		"""Load graph data with performance optimizations"""
		
		# Load entities with limit
		entities = await self.db_service.list_entities(
			tenant_id=tenant_id,
			knowledge_graph_id=knowledge_graph_id,
			limit=config.max_nodes * 2,  # Load extra for filtering
			include_embeddings=False  # Skip embeddings for performance
		)
		
		# Load relationships with limit
		relationships = await self.db_service.list_relationships(
			tenant_id=tenant_id,
			knowledge_graph_id=knowledge_graph_id,
			limit=config.max_edges * 2,  # Load extra for filtering
			confidence_threshold=config.confidence_threshold
		)
		
		return {
			"entities": entities,
			"relationships": relationships
		}
	
	async def _apply_filters(
		self,
		graph_data: Dict[str, Any],
		config: VisualizationConfig
	) -> Dict[str, Any]:
		"""Apply visualization filters"""
		
		entities = graph_data["entities"]
		relationships = graph_data["relationships"]
		
		# Filter by confidence threshold
		filtered_entities = [
			e for e in entities 
			if e.confidence_score >= config.confidence_threshold
		]
		
		filtered_relationships = [
			r for r in relationships
			if r.confidence_score >= config.confidence_threshold
		]
		
		# Limit to max counts
		filtered_entities = filtered_entities[:config.max_nodes]
		filtered_relationships = filtered_relationships[:config.max_edges]
		
		# Ensure relationship endpoints exist in filtered entities
		entity_ids = {e.canonical_entity_id for e in filtered_entities}
		final_relationships = [
			r for r in filtered_relationships
			if r.source_entity_id in entity_ids and r.target_entity_id in entity_ids
		]
		
		return {
			"entities": filtered_entities,
			"relationships": final_relationships
		}
	
	# ========================================================================
	# LAYOUT GENERATION
	# ========================================================================
	
	async def _generate_layout(
		self,
		graph_data: Dict[str, Any],
		config: VisualizationConfig
	) -> Dict[str, Any]:
		"""Generate node positions using specified layout algorithm"""
		
		if config.layout_algorithm == LayoutAlgorithm.SPRING:
			return await self._generate_spring_layout(graph_data, config)
		elif config.layout_algorithm == LayoutAlgorithm.CIRCULAR:
			return await self._generate_circular_layout(graph_data, config)
		elif config.layout_algorithm == LayoutAlgorithm.HIERARCHICAL:
			return await self._generate_hierarchical_layout(graph_data, config)
		else:
			# Default to force-directed layout
			return await self._generate_force_directed_layout(graph_data, config)
	
	async def _generate_spring_layout(
		self,
		graph_data: Dict[str, Any],
		config: VisualizationConfig
	) -> Dict[str, Any]:
		"""Generate spring-force layout"""
		
		entities = graph_data["entities"]
		relationships = graph_data["relationships"]
		
		# Build adjacency information
		adjacency = defaultdict(list)
		for rel in relationships:
			adjacency[rel.source_entity_id].append(rel.target_entity_id)
			adjacency[rel.target_entity_id].append(rel.source_entity_id)
		
		# Initialize random positions
		positions = {}
		for entity in entities:
			positions[entity.canonical_entity_id] = {
				"x": np.random.uniform(-config.width/2, config.width/2),
				"y": np.random.uniform(-config.height/2, config.height/2),
				"z": np.random.uniform(-100, 100) if config.enable_3d else 0
			}
		
		# Spring force simulation (simplified)
		for iteration in range(100):  # Simplified iteration count
			forces = defaultdict(lambda: {"x": 0, "y": 0, "z": 0})
			
			# Repulsive forces between all nodes
			for i, entity1 in enumerate(entities):
				for entity2 in entities[i+1:]:
					id1, id2 = entity1.canonical_entity_id, entity2.canonical_entity_id
					pos1, pos2 = positions[id1], positions[id2]
					
					dx = pos1["x"] - pos2["x"]
					dy = pos1["y"] - pos2["y"]
					dz = pos1["z"] - pos2["z"] if config.enable_3d else 0
					
					distance = math.sqrt(dx*dx + dy*dy + dz*dz)
					if distance > 0:
						repulsion = 1000 / (distance * distance)
						forces[id1]["x"] += (dx / distance) * repulsion
						forces[id1]["y"] += (dy / distance) * repulsion
						if config.enable_3d:
							forces[id1]["z"] += (dz / distance) * repulsion
						
						forces[id2]["x"] -= (dx / distance) * repulsion
						forces[id2]["y"] -= (dy / distance) * repulsion
						if config.enable_3d:
							forces[id2]["z"] -= (dz / distance) * repulsion
			
			# Attractive forces for connected nodes
			for rel in relationships:
				id1, id2 = rel.source_entity_id, rel.target_entity_id
				pos1, pos2 = positions[id1], positions[id2]
				
				dx = pos2["x"] - pos1["x"]
				dy = pos2["y"] - pos1["y"]
				dz = pos2["z"] - pos1["z"] if config.enable_3d else 0
				
				distance = math.sqrt(dx*dx + dy*dy + dz*dz)
				if distance > 0:
					attraction = distance * 0.01 * rel.strength
					forces[id1]["x"] += (dx / distance) * attraction
					forces[id1]["y"] += (dy / distance) * attraction
					if config.enable_3d:
						forces[id1]["z"] += (dz / distance) * attraction
					
					forces[id2]["x"] -= (dx / distance) * attraction
					forces[id2]["y"] -= (dy / distance) * attraction
					if config.enable_3d:
						forces[id2]["z"] -= (dz / distance) * attraction
			
			# Apply forces with damping
			for entity_id, force in forces.items():
				positions[entity_id]["x"] += force["x"] * config.damping
				positions[entity_id]["y"] += force["y"] * config.damping
				if config.enable_3d:
					positions[entity_id]["z"] += force["z"] * config.damping
		
		return {
			"algorithm": "spring",
			"positions": positions,
			"iterations": 100
		}
	
	async def _generate_circular_layout(
		self,
		graph_data: Dict[str, Any],
		config: VisualizationConfig
	) -> Dict[str, Any]:
		"""Generate circular layout"""
		
		entities = graph_data["entities"]
		positions = {}
		
		radius = min(config.width, config.height) / 2 * 0.8
		center_x, center_y = 0, 0
		
		for i, entity in enumerate(entities):
			angle = (2 * math.pi * i) / len(entities)
			positions[entity.canonical_entity_id] = {
				"x": center_x + radius * math.cos(angle),
				"y": center_y + radius * math.sin(angle),
				"z": 0
			}
		
		return {
			"algorithm": "circular",
			"positions": positions,
			"radius": radius
		}
	
	async def _generate_hierarchical_layout(
		self,
		graph_data: Dict[str, Any],
		config: VisualizationConfig
	) -> Dict[str, Any]:
		"""Generate hierarchical layout"""
		
		entities = graph_data["entities"]
		relationships = graph_data["relationships"]
		
		# Build hierarchy based on entity types or relationships
		levels = defaultdict(list)
		
		# Simple hierarchy by entity type
		type_order = ["person", "organization", "location", "concept", "event"]
		for entity in entities:
			level = type_order.index(entity.entity_type) if entity.entity_type in type_order else len(type_order)
			levels[level].append(entity)
		
		positions = {}
		level_height = config.height / (len(levels) + 1)
		
		for level, level_entities in levels.items():
			y_pos = -config.height/2 + (level + 1) * level_height
			
			if level_entities:
				entity_width = config.width / (len(level_entities) + 1)
				for i, entity in enumerate(level_entities):
					x_pos = -config.width/2 + (i + 1) * entity_width
					positions[entity.canonical_entity_id] = {
						"x": x_pos,
						"y": y_pos,
						"z": 0
					}
		
		return {
			"algorithm": "hierarchical",
			"positions": positions,
			"levels": len(levels)
		}
	
	async def _generate_force_directed_layout(
		self,
		graph_data: Dict[str, Any],
		config: VisualizationConfig
	) -> Dict[str, Any]:
		"""Generate force-directed layout (similar to spring but different parameters)"""
		# For now, use spring layout with different parameters
		return await self._generate_spring_layout(graph_data, config)
	
	async def _generate_temporal_layout(
		self,
		entities: List[GraphEntity],
		relationships: List[GraphRelationship],
		time_range: Tuple[datetime, datetime],
		config: VisualizationConfig
	) -> Dict[str, Any]:
		"""Generate temporal layout based on creation times"""
		
		positions = {}
		
		start_time = time_range[0]
		end_time = time_range[1]
		time_span = (end_time - start_time).total_seconds()
		
		for entity in entities:
			# Calculate x position based on creation time
			entity_time = entity.created_at or start_time
			time_offset = (entity_time - start_time).total_seconds()
			x_ratio = time_offset / time_span if time_span > 0 else 0
			x_pos = -config.width/2 + x_ratio * config.width
			
			# Y position based on entity type or random
			y_pos = np.random.uniform(-config.height/2, config.height/2)
			
			positions[entity.canonical_entity_id] = {
				"x": x_pos,
				"y": y_pos,
				"z": 0,
				"timestamp": entity_time.isoformat()
			}
		
		return {
			"algorithm": "temporal",
			"positions": positions,
			"time_range": [start_time.isoformat(), end_time.isoformat()]
		}
	
	# ========================================================================
	# STYLING AND APPEARANCE
	# ========================================================================
	
	def _initialize_node_styles(self) -> Dict[str, NodeStyle]:
		"""Initialize node styling templates"""
		return {
			"person": NodeStyle(
				size=15.0,
				color="#3498db",
				shape="circle",
				border_color="#2980b9"
			),
			"organization": NodeStyle(
				size=20.0,
				color="#e74c3c",
				shape="square",
				border_color="#c0392b"
			),
			"location": NodeStyle(
				size=12.0,
				color="#2ecc71",
				shape="triangle",
				border_color="#27ae60"
			),
			"concept": NodeStyle(
				size=10.0,
				color="#f39c12",
				shape="diamond",
				border_color="#e67e22"
			),
			"event": NodeStyle(
				size=18.0,
				color="#9b59b6",
				shape="circle",
				border_color="#8e44ad"
			),
			"default": NodeStyle()
		}
	
	def _initialize_edge_styles(self) -> Dict[str, EdgeStyle]:
		"""Initialize edge styling templates"""
		return {
			"strong": EdgeStyle(width=4.0, color="#2c3e50", opacity=0.9),
			"medium": EdgeStyle(width=2.5, color="#7f8c8d", opacity=0.7),
			"weak": EdgeStyle(width=1.5, color="#bdc3c7", opacity=0.5),
			"default": EdgeStyle()
		}
	
	def _initialize_layout_engines(self) -> Dict[str, Any]:
		"""Initialize layout engine configurations"""
		return {
			"spring": {"k": 1.0, "iterations": 100},
			"circular": {"radius_scale": 0.8},
			"hierarchical": {"level_separation": 100},
			"force_directed": {"charge": -300, "link_distance": 50}
		}
	
	async def _generate_styling(
		self,
		graph_data: Dict[str, Any],
		config: VisualizationConfig
	) -> Dict[str, Any]:
		"""Generate styling information for nodes and edges"""
		
		entities = graph_data["entities"]
		relationships = graph_data["relationships"]
		
		# Calculate size scales based on confidence or degree
		entity_degrees = defaultdict(int)
		for rel in relationships:
			entity_degrees[rel.source_entity_id] += 1
			entity_degrees[rel.target_entity_id] += 1
		
		max_degree = max(entity_degrees.values()) if entity_degrees else 1
		
		node_styles = {}
		for entity in entities:
			base_style = self.node_styles.get(entity.entity_type, self.node_styles["default"])
			
			# Scale size by degree and confidence
			degree = entity_degrees[entity.canonical_entity_id]
			size_factor = (degree / max_degree) * 0.5 + entity.confidence_score * 0.5
			scaled_size = config.node_size_range[0] + size_factor * (config.node_size_range[1] - config.node_size_range[0])
			
			node_styles[entity.canonical_entity_id] = {
				"size": scaled_size,
				"color": base_style.color,
				"border_color": base_style.border_color,
				"border_width": base_style.border_width,
				"shape": base_style.shape,
				"opacity": base_style.opacity,
				"label_color": base_style.label_color,
				"label_size": base_style.label_size
			}
		
		edge_styles = {}
		for rel in relationships:
			# Determine edge style based on strength and confidence
			strength_factor = rel.strength * rel.confidence_score
			
			if strength_factor > 0.8:
				base_style = self.edge_styles["strong"]
			elif strength_factor > 0.5:
				base_style = self.edge_styles["medium"]
			else:
				base_style = self.edge_styles["weak"]
			
			edge_styles[rel.canonical_relationship_id] = {
				"width": base_style.width,
				"color": base_style.color,
				"opacity": base_style.opacity,
				"style": base_style.style,
				"arrow_size": base_style.arrow_size,
				"curvature": base_style.curvature
			}
		
		return {
			"nodes": node_styles,
			"edges": edge_styles
		}
	
	async def _generate_temporal_styling(
		self,
		entities: List[GraphEntity],
		relationships: List[GraphRelationship],
		time_range: Tuple[datetime, datetime],
		config: VisualizationConfig
	) -> Dict[str, Any]:
		"""Generate temporal-specific styling"""
		
		styling = await self._generate_styling(
			{"entities": entities, "relationships": relationships}, 
			config
		)
		
		# Add temporal color gradients
		start_time = time_range[0]
		end_time = time_range[1]
		time_span = (end_time - start_time).total_seconds()
		
		for entity in entities:
			entity_time = entity.created_at or start_time
			time_ratio = (entity_time - start_time).total_seconds() / time_span if time_span > 0 else 0
			
			# Interpolate color based on time
			if entity.canonical_entity_id in styling["nodes"]:
				# Simple color interpolation from blue (old) to red (new)
				blue_intensity = int(255 * (1 - time_ratio))
				red_intensity = int(255 * time_ratio)
				styling["nodes"][entity.canonical_entity_id]["color"] = f"rgb({red_intensity}, 100, {blue_intensity})"
		
		return styling
	
	# ========================================================================
	# VISUALIZATION BUILDING
	# ========================================================================
	
	async def _build_visualization_nodes(
		self,
		entities: List[GraphEntity],
		layout_data: Dict[str, Any],
		styling_data: Dict[str, Any],
		config: VisualizationConfig
	) -> List[Dict[str, Any]]:
		"""Build visualization node objects"""
		
		nodes = []
		positions = layout_data["positions"]
		node_styles = styling_data["nodes"]
		
		for entity in entities:
			entity_id = entity.canonical_entity_id
			
			if entity_id not in positions:
				continue
			
			position = positions[entity_id]
			style = node_styles.get(entity_id, {})
			
			node = {
				"id": entity_id,
				"label": entity.canonical_name,
				"type": entity.entity_type,
				"position": position,
				"style": style,
				"data": {
					"canonical_name": entity.canonical_name,
					"aliases": entity.aliases,
					"properties": entity.properties,
					"confidence_score": entity.confidence_score,
					"created_at": entity.created_at.isoformat() if entity.created_at else None
				},
				"tooltip": self._generate_node_tooltip(entity),
				"selectable": config.enable_selection,
				"draggable": config.enable_drag
			}
			
			nodes.append(node)
		
		return nodes
	
	async def _build_visualization_edges(
		self,
		relationships: List[GraphRelationship],
		layout_data: Dict[str, Any],
		styling_data: Dict[str, Any],
		config: VisualizationConfig
	) -> List[Dict[str, Any]]:
		"""Build visualization edge objects"""
		
		edges = []
		edge_styles = styling_data["edges"]
		
		for rel in relationships:
			rel_id = rel.canonical_relationship_id
			style = edge_styles.get(rel_id, {})
			
			edge = {
				"id": rel_id,
				"source": rel.source_entity_id,
				"target": rel.target_entity_id,
				"label": rel.relationship_type,
				"type": rel.relationship_type,
				"style": style,
				"data": {
					"relationship_type": rel.relationship_type,
					"strength": rel.strength,
					"properties": rel.properties,
					"confidence_score": rel.confidence_score,
					"created_at": rel.created_at.isoformat() if rel.created_at else None
				},
				"tooltip": self._generate_edge_tooltip(rel),
				"selectable": config.enable_selection
			}
			
			edges.append(edge)
		
		return edges
	
	def _generate_node_tooltip(self, entity: GraphEntity) -> str:
		"""Generate tooltip HTML for node"""
		return f"""
		<div class="graph-tooltip">
			<h4>{entity.canonical_name}</h4>
			<p><strong>Type:</strong> {entity.entity_type}</p>
			<p><strong>Confidence:</strong> {entity.confidence_score:.2f}</p>
			<p><strong>Aliases:</strong> {', '.join(entity.aliases[:3])}{'...' if len(entity.aliases) > 3 else ''}</p>
		</div>
		"""
	
	def _generate_edge_tooltip(self, relationship: GraphRelationship) -> str:
		"""Generate tooltip HTML for edge"""
		return f"""
		<div class="graph-tooltip">
			<h4>{relationship.relationship_type}</h4>
			<p><strong>Strength:</strong> {relationship.strength:.2f}</p>
			<p><strong>Confidence:</strong> {relationship.confidence_score:.2f}</p>
		</div>
		"""
	
	# ========================================================================
	# CLUSTERING AND ANALYTICS
	# ========================================================================
	
	async def _generate_clusters(
		self,
		graph_data: Dict[str, Any],
		config: VisualizationConfig
	) -> List[Dict[str, Any]]:
		"""Generate graph clusters"""
		
		entities = graph_data["entities"]
		relationships = graph_data["relationships"]
		
		# Simple clustering by entity type
		clusters = defaultdict(list)
		for entity in entities:
			clusters[entity.entity_type].append(entity.canonical_entity_id)
		
		cluster_list = []
		colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22"]
		
		for i, (cluster_type, entity_ids) in enumerate(clusters.items()):
			if len(entity_ids) >= 2:  # Minimum cluster size
				cluster_list.append({
					"id": f"cluster_{i}",
					"label": cluster_type.title(),
					"type": cluster_type,
					"entity_ids": entity_ids,
					"size": len(entity_ids),
					"color": colors[i % len(colors)],
					"opacity": 0.2
				})
		
		return cluster_list
	
	async def _calculate_visualization_statistics(
		self,
		nodes: List[Dict[str, Any]],
		edges: List[Dict[str, Any]],
		clusters: List[Dict[str, Any]]
	) -> Dict[str, Any]:
		"""Calculate visualization statistics"""
		
		# Node statistics
		node_types = defaultdict(int)
		total_confidence = 0
		
		for node in nodes:
			node_types[node["type"]] += 1
			total_confidence += node["data"]["confidence_score"]
		
		avg_confidence = total_confidence / len(nodes) if nodes else 0
		
		# Edge statistics
		edge_types = defaultdict(int)
		total_strength = 0
		
		for edge in edges:
			edge_types[edge["type"]] += 1
			total_strength += edge["data"]["strength"]
		
		avg_strength = total_strength / len(edges) if edges else 0
		
		# Graph metrics
		degree_distribution = defaultdict(int)
		node_degrees = defaultdict(int)
		
		for edge in edges:
			node_degrees[edge["source"]] += 1
			node_degrees[edge["target"]] += 1
		
		for degree in node_degrees.values():
			degree_distribution[degree] += 1
		
		return {
			"node_count": len(nodes),
			"edge_count": len(edges),
			"cluster_count": len(clusters),
			"node_types": dict(node_types),
			"edge_types": dict(edge_types),
			"avg_node_confidence": avg_confidence,
			"avg_edge_strength": avg_strength,
			"degree_distribution": dict(degree_distribution),
			"density": (2 * len(edges)) / (len(nodes) * (len(nodes) - 1)) if len(nodes) > 1 else 0,
			"avg_degree": sum(node_degrees.values()) / len(nodes) if nodes else 0
		}
	
	# ========================================================================
	# EXPORT FUNCTIONS
	# ========================================================================
	
	async def _export_json(
		self,
		visualization_data: GraphVisualizationData,
		options: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Export as JSON format"""
		
		return {
			"format": "json",
			"data": {
				"nodes": visualization_data.nodes,
				"edges": visualization_data.edges,
				"clusters": visualization_data.clusters,
				"statistics": visualization_data.statistics,
				"metadata": visualization_data.metadata,
				"layout": visualization_data.layout_data,
				"styling": visualization_data.styling
			},
			"export_timestamp": datetime.utcnow().isoformat()
		}
	
	async def _export_graphml(
		self,
		visualization_data: GraphVisualizationData,
		options: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Export as GraphML format"""
		
		# Build GraphML XML structure
		graphml_data = f"""<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns
         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">
  
  <key id="name" for="node" attr.name="name" attr.type="string"/>
  <key id="type" for="node" attr.name="type" attr.type="string"/>
  <key id="confidence" for="node" attr.name="confidence" attr.type="double"/>
  <key id="x" for="node" attr.name="x" attr.type="double"/>
  <key id="y" for="node" attr.name="y" attr.type="double"/>
  
  <key id="relationship_type" for="edge" attr.name="relationship_type" attr.type="string"/>
  <key id="strength" for="edge" attr.name="strength" attr.type="double"/>
  
  <graph id="GraphRAG" edgedefault="directed">
"""
		
		# Add nodes
		for node in visualization_data.nodes:
			pos = node["position"]
			graphml_data += f"""    <node id="{node['id']}">
      <data key="name">{node['label']}</data>
      <data key="type">{node['type']}</data>
      <data key="confidence">{node['data']['confidence_score']}</data>
      <data key="x">{pos['x']}</data>
      <data key="y">{pos['y']}</data>
    </node>
"""
		
		# Add edges
		for edge in visualization_data.edges:
			graphml_data += f"""    <edge source="{edge['source']}" target="{edge['target']}">
      <data key="relationship_type">{edge['type']}</data>
      <data key="strength">{edge['data']['strength']}</data>
    </edge>
"""
		
		graphml_data += """  </graph>
</graphml>"""
		
		return {
			"format": "graphml",
			"data": graphml_data,
			"export_timestamp": datetime.utcnow().isoformat()
		}
	
	async def _export_svg(
		self,
		visualization_data: GraphVisualizationData,
		options: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Export as SVG format"""
		
		width = options.get("width", 1200)
		height = options.get("height", 800)
		
		svg_data = f"""<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" 
     refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#666" />
    </marker>
  </defs>
  
  <rect width="100%" height="100%" fill="white"/>
  
  <!-- Edges -->
"""
		
		# Add edges
		for edge in visualization_data.edges:
			source_node = next((n for n in visualization_data.nodes if n["id"] == edge["source"]), None)
			target_node = next((n for n in visualization_data.nodes if n["id"] == edge["target"]), None)
			
			if source_node and target_node:
				x1 = source_node["position"]["x"] + width/2
				y1 = source_node["position"]["y"] + height/2
				x2 = target_node["position"]["x"] + width/2
				y2 = target_node["position"]["y"] + height/2
				
				svg_data += f"""  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"
        stroke="{edge['style']['color']}" stroke-width="{edge['style']['width']}"
        marker-end="url(#arrowhead)"/>
"""
		
		svg_data += "\n  <!-- Nodes -->\n"
		
		# Add nodes
		for node in visualization_data.nodes:
			x = node["position"]["x"] + width/2
			y = node["position"]["y"] + height/2
			size = node["style"]["size"]
			color = node["style"]["color"]
			
			svg_data += f"""  <circle cx="{x}" cy="{y}" r="{size/2}"
        fill="{color}" stroke="{node['style']['border_color']}" 
        stroke-width="{node['style']['border_width']}"/>
  <text x="{x}" y="{y+5}" text-anchor="middle" 
        font-size="{node['style']['label_size']}" fill="{node['style']['label_color']}">
    {node['label'][:15]}{'...' if len(node['label']) > 15 else ''}
  </text>
"""
		
		svg_data += "</svg>"
		
		return {
			"format": "svg",
			"data": svg_data,
			"export_timestamp": datetime.utcnow().isoformat()
		}
	
	async def _export_cytoscape(
		self,
		visualization_data: GraphVisualizationData,
		options: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Export as Cytoscape.js format"""
		
		cytoscape_data = {
			"elements": {
				"nodes": [
					{
						"data": {
							"id": node["id"],
							"label": node["label"],
							"type": node["type"],
							**node["data"]
						},
						"position": {
							"x": node["position"]["x"],
							"y": node["position"]["y"]
						},
						"style": node["style"]
					}
					for node in visualization_data.nodes
				],
				"edges": [
					{
						"data": {
							"id": edge["id"],
							"source": edge["source"],
							"target": edge["target"],
							"label": edge["label"],
							"type": edge["type"],
							**edge["data"]
						},
						"style": edge["style"]
					}
					for edge in visualization_data.edges
				]
			},
			"layout": {
				"name": "preset"  # Use preset positions
			},
			"style": [
				{
					"selector": "node",
					"style": {
						"background-color": "data(color)",
						"label": "data(label)",
						"width": "data(size)",
						"height": "data(size)"
					}
				},
				{
					"selector": "edge",
					"style": {
						"width": "data(width)",
						"line-color": "data(color)",
						"target-arrow-color": "data(color)",
						"target-arrow-shape": "triangle",
						"curve-style": "bezier"
					}
				}
			]
		}
		
		return {
			"format": "cytoscape",
			"data": cytoscape_data,
			"export_timestamp": datetime.utcnow().isoformat()
		}
	
	async def _export_d3(
		self,
		visualization_data: GraphVisualizationData,
		options: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Export as D3.js format"""
		
		d3_data = {
			"nodes": [
				{
					"id": node["id"],
					"name": node["label"],
					"group": node["type"],
					"size": node["style"]["size"],
					"color": node["style"]["color"],
					"x": node["position"]["x"],
					"y": node["position"]["y"],
					**node["data"]
				}
				for node in visualization_data.nodes
			],
			"links": [
				{
					"source": edge["source"],
					"target": edge["target"],
					"type": edge["type"],
					"value": edge["data"]["strength"],
					"color": edge["style"]["color"],
					"width": edge["style"]["width"]
				}
				for edge in visualization_data.edges
			],
			"metadata": visualization_data.metadata
		}
		
		return {
			"format": "d3",
			"data": d3_data,
			"export_timestamp": datetime.utcnow().isoformat()
		}


__all__ = [
	'GraphVisualizationEngine',
	'VisualizationConfig',
	'VisualizationMode',
	'LayoutAlgorithm',
	'GraphVisualizationData',
	'NodeStyle',
	'EdgeStyle'
]