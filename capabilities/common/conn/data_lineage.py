"""
APG Connection Management - Data Lineage & Visualization

Advanced data lineage tracking, impact analysis, and interactive visualization
for complete data flow understanding across the organization.

Author: APG Platform Team
Version: 1.0.0
License: Proprietary - Datacraft © 2025
"""

import asyncio
import json
import dataclasses
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

@dataclasses.dataclass
class DataLineageNode:
	"""Represents a node in the data lineage graph."""

	id: str = dataclasses.field(default_factory=lambda: str(uuid4()))
	name: str = ""
	type: str = "entity"  # entity, transformation, flow, connection
	source_type: str = "unknown"  # database, api, file, stream

	# Hierarchy Information
	schema: Optional[str] = None
	table: Optional[str] = None
	field: Optional[str] = None

	# Metadata
	description: Optional[str] = None
	owner: Optional[str] = None
	tags: List[str] = dataclasses.field(default_factory=list)

	# Lineage Properties
	upstream_nodes: Set[str] = dataclasses.field(default_factory=set)
	downstream_nodes: Set[str] = dataclasses.field(default_factory=set)

	# Data Properties
	data_type: Optional[str] = None
	sensitive: bool = False
	pii: bool = False

	# Execution Context
	connection_id: Optional[str] = None
	flow_id: Optional[str] = None
	last_updated: datetime = dataclasses.field(default_factory=lambda: datetime.now(timezone.utc))

	# Quality Metrics
	data_quality_score: float = 1.0
	freshness_score: float = 1.0
	completeness_score: float = 1.0

@dataclasses.dataclass
class DataLineageEdge:
	"""Represents a relationship between lineage nodes."""

	id: str = dataclasses.field(default_factory=lambda: str(uuid4()))
	source_node_id: str = ""
	target_node_id: str = ""

	# Relationship Properties
	relationship_type: str = "derives_from"  # derives_from, transforms_to, feeds_into
	transformation_type: Optional[str] = None  # map, filter, aggregate, join
	transformation_logic: Optional[str] = None

	# Execution Context
	flow_id: Optional[str] = None
	connection_id: Optional[str] = None

	# Data Flow Properties
	volume_estimate: Optional[int] = None  # Records per day
	frequency: Optional[str] = None  # hourly, daily, real-time
	latency: Optional[float] = None  # Processing delay in seconds

	# Quality Impact
	quality_impact: float = 0.0  # -1 to 1, negative means quality degradation

	# Metadata
	created_at: datetime = dataclasses.field(default_factory=lambda: datetime.now(timezone.utc))
	last_execution: Optional[datetime] = None

@dataclasses.dataclass
class DataLineageGraph:
	"""Complete data lineage graph with analysis capabilities."""

	nodes: Dict[str, DataLineageNode] = dataclasses.field(default_factory=dict)
	edges: Dict[str, DataLineageEdge] = dataclasses.field(default_factory=dict)

	# Graph Metadata
	last_build: datetime = dataclasses.field(default_factory=lambda: datetime.now(timezone.utc))
	version: int = 1

	def add_node(self, node: DataLineageNode) -> str:
		"""Add a node to the lineage graph."""
		self.nodes[node.id] = node
		return node.id

	def add_edge(self, edge: DataLineageEdge) -> str:
		"""Add an edge to the lineage graph."""
		self.edges[edge.id] = edge

		# Update node relationships
		if edge.source_node_id in self.nodes:
			self.nodes[edge.source_node_id].downstream_nodes.add(edge.target_node_id)
		if edge.target_node_id in self.nodes:
			self.nodes[edge.target_node_id].upstream_nodes.add(edge.source_node_id)

		return edge.id

	def get_upstream_lineage(self, node_id: str, max_depth: int = 10) -> Dict[str, Any]:
		"""Get upstream lineage for a node."""
		if node_id not in self.nodes:
			return {"nodes": [], "edges": []}

		visited = set()
		lineage_nodes = {}
		lineage_edges = {}

		def traverse_upstream(current_id: str, depth: int):
			if depth >= max_depth or current_id in visited:
				return

			visited.add(current_id)
			if current_id in self.nodes:
				lineage_nodes[current_id] = self.nodes[current_id]

			# Find upstream edges
			for edge_id, edge in self.edges.items():
				if edge.target_node_id == current_id:
					lineage_edges[edge_id] = edge
					traverse_upstream(edge.source_node_id, depth + 1)

		traverse_upstream(node_id, 0)

		return {
			"nodes": list(lineage_nodes.values()),
			"edges": list(lineage_edges.values()),
			"depth": len(visited) - 1
		}

	def get_downstream_lineage(self, node_id: str, max_depth: int = 10) -> Dict[str, Any]:
		"""Get downstream lineage for a node."""
		if node_id not in self.nodes:
			return {"nodes": [], "edges": []}

		visited = set()
		lineage_nodes = {}
		lineage_edges = {}

		def traverse_downstream(current_id: str, depth: int):
			if depth >= max_depth or current_id in visited:
				return

			visited.add(current_id)
			if current_id in self.nodes:
				lineage_nodes[current_id] = self.nodes[current_id]

			# Find downstream edges
			for edge_id, edge in self.edges.items():
				if edge.source_node_id == current_id:
					lineage_edges[edge_id] = edge
					traverse_downstream(edge.target_node_id, depth + 1)

		traverse_downstream(node_id, 0)

		return {
			"nodes": list(lineage_nodes.values()),
			"edges": list(lineage_edges.values()),
			"depth": len(visited) - 1
		}

	def analyze_impact(self, node_id: str) -> Dict[str, Any]:
		"""Analyze impact of changes to a specific node."""
		downstream = self.get_downstream_lineage(node_id)

		impact_analysis = {
			"affected_nodes": len(downstream["nodes"]),
			"affected_flows": len(set(edge.flow_id for edge in downstream["edges"] if edge.flow_id)),
			"affected_connections": len(set(edge.connection_id for edge in downstream["edges"] if edge.connection_id)),
			"risk_level": "low",
			"recommendations": []
		}

		# Calculate risk level
		if impact_analysis["affected_nodes"] > 20:
			impact_analysis["risk_level"] = "high"
			impact_analysis["recommendations"].append("Consider phased rollout for changes")
		elif impact_analysis["affected_nodes"] > 10:
			impact_analysis["risk_level"] = "medium"
			impact_analysis["recommendations"].append("Test changes in staging environment")

		# Check for sensitive data
		sensitive_nodes = [node for node in downstream["nodes"] if node.sensitive or node.pii]
		if sensitive_nodes:
			impact_analysis["risk_level"] = "high"
			impact_analysis["recommendations"].append("Review data privacy implications")
			impact_analysis["sensitive_data_affected"] = len(sensitive_nodes)

		return impact_analysis

	def find_root_sources(self) -> List[DataLineageNode]:
		"""Find root source nodes (nodes with no upstream dependencies)."""
		return [node for node in self.nodes.values() if not node.upstream_nodes]

	def find_leaf_destinations(self) -> List[DataLineageNode]:
		"""Find leaf destination nodes (nodes with no downstream dependencies)."""
		return [node for node in self.nodes.values() if not node.downstream_nodes]

	def detect_cycles(self) -> List[List[str]]:
		"""Detect cycles in the lineage graph."""
		cycles = []
		visited = set()
		rec_stack = set()

		def has_cycle(node_id: str, path: List[str]) -> bool:
			if node_id in rec_stack:
				# Found cycle, extract it
				cycle_start = path.index(node_id)
				cycle = path[cycle_start:] + [node_id]
				cycles.append(cycle)
				return True

			if node_id in visited:
				return False

			visited.add(node_id)
			rec_stack.add(node_id)
			path.append(node_id)

			# Check downstream nodes
			for edge in self.edges.values():
				if edge.source_node_id == node_id:
					has_cycle(edge.target_node_id, path.copy())

			rec_stack.remove(node_id)
			return False

		for node_id in self.nodes:
			if node_id not in visited:
				has_cycle(node_id, [])

		return cycles

@dataclasses.dataclass
class DataLineageTracker:
	"""
	Main data lineage tracking system that builds and maintains
	the complete organizational data lineage graph.
	"""

	lineage_graph: DataLineageGraph = dataclasses.field(default_factory=DataLineageGraph)

	# Tracking State
	active_flows: Dict[str, Dict[str, Any]] = dataclasses.field(default_factory=dict)
	connection_schemas: Dict[str, Dict[str, Any]] = dataclasses.field(default_factory=dict)

	def _log_lineage_operation(self, operation: str) -> None:
		"""Log lineage operations following APG patterns."""
		print(f"Data lineage tracker: {operation}")

	async def track_connection(
		self,
		connection_id: str,
		connection_name: str,
		connection_type: str,
		schema_info: Dict[str, Any]
	) -> List[str]:
		"""Track a data connection and create lineage nodes."""
		self._log_lineage_operation(f"Tracking connection: {connection_name}")

		node_ids = []

		# Store connection schema
		self.connection_schemas[connection_id] = schema_info

		# Create connection node
		connection_node = DataLineageNode(
			name=connection_name,
			type="connection",
			source_type=connection_type,
			connection_id=connection_id,
			description=f"Data connection: {connection_name}"
		)
		self.lineage_graph.add_node(connection_node)
		node_ids.append(connection_node.id)

		# Create nodes for each table/stream in the connection
		for entity_name, entity_info in schema_info.items():
			entity_node = DataLineageNode(
				name=f"{connection_name}.{entity_name}",
				type="entity",
				source_type=connection_type,
				schema=connection_name,
				table=entity_name,
				connection_id=connection_id,
				description=entity_info.get("description", f"Entity: {entity_name}")
			)
			self.lineage_graph.add_node(entity_node)
			node_ids.append(entity_node.id)

			# Create edge from connection to entity
			connection_edge = DataLineageEdge(
				source_node_id=connection_node.id,
				target_node_id=entity_node.id,
				relationship_type="contains",
				connection_id=connection_id
			)
			self.lineage_graph.add_edge(connection_edge)

			# Create field-level nodes if schema is available
			if "fields" in entity_info:
				for field_name, field_info in entity_info["fields"].items():
					field_node = DataLineageNode(
						name=f"{connection_name}.{entity_name}.{field_name}",
						type="field",
						source_type=connection_type,
						schema=connection_name,
						table=entity_name,
						field=field_name,
						connection_id=connection_id,
						data_type=field_info.get("type"),
						sensitive=field_info.get("sensitive", False),
						pii=field_info.get("pii", False),
						description=field_info.get("description", f"Field: {field_name}")
					)
					self.lineage_graph.add_node(field_node)
					node_ids.append(field_node.id)

					# Create edge from entity to field
					field_edge = DataLineageEdge(
						source_node_id=entity_node.id,
						target_node_id=field_node.id,
						relationship_type="contains",
						connection_id=connection_id
					)
					self.lineage_graph.add_edge(field_edge)

		return node_ids

	async def track_flow_execution(
		self,
		flow_id: str,
		flow_name: str,
		source_connection_id: str,
		target_connection_id: str,
		transformations: List[Dict[str, Any]],
		field_mappings: Dict[str, str]
	) -> None:
		"""Track data flow execution and create lineage relationships."""
		self._log_lineage_operation(f"Tracking flow execution: {flow_name}")

		self.active_flows[flow_id] = {
			"name": flow_name,
			"source_connection_id": source_connection_id,
			"target_connection_id": target_connection_id,
			"transformations": transformations,
			"field_mappings": field_mappings,
			"last_execution": datetime.now(timezone.utc)
		}

		# Create flow node
		flow_node = DataLineageNode(
			name=flow_name,
			type="flow",
			source_type="flow",
			flow_id=flow_id,
			description=f"Data flow: {flow_name}"
		)
		self.lineage_graph.add_node(flow_node)

		# Create transformation nodes for complex transformations
		for i, transformation in enumerate(transformations):
			if transformation.get("type") in ["aggregate", "join", "complex"]:
				transform_node = DataLineageNode(
					name=f"{flow_name}_transform_{i}",
					type="transformation",
					source_type="transformation",
					flow_id=flow_id,
					description=f"Transformation: {transformation.get('description', transformation.get('type'))}"
				)
				self.lineage_graph.add_node(transform_node)

		# Create field-level lineage for mappings
		await self._create_field_lineage(flow_id, source_connection_id, target_connection_id, field_mappings)

	async def _create_field_lineage(
		self,
		flow_id: str,
		source_connection_id: str,
		target_connection_id: str,
		field_mappings: Dict[str, str]
	) -> None:
		"""Create detailed field-level lineage."""
		source_schema = self.connection_schemas.get(source_connection_id, {})
		target_schema = self.connection_schemas.get(target_connection_id, {})

		for source_field, target_field in field_mappings.items():
			# Find source and target field nodes
			source_node_id = None
			target_node_id = None

			for node in self.lineage_graph.nodes.values():
				if (node.connection_id == source_connection_id and
					node.field and source_field in node.name):
					source_node_id = node.id
				elif (node.connection_id == target_connection_id and
					  node.field and target_field in node.name):
					target_node_id = node.id

			# Create lineage edge if both nodes found
			if source_node_id and target_node_id:
				lineage_edge = DataLineageEdge(
					source_node_id=source_node_id,
					target_node_id=target_node_id,
					relationship_type="maps_to",
					transformation_type="field_mapping",
					flow_id=flow_id,
					connection_id=source_connection_id
				)
				self.lineage_graph.add_edge(lineage_edge)

	async def generate_lineage_visualization(
		self,
		node_id: Optional[str] = None,
		visualization_type: str = "full"  # full, upstream, downstream, impact
	) -> Dict[str, Any]:
		"""Generate data for lineage visualization."""
		self._log_lineage_operation(f"Generating {visualization_type} lineage visualization")

		if visualization_type == "full":
			nodes = list(self.lineage_graph.nodes.values())
			edges = list(self.lineage_graph.edges.values())
		elif node_id:
			if visualization_type == "upstream":
				lineage_data = self.lineage_graph.get_upstream_lineage(node_id)
			elif visualization_type == "downstream":
				lineage_data = self.lineage_graph.get_downstream_lineage(node_id)
			elif visualization_type == "impact":
				impact = self.lineage_graph.analyze_impact(node_id)
				lineage_data = self.lineage_graph.get_downstream_lineage(node_id)
				lineage_data["impact_analysis"] = impact
			else:
				lineage_data = {"nodes": [], "edges": []}

			nodes = lineage_data["nodes"]
			edges = lineage_data["edges"]
		else:
			nodes = []
			edges = []

		# Convert to visualization format
		viz_nodes = []
		viz_edges = []

		for node in nodes:
			viz_node = {
				"id": node.id,
				"label": node.name,
				"type": node.type,
				"source_type": node.source_type,
				"group": node.type,
				"metadata": {
					"description": node.description,
					"owner": node.owner,
					"tags": node.tags,
					"sensitive": node.sensitive,
					"pii": node.pii,
					"data_quality_score": node.data_quality_score,
					"freshness_score": node.freshness_score,
					"last_updated": node.last_updated.isoformat()
				}
			}

			# Add styling based on node type
			if node.type == "connection":
				viz_node["color"] = "#3498db"
				viz_node["size"] = 30
			elif node.type == "entity":
				viz_node["color"] = "#2ecc71"
				viz_node["size"] = 25
			elif node.type == "field":
				viz_node["color"] = "#f39c12"
				viz_node["size"] = 15
			elif node.type == "flow":
				viz_node["color"] = "#9b59b6"
				viz_node["size"] = 35
			elif node.type == "transformation":
				viz_node["color"] = "#e74c3c"
				viz_node["size"] = 20

			# Highlight sensitive data
			if node.sensitive or node.pii:
				viz_node["border"] = "#e74c3c"
				viz_node["border_width"] = 3

			viz_nodes.append(viz_node)

		for edge in edges:
			viz_edge = {
				"id": edge.id,
				"source": edge.source_node_id,
				"target": edge.target_node_id,
				"label": edge.relationship_type,
				"type": edge.relationship_type,
				"metadata": {
					"transformation_type": edge.transformation_type,
					"transformation_logic": edge.transformation_logic,
					"volume_estimate": edge.volume_estimate,
					"frequency": edge.frequency,
					"latency": edge.latency,
					"quality_impact": edge.quality_impact
				}
			}

			# Style edge based on type
			if edge.relationship_type == "maps_to":
				viz_edge["color"] = "#3498db"
				viz_edge["width"] = 2
			elif edge.relationship_type == "derives_from":
				viz_edge["color"] = "#2ecc71"
				viz_edge["width"] = 3
			elif edge.relationship_type == "contains":
				viz_edge["color"] = "#95a5a6"
				viz_edge["width"] = 1
				viz_edge["style"] = "dashed"

			viz_edges.append(viz_edge)

		# Add layout information
		layout_config = {
			"algorithm": "force_directed",
			"parameters": {
				"repulsion": 50,
				"attraction": 0.1,
				"damping": 0.9,
				"max_iterations": 1000
			}
		}

		# Generate summary statistics
		summary = {
			"total_nodes": len(viz_nodes),
			"total_edges": len(viz_edges),
			"node_types": {},
			"sensitive_data_nodes": len([n for n in nodes if n.sensitive or n.pii]),
			"data_quality_avg": sum(n.data_quality_score for n in nodes) / len(nodes) if nodes else 0,
			"last_updated": max((n.last_updated for n in nodes), default=datetime.now(timezone.utc)).isoformat()
		}

		for node in nodes:
			summary["node_types"][node.type] = summary["node_types"].get(node.type, 0) + 1

		return {
			"nodes": viz_nodes,
			"edges": viz_edges,
			"layout": layout_config,
			"summary": summary,
			"visualization_type": visualization_type,
			"generated_at": datetime.now(timezone.utc).isoformat()
		}

	async def get_data_catalog(self) -> Dict[str, Any]:
		"""Generate a comprehensive data catalog from lineage information."""
		catalog = {
			"entities": [],
			"connections": [],
			"flows": [],
			"summary": {
				"total_entities": 0,
				"total_fields": 0,
				"total_connections": 0,
				"total_flows": 0,
				"sensitive_fields": 0,
				"pii_fields": 0
			}
		}

		# Group nodes by type
		for node in self.lineage_graph.nodes.values():
			if node.type == "entity":
				entity_info = {
					"id": node.id,
					"name": node.name,
					"schema": node.schema,
					"table": node.table,
					"description": node.description,
					"owner": node.owner,
					"tags": node.tags,
					"connection_id": node.connection_id,
					"data_quality_score": node.data_quality_score,
					"freshness_score": node.freshness_score,
					"last_updated": node.last_updated.isoformat(),
					"fields": []
				}

				# Find associated fields
				for field_node in self.lineage_graph.nodes.values():
					if (field_node.type == "field" and
						field_node.schema == node.schema and
						field_node.table == node.table):
						field_info = {
							"name": field_node.field,
							"data_type": field_node.data_type,
							"sensitive": field_node.sensitive,
							"pii": field_node.pii,
							"description": field_node.description
						}
						entity_info["fields"].append(field_info)

						if field_node.sensitive:
							catalog["summary"]["sensitive_fields"] += 1
						if field_node.pii:
							catalog["summary"]["pii_fields"] += 1

				catalog["entities"].append(entity_info)
				catalog["summary"]["total_entities"] += 1
				catalog["summary"]["total_fields"] += len(entity_info["fields"])

			elif node.type == "connection":
				connection_info = {
					"id": node.id,
					"name": node.name,
					"source_type": node.source_type,
					"description": node.description,
					"connection_id": node.connection_id,
					"last_updated": node.last_updated.isoformat()
				}
				catalog["connections"].append(connection_info)
				catalog["summary"]["total_connections"] += 1

			elif node.type == "flow":
				flow_info = {
					"id": node.id,
					"name": node.name,
					"description": node.description,
					"flow_id": node.flow_id,
					"last_updated": node.last_updated.isoformat()
				}
				catalog["flows"].append(flow_info)
				catalog["summary"]["total_flows"] += 1

		return catalog

	async def search_lineage(
		self,
		query: str,
		search_type: str = "all"  # all, entities, fields, flows
	) -> List[Dict[str, Any]]:
		"""Search through lineage graph for entities, fields, or flows."""
		results = []
		query_lower = query.lower()

		for node in self.lineage_graph.nodes.values():
			match = False

			# Check search type filter
			if search_type != "all":
				if search_type == "entities" and node.type != "entity":
					continue
				elif search_type == "fields" and node.type != "field":
					continue
				elif search_type == "flows" and node.type != "flow":
					continue

			# Check name match
			if query_lower in node.name.lower():
				match = True

			# Check description match
			if node.description and query_lower in node.description.lower():
				match = True

			# Check tags match
			if any(query_lower in tag.lower() for tag in node.tags):
				match = True

			if match:
				result = {
					"id": node.id,
					"name": node.name,
					"type": node.type,
					"description": node.description,
					"tags": node.tags,
					"relevance_score": self._calculate_relevance_score(node, query_lower)
				}
				results.append(result)

		# Sort by relevance score
		results.sort(key=lambda x: x["relevance_score"], reverse=True)

		return results[:50]  # Return top 50 results

	def _calculate_relevance_score(self, node: DataLineageNode, query: str) -> float:
		"""Calculate relevance score for search results."""
		score = 0.0

		# Exact name match gets highest score
		if node.name.lower() == query:
			score += 1.0
		elif query in node.name.lower():
			score += 0.8

		# Description matches
		if node.description and query in node.description.lower():
			score += 0.3

		# Tag matches
		for tag in node.tags:
			if query in tag.lower():
				score += 0.2

		# Boost score for high-quality nodes
		score += node.data_quality_score * 0.1

		return score