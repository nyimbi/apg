"""
Identity Graph Intelligence

Advanced relationship analysis system for fraud detection, insider threat detection,
and access optimization using graph-based machine learning and network analysis.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from typing import Dict, Any, Optional, List, Tuple, Union, Set
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import hashlib
import json
import math
from pathlib import Path
from uuid_extensions import uuid7str
from pydantic import BaseModel, Field, ConfigDict
import numpy as np
from dataclasses import dataclass
from collections import defaultdict, deque
import networkx as nx
from itertools import combinations

from .enhanced_models import IdentityGraphNode, IdentityGraphEdge, IdentityRelationType

class GraphAlgorithm(str, Enum):
	"""Graph analysis algorithms"""
	PAGERANK = "pagerank"
	BETWEENNESS_CENTRALITY = "betweenness_centrality"
	CLUSTERING_COEFFICIENT = "clustering_coefficient"
	COMMUNITY_DETECTION = "community_detection"
	SHORTEST_PATH = "shortest_path"
	ANOMALY_DETECTION = "anomaly_detection"

class RiskIndicatorType(str, Enum):
	"""Types of risk indicators in identity graph"""
	SUSPICIOUS_ASSOCIATION = "suspicious_association"
	UNUSUAL_ACCESS_PATTERN = "unusual_access_pattern"
	VELOCITY_ANOMALY = "velocity_anomaly"
	PRIVILEGE_ESCALATION = "privilege_escalation"
	LATERAL_MOVEMENT = "lateral_movement"
	INSIDER_THREAT = "insider_threat"
	ACCOUNT_TAKEOVER = "account_takeover"
	COORDINATED_ATTACK = "coordinated_attack"

class ThreatLevel(str, Enum):
	"""Threat levels for risk indicators"""
	LOW = "low"
	MEDIUM = "medium"
	HIGH = "high"
	CRITICAL = "critical"

class GraphAnalysisResult(BaseModel):
	"""Result of graph analysis"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Analysis result identifier")
	analysis_type: GraphAlgorithm = Field(..., description="Type of analysis performed")
	target_node_id: Optional[str] = Field(default=None, description="Target node analyzed")
	
	# Results
	scores: Dict[str, float] = Field(default_factory=dict, description="Analysis scores by node")
	rankings: List[str] = Field(default_factory=list, description="Node rankings")
	clusters: Dict[str, List[str]] = Field(default_factory=dict, description="Detected clusters")
	paths: List[List[str]] = Field(default_factory=list, description="Important paths")
	
	# Analysis metadata
	node_count: int = Field(..., description="Number of nodes analyzed")
	edge_count: int = Field(..., description="Number of edges analyzed")
	execution_time_ms: float = Field(..., description="Analysis execution time")
	
	# Quality metrics
	confidence: float = Field(..., description="Confidence in results", ge=0.0, le=1.0)
	coverage: float = Field(..., description="Graph coverage", ge=0.0, le=1.0)
	
	analyzed_at: datetime = Field(default_factory=datetime.utcnow, description="Analysis timestamp")

class RiskIndicator(BaseModel):
	"""Identity graph risk indicator"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Risk indicator identifier")
	indicator_type: RiskIndicatorType = Field(..., description="Type of risk indicator")
	threat_level: ThreatLevel = Field(..., description="Assessed threat level")
	
	# Affected entities
	primary_node_id: str = Field(..., description="Primary node involved")
	related_node_ids: List[str] = Field(default_factory=list, description="Related nodes")
	affected_edges: List[str] = Field(default_factory=list, description="Affected edges")
	
	# Risk assessment
	risk_score: float = Field(..., description="Risk score", ge=0.0, le=1.0)
	confidence: float = Field(..., description="Confidence in assessment", ge=0.0, le=1.0)
	evidence: List[str] = Field(default_factory=list, description="Evidence supporting indicator")
	
	# Context
	detection_context: Dict[str, Any] = Field(default_factory=dict, description="Detection context")
	related_indicators: List[str] = Field(default_factory=list, description="Related risk indicators")
	
	# Timing and tracking
	first_detected: datetime = Field(default_factory=datetime.utcnow, description="First detection")
	last_observed: datetime = Field(default_factory=datetime.utcnow, description="Last observation")
	observation_count: int = Field(default=1, description="Number of times observed")
	
	# Response tracking
	acknowledged: bool = Field(default=False, description="Risk indicator acknowledged")
	mitigated: bool = Field(default=False, description="Risk indicator mitigated")
	false_positive: bool = Field(default=False, description="Marked as false positive")

class IdentityCluster(BaseModel):
	"""Cluster of related identities"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Cluster identifier")
	name: Optional[str] = Field(default=None, description="Cluster name/label")
	
	# Cluster composition
	member_nodes: List[str] = Field(..., description="Node IDs in cluster")
	core_nodes: List[str] = Field(default_factory=list, description="Core cluster members")
	peripheral_nodes: List[str] = Field(default_factory=list, description="Peripheral members")
	
	# Cluster properties
	cluster_type: str = Field(..., description="Type of cluster (family, team, location, etc.)")
	cohesion_score: float = Field(..., description="Internal cluster cohesion", ge=0.0, le=1.0)
	separation_score: float = Field(..., description="Separation from other clusters", ge=0.0, le=1.0)
	
	# Risk assessment
	collective_risk_score: float = Field(..., description="Cluster risk score", ge=0.0, le=1.0)
	risk_indicators: List[str] = Field(default_factory=list, description="Risk indicator IDs")
	
	# Temporal properties
	formation_date: datetime = Field(default_factory=datetime.utcnow, description="When cluster was detected")
	stability_score: float = Field(..., description="Cluster stability over time", ge=0.0, le=1.0)
	activity_level: float = Field(..., description="Cluster activity level", ge=0.0, le=1.0)

@dataclass
class GraphMetrics:
	"""Graph-wide metrics and statistics"""
	total_nodes: int
	total_edges: int
	avg_degree: float
	max_degree: int
	clustering_coefficient: float
	diameter: int
	connected_components: int
	density: float
	assortativity: float
	modularity: float

class IdentityGraphEngine:
	"""Main identity graph intelligence engine"""
	
	def __init__(self):
		# Graph storage
		self.graph = nx.MultiDiGraph()  # Support multiple edge types between nodes
		self._nodes: Dict[str, IdentityGraphNode] = {}
		self._edges: Dict[str, IdentityGraphEdge] = {}
		
		# Analysis results cache
		self._analysis_cache: Dict[str, GraphAnalysisResult] = {}
		self._risk_indicators: Dict[str, RiskIndicator] = {}
		self._clusters: Dict[str, IdentityCluster] = {}
		
		# Learning and patterns
		self._behavior_patterns: Dict[str, Dict[str, Any]] = {}
		self._anomaly_thresholds: Dict[str, float] = {
			'degree_centrality': 0.95,
			'betweenness_centrality': 0.9,
			'velocity_threshold': 0.8,
			'clustering_deviation': 0.85
		}
		
		# Performance tracking
		self._operation_times: Dict[str, List[float]] = {}
		self._cache_hit_rate = 0.0
	
	def _log_info(self, message: str, **kwargs):
		"""Log information message"""
		print(f"[IdentityGraph INFO] {message} {kwargs if kwargs else ''}")
	
	def _log_warning(self, message: str, **kwargs):
		"""Log warning message"""
		print(f"[IdentityGraph WARNING] {message} {kwargs if kwargs else ''}")
	
	def _log_error(self, message: str, **kwargs):
		"""Log error message"""
		print(f"[IdentityGraph ERROR] {message} {kwargs if kwargs else ''}")
	
	async def _time_operation(self, operation_name: str, operation_func):
		"""Time operations for performance monitoring"""
		start_time = asyncio.get_event_loop().time()
		result = await operation_func()
		end_time = asyncio.get_event_loop().time()
		
		duration_ms = (end_time - start_time) * 1000
		
		if operation_name not in self._operation_times:
			self._operation_times[operation_name] = []
		self._operation_times[operation_name].append(duration_ms)
		
		# Keep only last 100 measurements
		self._operation_times[operation_name] = self._operation_times[operation_name][-100:]
		
		return result, duration_ms
	
	async def add_identity_node(self, node: IdentityGraphNode) -> str:
		"""Add identity node to graph"""
		assert node.entity_id, "Entity ID is required"
		
		# Store node
		self._nodes[node.id] = node
		
		# Add to NetworkX graph
		self.graph.add_node(node.id, **{
			'entity_type': node.entity_type,
			'entity_id': node.entity_id,
			'risk_score': node.risk_score,
			'reputation_score': node.reputation_score,
			'attributes': node.attributes,
			'labels': list(node.labels)
		})
		
		self._log_info("Identity node added",
					   node_id=node.id,
					   entity_type=node.entity_type,
					   entity_id=node.entity_id)
		
		# Clear analysis cache
		self._analysis_cache.clear()
		
		return node.id
	
	async def add_relationship(self, edge: IdentityGraphEdge) -> str:
		"""Add relationship between identity nodes"""
		assert edge.source_node_id in self._nodes, "Source node must exist"
		assert edge.target_node_id in self._nodes, "Target node must exist"
		
		# Store edge
		self._edges[edge.id] = edge
		
		# Add to NetworkX graph
		self.graph.add_edge(
			edge.source_node_id,
			edge.target_node_id,
			key=edge.id,
			relationship_type=edge.relationship_type.value,
			strength=edge.strength,
			confidence=edge.confidence,
			first_observed=edge.first_observed,
			last_observed=edge.last_observed,
			observation_count=edge.observation_count
		)
		
		self._log_info("Relationship added",
					   edge_id=edge.id,
					   source=edge.source_node_id,
					   target=edge.target_node_id,
					   relationship_type=edge.relationship_type.value)
		
		# Clear analysis cache
		self._analysis_cache.clear()
		
		# Update relationship observation
		edge.update_observation()
		
		return edge.id
	
	async def analyze_identity_relationships(self, user_id: str) -> Dict[str, Any]:
		"""Analyze identity relationships for a specific user"""
		# Find user node
		user_nodes = [
			node for node in self._nodes.values()
			if node.entity_type == "user" and user_id in [node.entity_id, node.id]
		]
		
		if not user_nodes:
			self._log_warning("User node not found", user_id=user_id)
			return {"error": "User not found in identity graph"}
		
		user_node = user_nodes[0]
		
		self._log_info("Analyzing identity relationships", user_id=user_id, node_id=user_node.id)
		
		# Perform multiple graph analyses
		analyses = {}
		
		# 1. Centrality analysis
		centrality_result = await self._analyze_centrality(user_node.id)
		analyses["centrality"] = centrality_result
		
		# 2. Community detection
		community_result = await self._detect_communities(user_node.id)
		analyses["community"] = community_result
		
		# 3. Risk propagation analysis
		risk_analysis = await self._analyze_risk_propagation(user_node.id)
		analyses["risk_propagation"] = risk_analysis
		
		# 4. Anomaly detection
		anomaly_analysis = await self._detect_graph_anomalies(user_node.id)
		analyses["anomalies"] = anomaly_analysis
		
		# 5. Path analysis to sensitive resources
		path_analysis = await self._analyze_access_paths(user_node.id)
		analyses["access_paths"] = path_analysis
		
		# Generate risk indicators
		risk_indicators = await self._generate_risk_indicators(user_node.id, analyses)
		
		return {
			"user_id": user_id,
			"node_id": user_node.id,
			"analyses": analyses,
			"risk_indicators": [indicator.model_dump() for indicator in risk_indicators],
			"overall_risk_score": max([indicator.risk_score for indicator in risk_indicators], default=0.0),
			"analysis_timestamp": datetime.utcnow().isoformat()
		}
	
	async def _analyze_centrality(self, node_id: str) -> GraphAnalysisResult:
		"""Analyze node centrality measures"""
		if not self.graph.has_node(node_id):
			raise ValueError("Node not found in graph")
		
		self._log_info("Computing centrality measures", node_id=node_id)
		
		start_time = asyncio.get_event_loop().time()
		
		# Convert to undirected for centrality calculations
		undirected_graph = self.graph.to_undirected()
		
		# Calculate various centrality measures
		degree_centrality = nx.degree_centrality(undirected_graph)
		betweenness_centrality = nx.betweenness_centrality(undirected_graph, k=min(100, len(undirected_graph)))
		closeness_centrality = nx.closeness_centrality(undirected_graph)
		
		# PageRank for directed influence
		pagerank = nx.pagerank(self.graph, max_iter=50)
		
		# Combine scores
		scores = {}
		for node in undirected_graph.nodes():
			combined_score = (
				degree_centrality.get(node, 0.0) * 0.3 +
				betweenness_centrality.get(node, 0.0) * 0.3 +
				closeness_centrality.get(node, 0.0) * 0.2 +
				pagerank.get(node, 0.0) * 0.2
			)
			scores[node] = combined_score
		
		# Rank nodes
		rankings = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
		
		end_time = asyncio.get_event_loop().time()
		execution_time = (end_time - start_time) * 1000
		
		result = GraphAnalysisResult(
			analysis_type=GraphAlgorithm.PAGERANK,
			target_node_id=node_id,
			scores=scores,
			rankings=rankings[:20],  # Top 20
			node_count=len(undirected_graph.nodes()),
			edge_count=len(undirected_graph.edges()),
			execution_time_ms=execution_time,
			confidence=0.9,
			coverage=1.0
		)
		
		# Cache result
		cache_key = f"centrality_{node_id}"
		self._analysis_cache[cache_key] = result
		
		self._log_info("Centrality analysis complete",
					   node_id=node_id,
					   target_score=scores.get(node_id, 0.0),
					   rank_position=rankings.index(node_id) + 1 if node_id in rankings else -1)
		
		return result
	
	async def _detect_communities(self, node_id: str) -> GraphAnalysisResult:
		"""Detect communities/clusters in the graph"""
		if not self.graph.has_node(node_id):
			raise ValueError("Node not found in graph")
		
		self._log_info("Detecting communities", node_id=node_id)
		
		start_time = asyncio.get_event_loop().time()
		
		# Convert to undirected for community detection
		undirected_graph = self.graph.to_undirected()
		
		if len(undirected_graph) < 3:
			# Too few nodes for meaningful community detection
			return GraphAnalysisResult(
				analysis_type=GraphAlgorithm.COMMUNITY_DETECTION,
				target_node_id=node_id,
				clusters={"single_cluster": list(undirected_graph.nodes())},
				node_count=len(undirected_graph.nodes()),
				edge_count=len(undirected_graph.edges()),
				execution_time_ms=0,
				confidence=0.5,
				coverage=1.0
			)
		
		try:
			# Use Louvain method for community detection
			communities = nx.community.louvain_communities(undirected_graph, resolution=1.0)
			
			# Convert to cluster format
			clusters = {}
			user_community = None
			
			for i, community in enumerate(communities):
				cluster_id = f"community_{i}"
				clusters[cluster_id] = list(community)
				
				if node_id in community:
					user_community = cluster_id
			
			# Calculate modularity
			modularity = nx.community.modularity(undirected_graph, communities)
			
		except Exception as e:
			self._log_warning("Community detection failed, using fallback", error=str(e))
			# Fallback: simple connected components
			communities = list(nx.connected_components(undirected_graph))
			clusters = {f"component_{i}": list(comp) for i, comp in enumerate(communities)}
			modularity = 0.0
		
		end_time = asyncio.get_event_loop().time()
		execution_time = (end_time - start_time) * 1000
		
		result = GraphAnalysisResult(
			analysis_type=GraphAlgorithm.COMMUNITY_DETECTION,
			target_node_id=node_id,
			clusters=clusters,
			scores={"modularity": modularity},
			node_count=len(undirected_graph.nodes()),
			edge_count=len(undirected_graph.edges()),
			execution_time_ms=execution_time,
			confidence=0.8,
			coverage=1.0
		)
		
		# Cache result
		cache_key = f"communities_{node_id}"
		self._analysis_cache[cache_key] = result
		
		self._log_info("Community detection complete",
					   node_id=node_id,
					   communities_found=len(clusters),
					   modularity=modularity)
		
		return result
	
	async def _analyze_risk_propagation(self, node_id: str) -> Dict[str, Any]:
		"""Analyze how risk propagates through the graph from a node"""
		if not self.graph.has_node(node_id):
			return {"error": "Node not found"}
		
		self._log_info("Analyzing risk propagation", node_id=node_id)
		
		# Get node's risk score
		source_node = self._nodes[node_id]
		source_risk = source_node.risk_score
		
		# BFS to propagate risk with decay
		visited = set()
		risk_scores = {node_id: source_risk}
		queue = deque([(node_id, source_risk, 0)])  # (node, risk, depth)
		
		max_depth = 3  # Limit propagation depth
		decay_factor = 0.7  # Risk decreases by 30% per hop
		
		while queue:
			current_node, current_risk, depth = queue.popleft()
			
			if current_node in visited or depth >= max_depth:
				continue
			
			visited.add(current_node)
			
			# Propagate to neighbors
			for neighbor in self.graph.neighbors(current_node):
				if neighbor not in visited:
					# Calculate edge influence
					edges = self.graph[current_node][neighbor]
					max_strength = max(edge_data.get('strength', 0.5) for edge_data in edges.values())
					
					# Propagated risk = current_risk * edge_strength * decay
					propagated_risk = current_risk * max_strength * (decay_factor ** depth)
					
					# Combine with existing risk
					neighbor_node = self._nodes.get(neighbor)
					if neighbor_node:
						base_risk = neighbor_node.risk_score
						combined_risk = max(base_risk, propagated_risk)
						risk_scores[neighbor] = combined_risk
						
						queue.append((neighbor, combined_risk, depth + 1))
		
		# Identify high-risk propagation paths
		high_risk_nodes = [
			node for node, risk in risk_scores.items() 
			if risk > 0.7 and node != node_id
		]
		
		return {
			"source_node": node_id,
			"source_risk": source_risk,
			"risk_scores": risk_scores,
			"high_risk_nodes": high_risk_nodes,
			"propagation_depth": max_depth,
			"nodes_affected": len(risk_scores) - 1
		}
	
	async def _detect_graph_anomalies(self, node_id: str) -> Dict[str, Any]:
		"""Detect anomalies in graph structure and behavior around a node"""
		if not self.graph.has_node(node_id):
			return {"error": "Node not found"}
		
		self._log_info("Detecting graph anomalies", node_id=node_id)
		
		anomalies = []
		
		# 1. Degree anomaly detection
		degrees = dict(self.graph.degree())
		node_degree = degrees.get(node_id, 0)
		avg_degree = np.mean(list(degrees.values()))
		std_degree = np.std(list(degrees.values()))
		
		if std_degree > 0:
			degree_z_score = abs(node_degree - avg_degree) / std_degree
			if degree_z_score > 2.5:  # More than 2.5 standard deviations
				anomalies.append({
					"type": "unusual_degree",
					"severity": "high" if degree_z_score > 3.5 else "medium",
					"score": min(1.0, degree_z_score / 5.0),
					"details": f"Node degree ({node_degree}) is {degree_z_score:.1f} std devs from mean ({avg_degree:.1f})"
				})
		
		# 2. Relationship type anomaly
		node_edges = []
		for edge_id, edge in self._edges.items():
			if edge.source_node_id == node_id or edge.target_node_id == node_id:
				node_edges.append(edge)
		
		# Check for unusual relationship patterns
		relationship_types = [edge.relationship_type.value for edge in node_edges]
		type_counts = {}
		for rel_type in relationship_types:
			type_counts[rel_type] = type_counts.get(rel_type, 0) + 1
		
		# Flag if node has many different relationship types (potential hub)
		if len(type_counts) > 5:
			diversity_score = len(type_counts) / 10.0  # Normalize
			anomalies.append({
				"type": "high_relationship_diversity",
				"severity": "medium",
				"score": min(1.0, diversity_score),
				"details": f"Node has {len(type_counts)} different relationship types"
			})
		
		# 3. Temporal anomaly detection
		recent_edges = [
			edge for edge in node_edges
			if (datetime.utcnow() - edge.last_observed).days <= 7
		]
		
		if len(recent_edges) > len(node_edges) * 0.5:  # More than 50% of relationships are recent
			temporal_score = len(recent_edges) / len(node_edges) if node_edges else 0
			anomalies.append({
				"type": "recent_relationship_surge",
				"severity": "high" if temporal_score > 0.8 else "medium",
				"score": temporal_score,
				"details": f"{len(recent_edges)} out of {len(node_edges)} relationships formed in last 7 days"
			})
		
		# 4. Clustering coefficient anomaly
		try:
			clustering_coeff = nx.clustering(self.graph.to_undirected(), node_id)
			avg_clustering = nx.average_clustering(self.graph.to_undirected())
			
			if clustering_coeff < avg_clustering * 0.3 and node_degree > 3:
				# Low clustering with high degree might indicate hub/broker behavior
				anomalies.append({
					"type": "potential_broker_node",
					"severity": "medium",
					"score": 1.0 - (clustering_coeff / avg_clustering),
					"details": f"Low clustering coefficient ({clustering_coeff:.3f}) with degree {node_degree}"
				})
		except Exception:
			pass  # Skip clustering analysis if it fails
		
		# Calculate overall anomaly score
		overall_anomaly_score = max([a["score"] for a in anomalies], default=0.0)
		
		return {
			"target_node": node_id,
			"anomalies": anomalies,
			"overall_anomaly_score": overall_anomaly_score,
			"anomaly_count": len(anomalies)
		}
	
	async def _analyze_access_paths(self, node_id: str) -> Dict[str, Any]:
		"""Analyze paths from node to sensitive resources"""
		if not self.graph.has_node(node_id):
			return {"error": "Node not found"}
		
		self._log_info("Analyzing access paths", node_id=node_id)
		
		# Identify sensitive resource nodes
		sensitive_nodes = []
		for node_obj in self._nodes.values():
			if ("sensitive" in node_obj.labels or 
				node_obj.entity_type in ["admin_resource", "critical_system"] or
				"admin" in str(node_obj.attributes.get("role", "")).lower()):
				sensitive_nodes.append(node_obj.id)
		
		if not sensitive_nodes:
			return {
				"target_node": node_id,
				"sensitive_resources": [],
				"access_paths": [],
				"shortest_path_length": None
			}
		
		# Find shortest paths to sensitive resources
		access_paths = []
		path_lengths = []
		
		for sensitive_node in sensitive_nodes[:10]:  # Limit to top 10 sensitive resources
			try:
				if nx.has_path(self.graph, node_id, sensitive_node):
					path = nx.shortest_path(self.graph, node_id, sensitive_node)
					path_length = len(path) - 1  # Number of hops
					
					# Calculate path strength (minimum edge strength in path)
					path_strength = 1.0
					for i in range(len(path) - 1):
						edges = self.graph[path[i]][path[i+1]]
						min_strength = min(edge_data.get('strength', 0.5) for edge_data in edges.values())
						path_strength = min(path_strength, min_strength)
					
					access_paths.append({
						"target_resource": sensitive_node,
						"path": path,
						"path_length": path_length,
						"path_strength": path_strength,
						"risk_score": 1.0 - path_strength if path_length <= 2 else (1.0 - path_strength) * 0.5
					})
					path_lengths.append(path_length)
					
			except nx.NetworkXNoPath:
				continue  # No path exists
		
		# Sort paths by risk score
		access_paths.sort(key=lambda x: x["risk_score"], reverse=True)
		
		return {
			"target_node": node_id,
			"sensitive_resources": sensitive_nodes,
			"access_paths": access_paths[:5],  # Top 5 riskiest paths
			"shortest_path_length": min(path_lengths) if path_lengths else None,
			"average_path_length": np.mean(path_lengths) if path_lengths else None,
			"reachable_sensitive_resources": len(access_paths)
		}
	
	async def _generate_risk_indicators(self, node_id: str, analyses: Dict[str, Any]) -> List[RiskIndicator]:
		"""Generate risk indicators based on graph analyses"""
		risk_indicators = []
		
		# 1. High centrality risk
		centrality_analysis = analyses.get("centrality", {})
		if centrality_analysis and isinstance(centrality_analysis, GraphAnalysisResult):
			node_score = centrality_analysis.scores.get(node_id, 0.0)
			if node_score > self._anomaly_thresholds['degree_centrality']:
				risk_indicators.append(RiskIndicator(
					indicator_type=RiskIndicatorType.SUSPICIOUS_ASSOCIATION,
					threat_level=ThreatLevel.MEDIUM,
					primary_node_id=node_id,
					risk_score=node_score,
					confidence=0.8,
					evidence=[f"High centrality score: {node_score:.3f}"],
					detection_context={"analysis": "centrality", "threshold": self._anomaly_thresholds['degree_centrality']}
				))
		
		# 2. Anomaly-based risks
		anomaly_analysis = analyses.get("anomalies", {})
		if anomaly_analysis:
			for anomaly in anomaly_analysis.get("anomalies", []):
				if anomaly["score"] > 0.6:
					threat_level = ThreatLevel.HIGH if anomaly["severity"] == "high" else ThreatLevel.MEDIUM
					
					risk_indicators.append(RiskIndicator(
						indicator_type=RiskIndicatorType.UNUSUAL_ACCESS_PATTERN,
						threat_level=threat_level,
						primary_node_id=node_id,
						risk_score=anomaly["score"],
						confidence=0.7,
						evidence=[anomaly["details"]],
						detection_context={"analysis": "anomaly", "anomaly_type": anomaly["type"]}
					))
		
		# 3. Access path risks
		path_analysis = analyses.get("access_paths", {})
		if path_analysis:
			high_risk_paths = [
				path for path in path_analysis.get("access_paths", [])
				if path["risk_score"] > 0.7
			]
			
			if high_risk_paths:
				max_path_risk = max(path["risk_score"] for path in high_risk_paths)
				risk_indicators.append(RiskIndicator(
					indicator_type=RiskIndicatorType.PRIVILEGE_ESCALATION,
					threat_level=ThreatLevel.HIGH if max_path_risk > 0.8 else ThreatLevel.MEDIUM,
					primary_node_id=node_id,
					risk_score=max_path_risk,
					confidence=0.9,
					evidence=[f"Short path to {len(high_risk_paths)} sensitive resources"],
					detection_context={"analysis": "access_paths", "high_risk_paths": len(high_risk_paths)}
				))
		
		# 4. Risk propagation concerns
		risk_propagation = analyses.get("risk_propagation", {})
		if risk_propagation:
			high_risk_nodes = risk_propagation.get("high_risk_nodes", [])
			if len(high_risk_nodes) > 3:
				propagation_risk = min(1.0, len(high_risk_nodes) / 10.0)
				risk_indicators.append(RiskIndicator(
					indicator_type=RiskIndicatorType.LATERAL_MOVEMENT,
					threat_level=ThreatLevel.MEDIUM,
					primary_node_id=node_id,
					related_node_ids=high_risk_nodes,
					risk_score=propagation_risk,
					confidence=0.6,
					evidence=[f"Risk propagates to {len(high_risk_nodes)} connected nodes"],
					detection_context={"analysis": "risk_propagation"}
				))
		
		# Store risk indicators
		for indicator in risk_indicators:
			self._risk_indicators[indicator.id] = indicator
		
		return risk_indicators
	
	async def detect_insider_threats(self, time_window_days: int = 30) -> List[RiskIndicator]:
		"""Detect potential insider threats using graph analysis"""
		self._log_info("Detecting insider threats", time_window_days=time_window_days)
		
		cutoff_date = datetime.utcnow() - timedelta(days=time_window_days)
		insider_threats = []
		
		# Analyze each user node
		user_nodes = [node for node in self._nodes.values() if node.entity_type == "user"]
		
		for user_node in user_nodes:
			threat_indicators = []
			
			# 1. Unusual access pattern detection
			user_edges = [
				edge for edge in self._edges.values()
				if (edge.source_node_id == user_node.id or edge.target_node_id == user_node.id) and
				edge.last_observed >= cutoff_date
			]
			
			# Check for access outside normal hours
			unusual_time_accesses = []
			for edge in user_edges:
				access_hour = edge.last_observed.hour
				if access_hour < 6 or access_hour > 22:  # Outside normal business hours
					unusual_time_accesses.append(edge)
			
			if len(unusual_time_accesses) > len(user_edges) * 0.3:  # >30% unusual timing
				threat_indicators.append("Frequent access outside business hours")
			
			# 2. Rapid privilege escalation
			privileged_connections = []
			for edge in user_edges:
				other_node_id = edge.target_node_id if edge.source_node_id == user_node.id else edge.source_node_id
				other_node = self._nodes.get(other_node_id)
				
				if other_node and ("admin" in other_node.labels or other_node.risk_score > 0.7):
					privileged_connections.append(edge)
			
			# Check if privileged connections were formed recently
			recent_privileged = [
				edge for edge in privileged_connections
				if (datetime.utcnow() - edge.first_observed).days <= 7
			]
			
			if len(recent_privileged) > 2:
				threat_indicators.append("Recent connections to privileged resources")
			
			# 3. Data access velocity anomaly
			data_access_edges = [
				edge for edge in user_edges
				if edge.relationship_type in [IdentityRelationType.SAME_DEVICE, IdentityRelationType.SAME_LOCATION]
			]
			
			if len(data_access_edges) > 20:  # High volume of data access
				threat_indicators.append("High volume of data access relationships")
			
			# 4. Behavioral deviation
			# Check if user's graph position changed significantly
			try:
				current_centrality = nx.degree_centrality(self.graph.to_undirected()).get(user_node.id, 0.0)
				if current_centrality > 0.1:  # Significant centrality
					threat_indicators.append("High network centrality suggesting potential hub activity")
			except Exception:
				pass
			
			# Generate insider threat indicator if multiple red flags
			if len(threat_indicators) >= 2:
				threat_level = ThreatLevel.CRITICAL if len(threat_indicators) >= 4 else ThreatLevel.HIGH
				risk_score = min(1.0, len(threat_indicators) / 5.0)
				
				insider_threat = RiskIndicator(
					indicator_type=RiskIndicatorType.INSIDER_THREAT,
					threat_level=threat_level,
					primary_node_id=user_node.id,
					risk_score=risk_score,
					confidence=0.75,
					evidence=threat_indicators,
					detection_context={
						"time_window_days": time_window_days,
						"analysis_method": "graph_behavioral_analysis",
						"indicators_count": len(threat_indicators)
					}
				)
				
				insider_threats.append(insider_threat)
				self._risk_indicators[insider_threat.id] = insider_threat
		
		self._log_info("Insider threat detection complete",
					   threats_detected=len(insider_threats),
					   users_analyzed=len(user_nodes))
		
		return insider_threats
	
	async def detect_coordinated_attacks(self) -> List[RiskIndicator]:
		"""Detect coordinated attacks using graph clustering and timing analysis"""
		self._log_info("Detecting coordinated attacks")
		
		coordinated_attacks = []
		
		# Find suspicious clusters of activity
		recent_edges = [
			edge for edge in self._edges.values()
			if (datetime.utcnow() - edge.last_observed).days <= 1
		]
		
		if len(recent_edges) < 5:
			return coordinated_attacks  # Not enough recent activity
		
		# Group edges by time windows (1-hour windows)
		time_windows = defaultdict(list)
		for edge in recent_edges:
			hour_key = edge.last_observed.replace(minute=0, second=0, microsecond=0)
			time_windows[hour_key].append(edge)
		
		# Look for suspicious patterns in time windows
		for time_key, edges in time_windows.items():
			if len(edges) < 3:
				continue
			
			# Check if edges form a connected component (coordinated activity)
			involved_nodes = set()
			for edge in edges:
				involved_nodes.add(edge.source_node_id)
				involved_nodes.add(edge.target_node_id)
			
			if len(involved_nodes) >= 4:  # At least 4 nodes involved
				# Check if these nodes have similar characteristics
				node_types = []
				risk_scores = []
				
				for node_id in involved_nodes:
					node = self._nodes.get(node_id)
					if node:
						node_types.append(node.entity_type)
						risk_scores.append(node.risk_score)
				
				# Look for patterns
				suspicious_indicators = []
				
				# Multiple high-risk nodes active simultaneously
				high_risk_nodes = sum(1 for score in risk_scores if score > 0.6)
				if high_risk_nodes > len(involved_nodes) * 0.5:
					suspicious_indicators.append("Multiple high-risk entities active simultaneously")
				
				# Unusual entity type coordination
				if len(set(node_types)) == 1 and len(involved_nodes) > 5:
					suspicious_indicators.append("Coordinated activity among similar entity types")
				
				# Generate coordinated attack indicator
				if suspicious_indicators:
					attack_indicator = RiskIndicator(
						indicator_type=RiskIndicatorType.COORDINATED_ATTACK,
						threat_level=ThreatLevel.HIGH,
						primary_node_id=list(involved_nodes)[0],  # First node as primary
						related_node_ids=list(involved_nodes)[1:],
						risk_score=min(1.0, len(suspicious_indicators) / 3.0 + high_risk_nodes / len(involved_nodes)),
						confidence=0.7,
						evidence=suspicious_indicators + [f"Activity cluster at {time_key}"],
						detection_context={
							"time_window": time_key.isoformat(),
							"involved_nodes": len(involved_nodes),
							"activity_edges": len(edges)
						}
					)
					
					coordinated_attacks.append(attack_indicator)
					self._risk_indicators[attack_indicator.id] = attack_indicator
		
		self._log_info("Coordinated attack detection complete",
					   attacks_detected=len(coordinated_attacks),
					   time_windows_analyzed=len(time_windows))
		
		return coordinated_attacks
	
	async def get_graph_statistics(self) -> GraphMetrics:
		"""Get comprehensive graph statistics"""
		if self.graph.number_of_nodes() == 0:
			return GraphMetrics(0, 0, 0.0, 0, 0.0, 0, 0, 0.0, 0.0, 0.0)
		
		# Basic metrics
		total_nodes = self.graph.number_of_nodes()
		total_edges = self.graph.number_of_edges()
		
		# Degree statistics
		degrees = [degree for node, degree in self.graph.degree()]
		avg_degree = np.mean(degrees) if degrees else 0.0
		max_degree = max(degrees) if degrees else 0
		
		# Convert to undirected for some calculations
		undirected = self.graph.to_undirected()
		
		# Clustering coefficient
		try:
			clustering_coefficient = nx.average_clustering(undirected)
		except Exception:
			clustering_coefficient = 0.0
		
		# Diameter (maximum shortest path)
		try:
			if nx.is_connected(undirected):
				diameter = nx.diameter(undirected)
			else:
				# For disconnected graphs, use the largest component
				largest_cc = max(nx.connected_components(undirected), key=len)
				subgraph = undirected.subgraph(largest_cc)
				diameter = nx.diameter(subgraph) if len(subgraph) > 1 else 0
		except Exception:
			diameter = 0
		
		# Connected components
		connected_components = nx.number_connected_components(undirected)
		
		# Density
		try:
			density = nx.density(undirected)
		except Exception:
			density = 0.0
		
		# Assortativity (degree correlation)
		try:
			assortativity = nx.degree_assortativity_coefficient(undirected)
		except Exception:
			assortativity = 0.0
		
		# Modularity
		try:
			communities = list(nx.community.louvain_communities(undirected))
			modularity = nx.community.modularity(undirected, communities)
		except Exception:
			modularity = 0.0
		
		return GraphMetrics(
			total_nodes=total_nodes,
			total_edges=total_edges,
			avg_degree=avg_degree,
			max_degree=max_degree,
			clustering_coefficient=clustering_coefficient,
			diameter=diameter,
			connected_components=connected_components,
			density=density,
			assortativity=assortativity,
			modularity=modularity
		)
	
	def get_risk_indicators_summary(self, threat_levels: Optional[List[ThreatLevel]] = None) -> Dict[str, Any]:
		"""Get summary of risk indicators"""
		indicators = list(self._risk_indicators.values())
		
		if threat_levels:
			indicators = [ind for ind in indicators if ind.threat_level in threat_levels]
		
		# Group by type
		indicators_by_type = defaultdict(list)
		for indicator in indicators:
			indicators_by_type[indicator.indicator_type.value].append(indicator)
		
		# Summary statistics
		total_indicators = len(indicators)
		avg_risk_score = np.mean([ind.risk_score for ind in indicators]) if indicators else 0.0
		
		threat_level_counts = defaultdict(int)
		for indicator in indicators:
			threat_level_counts[indicator.threat_level.value] += 1
		
		return {
			"total_indicators": total_indicators,
			"avg_risk_score": avg_risk_score,
			"threat_level_distribution": dict(threat_level_counts),
			"indicators_by_type": {
				ind_type: len(type_indicators)
				for ind_type, type_indicators in indicators_by_type.items()
			},
			"recent_indicators": len([
				ind for ind in indicators
				if (datetime.utcnow() - ind.first_detected).days <= 7
			])
		}
	
	def clear_user_graph_data(self, user_id: str):
		"""Clear all graph data for a user (GDPR compliance)"""
		# Find user nodes
		user_node_ids = [
			node.id for node in self._nodes.values()
			if node.entity_type == "user" and user_id in [node.entity_id, node.id]
		]
		
		# Remove nodes from graph
		for node_id in user_node_ids:
			if self.graph.has_node(node_id):
				self.graph.remove_node(node_id)
			if node_id in self._nodes:
				del self._nodes[node_id]
		
		# Remove related edges
		edges_to_remove = []
		for edge_id, edge in self._edges.items():
			if edge.source_node_id in user_node_ids or edge.target_node_id in user_node_ids:
				edges_to_remove.append(edge_id)
		
		for edge_id in edges_to_remove:
			del self._edges[edge_id]
		
		# Clear risk indicators
		risk_indicators_to_remove = []
		for indicator_id, indicator in self._risk_indicators.items():
			if indicator.primary_node_id in user_node_ids:
				risk_indicators_to_remove.append(indicator_id)
		
		for indicator_id in risk_indicators_to_remove:
			del self._risk_indicators[indicator_id]
		
		# Clear caches
		self._analysis_cache.clear()
		
		self._log_info("User graph data cleared",
					   user_id=user_id,
					   nodes_removed=len(user_node_ids),
					   edges_removed=len(edges_to_remove),
					   indicators_removed=len(risk_indicators_to_remove))