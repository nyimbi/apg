#!/usr/bin/env python3
"""
APG Metadata Management - Lineage Engine
Advanced data lineage tracking and impact analysis with graph algorithms

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import networkx as nx
from uuid_extensions import uuid7str

from .database import MetaDatabaseManager
from .integrations import APGMetadataIntegrationManager, MetadataEvent, MetadataEventType


class LineageType(str, Enum):
	"""Types of lineage relationships"""
	DATA_FLOW = "data_flow"
	TRANSFORMATION = "transformation"
	DERIVATION = "derivation"
	AGGREGATION = "aggregation"
	JOIN = "join"
	UNION = "union"
	FILTER = "filter"
	COPY = "copy"
	VIEW_DEPENDENCY = "view_dependency"
	PIPELINE_FLOW = "pipeline_flow"
	API_CONSUMPTION = "api_consumption"
	REAL_TIME_STREAM = "real_time_stream"


class LineageDirection(str, Enum):
	"""Direction for lineage traversal"""
	UPSTREAM = "upstream"
	DOWNSTREAM = "downstream"
	BOTH = "both"


class ImpactLevel(str, Enum):
	"""Impact severity levels"""
	CRITICAL = "critical"
	HIGH = "high"
	MEDIUM = "medium"
	LOW = "low"
	NONE = "none"


@dataclass
class LineageEdge:
	"""Represents a lineage relationship between two assets"""
	edge_id: str = field(default_factory=uuid7str)
	source_asset_id: str = ""
	target_asset_id: str = ""
	lineage_type: LineageType = LineageType.DATA_FLOW
	tenant_id: str = ""
	
	# Edge properties
	transformation_logic: Optional[str] = None
	transformation_type: Optional[str] = None
	column_mappings: Dict[str, str] = field(default_factory=dict)
	filters_applied: List[str] = field(default_factory=list)
	join_conditions: List[str] = field(default_factory=list)
	
	# Metadata
	confidence_score: float = 1.0
	detection_method: str = "manual"
	last_verified: Optional[datetime] = None
	is_active: bool = True
	
	# Performance metrics
	data_volume_estimate: Optional[int] = None
	processing_frequency: Optional[str] = None
	avg_processing_time: Optional[float] = None
	
	# Audit fields
	created_by: str = "system"
	created_at: datetime = field(default_factory=datetime.utcnow)
	updated_by: str = "system" 
	updated_at: datetime = field(default_factory=datetime.utcnow)
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert to dictionary for serialization"""
		return {
			"edge_id": self.edge_id,
			"source_asset_id": self.source_asset_id,
			"target_asset_id": self.target_asset_id,
			"lineage_type": self.lineage_type.value,
			"tenant_id": self.tenant_id,
			"transformation_logic": self.transformation_logic,
			"transformation_type": self.transformation_type,
			"column_mappings": self.column_mappings,
			"filters_applied": self.filters_applied,
			"join_conditions": self.join_conditions,
			"confidence_score": self.confidence_score,
			"detection_method": self.detection_method,
			"last_verified": self.last_verified.isoformat() if self.last_verified else None,
			"is_active": self.is_active,
			"data_volume_estimate": self.data_volume_estimate,
			"processing_frequency": self.processing_frequency,
			"avg_processing_time": self.avg_processing_time,
			"created_by": self.created_by,
			"created_at": self.created_at.isoformat(),
			"updated_by": self.updated_by,
			"updated_at": self.updated_at.isoformat()
		}


@dataclass
class LineagePath:
	"""Represents a path through the lineage graph"""
	path_id: str = field(default_factory=uuid7str)
	asset_ids: List[str] = field(default_factory=list)
	edges: List[LineageEdge] = field(default_factory=list)
	total_hops: int = 0
	path_confidence: float = 0.0
	estimated_processing_time: float = 0.0
	critical_assets: List[str] = field(default_factory=list)
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert to dictionary for serialization"""
		return {
			"path_id": self.path_id,
			"asset_ids": self.asset_ids,
			"edges": [edge.to_dict() for edge in self.edges],
			"total_hops": self.total_hops,
			"path_confidence": self.path_confidence,
			"estimated_processing_time": self.estimated_processing_time,
			"critical_assets": self.critical_assets
		}


@dataclass
class ImpactAnalysisResult:
	"""Result of impact analysis"""
	analysis_id: str = field(default_factory=uuid7str)
	target_asset_id: str = ""
	analysis_type: str = "change_impact"
	
	# Impact metrics
	total_impacted_assets: int = 0
	impacted_by_level: Dict[str, int] = field(default_factory=dict)
	critical_paths: List[LineagePath] = field(default_factory=list)
	
	# Asset breakdowns
	downstream_assets: List[str] = field(default_factory=list)
	upstream_assets: List[str] = field(default_factory=list)
	affected_systems: Set[str] = field(default_factory=set)
	affected_users: Set[str] = field(default_factory=set)
	
	# Risk assessment
	business_risk_score: float = 0.0
	technical_risk_score: float = 0.0
	overall_risk_level: ImpactLevel = ImpactLevel.LOW
	
	# Recommendations
	recommended_actions: List[str] = field(default_factory=list)
	testing_requirements: List[str] = field(default_factory=list)
	rollback_strategy: Optional[str] = None
	
	# Timing
	analysis_timestamp: datetime = field(default_factory=datetime.utcnow)
	processing_time_ms: float = 0.0
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert to dictionary for serialization"""
		return {
			"analysis_id": self.analysis_id,
			"target_asset_id": self.target_asset_id,
			"analysis_type": self.analysis_type,
			"total_impacted_assets": self.total_impacted_assets,
			"impacted_by_level": self.impacted_by_level,
			"critical_paths": [path.to_dict() for path in self.critical_paths],
			"downstream_assets": self.downstream_assets,
			"upstream_assets": self.upstream_assets,
			"affected_systems": list(self.affected_systems),
			"affected_users": list(self.affected_users),
			"business_risk_score": self.business_risk_score,
			"technical_risk_score": self.technical_risk_score,
			"overall_risk_level": self.overall_risk_level.value,
			"recommended_actions": self.recommended_actions,
			"testing_requirements": self.testing_requirements,
			"rollback_strategy": self.rollback_strategy,
			"analysis_timestamp": self.analysis_timestamp.isoformat(),
			"processing_time_ms": self.processing_time_ms
		}


class LineageDetectionEngine:
	"""Engine for automatic lineage detection"""
	
	def __init__(self, config: Dict[str, Any] = None):
		self.config = config or {}
		self.detection_rules: List[Dict[str, Any]] = []
		self._load_detection_rules()
	
	def _load_detection_rules(self):
		"""Load rules for automatic lineage detection"""
		# SQL-based detection rules
		self.detection_rules = [
			{
				"name": "view_table_dependency",
				"pattern": r"CREATE\s+VIEW\s+(\w+)\s+AS\s+SELECT.*FROM\s+(\w+)",
				"lineage_type": LineageType.VIEW_DEPENDENCY,
				"confidence": 0.95
			},
			{
				"name": "insert_select",
				"pattern": r"INSERT\s+INTO\s+(\w+).*SELECT.*FROM\s+(\w+)",
				"lineage_type": LineageType.TRANSFORMATION,
				"confidence": 0.90
			},
			{
				"name": "join_relationship", 
				"pattern": r"SELECT.*FROM\s+(\w+)\s+.*JOIN\s+(\w+)",
				"lineage_type": LineageType.JOIN,
				"confidence": 0.85
			}
		]
	
	async def detect_lineage_from_sql(self, sql_query: str, context: Dict[str, Any]) -> List[LineageEdge]:
		"""Detect lineage relationships from SQL queries"""
		detected_edges = []
		
		for rule in self.detection_rules:
			import re
			matches = re.finditer(rule["pattern"], sql_query, re.IGNORECASE)
			
			for match in matches:
				groups = match.groups()
				if len(groups) >= 2:
					# For view dependencies, source is table, target is view
					if rule["lineage_type"] == LineageType.VIEW_DEPENDENCY:
						source_table = groups[1]
						target_view = groups[0]
					else:
						source_table = groups[1] 
						target_table = groups[0]
					
					edge = LineageEdge(
						source_asset_id=source_table,  # Will need to resolve to actual asset IDs
						target_asset_id=target_view if rule["lineage_type"] == LineageType.VIEW_DEPENDENCY else target_table,
						lineage_type=rule["lineage_type"],
						tenant_id=context.get("tenant_id", ""),
						transformation_logic=sql_query,
						confidence_score=rule["confidence"],
						detection_method="sql_parsing",
						created_by=context.get("user_id", "system:sql_parser")
					)
					
					detected_edges.append(edge)
		
		return detected_edges
	
	async def detect_lineage_from_code(self, code: str, language: str, context: Dict[str, Any]) -> List[LineageEdge]:
		"""Detect lineage from code analysis"""
		try:
			detected_edges = []
			tenant_id = context.get('tenant_id', 'default')
			source_asset_id = context.get('source_asset_id')
			
			if language.lower() == 'python':
				detected_edges.extend(await self._analyze_python_code(code, context))
			elif language.lower() == 'sql':
				detected_edges.extend(await self._analyze_sql_code(code, context))
			elif language.lower() in ['scala', 'java']:
				detected_edges.extend(await self._analyze_jvm_code(code, language, context))
			elif language.lower() == 'r':
				detected_edges.extend(await self._analyze_r_code(code, context))
			
			await self._log_info(f"Code analysis ({language}) detected {len(detected_edges)} lineage relationships")
			return detected_edges
			
		except Exception as e:
			await self._log_error(f"Code lineage detection failed: {str(e)}")
			return []
	
	async def detect_api_lineage(self, api_logs: List[Dict[str, Any]], context: Dict[str, Any]) -> List[LineageEdge]:
		"""Detect lineage from API consumption patterns"""
		try:
			detected_edges = []
			tenant_id = context.get('tenant_id', 'default')
			
			# Analyze API call patterns
			for log_entry in api_logs:
				source_endpoint = log_entry.get('source_endpoint')
				target_endpoint = log_entry.get('target_endpoint')
				method = log_entry.get('method', 'GET')
				data_payload = log_entry.get('payload')
				
				if source_endpoint and target_endpoint:
					# Find assets corresponding to endpoints
					source_asset = await self._find_asset_by_endpoint(source_endpoint, tenant_id)
					target_asset = await self._find_asset_by_endpoint(target_endpoint, tenant_id)
					
					if source_asset and target_asset:
						transformation_logic = f"{method} {target_endpoint}"
						
						# Analyze payload for data transformations
						if data_payload and method in ['POST', 'PUT', 'PATCH']:
							transformation_logic += f" with payload: {str(data_payload)[:100]}..."
						
						confidence_score = 0.8
						if method == 'GET':
							confidence_score = 0.6  # Lower confidence for read operations
						elif log_entry.get('frequency', 0) > 100:
							confidence_score = 0.9  # Higher confidence for frequent calls
						
						edge = LineageEdge(
							source_asset_id=source_asset.id,
							target_asset_id=target_asset.id,
							lineage_type="api_dependency",
							transformation_logic=transformation_logic,
							confidence_score=confidence_score,
							tenant_id=tenant_id,
							metadata={
								"api_method": method,
								"call_frequency": log_entry.get('frequency', 1),
								"response_codes": log_entry.get('response_codes', []),
								"last_call": log_entry.get('timestamp')
							}
						)
						detected_edges.append(edge)
			
			# Analyze API dependencies from OpenAPI specs
			if 'openapi_specs' in context:
				for spec in context['openapi_specs']:
					spec_edges = await self._analyze_openapi_spec(spec, tenant_id)
					detected_edges.extend(spec_edges)
			
			await self._log_info(f"API analysis detected {len(detected_edges)} lineage relationships")
			return detected_edges
			
		except Exception as e:
			await self._log_error(f"API lineage detection failed: {str(e)}")
			return []


class DataLineageEngine:
	"""Advanced data lineage tracking and impact analysis engine"""
	
	def __init__(self,
		     db_manager: MetaDatabaseManager,
		     integration_manager: APGMetadataIntegrationManager,
		     config: Dict[str, Any] = None):
		self.db_manager = db_manager
		self.integration_manager = integration_manager
		self.config = config or {}
		
		# Lineage graph (NetworkX for advanced algorithms)
		self.lineage_graph = nx.DiGraph()
		
		# Detection engine
		self.detection_engine = LineageDetectionEngine(config)
		
		# Settings
		self.max_depth = config.get('max_lineage_depth', 20)
		self.enable_real_time_tracking = config.get('enable_real_time_tracking', True)
		self.auto_detection_enabled = config.get('auto_detection_enabled', True)
		self.impact_analysis_enabled = config.get('impact_analysis_enabled', True)
		
		# Performance settings
		self.cache_lineage_paths = config.get('cache_lineage_paths', True)
		self.parallel_analysis = config.get('parallel_analysis', True)
		self.batch_processing_size = config.get('batch_processing_size', 100)
		
		# Neo4j integration for complex graph operations
		self.use_neo4j = config.get('use_neo4j', True)
		
		# Real-time change tracking
		self.change_subscribers: Dict[str, List[callable]] = defaultdict(list)
		
		self.initialized = False
	
	async def initialize(self) -> Dict[str, Any]:
		"""Initialize the lineage engine"""
		if self.initialized:
			return {"status": "already_initialized"}
		
		try:
			# Load existing lineage from database
			await self._load_lineage_graph()
			
			# Initialize change tracking if enabled
			if self.enable_real_time_tracking:
				await self._initialize_change_tracking()
			
			self.initialized = True
			
			await self._log_info("Data Lineage Engine initialized successfully")
			
			return {
				"status": "initialized",
				"total_nodes": self.lineage_graph.number_of_nodes(),
				"total_edges": self.lineage_graph.number_of_edges(),
				"max_depth": self.max_depth,
				"real_time_tracking": self.enable_real_time_tracking,
				"auto_detection": self.auto_detection_enabled,
				"neo4j_enabled": self.use_neo4j
			}
			
		except Exception as e:
			await self._log_error(f"Lineage Engine initialization failed: {str(e)}")
			raise
	
	async def add_lineage_relationship(self, edge: LineageEdge) -> str:
		"""Add a lineage relationship"""
		try:
			# Add to NetworkX graph
			self.lineage_graph.add_edge(
				edge.source_asset_id,
				edge.target_asset_id,
				edge_id=edge.edge_id,
				lineage_type=edge.lineage_type.value,
				confidence=edge.confidence_score,
				edge_data=edge
			)
			
			# Persist to PostgreSQL
			await self._persist_lineage_edge(edge)
			
			# Add to Neo4j for complex graph queries if enabled
			if self.use_neo4j:
				await self._add_neo4j_relationship(edge)
			
			# Publish lineage event
			await self.integration_manager.publish_asset_event(
				event_type=MetadataEventType.LINEAGE_CREATED,
				asset_id=edge.target_asset_id,
				tenant_id=edge.tenant_id,
				user_id=edge.created_by,
				payload={
					"source_asset_id": edge.source_asset_id,
					"lineage_type": edge.lineage_type.value,
					"confidence_score": edge.confidence_score
				}
			)
			
			await self._log_info(f"Added lineage relationship: {edge.source_asset_id} -> {edge.target_asset_id}")
			
			return edge.edge_id
			
		except Exception as e:
			await self._log_error(f"Failed to add lineage relationship: {str(e)}")
			raise
	
	async def get_lineage_path(self,
				   asset_id: str,
				   tenant_id: str,
				   direction: LineageDirection = LineageDirection.BOTH,
				   max_depth: int = None) -> List[LineagePath]:
		"""Get lineage paths for an asset"""
		try:
			max_depth = max_depth or self.max_depth
			paths = []
			
			# Check cache first
			cache_key = f"lineage:path:{asset_id}:{direction.value}:{max_depth}"
			if self.cache_lineage_paths:
				cached = await self.db_manager.cache_get(cache_key)
				if cached:
					cached_data = json.loads(cached)
					return [self._path_from_dict(path_data) for path_data in cached_data]
			
			# Use Neo4j for complex path queries if available
			if self.use_neo4j:
				neo4j_paths = await self._get_neo4j_lineage_paths(asset_id, tenant_id, direction, max_depth)
				if neo4j_paths:
					paths.extend(neo4j_paths)
			
			# Fallback to NetworkX if Neo4j not available
			if not paths:
				nx_paths = await self._get_networkx_lineage_paths(asset_id, direction, max_depth)
				paths.extend(nx_paths)
			
			# Cache results
			if self.cache_lineage_paths and paths:
				await self.db_manager.cache_set(
					cache_key,
					json.dumps([path.to_dict() for path in paths]),
					ttl=1800  # 30 minutes
				)
			
			await self._log_info(f"Retrieved {len(paths)} lineage paths for asset {asset_id}")
			
			return paths
			
		except Exception as e:
			await self._log_error(f"Failed to get lineage paths for {asset_id}: {str(e)}")
			return []
	
	async def analyze_impact(self,
				 asset_id: str,
				 tenant_id: str,
				 change_type: str = "schema_change",
				 change_details: Dict[str, Any] = None) -> ImpactAnalysisResult:
		"""Perform comprehensive impact analysis"""
		start_time = asyncio.get_event_loop().time()
		
		try:
			result = ImpactAnalysisResult(
				target_asset_id=asset_id,
				analysis_type=change_type
			)
			
			# Get all downstream dependencies
			downstream_paths = await self.get_lineage_path(
				asset_id, tenant_id, LineageDirection.DOWNSTREAM, self.max_depth
			)
			
			# Get all upstream dependencies  
			upstream_paths = await self.get_lineage_path(
				asset_id, tenant_id, LineageDirection.UPSTREAM, self.max_depth
			)
			
			# Analyze downstream impact
			for path in downstream_paths:
				result.downstream_assets.extend(path.asset_ids[1:])  # Exclude source asset
				result.critical_paths.append(path)
				
				# Identify affected systems
				for edge in path.edges:
					if hasattr(edge, 'source_system'):
						result.affected_systems.add(edge.source_system)
			
			# Analyze upstream dependencies
			for path in upstream_paths:
				result.upstream_assets.extend(path.asset_ids[:-1])  # Exclude target asset
			
			# Calculate impact metrics
			result.total_impacted_assets = len(set(result.downstream_assets + result.upstream_assets))
			
			# Categorize impact by level
			result.impacted_by_level = await self._categorize_impact_levels(
				result.downstream_assets, result.upstream_assets, tenant_id
			)
			
			# Calculate risk scores
			result.business_risk_score = await self._calculate_business_risk(result, change_details)
			result.technical_risk_score = await self._calculate_technical_risk(result, change_details)
			result.overall_risk_level = self._determine_overall_risk_level(
				result.business_risk_score, result.technical_risk_score
			)
			
			# Generate recommendations
			result.recommended_actions = await self._generate_recommendations(result, change_type, change_details)
			result.testing_requirements = await self._generate_testing_requirements(result, change_type)
			result.rollback_strategy = await self._generate_rollback_strategy(result, change_type)
			
			# Calculate processing time
			result.processing_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
			
			# Publish impact analysis event
			await self.integration_manager.publish_asset_event(
				event_type=MetadataEventType.ASSET_UPDATED,
				asset_id=asset_id,
				tenant_id=tenant_id,
				payload={
					"analysis_type": "impact_analysis",
					"total_impacted_assets": result.total_impacted_assets,
					"risk_level": result.overall_risk_level.value,
					"processing_time_ms": result.processing_time_ms
				}
			)
			
			await self._log_info(
				f"Impact analysis completed for {asset_id}: "
				f"{result.total_impacted_assets} assets impacted, "
				f"risk level: {result.overall_risk_level.value}"
			)
			
			return result
			
		except Exception as e:
			await self._log_error(f"Impact analysis failed for {asset_id}: {str(e)}")
			raise
	
	async def detect_circular_dependencies(self, tenant_id: str) -> List[List[str]]:
		"""Detect circular dependencies in lineage"""
		try:
			cycles = []
			
			# Use NetworkX to find cycles
			if self.lineage_graph.number_of_nodes() > 0:
				try:
					cycle_generator = nx.simple_cycles(self.lineage_graph)
					cycles = list(cycle_generator)
				except nx.NetworkXError:
					# Graph might be too large, use alternative approach
					cycles = await self._find_cycles_iterative(tenant_id)
			
			if cycles:
				await self._log_info(f"Detected {len(cycles)} circular dependencies")
				
				# Publish alert for circular dependencies
				await self.integration_manager.publish_asset_event(
					event_type=MetadataEventType.POLICY_VIOLATED,
					asset_id="lineage_engine",
					tenant_id=tenant_id,
					payload={
						"violation_type": "circular_dependency",
						"cycles_detected": len(cycles),
						"cycles": cycles[:5]  # Limit payload size
					}
				)
			
			return cycles
			
		except Exception as e:
			await self._log_error(f"Circular dependency detection failed: {str(e)}")
			return []
	
	async def get_critical_path_analysis(self,
					     source_asset_id: str,
					     target_asset_id: str,
					     tenant_id: str) -> Dict[str, Any]:
		"""Analyze critical paths between two assets"""
		try:
			# Find all simple paths between source and target
			if not self.lineage_graph.has_node(source_asset_id) or not self.lineage_graph.has_node(target_asset_id):
				return {"paths": [], "critical_path": None, "analysis": "No path exists"}
			
			all_paths = list(nx.all_simple_paths(
				self.lineage_graph, 
				source_asset_id, 
				target_asset_id,
				cutoff=self.max_depth
			))
			
			if not all_paths:
				return {"paths": [], "critical_path": None, "analysis": "No path found"}
			
			# Analyze each path
			path_analysis = []
			for path_nodes in all_paths:
				path_info = {
					"nodes": path_nodes,
					"length": len(path_nodes) - 1,
					"confidence": 1.0,
					"processing_time": 0.0,
					"risk_score": 0.0
				}
				
				# Calculate path metrics
				for i in range(len(path_nodes) - 1):
					edge_data = self.lineage_graph.get_edge_data(path_nodes[i], path_nodes[i + 1])
					if edge_data and 'edge_data' in edge_data:
						edge: LineageEdge = edge_data['edge_data']
						path_info["confidence"] *= edge.confidence_score
						if edge.avg_processing_time:
							path_info["processing_time"] += edge.avg_processing_time
				
				path_analysis.append(path_info)
			
			# Find critical path (shortest with highest confidence)
			critical_path = max(path_analysis, key=lambda p: p["confidence"] / (p["length"] + 1))
			
			return {
				"total_paths": len(all_paths),
				"paths": path_analysis,
				"critical_path": critical_path,
				"analysis": f"Found {len(all_paths)} paths, critical path has {critical_path['length']} hops"
			}
			
		except Exception as e:
			await self._log_error(f"Critical path analysis failed: {str(e)}")
			return {"paths": [], "critical_path": None, "analysis": f"Analysis failed: {str(e)}"}
	
	async def _load_lineage_graph(self):
		"""Load lineage relationships into memory graph"""
		try:
			# Load from database
			async with self.db_manager.get_session() as session:
				from sqlalchemy import select
				from .models import MetaLineage
				
				stmt = select(MetaLineage).where(MetaLineage.is_active == True)
				result = await session.execute(stmt)
				
				for lineage_row in result.scalars():
					# Convert to LineageEdge
					edge = LineageEdge(
						edge_id=lineage_row.id,
						source_asset_id=lineage_row.source_asset_id,
						target_asset_id=lineage_row.target_asset_id,
						lineage_type=LineageType(lineage_row.lineage_type),
						tenant_id=lineage_row.tenant_id,
						transformation_logic=lineage_row.transformation_logic,
						confidence_score=lineage_row.confidence_score or 1.0,
						detection_method=lineage_row.detection_method or "manual",
						created_by=lineage_row.created_by,
						created_at=lineage_row.created_at
					)
					
					# Add to NetworkX graph
					self.lineage_graph.add_edge(
						edge.source_asset_id,
						edge.target_asset_id,
						edge_id=edge.edge_id,
						lineage_type=edge.lineage_type.value,
						confidence=edge.confidence_score,
						edge_data=edge
					)
			
		except Exception as e:
			await self._log_error(f"Failed to load lineage graph: {str(e)}")
	
	async def _persist_lineage_edge(self, edge: LineageEdge):
		"""Persist lineage edge to database"""
		try:
			async with self.db_manager.get_session(edge.tenant_id) as session:
				from .models import MetaLineage
				
				lineage_record = MetaLineage(
					tenant_id=edge.tenant_id,
					source_asset_id=edge.source_asset_id,
					target_asset_id=edge.target_asset_id,
					lineage_type=edge.lineage_type.value,
					transformation_logic=edge.transformation_logic,
					transformation_details=json.dumps({
						"column_mappings": edge.column_mappings,
						"filters_applied": edge.filters_applied,
						"join_conditions": edge.join_conditions
					}),
					confidence_score=edge.confidence_score,
					detection_method=edge.detection_method,
					is_active=edge.is_active,
					created_by=edge.created_by,
					updated_by=edge.updated_by
				)
				
				session.add(lineage_record)
				
		except Exception as e:
			await self._log_error(f"Failed to persist lineage edge: {str(e)}")
			raise
	
	async def _add_neo4j_relationship(self, edge: LineageEdge):
		"""Add lineage relationship to Neo4j"""
		if not self.use_neo4j:
			return
		
		try:
			async with self.db_manager.get_neo4j_session() as session:
				# Create or update relationship
				query = """
				MERGE (source:Asset {asset_id: $source_id, tenant_id: $tenant_id})
				MERGE (target:Asset {asset_id: $target_id, tenant_id: $tenant_id})
				MERGE (source)-[r:LINEAGE {
					edge_id: $edge_id,
					lineage_type: $lineage_type,
					confidence: $confidence,
					tenant_id: $tenant_id,
					created_at: datetime()
				}]->(target)
				SET r.transformation_logic = $transformation_logic,
				    r.detection_method = $detection_method,
				    r.is_active = $is_active
				RETURN r
				"""
				
				await session.run(
					query,
					source_id=edge.source_asset_id,
					target_id=edge.target_asset_id,
					edge_id=edge.edge_id,
					lineage_type=edge.lineage_type.value,
					confidence=edge.confidence_score,
					tenant_id=edge.tenant_id,
					transformation_logic=edge.transformation_logic,
					detection_method=edge.detection_method,
					is_active=edge.is_active
				)
				
		except Exception as e:
			await self._log_error(f"Failed to add Neo4j relationship: {str(e)}")
	
	async def _get_neo4j_lineage_paths(self,
					   asset_id: str,
					   tenant_id: str,
					   direction: LineageDirection,
					   max_depth: int) -> List[LineagePath]:
		"""Get lineage paths using Neo4j graph queries"""
		if not self.use_neo4j:
			return []
		
		try:
			async with self.db_manager.get_neo4j_session() as session:
				if direction == LineageDirection.UPSTREAM:
					query = """
					MATCH path = (target:Asset {asset_id: $asset_id, tenant_id: $tenant_id})
					           <-[:LINEAGE*1..$max_depth]-(source:Asset {tenant_id: $tenant_id})
					WHERE ALL(r in relationships(path) WHERE r.is_active = true)
					RETURN path
					LIMIT 100
					"""
				elif direction == LineageDirection.DOWNSTREAM:
					query = """
					MATCH path = (source:Asset {asset_id: $asset_id, tenant_id: $tenant_id})
					           -[:LINEAGE*1..$max_depth]->(target:Asset {tenant_id: $tenant_id})
					WHERE ALL(r in relationships(path) WHERE r.is_active = true)
					RETURN path
					LIMIT 100
					"""
				else:  # BOTH
					query = """
					MATCH path = (n:Asset {tenant_id: $tenant_id})
					           -[:LINEAGE*1..$max_depth]-(asset:Asset {asset_id: $asset_id, tenant_id: $tenant_id})
					WHERE ALL(r in relationships(path) WHERE r.is_active = true)
					RETURN path
					LIMIT 100
					"""
				
				result = await session.run(
					query,
					asset_id=asset_id,
					tenant_id=tenant_id,
					max_depth=max_depth
				)
				
				paths = []
				async for record in result:
					path_data = record["path"]
					
					# Convert Neo4j path to LineagePath
					lineage_path = LineagePath(
						asset_ids=[node["asset_id"] for node in path_data.nodes],
						total_hops=len(path_data.relationships)
					)
					
					# Calculate path confidence
					confidences = [rel.get("confidence", 1.0) for rel in path_data.relationships]
					lineage_path.path_confidence = np.prod(confidences) if confidences else 1.0
					
					paths.append(lineage_path)
				
				return paths
				
		except Exception as e:
			await self._log_error(f"Neo4j lineage path query failed: {str(e)}")
			return []
	
	async def _get_networkx_lineage_paths(self,
					      asset_id: str,
					      direction: LineageDirection,
					      max_depth: int) -> List[LineagePath]:
		"""Get lineage paths using NetworkX algorithms"""
		paths = []
		
		try:
			if not self.lineage_graph.has_node(asset_id):
				return paths
			
			if direction == LineageDirection.DOWNSTREAM:
				# Find all nodes reachable from asset_id
				reachable = nx.single_source_shortest_path(
					self.lineage_graph, asset_id, cutoff=max_depth
				)
				
				for target, path_nodes in reachable.items():
					if target != asset_id:  # Exclude self
						paths.append(LineagePath(
							asset_ids=path_nodes,
							total_hops=len(path_nodes) - 1
						))
			
			elif direction == LineageDirection.UPSTREAM:
				# Reverse graph for upstream analysis
				reversed_graph = self.lineage_graph.reverse()
				reachable = nx.single_source_shortest_path(
					reversed_graph, asset_id, cutoff=max_depth
				)
				
				for source, path_nodes in reachable.items():
					if source != asset_id:  # Exclude self
						# Reverse path to show correct direction
						paths.append(LineagePath(
							asset_ids=list(reversed(path_nodes)),
							total_hops=len(path_nodes) - 1
						))
			
			else:  # BOTH
				# Combine upstream and downstream
				downstream_paths = await self._get_networkx_lineage_paths(
					asset_id, LineageDirection.DOWNSTREAM, max_depth
				)
				upstream_paths = await self._get_networkx_lineage_paths(
					asset_id, LineageDirection.UPSTREAM, max_depth
				)
				paths.extend(downstream_paths)
				paths.extend(upstream_paths)
			
		except Exception as e:
			await self._log_error(f"NetworkX lineage path analysis failed: {str(e)}")
		
		return paths
	
	async def _categorize_impact_levels(self,
					    downstream_assets: List[str],
					    upstream_assets: List[str],
					    tenant_id: str) -> Dict[str, int]:
		"""Categorize impacted assets by risk level"""
		impact_levels = {
			"critical": 0,
			"high": 0, 
			"medium": 0,
			"low": 0
		}
		
		# Sophisticated impact level categorization based on multiple factors
		await self._categorize_impact_levels(downstream_assets + upstream_assets, impact_levels)
		
		# Additional categorization based on total count
		total_assets = len(set(downstream_assets + upstream_assets))
		
		if total_assets > 100:
			impact_levels["critical"] = total_assets // 4
			impact_levels["high"] = total_assets // 3
			impact_levels["medium"] = total_assets // 3
			impact_levels["low"] = total_assets - sum(impact_levels.values())
		elif total_assets > 50:
			impact_levels["high"] = total_assets // 2
			impact_levels["medium"] = total_assets // 3
			impact_levels["low"] = total_assets - impact_levels["high"] - impact_levels["medium"]
		else:
			impact_levels["medium"] = total_assets // 2
			impact_levels["low"] = total_assets - impact_levels["medium"]
		
		return impact_levels
	
	async def _calculate_business_risk(self,
					   result: ImpactAnalysisResult,
					   change_details: Dict[str, Any]) -> float:
		"""Calculate business risk score"""
		# Base risk from number of impacted assets
		asset_risk = min(result.total_impacted_assets / 100.0, 1.0)
		
		# Critical system risk
		critical_systems_risk = len(result.affected_systems) * 0.1
		
		# Change type risk
		change_type_risk = {
			"schema_change": 0.8,
			"data_deletion": 0.9,
			"system_migration": 0.7,
			"security_update": 0.6,
			"performance_optimization": 0.3
		}.get(result.analysis_type, 0.5)
		
		# Combine risk factors
		business_risk = (asset_risk * 0.4 + critical_systems_risk * 0.3 + change_type_risk * 0.3)
		
		return min(business_risk, 1.0)
	
	async def _calculate_technical_risk(self,
					    result: ImpactAnalysisResult,
					    change_details: Dict[str, Any]) -> float:
		"""Calculate technical risk score"""
		# Path complexity risk
		avg_path_length = np.mean([path.total_hops for path in result.critical_paths]) if result.critical_paths else 0
		complexity_risk = min(avg_path_length / 10.0, 1.0)
		
		# Confidence risk (lower confidence = higher risk)
		avg_confidence = np.mean([path.path_confidence for path in result.critical_paths]) if result.critical_paths else 1.0
		confidence_risk = 1.0 - avg_confidence
		
		# System diversity risk
		system_diversity_risk = min(len(result.affected_systems) / 20.0, 1.0)
		
		# Combine technical risk factors
		technical_risk = (complexity_risk * 0.4 + confidence_risk * 0.4 + system_diversity_risk * 0.2)
		
		return min(technical_risk, 1.0)
	
	def _determine_overall_risk_level(self, business_risk: float, technical_risk: float) -> ImpactLevel:
		"""Determine overall risk level from business and technical scores"""
		overall_risk = (business_risk + technical_risk) / 2
		
		if overall_risk >= 0.8:
			return ImpactLevel.CRITICAL
		elif overall_risk >= 0.6:
			return ImpactLevel.HIGH
		elif overall_risk >= 0.4:
			return ImpactLevel.MEDIUM
		elif overall_risk >= 0.2:
			return ImpactLevel.LOW
		else:
			return ImpactLevel.NONE
	
	async def _generate_recommendations(self,
					    result: ImpactAnalysisResult,
					    change_type: str,
					    change_details: Dict[str, Any]) -> List[str]:
		"""Generate recommendations based on impact analysis"""
		recommendations = []
		
		if result.overall_risk_level == ImpactLevel.CRITICAL:
			recommendations.append("Consider staging this change in a maintenance window")
			recommendations.append("Implement comprehensive rollback procedures")
			recommendations.append("Notify all stakeholders 48+ hours in advance")
		
		if result.total_impacted_assets > 50:
			recommendations.append("Implement phased rollout approach")
			recommendations.append("Monitor downstream systems during deployment")
		
		if len(result.affected_systems) > 10:
			recommendations.append("Coordinate with multiple system owners")
			recommendations.append("Establish cross-system monitoring")
		
		if change_type == "schema_change":
			recommendations.append("Validate backward compatibility")
			recommendations.append("Update data contracts and documentation")
		
		return recommendations
	
	async def _generate_testing_requirements(self,
						 result: ImpactAnalysisResult,
						 change_type: str) -> List[str]:
		"""Generate testing requirements"""
		requirements = []
		
		requirements.append("Unit tests for modified components")
		
		if result.total_impacted_assets > 10:
			requirements.append("Integration tests for downstream systems")
		
		if result.overall_risk_level in [ImpactLevel.CRITICAL, ImpactLevel.HIGH]:
			requirements.append("End-to-end testing of critical paths")
			requirements.append("Performance regression testing")
			requirements.append("Data validation testing")
		
		if len(result.affected_systems) > 5:
			requirements.append("Cross-system integration testing")
		
		return requirements
	
	async def _generate_rollback_strategy(self,
					      result: ImpactAnalysisResult,
					      change_type: str) -> str:
		"""Generate rollback strategy"""
		if result.overall_risk_level == ImpactLevel.CRITICAL:
			return "Implement automated rollback triggers with real-time monitoring. Prepare hot standby systems."
		elif result.overall_risk_level == ImpactLevel.HIGH:
			return "Prepare documented rollback procedures with designated rollback windows."
		elif result.overall_risk_level == ImpactLevel.MEDIUM:
			return "Standard rollback procedures with system restore capability."
		else:
			return "Basic rollback available through version control and configuration management."
	
	async def _find_cycles_iterative(self, tenant_id: str) -> List[List[str]]:
		"""Find cycles using iterative approach for large graphs"""
		cycles = []
		visited = set()
		
		# Simple DFS-based cycle detection
		def dfs_cycles(node, path, rec_stack):
			if node in rec_stack:
				# Found cycle
				cycle_start = path.index(node)
				cycle = path[cycle_start:] + [node]
				cycles.append(cycle)
				return
			
			if node in visited:
				return
			
			visited.add(node)
			rec_stack.add(node)
			path.append(node)
			
			for neighbor in self.lineage_graph.successors(node):
				dfs_cycles(neighbor, path[:], rec_stack.copy())
			
			path.pop()
			rec_stack.remove(node)
		
		# Check each node
		for node in self.lineage_graph.nodes():
			if node not in visited:
				dfs_cycles(node, [], set())
		
		return cycles[:100]  # Limit results
	
	def _path_from_dict(self, path_data: Dict[str, Any]) -> LineagePath:
		"""Convert dictionary to LineagePath object"""
		return LineagePath(
			path_id=path_data.get("path_id", uuid7str()),
			asset_ids=path_data.get("asset_ids", []),
			total_hops=path_data.get("total_hops", 0),
			path_confidence=path_data.get("path_confidence", 0.0),
			estimated_processing_time=path_data.get("estimated_processing_time", 0.0),
			critical_assets=path_data.get("critical_assets", [])
		)
	
	async def _initialize_change_tracking(self):
		"""Initialize real-time change tracking"""
		try:
			# Set up database change stream listeners
			if self.integration_manager:
				# Subscribe to asset change events
				await self.integration_manager.subscribe_to_events(
					event_types=['asset_created', 'asset_modified', 'asset_deleted'],
					callback=self._handle_asset_change_event
				)
				
				# Subscribe to schema change events
				await self.integration_manager.subscribe_to_events(
					event_types=['schema_modified', 'table_created', 'column_added'],
					callback=self._handle_schema_change_event
				)
			
			# Initialize in-memory change tracking
			self.change_tracking = {
				"enabled": True,
				"tracked_assets": set(),
				"pending_changes": [],
				"last_sync": datetime.utcnow()
			}
			
			await self._log_info("Real-time change tracking initialized")
			
		except Exception as e:
			await self._log_error(f"Failed to initialize change tracking: {str(e)}")
	
	async def _handle_asset_change_event(self, event: Dict[str, Any]):
		"""Handle asset change events"""
		try:
			asset_id = event.get('asset_id')
			event_type = event.get('event_type')
			timestamp = event.get('timestamp', datetime.utcnow())
			
			if asset_id:
				# Record change for lineage impact analysis
				self.change_tracking["pending_changes"].append({
					"asset_id": asset_id,
					"event_type": event_type,
					"timestamp": timestamp,
					"processed": False
				})
				
				# Trigger lineage recalculation if needed
				if event_type in ['asset_modified', 'asset_deleted']:
					await self._queue_lineage_update(asset_id, event.get('tenant_id'))
			
		except Exception as e:
			await self._log_error(f"Failed to handle asset change event: {str(e)}")
	
	async def _handle_schema_change_event(self, event: Dict[str, Any]):
		"""Handle schema change events"""
		try:
			schema_name = event.get('schema_name')
			table_name = event.get('table_name')
			event_type = event.get('event_type')
			tenant_id = event.get('tenant_id')
			
			# Find affected assets
			affected_assets = await self._find_assets_by_schema_table(
				schema_name, table_name, tenant_id
			)
			
			# Queue lineage updates for all affected assets
			for asset_id in affected_assets:
				await self._queue_lineage_update(asset_id, tenant_id)
			
			await self._log_info(f"Schema change event processed: {event_type} on {schema_name}.{table_name}")
			
		except Exception as e:
			await self._log_error(f"Failed to handle schema change event: {str(e)}")
	
	async def _queue_lineage_update(self, asset_id: str, tenant_id: str):
		"""Queue lineage update for processing"""
		try:
			if self.integration_manager:
				await self.integration_manager.publish_event({
					"event_type": "lineage_update_requested",
					"asset_id": asset_id,
					"tenant_id": tenant_id,
					"timestamp": datetime.utcnow().isoformat(),
					"priority": "normal"
				})
		except Exception as e:
			await self._log_error(f"Failed to queue lineage update: {str(e)}")
	
	async def _find_assets_by_schema_table(self, schema_name: str, table_name: str, tenant_id: str) -> List[str]:
		"""Find assets matching schema and table name"""
		try:
			asset_ids = []
			async with self.db_manager.get_session(tenant_id) as session:
				from sqlalchemy import select, and_
				from .models import MetaAsset
				
				# Look for assets with matching schema/table information
				stmt = select(MetaAsset.id).where(
					and_(
						MetaAsset.tenant_id == tenant_id,
						MetaAsset.is_deleted == False,
						# Check if schema_info contains the schema/table
						MetaAsset.schema_info.contains({
							"schema_name": schema_name,
							"table_name": table_name
						})
					)
				)
				
				result = await session.execute(stmt)
				asset_ids = [row[0] for row in result.fetchall()]
			
			return asset_ids
			
		except Exception as e:
			await self._log_error(f"Failed to find assets by schema/table: {str(e)}")
			return []
	
	# === Code Analysis Methods ===
	
	async def _analyze_python_code(self, code: str, context: Dict[str, Any]) -> List[LineageEdge]:
		"""Analyze Python code for data lineage"""
		try:
			import ast
			import re
			
			edges = []
			tenant_id = context.get('tenant_id', 'default')
			source_asset_id = context.get('source_asset_id')
			
			# Parse Python AST for data operations
			try:
				tree = ast.parse(code)
			except SyntaxError:
				return []
			
			# Look for pandas operations
			pandas_patterns = [
				r'pd\.read_csv\([\'"]([^\'"]+)[\'"]',
				r'pd\.read_sql\([\'"]([^\'"]+)[\'"]',
				r'pd\.read_excel\([\'"]([^\'"]+)[\'"]',
				r'\.to_csv\([\'"]([^\'"]+)[\'"]',
				r'\.to_sql\([\'"]([^\'"]+)[\'"]'
			]
			
			for pattern in pandas_patterns:
				matches = re.findall(pattern, code)
				for match in matches:
					# Create lineage edge for data source/target
					if source_asset_id:
						edge = LineageEdge(
							source_asset_id=source_asset_id,
							target_asset_id=f"file_{match.replace('/', '_')}",
							lineage_type="python_data_operation",
							transformation_logic=f"Pandas operation: {pattern}",
							confidence_score=0.8,
							tenant_id=tenant_id
						)
						edges.append(edge)
			
			# Look for SQL queries in the code
			sql_patterns = [
				r'"""([^"]*SELECT[^"]*)"""',
				r"'''([^']*SELECT[^']*)'''",
				r'"([^"]*SELECT[^"]*)"',
				r"'([^']*SELECT[^']*)'",
			]
			
			for pattern in sql_patterns:
				matches = re.findall(pattern, code, re.IGNORECASE | re.MULTILINE | re.DOTALL)
				for sql_query in matches:
					tables = await self._extract_table_references(sql_query)
					for table_name in tables:
						if source_asset_id:
							edge = LineageEdge(
								source_asset_id=f"table_{table_name}",
								target_asset_id=source_asset_id,
								lineage_type="sql_query",
								transformation_logic=f"SQL in Python: {sql_query[:100]}...",
								confidence_score=0.9,
								tenant_id=tenant_id
							)
							edges.append(edge)
			
			return edges
			
		except Exception as e:
			await self._log_error(f"Python code analysis failed: {str(e)}")
			return []
	
	async def _analyze_sql_code(self, code: str, context: Dict[str, Any]) -> List[LineageEdge]:
		"""Analyze SQL code for data lineage"""
		try:
			edges = []
			tenant_id = context.get('tenant_id', 'default')
			source_asset_id = context.get('source_asset_id')
			
			# Extract table references
			source_tables = await self._extract_table_references(code)
			target_tables = await self._extract_target_tables(code)
			
			# Create edges for table dependencies
			for source_table in source_tables:
				for target_table in target_tables:
					edge = LineageEdge(
						source_asset_id=f"table_{source_table}",
						target_asset_id=f"table_{target_table}",
						lineage_type="sql_transformation",
						transformation_logic=code[:200] + "..." if len(code) > 200 else code,
						confidence_score=0.95,
						tenant_id=tenant_id
					)
					edges.append(edge)
			
			# If no target tables found but we have a source asset, use that
			if not target_tables and source_asset_id:
				for source_table in source_tables:
					edge = LineageEdge(
						source_asset_id=f"table_{source_table}",
						target_asset_id=source_asset_id,
						lineage_type="sql_dependency",
						transformation_logic=code[:200] + "..." if len(code) > 200 else code,
						confidence_score=0.9,
						tenant_id=tenant_id
					)
					edges.append(edge)
			
			return edges
			
		except Exception as e:
			await self._log_error(f"SQL code analysis failed: {str(e)}")
			return []
	
	async def _analyze_jvm_code(self, code: str, language: str, context: Dict[str, Any]) -> List[LineageEdge]:
		"""Analyze JVM-based code (Scala/Java) for data lineage"""
		try:
			import re
			
			edges = []
			tenant_id = context.get('tenant_id', 'default')
			source_asset_id = context.get('source_asset_id')
			
			# Spark/Scala patterns
			spark_patterns = [
				r'spark\.read\.(?:parquet|csv|json|table)\([\'"]([^\'"]+)[\'"]',
				r'\.write\.(?:parquet|csv|json|saveAsTable)\([\'"]([^\'"]+)[\'"]',
				r'spark\.sql\([\'"]([^\'"]+)[\'"]'
			]
			
			for pattern in spark_patterns:
				matches = re.findall(pattern, code)
				for match in matches:
					if source_asset_id:
						edge = LineageEdge(
							source_asset_id=source_asset_id,
							target_asset_id=f"spark_asset_{match.replace('.', '_')}",
							lineage_type=f"{language}_spark_operation",
							transformation_logic=f"Spark operation in {language}",
							confidence_score=0.85,
							tenant_id=tenant_id
						)
						edges.append(edge)
			
			return edges
			
		except Exception as e:
			await self._log_error(f"{language} code analysis failed: {str(e)}")
			return []
	
	async def _analyze_r_code(self, code: str, context: Dict[str, Any]) -> List[LineageEdge]:
		"""Analyze R code for data lineage"""
		try:
			import re
			
			edges = []
			tenant_id = context.get('tenant_id', 'default')
			source_asset_id = context.get('source_asset_id')
			
			# R data operation patterns
			r_patterns = [
				r'read\.csv\([\'"]([^\'"]+)[\'"]',
				r'read\.table\([\'"]([^\'"]+)[\'"]',
				r'write\.csv\([^,]+,\s*[\'"]([^\'"]+)[\'"]'
			]
			
			for pattern in r_patterns:
				matches = re.findall(pattern, code)
				for match in matches:
					if source_asset_id:
						edge = LineageEdge(
							source_asset_id=source_asset_id,
							target_asset_id=f"r_file_{match.replace('/', '_')}",
							lineage_type="r_data_operation",
							transformation_logic="R data file operation",
							confidence_score=0.8,
							tenant_id=tenant_id
						)
						edges.append(edge)
			
			return edges
			
		except Exception as e:
			await self._log_error(f"R code analysis failed: {str(e)}")
			return []
	
	async def _extract_table_references(self, sql: str) -> List[str]:
		"""Extract table references from SQL query"""
		try:
			import re
			
			# Simple regex-based table extraction
			# This is a basic implementation - production would use a proper SQL parser
			patterns = [
				r'\bFROM\s+([a-zA-Z_][a-zA-Z0-9_.]*)',
				r'\bJOIN\s+([a-zA-Z_][a-zA-Z0-9_.]*)',
				r'\bINTO\s+([a-zA-Z_][a-zA-Z0-9_.]*)',
				r'\bUPDATE\s+([a-zA-Z_][a-zA-Z0-9_.]*)',
				r'\bDELETE\s+FROM\s+([a-zA-Z_][a-zA-Z0-9_.]*)'
			]
			
			tables = set()
			for pattern in patterns:
				matches = re.findall(pattern, sql, re.IGNORECASE)
				for match in matches:
					# Clean up table name
					table_name = match.strip().lower()
					if table_name and not table_name.startswith('('):
						tables.add(table_name)
			
			return list(tables)
			
		except Exception as e:
			await self._log_error(f"Table extraction failed: {str(e)}")
			return []
	
	async def _extract_target_tables(self, sql: str) -> List[str]:
		"""Extract target tables from SQL query (INSERT, UPDATE, CREATE)"""
		try:
			import re
			
			patterns = [
				r'\bINTO\s+([a-zA-Z_][a-zA-Z0-9_.]*)',
				r'\bCREATE\s+(?:TABLE|VIEW)\s+([a-zA-Z_][a-zA-Z0-9_.]*)',
				r'\bUPDATE\s+([a-zA-Z_][a-zA-Z0-9_.]*)'
			]
			
			tables = set()
			for pattern in patterns:
				matches = re.findall(pattern, sql, re.IGNORECASE)
				for match in matches:
					table_name = match.strip().lower()
					if table_name:
						tables.add(table_name)
			
			return list(tables)
			
		except Exception as e:
			await self._log_error(f"Target table extraction failed: {str(e)}")
			return []
	
	async def _analyze_openapi_spec(self, spec: Dict[str, Any], tenant_id: str) -> List[LineageEdge]:
		"""Analyze OpenAPI specification for API lineage"""
		try:
			edges = []
			
			paths = spec.get('paths', {})
			for path, methods in paths.items():
				for method, details in methods.items():
					if isinstance(details, dict):
						# Look for data dependencies in request/response schemas
						request_schema = details.get('requestBody', {}).get('content', {})
						response_schema = details.get('responses', {}).get('200', {}).get('content', {})
						
						if request_schema or response_schema:
							edge = LineageEdge(
								source_asset_id=f"api_endpoint_{path.replace('/', '_')}",
								target_asset_id=f"api_operation_{method}_{path.replace('/', '_')}",
								lineage_type="api_schema",
								transformation_logic=f"OpenAPI {method.upper()} {path}",
								confidence_score=0.7,
								tenant_id=tenant_id,
								metadata={
									"openapi_path": path,
									"http_method": method.upper(),
									"has_request_body": bool(request_schema),
									"has_response_body": bool(response_schema)
								}
							)
							edges.append(edge)
			
			return edges
			
		except Exception as e:
			await self._log_error(f"OpenAPI spec analysis failed: {str(e)}")
			return []
	
	async def _find_asset_by_endpoint(self, endpoint: str, tenant_id: str):
		"""Find asset corresponding to API endpoint"""
		try:
			async with self.db_manager.get_session(tenant_id) as session:
				from sqlalchemy import select
				from .models import MetaAsset
				
				# Look for assets with matching endpoint information
				stmt = select(MetaAsset).where(
					MetaAsset.tenant_id == tenant_id,
					MetaAsset.is_deleted == False,
					MetaAsset.custom_attributes.contains({"endpoint": endpoint})
				)
				
				result = await session.execute(stmt)
				return result.scalar_one_or_none()
				
		except Exception as e:
			await self._log_error(f"Failed to find asset by endpoint: {str(e)}")
			return None
	
	async def _log_info(self, message: str):
		"""Log info message"""
		timestamp = datetime.utcnow().isoformat()
		print(f"[{timestamp}] META LINEAGE INFO: {message}")
	
	async def _categorize_impact_levels(self, assets: List[str], impact_levels: Dict[str, int]):
		"""Categorize asset impacts based on sophisticated analysis"""
		try:
			for asset_id in assets:
				# Get asset metadata for impact analysis
				impact_score = await self._calculate_asset_impact_score(asset_id)
				
				# Categorize based on impact score
				if impact_score >= 0.8:
					impact_levels["critical"] += 1
				elif impact_score >= 0.6:
					impact_levels["high"] += 1
				elif impact_score >= 0.3:
					impact_levels["medium"] += 1
				else:
					impact_levels["low"] += 1
					
		except Exception as e:
			await self._log_error(f"Impact level categorization failed: {str(e)}")
	
	async def _calculate_asset_impact_score(self, asset_id: str) -> float:
		"""Calculate comprehensive impact score for an asset"""
		try:
			score = 0.0
			
			async with self.db_manager.get_session() as session:
				from sqlalchemy import select
				from .models import MetaAsset, MetaQualityAssessment, MetaUserActivity
				
				# Get asset information
				stmt = select(MetaAsset).where(MetaAsset.id == asset_id)
				result = await session.execute(stmt)
				asset = result.scalar_one_or_none()
				
				if not asset:
					return 0.0
				
				# Factor 1: Asset type importance (databases > tables > views > files)
				type_scores = {
					'database': 0.3,
					'table': 0.25,
					'view': 0.2,
					'dataset': 0.25,
					'dashboard': 0.2,
					'report': 0.15,
					'file': 0.1,
					'api': 0.2
				}
				asset_type = getattr(asset, 'asset_type', 'file').lower()
				score += type_scores.get(asset_type, 0.1)
				
				# Factor 2: Quality score (higher quality = higher impact)
				if hasattr(asset, 'quality_score') and asset.quality_score:
					score += (asset.quality_score / 100.0) * 0.25
				
				# Factor 3: Update frequency (recently updated = more active/important)
				if hasattr(asset, 'updated_at') and asset.updated_at:
					days_since_update = (datetime.utcnow() - asset.updated_at).days
					if days_since_update < 1:
						score += 0.15  # Very recent
					elif days_since_update < 7:
						score += 0.1   # This week
					elif days_since_update < 30:
						score += 0.05  # This month
					# Older assets get no boost
				
				# Factor 4: Number of downstream dependencies (more dependents = higher impact)
				downstream_count = len(await self.get_downstream_assets(asset_id, asset.tenant_id, max_depth=3))
				if downstream_count > 20:
					score += 0.2
				elif downstream_count > 10:
					score += 0.15
				elif downstream_count > 5:
					score += 0.1
				elif downstream_count > 0:
					score += 0.05
				
				# Factor 5: Custom business criticality indicators
				if hasattr(asset, 'custom_attributes') and asset.custom_attributes:
					attrs = asset.custom_attributes
					if attrs.get('business_critical', False):
						score += 0.15
					if attrs.get('production_system', False):
						score += 0.1
					if attrs.get('customer_facing', False):
						score += 0.1
				
			return min(score, 1.0)  # Cap at 1.0
			
		except Exception as e:
			await self._log_error(f"Impact score calculation failed: {str(e)}")
			return 0.5  # Default neutral score
	
	async def _log_error(self, message: str):
		"""Log error message"""
		timestamp = datetime.utcnow().isoformat()
		print(f"[{timestamp}] META LINEAGE ERROR: {message}")


# Factory function for easy initialization
async def create_lineage_engine(
	db_manager: MetaDatabaseManager,
	integration_manager: APGMetadataIntegrationManager,
	config: Dict[str, Any] = None
) -> DataLineageEngine:
	"""Factory function to create and initialize lineage engine"""
	lineage_engine = DataLineageEngine(db_manager, integration_manager, config)
	await lineage_engine.initialize()
	return lineage_engine