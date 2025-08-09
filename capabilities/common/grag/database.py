"""
APG GraphRAG Capability - Database Service Layer

Revolutionary graph-based retrieval-augmented generation with Apache AGE integration.
Comprehensive database operations for knowledge graphs, entities, and relationships.

Author: Datacraft (nyimbi@gmail.com)
Copyright: © 2025 Datacraft  
Website: www.datacraft.co.ke
"""

from __future__ import annotations
import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import uuid

from sqlalchemy import and_, or_, func, text, select, update, delete, desc, asc
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import selectinload, joinedload
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError, DataError

from .models import (
	GrKnowledgeGraph, GrGraphEntity, GrGraphRelationship, GrGraphCommunity,
	GrQuery, GrResponse, GrCurationWorkflow, GrKnowledgeEdit,
	GrAnalytics, GrQueryPerformance, Base
)
from .views import (
	KnowledgeGraph, GraphEntity, GraphRelationship, GraphCommunity,
	GraphRAGQuery, GraphRAGResponse, PerformanceMetrics
)


logger = logging.getLogger(__name__)


class GraphRAGDatabaseError(Exception):
	"""Base exception for GraphRAG database operations"""
	def __init__(self, message: str, error_code: str = "DATABASE_ERROR", details: Optional[Dict[str, Any]] = None):
		super().__init__(message)
		self.error_code = error_code
		self.details = details or {}


class EntityNotFoundError(GraphRAGDatabaseError):
	"""Exception raised when entity is not found"""
	def __init__(self, entity_id: str, entity_type: str = "entity"):
		super().__init__(
			f"{entity_type.title()} with ID '{entity_id}' not found",
			"ENTITY_NOT_FOUND",
			{"entity_id": entity_id, "entity_type": entity_type}
		)


class RelationshipNotFoundError(GraphRAGDatabaseError):
	"""Exception raised when relationship is not found"""
	def __init__(self, relationship_id: str):
		super().__init__(
			f"Relationship with ID '{relationship_id}' not found",
			"RELATIONSHIP_NOT_FOUND",
			{"relationship_id": relationship_id}
		)


class GraphConsistencyError(GraphRAGDatabaseError):
	"""Exception raised when graph consistency is violated"""
	def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
		super().__init__(message, "GRAPH_CONSISTENCY_ERROR", details)


# ============================================================================
# DATABASE SERVICE CLASS
# ============================================================================

class GraphRAGDatabaseService:
	"""
	Comprehensive database service for GraphRAG operations with Apache AGE integration.
	
	Provides high-performance async operations for:
	- Knowledge graph CRUD operations
	- Entity and relationship management
	- Apache AGE graph database queries
	- Multi-hop graph traversal
	- Query caching and performance tracking
	- Multi-tenant data isolation
	"""
	
	def __init__(self, database_url: str, echo: bool = False):
		"""Initialize database service with connection pool"""
		self.database_url = database_url
		self.engine = create_async_engine(
			database_url,
			echo=echo,
			pool_size=20,
			max_overflow=30,
			pool_pre_ping=True,
			pool_recycle=3600,
			connect_args={
				"server_settings": {
					"jit": "off",  # Disable JIT for consistent performance
					"search_path": "ag_catalog,public"  # Include Apache AGE in search path
				}
			}
		)
		self.session_factory = async_sessionmaker(
			self.engine,
			class_=AsyncSession,
			expire_on_commit=False
		)
		self._graph_name = "graphrag_knowledge"
	
	@asynccontextmanager
	async def session(self):
		"""Async context manager for database sessions"""
		async with self.session_factory() as session:
			try:
				yield session
			except Exception:
				await session.rollback()
				raise
			finally:
				await session.close()
	
	async def initialize_database(self) -> None:
		"""Initialize database with Apache AGE graph"""
		async with self.engine.begin() as conn:
			# Load Apache AGE extension
			await conn.execute(text("CREATE EXTENSION IF NOT EXISTS age"))
			await conn.execute(text("LOAD 'age'"))
			await conn.execute(text("SET search_path = ag_catalog, '$user', public"))
			
			# Create GraphRAG graph if it doesn't exist
			try:
				await conn.execute(text(f"SELECT create_graph('{self._graph_name}')"))
				logger.info(f"Created Apache AGE graph: {self._graph_name}")
			except Exception as e:
				if "already exists" not in str(e):
					logger.error(f"Failed to create graph: {e}")
					raise
			
			# Create all tables
			await conn.run_sync(Base.metadata.create_all)
			logger.info("GraphRAG database initialized successfully")
	
	async def cleanup_database(self) -> None:
		"""Cleanup database connections"""
		await self.engine.dispose()
		logger.info("Database connections cleaned up")
	
	# ========================================================================
	# KNOWLEDGE GRAPH OPERATIONS
	# ========================================================================
	
	async def create_knowledge_graph(
		self,
		tenant_id: str,
		name: str,
		description: Optional[str] = None,
		metadata: Optional[Dict[str, Any]] = None
	) -> KnowledgeGraph:
		"""Create a new knowledge graph"""
		async with self.session() as session:
			try:
				# Check for duplicate names within tenant
				existing = await session.execute(
					select(GrKnowledgeGraph).where(
						and_(GrKnowledgeGraph.tenant_id == tenant_id, GrKnowledgeGraph.name == name)
					)
				)
				if existing.scalar_one_or_none():
					raise GraphRAGDatabaseError(
						f"Knowledge graph '{name}' already exists for tenant '{tenant_id}'",
						"DUPLICATE_GRAPH_NAME"
					)
				
				# Create new knowledge graph
				db_graph = GrKnowledgeGraph(
					tenant_id=tenant_id,
					name=name,
					description=description,
					metadata=metadata or {},
					quality_metrics={}
				)
				
				session.add(db_graph)
				await session.commit()
				await session.refresh(db_graph)
				
				logger.info(f"Created knowledge graph '{name}' for tenant '{tenant_id}'")
				return self._convert_db_to_pydantic_graph(db_graph)
				
			except IntegrityError as e:
				await session.rollback()
				raise GraphRAGDatabaseError(f"Failed to create knowledge graph: {e}", "INTEGRITY_ERROR")
	
	async def get_knowledge_graph(
		self,
		tenant_id: str,
		graph_id: str,
		include_stats: bool = False
	) -> KnowledgeGraph:
		"""Get knowledge graph by ID"""
		async with self.session() as session:
			# Build query with optional relationship loading
			query = select(GrKnowledgeGraph).where(
				and_(GrKnowledgeGraph.id == graph_id, GrKnowledgeGraph.tenant_id == tenant_id)
			)
			
			if include_stats:
				query = query.options(
					selectinload(GrKnowledgeGraph.entities),
					selectinload(GrKnowledgeGraph.relationships),
					selectinload(GrKnowledgeGraph.communities)
				)
			
			result = await session.execute(query)
			db_graph = result.scalar_one_or_none()
			
			if not db_graph:
				raise EntityNotFoundError(graph_id, "knowledge_graph")
			
			# Add real-time statistics if requested
			if include_stats:
				stats = await self._calculate_graph_statistics(session, graph_id)
				db_graph.metadata.update(stats)
			
			return self._convert_db_to_pydantic_graph(db_graph)
	
	async def list_knowledge_graphs(
		self,
		tenant_id: str,
		offset: int = 0,
		limit: int = 100,
		status_filter: Optional[str] = None
	) -> List[KnowledgeGraph]:
		"""List knowledge graphs for a tenant"""
		async with self.session() as session:
			query = select(GrKnowledgeGraph).where(GrKnowledgeGraph.tenant_id == tenant_id)
			
			if status_filter:
				query = query.where(GrKnowledgeGraph.status == status_filter)
			
			query = query.order_by(desc(GrKnowledgeGraph.updated_at)).offset(offset).limit(limit)
			
			result = await session.execute(query)
			db_graphs = result.scalars().all()
			
			return [self._convert_db_to_pydantic_graph(graph) for graph in db_graphs]
	
	async def update_knowledge_graph(
		self,
		tenant_id: str,
		graph_id: str,
		updates: Dict[str, Any]
	) -> KnowledgeGraph:
		"""Update knowledge graph"""
		async with self.session() as session:
			# Get existing graph
			query = select(GrKnowledgeGraph).where(
				and_(GrKnowledgeGraph.id == graph_id, GrKnowledgeGraph.tenant_id == tenant_id)
			)
			result = await session.execute(query)
			db_graph = result.scalar_one_or_none()
			
			if not db_graph:
				raise EntityNotFoundError(graph_id, "knowledge_graph")
			
			# Apply updates
			for key, value in updates.items():
				if hasattr(db_graph, key):
					setattr(db_graph, key, value)
			
			db_graph.updated_at = datetime.utcnow()
			await session.commit()
			await session.refresh(db_graph)
			
			logger.info(f"Updated knowledge graph '{graph_id}' for tenant '{tenant_id}'")
			return self._convert_db_to_pydantic_graph(db_graph)
	
	async def delete_knowledge_graph(self, tenant_id: str, graph_id: str) -> bool:
		"""Delete knowledge graph and all related data"""
		async with self.session() as session:
			# Check if graph exists
			query = select(GrKnowledgeGraph).where(
				and_(GrKnowledgeGraph.id == graph_id, GrKnowledgeGraph.tenant_id == tenant_id)
			)
			result = await session.execute(query)
			db_graph = result.scalar_one_or_none()
			
			if not db_graph:
				raise EntityNotFoundError(graph_id, "knowledge_graph")
			
			# Delete the graph (cascading will remove related entities)
			await session.delete(db_graph)
			await session.commit()
			
			logger.info(f"Deleted knowledge graph '{graph_id}' for tenant '{tenant_id}'")
			return True
	
	# ========================================================================
	# ENTITY OPERATIONS
	# ========================================================================
	
	async def create_entity(
		self,
		tenant_id: str,
		knowledge_graph_id: str,
		entity_id: str,
		entity_type: str,
		canonical_name: str,
		properties: Optional[Dict[str, Any]] = None,
		embeddings: Optional[List[float]] = None,
		aliases: Optional[List[str]] = None,
		confidence_score: float = 1.0
	) -> GraphEntity:
		"""Create a new entity in the knowledge graph"""
		async with self.session() as session:
			try:
				# Verify knowledge graph exists
				await self._verify_knowledge_graph(session, tenant_id, knowledge_graph_id)
				
				# Create database entity
				db_entity = GrGraphEntity(
					knowledge_graph_id=knowledge_graph_id,
					tenant_id=tenant_id,
					entity_id=entity_id,
					entity_type=entity_type,
					canonical_name=canonical_name,
					properties=properties or {},
					embeddings=embeddings,
					aliases=aliases or [],
					confidence_score=confidence_score
				)
				
				session.add(db_entity)
				await session.commit()
				await session.refresh(db_entity)
				
				# Create vertex in Apache AGE graph
				await self._create_age_vertex(session, db_entity)
				
				logger.info(f"Created entity '{entity_id}' in graph '{knowledge_graph_id}'")
				return self._convert_db_to_pydantic_entity(db_entity)
				
			except IntegrityError as e:
				await session.rollback()
				if "unique" in str(e).lower():
					raise GraphRAGDatabaseError(f"Entity '{entity_id}' already exists", "DUPLICATE_ENTITY")
				raise GraphRAGDatabaseError(f"Failed to create entity: {e}", "INTEGRITY_ERROR")
	
	async def get_entity(
		self,
		tenant_id: str,
		knowledge_graph_id: str,
		entity_id: str
	) -> GraphEntity:
		"""Get entity by ID"""
		async with self.session() as session:
			query = select(GrGraphEntity).where(
				and_(
					GrGraphEntity.tenant_id == tenant_id,
					GrGraphEntity.knowledge_graph_id == knowledge_graph_id,
					GrGraphEntity.entity_id == entity_id
				)
			)
			
			result = await session.execute(query)
			db_entity = result.scalar_one_or_none()
			
			if not db_entity:
				raise EntityNotFoundError(entity_id, "entity")
			
			return self._convert_db_to_pydantic_entity(db_entity)
	
	async def list_entities(
		self,
		tenant_id: str,
		knowledge_graph_id: str,
		entity_type: Optional[str] = None,
		offset: int = 0,
		limit: int = 100
	) -> List[GraphEntity]:
		"""List entities in knowledge graph"""
		async with self.session() as session:
			await self._verify_knowledge_graph(session, tenant_id, knowledge_graph_id)
			
			query = select(GrGraphEntity).where(
				and_(
					GrGraphEntity.tenant_id == tenant_id,
					GrGraphEntity.knowledge_graph_id == knowledge_graph_id,
					GrGraphEntity.status == 'active'
				)
			)
			
			if entity_type:
				query = query.where(GrGraphEntity.entity_type == entity_type)
			
			query = query.order_by(desc(GrGraphEntity.confidence_score)).offset(offset).limit(limit)
			
			result = await session.execute(query)
			db_entities = result.scalars().all()
			
			return [self._convert_db_to_pydantic_entity(entity) for entity in db_entities]
	
	async def update_entity(
		self,
		tenant_id: str,
		knowledge_graph_id: str,
		entity_id: str,
		updates: Dict[str, Any]
	) -> GraphEntity:
		"""Update entity properties"""
		async with self.session() as session:
			# Get existing entity
			query = select(GrGraphEntity).where(
				and_(
					GrGraphEntity.tenant_id == tenant_id,
					GrGraphEntity.knowledge_graph_id == knowledge_graph_id,
					GrGraphEntity.entity_id == entity_id
				)
			)
			result = await session.execute(query)
			db_entity = result.scalar_one_or_none()
			
			if not db_entity:
				raise EntityNotFoundError(entity_id, "entity")
			
			# Apply updates
			for key, value in updates.items():
				if hasattr(db_entity, key):
					setattr(db_entity, key, value)
			
			db_entity.updated_at = datetime.utcnow()
			await session.commit()
			await session.refresh(db_entity)
			
			# Update vertex in Apache AGE graph
			await self._update_age_vertex(session, db_entity)
			
			logger.info(f"Updated entity '{entity_id}' in graph '{knowledge_graph_id}'")
			return self._convert_db_to_pydantic_entity(db_entity)
	
	# ========================================================================
	# RELATIONSHIP OPERATIONS
	# ========================================================================
	
	async def create_relationship(
		self,
		tenant_id: str,
		knowledge_graph_id: str,
		relationship_id: str,
		source_entity_id: str,
		target_entity_id: str,
		relationship_type: str,
		strength: float = 1.0,
		properties: Optional[Dict[str, Any]] = None,
		confidence_score: float = 1.0
	) -> GraphRelationship:
		"""Create a new relationship between entities"""
		async with self.session() as session:
			try:
				# Verify entities exist
				source_entity = await self._get_entity_by_canonical_id(session, tenant_id, knowledge_graph_id, source_entity_id)
				target_entity = await self._get_entity_by_canonical_id(session, tenant_id, knowledge_graph_id, target_entity_id)
				
				if not source_entity or not target_entity:
					raise GraphRAGDatabaseError("Source or target entity not found", "ENTITY_NOT_FOUND")
				
				# Create database relationship
				db_relationship = GrGraphRelationship(
					knowledge_graph_id=knowledge_graph_id,
					tenant_id=tenant_id,
					relationship_id=relationship_id,
					source_entity_id=source_entity.id,
					target_entity_id=target_entity.id,
					relationship_type=relationship_type,
					strength=strength,
					properties=properties or {},
					confidence_score=confidence_score
				)
				
				session.add(db_relationship)
				await session.commit()
				await session.refresh(db_relationship)
				
				# Create edge in Apache AGE graph
				await self._create_age_edge(session, db_relationship, source_entity, target_entity)
				
				logger.info(f"Created relationship '{relationship_id}' in graph '{knowledge_graph_id}'")
				return self._convert_db_to_pydantic_relationship(db_relationship)
				
			except IntegrityError as e:
				await session.rollback()
				raise GraphRAGDatabaseError(f"Failed to create relationship: {e}", "INTEGRITY_ERROR")
	
	async def get_relationship(
		self,
		tenant_id: str,
		knowledge_graph_id: str,
		relationship_id: str
	) -> GraphRelationship:
		"""Get relationship by ID"""
		async with self.session() as session:
			query = select(GrGraphRelationship).where(
				and_(
					GrGraphRelationship.tenant_id == tenant_id,
					GrGraphRelationship.knowledge_graph_id == knowledge_graph_id,
					GrGraphRelationship.relationship_id == relationship_id
				)
			).options(
				joinedload(GrGraphRelationship.source_entity),
				joinedload(GrGraphRelationship.target_entity)
			)
			
			result = await session.execute(query)
			db_relationship = result.scalar_one_or_none()
			
			if not db_relationship:
				raise RelationshipNotFoundError(relationship_id)
			
			return self._convert_db_to_pydantic_relationship(db_relationship)
	
	async def list_relationships(
		self,
		tenant_id: str,
		knowledge_graph_id: str,
		entity_id: Optional[str] = None,
		relationship_type: Optional[str] = None,
		offset: int = 0,
		limit: int = 100
	) -> List[GraphRelationship]:
		"""List relationships in knowledge graph"""
		async with self.session() as session:
			await self._verify_knowledge_graph(session, tenant_id, knowledge_graph_id)
			
			query = select(GrGraphRelationship).where(
				and_(
					GrGraphRelationship.tenant_id == tenant_id,
					GrGraphRelationship.knowledge_graph_id == knowledge_graph_id,
					GrGraphRelationship.status == 'active'
				)
			).options(
				joinedload(GrGraphRelationship.source_entity),
				joinedload(GrGraphRelationship.target_entity)
			)
			
			# Filter by entity if specified
			if entity_id:
				entity = await self._get_entity_by_canonical_id(session, tenant_id, knowledge_graph_id, entity_id)
				if entity:
					query = query.where(
						or_(
							GrGraphRelationship.source_entity_id == entity.id,
							GrGraphRelationship.target_entity_id == entity.id
						)
					)
			
			if relationship_type:
				query = query.where(GrGraphRelationship.relationship_type == relationship_type)
			
			query = query.order_by(desc(GrGraphRelationship.strength)).offset(offset).limit(limit)
			
			result = await session.execute(query)
			db_relationships = result.scalars().all()
			
			return [self._convert_db_to_pydantic_relationship(rel) for rel in db_relationships]
	
	# ========================================================================
	# APACHE AGE GRAPH OPERATIONS
	# ========================================================================
	
	async def multi_hop_traversal(
		self,
		tenant_id: str,
		knowledge_graph_id: str,
		start_entity_id: str,
		max_hops: int = 3,
		relationship_types: Optional[List[str]] = None,
		min_strength: float = 0.0
	) -> List[Dict[str, Any]]:
		"""Perform multi-hop graph traversal using Apache AGE"""
		async with self.session() as session:
			await self._verify_knowledge_graph(session, tenant_id, knowledge_graph_id)
			
			# Get start entity
			start_entity = await self._get_entity_by_canonical_id(session, tenant_id, knowledge_graph_id, start_entity_id)
			if not start_entity:
				raise EntityNotFoundError(start_entity_id, "entity")
			
			# Build Cypher query for multi-hop traversal
			relationship_filter = ""
			if relationship_types:
				rel_types = "|".join(relationship_types)
				relationship_filter = f":{rel_types}"
			
			cypher_query = f"""
			MATCH path = (start)-[r{relationship_filter}*1..{max_hops}]-(end)
			WHERE id(start) = $start_id
			AND ALL(rel in relationships(path) WHERE rel.strength >= {min_strength})
			RETURN path, length(path) as path_length,
				   [rel in relationships(path) | rel.strength] as strengths
			ORDER BY path_length, reduce(s = 0, strength in strengths | s + strength) DESC
			LIMIT 100
			"""
			
			# Execute query with Apache AGE
			paths = await self._execute_cypher_query(
				session, 
				cypher_query,
				{"start_id": start_entity.id}
			)
			
			# Convert AGE results to structured format
			structured_paths = []
			for path_result in paths:
				structured_paths.append({
					"path_length": path_result.get("path_length", 0),
					"total_strength": sum(path_result.get("strengths", [])),
					"entities": self._extract_entities_from_path(path_result["path"]),
					"relationships": self._extract_relationships_from_path(path_result["path"])
				})
			
			logger.info(f"Found {len(structured_paths)} paths from entity '{start_entity_id}' with max {max_hops} hops")
			return structured_paths
	
	async def find_shortest_path(
		self,
		tenant_id: str,
		knowledge_graph_id: str,
		source_entity_id: str,
		target_entity_id: str
	) -> Optional[Dict[str, Any]]:
		"""Find shortest path between two entities using Apache AGE"""
		async with self.session() as session:
			await self._verify_knowledge_graph(session, tenant_id, knowledge_graph_id)
			
			# Get source and target entities
			source_entity = await self._get_entity_by_canonical_id(session, tenant_id, knowledge_graph_id, source_entity_id)
			target_entity = await self._get_entity_by_canonical_id(session, tenant_id, knowledge_graph_id, target_entity_id)
			
			if not source_entity or not target_entity:
				raise GraphRAGDatabaseError("Source or target entity not found", "ENTITY_NOT_FOUND")
			
			# Cypher query for shortest path
			cypher_query = """
			MATCH (source), (target), path = shortestPath((source)-[*]-(target))
			WHERE id(source) = $source_id AND id(target) = $target_id
			RETURN path, length(path) as path_length,
				   [rel in relationships(path) | rel.strength] as strengths
			"""
			
			# Execute query
			results = await self._execute_cypher_query(
				session,
				cypher_query,
				{"source_id": source_entity.id, "target_id": target_entity.id}
			)
			
			if not results:
				return None
			
			result = results[0]
			return {
				"path_length": result.get("path_length", 0),
				"total_strength": sum(result.get("strengths", [])),
				"entities": self._extract_entities_from_path(result["path"]),
				"relationships": self._extract_relationships_from_path(result["path"])
			}
	
	async def detect_communities(
		self,
		tenant_id: str,
		knowledge_graph_id: str,
		algorithm: str = "louvain",
		resolution: float = 1.0
	) -> List[GraphCommunity]:
		"""Detect communities in the knowledge graph"""
		async with self.session() as session:
			await self._verify_knowledge_graph(session, tenant_id, knowledge_graph_id)
			
			# Use NetworkX-based community detection for now
			# In production, this would use Apache AGE's graph algorithms
			communities_data = await self._detect_communities_networkx(
				session, tenant_id, knowledge_graph_id, algorithm, resolution
			)
			
			# Store communities in database
			communities = []
			for i, community_data in enumerate(communities_data):
				db_community = GrGraphCommunity(
					knowledge_graph_id=knowledge_graph_id,
					tenant_id=tenant_id,
					community_id=f"{algorithm}_community_{i}",
					name=f"Community {i+1}",
					algorithm=algorithm,
					members=community_data["members"],
					cohesion_score=community_data.get("cohesion_score"),
					size_metrics=community_data.get("size_metrics", {})
				)
				session.add(db_community)
				communities.append(self._convert_db_to_pydantic_community(db_community))
			
			await session.commit()
			
			logger.info(f"Detected {len(communities)} communities using {algorithm} algorithm")
			return communities
	
	# ========================================================================
	# QUERY OPERATIONS
	# ========================================================================
	
	async def create_query(
		self,
		tenant_id: str,
		knowledge_graph_id: str,
		query_text: str,
		query_type: str = "question_answering",
		query_embedding: Optional[List[float]] = None,
		context: Optional[Dict[str, Any]] = None,
		max_hops: int = 3
	) -> GraphRAGQuery:
		"""Create a new GraphRAG query"""
		async with self.session() as session:
			await self._verify_knowledge_graph(session, tenant_id, knowledge_graph_id)
			
			db_query = GrQuery(
				knowledge_graph_id=knowledge_graph_id,
				tenant_id=tenant_id,
				query_text=query_text,
				query_type=query_type,
				query_embedding=query_embedding,
				context=context or {},
				max_hops=max_hops
			)
			
			session.add(db_query)
			await session.commit()
			await session.refresh(db_query)
			
			logger.info(f"Created query '{db_query.id}' for graph '{knowledge_graph_id}'")
			return self._convert_db_to_pydantic_query(db_query)
	
	async def get_query(self, tenant_id: str, query_id: str) -> GraphRAGQuery:
		"""Get query by ID"""
		async with self.session() as session:
			query = select(GrQuery).where(
				and_(GrQuery.id == query_id, GrQuery.tenant_id == tenant_id)
			)
			
			result = await session.execute(query)
			db_query = result.scalar_one_or_none()
			
			if not db_query:
				raise EntityNotFoundError(query_id, "query")
			
			return self._convert_db_to_pydantic_query(db_query)
	
	async def update_query_status(
		self,
		tenant_id: str,
		query_id: str,
		status: str,
		processing_time_ms: Optional[int] = None
	) -> GraphRAGQuery:
		"""Update query status and processing time"""
		async with self.session() as session:
			query = select(GrQuery).where(
				and_(GrQuery.id == query_id, GrQuery.tenant_id == tenant_id)
			)
			
			result = await session.execute(query)
			db_query = result.scalar_one_or_none()
			
			if not db_query:
				raise EntityNotFoundError(query_id, "query")
			
			db_query.status = status
			if processing_time_ms is not None:
				db_query.processing_time_ms = processing_time_ms
			
			if status == "completed":
				db_query.completed_at = datetime.utcnow()
			
			await session.commit()
			await session.refresh(db_query)
			
			return self._convert_db_to_pydantic_query(db_query)
	
	# ========================================================================
	# ANALYTICS AND PERFORMANCE
	# ========================================================================
	
	async def record_performance_metrics(
		self,
		tenant_id: str,
		query_id: str,
		metrics: PerformanceMetrics
	) -> None:
		"""Record performance metrics for a query"""
		async with self.session() as session:
			db_performance = GrQueryPerformance(
				query_id=query_id,
				tenant_id=tenant_id,
				retrieval_time_ms=metrics.retrieval_time_ms,
				reasoning_time_ms=metrics.reasoning_time_ms,
				generation_time_ms=metrics.generation_time_ms,
				total_time_ms=metrics.total_time_ms,
				entities_retrieved=metrics.entities_retrieved,
				relationships_traversed=metrics.relationships_traversed,
				graph_hops=metrics.graph_hops,
				memory_usage_mb=metrics.memory_usage_mb,
				cache_hits=metrics.cache_hits,
				cache_misses=metrics.cache_misses,
				model_tokens=metrics.model_tokens
			)
			
			session.add(db_performance)
			await session.commit()
			
			logger.info(f"Recorded performance metrics for query '{query_id}'")
	
	async def get_graph_statistics(
		self,
		tenant_id: str,
		knowledge_graph_id: str
	) -> Dict[str, Any]:
		"""Get comprehensive statistics for a knowledge graph"""
		async with self.session() as session:
			stats = await self._calculate_graph_statistics(session, knowledge_graph_id)
			
			# Add performance statistics
			performance_stats = await self._calculate_performance_statistics(session, knowledge_graph_id)
			stats.update(performance_stats)
			
			return stats
	
	# ========================================================================
	# HELPER METHODS
	# ========================================================================
	
	async def _verify_knowledge_graph(
		self,
		session: AsyncSession,
		tenant_id: str,
		knowledge_graph_id: str
	) -> GrKnowledgeGraph:
		"""Verify knowledge graph exists and belongs to tenant"""
		query = select(GrKnowledgeGraph).where(
			and_(
				GrKnowledgeGraph.id == knowledge_graph_id,
				GrKnowledgeGraph.tenant_id == tenant_id
			)
		)
		
		result = await session.execute(query)
		db_graph = result.scalar_one_or_none()
		
		if not db_graph:
			raise EntityNotFoundError(knowledge_graph_id, "knowledge_graph")
		
		return db_graph
	
	async def _get_entity_by_canonical_id(
		self,
		session: AsyncSession,
		tenant_id: str,
		knowledge_graph_id: str,
		entity_id: str
	) -> Optional[GrGraphEntity]:
		"""Get entity by canonical entity ID"""
		query = select(GrGraphEntity).where(
			and_(
				GrGraphEntity.tenant_id == tenant_id,
				GrGraphEntity.knowledge_graph_id == knowledge_graph_id,
				GrGraphEntity.entity_id == entity_id
			)
		)
		
		result = await session.execute(query)
		return result.scalar_one_or_none()
	
	async def _calculate_graph_statistics(
		self,
		session: AsyncSession,
		knowledge_graph_id: str
	) -> Dict[str, Any]:
		"""Calculate real-time statistics for a knowledge graph"""
		# Count entities
		entity_count_query = select(func.count(GrGraphEntity.id)).where(
			and_(
				GrGraphEntity.knowledge_graph_id == knowledge_graph_id,
				GrGraphEntity.status == 'active'
			)
		)
		entity_count = await session.scalar(entity_count_query)
		
		# Count relationships
		rel_count_query = select(func.count(GrGraphRelationship.id)).where(
			and_(
				GrGraphRelationship.knowledge_graph_id == knowledge_graph_id,
				GrGraphRelationship.status == 'active'
			)
		)
		rel_count = await session.scalar(rel_count_query)
		
		# Count communities
		comm_count_query = select(func.count(GrGraphCommunity.id)).where(
			GrGraphCommunity.knowledge_graph_id == knowledge_graph_id
		)
		comm_count = await session.scalar(comm_count_query)
		
		# Calculate average confidence scores
		avg_entity_confidence_query = select(func.avg(GrGraphEntity.confidence_score)).where(
			and_(
				GrGraphEntity.knowledge_graph_id == knowledge_graph_id,
				GrGraphEntity.status == 'active'
			)
		)
		avg_entity_confidence = await session.scalar(avg_entity_confidence_query) or 0.0
		
		avg_rel_confidence_query = select(func.avg(GrGraphRelationship.confidence_score)).where(
			and_(
				GrGraphRelationship.knowledge_graph_id == knowledge_graph_id,
				GrGraphRelationship.status == 'active'
			)
		)
		avg_rel_confidence = await session.scalar(avg_rel_confidence_query) or 0.0
		
		return {
			"total_entities": entity_count or 0,
			"total_relationships": rel_count or 0,
			"total_communities": comm_count or 0,
			"average_entity_confidence": float(avg_entity_confidence),
			"average_relationship_confidence": float(avg_rel_confidence),
			"last_updated": datetime.utcnow().isoformat()
		}
	
	async def _calculate_performance_statistics(
		self,
		session: AsyncSession,
		knowledge_graph_id: str
	) -> Dict[str, Any]:
		"""Calculate performance statistics for a knowledge graph"""
		# Get queries for this graph in the last 24 hours
		yesterday = datetime.utcnow() - timedelta(days=1)
		
		recent_queries_query = select(GrQuery).where(
			and_(
				GrQuery.knowledge_graph_id == knowledge_graph_id,
				GrQuery.created_at >= yesterday
			)
		).options(selectinload(GrQuery.performance_logs))
		
		result = await session.execute(recent_queries_query)
		recent_queries = result.scalars().all()
		
		if not recent_queries:
			return {"recent_performance": {}}
		
		# Calculate performance metrics
		total_queries = len(recent_queries)
		completed_queries = len([q for q in recent_queries if q.status == "completed"])
		
		performance_logs = []
		for query in recent_queries:
			performance_logs.extend(query.performance_logs)
		
		if performance_logs:
			avg_total_time = sum(p.total_time_ms for p in performance_logs if p.total_time_ms) / len(performance_logs)
			avg_entities_retrieved = sum(p.entities_retrieved for p in performance_logs if p.entities_retrieved) / len(performance_logs)
		else:
			avg_total_time = 0
			avg_entities_retrieved = 0
		
		return {
			"recent_performance": {
				"total_queries_24h": total_queries,
				"completed_queries_24h": completed_queries,
				"success_rate": completed_queries / total_queries if total_queries > 0 else 0,
				"average_response_time_ms": avg_total_time,
				"average_entities_retrieved": avg_entities_retrieved
			}
		}
	
	# Additional Apache AGE helper methods would be implemented here
	async def _create_age_vertex(self, session: AsyncSession, entity: GrGraphEntity) -> None:
		"""Create vertex in Apache AGE graph"""
		# Implementation would create vertex using Apache AGE functions
		pass
	
	async def _create_age_edge(
		self,
		session: AsyncSession,
		relationship: GrGraphRelationship,
		source_entity: GrGraphEntity,
		target_entity: GrGraphEntity
	) -> None:
		"""Create edge in Apache AGE graph"""
		# Implementation would create edge using Apache AGE functions
		pass
	
	async def _update_age_vertex(self, session: AsyncSession, entity: GrGraphEntity) -> None:
		"""Update vertex in Apache AGE graph"""
		# Implementation would update vertex properties
		pass
	
	async def _execute_cypher_query(
		self,
		session: AsyncSession,
		query: str,
		parameters: Optional[Dict[str, Any]] = None
	) -> List[Dict[str, Any]]:
		"""Execute Cypher query using Apache AGE"""
		# Implementation would execute Cypher queries
		return []
	
	async def _detect_communities_networkx(
		self,
		session: AsyncSession,
		tenant_id: str,
		knowledge_graph_id: str,
		algorithm: str,
		resolution: float
	) -> List[Dict[str, Any]]:
		"""Detect communities using NetworkX algorithms"""
		# Implementation would use NetworkX for community detection
		return []
	
	def _extract_entities_from_path(self, path: Any) -> List[str]:
		"""Extract entity IDs from Apache AGE path result"""
		# Implementation would parse Apache AGE path format
		return []
	
	def _extract_relationships_from_path(self, path: Any) -> List[str]:
		"""Extract relationship IDs from Apache AGE path result"""
		# Implementation would parse Apache AGE path format
		return []
	
	# Conversion methods
	def _convert_db_to_pydantic_graph(self, db_graph: GrKnowledgeGraph) -> KnowledgeGraph:
		"""Convert database model to Pydantic model"""
		return KnowledgeGraph(
			graph_id=str(db_graph.id),
			tenant_id=db_graph.tenant_id,
			name=db_graph.name,
			description=db_graph.description,
			schema_version=db_graph.schema_version,
			metadata=db_graph.metadata,
			quality_metrics=db_graph.quality_metrics,
			status=db_graph.status,
			created_at=db_graph.created_at,
			updated_at=db_graph.updated_at
		)
	
	def _convert_db_to_pydantic_entity(self, db_entity: GrGraphEntity) -> GraphEntity:
		"""Convert database entity to Pydantic model"""
		return GraphEntity(
			entity_id=str(db_entity.id),
			tenant_id=db_entity.tenant_id,
			knowledge_graph_id=str(db_entity.knowledge_graph_id),
			canonical_entity_id=db_entity.entity_id,
			entity_type=db_entity.entity_type,
			canonical_name=db_entity.canonical_name,
			aliases=db_entity.aliases,
			properties=db_entity.properties,
			embeddings=db_entity.embeddings,
			confidence_score=float(db_entity.confidence_score),
			evidence_sources=db_entity.evidence_sources,
			provenance=db_entity.provenance,
			quality_score=float(db_entity.quality_score),
			status=db_entity.status,
			created_at=db_entity.created_at,
			updated_at=db_entity.updated_at
		)
	
	def _convert_db_to_pydantic_relationship(self, db_rel: GrGraphRelationship) -> GraphRelationship:
		"""Convert database relationship to Pydantic model"""
		return GraphRelationship(
			relationship_id=str(db_rel.id),
			tenant_id=db_rel.tenant_id,
			knowledge_graph_id=str(db_rel.knowledge_graph_id),
			canonical_relationship_id=db_rel.relationship_id,
			source_entity_id=str(db_rel.source_entity_id),
			target_entity_id=str(db_rel.target_entity_id),
			relationship_type=db_rel.relationship_type,
			strength=float(db_rel.strength),
			context=db_rel.context,
			properties=db_rel.properties,
			evidence_sources=db_rel.evidence_sources,
			provenance=db_rel.provenance,
			temporal_validity=db_rel.temporal_validity,
			confidence_score=float(db_rel.confidence_score),
			quality_score=float(db_rel.quality_score),
			status=db_rel.status,
			created_at=db_rel.created_at,
			updated_at=db_rel.updated_at
		)
	
	def _convert_db_to_pydantic_community(self, db_comm: GrGraphCommunity) -> GraphCommunity:
		"""Convert database community to Pydantic model"""
		return GraphCommunity(
			community_id=str(db_comm.id),
			tenant_id=db_comm.tenant_id,
			knowledge_graph_id=str(db_comm.knowledge_graph_id),
			canonical_community_id=db_comm.community_id,
			name=db_comm.name,
			description=db_comm.description,
			algorithm=db_comm.algorithm,
			members=db_comm.members,
			centrality_metrics=db_comm.centrality_metrics,
			cohesion_score=float(db_comm.cohesion_score) if db_comm.cohesion_score else None,
			size_metrics=db_comm.size_metrics,
			created_at=db_comm.created_at,
			updated_at=db_comm.updated_at
		)
	
	def _convert_db_to_pydantic_query(self, db_query: GrQuery) -> GraphRAGQuery:
		"""Convert database query to Pydantic model"""
		return GraphRAGQuery(
			query_id=str(db_query.id),
			tenant_id=db_query.tenant_id,
			knowledge_graph_id=str(db_query.knowledge_graph_id),
			query_text=db_query.query_text,
			query_type=db_query.query_type,
			query_embedding=db_query.query_embedding,
			context=db_query.context,
			max_hops=db_query.max_hops,
			status=db_query.status,
			processing_time_ms=db_query.processing_time_ms,
			created_at=db_query.created_at,
			completed_at=db_query.completed_at
		)


# ============================================================================
# EXPORTED FUNCTIONS
# ============================================================================

def create_database_service(database_url: str, echo: bool = False) -> GraphRAGDatabaseService:
	"""Factory function to create database service"""
	return GraphRAGDatabaseService(database_url, echo)


__all__ = [
	'GraphRAGDatabaseService',
	'GraphRAGDatabaseError',
	'EntityNotFoundError',
	'RelationshipNotFoundError',
	'GraphConsistencyError',
	'create_database_service',
]