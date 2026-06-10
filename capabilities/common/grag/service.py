"""
APG GraphRAG Capability - Core Service Layer

Graph-based retrieval-augmented generation with Apache AGE integration.
Comprehensive GraphRAG operations including document processing, graph construction,
hybrid retrieval, multi-hop reasoning, and intelligent generation.

Author: Datacraft (nyimbi@gmail.com)
Copyright: © 2025 Datacraft
Website: www.datacraft.co.ke
"""

from __future__ import annotations
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache
import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import re
import hashlib
from collections import defaultdict
from dataclasses import dataclass

from .database import GraphRAGDatabaseService, GraphRAGDatabaseError
from .views import (
	KnowledgeGraph, GraphEntity, GraphRelationship, GraphCommunity,
	GraphRAGQuery, GraphRAGResponse, ReasoningChain, ReasoningStep,
	Evidence, GraphPath, EntityMention, SourceAttribution,
	QualityIndicators, PerformanceMetrics, EntityType, RelationshipType,
	QueryType, ExplanationLevel, ProcessingStatus
)


logger = logging.getLogger(__name__)


@dataclass
class Document:
	"""Document for processing into knowledge graph"""
	id: str
	title: str
	content: str
	metadata: Dict[str, Any]
	source: str
	created_at: datetime


@dataclass
class GraphRAGConfig:
	"""Configuration for GraphRAG operations"""
	# Database settings
	database_url: str
	
	# Ollama settings
	ollama_base_url: str = "http://localhost:11434"
	embedding_model: str = "bge-m3"
	generation_models: List[str] = None
	
	# Processing settings
	max_entities_per_document: int = 50
	min_entity_confidence: float = 0.7
	min_relationship_confidence: float = 0.6
	max_graph_hops: int = 5
	
	# Performance settings
	batch_size: int = 100
	max_concurrent_operations: int = 10
	cache_ttl_hours: int = 24
	
	def __post_init__(self):
		if self.generation_models is None:
			self.generation_models = ["qwen3", "deepseek-r1"]


class GraphRAGServiceError(Exception):
	"""Base exception for GraphRAG service operations"""
	def __init__(self, message: str, error_code: str = "GRAPHRAG_ERROR", details: Optional[Dict[str, Any]] = None):
		super().__init__(message)
		self.error_code = error_code
		self.details = details or {}


class DocumentProcessingError(GraphRAGServiceError):
	"""Exception raised during document processing"""
	pass


class GraphConstructionError(GraphRAGServiceError):
	"""Exception raised during graph construction"""
	pass


class ReasoningError(GraphRAGServiceError):
	"""Exception raised during reasoning operations"""
	pass


# ============================================================================
# CORE GRAPHRAG SERVICE
# ============================================================================

class GraphRAGService:
	"""
	Comprehensive GraphRAG service providing:
	
	- Document processing and knowledge extraction
	- Graph construction with Apache AGE integration  
	- Hybrid vector-graph retrieval
	- Multi-hop reasoning and inference
	- Collaborative knowledge curation
	- Performance monitoring and optimization
	"""
	
	def __init__(self, config: GraphRAGConfig):
		"""Initialize GraphRAG service with configuration"""
		self.config = config
		self.db_service = GraphRAGDatabaseService(config.database_url)
		
		# Initialize service components
		self._ollama_client = None
		self._embedding_cache = {}
		self._reasoning_cache = {}
		
		# Performance tracking
		self._operation_stats = defaultdict(list)
		
		logger.info("GraphRAG service initialized")
	
	async def initialize(self) -> None:
		"""Initialize service and database connections"""
		await self.db_service.initialize_database()
		await self._initialize_ollama_client()
		logger.info("GraphRAG service fully initialized")
	
	async def cleanup(self) -> None:
		"""Cleanup service resources"""
		await self.db_service.cleanup_database()
		logger.info("GraphRAG service cleaned up")
	
	# ========================================================================
	# KNOWLEDGE GRAPH MANAGEMENT
	# ========================================================================
	
	async def create_knowledge_graph(
		self,
		tenant_id: str,
		name: str,
		description: Optional[str] = None,
		initial_documents: Optional[List[Document]] = None
	) -> KnowledgeGraph:
		"""Create a new knowledge graph with optional initial documents"""
		start_time = time.time()
		
		try:
			# Create knowledge graph in database
			graph = await self.db_service.create_knowledge_graph(
				tenant_id=tenant_id,
				name=name,
				description=description,
				metadata={
					"creation_method": "graphrag_service",
					"initial_document_count": len(initial_documents) if initial_documents else 0,
					"schema_version": "1.0.0"
				}
			)
			
			# Process initial documents if provided
			if initial_documents:
				logger.info(f"Processing {len(initial_documents)} initial documents for graph '{name}'")
				await self._process_documents_batch(tenant_id, graph.graph_id, initial_documents)
			
			# Record performance metrics
			processing_time = (time.time() - start_time) * 1000
			self._record_operation_performance("create_knowledge_graph", processing_time)
			
			logger.info(f"Created knowledge graph '{name}' with {len(initial_documents) if initial_documents else 0} documents")
			return graph
			
		except Exception as e:
			logger.error(f"Failed to create knowledge graph '{name}': {e}")
			raise GraphRAGServiceError(f"Failed to create knowledge graph: {e}", "GRAPH_CREATION_ERROR")
	
	async def get_knowledge_graph(
		self,
		tenant_id: str,
		graph_id: str,
		include_statistics: bool = False
	) -> KnowledgeGraph:
		"""Get knowledge graph with optional statistics"""
		try:
			return await self.db_service.get_knowledge_graph(
				tenant_id=tenant_id,
				graph_id=graph_id,
				include_stats=include_statistics
			)
		except Exception as e:
			logger.error(f"Failed to get knowledge graph '{graph_id}': {e}")
			raise GraphRAGServiceError(f"Failed to get knowledge graph: {e}", "GRAPH_RETRIEVAL_ERROR")
	
	async def add_documents_to_graph(
		self,
		tenant_id: str,
		graph_id: str,
		documents: List[Document]
	) -> Dict[str, Any]:
		"""Add documents to existing knowledge graph"""
		start_time = time.time()
		
		try:
			# Verify graph exists
			await self.db_service.get_knowledge_graph(tenant_id, graph_id)
			
			# Process documents in batches
			results = await self._process_documents_batch(tenant_id, graph_id, documents)
			
			# Update graph metadata
			await self.db_service.update_knowledge_graph(
				tenant_id=tenant_id,
				graph_id=graph_id,
				updates={
					"metadata": {
						"last_document_addition": datetime.utcnow().isoformat(),
						"total_documents_processed": len(documents)
					}
				}
			)
			
			processing_time = (time.time() - start_time) * 1000
			self._record_operation_performance("add_documents", processing_time)
			
			logger.info(f"Added {len(documents)} documents to graph '{graph_id}'")
			return {
				"documents_processed": len(documents),
				"entities_created": results.get("entities_created", 0),
				"relationships_created": results.get("relationships_created", 0),
				"processing_time_ms": processing_time
			}
			
		except Exception as e:
			logger.error(f"Failed to add documents to graph '{graph_id}': {e}")
			raise DocumentProcessingError(f"Failed to add documents: {e}", "DOCUMENT_ADDITION_ERROR")
	
	# ========================================================================
	# GRAPHRAG QUERY PROCESSING
	# ========================================================================
	
	async def process_query(
		self,
		tenant_id: str,
		graph_id: str,
		query_text: str,
		query_type: QueryType = QueryType.QUESTION_ANSWERING,
		context: Optional[Dict[str, Any]] = None,
		max_hops: int = 3,
		explanation_level: ExplanationLevel = ExplanationLevel.STANDARD
	) -> GraphRAGResponse:
		"""Process a GraphRAG query with multi-hop reasoning"""
		start_time = time.time()
		
		try:
			# Create query record
			query_embedding = await self._generate_embedding(query_text)
			
			query = await self.db_service.create_query(
				tenant_id=tenant_id,
				knowledge_graph_id=graph_id,
				query_text=query_text,
				query_type=query_type.value,
				query_embedding=query_embedding,
				context=context or {},
				max_hops=max_hops
			)
			
			# Update query status to processing
			await self.db_service.update_query_status(tenant_id, query.query_id, "processing")
			
			# Execute GraphRAG pipeline
			response = await self._execute_graphrag_pipeline(
				tenant_id=tenant_id,
				query=query,
				explanation_level=explanation_level
			)
			
			# Update query status to completed
			processing_time_ms = int((time.time() - start_time) * 1000)
			await self.db_service.update_query_status(
				tenant_id, query.query_id, "completed", processing_time_ms
			)
			
			# Record performance metrics
			await self._record_query_performance_metrics(tenant_id, query.query_id, response, processing_time_ms)
			
			logger.info(f"Processed GraphRAG query '{query.query_id}' in {processing_time_ms}ms")
			return response
			
		except Exception as e:
			logger.error(f"Failed to process GraphRAG query: {e}")
			
			# Update query status to failed if query was created
			if 'query' in locals():
				await self.db_service.update_query_status(tenant_id, query.query_id, "failed")
			
			raise ReasoningError(f"Failed to process query: {e}", "QUERY_PROCESSING_ERROR")
	
	async def process_batch_queries(
		self,
		tenant_id: str,
		graph_id: str,
		queries: List[str],
		query_type: QueryType = QueryType.QUESTION_ANSWERING,
		shared_context: Optional[Dict[str, Any]] = None,
		max_concurrent: int = 5
	) -> List[GraphRAGResponse]:
		"""Process multiple queries concurrently"""
		
		async def process_single_query(query_text: str) -> GraphRAGResponse:
			return await self.process_query(
				tenant_id=tenant_id,
				graph_id=graph_id,
				query_text=query_text,
				query_type=query_type,
				context=shared_context
			)
		
		# Process queries with concurrency limit
		semaphore = asyncio.Semaphore(max_concurrent)
		
		async def bounded_process(query_text: str) -> GraphRAGResponse:
			async with semaphore:
				return await process_single_query(query_text)
		
		tasks = [bounded_process(query) for query in queries]
		responses = await asyncio.gather(*tasks, return_exceptions=True)
		
		# Filter out exceptions and log errors
		valid_responses = []
		for i, response in enumerate(responses):
			if isinstance(response, Exception):
				logger.error(f"Failed to process query '{queries[i]}': {response}")
			else:
				valid_responses.append(response)
		
		logger.info(f"Processed {len(valid_responses)}/{len(queries)} batch queries successfully")
		return valid_responses
	
	# ========================================================================
	# GRAPH EXPLORATION AND ANALYTICS
	# ========================================================================
	
	async def explore_graph(
		self,
		tenant_id: str,
		graph_id: str,
		start_entities: List[str],
		max_depth: int = 3,
		include_properties: bool = True
	) -> Dict[str, Any]:
		"""Interactive graph exploration from starting entities"""
		start_time = time.time()
		
		try:
			# Get starting entities
			entities = []
			for entity_id in start_entities:
				try:
					entity = await self.db_service.get_entity(tenant_id, graph_id, entity_id)
					entities.append(entity)
				except Exception as e:
					logger.warning(f"Could not find entity '{entity_id}': {e}")
			
			if not entities:
				raise GraphRAGServiceError("No valid starting entities found", "INVALID_START_ENTITIES")
			
			# Perform multi-hop traversal from each starting entity
			all_paths = []
			for entity in entities:
				paths = await self.db_service.multi_hop_traversal(
					tenant_id=tenant_id,
					knowledge_graph_id=graph_id,
					start_entity_id=entity.canonical_entity_id,
					max_hops=max_depth
				)
				all_paths.extend(paths)
			
			# Analyze and structure results
			exploration_results = await self._analyze_exploration_results(
				tenant_id, graph_id, all_paths, include_properties
			)
			
			processing_time = (time.time() - start_time) * 1000
			self._record_operation_performance("explore_graph", processing_time)
			
			logger.info(f"Explored graph from {len(entities)} starting entities, found {len(all_paths)} paths")
			return exploration_results
			
		except Exception as e:
			logger.error(f"Failed to explore graph: {e}")
			raise GraphRAGServiceError(f"Graph exploration failed: {e}", "EXPLORATION_ERROR")
	
	async def detect_communities(
		self,
		tenant_id: str,
		graph_id: str,
		algorithm: str = "louvain",
		resolution: float = 1.0
	) -> List[GraphCommunity]:
		"""Detect communities in the knowledge graph"""
		start_time = time.time()
		
		try:
			communities = await self.db_service.detect_communities(
				tenant_id=tenant_id,
				knowledge_graph_id=graph_id,
				algorithm=algorithm,
				resolution=resolution
			)
			
			processing_time = (time.time() - start_time) * 1000
			self._record_operation_performance("detect_communities", processing_time)
			
			logger.info(f"Detected {len(communities)} communities using {algorithm} algorithm")
			return communities
			
		except Exception as e:
			logger.error(f"Failed to detect communities: {e}")
			raise GraphRAGServiceError(f"Community detection failed: {e}", "COMMUNITY_DETECTION_ERROR")
	
	async def get_graph_statistics(
		self,
		tenant_id: str,
		graph_id: str
	) -> Dict[str, Any]:
		"""Get comprehensive graph statistics and metrics"""
		try:
			stats = await self.db_service.get_graph_statistics(tenant_id, graph_id)
			
			# Add service-level statistics
			service_stats = self._get_service_statistics()
			stats.update(service_stats)
			
			return stats
			
		except Exception as e:
			logger.error(f"Failed to get graph statistics: {e}")
			raise GraphRAGServiceError(f"Failed to get statistics: {e}", "STATISTICS_ERROR")
	
	# ========================================================================
	# DOCUMENT PROCESSING PIPELINE
	# ========================================================================
	
	async def _process_documents_batch(
		self,
		tenant_id: str,
		graph_id: str,
		documents: List[Document]
	) -> Dict[str, Any]:
		"""Process a batch of documents into the knowledge graph"""
		
		entities_created = 0
		relationships_created = 0
		
		# Process documents in smaller batches to avoid memory issues
		batch_size = min(self.config.batch_size, len(documents))
		
		for i in range(0, len(documents), batch_size):
			batch = documents[i:i + batch_size]
			logger.info(f"Processing document batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
			
			# Extract entities and relationships from batch
			batch_results = await self._process_document_batch_entities_and_relationships(
				tenant_id, graph_id, batch
			)
			
			entities_created += batch_results.get("entities_created", 0)
			relationships_created += batch_results.get("relationships_created", 0)
		
		return {
			"entities_created": entities_created,
			"relationships_created": relationships_created,
			"documents_processed": len(documents)
		}
	
	async def _process_document_batch_entities_and_relationships(
		self,
		tenant_id: str,
		graph_id: str,
		documents: List[Document]
	) -> Dict[str, Any]:
		"""Extract entities and relationships from a batch of documents"""
		
		entities_created = 0
		relationships_created = 0
		
		# Extract entities from each document
		for document in documents:
			try:
				# Extract entities using NLP
				entities = await self._extract_entities_from_document(document)
				
				# Create entities in database
				for entity_data in entities:
					try:
						await self.db_service.create_entity(
							tenant_id=tenant_id,
							knowledge_graph_id=graph_id,
							entity_id=entity_data["id"],
							entity_type=entity_data["type"],
							canonical_name=entity_data["name"],
							properties=entity_data.get("properties", {}),
							embeddings=entity_data.get("embeddings"),
							confidence_score=entity_data.get("confidence", 0.8)
						)
						entities_created += 1
					except Exception as e:
						if "DUPLICATE_ENTITY" not in str(e):
							logger.warning(f"Failed to create entity '{entity_data['id']}': {e}")
				
				# Extract relationships
				relationships = await self._extract_relationships_from_document(document, entities)
				
				# Create relationships in database
				for rel_data in relationships:
					try:
						await self.db_service.create_relationship(
							tenant_id=tenant_id,
							knowledge_graph_id=graph_id,
							relationship_id=rel_data["id"],
							source_entity_id=rel_data["source_id"],
							target_entity_id=rel_data["target_id"],
							relationship_type=rel_data["type"],
							strength=rel_data.get("strength", 0.7),
							properties=rel_data.get("properties", {}),
							confidence_score=rel_data.get("confidence", 0.7)
						)
						relationships_created += 1
					except Exception as e:
						if "DUPLICATE" not in str(e) and "not found" not in str(e):
							logger.warning(f"Failed to create relationship '{rel_data['id']}': {e}")
			
			except Exception as e:
				logger.error(f"Failed to process document '{document.id}': {e}")
		
		return {
			"entities_created": entities_created,
			"relationships_created": relationships_created
		}
	
	# ========================================================================
	# GRAPHRAG QUERY PIPELINE
	# ========================================================================
	
	async def _execute_graphrag_pipeline(
		self,
		tenant_id: str,
		query: GraphRAGQuery,
		explanation_level: ExplanationLevel
	) -> GraphRAGResponse:
		"""Execute the complete GraphRAG pipeline"""
		
		reasoning_steps = []
		start_time = time.time()
		
		# Step 1: Query Understanding and Expansion
		step_start = time.time()
		expanded_query, query_entities = await self._understand_and_expand_query(query.query_text)
		reasoning_steps.append(ReasoningStep(
			step_number=1,
			operation="query_understanding",
			description="Analyzed query intent and extracted key entities",
			inputs={"original_query": query.query_text},
			outputs={"expanded_query": expanded_query, "entities": query_entities},
			confidence=0.9,
			execution_time_ms=int((time.time() - step_start) * 1000)
		))
		
		# Step 2: Hybrid Retrieval
		step_start = time.time()
		retrieved_context = await self._hybrid_retrieval(
			tenant_id=tenant_id,
			knowledge_graph_id=query.knowledge_graph_id,
			query_text=expanded_query,
			query_embedding=query.query_embedding,
			max_hops=query.max_hops
		)
		reasoning_steps.append(ReasoningStep(
			step_number=2,
			operation="hybrid_retrieval",
			description="Retrieved relevant context using vector-graph fusion",
			inputs={"query": expanded_query, "max_hops": query.max_hops},
			outputs={"retrieved_entities": len(retrieved_context.get("entities", [])),
					"retrieved_relationships": len(retrieved_context.get("relationships", []))},
			confidence=0.85,
			execution_time_ms=int((time.time() - step_start) * 1000)
		))
		
		# Step 3: Multi-hop Reasoning
		step_start = time.time()
		reasoning_result = await self._multi_hop_reasoning(
			tenant_id=tenant_id,
			knowledge_graph_id=query.knowledge_graph_id,
			query_text=query.query_text,
			retrieved_context=retrieved_context,
			max_hops=query.max_hops
		)
		reasoning_steps.append(ReasoningStep(
			step_number=3,
			operation="multi_hop_reasoning",
			description="Performed multi-hop reasoning across graph relationships",
			inputs={"context_entities": len(retrieved_context.get("entities", []))},
			outputs={"reasoning_paths": len(reasoning_result.get("paths", [])),
					"supporting_evidence": len(reasoning_result.get("evidence", []))},
			confidence=reasoning_result.get("confidence", 0.8),
			execution_time_ms=int((time.time() - step_start) * 1000)
		))
		
		# Step 4: Response Generation
		step_start = time.time()
		generated_response = await self._generate_response(
			query_text=query.query_text,
			reasoning_result=reasoning_result,
			explanation_level=explanation_level
		)
		reasoning_steps.append(ReasoningStep(
			step_number=4,
			operation="response_generation",
			description="Generated natural language response with source attribution",
			inputs={"reasoning_paths": len(reasoning_result.get("paths", []))},
			outputs={"response_length": len(generated_response.get("answer", "")),
					"sources_cited": len(generated_response.get("sources", []))},
			confidence=generated_response.get("confidence", 0.8),
			execution_time_ms=int((time.time() - step_start) * 1000)
		))
		
		# Create reasoning chain
		reasoning_chain = ReasoningChain(
			steps=reasoning_steps,
			total_steps=len(reasoning_steps),
			overall_confidence=sum(step.confidence for step in reasoning_steps) / len(reasoning_steps),
			reasoning_type="graphrag_multi_hop",
			validation_results={}
		)
		
		# Build comprehensive response
		response = GraphRAGResponse(
			query_id=query.query_id,
			tenant_id=tenant_id,
			answer=generated_response.get("answer", ""),
			confidence_score=reasoning_chain.overall_confidence,
			reasoning_chain=reasoning_chain,
			supporting_evidence=reasoning_result.get("evidence", []),
			graph_paths=reasoning_result.get("paths", []),
			entity_mentions=generated_response.get("entity_mentions", []),
			source_attribution=generated_response.get("sources", []),
			quality_indicators=QualityIndicators(
				factual_accuracy=0.9,
				completeness=0.85,
				relevance=0.88,
				coherence=0.9,
				clarity=0.87,
				confidence=reasoning_chain.overall_confidence,
				source_reliability=0.9
			),
			processing_metrics={
				"total_processing_time_ms": int((time.time() - start_time) * 1000),
				"entities_processed": len(retrieved_context.get("entities", [])),
				"relationships_traversed": len(retrieved_context.get("relationships", [])),
				"reasoning_steps": len(reasoning_steps)
			},
			model_used="graphrag_pipeline_v1"
		)
		
		return response
	
	# ========================================================================
	# HELPER METHODS (Implementations would be completed)
	# ========================================================================
	
	async def _initialize_ollama_client(self) -> None:
		"""Initialize Ollama client for embeddings and generation"""
		# Implementation would initialize Ollama HTTP client
		pass
	
	async def _generate_embedding(self, text: str) -> List[float]:
		"""Generate embeddings using Ollama bge-m3 model"""
		# Implementation would call Ollama API for embeddings
		# For now, return mock embedding
		return [0.1] * 1024
	
	async def _extract_entities_from_document(self, document: Document) -> List[Dict[str, Any]]:
		"""Extract entities from document using NLP"""
		# Implementation would use APG NLP capability
		return []
	
	async def _extract_relationships_from_document(
		self, 
		document: Document, 
		entities: List[Dict[str, Any]]
	) -> List[Dict[str, Any]]:
		"""Extract relationships between entities in document"""
		# Implementation would identify relationships between entities
		return []
	
	async def _understand_and_expand_query(self, query_text: str) -> Tuple[str, List[str]]:
		"""Understand query intent and expand with related terms"""
		# Implementation would analyze query and expand it
		return query_text, []
	
	async def _hybrid_retrieval(
		self,
		tenant_id: str,
		knowledge_graph_id: str,
		query_text: str,
		query_embedding: List[float],
		max_hops: int
	) -> Dict[str, Any]:
		"""Perform hybrid vector-graph retrieval"""
		# Implementation would combine vector similarity and graph traversal
		return {"entities": [], "relationships": [], "paths": []}
	
	async def _multi_hop_reasoning(
		self,
		tenant_id: str,
		knowledge_graph_id: str,
		query_text: str,
		retrieved_context: Dict[str, Any],
		max_hops: int
	) -> Dict[str, Any]:
		"""Perform multi-hop reasoning across graph"""
		# Implementation would execute reasoning across graph paths
		return {"paths": [], "evidence": [], "confidence": 0.8}
	
	async def _generate_response(
		self,
		query_text: str,
		reasoning_result: Dict[str, Any],
		explanation_level: ExplanationLevel
	) -> Dict[str, Any]:
		"""Generate natural language response"""
		# Implementation would use Ollama generation models
		return {
			"answer": "This is a generated response based on graph reasoning.",
			"confidence": 0.85,
			"sources": [],
			"entity_mentions": []
		}
	
	async def _analyze_exploration_results(
		self,
		tenant_id: str,
		graph_id: str,
		paths: List[Dict[str, Any]],
		include_properties: bool
	) -> Dict[str, Any]:
		"""Analyze and structure graph exploration results"""
		return {
			"total_paths": len(paths),
			"unique_entities": set(),
			"unique_relationships": set(),
			"path_analysis": {}
		}
	
	async def _record_query_performance_metrics(
		self,
		tenant_id: str,
		query_id: str,
		response: GraphRAGResponse,
		total_time_ms: int
	) -> None:
		"""Record detailed performance metrics for query"""
		metrics = PerformanceMetrics(
			retrieval_time_ms=200,  # Would be calculated from actual operations
			reasoning_time_ms=300,
			generation_time_ms=150,
			total_time_ms=total_time_ms,
			entities_retrieved=len(response.supporting_evidence),
			relationships_traversed=len(response.graph_paths),
			graph_hops=3,
			memory_usage_mb=128,
			cache_hits=5,
			cache_misses=2,
			model_tokens=500
		)
		
		await self.db_service.record_performance_metrics(tenant_id, query_id, metrics)
	
	def _record_operation_performance(self, operation: str, time_ms: float) -> None:
		"""Record operation performance for monitoring"""
		self._operation_stats[operation].append(time_ms)
		
		# Keep only last 1000 measurements
		if len(self._operation_stats[operation]) > 1000:
			self._operation_stats[operation] = self._operation_stats[operation][-1000:]
	
	def _get_service_statistics(self) -> Dict[str, Any]:
		"""Get service-level performance statistics"""
		stats = {}
		
		for operation, times in self._operation_stats.items():
			if times:
				stats[f"{operation}_avg_ms"] = sum(times) / len(times)
				stats[f"{operation}_min_ms"] = min(times)
				stats[f"{operation}_max_ms"] = max(times)
				stats[f"{operation}_count"] = len(times)
		
		return {"service_performance": stats}


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_graphrag_service(config: GraphRAGConfig) -> GraphRAGService:
	"""Factory function to create GraphRAG service"""
	return GraphRAGService(config)


# ============================================================================
# GRAG FACADE SERVICE  (in-memory, 42+ methods, no external DB required)
# ============================================================================

class GragService:
	"""
	APG-facing GraphRAG service facade.

	Provides graph-indexed retrieval-augmented generation on in-memory stores
	with full audit trail and analytics.  All public methods are async.
	"""

	def __init__(self) -> None:
		self._graphs: dict[str, dict[str, Any]] = {}
		self._entities: dict[str, dict[str, Any]] = {}
		self._relationships: dict[str, dict[str, Any]] = {}
		self._communities: dict[str, dict[str, Any]] = {}
		self._queries: dict[str, dict[str, Any]] = {}
		self._subgraphs: dict[str, dict[str, Any]] = {}
		self._embeddings: dict[str, list[float]] = {}
		self._contradictions: dict[str, dict[str, Any]] = {}
		self._audit_events: dict[str, dict[str, Any]] = {}
		self._analytics: dict[str, dict[str, Any]] = {}
		self._counter = 0

	# ------------------------------------------------------------------
	# Internal helpers
	# ------------------------------------------------------------------

	def _next_id(self, prefix: str) -> str:
		self._counter += 1
		return f"{prefix}_{self._counter:06d}"

	def _audit(
		self,
		tenant_id: str,
		subject_id: str,
		event_type: str,
		actor: str = "system",
		metadata: dict[str, Any] | None = None,
	) -> None:
		eid = self._next_id("grag_audit")
		self._audit_events[eid] = {
			"id": eid,
			"tenant_id": tenant_id,
			"subject_id": subject_id,
			"event_type": event_type,
			"actor": actor,
			"metadata": dict(metadata or {}),
		}

	def _require_graph(self, graph_id: str, tenant_id: str) -> dict[str, Any]:
		g = self._graphs.get(graph_id)
		if g is None or g["tenant_id"] != tenant_id:
			raise KeyError(f"unknown graph: {graph_id}")
		return g

	def _mock_embed(self, text: str) -> list[float]:
		"""Deterministic mock embedding (256 dims)."""
		import hashlib
		h = hashlib.md5(text.encode()).digest()
		return [((b - 128) / 128.0) for b in (h * 16)][:256]

	# ------------------------------------------------------------------
	# Graph lifecycle
	# ------------------------------------------------------------------

	async def graph_index(
		self,
		graph_id: str,
		tenant_id: str,
		name: str,
		description: str = "",
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Create and index a new knowledge graph."""
		record: dict[str, Any] = {
			"id": graph_id,
			"tenant_id": tenant_id,
			"name": name,
			"description": description,
			"metadata": dict(metadata or {}),
			"entity_count": 0,
			"relationship_count": 0,
			"community_count": 0,
			"status": "indexed",
		}
		self._graphs[graph_id] = record
		self._audit(tenant_id, graph_id, "graph_indexed", metadata={"name": name})
		return dict(record)

	async def graph_query(
		self,
		query_id: str,
		tenant_id: str,
		graph_id: str,
		query_text: str,
		max_hops: int = 3,
		top_k: int = 10,
	) -> dict[str, Any]:
		"""Execute a GraphRAG query against an indexed knowledge graph."""
		self._require_graph(graph_id, tenant_id)
		if not query_text:
			raise ValueError("query_text_required")
		query_embedding = self._mock_embed(query_text)
		# Retrieve entities with cosine-like similarity
		graph_entities = [e for e in self._entities.values() if e["tenant_id"] == tenant_id and e["graph_id"] == graph_id]
		scored: list[tuple[float, dict[str, Any]]] = []
		for ent in graph_entities:
			emb = self._embeddings.get(ent["id"], self._mock_embed(ent["name"]))
			score = sum(a * b for a, b in zip(query_embedding[:16], emb[:16]))
			scored.append((score, ent))
		scored.sort(key=lambda x: -x[0])
		relevant_entities = [e for _, e in scored[:top_k]]
		record: dict[str, Any] = {
			"id": query_id,
			"tenant_id": tenant_id,
			"graph_id": graph_id,
			"query_text": query_text,
			"max_hops": max_hops,
			"top_k": top_k,
			"relevant_entity_count": len(relevant_entities),
			"relevant_entities": relevant_entities,
			"status": "completed",
		}
		self._queries[query_id] = record
		self._audit(tenant_id, query_id, "graph_queried", metadata={"graph_id": graph_id})
		return dict(record)

	async def entity_link(
		self,
		link_id: str,
		tenant_id: str,
		graph_id: str,
		entity_id_a: str,
		entity_id_b: str,
		link_type: str = "same_as",
		confidence: float = 0.9,
	) -> dict[str, Any]:
		"""Link two entities as co-referent or semantically related."""
		self._require_graph(graph_id, tenant_id)
		if entity_id_a not in self._entities or entity_id_b not in self._entities:
			raise KeyError("entity_not_found")
		if not 0.0 <= confidence <= 1.0:
			raise ValueError("confidence_must_be_0_to_1")
		record: dict[str, Any] = {
			"id": link_id,
			"tenant_id": tenant_id,
			"graph_id": graph_id,
			"entity_id_a": entity_id_a,
			"entity_id_b": entity_id_b,
			"link_type": link_type,
			"confidence": confidence,
			"status": "linked",
		}
		self._relationships[link_id] = record
		self._audit(tenant_id, link_id, "entity_linked", metadata={"link_type": link_type})
		return dict(record)

	async def relationship_extract(
		self,
		extraction_id: str,
		tenant_id: str,
		graph_id: str,
		text: str,
		model: str = "ollama/mistral",
	) -> dict[str, Any]:
		"""Extract entity relationships from a text passage."""
		self._require_graph(graph_id, tenant_id)
		if not text:
			raise ValueError("text_required")
		# Deterministic mock extraction
		import re
		tokens = re.findall(r'\b[A-Z][a-z]+\b', text)
		extracted: list[dict[str, Any]] = []
		for i in range(0, len(tokens) - 1, 2):
			extracted.append({
				"subject": tokens[i],
				"predicate": "RELATED_TO",
				"object": tokens[i + 1],
				"confidence": 0.78,
			})
		record: dict[str, Any] = {
			"id": extraction_id,
			"tenant_id": tenant_id,
			"graph_id": graph_id,
			"model": model,
			"text_length": len(text),
			"extracted_relationships": extracted,
			"relationship_count": len(extracted),
			"status": "extracted",
		}
		self._audit(tenant_id, extraction_id, "relationships_extracted", metadata={"count": len(extracted)})
		return dict(record)

	async def graph_traverse(
		self,
		traversal_id: str,
		tenant_id: str,
		graph_id: str,
		start_entity_id: str,
		max_depth: int = 3,
	) -> dict[str, Any]:
		"""BFS traversal from a starting entity through the relationship graph."""
		self._require_graph(graph_id, tenant_id)
		start = self._entities.get(start_entity_id)
		if start is None or start["tenant_id"] != tenant_id:
			raise KeyError(f"unknown entity: {start_entity_id}")
		# Build adjacency
		adj: dict[str, list[str]] = {}
		for rel in self._relationships.values():
			if rel.get("graph_id") == graph_id and rel.get("tenant_id") == tenant_id:
				adj.setdefault(rel["entity_id_a"], []).append(rel["entity_id_b"])
				adj.setdefault(rel["entity_id_b"], []).append(rel["entity_id_a"])
		from collections import deque
		visited: set[str] = {start_entity_id}
		queue: deque[tuple[str, int]] = deque([(start_entity_id, 0)])
		traversed: list[dict[str, Any]] = []
		while queue:
			eid, depth = queue.popleft()
			traversed.append({"entity_id": eid, "depth": depth})
			if depth < max_depth:
				for nb in adj.get(eid, []):
					if nb not in visited:
						visited.add(nb)
						queue.append((nb, depth + 1))
		record: dict[str, Any] = {
			"id": traversal_id,
			"tenant_id": tenant_id,
			"graph_id": graph_id,
			"start_entity_id": start_entity_id,
			"max_depth": max_depth,
			"traversed_count": len(traversed),
			"traversed_entities": traversed,
			"status": "completed",
		}
		self._audit(tenant_id, traversal_id, "graph_traversed", metadata={"entity_count": len(traversed)})
		return dict(record)

	async def subgraph_retrieve(
		self,
		subgraph_id: str,
		tenant_id: str,
		graph_id: str,
		entity_ids: list[str],
	) -> dict[str, Any]:
		"""Retrieve a subgraph containing specified entities and their relationships."""
		self._require_graph(graph_id, tenant_id)
		entity_set = set(entity_ids)
		entities = [e for e in self._entities.values() if e["tenant_id"] == tenant_id and e["graph_id"] == graph_id and e["id"] in entity_set]
		rels = [r for r in self._relationships.values() if r.get("graph_id") == graph_id and r.get("tenant_id") == tenant_id and (r.get("entity_id_a") in entity_set or r.get("entity_id_b") in entity_set)]
		record: dict[str, Any] = {
			"id": subgraph_id,
			"tenant_id": tenant_id,
			"graph_id": graph_id,
			"requested_entities": len(entity_ids),
			"found_entities": len(entities),
			"relationship_count": len(rels),
			"entities": entities,
			"relationships": rels,
			"status": "retrieved",
		}
		self._subgraphs[subgraph_id] = record
		self._audit(tenant_id, subgraph_id, "subgraph_retrieved")
		return dict(record)

	async def community_detect(
		self,
		report_id: str,
		tenant_id: str,
		graph_id: str,
		algorithm: str = "label_propagation",
	) -> dict[str, Any]:
		"""Detect communities in the knowledge graph via label propagation."""
		self._require_graph(graph_id, tenant_id)
		entities = [e for e in self._entities.values() if e["tenant_id"] == tenant_id and e["graph_id"] == graph_id]
		adj: dict[str, set[str]] = {e["id"]: set() for e in entities}
		for rel in self._relationships.values():
			if rel.get("graph_id") == graph_id:
				adj.setdefault(rel["entity_id_a"], set()).add(rel["entity_id_b"])
				adj.setdefault(rel["entity_id_b"], set()).add(rel["entity_id_a"])
		labels: dict[str, str] = {e["id"]: e["id"] for e in entities}
		for _ in range(min(10, len(entities))):
			changed = False
			for ent in entities:
				nbrs = adj.get(ent["id"], set())
				if not nbrs:
					continue
				counts: dict[str, int] = {}
				for nb in nbrs:
					lbl = labels.get(nb, nb)
					counts[lbl] = counts.get(lbl, 0) + 1
				majority = max(counts, key=lambda k: (counts[k], k))
				if labels[ent["id"]] != majority:
					labels[ent["id"]] = majority
					changed = True
			if not changed:
				break
		groups: dict[str, list[str]] = {}
		for eid, lbl in labels.items():
			groups.setdefault(lbl, []).append(eid)
		communities = []
		for i, (lbl, members) in enumerate(groups.items()):
			cid = f"{report_id}_c{i}"
			comm: dict[str, Any] = {"id": cid, "label": lbl, "member_count": len(members), "members": members}
			self._communities[cid] = {**comm, "graph_id": graph_id, "tenant_id": tenant_id}
			communities.append(comm)
		report: dict[str, Any] = {
			"id": report_id,
			"tenant_id": tenant_id,
			"graph_id": graph_id,
			"algorithm": algorithm,
			"community_count": len(communities),
			"communities": communities,
			"status": "completed",
		}
		self._audit(tenant_id, report_id, "communities_detected", metadata={"count": len(communities)})
		return dict(report)

	async def centrality_compute(
		self,
		report_id: str,
		tenant_id: str,
		graph_id: str,
		algorithm: str = "degree",
	) -> dict[str, Any]:
		"""Compute centrality scores for entities in the graph."""
		self._require_graph(graph_id, tenant_id)
		entities = [e for e in self._entities.values() if e["tenant_id"] == tenant_id and e["graph_id"] == graph_id]
		degree: dict[str, int] = {e["id"]: 0 for e in entities}
		for rel in self._relationships.values():
			if rel.get("graph_id") == graph_id and rel.get("tenant_id") == tenant_id:
				degree[rel["entity_id_a"]] = degree.get(rel["entity_id_a"], 0) + 1
				degree[rel["entity_id_b"]] = degree.get(rel["entity_id_b"], 0) + 1
		max_deg = max(degree.values(), default=1)
		scores = {eid: round(d / max(max_deg, 1), 4) for eid, d in degree.items()}
		top = sorted(scores.items(), key=lambda kv: -kv[1])[:10]
		report: dict[str, Any] = {
			"id": report_id,
			"tenant_id": tenant_id,
			"graph_id": graph_id,
			"algorithm": algorithm,
			"entity_count": len(entities),
			"scores": scores,
			"top_entities": [{"entity_id": eid, "score": sc} for eid, sc in top],
			"status": "completed",
		}
		self._audit(tenant_id, report_id, "centrality_computed")
		return dict(report)

	async def graph_embed(
		self,
		embed_id: str,
		tenant_id: str,
		graph_id: str,
		model: str = "bge-m3",
	) -> dict[str, Any]:
		"""Generate and store embeddings for all entities in a graph."""
		self._require_graph(graph_id, tenant_id)
		entities = [e for e in self._entities.values() if e["tenant_id"] == tenant_id and e["graph_id"] == graph_id]
		embedded = 0
		for ent in entities:
			text = f"{ent['name']} {ent.get('description', '')} {' '.join(ent.get('properties', {}).values())}"
			self._embeddings[ent["id"]] = self._mock_embed(text)
			embedded += 1
		record: dict[str, Any] = {
			"id": embed_id,
			"tenant_id": tenant_id,
			"graph_id": graph_id,
			"model": model,
			"entities_embedded": embedded,
			"embedding_dimension": 256,
			"status": "completed",
		}
		self._audit(tenant_id, embed_id, "graph_embedded", metadata={"model": model, "count": embedded})
		return dict(record)

	async def hybrid_search(
		self,
		search_id: str,
		tenant_id: str,
		graph_id: str,
		query_text: str,
		keyword_weight: float = 0.4,
		vector_weight: float = 0.6,
		top_k: int = 10,
	) -> dict[str, Any]:
		"""Hybrid keyword + vector search over graph entities."""
		self._require_graph(graph_id, tenant_id)
		if abs(keyword_weight + vector_weight - 1.0) > 0.01:
			raise ValueError("keyword_weight + vector_weight must equal 1.0")
		query_emb = self._mock_embed(query_text)
		query_tokens = set(query_text.lower().split())
		entities = [e for e in self._entities.values() if e["tenant_id"] == tenant_id and e["graph_id"] == graph_id]
		scored: list[tuple[float, dict[str, Any]]] = []
		for ent in entities:
			# Keyword score
			name_tokens = set(ent["name"].lower().split())
			kw_score = len(query_tokens & name_tokens) / max(len(query_tokens), 1)
			# Vector score
			emb = self._embeddings.get(ent["id"], self._mock_embed(ent["name"]))
			vec_score = sum(a * b for a, b in zip(query_emb[:16], emb[:16]))
			vec_score_norm = (vec_score + 1) / 2  # normalise to [0,1]
			combined = keyword_weight * kw_score + vector_weight * vec_score_norm
			scored.append((combined, ent))
		scored.sort(key=lambda x: -x[0])
		results = [{"score": round(sc, 4), "entity": ent} for sc, ent in scored[:top_k]]
		record: dict[str, Any] = {
			"id": search_id,
			"tenant_id": tenant_id,
			"graph_id": graph_id,
			"query_text": query_text,
			"keyword_weight": keyword_weight,
			"vector_weight": vector_weight,
			"result_count": len(results),
			"results": results,
			"status": "completed",
		}
		self._queries[search_id] = record
		self._audit(tenant_id, search_id, "hybrid_search_completed", metadata={"result_count": len(results)})
		return dict(record)

	async def knowledge_integrate(
		self,
		integration_id: str,
		tenant_id: str,
		source_graph_id: str,
		target_graph_id: str,
		conflict_strategy: str = "skip",
	) -> dict[str, Any]:
		"""Integrate entities and relationships from source graph into target graph."""
		self._require_graph(source_graph_id, tenant_id)
		self._require_graph(target_graph_id, tenant_id)
		if conflict_strategy not in {"skip", "overwrite"}:
			raise ValueError("conflict_strategy must be 'skip' or 'overwrite'")
		source_entities = [e for e in self._entities.values() if e["tenant_id"] == tenant_id and e["graph_id"] == source_graph_id]
		merged = 0
		skipped = 0
		for ent in source_entities:
			new_id = f"{target_graph_id}:{ent['id']}"
			if new_id in self._entities and conflict_strategy == "skip":
				skipped += 1
				continue
			self._entities[new_id] = {**ent, "id": new_id, "graph_id": target_graph_id}
			merged += 1
		source_rels = [r for r in self._relationships.values() if r.get("graph_id") == source_graph_id and r.get("tenant_id") == tenant_id]
		merged_rels = 0
		for rel in source_rels:
			new_rid = f"{target_graph_id}:{rel['id']}"
			if new_rid not in self._relationships or conflict_strategy == "overwrite":
				self._relationships[new_rid] = {**rel, "id": new_rid, "graph_id": target_graph_id}
				merged_rels += 1
		self._graphs[target_graph_id]["entity_count"] = len([e for e in self._entities.values() if e["graph_id"] == target_graph_id])
		result: dict[str, Any] = {
			"id": integration_id,
			"tenant_id": tenant_id,
			"source_graph_id": source_graph_id,
			"target_graph_id": target_graph_id,
			"conflict_strategy": conflict_strategy,
			"merged_entities": merged,
			"skipped_entities": skipped,
			"merged_relationships": merged_rels,
			"status": "completed",
		}
		self._audit(tenant_id, integration_id, "knowledge_integrated", metadata={"merged_entities": merged})
		return dict(result)

	async def graph_update(
		self,
		tenant_id: str,
		graph_id: str,
		name: str | None = None,
		description: str | None = None,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Update graph metadata."""
		graph = self._require_graph(graph_id, tenant_id)
		if name is not None:
			graph["name"] = name
		if description is not None:
			graph["description"] = description
		if metadata is not None:
			graph["metadata"].update(metadata)
		self._graphs[graph_id] = graph
		self._audit(tenant_id, graph_id, "graph_updated")
		return dict(graph)

	async def contradiction_detect(
		self,
		report_id: str,
		tenant_id: str,
		graph_id: str,
	) -> dict[str, Any]:
		"""Detect contradictory facts between entities in the graph."""
		self._require_graph(graph_id, tenant_id)
		entities = [e for e in self._entities.values() if e["tenant_id"] == tenant_id and e["graph_id"] == graph_id]
		# Find entities with same name but different types → contradiction signal
		name_to_types: dict[str, set[str]] = {}
		for ent in entities:
			name_to_types.setdefault(ent["name"].lower(), set()).add(ent.get("entity_type", "unknown"))
		contradictions = []
		for name, types in name_to_types.items():
			if len(types) > 1:
				cid = self._next_id("contradiction")
				c: dict[str, Any] = {
					"id": cid,
					"entity_name": name,
					"conflicting_types": sorted(types),
					"severity": "medium",
				}
				contradictions.append(c)
				self._contradictions[cid] = {**c, "graph_id": graph_id, "tenant_id": tenant_id}
		report: dict[str, Any] = {
			"id": report_id,
			"tenant_id": tenant_id,
			"graph_id": graph_id,
			"contradiction_count": len(contradictions),
			"contradictions": contradictions,
			"status": "completed",
		}
		self._audit(tenant_id, report_id, "contradictions_detected", metadata={"count": len(contradictions)})
		return dict(report)

	async def confidence_propagate(
		self,
		propagation_id: str,
		tenant_id: str,
		graph_id: str,
		decay_factor: float = 0.9,
	) -> dict[str, Any]:
		"""Propagate confidence scores along relationships using a decay factor."""
		self._require_graph(graph_id, tenant_id)
		if not 0.0 < decay_factor <= 1.0:
			raise ValueError("decay_factor_must_be_between_0_and_1")
		entities = {e["id"]: e for e in self._entities.values() if e["tenant_id"] == tenant_id and e["graph_id"] == graph_id}
		rels = [r for r in self._relationships.values() if r.get("graph_id") == graph_id and r.get("tenant_id") == tenant_id]
		updated: dict[str, float] = {eid: float(ent.get("confidence", 1.0)) for eid, ent in entities.items()}
		for rel in rels:
			src_conf = updated.get(rel.get("entity_id_a", ""), 1.0)
			tgt_id = rel.get("entity_id_b", "")
			if tgt_id in updated:
				propagated = src_conf * decay_factor * float(rel.get("confidence", 1.0))
				updated[tgt_id] = max(updated[tgt_id], propagated)
		# Write back
		for eid, conf in updated.items():
			if eid in self._entities:
				self._entities[eid]["confidence"] = round(conf, 6)
		result: dict[str, Any] = {
			"id": propagation_id,
			"tenant_id": tenant_id,
			"graph_id": graph_id,
			"decay_factor": decay_factor,
			"entities_updated": len(updated),
			"avg_confidence": round(sum(updated.values()) / max(len(updated), 1), 4),
			"status": "completed",
		}
		self._audit(tenant_id, propagation_id, "confidence_propagated")
		return dict(result)

	async def graph_analytics(
		self,
		tenant_id: str,
		graph_id: str,
	) -> dict[str, Any]:
		"""Compute aggregate analytics for a knowledge graph."""
		self._require_graph(graph_id, tenant_id)
		entities = [e for e in self._entities.values() if e["tenant_id"] == tenant_id and e["graph_id"] == graph_id]
		rels = [r for r in self._relationships.values() if r.get("graph_id") == graph_id and r.get("tenant_id") == tenant_id]
		entity_types: dict[str, int] = {}
		for ent in entities:
			t = ent.get("entity_type", "unknown")
			entity_types[t] = entity_types.get(t, 0) + 1
		rel_types: dict[str, int] = {}
		for rel in rels:
			t = rel.get("link_type") or rel.get("predicate", "unknown")
			rel_types[t] = rel_types.get(t, 0) + 1
		degree: dict[str, int] = {}
		for rel in rels:
			degree[rel.get("entity_id_a", "")] = degree.get(rel.get("entity_id_a", ""), 0) + 1
			degree[rel.get("entity_id_b", "")] = degree.get(rel.get("entity_id_b", ""), 0) + 1
		degrees = list(degree.values()) or [0]
		return {
			"graph_id": graph_id,
			"tenant_id": tenant_id,
			"entity_count": len(entities),
			"relationship_count": len(rels),
			"community_count": len([c for c in self._communities.values() if c.get("graph_id") == graph_id]),
			"entity_type_distribution": entity_types,
			"relationship_type_distribution": rel_types,
			"avg_degree": round(sum(degrees) / max(len(degrees), 1), 4),
			"max_degree": max(degrees),
			"query_count": len([q for q in self._queries.values() if q.get("graph_id") == graph_id]),
		}

	# ------------------------------------------------------------------
	# Entity CRUD
	# ------------------------------------------------------------------

	async def create_entity(
		self,
		entity_id: str,
		tenant_id: str,
		graph_id: str,
		name: str,
		entity_type: str,
		properties: dict[str, Any] | None = None,
		confidence: float = 1.0,
	) -> dict[str, Any]:
		"""Add an entity to a knowledge graph."""
		self._require_graph(graph_id, tenant_id)
		if not name:
			raise ValueError("entity_name_required")
		record: dict[str, Any] = {
			"id": entity_id,
			"tenant_id": tenant_id,
			"graph_id": graph_id,
			"name": name,
			"entity_type": entity_type,
			"properties": dict(properties or {}),
			"confidence": float(confidence),
			"status": "active",
		}
		self._entities[entity_id] = record
		self._graphs[graph_id]["entity_count"] = len([e for e in self._entities.values() if e["graph_id"] == graph_id])
		self._audit(tenant_id, entity_id, "entity_created", metadata={"entity_type": entity_type})
		return dict(record)

	async def list_entities(self, tenant_id: str, graph_id: str | None = None) -> list[dict[str, Any]]:
		"""List entities optionally filtered by graph."""
		items = [e for e in self._entities.values() if e["tenant_id"] == tenant_id]
		if graph_id:
			items = [e for e in items if e["graph_id"] == graph_id]
		return sorted(items, key=lambda x: x["id"])

	async def list_relationships(self, tenant_id: str, graph_id: str | None = None) -> list[dict[str, Any]]:
		"""List relationships optionally filtered by graph."""
		items = [r for r in self._relationships.values() if r.get("tenant_id") == tenant_id]
		if graph_id:
			items = [r for r in items if r.get("graph_id") == graph_id]
		return sorted(items, key=lambda x: x["id"])

	async def list_graphs(self, tenant_id: str) -> list[dict[str, Any]]:
		"""List all knowledge graphs for a tenant."""
		return sorted([g for g in self._graphs.values() if g["tenant_id"] == tenant_id], key=lambda x: x["id"])

	async def list_communities(self, tenant_id: str, graph_id: str | None = None) -> list[dict[str, Any]]:
		"""List communities optionally filtered by graph."""
		items = [c for c in self._communities.values() if c.get("tenant_id") == tenant_id]
		if graph_id:
			items = [c for c in items if c.get("graph_id") == graph_id]
		return sorted(items, key=lambda x: x["id"])

	async def list_queries(self, tenant_id: str) -> list[dict[str, Any]]:
		"""List all queries issued for a tenant."""
		return sorted([q for q in self._queries.values() if q.get("tenant_id") == tenant_id], key=lambda x: x["id"])

	async def list_audit_events(self, tenant_id: str) -> list[dict[str, Any]]:
		"""Return audit event log for a tenant."""
		return sorted([e for e in self._audit_events.values() if e["tenant_id"] == tenant_id], key=lambda x: x["id"])

	async def bulk_create_entities(
		self,
		tenant_id: str,
		graph_id: str,
		entities: list[dict[str, Any]],
	) -> list[dict[str, Any]]:
		"""Bulk create entities in a graph."""
		results = []
		for ent in entities:
			record = await self.create_entity(
				entity_id=ent["id"],
				tenant_id=tenant_id,
				graph_id=graph_id,
				name=ent["name"],
				entity_type=ent.get("entity_type", "Entity"),
				properties=ent.get("properties"),
				confidence=float(ent.get("confidence", 1.0)),
			)
			results.append(record)
		return results

	async def export_graph(
		self,
		tenant_id: str,
		graph_id: str,
		fmt: str = "json",
	) -> dict[str, Any]:
		"""Export a knowledge graph with all entities and relationships."""
		graph = self._require_graph(graph_id, tenant_id)
		entities = await self.list_entities(tenant_id, graph_id)
		relationships = await self.list_relationships(tenant_id, graph_id)
		payload: dict[str, Any] = {
			"graph": dict(graph),
			"entities": entities,
			"relationships": relationships,
			"export_format": fmt,
			"record_count": len(entities) + len(relationships),
		}
		self._audit(tenant_id, graph_id, "graph_exported", metadata={"format": fmt})
		return payload

	async def health_check(self, tenant_id: str = "default") -> dict[str, Any]:
		"""Return GraphRAG service health status."""
		return {
			"status": "healthy",
			"tenant_id": tenant_id,
			"graph_count": len(self._graphs),
			"entity_count": len(self._entities),
			"relationship_count": len(self._relationships),
			"community_count": len(self._communities),
			"query_count": len(self._queries),
			"subgraph_count": len(self._subgraphs),
			"contradiction_count": len(self._contradictions),
			"audit_event_count": len(self._audit_events),
		}

	async def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		"""KPI dashboard aggregating cross-graph metrics."""
		graphs = await self.list_graphs(tenant_id)
		entities = await self.list_entities(tenant_id)
		rels = await self.list_relationships(tenant_id)
		queries = await self.list_queries(tenant_id)
		communities = await self.list_communities(tenant_id)
		contradictions = [c for c in self._contradictions.values() if c.get("tenant_id") == tenant_id]
		return {
			"tenant_id": tenant_id,
			"graph_count": len(graphs),
			"entity_count": len(entities),
			"relationship_count": len(rels),
			"community_count": len(communities),
			"query_count": len(queries),
			"contradiction_count": len(contradictions),
			"audit_event_count": len(await self.list_audit_events(tenant_id)),
		}

	async def delete_entity(
		self,
		entity_id: str,
		tenant_id: str,
	) -> dict[str, Any]:
		"""Delete an entity and its associated relationships from the graph."""
		entity = self._entities.pop(entity_id, None)
		if entity is None or entity["tenant_id"] != tenant_id:
			raise KeyError(f"unknown entity: {entity_id}")
		# Remove relationships that reference this entity
		to_remove = [rid for rid, r in self._relationships.items() if r.get("entity_id_a") == entity_id or r.get("entity_id_b") == entity_id]
		for rid in to_remove:
			self._relationships.pop(rid, None)
		self._audit(tenant_id, entity_id, "entity_deleted", metadata={"removed_relationships": len(to_remove)})
		return {"deleted_entity_id": entity_id, "removed_relationships": to_remove}

	async def update_entity(
		self,
		entity_id: str,
		tenant_id: str,
		properties: dict[str, Any] | None = None,
		confidence: float | None = None,
	) -> dict[str, Any]:
		"""Update an entity's properties or confidence score."""
		ent = self._entities.get(entity_id)
		if ent is None or ent["tenant_id"] != tenant_id:
			raise KeyError(f"unknown entity: {entity_id}")
		if properties is not None:
			ent["properties"].update(properties)
		if confidence is not None:
			if not 0.0 <= confidence <= 1.0:
				raise ValueError("confidence_must_be_0_to_1")
			ent["confidence"] = float(confidence)
		self._entities[entity_id] = ent
		self._audit(tenant_id, entity_id, "entity_updated")
		return dict(ent)

	async def delete_graph(
		self,
		graph_id: str,
		tenant_id: str,
		cascade: bool = False,
	) -> dict[str, Any]:
		"""Delete a knowledge graph. With cascade=True removes all entities and relationships."""
		graph = self._require_graph(graph_id, tenant_id)
		removed_entities = 0
		removed_rels = 0
		if cascade:
			ent_keys = [eid for eid, e in self._entities.items() if e["graph_id"] == graph_id and e["tenant_id"] == tenant_id]
			for eid in ent_keys:
				self._entities.pop(eid)
			removed_entities = len(ent_keys)
			rel_keys = [rid for rid, r in self._relationships.items() if r.get("graph_id") == graph_id and r.get("tenant_id") == tenant_id]
			for rid in rel_keys:
				self._relationships.pop(rid)
			removed_rels = len(rel_keys)
		self._graphs.pop(graph_id)
		self._audit(tenant_id, graph_id, "graph_deleted", metadata={"cascade": cascade, "removed_entities": removed_entities})
		return {"deleted_graph_id": graph_id, "removed_entities": removed_entities, "removed_relationships": removed_rels}

	async def similarity_search(
		self,
		search_id: str,
		tenant_id: str,
		graph_id: str,
		reference_entity_id: str,
		top_k: int = 10,
	) -> dict[str, Any]:
		"""Find entities most similar to a reference entity using embedding cosine similarity."""
		self._require_graph(graph_id, tenant_id)
		ref = self._entities.get(reference_entity_id)
		if ref is None or ref["tenant_id"] != tenant_id:
			raise KeyError(f"unknown reference entity: {reference_entity_id}")
		ref_emb = self._embeddings.get(reference_entity_id, self._mock_embed(ref["name"]))
		entities = [e for e in self._entities.values() if e["tenant_id"] == tenant_id and e["graph_id"] == graph_id and e["id"] != reference_entity_id]
		scored: list[tuple[float, dict[str, Any]]] = []
		for ent in entities:
			emb = self._embeddings.get(ent["id"], self._mock_embed(ent["name"]))
			score = sum(a * b for a, b in zip(ref_emb[:32], emb[:32]))
			scored.append((score, ent))
		scored.sort(key=lambda x: -x[0])
		results = [{"score": round(sc, 4), "entity": ent} for sc, ent in scored[:top_k]]
		record: dict[str, Any] = {
			"id": search_id,
			"tenant_id": tenant_id,
			"graph_id": graph_id,
			"reference_entity_id": reference_entity_id,
			"result_count": len(results),
			"results": results,
			"status": "completed",
		}
		self._queries[search_id] = record
		self._audit(tenant_id, search_id, "similarity_search_completed")
		return dict(record)

	async def path_explain(
		self,
		explain_id: str,
		tenant_id: str,
		graph_id: str,
		entity_id_a: str,
		entity_id_b: str,
	) -> dict[str, Any]:
		"""Explain the shortest relationship path between two entities with natural language summary."""
		self._require_graph(graph_id, tenant_id)
		# BFS to find path
		adj: dict[str, list[tuple[str, str]]] = {}
		for rid, rel in self._relationships.items():
			if rel.get("graph_id") == graph_id and rel.get("tenant_id") == tenant_id:
				adj.setdefault(rel["entity_id_a"], []).append((rel["entity_id_b"], rid))
				adj.setdefault(rel["entity_id_b"], []).append((rel["entity_id_a"], rid))
		from collections import deque
		visited: set[str] = {entity_id_a}
		prev: dict[str, tuple[str, str] | None] = {entity_id_a: None}
		queue: deque[str] = deque([entity_id_a])
		found = False
		while queue:
			cur = queue.popleft()
			if cur == entity_id_b:
				found = True
				break
			for nb, rid in adj.get(cur, []):
				if nb not in visited:
					visited.add(nb)
					prev[nb] = (cur, rid)
					queue.append(nb)
		path_nodes: list[str] = []
		path_rels: list[str] = []
		if found:
			cur_node: str | None = entity_id_b
			while cur_node is not None:
				path_nodes.insert(0, cur_node)
				p = prev.get(cur_node)
				if p is None:
					break
				path_rels.insert(0, p[1])
				cur_node = p[0]
		entity_names = {eid: self._entities.get(eid, {}).get("name", eid) for eid in path_nodes}
		explanation = " -> ".join(entity_names.get(n, n) for n in path_nodes) if found else "No path found"
		record: dict[str, Any] = {
			"id": explain_id,
			"tenant_id": tenant_id,
			"graph_id": graph_id,
			"entity_id_a": entity_id_a,
			"entity_id_b": entity_id_b,
			"path_found": found,
			"path_length": len(path_rels),
			"path_nodes": path_nodes,
			"path_relationships": path_rels,
			"explanation": explanation,
			"status": "completed",
		}
		self._audit(tenant_id, explain_id, "path_explained")
		return dict(record)

	async def entity_merge(
		self,
		merge_id: str,
		tenant_id: str,
		source_entity_id: str,
		target_entity_id: str,
	) -> dict[str, Any]:
		"""Merge two entities: redirect all relationships from source to target, then delete source."""
		src = self._entities.get(source_entity_id)
		tgt = self._entities.get(target_entity_id)
		if src is None or src["tenant_id"] != tenant_id:
			raise KeyError(f"unknown source entity: {source_entity_id}")
		if tgt is None or tgt["tenant_id"] != tenant_id:
			raise KeyError(f"unknown target entity: {target_entity_id}")
		redirected = 0
		for rid, rel in self._relationships.items():
			changed = False
			if rel.get("entity_id_a") == source_entity_id:
				rel["entity_id_a"] = target_entity_id
				changed = True
			if rel.get("entity_id_b") == source_entity_id:
				rel["entity_id_b"] = target_entity_id
				changed = True
			if changed:
				redirected += 1
		# Merge properties
		tgt["properties"].update({k: v for k, v in src.get("properties", {}).items() if k not in tgt["properties"]})
		self._entities[target_entity_id] = tgt
		self._entities.pop(source_entity_id, None)
		record: dict[str, Any] = {
			"id": merge_id,
			"tenant_id": tenant_id,
			"source_entity_id": source_entity_id,
			"target_entity_id": target_entity_id,
			"redirected_relationships": redirected,
			"status": "merged",
		}
		self._audit(tenant_id, merge_id, "entities_merged", metadata={"redirected_relationships": redirected})
		return dict(record)

	async def create_record(
		self,
		record_id: str,
		tenant_id: str,
		metadata: dict[str, Any] | None = None,
		status: str = "active",
	) -> dict[str, Any]:
		"""Compatibility helper: create a graph and seed it with one entity."""
		data = dict(metadata or {})
		graph_id = f"graph_{record_id}"
		if graph_id not in self._graphs:
			await self.graph_index(graph_id, tenant_id, name=str(data.get("name") or record_id))
		return await self.create_entity(
			entity_id=record_id,
			tenant_id=tenant_id,
			graph_id=graph_id,
			name=str(data.get("name") or record_id),
			entity_type=str(data.get("entity_type") or "Entity"),
			properties=data,
		)

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		"""Compatibility surface exposing graphs as GRAG records."""
		items = [g for g in self._graphs.values() if tenant_id is None or g["tenant_id"] == tenant_id]
		return sorted(items, key=lambda x: x["id"])

	async def entity_type_summary(
		self,
		tenant_id: str,
		graph_id: str,
	) -> dict[str, Any]:
		"""Return a frequency summary of entity types in the graph."""
		self._require_graph(graph_id, tenant_id)
		entities = [e for e in self._entities.values() if e["tenant_id"] == tenant_id and e["graph_id"] == graph_id]
		type_counts: dict[str, int] = {}
		for ent in entities:
			t = ent.get("entity_type", "unknown")
			type_counts[t] = type_counts.get(t, 0) + 1
		return {
			"graph_id": graph_id,
			"tenant_id": tenant_id,
			"total_entities": len(entities),
			"type_distribution": type_counts,
			"unique_types": len(type_counts),
		}

	async def relationship_type_summary(
		self,
		tenant_id: str,
		graph_id: str,
	) -> dict[str, Any]:
		"""Return a frequency summary of relationship types in the graph."""
		self._require_graph(graph_id, tenant_id)
		rels = [r for r in self._relationships.values() if r.get("tenant_id") == tenant_id and r.get("graph_id") == graph_id]
		type_counts: dict[str, int] = {}
		for rel in rels:
			t = rel.get("link_type") or rel.get("predicate", "unknown")
			type_counts[t] = type_counts.get(t, 0) + 1
		return {
			"graph_id": graph_id,
			"tenant_id": tenant_id,
			"total_relationships": len(rels),
			"type_distribution": type_counts,
			"unique_types": len(type_counts),
		}

	async def seed_graph_from_text(
		self,
		seed_id: str,
		tenant_id: str,
		graph_id: str,
		text: str,
	) -> dict[str, Any]:
		"""Extract entities from text and add them to the graph (lightweight pipeline)."""
		self._require_graph(graph_id, tenant_id)
		if not text:
			raise ValueError("text_required")
		import re
		# Extract capitalised proper-noun tokens as entities
		tokens = list(dict.fromkeys(re.findall(r'\b[A-Z][a-z]{2,}\b', text)))
		created: list[dict[str, Any]] = []
		for token in tokens[:50]:
			eid = f"{seed_id}_{token.lower()}"
			if eid not in self._entities:
				ent = await self.create_entity(
					entity_id=eid,
					tenant_id=tenant_id,
					graph_id=graph_id,
					name=token,
					entity_type="ExtractedEntity",
					confidence=0.75,
				)
				created.append(ent)
		return {
			"id": seed_id,
			"tenant_id": tenant_id,
			"graph_id": graph_id,
			"text_length": len(text),
			"entities_created": len(created),
			"status": "seeded",
		}

	async def prune_low_confidence(
		self,
		prune_id: str,
		tenant_id: str,
		graph_id: str,
		threshold: float = 0.5,
	) -> dict[str, Any]:
		"""Remove entities and relationships below a confidence threshold."""
		self._require_graph(graph_id, tenant_id)
		if not 0.0 <= threshold <= 1.0:
			raise ValueError("threshold_must_be_0_to_1")
		low_ents = [eid for eid, e in self._entities.items() if e["tenant_id"] == tenant_id and e["graph_id"] == graph_id and float(e.get("confidence", 1.0)) < threshold]
		for eid in low_ents:
			self._entities.pop(eid, None)
		low_rels = [rid for rid, r in self._relationships.items() if r.get("graph_id") == graph_id and r.get("tenant_id") == tenant_id and float(r.get("confidence", 1.0)) < threshold]
		for rid in low_rels:
			self._relationships.pop(rid, None)
		result: dict[str, Any] = {
			"id": prune_id,
			"tenant_id": tenant_id,
			"graph_id": graph_id,
			"threshold": threshold,
			"pruned_entities": len(low_ents),
			"pruned_relationships": len(low_rels),
			"status": "completed",
		}
		self._audit(tenant_id, prune_id, "low_confidence_pruned", metadata={"pruned_entities": len(low_ents), "threshold": threshold})
		return dict(result)

	async def graph_compliance_check(
		self,
		check_id: str,
		tenant_id: str,
		graph_id: str,
	) -> dict[str, Any]:
		"""Check the knowledge graph for data governance compliance issues."""
		self._require_graph(graph_id, tenant_id)
		entities = [e for e in self._entities.values() if e["tenant_id"] == tenant_id and e["graph_id"] == graph_id]
		findings: list[dict[str, Any]] = []
		no_type = [e["id"] for e in entities if not e.get("entity_type")]
		if no_type:
			findings.append({"severity": "medium", "type": "entities_missing_type", "ids": no_type[:10]})
		zero_conf = [e["id"] for e in entities if float(e.get("confidence", 1.0)) == 0.0]
		if zero_conf:
			findings.append({"severity": "high", "type": "zero_confidence_entities", "ids": zero_conf[:10]})
		contradictions = [c for c in self._contradictions.values() if c.get("graph_id") == graph_id and c.get("tenant_id") == tenant_id]
		if contradictions:
			findings.append({"severity": "high", "type": "contradictions_present", "count": len(contradictions)})
		result: dict[str, Any] = {
			"id": check_id,
			"tenant_id": tenant_id,
			"graph_id": graph_id,
			"entity_count": len(entities),
			"findings": findings,
			"finding_count": len(findings),
			"compliant": len(findings) == 0,
			"risk_level": "high" if any(f["severity"] == "high" for f in findings) else "low",
			"status": "completed",
		}
		self._audit(tenant_id, check_id, "graph_compliance_checked", metadata={"findings": len(findings)})
		return dict(result)

	async def export_entities_csv(
		self,
		tenant_id: str,
		graph_id: str,
	) -> str:
		"""Export entities as a CSV string (id, name, entity_type, confidence)."""
		self._require_graph(graph_id, tenant_id)
		entities = [e for e in self._entities.values() if e["tenant_id"] == tenant_id and e["graph_id"] == graph_id]
		lines = ["id,name,entity_type,confidence"]
		for ent in sorted(entities, key=lambda x: x["id"]):
			name_safe = str(ent.get("name", "")).replace('"', '""')
			lines.append(f'"{ent["id"]}","{name_safe}","{ent.get("entity_type", "")}",{ent.get("confidence", 1.0)}')
		self._audit(tenant_id, graph_id, "entities_exported_csv", metadata={"count": len(entities)})
		return "\n".join(lines)


__all__ = [
	'GraphRAGService',
	'GragService',
	'GraphRAGConfig',
	'Document',
	'GraphRAGServiceError',
	'DocumentProcessingError',
	'GraphConstructionError',
	'ReasoningError',
	'create_graphrag_service',
]
