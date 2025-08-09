"""
APG GraphRAG Capability - Hybrid Vector-Graph Retrieval System

Revolutionary hybrid retrieval combining vector similarity search with graph traversal.
Optimizes retrieval through intelligent fusion of dense embeddings and graph relationships.

Author: Datacraft (nyimbi@gmail.com)
Copyright: © 2025 Datacraft
Website: www.datacraft.co.ke
"""

from __future__ import annotations
import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from scipy.spatial.distance import cosine

from .database import GraphRAGDatabaseService
from .views import (
	GraphEntity, GraphRelationship, GraphPath, Evidence,
	RetrievalConfig, PerformanceMetrics
)


logger = logging.getLogger(__name__)


@dataclass
class RetrievalCandidate:
	"""Candidate entity or relationship for retrieval"""
	item_id: str
	item_type: str  # entity, relationship
	content: str
	embeddings: List[float]
	metadata: Dict[str, Any]
	confidence_score: float
	source_type: str  # vector_similarity, graph_traversal, hybrid


@dataclass
class RetrievalResult:
	"""Result from hybrid retrieval operation"""
	entities: List[GraphEntity]
	relationships: List[GraphRelationship]
	graph_paths: List[GraphPath]
	evidence: List[Evidence]
	retrieval_scores: Dict[str, float]
	retrieval_metadata: Dict[str, Any]
	performance_metrics: Dict[str, Any]


class HybridRetrievalEngine:
	"""
	Advanced hybrid retrieval engine combining:
	
	- Dense vector similarity search using bge-m3 embeddings
	- Graph traversal and relationship exploration
	- Intelligent ranking fusion algorithms
	- Context-aware retrieval optimization
	- Performance monitoring and caching
	"""
	
	def __init__(self, db_service: GraphRAGDatabaseService, config: Optional[Dict[str, Any]] = None):
		"""Initialize hybrid retrieval engine"""
		self.db_service = db_service
		self.config = config or {}
		
		# Retrieval parameters
		self.vector_weight = self.config.get("vector_weight", 0.6)
		self.graph_weight = self.config.get("graph_weight", 0.4)
		self.similarity_threshold = self.config.get("similarity_threshold", 0.7)
		self.max_candidates = self.config.get("max_candidates", 100)
		self.diversity_factor = self.config.get("diversity_factor", 0.3)
		
		# Performance tracking
		self._retrieval_cache = {}
		self._performance_stats = defaultdict(list)
		
		logger.info("Hybrid retrieval engine initialized")
	
	async def hybrid_retrieve(
		self,
		tenant_id: str,
		knowledge_graph_id: str,
		query_text: str,
		query_embedding: List[float],
		retrieval_config: RetrievalConfig,
		context: Optional[Dict[str, Any]] = None
	) -> RetrievalResult:
		"""
		Perform hybrid vector-graph retrieval optimized for GraphRAG queries
		
		Args:
			tenant_id: Tenant identifier
			knowledge_graph_id: Knowledge graph identifier
			query_text: Original query text
			query_embedding: Query embedding from bge-m3
			retrieval_config: Retrieval configuration parameters
			context: Optional context for retrieval optimization
			
		Returns:
			RetrievalResult with entities, relationships, and paths
		"""
		start_time = time.time()
		
		try:
			# Check cache first
			cache_key = self._generate_cache_key(
				tenant_id, knowledge_graph_id, query_text, retrieval_config
			)
			
			if cache_key in self._retrieval_cache:
				logger.info("Returning cached retrieval result")
				return self._retrieval_cache[cache_key]
			
			# Step 1: Vector similarity search
			vector_start = time.time()
			vector_candidates = await self._vector_similarity_search(
				tenant_id=tenant_id,
				knowledge_graph_id=knowledge_graph_id,
				query_embedding=query_embedding,
				max_results=retrieval_config.max_entities,
				similarity_threshold=retrieval_config.similarity_threshold
			)
			vector_time = (time.time() - vector_start) * 1000
			
			# Step 2: Graph traversal from top vector candidates
			graph_start = time.time()
			graph_candidates = await self._graph_traversal_search(
				tenant_id=tenant_id,
				knowledge_graph_id=knowledge_graph_id,
				seed_entities=vector_candidates[:10],  # Use top 10 as seeds
				max_hops=3,
				relationship_types=retrieval_config.relationship_types
			)
			graph_time = (time.time() - graph_start) * 1000
			
			# Step 3: Fusion and ranking
			fusion_start = time.time()
			fused_results = await self._fusion_ranking(
				query_embedding=query_embedding,
				vector_candidates=vector_candidates,
				graph_candidates=graph_candidates,
				retrieval_config=retrieval_config,
				context=context
			)
			fusion_time = (time.time() - fusion_start) * 1000
			
			# Step 4: Diversity optimization
			diversity_start = time.time()
			optimized_results = await self._diversity_optimization(
				candidates=fused_results,
				diversity_factor=retrieval_config.diversity_factor,
				max_results=retrieval_config.max_entities
			)
			diversity_time = (time.time() - diversity_start) * 1000
			
			# Step 5: Build comprehensive result
			result = await self._build_retrieval_result(
				tenant_id=tenant_id,
				knowledge_graph_id=knowledge_graph_id,
				optimized_candidates=optimized_results,
				query_text=query_text,
				retrieval_config=retrieval_config
			)
			
			# Add performance metrics
			total_time = (time.time() - start_time) * 1000
			result.performance_metrics = {
				"total_time_ms": total_time,
				"vector_search_time_ms": vector_time,
				"graph_traversal_time_ms": graph_time,
				"fusion_ranking_time_ms": fusion_time,
				"diversity_optimization_time_ms": diversity_time,
				"vector_candidates": len(vector_candidates),
				"graph_candidates": len(graph_candidates),
				"final_results": len(result.entities)
			}
			
			# Cache result
			self._retrieval_cache[cache_key] = result
			
			# Record performance statistics
			self._record_performance("hybrid_retrieve", total_time)
			
			logger.info(f"Hybrid retrieval completed in {total_time:.1f}ms, found {len(result.entities)} entities")
			return result
			
		except Exception as e:
			logger.error(f"Hybrid retrieval failed: {e}")
			raise
	
	async def contextual_expand_query(
		self,
		tenant_id: str,
		knowledge_graph_id: str,
		query_text: str,
		query_embedding: List[float],
		expansion_depth: int = 2
	) -> Tuple[str, List[float], Dict[str, Any]]:
		"""
		Expand query using graph context and semantic relationships
		
		Args:
			tenant_id: Tenant identifier
			knowledge_graph_id: Knowledge graph identifier
			query_text: Original query text
			query_embedding: Query embedding
			expansion_depth: Depth of semantic expansion
			
		Returns:
			Tuple of (expanded_query, expanded_embedding, expansion_metadata)
		"""
		try:
			# Find semantically related entities
			related_entities = await self._find_semantically_related_entities(
				tenant_id=tenant_id,
				knowledge_graph_id=knowledge_graph_id,
				query_embedding=query_embedding,
				max_entities=20
			)
			
			# Extract expansion terms from related entities
			expansion_terms = []
			for entity in related_entities:
				expansion_terms.extend(entity.aliases)
				expansion_terms.append(entity.canonical_name)
			
			# Remove duplicates and filter by relevance
			unique_terms = list(set(expansion_terms))
			relevant_terms = [term for term in unique_terms if len(term) > 2][:10]
			
			# Build expanded query
			expanded_query = query_text
			if relevant_terms:
				expanded_query += " " + " ".join(relevant_terms)
			
			# Generate expanded embedding (would be done via Ollama in production)
			expanded_embedding = query_embedding  # Simplified for now
			
			expansion_metadata = {
				"original_query": query_text,
				"expansion_terms": relevant_terms,
				"related_entities_count": len(related_entities),
				"expansion_depth": expansion_depth
			}
			
			logger.info(f"Query expanded with {len(relevant_terms)} terms from {len(related_entities)} related entities")
			return expanded_query, expanded_embedding, expansion_metadata
			
		except Exception as e:
			logger.error(f"Query expansion failed: {e}")
			return query_text, query_embedding, {}
	
	async def multi_hop_exploration(
		self,
		tenant_id: str,
		knowledge_graph_id: str,
		start_entities: List[str],
		max_hops: int = 3,
		path_constraints: Optional[Dict[str, Any]] = None
	) -> List[GraphPath]:
		"""
		Explore multi-hop paths from starting entities with constraints
		
		Args:
			tenant_id: Tenant identifier
			knowledge_graph_id: Knowledge graph identifier
			start_entities: Starting entity IDs
			max_hops: Maximum number of hops
			path_constraints: Optional constraints on paths
			
		Returns:
			List of discovered graph paths
		"""
		try:
			all_paths = []
			
			# Explore from each starting entity
			for entity_id in start_entities:
				entity_paths = await self.db_service.multi_hop_traversal(
					tenant_id=tenant_id,
					knowledge_graph_id=knowledge_graph_id,
					start_entity_id=entity_id,
					max_hops=max_hops
				)
				
				# Convert to GraphPath objects
				for path_data in entity_paths:
					graph_path = GraphPath(
						entities=path_data.get("entities", []),
						relationships=path_data.get("relationships", []),
						path_length=path_data.get("path_length", 0),
						path_strength=path_data.get("total_strength", 0.0),
						semantic_coherence=self._calculate_semantic_coherence(path_data),
						confidence=min(1.0, path_data.get("total_strength", 0.0) / path_data.get("path_length", 1))
					)
					all_paths.append(graph_path)
			
			# Apply path constraints if specified
			if path_constraints:
				all_paths = self._apply_path_constraints(all_paths, path_constraints)
			
			# Sort by relevance and strength
			all_paths.sort(key=lambda p: (p.path_strength, p.semantic_coherence), reverse=True)
			
			logger.info(f"Explored {len(all_paths)} multi-hop paths from {len(start_entities)} starting entities")
			return all_paths[:100]  # Limit to top 100 paths
			
		except Exception as e:
			logger.error(f"Multi-hop exploration failed: {e}")
			return []
	
	# ========================================================================
	# VECTOR SIMILARITY SEARCH
	# ========================================================================
	
	async def _vector_similarity_search(
		self,
		tenant_id: str,
		knowledge_graph_id: str,
		query_embedding: List[float],
		max_results: int = 50,
		similarity_threshold: float = 0.7
	) -> List[RetrievalCandidate]:
		"""Perform vector similarity search using entity embeddings"""
		try:
			# Get entities with embeddings from database
			entities = await self.db_service.list_entities(
				tenant_id=tenant_id,
				knowledge_graph_id=knowledge_graph_id,
				limit=1000  # Search in larger pool for better results
			)
			
			candidates = []
			
			# Calculate similarity scores
			for entity in entities:
				if entity.embeddings and len(entity.embeddings) == len(query_embedding):
					# Calculate cosine similarity
					similarity = 1 - cosine(query_embedding, entity.embeddings)
					
					if similarity >= similarity_threshold:
						candidate = RetrievalCandidate(
							item_id=entity.canonical_entity_id,
							item_type="entity",
							content=entity.canonical_name,
							embeddings=entity.embeddings,
							metadata={
								"entity_type": entity.entity_type,
								"properties": entity.properties,
								"aliases": entity.aliases
							},
							confidence_score=similarity,
							source_type="vector_similarity"
						)
						candidates.append(candidate)
			
			# Sort by similarity and return top results
			candidates.sort(key=lambda c: c.confidence_score, reverse=True)
			
			logger.info(f"Vector search found {len(candidates)} candidates above threshold {similarity_threshold}")
			return candidates[:max_results]
			
		except Exception as e:
			logger.error(f"Vector similarity search failed: {e}")
			return []
	
	# ========================================================================
	# GRAPH TRAVERSAL SEARCH
	# ========================================================================
	
	async def _graph_traversal_search(
		self,
		tenant_id: str,
		knowledge_graph_id: str,
		seed_entities: List[RetrievalCandidate],
		max_hops: int = 3,
		relationship_types: Optional[List[str]] = None
	) -> List[RetrievalCandidate]:
		"""Perform graph traversal from seed entities"""
		try:
			graph_candidates = []
			explored_entities = set()
			
			# Traverse from each seed entity
			for seed in seed_entities:
				if seed.item_id in explored_entities:
					continue
				
				explored_entities.add(seed.item_id)
				
				# Get paths from this seed entity
				paths = await self.db_service.multi_hop_traversal(
					tenant_id=tenant_id,
					knowledge_graph_id=knowledge_graph_id,
					start_entity_id=seed.item_id,
					max_hops=max_hops,
					relationship_types=relationship_types
				)
				
				# Extract entities and relationships from paths
				for path_data in paths:
					# Add entities from path
					for entity_id in path_data.get("entities", []):
						if entity_id not in explored_entities:
							try:
								entity = await self.db_service.get_entity(
									tenant_id, knowledge_graph_id, entity_id
								)
								
								candidate = RetrievalCandidate(
									item_id=entity.canonical_entity_id,
									item_type="entity",
									content=entity.canonical_name,
									embeddings=entity.embeddings or [],
									metadata={
										"entity_type": entity.entity_type,
										"properties": entity.properties,
										"path_distance": path_data.get("path_length", 0),
										"path_strength": path_data.get("total_strength", 0.0)
									},
									confidence_score=max(0.1, 1.0 - (path_data.get("path_length", 1) * 0.2)),
									source_type="graph_traversal"
								)
								graph_candidates.append(candidate)
								explored_entities.add(entity_id)
								
							except Exception as e:
								logger.warning(f"Could not retrieve entity {entity_id}: {e}")
					
					# Add relationships from path
					for rel_id in path_data.get("relationships", []):
						try:
							relationship = await self.db_service.get_relationship(
								tenant_id, knowledge_graph_id, rel_id
							)
							
							candidate = RetrievalCandidate(
								item_id=relationship.canonical_relationship_id,
								item_type="relationship",
								content=f"{relationship.relationship_type}: {relationship.source_entity_id} -> {relationship.target_entity_id}",
								embeddings=[],
								metadata={
									"relationship_type": relationship.relationship_type,
									"strength": relationship.strength,
									"properties": relationship.properties,
									"path_distance": path_data.get("path_length", 0)
								},
								confidence_score=float(relationship.strength),
								source_type="graph_traversal"
							)
							graph_candidates.append(candidate)
							
						except Exception as e:
							logger.warning(f"Could not retrieve relationship {rel_id}: {e}")
			
			logger.info(f"Graph traversal found {len(graph_candidates)} candidates from {len(seed_entities)} seeds")
			return graph_candidates
			
		except Exception as e:
			logger.error(f"Graph traversal search failed: {e}")
			return []
	
	# ========================================================================
	# FUSION AND RANKING
	# ========================================================================
	
	async def _fusion_ranking(
		self,
		query_embedding: List[float],
		vector_candidates: List[RetrievalCandidate],
		graph_candidates: List[RetrievalCandidate],
		retrieval_config: RetrievalConfig,
		context: Optional[Dict[str, Any]] = None
	) -> List[RetrievalCandidate]:
		"""Fuse and rank candidates from vector and graph search"""
		try:
			# Combine all candidates
			all_candidates = vector_candidates + graph_candidates
			
			# Remove duplicates based on item_id
			unique_candidates = {}
			for candidate in all_candidates:
				if candidate.item_id not in unique_candidates:
					unique_candidates[candidate.item_id] = candidate
				else:
					# Merge scores for duplicates
					existing = unique_candidates[candidate.item_id]
					if candidate.confidence_score > existing.confidence_score:
						unique_candidates[candidate.item_id] = candidate
			
			candidates = list(unique_candidates.values())
			
			# Calculate hybrid scores
			for candidate in candidates:
				vector_score = 0.0
				graph_score = 0.0
				
				if candidate.source_type == "vector_similarity":
					vector_score = candidate.confidence_score
					# Calculate implicit graph score based on connectivity
					graph_score = self._calculate_connectivity_score(candidate)
				elif candidate.source_type == "graph_traversal":
					graph_score = candidate.confidence_score
					# Calculate vector score if embeddings available
					if candidate.embeddings and len(candidate.embeddings) == len(query_embedding):
						vector_score = 1 - cosine(query_embedding, candidate.embeddings)
				
				# Weighted fusion
				hybrid_score = (self.vector_weight * vector_score + 
							   self.graph_weight * graph_score)
				
				# Apply context boosting if available
				if context:
					context_boost = self._calculate_context_boost(candidate, context)
					hybrid_score *= (1 + context_boost)
				
				candidate.confidence_score = hybrid_score
			
			# Sort by hybrid score
			candidates.sort(key=lambda c: c.confidence_score, reverse=True)
			
			logger.info(f"Fusion ranking processed {len(candidates)} unique candidates")
			return candidates
			
		except Exception as e:
			logger.error(f"Fusion ranking failed: {e}")
			return vector_candidates + graph_candidates  # Fallback
	
	# ========================================================================
	# DIVERSITY OPTIMIZATION
	# ========================================================================
	
	async def _diversity_optimization(
		self,
		candidates: List[RetrievalCandidate],
		diversity_factor: float = 0.3,
		max_results: int = 50
	) -> List[RetrievalCandidate]:
		"""Optimize candidate selection for diversity while maintaining relevance"""
		try:
			if len(candidates) <= max_results:
				return candidates
			
			selected = []
			remaining = candidates.copy()
			
			# Always select the top candidate
			if remaining:
				selected.append(remaining.pop(0))
			
			# Iteratively select diverse candidates
			while len(selected) < max_results and remaining:
				best_candidate = None
				best_score = -1
				
				for candidate in remaining:
					# Calculate diversity score
					diversity_score = self._calculate_diversity_score(candidate, selected)
					
					# Combine relevance and diversity
					combined_score = (
						(1 - diversity_factor) * candidate.confidence_score +
						diversity_factor * diversity_score
					)
					
					if combined_score > best_score:
						best_score = combined_score
						best_candidate = candidate
				
				if best_candidate:
					selected.append(best_candidate)
					remaining.remove(best_candidate)
				else:
					break
			
			logger.info(f"Diversity optimization selected {len(selected)} from {len(candidates)} candidates")
			return selected
			
		except Exception as e:
			logger.error(f"Diversity optimization failed: {e}")
			return candidates[:max_results]  # Fallback
	
	# ========================================================================
	# RESULT BUILDING
	# ========================================================================
	
	async def _build_retrieval_result(
		self,
		tenant_id: str,
		knowledge_graph_id: str,
		optimized_candidates: List[RetrievalCandidate],
		query_text: str,
		retrieval_config: RetrievalConfig
	) -> RetrievalResult:
		"""Build comprehensive retrieval result from optimized candidates"""
		try:
			entities = []
			relationships = []
			evidence = []
			graph_paths = []
			
			# Separate entities and relationships
			entity_candidates = [c for c in optimized_candidates if c.item_type == "entity"]
			relationship_candidates = [c for c in optimized_candidates if c.item_type == "relationship"]
			
			# Build entities
			for candidate in entity_candidates:
				try:
					entity = await self.db_service.get_entity(
						tenant_id, knowledge_graph_id, candidate.item_id
					)
					entities.append(entity)
					
					# Create evidence
					evidence.append(Evidence(
						source_type="entity",
						source_id=entity.canonical_entity_id,
						content=entity.canonical_name,
						confidence=candidate.confidence_score,
						relevance_score=candidate.confidence_score,
						provenance={"retrieval_method": candidate.source_type}
					))
					
				except Exception as e:
					logger.warning(f"Could not build entity result for {candidate.item_id}: {e}")
			
			# Build relationships
			for candidate in relationship_candidates:
				try:
					relationship = await self.db_service.get_relationship(
						tenant_id, knowledge_graph_id, candidate.item_id
					)
					relationships.append(relationship)
					
					# Create evidence
					evidence.append(Evidence(
						source_type="relationship",
						source_id=relationship.canonical_relationship_id,
						content=f"{relationship.relationship_type}: {relationship.source_entity_id} -> {relationship.target_entity_id}",
						confidence=candidate.confidence_score,
						relevance_score=candidate.confidence_score,
						provenance={"retrieval_method": candidate.source_type}
					))
					
				except Exception as e:
					logger.warning(f"Could not build relationship result for {candidate.item_id}: {e}")
			
			# Build graph paths (simplified)
			if entities:
				sample_paths = await self._build_sample_paths(
					tenant_id, knowledge_graph_id, entities[:5]
				)
				graph_paths.extend(sample_paths)
			
			# Calculate retrieval scores
			retrieval_scores = {
				"average_confidence": sum(c.confidence_score for c in optimized_candidates) / len(optimized_candidates) if optimized_candidates else 0.0,
				"vector_ratio": len([c for c in optimized_candidates if c.source_type == "vector_similarity"]) / len(optimized_candidates) if optimized_candidates else 0.0,
				"graph_ratio": len([c for c in optimized_candidates if c.source_type == "graph_traversal"]) / len(optimized_candidates) if optimized_candidates else 0.0,
				"entity_count": len(entities),
				"relationship_count": len(relationships)
			}
			
			# Build metadata
			retrieval_metadata = {
				"query_text": query_text,
				"retrieval_config": retrieval_config.dict(),
				"total_candidates_processed": len(optimized_candidates),
				"retrieval_timestamp": datetime.utcnow().isoformat()
			}
			
			return RetrievalResult(
				entities=entities,
				relationships=relationships,
				graph_paths=graph_paths,
				evidence=evidence,
				retrieval_scores=retrieval_scores,
				retrieval_metadata=retrieval_metadata,
				performance_metrics={}  # Will be filled by caller
			)
			
		except Exception as e:
			logger.error(f"Failed to build retrieval result: {e}")
			raise
	
	# ========================================================================
	# HELPER METHODS
	# ========================================================================
	
	def _generate_cache_key(
		self,
		tenant_id: str,
		knowledge_graph_id: str,
		query_text: str,
		retrieval_config: RetrievalConfig
	) -> str:
		"""Generate cache key for retrieval results"""
		import hashlib
		key_data = f"{tenant_id}:{knowledge_graph_id}:{query_text}:{retrieval_config.dict()}"
		return hashlib.md5(key_data.encode()).hexdigest()
	
	async def _find_semantically_related_entities(
		self,
		tenant_id: str,
		knowledge_graph_id: str,
		query_embedding: List[float],
		max_entities: int = 20
	) -> List[GraphEntity]:
		"""Find entities semantically related to query"""
		# Implementation would use vector similarity
		return []
	
	def _calculate_semantic_coherence(self, path_data: Dict[str, Any]) -> float:
		"""Calculate semantic coherence of a graph path"""
		# Implementation would analyze entity and relationship types for coherence
		return 0.8
	
	def _apply_path_constraints(
		self,
		paths: List[GraphPath],
		constraints: Dict[str, Any]
	) -> List[GraphPath]:
		"""Apply constraints to filter graph paths"""
		# Implementation would filter paths based on constraints
		return paths
	
	def _calculate_connectivity_score(self, candidate: RetrievalCandidate) -> float:
		"""Calculate connectivity score for entity"""
		# Implementation would analyze entity connectivity in graph
		return 0.5
	
	def _calculate_context_boost(
		self,
		candidate: RetrievalCandidate,
		context: Dict[str, Any]
	) -> float:
		"""Calculate context boost for candidate"""
		# Implementation would boost scores based on context
		return 0.1
	
	def _calculate_diversity_score(
		self,
		candidate: RetrievalCandidate,
		selected: List[RetrievalCandidate]
	) -> float:
		"""Calculate diversity score for candidate"""
		# Implementation would calculate how different candidate is from selected
		return 0.7
	
	async def _build_sample_paths(
		self,
		tenant_id: str,
		knowledge_graph_id: str,
		entities: List[GraphEntity]
	) -> List[GraphPath]:
		"""Build sample graph paths from entities"""
		# Implementation would find paths between entities
		return []
	
	def _record_performance(self, operation: str, time_ms: float) -> None:
		"""Record performance statistics"""
		self._performance_stats[operation].append(time_ms)
		
		# Keep only last 1000 measurements
		if len(self._performance_stats[operation]) > 1000:
			self._performance_stats[operation] = self._performance_stats[operation][-1000:]


__all__ = [
	'HybridRetrievalEngine',
	'RetrievalCandidate',
	'RetrievalResult',
]