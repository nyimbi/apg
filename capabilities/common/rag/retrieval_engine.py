"""
APG RAG Intelligent Retrieval Engine

Advanced retrieval system with context-aware ranking, multi-stage retrieval,
and intelligent query expansion using PostgreSQL + pgvector + pgai.
"""

import asyncio
import logging
import time
import math
from typing import Dict, Any, List, Optional, Tuple, Union, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import re
from uuid_extensions import uuid7str
from collections import defaultdict, Counter

# Database imports
import asyncpg
from asyncpg import Pool

# APG imports
from .models import (
	RetrievalRequest, RetrievalResult, RetrievalMethod,
	KnowledgeBase, Document, DocumentChunk, APGBaseModel
)
from .vector_service import VectorService
from .ollama_integration import AdvancedOllamaIntegration, RequestPriority

class QueryType(str, Enum):
	"""Types of queries for optimization"""
	FACTUAL = "factual"
	CONCEPTUAL = "conceptual"
	PROCEDURAL = "procedural"
	COMPARATIVE = "comparative"
	ANALYTICAL = "analytical"
	CONVERSATIONAL = "conversational"

class RetrievalStrategy(str, Enum):
	"""Retrieval strategies for different scenarios"""
	PRECISION_FOCUSED = "precision_focused"
	RECALL_FOCUSED = "recall_focused"
	BALANCED = "balanced"
	DIVERSITY_FOCUSED = "diversity_focused"
	RECENCY_FOCUSED = "recency_focused"

@dataclass
class QueryAnalysis:
	"""Results of query analysis"""
	query_type: QueryType
	intent: str
	entities: List[str]
	keywords: List[str]
	complexity_score: float
	temporal_indicators: List[str]
	question_words: List[str]
	domain_hints: List[str]

@dataclass
class RetrievalConfig:
	"""Configuration for retrieval operations"""
	# Basic parameters
	default_k: int = 10
	max_k: int = 100
	default_similarity_threshold: float = 0.7
	min_similarity_threshold: float = 0.3
	
	# Multi-stage retrieval
	enable_multi_stage: bool = True
	first_stage_multiplier: float = 3.0  # Retrieve 3x more in first stage
	
	# Query expansion
	enable_query_expansion: bool = True
	max_expansion_terms: int = 5
	expansion_similarity_threshold: float = 0.8
	
	# Ranking factors
	similarity_weight: float = 0.4
	recency_weight: float = 0.15
	authority_weight: float = 0.15
	diversity_weight: float = 0.15
	context_weight: float = 0.15
	
	# Performance
	max_query_time_ms: float = 500.0
	enable_caching: bool = True
	cache_ttl_minutes: int = 30

@dataclass
class RetrievalContext:
	"""Context for retrieval operations"""
	user_id: Optional[str] = None
	session_id: Optional[str] = None
	conversation_history: List[str] = field(default_factory=list)
	user_preferences: Dict[str, Any] = field(default_factory=dict)
	domain_context: Optional[str] = None
	temporal_context: Optional[datetime] = None

@dataclass
class RankedResult:
	"""A single ranked retrieval result"""
	chunk_id: str
	document_id: str
	content: str
	similarity_score: float
	relevance_score: float
	authority_score: float
	recency_score: float
	diversity_score: float
	final_score: float
	metadata: Dict[str, Any] = field(default_factory=dict)

class QueryAnalyzer:
	"""Analyzes queries to optimize retrieval strategy"""
	
	def __init__(self):
		self.logger = logging.getLogger(__name__)
		
		# Query patterns
		self.question_words = {'what', 'when', 'where', 'who', 'why', 'how', 'which', 'whose'}
		self.temporal_indicators = {
			'recent', 'latest', 'current', 'today', 'yesterday', 'last', 'previous',
			'new', 'old', 'before', 'after', 'since', 'until', 'now', 'then'
		}
		self.comparison_words = {
			'compare', 'versus', 'vs', 'difference', 'similar', 'unlike', 'contrast',
			'better', 'worse', 'best', 'worst', 'more', 'less', 'than'
		}
		self.procedural_words = {
			'how', 'step', 'process', 'procedure', 'method', 'way', 'guide',
			'instructions', 'tutorial', 'walkthrough'
		}
	
	async def analyze_query(self, query: str, context: Optional[RetrievalContext] = None) -> QueryAnalysis:
		"""Comprehensive query analysis"""
		query_lower = query.lower()
		words = query_lower.split()
		
		# Detect query type
		query_type = self._classify_query_type(query_lower, words)
		
		# Extract components
		entities = await self._extract_entities(query)
		keywords = self._extract_keywords(query, words)
		question_words = [w for w in words if w in self.question_words]
		temporal_indicators = [w for w in words if w in self.temporal_indicators]
		
		# Calculate complexity
		complexity_score = self._calculate_complexity(query, words, entities)
		
		# Determine intent
		intent = self._determine_intent(query_type, question_words, words)
		
		# Extract domain hints
		domain_hints = self._extract_domain_hints(query, context)
		
		return QueryAnalysis(
			query_type=query_type,
			intent=intent,
			entities=entities,
			keywords=keywords,
			complexity_score=complexity_score,
			temporal_indicators=temporal_indicators,
			question_words=question_words,
			domain_hints=domain_hints
		)
	
	def _classify_query_type(self, query_lower: str, words: List[str]) -> QueryType:
		"""Classify the type of query"""
		# Check for comparison indicators
		if any(word in query_lower for word in self.comparison_words):
			return QueryType.COMPARATIVE
		
		# Check for procedural indicators
		if any(word in query_lower for word in self.procedural_words):
			return QueryType.PROCEDURAL
		
		# Check for analytical indicators
		if any(word in query_lower for word in ['analyze', 'analysis', 'trend', 'pattern', 'correlation']):
			return QueryType.ANALYTICAL
		
		# Check for conversational indicators
		if len(words) > 15 or any(word in query_lower for word in ['tell me', 'explain', 'discuss']):
			return QueryType.CONVERSATIONAL
		
		# Check for conceptual indicators
		if any(word in query_lower for word in ['concept', 'theory', 'principle', 'meaning', 'definition']):
			return QueryType.CONCEPTUAL
		
		# Default to factual
		return QueryType.FACTUAL
	
	async def _extract_entities(self, query: str) -> List[str]:
		"""Extract named entities from query"""
		# Simple entity extraction (could be enhanced with NLP capability)
		entities = []
		
		# Extract proper nouns (capitalized words)
		words = query.split()
		for word in words:
			if word[0].isupper() and len(word) > 2:
				entities.append(word)
		
		# Extract quoted phrases
		quoted_pattern = r'"([^"]*)"'
		quoted_matches = re.findall(quoted_pattern, query)
		entities.extend(quoted_matches)
		
		return entities
	
	def _extract_keywords(self, query: str, words: List[str]) -> List[str]:
		"""Extract important keywords from query"""
		# Simple keyword extraction
		stop_words = {
			'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
			'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
			'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
			'may', 'might', 'can', 'this', 'that', 'these', 'those'
		}
		
		keywords = []
		for word in words:
			clean_word = re.sub(r'[^\w]', '', word.lower())
			if len(clean_word) > 2 and clean_word not in stop_words:
				keywords.append(clean_word)
		
		return keywords
	
	def _calculate_complexity(self, query: str, words: List[str], entities: List[str]) -> float:
		"""Calculate query complexity score (0-1)"""
		factors = []
		
		# Length factor
		length_score = min(len(words) / 20.0, 1.0)
		factors.append(length_score)
		
		# Entity factor
		entity_score = min(len(entities) / 5.0, 1.0)
		factors.append(entity_score)
		
		# Nested question factor
		question_count = query.count('?')
		nested_score = min(question_count / 3.0, 1.0)
		factors.append(nested_score)
		
		# Conjunction factor (complex relationships)
		conjunctions = query.lower().count(' and ') + query.lower().count(' or ')
		conjunction_score = min(conjunctions / 3.0, 1.0)
		factors.append(conjunction_score)
		
		return sum(factors) / len(factors)
	
	def _determine_intent(self, query_type: QueryType, question_words: List[str], words: List[str]) -> str:
		"""Determine the user's intent"""
		if question_words:
			primary_question = question_words[0]
			if primary_question in ['what', 'which']:
				return 'identify'
			elif primary_question == 'how':
				return 'explain_process'
			elif primary_question == 'why':
				return 'explain_reason'
			elif primary_question in ['when', 'where']:
				return 'locate'
			elif primary_question == 'who':
				return 'identify_person'
		
		# Fallback based on query type
		intent_map = {
			QueryType.FACTUAL: 'find_facts',
			QueryType.CONCEPTUAL: 'understand_concept',
			QueryType.PROCEDURAL: 'learn_process',
			QueryType.COMPARATIVE: 'compare_options',
			QueryType.ANALYTICAL: 'analyze_data',
			QueryType.CONVERSATIONAL: 'explore_topic'
		}
		
		return intent_map.get(query_type, 'general_search')
	
	def _extract_domain_hints(self, query: str, context: Optional[RetrievalContext]) -> List[str]:
		"""Extract domain-specific hints"""
		domain_hints = []
		
		# Technical domains
		tech_indicators = ['api', 'code', 'programming', 'software', 'system', 'database']
		if any(indicator in query.lower() for indicator in tech_indicators):
			domain_hints.append('technical')
		
		# Business domains
		business_indicators = ['revenue', 'profit', 'customer', 'market', 'strategy', 'sales']
		if any(indicator in query.lower() for indicator in business_indicators):
			domain_hints.append('business')
		
		# Add context domain if available
		if context and context.domain_context:
			domain_hints.append(context.domain_context)
		
		return domain_hints

class RetrievalRanker:
	"""Advanced ranking system for retrieval results"""
	
	def __init__(self, config: RetrievalConfig):
		self.config = config
		self.logger = logging.getLogger(__name__)
	
	async def rank_results(self,
	                      raw_results: List[Dict[str, Any]],
	                      query_analysis: QueryAnalysis,
	                      context: Optional[RetrievalContext] = None) -> List[RankedResult]:
		"""Rank retrieval results using multiple factors"""
		
		if not raw_results:
			return []
		
		ranked_results = []
		
		# Calculate various scores for each result
		for result in raw_results:
			# Base similarity score
			similarity_score = result.get('similarity_score', 0.0)
			
			# Calculate additional scoring factors
			authority_score = await self._calculate_authority_score(result)
			recency_score = self._calculate_recency_score(result)
			diversity_score = 0.0  # Will be calculated across all results
			relevance_score = await self._calculate_relevance_score(result, query_analysis)
			
			# Create ranked result
			ranked_result = RankedResult(
				chunk_id=result['chunk_id'],
				document_id=result['document_id'],
				content=result['content'],
				similarity_score=similarity_score,
				relevance_score=relevance_score,
				authority_score=authority_score,
				recency_score=recency_score,
				diversity_score=diversity_score,
				final_score=0.0,  # Will be calculated after diversity
				metadata=result
			)
			
			ranked_results.append(ranked_result)
		
		# Calculate diversity scores
		await self._calculate_diversity_scores(ranked_results)
		
		# Calculate final scores and sort
		for result in ranked_results:
			result.final_score = self._calculate_final_score(result, query_analysis)
		
		# Sort by final score
		ranked_results.sort(key=lambda x: x.final_score, reverse=True)
		
		return ranked_results
	
	async def _calculate_authority_score(self, result: Dict[str, Any]) -> float:
		"""Calculate authority score based on document characteristics"""
		score = 0.5  # Base score
		
		# Document length factor
		content_length = len(result.get('content', ''))
		length_score = min(content_length / 2000.0, 1.0) * 0.2
		score += length_score
		
		# Section level factor (higher level = more authoritative)
		section_level = result.get('section_level', 0)
		if section_level > 0:
			level_score = min(section_level / 5.0, 1.0) * 0.3
			score += level_score
		
		return min(score, 1.0)
	
	def _calculate_recency_score(self, result: Dict[str, Any]) -> float:
		"""Calculate recency score based on document age"""
		# This would typically use document creation/modification date
		# For now, return a default score
		return 0.5
	
	async def _calculate_relevance_score(self, result: Dict[str, Any], query_analysis: QueryAnalysis) -> float:
		"""Calculate content relevance beyond similarity"""
		content = result.get('content', '').lower()
		score = 0.0
		
		# Keyword matching
		keyword_matches = sum(1 for keyword in query_analysis.keywords if keyword in content)
		keyword_score = min(keyword_matches / max(len(query_analysis.keywords), 1), 1.0) * 0.4
		score += keyword_score
		
		# Entity matching
		entity_matches = sum(1 for entity in query_analysis.entities 
		                    if entity.lower() in content)
		entity_score = min(entity_matches / max(len(query_analysis.entities), 1), 1.0) * 0.3
		score += entity_score
		
		# Domain relevance
		if query_analysis.domain_hints:
			domain_matches = sum(1 for hint in query_analysis.domain_hints 
			                   if hint in content)
			domain_score = min(domain_matches / len(query_analysis.domain_hints), 1.0) * 0.3
			score += domain_score
		
		return min(score, 1.0)
	
	async def _calculate_diversity_scores(self, results: List[RankedResult]) -> None:
		"""Calculate diversity scores to avoid redundant results"""
		if len(results) <= 1:
			for result in results:
				result.diversity_score = 1.0
			return
		
		# Simple content-based diversity
		for i, result in enumerate(results):
			diversity_sum = 0.0
			
			for j, other_result in enumerate(results):
				if i != j:
					# Calculate content similarity
					similarity = self._calculate_content_similarity(
						result.content, other_result.content
					)
					diversity_sum += (1.0 - similarity)
			
			# Average diversity score
			result.diversity_score = diversity_sum / (len(results) - 1)
	
	def _calculate_content_similarity(self, content1: str, content2: str) -> float:
		"""Calculate similarity between two pieces of content"""
		words1 = set(content1.lower().split())
		words2 = set(content2.lower().split())
		
		if not words1 or not words2:
			return 0.0
		
		intersection = len(words1.intersection(words2))
		union = len(words1.union(words2))
		
		return intersection / union if union > 0 else 0.0
	
	def _calculate_final_score(self, result: RankedResult, query_analysis: QueryAnalysis) -> float:
		"""Calculate final weighted score"""
		# Adjust weights based on query type
		weights = self._get_weights_for_query_type(query_analysis.query_type)
		
		final_score = (
			result.similarity_score * weights['similarity'] +
			result.relevance_score * weights['relevance'] +
			result.authority_score * weights['authority'] +
			result.recency_score * weights['recency'] +
			result.diversity_score * weights['diversity']
		)
		
		return final_score
	
	def _get_weights_for_query_type(self, query_type: QueryType) -> Dict[str, float]:
		"""Get scoring weights optimized for query type"""
		base_weights = {
			'similarity': self.config.similarity_weight,
			'relevance': 0.25,  # Not in config, calculated
			'authority': self.config.authority_weight,
			'recency': self.config.recency_weight,
			'diversity': self.config.diversity_weight
		}
		
		# Adjust weights based on query type
		if query_type == QueryType.FACTUAL:
			base_weights['authority'] *= 1.3
			base_weights['similarity'] *= 1.2
		elif query_type == QueryType.PROCEDURAL:
			base_weights['authority'] *= 1.5
			base_weights['diversity'] *= 0.8
		elif query_type == QueryType.CONVERSATIONAL:
			base_weights['diversity'] *= 1.3
			base_weights['authority'] *= 0.9
		elif query_type == QueryType.COMPARATIVE:
			base_weights['diversity'] *= 1.4
			base_weights['relevance'] *= 1.2
		
		# Normalize weights
		total = sum(base_weights.values())
		return {k: v / total for k, v in base_weights.items()}

class IntelligentRetrievalEngine:
	"""Main retrieval engine with intelligent optimization"""
	
	def __init__(self,
	             config: RetrievalConfig,
	             vector_service: VectorService,
	             ollama_integration: AdvancedOllamaIntegration,
	             db_pool: Pool,
	             tenant_id: str,
	             capability_id: str = "rag"):
		
		self.config = config
		self.vector_service = vector_service
		self.ollama_integration = ollama_integration
		self.db_pool = db_pool
		self.tenant_id = tenant_id
		self.capability_id = capability_id
		
		# Core components
		self.query_analyzer = QueryAnalyzer()
		self.ranker = RetrievalRanker(config)
		
		# Caching
		self.query_cache = {}  # Simple in-memory cache
		self.expansion_cache = {}
		
		# Statistics
		self.stats = {
			'total_queries': 0,
			'cache_hits': 0,
			'cache_misses': 0,
			'average_query_time_ms': 0.0,
			'multi_stage_queries': 0,
			'expanded_queries': 0
		}
		
		self.logger = logging.getLogger(__name__)
	
	async def retrieve(self,
	                  request: RetrievalRequest,
	                  context: Optional[RetrievalContext] = None) -> RetrievalResult:
		"""Main retrieval method with intelligent optimization"""
		start_time = time.time()
		request_id = uuid7str()
		
		try:
			self.logger.info(f"[{request_id}] Starting retrieval for query: {request.query_text[:100]}...")
			
			# Check cache first
			cache_key = self._generate_cache_key(request)
			if self.config.enable_caching and cache_key in self.query_cache:
				cached_result, cached_time = self.query_cache[cache_key]
				if (datetime.now() - cached_time).total_seconds() < self.config.cache_ttl_minutes * 60:
					self.stats['cache_hits'] += 1
					self.logger.info(f"[{request_id}] Cache hit for query")
					return cached_result
			
			self.stats['cache_misses'] += 1
			
			# Analyze query
			query_analysis = await self.query_analyzer.analyze_query(request.query_text, context)
			self.logger.debug(f"[{request_id}] Query analysis: {query_analysis.query_type}, intent: {query_analysis.intent}")
			
			# Generate query embedding
			query_embedding = await self._generate_query_embedding(request.query_text)
			
			# Determine retrieval strategy
			strategy = self._select_retrieval_strategy(query_analysis, request)
			
			# Execute retrieval based on method and strategy
			raw_results = await self._execute_retrieval(
				request, query_embedding, query_analysis, strategy, request_id
			)
			
			# Rank results
			ranked_results = await self.ranker.rank_results(raw_results, query_analysis, context)
			
			# Limit to requested number
			final_results = ranked_results[:request.k_retrievals]
			
			# Create retrieval result
			processing_time_ms = (time.time() - start_time) * 1000
			
			result = RetrievalResult(
				tenant_id=self.tenant_id,
				query_text=request.query_text,
				query_embedding=query_embedding,
				query_hash=self._generate_query_hash(request.query_text),
				knowledge_base_id=request.knowledge_base_id,
				k_retrievals=request.k_retrievals,
				similarity_threshold=request.similarity_threshold,
				retrieved_chunk_ids=[r.chunk_id for r in final_results],
				similarity_scores=[r.similarity_score for r in final_results],
				retrieval_method=request.retrieval_method,
				retrieval_time_ms=int(processing_time_ms),
				total_candidates=len(raw_results),
				result_quality_score=self._calculate_result_quality(final_results),
				diversity_score=self._calculate_diversity_metric(final_results)
			)
			
			# Cache result
			if self.config.enable_caching:
				self.query_cache[cache_key] = (result, datetime.now())
			
			# Update statistics
			self._update_stats(processing_time_ms, query_analysis, strategy)
			
			self.logger.info(f"[{request_id}] Retrieval completed: {len(final_results)} results in {processing_time_ms:.1f}ms")
			return result
		
		except Exception as e:
			processing_time_ms = (time.time() - start_time) * 1000
			self.logger.error(f"[{request_id}] Retrieval failed: {str(e)}")
			
			# Return empty result with error info
			return RetrievalResult(
				tenant_id=self.tenant_id,
				query_text=request.query_text,
				query_embedding=[0.0] * 1024,
				query_hash=self._generate_query_hash(request.query_text),
				knowledge_base_id=request.knowledge_base_id,
				k_retrievals=request.k_retrievals,
				similarity_threshold=request.similarity_threshold,
				retrieved_chunk_ids=[],
				similarity_scores=[],
				retrieval_method=request.retrieval_method,
				retrieval_time_ms=int(processing_time_ms),
				total_candidates=0,
				result_quality_score=0.0,
				diversity_score=0.0
			)
	
	async def _generate_query_embedding(self, query_text: str) -> List[float]:
		"""Generate embedding for query text"""
		embedding_result = {}
		
		def embedding_callback(result):
			if result['success'] and result['embeddings']:
				embedding_result['embedding'] = result['embeddings'][0]
		
		# Request embedding
		await self.ollama_integration.generate_embeddings_async(
			texts=[query_text],
			model="bge-m3",
			tenant_id=self.tenant_id,
			capability_id=self.capability_id,
			priority=RequestPriority.HIGH,
			callback=embedding_callback
		)
		
		# Wait for result
		max_wait = 10.0
		wait_start = time.time()
		while 'embedding' not in embedding_result and (time.time() - wait_start) < max_wait:
			await asyncio.sleep(0.1)
		
		return embedding_result.get('embedding', [0.0] * 1024)
	
	def _select_retrieval_strategy(self, query_analysis: QueryAnalysis, request: RetrievalRequest) -> RetrievalStrategy:
		"""Select optimal retrieval strategy based on query analysis"""
		
		# High complexity queries benefit from recall-focused approach
		if query_analysis.complexity_score > 0.7:
			return RetrievalStrategy.RECALL_FOCUSED
		
		# Procedural queries need precise, authoritative results
		if query_analysis.query_type == QueryType.PROCEDURAL:
			return RetrievalStrategy.PRECISION_FOCUSED
		
		# Comparative queries benefit from diversity
		if query_analysis.query_type == QueryType.COMPARATIVE:
			return RetrievalStrategy.DIVERSITY_FOCUSED
		
		# Temporal indicators suggest recency focus
		if query_analysis.temporal_indicators:
			return RetrievalStrategy.RECENCY_FOCUSED
		
		# Default to balanced approach
		return RetrievalStrategy.BALANCED
	
	async def _execute_retrieval(self,
	                           request: RetrievalRequest,
	                           query_embedding: List[float],
	                           query_analysis: QueryAnalysis,
	                           strategy: RetrievalStrategy,
	                           request_id: str) -> List[Dict[str, Any]]:
		"""Execute retrieval based on method and strategy"""
		
		if request.retrieval_method == RetrievalMethod.VECTOR_SIMILARITY:
			return await self._vector_retrieval(request, query_embedding, strategy)
		
		elif request.retrieval_method == RetrievalMethod.HYBRID_SEARCH:
			return await self._hybrid_retrieval(request, query_embedding, query_analysis, strategy)
		
		elif request.retrieval_method == RetrievalMethod.KNOWLEDGE_GRAPH:
			return await self._knowledge_graph_retrieval(request, query_analysis, strategy)
		
		elif request.retrieval_method == RetrievalMethod.FULL_TEXT:
			return await self._full_text_retrieval(request, query_analysis, strategy)
		
		else:
			# Default to vector similarity
			return await self._vector_retrieval(request, query_embedding, strategy)
	
	async def _vector_retrieval(self,
	                          request: RetrievalRequest,
	                          query_embedding: List[float],
	                          strategy: RetrievalStrategy) -> List[Dict[str, Any]]:
		"""Pure vector similarity retrieval"""
		
		# Adjust parameters based on strategy
		k = request.k_retrievals
		threshold = request.similarity_threshold
		
		if strategy == RetrievalStrategy.RECALL_FOCUSED:
			k = min(int(k * 2), self.config.max_k)
			threshold = max(threshold - 0.1, self.config.min_similarity_threshold)
		elif strategy == RetrievalStrategy.PRECISION_FOCUSED:
			threshold = min(threshold + 0.1, 1.0)
		
		# Multi-stage retrieval if enabled
		if self.config.enable_multi_stage and k > 10:
			self.stats['multi_stage_queries'] += 1
			
			# First stage: retrieve more candidates
			first_stage_k = min(int(k * self.config.first_stage_multiplier), self.config.max_k)
			first_stage_results = await self.vector_service.vector_search(
				query_embedding=query_embedding,
				knowledge_base_id=request.knowledge_base_id,
				k=first_stage_k,
				similarity_threshold=threshold * 0.8,  # Lower threshold for first stage
				filters=request.filters
			)
			
			# Second stage: re-rank and filter
			return first_stage_results[:k]
		else:
			# Single-stage retrieval
			return await self.vector_service.vector_search(
				query_embedding=query_embedding,
				knowledge_base_id=request.knowledge_base_id,
				k=k,
				similarity_threshold=threshold,
				filters=request.filters
			)
	
	async def _hybrid_retrieval(self,
	                          request: RetrievalRequest,
	                          query_embedding: List[float],
	                          query_analysis: QueryAnalysis,
	                          strategy: RetrievalStrategy) -> List[Dict[str, Any]]:
		"""Hybrid vector + text retrieval"""
		
		# Adjust weights based on strategy and query type
		vector_weight = 0.7
		text_weight = 0.3
		
		if query_analysis.query_type == QueryType.FACTUAL:
			vector_weight = 0.8
			text_weight = 0.2
		elif query_analysis.query_type == QueryType.PROCEDURAL:
			vector_weight = 0.6
			text_weight = 0.4
		
		# Query expansion if enabled
		expanded_query = request.query_text
		if self.config.enable_query_expansion:
			expanded_query = await self._expand_query(request.query_text, query_analysis)
			if expanded_query != request.query_text:
				self.stats['expanded_queries'] += 1
		
		return await self.vector_service.hybrid_search(
			query_text=expanded_query,
			query_embedding=query_embedding,
			knowledge_base_id=request.knowledge_base_id,
			k=request.k_retrievals,
			vector_weight=vector_weight,
			text_weight=text_weight,
			similarity_threshold=request.similarity_threshold
		)
	
	async def _knowledge_graph_retrieval(self,
	                                   request: RetrievalRequest,
	                                   query_analysis: QueryAnalysis,
	                                   strategy: RetrievalStrategy) -> List[Dict[str, Any]]:
		"""Knowledge graph-based retrieval"""
		results = []
		
		try:
			async with self.db_pool.acquire() as conn:
				# Search for entities mentioned in query
				for entity in query_analysis.entities:
					entity_results = await conn.fetch("""
						SELECT * FROM rg_entity_search($1, $2, $3, NULL)
					""", self.tenant_id, request.knowledge_base_id, entity)
					
					for entity_row in entity_results:
						# Get chunks from documents related to this entity
						chunk_results = await conn.fetch("""
							SELECT 
								c.id as chunk_id,
								c.document_id,
								c.content,
								c.chunk_index,
								c.character_count,
								c.section_title,
								c.section_level,
								d.title as document_title,
								d.filename as document_filename,
								0.8 as similarity_score
							FROM apg_rag_document_chunks c
							JOIN apg_rag_documents d ON c.document_id = d.id  
							WHERE d.id = ANY($1) 
								AND c.tenant_id = $2
								AND c.knowledge_base_id = $3
							LIMIT $4
						""", entity_row['related_documents'], self.tenant_id, 
						     request.knowledge_base_id, request.k_retrievals)
						
						for chunk_row in chunk_results:
							result = dict(chunk_row)
							results.append(result)
		
		except Exception as e:
			self.logger.error(f"Knowledge graph retrieval failed: {str(e)}")
		
		return results[:request.k_retrievals]
	
	async def _full_text_retrieval(self,
	                             request: RetrievalRequest,
	                             query_analysis: QueryAnalysis,
	                             strategy: RetrievalStrategy) -> List[Dict[str, Any]]:
		"""Full-text search retrieval"""
		try:
			async with self.db_pool.acquire() as conn:
				# Prepare search query
				search_query = request.query_text
				if query_analysis.keywords:
					# Enhance with keywords
					search_query += " " + " ".join(query_analysis.keywords)
				
				results = await conn.fetch("""
					SELECT 
						c.id as chunk_id,
						c.document_id,
						c.content,
						c.chunk_index,
						c.character_count,
						c.section_title,
						c.section_level,
						d.title as document_title,
						d.filename as document_filename,
						ts_rank_cd(to_tsvector('english', c.content), plainto_tsquery('english', $1)) as similarity_score
					FROM apg_rag_document_chunks c
					JOIN apg_rag_documents d ON c.document_id = d.id
					WHERE c.tenant_id = $2
						AND c.knowledge_base_id = $3
						AND to_tsvector('english', c.content) @@ plainto_tsquery('english', $1)
					ORDER BY similarity_score DESC
					LIMIT $4
				""", search_query, self.tenant_id, request.knowledge_base_id, request.k_retrievals)
				
				return [dict(row) for row in results]
		
		except Exception as e:
			self.logger.error(f"Full-text retrieval failed: {str(e)}")
			return []
	
	async def _expand_query(self, query: str, query_analysis: QueryAnalysis) -> str:
		"""Expand query with related terms"""
		# Check cache first
		if query in self.expansion_cache:
			return self.expansion_cache[query]
		
		# Simple expansion using keywords and entities
		expansion_terms = []
		
		# Add synonyms or related terms (this could be enhanced with a thesaurus)
		for keyword in query_analysis.keywords[:3]:  # Limit to top 3 keywords
			# Simple synonym mapping (could be enhanced)
			synonyms = self._get_simple_synonyms(keyword)
			expansion_terms.extend(synonyms[:2])  # Add top 2 synonyms
		
		# Limit total expansion terms
		expansion_terms = expansion_terms[:self.config.max_expansion_terms]
		
		expanded_query = query
		if expansion_terms:
			expanded_query = f"{query} {' '.join(expansion_terms)}"
		
		# Cache the expansion
		self.expansion_cache[query] = expanded_query
		
		return expanded_query
	
	def _get_simple_synonyms(self, word: str) -> List[str]:
		"""Get simple synonyms for a word"""
		# Simple synonym mapping (could be enhanced with proper thesaurus)
		synonym_map = {
			'good': ['excellent', 'great'],
			'bad': ['poor', 'terrible'],
			'big': ['large', 'huge'],
			'small': ['tiny', 'little'],
			'fast': ['quick', 'rapid'],
			'slow': ['gradual', 'delayed']
		}
		
		return synonym_map.get(word.lower(), [])
	
	def _generate_cache_key(self, request: RetrievalRequest) -> str:
		"""Generate cache key for request"""
		import hashlib
		key_data = f"{request.query_text}:{request.knowledge_base_id}:{request.k_retrievals}:{request.similarity_threshold}:{request.retrieval_method.value}"
		if request.filters:
			key_data += f":{json.dumps(request.filters, sort_keys=True)}"
		return hashlib.md5(key_data.encode()).hexdigest()
	
	def _generate_query_hash(self, query: str) -> str:
		"""Generate hash for query text"""
		import hashlib
		return hashlib.md5(query.encode()).hexdigest()
	
	def _calculate_result_quality(self, results: List[RankedResult]) -> float:
		"""Calculate overall quality score for results"""
		if not results:
			return 0.0
		
		# Average of top scores
		top_scores = [r.final_score for r in results[:5]]  # Top 5 results
		return sum(top_scores) / len(top_scores)
	
	def _calculate_diversity_metric(self, results: List[RankedResult]) -> float:
		"""Calculate diversity metric for results"""
		if not results:
			return 0.0
		
		diversity_scores = [r.diversity_score for r in results]
		return sum(diversity_scores) / len(diversity_scores)
	
	def _update_stats(self, processing_time_ms: float, query_analysis: QueryAnalysis, strategy: RetrievalStrategy) -> None:
		"""Update retrieval statistics"""
		self.stats['total_queries'] += 1
		
		# Update average processing time
		current_avg = self.stats['average_query_time_ms']
		total_queries = self.stats['total_queries']
		self.stats['average_query_time_ms'] = (
			(current_avg * (total_queries - 1) + processing_time_ms) / total_queries
		)
	
	def get_statistics(self) -> Dict[str, Any]:
		"""Get comprehensive retrieval statistics"""
		cache_hit_rate = self.stats['cache_hits'] / max(1, self.stats['cache_hits'] + self.stats['cache_misses'])
		
		return {
			**self.stats,
			'cache_hit_rate': cache_hit_rate,
			'expansion_rate': self.stats['expanded_queries'] / max(1, self.stats['total_queries']),
			'multi_stage_rate': self.stats['multi_stage_queries'] / max(1, self.stats['total_queries']),
			'cache_size': len(self.query_cache),
			'expansion_cache_size': len(self.expansion_cache)
		}
	
	async def clear_caches(self) -> None:
		"""Clear all caches"""
		self.query_cache.clear()
		self.expansion_cache.clear()
		self.logger.info("Retrieval caches cleared")

# Factory function for APG integration  
async def create_retrieval_engine(
	tenant_id: str,
	capability_id: str,
	vector_service: VectorService,
	ollama_integration: AdvancedOllamaIntegration,
	db_pool: Pool,
	config: RetrievalConfig = None
) -> IntelligentRetrievalEngine:
	"""Create intelligent retrieval engine"""
	if config is None:
		config = RetrievalConfig()
	
	engine = IntelligentRetrievalEngine(
		config, vector_service, ollama_integration, db_pool, tenant_id, capability_id
	)
	
	return engine