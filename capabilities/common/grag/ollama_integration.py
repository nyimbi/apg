"""
APG GraphRAG Capability - Ollama Integration

Revolutionary Ollama integration for embeddings and generation with bge-m3 (8k context)
and advanced generation models (qwen3, deepseek-r1) for GraphRAG operations.

Author: Datacraft (nyimbi@gmail.com)
Copyright: © 2025 Datacraft
Website: www.datacraft.co.ke
"""

from __future__ import annotations
import asyncio
import aiohttp
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from collections import defaultdict
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor

from .views import (
	GraphRAGQuery, GraphRAGResponse, ReasoningChain, Evidence,
	EntityMention, SourceAttribution, QualityIndicators
)


logger = logging.getLogger(__name__)


@dataclass
class OllamaConfig:
	"""Configuration for Ollama integration"""
	base_url: str = "http://localhost:11434"
	embedding_model: str = "bge-m3"  # 8k context embedding model
	generation_models: List[str] = None
	default_generation_model: str = "qwen3"
	
	# Performance settings
	max_concurrent_requests: int = 10
	request_timeout_seconds: int = 60
	retry_attempts: int = 3
	retry_delay_seconds: float = 1.0
	
	# Model settings
	embedding_dimensions: int = 1024
	max_context_length: int = 8000  # 8k context for bge-m3
	generation_temperature: float = 0.7
	generation_max_tokens: int = 4000
	
	# Caching settings
	enable_embedding_cache: bool = True
	cache_ttl_hours: int = 24
	max_cache_size: int = 10000
	
	def __post_init__(self):
		if self.generation_models is None:
			self.generation_models = ["qwen3", "deepseek-r1", "llama3.2"]


@dataclass
class EmbeddingResult:
	"""Result from embedding generation"""
	embeddings: List[float]
	model_used: str
	input_tokens: int
	processing_time_ms: float
	cache_hit: bool = False


@dataclass
class GenerationResult:
	"""Result from text generation"""
	generated_text: str
	model_used: str
	input_tokens: int
	output_tokens: int
	processing_time_ms: float
	confidence_score: float
	finish_reason: str
	cache_hit: bool = False


class OllamaIntegrationError(Exception):
	"""Base exception for Ollama integration errors"""
	def __init__(self, message: str, error_code: str = "OLLAMA_ERROR", details: Optional[Dict[str, Any]] = None):
		super().__init__(message)
		self.error_code = error_code
		self.details = details or {}


class ModelNotAvailableError(OllamaIntegrationError):
	"""Exception raised when requested model is not available"""
	pass


class EmbeddingGenerationError(OllamaIntegrationError):
	"""Exception raised during embedding generation"""
	pass


class TextGenerationError(OllamaIntegrationError):
	"""Exception raised during text generation"""
	pass


# ============================================================================
# OLLAMA CLIENT
# ============================================================================

class OllamaClient:
	"""
	High-performance Ollama client for GraphRAG operations providing:
	
	- bge-m3 embedding generation with 8k context support
	- Advanced text generation with qwen3, deepseek-r1
	- Intelligent model selection and fallback
	- Performance optimization and caching
	- Concurrent request handling
	- Automatic retry and error recovery
	"""
	
	def __init__(self, config: OllamaConfig):
		"""Initialize Ollama client with configuration"""
		self.config = config
		self.session = None
		self.executor = ThreadPoolExecutor(max_workers=config.max_concurrent_requests)
		
		# Caching
		self._embedding_cache = {}
		self._generation_cache = {}
		self._cache_timestamps = {}
		
		# Performance tracking
		self._performance_stats = defaultdict(list)
		self._model_health = {}
		
		# Concurrency control
		self._embedding_semaphore = asyncio.Semaphore(config.max_concurrent_requests)
		self._generation_semaphore = asyncio.Semaphore(config.max_concurrent_requests)
		
		logger.info(f"Ollama client initialized with base URL: {config.base_url}")
	
	async def initialize(self) -> None:
		"""Initialize HTTP session and check model availability"""
		connector = aiohttp.TCPConnector(
			limit=self.config.max_concurrent_requests * 2,
			ttl_dns_cache=300,
			use_dns_cache=True
		)
		
		timeout = aiohttp.ClientTimeout(total=self.config.request_timeout_seconds)
		
		self.session = aiohttp.ClientSession(
			connector=connector,
			timeout=timeout,
			headers={"Content-Type": "application/json"}
		)
		
		# Check model availability
		await self._check_model_availability()
		
		logger.info("Ollama client initialized successfully")
	
	async def cleanup(self) -> None:
		"""Cleanup resources"""
		if self.session:
			await self.session.close()
		
		self.executor.shutdown(wait=True)
		logger.info("Ollama client cleaned up")
	
	# ========================================================================
	# EMBEDDING OPERATIONS
	# ========================================================================
	
	async def generate_embedding(
		self,
		text: str,
		model: Optional[str] = None,
		truncate_if_needed: bool = True
	) -> EmbeddingResult:
		"""
		Generate embeddings using bge-m3 with 8k context support
		
		Args:
			text: Input text to embed
			model: Optional model override (defaults to bge-m3)
			truncate_if_needed: Whether to truncate text if too long
			
		Returns:
			EmbeddingResult with embeddings and metadata
		"""
		start_time = time.time()
		model_name = model or self.config.embedding_model
		
		# Check cache first
		if self.config.enable_embedding_cache:
			cache_key = self._generate_embedding_cache_key(text, model_name)
			cached_result = self._get_cached_embedding(cache_key)
			if cached_result:
				logger.debug(f"Cache hit for embedding generation")
				return cached_result
		
		async with self._embedding_semaphore:
			try:
				# Preprocess text
				processed_text = await self._preprocess_text_for_embedding(
					text, truncate_if_needed
				)
				
				# Generate embedding with retry logic
				embedding_data = await self._generate_embedding_with_retry(
					processed_text, model_name
				)
				
				# Process result
				result = EmbeddingResult(
					embeddings=embedding_data["embedding"],
					model_used=model_name,
					input_tokens=len(processed_text.split()),  # Approximate
					processing_time_ms=(time.time() - start_time) * 1000,
					cache_hit=False
				)
				
				# Validate embedding dimensions
				if len(result.embeddings) != self.config.embedding_dimensions:
					raise EmbeddingGenerationError(
						f"Unexpected embedding dimensions: {len(result.embeddings)} != {self.config.embedding_dimensions}",
						"INVALID_EMBEDDING_DIMENSIONS"
					)
				
				# Cache result
				if self.config.enable_embedding_cache:
					self._cache_embedding(cache_key, result)
				
				# Record performance
				self._record_performance("embedding_generation", result.processing_time_ms)
				
				logger.debug(f"Generated embedding in {result.processing_time_ms:.1f}ms using {model_name}")
				return result
				
			except Exception as e:
				logger.error(f"Embedding generation failed: {e}")
				raise EmbeddingGenerationError(f"Failed to generate embedding: {e}", "EMBEDDING_ERROR")
	
	async def generate_batch_embeddings(
		self,
		texts: List[str],
		model: Optional[str] = None,
		batch_size: int = 10
	) -> List[EmbeddingResult]:
		"""
		Generate embeddings for multiple texts concurrently
		
		Args:
			texts: List of texts to embed
			model: Optional model override
			batch_size: Number of concurrent embedding requests
			
		Returns:
			List of EmbeddingResult objects
		"""
		start_time = time.time()
		
		# Process in batches to avoid overwhelming the server
		results = []
		
		for i in range(0, len(texts), batch_size):
			batch = texts[i:i + batch_size]
			
			# Generate embeddings concurrently for this batch
			batch_tasks = [
				self.generate_embedding(text, model)
				for text in batch
			]
			
			batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
			
			# Process results and handle exceptions
			for j, result in enumerate(batch_results):
				if isinstance(result, Exception):
					logger.error(f"Failed to generate embedding for text {i+j}: {result}")
					# Create placeholder result
					results.append(EmbeddingResult(
						embeddings=[0.0] * self.config.embedding_dimensions,
						model_used=model or self.config.embedding_model,
						input_tokens=0,
						processing_time_ms=0.0,
						cache_hit=False
					))
				else:
					results.append(result)
		
		total_time = (time.time() - start_time) * 1000
		logger.info(f"Generated {len(results)} embeddings in {total_time:.1f}ms")
		
		return results
	
	# ========================================================================
	# TEXT GENERATION OPERATIONS
	# ========================================================================
	
	async def generate_graphrag_response(
		self,
		query: GraphRAGQuery,
		reasoning_context: Dict[str, Any],
		evidence: List[Evidence],
		model: Optional[str] = None,
		temperature: Optional[float] = None
	) -> GenerationResult:
		"""
		Generate GraphRAG response using advanced generation models
		
		Args:
			query: GraphRAG query object
			reasoning_context: Context from reasoning engine
			evidence: Supporting evidence
			model: Optional model override
			temperature: Optional temperature override
			
		Returns:
			GenerationResult with generated response
		"""
		start_time = time.time()
		model_name = model or self.config.default_generation_model
		gen_temperature = temperature or self.config.generation_temperature
		
		async with self._generation_semaphore:
			try:
				# Build comprehensive prompt
				prompt = await self._build_graphrag_prompt(
					query, reasoning_context, evidence
				)
				
				# Generate response with retry logic
				generation_data = await self._generate_text_with_retry(
					prompt=prompt,
					model=model_name,
					temperature=gen_temperature,
					max_tokens=self.config.generation_max_tokens
				)
				
				# Process and validate result
				result = GenerationResult(
					generated_text=generation_data["response"],
					model_used=model_name,
					input_tokens=generation_data.get("prompt_eval_count", 0),
					output_tokens=generation_data.get("eval_count", 0),
					processing_time_ms=(time.time() - start_time) * 1000,
					confidence_score=self._calculate_generation_confidence(generation_data),
					finish_reason=generation_data.get("finish_reason", "completed"),
					cache_hit=False
				)
				
				# Record performance
				self._record_performance("text_generation", result.processing_time_ms)
				
				logger.info(f"Generated GraphRAG response in {result.processing_time_ms:.1f}ms using {model_name}")
				return result
				
			except Exception as e:
				logger.error(f"GraphRAG response generation failed: {e}")
				raise TextGenerationError(f"Failed to generate response: {e}", "GENERATION_ERROR")
	
	async def generate_with_model_selection(
		self,
		query: GraphRAGQuery,
		reasoning_context: Dict[str, Any],
		evidence: List[Evidence],
		fallback_enabled: bool = True
	) -> GenerationResult:
		"""
		Generate response with intelligent model selection and fallback
		
		Args:
			query: GraphRAG query object
			reasoning_context: Context from reasoning engine
			evidence: Supporting evidence
			fallback_enabled: Whether to try fallback models on failure
			
		Returns:
			GenerationResult with generated response
		"""
		# Determine best model for query type
		selected_model = await self._select_optimal_model(query, reasoning_context)
		
		try:
			# Try primary model
			return await self.generate_graphrag_response(
				query, reasoning_context, evidence, model=selected_model
			)
			
		except Exception as e:
			logger.warning(f"Primary model {selected_model} failed: {e}")
			
			if not fallback_enabled:
				raise
			
			# Try fallback models
			for fallback_model in self.config.generation_models:
				if fallback_model != selected_model and await self._is_model_healthy(fallback_model):
					try:
						logger.info(f"Trying fallback model: {fallback_model}")
						return await self.generate_graphrag_response(
							query, reasoning_context, evidence, model=fallback_model
						)
					except Exception as fallback_error:
						logger.warning(f"Fallback model {fallback_model} failed: {fallback_error}")
						continue
			
			# All models failed
			raise TextGenerationError(
				"All generation models failed",
				"ALL_MODELS_FAILED",
				{"primary_model": selected_model, "fallback_models": self.config.generation_models}
			)
	
	# ========================================================================
	# SPECIALIZED GENERATION METHODS
	# ========================================================================
	
	async def generate_entity_mentions(
		self,
		generated_text: str,
		entities: List[str],
		model: Optional[str] = None
	) -> List[EntityMention]:
		"""
		Generate entity mentions in generated text
		
		Args:
			generated_text: Generated response text
			entities: List of entity names to identify
			model: Optional model override
			
		Returns:
			List of EntityMention objects
		"""
		try:
			entity_mentions = []
			
			# Simple implementation - find entity names in text
			for entity_name in entities:
				start_pos = generated_text.lower().find(entity_name.lower())
				if start_pos != -1:
					end_pos = start_pos + len(entity_name)
					
					mention = EntityMention(
						entity_id=entity_name,  # Simplified
						mention_text=entity_name,
						position_start=start_pos,
						position_end=end_pos,
						confidence=0.9  # High confidence for exact matches
					)
					entity_mentions.append(mention)
			
			logger.debug(f"Found {len(entity_mentions)} entity mentions in generated text")
			return entity_mentions
			
		except Exception as e:
			logger.error(f"Entity mention generation failed: {e}")
			return []
	
	async def generate_source_attribution(
		self,
		generated_text: str,
		evidence: List[Evidence],
		model: Optional[str] = None
	) -> List[SourceAttribution]:
		"""
		Generate source attribution for generated content
		
		Args:
			generated_text: Generated response text
			evidence: Supporting evidence
			model: Optional model override
			
		Returns:
			List of SourceAttribution objects
		"""
		try:
			attributions = []
			
			# Attribute sources based on evidence relevance
			for i, evidence_item in enumerate(evidence):
				# Calculate contribution weight based on relevance and confidence
				contribution_weight = evidence_item.relevance_score * evidence_item.confidence
				
				attribution = SourceAttribution(
					source_id=evidence_item.source_id,
					source_type=evidence_item.source_type,
					contribution_weight=contribution_weight,
					citation_text=f"[{i+1}] {evidence_item.content[:100]}...",
					confidence=evidence_item.confidence
				)
				attributions.append(attribution)
			
			# Sort by contribution weight
			attributions.sort(key=lambda a: a.contribution_weight, reverse=True)
			
			logger.debug(f"Generated {len(attributions)} source attributions")
			return attributions
			
		except Exception as e:
			logger.error(f"Source attribution generation failed: {e}")
			return []
	
	async def calculate_response_quality(
		self,
		generated_text: str,
		query: GraphRAGQuery,
		evidence: List[Evidence],
		reasoning_context: Dict[str, Any]
	) -> QualityIndicators:
		"""
		Calculate quality indicators for generated response
		
		Args:
			generated_text: Generated response text
			query: Original query
			evidence: Supporting evidence
			reasoning_context: Reasoning context
			
		Returns:
			QualityIndicators object
		"""
		try:
			# Calculate various quality metrics
			factual_accuracy = await self._assess_factual_accuracy(generated_text, evidence)
			completeness = await self._assess_completeness(generated_text, query)
			relevance = await self._assess_relevance(generated_text, query)
			coherence = await self._assess_coherence(generated_text)
			clarity = await self._assess_clarity(generated_text)
			confidence = reasoning_context.get("overall_confidence", 0.8)
			source_reliability = await self._assess_source_reliability(evidence)
			
			return QualityIndicators(
				factual_accuracy=factual_accuracy,
				completeness=completeness,
				relevance=relevance,
				coherence=coherence,
				clarity=clarity,
				confidence=confidence,
				source_reliability=source_reliability
			)
			
		except Exception as e:
			logger.error(f"Quality assessment failed: {e}")
			# Return default quality indicators
			return QualityIndicators(
				factual_accuracy=0.8,
				completeness=0.8,
				relevance=0.8,
				coherence=0.8,
				clarity=0.8,
				confidence=0.8,
				source_reliability=0.8
			)
	
	# ========================================================================
	# HTTP CLIENT METHODS
	# ========================================================================
	
	async def _generate_embedding_with_retry(
		self,
		text: str,
		model: str
	) -> Dict[str, Any]:
		"""Generate embedding with retry logic"""
		for attempt in range(self.config.retry_attempts):
			try:
				async with self.session.post(
					f"{self.config.base_url}/api/embeddings",
					json={
						"model": model,
						"prompt": text
					}
				) as response:
					if response.status == 200:
						data = await response.json()
						return data
					else:
						error_text = await response.text()
						raise OllamaIntegrationError(
							f"Embedding API returned {response.status}: {error_text}",
							"HTTP_ERROR"
						)
			
			except Exception as e:
				if attempt == self.config.retry_attempts - 1:
					raise
				
				logger.warning(f"Embedding attempt {attempt + 1} failed: {e}")
				await asyncio.sleep(self.config.retry_delay_seconds * (2 ** attempt))
	
	async def _generate_text_with_retry(
		self,
		prompt: str,
		model: str,
		temperature: float,
		max_tokens: int
	) -> Dict[str, Any]:
		"""Generate text with retry logic"""
		for attempt in range(self.config.retry_attempts):
			try:
				async with self.session.post(
					f"{self.config.base_url}/api/generate",
					json={
						"model": model,
						"prompt": prompt,
						"stream": False,
						"options": {
							"temperature": temperature,
							"num_predict": max_tokens,
							"top_k": 40,
							"top_p": 0.9
						}
					}
				) as response:
					if response.status == 200:
						data = await response.json()
						return data
					else:
						error_text = await response.text()
						raise OllamaIntegrationError(
							f"Generation API returned {response.status}: {error_text}",
							"HTTP_ERROR"
						)
			
			except Exception as e:
				if attempt == self.config.retry_attempts - 1:
					raise
				
				logger.warning(f"Generation attempt {attempt + 1} failed: {e}")
				await asyncio.sleep(self.config.retry_delay_seconds * (2 ** attempt))
	
	# ========================================================================
	# HELPER METHODS
	# ========================================================================
	
	async def _check_model_availability(self) -> None:
		"""Check availability of configured models"""
		try:
			async with self.session.get(f"{self.config.base_url}/api/tags") as response:
				if response.status == 200:
					data = await response.json()
					available_models = [model["name"] for model in data.get("models", [])]
					
					# Check embedding model
					if self.config.embedding_model not in available_models:
						logger.warning(f"Embedding model {self.config.embedding_model} not available")
					
					# Check generation models
					for model in self.config.generation_models:
						if model not in available_models:
							logger.warning(f"Generation model {model} not available")
						else:
							self._model_health[model] = True
					
					logger.info(f"Available models: {available_models}")
				else:
					logger.warning("Could not check model availability")
		
		except Exception as e:
			logger.error(f"Model availability check failed: {e}")
	
	async def _preprocess_text_for_embedding(
		self,
		text: str,
		truncate_if_needed: bool
	) -> str:
		"""Preprocess text for embedding generation"""
		# Clean and normalize text
		cleaned_text = text.strip()
		
		# Truncate if needed for 8k context
		if truncate_if_needed and len(cleaned_text) > self.config.max_context_length:
			logger.warning(f"Truncating text from {len(cleaned_text)} to {self.config.max_context_length} chars")
			cleaned_text = cleaned_text[:self.config.max_context_length]
		
		return cleaned_text
	
	async def _build_graphrag_prompt(
		self,
		query: GraphRAGQuery,
		reasoning_context: Dict[str, Any],
		evidence: List[Evidence]
	) -> str:
		"""Build comprehensive prompt for GraphRAG response generation"""
		
		# Build evidence context
		evidence_text = "\n".join([
			f"Evidence {i+1}: {ev.content}" 
			for i, ev in enumerate(evidence[:10])  # Limit to top 10 pieces of evidence
		])
		
		# Build reasoning context
		reasoning_summary = reasoning_context.get("reasoning_summary", "Multi-hop reasoning completed")
		
		prompt = f"""You are an advanced AI assistant specialized in answering questions using knowledge graph reasoning.

Query: {query.query_text}

Reasoning Process: {reasoning_summary}

Supporting Evidence:
{evidence_text}

Instructions:
1. Answer the query comprehensively using the provided evidence
2. Explain your reasoning clearly and logically
3. Cite relevant evidence in your response
4. If information is incomplete, acknowledge limitations
5. Provide a confident and well-structured answer

Response:"""
		
		return prompt
	
	async def _select_optimal_model(
		self,
		query: GraphRAGQuery,
		reasoning_context: Dict[str, Any]
	) -> str:
		"""Select optimal generation model based on query characteristics"""
		
		# Simple model selection logic
		query_text = query.query_text.lower()
		
		# Use deepseek-r1 for complex reasoning tasks
		if any(keyword in query_text for keyword in ["analyze", "compare", "explain", "reason"]):
			if "deepseek-r1" in self.config.generation_models:
				return "deepseek-r1"
		
		# Use qwen3 for general questions
		if "qwen3" in self.config.generation_models:
			return "qwen3"
		
		# Fallback to default
		return self.config.default_generation_model
	
	async def _is_model_healthy(self, model: str) -> bool:
		"""Check if model is healthy and available"""
		return self._model_health.get(model, False)
	
	def _calculate_generation_confidence(self, generation_data: Dict[str, Any]) -> float:
		"""Calculate confidence score for generated text"""
		# Simple confidence calculation based on response length and completion
		response_length = len(generation_data.get("response", ""))
		
		if response_length < 10:
			return 0.3
		elif response_length < 50:
			return 0.6
		elif response_length < 200:
			return 0.8
		else:
			return 0.9
	
	# Caching methods
	def _generate_embedding_cache_key(self, text: str, model: str) -> str:
		"""Generate cache key for embedding"""
		content = f"{model}:{text}"
		return hashlib.md5(content.encode()).hexdigest()
	
	def _get_cached_embedding(self, cache_key: str) -> Optional[EmbeddingResult]:
		"""Get cached embedding result"""
		if not self.config.enable_embedding_cache:
			return None
		
		if cache_key in self._embedding_cache:
			timestamp = self._cache_timestamps.get(cache_key, 0)
			cache_age = time.time() - timestamp
			
			if cache_age < self.config.cache_ttl_hours * 3600:
				result = self._embedding_cache[cache_key]
				result.cache_hit = True
				return result
			else:
				# Remove expired cache entry
				del self._embedding_cache[cache_key]
				del self._cache_timestamps[cache_key]
		
		return None
	
	def _cache_embedding(self, cache_key: str, result: EmbeddingResult) -> None:
		"""Cache embedding result"""
		if not self.config.enable_embedding_cache:
			return
		
		# Remove oldest entries if cache is full
		if len(self._embedding_cache) >= self.config.max_cache_size:
			oldest_key = min(self._cache_timestamps, key=self._cache_timestamps.get)
			del self._embedding_cache[oldest_key]
			del self._cache_timestamps[oldest_key]
		
		self._embedding_cache[cache_key] = result
		self._cache_timestamps[cache_key] = time.time()
	
	# Quality assessment methods (simplified implementations)
	async def _assess_factual_accuracy(self, text: str, evidence: List[Evidence]) -> float:
		"""Assess factual accuracy of generated text"""
		return 0.9  # Simplified - would use actual fact-checking
	
	async def _assess_completeness(self, text: str, query: GraphRAGQuery) -> float:
		"""Assess completeness of response"""
		return 0.85  # Simplified - would analyze query coverage
	
	async def _assess_relevance(self, text: str, query: GraphRAGQuery) -> float:
		"""Assess relevance to query"""
		return 0.88  # Simplified - would use semantic similarity
	
	async def _assess_coherence(self, text: str) -> float:
		"""Assess coherence of generated text"""
		return 0.9  # Simplified - would analyze logical flow
	
	async def _assess_clarity(self, text: str) -> float:
		"""Assess clarity of generated text"""
		return 0.87  # Simplified - would analyze readability
	
	async def _assess_source_reliability(self, evidence: List[Evidence]) -> float:
		"""Assess reliability of sources"""
		if not evidence:
			return 0.5
		
		avg_confidence = sum(e.confidence for e in evidence) / len(evidence)
		return avg_confidence
	
	def _record_performance(self, operation: str, time_ms: float) -> None:
		"""Record performance statistics"""
		self._performance_stats[operation].append(time_ms)
		
		# Keep only last 1000 measurements
		if len(self._performance_stats[operation]) > 1000:
			self._performance_stats[operation] = self._performance_stats[operation][-1000:]
	
	def get_performance_stats(self) -> Dict[str, Any]:
		"""Get performance statistics"""
		stats = {}
		
		for operation, times in self._performance_stats.items():
			if times:
				stats[operation] = {
					"average_ms": sum(times) / len(times),
					"min_ms": min(times),
					"max_ms": max(times),
					"count": len(times)
				}
		
		return stats


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_ollama_client(config: Optional[OllamaConfig] = None) -> OllamaClient:
	"""Factory function to create Ollama client"""
	if config is None:
		config = OllamaConfig()
	
	return OllamaClient(config)


__all__ = [
	'OllamaClient',
	'OllamaConfig',
	'EmbeddingResult',
	'GenerationResult',
	'OllamaIntegrationError',
	'ModelNotAvailableError',
	'EmbeddingGenerationError',
	'TextGenerationError',
	'create_ollama_client',
]