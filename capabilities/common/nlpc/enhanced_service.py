"""
APG Enhanced NLP Service

Advanced NLP service with intelligent model orchestration, ensemble processing,
and enterprise-grade performance optimization.

Builds on the foundation service with sophisticated model management,
automatic failover, and multi-model ensemble capabilities.
"""

import asyncio
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, AsyncGenerator, Callable
from contextlib import asynccontextmanager
from collections import defaultdict, deque
import numpy as np
from dataclasses import dataclass
import concurrent.futures
from uuid_extensions import uuid7str

from models import (
	TextDocument, NLPModel, ProcessingRequest, ProcessingResult,
	StreamingSession, StreamingChunk, SystemHealth,
	NLPTaskType, ModelProvider, ProcessingStatus, QualityLevel, LanguageCode
)
from service import NLPService, ModelConfig
from model_registry import ModelRegistry, ModelStatus, LoadBalanceStrategy

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class EnsembleConfig:
	"""Configuration for ensemble processing"""
	enabled: bool = True
	min_models: int = 2
	max_models: int = 4
	confidence_threshold: float = 0.8
	consensus_method: str = "weighted_voting"  # weighted_voting, majority, best_confidence
	timeout_seconds: int = 30

@dataclass
class ProcessingPipeline:
	"""Processing pipeline configuration"""
	preprocessing_steps: List[str]
	model_selection_strategy: str
	postprocessing_steps: List[str]
	validation_rules: List[str]
	fallback_enabled: bool = True

class EnhancedNLPService(NLPService):
	"""
	Enhanced NLP service with intelligent model orchestration.
	
	Advanced Features:
	- Multi-model ensemble processing
	- Intelligent model selection and load balancing
	- Automatic failover and retry mechanisms
	- Performance-based model routing
	- Real-time model health monitoring
	- Batch processing optimization
	- Advanced caching strategies
	"""
	
	def __init__(self, tenant_id: str, config: Optional[ModelConfig] = None):
		"""Initialize enhanced NLP service"""
		super().__init__(tenant_id, config)
		
		# Enhanced configuration
		self.enhanced_config = {
			"ensemble_processing": True,
			"max_retry_attempts": 3,
			"model_timeout_seconds": 60,
			"batch_processing": True,
			"max_batch_size": 50,
			"cache_enabled": True,
			"cache_ttl_seconds": 3600,
			"load_balance_strategy": "weighted_performance",
			"health_check_interval": 30,
			"performance_monitoring": True
		}
		
		# Update with user config
		if config:
			self.enhanced_config.update(getattr(config, '__dict__', {}))
		
		# Initialize model registry
		self.model_registry = ModelRegistry(
			tenant_id=tenant_id,
			config={
				"load_balance_strategy": self.enhanced_config["load_balance_strategy"],
				"health_check_interval": self.enhanced_config["health_check_interval"]
			}
		)
		
		# Ensemble processing
		self.ensemble_config = EnsembleConfig()
		
		# Processing pipelines by task type
		self.processing_pipelines: Dict[NLPTaskType, ProcessingPipeline] = {}
		
		# Performance optimization
		self._request_cache: Dict[str, Any] = {}
		self._batch_queue: asyncio.Queue = asyncio.Queue()
		self._batch_processor_task: Optional[asyncio.Task] = None
		
		# Monitoring and metrics
		self._performance_tracker = defaultdict(lambda: deque(maxlen=1000))
		self._error_tracker = defaultdict(lambda: deque(maxlen=100))
		
		# Concurrency management
		self._semaphore = asyncio.Semaphore(100)  # Limit concurrent requests
		self._model_selection_lock = asyncio.Lock()
		
		self._log_enhanced_service_initialized()
	
	def _log_enhanced_service_initialized(self) -> None:
		"""Log enhanced service initialization"""
		logger.info(f"Enhanced NLP service initialized for tenant: {self.tenant_id}")
		logger.info(f"Ensemble processing: {self.enhanced_config['ensemble_processing']}")
		logger.info(f"Batch processing: {self.enhanced_config['batch_processing']}")
		logger.info(f"Load balancing: {self.enhanced_config['load_balance_strategy']}")
	
	async def initialize_enhanced_models(self) -> None:
		"""Initialize enhanced model management"""
		await super().initialize_models()
		
		# Register existing models with the registry
		for model_id, model_info in self._models.items():
			if model_id in self._model_metadata:
				metadata = self._model_metadata[model_id]
				await self.model_registry.register_model(
					model_metadata=metadata,
					model_instance=model_info,
					load_priority=1,
					max_concurrent=10
				)
		
		# Start health monitoring
		await self.model_registry.start_health_monitoring()
		
		# Start batch processor if enabled
		if self.enhanced_config["batch_processing"]:
			self._batch_processor_task = asyncio.create_task(self._batch_processor_loop())
		
		# Initialize processing pipelines
		await self._initialize_processing_pipelines()
		
		self._log_enhanced_models_initialized()
	
	def _log_enhanced_models_initialized(self) -> None:
		"""Log enhanced models initialization"""
		logger.info("Enhanced model management initialized")
	
	async def _initialize_processing_pipelines(self) -> None:
		"""Initialize task-specific processing pipelines"""
		
		# Sentiment Analysis Pipeline
		self.processing_pipelines[NLPTaskType.SENTIMENT_ANALYSIS] = ProcessingPipeline(
			preprocessing_steps=["text_normalization", "emoji_handling", "negation_handling"],
			model_selection_strategy="accuracy_weighted",
			postprocessing_steps=["confidence_calibration", "result_validation"],
			validation_rules=["confidence_threshold_0.3", "sentiment_consistency"]
		)
		
		# Entity Recognition Pipeline
		self.processing_pipelines[NLPTaskType.NAMED_ENTITY_RECOGNITION] = ProcessingPipeline(
			preprocessing_steps=["text_normalization", "tokenization", "case_restoration"],
			model_selection_strategy="ensemble_voting",
			postprocessing_steps=["entity_linking", "disambiguation", "confidence_scoring"],
			validation_rules=["entity_overlap_check", "type_consistency"]
		)
		
		# Text Classification Pipeline
		self.processing_pipelines[NLPTaskType.TEXT_CLASSIFICATION] = ProcessingPipeline(
			preprocessing_steps=["text_cleaning", "feature_extraction", "dimensionality_reduction"],
			model_selection_strategy="performance_based",
			postprocessing_steps=["class_probability_normalization", "hierarchy_validation"],
			validation_rules=["probability_sum_check", "class_consistency"]
		)
		
		# Summarization Pipeline
		self.processing_pipelines[NLPTaskType.TEXT_SUMMARIZATION] = ProcessingPipeline(
			preprocessing_steps=["sentence_segmentation", "importance_scoring", "redundancy_removal"],
			model_selection_strategy="length_optimized",
			postprocessing_steps=["coherence_checking", "length_validation", "readability_scoring"],
			validation_rules=["summary_length_check", "coherence_threshold_0.7"]
		)
		
		logger.info(f"Initialized {len(self.processing_pipelines)} processing pipelines")
	
	async def process_text_enhanced(self, request: ProcessingRequest) -> ProcessingResult:
		"""
		Enhanced text processing with intelligent model selection and ensemble processing.
		"""
		assert request.tenant_id == self.tenant_id, "Request tenant must match service tenant"
		
		async with self._semaphore:  # Limit concurrency
			start_time = time.time()
			request_hash = self._calculate_request_hash(request)
			
			# Check cache first
			if self.enhanced_config["cache_enabled"]:
				cached_result = self._get_cached_result(request_hash)
				if cached_result:
					self._log_cache_hit(request.id)
					return cached_result
			
			try:
				# Get processing pipeline for task
				pipeline = self.processing_pipelines.get(request.task_type)
				
				# Preprocess text
				processed_text = await self._preprocess_text(request, pipeline)
				
				# Select optimal model(s)
				selected_models = await self._select_models_for_request(request)
				
				if not selected_models:
					raise RuntimeError(f"No available models for task: {request.task_type}")
				
				# Process with ensemble if enabled and applicable
				if (self.enhanced_config["ensemble_processing"] and 
					len(selected_models) >= self.ensemble_config.min_models and
					request.quality_level in [QualityLevel.ACCURATE, QualityLevel.BEST]):
					
					result = await self._process_with_ensemble(
						processed_text, request, selected_models, pipeline
					)
				else:
					# Single model processing with retry
					result = await self._process_with_retry(
						processed_text, request, selected_models[0], pipeline
					)
				
				# Postprocess results
				final_result = await self._postprocess_results(result, pipeline)
				
				# Cache result
				if self.enhanced_config["cache_enabled"]:
					self._cache_result(request_hash, final_result)
				
				# Update performance metrics
				processing_time = (time.time() - start_time) * 1000
				await self._update_enhanced_metrics(request, final_result, processing_time)
				
				self._log_enhanced_processing_complete(request.id, processing_time)
				
				return final_result
				
			except Exception as e:
				error_result = await self._handle_processing_error(request, str(e), start_time)
				return error_result
	
	def _calculate_request_hash(self, request: ProcessingRequest) -> str:
		"""Calculate hash for request caching"""
		import hashlib
		
		content = request.text_content or f"doc_{request.document_id}"
		hash_input = f"{content}_{request.task_type}_{request.quality_level}_{request.language}"
		return hashlib.md5(hash_input.encode()).hexdigest()
	
	def _get_cached_result(self, request_hash: str) -> Optional[ProcessingResult]:
		"""Get cached processing result"""
		if request_hash in self._request_cache:
			cached_data = self._request_cache[request_hash]
			if datetime.utcnow() - cached_data["timestamp"] < timedelta(seconds=self.enhanced_config["cache_ttl_seconds"]):
				return cached_data["result"]
			else:
				del self._request_cache[request_hash]  # Expired
		return None
	
	def _cache_result(self, request_hash: str, result: ProcessingResult) -> None:
		"""Cache processing result"""
		self._request_cache[request_hash] = {
			"result": result,
			"timestamp": datetime.utcnow()
		}
		
		# Cleanup old cache entries if too many
		if len(self._request_cache) > 1000:
			oldest_keys = sorted(
				self._request_cache.keys(),
				key=lambda k: self._request_cache[k]["timestamp"]
			)[:100]
			for key in oldest_keys:
				del self._request_cache[key]
	
	def _log_cache_hit(self, request_id: str) -> None:
		"""Log cache hit"""
		logger.debug(f"Cache hit for request: {request_id}")
	
	async def _preprocess_text(self, 
							   request: ProcessingRequest,
							   pipeline: Optional[ProcessingPipeline]) -> str:
		"""Preprocess text according to pipeline configuration"""
		text = request.text_content or await self._prepare_text_content(request)
		
		if not pipeline:
			return text
		
		processed_text = text
		
		for step in pipeline.preprocessing_steps:
			if step == "text_normalization":
				processed_text = self._normalize_text(processed_text)
			elif step == "emoji_handling":
				processed_text = self._handle_emojis(processed_text)
			elif step == "negation_handling":
				processed_text = self._handle_negations(processed_text)
			elif step == "tokenization":
				# Keep as text for now, tokenization happens in model
				pass
			elif step == "case_restoration":
				processed_text = self._restore_case(processed_text)
			elif step == "text_cleaning":
				processed_text = self._clean_text(processed_text)
		
		return processed_text
	
	def _normalize_text(self, text: str) -> str:
		"""Normalize text for processing"""
		import re
		# Remove extra whitespace
		text = re.sub(r'\s+', ' ', text)
		# Fix common encoding issues
		text = text.replace('"', '"').replace('"', '"').replace(''', "'").replace(''', "'")
		return text.strip()
	
	def _handle_emojis(self, text: str) -> str:
		"""Handle emojis in text"""
		# For now, keep emojis as they can carry sentiment information
		return text
	
	def _handle_negations(self, text: str) -> str:
		"""Handle negation patterns"""
		import re
		# Mark negations for better sentiment analysis
		negation_patterns = [
			(r'\bnot\s+(\w+)', r'NOT_\1'),
			(r'\bno\s+(\w+)', r'NO_\1'),  
			(r'\bnever\s+(\w+)', r'NEVER_\1'),
			(r"n't\s+(\w+)", r"NOT_\1")
		]
		
		for pattern, replacement in negation_patterns:
			text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
		
		return text
	
	def _restore_case(self, text: str) -> str:
		"""Restore proper case for entities"""
		# Simple implementation - would be more sophisticated in practice
		return text
	
	def _clean_text(self, text: str) -> str:
		"""Clean text for processing"""
		import re
		# Remove URLs
		text = re.sub(r'http\S+|www\S+', '[URL]', text)
		# Remove excessive punctuation
		text = re.sub(r'[!?]{2,}', '!!', text)
		text = re.sub(r'[.]{3,}', '...', text)
		return text
	
	async def _select_models_for_request(self, request: ProcessingRequest) -> List[str]:
		"""Select optimal models for processing request"""
		async with self._model_selection_lock:
			
			# Primary model selection
			primary_model = await self.model_registry.select_model(
				task_type=request.task_type,
				language=request.language,
				quality_level=request.quality_level,
				preferred_provider=request.preferred_provider,
				preferred_model_id=request.preferred_model
			)
			
			if not primary_model:
				return []
			
			selected_models = [primary_model]
			
			# For ensemble processing, select additional models
			if (self.enhanced_config["ensemble_processing"] and 
				request.quality_level in [QualityLevel.ACCURATE, QualityLevel.BEST]):
				
				# Get all candidate models for the task
				all_candidates = await self._get_all_task_candidates(request.task_type, request.language)
				
				# Filter out already selected models
				ensemble_candidates = [m for m in all_candidates if m != primary_model]
				
				# Select top models by performance
				ensemble_models = await self._select_ensemble_models(
					ensemble_candidates, 
					request.quality_level,
					self.ensemble_config.max_models - 1
				)
				
				selected_models.extend(ensemble_models)
			
			self._log_models_selected(request.id, selected_models)
			return selected_models
	
	async def _get_all_task_candidates(self, 
									   task_type: NLPTaskType, 
									   language: Optional[LanguageCode]) -> List[str]:
		"""Get all available models for a task"""
		candidates = []
		
		# This would normally query the model registry more efficiently
		# For now, we'll simulate by selecting multiple times with different strategies
		strategies = [LoadBalanceStrategy.FASTEST_RESPONSE, LoadBalanceStrategy.HIGHEST_ACCURACY]
		
		for strategy in strategies:
			# Temporarily change strategy
			old_strategy = self.model_registry._load_balance_strategy
			self.model_registry._load_balance_strategy = strategy
			
			model = await self.model_registry.select_model(
				task_type=task_type,
				language=language
			)
			
			if model and model not in candidates:
				candidates.append(model)
			
			# Restore original strategy
			self.model_registry._load_balance_strategy = old_strategy
		
		return candidates
	
	async def _select_ensemble_models(self, 
									  candidates: List[str],
									  quality_level: QualityLevel,
									  max_count: int) -> List[str]:
		"""Select best models for ensemble processing"""
		if not candidates:
			return []
		
		# Score models based on performance metrics
		model_scores = []
		
		for model_id in candidates:
			model_details = self.model_registry.get_model_details(model_id)
			if not model_details or not model_details["status"]["is_available"]:
				continue
			
			perf = model_details["performance"]
			
			# Calculate composite score
			accuracy_score = perf.get("success_rate_percent", 50) / 100
			speed_score = max(0, 1.0 - (perf.get("avg_latency_ms", 1000) / 1000))
			health_score = model_details["performance"].get("health_score", 0.5)
			
			# Weight based on quality level
			if quality_level == QualityLevel.ACCURATE:
				composite_score = accuracy_score * 0.6 + speed_score * 0.2 + health_score * 0.2
			else:  # BEST
				composite_score = accuracy_score * 0.5 + speed_score * 0.2 + health_score * 0.3
			
			model_scores.append((model_id, composite_score))
		
		# Sort by score and take top models
		model_scores.sort(key=lambda x: x[1], reverse=True)
		selected = [model_id for model_id, score in model_scores[:max_count]]
		
		return selected
	
	def _log_models_selected(self, request_id: str, models: List[str]) -> None:
		"""Log model selection"""
		logger.debug(f"Selected {len(models)} models for request {request_id}: {models}")
	
	async def _process_with_ensemble(self, 
									text: str,
									request: ProcessingRequest,
									model_ids: List[str],
									pipeline: Optional[ProcessingPipeline]) -> ProcessingResult:
		"""Process text using ensemble of models"""
		
		# Process with each model concurrently
		tasks = []
		for model_id in model_ids:
			task = asyncio.create_task(
				self._process_with_single_model(text, request, model_id, pipeline)
			)
			tasks.append((model_id, task))
		
		# Collect results with timeout
		model_results = []
		
		try:
			done, pending = await asyncio.wait(
				[task for _, task in tasks],
				timeout=self.ensemble_config.timeout_seconds,
				return_when=asyncio.ALL_COMPLETED
			)
			
			# Cancel pending tasks
			for task in pending:
				task.cancel()
			
			# Collect successful results
			for model_id, task in tasks:
				if task in done:
					try:
						result = await task
						if result.is_successful:
							model_results.append((model_id, result))
					except Exception as e:
						self._log_ensemble_model_failed(model_id, str(e))
			
		except asyncio.TimeoutError:
			# Cancel all tasks on timeout
			for _, task in tasks:
				task.cancel()
		
		if not model_results:
			raise RuntimeError("All ensemble models failed")
		
		# Combine results using configured consensus method
		final_result = await self._combine_ensemble_results(
			model_results, request, self.ensemble_config.consensus_method
		)
		
		self._log_ensemble_processing_complete(request.id, len(model_results))
		
		return final_result
	
	def _log_ensemble_model_failed(self, model_id: str, error: str) -> None:
		"""Log ensemble model failure"""
		logger.warning(f"Ensemble model failed {model_id}: {error}")
	
	def _log_ensemble_processing_complete(self, request_id: str, model_count: int) -> None:
		"""Log ensemble processing completion"""
		logger.info(f"Ensemble processing complete for {request_id}: {model_count} models")
	
	async def _process_with_single_model(self, 
										text: str,
										request: ProcessingRequest,
										model_id: str,
										pipeline: Optional[ProcessingPipeline]) -> ProcessingResult:
		"""Process text with a single model"""
		
		async with self.model_registry.acquire_model(model_id) as instance:
			start_time = time.time()
			
			try:
				# Execute processing based on provider
				provider = instance.metadata.provider
				
				if provider == ModelProvider.OLLAMA:
					results = await self._process_with_ollama(
						text, request.task_type, instance.instance, request.parameters
					)
				elif provider == ModelProvider.TRANSFORMERS:
					results = await self._process_with_transformers(
						text, request.task_type, instance.instance, request.parameters
					)
				elif provider == ModelProvider.SPACY:
					results = await self._process_with_spacy(
						text, request.task_type, instance.instance, request.parameters
					)
				else:
					raise ValueError(f"Unsupported provider: {provider}")
				
				processing_time = (time.time() - start_time) * 1000
				
				# Update model metrics
				await self.model_registry.update_model_metrics(
					model_id=model_id,
					latency_ms=processing_time,
					success=True,
					confidence=results.get("confidence", 0.0)
				)
				
				# Create result
				result = ProcessingResult(
					request_id=request.id,
					tenant_id=self.tenant_id,
					task_type=request.task_type,
					model_used=model_id,
					provider_used=provider,
					processing_time_ms=processing_time,
					total_time_ms=processing_time,
					results=results,
					confidence_score=results.get("confidence", 0.0),
					status=ProcessingStatus.COMPLETED
				)
				
				return result
				
			except Exception as e:
				processing_time = (time.time() - start_time) * 1000
				
				# Update model metrics for failure
				await self.model_registry.update_model_metrics(
					model_id=model_id,
					latency_ms=processing_time,
					success=False
				)
				
				raise e
	
	async def _process_with_retry(self, 
								 text: str,
								 request: ProcessingRequest,
								 model_id: str,
								 pipeline: Optional[ProcessingPipeline]) -> ProcessingResult:
		"""Process with retry mechanism"""
		
		last_error = None
		
		for attempt in range(self.enhanced_config["max_retry_attempts"]):
			try:
				result = await self._process_with_single_model(text, request, model_id, pipeline)
				return result
			
			except Exception as e:
				last_error = e
				self._log_retry_attempt(request.id, attempt + 1, str(e))
				
				if attempt < self.enhanced_config["max_retry_attempts"] - 1:
					# Wait before retry with exponential backoff
					wait_time = 2 ** attempt
					await asyncio.sleep(wait_time)
				else:
					# Last attempt - try fallback model if available
					fallback_model = await self._select_fallback_model(request.task_type, model_id)
					if fallback_model:
						try:
							result = await self._process_with_single_model(text, request, fallback_model, pipeline)
							self._log_fallback_success(request.id, fallback_model)
							return result
						except Exception as fallback_error:
							self._log_fallback_failed(request.id, str(fallback_error))
		
		raise last_error
	
	def _log_retry_attempt(self, request_id: str, attempt: int, error: str) -> None:
		"""Log retry attempt"""
		logger.warning(f"Retry attempt {attempt} for request {request_id}: {error}")
	
	def _log_fallback_success(self, request_id: str, fallback_model: str) -> None:
		"""Log successful fallback"""
		logger.info(f"Fallback successful for request {request_id} using model {fallback_model}")
	
	def _log_fallback_failed(self, request_id: str, error: str) -> None:
		"""Log fallback failure"""
		logger.error(f"Fallback failed for request {request_id}: {error}")
	
	async def _select_fallback_model(self, task_type: NLPTaskType, failed_model_id: str) -> Optional[str]:
		"""Select fallback model when primary model fails"""
		
		fallback_model = await self.model_registry.select_model(
			task_type=task_type,
			quality_level=QualityLevel.FAST  # Prefer fast models for fallback
		)
		
		# Make sure we don't select the same failed model
		if fallback_model == failed_model_id:
			# Try to get another candidate
			candidates = await self._get_all_task_candidates(task_type, None)
			for candidate in candidates:
				if candidate != failed_model_id:
					return candidate
		
		return fallback_model
	
	async def _combine_ensemble_results(self, 
									   model_results: List[tuple[str, ProcessingResult]],
									   request: ProcessingRequest,
									   consensus_method: str) -> ProcessingResult:
		"""Combine results from ensemble of models"""
		
		if len(model_results) == 1:
			return model_results[0][1]
		
		if consensus_method == "weighted_voting":
			return await self._weighted_voting_consensus(model_results, request)
		elif consensus_method == "majority":
			return await self._majority_consensus(model_results, request)
		elif consensus_method == "best_confidence":
			return await self._best_confidence_consensus(model_results, request)
		else:
			# Default to weighted voting
			return await self._weighted_voting_consensus(model_results, request)
	
	async def _weighted_voting_consensus(self, 
										model_results: List[tuple[str, ProcessingResult]],
										request: ProcessingRequest) -> ProcessingResult:
		"""Combine results using weighted voting based on model performance"""
		
		# Get weights based on model performance
		weights = []
		for model_id, result in model_results:
			model_details = self.model_registry.get_model_details(model_id)
			if model_details:
				performance = model_details["performance"]
				weight = (performance["success_rate_percent"] / 100) * (1.0 - performance["avg_latency_ms"] / 1000)
				weights.append(max(weight, 0.1))  # Minimum weight of 0.1
			else:
				weights.append(0.5)  # Default weight
		
		# Normalize weights
		total_weight = sum(weights)
		normalized_weights = [w / total_weight for w in weights]
		
		# Combine results based on task type
		if request.task_type == NLPTaskType.SENTIMENT_ANALYSIS:
			return await self._combine_sentiment_results(model_results, normalized_weights, request)
		elif request.task_type == NLPTaskType.TEXT_CLASSIFICATION:
			return await self._combine_classification_results(model_results, normalized_weights, request)
		elif request.task_type == NLPTaskType.NAMED_ENTITY_RECOGNITION:
			return await self._combine_ner_results(model_results, normalized_weights, request)
		else:
			# Default: return highest confidence result
			return max(model_results, key=lambda x: x[1].confidence_score)[1]
	
	async def _combine_sentiment_results(self, 
										model_results: List[tuple[str, ProcessingResult]],
										weights: List[float],
										request: ProcessingRequest) -> ProcessingResult:
		"""Combine sentiment analysis results"""
		
		# Aggregate sentiment scores
		sentiment_scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
		total_confidence = 0.0
		
		for (model_id, result), weight in zip(model_results, weights):
			results_data = result.results
			
			if "sentiment" in results_data and "confidence" in results_data:
				sentiment = results_data["sentiment"]
				confidence = results_data["confidence"]
				
				# Add weighted vote
				sentiment_scores[sentiment] += weight * confidence
				total_confidence += weight * confidence
		
		# Determine final sentiment
		final_sentiment = max(sentiment_scores, key=sentiment_scores.get)
		final_confidence = total_confidence / sum(weights) if sum(weights) > 0 else 0.0
		
		# Create combined result
		combined_results = {
			"sentiment": final_sentiment,
			"confidence": final_confidence,
			"scores": sentiment_scores,
			"ensemble_method": "weighted_voting",
			"model_count": len(model_results)
		}
		
		# Use first result as template
		base_result = model_results[0][1]
		
		return ProcessingResult(
			request_id=request.id,
			tenant_id=self.tenant_id,
			task_type=request.task_type,
			model_used="ensemble",
			provider_used=ModelProvider.CUSTOM,
			processing_time_ms=sum(r.processing_time_ms for _, r in model_results) / len(model_results),
			total_time_ms=max(r.total_time_ms for _, r in model_results),
			results=combined_results,
			confidence_score=final_confidence,
			status=ProcessingStatus.COMPLETED,
			metadata={"ensemble_models": [m for m, _ in model_results]}
		)
	
	async def _combine_classification_results(self, 
											 model_results: List[tuple[str, ProcessingResult]],
											 weights: List[float],
											 request: ProcessingRequest) -> ProcessingResult:
		"""Combine text classification results"""
		
		# Aggregate class probabilities
		class_scores = defaultdict(float)
		total_confidence = 0.0
		
		for (model_id, result), weight in zip(model_results, weights):
			results_data = result.results
			
			if "predicted_class" in results_data and "confidence" in results_data:
				predicted_class = results_data["predicted_class"]
				confidence = results_data["confidence"]
				
				# Add weighted vote
				class_scores[predicted_class] += weight * confidence
				total_confidence += weight * confidence
		
		# Determine final classification
		final_class = max(class_scores, key=class_scores.get) if class_scores else "unknown"
		final_confidence = total_confidence / sum(weights) if sum(weights) > 0 else 0.0
		
		# Create combined result
		combined_results = {
			"predicted_class": final_class,
			"confidence": final_confidence,
			"class_probabilities": dict(class_scores),
			"ensemble_method": "weighted_voting",
			"model_count": len(model_results)
		}
		
		# Use first result as template
		base_result = model_results[0][1]
		
		return ProcessingResult(
			request_id=request.id,
			tenant_id=self.tenant_id,
			task_type=request.task_type,
			model_used="ensemble",
			provider_used=ModelProvider.CUSTOM,
			processing_time_ms=sum(r.processing_time_ms for _, r in model_results) / len(model_results),
			total_time_ms=max(r.total_time_ms for _, r in model_results),
			results=combined_results,
			confidence_score=final_confidence,
			status=ProcessingStatus.COMPLETED,
			metadata={"ensemble_models": [m for m, _ in model_results]}
		)
	
	async def _combine_ner_results(self, 
								  model_results: List[tuple[str, ProcessingResult]],
								  weights: List[float],
								  request: ProcessingRequest) -> ProcessingResult:
		"""Combine named entity recognition results"""
		
		# Collect all entities from all models
		all_entities = []
		
		for (model_id, result), weight in zip(model_results, weights):
			results_data = result.results
			
			if "entities" in results_data:
				entities = results_data["entities"]
				for entity in entities:
					entity_copy = entity.copy()
					entity_copy["model_weight"] = weight
					entity_copy["source_model"] = model_id
					all_entities.append(entity_copy)
		
		# Merge overlapping entities
		merged_entities = self._merge_overlapping_entities(all_entities)
		
		# Create combined result
		combined_results = {
			"entities": merged_entities,
			"entity_count": len(merged_entities),
			"entity_types": list(set(e["label"] for e in merged_entities if "label" in e)),
			"ensemble_method": "weighted_merging",
			"model_count": len(model_results)
		}
		
		# Calculate average confidence
		avg_confidence = sum(e.get("confidence", 0.5) for e in merged_entities) / max(len(merged_entities), 1)
		
		# Use first result as template
		base_result = model_results[0][1]
		
		return ProcessingResult(
			request_id=request.id,
			tenant_id=self.tenant_id,
			task_type=request.task_type,
			model_used="ensemble",
			provider_used=ModelProvider.CUSTOM,
			processing_time_ms=sum(r.processing_time_ms for _, r in model_results) / len(model_results),
			total_time_ms=max(r.total_time_ms for _, r in model_results),
			results=combined_results,
			confidence_score=avg_confidence,
			status=ProcessingStatus.COMPLETED,
			metadata={"ensemble_models": [m for m, _ in model_results]}
		)
	
	def _merge_overlapping_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
		"""Merge overlapping entities from multiple models"""
		if not entities:
			return []
		
		# Sort entities by start position
		sorted_entities = sorted(entities, key=lambda e: e.get("start", 0))
		
		merged = []
		current_entity = None
		
		for entity in sorted_entities:
			if current_entity is None:
				current_entity = entity.copy()
			else:
				# Check for overlap
				current_start = current_entity.get("start", 0)
				current_end = current_entity.get("end", 0)
				entity_start = entity.get("start", 0)
				entity_end = entity.get("end", 0)
				
				# If overlapping and same type, merge
				if (entity_start <= current_end and 
					current_entity.get("label") == entity.get("label")):
					
					# Extend boundaries
					current_entity["start"] = min(current_start, entity_start)
					current_entity["end"] = max(current_end, entity_end)
					
					# Combine confidence (weighted average)
					current_weight = current_entity.get("model_weight", 1.0)
					entity_weight = entity.get("model_weight", 1.0)
					current_conf = current_entity.get("confidence", 0.5)
					entity_conf = entity.get("confidence", 0.5)
					
					total_weight = current_weight + entity_weight
					combined_confidence = (current_conf * current_weight + entity_conf * entity_weight) / total_weight
					current_entity["confidence"] = combined_confidence
					current_entity["model_weight"] = total_weight
				else:
					# No overlap, finalize current entity
					merged.append(current_entity)
					current_entity = entity.copy()
		
		# Add the last entity
		if current_entity:
			merged.append(current_entity)
		
		return merged
	
	async def _majority_consensus(self, 
								 model_results: List[tuple[str, ProcessingResult]],
								 request: ProcessingRequest) -> ProcessingResult:
		"""Simple majority voting consensus"""
		
		if request.task_type == NLPTaskType.SENTIMENT_ANALYSIS:
			# Count sentiment votes
			votes = {}
			for _, result in model_results:
				sentiment = result.results.get("sentiment", "neutral")
				votes[sentiment] = votes.get(sentiment, 0) + 1
			
			# Get majority sentiment
			majority_sentiment = max(votes, key=votes.get)
			
			# Calculate average confidence for majority sentiment
			majority_confidences = []
			for _, result in model_results:
				if result.results.get("sentiment") == majority_sentiment:
					majority_confidences.append(result.results.get("confidence", 0.5))
			
			avg_confidence = sum(majority_confidences) / len(majority_confidences)
			
			combined_results = {
				"sentiment": majority_sentiment,
				"confidence": avg_confidence,
				"votes": votes,
				"ensemble_method": "majority_voting"
			}
			
		else:
			# For other tasks, return highest confidence result
			best_result = max(model_results, key=lambda x: x[1].confidence_score)
			return best_result[1]
		
		# Create result using first result as template
		base_result = model_results[0][1]
		
		return ProcessingResult(
			request_id=request.id,
			tenant_id=self.tenant_id,
			task_type=request.task_type,
			model_used="ensemble",
			provider_used=ModelProvider.CUSTOM,
			processing_time_ms=sum(r.processing_time_ms for _, r in model_results) / len(model_results),
			total_time_ms=max(r.total_time_ms for _, r in model_results),
			results=combined_results,
			confidence_score=avg_confidence,
			status=ProcessingStatus.COMPLETED
		)
	
	async def _best_confidence_consensus(self, 
									    model_results: List[tuple[str, ProcessingResult]],
									    request: ProcessingRequest) -> ProcessingResult:
		"""Select result with highest confidence"""
		
		best_result = max(model_results, key=lambda x: x[1].confidence_score)
		
		# Add ensemble metadata
		result = best_result[1]
		result.metadata = result.metadata or {}
		result.metadata["ensemble_method"] = "best_confidence"
		result.metadata["model_count"] = len(model_results)
		result.metadata["selected_model"] = best_result[0]
		
		return result
	
	async def _postprocess_results(self, 
								  result: ProcessingResult,
								  pipeline: Optional[ProcessingPipeline]) -> ProcessingResult:
		"""Postprocess results according to pipeline configuration"""
		
		if not pipeline:
			return result
		
		processed_result = result
		
		for step in pipeline.postprocessing_steps:
			if step == "confidence_calibration":
				processed_result = self._calibrate_confidence(processed_result)
			elif step == "result_validation":
				processed_result = self._validate_result(processed_result, pipeline)
			elif step == "entity_linking":
				processed_result = await self._link_entities(processed_result)
			elif step == "coherence_checking":
				processed_result = self._check_coherence(processed_result)
		
		return processed_result
	
	def _calibrate_confidence(self, result: ProcessingResult) -> ProcessingResult:
		"""Calibrate confidence scores based on historical performance"""
		# Simple confidence calibration - in practice would use more sophisticated methods
		original_confidence = result.confidence_score
		
		# Apply calibration based on task type
		if result.task_type == NLPTaskType.SENTIMENT_ANALYSIS:
			# Sentiment analysis tends to be overconfident
			calibrated_confidence = original_confidence * 0.9
		elif result.task_type == NLPTaskType.NAMED_ENTITY_RECOGNITION:
			# NER tends to be underconfident
			calibrated_confidence = min(1.0, original_confidence * 1.1)
		else:
			calibrated_confidence = original_confidence
		
		result.confidence_score = calibrated_confidence
		
		# Update in results dict if present
		if "confidence" in result.results:
			result.results["confidence"] = calibrated_confidence
		
		return result
	
	def _validate_result(self, 
						result: ProcessingResult,
						pipeline: ProcessingPipeline) -> ProcessingResult:
		"""Validate result against pipeline rules"""
		
		validation_passed = True
		validation_issues = []
		
		for rule in pipeline.validation_rules:
			if rule.startswith("confidence_threshold_"):
				threshold = float(rule.split("_")[-1])
				if result.confidence_score < threshold:
					validation_passed = False
					validation_issues.append(f"Confidence below threshold: {result.confidence_score} < {threshold}")
			
			elif rule == "sentiment_consistency":
				# Check if sentiment and confidence are consistent
				if result.task_type == NLPTaskType.SENTIMENT_ANALYSIS:
					confidence = result.confidence_score
					sentiment = result.results.get("sentiment", "neutral")
					
					# Very low confidence should tend toward neutral
					if confidence < 0.3 and sentiment != "neutral":
						validation_issues.append("Low confidence but non-neutral sentiment")
			
			elif rule == "probability_sum_check":
				# Check if probabilities sum to ~1.0
				if "class_probabilities" in result.results:
					probs = result.results["class_probabilities"]
					total_prob = sum(probs.values())
					if abs(total_prob - 1.0) > 0.1:
						validation_issues.append(f"Probabilities don't sum to 1.0: {total_prob}")
		
		# Add validation metadata
		result.metadata = result.metadata or {}
		result.metadata["validation_passed"] = validation_passed
		if validation_issues:
			result.metadata["validation_issues"] = validation_issues
		
		return result
	
	async def _link_entities(self, result: ProcessingResult) -> ProcessingResult:
		"""Link entities to knowledge bases"""
		# Placeholder for entity linking - would integrate with knowledge bases
		if result.task_type == NLPTaskType.NAMED_ENTITY_RECOGNITION:
			entities = result.results.get("entities", [])
			for entity in entities:
				entity_type = entity.get("label")
				entity_text = entity.get("text", "")
				
				# Simple knowledge linking (in practice would query actual KBs)
				if entity_type == "PERSON":
					entity["wikipedia_url"] = f"https://en.wikipedia.org/wiki/{entity_text.replace(' ', '_')}"
				elif entity_type == "ORG":
					entity["wikipedia_url"] = f"https://en.wikipedia.org/wiki/{entity_text.replace(' ', '_')}"
		
		return result
	
	def _check_coherence(self, result: ProcessingResult) -> ProcessingResult:
		"""Check coherence of generated text"""
		if result.task_type == NLPTaskType.TEXT_SUMMARIZATION:
			summary = result.results.get("summary", "")
			
			# Simple coherence check - count sentence connections
			sentences = summary.split(".")
			coherence_score = 0.8  # Placeholder - would use actual coherence metrics
			
			result.metadata = result.metadata or {}
			result.metadata["coherence_score"] = coherence_score
		
		return result
	
	async def _handle_processing_error(self, 
									  request: ProcessingRequest,
									  error_message: str,
									  start_time: float) -> ProcessingResult:
		"""Handle processing errors and create error result"""
		
		processing_time = (time.time() - start_time) * 1000
		
		# Track error
		self._error_tracker[request.task_type].append({
			"timestamp": datetime.utcnow(),
			"error": error_message,
			"request_id": request.id
		})
		
		error_result = ProcessingResult(
			request_id=request.id,
			tenant_id=self.tenant_id,
			task_type=request.task_type,
			model_used="unknown",
			provider_used=ModelProvider.CUSTOM,
			processing_time_ms=processing_time,
			total_time_ms=processing_time,
			results={},
			status=ProcessingStatus.FAILED,
			error_message=error_message,
			error_code="PROCESSING_FAILED"
		)
		
		self._log_enhanced_processing_error(request.id, error_message)
		
		return error_result
	
	def _log_enhanced_processing_error(self, request_id: str, error: str) -> None:
		"""Log enhanced processing error"""
		logger.error(f"Enhanced processing failed for request {request_id}: {error}")
	
	def _log_enhanced_processing_complete(self, request_id: str, processing_time: float) -> None:
		"""Log enhanced processing completion"""
		logger.info(f"Enhanced processing complete for {request_id}: {processing_time:.2f}ms")
	
	async def _update_enhanced_metrics(self, 
									  request: ProcessingRequest,
									  result: ProcessingResult,
									  processing_time: float) -> None:
		"""Update enhanced performance metrics"""
		
		# Track performance by task type
		self._performance_tracker[request.task_type].append({
			"timestamp": datetime.utcnow(),
			"processing_time_ms": processing_time,
			"success": result.is_successful,
			"confidence": result.confidence_score,
			"model_used": result.model_used,
			"quality_level": request.quality_level
		})
	
	async def _batch_processor_loop(self) -> None:
		"""Background batch processor for high-throughput processing"""
		
		while True:
			try:
				batch_requests = []
				
				# Collect batch of requests (with timeout)
				try:
					# Get first request (blocking)
					first_request = await asyncio.wait_for(
						self._batch_queue.get(), 
						timeout=5.0
					)
					batch_requests.append(first_request)
					
					# Collect additional requests (non-blocking)
					while len(batch_requests) < self.enhanced_config["max_batch_size"]:
						try:
							request = self._batch_queue.get_nowait()
							batch_requests.append(request)
						except asyncio.QueueEmpty:
							break
				
				except asyncio.TimeoutError:
					# No requests in queue
					continue
				
				# Process batch
				if batch_requests:
					await self._process_batch(batch_requests)
			
			except asyncio.CancelledError:
				break
			except Exception as e:
				logger.error(f"Batch processor error: {e}")
				await asyncio.sleep(1)
	
	async def _process_batch(self, batch_requests: List[tuple]) -> None:
		"""Process a batch of requests efficiently"""
		
		# Group requests by task type and model
		task_groups = defaultdict(list)
		
		for request_data in batch_requests:
			request, future = request_data
			task_groups[request.task_type].append((request, future))
		
		# Process each task group
		for task_type, requests in task_groups.items():
			try:
				# Select optimal model for batch
				model_id = await self.model_registry.select_model(
					task_type=task_type,
					quality_level=QualityLevel.FAST  # Prefer speed for batch processing
				)
				
				if model_id:
					# Process requests in parallel with the same model
					tasks = []
					for request, future in requests:
						task = asyncio.create_task(
							self._process_single_batch_request(request, model_id)
						)
						tasks.append((task, future))
					
					# Wait for all to complete
					for task, future in tasks:
						try:
							result = await task
							future.set_result(result)
						except Exception as e:
							future.set_exception(e)
				else:
					# No model available - set errors
					for request, future in requests:
						error = RuntimeError(f"No model available for task: {task_type}")
						future.set_exception(error)
			
			except Exception as e:
				# Set error for all requests in group
				for request, future in requests:
					future.set_exception(e)
	
	async def _process_single_batch_request(self, 
										   request: ProcessingRequest,
										   model_id: str) -> ProcessingResult:
		"""Process single request in batch mode"""
		
		# Use faster processing path for batch
		async with self.model_registry.acquire_model(model_id) as instance:
			start_time = time.time()
			
			# Get text content
			text = request.text_content or await self._prepare_text_content(request)
			
			# Basic preprocessing
			processed_text = self._normalize_text(text)
			
			# Process with model
			provider = instance.metadata.provider
			
			if provider == ModelProvider.OLLAMA:
				results = await self._process_with_ollama(
					processed_text, request.task_type, instance.instance, request.parameters
				)
			elif provider == ModelProvider.TRANSFORMERS:
				results = await self._process_with_transformers(
					processed_text, request.task_type, instance.instance, request.parameters
				)
			elif provider == ModelProvider.SPACY:
				results = await self._process_with_spacy(
					processed_text, request.task_type, instance.instance, request.parameters
				)
			else:
				raise ValueError(f"Unsupported provider: {provider}")
			
			processing_time = (time.time() - start_time) * 1000
			
			# Update metrics
			await self.model_registry.update_model_metrics(
				model_id=model_id,
				latency_ms=processing_time,
				success=True,
				confidence=results.get("confidence", 0.0)
			)
			
			# Create result
			result = ProcessingResult(
				request_id=request.id,
				tenant_id=self.tenant_id,
				task_type=request.task_type,
				model_used=model_id,
				provider_used=provider,
				processing_time_ms=processing_time,
				total_time_ms=processing_time,
				results=results,
				confidence_score=results.get("confidence", 0.0),
				status=ProcessingStatus.COMPLETED
			)
			
			return result
	
	async def get_enhanced_system_health(self) -> Dict[str, Any]:
		"""Get enhanced system health with registry information"""
		
		# Get base health
		base_health = await self.get_system_health()
		
		# Get registry stats
		registry_stats = self.model_registry.get_registry_stats()
		
		# Calculate enhanced metrics
		recent_performance = []
		for task_type, metrics in self._performance_tracker.items():
			recent_metrics = [m for m in metrics if 
							 (datetime.utcnow() - m["timestamp"]).seconds < 300]  # Last 5 minutes
			
			if recent_metrics:
				avg_latency = sum(m["processing_time_ms"] for m in recent_metrics) / len(recent_metrics)
				success_rate = sum(1 for m in recent_metrics if m["success"]) / len(recent_metrics) * 100
				avg_confidence = sum(m["confidence"] for m in recent_metrics) / len(recent_metrics)
				
				recent_performance.append({
					"task_type": task_type.value,
					"request_count": len(recent_metrics),
					"avg_latency_ms": round(avg_latency, 2),
					"success_rate_percent": round(success_rate, 2),
					"avg_confidence": round(avg_confidence, 3)
				})
		
		# Enhanced health data
		enhanced_health = {
			"base_health": base_health.__dict__ if hasattr(base_health, '__dict__') else base_health,
			"model_registry": registry_stats,
			"recent_performance": recent_performance,
			"cache_stats": {
				"cache_size": len(self._request_cache),
				"max_cache_size": 1000,
				"cache_hit_rate": "not_implemented"  # Would track in practice
			},
			"batch_processing": {
				"enabled": self.enhanced_config["batch_processing"],
				"queue_size": self._batch_queue.qsize() if self._batch_queue else 0,
				"max_batch_size": self.enhanced_config["max_batch_size"]
			},
			"ensemble_processing": {
				"enabled": self.enhanced_config["ensemble_processing"],
				"min_models": self.ensemble_config.min_models,
				"max_models": self.ensemble_config.max_models,
				"consensus_method": self.ensemble_config.consensus_method
			}
		}
		
		return enhanced_health
	
	async def cleanup_enhanced(self) -> None:
		"""Enhanced cleanup with registry cleanup"""
		
		# Stop batch processor
		if self._batch_processor_task:
			self._batch_processor_task.cancel()
			try:
				await self._batch_processor_task
			except asyncio.CancelledError:
				pass
		
		# Cleanup model registry
		await self.model_registry.cleanup()
		
		# Clear caches and metrics
		self._request_cache.clear()
		self._performance_tracker.clear()
		self._error_tracker.clear()
		
		# Call parent cleanup
		await super().cleanup()
		
		logger.info("Enhanced NLP service cleanup completed")

# Export main class
__all__ = ["EnhancedNLPService", "EnsembleConfig", "ProcessingPipeline"]