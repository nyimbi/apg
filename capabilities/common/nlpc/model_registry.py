"""
APG NLP Model Registry

Enterprise model management system for NLP capabilities with intelligent
orchestration, performance monitoring, and automatic failover.

Supports multiple model providers: Ollama, Transformers, spaCy, NLTK, and custom models.
"""

import asyncio
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Set, Callable, AsyncGenerator
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import hashlib
import psutil
import threading
from contextlib import asynccontextmanager
from uuid_extensions import uuid7str

from models import (
	NLPModel, ModelProvider, NLPTaskType, LanguageCode, 
	ProcessingStatus, QualityLevel
)

# Configure logging
logger = logging.getLogger(__name__)

class ModelStatus(str, Enum):
	"""Model operational status"""
	INITIALIZING = "initializing"
	LOADING = "loading"
	READY = "ready"
	BUSY = "busy"
	ERROR = "error"
	UNLOADING = "unloading"
	OFFLINE = "offline"

class LoadBalanceStrategy(str, Enum):
	"""Load balancing strategies for model selection"""
	ROUND_ROBIN = "round_robin"
	LEAST_LOADED = "least_loaded"
	FASTEST_RESPONSE = "fastest_response"
	HIGHEST_ACCURACY = "highest_accuracy"
	WEIGHTED_PERFORMANCE = "weighted_performance"

@dataclass
class ModelMetrics:
	"""Real-time model performance metrics"""
	request_count: int = 0
	success_count: int = 0
	error_count: int = 0
	total_latency_ms: float = 0.0
	avg_latency_ms: float = 0.0
	min_latency_ms: float = float('inf')
	max_latency_ms: float = 0.0
	accuracy_scores: deque = field(default_factory=lambda: deque(maxlen=100))
	confidence_scores: deque = field(default_factory=lambda: deque(maxlen=100))
	memory_usage_mb: float = 0.0
	cpu_usage_percent: float = 0.0
	last_used: Optional[datetime] = None
	health_score: float = 1.0
	
	def update_request(self, latency_ms: float, success: bool, accuracy: float = None, confidence: float = None):
		"""Update metrics with new request data"""
		self.request_count += 1
		self.last_used = datetime.utcnow()
		
		if success:
			self.success_count += 1
			self.total_latency_ms += latency_ms
			self.avg_latency_ms = self.total_latency_ms / self.success_count
			self.min_latency_ms = min(self.min_latency_ms, latency_ms)
			self.max_latency_ms = max(self.max_latency_ms, latency_ms)
			
			if accuracy is not None:
				self.accuracy_scores.append(accuracy)
			if confidence is not None:
				self.confidence_scores.append(confidence)
		else:
			self.error_count += 1
		
		# Update health score based on success rate
		success_rate = self.success_count / self.request_count
		self.health_score = success_rate * 0.7 + (1.0 - min(self.avg_latency_ms / 1000, 1.0)) * 0.3

@dataclass  
class ModelInstance:
	"""Individual model instance with state and metrics"""
	id: str
	metadata: NLPModel
	instance: Any  # Actual model object
	status: ModelStatus = ModelStatus.INITIALIZING
	metrics: ModelMetrics = field(default_factory=ModelMetrics)
	load_priority: int = 0
	max_concurrent: int = 10
	current_requests: int = 0
	initialization_time: Optional[datetime] = None
	last_health_check: Optional[datetime] = None
	
	@property
	def is_available(self) -> bool:
		"""Check if model is available for processing"""
		return (self.status == ModelStatus.READY and 
				self.current_requests < self.max_concurrent and
				self.metrics.health_score > 0.5)
	
	@property
	def load_factor(self) -> float:
		"""Calculate current load factor (0.0 to 1.0)"""
		return self.current_requests / self.max_concurrent if self.max_concurrent > 0 else 1.0

class ModelRegistry:
	"""
	Enterprise model registry for intelligent NLP model management.
	
	Features:
	- Multi-provider model support (Ollama, Transformers, spaCy, NLTK)
	- Intelligent load balancing and failover
	- Real-time performance monitoring
	- Automatic model scaling and health management
	- Task-specific model selection
	"""
	
	def __init__(self, tenant_id: str, config: Dict[str, Any] = None):
		"""Initialize model registry"""
		assert tenant_id, "Tenant ID is required for model registry"
		
		self.tenant_id = tenant_id
		self.config = config or {}
		
		# Model storage
		self._models: Dict[str, ModelInstance] = {}
		self._models_by_provider: Dict[ModelProvider, List[str]] = defaultdict(list)
		self._models_by_task: Dict[NLPTaskType, List[str]] = defaultdict(list)
		self._models_by_language: Dict[LanguageCode, List[str]] = defaultdict(list)
		
		# Load balancing and selection
		self._load_balance_strategy = LoadBalanceStrategy(
			self.config.get("load_balance_strategy", "weighted_performance")
		)
		self._round_robin_counters: Dict[str, int] = defaultdict(int)
		
		# Monitoring and health
		self._health_check_interval = self.config.get("health_check_interval", 60)
		self._health_check_task: Optional[asyncio.Task] = None
		self._performance_history: deque = deque(maxlen=1000)
		
		# Concurrency management
		self._model_locks: Dict[str, asyncio.Lock] = {}
		self._initialization_lock = asyncio.Lock()
		
		self._log_registry_initialized()
	
	def _log_registry_initialized(self) -> None:
		"""Log registry initialization"""
		logger.info(f"Model registry initialized for tenant: {self.tenant_id}")
		logger.info(f"Load balance strategy: {self._load_balance_strategy}")
		logger.info(f"Health check interval: {self._health_check_interval}s")
	
	async def register_model(self, 
						   model_metadata: NLPModel, 
						   model_instance: Any,
						   load_priority: int = 0,
						   max_concurrent: int = 10) -> str:
		"""Register a new model instance"""
		assert model_metadata.tenant_id == self.tenant_id, "Model tenant must match registry tenant"
		
		async with self._initialization_lock:
			model_id = model_metadata.id
			
			# Create model instance wrapper
			instance = ModelInstance(
				id=model_id,
				metadata=model_metadata,
				instance=model_instance,
				load_priority=load_priority,
				max_concurrent=max_concurrent,
				initialization_time=datetime.utcnow()
			)
			
			# Create async lock for this model
			self._model_locks[model_id] = asyncio.Lock()
			
			# Store in registry
			self._models[model_id] = instance
			
			# Index by provider
			self._models_by_provider[model_metadata.provider].append(model_id)
			
			# Index by supported tasks
			for task in model_metadata.supported_tasks:
				self._models_by_task[task].append(model_id)
			
			# Index by supported languages
			for language in model_metadata.supported_languages:
				self._models_by_language[language].append(model_id)
			
			# Test model availability
			await self._health_check_model(model_id)
			
			self._log_model_registered(model_id, model_metadata.provider, len(model_metadata.supported_tasks))
			
			return model_id
	
	def _log_model_registered(self, model_id: str, provider: ModelProvider, task_count: int) -> None:
		"""Log model registration"""
		logger.info(f"Model registered: {model_id} ({provider}, {task_count} tasks)")
	
	async def unregister_model(self, model_id: str) -> bool:
		"""Unregister a model and cleanup resources"""
		if model_id not in self._models:
			return False
		
		async with self._initialization_lock:
			instance = self._models[model_id]
			
			# Set status to unloading
			instance.status = ModelStatus.UNLOADING
			
			# Wait for current requests to complete (with timeout)
			max_wait = 30  # seconds
			wait_count = 0
			while instance.current_requests > 0 and wait_count < max_wait:
				await asyncio.sleep(1)
				wait_count += 1
			
			# Remove from indexes
			metadata = instance.metadata
			self._models_by_provider[metadata.provider].remove(model_id)
			
			for task in metadata.supported_tasks:
				if model_id in self._models_by_task[task]:
					self._models_by_task[task].remove(model_id)
			
			for language in metadata.supported_languages:
				if model_id in self._models_by_language[language]:
					self._models_by_language[language].remove(model_id)
			
			# Cleanup
			del self._models[model_id]
			if model_id in self._model_locks:
				del self._model_locks[model_id]
			
			self._log_model_unregistered(model_id)
			
			return True
	
	def _log_model_unregistered(self, model_id: str) -> None:
		"""Log model unregistration"""
		logger.info(f"Model unregistered: {model_id}")
	
	async def select_model(self, 
						  task_type: NLPTaskType,
						  language: Optional[LanguageCode] = None,
						  quality_level: QualityLevel = QualityLevel.BALANCED,
						  preferred_provider: Optional[ModelProvider] = None,
						  preferred_model_id: Optional[str] = None) -> Optional[str]:
		"""Intelligently select the best model for a task"""
		
		# First, try preferred model if specified
		if preferred_model_id and preferred_model_id in self._models:
			instance = self._models[preferred_model_id]
			if (task_type in instance.metadata.supported_tasks and 
				instance.is_available):
				return preferred_model_id
		
		# Get candidate models for the task
		candidate_ids = self._models_by_task.get(task_type, [])
		
		if not candidate_ids:
			self._log_no_models_for_task(task_type)
			return None
		
		# Filter by language if specified
		if language and language != LanguageCode.AUTO:
			language_candidates = self._models_by_language.get(language, [])
			candidate_ids = [id for id in candidate_ids if id in language_candidates]
		
		# Filter by provider if specified
		if preferred_provider:
			provider_candidates = self._models_by_provider.get(preferred_provider, [])
			candidate_ids = [id for id in candidate_ids if id in provider_candidates]
		
		# Filter to only available models
		available_candidates = [
			id for id in candidate_ids 
			if id in self._models and self._models[id].is_available
		]
		
		if not available_candidates:
			self._log_no_available_models(task_type)
			return None
		
		# Select based on quality level and load balancing strategy
		selected_id = await self._select_by_strategy(
			available_candidates, quality_level, task_type
		)
		
		if selected_id:
			self._log_model_selected(selected_id, task_type, self._load_balance_strategy)
		
		return selected_id
	
	def _log_no_models_for_task(self, task_type: NLPTaskType) -> None:
		"""Log no models available for task"""
		logger.warning(f"No models registered for task: {task_type}")
	
	def _log_no_available_models(self, task_type: NLPTaskType) -> None:
		"""Log no available models"""
		logger.warning(f"No available models for task: {task_type}")
	
	def _log_model_selected(self, model_id: str, task_type: NLPTaskType, strategy: LoadBalanceStrategy) -> None:
		"""Log model selection"""
		logger.debug(f"Model selected: {model_id} for {task_type} using {strategy}")
	
	async def _select_by_strategy(self, 
								 candidates: List[str],
								 quality_level: QualityLevel,
								 task_type: NLPTaskType) -> Optional[str]:
		"""Select model using configured strategy"""
		
		if self._load_balance_strategy == LoadBalanceStrategy.ROUND_ROBIN:
			return self._select_round_robin(candidates, task_type)
		
		elif self._load_balance_strategy == LoadBalanceStrategy.LEAST_LOADED:
			return self._select_least_loaded(candidates)
		
		elif self._load_balance_strategy == LoadBalanceStrategy.FASTEST_RESPONSE:
			return self._select_fastest_response(candidates)
		
		elif self._load_balance_strategy == LoadBalanceStrategy.HIGHEST_ACCURACY:
			return self._select_highest_accuracy(candidates)
		
		elif self._load_balance_strategy == LoadBalanceStrategy.WEIGHTED_PERFORMANCE:
			return await self._select_weighted_performance(candidates, quality_level)
		
		else:
			# Default to first available
			return candidates[0] if candidates else None
	
	def _select_round_robin(self, candidates: List[str], task_type: NLPTaskType) -> str:
		"""Select model using round-robin strategy"""
		task_key = str(task_type)
		counter = self._round_robin_counters[task_key]
		selected = candidates[counter % len(candidates)]
		self._round_robin_counters[task_key] = (counter + 1) % len(candidates)
		return selected
	
	def _select_least_loaded(self, candidates: List[str]) -> str:
		"""Select model with lowest current load"""
		best_model = None
		min_load = float('inf')
		
		for model_id in candidates:
			instance = self._models[model_id]
			load = instance.load_factor
			
			if load < min_load:
				min_load = load
				best_model = model_id
		
		return best_model
	
	def _select_fastest_response(self, candidates: List[str]) -> str:
		"""Select model with fastest average response time"""
		best_model = None
		min_latency = float('inf')
		
		for model_id in candidates:
			instance = self._models[model_id]
			latency = instance.metrics.avg_latency_ms or float('inf')
			
			if latency < min_latency:
				min_latency = latency
				best_model = model_id
		
		return best_model
	
	def _select_highest_accuracy(self, candidates: List[str]) -> str:
		"""Select model with highest accuracy"""
		best_model = None
		max_accuracy = 0.0
		
		for model_id in candidates:
			instance = self._models[model_id]
			accuracy = (sum(instance.metrics.accuracy_scores) / len(instance.metrics.accuracy_scores)
					   if instance.metrics.accuracy_scores else instance.metadata.accuracy_score)
			
			if accuracy > max_accuracy:
				max_accuracy = accuracy
				best_model = model_id
		
		return best_model
	
	async def _select_weighted_performance(self, 
										  candidates: List[str], 
										  quality_level: QualityLevel) -> str:
		"""Select model using weighted performance score"""
		best_model = None
		best_score = -1.0
		
		# Quality level weights: [speed, accuracy, reliability]
		weights = {
			QualityLevel.FAST: [0.6, 0.2, 0.2],
			QualityLevel.BALANCED: [0.3, 0.4, 0.3], 
			QualityLevel.ACCURATE: [0.1, 0.7, 0.2],
			QualityLevel.BEST: [0.1, 0.6, 0.3]
		}
		
		w_speed, w_accuracy, w_reliability = weights[quality_level]
		
		for model_id in candidates:
			instance = self._models[model_id]
			metrics = instance.metrics
			
			# Speed score (inverted latency, normalized)
			speed_score = max(0, 1.0 - (metrics.avg_latency_ms / 1000.0)) if metrics.avg_latency_ms > 0 else 0.5
			
			# Accuracy score
			accuracy_score = (sum(metrics.accuracy_scores) / len(metrics.accuracy_scores)
							if metrics.accuracy_scores else instance.metadata.accuracy_score)
			
			# Reliability score (health + success rate)
			success_rate = metrics.success_count / metrics.request_count if metrics.request_count > 0 else 1.0
			reliability_score = (metrics.health_score + success_rate) / 2.0
			
			# Weighted composite score
			composite_score = (w_speed * speed_score + 
							  w_accuracy * accuracy_score + 
							  w_reliability * reliability_score)
			
			# Apply load factor penalty
			load_penalty = instance.load_factor * 0.2
			final_score = composite_score - load_penalty
			
			if final_score > best_score:
				best_score = final_score
				best_model = model_id
		
		return best_model
	
	@asynccontextmanager
	async def acquire_model(self, model_id: str):
		"""Acquire exclusive access to a model for processing"""
		if model_id not in self._models:
			raise ValueError(f"Model not found: {model_id}")
		
		instance = self._models[model_id]
		
		# Check if model is available
		if not instance.is_available:
			raise RuntimeError(f"Model not available: {model_id} (status: {instance.status})")
		
		# Acquire model lock
		async with self._model_locks[model_id]:
			# Increment request counter
			instance.current_requests += 1
			instance.status = ModelStatus.BUSY
			
			try:
				yield instance
			finally:
				# Decrement request counter
				instance.current_requests -= 1
				if instance.current_requests == 0:
					instance.status = ModelStatus.READY
	
	async def update_model_metrics(self, 
								  model_id: str,
								  latency_ms: float,
								  success: bool,
								  accuracy: Optional[float] = None,
								  confidence: Optional[float] = None) -> None:
		"""Update model performance metrics"""
		if model_id not in self._models:
			return
		
		instance = self._models[model_id]
		instance.metrics.update_request(latency_ms, success, accuracy, confidence)
		
		# Store performance history
		self._performance_history.append({
			"timestamp": datetime.utcnow(),
			"model_id": model_id,
			"latency_ms": latency_ms,
			"success": success,
			"accuracy": accuracy,
			"confidence": confidence
		})
	
	async def _health_check_model(self, model_id: str) -> bool:
		"""Perform health check on a specific model"""
		if model_id not in self._models:
			return False
		
		instance = self._models[model_id]
		
		try:
			# Basic availability check
			if instance.instance is None:
				instance.status = ModelStatus.ERROR
				return False
			
			# Update last health check
			instance.last_health_check = datetime.utcnow()
			
			# Check system resources
			instance.metrics.memory_usage_mb = psutil.Process().memory_info().rss / 1024 / 1024
			instance.metrics.cpu_usage_percent = psutil.cpu_percent(interval=0.1)
			
			# If model was in error state and passes basic checks, mark as ready
			if instance.status == ModelStatus.ERROR:
				instance.status = ModelStatus.READY
			
			return True
			
		except Exception as e:
			instance.status = ModelStatus.ERROR
			self._log_model_health_check_failed(model_id, str(e))
			return False
	
	def _log_model_health_check_failed(self, model_id: str, error: str) -> None:
		"""Log model health check failure"""
		logger.error(f"Health check failed for model {model_id}: {error}")
	
	async def start_health_monitoring(self) -> None:
		"""Start background health monitoring"""
		if self._health_check_task is not None:
			return  # Already running
		
		self._health_check_task = asyncio.create_task(self._health_check_loop())
		self._log_health_monitoring_started()
	
	def _log_health_monitoring_started(self) -> None:
		"""Log health monitoring start"""
		logger.info("Model health monitoring started")
	
	async def stop_health_monitoring(self) -> None:
		"""Stop background health monitoring"""
		if self._health_check_task is not None:
			self._health_check_task.cancel()
			try:
				await self._health_check_task
			except asyncio.CancelledError:
				pass
			self._health_check_task = None
		
		self._log_health_monitoring_stopped()
	
	def _log_health_monitoring_stopped(self) -> None:
		"""Log health monitoring stop"""
		logger.info("Model health monitoring stopped")
	
	async def _health_check_loop(self) -> None:
		"""Background health check loop"""
		while True:
			try:
				# Check all registered models
				for model_id in list(self._models.keys()):
					await self._health_check_model(model_id)
				
				# Wait for next check
				await asyncio.sleep(self._health_check_interval)
				
			except asyncio.CancelledError:
				break
			except Exception as e:
				logger.error(f"Health check loop error: {e}")
				await asyncio.sleep(5)  # Brief pause on error
	
	def get_registry_stats(self) -> Dict[str, Any]:
		"""Get comprehensive registry statistics"""
		total_models = len(self._models)
		ready_models = sum(1 for m in self._models.values() if m.status == ModelStatus.READY)
		busy_models = sum(1 for m in self._models.values() if m.status == ModelStatus.BUSY)
		error_models = sum(1 for m in self._models.values() if m.status == ModelStatus.ERROR)
		
		# Provider distribution
		provider_counts = {
			provider.value: len(model_ids) 
			for provider, model_ids in self._models_by_provider.items()
		}
		
		# Task coverage
		task_coverage = {
			task.value: len(model_ids)
			for task, model_ids in self._models_by_task.items()
		}
		
		# Performance aggregation
		total_requests = sum(m.metrics.request_count for m in self._models.values())
		total_success = sum(m.metrics.success_count for m in self._models.values())
		avg_success_rate = (total_success / total_requests * 100) if total_requests > 0 else 0
		
		avg_latency = sum(m.metrics.avg_latency_ms for m in self._models.values() 
						 if m.metrics.avg_latency_ms > 0) / max(ready_models, 1)
		
		return {
			"registry_id": f"nlp_registry_{self.tenant_id}",
			"total_models": total_models,
			"models_by_status": {
				"ready": ready_models,
				"busy": busy_models, 
				"error": error_models,
				"other": total_models - ready_models - busy_models - error_models
			},
			"provider_distribution": provider_counts,
			"task_coverage": task_coverage,
			"performance_summary": {
				"total_requests": total_requests,
				"success_rate_percent": round(avg_success_rate, 2),
				"average_latency_ms": round(avg_latency, 2)
			},
			"load_balance_strategy": self._load_balance_strategy.value,
			"health_check_interval": self._health_check_interval,
			"monitoring_active": self._health_check_task is not None
		}
	
	def get_model_details(self, model_id: str) -> Optional[Dict[str, Any]]:
		"""Get detailed information about a specific model"""
		if model_id not in self._models:
			return None
		
		instance = self._models[model_id]
		metrics = instance.metrics
		
		return {
			"model_id": model_id,
			"metadata": {
				"name": instance.metadata.name,
				"provider": instance.metadata.provider.value,
				"model_key": instance.metadata.model_key,
				"supported_tasks": [task.value for task in instance.metadata.supported_tasks],
				"supported_languages": [lang.value for lang in instance.metadata.supported_languages]
			},
			"status": {
				"current_status": instance.status.value,
				"is_available": instance.is_available,
				"current_requests": instance.current_requests,
				"max_concurrent": instance.max_concurrent,
				"load_factor": round(instance.load_factor, 3)
			},
			"performance": {
				"request_count": metrics.request_count,
				"success_count": metrics.success_count,
				"error_count": metrics.error_count,
				"success_rate_percent": round(metrics.success_count / max(metrics.request_count, 1) * 100, 2),
				"avg_latency_ms": round(metrics.avg_latency_ms, 2),
				"min_latency_ms": round(metrics.min_latency_ms, 2) if metrics.min_latency_ms != float('inf') else None,
				"max_latency_ms": round(metrics.max_latency_ms, 2),
				"health_score": round(metrics.health_score, 3),
				"last_used": metrics.last_used.isoformat() if metrics.last_used else None
			},
			"resources": {
				"memory_usage_mb": round(metrics.memory_usage_mb, 1),
				"cpu_usage_percent": round(metrics.cpu_usage_percent, 1)
			},
			"timestamps": {
				"initialized_at": instance.initialization_time.isoformat() if instance.initialization_time else None,
				"last_health_check": instance.last_health_check.isoformat() if instance.last_health_check else None
			}
		}
	
	async def cleanup(self) -> None:
		"""Cleanup registry resources"""
		# Stop health monitoring
		await self.stop_health_monitoring()
		
		# Unregister all models
		model_ids = list(self._models.keys())
		for model_id in model_ids:
			await self.unregister_model(model_id)
		
		# Clear data structures
		self._models.clear()
		self._models_by_provider.clear()
		self._models_by_task.clear()
		self._models_by_language.clear()
		self._model_locks.clear()
		self._round_robin_counters.clear()
		self._performance_history.clear()
		
		logger.info(f"Model registry cleanup completed for tenant: {self.tenant_id}")

# Export main classes
__all__ = ["ModelRegistry", "ModelInstance", "ModelMetrics", "ModelStatus", "LoadBalanceStrategy"]