"""
APG RAG Ollama Client

High-performance Ollama client for bge-m3 embeddings and qwen3/deepseek-r1 generation
with connection pooling, retry mechanisms, and comprehensive error handling.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import httpx
from contextlib import asynccontextmanager
import hashlib
from uuid_extensions import uuid7str

# APG-specific imports (will be available in APG environment)
try:
	from pydantic import BaseModel, Field, ConfigDict, AfterValidator
	from typing import Annotated
except ImportError:
	# Fallback for development
	pass

@dataclass
class OllamaConfig:
	"""Configuration for Ollama client"""
	base_url: str = "http://localhost:11434"
	embedding_model: str = "bge-m3"
	generation_models: List[str] = field(default_factory=lambda: ["qwen3", "deepseek-r1"])
	
	# Connection pooling
	max_connections: int = 100
	max_keepalive_connections: int = 20
	keepalive_expiry: float = 30.0
	
	# Timeouts
	connect_timeout: float = 10.0
	read_timeout: float = 300.0  # 5 minutes for large generation tasks
	write_timeout: float = 10.0
	
	# Retry configuration
	max_retries: int = 3
	retry_delay: float = 1.0
	backoff_factor: float = 2.0
	
	# Performance
	embedding_batch_size: int = 32
	generation_concurrent_limit: int = 10
	
	# Health check
	health_check_interval: float = 30.0
	health_check_timeout: float = 5.0
	
	# Caching
	enable_embedding_cache: bool = True
	cache_ttl_seconds: int = 3600  # 1 hour
	max_cache_size: int = 10000

@dataclass
class EmbeddingRequest:
	"""Request for generating embeddings"""
	texts: List[str]
	model: str = "bge-m3"
	normalize: bool = True
	tenant_id: Optional[str] = None
	request_id: str = field(default_factory=uuid7str)

@dataclass
class EmbeddingResponse:
	"""Response containing embeddings"""
	embeddings: List[List[float]]
	model: str
	dimensions: int
	processing_time_ms: float
	request_id: str
	cached: bool = False

@dataclass
class GenerationRequest:
	"""Request for text generation"""
	prompt: str
	model: str = "qwen3"
	max_tokens: int = 2048
	temperature: float = 0.7
	top_p: float = 0.9
	stop_sequences: List[str] = field(default_factory=list)
	stream: bool = False
	tenant_id: Optional[str] = None
	request_id: str = field(default_factory=uuid7str)

@dataclass
class GenerationResponse:
	"""Response containing generated text"""
	text: str
	model: str
	tokens_used: int
	processing_time_ms: float
	request_id: str
	finish_reason: str = "completed"

class OllamaConnectionError(Exception):
	"""Ollama connection related errors"""
	pass

class OllamaModelError(Exception):
	"""Ollama model related errors"""
	pass

class OllamaTimeoutError(Exception):
	"""Ollama timeout errors"""
	pass

class EmbeddingCache:
	"""Simple in-memory cache for embeddings"""
	
	def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
		self.cache: Dict[str, Tuple[List[List[float]], float]] = {}
		self.max_size = max_size
		self.ttl_seconds = ttl_seconds
		self._lock = asyncio.Lock()
	
	def _generate_key(self, texts: List[str], model: str) -> str:
		"""Generate cache key for text list and model"""
		content = f"{model}::{json.dumps(sorted(texts))}"
		return hashlib.md5(content.encode()).hexdigest()
	
	async def get(self, texts: List[str], model: str) -> Optional[List[List[float]]]:
		"""Get embeddings from cache if available and not expired"""
		key = self._generate_key(texts, model)
		
		async with self._lock:
			if key in self.cache:
				embeddings, timestamp = self.cache[key]
				if time.time() - timestamp < self.ttl_seconds:
					return embeddings
				else:
					del self.cache[key]
		
		return None
	
	async def set(self, texts: List[str], model: str, embeddings: List[List[float]]) -> None:
		"""Store embeddings in cache"""
		key = self._generate_key(texts, model)
		
		async with self._lock:
			# Evict old entries if cache is full
			if len(self.cache) >= self.max_size:
				# Remove oldest entries (simple FIFO)
				oldest_keys = list(self.cache.keys())[:len(self.cache) - self.max_size + 1]
				for old_key in oldest_keys:
					del self.cache[old_key]
			
			self.cache[key] = (embeddings, time.time())
	
	async def clear(self) -> None:
		"""Clear all cached embeddings"""
		async with self._lock:
			self.cache.clear()

class ModelHealthMonitor:
	"""Monitor health of Ollama models"""
	
	def __init__(self, client_config: OllamaConfig):
		self.config = client_config
		self.model_health: Dict[str, bool] = {}
		self.last_health_check: Dict[str, datetime] = {}
		self._lock = asyncio.Lock()
	
	async def check_model_health(self, client: httpx.AsyncClient, model: str) -> bool:
		"""Check if a specific model is healthy"""
		try:
			response = await client.post(
				f"{self.config.base_url}/api/show",
				json={"name": model},
				timeout=self.config.health_check_timeout
			)
			
			if response.status_code == 200:
				model_info = response.json()
				return "error" not in model_info
			
			return False
			
		except Exception as e:
			logging.warning(f"Health check failed for model {model}: {str(e)}")
			return False
	
	async def update_model_health(self, client: httpx.AsyncClient) -> None:
		"""Update health status for all configured models"""
		all_models = [self.config.embedding_model] + self.config.generation_models
		
		async with self._lock:
			for model in all_models:
				health = await self.check_model_health(client, model)
				self.model_health[model] = health
				self.last_health_check[model] = datetime.now()
	
	def is_model_healthy(self, model: str) -> bool:
		"""Check if model is currently healthy"""
		return self.model_health.get(model, False)
	
	def get_healthy_models(self, model_list: List[str]) -> List[str]:
		"""Get list of healthy models from given list"""
		return [model for model in model_list if self.is_model_healthy(model)]

class OllamaClient:
	"""High-performance Ollama client with advanced features"""
	
	def __init__(self, config: Optional[OllamaConfig] = None):
		self.config = config or OllamaConfig()
		self.embedding_cache = EmbeddingCache(
			max_size=self.config.max_cache_size,
			ttl_seconds=self.config.cache_ttl_seconds
		) if self.config.enable_embedding_cache else None
		
		self.health_monitor = ModelHealthMonitor(self.config)
		self._client: Optional[httpx.AsyncClient] = None
		self._generation_semaphore = asyncio.Semaphore(self.config.generation_concurrent_limit)
		self._last_health_check = datetime.min
		
		# Logging
		self.logger = logging.getLogger(__name__)
		
		# Statistics
		self.stats = {
			'embedding_requests': 0,
			'generation_requests': 0,
			'cache_hits': 0,
			'cache_misses': 0,
			'errors': 0,
			'retries': 0
		}
	
	@asynccontextmanager
	async def _get_client(self):
		"""Get or create HTTP client with connection pooling"""
		if self._client is None:
			limits = httpx.Limits(
				max_connections=self.config.max_connections,
				max_keepalive_connections=self.config.max_keepalive_connections,
				keepalive_expiry=self.config.keepalive_expiry
			)
			
			timeout = httpx.Timeout(
				connect=self.config.connect_timeout,
				read=self.config.read_timeout,
				write=self.config.write_timeout
			)
			
			self._client = httpx.AsyncClient(
				limits=limits,
				timeout=timeout,
				headers={'Content-Type': 'application/json'}
			)
		
		try:
			yield self._client
		finally:
			pass  # Keep client open for reuse
	
	async def _ensure_health_check(self) -> None:
		"""Ensure models are health checked recently"""
		now = datetime.now()
		if (now - self._last_health_check).total_seconds() > self.config.health_check_interval:
			async with self._get_client() as client:
				await self.health_monitor.update_model_health(client)
			self._last_health_check = now
	
	async def _retry_request(self, request_func, *args, **kwargs) -> Any:
		"""Execute request with retry logic"""
		last_exception = None
		
		for attempt in range(self.config.max_retries + 1):
			try:
				return await request_func(*args, **kwargs)
			
			except (httpx.TimeoutException, httpx.ConnectError) as e:
				last_exception = e
				if attempt < self.config.max_retries:
					delay = self.config.retry_delay * (self.config.backoff_factor ** attempt)
					self.logger.warning(f"Request attempt {attempt + 1} failed, retrying in {delay}s: {str(e)}")
					self.stats['retries'] += 1
					await asyncio.sleep(delay)
				else:
					break
			
			except Exception as e:
				# Don't retry on non-network errors
				self.stats['errors'] += 1
				raise e
		
		self.stats['errors'] += 1
		if isinstance(last_exception, httpx.TimeoutException):
			raise OllamaTimeoutError(f"Request timed out after {self.config.max_retries} attempts")
		else:
			raise OllamaConnectionError(f"Connection failed after {self.config.max_retries} attempts: {str(last_exception)}")
	
	def _log_request_start(self, request_type: str, request_id: str, details: str = "") -> None:
		"""Log request start for audit trail"""
		self.logger.info(f"[{request_id}] Starting {request_type} request {details}")
	
	def _log_request_complete(self, request_type: str, request_id: str, processing_time_ms: float, cached: bool = False) -> None:
		"""Log request completion for audit trail"""
		cache_info = " (cached)" if cached else ""
		self.logger.info(f"[{request_id}] Completed {request_type} request in {processing_time_ms:.1f}ms{cache_info}")
	
	def _log_request_error(self, request_type: str, request_id: str, error: str) -> None:
		"""Log request error for debugging"""
		self.logger.error(f"[{request_id}] {request_type} request failed: {error}")
	
	async def generate_embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
		"""Generate embeddings using bge-m3 model"""
		start_time = time.time()
		self._log_request_start("embedding", request.request_id, f"for {len(request.texts)} texts")
		
		try:
			await self._ensure_health_check()
			
			# Check if model is healthy
			if not self.health_monitor.is_model_healthy(request.model):
				raise OllamaModelError(f"Model {request.model} is not healthy")
			
			# Check cache first
			if self.embedding_cache:
				cached_embeddings = await self.embedding_cache.get(request.texts, request.model)
				if cached_embeddings is not None:
					processing_time_ms = (time.time() - start_time) * 1000
					self.stats['cache_hits'] += 1
					self._log_request_complete("embedding", request.request_id, processing_time_ms, cached=True)
					
					return EmbeddingResponse(
						embeddings=cached_embeddings,
						model=request.model,
						dimensions=len(cached_embeddings[0]) if cached_embeddings else 1024,
						processing_time_ms=processing_time_ms,
						request_id=request.request_id,
						cached=True
					)
				else:
					self.stats['cache_misses'] += 1
			
			# Process embeddings in batches for performance
			all_embeddings = []
			
			for i in range(0, len(request.texts), self.config.embedding_batch_size):
				batch_texts = request.texts[i:i + self.config.embedding_batch_size]
				batch_embeddings = await self._generate_embedding_batch(batch_texts, request.model)
				all_embeddings.extend(batch_embeddings)
			
			# Cache results if caching is enabled
			if self.embedding_cache:
				await self.embedding_cache.set(request.texts, request.model, all_embeddings)
			
			processing_time_ms = (time.time() - start_time) * 1000
			self.stats['embedding_requests'] += 1
			self._log_request_complete("embedding", request.request_id, processing_time_ms)
			
			return EmbeddingResponse(
				embeddings=all_embeddings,
				model=request.model,
				dimensions=len(all_embeddings[0]) if all_embeddings else 1024,
				processing_time_ms=processing_time_ms,
				request_id=request.request_id,
				cached=False
			)
		
		except Exception as e:
			self._log_request_error("embedding", request.request_id, str(e))
			raise
	
	async def _generate_embedding_batch(self, texts: List[str], model: str) -> List[List[float]]:
		"""Generate embeddings for a batch of texts"""
		async def _make_embedding_request():
			async with self._get_client() as client:
				response = await client.post(
					f"{self.config.base_url}/api/embeddings",
					json={
						"model": model,
						"prompt": texts[0] if len(texts) == 1 else texts,
						"options": {
							"num_predict": -1,  # Generate full embeddings
							"temperature": 0.0   # Deterministic embeddings
						}
					}
				)
				
				if response.status_code != 200:
					raise OllamaModelError(f"Embedding request failed: {response.status_code} - {response.text}")
				
				result = response.json()
				
				# Handle single text vs batch response format
				if "embedding" in result:
					return [result["embedding"]]
				elif "embeddings" in result:
					return result["embeddings"]
				else:
					raise OllamaModelError(f"Unexpected response format: {result}")
		
		return await self._retry_request(_make_embedding_request)
	
	async def generate_text(self, request: GenerationRequest) -> GenerationResponse:
		"""Generate text using qwen3 or deepseek-r1 models"""
		start_time = time.time()
		self._log_request_start("generation", request.request_id, f"with model {request.model}")
		
		try:
			await self._ensure_health_check()
			
			# Check if requested model is healthy, fallback to healthy alternatives
			available_models = self.health_monitor.get_healthy_models([request.model] + self.config.generation_models)
			if not available_models:
				raise OllamaModelError("No healthy generation models available")
			
			selected_model = available_models[0]  # Use first healthy model
			if selected_model != request.model:
				self.logger.info(f"[{request.request_id}] Fallback from {request.model} to {selected_model}")
			
			# Use semaphore to limit concurrent generation requests
			async with self._generation_semaphore:
				response = await self._generate_text_request(request, selected_model)
			
			processing_time_ms = (time.time() - start_time) * 1000
			self.stats['generation_requests'] += 1
			self._log_request_complete("generation", request.request_id, processing_time_ms)
			
			return GenerationResponse(
				text=response["response"],
				model=selected_model,
				tokens_used=response.get("eval_count", 0),
				processing_time_ms=processing_time_ms,
				request_id=request.request_id,
				finish_reason=response.get("done_reason", "completed")
			)
		
		except Exception as e:
			self._log_request_error("generation", request.request_id, str(e))
			raise
	
	async def _generate_text_request(self, request: GenerationRequest, model: str) -> Dict[str, Any]:
		"""Make text generation request to Ollama"""
		async def _make_generation_request():
			async with self._get_client() as client:
				payload = {
					"model": model,
					"prompt": request.prompt,
					"stream": request.stream,
					"options": {
						"num_predict": request.max_tokens,
						"temperature": request.temperature,
						"top_p": request.top_p,
					}
				}
				
				if request.stop_sequences:
					payload["options"]["stop"] = request.stop_sequences
				
				response = await client.post(
					f"{self.config.base_url}/api/generate",
					json=payload
				)
				
				if response.status_code != 200:
					raise OllamaModelError(f"Generation request failed: {response.status_code} - {response.text}")
				
				return response.json()
		
		return await self._retry_request(_make_generation_request)
	
	async def list_models(self) -> List[Dict[str, Any]]:
		"""List available models in Ollama"""
		async def _list_models_request():
			async with self._get_client() as client:
				response = await client.get(f"{self.config.base_url}/api/tags")
				
				if response.status_code != 200:
					raise OllamaConnectionError(f"Failed to list models: {response.status_code}")
				
				return response.json().get("models", [])
		
		return await self._retry_request(_list_models_request)
	
	async def get_model_info(self, model: str) -> Dict[str, Any]:
		"""Get detailed information about a specific model"""
		async def _get_model_info_request():
			async with self._get_client() as client:
				response = await client.post(
					f"{self.config.base_url}/api/show",
					json={"name": model}
				)
				
				if response.status_code != 200:
					raise OllamaModelError(f"Failed to get model info for {model}: {response.status_code}")
				
				return response.json()
		
		return await self._retry_request(_get_model_info_request)
	
	async def health_check(self) -> Dict[str, Any]:
		"""Comprehensive health check of Ollama service and models"""
		health_info = {
			"service_healthy": False,
			"models": {},
			"statistics": self.stats.copy(),
			"timestamp": datetime.now().isoformat()
		}
		
		try:
			# Check service availability
			async with self._get_client() as client:
				response = await client.get(f"{self.config.base_url}/api/tags", timeout=5.0)
				health_info["service_healthy"] = response.status_code == 200
			
			# Update model health if service is available
			if health_info["service_healthy"]:
				await self.health_monitor.update_model_health(self._client)
				health_info["models"] = self.health_monitor.model_health.copy()
			
		except Exception as e:
			health_info["error"] = str(e)
		
		return health_info
	
	async def close(self) -> None:
		"""Close the client and cleanup resources"""
		if self._client:
			await self._client.aclose()
			self._client = None
		
		if self.embedding_cache:
			await self.embedding_cache.clear()
		
		self.logger.info("Ollama client closed successfully")
	
	def get_statistics(self) -> Dict[str, Any]:
		"""Get client usage statistics"""
		return {
			**self.stats,
			"cache_hit_rate": self.stats['cache_hits'] / max(1, self.stats['cache_hits'] + self.stats['cache_misses']),
			"model_health": self.health_monitor.model_health.copy(),
			"last_health_check": {
				model: timestamp.isoformat() 
				for model, timestamp in self.health_monitor.last_health_check.items()
			}
		}

# Async context manager for easy client lifecycle management
@asynccontextmanager
async def ollama_client(config: Optional[OllamaConfig] = None):
	"""Async context manager for Ollama client"""
	client = OllamaClient(config)
	try:
		yield client
	finally:
		await client.close()

# Factory function for APG integration
async def create_ollama_client(
	base_url: str = "http://localhost:11434",
	embedding_model: str = "bge-m3",
	generation_models: List[str] = None,
	**kwargs
) -> OllamaClient:
	"""Factory function to create configured Ollama client"""
	if generation_models is None:
		generation_models = ["qwen3", "deepseek-r1"]
	
	config = OllamaConfig(
		base_url=base_url,
		embedding_model=embedding_model,
		generation_models=generation_models,
		**kwargs
	)
	
	client = OllamaClient(config)
	
	# Initial health check
	await client._ensure_health_check()
	
	return client