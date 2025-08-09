"""
APG RAG Advanced Ollama Integration

Production-ready Ollama integration with advanced queuing, connection pooling,
load balancing, and comprehensive monitoring for enterprise RAG operations.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import httpx
from contextlib import asynccontextmanager
import hashlib
from uuid_extensions import uuid7str
import heapq
from collections import defaultdict, deque

# APG-specific imports (will be available in APG environment)
try:
	from pydantic import BaseModel, Field, ConfigDict, AfterValidator
	from typing import Annotated
except ImportError:
	# Fallback for development
	pass

class RequestPriority(Enum):
	"""Request priority levels for queue management"""
	LOW = 1
	NORMAL = 2
	HIGH = 3
	CRITICAL = 4

class QueueStatus(Enum):
	"""Queue status for monitoring"""
	ACTIVE = "active"
	PAUSED = "paused"
	DRAINING = "draining"
	STOPPED = "stopped"

@dataclass
class OllamaRequest:
	"""Base request for Ollama operations"""
	request_id: str = field(default_factory=uuid7str)
	tenant_id: str = ""
	capability_id: str = ""
	priority: RequestPriority = RequestPriority.NORMAL
	created_at: datetime = field(default_factory=datetime.now)
	timeout: float = 300.0
	retries_remaining: int = 3
	callback: Optional[Callable] = None
	
	def __lt__(self, other):
		"""For priority queue ordering"""
		return (self.priority.value, self.created_at) > (other.priority.value, other.created_at)

@dataclass
class EmbeddingQueueRequest(OllamaRequest):
	"""Embedding request for queue processing"""
	texts: List[str] = field(default_factory=list)
	model: str = "bge-m3"
	normalize: bool = True
	batch_optimize: bool = True

@dataclass
class GenerationQueueRequest(OllamaRequest):
	"""Generation request for queue processing"""
	prompt: str = ""
	model: str = "qwen3"
	max_tokens: int = 2048
	temperature: float = 0.7
	top_p: float = 0.9
	stop_sequences: List[str] = field(default_factory=list)
	stream: bool = False

@dataclass
class QueueMetrics:
	"""Queue performance metrics"""
	total_requests: int = 0
	completed_requests: int = 0
	failed_requests: int = 0
	current_queue_size: int = 0
	average_wait_time_ms: float = 0.0
	average_processing_time_ms: float = 0.0
	throughput_per_minute: float = 0.0
	last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class LoadBalancerConfig:
	"""Load balancer configuration"""
	strategy: str = "round_robin"  # round_robin, least_connections, weighted
	health_check_interval: float = 30.0
	failure_threshold: int = 3
	recovery_threshold: int = 2
	circuit_breaker_timeout: float = 60.0

class OllamaEndpoint:
	"""Represents an Ollama endpoint with health tracking"""
	
	def __init__(self, url: str, weight: float = 1.0):
		self.url = url
		self.weight = weight
		self.active_connections = 0
		self.total_requests = 0
		self.failed_requests = 0
		self.last_health_check = datetime.min
		self.is_healthy = True
		self.circuit_breaker_until = datetime.min
		self.response_times = deque(maxlen=100)  # Last 100 response times
	
	@property
	def average_response_time(self) -> float:
		"""Get average response time in milliseconds"""
		return sum(self.response_times) / len(self.response_times) if self.response_times else 0.0
	
	@property
	def failure_rate(self) -> float:
		"""Get failure rate as percentage"""
		return (self.failed_requests / max(1, self.total_requests)) * 100
	
	def record_request(self, processing_time_ms: float, success: bool):
		"""Record request metrics"""
		self.total_requests += 1
		if not success:
			self.failed_requests += 1
		else:
			self.response_times.append(processing_time_ms)
	
	def is_available(self) -> bool:
		"""Check if endpoint is available for requests"""
		return self.is_healthy and datetime.now() > self.circuit_breaker_until

class LoadBalancer:
	"""Advanced load balancer for Ollama endpoints"""
	
	def __init__(self, config: LoadBalancerConfig):
		self.config = config
		self.endpoints: List[OllamaEndpoint] = []
		self.round_robin_index = 0
		self.logger = logging.getLogger(__name__)
	
	def add_endpoint(self, url: str, weight: float = 1.0):
		"""Add an endpoint to the load balancer"""
		endpoint = OllamaEndpoint(url, weight)
		self.endpoints.append(endpoint)
		self.logger.info(f"Added Ollama endpoint: {url} with weight {weight}")
	
	def select_endpoint(self) -> Optional[OllamaEndpoint]:
		"""Select best available endpoint based on strategy"""
		available_endpoints = [ep for ep in self.endpoints if ep.is_available()]
		
		if not available_endpoints:
			return None
		
		if self.config.strategy == "round_robin":
			endpoint = available_endpoints[self.round_robin_index % len(available_endpoints)]
			self.round_robin_index += 1
			return endpoint
		
		elif self.config.strategy == "least_connections":
			return min(available_endpoints, key=lambda ep: ep.active_connections)
		
		elif self.config.strategy == "weighted":
			# Weighted random selection based on inverse response time
			weights = []
			for ep in available_endpoints:
				avg_time = ep.average_response_time or 1.0
				weight = ep.weight / avg_time
				weights.append(weight)
			
			total_weight = sum(weights)
			if total_weight > 0:
				import random
				r = random.uniform(0, total_weight)
				cumulative_weight = 0
				for i, weight in enumerate(weights):
					cumulative_weight += weight
					if r <= cumulative_weight:
						return available_endpoints[i]
			
			return available_endpoints[0]
		
		return available_endpoints[0]
	
	async def health_check_all(self, client: httpx.AsyncClient):
		"""Perform health check on all endpoints"""
		for endpoint in self.endpoints:
			await self._health_check_endpoint(client, endpoint)
	
	async def _health_check_endpoint(self, client: httpx.AsyncClient, endpoint: OllamaEndpoint):
		"""Health check for individual endpoint"""
		try:
			response = await client.get(
				f"{endpoint.url}/api/tags",
				timeout=self.config.health_check_interval / 2
			)
			
			was_healthy = endpoint.is_healthy
			endpoint.is_healthy = response.status_code == 200
			endpoint.last_health_check = datetime.now()
			
			# Circuit breaker recovery
			if endpoint.is_healthy and not was_healthy:
				endpoint.circuit_breaker_until = datetime.min
				self.logger.info(f"Endpoint {endpoint.url} recovered")
			
		except Exception as e:
			endpoint.is_healthy = False
			endpoint.last_health_check = datetime.now()
			
			# Circuit breaker activation
			if endpoint.failure_rate > self.config.failure_threshold * 10:
				endpoint.circuit_breaker_until = datetime.now() + timedelta(seconds=self.config.circuit_breaker_timeout)
				self.logger.warning(f"Circuit breaker activated for {endpoint.url}: {str(e)}")

class RequestQueue:
	"""Advanced priority queue with batching and throttling"""
	
	def __init__(self, max_size: int = 10000, batch_size: int = 32):
		self.max_size = max_size
		self.batch_size = batch_size
		self.queue: List[OllamaRequest] = []
		self.status = QueueStatus.ACTIVE
		self.metrics = QueueMetrics()
		self._lock = asyncio.Lock()
		self._condition = asyncio.Condition(self._lock)
		self.logger = logging.getLogger(__name__)
	
	async def enqueue(self, request: OllamaRequest) -> bool:
		"""Add request to queue with priority ordering"""
		async with self._lock:
			if len(self.queue) >= self.max_size:
				self.logger.warning(f"Queue full, rejecting request {request.request_id}")
				return False
			
			heapq.heappush(self.queue, request)
			self.metrics.total_requests += 1
			self.metrics.current_queue_size = len(self.queue)
			
			# Notify waiting consumers
			self._condition.notify()
			
			self.logger.debug(f"Enqueued request {request.request_id} with priority {request.priority}")
			return True
	
	async def dequeue_batch(self, timeout: float = 1.0) -> List[OllamaRequest]:
		"""Dequeue a batch of requests for processing"""
		async with self._condition:
			# Wait for requests or timeout
			try:
				await asyncio.wait_for(
					self._condition.wait_for(lambda: len(self.queue) > 0 or self.status != QueueStatus.ACTIVE),
					timeout=timeout
				)
			except asyncio.TimeoutError:
				return []
			
			if self.status != QueueStatus.ACTIVE or not self.queue:
				return []
			
			# Dequeue batch of requests
			batch = []
			batch_size = min(self.batch_size, len(self.queue))
			
			for _ in range(batch_size):
				if self.queue:
					request = heapq.heappop(self.queue)
					batch.append(request)
			
			self.metrics.current_queue_size = len(self.queue)
			return batch
	
	async def pause(self):
		"""Pause queue processing"""
		async with self._lock:
			self.status = QueueStatus.PAUSED
			self.logger.info("Queue paused")
	
	async def resume(self):
		"""Resume queue processing"""
		async with self._lock:
			self.status = QueueStatus.ACTIVE
			self._condition.notify_all()
			self.logger.info("Queue resumed")
	
	async def drain(self):
		"""Drain remaining requests"""
		async with self._lock:
			self.status = QueueStatus.DRAINING
			self._condition.notify_all()
			self.logger.info("Queue draining")
	
	def get_metrics(self) -> QueueMetrics:
		"""Get current queue metrics"""
		self.metrics.last_updated = datetime.now()
		return self.metrics

class OllamaConnectionPool:
	"""Advanced connection pool with per-endpoint management"""
	
	def __init__(self, max_connections_per_endpoint: int = 20):
		self.max_connections_per_endpoint = max_connections_per_endpoint
		self.pools: Dict[str, httpx.AsyncClient] = {}
		self.connection_counts: Dict[str, int] = defaultdict(int)
		self._locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
	
	async def get_client(self, endpoint_url: str) -> httpx.AsyncClient:
		"""Get or create HTTP client for endpoint"""
		async with self._locks[endpoint_url]:
			if endpoint_url not in self.pools:
				limits = httpx.Limits(
					max_connections=self.max_connections_per_endpoint,
					max_keepalive_connections=self.max_connections_per_endpoint // 2,
					keepalive_expiry=60.0
				)
				
				timeout = httpx.Timeout(connect=10.0, read=300.0, write=10.0)
				
				self.pools[endpoint_url] = httpx.AsyncClient(
					limits=limits,
					timeout=timeout,
					headers={'Content-Type': 'application/json'}
				)
			
			self.connection_counts[endpoint_url] += 1
			return self.pools[endpoint_url]
	
	async def release_client(self, endpoint_url: str):
		"""Release client connection"""
		async with self._locks[endpoint_url]:
			self.connection_counts[endpoint_url] = max(0, self.connection_counts[endpoint_url] - 1)
	
	async def close_all(self):
		"""Close all connection pools"""
		for client in self.pools.values():
			await client.aclose()
		self.pools.clear()
		self.connection_counts.clear()

class AdvancedOllamaIntegration:
	"""Production-ready Ollama integration with advanced features"""
	
	def __init__(self, 
	             endpoints: List[str] = None, 
	             load_balancer_config: LoadBalancerConfig = None,
	             queue_config: Dict[str, Any] = None):
		
		# Configuration
		self.endpoints = endpoints or ["http://localhost:11434"]
		self.load_balancer_config = load_balancer_config or LoadBalancerConfig()
		self.queue_config = queue_config or {"max_size": 10000, "batch_size": 32}
		
		# Core components
		self.load_balancer = LoadBalancer(self.load_balancer_config)
		self.embedding_queue = RequestQueue(**self.queue_config)
		self.generation_queue = RequestQueue(**self.queue_config)
		self.connection_pool = OllamaConnectionPool()
		
		# Processing state
		self.is_running = False
		self.worker_tasks: List[asyncio.Task] = []
		
		# Monitoring
		self.logger = logging.getLogger(__name__)
		self.metrics = {
			'total_requests': 0,
			'successful_requests': 0,
			'failed_requests': 0,
			'average_latency_ms': 0.0,
			'queue_wait_times': deque(maxlen=1000),
			'processing_times': deque(maxlen=1000)
		}
		
		# Initialize load balancer with endpoints
		for endpoint in self.endpoints:
			self.load_balancer.add_endpoint(endpoint)
	
	async def start(self):
		"""Start the Ollama integration service"""
		if self.is_running:
			return
		
		self.is_running = True
		self.logger.info("Starting Advanced Ollama Integration")
		
		# Start worker tasks
		self.worker_tasks = [
			asyncio.create_task(self._embedding_worker()),
			asyncio.create_task(self._generation_worker()),
			asyncio.create_task(self._health_monitor()),
			asyncio.create_task(self._metrics_collector())
		]
		
		self.logger.info("Ollama integration started successfully")
	
	async def stop(self):
		"""Stop the Ollama integration service"""
		if not self.is_running:
			return
		
		self.is_running = False
		self.logger.info("Stopping Advanced Ollama Integration")
		
		# Drain queues
		await self.embedding_queue.drain()
		await self.generation_queue.drain()
		
		# Cancel worker tasks
		for task in self.worker_tasks:
			task.cancel()
		
		# Wait for tasks to complete
		await asyncio.gather(*self.worker_tasks, return_exceptions=True)
		
		# Close connection pools
		await self.connection_pool.close_all()
		
		self.logger.info("Ollama integration stopped")
	
	async def _embedding_worker(self):
		"""Worker for processing embedding requests"""
		while self.is_running:
			try:
				batch = await self.embedding_queue.dequeue_batch(timeout=1.0)
				if not batch:
					continue
				
				# Group by capability and tenant for optimization
				grouped_requests = self._group_requests_by_context(batch)
				
				for group_key, requests in grouped_requests.items():
					await self._process_embedding_batch(requests)
			
			except Exception as e:
				self.logger.error(f"Embedding worker error: {str(e)}")
				await asyncio.sleep(1.0)
	
	async def _generation_worker(self):
		"""Worker for processing generation requests"""
		while self.is_running:
			try:
				batch = await self.generation_queue.dequeue_batch(timeout=1.0)
				if not batch:
					continue
				
				# Process generation requests individually for now
				# TODO: Implement batch generation if Ollama supports it
				for request in batch:
					await self._process_generation_request(request)
			
			except Exception as e:
				self.logger.error(f"Generation worker error: {str(e)}")
				await asyncio.sleep(1.0)
	
	async def _health_monitor(self):
		"""Monitor health of all endpoints"""
		while self.is_running:
			try:
				# Use any available client for health checks
				endpoint = self.load_balancer.select_endpoint()
				if endpoint:
					client = await self.connection_pool.get_client(endpoint.url)
					await self.load_balancer.health_check_all(client)
					await self.connection_pool.release_client(endpoint.url)
				
				await asyncio.sleep(self.load_balancer_config.health_check_interval)
			
			except Exception as e:
				self.logger.error(f"Health monitor error: {str(e)}")
				await asyncio.sleep(30.0)
	
	async def _metrics_collector(self):
		"""Collect and update performance metrics"""
		while self.is_running:
			try:
				# Update metrics
				if self.metrics['processing_times']:
					self.metrics['average_latency_ms'] = sum(self.metrics['processing_times']) / len(self.metrics['processing_times'])
				
				# Log metrics periodically
				self.logger.info(f"Metrics: {self._get_metrics_summary()}")
				
				await asyncio.sleep(60.0)  # Update every minute
			
			except Exception as e:
				self.logger.error(f"Metrics collector error: {str(e)}")
				await asyncio.sleep(60.0)
	
	def _group_requests_by_context(self, requests: List[OllamaRequest]) -> Dict[str, List[OllamaRequest]]:
		"""Group requests by tenant and capability for batch optimization"""
		groups = defaultdict(list)
		for request in requests:
			key = f"{request.tenant_id}:{request.capability_id}"
			groups[key].append(request)
		return dict(groups)
	
	async def _process_embedding_batch(self, requests: List[EmbeddingQueueRequest]):
		"""Process a batch of embedding requests"""
		start_time = time.time()
		
		try:
			# Select endpoint
			endpoint = self.load_balancer.select_endpoint()
			if not endpoint:
				raise Exception("No healthy endpoints available")
			
			# Get client
			client = await self.connection_pool.get_client(endpoint.url)
			endpoint.active_connections += 1
			
			try:
				# Combine texts from all requests in batch
				all_texts = []
				request_text_ranges = []
				
				for request in requests:
					start_idx = len(all_texts)
					all_texts.extend(request.texts)
					end_idx = len(all_texts)
					request_text_ranges.append((request, start_idx, end_idx))
				
				# Make batch embedding request
				response = await client.post(
					f"{endpoint.url}/api/embeddings",
					json={
						"model": requests[0].model,  # Assume same model for batch
						"prompt": all_texts,
						"options": {
							"num_predict": -1,
							"temperature": 0.0
						}
					}
				)
				
				if response.status_code != 200:
					raise Exception(f"Embedding request failed: {response.status_code}")
				
				result = response.json()
				embeddings = result.get("embeddings", [result.get("embedding", [])])
				
				# Distribute results back to individual requests
				for request, start_idx, end_idx in request_text_ranges:
					request_embeddings = embeddings[start_idx:end_idx]
					
					processing_time_ms = (time.time() - start_time) * 1000
					
					# Call callback if provided
					if request.callback:
						await request.callback({
							"request_id": request.request_id,
							"embeddings": request_embeddings,
							"model": request.model,
							"processing_time_ms": processing_time_ms,
							"success": True
						})
					
					# Update metrics
					self.metrics['successful_requests'] += 1
					self.metrics['processing_times'].append(processing_time_ms)
				
				# Record endpoint metrics
				endpoint.record_request(processing_time_ms, True)
				
			finally:
				endpoint.active_connections -= 1
				await self.connection_pool.release_client(endpoint.url)
		
		except Exception as e:
			processing_time_ms = (time.time() - start_time) * 1000
			
			# Handle failures for all requests in batch
			for request in requests:
				if request.callback:
					await request.callback({
						"request_id": request.request_id,
						"error": str(e),
						"processing_time_ms": processing_time_ms,
						"success": False
					})
				
				self.metrics['failed_requests'] += 1
			
			self.logger.error(f"Batch embedding processing failed: {str(e)}")
	
	async def _process_generation_request(self, request: GenerationQueueRequest):
		"""Process individual generation request"""
		start_time = time.time()
		
		try:
			# Select endpoint
			endpoint = self.load_balancer.select_endpoint()
			if not endpoint:
				raise Exception("No healthy endpoints available")
			
			# Get client
			client = await self.connection_pool.get_client(endpoint.url)
			endpoint.active_connections += 1
			
			try:
				# Make generation request
				response = await client.post(
					f"{endpoint.url}/api/generate",
					json={
						"model": request.model,
						"prompt": request.prompt,
						"stream": request.stream,
						"options": {
							"num_predict": request.max_tokens,
							"temperature": request.temperature,
							"top_p": request.top_p,
							"stop": request.stop_sequences
						}
					}
				)
				
				if response.status_code != 200:
					raise Exception(f"Generation request failed: {response.status_code}")
				
				result = response.json()
				processing_time_ms = (time.time() - start_time) * 1000
				
				# Call callback if provided
				if request.callback:
					await request.callback({
						"request_id": request.request_id,
						"text": result.get("response", ""),
						"model": request.model,
						"tokens_used": result.get("eval_count", 0),
						"processing_time_ms": processing_time_ms,
						"success": True
					})
				
				# Update metrics
				self.metrics['successful_requests'] += 1
				self.metrics['processing_times'].append(processing_time_ms)
				
				# Record endpoint metrics
				endpoint.record_request(processing_time_ms, True)
				
			finally:
				endpoint.active_connections -= 1
				await self.connection_pool.release_client(endpoint.url)
		
		except Exception as e:
			processing_time_ms = (time.time() - start_time) * 1000
			
			# Handle failure
			if request.callback:
				await request.callback({
					"request_id": request.request_id,
					"error": str(e),
					"processing_time_ms": processing_time_ms,
					"success": False
				})
			
			self.metrics['failed_requests'] += 1
			self.logger.error(f"Generation processing failed: {str(e)}")
	
	async def generate_embeddings_async(self, 
	                                   texts: List[str], 
	                                   model: str = "bge-m3",
	                                   tenant_id: str = "",
	                                   capability_id: str = "",
	                                   priority: RequestPriority = RequestPriority.NORMAL,
	                                   callback: Optional[Callable] = None) -> str:
		"""Queue embedding generation request"""
		request = EmbeddingQueueRequest(
			texts=texts,
			model=model,
			tenant_id=tenant_id,
			capability_id=capability_id,
			priority=priority,
			callback=callback
		)
		
		success = await self.embedding_queue.enqueue(request)
		if not success:
			raise Exception("Failed to enqueue embedding request - queue full")
		
		return request.request_id
	
	async def generate_text_async(self,
	                             prompt: str,
	                             model: str = "qwen3",
	                             tenant_id: str = "",
	                             capability_id: str = "",
	                             priority: RequestPriority = RequestPriority.NORMAL,
	                             callback: Optional[Callable] = None,
	                             **kwargs) -> str:
		"""Queue text generation request"""
		request = GenerationQueueRequest(
			prompt=prompt,
			model=model,
			tenant_id=tenant_id,
			capability_id=capability_id,
			priority=priority,
			callback=callback,
			**kwargs
		)
		
		success = await self.generation_queue.enqueue(request)
		if not success:
			raise Exception("Failed to enqueue generation request - queue full")
		
		return request.request_id
	
	def _get_metrics_summary(self) -> Dict[str, Any]:
		"""Get comprehensive metrics summary"""
		embedding_metrics = self.embedding_queue.get_metrics()
		generation_metrics = self.generation_queue.get_metrics()
		
		return {
			"total_requests": self.metrics['total_requests'],
			"successful_requests": self.metrics['successful_requests'],
			"failed_requests": self.metrics['failed_requests'],
			"success_rate": self.metrics['successful_requests'] / max(1, self.metrics['total_requests']),
			"average_latency_ms": self.metrics['average_latency_ms'],
			"embedding_queue_size": embedding_metrics.current_queue_size,
			"generation_queue_size": generation_metrics.current_queue_size,
			"healthy_endpoints": len([ep for ep in self.load_balancer.endpoints if ep.is_available()]),
			"total_endpoints": len(self.load_balancer.endpoints)
		}
	
	async def get_system_status(self) -> Dict[str, Any]:
		"""Get comprehensive system status"""
		return {
			"service_status": "running" if self.is_running else "stopped",
			"metrics": self._get_metrics_summary(),
			"endpoints": [
				{
					"url": ep.url,
					"healthy": ep.is_healthy,
					"active_connections": ep.active_connections,
					"total_requests": ep.total_requests,
					"failure_rate": ep.failure_rate,
					"avg_response_time_ms": ep.average_response_time
				}
				for ep in self.load_balancer.endpoints
			],
			"queues": {
				"embedding": {
					"status": self.embedding_queue.status.value,
					"size": self.embedding_queue.metrics.current_queue_size,
					"total_processed": self.embedding_queue.metrics.completed_requests
				},
				"generation": {
					"status": self.generation_queue.status.value,
					"size": self.generation_queue.metrics.current_queue_size,
					"total_processed": self.generation_queue.metrics.completed_requests
				}
			},
			"timestamp": datetime.now().isoformat()
		}

# Async context manager for easy lifecycle management
@asynccontextmanager
async def advanced_ollama_integration(endpoints: List[str] = None, **kwargs):
	"""Context manager for Advanced Ollama Integration"""
	integration = AdvancedOllamaIntegration(endpoints=endpoints, **kwargs)
	await integration.start()
	try:
		yield integration
	finally:
		await integration.stop()

# Factory function for APG integration
async def create_ollama_integration(endpoints: List[str] = None, **kwargs) -> AdvancedOllamaIntegration:
	"""Factory function to create and start Ollama integration"""
	if endpoints is None:
		endpoints = ["http://localhost:11434"]
	
	integration = AdvancedOllamaIntegration(endpoints=endpoints, **kwargs)
	await integration.start()
	return integration