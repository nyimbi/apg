"""
Audio Processing Performance Optimization & Scaling

Advanced performance optimization, caching, load balancing, and scaling
mechanisms for production deployment of the audio processing capability.

Copyright © 2025 Datacraft
Author: APG Development Team
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import queue
import logging
import weakref
from contextlib import asynccontextmanager

import redis
import psutil
import numpy as np
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry

from .models import ProcessingStatus, APTranscriptionJob, APVoiceSynthesisJob
from uuid_extensions import uuid7str


# Performance Metrics Collection
METRICS_REGISTRY = CollectorRegistry()

PROCESSING_REQUESTS_TOTAL = Counter(
	'audio_processing_requests_total', 
	'Total audio processing requests', 
	['operation_type', 'status', 'tenant_id'],
	registry=METRICS_REGISTRY
)

PROCESSING_DURATION_SECONDS = Histogram(
	'audio_processing_duration_seconds',
	'Time spent processing audio requests',
	['operation_type', 'tenant_id'],
	registry=METRICS_REGISTRY
)

ACTIVE_PROCESSING_JOBS = Gauge(
	'audio_processing_active_jobs',
	'Number of currently active processing jobs',
	['operation_type', 'tenant_id'],
	registry=METRICS_REGISTRY
)

CACHE_HIT_RATE = Gauge(
	'audio_processing_cache_hit_rate',
	'Cache hit rate for audio processing operations',
	['cache_type', 'tenant_id'],
	registry=METRICS_REGISTRY
)


@dataclass
class PerformanceMetrics:
	"""Performance metrics data structure"""
	operation_type: str
	tenant_id: str
	start_time: float
	end_time: Optional[float] = None
	status: str = "in_progress"
	cache_hit: bool = False
	resource_usage: Optional[Dict[str, float]] = None


class ResourceMonitor:
	"""System resource monitoring and alerting"""
	
	def __init__(self):
		self.cpu_threshold = 80.0  # CPU usage percentage
		self.memory_threshold = 85.0  # Memory usage percentage
		self.disk_threshold = 90.0  # Disk usage percentage
		self.monitoring_interval = 10.0  # seconds
		self._monitoring_task: Optional[asyncio.Task] = None
		self._alerts: deque = deque(maxlen=100)
		self._logger = logging.getLogger(__name__)
	
	async def start_monitoring(self) -> None:
		"""Start resource monitoring"""
		if self._monitoring_task is not None:
			return
		
		self._monitoring_task = asyncio.create_task(self._monitor_resources())
		self._logger.info("Resource monitoring started")
	
	async def stop_monitoring(self) -> None:
		"""Stop resource monitoring"""
		if self._monitoring_task is not None:
			self._monitoring_task.cancel()
			try:
				await self._monitoring_task
			except asyncio.CancelledError:
				pass
			self._monitoring_task = None
		
		self._logger.info("Resource monitoring stopped")
	
	async def _monitor_resources(self) -> None:
		"""Monitor system resources continuously"""
		while True:
			try:
				# CPU usage
				cpu_percent = psutil.cpu_percent(interval=1)
				
				# Memory usage
				memory = psutil.virtual_memory()
				memory_percent = memory.percent
				
				# Disk usage
				disk = psutil.disk_usage('/')
				disk_percent = (disk.used / disk.total) * 100
				
				# Check thresholds and generate alerts
				if cpu_percent > self.cpu_threshold:
					await self._generate_alert("cpu", cpu_percent, self.cpu_threshold)
				
				if memory_percent > self.memory_threshold:
					await self._generate_alert("memory", memory_percent, self.memory_threshold)
				
				if disk_percent > self.disk_threshold:
					await self._generate_alert("disk", disk_percent, self.disk_threshold)
				
				# Update metrics
				self._update_resource_metrics(cpu_percent, memory_percent, disk_percent)
				
				await asyncio.sleep(self.monitoring_interval)
			
			except Exception as e:
				self._logger.error(f"Error in resource monitoring: {e}")
				await asyncio.sleep(self.monitoring_interval)
	
	async def _generate_alert(self, resource_type: str, current_value: float, threshold: float) -> None:
		"""Generate resource usage alert"""
		alert = {
			'timestamp': datetime.utcnow(),
			'type': 'resource_alert',
			'resource': resource_type,
			'current_value': current_value,
			'threshold': threshold,
			'severity': 'high' if current_value > threshold * 1.1 else 'medium'
		}
		
		self._alerts.append(alert)
		self._logger.warning(f"Resource alert: {resource_type} usage at {current_value}% (threshold: {threshold}%)")
	
	def _update_resource_metrics(self, cpu: float, memory: float, disk: float) -> None:
		"""Update Prometheus metrics"""
		# These would be actual Prometheus gauges in production
		pass
	
	def get_recent_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
		"""Get recent resource alerts"""
		return list(self._alerts)[-limit:]


class CacheManager:
	"""Multi-level caching system for audio processing"""
	
	def __init__(self, redis_url: str = "redis://localhost:6379/0"):
		self.redis_client = redis.from_url(redis_url, decode_responses=True)
		self.local_cache: Dict[str, Any] = {}
		self.cache_stats = defaultdict(lambda: {'hits': 0, 'misses': 0})
		self._cache_lock = threading.RLock()
		self._logger = logging.getLogger(__name__)
		
		# Cache configuration
		self.local_cache_max_size = 1000
		self.local_cache_ttl = 300  # 5 minutes
		self.redis_cache_ttl = 3600  # 1 hour
	
	async def get(self, key: str, tenant_id: str, cache_type: str = "general") -> Optional[Any]:
		"""Get value from cache with fallback layers"""
		full_key = f"{tenant_id}:{key}"
		
		# Try local cache first
		with self._cache_lock:
			if full_key in self.local_cache:
				entry = self.local_cache[full_key]
				if entry['expires_at'] > time.time():
					self.cache_stats[cache_type]['hits'] += 1
					self._update_cache_hit_rate(cache_type, tenant_id, True)
					return entry['value']
				else:
					del self.local_cache[full_key]
		
		# Try Redis cache
		try:
			value = await asyncio.get_event_loop().run_in_executor(
				None, self.redis_client.get, full_key
			)
			if value is not None:
				# Store in local cache for faster access
				await self._store_local(full_key, value)
				self.cache_stats[cache_type]['hits'] += 1
				self._update_cache_hit_rate(cache_type, tenant_id, True)
				return value
		except Exception as e:
			self._logger.warning(f"Redis cache error: {e}")
		
		# Cache miss
		self.cache_stats[cache_type]['misses'] += 1
		self._update_cache_hit_rate(cache_type, tenant_id, False)
		return None
	
	async def set(self, key: str, value: Any, tenant_id: str, ttl: Optional[int] = None) -> None:
		"""Set value in cache layers"""
		full_key = f"{tenant_id}:{key}"
		ttl = ttl or self.redis_cache_ttl
		
		# Store in Redis
		try:
			await asyncio.get_event_loop().run_in_executor(
				None, self.redis_client.setex, full_key, ttl, value
			)
		except Exception as e:
			self._logger.warning(f"Redis cache set error: {e}")
		
		# Store in local cache
		await self._store_local(full_key, value, self.local_cache_ttl)
	
	async def _store_local(self, key: str, value: Any, ttl: int = None) -> None:
		"""Store value in local cache"""
		ttl = ttl or self.local_cache_ttl
		
		with self._cache_lock:
			# Evict oldest entries if cache is full
			if len(self.local_cache) >= self.local_cache_max_size:
				oldest_key = min(self.local_cache.keys(), 
								key=lambda k: self.local_cache[k]['created_at'])
				del self.local_cache[oldest_key]
			
			self.local_cache[key] = {
				'value': value,
				'created_at': time.time(),
				'expires_at': time.time() + ttl
			}
	
	def _update_cache_hit_rate(self, cache_type: str, tenant_id: str, hit: bool) -> None:
		"""Update cache hit rate metrics"""
		stats = self.cache_stats[cache_type]
		total_requests = stats['hits'] + stats['misses']
		
		if total_requests > 0:
			hit_rate = stats['hits'] / total_requests
			CACHE_HIT_RATE.labels(cache_type=cache_type, tenant_id=tenant_id).set(hit_rate)
	
	async def invalidate(self, pattern: str, tenant_id: str) -> int:
		"""Invalidate cache entries matching pattern"""
		full_pattern = f"{tenant_id}:{pattern}"
		
		# Invalidate local cache
		invalidated_local = 0
		with self._cache_lock:
			keys_to_remove = [k for k in self.local_cache.keys() if pattern in k]
			for key in keys_to_remove:
				del self.local_cache[key]
				invalidated_local += 1
		
		# Invalidate Redis cache
		try:
			keys = await asyncio.get_event_loop().run_in_executor(
				None, self.redis_client.keys, full_pattern
			)
			if keys:
				await asyncio.get_event_loop().run_in_executor(
					None, self.redis_client.delete, *keys
				)
		except Exception as e:
			self._logger.warning(f"Redis cache invalidation error: {e}")
		
		return invalidated_local + len(keys) if 'keys' in locals() else invalidated_local
	
	def get_cache_stats(self) -> Dict[str, Any]:
		"""Get cache statistics"""
		return dict(self.cache_stats)


class LoadBalancer:
	"""Load balancing for audio processing operations"""
	
	def __init__(self):
		self.worker_pools: Dict[str, ThreadPoolExecutor] = {}
		self.process_pools: Dict[str, ProcessPoolExecutor] = {}
		self.active_jobs: Dict[str, int] = defaultdict(int)
		self.worker_health: Dict[str, bool] = {}
		self._load_lock = threading.RLock()
		self._logger = logging.getLogger(__name__)
		
		# Initialize worker pools
		self._initialize_worker_pools()
	
	def _initialize_worker_pools(self) -> None:
		"""Initialize worker pools for different operation types"""
		cpu_count = psutil.cpu_count()
		
		# Thread pools for I/O bound operations
		self.worker_pools['transcription'] = ThreadPoolExecutor(
			max_workers=min(cpu_count * 2, 16),
			thread_name_prefix='transcription'
		)
		self.worker_pools['synthesis'] = ThreadPoolExecutor(
			max_workers=min(cpu_count * 2, 12),
			thread_name_prefix='synthesis'
		)
		self.worker_pools['analysis'] = ThreadPoolExecutor(
			max_workers=min(cpu_count, 8),
			thread_name_prefix='analysis'
		)
		
		# Process pools for CPU-intensive operations
		self.process_pools['enhancement'] = ProcessPoolExecutor(
			max_workers=min(cpu_count, 4)
		)
		self.process_pools['voice_cloning'] = ProcessPoolExecutor(
			max_workers=min(cpu_count // 2, 2)
		)
		
		# Initialize health status
		for pool_name in list(self.worker_pools.keys()) + list(self.process_pools.keys()):
			self.worker_health[pool_name] = True
		
		self._logger.info(f"Initialized worker pools: {list(self.worker_pools.keys())}, "
						 f"process pools: {list(self.process_pools.keys())}")
	
	async def submit_job(self, operation_type: str, func: Callable, *args, **kwargs) -> Any:
		"""Submit job to appropriate worker pool with load balancing"""
		with self._load_lock:
			# Check if worker is healthy
			if not self.worker_health.get(operation_type, False):
				raise RuntimeError(f"Worker pool '{operation_type}' is unhealthy")
			
			# Increment active job count
			self.active_jobs[operation_type] += 1
		
		try:
			# Choose appropriate executor
			if operation_type in self.worker_pools:
				executor = self.worker_pools[operation_type]
			elif operation_type in self.process_pools:
				executor = self.process_pools[operation_type]
			else:
				raise ValueError(f"Unknown operation type: {operation_type}")
			
			# Submit job
			loop = asyncio.get_event_loop()
			result = await loop.run_in_executor(executor, func, *args, **kwargs)
			
			return result
		
		except Exception as e:
			self._logger.error(f"Job execution failed for {operation_type}: {e}")
			raise
		
		finally:
			with self._load_lock:
				self.active_jobs[operation_type] -= 1
	
	def get_load_stats(self) -> Dict[str, Any]:
		"""Get current load balancing statistics"""
		with self._load_lock:
			stats = {
				'active_jobs': dict(self.active_jobs),
				'worker_health': dict(self.worker_health),
				'pool_sizes': {
					**{name: pool._max_workers for name, pool in self.worker_pools.items()},
					**{name: pool._max_workers for name, pool in self.process_pools.items()}
				}
			}
		
		return stats
	
	async def health_check(self) -> Dict[str, bool]:
		"""Perform health check on all worker pools"""
		health_status = {}
		
		for pool_name, pool in self.worker_pools.items():
			try:
				# Submit a simple test job
				future = pool.submit(lambda: True)
				result = await asyncio.wait_for(
					asyncio.wrap_future(future), timeout=5.0
				)
				health_status[pool_name] = result
			except Exception as e:
				self._logger.warning(f"Health check failed for {pool_name}: {e}")
				health_status[pool_name] = False
		
		for pool_name, pool in self.process_pools.items():
			try:
				# Submit a simple test job
				future = pool.submit(lambda: True)
				result = await asyncio.wait_for(
					asyncio.wrap_future(future), timeout=10.0
				)
				health_status[pool_name] = result
			except Exception as e:
				self._logger.warning(f"Health check failed for {pool_name}: {e}")
				health_status[pool_name] = False
		
		# Update health status
		with self._load_lock:
			self.worker_health.update(health_status)
		
		return health_status
	
	async def shutdown(self) -> None:
		"""Gracefully shutdown all worker pools"""
		self._logger.info("Shutting down worker pools...")
		
		# Shutdown thread pools
		for name, pool in self.worker_pools.items():
			pool.shutdown(wait=True)
			self._logger.info(f"Thread pool '{name}' shutdown complete")
		
		# Shutdown process pools
		for name, pool in self.process_pools.items():
			pool.shutdown(wait=True)
			self._logger.info(f"Process pool '{name}' shutdown complete")


class PerformanceOptimizer:
	"""Main performance optimization coordinator"""
	
	def __init__(self, redis_url: str = "redis://localhost:6379/0"):
		self.resource_monitor = ResourceMonitor()
		self.cache_manager = CacheManager(redis_url)
		self.load_balancer = LoadBalancer()
		self.metrics: List[PerformanceMetrics] = []
		self._metrics_lock = threading.RLock()
		self._logger = logging.getLogger(__name__)
		
		# Performance thresholds
		self.max_processing_time = {
			'transcription': 120.0,  # 2 minutes
			'synthesis': 60.0,       # 1 minute
			'analysis': 180.0,       # 3 minutes
			'enhancement': 300.0     # 5 minutes
		}
	
	async def initialize(self) -> None:
		"""Initialize performance optimization components"""
		await self.resource_monitor.start_monitoring()
		self._logger.info("Performance optimizer initialized")
	
	async def shutdown(self) -> None:
		"""Shutdown performance optimization components"""
		await self.resource_monitor.stop_monitoring()
		await self.load_balancer.shutdown()
		self._logger.info("Performance optimizer shutdown complete")
	
	@asynccontextmanager
	async def track_operation(self, operation_type: str, tenant_id: str):
		"""Context manager for tracking operation performance"""
		metrics = PerformanceMetrics(
			operation_type=operation_type,
			tenant_id=tenant_id,
			start_time=time.time()
		)
		
		# Update active jobs metric
		ACTIVE_PROCESSING_JOBS.labels(
			operation_type=operation_type, 
			tenant_id=tenant_id
		).inc()
		
		try:
			yield metrics
			metrics.status = "completed"
		except Exception as e:
			metrics.status = "failed"
			raise
		finally:
			metrics.end_time = time.time()
			
			# Update metrics
			ACTIVE_PROCESSING_JOBS.labels(
				operation_type=operation_type, 
				tenant_id=tenant_id
			).dec()
			
			PROCESSING_REQUESTS_TOTAL.labels(
				operation_type=operation_type,
				status=metrics.status,
				tenant_id=tenant_id
			).inc()
			
			if metrics.end_time:
				duration = metrics.end_time - metrics.start_time
				PROCESSING_DURATION_SECONDS.labels(
					operation_type=operation_type,
					tenant_id=tenant_id
				).observe(duration)
			
			# Store metrics
			with self._metrics_lock:
				self.metrics.append(metrics)
				# Keep only recent metrics
				if len(self.metrics) > 10000:
					self.metrics = self.metrics[-5000:]
	
	async def optimize_processing(self, operation_type: str, func: Callable, *args, **kwargs) -> Any:
		"""Optimize processing with caching, load balancing, and monitoring"""
		tenant_id = kwargs.get('tenant_id', 'default')
		
		# Generate cache key
		cache_key = self._generate_cache_key(operation_type, args, kwargs)
		
		# Try cache first
		cached_result = await self.cache_manager.get(cache_key, tenant_id, operation_type)
		if cached_result is not None:
			return cached_result
		
		# Process with load balancing and monitoring
		async with self.track_operation(operation_type, tenant_id) as metrics:
			result = await self.load_balancer.submit_job(operation_type, func, *args, **kwargs)
			
			# Cache result if processing was successful
			if metrics.status == "completed":
				await self.cache_manager.set(cache_key, result, tenant_id)
			
			return result
	
	def _generate_cache_key(self, operation_type: str, args: tuple, kwargs: dict) -> str:
		"""Generate cache key for operation"""
		# Create a deterministic cache key
		key_parts = [operation_type]
		
		# Add relevant args/kwargs for cache key
		if 'audio_source' in kwargs:
			audio_source = kwargs['audio_source']
			if isinstance(audio_source, dict) and 'file_path' in audio_source:
				key_parts.append(audio_source['file_path'])
		
		if 'text_content' in kwargs:
			# Use hash of text content for cache key
			import hashlib
			text_hash = hashlib.md5(kwargs['text_content'].encode()).hexdigest()[:8]
			key_parts.append(f"text_{text_hash}")
		
		# Add other relevant parameters
		for param in ['language_code', 'provider', 'voice_id', 'emotion']:
			if param in kwargs:
				key_parts.append(f"{param}_{kwargs[param]}")
		
		return ":".join(key_parts)
	
	async def get_performance_report(self, hours: int = 24) -> Dict[str, Any]:
		"""Generate comprehensive performance report"""
		cutoff_time = time.time() - (hours * 3600)
		
		with self._metrics_lock:
			recent_metrics = [m for m in self.metrics if m.start_time > cutoff_time]
		
		# Aggregate statistics
		stats_by_operation = defaultdict(lambda: {
			'total_requests': 0,
			'successful_requests': 0,
			'failed_requests': 0,
			'avg_duration': 0.0,
			'max_duration': 0.0,
			'min_duration': float('inf')
		})
		
		for metric in recent_metrics:
			op_stats = stats_by_operation[metric.operation_type]
			op_stats['total_requests'] += 1
			
			if metric.status == 'completed':
				op_stats['successful_requests'] += 1
			else:
				op_stats['failed_requests'] += 1
			
			if metric.end_time:
				duration = metric.end_time - metric.start_time
				op_stats['max_duration'] = max(op_stats['max_duration'], duration)
				op_stats['min_duration'] = min(op_stats['min_duration'], duration)
		
		# Calculate averages
		for op_type, stats in stats_by_operation.items():
			if stats['successful_requests'] > 0:
				total_duration = sum(
					m.end_time - m.start_time 
					for m in recent_metrics 
					if m.operation_type == op_type and m.end_time and m.status == 'completed'
				)
				stats['avg_duration'] = total_duration / stats['successful_requests']
			
			if stats['min_duration'] == float('inf'):
				stats['min_duration'] = 0.0
		
		report = {
			'report_period_hours': hours,
			'total_metrics_collected': len(recent_metrics),
			'operations_stats': dict(stats_by_operation),
			'resource_alerts': self.resource_monitor.get_recent_alerts(),
			'cache_stats': self.cache_manager.get_cache_stats(),
			'load_balancer_stats': self.load_balancer.get_load_stats(),
			'generated_at': datetime.utcnow().isoformat()
		}
		
		return report


# Factory function for creating performance optimizer
def create_performance_optimizer(redis_url: str = "redis://localhost:6379/0") -> PerformanceOptimizer:
	"""Create and configure performance optimizer"""
	return PerformanceOptimizer(redis_url)


# Performance decorators for easy integration
def performance_optimized(operation_type: str):
	"""Decorator for automatic performance optimization"""
	def decorator(func):
		async def wrapper(*args, **kwargs):
			# Get or create performance optimizer instance
			optimizer = getattr(wrapper, '_optimizer', None)
			if optimizer is None:
				optimizer = create_performance_optimizer()
				wrapper._optimizer = optimizer
				await optimizer.initialize()
			
			return await optimizer.optimize_processing(operation_type, func, *args, **kwargs)
		
		return wrapper
	return decorator


# Auto-scaling configuration
class AutoScaler:
	"""Auto-scaling based on load and resource usage"""
	
	def __init__(self, performance_optimizer: PerformanceOptimizer):
		self.optimizer = performance_optimizer
		self.scaling_rules = {
			'cpu_threshold': 70.0,
			'memory_threshold': 80.0,
			'queue_length_threshold': 50,
			'response_time_threshold': 30.0
		}
		self._logger = logging.getLogger(__name__)
	
	async def evaluate_scaling_needs(self) -> Dict[str, Any]:
		"""Evaluate if scaling is needed based on current metrics"""
		load_stats = self.optimizer.load_balancer.get_load_stats()
		performance_report = await self.optimizer.get_performance_report(hours=1)
		
		scaling_recommendations = {
			'scale_up': [],
			'scale_down': [],
			'current_status': 'stable'
		}
		
		# Analyze load and performance
		for operation_type, active_jobs in load_stats['active_jobs'].items():
			op_stats = performance_report['operations_stats'].get(operation_type, {})
			
			# Check if scaling up is needed
			if (active_jobs > self.scaling_rules['queue_length_threshold'] or
				op_stats.get('avg_duration', 0) > self.scaling_rules['response_time_threshold']):
				scaling_recommendations['scale_up'].append({
					'operation_type': operation_type,
					'reason': 'high_load',
					'current_jobs': active_jobs,
					'avg_duration': op_stats.get('avg_duration', 0)
				})
		
		if scaling_recommendations['scale_up']:
			scaling_recommendations['current_status'] = 'scale_up_needed'
		
		return scaling_recommendations