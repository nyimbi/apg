"""
Computer Vision & Visual Intelligence - Performance Optimization & Monitoring

Comprehensive performance optimization including caching strategies, load balancing,
auto-scaling, resource monitoring, and observability with enterprise-grade metrics
collection, alerting, and performance tuning for computer vision workloads.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import asyncio
import json
import time
from collections import deque, defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from threading import Lock
import hashlib
import gc

import redis
import psutil
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
import structlog

from .models import ProcessingStatus, ProcessingType


# Performance Metrics Configuration
@dataclass
class PerformanceConfig:
	"""Performance configuration settings"""
	cache_ttl_seconds: int = 3600
	max_cache_size_mb: int = 1024
	max_concurrent_jobs: int = 50
	worker_pool_size: int = 10
	batch_size_limit: int = 100
	auto_scaling_enabled: bool = True
	monitoring_interval_seconds: int = 30
	performance_log_level: str = "INFO"
	
	# Resource limits
	max_memory_usage_percent: float = 80.0
	max_cpu_usage_percent: float = 75.0
	max_disk_usage_percent: float = 85.0
	
	# Auto-scaling thresholds
	scale_up_cpu_threshold: float = 70.0
	scale_down_cpu_threshold: float = 30.0
	scale_up_memory_threshold: float = 80.0
	scale_down_memory_threshold: float = 40.0
	scale_up_queue_threshold: int = 10
	scale_down_queue_threshold: int = 2


class CVPerformanceMetrics:
	"""
	Performance Metrics Collection and Analysis
	
	Comprehensive metrics collection for computer vision processing including
	processing times, throughput, resource utilization, and business metrics
	with Prometheus integration and real-time monitoring capabilities.
	"""
	
	def __init__(self, config: PerformanceConfig):
		self.config = config
		self.logger = structlog.get_logger("cv_performance")
		
		# Prometheus metrics registry
		self.registry = CollectorRegistry()
		
		# Processing metrics
		self.processing_duration = Histogram(
			'cv_processing_duration_seconds',
			'Time spent processing computer vision jobs',
			['processing_type', 'status', 'tenant_id'],
			registry=self.registry,
			buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
		)
		
		self.jobs_total = Counter(
			'cv_jobs_total',
			'Total number of computer vision jobs',
			['processing_type', 'status', 'tenant_id'],
			registry=self.registry
		)
		
		self.api_requests = Counter(
			'cv_api_requests_total',
			'Total API requests',
			['endpoint', 'method', 'status_code'],
			registry=self.registry
		)
		
		self.api_duration = Histogram(
			'cv_api_request_duration_seconds',
			'API request duration',
			['endpoint', 'method'],
			registry=self.registry,
			buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
		)
		
		# Resource metrics
		self.cpu_usage = Gauge(
			'cv_cpu_usage_percent',
			'CPU usage percentage',
			registry=self.registry
		)
		
		self.memory_usage = Gauge(
			'cv_memory_usage_percent',
			'Memory usage percentage',
			registry=self.registry
		)
		
		self.active_jobs = Gauge(
			'cv_active_jobs',
			'Number of active processing jobs',
			['tenant_id'],
			registry=self.registry
		)
		
		self.queue_length = Gauge(
			'cv_queue_length',
			'Processing queue length',
			registry=self.registry
		)
		
		# Cache metrics
		self.cache_hits = Counter(
			'cv_cache_hits_total',
			'Cache hit count',
			['cache_type'],
			registry=self.registry
		)
		
		self.cache_misses = Counter(
			'cv_cache_misses_total',
			'Cache miss count',
			['cache_type'],
			registry=self.registry
		)
		
		# Model performance metrics
		self.model_inference_duration = Histogram(
			'cv_model_inference_duration_seconds',
			'Model inference time',
			['model_name', 'model_type'],
			registry=self.registry,
			buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
		)
		
		self.model_accuracy = Gauge(
			'cv_model_accuracy',
			'Model accuracy score',
			['model_name', 'model_type'],
			registry=self.registry
		)
		
		# Business metrics
		self.processing_volume = Counter(
			'cv_processing_volume_total',
			'Total data processed',
			['unit', 'tenant_id'],
			registry=self.registry
		)
		
		# Performance tracking
		self.recent_processing_times = deque(maxlen=1000)
		self.hourly_stats = defaultdict(lambda: defaultdict(int))
		self.daily_stats = defaultdict(lambda: defaultdict(int))
		
		# Resource monitoring
		self.resource_history = deque(maxlen=720)  # 6 hours at 30s intervals
		self.alert_conditions = []
		
		self.logger.info("Performance metrics initialized", 
		                config=self.config.__dict__)
	
	def record_job_processing(
		self,
		processing_type: ProcessingType,
		status: ProcessingStatus,
		duration_seconds: float,
		tenant_id: str,
		model_name: Optional[str] = None,
		accuracy_score: Optional[float] = None,
		data_size_bytes: Optional[int] = None
	) -> None:
		"""Record job processing metrics"""
		try:
			# Record processing duration
			self.processing_duration.labels(
				processing_type=processing_type.value,
				status=status.value,
				tenant_id=tenant_id
			).observe(duration_seconds)
			
			# Increment job counter
			self.jobs_total.labels(
				processing_type=processing_type.value,
				status=status.value,
				tenant_id=tenant_id
			).inc()
			
			# Record model performance if available
			if model_name and accuracy_score is not None:
				self.model_accuracy.labels(
					model_name=model_name,
					model_type=processing_type.value
				).set(accuracy_score)
			
			# Record processing volume
			if data_size_bytes:
				self.processing_volume.labels(
					unit='bytes',
					tenant_id=tenant_id
				).inc(data_size_bytes)
			
			# Track recent processing times for analysis
			self.recent_processing_times.append({
				'timestamp': datetime.utcnow(),
				'duration': duration_seconds,
				'type': processing_type.value,
				'status': status.value,
				'tenant_id': tenant_id
			})
			
			# Update hourly stats
			hour_key = datetime.utcnow().strftime('%Y-%m-%d-%H')
			self.hourly_stats[hour_key]['total_jobs'] += 1
			self.hourly_stats[hour_key]['total_duration'] += duration_seconds
			
			if status == ProcessingStatus.COMPLETED:
				self.hourly_stats[hour_key]['successful_jobs'] += 1
			elif status == ProcessingStatus.FAILED:
				self.hourly_stats[hour_key]['failed_jobs'] += 1
			
			self.logger.debug("Recorded job processing metrics",
			                 processing_type=processing_type.value,
			                 status=status.value,
			                 duration=duration_seconds,
			                 tenant_id=tenant_id)
			
		except Exception as e:
			self.logger.error("Failed to record job processing metrics",
			                 error=str(e))
	
	def record_api_request(
		self,
		endpoint: str,
		method: str,
		status_code: int,
		duration_seconds: float
	) -> None:
		"""Record API request metrics"""
		try:
			self.api_requests.labels(
				endpoint=endpoint,
				method=method,
				status_code=status_code
			).inc()
			
			self.api_duration.labels(
				endpoint=endpoint,
				method=method
			).observe(duration_seconds)
			
			self.logger.debug("Recorded API request metrics",
			                 endpoint=endpoint,
			                 method=method,
			                 status_code=status_code,
			                 duration=duration_seconds)
			
		except Exception as e:
			self.logger.error("Failed to record API request metrics",
			                 error=str(e))
	
	def record_model_inference(
		self,
		model_name: str,
		model_type: str,
		inference_time_seconds: float
	) -> None:
		"""Record model inference performance"""
		try:
			self.model_inference_duration.labels(
				model_name=model_name,
				model_type=model_type
			).observe(inference_time_seconds)
			
			self.logger.debug("Recorded model inference metrics",
			                 model_name=model_name,
			                 model_type=model_type,
			                 inference_time=inference_time_seconds)
			
		except Exception as e:
			self.logger.error("Failed to record model inference metrics",
			                 error=str(e))
	
	def update_resource_metrics(self) -> None:
		"""Update system resource metrics"""
		try:
			# CPU usage
			cpu_percent = psutil.cpu_percent(interval=1)
			self.cpu_usage.set(cpu_percent)
			
			# Memory usage
			memory = psutil.virtual_memory()
			memory_percent = memory.percent
			self.memory_usage.set(memory_percent)
			
			# Store resource history
			self.resource_history.append({
				'timestamp': datetime.utcnow(),
				'cpu_percent': cpu_percent,
				'memory_percent': memory_percent,
				'memory_available_gb': memory.available / (1024**3),
				'disk_usage_percent': psutil.disk_usage('/').percent
			})
			
			# Check alert conditions
			self._check_resource_alerts(cpu_percent, memory_percent)
			
			self.logger.debug("Updated resource metrics",
			                 cpu_percent=cpu_percent,
			                 memory_percent=memory_percent)
			
		except Exception as e:
			self.logger.error("Failed to update resource metrics",
			                 error=str(e))
	
	def update_active_jobs_count(self, count: int, tenant_id: str = "default") -> None:
		"""Update active jobs count"""
		self.active_jobs.labels(tenant_id=tenant_id).set(count)
	
	def update_queue_length(self, length: int) -> None:
		"""Update processing queue length"""
		self.queue_length.set(length)
	
	def record_cache_hit(self, cache_type: str) -> None:
		"""Record cache hit"""
		self.cache_hits.labels(cache_type=cache_type).inc()
	
	def record_cache_miss(self, cache_type: str) -> None:
		"""Record cache miss"""
		self.cache_misses.labels(cache_type=cache_type).inc()
	
	def _check_resource_alerts(self, cpu_percent: float, memory_percent: float) -> None:
		"""Check for resource alert conditions"""
		alerts = []
		
		if cpu_percent > self.config.max_cpu_usage_percent:
			alerts.append(f"High CPU usage: {cpu_percent:.1f}%")
		
		if memory_percent > self.config.max_memory_usage_percent:
			alerts.append(f"High memory usage: {memory_percent:.1f}%")
		
		if alerts:
			self.alert_conditions.extend(alerts)
			self.logger.warning("Resource alerts detected", alerts=alerts)
	
	def get_performance_summary(self) -> Dict[str, Any]:
		"""Get performance summary statistics"""
		try:
			current_hour = datetime.utcnow().strftime('%Y-%m-%d-%H')
			hour_stats = self.hourly_stats.get(current_hour, {})
			
			# Calculate recent average processing time
			recent_times = [p['duration'] for p in self.recent_processing_times 
			               if p['timestamp'] > datetime.utcnow() - timedelta(minutes=10)]
			avg_processing_time = sum(recent_times) / len(recent_times) if recent_times else 0
			
			# Calculate success rate
			successful = hour_stats.get('successful_jobs', 0)
			total = hour_stats.get('total_jobs', 0)
			success_rate = successful / total if total > 0 else 0
			
			# Get latest resource usage
			latest_resources = self.resource_history[-1] if self.resource_history else {}
			
			return {
				'timestamp': datetime.utcnow().isoformat(),
				'processing_stats': {
					'avg_processing_time_seconds': avg_processing_time,
					'jobs_this_hour': total,
					'successful_jobs_this_hour': successful,
					'failed_jobs_this_hour': hour_stats.get('failed_jobs', 0),
					'success_rate': success_rate,
					'total_processing_time_this_hour': hour_stats.get('total_duration', 0)
				},
				'resource_usage': {
					'cpu_percent': latest_resources.get('cpu_percent', 0),
					'memory_percent': latest_resources.get('memory_percent', 0),
					'memory_available_gb': latest_resources.get('memory_available_gb', 0),
					'disk_usage_percent': latest_resources.get('disk_usage_percent', 0)
				},
				'queue_stats': {
					'active_jobs': sum(self.active_jobs._value.values()),
					'queue_length': self.queue_length._value.get()
				},
				'alerts': self.alert_conditions[-10:] if self.alert_conditions else []
			}
			
		except Exception as e:
			self.logger.error("Failed to generate performance summary",
			                 error=str(e))
			return {'error': str(e)}
	
	def export_prometheus_metrics(self) -> str:
		"""Export metrics in Prometheus format"""
		return generate_latest(self.registry).decode('utf-8')


class CVCacheManager:
	"""
	Advanced Caching Manager
	
	Multi-level caching system for computer vision processing including
	local memory cache, distributed Redis cache, and intelligent cache
	invalidation with performance optimization and memory management.
	"""
	
	def __init__(self, config: PerformanceConfig, metrics: CVPerformanceMetrics):
		self.config = config
		self.metrics = metrics
		self.logger = structlog.get_logger("cv_cache")
		
		# Local memory cache
		self.local_cache: Dict[str, Dict[str, Any]] = {}
		self.cache_access_times: Dict[str, datetime] = {}
		self.cache_hit_counts: Dict[str, int] = defaultdict(int)
		self.cache_lock = Lock()
		
		# Redis distributed cache
		try:
			self.redis_client = redis.Redis(
				host='localhost',
				port=6379,
				db=0,
				decode_responses=True,
				socket_timeout=5,
				retry_on_timeout=True
			)
			self.redis_available = True
			self.redis_client.ping()  # Test connection
		except Exception as e:
			self.logger.warning("Redis not available, using local cache only",
			                   error=str(e))
			self.redis_available = False
		
		# Cache statistics
		self.cache_stats = {
			'local_hits': 0,
			'local_misses': 0,
			'redis_hits': 0,
			'redis_misses': 0,
			'evictions': 0,
			'total_size_bytes': 0
		}
		
		self.logger.info("Cache manager initialized",
		                local_cache_enabled=True,
		                redis_enabled=self.redis_available)
	
	def _generate_cache_key(self, prefix: str, **kwargs) -> str:
		"""Generate cache key from parameters"""
		key_data = json.dumps(kwargs, sort_keys=True)
		key_hash = hashlib.sha256(key_data.encode()).hexdigest()[:16]
		return f"cv:{prefix}:{key_hash}"
	
	async def get_processing_result(
		self,
		file_path: str,
		processing_type: ProcessingType,
		parameters: Dict[str, Any],
		tenant_id: str
	) -> Optional[Dict[str, Any]]:
		"""Get cached processing result"""
		cache_key = self._generate_cache_key(
			"processing",
			file_path=file_path,
			processing_type=processing_type.value,
			parameters=parameters,
			tenant_id=tenant_id
		)
		
		try:
			# Try local cache first
			result = await self._get_from_local_cache(cache_key)
			if result is not None:
				self.metrics.record_cache_hit("local")
				self.cache_stats['local_hits'] += 1
				return result
			
			# Try Redis cache
			if self.redis_available:
				result = await self._get_from_redis_cache(cache_key)
				if result is not None:
					# Store in local cache for faster access
					await self._store_in_local_cache(cache_key, result)
					self.metrics.record_cache_hit("redis")
					self.cache_stats['redis_hits'] += 1
					return result
			
			# Cache miss
			self.metrics.record_cache_miss("processing")
			self.cache_stats['local_misses'] += 1
			if self.redis_available:
				self.cache_stats['redis_misses'] += 1
			
			return None
			
		except Exception as e:
			self.logger.error("Failed to get cached result",
			                 cache_key=cache_key,
			                 error=str(e))
			return None
	
	async def store_processing_result(
		self,
		file_path: str,
		processing_type: ProcessingType,
		parameters: Dict[str, Any],
		tenant_id: str,
		result: Dict[str, Any]
	) -> None:
		"""Store processing result in cache"""
		cache_key = self._generate_cache_key(
			"processing",
			file_path=file_path,
			processing_type=processing_type.value,
			parameters=parameters,
			tenant_id=tenant_id
		)
		
		try:
			# Store in local cache
			await self._store_in_local_cache(cache_key, result)
			
			# Store in Redis cache
			if self.redis_available:
				await self._store_in_redis_cache(cache_key, result)
			
			self.logger.debug("Stored processing result in cache",
			                 cache_key=cache_key,
			                 result_size=len(json.dumps(result)))
			
		except Exception as e:
			self.logger.error("Failed to store result in cache",
			                 cache_key=cache_key,
			                 error=str(e))
	
	async def get_model_cache(self, model_name: str, model_version: str) -> Optional[Any]:
		"""Get cached model instance"""
		cache_key = self._generate_cache_key(
			"model",
			model_name=model_name,
			model_version=model_version
		)
		
		# Models are only cached locally due to size
		result = await self._get_from_local_cache(cache_key)
		if result:
			self.metrics.record_cache_hit("model")
		else:
			self.metrics.record_cache_miss("model")
		
		return result
	
	async def store_model_cache(
		self,
		model_name: str,
		model_version: str,
		model_instance: Any
	) -> None:
		"""Store model instance in cache"""
		cache_key = self._generate_cache_key(
			"model",
			model_name=model_name,
			model_version=model_version
		)
		
		try:
			await self._store_in_local_cache(cache_key, model_instance)
			self.logger.debug("Stored model in cache",
			                 model_name=model_name,
			                 model_version=model_version)
		except Exception as e:
			self.logger.error("Failed to store model in cache",
			                 model_name=model_name,
			                 error=str(e))
	
	async def _get_from_local_cache(self, cache_key: str) -> Optional[Any]:
		"""Get value from local memory cache"""
		with self.cache_lock:
			if cache_key in self.local_cache:
				# Check expiration
				cache_entry = self.local_cache[cache_key]
				if cache_entry['expires_at'] > datetime.utcnow():
					# Update access time
					self.cache_access_times[cache_key] = datetime.utcnow()
					self.cache_hit_counts[cache_key] += 1
					return cache_entry['value']
				else:
					# Remove expired entry
					del self.local_cache[cache_key]
					if cache_key in self.cache_access_times:
						del self.cache_access_times[cache_key]
					if cache_key in self.cache_hit_counts:
						del self.cache_hit_counts[cache_key]
		
		return None
	
	async def _store_in_local_cache(self, cache_key: str, value: Any) -> None:
		"""Store value in local memory cache"""
		with self.cache_lock:
			# Check cache size and evict if necessary
			await self._evict_if_necessary()
			
			expires_at = datetime.utcnow() + timedelta(seconds=self.config.cache_ttl_seconds)
			
			self.local_cache[cache_key] = {
				'value': value,
				'expires_at': expires_at,
				'created_at': datetime.utcnow(),
				'size_bytes': self._estimate_size(value)
			}
			
			self.cache_access_times[cache_key] = datetime.utcnow()
			self.cache_hit_counts[cache_key] = 0
			
			# Update cache size statistics
			self._update_cache_size_stats()
	
	async def _get_from_redis_cache(self, cache_key: str) -> Optional[Any]:
		"""Get value from Redis cache"""
		try:
			cached_data = self.redis_client.get(cache_key)
			if cached_data:
				return json.loads(cached_data)
		except Exception as e:
			self.logger.warning("Redis cache get failed",
			                   cache_key=cache_key,
			                   error=str(e))
		return None
	
	async def _store_in_redis_cache(self, cache_key: str, value: Any) -> None:
		"""Store value in Redis cache"""
		try:
			serialized_value = json.dumps(value, default=str)
			self.redis_client.setex(
				cache_key,
				self.config.cache_ttl_seconds,
				serialized_value
			)
		except Exception as e:
			self.logger.warning("Redis cache store failed",
			                   cache_key=cache_key,
			                   error=str(e))
	
	async def _evict_if_necessary(self) -> None:
		"""Evict least recently used items if cache is full"""
		current_size_mb = self._get_current_cache_size_mb()
		
		if current_size_mb > self.config.max_cache_size_mb:
			# Sort by access time (LRU)
			sorted_keys = sorted(
				self.cache_access_times.items(),
				key=lambda x: x[1]
			)
			
			# Evict oldest 25% of entries
			eviction_count = len(sorted_keys) // 4
			for cache_key, _ in sorted_keys[:eviction_count]:
				if cache_key in self.local_cache:
					del self.local_cache[cache_key]
				if cache_key in self.cache_access_times:
					del self.cache_access_times[cache_key]
				if cache_key in self.cache_hit_counts:
					del self.cache_hit_counts[cache_key]
				
				self.cache_stats['evictions'] += 1
			
			self.logger.info("Evicted cache entries",
			                eviction_count=eviction_count,
			                cache_size_mb=current_size_mb)
	
	def _estimate_size(self, value: Any) -> int:
		"""Estimate memory size of cached value"""
		try:
			if isinstance(value, str):
				return len(value.encode('utf-8'))
			elif isinstance(value, (dict, list)):
				return len(json.dumps(value, default=str).encode('utf-8'))
			else:
				return 1024  # Default estimate
		except Exception:
			return 1024
	
	def _get_current_cache_size_mb(self) -> float:
		"""Get current cache size in MB"""
		total_bytes = sum(
			entry['size_bytes'] for entry in self.local_cache.values()
		)
		return total_bytes / (1024 * 1024)
	
	def _update_cache_size_stats(self) -> None:
		"""Update cache size statistics"""
		self.cache_stats['total_size_bytes'] = sum(
			entry['size_bytes'] for entry in self.local_cache.values()
		)
	
	def get_cache_stats(self) -> Dict[str, Any]:
		"""Get cache performance statistics"""
		with self.cache_lock:
			total_requests = (
				self.cache_stats['local_hits'] + 
				self.cache_stats['local_misses']
			)
			
			hit_rate = (
				self.cache_stats['local_hits'] / total_requests 
				if total_requests > 0 else 0
			)
			
			return {
				'local_cache': {
					'entries': len(self.local_cache),
					'size_mb': self._get_current_cache_size_mb(),
					'max_size_mb': self.config.max_cache_size_mb,
					'hit_rate': hit_rate,
					'hits': self.cache_stats['local_hits'],
					'misses': self.cache_stats['local_misses'],
					'evictions': self.cache_stats['evictions']
				},
				'redis_cache': {
					'enabled': self.redis_available,
					'hits': self.cache_stats['redis_hits'],
					'misses': self.cache_stats['redis_misses']
				} if self.redis_available else {'enabled': False},
				'performance': {
					'overall_hit_rate': hit_rate,
					'total_requests': total_requests
				}
			}
	
	async def clear_cache(self, pattern: Optional[str] = None) -> int:
		"""Clear cache entries matching pattern"""
		cleared_count = 0
		
		with self.cache_lock:
			if pattern:
				# Clear entries matching pattern
				keys_to_remove = [
					key for key in self.local_cache.keys()
					if pattern in key
				]
			else:
				# Clear all entries
				keys_to_remove = list(self.local_cache.keys())
			
			for key in keys_to_remove:
				if key in self.local_cache:
					del self.local_cache[key]
				if key in self.cache_access_times:
					del self.cache_access_times[key]
				if key in self.cache_hit_counts:
					del self.cache_hit_counts[key]
				cleared_count += 1
		
		# Clear Redis cache if available
		if self.redis_available and pattern:
			try:
				redis_keys = self.redis_client.keys(f"cv:*{pattern}*")
				if redis_keys:
					self.redis_client.delete(*redis_keys)
					cleared_count += len(redis_keys)
			except Exception as e:
				self.logger.warning("Failed to clear Redis cache",
				                   pattern=pattern,
				                   error=str(e))
		
		self.logger.info("Cache cleared",
		                pattern=pattern,
		                cleared_count=cleared_count)
		
		return cleared_count


class CVAutoScaler:
	"""
	Auto-Scaling Manager
	
	Intelligent auto-scaling system for computer vision workloads based on
	CPU usage, memory consumption, queue length, and processing demand with
	Kubernetes integration and predictive scaling capabilities.
	"""
	
	def __init__(self, config: PerformanceConfig, metrics: CVPerformanceMetrics):
		self.config = config
		self.metrics = metrics
		self.logger = structlog.get_logger("cv_autoscaler")
		
		# Scaling state
		self.current_replicas = 3  # Default starting replicas
		self.min_replicas = 2
		self.max_replicas = 20
		self.last_scale_time = datetime.utcnow()
		self.scale_cooldown_minutes = 5
		
		# Scaling decision history
		self.scaling_history = deque(maxlen=100)
		self.resource_samples = deque(maxlen=10)  # For averaging
		
		self.logger.info("Auto-scaler initialized",
		                enabled=self.config.auto_scaling_enabled,
		                current_replicas=self.current_replicas)
	
	async def evaluate_scaling_needs(self) -> Optional[Dict[str, Any]]:
		"""Evaluate if scaling action is needed"""
		if not self.config.auto_scaling_enabled:
			return None
		
		try:
			# Collect current metrics
			current_metrics = await self._collect_scaling_metrics()
			self.resource_samples.append(current_metrics)
			
			# Calculate averages over recent samples
			avg_metrics = self._calculate_average_metrics()
			
			# Make scaling decision
			scaling_decision = self._make_scaling_decision(avg_metrics)
			
			if scaling_decision:
				# Check cooldown period
				if self._can_scale():
					await self._execute_scaling_action(scaling_decision)
					return scaling_decision
				else:
					self.logger.debug("Scaling action skipped due to cooldown")
			
			return None
			
		except Exception as e:
			self.logger.error("Failed to evaluate scaling needs",
			                 error=str(e))
			return None
	
	async def _collect_scaling_metrics(self) -> Dict[str, float]:
		"""Collect metrics for scaling decisions"""
		try:
			# Get system resource usage
			cpu_percent = psutil.cpu_percent(interval=1)
			memory = psutil.virtual_memory()
			memory_percent = memory.percent
			
			# Get queue metrics
			queue_length = self.metrics.queue_length._value.get()
			active_jobs = sum(self.metrics.active_jobs._value.values())
			
			# Calculate recent processing rate
			recent_jobs = [
				p for p in self.metrics.recent_processing_times
				if p['timestamp'] > datetime.utcnow() - timedelta(minutes=5)
			]
			processing_rate = len(recent_jobs) / 5.0  # Jobs per minute
			
			return {
				'cpu_percent': cpu_percent,
				'memory_percent': memory_percent,
				'queue_length': queue_length,
				'active_jobs': active_jobs,
				'processing_rate': processing_rate,
				'timestamp': datetime.utcnow().timestamp()
			}
			
		except Exception as e:
			self.logger.error("Failed to collect scaling metrics",
			                 error=str(e))
			return {}
	
	def _calculate_average_metrics(self) -> Dict[str, float]:
		"""Calculate average metrics over recent samples"""
		if not self.resource_samples:
			return {}
		
		metrics_sum = defaultdict(float)
		sample_count = len(self.resource_samples)
		
		for sample in self.resource_samples:
			for key, value in sample.items():
				if isinstance(value, (int, float)) and key != 'timestamp':
					metrics_sum[key] += value
		
		return {
			key: value / sample_count 
			for key, value in metrics_sum.items()
		}
	
	def _make_scaling_decision(self, metrics: Dict[str, float]) -> Optional[Dict[str, Any]]:
		"""Make scaling decision based on metrics"""
		if not metrics:
			return None
		
		scale_up_reasons = []
		scale_down_reasons = []
		
		# CPU-based scaling
		cpu_percent = metrics.get('cpu_percent', 0)
		if cpu_percent > self.config.scale_up_cpu_threshold:
			scale_up_reasons.append(f"High CPU usage: {cpu_percent:.1f}%")
		elif cpu_percent < self.config.scale_down_cpu_threshold:
			scale_down_reasons.append(f"Low CPU usage: {cpu_percent:.1f}%")
		
		# Memory-based scaling
		memory_percent = metrics.get('memory_percent', 0)
		if memory_percent > self.config.scale_up_memory_threshold:
			scale_up_reasons.append(f"High memory usage: {memory_percent:.1f}%")
		elif memory_percent < self.config.scale_down_memory_threshold:
			scale_down_reasons.append(f"Low memory usage: {memory_percent:.1f}%")
		
		# Queue-based scaling
		queue_length = metrics.get('queue_length', 0)
		if queue_length > self.config.scale_up_queue_threshold:
			scale_up_reasons.append(f"Long queue: {queue_length} jobs")
		elif queue_length < self.config.scale_down_queue_threshold:
			scale_down_reasons.append(f"Short queue: {queue_length} jobs")
		
		# Processing rate considerations
		processing_rate = metrics.get('processing_rate', 0)
		active_jobs = metrics.get('active_jobs', 0)
		
		if processing_rate > 0 and active_jobs / processing_rate > 10:  # >10 min backlog
			scale_up_reasons.append(f"High processing backlog")
		
		# Make decision
		if len(scale_up_reasons) >= 2 and self.current_replicas < self.max_replicas:
			target_replicas = min(self.current_replicas + 1, self.max_replicas)
			return {
				'action': 'scale_up',
				'current_replicas': self.current_replicas,
				'target_replicas': target_replicas,
				'reasons': scale_up_reasons,
				'metrics': metrics
			}
		elif len(scale_down_reasons) >= 2 and self.current_replicas > self.min_replicas:
			target_replicas = max(self.current_replicas - 1, self.min_replicas)
			return {
				'action': 'scale_down',
				'current_replicas': self.current_replicas,
				'target_replicas': target_replicas,
				'reasons': scale_down_reasons,
				'metrics': metrics
			}
		
		return None
	
	def _can_scale(self) -> bool:
		"""Check if scaling action is allowed based on cooldown"""
		time_since_last_scale = datetime.utcnow() - self.last_scale_time
		return time_since_last_scale.total_seconds() > (self.scale_cooldown_minutes * 60)
	
	async def _execute_scaling_action(self, decision: Dict[str, Any]) -> None:
		"""Execute scaling action"""
		try:
			action = decision['action']
			target_replicas = decision['target_replicas']
			
			# In a real implementation, this would call Kubernetes API
			# For now, we simulate the scaling action
			self.logger.info("Executing scaling action",
			                action=action,
			                current_replicas=self.current_replicas,
			                target_replicas=target_replicas,
			                reasons=decision['reasons'])
			
			# Update state
			self.current_replicas = target_replicas
			self.last_scale_time = datetime.utcnow()
			
			# Record scaling event
			self.scaling_history.append({
				'timestamp': datetime.utcnow(),
				'action': action,
				'from_replicas': decision['current_replicas'],
				'to_replicas': target_replicas,
				'reasons': decision['reasons'],
				'metrics': decision['metrics']
			})
			
			# In real implementation:
			# await self._kubernetes_scale_deployment(target_replicas)
			
		except Exception as e:
			self.logger.error("Failed to execute scaling action",
			                 decision=decision,
			                 error=str(e))
	
	def get_scaling_status(self) -> Dict[str, Any]:
		"""Get current scaling status and history"""
		return {
			'enabled': self.config.auto_scaling_enabled,
			'current_replicas': self.current_replicas,
			'min_replicas': self.min_replicas,
			'max_replicas': self.max_replicas,
			'last_scale_time': self.last_scale_time.isoformat(),
			'cooldown_minutes': self.scale_cooldown_minutes,
			'can_scale': self._can_scale(),
			'recent_scaling_events': list(self.scaling_history)[-5:],
			'current_metrics': self.resource_samples[-1] if self.resource_samples else {}
		}


class CVPerformanceMonitor:
	"""
	Comprehensive Performance Monitor
	
	Central performance monitoring system that coordinates metrics collection,
	caching, auto-scaling, and alerting with real-time monitoring, historical
	analysis, and predictive insights for computer vision workloads.
	"""
	
	def __init__(self, config: Optional[PerformanceConfig] = None):
		self.config = config or PerformanceConfig()
		self.logger = structlog.get_logger("cv_performance_monitor")
		
		# Initialize components
		self.metrics = CVPerformanceMetrics(self.config)
		self.cache_manager = CVCacheManager(self.config, self.metrics)
		self.auto_scaler = CVAutoScaler(self.config, self.metrics)
		
		# Monitoring state
		self.monitoring_active = False
		self.monitoring_task: Optional[asyncio.Task] = None
		
		self.logger.info("Performance monitor initialized",
		                config=self.config.__dict__)
	
	async def start_monitoring(self) -> None:
		"""Start performance monitoring"""
		if self.monitoring_active:
			self.logger.warning("Performance monitoring already active")
			return
		
		self.monitoring_active = True
		self.monitoring_task = asyncio.create_task(self._monitoring_loop())
		
		self.logger.info("Performance monitoring started")
	
	async def stop_monitoring(self) -> None:
		"""Stop performance monitoring"""
		if not self.monitoring_active:
			return
		
		self.monitoring_active = False
		
		if self.monitoring_task:
			self.monitoring_task.cancel()
			try:
				await self.monitoring_task
			except asyncio.CancelledError:
				pass
		
		self.logger.info("Performance monitoring stopped")
	
	async def _monitoring_loop(self) -> None:
		"""Main monitoring loop"""
		while self.monitoring_active:
			try:
				# Update resource metrics
				self.metrics.update_resource_metrics()
				
				# Evaluate auto-scaling needs
				scaling_decision = await self.auto_scaler.evaluate_scaling_needs()
				if scaling_decision:
					self.logger.info("Auto-scaling action taken",
					                decision=scaling_decision)
				
				# Perform cache maintenance
				gc.collect()  # Force garbage collection
				
				await asyncio.sleep(self.config.monitoring_interval_seconds)
				
			except asyncio.CancelledError:
				break
			except Exception as e:
				self.logger.error("Error in monitoring loop",
				                 error=str(e))
				await asyncio.sleep(5)  # Brief pause before retrying
	
	def get_comprehensive_status(self) -> Dict[str, Any]:
		"""Get comprehensive performance status"""
		return {
			'timestamp': datetime.utcnow().isoformat(),
			'monitoring_active': self.monitoring_active,
			'performance_summary': self.metrics.get_performance_summary(),
			'cache_stats': self.cache_manager.get_cache_stats(),
			'scaling_status': self.auto_scaler.get_scaling_status(),
			'config': self.config.__dict__
		}


# Export main classes
__all__ = [
	'PerformanceConfig',
	'CVPerformanceMetrics', 
	'CVCacheManager',
	'CVAutoScaler',
	'CVPerformanceMonitor'
]