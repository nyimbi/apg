#!/usr/bin/env python3
"""APG Cash Management - Advanced Performance Optimizer

World-class performance optimization system with intelligent caching,
connection pooling, and real-time performance monitoring.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero | APG Platform Architect
"""

import asyncio
import time
import hashlib
import json
import statistics
from typing import Dict, Any, Optional, List, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
from contextlib import asynccontextmanager

import redis.asyncio as redis
import asyncpg
from pydantic import BaseModel, Field, ConfigDict
from uuid_extensions import uuid7str
import psutil
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CacheStrategy(str, Enum):
	"""Cache strategy types."""
	LRU = "lru"
	LFU = "lfu"
	TTL = "ttl"
	ADAPTIVE = "adaptive"
	PREDICTIVE = "predictive"

class PerformanceMetric(str, Enum):
	"""Performance metrics for optimization."""
	LATENCY = "latency"
	THROUGHPUT = "throughput"
	MEMORY_USAGE = "memory_usage"
	CPU_USAGE = "cpu_usage"
	CACHE_HIT_RATE = "cache_hit_rate"
	DB_CONNECTION_POOL = "db_connection_pool"
	ERROR_RATE = "error_rate"

@dataclass
class PerformanceSnapshot:
	"""Performance metrics snapshot."""
	timestamp: datetime
	latency_ms: float
	throughput_rps: float
	memory_usage_mb: float
	cpu_usage_percent: float
	cache_hit_rate: float
	active_connections: int
	error_rate: float
	metadata: Dict[str, Any] = field(default_factory=dict)

class CacheConfiguration(BaseModel):
	"""Advanced cache configuration."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	strategy: CacheStrategy = CacheStrategy.ADAPTIVE
	max_memory_mb: int = Field(default=512, ge=64, le=8192)
	default_ttl_seconds: int = Field(default=3600, ge=60, le=86400)
	max_keys: int = Field(default=100000, ge=1000, le=1000000)
	compression_enabled: bool = True
	compression_threshold_bytes: int = Field(default=1024, ge=256)
	
	# Adaptive caching parameters
	access_pattern_window: int = Field(default=1000, ge=100)
	prediction_horizon_minutes: int = Field(default=30, ge=5, le=240)
	learning_rate: float = Field(default=0.01, ge=0.001, le=0.1)
	
	# Performance thresholds
	eviction_threshold: float = Field(default=0.9, ge=0.7, le=0.95)
	warming_threshold: float = Field(default=0.6, ge=0.3, le=0.8)

class ConnectionPoolConfiguration(BaseModel):
	"""Advanced connection pool configuration."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	min_connections: int = Field(default=5, ge=1, le=50)
	max_connections: int = Field(default=50, ge=10, le=500)
	connection_timeout_seconds: int = Field(default=30, ge=5, le=120)
	idle_timeout_seconds: int = Field(default=300, ge=60, le=3600)
	max_lifetime_seconds: int = Field(default=3600, ge=300, le=14400)
	
	# Health monitoring
	health_check_interval_seconds: int = Field(default=30, ge=10, le=300)
	max_retries: int = Field(default=3, ge=1, le=10)
	retry_delay_seconds: float = Field(default=1.0, ge=0.1, le=10.0)
	
	# Load balancing
	load_balancing_strategy: str = Field(default="round_robin")
	connection_reuse_enabled: bool = True

class PerformanceOptimizer:
	"""Advanced performance optimization engine."""
	
	def __init__(
		self,
		tenant_id: str,
		redis_url: str = "redis://localhost:6379/0",
		cache_config: Optional[CacheConfiguration] = None,
		pool_config: Optional[ConnectionPoolConfiguration] = None
	):
		self.tenant_id = tenant_id
		self.redis_url = redis_url
		self.cache_config = cache_config or CacheConfiguration()
		self.pool_config = pool_config or ConnectionPoolConfiguration()
		
		# Performance tracking
		self.performance_history: List[PerformanceSnapshot] = []
		self.cache_stats = {
			'hits': 0,
			'misses': 0,
			'evictions': 0,
			'memory_usage': 0
		}
		
		# Connection pools
		self.db_pools: Dict[str, asyncpg.Pool] = {}
		self.redis_pool: Optional[redis.ConnectionPool] = None
		
		# Performance monitors
		self.active_monitors: Dict[str, asyncio.Task] = {}
		self.optimization_tasks: Dict[str, asyncio.Task] = {}
		
		# Cache intelligence
		self.access_patterns: Dict[str, List[float]] = {}
		self.prediction_model: Optional[Dict[str, Any]] = None
		
		logger.info(f"Initialized PerformanceOptimizer for tenant {tenant_id}")
	
	async def initialize(self) -> None:
		"""Initialize performance optimization system."""
		try:
			# Initialize Redis connection pool
			await self._initialize_redis_pool()
			
			# Initialize database connection pools
			await self._initialize_db_pools()
			
			# Start performance monitoring
			await self._start_performance_monitoring()
			
			# Initialize cache intelligence
			await self._initialize_cache_intelligence()
			
			logger.info("Performance optimization system initialized")
			
		except Exception as e:
			logger.error(f"Failed to initialize performance optimizer: {e}")
			raise
	
	async def _initialize_redis_pool(self) -> None:
		"""Initialize Redis connection pool with optimization."""
		self.redis_pool = redis.ConnectionPool.from_url(
			self.redis_url,
			max_connections=self.pool_config.max_connections,
			socket_connect_timeout=self.pool_config.connection_timeout_seconds,
			socket_timeout=self.pool_config.idle_timeout_seconds,
			health_check_interval=self.pool_config.health_check_interval_seconds,
			retry_on_timeout=True,
			retry_on_error=[redis.ConnectionError, redis.TimeoutError]
		)
		logger.info("Redis connection pool initialized")
	
	async def _initialize_db_pools(self) -> None:
		"""Initialize database connection pools."""
		# Primary database pool
		self.db_pools['primary'] = await asyncpg.create_pool(
			database="apg_cash_production",
			user="apg_user",
			host="localhost",
			port=5432,
			min_size=self.pool_config.min_connections,
			max_size=self.pool_config.max_connections,
			command_timeout=self.pool_config.connection_timeout_seconds,
			max_inactive_connection_lifetime=self.pool_config.idle_timeout_seconds,
			max_queries=1000,
			setup=self._setup_db_connection
		)
		
		# Read replica pool
		self.db_pools['replica'] = await asyncpg.create_pool(
			database="apg_cash_production",
			user="apg_readonly",
			host="localhost",
			port=5433,
			min_size=max(1, self.pool_config.min_connections // 2),
			max_size=self.pool_config.max_connections,
			command_timeout=self.pool_config.connection_timeout_seconds,
			max_inactive_connection_lifetime=self.pool_config.idle_timeout_seconds,
			setup=self._setup_db_connection
		)
		
		logger.info("Database connection pools initialized")
	
	async def _setup_db_connection(self, connection) -> None:
		"""Setup individual database connection."""
		await connection.execute("SET statement_timeout = '30s'")
		await connection.execute("SET lock_timeout = '10s'")
		await connection.execute("SET idle_in_transaction_session_timeout = '60s'")
	
	async def _start_performance_monitoring(self) -> None:
		"""Start performance monitoring tasks."""
		self.active_monitors['system_metrics'] = asyncio.create_task(
			self._monitor_system_metrics()
		)
		self.active_monitors['cache_performance'] = asyncio.create_task(
			self._monitor_cache_performance()
		)
		self.active_monitors['db_performance'] = asyncio.create_task(
			self._monitor_db_performance()
		)
		
		logger.info("Performance monitoring started")
	
	async def _initialize_cache_intelligence(self) -> None:
		"""Initialize intelligent caching system."""
		# Load existing access patterns
		redis_client = redis.Redis(connection_pool=self.redis_pool)
		
		try:
			patterns_key = f"cache:patterns:{self.tenant_id}"
			patterns_data = await redis_client.get(patterns_key)
			
			if patterns_data:
				self.access_patterns = json.loads(patterns_data.decode())
				logger.info("Loaded existing access patterns")
			
			# Initialize prediction model
			await self._build_prediction_model()
			
		except Exception as e:
			logger.warning(f"Could not load access patterns: {e}")
		finally:
			await redis_client.close()
	
	async def get_optimized(
		self,
		key: str,
		fetch_function: Callable,
		ttl_seconds: Optional[int] = None,
		cache_strategy: Optional[CacheStrategy] = None
	) -> Any:
		"""Get data with intelligent caching optimization."""
		start_time = time.time()
		
		try:
			# Record access pattern
			await self._record_access_pattern(key)
			
			# Check cache first
			cached_value = await self._get_from_cache(key)
			if cached_value is not None:
				self.cache_stats['hits'] += 1
				latency = (time.time() - start_time) * 1000
				await self._record_cache_hit(key, latency)
				return cached_value
			
			# Cache miss - fetch data
			self.cache_stats['misses'] += 1
			value = await fetch_function()
			
			# Determine optimal TTL and strategy
			optimal_ttl = await self._calculate_optimal_ttl(key, ttl_seconds)
			optimal_strategy = cache_strategy or await self._determine_cache_strategy(key)
			
			# Store in cache
			await self._set_in_cache(key, value, optimal_ttl, optimal_strategy)
			
			latency = (time.time() - start_time) * 1000
			await self._record_cache_miss(key, latency)
			
			return value
			
		except Exception as e:
			logger.error(f"Error in optimized get for key {key}: {e}")
			# Fallback to direct fetch
			return await fetch_function()
	
	async def _get_from_cache(self, key: str) -> Optional[Any]:
		"""Get value from cache with compression support."""
		redis_client = redis.Redis(connection_pool=self.redis_pool)
		
		try:
			cache_key = f"cache:{self.tenant_id}:{key}"
			
			# Check if compressed
			metadata_key = f"{cache_key}:meta"
			metadata = await redis_client.hgetall(metadata_key)
			
			raw_value = await redis_client.get(cache_key)
			if raw_value is None:
				return None
			
			# Decompress if needed
			if metadata.get(b'compressed') == b'true':
				import zlib
				raw_value = zlib.decompress(raw_value)
			
			# Deserialize
			value = json.loads(raw_value.decode())
			return value
			
		except Exception as e:
			logger.warning(f"Cache get error for key {key}: {e}")
			return None
		finally:
			await redis_client.close()
	
	async def _set_in_cache(
		self,
		key: str,
		value: Any,
		ttl_seconds: int,
		strategy: CacheStrategy
	) -> None:
		"""Set value in cache with optimization."""
		redis_client = redis.Redis(connection_pool=self.redis_pool)
		
		try:
			cache_key = f"cache:{self.tenant_id}:{key}"
			serialized_value = json.dumps(value).encode()
			
			# Compress if beneficial
			compressed = False
			if (len(serialized_value) > self.cache_config.compression_threshold_bytes 
				and self.cache_config.compression_enabled):
				import zlib
				compressed_value = zlib.compress(serialized_value)
				if len(compressed_value) < len(serialized_value) * 0.9:  # 10% compression minimum
					serialized_value = compressed_value
					compressed = True
			
			# Store value with TTL
			await redis_client.setex(cache_key, ttl_seconds, serialized_value)
			
			# Store metadata
			metadata_key = f"{cache_key}:meta"
			metadata = {
				'strategy': strategy.value,
				'compressed': str(compressed).lower(),
				'size_bytes': len(serialized_value),
				'created_at': datetime.now().isoformat(),
				'access_count': 0
			}
			await redis_client.hset(metadata_key, mapping=metadata)
			await redis_client.expire(metadata_key, ttl_seconds)
			
		except Exception as e:
			logger.warning(f"Cache set error for key {key}: {e}")
		finally:
			await redis_client.close()
	
	async def _record_access_pattern(self, key: str) -> None:
		"""Record access pattern for predictive caching."""
		now = time.time()
		
		if key not in self.access_patterns:
			self.access_patterns[key] = []
		
		self.access_patterns[key].append(now)
		
		# Keep only recent accesses
		cutoff = now - (self.cache_config.access_pattern_window * 60)
		self.access_patterns[key] = [
			t for t in self.access_patterns[key] if t > cutoff
		]
		
		# Periodically save patterns
		if len(self.access_patterns[key]) % 10 == 0:
			await self._save_access_patterns()
	
	async def _calculate_optimal_ttl(self, key: str, default_ttl: Optional[int]) -> int:
		"""Calculate optimal TTL based on access patterns."""
		if default_ttl:
			return default_ttl
		
		if key not in self.access_patterns or len(self.access_patterns[key]) < 2:
			return self.cache_config.default_ttl_seconds
		
		# Calculate access frequency
		accesses = self.access_patterns[key]
		if len(accesses) >= 2:
			intervals = [accesses[i] - accesses[i-1] for i in range(1, len(accesses))]
			avg_interval = statistics.mean(intervals)
			
			# TTL should be roughly 2-3x the average access interval
			optimal_ttl = int(avg_interval * 2.5)
			
			# Clamp to reasonable bounds
			optimal_ttl = max(60, min(optimal_ttl, 86400))
			return optimal_ttl
		
		return self.cache_config.default_ttl_seconds
	
	async def _determine_cache_strategy(self, key: str) -> CacheStrategy:
		"""Determine optimal caching strategy for key."""
		if self.cache_config.strategy != CacheStrategy.ADAPTIVE:
			return self.cache_config.strategy
		
		if key not in self.access_patterns:
			return CacheStrategy.LRU
		
		accesses = self.access_patterns[key]
		
		# High frequency = LFU
		if len(accesses) > 50:
			return CacheStrategy.LFU
		
		# Regular pattern = TTL
		if len(accesses) >= 3:
			intervals = [accesses[i] - accesses[i-1] for i in range(1, len(accesses))]
			if statistics.stdev(intervals) < statistics.mean(intervals) * 0.3:
				return CacheStrategy.TTL
		
		# Default to LRU
		return CacheStrategy.LRU
	
	async def _monitor_system_metrics(self) -> None:
		"""Monitor system performance metrics."""
		while True:
			try:
				# Collect system metrics
				cpu_percent = psutil.cpu_percent(interval=1)
				memory = psutil.virtual_memory()
				
				# Calculate throughput (requests per second)
				current_time = time.time()
				if hasattr(self, '_last_request_count'):
					time_delta = current_time - self._last_metric_time
					request_delta = self._request_count - self._last_request_count
					throughput = request_delta / time_delta if time_delta > 0 else 0
				else:
					throughput = 0
				
				self._last_request_count = getattr(self, '_request_count', 0)
				self._last_metric_time = current_time
				
				# Calculate cache hit rate
				total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
				cache_hit_rate = (self.cache_stats['hits'] / total_requests) if total_requests > 0 else 0
				
				# Create performance snapshot
				snapshot = PerformanceSnapshot(
					timestamp=datetime.now(),
					latency_ms=0,  # Will be updated by request tracking
					throughput_rps=throughput,
					memory_usage_mb=memory.used / (1024 * 1024),
					cpu_usage_percent=cpu_percent,
					cache_hit_rate=cache_hit_rate,
					active_connections=sum(pool.get_size() for pool in self.db_pools.values()),
					error_rate=0,  # Will be updated by error tracking
					metadata={
						'memory_percent': memory.percent,
						'cache_stats': self.cache_stats.copy()
					}
				)
				
				self.performance_history.append(snapshot)
				
				# Keep only recent history
				cutoff = datetime.now() - timedelta(hours=24)
				self.performance_history = [
					s for s in self.performance_history if s.timestamp > cutoff
				]
				
				# Trigger optimization if needed
				await self._check_optimization_triggers(snapshot)
				
				await asyncio.sleep(30)  # Monitor every 30 seconds
				
			except Exception as e:
				logger.error(f"System metrics monitoring error: {e}")
				await asyncio.sleep(60)
	
	async def _monitor_cache_performance(self) -> None:
		"""Monitor cache performance and optimize."""
		while True:
			try:
				redis_client = redis.Redis(connection_pool=self.redis_pool)
				
				# Get Redis memory info
				redis_info = await redis_client.info('memory')
				memory_usage = redis_info.get('used_memory', 0)
				max_memory = redis_info.get('maxmemory', 0)
				
				self.cache_stats['memory_usage'] = memory_usage
				
				# Check if memory usage is high
				if max_memory > 0 and memory_usage / max_memory > self.cache_config.eviction_threshold:
					await self._trigger_cache_optimization()
				
				# Predictive cache warming
				if memory_usage / max_memory < self.cache_config.warming_threshold:
					await self._trigger_cache_warming()
				
				await redis_client.close()
				await asyncio.sleep(60)  # Check every minute
				
			except Exception as e:
				logger.error(f"Cache performance monitoring error: {e}")
				await asyncio.sleep(120)
	
	async def _monitor_db_performance(self) -> None:
		"""Monitor database performance."""
		while True:
			try:
				for pool_name, pool in self.db_pools.items():
					# Check pool health
					pool_size = pool.get_size()
					pool_idle = pool.get_idle_size()
					
					if pool_idle == 0 and pool_size < self.pool_config.max_connections:
						logger.warning(f"Database pool {pool_name} may need scaling")
					
					# Log pool statistics
					logger.debug(f"Pool {pool_name}: size={pool_size}, idle={pool_idle}")
				
				await asyncio.sleep(120)  # Check every 2 minutes
				
			except Exception as e:
				logger.error(f"Database performance monitoring error: {e}")
				await asyncio.sleep(180)
	
	async def _check_optimization_triggers(self, snapshot: PerformanceSnapshot) -> None:
		"""Check if optimization should be triggered."""
		# High CPU usage trigger
		if snapshot.cpu_usage_percent > 80:
			await self._trigger_cpu_optimization()
		
		# Low cache hit rate trigger
		if snapshot.cache_hit_rate < 0.6 and self.cache_stats['hits'] + self.cache_stats['misses'] > 100:
			await self._trigger_cache_optimization()
		
		# High memory usage trigger
		if snapshot.memory_usage_mb > 4096:  # 4GB threshold
			await self._trigger_memory_optimization()
	
	async def _trigger_cache_optimization(self) -> None:
		"""Trigger cache optimization."""
		if 'cache_optimization' in self.optimization_tasks:
			return  # Already running
		
		self.optimization_tasks['cache_optimization'] = asyncio.create_task(
			self._optimize_cache()
		)
		logger.info("Triggered cache optimization")
	
	async def _trigger_cache_warming(self) -> None:
		"""Trigger predictive cache warming."""
		if 'cache_warming' in self.optimization_tasks:
			return
		
		self.optimization_tasks['cache_warming'] = asyncio.create_task(
			self._warm_cache()
		)
		logger.info("Triggered cache warming")
	
	async def _trigger_cpu_optimization(self) -> None:
		"""Trigger CPU optimization."""
		logger.info("High CPU usage detected - optimizing connections")
		
		# Reduce connection pool sizes temporarily
		for pool in self.db_pools.values():
			if hasattr(pool, '_min_size'):
				pool._min_size = max(1, pool._min_size - 1)
	
	async def _trigger_memory_optimization(self) -> None:
		"""Trigger memory optimization."""
		logger.info("High memory usage detected - optimizing cache")
		
		# Force cache cleanup
		await self._optimize_cache()
		
		# Reduce cache size temporarily
		self.cache_config.max_memory_mb = int(self.cache_config.max_memory_mb * 0.8)
	
	async def _optimize_cache(self) -> None:
		"""Optimize cache performance."""
		try:
			redis_client = redis.Redis(connection_pool=self.redis_pool)
			
			# Get all cache keys for tenant
			pattern = f"cache:{self.tenant_id}:*"
			keys = await redis_client.keys(pattern)
			
			# Analyze access patterns and remove stale entries
			removed_count = 0
			for key in keys:
				if b':meta' in key:
					continue
				
				# Get metadata
				meta_key = key + b':meta'
				metadata = await redis_client.hgetall(meta_key)
				
				if not metadata:
					await redis_client.delete(key)
					removed_count += 1
					continue
				
				# Check access count
				access_count = int(metadata.get(b'access_count', b'0'))
				if access_count == 0:
					# Never accessed - remove
					await redis_client.delete(key, meta_key)
					removed_count += 1
			
			logger.info(f"Cache optimization completed - removed {removed_count} stale entries")
			
			await redis_client.close()
			
		except Exception as e:
			logger.error(f"Cache optimization error: {e}")
		finally:
			if 'cache_optimization' in self.optimization_tasks:
				del self.optimization_tasks['cache_optimization']
	
	async def _warm_cache(self) -> None:
		"""Warm cache with predicted requests."""
		try:
			# Use prediction model to identify likely cache misses
			predicted_keys = await self._predict_cache_needs()
			
			for key in predicted_keys:
				# Check if key is already cached
				cached = await self._get_from_cache(key)
				if cached is None:
					# Key would likely be requested - trigger warming
					logger.debug(f"Pre-warming cache for predicted key: {key}")
					# This would trigger the actual data fetch and cache
					# Implementation depends on specific use case
			
			logger.info(f"Cache warming completed for {len(predicted_keys)} keys")
			
		except Exception as e:
			logger.error(f"Cache warming error: {e}")
		finally:
			if 'cache_warming' in self.optimization_tasks:
				del self.optimization_tasks['cache_warming']
	
	async def _build_prediction_model(self) -> None:
		"""Build ML model for cache prediction."""
		if len(self.access_patterns) < 10:
			return
		
		# Simple frequency-based prediction model
		model = {}
		for key, accesses in self.access_patterns.items():
			if len(accesses) >= 3:
				# Calculate access frequency and pattern
				intervals = [accesses[i] - accesses[i-1] for i in range(1, len(accesses))]
				avg_interval = statistics.mean(intervals)
				
				# Predict next access time
				last_access = accesses[-1]
				predicted_next = last_access + avg_interval
				
				model[key] = {
					'predicted_next_access': predicted_next,
					'frequency': len(accesses) / (accesses[-1] - accesses[0]) if len(accesses) > 1 else 0,
					'confidence': 1.0 / (statistics.stdev(intervals) + 1) if len(intervals) > 1 else 0.5
				}
		
		self.prediction_model = model
		logger.info(f"Built prediction model for {len(model)} keys")
	
	async def _predict_cache_needs(self) -> List[str]:
		"""Predict which keys will be needed soon."""
		if not self.prediction_model:
			await self._build_prediction_model()
		
		if not self.prediction_model:
			return []
		
		now = time.time()
		horizon = self.cache_config.prediction_horizon_minutes * 60
		
		predicted_keys = []
		for key, prediction in self.prediction_model.items():
			predicted_time = prediction['predicted_next_access']
			confidence = prediction['confidence']
			
			# If predicted access is within horizon and confidence is high
			if (predicted_time - now) <= horizon and confidence > 0.5:
				predicted_keys.append(key)
		
		return predicted_keys
	
	async def _save_access_patterns(self) -> None:
		"""Save access patterns to Redis for persistence."""
		try:
			redis_client = redis.Redis(connection_pool=self.redis_pool)
			
			patterns_key = f"cache:patterns:{self.tenant_id}"
			patterns_data = json.dumps(self.access_patterns)
			
			await redis_client.setex(patterns_key, 86400, patterns_data)  # 24 hour TTL
			await redis_client.close()
			
		except Exception as e:
			logger.warning(f"Could not save access patterns: {e}")
	
	async def _record_cache_hit(self, key: str, latency_ms: float) -> None:
		"""Record cache hit metrics."""
		# Update access count in metadata
		try:
			redis_client = redis.Redis(connection_pool=self.redis_pool)
			metadata_key = f"cache:{self.tenant_id}:{key}:meta"
			await redis_client.hincrby(metadata_key, 'access_count', 1)
			await redis_client.close()
		except:
			pass
	
	async def _record_cache_miss(self, key: str, latency_ms: float) -> None:
		"""Record cache miss metrics."""
		# Cache miss recorded in stats already
		pass
	
	@asynccontextmanager
	async def get_db_connection(self, pool_name: str = 'primary'):
		"""Get database connection with automatic management."""
		if pool_name not in self.db_pools:
			raise ValueError(f"Unknown database pool: {pool_name}")
		
		pool = self.db_pools[pool_name]
		async with pool.acquire() as connection:
			yield connection
	
	async def get_performance_report(self) -> Dict[str, Any]:
		"""Generate comprehensive performance report."""
		if not self.performance_history:
			return {"status": "No performance data available"}
		
		recent_snapshots = self.performance_history[-100:]  # Last 100 snapshots
		
		# Calculate averages
		avg_latency = statistics.mean(s.latency_ms for s in recent_snapshots if s.latency_ms > 0)
		avg_throughput = statistics.mean(s.throughput_rps for s in recent_snapshots)
		avg_cpu = statistics.mean(s.cpu_usage_percent for s in recent_snapshots)
		avg_memory = statistics.mean(s.memory_usage_mb for s in recent_snapshots)
		avg_cache_hit_rate = statistics.mean(s.cache_hit_rate for s in recent_snapshots)
		
		return {
			"performance_summary": {
				"average_latency_ms": round(avg_latency, 2),
				"average_throughput_rps": round(avg_throughput, 2),
				"average_cpu_percent": round(avg_cpu, 2),
				"average_memory_mb": round(avg_memory, 2),
				"average_cache_hit_rate": round(avg_cache_hit_rate, 3)
			},
			"cache_statistics": self.cache_stats,
			"connection_pools": {
				name: {
					"size": pool.get_size(),
					"idle": pool.get_idle_size(),
					"max_size": self.pool_config.max_connections
				}
				for name, pool in self.db_pools.items()
			},
			"optimization_status": {
				"active_monitors": list(self.active_monitors.keys()),
				"running_optimizations": list(self.optimization_tasks.keys()),
				"cache_strategy": self.cache_config.strategy.value,
				"prediction_model_keys": len(self.prediction_model) if self.prediction_model else 0
			},
			"recommendations": await self._generate_recommendations()
		}
	
	async def _generate_recommendations(self) -> List[str]:
		"""Generate performance optimization recommendations."""
		recommendations = []
		
		if not self.performance_history:
			return recommendations
		
		recent = self.performance_history[-10:]
		
		# CPU recommendations
		avg_cpu = statistics.mean(s.cpu_usage_percent for s in recent)
		if avg_cpu > 80:
			recommendations.append("Consider scaling up CPU resources or optimizing query performance")
		elif avg_cpu < 20:
			recommendations.append("CPU resources may be over-provisioned")
		
		# Memory recommendations
		avg_memory = statistics.mean(s.memory_usage_mb for s in recent)
		if avg_memory > 6144:  # 6GB
			recommendations.append("Consider increasing memory allocation or optimizing cache usage")
		
		# Cache recommendations
		total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
		if total_requests > 100:
			hit_rate = self.cache_stats['hits'] / total_requests
			if hit_rate < 0.6:
				recommendations.append("Cache hit rate is low - consider adjusting TTL or cache strategy")
			elif hit_rate > 0.95:
				recommendations.append("Excellent cache performance - consider increasing cache size")
		
		# Connection pool recommendations
		for name, pool in self.db_pools.items():
			utilization = (pool.get_size() - pool.get_idle_size()) / pool.get_size()
			if utilization > 0.9:
				recommendations.append(f"Database pool '{name}' is highly utilized - consider increasing pool size")
		
		return recommendations
	
	async def cleanup(self) -> None:
		"""Cleanup resources."""
		# Stop monitoring tasks
		for monitor in self.active_monitors.values():
			monitor.cancel()
		
		# Stop optimization tasks
		for task in self.optimization_tasks.values():
			task.cancel()
		
		# Close connection pools
		for pool in self.db_pools.values():
			await pool.close()
		
		# Close Redis pool
		if self.redis_pool:
			await self.redis_pool.disconnect()
		
		# Save final access patterns
		await self._save_access_patterns()
		
		logger.info("Performance optimizer cleanup completed")

# Global performance optimizer instance
_performance_optimizer: Optional[PerformanceOptimizer] = None

async def get_performance_optimizer(tenant_id: str) -> PerformanceOptimizer:
	"""Get or create performance optimizer instance."""
	global _performance_optimizer
	
	if _performance_optimizer is None or _performance_optimizer.tenant_id != tenant_id:
		_performance_optimizer = PerformanceOptimizer(tenant_id)
		await _performance_optimizer.initialize()
	
	return _performance_optimizer

async def optimized_query(
	tenant_id: str,
	query_key: str,
	query_function: Callable,
	ttl_seconds: Optional[int] = None
) -> Any:
	"""Execute query with performance optimization."""
	optimizer = await get_performance_optimizer(tenant_id)
	return await optimizer.get_optimized(query_key, query_function, ttl_seconds)

# Performance monitoring decorator
def monitor_performance(func):
	"""Decorator to monitor function performance."""
	async def wrapper(*args, **kwargs):
		start_time = time.time()
		
		try:
			result = await func(*args, **kwargs)
			success = True
		except Exception as e:
			success = False
			raise
		finally:
			latency = (time.time() - start_time) * 1000
			
			# Record performance metrics
			# This would integrate with the performance optimizer
			logger.debug(f"Function {func.__name__} completed in {latency:.2f}ms, success={success}")
		
		return result
	
	return wrapper

if __name__ == "__main__":
	async def main():
		# Example usage
		optimizer = PerformanceOptimizer("demo_tenant")
		await optimizer.initialize()
		
		# Simulate some cache operations
		async def fetch_data():
			await asyncio.sleep(0.1)  # Simulate database query
			return {"data": "test_value", "timestamp": time.time()}
		
		# Test optimized caching
		result1 = await optimizer.get_optimized("test_key", fetch_data)
		result2 = await optimizer.get_optimized("test_key", fetch_data)  # Should hit cache
		
		print("Cache hit:", result1 == result2)
		
		# Get performance report
		report = await optimizer.get_performance_report()
		print("Performance report:", json.dumps(report, indent=2))
		
		await optimizer.cleanup()
	
	asyncio.run(main())