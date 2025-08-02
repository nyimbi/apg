"""
APG Workflow Performance Optimizer

Comprehensive performance optimization system including database query optimization,
caching strategies, connection pooling, memory management, and system monitoring.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import hashlib
import pickle
import weakref
from functools import wraps
import gc
import psutil
import threading
from collections import defaultdict, OrderedDict
import json

from pydantic import BaseModel, Field, ConfigDict
from sqlalchemy.pool import QueuePool, StaticPool
from sqlalchemy.engine import create_engine, Engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy import text, event
import redis
import aioredis
from cachetools import TTLCache, LRUCache
import pymemcache.client.base

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
	"""Performance metrics collection."""
	
	# Database metrics
	db_query_count: int = 0
	db_query_time_total: float = 0.0
	db_connection_pool_size: int = 0
	db_connection_pool_checked_out: int = 0
	
	# Cache metrics
	cache_hits: int = 0
	cache_misses: int = 0
	cache_size: int = 0
	cache_memory_usage: int = 0
	
	# Memory metrics
	memory_usage_mb: float = 0.0
	memory_peak_mb: float = 0.0
	gc_collections: int = 0
	
	# System metrics
	cpu_usage_percent: float = 0.0
	disk_io_read_mb: float = 0.0
	disk_io_write_mb: float = 0.0
	
	# Workflow metrics
	active_workflows: int = 0
	completed_workflows: int = 0
	failed_workflows: int = 0
	avg_workflow_duration: float = 0.0
	
	# API metrics
	api_requests_total: int = 0
	api_requests_per_second: float = 0.0
	api_avg_response_time: float = 0.0
	api_error_rate: float = 0.0
	
	# Timestamp
	timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class OptimizationConfig(BaseModel):
	"""Performance optimization configuration."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	# Database optimization
	db_pool_size: int = Field(default=20, ge=5, le=100, description="Database connection pool size")
	db_max_overflow: int = Field(default=30, ge=0, le=200, description="Maximum pool overflow")
	db_pool_timeout: int = Field(default=30, ge=5, le=300, description="Pool timeout in seconds")
	db_pool_recycle: int = Field(default=3600, ge=300, le=86400, description="Pool recycle time in seconds")
	enable_query_optimization: bool = Field(default=True, description="Enable query optimization")
	
	# Caching configuration
	enable_redis_cache: bool = Field(default=True, description="Enable Redis caching")
	redis_url: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")
	cache_ttl_seconds: int = Field(default=3600, ge=60, le=86400, description="Default cache TTL")
	max_cache_size: int = Field(default=10000, ge=100, le=1000000, description="Maximum cache entries")
	
	# Memory management
	enable_memory_optimization: bool = Field(default=True, description="Enable memory optimization")
	max_memory_usage_mb: int = Field(default=2048, ge=256, le=16384, description="Maximum memory usage in MB")
	gc_collection_threshold: int = Field(default=100, ge=10, le=1000, description="GC collection threshold")
	
	# Monitoring
	metrics_collection_interval: int = Field(default=60, ge=10, le=3600, description="Metrics collection interval")
	enable_performance_monitoring: bool = Field(default=True, description="Enable performance monitoring")
	
	# API optimization
	enable_request_batching: bool = Field(default=True, description="Enable request batching")
	batch_size: int = Field(default=50, ge=1, le=1000, description="Request batch size")
	request_timeout: int = Field(default=30, ge=5, le=300, description="Request timeout in seconds")

class DatabaseOptimizer:
	"""Database performance optimizer."""
	
	def __init__(self, config: OptimizationConfig):
		self.config = config
		self.engines: Dict[str, Engine] = {}
		self.session_factories: Dict[str, sessionmaker] = {}
		self.query_cache: TTLCache = TTLCache(maxsize=1000, ttl=300)
		self.query_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
			'count': 0,
			'total_time': 0.0,
			'avg_time': 0.0,
			'last_executed': None
		})
	
	def create_optimized_engine(self, database_url: str, engine_name: str = "default") -> Engine:
		"""Create an optimized database engine with connection pooling."""
		try:
			# Configure connection pooling
			engine = create_engine(
				database_url,
				poolclass=QueuePool,
				pool_size=self.config.db_pool_size,
				max_overflow=self.config.db_max_overflow,
				pool_timeout=self.config.db_pool_timeout,
				pool_recycle=self.config.db_pool_recycle,
				pool_pre_ping=True,  # Validate connections
				echo=False,  # Disable SQL logging for performance
				future=True  # Use 2.0 style
			)
			
			# Add query monitoring
			if self.config.enable_performance_monitoring:
				self._setup_query_monitoring(engine)
			
			self.engines[engine_name] = engine
			
			# Create session factory
			self.session_factories[engine_name] = scoped_session(
				sessionmaker(bind=engine, expire_on_commit=False)
			)
			
			logger.info(f"Created optimized database engine: {engine_name}")
			return engine
			
		except Exception as e:
			logger.error(f"Failed to create optimized engine: {e}")
			raise
	
	def _setup_query_monitoring(self, engine: Engine) -> None:
		"""Setup query performance monitoring."""
		
		@event.listens_for(engine, "before_cursor_execute")
		def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
			context._query_start_time = time.time()
		
		@event.listens_for(engine, "after_cursor_execute")
		def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
			total_time = time.time() - context._query_start_time
			
			# Generate query hash for tracking
			query_hash = hashlib.md5(statement.encode()).hexdigest()[:8]
			
			stats = self.query_stats[query_hash]
			stats['count'] += 1
			stats['total_time'] += total_time
			stats['avg_time'] = stats['total_time'] / stats['count']
			stats['last_executed'] = datetime.now(timezone.utc)
			stats['statement'] = statement[:200]  # Store truncated statement
			
			# Log slow queries
			if total_time > 1.0:  # Log queries taking more than 1 second
				logger.warning(f"Slow query detected: {total_time:.3f}s - {statement[:100]}...")
	
	def get_session(self, engine_name: str = "default"):
		"""Get an optimized database session."""
		if engine_name not in self.session_factories:
			raise ValueError(f"Engine {engine_name} not found")
		
		return self.session_factories[engine_name]()
	
	@asynccontextmanager
	async def get_async_session(self, engine_name: str = "default"):
		"""Get an async database session with automatic cleanup."""
		session = self.get_session(engine_name)
		try:
			yield session
			session.commit()
		except Exception:
			session.rollback()
			raise
		finally:
			session.close()
	
	def optimize_query(self, query: str, parameters: Optional[Dict] = None) -> str:
		"""Apply query optimizations."""
		if not self.config.enable_query_optimization:
			return query
		
		# Check cache first
		cache_key = hashlib.md5(f"{query}{parameters}".encode()).hexdigest()
		if cache_key in self.query_cache:
			return self.query_cache[cache_key]
		
		optimized_query = query
		
		# Add LIMIT if missing for potentially large result sets
		if ("SELECT" in query.upper() and 
			"LIMIT" not in query.upper() and 
			"COUNT(" not in query.upper()):
			optimized_query += " LIMIT 1000"
		
		# Add appropriate indexes hints (PostgreSQL specific)
		if "WHERE" in query.upper() and "EXPLAIN" not in query.upper():
			# Analyze query for potential index usage
			where_clause = query.upper().split("WHERE")[1].split("ORDER BY")[0] if "ORDER BY" in query.upper() else query.upper().split("WHERE")[1]
			
			# Common optimization patterns
			if "workflow_id" in where_clause.lower():
				optimized_query = f"/*+ INDEX(workflow_definitions_workflow_id_idx) */ {optimized_query}"
			elif "user_id" in where_clause.lower():
				optimized_query = f"/*+ INDEX(workflow_executions_user_id_idx) */ {optimized_query}"
			elif "created_at" in where_clause.lower() or "timestamp" in where_clause.lower():
				optimized_query = f"/*+ INDEX(workflow_executions_created_at_idx) */ {optimized_query}"
			elif "status" in where_clause.lower():
				optimized_query = f"/*+ INDEX(workflow_executions_status_idx) */ {optimized_query}"
		
		self.query_cache[cache_key] = optimized_query
		return optimized_query
	
	def get_performance_stats(self) -> Dict[str, Any]:
		"""Get database performance statistics."""
		stats = {
			'total_queries': sum(stat['count'] for stat in self.query_stats.values()),
			'avg_query_time': 0.0,
			'slow_queries': 0,
			'query_cache_size': len(self.query_cache),
			'engines': {}
		}
		
		# Calculate average query time
		if stats['total_queries'] > 0:
			total_time = sum(stat['total_time'] for stat in self.query_stats.values())
			stats['avg_query_time'] = total_time / stats['total_queries']
		
		# Count slow queries
		stats['slow_queries'] = sum(1 for stat in self.query_stats.values() 
									if stat['avg_time'] > 1.0)
		
		# Engine statistics
		for name, engine in self.engines.items():
			pool = engine.pool
			stats['engines'][name] = {
				'pool_size': pool.size() if hasattr(pool, 'size') else 0,
				'checked_out': pool.checkedout() if hasattr(pool, 'checkedout') else 0,
				'overflow': pool.overflow() if hasattr(pool, 'overflow') else 0,
				'invalidated': pool.invalidated() if hasattr(pool, 'invalidated') else 0
			}
		
		return stats

class CacheManager:
	"""Multi-level caching system."""
	
	def __init__(self, config: OptimizationConfig):
		self.config = config
		self.memory_cache: TTLCache = TTLCache(
			maxsize=config.max_cache_size,
			ttl=config.cache_ttl_seconds
		)
		self.redis_client: Optional[redis.Redis] = None
		self.async_redis_client: Optional[aioredis.Redis] = None
		self.cache_stats = {
			'hits': 0,
			'misses': 0,
			'sets': 0,
			'deletes': 0
		}
		
		if config.enable_redis_cache:
			self._setup_redis()
	
	def _setup_redis(self) -> None:
		"""Setup Redis connection."""
		try:
			self.redis_client = redis.from_url(
				self.config.redis_url,
				decode_responses=True,
				socket_connect_timeout=5,
				socket_timeout=5,
				retry_on_timeout=True
			)
			
			# Test connection
			self.redis_client.ping()
			logger.info("Redis cache connection established")
			
		except Exception as e:
			logger.warning(f"Failed to connect to Redis: {e}. Using memory cache only.")
			self.redis_client = None
	
	async def _setup_async_redis(self) -> None:
		"""Setup async Redis connection."""
		if not self.config.enable_redis_cache:
			return
		
		try:
			self.async_redis_client = aioredis.from_url(
				self.config.redis_url,
				decode_responses=True,
				socket_connect_timeout=5,
				socket_timeout=5,
				retry_on_timeout=True
			)
			
			# Test connection
			await self.async_redis_client.ping()
			logger.info("Async Redis cache connection established")
			
		except Exception as e:
			logger.warning(f"Failed to connect to async Redis: {e}")
			self.async_redis_client = None
	
	def get(self, key: str, default: Any = None) -> Any:
		"""Get value from cache with multi-level fallback."""
		# Try memory cache first
		if key in self.memory_cache:
			self.cache_stats['hits'] += 1
			return self.memory_cache[key]
		
		# Try Redis cache
		if self.redis_client:
			try:
				value = self.redis_client.get(key)
				if value is not None:
					# Deserialize and store in memory cache
					deserialized = pickle.loads(value.encode('latin-1'))
					self.memory_cache[key] = deserialized
					self.cache_stats['hits'] += 1
					return deserialized
			except Exception as e:
				logger.debug(f"Redis cache get error: {e}")
		
		self.cache_stats['misses'] += 1
		return default
	
	async def aget(self, key: str, default: Any = None) -> Any:
		"""Async get value from cache."""
		# Try memory cache first
		if key in self.memory_cache:
			self.cache_stats['hits'] += 1
			return self.memory_cache[key]
		
		# Try async Redis cache
		if not self.async_redis_client:
			await self._setup_async_redis()
		
		if self.async_redis_client:
			try:
				value = await self.async_redis_client.get(key)
				if value is not None:
					# Deserialize and store in memory cache
					deserialized = pickle.loads(value.encode('latin-1'))
					self.memory_cache[key] = deserialized
					self.cache_stats['hits'] += 1
					return deserialized
			except Exception as e:
				logger.debug(f"Async Redis cache get error: {e}")
		
		self.cache_stats['misses'] += 1
		return default
	
	def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
		"""Set value in cache with multi-level storage."""
		ttl = ttl or self.config.cache_ttl_seconds
		
		# Store in memory cache
		self.memory_cache[key] = value
		
		# Store in Redis cache
		if self.redis_client:
			try:
				serialized = pickle.dumps(value).decode('latin-1')
				self.redis_client.setex(key, ttl, serialized)
			except Exception as e:
				logger.debug(f"Redis cache set error: {e}")
		
		self.cache_stats['sets'] += 1
	
	async def aset(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
		"""Async set value in cache."""
		ttl = ttl or self.config.cache_ttl_seconds
		
		# Store in memory cache
		self.memory_cache[key] = value
		
		# Store in async Redis cache
		if not self.async_redis_client:
			await self._setup_async_redis()
		
		if self.async_redis_client:
			try:
				serialized = pickle.dumps(value).decode('latin-1')
				await self.async_redis_client.setex(key, ttl, serialized)
			except Exception as e:
				logger.debug(f"Async Redis cache set error: {e}")
		
		self.cache_stats['sets'] += 1
	
	def delete(self, key: str) -> None:
		"""Delete value from cache."""
		# Remove from memory cache
		self.memory_cache.pop(key, None)
		
		# Remove from Redis cache
		if self.redis_client:
			try:
				self.redis_client.delete(key)
			except Exception as e:
				logger.debug(f"Redis cache delete error: {e}")
		
		self.cache_stats['deletes'] += 1
	
	async def adelete(self, key: str) -> None:
		"""Async delete value from cache."""
		# Remove from memory cache
		self.memory_cache.pop(key, None)
		
		# Remove from async Redis cache
		if not self.async_redis_client:
			await self._setup_async_redis()
		
		if self.async_redis_client:
			try:
				await self.async_redis_client.delete(key)
			except Exception as e:
				logger.debug(f"Async Redis cache delete error: {e}")
		
		self.cache_stats['deletes'] += 1
	
	def clear(self) -> None:
		"""Clear all caches."""
		self.memory_cache.clear()
		
		if self.redis_client:
			try:
				self.redis_client.flushdb()
			except Exception as e:
				logger.debug(f"Redis cache clear error: {e}")
	
	def get_stats(self) -> Dict[str, Any]:
		"""Get cache performance statistics."""
		total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
		hit_rate = (self.cache_stats['hits'] / total_requests * 100) if total_requests > 0 else 0
		
		return {
			'memory_cache_size': len(self.memory_cache),
			'memory_cache_maxsize': self.memory_cache.maxsize,
			'hit_rate_percent': hit_rate,
			'total_requests': total_requests,
			**self.cache_stats,
			'redis_connected': self.redis_client is not None
		}

class MemoryOptimizer:
	"""Memory management and optimization."""
	
	def __init__(self, config: OptimizationConfig):
		self.config = config
		self.memory_stats = {
			'peak_usage_mb': 0.0,
			'gc_collections': 0,
			'object_count': 0
		}
		self.weak_refs: Dict[str, weakref.WeakSet] = defaultdict(weakref.WeakSet)
		
		if config.enable_memory_optimization:
			self._setup_memory_monitoring()
	
	def _setup_memory_monitoring(self) -> None:
		"""Setup memory monitoring and automatic cleanup."""
		# Setup garbage collection thresholds
		gc.set_threshold(
			self.config.gc_collection_threshold,
			self.config.gc_collection_threshold // 10,
			self.config.gc_collection_threshold // 20
		)
		
		# Start memory monitoring thread
		self.monitoring_thread = threading.Thread(target=self._memory_monitor_loop, daemon=True)
		self.monitoring_thread.start()
	
	def _memory_monitor_loop(self) -> None:
		"""Background memory monitoring loop."""
		while True:
			try:
				# Get current memory usage
				process = psutil.Process()
				memory_mb = process.memory_info().rss / 1024 / 1024
				
				# Update peak usage
				if memory_mb > self.memory_stats['peak_usage_mb']:
					self.memory_stats['peak_usage_mb'] = memory_mb
				
				# Check if memory usage is too high
				if memory_mb > self.config.max_memory_usage_mb:
					logger.warning(f"High memory usage detected: {memory_mb:.1f}MB")
					self.force_cleanup()
				
				# Update object count
				self.memory_stats['object_count'] = len(gc.get_objects())
				
				time.sleep(60)  # Check every minute
				
			except Exception as e:
				logger.error(f"Memory monitoring error: {e}")
				time.sleep(60)
	
	def register_object(self, obj: Any, category: str = "default") -> None:
		"""Register object for memory tracking."""
		try:
			self.weak_refs[category].add(obj)
		except TypeError:
			# Object not weakly referenceable
			pass
	
	def force_cleanup(self) -> int:
		"""Force garbage collection and cleanup."""
		# Clear weak references to dead objects
		for category in self.weak_refs:
			# WeakSet automatically removes dead references
			pass
		
		# Force garbage collection
		collected = gc.collect()
		self.memory_stats['gc_collections'] += 1
		
		logger.info(f"Forced cleanup collected {collected} objects")
		return collected
	
	def get_memory_usage(self) -> Dict[str, Any]:
		"""Get current memory usage statistics."""
		try:
			process = psutil.Process()
			memory_info = process.memory_info()
			
			return {
				'rss_mb': memory_info.rss / 1024 / 1024,
				'vms_mb': memory_info.vms / 1024 / 1024,
				'peak_usage_mb': self.memory_stats['peak_usage_mb'],
				'gc_collections': self.memory_stats['gc_collections'],
				'object_count': len(gc.get_objects()),
				'tracked_objects': {
					category: len(refs) for category, refs in self.weak_refs.items()
				}
			}
		except Exception as e:
			logger.error(f"Failed to get memory usage: {e}")
			return {}

class PerformanceOptimizer:
	"""Main performance optimization coordinator."""
	
	def __init__(self, config: OptimizationConfig):
		self.config = config
		self.db_optimizer = DatabaseOptimizer(config)
		self.cache_manager = CacheManager(config)
		self.memory_optimizer = MemoryOptimizer(config)
		
		self.metrics_history: List[PerformanceMetrics] = []
		self.monitoring_active = False
		self.monitoring_task: Optional[asyncio.Task] = None
		
		logger.info("Performance optimizer initialized")
	
	async def start_monitoring(self) -> None:
		"""Start performance monitoring."""
		if self.monitoring_active:
			return
		
		self.monitoring_active = True
		self.monitoring_task = asyncio.create_task(self._monitoring_loop())
		logger.info("Performance monitoring started")
	
	async def stop_monitoring(self) -> None:
		"""Stop performance monitoring."""
		self.monitoring_active = False
		if self.monitoring_task:
			self.monitoring_task.cancel()
			try:
				await self.monitoring_task
			except asyncio.CancelledError:
				pass
		logger.info("Performance monitoring stopped")
	
	async def _monitoring_loop(self) -> None:
		"""Main monitoring loop."""
		while self.monitoring_active:
			try:
				metrics = await self.collect_metrics()
				self.metrics_history.append(metrics)
				
				# Keep only last 24 hours of metrics (assuming 1-minute intervals)
				if len(self.metrics_history) > 1440:
					self.metrics_history = self.metrics_history[-1440:]
				
				# Check for performance issues
				await self._check_performance_alerts(metrics)
				
				await asyncio.sleep(self.config.metrics_collection_interval)
				
			except asyncio.CancelledError:
				break
			except Exception as e:
				logger.error(f"Performance monitoring error: {e}")
				await asyncio.sleep(60)
	
	async def collect_metrics(self) -> PerformanceMetrics:
		"""Collect comprehensive performance metrics."""
		metrics = PerformanceMetrics()
		
		try:
			# Database metrics
			db_stats = self.db_optimizer.get_performance_stats()
			metrics.db_query_count = db_stats.get('total_queries', 0)
			metrics.db_query_time_total = db_stats.get('avg_query_time', 0.0)
			
			# Cache metrics
			cache_stats = self.cache_manager.get_stats()
			metrics.cache_hits = cache_stats.get('hits', 0)
			metrics.cache_misses = cache_stats.get('misses', 0)
			metrics.cache_size = cache_stats.get('memory_cache_size', 0)
			
			# Memory metrics
			memory_stats = self.memory_optimizer.get_memory_usage()
			metrics.memory_usage_mb = memory_stats.get('rss_mb', 0.0)
			metrics.memory_peak_mb = memory_stats.get('peak_usage_mb', 0.0)
			metrics.gc_collections = memory_stats.get('gc_collections', 0)
			
			# System metrics
			try:
				process = psutil.Process()
				metrics.cpu_usage_percent = process.cpu_percent()
				
				io_counters = process.io_counters()
				metrics.disk_io_read_mb = io_counters.read_bytes / 1024 / 1024
				metrics.disk_io_write_mb = io_counters.write_bytes / 1024 / 1024
			except Exception as e:
				logger.debug(f"Failed to collect system metrics: {e}")
			
		except Exception as e:
			logger.error(f"Failed to collect performance metrics: {e}")
		
		return metrics
	
	async def _check_performance_alerts(self, metrics: PerformanceMetrics) -> None:
		"""Check for performance issues and alert."""
		alerts = []
		
		# Memory usage alert
		if metrics.memory_usage_mb > self.config.max_memory_usage_mb * 0.9:
			alerts.append(f"High memory usage: {metrics.memory_usage_mb:.1f}MB")
		
		# CPU usage alert
		if metrics.cpu_usage_percent > 90:
			alerts.append(f"High CPU usage: {metrics.cpu_usage_percent:.1f}%")
		
		# Cache hit rate alert
		total_requests = metrics.cache_hits + metrics.cache_misses
		if total_requests > 100:
			hit_rate = metrics.cache_hits / total_requests * 100
			if hit_rate < 50:
				alerts.append(f"Low cache hit rate: {hit_rate:.1f}%")
		
		# Database performance alert
		if metrics.db_query_time_total > 5.0:
			alerts.append(f"Slow database queries: {metrics.db_query_time_total:.3f}s avg")
		
		if alerts:
			logger.warning(f"Performance alerts: {'; '.join(alerts)}")
	
	def get_performance_report(self) -> Dict[str, Any]:
		"""Generate comprehensive performance report."""
		if not self.metrics_history:
			return {"error": "No metrics available"}
		
		latest = self.metrics_history[-1]
		
		# Calculate trends if we have enough data
		trends = {}
		if len(self.metrics_history) >= 2:
			previous = self.metrics_history[-2]
			trends = {
				'memory_trend': latest.memory_usage_mb - previous.memory_usage_mb,
				'cpu_trend': latest.cpu_usage_percent - previous.cpu_usage_percent,
				'query_count_trend': latest.db_query_count - previous.db_query_count
			}
		
		return {
			'current_metrics': {
				'memory_usage_mb': latest.memory_usage_mb,
				'memory_peak_mb': latest.memory_peak_mb,
				'cpu_usage_percent': latest.cpu_usage_percent,
				'db_query_count': latest.db_query_count,
				'cache_hit_rate': (latest.cache_hits / (latest.cache_hits + latest.cache_misses) * 100) 
								 if (latest.cache_hits + latest.cache_misses) > 0 else 0,
				'active_workflows': latest.active_workflows,
				'timestamp': latest.timestamp.isoformat()
			},
			'trends': trends,
			'database_stats': self.db_optimizer.get_performance_stats(),
			'cache_stats': self.cache_manager.get_stats(),
			'memory_stats': self.memory_optimizer.get_memory_usage(),
			'metrics_count': len(self.metrics_history)
		}
	
	def cache_decorator(self, ttl: int = None, key_prefix: str = ""):
		"""Decorator for caching function results."""
		def decorator(func: Callable) -> Callable:
			@wraps(func)
			async def async_wrapper(*args, **kwargs):
				# Generate cache key
				key = f"{key_prefix}{func.__name__}:{hash(str(args) + str(kwargs))}"
				
				# Try to get from cache
				result = await self.cache_manager.aget(key)
				if result is not None:
					return result
				
				# Execute function and cache result
				result = await func(*args, **kwargs)
				await self.cache_manager.aset(key, result, ttl)
				return result
			
			@wraps(func)
			def sync_wrapper(*args, **kwargs):
				# Generate cache key
				key = f"{key_prefix}{func.__name__}:{hash(str(args) + str(kwargs))}"
				
				# Try to get from cache
				result = self.cache_manager.get(key)
				if result is not None:
					return result
				
				# Execute function and cache result
				result = func(*args, **kwargs)
				self.cache_manager.set(key, result, ttl)
				return result
			
			# Return appropriate wrapper based on function type
			if asyncio.iscoroutinefunction(func):
				return async_wrapper
			else:
				return sync_wrapper
		
		return decorator
	
	def optimize_async_function(self, func: Callable) -> Callable:
		"""Decorator to optimize async function performance."""
		@wraps(func)
		async def wrapper(*args, **kwargs):
			start_time = time.time()
			
			try:
				# Register function call for monitoring
				self.memory_optimizer.register_object(func, "async_functions")
				
				# Execute function
				result = await func(*args, **kwargs)
				
				# Log slow functions
				execution_time = time.time() - start_time
				if execution_time > 5.0:
					logger.warning(f"Slow async function: {func.__name__} took {execution_time:.3f}s")
				
				return result
				
			except Exception as e:
				logger.error(f"Error in optimized function {func.__name__}: {e}")
				raise
		
		return wrapper

# Global performance optimizer instance
_performance_optimizer: Optional[PerformanceOptimizer] = None

def get_performance_optimizer() -> PerformanceOptimizer:
	"""Get the global performance optimizer instance."""
	global _performance_optimizer
	if _performance_optimizer is None:
		config = OptimizationConfig()
		_performance_optimizer = PerformanceOptimizer(config)
	return _performance_optimizer

def performance_cache(ttl: int = 3600, key_prefix: str = ""):
	"""Decorator for performance caching."""
	optimizer = get_performance_optimizer()
	return optimizer.cache_decorator(ttl=ttl, key_prefix=key_prefix)

def optimize_performance(func: Callable) -> Callable:
	"""Decorator for general performance optimization."""
	optimizer = get_performance_optimizer()
	return optimizer.optimize_async_function(func)