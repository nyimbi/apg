"""
APG Connection Management Performance Optimization
Advanced caching, connection pooling, and performance enhancements

Author: Nyimbi Odero
Company: Datacraft
Copyright: © 2025
"""

import asyncio
import time
import hashlib
import json
import pickle
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Union, Callable, TypeVar, Generic, Tuple
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps, lru_cache
from contextlib import asynccontextmanager
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from collections import defaultdict, deque
import weakref

# Redis imports for distributed caching
try:
    import redis.asyncio as aioredis
    import redis
    from redis.connection import ConnectionPool
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis not available. Using in-memory caching only.")

# SQLAlchemy imports for connection pooling
try:
    from sqlalchemy import text
    from sqlalchemy.pool import QueuePool, StaticPool, NullPool
    from sqlalchemy.engine import create_engine
    from sqlalchemy.orm import sessionmaker
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

from .error_handling import APGError, ErrorContext
from .monitoring import global_metrics_collector, monitor_performance

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CacheStrategy(str, Enum):
    """Cache invalidation strategies"""
    TTL = "ttl"                    # Time-to-live
    LRU = "lru"                   # Least Recently Used
    LFU = "lfu"                   # Least Frequently Used
    WRITE_THROUGH = "write_through"  # Write to cache and storage
    WRITE_BACK = "write_back"        # Write to cache, async to storage
    READ_THROUGH = "read_through"    # Read from cache, fallback to storage


class PerformanceLevel(str, Enum):
    """Performance optimization levels"""
    BASIC = "basic"
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"
    EXTREME = "extreme"


@dataclass
class CacheConfig:
    """Cache configuration settings"""
    strategy: CacheStrategy = CacheStrategy.TTL
    ttl_seconds: int = 3600
    max_size: int = 10000
    enable_compression: bool = True
    enable_encryption: bool = False
    namespace: str = "apg:conn"
    serializer: str = "json"  # json, pickle, msgpack


@dataclass
class PerformanceConfig:
    """Performance optimization configuration"""
    level: PerformanceLevel = PerformanceLevel.STANDARD
    enable_async_processing: bool = True
    max_worker_threads: int = 10
    max_worker_processes: int = 4
    enable_connection_pooling: bool = True
    pool_size: int = 20
    pool_overflow: int = 30
    enable_query_caching: bool = True
    enable_result_streaming: bool = True
    enable_batch_processing: bool = True
    batch_size: int = 1000
    enable_prefetching: bool = True
    prefetch_size: int = 100


class CacheEntry(Generic[T]):
    """Cache entry with metadata"""

    def __init__(self, value: T, created_at: datetime = None, ttl: int = None,
                 access_count: int = 0, last_accessed: datetime = None):
        self.value = value
        self.created_at = created_at or datetime.now(timezone.utc)
        self.ttl = ttl
        self.access_count = access_count
        self.last_accessed = last_accessed or self.created_at
        self.size = self._calculate_size(value)

    def _calculate_size(self, value: Any) -> int:
        """Estimate memory size of cached value"""
        try:
            if isinstance(value, (str, bytes)):
                return len(value)
            elif isinstance(value, (dict, list, tuple)):
                return len(json.dumps(value, default=str).encode())
            else:
                return len(pickle.dumps(value))
        except Exception:
            return 1024  # Default size estimate

    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        if not self.ttl:
            return False
        age = (datetime.now(timezone.utc) - self.created_at).total_seconds()
        return age > self.ttl

    def touch(self):
        """Update access metadata"""
        self.access_count += 1
        self.last_accessed = datetime.now(timezone.utc)


class InMemoryCache(Generic[T]):
    """High-performance in-memory cache with multiple eviction strategies"""

    def __init__(self, config: CacheConfig):
        self.config = config
        self._cache: Dict[str, CacheEntry[T]] = {}
        self._access_order: deque = deque()  # For LRU
        self._access_count: Dict[str, int] = defaultdict(int)  # For LFU
        self._lock = asyncio.Lock()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'memory_usage': 0
        }

    async def get(self, key: str) -> Optional[T]:
        """Get value from cache"""
        async with self._lock:
            if key not in self._cache:
                self._stats['misses'] += 1
                return None

            entry = self._cache[key]

            # Check expiration
            if entry.is_expired():
                await self._evict_key(key)
                self._stats['misses'] += 1
                return None

            # Update access metadata
            entry.touch()
            self._update_access_order(key)
            self._stats['hits'] += 1

            return entry.value

    async def set(self, key: str, value: T, ttl: Optional[int] = None) -> None:
        """Set value in cache"""
        async with self._lock:
            entry = CacheEntry(value, ttl=ttl or self.config.ttl_seconds)

            # Check if we need to evict
            if len(self._cache) >= self.config.max_size and key not in self._cache:
                await self._evict_entries(1)

            self._cache[key] = entry
            self._update_access_order(key)
            self._update_memory_usage()

    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        async with self._lock:
            if key in self._cache:
                await self._evict_key(key)
                return True
            return False

    async def clear(self) -> None:
        """Clear all cache entries"""
        async with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._access_count.clear()
            self._stats['memory_usage'] = 0

    async def _evict_key(self, key: str) -> None:
        """Evict a specific key"""
        if key in self._cache:
            del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)
            if key in self._access_count:
                del self._access_count[key]
            self._stats['evictions'] += 1
            self._update_memory_usage()

    async def _evict_entries(self, count: int) -> None:
        """Evict entries based on strategy"""
        if self.config.strategy == CacheStrategy.LRU:
            await self._evict_lru(count)
        elif self.config.strategy == CacheStrategy.LFU:
            await self._evict_lfu(count)
        else:
            await self._evict_ttl(count)

    async def _evict_lru(self, count: int) -> None:
        """Evict least recently used entries"""
        for _ in range(min(count, len(self._access_order))):
            if self._access_order:
                key = self._access_order.popleft()
                if key in self._cache:
                    del self._cache[key]
                    self._stats['evictions'] += 1

    async def _evict_lfu(self, count: int) -> None:
        """Evict least frequently used entries"""
        if not self._access_count:
            return

        # Sort by access count and evict least used
        sorted_keys = sorted(self._access_count.items(), key=lambda x: x[1])
        for i in range(min(count, len(sorted_keys))):
            key = sorted_keys[i][0]
            await self._evict_key(key)

    async def _evict_ttl(self, count: int) -> None:
        """Evict expired entries first, then oldest"""
        # First evict expired entries
        expired_keys = []
        for key, entry in self._cache.items():
            if entry.is_expired():
                expired_keys.append(key)

        for key in expired_keys[:count]:
            await self._evict_key(key)

        # If we still need to evict more, evict oldest
        remaining = count - len(expired_keys)
        if remaining > 0:
            oldest_keys = sorted(self._cache.items(),
                               key=lambda x: x[1].created_at)[:remaining]
            for key, _ in oldest_keys:
                await self._evict_key(key)

    def _update_access_order(self, key: str) -> None:
        """Update LRU access order"""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
        self._access_count[key] += 1

    def _update_memory_usage(self) -> None:
        """Update memory usage statistics"""
        self._stats['memory_usage'] = sum(
            entry.size for entry in self._cache.values()
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self._stats['hits'] + self._stats['misses']
        hit_rate = (self._stats['hits'] / total_requests * 100) if total_requests > 0 else 0

        return {
            **self._stats,
            'hit_rate_percent': round(hit_rate, 2),
            'cache_size': len(self._cache),
            'max_size': self.config.max_size
        }


class DistributedCache:
    """Redis-based distributed cache for multi-instance deployments"""

    def __init__(self, config: CacheConfig, redis_url: str = None):
        self.config = config
        self.redis_url = redis_url or "redis://localhost:6379/0"
        self._redis: Optional[aioredis.Redis] = None
        self._connection_pool: Optional[aioredis.ConnectionPool] = None

        if not REDIS_AVAILABLE:
            raise APGError(
                message="Redis not available for distributed caching",
                context=ErrorContext(tenant_id="system", operation="init_distributed_cache")
            )

    async def connect(self):
        """Initialize Redis connection"""
        try:
            self._connection_pool = aioredis.ConnectionPool.from_url(
                self.redis_url,
                max_connections=20,
                retry_on_timeout=True,
                health_check_interval=30
            )
            self._redis = aioredis.Redis(connection_pool=self._connection_pool)

            # Test connection
            await self._redis.ping()
            logger.info("Redis distributed cache connected successfully")

        except Exception as e:
            raise APGError(
                message=f"Failed to connect to Redis: {str(e)}",
                context=ErrorContext(tenant_id="system", operation="connect_redis"),
                cause=e
            )

    async def get(self, key: str) -> Optional[Any]:
        """Get value from distributed cache"""
        try:
            if not self._redis:
                await self.connect()

            cache_key = f"{self.config.namespace}:{key}"
            data = await self._redis.get(cache_key)

            if data is None:
                return None

            # Deserialize data
            if self.config.serializer == "json":
                return json.loads(data)
            elif self.config.serializer == "pickle":
                return pickle.loads(data)
            else:
                return data.decode() if isinstance(data, bytes) else data

        except Exception as e:
            logger.error(f"Error getting from distributed cache: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in distributed cache"""
        try:
            if not self._redis:
                await self.connect()

            cache_key = f"{self.config.namespace}:{key}"

            # Serialize data
            if self.config.serializer == "json":
                data = json.dumps(value, default=str)
            elif self.config.serializer == "pickle":
                data = pickle.dumps(value)
            else:
                data = str(value)

            ttl_seconds = ttl or self.config.ttl_seconds
            await self._redis.setex(cache_key, ttl_seconds, data)

        except Exception as e:
            logger.error(f"Error setting in distributed cache: {e}")

    async def delete(self, key: str) -> bool:
        """Delete key from distributed cache"""
        try:
            if not self._redis:
                await self.connect()

            cache_key = f"{self.config.namespace}:{key}"
            result = await self._redis.delete(cache_key)
            return result > 0

        except Exception as e:
            logger.error(f"Error deleting from distributed cache: {e}")
            return False

    async def clear_namespace(self) -> None:
        """Clear all keys in namespace"""
        try:
            if not self._redis:
                await self.connect()

            pattern = f"{self.config.namespace}:*"
            keys = await self._redis.keys(pattern)

            if keys:
                await self._redis.delete(*keys)

        except Exception as e:
            logger.error(f"Error clearing cache namespace: {e}")

    async def close(self):
        """Close Redis connection"""
        if self._connection_pool:
            await self._connection_pool.disconnect()


class CacheManager:
    """Unified cache manager with fallback strategies"""

    def __init__(self, config: CacheConfig, redis_url: str = None):
        self.config = config
        self.memory_cache = InMemoryCache[Any](config)
        self.distributed_cache: Optional[DistributedCache] = None

        # Initialize distributed cache if Redis is available
        if REDIS_AVAILABLE and redis_url:
            try:
                self.distributed_cache = DistributedCache(config, redis_url)
            except Exception as e:
                logger.warning(f"Failed to initialize distributed cache: {e}")

    async def get(self, key: str) -> Optional[Any]:
        """Get with multi-level cache strategy"""
        # Try memory cache first
        value = await self.memory_cache.get(key)
        if value is not None:
            return value

        # Try distributed cache
        if self.distributed_cache:
            value = await self.distributed_cache.get(key)
            if value is not None:
                # Populate memory cache
                await self.memory_cache.set(key, value)
                return value

        return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set in all available caches"""
        # Set in memory cache
        await self.memory_cache.set(key, value, ttl)

        # Set in distributed cache
        if self.distributed_cache:
            await self.distributed_cache.set(key, value, ttl)

    async def delete(self, key: str) -> bool:
        """Delete from all caches"""
        memory_deleted = await self.memory_cache.delete(key)
        distributed_deleted = True

        if self.distributed_cache:
            distributed_deleted = await self.distributed_cache.delete(key)

        return memory_deleted or distributed_deleted

    async def clear(self) -> None:
        """Clear all caches"""
        await self.memory_cache.clear()

        if self.distributed_cache:
            await self.distributed_cache.clear_namespace()

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        return {
            'memory_cache': self.memory_cache.get_stats(),
            'distributed_cache_available': self.distributed_cache is not None
        }


class ConnectionPoolManager:
    """Advanced database connection pooling"""

    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.pools: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def get_pool(self, connection_string: str, pool_name: str = "default") -> Any:
        """Get or create connection pool"""
        with self._lock:
            if pool_name not in self.pools:
                self.pools[pool_name] = self._create_pool(connection_string)
            return self.pools[pool_name]

    def _create_pool(self, connection_string: str) -> Any:
        """Create optimized connection pool"""
        if not SQLALCHEMY_AVAILABLE:
            raise APGError(
                message="SQLAlchemy not available for connection pooling",
                context=ErrorContext(tenant_id="system", operation="create_connection_pool")
            )

        # Configure pool based on performance level
        if self.config.level == PerformanceLevel.EXTREME:
            pool_class = QueuePool
            pool_size = self.config.pool_size * 2
            max_overflow = self.config.pool_overflow * 2
            pool_pre_ping = True
            pool_recycle = 3600
        elif self.config.level == PerformanceLevel.AGGRESSIVE:
            pool_class = QueuePool
            pool_size = self.config.pool_size
            max_overflow = self.config.pool_overflow
            pool_pre_ping = True
            pool_recycle = 7200
        else:
            pool_class = QueuePool
            pool_size = max(5, self.config.pool_size // 2)
            max_overflow = max(10, self.config.pool_overflow // 2)
            pool_pre_ping = False
            pool_recycle = 14400

        engine = create_engine(
            connection_string,
            poolclass=pool_class,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=pool_pre_ping,
            pool_recycle=pool_recycle,
            echo=False,  # Set to True for SQL debugging
            future=True
        )

        return sessionmaker(bind=engine)

    def get_pool_stats(self, pool_name: str = "default") -> Dict[str, Any]:
        """Get connection pool statistics"""
        if pool_name not in self.pools:
            return {}

        pool = self.pools[pool_name].get_bind().pool
        return {
            'size': pool.size(),
            'checked_in': pool.checkedin(),
            'checked_out': pool.checkedout(),
            'overflow': pool.overflow(),
            'invalid': pool.invalid()
        }


class AsyncTaskManager:
    """Advanced async task and worker management"""

    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.thread_pool = ThreadPoolExecutor(max_workers=config.max_worker_threads)
        self.process_pool = ProcessPoolExecutor(max_workers=config.max_worker_processes) if config.max_worker_processes > 0 else None
        self.task_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)
        self.result_cache: Dict[str, Any] = {}
        self._workers_running = False

    async def start_workers(self):
        """Start background task workers"""
        if self._workers_running:
            return

        self._workers_running = True

        # Start async task workers
        for i in range(self.config.max_worker_threads):
            asyncio.create_task(self._async_worker(f"worker-{i}"))

        logger.info(f"Started {self.config.max_worker_threads} async task workers")

    async def _async_worker(self, worker_name: str):
        """Background async task worker"""
        while self._workers_running:
            try:
                # Get task from queue with timeout
                task_data = await asyncio.wait_for(
                    self.task_queue.get(),
                    timeout=1.0
                )

                task_id = task_data['id']
                func = task_data['func']
                args = task_data.get('args', ())
                kwargs = task_data.get('kwargs', {})

                # Execute task
                start_time = time.time()
                try:
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        result = func(*args, **kwargs)

                    self.result_cache[task_id] = {
                        'status': 'completed',
                        'result': result,
                        'duration': time.time() - start_time
                    }

                except Exception as e:
                    self.result_cache[task_id] = {
                        'status': 'failed',
                        'error': str(e),
                        'duration': time.time() - start_time
                    }

                self.task_queue.task_done()

            except asyncio.TimeoutError:
                # No tasks available, continue
                continue
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")

    async def submit_task(self, func: Callable, *args, **kwargs) -> str:
        """Submit async task for background processing"""
        task_id = hashlib.md5(f"{func.__name__}:{time.time()}".encode()).hexdigest()

        task_data = {
            'id': task_id,
            'func': func,
            'args': args,
            'kwargs': kwargs,
            'submitted_at': time.time()
        }

        await self.task_queue.put(task_data)
        return task_id

    def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task result by ID"""
        return self.result_cache.get(task_id)

    async def run_in_thread(self, func: Callable, *args, **kwargs) -> Any:
        """Run CPU-intensive task in thread pool"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_pool, func, *args, **kwargs)

    async def run_in_process(self, func: Callable, *args, **kwargs) -> Any:
        """Run CPU-intensive task in process pool"""
        if not self.process_pool:
            raise APGError(
                message="Process pool not available",
                context=ErrorContext(tenant_id="system", operation="run_in_process")
            )

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.process_pool, func, *args, **kwargs)

    async def stop_workers(self):
        """Stop background workers"""
        self._workers_running = False

        # Wait for queue to be empty
        await self.task_queue.join()

        # Shutdown executors
        self.thread_pool.shutdown(wait=True)
        if self.process_pool:
            self.process_pool.shutdown(wait=True)


class PerformanceOptimizer:
    """Main performance optimization coordinator"""

    def __init__(self,
                 cache_config: CacheConfig = None,
                 perf_config: PerformanceConfig = None,
                 redis_url: str = None):

        self.cache_config = cache_config or CacheConfig()
        self.perf_config = perf_config or PerformanceConfig()

        # Initialize components
        self.cache_manager = CacheManager(self.cache_config, redis_url)
        self.connection_pool_manager = ConnectionPoolManager(self.perf_config)
        self.task_manager = AsyncTaskManager(self.perf_config)

        # Performance tracking
        self._performance_metrics = {
            'cache_hits': 0,
            'cache_misses': 0,
            'async_tasks': 0,
            'connection_pool_usage': 0,
            'query_executions': 0,
            'query_failures': 0,
            'query_skips': 0
        }
        self.query_executors: Dict[str, Callable[[str, Optional[Dict[str, Any]]], Any]] = {}

    async def initialize(self):
        """Initialize performance components"""
        # Start async task workers
        if self.perf_config.enable_async_processing:
            await self.task_manager.start_workers()

        # Connect distributed cache
        if self.cache_manager.distributed_cache:
            await self.cache_manager.distributed_cache.connect()

        logger.info(f"Performance optimizer initialized with level: {self.perf_config.level.value}")

    async def shutdown(self):
        """Shutdown performance components"""
        await self.task_manager.stop_workers()

        if self.cache_manager.distributed_cache:
            await self.cache_manager.distributed_cache.close()

    # Caching decorators and utilities

    def cached(self, ttl: int = None, key_func: Callable = None):
        """Decorator for caching function results"""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = f"{func.__module__}.{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"

                # Try to get from cache
                cached_result = await self.cache_manager.get(cache_key)
                if cached_result is not None:
                    self._performance_metrics['cache_hits'] += 1
                    return cached_result

                # Execute function
                result = await func(*args, **kwargs)

                # Cache result
                await self.cache_manager.set(cache_key, result, ttl)
                self._performance_metrics['cache_misses'] += 1

                return result

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                # For sync functions, use a simplified caching approach
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = f"{func.__module__}.{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"

                # Simple in-memory cache for sync functions
                if not hasattr(sync_wrapper, '_cache'):
                    sync_wrapper._cache = {}

                if cache_key in sync_wrapper._cache:
                    entry_time, cached_result = sync_wrapper._cache[cache_key]
                    if time.time() - entry_time < (ttl or self.cache_config.ttl_seconds):
                        return cached_result

                result = func(*args, **kwargs)
                sync_wrapper._cache[cache_key] = (time.time(), result)

                return result

            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator

    def batch_process(self, batch_size: int = None):
        """Decorator for batch processing"""
        def decorator(func):
            @wraps(func)
            async def wrapper(items, *args, **kwargs):
                size = batch_size or self.perf_config.batch_size
                results = []

                for i in range(0, len(items), size):
                    batch = items[i:i + size]
                    batch_result = await func(batch, *args, **kwargs)
                    results.extend(batch_result if isinstance(batch_result, list) else [batch_result])

                return results
            return wrapper
        return decorator

    @asynccontextmanager
    async def database_session(self, connection_string: str, pool_name: str = "default"):
        """Context manager for database sessions with connection pooling"""
        if self.perf_config.enable_connection_pooling:
            session_factory = self.connection_pool_manager.get_pool(connection_string, pool_name)
            session = session_factory()
        else:
            # Fallback to direct connection
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker
            engine = create_engine(connection_string)
            session_factory = sessionmaker(bind=engine)
            session = session_factory()

        try:
            self._performance_metrics['connection_pool_usage'] += 1
            yield session
        finally:
            session.close()

    async def prefetch_data(self, keys: List[str], fetch_func: Callable) -> Dict[str, Any]:
        """Prefetch and cache data for multiple keys"""
        if not self.perf_config.enable_prefetching:
            return {}

        results = {}
        uncached_keys = []

        # Check cache for existing data
        for key in keys:
            cached_value = await self.cache_manager.get(key)
            if cached_value is not None:
                results[key] = cached_value
            else:
                uncached_keys.append(key)

        # Fetch uncached data in batches
        if uncached_keys:
            batch_size = self.perf_config.prefetch_size
            for i in range(0, len(uncached_keys), batch_size):
                batch_keys = uncached_keys[i:i + batch_size]

                # Fetch data for batch
                if asyncio.iscoroutinefunction(fetch_func):
                    batch_data = await fetch_func(batch_keys)
                else:
                    batch_data = await self.task_manager.run_in_thread(fetch_func, batch_keys)

                # Cache and add to results
                for key, value in batch_data.items():
                    await self.cache_manager.set(key, value)
                    results[key] = value

        return results

    def register_query_executor(
        self,
        name: str,
        executor: Callable[[str, Optional[Dict[str, Any]]], Any]
    ) -> None:
        """Register a callable used by query optimization to execute real queries."""
        if not name:
            raise ValueError("Query executor name is required")
        if not callable(executor):
            raise ValueError("Query executor must be callable")

        self.query_executors[name] = executor

    def unregister_query_executor(self, name: str) -> bool:
        """Remove a registered query executor."""
        return self.query_executors.pop(name, None) is not None

    def _normalize_query_params(self, params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Return a stable mapping for query parameters."""
        return dict(params or {})

    def _make_query_cache_key(
        self,
        query: str,
        params: Optional[Dict[str, Any]],
        executor_name: str,
        pool_name: Optional[str]
    ) -> str:
        """Build a deterministic cache key for query, params, and execution route."""
        payload = {
            'query': query,
            'params': self._normalize_query_params(params),
            'executor_name': executor_name,
            'pool_name': pool_name
        }
        encoded = json.dumps(payload, sort_keys=True, default=str, separators=(',', ':'))
        return f"query:{hashlib.sha256(encoded.encode()).hexdigest()}"

    def _resolve_query_executor(
        self,
        executor: Optional[Callable[[str, Optional[Dict[str, Any]]], Any]],
        executor_name: str
    ) -> Tuple[str, Optional[Callable[[str, Optional[Dict[str, Any]]], Any]], Optional[str]]:
        """Resolve an inline or registered executor for query execution."""
        if executor is not None:
            if not callable(executor):
                return "inline", None, "Inline query executor is not callable"
            return "inline", executor, None

        if executor_name in self.query_executors:
            return executor_name, self.query_executors[executor_name], None

        if executor_name != "default":
            return executor_name, None, f"Query executor '{executor_name}' is not registered"

        return executor_name, None, "No query executor registered"

    async def _execute_query_with_executor(
        self,
        executor: Callable[[str, Optional[Dict[str, Any]]], Any],
        query: str,
        params: Optional[Dict[str, Any]]
    ) -> Any:
        """Execute a query through a caller-provided async or sync executor."""
        if asyncio.iscoroutinefunction(executor):
            return await executor(query, params)
        return await self.task_manager.run_in_thread(executor, query, params)

    async def _execute_query_with_pool(
        self,
        query: str,
        params: Optional[Dict[str, Any]],
        pool_name: str
    ) -> Optional[Dict[str, Any]]:
        """Execute a SQLAlchemy-backed query through an existing named pool."""
        if not SQLALCHEMY_AVAILABLE:
            return None
        if pool_name not in self.connection_pool_manager.pools:
            return None

        return await self.task_manager.run_in_thread(
            self._execute_query_with_pool_sync,
            query,
            params,
            pool_name
        )

    def _execute_query_with_pool_sync(
        self,
        query: str,
        params: Optional[Dict[str, Any]],
        pool_name: str
    ) -> Dict[str, Any]:
        """Synchronous SQLAlchemy query execution used from the worker thread pool."""
        session_factory = self.connection_pool_manager.pools[pool_name]
        session = session_factory()
        try:
            result = session.execute(text(query), params or {})
            if getattr(result, "returns_rows", False):
                rows = [dict(row._mapping) for row in result]
                return {"rows": rows, "row_count": len(rows)}

            session.commit()
            return {"row_count": getattr(result, "rowcount", 0)}
        finally:
            session.close()

    def _result_row_count(self, result: Any) -> Optional[int]:
        """Infer row count from common executor result shapes."""
        if isinstance(result, list):
            return len(result)
        if isinstance(result, tuple):
            return len(result)
        if isinstance(result, dict):
            if isinstance(result.get("rows"), list):
                return len(result["rows"])
            if "row_count" in result:
                return result["row_count"]
            if "rowcount" in result:
                return result["rowcount"]
        return None

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        return {
            'cache_stats': self.cache_manager.get_stats(),
            'connection_pool_stats': {
                name: self.connection_pool_manager.get_pool_stats(name)
                for name in self.connection_pool_manager.pools.keys()
            },
            'performance_metrics': self._performance_metrics,
            'config': {
                'performance_level': self.perf_config.level.value,
                'cache_strategy': self.cache_config.strategy.value,
                'async_processing': self.perf_config.enable_async_processing,
                'connection_pooling': self.perf_config.enable_connection_pooling
            }
        }


# Global performance optimizer instance
global_performance_optimizer = PerformanceOptimizer()


# Convenience functions and decorators
def cached(ttl: int = None, key_func: Callable = None):
    """Global caching decorator"""
    return global_performance_optimizer.cached(ttl, key_func)


def batch_process(batch_size: int = None):
    """Global batch processing decorator"""
    return global_performance_optimizer.batch_process(batch_size)


def register_query_executor(
    name: str,
    executor: Callable[[str, Optional[Dict[str, Any]]], Any]
) -> None:
    """Register a global query executor used by optimize_query_performance."""
    global_performance_optimizer.register_query_executor(name, executor)


def unregister_query_executor(name: str) -> bool:
    """Unregister a global query executor."""
    return global_performance_optimizer.unregister_query_executor(name)


async def get_performance_stats() -> Dict[str, Any]:
    """Get global performance statistics"""
    return global_performance_optimizer.get_performance_stats()


# Performance monitoring integration
@monitor_performance("performance_optimization")
async def optimize_query_performance(
    query: str,
    params: Dict[str, Any] = None,
    *,
    executor: Callable[[str, Optional[Dict[str, Any]]], Any] = None,
    executor_name: str = "default",
    pool_name: Optional[str] = None,
    cache_ttl: int = 300
) -> Dict[str, Any]:
    """Optimize database query performance with caching and connection pooling"""

    optimizer = global_performance_optimizer
    normalized_params = optimizer._normalize_query_params(params)
    executor_label, resolved_executor, executor_error = optimizer._resolve_query_executor(executor, executor_name)
    cache_key = optimizer._make_query_cache_key(query, normalized_params, executor_label, pool_name)

    # Try cache first
    cached_result = await optimizer.cache_manager.get(cache_key)
    if cached_result is not None:
        optimizer._performance_metrics['cache_hits'] += 1
        if isinstance(cached_result, dict):
            response = dict(cached_result)
            response['cached'] = True
            response['cache_hit_at'] = datetime.now(timezone.utc).isoformat()
            return response
        return {
            'query': query,
            'params': normalized_params,
            'executed_at': datetime.now(timezone.utc).isoformat(),
            'cached': True,
            'executed': True,
            'execution_strategy': 'cache',
            'result': cached_result
        }

    optimizer._performance_metrics['cache_misses'] += 1
    started_at = time.perf_counter()

    result_payload = None
    execution_strategy = None

    try:
        if resolved_executor is not None:
            execution_strategy = f"executor:{executor_label}"
            result_payload = await optimizer._execute_query_with_executor(
                resolved_executor,
                query,
                normalized_params
            )
        elif pool_name:
            result_payload = await optimizer._execute_query_with_pool(query, normalized_params, pool_name)
            if result_payload is not None:
                execution_strategy = f"pool:{pool_name}"
    except Exception:
        optimizer._performance_metrics['query_failures'] += 1
        raise

    if execution_strategy is None:
        optimizer._performance_metrics['query_skips'] += 1
        reason = executor_error
        if pool_name:
            reason = f"SQLAlchemy pool '{pool_name}' is not available"
        return {
            'query': query,
            'params': normalized_params,
            'executed_at': datetime.now(timezone.utc).isoformat(),
            'cached': False,
            'executed': False,
            'status': 'not_executed',
            'execution_strategy': 'unavailable',
            'reason': reason
        }

    duration_ms = round((time.perf_counter() - started_at) * 1000, 3)
    result = {
        'query': query,
        'params': normalized_params,
        'executed_at': datetime.now(timezone.utc).isoformat(),
        'cached': False,
        'executed': True,
        'execution_strategy': execution_strategy,
        'duration_ms': duration_ms,
        'row_count': optimizer._result_row_count(result_payload),
        'result': result_payload
    }
    optimizer._performance_metrics['query_executions'] += 1

    # Cache result
    if cache_ttl and cache_ttl > 0:
        await optimizer.cache_manager.set(cache_key, result, ttl=cache_ttl)

    return result


async def warm_cache(keys: List[str], data_source: Callable) -> int:
    """Warm up cache with frequently accessed data"""
    warmed_count = 0

    for key in keys:
        try:
            # Check if already cached
            if await global_performance_optimizer.cache_manager.get(key) is None:
                # Fetch and cache data
                if asyncio.iscoroutinefunction(data_source):
                    data = await data_source(key)
                else:
                    data = await global_performance_optimizer.task_manager.run_in_thread(data_source, key)

                await global_performance_optimizer.cache_manager.set(key, data)
                warmed_count += 1

        except Exception as e:
            logger.warning(f"Failed to warm cache for key {key}: {e}")

    return warmed_count
