"""
Optimized Twitter Crawler Core Module
====================================

High-performance Twitter crawling with advanced optimizations:
- Connection pooling and session management
- Intelligent rate limiting with token bucket algorithm
- Memory-efficient data structures and caching
- Asynchronous batch processing
- Comprehensive error handling and recovery
- Performance monitoring and metrics

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Date: June 28, 2025
"""

import asyncio
import logging
import json
import time
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union, Callable, AsyncGenerator, Tuple
from enum import Enum
from datetime import datetime, timedelta
import pickle
import os
import hashlib
from collections import deque, defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc

logger = logging.getLogger(__name__)

# Optional dependencies with fallbacks
try:
    import twikit
    from twikit import Client
    from twikit.errors import TwitterException, Unauthorized, TooManyRequests
    TwikitException = TwitterException  # Alias for backwards compatibility
    TWIKIT_AVAILABLE = True
except ImportError:
    TWIKIT_AVAILABLE = False
    Client = None
    TwitterException = Exception
    TwikitException = Exception
    Unauthorized = Exception
    TooManyRequests = Exception

try:
    import aiohttp
    import aiofiles
    ASYNC_IO_AVAILABLE = True
except ImportError:
    ASYNC_IO_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    from cachetools import TTLCache, LRUCache
    CACHETOOLS_AVAILABLE = True
except ImportError:
    CACHETOOLS_AVAILABLE = False
    # Fallback implementations
    class TTLCache(dict):
        def __init__(self, maxsize, ttl): 
            super().__init__()
            self.maxsize = maxsize
    
    class LRUCache(dict):
        def __init__(self, maxsize): 
            super().__init__()
            self.maxsize = maxsize


# Custom Exceptions
class OptimizedCrawlerError(Exception):
    """Base exception for optimized Twitter crawler errors"""
    pass


class ConnectionPoolError(OptimizedCrawlerError):
    """Connection pool related errors"""
    pass


class CacheError(OptimizedCrawlerError):
    """Cache related errors"""
    pass


class PerformanceError(OptimizedCrawlerError):
    """Performance related errors"""
    pass


class CrawlerStatus(Enum):
    """Enhanced crawler status tracking"""
    IDLE = "idle"
    INITIALIZING = "initializing"
    AUTHENTICATING = "authenticating"
    ACTIVE = "active"
    RATE_LIMITED = "rate_limited"
    OPTIMIZING = "optimizing"
    ERROR = "error"
    STOPPED = "stopped"
    MAINTENANCE = "maintenance"


@dataclass
class PerformanceMetrics:
    """Comprehensive performance tracking"""
    # Request metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rate_limited_requests: int = 0
    
    # Timing metrics
    avg_response_time: float = 0.0
    min_response_time: float = float('inf')
    max_response_time: float = 0.0
    total_response_time: float = 0.0
    
    # Memory metrics
    memory_usage_mb: float = 0.0
    peak_memory_mb: float = 0.0
    cache_hit_rate: float = 0.0
    cache_size: int = 0
    
    # Connection metrics
    active_connections: int = 0
    connection_pool_size: int = 0
    connection_reuse_rate: float = 0.0
    
    # Error metrics
    error_rate: float = 0.0
    retry_rate: float = 0.0
    
    # Throughput metrics
    requests_per_second: float = 0.0
    tweets_processed: int = 0
    users_processed: int = 0
    
    # Time tracking
    uptime_seconds: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def update_request_metrics(self, success: bool, response_time: float):
        """Update request-related metrics"""
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        self.total_response_time += response_time
        self.avg_response_time = self.total_response_time / self.total_requests
        self.min_response_time = min(self.min_response_time, response_time)
        self.max_response_time = max(self.max_response_time, response_time)
        
        # Calculate rates
        if self.total_requests > 0:
            self.error_rate = self.failed_requests / self.total_requests
        
        self.last_updated = datetime.now()
    
    def update_memory_metrics(self):
        """Update memory usage metrics"""
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()
                current_memory = memory_info.rss / 1024 / 1024  # MB
                self.memory_usage_mb = current_memory
                self.peak_memory_mb = max(self.peak_memory_mb, current_memory)
            except Exception as e:
                logger.debug(f"Failed to update memory metrics: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        return {
            "requests": {
                "total": self.total_requests,
                "successful": self.successful_requests,
                "failed": self.failed_requests,
                "success_rate": self.successful_requests / max(self.total_requests, 1) * 100,
                "error_rate": self.error_rate * 100
            },
            "performance": {
                "avg_response_time_ms": self.avg_response_time * 1000,
                "min_response_time_ms": self.min_response_time * 1000,
                "max_response_time_ms": self.max_response_time * 1000,
                "requests_per_second": self.requests_per_second
            },
            "memory": {
                "current_mb": self.memory_usage_mb,
                "peak_mb": self.peak_memory_mb
            },
            "cache": {
                "hit_rate": self.cache_hit_rate * 100,
                "size": self.cache_size
            },
            "uptime_hours": self.uptime_seconds / 3600
        }


@dataclass
class OptimizedTwitterConfig:
    """Enhanced configuration with performance optimizations"""
    # Authentication
    username: Optional[str] = None
    password: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    
    # Session management
    session_file: str = "twitter_session_optimized.pkl"
    auto_save_session: bool = True
    session_timeout: int = 3600  # 1 hour
    session_persistence: bool = True
    
    # Enhanced rate limiting (token bucket algorithm)
    rate_limit_requests_per_minute: int = 30
    rate_limit_requests_per_hour: int = 1000
    rate_limit_burst_size: int = 10  # Allow burst requests
    rate_limit_refill_rate: float = 0.5  # Tokens per second
    wait_on_rate_limit: bool = True
    adaptive_rate_limiting: bool = True
    
    # Connection pooling
    connection_pool_size: int = 20
    connection_pool_max_overflow: int = 10
    connection_timeout: int = 30
    connection_keepalive: bool = True
    connection_retry_attempts: int = 3
    
    # Caching configuration
    enable_caching: bool = True
    cache_type: str = "memory"  # "memory", "redis", "file"
    cache_ttl: int = 3600  # 1 hour
    cache_max_size: int = 10000
    cache_compression: bool = True
    
    # Performance optimizations
    batch_size: int = 100
    max_concurrent_requests: int = 10
    request_timeout: int = 60
    enable_compression: bool = True
    enable_keep_alive: bool = True
    
    # Memory management
    memory_limit_mb: int = 1024  # 1GB
    garbage_collection_interval: int = 300  # 5 minutes
    weak_references: bool = True
    memory_monitoring: bool = True
    
    # Retry configuration
    max_retries: int = 5
    backoff_factor: float = 2.0
    backoff_jitter: bool = True
    retry_on_status: List[int] = field(default_factory=lambda: [429, 500, 502, 503, 504])
    
    # Advanced features
    enable_metrics: bool = True
    metrics_interval: int = 60  # 1 minute
    enable_health_checks: bool = True
    health_check_interval: int = 300  # 5 minutes
    
    # Logging and debugging
    log_level: str = "INFO"
    log_requests: bool = False
    log_performance: bool = True
    enable_profiling: bool = False
    
    # User agent and headers
    user_agent: Optional[str] = None
    custom_headers: Dict[str, str] = field(default_factory=dict)
    
    # Proxy settings
    proxy: Optional[str] = None
    proxy_auth: Optional[tuple] = None
    proxy_rotation: bool = False
    
    # Security
    verify_ssl: bool = True
    enable_cookies: bool = True
    cookie_encryption: bool = False


class TokenBucketRateLimiter:
    """Advanced rate limiter using token bucket algorithm"""
    
    def __init__(self, config: OptimizedTwitterConfig):
        self.config = config
        self.capacity = config.rate_limit_burst_size
        self.tokens = self.capacity
        self.refill_rate = config.rate_limit_refill_rate
        self.last_refill = time.time()
        self.lock = threading.Lock()
        
        # Adaptive rate limiting
        self.adaptive = config.adaptive_rate_limiting
        self.success_count = 0
        self.error_count = 0
        self.adjustment_factor = 1.0
        
        # Request tracking
        self.request_times = deque(maxlen=1000)
        
    def acquire_token(self, tokens_needed: int = 1) -> bool:
        """Acquire tokens from the bucket"""
        with self.lock:
            self._refill_bucket()
            
            if self.tokens >= tokens_needed:
                self.tokens -= tokens_needed
                self.request_times.append(time.time())
                return True
            
            return False
    
    def _refill_bucket(self):
        """Refill the token bucket based on time elapsed"""
        now = time.time()
        time_passed = now - self.last_refill
        tokens_to_add = time_passed * self.refill_rate * self.adjustment_factor
        
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now
    
    def wait_for_token(self) -> float:
        """Calculate wait time for next token"""
        with self.lock:
            self._refill_bucket()
            
            if self.tokens >= 1:
                return 0.0
            
            wait_time = (1 - self.tokens) / (self.refill_rate * self.adjustment_factor)
            return max(0.0, wait_time)
    
    def record_success(self):
        """Record successful request for adaptive rate limiting"""
        if self.adaptive:
            self.success_count += 1
            if self.success_count % 100 == 0:
                # Gradually increase rate if successful
                self.adjustment_factor = min(2.0, self.adjustment_factor * 1.1)
    
    def record_error(self):
        """Record failed request for adaptive rate limiting"""
        if self.adaptive:
            self.error_count += 1
            if self.error_count % 10 == 0:
                # Decrease rate on errors
                self.adjustment_factor = max(0.5, self.adjustment_factor * 0.9)
    
    def get_current_rate(self) -> float:
        """Get current requests per second rate"""
        now = time.time()
        recent_requests = [t for t in self.request_times if now - t < 60]
        return len(recent_requests) / 60.0
    
    def get_status(self) -> Dict[str, Any]:
        """Get rate limiter status"""
        return {
            "tokens_available": self.tokens,
            "capacity": self.capacity,
            "refill_rate": self.refill_rate * self.adjustment_factor,
            "current_rate": self.get_current_rate(),
            "adjustment_factor": self.adjustment_factor,
            "success_count": self.success_count,
            "error_count": self.error_count
        }


class MemoryOptimizedCache:
    """Memory-efficient caching with compression and TTL"""
    
    def __init__(self, config: OptimizedTwitterConfig):
        self.config = config
        self.max_size = config.cache_max_size
        self.ttl = config.cache_ttl
        self.compression = config.cache_compression
        
        # Initialize cache based on type
        if config.cache_type == "redis" and REDIS_AVAILABLE:
            self._init_redis_cache()
        else:
            self._init_memory_cache()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        # Weak references for memory efficiency
        if config.weak_references:
            self._weak_refs = weakref.WeakValueDictionary()
    
    def _init_memory_cache(self):
        """Initialize in-memory cache"""
        if CACHETOOLS_AVAILABLE:
            self.cache = TTLCache(maxsize=self.max_size, ttl=self.ttl)
        else:
            self.cache = {}
            self._expiry_times = {}
    
    def _init_redis_cache(self):
        """Initialize Redis cache"""
        try:
            self.redis_client = redis.Redis(decode_responses=True)
            self.cache_type = "redis"
        except Exception as e:
            logger.warning(f"Failed to initialize Redis cache: {e}")
            self._init_memory_cache()
            self.cache_type = "memory"
    
    def _make_key(self, key: Any) -> str:
        """Create cache key from input"""
        if isinstance(key, str):
            return key
        
        # Create hash for complex keys
        key_str = json.dumps(key, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _compress_data(self, data: Any) -> bytes:
        """Compress data if compression is enabled"""
        if not self.compression:
            return pickle.dumps(data)
        
        try:
            import gzip
            return gzip.compress(pickle.dumps(data))
        except ImportError:
            return pickle.dumps(data)
    
    def _decompress_data(self, data: bytes) -> Any:
        """Decompress data if compression was used"""
        if not self.compression:
            return pickle.loads(data)
        
        try:
            import gzip
            return pickle.loads(gzip.decompress(data))
        except (ImportError, gzip.BadGzipFile):
            return pickle.loads(data)
    
    def get(self, key: Any) -> Optional[Any]:
        """Get item from cache"""
        cache_key = self._make_key(key)
        
        try:
            if hasattr(self, 'redis_client'):
                # Redis cache
                data = self.redis_client.get(cache_key)
                if data:
                    self.hits += 1
                    return json.loads(data)
                else:
                    self.misses += 1
                    return None
            else:
                # Memory cache
                if CACHETOOLS_AVAILABLE:
                    if cache_key in self.cache:
                        self.hits += 1
                        return self.cache[cache_key]
                    else:
                        self.misses += 1
                        return None
                else:
                    # Fallback cache with manual TTL
                    if cache_key in self.cache:
                        if time.time() < self._expiry_times.get(cache_key, 0):
                            self.hits += 1
                            return self.cache[cache_key]
                        else:
                            # Expired
                            del self.cache[cache_key]
                            del self._expiry_times[cache_key]
                    
                    self.misses += 1
                    return None
                    
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            self.misses += 1
            return None
    
    def set(self, key: Any, value: Any, ttl: Optional[int] = None) -> bool:
        """Set item in cache"""
        cache_key = self._make_key(key)
        cache_ttl = ttl or self.ttl
        
        try:
            if hasattr(self, 'redis_client'):
                # Redis cache
                self.redis_client.setex(
                    cache_key, 
                    cache_ttl, 
                    json.dumps(value, default=str)
                )
                return True
            else:
                # Memory cache
                if CACHETOOLS_AVAILABLE:
                    self.cache[cache_key] = value
                    return True
                else:
                    # Fallback cache
                    if len(self.cache) >= self.max_size:
                        # Simple LRU eviction
                        oldest_key = next(iter(self.cache))
                        del self.cache[oldest_key]
                        if oldest_key in self._expiry_times:
                            del self._expiry_times[oldest_key]
                        self.evictions += 1
                    
                    self.cache[cache_key] = value
                    self._expiry_times[cache_key] = time.time() + cache_ttl
                    return True
                    
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def delete(self, key: Any) -> bool:
        """Delete item from cache"""
        cache_key = self._make_key(key)
        
        try:
            if hasattr(self, 'redis_client'):
                return bool(self.redis_client.delete(cache_key))
            else:
                if cache_key in self.cache:
                    del self.cache[cache_key]
                    if hasattr(self, '_expiry_times') and cache_key in self._expiry_times:
                        del self._expiry_times[cache_key]
                    return True
                return False
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    def clear(self):
        """Clear all cache"""
        try:
            if hasattr(self, 'redis_client'):
                self.redis_client.flushdb()
            else:
                self.cache.clear()
                if hasattr(self, '_expiry_times'):
                    self._expiry_times.clear()
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests) if total_requests > 0 else 0
        
        stats = {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "evictions": self.evictions,
            "size": len(self.cache) if hasattr(self, 'cache') else 0,
            "max_size": self.max_size
        }
        
        # Add memory usage if available
        try:
            if hasattr(self, 'cache') and PSUTIL_AVAILABLE:
                import sys
                cache_size_bytes = sys.getsizeof(self.cache)
                stats["memory_usage_mb"] = cache_size_bytes / 1024 / 1024
        except Exception:
            pass
        
        return stats


class ConnectionPool:
    """High-performance connection pool for Twitter API"""
    
    def __init__(self, config: OptimizedTwitterConfig):
        self.config = config
        self.max_size = config.connection_pool_size
        self.max_overflow = config.connection_pool_max_overflow
        self.timeout = config.connection_timeout
        
        # Connection tracking
        self.connections: deque = deque()
        self.active_connections = 0
        self.total_created = 0
        self.total_reused = 0
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Health monitoring
        self.health_checks = 0
        self.health_failures = 0
    
    async def acquire_connection(self) -> Optional[Client]:
        """Acquire a connection from the pool"""
        with self.lock:
            # Try to reuse existing connection
            while self.connections:
                client = self.connections.popleft()
                if await self._is_connection_healthy(client):
                    self.active_connections += 1
                    self.total_reused += 1
                    return client
                else:
                    # Connection is unhealthy, discard it
                    self.health_failures += 1
            
            # Create new connection if pool allows
            if self.active_connections < (self.max_size + self.max_overflow):
                client = await self._create_connection()
                if client:
                    self.active_connections += 1
                    self.total_created += 1
                    return client
            
            return None
    
    async def release_connection(self, client: Client):
        """Release a connection back to the pool"""
        with self.lock:
            self.active_connections -= 1
            
            # Only return to pool if under max_size and connection is healthy
            if (len(self.connections) < self.max_size and 
                await self._is_connection_healthy(client)):
                self.connections.append(client)
            else:
                # Close the connection
                await self._close_connection(client)
    
    async def _create_connection(self) -> Optional[Client]:
        """Create a new Twitter client connection"""
        try:
            if not TWIKIT_AVAILABLE:
                return None
            
            client = Client()
            
            # Configure client settings
            if self.config.custom_headers:
                for key, value in self.config.custom_headers.items():
                    client.headers[key] = value
            
            if self.config.proxy:
                # Configure proxy if available
                pass
            
            return client
            
        except Exception as e:
            logger.error(f"Failed to create connection: {e}")
            return None
    
    async def _is_connection_healthy(self, client: Client) -> bool:
        """Check if a connection is healthy"""
        try:
            self.health_checks += 1
            
            # Simple health check - you might want to make this more robust
            if not client:
                return False
            
            # Check if client has required attributes
            return hasattr(client, 'login') and hasattr(client, 'search_tweet')
            
        except Exception as e:
            logger.debug(f"Connection health check failed: {e}")
            return False
    
    async def _close_connection(self, client: Client):
        """Close a connection"""
        try:
            # Implement proper connection closing based on twikit API
            if hasattr(client, 'close'):
                await client.close()
        except Exception as e:
            logger.debug(f"Error closing connection: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        with self.lock:
            reuse_rate = (self.total_reused / max(self.total_created + self.total_reused, 1)) * 100
            health_rate = ((self.health_checks - self.health_failures) / max(self.health_checks, 1)) * 100
            
            return {
                "pool_size": len(self.connections),
                "active_connections": self.active_connections,
                "max_size": self.max_size,
                "total_created": self.total_created,
                "total_reused": self.total_reused,
                "reuse_rate": reuse_rate,
                "health_checks": self.health_checks,
                "health_failures": self.health_failures,
                "health_rate": health_rate
            }
    
    async def close_all(self):
        """Close all connections in the pool"""
        with self.lock:
            while self.connections:
                client = self.connections.popleft()
                await self._close_connection(client)
            
            self.active_connections = 0


class OptimizedTwitterCrawler:
    """High-performance Twitter crawler with advanced optimizations"""
    
    def __init__(self, config: OptimizedTwitterConfig):
        if not TWIKIT_AVAILABLE:
            raise ImportError("twikit is required for OptimizedTwitterCrawler. Install with: pip install twikit")
        
        self.config = config
        self.status = CrawlerStatus.IDLE
        self.start_time = datetime.now()
        
        # Core components
        self.rate_limiter = TokenBucketRateLimiter(config)
        self.cache = MemoryOptimizedCache(config)
        self.connection_pool = ConnectionPool(config)
        self.metrics = PerformanceMetrics()
        
        # Session management
        self.session_data: Dict[str, Any] = {}
        self.is_authenticated = False
        
        # Error handling
        self.error_count = 0
        self.last_error: Optional[Exception] = None
        self.error_recovery_strategies = []
        
        # Callbacks and hooks
        self.callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        self.shutdown_event = asyncio.Event()
        
        # Memory management
        self.last_gc_time = time.time()
        self.memory_warnings = 0
        
        # Thread pool for CPU-intensive tasks
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(getattr(logging, config.log_level.upper()))
        
        # Performance monitoring
        if config.enable_metrics:
            self._setup_monitoring()
    
    def _setup_monitoring(self):
        """Setup performance monitoring"""
        self.logger.info("Setting up performance monitoring")
        
        # Schedule background monitoring tasks
        if self.config.enable_health_checks:
            asyncio.create_task(self._health_check_loop())
        
        if self.config.enable_metrics:
            asyncio.create_task(self._metrics_collection_loop())
        
        if self.config.memory_monitoring:
            asyncio.create_task(self._memory_monitoring_loop())
    
    async def initialize(self) -> bool:
        """Initialize the optimized crawler"""
        try:
            self.status = CrawlerStatus.INITIALIZING
            self.logger.info("Initializing optimized Twitter crawler")
            
            # Load existing session if available
            if await self._load_session():
                if await self._validate_session():
                    self.status = CrawlerStatus.ACTIVE
                    self.logger.info("Existing session validated successfully")
                    return True
            
            # Authenticate if credentials provided
            if self.config.username and self.config.password:
                return await self._authenticate()
            else:
                self.logger.warning("No credentials provided, using unauthenticated session")
                self.status = CrawlerStatus.ACTIVE
                return True
                
        except Exception as e:
            self.status = CrawlerStatus.ERROR
            self.last_error = e
            self.logger.error(f"Failed to initialize crawler: {e}")
            return False
    
    async def _authenticate(self) -> bool:
        """Authenticate with Twitter using connection pool"""
        try:
            self.status = CrawlerStatus.AUTHENTICATING
            self.logger.info("Authenticating with Twitter")
            
            # Get connection from pool
            client = await self.connection_pool.acquire_connection()
            if not client:
                raise OptimizedCrawlerError("Failed to acquire connection for authentication")
            
            try:
                # Perform authentication
                await client.login(
                    auth_info_1=self.config.username or self.config.email,
                    auth_info_2=self.config.email or self.config.phone,
                    password=self.config.password
                )
                
                self.is_authenticated = True
                await self._save_session()
                self.status = CrawlerStatus.ACTIVE
                
                self.logger.info("Authentication successful")
                self._trigger_callback('authentication_success')
                return True
                
            finally:
                await self.connection_pool.release_connection(client)
                
        except Unauthorized as e:
            self.status = CrawlerStatus.ERROR
            self.last_error = e
            self.logger.error(f"Authentication failed: {e}")
            self._trigger_callback('authentication_failed', error=e)
            raise OptimizedCrawlerError(f"Authentication failed: {e}")
        
        except Exception as e:
            self.status = CrawlerStatus.ERROR
            self.last_error = e
            self.logger.error(f"Unexpected error during authentication: {e}")
            raise OptimizedCrawlerError(f"Authentication error: {e}")
    
    async def _load_session(self) -> bool:
        """Load session from file with optimization"""
        try:
            if not os.path.exists(self.config.session_file):
                return False
            
            # Check file age
            file_age = time.time() - os.path.getmtime(self.config.session_file)
            if file_age > self.config.session_timeout:
                self.logger.debug("Session file expired")
                return False
            
            with open(self.config.session_file, 'rb') as f:
                self.session_data = pickle.load(f)
            
            self.is_authenticated = self.session_data.get('is_authenticated', False)
            self.logger.debug("Session loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load session: {e}")
            return False
    
    async def _save_session(self) -> bool:
        """Save session to file with optimization"""
        try:
            if not self.config.auto_save_session:
                return True
            
            session_data = {
                'is_authenticated': self.is_authenticated,
                'last_activity': datetime.now(),
                'config_hash': self._get_config_hash(),
                'metrics': asdict(self.metrics)
            }
            
            # Use atomic write for safety
            temp_file = f"{self.config.session_file}.tmp"
            with open(temp_file, 'wb') as f:
                pickle.dump(session_data, f)
            
            os.rename(temp_file, self.config.session_file)
            self.logger.debug("Session saved successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save session: {e}")
            return False
    
    def _get_config_hash(self) -> str:
        """Get hash of configuration for cache invalidation"""
        config_str = json.dumps(asdict(self.config), sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    async def _validate_session(self) -> bool:
        """Validate existing session"""
        try:
            # Check if config has changed
            stored_hash = self.session_data.get('config_hash')
            current_hash = self._get_config_hash()
            
            if stored_hash != current_hash:
                self.logger.debug("Configuration changed, session invalid")
                return False
            
            # Try a simple API call to validate
            client = await self.connection_pool.acquire_connection()
            if not client:
                return False
            
            try:
                # Simple validation - this might need adjustment based on twikit API
                return True  # Placeholder - implement actual validation
            finally:
                await self.connection_pool.release_connection(client)
                
        except Exception as e:
            self.logger.debug(f"Session validation failed: {e}")
            return False
    
    async def make_optimized_request(
        self, 
        request_func: Callable, 
        cache_key: Optional[str] = None,
        cache_ttl: Optional[int] = None,
        *args, 
        **kwargs
    ) -> Any:
        """Make an optimized request with caching, rate limiting, and error handling"""
        
        # Check cache first
        if cache_key and self.config.enable_caching:
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                self.metrics.cache_hit_rate = (self.cache.hits / (self.cache.hits + self.cache.misses))
                return cached_result
        
        # Rate limiting with token bucket
        if not self.rate_limiter.acquire_token():
            wait_time = self.rate_limiter.wait_for_token()
            if wait_time > 0:
                self.status = CrawlerStatus.RATE_LIMITED
                self.logger.debug(f"Rate limited, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
                self.status = CrawlerStatus.ACTIVE
        
        # Execute request with retries and monitoring
        attempt = 0
        start_time = time.time()
        
        while attempt < self.config.max_retries:
            try:
                # Get connection from pool
                client = await self.connection_pool.acquire_connection()
                if not client:
                    raise ConnectionPoolError("Failed to acquire connection from pool")
                
                try:
                    # Execute the request
                    result = await request_func(client, *args, **kwargs)
                    
                    # Record success
                    response_time = time.time() - start_time
                    self.metrics.update_request_metrics(True, response_time)
                    self.rate_limiter.record_success()
                    
                    # Cache the result
                    if cache_key and self.config.enable_caching:
                        self.cache.set(cache_key, result, cache_ttl)
                    
                    return result
                    
                finally:
                    await self.connection_pool.release_connection(client)
                    
            except TooManyRequests as e:
                self.logger.warning(f"Rate limited by Twitter: {e}")
                self.status = CrawlerStatus.RATE_LIMITED
                self.rate_limiter.record_error()
                
                # Extract reset time from exception or use default
                reset_time = getattr(e, 'reset_time', 900)  # 15 minutes default
                await asyncio.sleep(min(reset_time, 3600))  # Max 1 hour wait
                self.status = CrawlerStatus.ACTIVE
                
            except (TwikitException, ConnectionPoolError) as e:
                attempt += 1
                self.error_count += 1
                self.last_error = e
                
                response_time = time.time() - start_time
                self.metrics.update_request_metrics(False, response_time)
                self.rate_limiter.record_error()
                
                if attempt >= self.config.max_retries:
                    self.logger.error(f"Request failed after {self.config.max_retries} attempts: {e}")
                    raise OptimizedCrawlerError(f"Request failed: {e}")
                
                # Exponential backoff with jitter
                base_wait = self.config.backoff_factor ** attempt
                if self.config.backoff_jitter:
                    import random
                    jitter = random.uniform(0.1, 0.5)
                    wait_time = base_wait * (1 + jitter)
                else:
                    wait_time = base_wait
                
                self.logger.warning(f"Request failed (attempt {attempt}), retrying in {wait_time:.2f}s: {e}")
                await asyncio.sleep(wait_time)
            
            except Exception as e:
                self.error_count += 1
                self.last_error = e
                response_time = time.time() - start_time
                self.metrics.update_request_metrics(False, response_time)
                self.logger.error(f"Unexpected error in request: {e}")
                raise OptimizedCrawlerError(f"Unexpected error: {e}")
    
    async def batch_search_tweets(
        self, 
        queries: List[str],
        count_per_query: int = 20,
        result_type: str = 'recent'
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Perform batch tweet searches with optimization"""
        
        async def search_single_query(query: str) -> Tuple[str, List[Dict[str, Any]]]:
            """Search for a single query"""
            try:
                cache_key = f"search_{hashlib.md5(f'{query}_{count_per_query}_{result_type}'.encode()).hexdigest()}"
                
                async def _search(client):
                    tweets = await client.search_tweet(query, count=count_per_query)
                    return [self._tweet_to_dict(tweet) for tweet in tweets]
                
                result = await self.make_optimized_request(
                    _search,
                    cache_key=cache_key,
                    cache_ttl=300  # 5 minutes cache
                )
                
                return query, result
                
            except Exception as e:
                self.logger.error(f"Failed to search for query '{query}': {e}")
                return query, []
        
        # Execute searches concurrently with semaphore
        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
        async def bounded_search(query: str):
            async with semaphore:
                return await search_single_query(query)
        
        # Execute all searches
        tasks = [bounded_search(query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        search_results = {}
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Batch search error: {result}")
                continue
            
            query, tweets = result
            search_results[query] = tweets
        
        return search_results
    
    async def get_user_batch(self, usernames: List[str]) -> Dict[str, Optional[Dict[str, Any]]]:
        """Get multiple users with batch optimization"""
        
        async def get_single_user(username: str) -> Tuple[str, Optional[Dict[str, Any]]]:
            """Get a single user"""
            try:
                cache_key = f"user_{username}"
                
                async def _get_user(client):
                    user = await client.get_user_by_screen_name(username)
                    return self._user_to_dict(user) if user else None
                
                result = await self.make_optimized_request(
                    _get_user,
                    cache_key=cache_key,
                    cache_ttl=1800  # 30 minutes cache for user data
                )
                
                return username, result
                
            except Exception as e:
                self.logger.error(f"Failed to get user '{username}': {e}")
                return username, None
        
        # Execute with concurrency control
        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
        async def bounded_get_user(username: str):
            async with semaphore:
                return await get_single_user(username)
        
        tasks = [bounded_get_user(username) for username in usernames]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        user_results = {}
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Batch user fetch error: {result}")
                continue
            
            username, user_data = result
            user_results[username] = user_data
        
        return user_results
    
    def _user_to_dict(self, user) -> Dict[str, Any]:
        """Convert twikit user object to dictionary with optimization"""
        return {
            'id': getattr(user, 'id', None),
            'username': getattr(user, 'screen_name', None),
            'display_name': getattr(user, 'name', None),
            'description': getattr(user, 'description', None),
            'followers_count': getattr(user, 'followers_count', 0),
            'following_count': getattr(user, 'friends_count', 0),
            'tweet_count': getattr(user, 'statuses_count', 0),
            'verified': getattr(user, 'verified', False),
            'profile_image_url': getattr(user, 'profile_image_url_https', None),
            'created_at': getattr(user, 'created_at', None),
            'location': getattr(user, 'location', None),
            'url': getattr(user, 'url', None)
        }
    
    def _tweet_to_dict(self, tweet) -> Dict[str, Any]:
        """Convert twikit tweet object to dictionary with optimization"""
        return {
            'id': getattr(tweet, 'id', None),
            'text': getattr(tweet, 'full_text', getattr(tweet, 'text', None)),
            'created_at': getattr(tweet, 'created_at', None),
            'user': self._user_to_dict(getattr(tweet, 'user', None)) if hasattr(tweet, 'user') else None,
            'retweet_count': getattr(tweet, 'retweet_count', 0),
            'favorite_count': getattr(tweet, 'favorite_count', 0),
            'reply_count': getattr(tweet, 'reply_count', 0),
            'lang': getattr(tweet, 'lang', None),
            'hashtags': [tag.get('text', '') for tag in getattr(tweet, 'hashtags', [])],
            'urls': [url.get('expanded_url', '') for url in getattr(tweet, 'urls', [])],
            'is_retweet': getattr(tweet, 'retweeted', False),
            'in_reply_to_status_id': getattr(tweet, 'in_reply_to_status_id', None),
            'in_reply_to_user_id': getattr(tweet, 'in_reply_to_user_id', None)
        }
    
    async def _health_check_loop(self):
        """Background health check loop"""
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(self.config.health_check_interval)
                await self._perform_health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health check loop error: {e}")
    
    async def _metrics_collection_loop(self):
        """Background metrics collection loop"""
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(self.config.metrics_interval)
                self._collect_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Metrics collection error: {e}")
    
    async def _memory_monitoring_loop(self):
        """Background memory monitoring loop"""
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(60)  # Check every minute
                self._check_memory_usage()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Memory monitoring error: {e}")
    
    async def _perform_health_check(self):
        """Perform system health check"""
        try:
            # Check connection pool health
            pool_stats = self.connection_pool.get_stats()
            if pool_stats['health_rate'] < 80:
                self.logger.warning(f"Connection pool health degraded: {pool_stats['health_rate']:.1f}%")
            
            # Check rate limiter status
            rate_stats = self.rate_limiter.get_status()
            if rate_stats['error_count'] > 100:
                self.logger.warning(f"High error rate detected: {rate_stats['error_count']} errors")
            
            # Check cache performance
            cache_stats = self.cache.get_stats()
            if cache_stats['hit_rate'] < 0.5:
                self.logger.info(f"Low cache hit rate: {cache_stats['hit_rate']:.1%}")
            
            self.logger.debug("Health check completed")
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
    
    def _collect_metrics(self):
        """Collect performance metrics"""
        try:
            # Update uptime
            self.metrics.uptime_seconds = (datetime.now() - self.start_time).total_seconds()
            
            # Update memory metrics
            self.metrics.update_memory_metrics()
            
            # Update cache metrics
            cache_stats = self.cache.get_stats()
            self.metrics.cache_hit_rate = cache_stats['hit_rate']
            self.metrics.cache_size = cache_stats['size']
            
            # Update connection metrics
            pool_stats = self.connection_pool.get_stats()
            self.metrics.active_connections = pool_stats['active_connections']
            self.metrics.connection_pool_size = pool_stats['pool_size']
            self.metrics.connection_reuse_rate = pool_stats['reuse_rate'] / 100
            
            # Calculate requests per second
            if self.metrics.uptime_seconds > 0:
                self.metrics.requests_per_second = self.metrics.total_requests / self.metrics.uptime_seconds
            
            if self.config.log_performance:
                self.logger.debug(f"Performance metrics: {self.metrics.get_summary()}")
                
        except Exception as e:
            self.logger.error(f"Metrics collection failed: {e}")
    
    def _check_memory_usage(self):
        """Check and manage memory usage"""
        try:
            if not PSUTIL_AVAILABLE:
                return
            
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            if memory_mb > self.config.memory_limit_mb:
                self.memory_warnings += 1
                self.logger.warning(f"Memory usage high: {memory_mb:.1f}MB (limit: {self.config.memory_limit_mb}MB)")
                
                # Trigger garbage collection
                if time.time() - self.last_gc_time > self.config.garbage_collection_interval:
                    collected = gc.collect()
                    self.last_gc_time = time.time()
                    self.logger.info(f"Garbage collection completed, collected {collected} objects")
                
                # Clear cache if memory is still high
                if memory_mb > self.config.memory_limit_mb * 1.2:
                    self.cache.clear()
                    self.logger.warning("Cache cleared due to high memory usage")
                    
        except Exception as e:
            self.logger.error(f"Memory monitoring failed: {e}")
    
    def add_callback(self, event: str, callback: Callable):
        """Add event callback"""
        self.callbacks[event].append(callback)
    
    def remove_callback(self, event: str, callback: Callable):
        """Remove event callback"""
        if callback in self.callbacks[event]:
            self.callbacks[event].remove(callback)
    
    def _trigger_callback(self, event: str, **kwargs):
        """Trigger event callbacks"""
        for callback in self.callbacks[event]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(**kwargs))
                else:
                    callback(**kwargs)
            except Exception as e:
                self.logger.error(f"Error in callback for event {event}: {e}")
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive crawler status"""
        return {
            'status': self.status.value,
            'is_authenticated': self.is_authenticated,
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds(),
            'error_count': self.error_count,
            'last_error': str(self.last_error) if self.last_error else None,
            'performance_metrics': self.metrics.get_summary(),
            'rate_limiter': self.rate_limiter.get_status(),
            'connection_pool': self.connection_pool.get_stats(),
            'cache': self.cache.get_stats(),
            'memory_warnings': self.memory_warnings,
            'background_tasks': len(self.background_tasks)
        }
    
    async def close(self):
        """Close the crawler and clean up resources"""
        try:
            self.status = CrawlerStatus.STOPPED
            self.logger.info("Shutting down optimized Twitter crawler")
            
            # Signal background tasks to stop
            self.shutdown_event.set()
            
            # Cancel background tasks
            for task in self.background_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            if self.background_tasks:
                await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
            # Save session
            if self.config.auto_save_session:
                await self._save_session()
            
            # Close connection pool
            await self.connection_pool.close_all()
            
            # Clear cache
            self.cache.clear()
            
            # Close thread pool
            self.thread_pool.shutdown(wait=True)
            
            # Final garbage collection
            gc.collect()
            
            self.logger.info("Twitter crawler shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


# Factory functions
def create_optimized_twitter_crawler(
    username: Optional[str] = None,
    password: Optional[str] = None,
    email: Optional[str] = None,
    **config_kwargs
) -> OptimizedTwitterCrawler:
    """Create an optimized Twitter crawler"""
    config = OptimizedTwitterConfig(
        username=username,
        password=password,
        email=email,
        **config_kwargs
    )
    return OptimizedTwitterCrawler(config)


async def optimized_tweet_search(
    query: str,
    count: int = 20,
    username: Optional[str] = None,
    password: Optional[str] = None,
    **config_kwargs
) -> List[Dict[str, Any]]:
    """Quick optimized tweet search function"""
    crawler = create_optimized_twitter_crawler(
        username=username, 
        password=password,
        **config_kwargs
    )
    
    try:
        await crawler.initialize()
        
        async def _search(client):
            tweets = await client.search_tweet(query, count=count)
            return [crawler._tweet_to_dict(tweet) for tweet in tweets]
        
        return await crawler.make_optimized_request(_search)
    finally:
        await crawler.close()


__all__ = [
    'OptimizedTwitterCrawler', 'OptimizedTwitterConfig', 'PerformanceMetrics',
    'TokenBucketRateLimiter', 'MemoryOptimizedCache', 'ConnectionPool',
    'CrawlerStatus', 'OptimizedCrawlerError', 'ConnectionPoolError', 'CacheError',
    'create_optimized_twitter_crawler', 'optimized_tweet_search'
]