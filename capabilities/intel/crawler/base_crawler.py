"""
Enhanced Base Crawler with Advanced Resource Management
======================================================

A production-ready base crawler implementation that provides robust resource management,
error handling, rate limiting, and monitoring capabilities. This serves as the foundation
for all specialized crawlers in the Lindela system.

Key Features:
- Automatic resource cleanup and connection pooling
- Advanced rate limiting with token bucket algorithm
- Circuit breaker pattern for fault tolerance
- Comprehensive metrics and performance monitoring
- Memory-efficient streaming for large responses
- Retry logic with exponential backoff
- Request deduplication and caching
- Concurrent request management with semaphores
- Unified configuration system integration

Architecture:
- Uses asyncio for high-performance concurrent operations
- Implements context managers for resource safety
- Provides hooks for extensibility and customization
- Supports both streaming and buffered responses
- Integrates with unified configuration management

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Version: 2.1.0
License: MIT
"""

import asyncio
import aiohttp
import logging
import time
import hashlib
import json
from typing import Dict, List, Optional, Any, Callable, Union, AsyncIterator, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from urllib.parse import urlparse, urljoin
from collections import deque, defaultdict
from enum import Enum
import weakref
from contextlib import asynccontextmanager
import pickle
from pathlib import Path
import aiofiles
import ssl
from abc import ABC, abstractmethod

# Import unified configuration system
try:
    from ..utils.config import (
        UnifiedCrawlerConfiguration,
        CrawlerPerformanceConfiguration,
        CrawlerType
    )
    UNIFIED_CONFIG_AVAILABLE = True
except ImportError:
    UNIFIED_CONFIG_AVAILABLE = False
    UnifiedCrawlerConfiguration = None

# Configure logging
logger = logging.getLogger(__name__)


class RequestPriority(Enum):
    """Priority levels for request scheduling."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


class ResponseType(Enum):
    """Types of responses the crawler can handle."""
    HTML = "html"
    JSON = "json"
    XML = "xml"
    BINARY = "binary"
    TEXT = "text"


@dataclass
class CrawlRequest:
    """Represents a crawl request with metadata."""
    url: str
    method: str = "GET"
    headers: Optional[Dict[str, str]] = None
    data: Optional[Union[str, bytes, Dict]] = None
    params: Optional[Dict[str, str]] = None
    priority: RequestPriority = RequestPriority.NORMAL
    retry_count: int = 0
    max_retries: int = 3
    timeout: Optional[float] = None
    callback: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def __hash__(self):
        """Make request hashable for deduplication."""
        return hash((self.url, self.method, json.dumps(self.params or {})))


@dataclass
class CrawlResponse:
    """Represents a crawl response with metadata."""
    request: CrawlRequest
    status_code: int
    headers: Dict[str, str]
    content: Optional[Union[str, bytes]] = None
    content_type: Optional[str] = None
    response_type: ResponseType = ResponseType.HTML
    encoding: Optional[str] = None
    response_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success(self) -> bool:
        """Check if response was successful."""
        return 200 <= self.status_code < 300 and self.error is None
    
    def json(self) -> Optional[Dict]:
        """Parse response as JSON."""
        if self.content and isinstance(self.content, str):
            try:
                return json.loads(self.content)
            except json.JSONDecodeError:
                return None
        return None


class RateLimiter:
    """Token bucket rate limiter for controlling request rate."""
    
    def __init__(self, rate: float, burst: int = 1):
        """
        Initialize rate limiter.
        
        Args:
            rate: Requests per second
            burst: Maximum burst size
        """
        self.rate = rate
        self.burst = burst
        self.tokens = burst
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> float:
        """
        Acquire tokens, waiting if necessary.
        
        Returns:
            Wait time in seconds
        """
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
            self.last_update = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return 0.0
            
            wait_time = (tokens - self.tokens) / self.rate
            await asyncio.sleep(wait_time)
            self.tokens = 0
            return wait_time


class CircuitBreaker:
    """Circuit breaker for handling failures gracefully."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "closed"  # closed, open, half-open
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        async with self._lock:
            if self.state == "open":
                if self._should_attempt_reset():
                    self.state = "half-open"
                else:
                    raise Exception(f"Circuit breaker is open (failures: {self.failure_count})")
        
        try:
            result = await func(*args, **kwargs)
            async with self._lock:
                self._on_success()
            return result
        except self.expected_exception as e:
            async with self._lock:
                self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset."""
        return (
            self.last_failure_time is not None and
            time.time() - self.last_failure_time >= self.recovery_timeout
        )
    
    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        self.state = "closed"
        self.last_failure_time = None
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "open"


class RequestDeduplicator:
    """Deduplicates requests to avoid redundant crawling."""
    
    def __init__(self, cache_size: int = 10000, ttl: float = 3600):
        self.cache_size = cache_size
        self.ttl = ttl
        self.seen: Dict[int, float] = {}
        self._cleanup_interval = 300  # 5 minutes
        self._cleanup_task: Optional[asyncio.Task] = None
    
    def is_duplicate(self, request: CrawlRequest) -> bool:
        """Check if request is a duplicate."""
        req_hash = hash(request)
        now = time.time()
        
        if req_hash in self.seen:
            if now - self.seen[req_hash] < self.ttl:
                return True
            else:
                del self.seen[req_hash]
        
        # Add to seen
        self.seen[req_hash] = now
        
        # Enforce cache size limit
        if len(self.seen) > self.cache_size:
            # Remove oldest entries
            sorted_items = sorted(self.seen.items(), key=lambda x: x[1])
            for key, _ in sorted_items[:len(self.seen) - self.cache_size]:
                del self.seen[key]
        
        return False
    
    async def start_cleanup(self):
        """Start periodic cleanup of expired entries."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def stop_cleanup(self):
        """Stop cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
    
    async def _cleanup_loop(self):
        """Periodically clean up expired entries."""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)
                now = time.time()
                expired = [k for k, v in self.seen.items() if now - v > self.ttl]
                for key in expired:
                    del self.seen[key]
                logger.debug(f"Cleaned up {len(expired)} expired request hashes")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in deduplicator cleanup: {e}")


class ResourcePool:
    """Generic resource pool for managing limited resources."""
    
    def __init__(self, factory: Callable, max_size: int = 10, min_size: int = 1):
        self.factory = factory
        self.max_size = max_size
        self.min_size = min_size
        self.pool: deque = deque()
        self.in_use: Set = set()
        self._lock = asyncio.Lock()
        self._created = 0
        self._stats = {
            'created': 0,
            'destroyed': 0,
            'acquired': 0,
            'released': 0,
            'reused': 0
        }
    
    async def acquire(self):
        """Acquire a resource from the pool."""
        async with self._lock:
            # Try to get from pool
            while self.pool:
                resource = self.pool.popleft()
                if await self._validate_resource(resource):
                    self.in_use.add(id(resource))
                    self._stats['acquired'] += 1
                    self._stats['reused'] += 1
                    return resource
                else:
                    await self._destroy_resource(resource)
            
            # Create new resource if under limit
            if self._created < self.max_size:
                resource = await self.factory()
                self._created += 1
                self.in_use.add(id(resource))
                self._stats['created'] += 1
                self._stats['acquired'] += 1
                return resource
            
            # Wait for resource to be released
            while not self.pool:
                await asyncio.sleep(0.1)
            
            return await self.acquire()
    
    async def release(self, resource):
        """Release a resource back to the pool."""
        async with self._lock:
            resource_id = id(resource)
            if resource_id in self.in_use:
                self.in_use.remove(resource_id)
                self._stats['released'] += 1
                
                if len(self.pool) < self.max_size:
                    self.pool.append(resource)
                else:
                    await self._destroy_resource(resource)
    
    async def _validate_resource(self, resource) -> bool:
        """Validate if resource is still usable."""
        # Override in subclass for specific validation
        return True
    
    async def _destroy_resource(self, resource):
        """Destroy a resource."""
        # Override in subclass for cleanup
        self._created -= 1
        self._stats['destroyed'] += 1
    
    async def close_all(self):
        """Close all resources in the pool."""
        async with self._lock:
            # Destroy pooled resources
            while self.pool:
                resource = self.pool.popleft()
                await self._destroy_resource(resource)
            
            # Note: in_use resources should be released by their users
            if self.in_use:
                logger.warning(f"{len(self.in_use)} resources still in use during pool closure")
    
    def get_stats(self) -> Dict[str, int]:
        """Get pool statistics."""
        return {
            **self._stats,
            'pool_size': len(self.pool),
            'in_use': len(self.in_use),
            'total_created': self._created
        }


class ConnectionPool(ResourcePool):
    """HTTP connection pool using aiohttp."""
    
    def __init__(self, max_size: int = 100, timeout: int = 30):
        self.timeout = timeout
        super().__init__(self._create_connector, max_size)
    
    async def _create_connector(self) -> aiohttp.TCPConnector:
        """Create a new connector."""
        return aiohttp.TCPConnector(
            limit=100,
            limit_per_host=10,
            ttl_dns_cache=300,
            enable_cleanup_closed=True,
            force_close=True,
            ssl=ssl.create_default_context()
        )
    
    async def _validate_resource(self, connector: aiohttp.TCPConnector) -> bool:
        """Validate connector is still usable."""
        return not connector.closed
    
    async def _destroy_resource(self, connector: aiohttp.TCPConnector):
        """Close connector."""
        await super()._destroy_resource(connector)
        if not connector.closed:
            await connector.close()


class BaseCrawler(ABC):
    """
    Abstract base crawler with advanced resource management.
    
    This class provides:
    - Connection pooling
    - Rate limiting
    - Circuit breaker
    - Request deduplication
    - Retry logic
    - Performance monitoring
    - Resource cleanup
    """
    
    def __init__(
        self,
        rate_limit: float = 10.0,
        max_concurrent: int = 10,
        timeout: int = 30,
        max_retries: int = 3,
        user_agent: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        cache_responses: bool = False,
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize base crawler.
        
        Args:
            rate_limit: Requests per second
            max_concurrent: Maximum concurrent requests
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            user_agent: Custom user agent
            headers: Default headers
            cache_responses: Whether to cache responses
            cache_dir: Directory for response cache
        """
        self.rate_limit = rate_limit
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache_responses = cache_responses
        self.cache_dir = cache_dir or Path.home() / ".crawler_cache"
        
        # Default headers
        self.default_headers = {
            'User-Agent': user_agent or 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        if headers:
            self.default_headers.update(headers)
        
        # Resource management
        self.rate_limiter = RateLimiter(rate_limit, burst=int(rate_limit * 2))
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.connection_pool = ConnectionPool(max_size=max_concurrent * 2)
        self.deduplicator = RequestDeduplicator()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Request queue
        self.request_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.active_requests: Set[int] = set()
        
        # Metrics
        self.metrics = defaultdict(int)
        self.domain_metrics = defaultdict(lambda: defaultdict(int))
        self.start_time = time.time()
        
        # Session management
        self._session: Optional[aiohttp.ClientSession] = None
        self._workers: List[asyncio.Task] = []
        self._running = False
        
        # Cache setup
        if self.cache_responses:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def start(self):
        """Start the crawler."""
        if self._running:
            return
        
        self._running = True
        
        # Start deduplicator cleanup
        await self.deduplicator.start_cleanup()
        
        # Start worker tasks
        for i in range(min(self.max_concurrent, 5)):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self._workers.append(worker)
        
        logger.info(f"Crawler started with {len(self._workers)} workers")
    
    async def close(self):
        """Close the crawler and cleanup resources."""
        self._running = False
        
        # Cancel workers
        for worker in self._workers:
            worker.cancel()
        
        # Wait for workers to finish
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        
        # Stop deduplicator cleanup
        await self.deduplicator.stop_cleanup()
        
        # Close session
        if self._session and not self._session.closed:
            await self._session.close()
        
        # Close connection pool
        await self.connection_pool.close_all()
        
        # Log final metrics
        self._log_metrics()
        
        logger.info("Crawler closed")
    
    async def crawl(self, request: Union[str, CrawlRequest]) -> CrawlResponse:
        """
        Crawl a single URL or request.
        
        Args:
            request: URL string or CrawlRequest object
            
        Returns:
            CrawlResponse object
        """
        if isinstance(request, str):
            request = CrawlRequest(url=request)
        
        # Check for duplicates
        if self.deduplicator.is_duplicate(request):
            self.metrics['duplicates_skipped'] += 1
            return CrawlResponse(
                request=request,
                status_code=0,
                headers={},
                error="Duplicate request skipped"
            )
        
        # Check cache
        if self.cache_responses:
            cached = await self._get_cached_response(request)
            if cached:
                self.metrics['cache_hits'] += 1
                return cached
        
        # Add to queue
        await self.request_queue.put((request.priority.value, id(request), request))
        
        # Wait for result
        while id(request) in self.active_requests:
            await asyncio.sleep(0.1)
        
        # Get result (this is simplified, in practice you'd use a result store)
        # For now, perform the request directly
        return await self._perform_request(request)
    
    async def crawl_many(
        self,
        requests: List[Union[str, CrawlRequest]],
        return_exceptions: bool = False
    ) -> List[Union[CrawlResponse, Exception]]:
        """
        Crawl multiple URLs concurrently.
        
        Args:
            requests: List of URLs or CrawlRequest objects
            return_exceptions: Whether to return exceptions instead of raising
            
        Returns:
            List of CrawlResponse objects or exceptions
        """
        tasks = []
        for request in requests:
            if isinstance(request, str):
                request = CrawlRequest(url=request)
            tasks.append(self.crawl(request))
        
        return await asyncio.gather(*tasks, return_exceptions=return_exceptions)
    
    async def stream_response(
        self,
        request: Union[str, CrawlRequest],
        chunk_size: int = 8192
    ) -> AsyncIterator[bytes]:
        """
        Stream response content in chunks.
        
        Args:
            request: URL or CrawlRequest
            chunk_size: Size of chunks to yield
            
        Yields:
            Bytes chunks
        """
        if isinstance(request, str):
            request = CrawlRequest(url=request)
        
        async with self.semaphore:
            await self.rate_limiter.acquire()
            
            connector = await self.connection_pool.acquire()
            session = await self._get_session(connector)
            
            try:
                async with session.request(
                    request.method,
                    request.url,
                    headers=self._merge_headers(request.headers),
                    data=request.data,
                    params=request.params,
                    timeout=aiohttp.ClientTimeout(total=request.timeout or self.timeout)
                ) as response:
                    async for chunk in response.content.iter_chunked(chunk_size):
                        yield chunk
            finally:
                await self.connection_pool.release(connector)
    
    async def _worker(self, name: str):
        """Worker task that processes requests from the queue."""
        logger.debug(f"Worker {name} started")
        
        while self._running:
            try:
                # Get request from queue
                priority, req_id, request = await asyncio.wait_for(
                    self.request_queue.get(),
                    timeout=1.0
                )
                
                self.active_requests.add(req_id)
                
                try:
                    # Process request
                    await self._perform_request(request)
                finally:
                    self.active_requests.remove(req_id)
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Worker {name} error: {e}")
        
        logger.debug(f"Worker {name} stopped")
    
    async def _perform_request(self, request: CrawlRequest) -> CrawlResponse:
        """Perform actual HTTP request with retries."""
        domain = urlparse(request.url).netloc
        
        # Get or create circuit breaker for domain
        if domain not in self.circuit_breakers:
            self.circuit_breakers[domain] = CircuitBreaker()
        
        circuit_breaker = self.circuit_breakers[domain]
        
        for attempt in range(request.max_retries + 1):
            try:
                # Use circuit breaker
                response = await circuit_breaker.call(
                    self._execute_request,
                    request
                )
                
                # Cache successful response
                if self.cache_responses and response.success:
                    await self._cache_response(response)
                
                return response
                
            except Exception as e:
                if attempt == request.max_retries:
                    self.metrics['failed_requests'] += 1
                    self.domain_metrics[domain]['failures'] += 1
                    
                    return CrawlResponse(
                        request=request,
                        status_code=0,
                        headers={},
                        error=str(e)
                    )
                
                # Exponential backoff
                wait_time = min(2 ** attempt, 60)
                logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)
    
    async def _execute_request(self, request: CrawlRequest) -> CrawlResponse:
        """Execute single HTTP request."""
        start_time = time.time()
        
        async with self.semaphore:
            # Rate limiting
            wait_time = await self.rate_limiter.acquire()
            if wait_time > 0:
                self.metrics['rate_limit_waits'] += 1
            
            connector = await self.connection_pool.acquire()
            session = await self._get_session(connector)
            
            try:
                # Perform request
                async with session.request(
                    request.method,
                    request.url,
                    headers=self._merge_headers(request.headers),
                    data=request.data,
                    params=request.params,
                    timeout=aiohttp.ClientTimeout(total=request.timeout or self.timeout),
                    allow_redirects=True
                ) as response:
                    # Read content
                    content = await response.read()
                    
                    # Detect encoding
                    encoding = response.charset or 'utf-8'
                    
                    # Decode if text
                    content_type = response.headers.get('Content-Type', '')
                    if 'text' in content_type or 'json' in content_type or 'xml' in content_type:
                        try:
                            content = content.decode(encoding)
                            response_type = self._detect_response_type(content_type, content)
                        except UnicodeDecodeError:
                            response_type = ResponseType.BINARY
                    else:
                        response_type = ResponseType.BINARY
                    
                    # Update metrics
                    response_time = time.time() - start_time
                    self.metrics['total_requests'] += 1
                    self.metrics['total_bytes'] += len(content)
                    self.metrics['total_response_time'] += response_time
                    
                    domain = urlparse(request.url).netloc
                    self.domain_metrics[domain]['requests'] += 1
                    self.domain_metrics[domain]['bytes'] += len(content)
                    
                    # Process response through hook
                    response_obj = CrawlResponse(
                        request=request,
                        status_code=response.status,
                        headers=dict(response.headers),
                        content=content,
                        content_type=content_type,
                        response_type=response_type,
                        encoding=encoding,
                        response_time=response_time
                    )
                    
                    # Call response processing hook
                    processed = await self.process_response(response_obj)
                    
                    return processed
                    
            except asyncio.TimeoutError:
                self.metrics['timeouts'] += 1
                raise Exception(f"Request timeout after {request.timeout or self.timeout}s")
            except aiohttp.ClientError as e:
                self.metrics['client_errors'] += 1
                raise Exception(f"Client error: {e}")
            except Exception as e:
                self.metrics['other_errors'] += 1
                raise
            finally:
                await self.connection_pool.release(connector)
    
    async def _get_session(self, connector: aiohttp.TCPConnector) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                connector=connector,
                connector_owner=False
            )
        return self._session
    
    def _merge_headers(self, custom_headers: Optional[Dict[str, str]]) -> Dict[str, str]:
        """Merge custom headers with defaults."""
        headers = self.default_headers.copy()
        if custom_headers:
            headers.update(custom_headers)
        return headers
    
    def _detect_response_type(self, content_type: str, content: str) -> ResponseType:
        """Detect response type from content type and content."""
        content_type = content_type.lower()
        
        if 'json' in content_type:
            return ResponseType.JSON
        elif 'xml' in content_type:
            return ResponseType.XML
        elif 'html' in content_type:
            return ResponseType.HTML
        
        # Try to detect from content
        content_start = content[:100].strip()
        if content_start.startswith('{') or content_start.startswith('['):
            return ResponseType.JSON
        elif content_start.startswith('<?xml') or content_start.startswith('<xml'):
            return ResponseType.XML
        elif content_start.startswith('<!DOCTYPE') or content_start.startswith('<html'):
            return ResponseType.HTML
        
        return ResponseType.TEXT
    
    async def _get_cached_response(self, request: CrawlRequest) -> Optional[CrawlResponse]:
        """Get cached response if available."""
        cache_key = hashlib.md5(
            f"{request.method}:{request.url}:{json.dumps(request.params or {})}".encode()
        ).hexdigest()
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            try:
                async with aiofiles.open(cache_file, 'rb') as f:
                    data = await f.read()
                    return pickle.loads(data)
            except Exception as e:
                logger.error(f"Failed to load cached response: {e}")
        
        return None
    
    async def _cache_response(self, response: CrawlResponse):
        """Cache response to disk."""
        cache_key = hashlib.md5(
            f"{response.request.method}:{response.request.url}:{json.dumps(response.request.params or {})}".encode()
        ).hexdigest()
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            async with aiofiles.open(cache_file, 'wb') as f:
                await f.write(pickle.dumps(response))
        except Exception as e:
            logger.error(f"Failed to cache response: {e}")
    
    def _log_metrics(self):
        """Log crawler metrics."""
        elapsed = time.time() - self.start_time
        total_requests = self.metrics['total_requests']
        
        if total_requests > 0:
            logger.info(f"Crawler metrics after {elapsed:.2f}s:")
            logger.info(f"  Total requests: {total_requests}")
            logger.info(f"  Success rate: {(total_requests - self.metrics['failed_requests']) / total_requests * 100:.2f}%")
            logger.info(f"  Average response time: {self.metrics['total_response_time'] / total_requests:.2f}s")
            logger.info(f"  Total data: {self.metrics['total_bytes'] / (1024*1024):.2f} MB")
            logger.info(f"  Requests/second: {total_requests / elapsed:.2f}")
            logger.info(f"  Cache hits: {self.metrics.get('cache_hits', 0)}")
            logger.info(f"  Duplicates skipped: {self.metrics.get('duplicates_skipped', 0)}")
            
            # Domain statistics
            logger.info("Domain statistics:")
            for domain, stats in sorted(self.domain_metrics.items(), 
                                       key=lambda x: x[1]['requests'], 
                                       reverse=True)[:10]:
                logger.info(f"  {domain}: {stats['requests']} requests, "
                          f"{stats['bytes'] / (1024*1024):.2f} MB, "
                          f"{stats.get('failures', 0)} failures")
    
    @abstractmethod
    async def process_response(self, response: CrawlResponse) -> CrawlResponse:
        """
        Process response before returning.
        
        This is a hook for subclasses to implement custom processing logic.
        
        Args:
            response: The raw response
            
        Returns:
            Processed response
        """
        pass
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        elapsed = time.time() - self.start_time
        total_requests = self.metrics['total_requests']
        
        return {
            'elapsed_time': elapsed,
            'total_requests': total_requests,
            'failed_requests': self.metrics.get('failed_requests', 0),
            'success_rate': (total_requests - self.metrics.get('failed_requests', 0)) / total_requests * 100 if total_requests > 0 else 0,
            'average_response_time': self.metrics['total_response_time'] / total_requests if total_requests > 0 else 0,
            'total_bytes': self.metrics.get('total_bytes', 0),
            'requests_per_second': total_requests / elapsed if elapsed > 0 else 0,
            'cache_hits': self.metrics.get('cache_hits', 0),
            'duplicates_skipped': self.metrics.get('duplicates_skipped', 0),
            'timeouts': self.metrics.get('timeouts', 0),
            'rate_limit_waits': self.metrics.get('rate_limit_waits', 0),
            'connection_pool_stats': self.connection_pool.get_stats(),
            'domain_count': len(self.domain_metrics)
        }


class SimpleCrawler(BaseCrawler):
    """Simple crawler implementation for basic web crawling."""
    
    async def process_response(self, response: CrawlResponse) -> CrawlResponse:
        """Simple pass-through processing."""
        return response


# Example usage
async def example_usage():
    """Example of how to use the base crawler."""
    
    # Create crawler instance
    async with SimpleCrawler(
        rate_limit=10.0,
        max_concurrent=5,
        timeout=30,
        cache_responses=True
    ) as crawler:
        
        # Crawl single URL
        response = await crawler.crawl("https://example.com")
        print(f"Status: {response.status_code}")
        print(f"Content length: {len(response.content) if response.content else 0}")
        
        # Crawl multiple URLs
        urls = [
            "https://example.com/page1",
            "https://example.com/page2",
            "https://example.com/page3"
        ]
        
        responses = await crawler.crawl_many(urls)
        for response in responses:
            if isinstance(response, CrawlResponse):
                print(f"{response.request.url}: {response.status_code}")
            else:
                print(f"Error: {response}")
        
        # Stream large file
        async for chunk in crawler.stream_response("https://example.com/largefile.zip"):
            # Process chunk
            pass
        
        # Get metrics
        metrics = crawler.get_metrics()
        print(f"Total requests: {metrics['total_requests']}")
        print(f"Success rate: {metrics['success_rate']:.2f}%")


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())