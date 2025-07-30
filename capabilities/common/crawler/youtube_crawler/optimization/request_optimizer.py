"""
Request Optimizer Module
========================

Advanced request optimization for YouTube API with connection pooling,
retry mechanisms, and intelligent request management.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
"""

import asyncio
import logging
import time
import random
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import aiohttp
import json

logger = logging.getLogger(__name__)


class RetryStrategy(Enum):
    """Retry strategy types."""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    ADAPTIVE = "adaptive"


@dataclass
class RequestConfig:
    """Configuration for request optimization."""
    max_concurrent_requests: int = 10
    connection_timeout: float = 30.0
    read_timeout: float = 60.0
    total_timeout: float = 300.0
    max_retries: int = 3
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    base_delay: float = 1.0
    max_delay: float = 60.0
    jitter: bool = True
    enable_connection_pooling: bool = True
    pool_size: int = 100
    pool_ttl: int = 300
    enable_compression: bool = True
    user_agent: str = "YouTube-Crawler/1.0"


@dataclass
class RequestResult:
    """Result of a request operation."""
    success: bool
    data: Optional[Any] = None
    status_code: Optional[int] = None
    error: Optional[str] = None
    retry_count: int = 0
    total_time: float = 0.0
    response_time: float = 0.0
    from_cache: bool = False


@dataclass
class ConnectionStats:
    """Connection pool statistics."""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    created_connections: int = 0
    closed_connections: int = 0
    connection_errors: int = 0
    pool_hits: int = 0
    pool_misses: int = 0


class ConnectionPool:
    """Advanced connection pool for HTTP requests."""
    
    def __init__(self, config: RequestConfig):
        self.config = config
        self.connector = None
        self.session = None
        self.stats = ConnectionStats()
        self.created_at = time.time()
        self.last_cleanup = time.time()
        self._lock = asyncio.Lock()
    
    async def initialize(self):
        """Initialize the connection pool."""
        if self.session is None:
            connector_kwargs = {
                'limit': self.config.pool_size,
                'limit_per_host': min(self.config.pool_size, 30),
                'ttl_dns_cache': 300,
                'use_dns_cache': True,
                'enable_cleanup_closed': True,
            }
            
            if self.config.enable_compression:
                connector_kwargs['enable_cleanup_closed'] = True
            
            self.connector = aiohttp.TCPConnector(**connector_kwargs)
            
            timeout = aiohttp.ClientTimeout(
                total=self.config.total_timeout,
                connect=self.config.connection_timeout,
                sock_read=self.config.read_timeout
            )
            
            headers = {
                'User-Agent': self.config.user_agent,
                'Accept-Encoding': 'gzip, deflate' if self.config.enable_compression else 'identity'
            }
            
            self.session = aiohttp.ClientSession(
                connector=self.connector,
                timeout=timeout,
                headers=headers,
                trust_env=True
            )
            
            self.stats.created_connections += 1
            logger.debug("Connection pool initialized")
    
    async def close(self):
        """Close the connection pool."""
        if self.session:
            await self.session.close()
            self.session = None
            self.stats.closed_connections += 1
        
        if self.connector:
            await self.connector.close()
            self.connector = None
    
    async def get_session(self) -> aiohttp.ClientSession:
        """Get the session, initializing if necessary."""
        if self.session is None or self.session.closed:
            async with self._lock:
                if self.session is None or self.session.closed:
                    await self.initialize()
        
        return self.session
    
    async def cleanup_if_needed(self):
        """Cleanup connections if TTL expired."""
        current_time = time.time()
        if current_time - self.last_cleanup > self.config.pool_ttl:
            async with self._lock:
                if self.session and hasattr(self.connector, '_conns'):
                    # Force cleanup of idle connections
                    for conns in self.connector._conns.values():
                        expired_conns = []
                        for conn in conns:
                            if hasattr(conn, '_created') and current_time - conn._created > self.config.pool_ttl:
                                expired_conns.append(conn)
                        
                        for conn in expired_conns:
                            conns.remove(conn)
                            conn.close()
                
                self.last_cleanup = current_time
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        stats_dict = {
            'total_connections': self.stats.created_connections,
            'connection_errors': self.stats.connection_errors,
            'pool_age': time.time() - self.created_at,
            'last_cleanup': time.time() - self.last_cleanup
        }
        
        if self.connector and hasattr(self.connector, '_conns'):
            active = sum(len(conns) for conns in self.connector._conns.values())
            stats_dict.update({
                'active_connections': active,
                'pool_limit': self.config.pool_size
            })
        
        return stats_dict


class RetryManager:
    """Advanced retry management with multiple strategies."""
    
    def __init__(self, config: RequestConfig):
        self.config = config
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'retry_attempts': 0,
            'total_retry_time': 0.0,
            'retry_distribution': [0] * (config.max_retries + 1)
        }
    
    async def execute_with_retry(self, request_func: Callable, *args, **kwargs) -> RequestResult:
        """Execute a request with retry logic."""
        start_time = time.time()
        last_error = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                # Calculate delay for this attempt
                if attempt > 0:
                    delay = self._calculate_delay(attempt)
                    logger.debug(f"Retrying request after {delay:.2f}s (attempt {attempt + 1})")
                    await asyncio.sleep(delay)
                    self.stats['retry_attempts'] += 1
                    self.stats['total_retry_time'] += delay
                
                # Execute the request
                request_start = time.time()
                result = await request_func(*args, **kwargs)
                request_time = time.time() - request_start
                
                # Success
                total_time = time.time() - start_time
                self.stats['successful_requests'] += 1
                self.stats['retry_distribution'][attempt] += 1
                
                return RequestResult(
                    success=True,
                    data=result,
                    retry_count=attempt,
                    total_time=total_time,
                    response_time=request_time
                )
                
            except Exception as e:
                last_error = e
                
                # Check if we should retry this error
                if not self._should_retry(e, attempt):
                    break
                
                logger.debug(f"Request failed (attempt {attempt + 1}): {e}")
        
        # All retries exhausted
        total_time = time.time() - start_time
        self.stats['failed_requests'] += 1
        self.stats['retry_distribution'][self.config.max_retries] += 1
        
        return RequestResult(
            success=False,
            error=str(last_error),
            retry_count=self.config.max_retries,
            total_time=total_time
        )
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt based on strategy."""
        if self.config.retry_strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.config.base_delay * (2 ** (attempt - 1))
        elif self.config.retry_strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.config.base_delay * attempt
        elif self.config.retry_strategy == RetryStrategy.FIXED_DELAY:
            delay = self.config.base_delay
        elif self.config.retry_strategy == RetryStrategy.ADAPTIVE:
            # Adaptive based on recent success rate
            success_rate = self._get_recent_success_rate()
            multiplier = 2.0 if success_rate < 0.5 else 1.0
            delay = self.config.base_delay * (multiplier ** (attempt - 1))
        else:
            delay = self.config.base_delay
        
        # Apply jitter to avoid thundering herd
        if self.config.jitter:
            jitter_factor = random.uniform(0.5, 1.5)
            delay *= jitter_factor
        
        return min(delay, self.config.max_delay)
    
    def _should_retry(self, error: Exception, attempt: int) -> bool:
        """Determine if an error should trigger a retry."""
        if attempt >= self.config.max_retries:
            return False
        
        # Retry on network errors, timeouts, and server errors
        retry_conditions = [
            isinstance(error, (aiohttp.ClientError, asyncio.TimeoutError)),
            hasattr(error, 'status') and 500 <= error.status < 600,
            hasattr(error, 'status') and error.status == 429,  # Rate limit
            hasattr(error, 'status') and error.status == 503,  # Service unavailable
        ]
        
        return any(retry_conditions)
    
    def _get_recent_success_rate(self) -> float:
        """Calculate recent success rate for adaptive strategy."""
        total = self.stats['successful_requests'] + self.stats['failed_requests']
        if total == 0:
            return 1.0
        return self.stats['successful_requests'] / total
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retry manager statistics."""
        total_requests = self.stats['successful_requests'] + self.stats['failed_requests']
        
        return {
            'total_requests': total_requests,
            'success_rate': (self.stats['successful_requests'] / total_requests) if total_requests > 0 else 0.0,
            'retry_rate': (self.stats['retry_attempts'] / total_requests) if total_requests > 0 else 0.0,
            'avg_retry_time': (self.stats['total_retry_time'] / self.stats['retry_attempts']) if self.stats['retry_attempts'] > 0 else 0.0,
            'retry_distribution': self.stats['retry_distribution'].copy(),
            'config': {
                'max_retries': self.config.max_retries,
                'strategy': self.config.retry_strategy.value,
                'base_delay': self.config.base_delay,
                'max_delay': self.config.max_delay
            }
        }
    
    def reset_stats(self):
        """Reset retry statistics."""
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'retry_attempts': 0,
            'total_retry_time': 0.0,
            'retry_distribution': [0] * (self.config.max_retries + 1)
        }


class RequestOptimizer:
    """Main request optimizer with connection pooling and retry management."""
    
    def __init__(self, config: Optional[RequestConfig] = None):
        self.config = config or RequestConfig()
        self.connection_pool = ConnectionPool(self.config)
        self.retry_manager = RetryManager(self.config)
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
        # Request statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_response_time': 0.0,
            'requests_by_method': {},
            'responses_by_status': {},
            'concurrent_requests': 0,
            'max_concurrent_reached': 0
        }
        
        self._active_requests = 0
        self._lock = asyncio.Lock()
    
    async def initialize(self):
        """Initialize the request optimizer."""
        await self.connection_pool.initialize()
    
    async def close(self):
        """Close the request optimizer."""
        await self.connection_pool.close()
    
    async def get(self, url: str, params: Optional[Dict] = None, 
                  headers: Optional[Dict] = None, **kwargs) -> RequestResult:
        """Optimized GET request."""
        return await self._make_request('GET', url, params=params, headers=headers, **kwargs)
    
    async def post(self, url: str, data: Optional[Any] = None, json_data: Optional[Dict] = None,
                   headers: Optional[Dict] = None, **kwargs) -> RequestResult:
        """Optimized POST request."""
        return await self._make_request('POST', url, data=data, json=json_data, headers=headers, **kwargs)
    
    async def put(self, url: str, data: Optional[Any] = None, json_data: Optional[Dict] = None,
                  headers: Optional[Dict] = None, **kwargs) -> RequestResult:
        """Optimized PUT request."""
        return await self._make_request('PUT', url, data=data, json=json_data, headers=headers, **kwargs)
    
    async def delete(self, url: str, headers: Optional[Dict] = None, **kwargs) -> RequestResult:
        """Optimized DELETE request."""
        return await self._make_request('DELETE', url, headers=headers, **kwargs)
    
    async def batch_requests(self, requests: List[Tuple[str, str, Dict]]) -> List[RequestResult]:
        """Execute multiple requests concurrently."""
        tasks = []
        
        for method, url, kwargs in requests:
            task = asyncio.create_task(self._make_request(method, url, **kwargs))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to failed RequestResults
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append(RequestResult(
                    success=False,
                    error=str(result)
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _make_request(self, method: str, url: str, **kwargs) -> RequestResult:
        """Make an optimized HTTP request with retry logic."""
        async with self.semaphore:
            async with self._lock:
                self._active_requests += 1
                self.stats['concurrent_requests'] = self._active_requests
                self.stats['max_concurrent_reached'] = max(
                    self.stats['max_concurrent_reached'],
                    self._active_requests
                )
            
            try:
                # Cleanup connections if needed
                await self.connection_pool.cleanup_if_needed()
                
                # Execute request with retry
                result = await self.retry_manager.execute_with_retry(
                    self._execute_single_request,
                    method, url, **kwargs
                )
                
                # Update statistics
                await self._update_stats(method, result)
                
                return result
                
            finally:
                async with self._lock:
                    self._active_requests -= 1
                    self.stats['concurrent_requests'] = self._active_requests
    
    async def _execute_single_request(self, method: str, url: str, **kwargs) -> Any:
        """Execute a single HTTP request."""
        session = await self.connection_pool.get_session()
        
        # Prepare request arguments
        request_kwargs = kwargs.copy()
        
        # Handle JSON data
        if 'json' in request_kwargs:
            request_kwargs['json'] = request_kwargs.pop('json')
        
        # Make the request
        async with session.request(method, url, **request_kwargs) as response:
            # Check for HTTP errors
            if response.status >= 400:
                error_text = await response.text()
                error = aiohttp.ClientResponseError(
                    request_info=response.request_info,
                    history=response.history,
                    status=response.status,
                    message=f"HTTP {response.status}: {error_text[:200]}"
                )
                error.status = response.status
                raise error
            
            # Parse response based on content type
            content_type = response.headers.get('content-type', '').lower()
            
            if 'application/json' in content_type:
                data = await response.json()
            elif 'text/' in content_type:
                data = await response.text()
            else:
                data = await response.read()
            
            return {
                'data': data,
                'status': response.status,
                'headers': dict(response.headers),
                'url': str(response.url)
            }
    
    async def _update_stats(self, method: str, result: RequestResult):
        """Update request statistics."""
        async with self._lock:
            self.stats['total_requests'] += 1
            
            if result.success:
                self.stats['successful_requests'] += 1
                self.stats['total_response_time'] += result.response_time
            else:
                self.stats['failed_requests'] += 1
            
            # Track by method
            self.stats['requests_by_method'][method] = \
                self.stats['requests_by_method'].get(method, 0) + 1
            
            # Track by status code
            if result.status_code:
                self.stats['responses_by_status'][result.status_code] = \
                    self.stats['responses_by_status'].get(result.status_code, 0) + 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive request optimizer statistics."""
        total_requests = self.stats['total_requests']
        successful_requests = self.stats['successful_requests']
        
        optimizer_stats = {
            'request_stats': {
                'total_requests': total_requests,
                'successful_requests': successful_requests,
                'failed_requests': self.stats['failed_requests'],
                'success_rate': (successful_requests / total_requests) if total_requests > 0 else 0.0,
                'avg_response_time': (self.stats['total_response_time'] / successful_requests) if successful_requests > 0 else 0.0,
                'current_concurrent': self.stats['concurrent_requests'],
                'max_concurrent_reached': self.stats['max_concurrent_reached'],
                'requests_by_method': dict(self.stats['requests_by_method']),
                'responses_by_status': dict(self.stats['responses_by_status'])
            },
            'connection_pool': self.connection_pool.get_stats(),
            'retry_manager': self.retry_manager.get_stats(),
            'config': {
                'max_concurrent': self.config.max_concurrent_requests,
                'connection_timeout': self.config.connection_timeout,
                'read_timeout': self.config.read_timeout,
                'total_timeout': self.config.total_timeout,
                'max_retries': self.config.max_retries,
                'retry_strategy': self.config.retry_strategy.value
            }
        }
        
        return optimizer_stats
    
    def reset_stats(self):
        """Reset all statistics."""
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_response_time': 0.0,
            'requests_by_method': {},
            'responses_by_status': {},
            'concurrent_requests': 0,
            'max_concurrent_reached': 0
        }
        self.retry_manager.reset_stats()
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Utility functions
async def create_optimized_session(config: Optional[RequestConfig] = None) -> RequestOptimizer:
    """Create and initialize an optimized request session."""
    optimizer = RequestOptimizer(config)
    await optimizer.initialize()
    return optimizer


def create_request_config(preset: str = 'balanced') -> RequestConfig:
    """Create request configuration with predefined presets."""
    presets = {
        'conservative': RequestConfig(
            max_concurrent_requests=3,
            connection_timeout=30.0,
            read_timeout=60.0,
            max_retries=5,
            base_delay=2.0,
            max_delay=120.0
        ),
        'balanced': RequestConfig(
            max_concurrent_requests=10,
            connection_timeout=30.0,
            read_timeout=60.0,
            max_retries=3,
            base_delay=1.0,
            max_delay=60.0
        ),
        'aggressive': RequestConfig(
            max_concurrent_requests=20,
            connection_timeout=15.0,
            read_timeout=30.0,
            max_retries=2,
            base_delay=0.5,
            max_delay=30.0
        )
    }
    
    return presets.get(preset, presets['balanced'])


__all__ = [
    'RequestOptimizer',
    'ConnectionPool',
    'RetryManager',
    'RequestConfig',
    'RequestResult',
    'RetryStrategy',
    'ConnectionStats',
    'create_optimized_session',
    'create_request_config'
]