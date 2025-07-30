"""
Stealth Integration Module
==========================

Integration module for connecting the Google News crawler with the existing
stealth orchestrator system, providing seamless anti-detection capabilities.

This module provides:
- Integration with the unified stealth orchestrator
- CloudScraper priority fallback system
- Request routing and load balancing
- Error handling and recovery
- Performance monitoring
- Configuration management

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
License: MIT
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from urllib.parse import urlparse
from dataclasses import dataclass
from enum import Enum

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

logger = logging.getLogger(__name__)

class RequestMethod(Enum):
    """HTTP request methods."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"

class StealthMode(Enum):
    """Stealth operation modes."""
    CLOUDSCRAPER_ONLY = "cloudscraper_only"
    BASIC_ONLY = "basic_only"
    INTELLIGENT_FALLBACK = "intelligent_fallback"
    ROUND_ROBIN = "round_robin"

@dataclass
class RequestConfig:
    """Configuration for individual requests."""
    method: RequestMethod = RequestMethod.GET
    headers: Optional[Dict[str, str]] = None
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    follow_redirects: bool = True
    verify_ssl: bool = True
    user_agent: Optional[str] = None

@dataclass
class RequestResult:
    """Result of a stealth request."""
    success: bool
    status_code: Optional[int] = None
    content: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    response_time: float = 0.0
    method_used: str = "unknown"
    error: Optional[str] = None
    attempts: int = 1

class StealthIntegrationManager:
    """Manager for stealth integration with Google News crawler."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize stealth integration manager."""
        self.config = config or {}

        # Stealth orchestrator reference
        self._stealth_orchestrator = None
        self._basic_session = None

        # Configuration
        self.stealth_mode = StealthMode(self.config.get('stealth_mode', 'intelligent_fallback'))
        self.max_concurrent_requests = self.config.get('max_concurrent_requests', 10)
        self.default_timeout = self.config.get('default_timeout', 30)
        self.enable_caching = self.config.get('enable_caching', True)
        self.cache_ttl = self.config.get('cache_ttl', 300)  # 5 minutes

        # Performance tracking
        self.request_stats = {
            'total_requests': 0,
            'cloudscraper_success': 0,
            'basic_success': 0,
            'failures': 0,
            'average_response_time': 0.0,
            'success_rate': 0.0
        }

        # Request cache
        self._request_cache = {} if self.enable_caching else None

        # Rate limiting
        self._rate_limiter = asyncio.Semaphore(self.max_concurrent_requests)

        # Default headers
        self.default_headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }

    async def initialize(self, stealth_orchestrator=None):
        """Initialize the stealth integration system."""
        try:
            # Set stealth orchestrator reference
            if stealth_orchestrator:
                self._stealth_orchestrator = stealth_orchestrator
                logger.info("Stealth orchestrator connected successfully")
            else:
                # Try to import and create stealth orchestrator
                await self._initialize_stealth_orchestrator()

            # Initialize basic HTTP session
            if AIOHTTP_AVAILABLE:
                await self._initialize_basic_session()

            logger.info(f"Stealth integration initialized with mode: {self.stealth_mode.value}")

        except Exception as e:
            logger.error(f"Failed to initialize stealth integration: {e}")
            raise

    async def _initialize_stealth_orchestrator(self):
        """Initialize stealth orchestrator if not provided."""
        try:
            # Try to import from the news crawler stealth system
            try:
                from lindela.packages_enhanced.crawlers.news_crawler.stealth.unified_stealth_orchestrator import UnifiedStealthOrchestrator
            except ImportError:
                try:
                    # Try relative import
                    from ..news_crawler.stealth.unified_stealth_orchestrator import UnifiedStealthOrchestrator
                except ImportError:
                    # Create a mock stealth orchestrator for development
                    class MockStealthOrchestrator:
                        async def initialize(self):
                            pass
                        async def handle_request_with_stealth(self, url, data, **kwargs):
                            # Return a mock response
                            class MockResponse:
                                text = "<html><body>Mock stealth response</body></html>"
                                status_code = 200
                                headers = {}
                            return MockResponse()
                        async def close(self):
                            pass

                    UnifiedStealthOrchestrator = MockStealthOrchestrator
                    logger.warning("Using mock stealth orchestrator - stealth features disabled")

            orchestrator_config = self.config.get('stealth_orchestrator', {})
            self._stealth_orchestrator = UnifiedStealthOrchestrator(orchestrator_config)
            await self._stealth_orchestrator.initialize()

            logger.info("Created and initialized stealth orchestrator")

        except ImportError as e:
            logger.warning(f"Could not import stealth orchestrator: {e}")
            logger.info("Running without stealth capabilities")
        except Exception as e:
            logger.error(f"Failed to initialize stealth orchestrator: {e}")

    async def _initialize_basic_session(self):
        """Initialize basic HTTP session."""
        if not AIOHTTP_AVAILABLE:
            logger.warning("aiohttp not available - basic session disabled")
            return

        try:
            connector = aiohttp.TCPConnector(
                limit=self.max_concurrent_requests,
                ttl_dns_cache=300,
                use_dns_cache=True,
            )

            timeout = aiohttp.ClientTimeout(total=self.default_timeout)

            self._basic_session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers=self.default_headers
            )
        except Exception as e:
            logger.error(f"Failed to create aiohttp session: {e}")
            self._basic_session = None

        logger.info("Basic HTTP session initialized")

    async def handle_request(self,
                           url: str,
                           config: Optional[RequestConfig] = None,
                           **kwargs) -> RequestResult:
        """Handle a request using the configured stealth strategy."""

        config = config or RequestConfig()
        start_time = time.time()

        # Check cache first
        if self.enable_caching and config.method == RequestMethod.GET:
            cached_result = self._get_cached_result(url)
            if cached_result:
                logger.debug(f"Cache hit for {url}")
                return cached_result

        async with self._rate_limiter:
            try:
                result = await self._execute_request(url, config, **kwargs)

                # Update statistics
                self._update_stats(result, time.time() - start_time)

                # Cache successful results
                if (self.enable_caching and
                    result.success and
                    config.method == RequestMethod.GET):
                    self._cache_result(url, result)

                return result

            except Exception as e:
                logger.error(f"Request handling failed for {url}: {e}")
                self._update_stats(None, time.time() - start_time, error=True)

                return RequestResult(
                    success=False,
                    error=str(e),
                    response_time=time.time() - start_time
                )

    async def _execute_request(self,
                             url: str,
                             config: RequestConfig,
                             **kwargs) -> RequestResult:
        """Execute request using the configured strategy."""

        if self.stealth_mode == StealthMode.CLOUDSCRAPER_ONLY:
            return await self._request_with_stealth(url, config, **kwargs)

        elif self.stealth_mode == StealthMode.BASIC_ONLY:
            return await self._request_with_basic(url, config, **kwargs)

        elif self.stealth_mode == StealthMode.INTELLIGENT_FALLBACK:
            return await self._request_with_intelligent_fallback(url, config, **kwargs)

        elif self.stealth_mode == StealthMode.ROUND_ROBIN:
            return await self._request_with_round_robin(url, config, **kwargs)

        else:
            raise ValueError(f"Unknown stealth mode: {self.stealth_mode}")

    async def _request_with_stealth(self,
                                  url: str,
                                  config: RequestConfig,
                                  **kwargs) -> RequestResult:
        """Make request using stealth orchestrator."""

        if not self._stealth_orchestrator:
            return RequestResult(
                success=False,
                error="Stealth orchestrator not available"
            )

        try:
            # Prepare request parameters
            request_params = {
                'method': config.method.value,
                'headers': {**self.default_headers, **(config.headers or {})},
                'timeout': config.timeout,
                'verify_ssl': config.verify_ssl,
                'follow_redirects': config.follow_redirects,
            }

            if config.user_agent:
                request_params['headers']['User-Agent'] = config.user_agent

            # Make request through stealth orchestrator
            response = await self._stealth_orchestrator.handle_request_with_stealth(
                url, None, **request_params
            )

            if response and hasattr(response, 'text'):
                content = response.text if hasattr(response, 'text') else str(response)
                status_code = getattr(response, 'status_code', 200)
                headers = getattr(response, 'headers', {})

                return RequestResult(
                    success=True,
                    status_code=status_code,
                    content=content,
                    headers=dict(headers) if headers else {},
                    method_used='cloudscraper'
                )
            else:
                return RequestResult(
                    success=False,
                    error="Invalid response from stealth orchestrator"
                )

        except Exception as e:
            logger.debug(f"Stealth request failed for {url}: {e}")
            return RequestResult(
                success=False,
                error=f"Stealth request failed: {e}",
                method_used='cloudscraper'
            )

    async def _request_with_basic(self,
                                url: str,
                                config: RequestConfig,
                                **kwargs) -> RequestResult:
        """Make request using basic HTTP client."""

        if not self._basic_session:
            return RequestResult(
                success=False,
                error="Basic HTTP session not available"
            )

        try:
            # Prepare request parameters
            headers = {**self.default_headers, **(config.headers or {})}
            if config.user_agent:
                headers['User-Agent'] = config.user_agent

            # Make request
            async with self._basic_session.request(
                method=config.method.value,
                url=url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=config.timeout),
                ssl=config.verify_ssl,
                allow_redirects=config.follow_redirects,
                **kwargs
            ) as response:

                content = await response.text()

                return RequestResult(
                    success=response.status < 400,
                    status_code=response.status,
                    content=content,
                    headers=dict(response.headers),
                    method_used='basic'
                )

        except asyncio.TimeoutError:
            return RequestResult(
                success=False,
                error="Request timeout",
                method_used='basic'
            )
        except Exception as e:
            logger.debug(f"Basic request failed for {url}: {e}")
            return RequestResult(
                success=False,
                error=f"Basic request failed: {e}",
                method_used='basic'
            )

    async def _request_with_intelligent_fallback(self,
                                               url: str,
                                               config: RequestConfig,
                                               **kwargs) -> RequestResult:
        """Make request with intelligent fallback strategy."""

        # Try stealth first
        if self._stealth_orchestrator:
            result = await self._request_with_stealth(url, config, **kwargs)
            if result.success:
                return result

            logger.debug(f"Stealth request failed for {url}, trying basic")

        # Fallback to basic
        result = await self._request_with_basic(url, config, **kwargs)
        if result.success:
            return result

        # If both failed, return the more informative error
        return RequestResult(
            success=False,
            error="Both stealth and basic requests failed",
            method_used='fallback'
        )

    async def _request_with_round_robin(self,
                                      url: str,
                                      config: RequestConfig,
                                      **kwargs) -> RequestResult:
        """Make request using round-robin strategy."""

        # Simple round-robin based on request count
        use_stealth = (self.request_stats['total_requests'] % 2) == 0

        if use_stealth and self._stealth_orchestrator:
            return await self._request_with_stealth(url, config, **kwargs)
        else:
            return await self._request_with_basic(url, config, **kwargs)

    def _get_cached_result(self, url: str) -> Optional[RequestResult]:
        """Get cached result if available and not expired."""
        if not self._request_cache:
            return None

        cache_key = self._generate_cache_key(url)
        cached_data = self._request_cache.get(cache_key)

        if cached_data:
            result, timestamp = cached_data
            if time.time() - timestamp < self.cache_ttl:
                return result
            else:
                # Remove expired entry
                del self._request_cache[cache_key]

        return None

    def _cache_result(self, url: str, result: RequestResult):
        """Cache successful result."""
        if not self._request_cache:
            return

        cache_key = self._generate_cache_key(url)
        self._request_cache[cache_key] = (result, time.time())

        # Simple cache size management
        if len(self._request_cache) > 1000:
            # Remove oldest entries
            sorted_items = sorted(
                self._request_cache.items(),
                key=lambda x: x[1][1]  # Sort by timestamp
            )

            # Keep newest 800 entries
            self._request_cache = dict(sorted_items[-800:])

    def _generate_cache_key(self, url: str) -> str:
        """Generate cache key for URL."""
        # Simple URL-based key (could be enhanced with headers, etc.)
        return f"url:{hash(url)}"

    def _update_stats(self, result: Optional[RequestResult], response_time: float, error: bool = False):
        """Update performance statistics."""
        self.request_stats['total_requests'] += 1

        if error or not result or not result.success:
            self.request_stats['failures'] += 1
        else:
            if result.method_used == 'cloudscraper':
                self.request_stats['cloudscraper_success'] += 1
            elif result.method_used == 'basic':
                self.request_stats['basic_success'] += 1

        # Update average response time
        total_requests = self.request_stats['total_requests']
        current_avg = self.request_stats['average_response_time']
        self.request_stats['average_response_time'] = (
            (current_avg * (total_requests - 1) + response_time) / total_requests
        )

        # Update success rate
        successful_requests = (
            self.request_stats['cloudscraper_success'] +
            self.request_stats['basic_success']
        )
        self.request_stats['success_rate'] = successful_requests / total_requests

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.request_stats.copy()

        # Add derived metrics
        if stats['total_requests'] > 0:
            stats['cloudscraper_success_rate'] = (
                stats['cloudscraper_success'] / stats['total_requests']
            )
            stats['basic_success_rate'] = (
                stats['basic_success'] / stats['total_requests']
            )
            stats['failure_rate'] = stats['failures'] / stats['total_requests']

        # Add configuration info
        stats['stealth_mode'] = self.stealth_mode.value
        stats['stealth_available'] = self._stealth_orchestrator is not None
        stats['basic_available'] = self._basic_session is not None
        stats['caching_enabled'] = self.enable_caching

        if self.enable_caching and self._request_cache:
            stats['cache_size'] = len(self._request_cache)

        return stats

    def clear_cache(self):
        """Clear request cache."""
        if self._request_cache:
            self._request_cache.clear()
            logger.info("Request cache cleared")

    def reset_stats(self):
        """Reset performance statistics."""
        self.request_stats = {
            'total_requests': 0,
            'cloudscraper_success': 0,
            'basic_success': 0,
            'failures': 0,
            'average_response_time': 0.0,
            'success_rate': 0.0
        }
        logger.info("Performance statistics reset")

    async def close(self):
        """Close the stealth integration manager."""
        try:
            # Close basic session
            if self._basic_session:
                await self._basic_session.close()
                logger.info("Basic HTTP session closed")

            # Close stealth orchestrator if we created it
            if (self._stealth_orchestrator and
                hasattr(self._stealth_orchestrator, 'close')):
                await self._stealth_orchestrator.close()
                logger.info("Stealth orchestrator closed")

            # Clear cache
            if self.enable_caching:
                self.clear_cache()

            logger.info("Stealth integration manager closed successfully")

        except Exception as e:
            logger.error(f"Error closing stealth integration manager: {e}")

    def __del__(self):
        """Cleanup on deletion."""
        if self._basic_session and not self._basic_session.closed:
            logger.warning("StealthIntegrationManager was not properly closed")

# Factory functions for easy integration

async def create_stealth_integration(config: Optional[Dict[str, Any]] = None,
                                   stealth_orchestrator=None) -> StealthIntegrationManager:
    """Create and initialize stealth integration manager."""
    manager = StealthIntegrationManager(config)
    await manager.initialize(stealth_orchestrator)
    return manager

def get_default_config() -> Dict[str, Any]:
    """Get default configuration for stealth integration."""
    return {
        'stealth_mode': 'intelligent_fallback',
        'max_concurrent_requests': 10,
        'default_timeout': 30,
        'enable_caching': True,
        'cache_ttl': 300,
        'stealth_orchestrator': {
            'cloudflare_solver': 'cloudscraper',
            'max_retries': 3,
            'retry_delay': 1.0,
            'success_threshold': 0.8,
        }
    }

# Integration helper for Google News client
class GoogleNewsStealthMixin:
    """Mixin class to add stealth capabilities to Google News client."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stealth_integration = None

    async def _initialize_stealth_integration(self, config: Optional[Dict[str, Any]] = None):
        """Initialize stealth integration for Google News client."""
        if not self._stealth_integration:
            stealth_config = config or get_default_config()
            self._stealth_integration = await create_stealth_integration(
                stealth_config,
                getattr(self, 'stealth_orchestrator', None)
            )

    async def _fetch_with_integrated_stealth(self, url: str, config: Optional[RequestConfig] = None) -> Optional[str]:
        """Fetch URL using integrated stealth capabilities."""
        if not self._stealth_integration:
            await self._initialize_stealth_integration()

        result = await self._stealth_integration.handle_request(url, config)

        if result.success:
            return result.content
        else:
            logger.warning(f"Stealth request failed for {url}: {result.error}")
            return None

    def get_stealth_stats(self) -> Dict[str, Any]:
        """Get stealth integration statistics."""
        if self._stealth_integration:
            return self._stealth_integration.get_performance_stats()
        return {}

    async def _cleanup_stealth_integration(self):
        """Cleanup stealth integration."""
        if self._stealth_integration:
            await self._stealth_integration.close()
