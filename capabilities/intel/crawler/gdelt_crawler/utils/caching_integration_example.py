"""
GDELT Caching Integration Example
================================

Example demonstrating how to integrate the central caching utilities
into the GDELT crawler instead of using local caching implementations.

This example shows how to replace the simple dictionary-based caching
in the GDELT API client with the comprehensive caching system from
packages_enhanced.utils.caching.

Key Benefits:
- Advanced caching strategies (LRU, LFU, TTL, spatial-aware)
- Multiple backends (memory, Redis, file system)
- Compression and encryption support
- Performance monitoring and analytics
- Distributed caching capabilities
- Automatic cache warming and invalidation

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Version: 1.0.0
License: MIT
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import hashlib
from urllib.parse import urlencode

# Import central caching utilities instead of implementing local caching
from ....utils.caching import (
    CacheManager,
    CacheConfig,
    create_cache_manager,
    cache_decorator,
    spatial_cache_decorator
)

# Import other central utilities
from ....utils.monitoring import MonitoringManager
from ....utils.performance import PerformanceProfiler

# Configure logging
logger = logging.getLogger(__name__)


class EnhancedGDELTClient:
    """
    Enhanced GDELT client using central caching utilities.

    This replaces the local dictionary-based caching with the comprehensive
    caching system from packages_enhanced.utils.caching.
    """

    def __init__(
        self,
        rate_limit: float = 10.0,
        timeout: int = 60,
        max_concurrent: int = 5,
        cache_config: Optional[Dict[str, Any]] = None,
        enable_monitoring: bool = True
    ):
        """
        Initialize enhanced GDELT client with central caching.

        Args:
            rate_limit: API rate limit (requests per second)
            timeout: Request timeout in seconds
            max_concurrent: Maximum concurrent requests
            cache_config: Configuration for caching system
            enable_monitoring: Whether to enable performance monitoring
        """
        self.rate_limit = rate_limit
        self.timeout = timeout
        self.max_concurrent = max_concurrent
        self.enable_monitoring = enable_monitoring

        # Initialize caching system
        self.cache_manager = None
        self.cache_config = cache_config or self._get_default_cache_config()

        # Initialize monitoring
        self.monitoring_manager = None
        self.performance_profiler = None

        # Client state
        self.session = None
        self._initialized = False

    def _get_default_cache_config(self) -> Dict[str, Any]:
        """Get default cache configuration for GDELT data."""
        return {
            'strategy': 'lru',
            'max_size': 10000,
            'ttl': 3600,  # 1 hour TTL for API responses
            'backend': 'memory',
            'compression': True,
            'enable_monitoring': True,
            'spatial_optimization': True,  # Enable spatial caching for geographic data
            'batch_operations': True,
            'cache_warming': {
                'enabled': True,
                'preload_patterns': [
                    'conflict*',
                    'ukraine*',
                    'africa*'
                ]
            }
        }

    async def initialize(self):
        """Initialize the client with caching and monitoring systems."""
        if self._initialized:
            return

        try:
            # Initialize cache manager
            self.cache_manager = await create_cache_manager(self.cache_config)
            logger.info("Cache manager initialized with central utilities")

            # Initialize monitoring if enabled
            if self.enable_monitoring:
                self.monitoring_manager = MonitoringManager()
                await self.monitoring_manager.initialize()

                self.performance_profiler = PerformanceProfiler()
                logger.info("Monitoring and performance profiling initialized")

            # Warm up cache with common queries
            await self._warm_cache()

            self._initialized = True
            logger.info("Enhanced GDELT client initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize enhanced GDELT client: {e}")
            raise

    async def _warm_cache(self):
        """Warm up cache with common GDELT queries."""
        try:
            # Example of cache warming with common conflict-related queries
            common_queries = [
                {'query': 'conflict', 'timespan': '1d'},
                {'query': 'ukraine war', 'timespan': '1d'},
                {'query': 'africa violence', 'timespan': '1d'},
                {'query': 'terrorism', 'timespan': '1d'}
            ]

            logger.info("Starting cache warming with common queries")

            # This would normally make actual API calls, but for this example
            # we'll just register the patterns in the cache
            for query_params in common_queries:
                cache_key = self._generate_cache_key(query_params)
                await self.cache_manager.register_pattern(cache_key)

            logger.info(f"Cache warmed with {len(common_queries)} common query patterns")

        except Exception as e:
            logger.warning(f"Cache warming failed: {e}")

    def _generate_cache_key(self, params: Dict[str, Any]) -> str:
        """
        Generate cache key from query parameters.

        Uses a more sophisticated approach than simple parameter hashing,
        incorporating spatial and temporal context for better cache efficiency.
        """
        # Sort parameters for consistent key generation
        sorted_params = sorted(params.items())
        param_str = urlencode(sorted_params)

        # Add temporal context for better cache organization
        if 'timespan' in params:
            timespan = params['timespan']
            # Normalize timespan to standard format
            if timespan.endswith('d'):
                param_str += f"_daily_{timespan}"
            elif timespan.endswith('h'):
                param_str += f"_hourly_{timespan}"

        # Add spatial context if available
        if any(key in params for key in ['country', 'region', 'location']):
            spatial_context = params.get('country', params.get('region', params.get('location', '')))
            param_str += f"_spatial_{spatial_context}"

        return hashlib.sha256(param_str.encode()).hexdigest()

    @cache_decorator(ttl=3600, key_prefix='gdelt_query')
    async def query_with_caching(
        self,
        query: str,
        timespan: str = '1d',
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Query GDELT with automatic caching using decorator.

        The cache_decorator automatically handles caching logic,
        including key generation, TTL management, and cache statistics.
        """
        if self.enable_monitoring:
            with self.performance_profiler.profile('gdelt_api_query'):
                return await self._execute_query(query, timespan, **kwargs)
        else:
            return await self._execute_query(query, timespan, **kwargs)

    async def _execute_query(
        self,
        query: str,
        timespan: str,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Execute the actual GDELT API query."""
        # This would contain the actual API call logic
        # For this example, we'll return mock data

        logger.info(f"Executing GDELT query: {query} (timespan: {timespan})")

        # Simulate API delay
        await asyncio.sleep(0.1)

        # Return mock data
        return [
            {
                'url': f'https://example.com/article1',
                'title': f'Article about {query}',
                'date': datetime.now().isoformat(),
                'location': 'Global',
                'sentiment': 0.2
            }
        ]

    @spatial_cache_decorator(
        ttl=7200,  # 2 hours TTL for location-based queries
        spatial_key='coordinates',
        radius_km=50  # Cache hits within 50km radius
    )
    async def query_by_location(
        self,
        latitude: float,
        longitude: float,
        radius_km: float = 100,
        timespan: str = '1d'
    ) -> List[Dict[str, Any]]:
        """
        Query GDELT by geographic location with spatial caching.

        The spatial_cache_decorator provides geographic-aware caching
        that can return cached results for nearby locations.
        """
        coordinates = {'latitude': latitude, 'longitude': longitude}

        if self.enable_monitoring:
            with self.performance_profiler.profile('gdelt_location_query'):
                return await self._execute_location_query(
                    latitude, longitude, radius_km, timespan
                )
        else:
            return await self._execute_location_query(
                latitude, longitude, radius_km, timespan
            )

    async def _execute_location_query(
        self,
        latitude: float,
        longitude: float,
        radius_km: float,
        timespan: str
    ) -> List[Dict[str, Any]]:
        """Execute location-based GDELT query."""
        logger.info(f"Executing location query: {latitude}, {longitude} (radius: {radius_km}km)")

        # Simulate API delay
        await asyncio.sleep(0.2)

        # Return mock data
        return [
            {
                'url': 'https://example.com/local_article',
                'title': f'Local event near {latitude}, {longitude}',
                'date': datetime.now().isoformat(),
                'latitude': latitude + 0.01,
                'longitude': longitude + 0.01,
                'sentiment': -0.3
            }
        ]

    async def batch_query_with_caching(
        self,
        queries: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """
        Execute multiple queries efficiently using batch caching.

        Leverages the central caching system's batch operations
        for improved performance.
        """
        if not self._initialized:
            await self.initialize()

        # Check cache for all queries first
        cache_keys = [self._generate_cache_key(query) for query in queries]
        cached_results = await self.cache_manager.get_batch(cache_keys)

        # Identify queries that need to be executed
        queries_to_execute = []
        results = [None] * len(queries)

        for i, (query, cached_result) in enumerate(zip(queries, cached_results)):
            if cached_result is not None:
                results[i] = cached_result
                logger.debug(f"Cache hit for query {i}")
            else:
                queries_to_execute.append((i, query))

        # Execute uncached queries
        if queries_to_execute:
            logger.info(f"Executing {len(queries_to_execute)} uncached queries")

            tasks = []
            for i, query in queries_to_execute:
                task = self._execute_query(**query)
                tasks.append((i, task))

            # Execute queries concurrently
            executed_results = await asyncio.gather(
                *[task for _, task in tasks],
                return_exceptions=True
            )

            # Store results in cache and update results list
            cache_operations = []
            for (i, _), result in zip(tasks, executed_results):
                if not isinstance(result, Exception):
                    results[i] = result
                    cache_key = cache_keys[i]
                    cache_operations.append((cache_key, result))
                else:
                    logger.error(f"Query {i} failed: {result}")
                    results[i] = []

            # Batch store in cache
            if cache_operations:
                await self.cache_manager.set_batch(cache_operations)

        return results

    async def get_cache_statistics(self) -> Dict[str, Any]:
        """Get detailed cache performance statistics."""
        if not self.cache_manager:
            return {}

        stats = await self.cache_manager.get_statistics()

        # Add GDELT-specific metrics
        gdelt_stats = {
            'gdelt_specific': {
                'api_queries_cached': stats.get('hits', 0),
                'cache_efficiency': stats.get('hit_rate', 0.0),
                'spatial_cache_hits': stats.get('spatial_hits', 0),
                'cache_size_mb': stats.get('memory_usage_mb', 0)
            }
        }

        return {**stats, **gdelt_stats}

    async def optimize_cache(self):
        """Optimize cache performance based on usage patterns."""
        if not self.cache_manager:
            return

        logger.info("Optimizing GDELT cache based on usage patterns")

        # Get usage statistics
        stats = await self.get_cache_statistics()

        # Optimize based on patterns
        if stats.get('hit_rate', 0) < 0.5:
            # Low hit rate - increase cache size or adjust TTL
            await self.cache_manager.optimize_settings({
                'max_size': int(self.cache_config.get('max_size', 10000) * 1.5),
                'ttl': int(self.cache_config.get('ttl', 3600) * 1.2)
            })
            logger.info("Increased cache size and TTL due to low hit rate")

        # Cleanup expired entries
        await self.cache_manager.cleanup_expired()

        logger.info("Cache optimization completed")

    async def close(self):
        """Clean up resources."""
        if self.cache_manager:
            await self.cache_manager.close()

        if self.monitoring_manager:
            await self.monitoring_manager.close()

        logger.info("Enhanced GDELT client closed")


# Example usage and integration
async def example_usage():
    """Example of how to use the enhanced GDELT client with central caching."""

    # Create client with custom cache configuration
    cache_config = {
        'strategy': 'lru',
        'max_size': 20000,
        'ttl': 7200,  # 2 hours
        'backend': 'redis',  # Use Redis for persistent caching
        'compression': True,
        'enable_monitoring': True
    }

    client = EnhancedGDELTClient(
        rate_limit=15.0,
        cache_config=cache_config,
        enable_monitoring=True
    )

    try:
        # Initialize client
        await client.initialize()

        # Single query with automatic caching
        results = await client.query_with_caching(
            query='ukraine conflict',
            timespan='1d'
        )
        print(f"Found {len(results)} articles")

        # Location-based query with spatial caching
        location_results = await client.query_by_location(
            latitude=50.4501,  # Kyiv coordinates
            longitude=30.5234,
            radius_km=100,
            timespan='1d'
        )
        print(f"Found {len(location_results)} location-based articles")

        # Batch queries
        batch_queries = [
            {'query': 'conflict', 'timespan': '1d'},
            {'query': 'terrorism', 'timespan': '1d'},
            {'query': 'violence', 'timespan': '1d'}
        ]

        batch_results = await client.batch_query_with_caching(batch_queries)
        print(f"Batch results: {len(batch_results)} query sets")

        # Get cache performance statistics
        cache_stats = await client.get_cache_statistics()
        print(f"Cache hit rate: {cache_stats.get('hit_rate', 0):.2%}")

        # Optimize cache performance
        await client.optimize_cache()

    finally:
        await client.close()


# Factory function for easy integration
async def create_enhanced_gdelt_client(
    cache_backend: str = 'memory',
    enable_redis: bool = False,
    enable_monitoring: bool = True
) -> EnhancedGDELTClient:
    """
    Factory function to create enhanced GDELT client with optimal settings.

    Args:
        cache_backend: Cache backend ('memory', 'redis', 'file')
        enable_redis: Whether to enable Redis caching
        enable_monitoring: Whether to enable performance monitoring

    Returns:
        Configured and initialized GDELT client
    """
    cache_config = {
        'strategy': 'adaptive',  # Use adaptive caching strategy
        'max_size': 50000,
        'ttl': 3600,
        'backend': 'redis' if enable_redis else cache_backend,
        'compression': True,
        'enable_monitoring': enable_monitoring,
        'spatial_optimization': True,
        'distributed': enable_redis,
        'backup_enabled': True
    }

    client = EnhancedGDELTClient(
        rate_limit=20.0,
        cache_config=cache_config,
        enable_monitoring=enable_monitoring
    )

    await client.initialize()
    return client


if __name__ == "__main__":
    asyncio.run(example_usage())
