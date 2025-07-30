"""
YouTube Crawler Optimization Module
=====================================

Performance optimization components for YouTube content crawling.
Provides caching, rate limiting, request optimization, and monitoring.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
"""

from ....utils.caching import CacheManager
from .youtube_cache_wrapper import YouTubeCacheWrapper
from .rate_limiter import RateLimiter, TokenBucketLimiter, SlidingWindowLimiter
from .request_optimizer import RequestOptimizer, ConnectionPool, RetryManager
from .batch_processor import BatchProcessor, BatchConfig, BatchResult
from .performance_monitor import PerformanceMonitor, MetricsCollector, HealthChecker

__all__ = [
    # Cache management
    'CacheManager',
    'YouTubeCacheWrapper',
    
    # Rate limiting
    'RateLimiter',
    'TokenBucketLimiter',
    'SlidingWindowLimiter',
    
    # Request optimization
    'RequestOptimizer',
    'ConnectionPool',
    'RetryManager',
    
    # Batch processing
    'BatchProcessor',
    'BatchConfig',
    'BatchResult',
    
    # Performance monitoring
    'PerformanceMonitor',
    'MetricsCollector',
    'HealthChecker'
]

# Version information
__version__ = "1.0.0"
__author__ = "Nyimbi Odero"
__email__ = "nyimbi@datacraft.co.ke"
__company__ = "Datacraft"

# Optimization constants
OPTIMIZATION_DEFAULTS = {
    'cache_ttl': 3600,  # 1 hour
    'rate_limit_per_minute': 60,
    'max_concurrent_requests': 10,
    'batch_size': 50,
    'retry_attempts': 3,
    'connection_timeout': 30,
    'read_timeout': 60
}

def get_optimization_defaults():
    """Get default optimization settings."""
    return OPTIMIZATION_DEFAULTS.copy()

async def create_optimized_client_config():
    """Create optimized configuration for YouTube client."""
    cache_manager = await CacheManager.create()
    youtube_cache = YouTubeCacheWrapper(cache_manager)
    
    return {
        'cache_manager': youtube_cache,
        'rate_limiter': RateLimiter(requests_per_minute=OPTIMIZATION_DEFAULTS['rate_limit_per_minute']),
        'request_optimizer': RequestOptimizer(
            max_concurrent=OPTIMIZATION_DEFAULTS['max_concurrent_requests'],
            timeout=OPTIMIZATION_DEFAULTS['connection_timeout']
        ),
        'batch_processor': BatchProcessor(batch_size=OPTIMIZATION_DEFAULTS['batch_size']),
        'performance_monitor': PerformanceMonitor()
    }

# Performance presets
PERFORMANCE_PRESETS = {
    'conservative': {
        'rate_limit_per_minute': 30,
        'max_concurrent_requests': 3,
        'batch_size': 10,
        'cache_ttl': 7200  # 2 hours
    },
    'balanced': {
        'rate_limit_per_minute': 60,
        'max_concurrent_requests': 10,
        'batch_size': 50,
        'cache_ttl': 3600  # 1 hour
    },
    'aggressive': {
        'rate_limit_per_minute': 120,
        'max_concurrent_requests': 20,
        'batch_size': 100,
        'cache_ttl': 1800  # 30 minutes
    }
}

def get_performance_preset(preset_name: str):
    """Get performance preset configuration."""
    return PERFORMANCE_PRESETS.get(preset_name, PERFORMANCE_PRESETS['balanced'])

def optimize_for_environment(environment: str):
    """Get optimized settings for specific environment."""
    env_settings = {
        'development': get_performance_preset('conservative'),
        'testing': get_performance_preset('balanced'),
        'staging': get_performance_preset('balanced'),
        'production': get_performance_preset('aggressive')
    }
    
    return env_settings.get(environment, env_settings['production'])

# Health check for optimization components
def health_check():
    """Perform health check on optimization components."""
    results = {
        'status': 'healthy',
        'components': {},
        'timestamp': None
    }
    
    try:
        from datetime import datetime
        results['timestamp'] = datetime.utcnow().isoformat()
        
        # Test cache manager (async)
        try:
            import asyncio
            async def test_cache():
                cache = await CacheManager.create()
                await cache.set('test_key', 'test_value', ttl=1)
                value = await cache.get('test_key')
                return value == 'test_value'
            
            if asyncio.run(test_cache()):
                results['components']['cache_manager'] = 'healthy'
            else:
                results['components']['cache_manager'] = 'degraded'
        except Exception as e:
            results['components']['cache_manager'] = f'error: {e}'
        
        # Test rate limiter
        try:
            limiter = RateLimiter(requests_per_minute=60)
            if limiter.can_proceed():
                results['components']['rate_limiter'] = 'healthy'
            else:
                results['components']['rate_limiter'] = 'degraded'
        except Exception as e:
            results['components']['rate_limiter'] = f'error: {e}'
        
        # Test request optimizer
        try:
            optimizer = RequestOptimizer()
            results['components']['request_optimizer'] = 'healthy'
        except Exception as e:
            results['components']['request_optimizer'] = f'error: {e}'
        
        # Test batch processor
        try:
            processor = BatchProcessor()
            results['components']['batch_processor'] = 'healthy'
        except Exception as e:
            results['components']['batch_processor'] = f'error: {e}'
        
        # Test performance monitor
        try:
            monitor = PerformanceMonitor()
            results['components']['performance_monitor'] = 'healthy'
        except Exception as e:
            results['components']['performance_monitor'] = f'error: {e}'
        
        # Overall status
        error_components = [k for k, v in results['components'].items() if v.startswith('error')]
        degraded_components = [k for k, v in results['components'].items() if v == 'degraded']
        
        if error_components:
            results['status'] = 'error'
            results['error_components'] = error_components
        elif degraded_components:
            results['status'] = 'degraded'
            results['degraded_components'] = degraded_components
    
    except Exception as e:
        results['status'] = 'error'
        results['error'] = str(e)
    
    return results