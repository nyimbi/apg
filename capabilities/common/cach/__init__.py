#!/usr/bin/env python3
"""
APG Cache Management (CACH) Capability
Revolutionary AI-powered cache management with autonomous optimization

This capability provides:
- Autonomous Cache Intelligence with AI-powered self-optimization
- Predictive Content Delivery with ML-driven prefetching  
- Intelligent Cache Warming with smart cold start elimination
- Adaptive Multi-Tier Orchestration with dynamic management
- Content-Aware Optimization with semantic understanding
- Real-Time Performance Analytics with live insights
- Zero-Configuration Intelligence with self-configuring policies
- Distributed Consensus Optimization with smart consistency
- Behavior-Driven Security with adaptive policies
- Quantum-Ready Architecture with future-proof platform

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

from .service import CacheService, CacheServiceConfig, create_cache_service
from .models import (
	CacheEntry, CacheCluster, CachePolicy, CacheMetrics, AIOptimizationResult,
	CacheBackendType, CompressionAlgorithm, EvictionPolicy, CacheAccessPattern,
	SecurityLevel, CacheTier
)
from .blueprint import (
	cache_blueprint, CacheManagementView, CacheAPIView, CacheChartView,
	CAPABILITY_METADATA, register_with_appbuilder
)

# Capability metadata for APG composition engine
__capability_name__ = "cach"
__capability_version__ = "1.0.0"
__capability_description__ = "AI-powered cache management with autonomous optimization"
__capability_dependencies__ = ["auth", "audl", "mten", "moni", "conf"]
__capability_optional_dependencies__ = ["aicr", "pred", "anom", "agnt"]

# Export main components
__all__ = [
	# Service components
	'CacheService',
	'CacheServiceConfig', 
	'create_cache_service',
	
	# Data models
	'CacheEntry',
	'CacheCluster',
	'CachePolicy',
	'CacheMetrics',
	'AIOptimizationResult',
	
	# Enums
	'CacheBackendType',
	'CompressionAlgorithm',
	'EvictionPolicy',
	'CacheAccessPattern',
	'SecurityLevel',
	'CacheTier',
	
	# Flask integration
	'cache_blueprint',
	'CacheManagementView',
	'CacheAPIView',
	'CacheChartView',
	'register_with_appbuilder',
	
	# APG metadata
	'CAPABILITY_METADATA',
	'__capability_name__',
	'__capability_version__',
	'__capability_description__',
	'__capability_dependencies__',
	'__capability_optional_dependencies__'
]

# APG capability initialization function
async def initialize_capability(config: dict = None) -> CacheService:
	"""
	Initialize the cache management capability
	Called by APG composition engine during capability loading
	"""
	cache_config = CacheServiceConfig()
	
	if config:
		# Update configuration from APG
		for key, value in config.items():
			if hasattr(cache_config, key):
				setattr(cache_config, key, value)
	
	# Create and initialize service
	service = CacheService(cache_config)
	await service.initialize(config)
	
	return service


# APG capability health check
async def health_check() -> dict:
	"""
	Health check for APG monitoring
	Returns capability health status and metrics
	"""
	try:
		from .service import _cache_service
		
		if _cache_service is None or not _cache_service.running:
			return {
				'healthy': False,
				'capability': 'cach',
				'status': 'not_running',
				'timestamp': None
			}
		
		stats = await _cache_service.get_stats()
		
		return {
			'healthy': True,
			'capability': 'cach',
			'status': 'running',
			'stats': {
				'total_entries': stats.get('total_entries', 0),
				'hit_rate': stats.get('hit_rate', 0),
				'memory_utilization': stats.get('memory_utilization', 0),
				'total_operations': stats.get('total_operations', 0)
			},
			'timestamp': stats.get('timestamp')
		}
		
	except Exception as e:
		return {
			'healthy': False,
			'capability': 'cach',
			'status': 'error',
			'error': str(e),
			'timestamp': None
		}


# APG capability export functions for composition
async def get_cache_value(key: str, namespace: str = "default", tenant_id: str = None) -> any:
	"""Export function: Get cache value for other capabilities"""
	from .service import _cache_service
	if _cache_service:
		return await _cache_service.get(key=key, namespace=namespace, tenant_id=tenant_id)
	return None


async def set_cache_value(key: str, value: any, ttl_seconds: int = None, 
						  namespace: str = "default", tenant_id: str = None) -> bool:
	"""Export function: Set cache value for other capabilities"""
	from .service import _cache_service
	if _cache_service:
		return await _cache_service.set(
			key=key, value=value, ttl_seconds=ttl_seconds, 
			namespace=namespace, tenant_id=tenant_id
		)
	return False


async def delete_cache_value(key: str, namespace: str = "default", tenant_id: str = None) -> bool:
	"""Export function: Delete cache value for other capabilities"""
	from .service import _cache_service
	if _cache_service:
		return await _cache_service.delete(key=key, namespace=namespace, tenant_id=tenant_id)
	return False


async def get_cache_stats() -> dict:
	"""Export function: Get cache statistics for monitoring"""
	from .service import _cache_service
	if _cache_service:
		return await _cache_service.get_stats()
	return {}


# Event handlers for APG composition engine
async def handle_cache_set(event_data: dict) -> None:
	"""Handle cache set events from other capabilities"""
	key = event_data.get('key')
	value = event_data.get('value') 
	if key and value is not None:
		await set_cache_value(key, value, 
							  ttl_seconds=event_data.get('ttl_seconds'),
							  namespace=event_data.get('namespace', 'default'),
							  tenant_id=event_data.get('tenant_id'))


async def handle_cache_get(event_data: dict) -> any:
	"""Handle cache get events from other capabilities"""
	key = event_data.get('key')
	if key:
		return await get_cache_value(key,
									 namespace=event_data.get('namespace', 'default'),
									 tenant_id=event_data.get('tenant_id'))
	return None


async def handle_cache_delete(event_data: dict) -> bool:
	"""Handle cache delete events from other capabilities"""
	key = event_data.get('key')
	if key:
		return await delete_cache_value(key,
										namespace=event_data.get('namespace', 'default'), 
										tenant_id=event_data.get('tenant_id'))
	return False


async def handle_cache_optimized(event_data: dict) -> None:
	"""Handle cache optimization completion events"""
	# Log optimization results or trigger additional actions
	pass
