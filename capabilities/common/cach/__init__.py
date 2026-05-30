#!/usr/bin/env python3
"""
APG Cache Management (CACH) Capability
Tenant-scoped cache governance, lifecycle records, and runtime adapters

This capability provides:
- Cache namespace lifecycle governance
- Cache entry admission, freshness, and invalidation records
- Warming and eviction review workflows
- Deterministic cache guardrail evaluation
- Generated-application UI and theme metadata
- Adapter boundaries for memory, Redis-compatible, edge, CDN, and query caches

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

from .capability_contract import (
	get_capability_contract,
	evaluate_capability_rules
)
try:
	from .service import (
		CacheAuditEventRecord,
		CacheEntryRecord,
		CacheEvictionReviewRecord,
		CacheGovernanceService,
		CacheNamespaceRecord,
		CacheService,
		CacheServiceConfig,
		CacheWarmingPlanRecord,
		create_cache_service,
	)
	_SERVICE_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
	if exc.name not in {'lz4', 'zstandard'}:
		raise
	CacheService = None
	CacheServiceConfig = None
	CacheGovernanceService = None
	CacheNamespaceRecord = None
	CacheEntryRecord = None
	CacheWarmingPlanRecord = None
	CacheEvictionReviewRecord = None
	CacheAuditEventRecord = None
	_SERVICE_IMPORT_ERROR = exc

	def create_cache_service(*args, **kwargs):
		"""Require optional compression dependencies before service creation."""
		raise ModuleNotFoundError(
			"CACH service requires optional compression dependencies: lz4, zstandard"
		) from _SERVICE_IMPORT_ERROR

from .models import (
	CacheEntry, CacheCluster, CachePolicy, CacheMetrics, AIOptimizationResult,
	CacheBackendType, CompressionAlgorithm, EvictionPolicy, CacheAccessPattern,
	SecurityLevel, CacheTier
)
try:
	from .blueprint import (
		cache_blueprint, CacheManagementView, CacheAPIView, CacheChartView,
		CAPABILITY_METADATA, register_with_appbuilder
	)
	_BLUEPRINT_IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:
	cache_blueprint = None
	CacheManagementView = None
	CacheAPIView = None
	CacheChartView = None
	_BLUEPRINT_IMPORT_ERROR = exc
	CAPABILITY_METADATA = {
		"name": "cach",
		"display_name": "Cache Management",
		"description": "Tenant-scoped cache governance, warming, eviction review, and runtime adapter control",
		"version": "1.0.0",
		"category": "infrastructure",
		"dependencies": ["auth", "audl", "mten", "moni", "conf"],
		"optional_dependencies": ["aicr", "pred", "anom", "agnt"]
	}

	def register_with_appbuilder(*args, **kwargs):
		"""Require optional UI/runtime dependencies before AppBuilder registration."""
		raise ImportError("CACH UI integration requires optional runtime dependencies") from _BLUEPRINT_IMPORT_ERROR

# Capability metadata for APG composition engine
__capability_name__ = "cach"
__capability_version__ = "1.0.0"
__capability_description__ = "Tenant-scoped cache governance, warming, eviction review, and runtime adapter control"
__capability_dependencies__ = ["auth", "audl", "mten", "moni", "conf"]
__capability_optional_dependencies__ = ["aicr", "pred", "anom", "agnt"]

# Export main components
__all__ = [
	# Service components
	'CacheService',
	'CacheServiceConfig', 
	'CacheGovernanceService',
	'CacheNamespaceRecord',
	'CacheEntryRecord',
	'CacheWarmingPlanRecord',
	'CacheEvictionReviewRecord',
	'CacheAuditEventRecord',
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
	'register_capability',
	'get_capability_info',
	'get_capability_contract',
	'evaluate_capability_rules',
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
	if CacheService is None or CacheServiceConfig is None:
		raise ModuleNotFoundError(
			"CACH service requires optional compression dependencies: lz4, zstandard"
		) from _SERVICE_IMPORT_ERROR
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


def register_capability() -> dict:
	"""Register cache management with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "cach",
		"aliases": ["cache_management", "cache", "caching_layer"],
		"display_name": "Cache Management",
		"description": __capability_description__,
		"version": __capability_version__,
		"dependencies": __capability_dependencies__,
		"optional_dependencies": __capability_optional_dependencies__,
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"cache_operations": "Read, write, delete, and inspect tenant-aware cache entries",
			"cache_namespace_lifecycle": "Register, disable, retire, and govern tenant cache namespaces",
			"cache_policy_governance": "Apply namespace, TTL, eviction, and security policies",
			"intelligent_warming": "Warm cache namespaces from configured data sources",
			"eviction_review": "Capture memory pressure, eviction plans, independent review, and audit notes",
			"adaptive_optimization": "Tune cache tiers from performance and access signals",
			"capability_rules": "Evaluate deterministic cache governance rules",
			"visual_theming": "Apply cache-control theme tokens and components"
		},
		"endpoints": {
			"entries": "/cach/api/v1/entries",
			"namespaces": "/cach/api/v1/namespaces",
			"policies": "/cach/api/v1/policies",
			"warming": "/cach/api/v1/warming",
			"evictions": "/cach/api/v1/evictions",
			"tiers": "/cach/api/v1/tiers",
			"adapters": "/cach/api/v1/adapters",
			"audit": "/cach/api/v1/audit",
			"analytics": "/cach/api/v1/analytics",
			"health": "/cach/api/v1/health"
		},
		"ui_components": {
			route["name"]: route["path"]
			for route in contract["ui"]["routes"]
		},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"permissions": [
			"cach:view",
			"cach:read",
			"cach:write",
			"cach:delete",
			"cach:manage_namespaces",
			"cach:manage_policies",
			"cach:warm",
			"cach:review_eviction",
			"cach:view_analytics",
			"cach:admin"
		]
	}


def get_capability_info() -> dict:
	"""Get CACH capability information for composition and marketplace discovery."""
	return {
		"metadata": CAPABILITY_METADATA,
		"contract": get_capability_contract(),
		"features": [
			"Tenant-aware cache namespaces",
			"Deterministic cache admission guardrails",
			"Cache warming request and review workflows",
			"Eviction and capacity review evidence",
			"Generated-application UI and theme metadata",
			"Backend-neutral cache adapter boundaries"
		]
	}


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
