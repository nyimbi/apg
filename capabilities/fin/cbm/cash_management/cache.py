"""
APG Cash Management - Caching Layer

APG-compatible caching infrastructure for high-performance operations.
Implements CLAUDE.md standards with async patterns and APG integration.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

import aioredis
from pydantic import BaseModel


class CashCacheManager:
	"""
	APG-compatible caching layer for cash management operations.
	
	Provides high-performance caching with automatic invalidation,
	cache warming, and APG multi-tenant isolation.
	"""
	
	def __init__(self, redis_client: aioredis.Redis, tenant_id: str):
		"""Initialize cache manager with APG multi-tenant support."""
		self.redis = redis_client
		self.tenant_id = tenant_id
		self.cache_prefix = f"cm:{tenant_id}"
		self._log_cache_init()
	
	# =========================================================================
	# Bank Data Caching
	# =========================================================================
	
	async def cache_bank(self, bank_id: str, bank_data: Dict[str, Any], ttl: int = 3600) -> None:
		"""Cache bank data with TTL."""
		assert bank_id is not None, "bank_id required for caching"
		
		cache_key = self._get_bank_key(bank_id)
		serialized_data = self._serialize_cache_data(bank_data)
		
		await self.redis.setex(cache_key, ttl, serialized_data)
		self._log_cache_set("bank", bank_id, ttl)
	
	async def get_cached_bank(self, bank_id: str) -> Optional[Dict[str, Any]]:
		"""Retrieve cached bank data."""
		assert bank_id is not None, "bank_id required for cache retrieval"
		
		cache_key = self._get_bank_key(bank_id)
		cached_data = await self.redis.get(cache_key)
		
		if cached_data:
			self._log_cache_hit("bank", bank_id)
			return self._deserialize_cache_data(cached_data)
		
		self._log_cache_miss("bank", bank_id)
		return None
	
	async def invalidate_bank_cache(self, bank_id: Optional[str] = None) -> int:
		"""Invalidate bank cache entries."""
		if bank_id:
			# Invalidate specific bank
			cache_key = self._get_bank_key(bank_id)
			deleted_count = await self.redis.delete(cache_key)
			self._log_cache_invalidate("bank", bank_id)
		else:
			# Invalidate all banks for tenant
			pattern = f"{self.cache_prefix}:bank:*"
			keys = await self.redis.keys(pattern)
			deleted_count = await self.redis.delete(*keys) if keys else 0
			self._log_cache_invalidate("banks", "all")
		
		return deleted_count
	
	# =========================================================================
	# Cash Account Caching
	# =========================================================================
	
	async def cache_account(self, account_id: str, account_data: Dict[str, Any], ttl: int = 1800) -> None:
		"""Cache cash account data with TTL."""
		assert account_id is not None, "account_id required for caching"
		
		cache_key = self._get_account_key(account_id)
		serialized_data = self._serialize_cache_data(account_data)
		
		await self.redis.setex(cache_key, ttl, serialized_data)
		self._log_cache_set("account", account_id, ttl)
	
	async def get_cached_account(self, account_id: str) -> Optional[Dict[str, Any]]:
		"""Retrieve cached account data."""
		assert account_id is not None, "account_id required for cache retrieval"
		
		cache_key = self._get_account_key(account_id)
		cached_data = await self.redis.get(cache_key)
		
		if cached_data:
			self._log_cache_hit("account", account_id)
			return self._deserialize_cache_data(cached_data)
		
		self._log_cache_miss("account", account_id)
		return None
	
	async def cache_account_balance(self, account_id: str, balance_data: Dict[str, Any], ttl: int = 300) -> None:
		"""Cache real-time account balance with short TTL."""
		assert account_id is not None, "account_id required for balance caching"
		
		cache_key = self._get_balance_key(account_id)
		serialized_data = self._serialize_cache_data(balance_data)
		
		await self.redis.setex(cache_key, ttl, serialized_data)
		self._log_cache_set("balance", account_id, ttl)
	
	async def get_cached_balance(self, account_id: str) -> Optional[Dict[str, Any]]:
		"""Retrieve cached balance data."""
		assert account_id is not None, "account_id required for balance retrieval"
		
		cache_key = self._get_balance_key(account_id)
		cached_data = await self.redis.get(cache_key)
		
		if cached_data:
			self._log_cache_hit("balance", account_id)
			return self._deserialize_cache_data(cached_data)
		
		self._log_cache_miss("balance", account_id)
		return None
	
	async def invalidate_account_cache(self, account_id: Optional[str] = None) -> int:
		"""Invalidate account cache entries."""
		if account_id:
			# Invalidate specific account and its balance
			account_key = self._get_account_key(account_id)
			balance_key = self._get_balance_key(account_id)
			deleted_count = await self.redis.delete(account_key, balance_key)
			self._log_cache_invalidate("account", account_id)
		else:
			# Invalidate all accounts for tenant
			pattern = f"{self.cache_prefix}:account:*"
			keys = await self.redis.keys(pattern)
			balance_pattern = f"{self.cache_prefix}:balance:*"
			balance_keys = await self.redis.keys(balance_pattern)
			all_keys = keys + balance_keys
			deleted_count = await self.redis.delete(*all_keys) if all_keys else 0
			self._log_cache_invalidate("accounts", "all")
		
		return deleted_count
	
	# =========================================================================
	# Cash Position Caching
	# =========================================================================
	
	async def cache_global_position(self, position_data: Dict[str, Any], ttl: int = 600) -> None:
		"""Cache global cash position with TTL."""
		cache_key = self._get_global_position_key()
		serialized_data = self._serialize_cache_data(position_data)
		
		await self.redis.setex(cache_key, ttl, serialized_data)
		self._log_cache_set("global_position", "current", ttl)
	
	async def get_cached_global_position(self) -> Optional[Dict[str, Any]]:
		"""Retrieve cached global cash position."""
		cache_key = self._get_global_position_key()
		cached_data = await self.redis.get(cache_key)
		
		if cached_data:
			self._log_cache_hit("global_position", "current")
			return self._deserialize_cache_data(cached_data)
		
		self._log_cache_miss("global_position", "current")
		return None
	
	async def cache_entity_position(self, entity_id: str, position_data: Dict[str, Any], ttl: int = 600) -> None:
		"""Cache entity-specific cash position."""
		assert entity_id is not None, "entity_id required for position caching"
		
		cache_key = self._get_entity_position_key(entity_id)
		serialized_data = self._serialize_cache_data(position_data)
		
		await self.redis.setex(cache_key, ttl, serialized_data)
		self._log_cache_set("entity_position", entity_id, ttl)
	
	async def get_cached_entity_position(self, entity_id: str) -> Optional[Dict[str, Any]]:
		"""Retrieve cached entity position."""
		assert entity_id is not None, "entity_id required for position retrieval"
		
		cache_key = self._get_entity_position_key(entity_id)
		cached_data = await self.redis.get(cache_key)
		
		if cached_data:
			self._log_cache_hit("entity_position", entity_id)
			return self._deserialize_cache_data(cached_data)
		
		self._log_cache_miss("entity_position", entity_id)
		return None
	
	async def invalidate_position_cache(self, entity_id: Optional[str] = None) -> int:
		"""Invalidate position cache entries."""
		if entity_id:
			# Invalidate specific entity position
			cache_key = self._get_entity_position_key(entity_id)
			deleted_count = await self.redis.delete(cache_key)
			self._log_cache_invalidate("entity_position", entity_id)
		else:
			# Invalidate all positions for tenant
			global_key = self._get_global_position_key()
			pattern = f"{self.cache_prefix}:position:*"
			keys = await self.redis.keys(pattern)
			all_keys = [global_key] + keys
			deleted_count = await self.redis.delete(*all_keys) if all_keys else 0
			self._log_cache_invalidate("positions", "all")
		
		return deleted_count
	
	# =========================================================================
	# Forecast Caching
	# =========================================================================
	
	async def cache_forecast(self, forecast_id: str, forecast_data: Dict[str, Any], ttl: int = 7200) -> None:
		"""Cache forecast data with TTL."""
		assert forecast_id is not None, "forecast_id required for caching"
		
		cache_key = self._get_forecast_key(forecast_id)
		serialized_data = self._serialize_cache_data(forecast_data)
		
		await self.redis.setex(cache_key, ttl, serialized_data)
		self._log_cache_set("forecast", forecast_id, ttl)
	
	async def get_cached_forecast(self, forecast_id: str) -> Optional[Dict[str, Any]]:
		"""Retrieve cached forecast data."""
		assert forecast_id is not None, "forecast_id required for forecast retrieval"
		
		cache_key = self._get_forecast_key(forecast_id)
		cached_data = await self.redis.get(cache_key)
		
		if cached_data:
			self._log_cache_hit("forecast", forecast_id)
			return self._deserialize_cache_data(cached_data)
		
		self._log_cache_miss("forecast", forecast_id)
		return None
	
	async def cache_forecast_accuracy(self, accuracy_data: Dict[str, Any], ttl: int = 3600) -> None:
		"""Cache forecast accuracy metrics."""
		cache_key = self._get_forecast_accuracy_key()
		serialized_data = self._serialize_cache_data(accuracy_data)
		
		await self.redis.setex(cache_key, ttl, serialized_data)
		self._log_cache_set("forecast_accuracy", "metrics", ttl)
	
	async def get_cached_forecast_accuracy(self) -> Optional[Dict[str, Any]]:
		"""Retrieve cached forecast accuracy metrics."""
		cache_key = self._get_forecast_accuracy_key()
		cached_data = await self.redis.get(cache_key)
		
		if cached_data:
			self._log_cache_hit("forecast_accuracy", "metrics")
			return self._deserialize_cache_data(cached_data)
		
		self._log_cache_miss("forecast_accuracy", "metrics")
		return None
	
	# =========================================================================
	# Investment Opportunity Caching
	# =========================================================================
	
	async def cache_investment_opportunities(self, opportunities_data: List[Dict[str, Any]], ttl: int = 900) -> None:
		"""Cache investment opportunities with short TTL."""
		cache_key = self._get_opportunities_key()
		serialized_data = self._serialize_cache_data(opportunities_data)
		
		await self.redis.setex(cache_key, ttl, serialized_data)
		self._log_cache_set("opportunities", "current", ttl)
	
	async def get_cached_investment_opportunities(self) -> Optional[List[Dict[str, Any]]]:
		"""Retrieve cached investment opportunities."""
		cache_key = self._get_opportunities_key()
		cached_data = await self.redis.get(cache_key)
		
		if cached_data:
			self._log_cache_hit("opportunities", "current")
			return self._deserialize_cache_data(cached_data)
		
		self._log_cache_miss("opportunities", "current")
		return None
	
	async def cache_optimization_result(self, optimization_key: str, result_data: Dict[str, Any], ttl: int = 1800) -> None:
		"""Cache investment optimization results."""
		assert optimization_key is not None, "optimization_key required for caching"
		
		cache_key = self._get_optimization_key(optimization_key)
		serialized_data = self._serialize_cache_data(result_data)
		
		await self.redis.setex(cache_key, ttl, serialized_data)
		self._log_cache_set("optimization", optimization_key, ttl)
	
	async def get_cached_optimization_result(self, optimization_key: str) -> Optional[Dict[str, Any]]:
		"""Retrieve cached optimization result."""
		assert optimization_key is not None, "optimization_key required for retrieval"
		
		cache_key = self._get_optimization_key(optimization_key)
		cached_data = await self.redis.get(cache_key)
		
		if cached_data:
			self._log_cache_hit("optimization", optimization_key)
			return self._deserialize_cache_data(cached_data)
		
		self._log_cache_miss("optimization", optimization_key)
		return None
	
	# =========================================================================
	# Alert Caching
	# =========================================================================
	
	async def cache_active_alerts(self, alerts_data: List[Dict[str, Any]], ttl: int = 300) -> None:
		"""Cache active alerts with short TTL."""
		cache_key = self._get_alerts_key()
		serialized_data = self._serialize_cache_data(alerts_data)
		
		await self.redis.setex(cache_key, ttl, serialized_data)
		self._log_cache_set("alerts", "active", ttl)
	
	async def get_cached_active_alerts(self) -> Optional[List[Dict[str, Any]]]:
		"""Retrieve cached active alerts."""
		cache_key = self._get_alerts_key()
		cached_data = await self.redis.get(cache_key)
		
		if cached_data:
			self._log_cache_hit("alerts", "active")
			return self._deserialize_cache_data(cached_data)
		
		self._log_cache_miss("alerts", "active")
		return None
	
	async def invalidate_alerts_cache(self) -> int:
		"""Invalidate alerts cache."""
		cache_key = self._get_alerts_key()
		deleted_count = await self.redis.delete(cache_key)
		self._log_cache_invalidate("alerts", "active")
		return deleted_count
	
	# =========================================================================
	# Session and Rate Limiting
	# =========================================================================
	
	async def set_rate_limit(self, user_id: str, operation: str, limit: int, window_seconds: int) -> bool:
		"""Implement rate limiting for operations."""
		assert user_id is not None, "user_id required for rate limiting"
		assert operation is not None, "operation required for rate limiting"
		
		rate_key = self._get_rate_limit_key(user_id, operation)
		
		# Use Redis sliding window rate limiting
		current_time = datetime.utcnow().timestamp()
		window_start = current_time - window_seconds
		
		# Remove old entries
		await self.redis.zremrangebyscore(rate_key, 0, window_start)
		
		# Count current requests
		current_count = await self.redis.zcard(rate_key)
		
		if current_count >= limit:
			self._log_rate_limit_exceeded(user_id, operation, current_count, limit)
			return False
		
		# Add current request
		await self.redis.zadd(rate_key, {str(current_time): current_time})
		await self.redis.expire(rate_key, window_seconds)
		
		self._log_rate_limit_allowed(user_id, operation, current_count + 1, limit)
		return True
	
	async def get_rate_limit_status(self, user_id: str, operation: str, window_seconds: int) -> Dict[str, int]:
		"""Get current rate limit status."""
		assert user_id is not None, "user_id required for rate limit status"
		assert operation is not None, "operation required for rate limit status"
		
		rate_key = self._get_rate_limit_key(user_id, operation)
		
		current_time = datetime.utcnow().timestamp()
		window_start = current_time - window_seconds
		
		# Count current requests in window
		current_count = await self.redis.zcount(rate_key, window_start, current_time)
		
		# Get time of oldest request
		oldest_requests = await self.redis.zrange(rate_key, 0, 0, withscores=True)
		reset_time = int(oldest_requests[0][1] + window_seconds) if oldest_requests else int(current_time + window_seconds)
		
		return {
			'current_count': current_count,
			'reset_time': reset_time,
			'window_seconds': window_seconds
		}
	
	# =========================================================================
	# Cache Warming and Preloading
	# =========================================================================
	
	async def warm_cache(self, warm_config: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
		"""Warm cache with frequently accessed data."""
		warm_config = warm_config or {
			'banks': True,
			'accounts': True,
			'positions': True,
			'alerts': True
		}
		
		warmed_counts = {}
		
		# This would integrate with the service layer to preload data
		# Simplified implementation for demonstration
		
		if warm_config.get('banks'):
			# Would preload bank data
			warmed_counts['banks'] = 0
		
		if warm_config.get('accounts'):
			# Would preload account data
			warmed_counts['accounts'] = 0
		
		if warm_config.get('positions'):
			# Would preload position data
			warmed_counts['positions'] = 0
		
		if warm_config.get('alerts'):
			# Would preload alert data
			warmed_counts['alerts'] = 0
		
		self._log_cache_warmed(warmed_counts)
		return warmed_counts
	
	# =========================================================================
	# Cache Management and Statistics
	# =========================================================================
	
	async def get_cache_stats(self) -> Dict[str, Any]:
		"""Get cache statistics for monitoring."""
		# Get Redis info
		redis_info = await self.redis.info()
		
		# Get tenant-specific key counts
		patterns = [
			f"{self.cache_prefix}:bank:*",
			f"{self.cache_prefix}:account:*",
			f"{self.cache_prefix}:balance:*",
			f"{self.cache_prefix}:position:*",
			f"{self.cache_prefix}:forecast:*",
			f"{self.cache_prefix}:alerts:*",
			f"{self.cache_prefix}:opportunities:*",
			f"{self.cache_prefix}:optimization:*",
			f"{self.cache_prefix}:rate_limit:*"
		]
		
		key_counts = {}
		total_keys = 0
		
		for pattern in patterns:
			keys = await self.redis.keys(pattern)
			cache_type = pattern.split(':')[2].rstrip('*')
			key_counts[cache_type] = len(keys)
			total_keys += len(keys)
		
		return {
			'tenant_id': self.tenant_id,
			'total_keys': total_keys,
			'key_counts_by_type': key_counts,
			'redis_used_memory': redis_info.get('used_memory', 0),
			'redis_used_memory_human': redis_info.get('used_memory_human', '0B'),
			'redis_connected_clients': redis_info.get('connected_clients', 0),
			'redis_keyspace_hits': redis_info.get('keyspace_hits', 0),
			'redis_keyspace_misses': redis_info.get('keyspace_misses', 0),
			'cache_hit_ratio': self._calculate_hit_ratio(
				redis_info.get('keyspace_hits', 0),
				redis_info.get('keyspace_misses', 0)
			)
		}
	
	async def clear_tenant_cache(self) -> int:
		"""Clear all cache for tenant."""
		pattern = f"{self.cache_prefix}:*"
		keys = await self.redis.keys(pattern)
		deleted_count = await self.redis.delete(*keys) if keys else 0
		
		self._log_cache_cleared(deleted_count)
		return deleted_count
	
	async def set_cache_health_check(self) -> bool:
		"""Set cache health check marker."""
		health_key = f"{self.cache_prefix}:health"
		health_data = {
			'timestamp': datetime.utcnow().isoformat(),
			'tenant_id': self.tenant_id,
			'status': 'healthy'
		}
		
		try:
			await self.redis.setex(health_key, 60, json.dumps(health_data))
			return True
		except Exception:
			return False
	
	async def get_cache_health(self) -> Dict[str, Any]:
		"""Get cache health status."""
		health_key = f"{self.cache_prefix}:health"
		
		try:
			health_data = await self.redis.get(health_key)
			if health_data:
				return json.loads(health_data)
			else:
				return {'status': 'unknown', 'timestamp': None}
		except Exception as e:
			return {'status': 'error', 'error': str(e)}
	
	# =========================================================================
	# Private Helper Methods
	# =========================================================================
	
	def _get_bank_key(self, bank_id: str) -> str:
		"""Generate cache key for bank data."""
		return f"{self.cache_prefix}:bank:{bank_id}"
	
	def _get_account_key(self, account_id: str) -> str:
		"""Generate cache key for account data."""
		return f"{self.cache_prefix}:account:{account_id}"
	
	def _get_balance_key(self, account_id: str) -> str:
		"""Generate cache key for balance data."""
		return f"{self.cache_prefix}:balance:{account_id}"
	
	def _get_global_position_key(self) -> str:
		"""Generate cache key for global position."""
		return f"{self.cache_prefix}:position:global"
	
	def _get_entity_position_key(self, entity_id: str) -> str:
		"""Generate cache key for entity position."""
		return f"{self.cache_prefix}:position:entity:{entity_id}"
	
	def _get_forecast_key(self, forecast_id: str) -> str:
		"""Generate cache key for forecast data."""
		return f"{self.cache_prefix}:forecast:{forecast_id}"
	
	def _get_forecast_accuracy_key(self) -> str:
		"""Generate cache key for forecast accuracy."""
		return f"{self.cache_prefix}:forecast:accuracy"
	
	def _get_opportunities_key(self) -> str:
		"""Generate cache key for investment opportunities."""
		return f"{self.cache_prefix}:opportunities"
	
	def _get_optimization_key(self, optimization_key: str) -> str:
		"""Generate cache key for optimization results."""
		return f"{self.cache_prefix}:optimization:{optimization_key}"
	
	def _get_alerts_key(self) -> str:
		"""Generate cache key for alerts."""
		return f"{self.cache_prefix}:alerts:active"
	
	def _get_rate_limit_key(self, user_id: str, operation: str) -> str:
		"""Generate cache key for rate limiting."""
		return f"{self.cache_prefix}:rate_limit:{user_id}:{operation}"
	
	def _serialize_cache_data(self, data: Any) -> str:
		"""Serialize data for cache storage."""
		class CacheEncoder(json.JSONEncoder):
			def default(self, obj):
				if isinstance(obj, Decimal):
					return float(obj)
				elif isinstance(obj, (datetime,)):
					return obj.isoformat()
				elif isinstance(obj, UUID):
					return str(obj)
				elif isinstance(obj, BaseModel):
					return obj.model_dump()
				return super().default(obj)
		
		return json.dumps(data, cls=CacheEncoder)
	
	def _deserialize_cache_data(self, cached_data: str) -> Any:
		"""Deserialize data from cache storage."""
		return json.loads(cached_data)
	
	def _calculate_hit_ratio(self, hits: int, misses: int) -> float:
		"""Calculate cache hit ratio."""
		total = hits + misses
		return (hits / total * 100) if total > 0 else 0.0
	
	# Logging methods
	def _log_cache_init(self) -> None:
		"""Log cache initialization."""
		print(f"CashCacheManager initialized for tenant: {self.tenant_id}")
	
	def _log_cache_set(self, cache_type: str, key: str, ttl: int) -> None:
		"""Log cache set operation."""
		print(f"Cache SET {cache_type}:{key} TTL={ttl}s")
	
	def _log_cache_hit(self, cache_type: str, key: str) -> None:
		"""Log cache hit."""
		print(f"Cache HIT {cache_type}:{key}")
	
	def _log_cache_miss(self, cache_type: str, key: str) -> None:
		"""Log cache miss."""
		print(f"Cache MISS {cache_type}:{key}")
	
	def _log_cache_invalidate(self, cache_type: str, key: str) -> None:
		"""Log cache invalidation."""
		print(f"Cache INVALIDATE {cache_type}:{key}")
	
	def _log_rate_limit_exceeded(self, user_id: str, operation: str, current: int, limit: int) -> None:
		"""Log rate limit exceeded."""
		print(f"Rate limit EXCEEDED {user_id}:{operation} {current}/{limit}")
	
	def _log_rate_limit_allowed(self, user_id: str, operation: str, current: int, limit: int) -> None:
		"""Log rate limit allowed."""
		print(f"Rate limit OK {user_id}:{operation} {current}/{limit}")
	
	def _log_cache_warmed(self, counts: Dict[str, int]) -> None:
		"""Log cache warming."""
		total = sum(counts.values())
		print(f"Cache warmed: {total} items - {counts}")
	
	def _log_cache_cleared(self, count: int) -> None:
		"""Log cache clearing."""
		print(f"Cache cleared: {count} keys for tenant {self.tenant_id}")


# Export cache manager
__all__ = [
	'CashCacheManager'
]