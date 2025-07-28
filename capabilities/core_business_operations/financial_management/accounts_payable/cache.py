"""
APG Core Financials - Accounts Payable Caching Layer

High-performance caching implementation for AP operations using Redis
with APG platform integration and CLAUDE.md compliance.

© 2025 Datacraft. All rights reserved.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Union
from functools import wraps

import redis.asyncio as redis
from redis.asyncio import Redis

from .models import APVendor, APInvoice, APPayment


class APCacheService:
	"""High-performance caching service for AP operations"""
	
	def __init__(self, redis_url: str = "redis://localhost:6379"):
		self.redis_url = redis_url
		self.redis_client: Redis | None = None
		self._connected = False
		
	async def connect(self) -> None:
		"""Connect to Redis cache"""
		assert self.redis_url is not None, "Redis URL must be provided"
		
		try:
			self.redis_client = redis.from_url(
				self.redis_url,
				encoding="utf-8",
				decode_responses=True,
				max_connections=20
			)
			
			# Test connection
			await self.redis_client.ping()
			self._connected = True
			await self._log_cache_connection("Connected to Redis cache successfully")
			
		except Exception as e:
			await self._log_cache_connection(f"Failed to connect to Redis: {str(e)}")
			self._connected = False
			# Use in-memory fallback
			self._memory_cache: Dict[str, Any] = {}
	
	async def disconnect(self) -> None:
		"""Disconnect from Redis cache"""
		if self.redis_client:
			await self.redis_client.close()
		self._connected = False
		await self._log_cache_connection("Disconnected from Redis cache")
	
	async def get(self, key: str) -> Any | None:
		"""Get value from cache"""
		assert key is not None, "Cache key must be provided"
		
		if not self._connected:
			return self._memory_cache.get(key)
		
		try:
			value = await self.redis_client.get(key)
			if value:
				return json.loads(value)
			return None
		except Exception as e:
			await self._log_cache_operation(f"Cache get failed for key {key}: {str(e)}")
			return None
	
	async def set(
		self, 
		key: str, 
		value: Any, 
		ttl_seconds: int = 300
	) -> bool:
		"""Set value in cache with TTL"""
		assert key is not None, "Cache key must be provided"
		assert value is not None, "Cache value must be provided"
		assert ttl_seconds > 0, "TTL must be positive"
		
		if not self._connected:
			self._memory_cache[key] = value
			return True
		
		try:
			serialized_value = json.dumps(value, default=str)
			await self.redis_client.setex(key, ttl_seconds, serialized_value)
			return True
		except Exception as e:
			await self._log_cache_operation(f"Cache set failed for key {key}: {str(e)}")
			return False
	
	async def delete(self, key: str) -> bool:
		"""Delete value from cache"""
		assert key is not None, "Cache key must be provided"
		
		if not self._connected:
			return self._memory_cache.pop(key, None) is not None
		
		try:
			result = await self.redis_client.delete(key)
			return result > 0
		except Exception as e:
			await self._log_cache_operation(f"Cache delete failed for key {key}: {str(e)}")
			return False
	
	async def invalidate_pattern(self, pattern: str) -> int:
		"""Invalidate all keys matching pattern"""
		assert pattern is not None, "Pattern must be provided"
		
		if not self._connected:
			# Simple pattern matching for memory cache
			keys_to_delete = [k for k in self._memory_cache.keys() if pattern in k]
			for key in keys_to_delete:
				del self._memory_cache[key]
			return len(keys_to_delete)
		
		try:
			keys = await self.redis_client.keys(pattern)
			if keys:
				return await self.redis_client.delete(*keys)
			return 0
		except Exception as e:
			await self._log_cache_operation(f"Cache pattern invalidation failed for {pattern}: {str(e)}")
			return 0
	
	# Vendor-specific caching methods
	
	async def cache_vendor(self, vendor: APVendor, ttl_seconds: int = 600) -> bool:
		"""Cache vendor data"""
		assert vendor is not None, "Vendor must be provided"
		assert vendor.id is not None, "Vendor ID must be set"
		
		cache_key = f"ap:vendor:{vendor.id}"
		vendor_data = {
			"id": vendor.id,
			"vendor_code": vendor.vendor_code,
			"legal_name": vendor.legal_name,
			"status": vendor.status.value,
			"tenant_id": vendor.tenant_id,
			"cached_at": datetime.utcnow().isoformat()
		}
		
		return await self.set(cache_key, vendor_data, ttl_seconds)
	
	async def get_cached_vendor(self, vendor_id: str) -> Dict[str, Any] | None:
		"""Get cached vendor data"""
		assert vendor_id is not None, "Vendor ID must be provided"
		
		cache_key = f"ap:vendor:{vendor_id}"
		return await self.get(cache_key)
	
	async def invalidate_vendor_cache(self, vendor_id: str) -> bool:
		"""Invalidate vendor cache"""
		assert vendor_id is not None, "Vendor ID must be provided"
		
		# Invalidate vendor and related caches
		patterns = [
			f"ap:vendor:{vendor_id}",
			f"ap:vendor_invoices:{vendor_id}:*",
			f"ap:vendor_payments:{vendor_id}:*"
		]
		
		total_deleted = 0
		for pattern in patterns:
			total_deleted += await self.invalidate_pattern(pattern)
		
		return total_deleted > 0
	
	# Invoice-specific caching methods
	
	async def cache_invoice_processing_result(
		self, 
		invoice_id: str, 
		result: Dict[str, Any], 
		ttl_seconds: int = 1800
	) -> bool:
		"""Cache invoice processing result"""
		assert invoice_id is not None, "Invoice ID must be provided"
		assert result is not None, "Processing result must be provided"
		
		cache_key = f"ap:invoice_processing:{invoice_id}"
		return await self.set(cache_key, result, ttl_seconds)
	
	async def get_cached_invoice_processing(self, invoice_id: str) -> Dict[str, Any] | None:
		"""Get cached invoice processing result"""
		assert invoice_id is not None, "Invoice ID must be provided"
		
		cache_key = f"ap:invoice_processing:{invoice_id}"
		return await self.get(cache_key)
	
	# Payment-specific caching methods
	
	async def cache_payment_status(
		self, 
		payment_id: str, 
		status_data: Dict[str, Any], 
		ttl_seconds: int = 300
	) -> bool:
		"""Cache payment status data"""
		assert payment_id is not None, "Payment ID must be provided"
		assert status_data is not None, "Status data must be provided"
		
		cache_key = f"ap:payment_status:{payment_id}"
		return await self.set(cache_key, status_data, ttl_seconds)
	
	async def get_cached_payment_status(self, payment_id: str) -> Dict[str, Any] | None:
		"""Get cached payment status"""
		assert payment_id is not None, "Payment ID must be provided"
		
		cache_key = f"ap:payment_status:{payment_id}"
		return await self.get(cache_key)
	
	# Analytics and dashboard caching
	
	async def cache_dashboard_data(
		self, 
		tenant_id: str, 
		dashboard_data: Dict[str, Any], 
		ttl_seconds: int = 300
	) -> bool:
		"""Cache dashboard analytics data"""
		assert tenant_id is not None, "Tenant ID must be provided"
		assert dashboard_data is not None, "Dashboard data must be provided"
		
		cache_key = f"ap:dashboard:{tenant_id}"
		return await self.set(cache_key, dashboard_data, ttl_seconds)
	
	async def get_cached_dashboard_data(self, tenant_id: str) -> Dict[str, Any] | None:
		"""Get cached dashboard data"""
		assert tenant_id is not None, "Tenant ID must be provided"
		
		cache_key = f"ap:dashboard:{tenant_id}"
		return await self.get(cache_key)
	
	async def cache_aging_report(
		self, 
		tenant_id: str, 
		aging_data: Dict[str, Any], 
		ttl_seconds: int = 600
	) -> bool:
		"""Cache aging report data"""
		assert tenant_id is not None, "Tenant ID must be provided"
		assert aging_data is not None, "Aging data must be provided"
		
		cache_key = f"ap:aging:{tenant_id}"
		return await self.set(cache_key, aging_data, ttl_seconds)
	
	async def get_cached_aging_report(self, tenant_id: str) -> Dict[str, Any] | None:
		"""Get cached aging report"""
		assert tenant_id is not None, "Tenant ID must be provided"
		
		cache_key = f"ap:aging:{tenant_id}"
		return await self.get(cache_key)
	
	# Cache health and monitoring
	
	async def get_cache_stats(self) -> Dict[str, Any]:
		"""Get cache performance statistics"""
		if not self._connected:
			return {
				"status": "memory_fallback",
				"memory_keys": len(self._memory_cache),
				"connected": False
			}
		
		try:
			info = await self.redis_client.info()
			return {
				"status": "connected",
				"connected_clients": info.get("connected_clients", 0),
				"used_memory": info.get("used_memory_human", "Unknown"),
				"hits": info.get("keyspace_hits", 0),
				"misses": info.get("keyspace_misses", 0),
				"connected": True
			}
		except Exception as e:
			return {
				"status": "error",
				"error": str(e),
				"connected": False
			}
	
	async def _log_cache_connection(self, message: str) -> None:
		"""Log cache connection events"""
		print(f"AP Cache: {message}")
	
	async def _log_cache_operation(self, message: str) -> None:
		"""Log cache operation events"""
		print(f"AP Cache Operation: {message}")


# Caching decorators for service methods

def cache_result(ttl_seconds: int = 300, key_template: str = None):
	"""Decorator to cache method results"""
	def decorator(func):
		@wraps(func)
		async def wrapper(*args, **kwargs):
			# Extract cache service from first argument (self)
			cache_service = getattr(args[0], 'cache_service', None)
			if not cache_service:
				# No cache service available, execute directly
				return await func(*args, **kwargs)
			
			# Generate cache key
			if key_template:
				# Use provided template with method arguments
				cache_key = key_template.format(*args[1:], **kwargs)
			else:
				# Default key based on function name and arguments
				cache_key = f"ap:{func.__name__}:{hash(str(args[1:]) + str(kwargs))}"
			
			# Try to get from cache first
			cached_result = await cache_service.get(cache_key)
			if cached_result is not None:
				return cached_result
			
			# Execute function and cache result
			result = await func(*args, **kwargs)
			if result is not None:
				await cache_service.set(cache_key, result, ttl_seconds)
			
			return result
		return wrapper
	return decorator


def cache_invalidate(pattern: str = None):
	"""Decorator to invalidate cache after method execution"""
	def decorator(func):
		@wraps(func)
		async def wrapper(*args, **kwargs):
			# Execute function first
			result = await func(*args, **kwargs)
			
			# Invalidate cache after successful execution
			cache_service = getattr(args[0], 'cache_service', None)
			if cache_service and pattern:
				invalidation_pattern = pattern.format(*args[1:], **kwargs)
				await cache_service.invalidate_pattern(invalidation_pattern)
			
			return result
		return wrapper
	return decorator


# Global cache service instance
_global_cache_service: APCacheService | None = None


async def get_cache_service() -> APCacheService:
	"""Get global cache service instance"""
	global _global_cache_service
	
	if _global_cache_service is None:
		_global_cache_service = APCacheService()
		await _global_cache_service.connect()
	
	return _global_cache_service


async def shutdown_cache_service() -> None:
	"""Shutdown global cache service"""
	global _global_cache_service
	
	if _global_cache_service:
		await _global_cache_service.disconnect()
		_global_cache_service = None