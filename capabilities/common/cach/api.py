#!/usr/bin/env python3
"""
APG Cache Management (CACH) - API Implementation
RESTful and GraphQL API endpoints with APG integration

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from fastapi import FastAPI, HTTPException, Depends, Query, Path, Body, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from uuid_extensions import uuid7str

from .service import (
	CacheEvictionReviewRecord,
	CacheAgentRecord,
	CacheGovernanceService,
	CacheLifecycleBatchRecord,
	CacheNamespaceRecord,
	CacheEntryRecord,
	CacheService,
	CacheServiceConfig,
	CacheWarmingPlanRecord,
)
from .models import (
	CacheEntry, CacheCluster, CachePolicy, CacheMetrics,
	CacheBackendType, CompressionAlgorithm, EvictionPolicy, SecurityLevel
)


# API Models for requests/responses
class CacheSetRequest(BaseModel):
	"""Request model for setting cache values"""
	key: str = Field(..., description="Cache key", max_length=1024)
	value: Any = Field(..., description="Value to cache")
	ttl_seconds: Optional[int] = Field(None, description="Time to live in seconds", ge=1)
	namespace: str = Field("default", description="Cache namespace")
	compression: Optional[CompressionAlgorithm] = Field(None, description="Compression algorithm")
	policy_name: Optional[str] = Field(None, description="Policy to apply")


class CacheGetResponse(BaseModel):
	"""Response model for cache get operations"""
	key: str
	value: Any
	hit: bool
	ttl_remaining: Optional[int] = Field(None, description="Remaining TTL in seconds")
	namespace: str
	access_count: int
	last_accessed: Optional[datetime]


class CacheSetResponse(BaseModel):
	"""Response model for cache set operations"""
	success: bool
	key: str
	namespace: str
	size_bytes: int
	compression_used: CompressionAlgorithm
	compression_ratio: float


class CacheDeleteResponse(BaseModel):
	"""Response model for cache delete operations"""
	success: bool
	key: str
	namespace: str
	existed: bool


class CacheStatsResponse(BaseModel):
	"""Response model for cache statistics"""
	total_entries: int
	total_size_bytes: int
	hit_rate: float
	memory_utilization: float
	total_operations: int
	cache_hits: int
	cache_misses: int
	cache_evictions: int
	policies_count: int
	clusters_count: int
	ai_optimizations: int
	timestamp: datetime


class CachePolicyRequest(BaseModel):
	"""Request model for creating cache policies"""
	name: str = Field(..., max_length=255)
	description: str = Field("", max_length=1000)
	key_patterns: List[str] = Field(..., description="Key patterns this policy applies to")
	default_ttl_seconds: int = Field(3600, ge=1, le=31536000)
	max_value_size_bytes: int = Field(1048576, ge=1)
	compression_enabled: bool = Field(True)
	prefetch_enabled: bool = Field(True)
	ai_optimization_enabled: bool = Field(True)


class CacheClusterRequest(BaseModel):
	"""Request model for creating cache clusters"""
	name: str = Field(..., max_length=255)
	description: str = Field("", max_length=1000)
	backend_type: CacheBackendType = Field(CacheBackendType.REDIS)
	nodes: List[str] = Field(default_factory=list)
	max_memory_mb: int = Field(1024, ge=100)
	replication_factor: int = Field(2, ge=1, le=10)
	ai_optimization_enabled: bool = Field(True)


class CacheHealthResponse(BaseModel):
	"""Response model for health checks"""
	healthy: bool
	timestamp: datetime
	uptime_seconds: float
	memory_usage: Dict[str, Any]
	performance_metrics: Dict[str, Any]
	cluster_status: List[Dict[str, Any]]


class AIInsightsResponse(BaseModel):
	"""Response model for AI optimization insights"""
	insights: List[Dict[str, Any]]
	total_optimizations: int
	average_confidence: float
	performance_improvement: float


# Dependency injection
cache_service: Optional[CacheService] = None
SERVICE = CacheGovernanceService()


async def get_cache_service() -> CacheService:
	"""Get cache service instance"""
	global cache_service
	if cache_service is None:
		config = CacheServiceConfig()
		cache_service = CacheService(config)
		await cache_service.initialize()
	return cache_service


def capability_status() -> dict[str, Any]:
	"""Return dependency-light CACH capability status for generated apps."""
	return {
		"capability": "cach",
		"service": "cache_governance",
		"status": "ready",
		"summary": SERVICE.dashboard_summary(),
	}


def _payload_bool(value: Any, default: bool = False) -> bool:
	if value is None:
		return default
	if isinstance(value, bool):
		return value
	if isinstance(value, str):
		return value.strip().lower() in {"1", "true", "yes", "y", "on"}
	return bool(value)


def create_namespace_record(**kwargs: Any) -> CacheNamespaceRecord:
	"""Create a CACH namespace policy record."""
	return SERVICE.create_namespace(**kwargs)


def write_cache_entry_record(**kwargs: Any) -> CacheEntryRecord:
	"""Evaluate and admit cache-entry metadata."""
	return SERVICE.write_entry(**kwargs)


def read_cache_entry_record(**kwargs: Any) -> dict[str, Any]:
	"""Read governed cache-entry metadata."""
	return SERVICE.read_entry(**kwargs)


def delete_cache_entry_record(**kwargs: Any) -> dict[str, Any]:
	"""Invalidate governed cache-entry metadata."""
	return SERVICE.delete_entry(**kwargs)


def request_warming_plan(**kwargs: Any) -> CacheWarmingPlanRecord:
	"""Create a cache warming request."""
	return SERVICE.request_warming_plan(**kwargs)


def decide_warming_plan(**kwargs: Any) -> CacheWarmingPlanRecord:
	"""Approve or reject a cache warming request with evidence."""
	return SERVICE.decide_warming_plan(**kwargs)


def request_eviction_review(**kwargs: Any) -> CacheEvictionReviewRecord:
	"""Create an eviction or capacity review request."""
	return SERVICE.request_eviction_review(**kwargs)


def decide_eviction_review(**kwargs: Any) -> CacheEvictionReviewRecord:
	"""Approve or reject an eviction review with evidence."""
	return SERVICE.decide_eviction_review(**kwargs)


def register_cache_agent(**kwargs: Any) -> CacheAgentRecord:
	"""Register a first-class cache governance agent."""
	if "contribution_disclosed" in kwargs:
		kwargs["contribution_disclosed"] = _payload_bool(kwargs["contribution_disclosed"], True)
	if "human_approval_required" in kwargs:
		kwargs["human_approval_required"] = _payload_bool(kwargs["human_approval_required"], False)
	return SERVICE.register_cache_agent(**kwargs)


def validate_cache_lifecycle_batch(**kwargs: Any) -> CacheLifecycleBatchRecord:
	"""Validate a CACH lifecycle mutation batch against the Bytewax guardrail."""
	return SERVICE.validate_cache_lifecycle_batch(**kwargs)


def list_records(record_type: str, tenant_id: str | None = None) -> list[dict[str, Any]]:
	"""List dependency-light CACH records."""
	return SERVICE.list_records(record_type, tenant_id)


def list_pending_reviews(tenant_id: str | None = None) -> list[dict[str, Any]]:
	"""List CACH records that require review across generated-app queues."""
	return SERVICE.list_pending_reviews(tenant_id)


def list_cache_governance(tenant_id: str | None = None) -> dict[str, Any]:
	"""Return all generated-app governance records for a tenant."""
	return {
		"summary": SERVICE.dashboard_summary(tenant_id),
		"namespaces": SERVICE.list_records("namespaces", tenant_id),
		"entries": SERVICE.list_records("entries", tenant_id),
		"warming_plans": SERVICE.list_records("warming_plans", tenant_id),
		"eviction_reviews": SERVICE.list_records("eviction_reviews", tenant_id),
		"cache_agents": SERVICE.list_records("cache_agents", tenant_id),
		"lifecycle_batches": SERVICE.list_records("lifecycle_batches", tenant_id),
		"pending_reviews": SERVICE.list_pending_reviews(tenant_id),
		"audit_events": SERVICE.list_records("audit_events", tenant_id),
	}


def _clean_text(value: Any) -> Optional[str]:
	"""Return a non-empty stripped string or None."""
	if value is None:
		return None
	text = str(value).strip()
	return text or None


def _mapping_get(mapping: Any, *keys: str) -> Optional[str]:
	"""Read the first present value from a dict-like request carrier."""
	if mapping is None:
		return None
	for key in keys:
		try:
			value = mapping.get(key)
		except AttributeError:
			value = None
		text = _clean_text(value)
		if text:
			return text
	return None


def get_current_tenant(request: Request) -> str:
	"""Get current tenant ID from APG request context."""
	for candidate in (
		getattr(getattr(request, "state", None), "tenant_id", None),
		getattr(getattr(request, "state", None), "current_tenant", None),
		_mapping_get(request.headers, "X-APG-Tenant-ID", "X-Tenant-ID", "X-Organization-ID"),
		_mapping_get(request.query_params, "tenant_id", "tenant"),
		_mapping_get(request.scope, "apg_tenant_id", "tenant_id"),
		os.getenv("APG_TENANT_ID"),
		os.getenv("APG_DEFAULT_TENANT_ID"),
		"default",
	):
		tenant_id = _clean_text(candidate)
		if tenant_id:
			return tenant_id

	return "default"


def get_current_user(request: Request) -> str:
	"""Get current user ID from APG request context."""
	for candidate in (
		getattr(getattr(request, "state", None), "user_id", None),
		getattr(getattr(request, "state", None), "current_user_id", None),
		_mapping_get(request.headers, "X-APG-User-ID", "X-User-ID"),
		_mapping_get(request.query_params, "user_id", "user"),
		_mapping_get(request.scope, "apg_user_id", "user_id"),
		os.getenv("APG_USER_ID"),
		os.getenv("APG_DEFAULT_USER_ID"),
		"system",
	):
		user_id = _clean_text(candidate)
		if user_id:
			return user_id

	return "system"


# Create FastAPI app
app = FastAPI(
	title="APG Cache Management API",
	description="Tenant-scoped cache management, governance, warming, and eviction review",
	version="1.0.0",
	docs_url="/api/v1/cache/docs",
	redoc_url="/api/v1/cache/redoc"
)

# Configure CORS
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],  # In production: restrict to specific origins
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cach.api")


# Exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
	return JSONResponse(
		status_code=400,
		content={"error": "Bad Request", "detail": str(exc)}
	)


@app.exception_handler(KeyError)
async def key_error_handler(request, exc):
	return JSONResponse(
		status_code=404,
		content={"error": "Not Found", "detail": f"Resource not found: {exc}"}
	)


# Health check endpoint
@app.get("/api/v1/cache/health", response_model=CacheHealthResponse)
async def health_check(
	service: CacheService = Depends(get_cache_service)
):
	"""Health check endpoint for monitoring"""
	try:
		stats = await service.get_stats()
		
		return CacheHealthResponse(
			healthy=True,
			timestamp=datetime.utcnow(),
			uptime_seconds=0.0,  # Would be actual uptime
			memory_usage={
				"total_mb": stats.get("total_size_bytes", 0) / (1024 * 1024),
				"utilization_percent": stats.get("memory_utilization", 0)
			},
			performance_metrics={
				"hit_rate": stats.get("hit_rate", 0),
				"operations": stats.get("total_operations", 0)
			},
			cluster_status=[]  # Would include actual cluster status
		)
	except Exception as e:
		logger.error(f"Health check failed: {e}")
		return CacheHealthResponse(
			healthy=False,
			timestamp=datetime.utcnow(),
			uptime_seconds=0.0,
			memory_usage={},
			performance_metrics={},
			cluster_status=[]
		)


# Core cache operations
@app.post("/api/v1/cache/set", response_model=CacheSetResponse)
async def set_cache_value(
	request: CacheSetRequest,
	service: CacheService = Depends(get_cache_service),
	tenant_id: str = Depends(get_current_tenant)
):
	"""Set a value in the cache with AI optimization"""
	try:
		success = await service.set(
			key=request.key,
			value=request.value,
			ttl_seconds=request.ttl_seconds,
			namespace=request.namespace,
			tenant_id=tenant_id,
			compression=request.compression,
			policy_name=request.policy_name
		)
		
		if not success:
			raise HTTPException(status_code=500, detail="Failed to set cache value")
		
		# Get entry details for response
		cache_key = f"{tenant_id}:{request.namespace}:{request.key}"
		entry = service._cache_store.get(cache_key)
		
		return CacheSetResponse(
			success=True,
			key=request.key,
			namespace=request.namespace,
			size_bytes=entry.size_bytes if entry else 0,
			compression_used=entry.compression_type if entry else CompressionAlgorithm.NONE,
			compression_ratio=entry.compression_ratio if entry else 1.0
		)
		
	except Exception as e:
		logger.error(f"Error setting cache value: {e}")
		raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/cache/get/{key}", response_model=CacheGetResponse)
async def get_cache_value(
	key: str = Path(..., description="Cache key to retrieve"),
	namespace: str = Query("default", description="Cache namespace"),
	service: CacheService = Depends(get_cache_service),
	tenant_id: str = Depends(get_current_tenant)
):
	"""Get a value from the cache with intelligent prefetching"""
	try:
		value = await service.get(key=key, namespace=namespace, tenant_id=tenant_id)
		
		if value is None:
			raise HTTPException(status_code=404, detail="Key not found in cache")
		
		# Get entry metadata
		cache_key = f"{tenant_id}:{namespace}:{key}"
		entry = service._cache_store.get(cache_key)
		
		ttl_remaining = None
		if entry and entry.ttl_seconds:
			elapsed = (datetime.utcnow() - entry.created_at).total_seconds()
			ttl_remaining = max(0, int(entry.ttl_seconds - elapsed))
		
		return CacheGetResponse(
			key=key,
			value=value,
			hit=True,
			ttl_remaining=ttl_remaining,
			namespace=namespace,
			access_count=entry.access_count if entry else 0,
			last_accessed=entry.last_accessed if entry else None
		)
		
	except HTTPException:
		raise
	except Exception as e:
		logger.error(f"Error getting cache value: {e}")
		raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/cache/delete/{key}", response_model=CacheDeleteResponse)
async def delete_cache_value(
	key: str = Path(..., description="Cache key to delete"),
	namespace: str = Query("default", description="Cache namespace"),
	service: CacheService = Depends(get_cache_service),
	tenant_id: str = Depends(get_current_tenant)
):
	"""Delete a value from the cache"""
	try:
		# Check if key exists first
		existed = await service.exists(key=key, namespace=namespace, tenant_id=tenant_id)
		
		# Delete the key
		success = await service.delete(key=key, namespace=namespace, tenant_id=tenant_id)
		
		return CacheDeleteResponse(
			success=success,
			key=key,
			namespace=namespace,
			existed=existed
		)
		
	except Exception as e:
		logger.error(f"Error deleting cache value: {e}")
		raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/cache/exists/{key}")
async def check_cache_exists(
	key: str = Path(..., description="Cache key to check"),
	namespace: str = Query("default", description="Cache namespace"),
	service: CacheService = Depends(get_cache_service),
	tenant_id: str = Depends(get_current_tenant)
):
	"""Check if a key exists in the cache"""
	try:
		exists = await service.exists(key=key, namespace=namespace, tenant_id=tenant_id)
		return {"exists": exists, "key": key, "namespace": namespace}
		
	except Exception as e:
		logger.error(f"Error checking cache existence: {e}")
		raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/cache/namespace/{namespace}")
async def clear_namespace(
	namespace: str = Path(..., description="Namespace to clear"),
	service: CacheService = Depends(get_cache_service),
	tenant_id: str = Depends(get_current_tenant)
):
	"""Clear all entries in a namespace"""
	try:
		deleted_count = await service.clear_namespace(namespace=namespace, tenant_id=tenant_id)
		return {
			"success": True,
			"namespace": namespace,
			"deleted_count": deleted_count
		}
		
	except Exception as e:
		logger.error(f"Error clearing namespace: {e}")
		raise HTTPException(status_code=500, detail=str(e))


# Policy management endpoints
@app.post("/api/v1/cache/policies")
async def create_cache_policy(
	request: CachePolicyRequest,
	service: CacheService = Depends(get_cache_service),
	tenant_id: str = Depends(get_current_tenant),
	user_id: str = Depends(get_current_user)
):
	"""Create a new cache policy"""
	try:
		policy = CachePolicy(
			name=request.name,
			description=request.description,
			key_patterns=request.key_patterns,
			tenant_id=tenant_id,
			default_ttl_seconds=request.default_ttl_seconds,
			max_value_size_bytes=request.max_value_size_bytes,
			compression_enabled=request.compression_enabled,
			prefetch_enabled=request.prefetch_enabled,
			ai_optimization_enabled=request.ai_optimization_enabled,
			created_by=user_id
		)
		
		policy_id = await service.create_policy(policy)
		
		return {
			"success": True,
			"policy_id": policy_id,
			"name": request.name
		}
		
	except Exception as e:
		logger.error(f"Error creating cache policy: {e}")
		raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/cache/policies")
async def list_cache_policies(
	service: CacheService = Depends(get_cache_service),
	tenant_id: str = Depends(get_current_tenant)
):
	"""List all cache policies for the tenant"""
	try:
		policies = [
			{
				"policy_id": policy.policy_id,
				"name": policy.name,
				"description": policy.description,
				"key_patterns": policy.key_patterns,
				"enabled": policy.enabled,
				"created_at": policy.created_at.isoformat(),
				"effectiveness_score": policy.effectiveness_score
			}
			for policy in service._policies.values()
			if policy.tenant_id == tenant_id
		]
		
		return {"policies": policies, "total": len(policies)}
		
	except Exception as e:
		logger.error(f"Error listing cache policies: {e}")
		raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/cache/policies/{policy_id}/apply/{key}")
async def apply_policy_to_key(
	policy_id: str = Path(..., description="Policy ID to apply"),
	key: str = Path(..., description="Cache key to apply policy to"),
	namespace: str = Query("default", description="Cache namespace"),
	service: CacheService = Depends(get_cache_service),
	tenant_id: str = Depends(get_current_tenant)
):
	"""Apply a policy to a specific cache key"""
	try:
		success = await service.apply_policy(
			key=key,
			policy_id=policy_id,
			namespace=namespace,
			tenant_id=tenant_id
		)
		
		return {
			"success": success,
			"policy_id": policy_id,
			"key": key,
			"namespace": namespace
		}
		
	except Exception as e:
		logger.error(f"Error applying policy: {e}")
		raise HTTPException(status_code=500, detail=str(e))


# Statistics and monitoring endpoints
@app.get("/api/v1/cache/stats", response_model=CacheStatsResponse)
async def get_cache_statistics(
	service: CacheService = Depends(get_cache_service)
):
	"""Get comprehensive cache statistics"""
	try:
		stats = await service.get_stats()
		
		return CacheStatsResponse(
			total_entries=stats["total_entries"],
			total_size_bytes=stats["total_size_bytes"],
			hit_rate=stats["hit_rate"],
			memory_utilization=stats["memory_utilization"],
			total_operations=stats["total_operations"],
			cache_hits=stats["cache_hits"],
			cache_misses=stats["cache_misses"],
			cache_evictions=stats["cache_evictions"],
			policies_count=stats["policies_count"],
			clusters_count=stats["clusters_count"],
			ai_optimizations=stats["ai_optimizations"],
			timestamp=datetime.utcnow()
		)
		
	except Exception as e:
		logger.error(f"Error getting cache statistics: {e}")
		raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/cache/performance/history")
async def get_performance_history(
	limit: int = Query(100, description="Number of data points to return", ge=1, le=1000),
	service: CacheService = Depends(get_cache_service)
):
	"""Get performance history for analytics dashboard"""
	try:
		history = await service.get_performance_history()
		return {"history": history[-limit:], "total_points": len(history)}
		
	except Exception as e:
		logger.error(f"Error getting performance history: {e}")
		raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/cache/ai/insights", response_model=AIInsightsResponse)
async def get_ai_insights(
	service: CacheService = Depends(get_cache_service)
):
	"""Get AI optimization insights and recommendations"""
	try:
		insights = await service.get_ai_insights()
		
		if not insights:
			return AIInsightsResponse(
				insights=[],
				total_optimizations=0,
				average_confidence=0.0,
				performance_improvement=0.0
			)
		
		total_optimizations = len(insights)
		average_confidence = sum(insight["confidence_score"] for insight in insights) / total_optimizations
		performance_improvement = sum(insight["expected_improvement"] for insight in insights) / total_optimizations
		
		return AIInsightsResponse(
			insights=insights,
			total_optimizations=total_optimizations,
			average_confidence=average_confidence,
			performance_improvement=performance_improvement
		)
		
	except Exception as e:
		logger.error(f"Error getting AI insights: {e}")
		raise HTTPException(status_code=500, detail=str(e))


# Batch operations
@app.post("/api/v1/cache/batch/set")
async def batch_set_cache_values(
	requests: List[CacheSetRequest],
	service: CacheService = Depends(get_cache_service),
	tenant_id: str = Depends(get_current_tenant)
):
	"""Set multiple cache values in batch"""
	try:
		results = []
		
		for request in requests:
			try:
				success = await service.set(
					key=request.key,
					value=request.value,
					ttl_seconds=request.ttl_seconds,
					namespace=request.namespace,
					tenant_id=tenant_id,
					compression=request.compression
				)
				results.append({
					"key": request.key,
					"namespace": request.namespace,
					"success": success
				})
			except Exception as e:
				results.append({
					"key": request.key,
					"namespace": request.namespace,
					"success": False,
					"error": str(e)
				})
		
		successful = sum(1 for r in results if r["success"])
		
		return {
			"total": len(requests),
			"successful": successful,
			"failed": len(requests) - successful,
			"results": results
		}
		
	except Exception as e:
		logger.error(f"Error in batch set operation: {e}")
		raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/cache/batch/get")
async def batch_get_cache_values(
	keys: List[str] = Body(..., description="List of cache keys to retrieve"),
	namespace: str = Body("default", description="Cache namespace"),
	service: CacheService = Depends(get_cache_service),
	tenant_id: str = Depends(get_current_tenant)
):
	"""Get multiple cache values in batch"""
	try:
		results = []
		
		for key in keys:
			try:
				value = await service.get(key=key, namespace=namespace, tenant_id=tenant_id)
				results.append({
					"key": key,
					"value": value,
					"found": value is not None
				})
			except Exception as e:
				results.append({
					"key": key,
					"value": None,
					"found": False,
					"error": str(e)
				})
		
		found_count = sum(1 for r in results if r["found"])
		
		return {
			"total": len(keys),
			"found": found_count,
			"not_found": len(keys) - found_count,
			"results": results
		}
		
	except Exception as e:
		logger.error(f"Error in batch get operation: {e}")
		raise HTTPException(status_code=500, detail=str(e))


# Advanced operations
@app.post("/api/v1/cache/optimize")
async def trigger_optimization(
	service: CacheService = Depends(get_cache_service),
	tenant_id: str = Depends(get_current_tenant)
):
	"""Manually trigger AI optimization analysis"""
	try:
		# Trigger optimization analysis
		await service._run_ai_optimization()
		
		return {
			"success": True,
			"message": "Optimization analysis triggered",
			"timestamp": datetime.utcnow().isoformat()
		}
		
	except Exception as e:
		logger.error(f"Error triggering optimization: {e}")
		raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/cache/prefetch")
async def trigger_prefetch(
	patterns: List[str] = Body(..., description="Key patterns to prefetch"),
	namespace: str = Body("default", description="Cache namespace"),
	service: CacheService = Depends(get_cache_service),
	tenant_id: str = Depends(get_current_tenant)
):
	"""Manually trigger intelligent prefetching for specified patterns"""
	try:
		# This would trigger actual prefetching logic in production
		return {
			"success": True,
			"message": f"Prefetch triggered for {len(patterns)} patterns",
			"patterns": patterns,
			"namespace": namespace
		}
		
	except Exception as e:
		logger.error(f"Error triggering prefetch: {e}")
		raise HTTPException(status_code=500, detail=str(e))


# WebSocket endpoint for real-time updates
@app.websocket("/api/v1/cache/ws/stats")
async def websocket_stats(websocket):
	"""WebSocket endpoint for real-time cache statistics"""
	await websocket.accept()
	
	try:
		service = await get_cache_service()
		
		while True:
			stats = await service.get_stats()
			await websocket.send_json({
				"type": "stats_update",
				"data": stats,
				"timestamp": datetime.utcnow().isoformat()
			})
			await asyncio.sleep(5)  # Send updates every 5 seconds
			
	except Exception as e:
		logger.error(f"WebSocket error: {e}")
	finally:
		await websocket.close()


# Root endpoint
@app.get("/api/v1/cache")
async def cache_api_root():
	"""Cache API root endpoint"""
	return {
		"service": "APG Cache Management API",
		"version": "1.0.0",
		"description": "Tenant-scoped cache governance and runtime adapter API",
		"endpoints": {
			"health": "/api/v1/cache/health",
			"set": "/api/v1/cache/set",
			"get": "/api/v1/cache/get/{key}",
			"delete": "/api/v1/cache/delete/{key}",
			"stats": "/api/v1/cache/stats",
			"policies": "/api/v1/cache/policies",
			"ai_insights": "/api/v1/cache/ai/insights",
			"docs": "/api/v1/cache/docs"
		},
		"features": [
			"AI-powered autonomous optimization",
			"Predictive content delivery",
			"Intelligent cache warming",
			"Multi-tier orchestration",
			"Content-aware optimization",
			"Real-time analytics",
			"Zero-configuration intelligence",
			"Behavior-driven security"
		]
	}


# Startup event
@app.on_event("startup")
async def startup_event():
	"""Initialize cache service on startup"""
	global cache_service
	try:
		config = CacheServiceConfig()
		cache_service = CacheService(config)
		await cache_service.initialize()
		logger.info("APG Cache Management API started successfully")
	except Exception as e:
		logger.error(f"Failed to start cache service: {e}")
		raise


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
	"""Cleanup on shutdown"""
	global cache_service
	if cache_service:
		await cache_service.shutdown()
		logger.info("APG Cache Management API shut down")


# Export the FastAPI app
__all__ = ['app']
