"""
APG Customer Relationship Management - Enterprise API Gateway

This module provides enterprise-grade API gateway capabilities including
rate limiting, request/response transformation, authentication, monitoring,
caching, and advanced routing for the CRM system.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from uuid import uuid4
import hashlib
import redis.asyncio as aioredis

from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from uuid_extensions import uuid7str

from .views import CRMResponse, CRMError


logger = logging.getLogger(__name__)


class RateLimitRule(BaseModel):
	"""Rate limiting rule configuration"""
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	rule_name: str
	description: Optional[str] = None
	resource_pattern: str  # URL pattern or resource identifier
	rate_limit_type: str = "requests_per_minute"  # requests_per_minute, requests_per_hour, bandwidth
	limit_value: int
	window_size_seconds: int = 60
	burst_limit: Optional[int] = None  # Allow short bursts above limit
	scope: str = "tenant"  # tenant, user, ip, api_key
	enforcement_action: str = "reject"  # reject, delay, throttle
	override_headers: Dict[str, str] = Field(default_factory=dict)
	exception_conditions: List[Dict[str, Any]] = Field(default_factory=list)
	is_active: bool = True
	priority: int = 100  # Higher priority rules are evaluated first
	created_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str


class APIEndpoint(BaseModel):
	"""API endpoint configuration"""
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	endpoint_path: str
	http_methods: List[str] = Field(default_factory=list)
	description: Optional[str] = None
	version: str = "v1"
	is_public: bool = False
	authentication_required: bool = True
	authorization_rules: List[Dict[str, Any]] = Field(default_factory=list)
	rate_limit_rules: List[str] = Field(default_factory=list)  # Rule IDs
	transformation_rules: Dict[str, Any] = Field(default_factory=dict)
	caching_config: Dict[str, Any] = Field(default_factory=dict)
	monitoring_config: Dict[str, Any] = Field(default_factory=dict)
	deprecated: bool = False
	deprecation_date: Optional[datetime] = None
	replacement_endpoint: Optional[str] = None
	tags: List[str] = Field(default_factory=list)
	is_active: bool = True
	created_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str


class APIRequest(BaseModel):
	"""API request tracking"""
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	endpoint_id: Optional[str] = None
	request_path: str
	http_method: str
	client_ip: str
	user_agent: Optional[str] = None
	user_id: Optional[str] = None
	api_key_id: Optional[str] = None
	request_headers: Dict[str, str] = Field(default_factory=dict)
	request_body_size: int = 0
	response_status: int = 200
	response_body_size: int = 0
	response_time_ms: float = 0.0
	rate_limit_applied: bool = False
	cache_hit: bool = False
	errors: List[str] = Field(default_factory=list)
	metadata: Dict[str, Any] = Field(default_factory=dict)
	timestamp: datetime = Field(default_factory=datetime.utcnow)


class APIGatewayMetrics(BaseModel):
	"""API gateway metrics aggregation"""
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	metric_date: datetime
	total_requests: int = 0
	successful_requests: int = 0
	failed_requests: int = 0
	rate_limited_requests: int = 0
	cached_requests: int = 0
	avg_response_time_ms: float = 0.0
	total_bandwidth_bytes: int = 0
	unique_clients: int = 0
	top_endpoints: List[Dict[str, Any]] = Field(default_factory=list)
	error_breakdown: Dict[str, int] = Field(default_factory=dict)
	performance_percentiles: Dict[str, float] = Field(default_factory=dict)
	created_at: datetime = Field(default_factory=datetime.utcnow)


class RateLimiter:
	"""Advanced rate limiting implementation"""
	
	def __init__(self, redis_client: aioredis.Redis):
		self.redis = redis_client
		self.sliding_window_script = """
		local key = KEYS[1]
		local window = tonumber(ARGV[1])
		local limit = tonumber(ARGV[2])
		local current_time = tonumber(ARGV[3])
		
		-- Remove expired entries
		redis.call('ZREMRANGEBYSCORE', key, 0, current_time - window)
		
		-- Count current requests
		local current_count = redis.call('ZCARD', key)
		
		if current_count < limit then
			-- Add current request
			redis.call('ZADD', key, current_time, current_time)
			redis.call('EXPIRE', key, window)
			return {1, limit - current_count - 1}
		else
			return {0, 0}
		end
		"""
	
	async def check_rate_limit(
		self, 
		key: str, 
		limit: int, 
		window_seconds: int,
		burst_limit: Optional[int] = None
	) -> Tuple[bool, int]:
		"""
		Check if request should be rate limited
		Returns (allowed, remaining_requests)
		"""
		try:
			current_time = time.time()
			
			# Use sliding window algorithm
			result = await self.redis.eval(
				self.sliding_window_script,
				1,  # Number of keys
				f"rate_limit:{key}",
				window_seconds,
				limit,
				current_time
			)
			
			allowed = bool(result[0])
			remaining = int(result[1])
			
			# Check burst limit if configured
			if burst_limit and allowed:
				burst_key = f"burst:{key}"
				burst_count = await self.redis.incr(burst_key)
				await self.redis.expire(burst_key, 1)  # 1 second burst window
				
				if burst_count > burst_limit:
					allowed = False
					remaining = 0
			
			return allowed, remaining
			
		except Exception as e:
			logger.error(f"Rate limit check failed: {str(e)}")
			# Fail open - allow request if rate limiter is down
			return True, limit


class RequestTransformer:
	"""Request/response transformation engine"""
	
	def __init__(self):
		self.transformations = {}
	
	async def transform_request(
		self, 
		request: Request, 
		transformation_rules: Dict[str, Any]
	) -> Request:
		"""Apply request transformations"""
		try:
			if not transformation_rules:
				return request
			
			# Header transformations
			if "headers" in transformation_rules:
				for rule in transformation_rules["headers"]:
					if rule["action"] == "add":
						request.headers[rule["name"]] = rule["value"]
					elif rule["action"] == "remove":
						request.headers.pop(rule["name"], None)
					elif rule["action"] == "rename":
						if rule["old_name"] in request.headers:
							request.headers[rule["new_name"]] = request.headers.pop(rule["old_name"])
			
			# Query parameter transformations
			if "query_params" in transformation_rules:
				# Similar transformations for query parameters
				pass
			
			# Body transformations
			if "body" in transformation_rules:
				# JSON body transformations
				pass
			
			return request
			
		except Exception as e:
			logger.error(f"Request transformation failed: {str(e)}")
			return request
	
	async def transform_response(
		self, 
		response: Response, 
		transformation_rules: Dict[str, Any]
	) -> Response:
		"""Apply response transformations"""
		try:
			if not transformation_rules:
				return response
			
			# Response transformations implementation
			return response
			
		except Exception as e:
			logger.error(f"Response transformation failed: {str(e)}")
			return response


class CacheManager:
	"""Advanced caching with intelligent invalidation"""
	
	def __init__(self, redis_client: aioredis.Redis):
		self.redis = redis_client
	
	async def get_cached_response(
		self, 
		cache_key: str
	) -> Optional[Dict[str, Any]]:
		"""Get cached response if available"""
		try:
			cached_data = await self.redis.get(f"cache:{cache_key}")
			if cached_data:
				return json.loads(cached_data)
			return None
			
		except Exception as e:
			logger.error(f"Cache retrieval failed: {str(e)}")
			return None
	
	async def cache_response(
		self, 
		cache_key: str, 
		response_data: Dict[str, Any], 
		ttl_seconds: int
	) -> None:
		"""Cache response with TTL"""
		try:
			await self.redis.setex(
				f"cache:{cache_key}",
				ttl_seconds,
				json.dumps(response_data, default=str)
			)
			
		except Exception as e:
			logger.error(f"Response caching failed: {str(e)}")
	
	async def invalidate_cache(
		self, 
		pattern: str
	) -> None:
		"""Invalidate cache entries matching pattern"""
		try:
			keys = await self.redis.keys(f"cache:{pattern}")
			if keys:
				await self.redis.delete(*keys)
				
		except Exception as e:
			logger.error(f"Cache invalidation failed: {str(e)}")


class APIGateway:
	"""Enterprise API Gateway with advanced features"""
	
	def __init__(self, db_pool, redis_url: str = "redis://localhost:6379"):
		self.db_pool = db_pool
		self.redis = None
		self.rate_limiter = None
		self.transformer = RequestTransformer()
		self.cache_manager = None
		self.rate_limit_rules = {}
		self.endpoint_configs = {}
		
	async def initialize(self) -> None:
		"""Initialize the API gateway"""
		try:
			# Initialize Redis connection
			self.redis = aioredis.from_url("redis://localhost:6379")
			self.rate_limiter = RateLimiter(self.redis)
			self.cache_manager = CacheManager(self.redis)
			
			# Load rate limiting rules
			await self._load_rate_limit_rules()
			
			# Load endpoint configurations
			await self._load_endpoint_configs()
			
			logger.info("✅ API Gateway initialized successfully")
			
		except Exception as e:
			logger.error(f"Failed to initialize API Gateway: {str(e)}")
			raise CRMError(f"API Gateway initialization failed: {str(e)}")
	
	async def create_rate_limit_rule(
		self,
		tenant_id: str,
		rule_name: str,
		resource_pattern: str,
		rate_limit_type: str,
		limit_value: int,
		created_by: str,
		description: Optional[str] = None,
		window_size_seconds: int = 60,
		burst_limit: Optional[int] = None,
		scope: str = "tenant",
		enforcement_action: str = "reject"
	) -> RateLimitRule:
		"""Create a new rate limiting rule"""
		try:
			rule = RateLimitRule(
				tenant_id=tenant_id,
				rule_name=rule_name,
				description=description,
				resource_pattern=resource_pattern,
				rate_limit_type=rate_limit_type,
				limit_value=limit_value,
				window_size_seconds=window_size_seconds,
				burst_limit=burst_limit,
				scope=scope,
				enforcement_action=enforcement_action,
				created_by=created_by
			)
			
			async with self.db_pool.acquire() as conn:
				await conn.execute("""
					INSERT INTO crm_rate_limit_rules (
						id, tenant_id, rule_name, description, resource_pattern,
						rate_limit_type, limit_value, window_size_seconds, burst_limit,
						scope, enforcement_action, override_headers, exception_conditions,
						is_active, priority, created_at, created_by
					) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
				""",
				rule.id, rule.tenant_id, rule.rule_name, rule.description,
				rule.resource_pattern, rule.rate_limit_type, rule.limit_value,
				rule.window_size_seconds, rule.burst_limit, rule.scope,
				rule.enforcement_action, json.dumps(rule.override_headers),
				json.dumps(rule.exception_conditions), rule.is_active,
				rule.priority, rule.created_at, rule.created_by)
			
			# Update in-memory cache
			self.rate_limit_rules[rule.id] = rule
			
			logger.info(f"Created rate limit rule: {rule.rule_name} for tenant {tenant_id}")
			return rule
			
		except Exception as e:
			logger.error(f"Failed to create rate limit rule: {str(e)}")
			raise CRMError(f"Failed to create rate limit rule: {str(e)}")
	
	async def register_endpoint(
		self,
		tenant_id: str,
		endpoint_path: str,
		http_methods: List[str],
		created_by: str,
		description: Optional[str] = None,
		version: str = "v1",
		is_public: bool = False,
		authentication_required: bool = True,
		rate_limit_rules: Optional[List[str]] = None,
		caching_config: Optional[Dict[str, Any]] = None
	) -> APIEndpoint:
		"""Register a new API endpoint"""
		try:
			endpoint = APIEndpoint(
				tenant_id=tenant_id,
				endpoint_path=endpoint_path,
				http_methods=http_methods,
				description=description,
				version=version,
				is_public=is_public,
				authentication_required=authentication_required,
				rate_limit_rules=rate_limit_rules or [],
				caching_config=caching_config or {},
				created_by=created_by
			)
			
			async with self.db_pool.acquire() as conn:
				await conn.execute("""
					INSERT INTO crm_api_endpoints (
						id, tenant_id, endpoint_path, http_methods, description, version,
						is_public, authentication_required, authorization_rules, rate_limit_rules,
						transformation_rules, caching_config, monitoring_config, deprecated,
						deprecation_date, replacement_endpoint, tags, is_active, created_at, created_by
					) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20)
				""",
				endpoint.id, endpoint.tenant_id, endpoint.endpoint_path,
				json.dumps(endpoint.http_methods), endpoint.description, endpoint.version,
				endpoint.is_public, endpoint.authentication_required,
				json.dumps(endpoint.authorization_rules), json.dumps(endpoint.rate_limit_rules),
				json.dumps(endpoint.transformation_rules), json.dumps(endpoint.caching_config),
				json.dumps(endpoint.monitoring_config), endpoint.deprecated,
				endpoint.deprecation_date, endpoint.replacement_endpoint,
				json.dumps(endpoint.tags), endpoint.is_active, endpoint.created_at, endpoint.created_by)
			
			# Update in-memory cache
			self.endpoint_configs[endpoint.endpoint_path] = endpoint
			
			logger.info(f"Registered API endpoint: {endpoint.endpoint_path} for tenant {tenant_id}")
			return endpoint
			
		except Exception as e:
			logger.error(f"Failed to register API endpoint: {str(e)}")
			raise CRMError(f"Failed to register API endpoint: {str(e)}")
	
	async def process_request(
		self,
		request: Request,
		tenant_id: str,
		user_id: Optional[str] = None
	) -> Tuple[bool, Optional[Response], Dict[str, Any]]:
		"""
		Process incoming request through gateway
		Returns (should_continue, early_response, metadata)
		"""
		request_metadata = {
			"gateway_processed": True,
			"rate_limited": False,
			"cached": False,
			"transformed": False,
			"processing_time_ms": 0.0
		}
		
		start_time = time.time()
		
		try:
			# Get endpoint configuration
			endpoint_config = await self._get_endpoint_config(request.url.path)
			
			# Check rate limiting
			rate_limit_result = await self._check_rate_limits(
				request, tenant_id, user_id, endpoint_config
			)
			
			if not rate_limit_result["allowed"]:
				request_metadata["rate_limited"] = True
				return False, JSONResponse(
					status_code=429,
					content={"error": "Rate limit exceeded", "retry_after": rate_limit_result["retry_after"]},
					headers={"X-RateLimit-Remaining": "0"}
				), request_metadata
			
			# Check cache
			if endpoint_config and endpoint_config.caching_config.get("enabled", False):
				cache_key = await self._generate_cache_key(request, tenant_id)
				cached_response = await self.cache_manager.get_cached_response(cache_key)
				
				if cached_response:
					request_metadata["cached"] = True
					return False, JSONResponse(
						content=cached_response["content"],
						status_code=cached_response["status_code"],
						headers=cached_response.get("headers", {})
					), request_metadata
			
			# Apply request transformations
			if endpoint_config and endpoint_config.transformation_rules:
				transformed_request = await self.transformer.transform_request(
					request, endpoint_config.transformation_rules
				)
				request_metadata["transformed"] = True
				request = transformed_request
			
			# Log request
			await self._log_api_request(request, tenant_id, user_id, endpoint_config)
			
			processing_time = (time.time() - start_time) * 1000
			request_metadata["processing_time_ms"] = processing_time
			
			return True, None, request_metadata
			
		except Exception as e:
			logger.error(f"Request processing failed: {str(e)}")
			return False, JSONResponse(
				status_code=500,
				content={"error": "Gateway processing failed"}
			), request_metadata
	
	async def process_response(
		self,
		request: Request,
		response: Response,
		tenant_id: str,
		request_metadata: Dict[str, Any]
	) -> Response:
		"""Process outgoing response through gateway"""
		try:
			endpoint_config = await self._get_endpoint_config(request.url.path)
			
			# Apply response transformations
			if endpoint_config and endpoint_config.transformation_rules:
				response = await self.transformer.transform_response(
					response, endpoint_config.transformation_rules
				)
			
			# Cache response if configured
			if (endpoint_config and 
				endpoint_config.caching_config.get("enabled", False) and
				response.status_code == 200):
				
				cache_key = await self._generate_cache_key(request, tenant_id)
				ttl = endpoint_config.caching_config.get("ttl_seconds", 300)
				
				# Extract response content for caching
				response_content = {
					"content": response.body.decode() if hasattr(response, 'body') else {},
					"status_code": response.status_code,
					"headers": dict(response.headers)
				}
				
				await self.cache_manager.cache_response(cache_key, response_content, ttl)
			
			# Add gateway headers
			response.headers["X-Gateway-Processed"] = "true"
			response.headers["X-Gateway-Version"] = "1.0"
			
			if request_metadata.get("rate_limited"):
				response.headers["X-Rate-Limited"] = "true"
			
			if request_metadata.get("cached"):
				response.headers["X-Cache-Hit"] = "true"
			
			return response
			
		except Exception as e:
			logger.error(f"Response processing failed: {str(e)}")
			return response
	
	async def get_metrics(
		self,
		tenant_id: str,
		start_date: datetime,
		end_date: datetime
	) -> APIGatewayMetrics:
		"""Get API gateway metrics for a time period"""
		try:
			async with self.db_pool.acquire() as conn:
				# Aggregate metrics from API requests
				metrics_row = await conn.fetchrow("""
					SELECT 
						COUNT(*) as total_requests,
						SUM(CASE WHEN response_status < 400 THEN 1 ELSE 0 END) as successful_requests,
						SUM(CASE WHEN response_status >= 400 THEN 1 ELSE 0 END) as failed_requests,
						SUM(CASE WHEN rate_limit_applied THEN 1 ELSE 0 END) as rate_limited_requests,
						SUM(CASE WHEN cache_hit THEN 1 ELSE 0 END) as cached_requests,
						AVG(response_time_ms) as avg_response_time_ms,
						SUM(request_body_size + response_body_size) as total_bandwidth_bytes,
						COUNT(DISTINCT client_ip) as unique_clients
					FROM crm_api_requests 
					WHERE tenant_id = $1 AND timestamp BETWEEN $2 AND $3
				""", tenant_id, start_date, end_date)
				
				# Get top endpoints
				top_endpoints_rows = await conn.fetch("""
					SELECT request_path, COUNT(*) as request_count,
						   AVG(response_time_ms) as avg_response_time
					FROM crm_api_requests 
					WHERE tenant_id = $1 AND timestamp BETWEEN $2 AND $3
					GROUP BY request_path
					ORDER BY request_count DESC
					LIMIT 10
				""", tenant_id, start_date, end_date)
				
				top_endpoints = [
					{
						"path": row["request_path"],
						"request_count": row["request_count"],
						"avg_response_time": float(row["avg_response_time"] or 0)
					}
					for row in top_endpoints_rows
				]
				
				metrics = APIGatewayMetrics(
					tenant_id=tenant_id,
					metric_date=datetime.utcnow(),
					total_requests=metrics_row["total_requests"] or 0,
					successful_requests=metrics_row["successful_requests"] or 0,
					failed_requests=metrics_row["failed_requests"] or 0,
					rate_limited_requests=metrics_row["rate_limited_requests"] or 0,
					cached_requests=metrics_row["cached_requests"] or 0,
					avg_response_time_ms=float(metrics_row["avg_response_time_ms"] or 0),
					total_bandwidth_bytes=metrics_row["total_bandwidth_bytes"] or 0,
					unique_clients=metrics_row["unique_clients"] or 0,
					top_endpoints=top_endpoints
				)
				
				return metrics
				
		except Exception as e:
			logger.error(f"Failed to get gateway metrics: {str(e)}")
			raise CRMError(f"Failed to get gateway metrics: {str(e)}")
	
	# Helper methods
	
	async def _load_rate_limit_rules(self) -> None:
		"""Load rate limiting rules from database"""
		try:
			async with self.db_pool.acquire() as conn:
				rows = await conn.fetch("""
					SELECT * FROM crm_rate_limit_rules WHERE is_active = true
					ORDER BY priority DESC
				""")
				
				for row in rows:
					rule = RateLimitRule(**dict(row))
					self.rate_limit_rules[rule.id] = rule
					
		except Exception as e:
			logger.error(f"Failed to load rate limit rules: {str(e)}")
	
	async def _load_endpoint_configs(self) -> None:
		"""Load endpoint configurations from database"""
		try:
			async with self.db_pool.acquire() as conn:
				rows = await conn.fetch("""
					SELECT * FROM crm_api_endpoints WHERE is_active = true
				""")
				
				for row in rows:
					endpoint = APIEndpoint(**dict(row))
					self.endpoint_configs[endpoint.endpoint_path] = endpoint
					
		except Exception as e:
			logger.error(f"Failed to load endpoint configs: {str(e)}")
	
	async def _get_endpoint_config(self, path: str) -> Optional[APIEndpoint]:
		"""Get endpoint configuration for a path"""
		# Simple path matching - could be enhanced with pattern matching
		return self.endpoint_configs.get(path)
	
	async def _check_rate_limits(
		self,
		request: Request,
		tenant_id: str,
		user_id: Optional[str],
		endpoint_config: Optional[APIEndpoint]
	) -> Dict[str, Any]:
		"""Check all applicable rate limits"""
		try:
			# Find applicable rules
			applicable_rules = []
			
			for rule in self.rate_limit_rules.values():
				if rule.tenant_id == tenant_id and self._rule_matches_request(rule, request):
					applicable_rules.append(rule)
			
			# Check each rule
			for rule in sorted(applicable_rules, key=lambda r: r.priority, reverse=True):
				rate_limit_key = await self._generate_rate_limit_key(rule, request, tenant_id, user_id)
				
				allowed, remaining = await self.rate_limiter.check_rate_limit(
					rate_limit_key,
					rule.limit_value,
					rule.window_size_seconds,
					rule.burst_limit
				)
				
				if not allowed:
					return {
						"allowed": False,
						"rule_id": rule.id,
						"retry_after": rule.window_size_seconds,
						"remaining": remaining
					}
			
			return {"allowed": True, "remaining": 1000}  # Default high limit
			
		except Exception as e:
			logger.error(f"Rate limit check failed: {str(e)}")
			return {"allowed": True, "remaining": 1000}  # Fail open
	
	def _rule_matches_request(self, rule: RateLimitRule, request: Request) -> bool:
		"""Check if a rate limit rule applies to the request"""
		# Simple pattern matching - could be enhanced with regex
		return rule.resource_pattern in str(request.url.path)
	
	async def _generate_rate_limit_key(
		self,
		rule: RateLimitRule,
		request: Request,
		tenant_id: str,
		user_id: Optional[str]
	) -> str:
		"""Generate rate limiting key based on scope"""
		if rule.scope == "tenant":
			return f"tenant:{tenant_id}:{rule.id}"
		elif rule.scope == "user" and user_id:
			return f"user:{user_id}:{rule.id}"
		elif rule.scope == "ip":
			client_ip = request.client.host if request.client else "unknown"
			return f"ip:{client_ip}:{rule.id}"
		else:
			return f"global:{rule.id}"
	
	async def _generate_cache_key(self, request: Request, tenant_id: str) -> str:
		"""Generate cache key for request"""
		key_data = f"{tenant_id}:{request.method}:{request.url.path}:{str(request.query_params)}"
		return hashlib.md5(key_data.encode()).hexdigest()
	
	async def _log_api_request(
		self,
		request: Request,
		tenant_id: str,
		user_id: Optional[str],
		endpoint_config: Optional[APIEndpoint]
	) -> None:
		"""Log API request for monitoring and analytics"""
		try:
			api_request = APIRequest(
				tenant_id=tenant_id,
				endpoint_id=endpoint_config.id if endpoint_config else None,
				request_path=str(request.url.path),
				http_method=request.method,
				client_ip=request.client.host if request.client else "unknown",
				user_agent=request.headers.get("user-agent"),
				user_id=user_id,
				request_headers=dict(request.headers),
				request_body_size=int(request.headers.get("content-length", 0))
			)
			
			async with self.db_pool.acquire() as conn:
				await conn.execute("""
					INSERT INTO crm_api_requests (
						id, tenant_id, endpoint_id, request_path, http_method,
						client_ip, user_agent, user_id, api_key_id, request_headers,
						request_body_size, timestamp
					) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
				""",
				api_request.id, api_request.tenant_id, api_request.endpoint_id,
				api_request.request_path, api_request.http_method, api_request.client_ip,
				api_request.user_agent, api_request.user_id, api_request.api_key_id,
				json.dumps(api_request.request_headers), api_request.request_body_size,
				api_request.timestamp)
				
		except Exception as e:
			logger.error(f"Failed to log API request: {str(e)}")


class APIGatewayMiddleware(BaseHTTPMiddleware):
	"""FastAPI middleware for API Gateway integration"""
	
	def __init__(self, app: FastAPI, gateway: APIGateway):
		super().__init__(app)
		self.gateway = gateway
	
	async def dispatch(self, request: Request, call_next: Callable) -> Response:
		"""Process request through API gateway"""
		try:
			# Extract tenant and user from headers or auth
			tenant_id = request.headers.get("X-Tenant-ID", "default")
			user_id = request.headers.get("X-User-ID")
			
			# Process request through gateway
			should_continue, early_response, metadata = await self.gateway.process_request(
				request, tenant_id, user_id
			)
			
			if not should_continue and early_response:
				return early_response
			
			# Continue with normal processing
			response = await call_next(request)
			
			# Process response through gateway
			processed_response = await self.gateway.process_response(
				request, response, tenant_id, metadata
			)
			
			return processed_response
			
		except Exception as e:
			logger.error(f"Gateway middleware error: {str(e)}")
			# Continue without gateway processing if there's an error
			return await call_next(request)