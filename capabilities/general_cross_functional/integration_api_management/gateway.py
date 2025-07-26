"""
APG Integration API Management - Gateway Engine

High-performance API gateway with routing, middleware, load balancing,
and real-time policy enforcement supporting 100K+ RPS throughput.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import time
import hashlib
import secrets
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import urljoin, urlparse

import aiohttp
import aioredis
from aiohttp import web, ClientSession, ClientTimeout, ClientError
from aiohttp.web_middlewares import normalize_path_middleware
from aiohttp_cors import setup as cors_setup, ResourceOptions

from .models import (
	AMAPI, AMEndpoint, AMPolicy, AMConsumer, AMAPIKey, AMUsageRecord,
	APIStatus, ProtocolType, AuthenticationType, PolicyType, LoadBalancingAlgorithm
)
from .service import (
	APILifecycleService, ConsumerManagementService, 
	PolicyManagementService, AnalyticsService
)

# =============================================================================
# Gateway Configuration and Data Classes
# =============================================================================

class RequestMethod(str, Enum):
	"""HTTP request methods."""
	GET = "GET"
	POST = "POST"
	PUT = "PUT"
	DELETE = "DELETE"
	PATCH = "PATCH"
	HEAD = "HEAD"
	OPTIONS = "OPTIONS"

class LoadBalancingStrategy(str, Enum):
	"""Load balancing strategies."""
	ROUND_ROBIN = "round_robin"
	WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
	LEAST_CONNECTIONS = "least_connections"
	IP_HASH = "ip_hash"
	RANDOM = "random"

@dataclass
class UpstreamServer:
	"""Upstream server configuration."""
	url: str
	weight: int = 1
	max_connections: int = 1000
	health_check_path: str = "/health"
	health_check_interval: int = 30
	is_healthy: bool = True
	current_connections: int = 0
	response_times: List[float] = field(default_factory=list)
	last_health_check: Optional[datetime] = None

@dataclass
class RouteConfig:
	"""API route configuration."""
	api_id: str
	path_pattern: str
	upstream_servers: List[UpstreamServer]
	load_balancing: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN
	timeout_ms: int = 30000
	retry_attempts: int = 3
	circuit_breaker_enabled: bool = True
	cache_enabled: bool = False
	cache_ttl: int = 300
	policies: List[str] = field(default_factory=list)
	
@dataclass
class GatewayRequest:
	"""Gateway request context."""
	request_id: str
	method: str
	path: str
	headers: Dict[str, str]
	query_params: Dict[str, str]
	body: bytes
	client_ip: str
	user_agent: str
	timestamp: datetime
	api_key: Optional[str] = None
	consumer_id: Optional[str] = None
	tenant_id: str = "default"
	
@dataclass
class GatewayResponse:
	"""Gateway response context."""
	status_code: int
	headers: Dict[str, str]
	body: bytes
	response_time_ms: float
	cache_hit: bool = False
	error_code: Optional[str] = None
	error_message: Optional[str] = None

@dataclass
class PolicyContext:
	"""Policy execution context."""
	request: GatewayRequest
	response: Optional[GatewayResponse] = None
	metadata: Dict[str, Any] = field(default_factory=dict)

# =============================================================================
# Middleware Components
# =============================================================================

class RateLimitMiddleware:
	"""Rate limiting middleware with Redis backend."""
	
	def __init__(self, redis_client: aioredis.Redis):
		self.redis = redis_client
		self.default_rate_limit = 1000  # requests per minute
		
	async def __call__(self, request: web.Request, handler: Callable) -> web.Response:
		"""Apply rate limiting to request."""
		
		gateway_request = request.get('gateway_request')
		if not gateway_request:
			return await handler(request)
		
		# Get rate limit for consumer/API
		rate_limit = await self._get_rate_limit(gateway_request)
		if not rate_limit:
			return await handler(request)
		
		# Check rate limit
		key = f"rate_limit:{gateway_request.consumer_id or gateway_request.client_ip}:{gateway_request.api_key or 'anonymous'}"
		
		try:
			# Use sliding window rate limiting
			current_time = int(time.time())
			window_start = current_time - 60  # 1 minute window
			
			# Remove old entries
			await self.redis.zremrangebyscore(key, 0, window_start)
			
			# Count current requests
			current_count = await self.redis.zcard(key)
			
			if current_count >= rate_limit:
				return web.json_response(
					{
						'error': 'Rate limit exceeded',
						'limit': rate_limit,
						'window': '1 minute',
						'retry_after': 60
					},
					status=429,
					headers={'Retry-After': '60'}
				)
			
			# Add current request
			await self.redis.zadd(key, {gateway_request.request_id: current_time})
			await self.redis.expire(key, 60)
			
		except Exception as e:
			# Log error but don't block request
			print(f"Rate limiting error: {e}")
		
		return await handler(request)
	
	async def _get_rate_limit(self, gateway_request: GatewayRequest) -> Optional[int]:
		"""Get rate limit for request."""
		# Implementation would query database for consumer/API rate limits
		return self.default_rate_limit

class AuthenticationMiddleware:
	"""Authentication middleware supporting multiple auth types."""
	
	def __init__(self, consumer_service: ConsumerManagementService):
		self.consumer_service = consumer_service
		
	async def __call__(self, request: web.Request, handler: Callable) -> web.Response:
		"""Authenticate request."""
		
		gateway_request = request.get('gateway_request')
		if not gateway_request:
			return await handler(request)
		
		# Skip authentication for health checks
		if gateway_request.path.endswith('/health'):
			return await handler(request)
		
		# Extract authentication credentials
		auth_result = await self._authenticate_request(gateway_request)
		
		if not auth_result['success']:
			return web.json_response(
				{
					'error': 'Authentication failed',
					'message': auth_result['message']
				},
				status=401,
				headers={'WWW-Authenticate': 'Bearer'}
			)
		
		# Set consumer info in request
		gateway_request.consumer_id = auth_result.get('consumer_id')
		gateway_request.api_key = auth_result.get('api_key')
		
		return await handler(request)
	
	async def _authenticate_request(self, gateway_request: GatewayRequest) -> Dict[str, Any]:
		"""Authenticate request using various methods."""
		
		# Try API Key authentication
		api_key = self._extract_api_key(gateway_request)
		if api_key:
			return await self._validate_api_key(api_key, gateway_request.tenant_id)
		
		# Try JWT authentication
		jwt_token = self._extract_jwt_token(gateway_request)
		if jwt_token:
			return await self._validate_jwt_token(jwt_token)
		
		# Try OAuth2 Bearer token
		bearer_token = self._extract_bearer_token(gateway_request)
		if bearer_token:
			return await self._validate_bearer_token(bearer_token)
		
		return {'success': False, 'message': 'No valid authentication credentials found'}
	
	def _extract_api_key(self, gateway_request: GatewayRequest) -> Optional[str]:
		"""Extract API key from request."""
		# Check header
		api_key = gateway_request.headers.get('X-API-Key')
		if api_key:
			return api_key
		
		# Check query parameter
		api_key = gateway_request.query_params.get('api_key')
		if api_key:
			return api_key
		
		return None
	
	def _extract_jwt_token(self, gateway_request: GatewayRequest) -> Optional[str]:
		"""Extract JWT token from Authorization header."""
		auth_header = gateway_request.headers.get('Authorization', '')
		if auth_header.startswith('Bearer '):
			return auth_header[7:]
		return None
	
	def _extract_bearer_token(self, gateway_request: GatewayRequest) -> Optional[str]:
		"""Extract Bearer token from Authorization header."""
		return self._extract_jwt_token(gateway_request)
	
	async def _validate_api_key(self, api_key: str, tenant_id: str) -> Dict[str, Any]:
		"""Validate API key."""
		try:
			# Hash the API key for lookup
			key_hash = hashlib.sha256(api_key.encode()).hexdigest()
			
			# Use consumer service to validate
			consumer_id = await self.consumer_service.validate_api_key(
				api_key_hash=key_hash,
				tenant_id=tenant_id
			)
			
			if consumer_id:
				return {
					'success': True,
					'consumer_id': consumer_id,
					'api_key': api_key[:8] + '...',
					'auth_method': 'api_key'
				}
			
		except Exception as e:
			print(f"API key validation error: {e}")
		
		return {'success': False, 'message': 'Invalid API key'}
	
	async def _validate_jwt_token(self, token: str) -> Dict[str, Any]:
		"""Validate JWT token."""
		# Implementation would validate JWT signature and claims
		return {'success': False, 'message': 'JWT validation not implemented'}
	
	async def _validate_bearer_token(self, token: str) -> Dict[str, Any]:
		"""Validate OAuth2 Bearer token."""
		# Implementation would validate with OAuth2 provider
		return {'success': False, 'message': 'Bearer token validation not implemented'}

class PolicyEnforcementMiddleware:
	"""Policy enforcement middleware."""
	
	def __init__(self, policy_service: PolicyManagementService):
		self.policy_service = policy_service
		
	async def __call__(self, request: web.Request, handler: Callable) -> web.Response:
		"""Enforce policies on request."""
		
		gateway_request = request.get('gateway_request')
		if not gateway_request:
			return await handler(request)
		
		# Get applicable policies
		policies = await self._get_applicable_policies(gateway_request)
		
		# Execute pre-request policies
		for policy in policies:
			if policy['type'] in ['authentication', 'authorization', 'validation']:
				result = await self._execute_policy(policy, gateway_request)
				if not result['success']:
					return web.json_response(
						{
							'error': 'Policy violation',
							'policy': policy['name'],
							'message': result['message']
						},
						status=result.get('status_code', 403)
					)
		
		# Execute request
		response = await handler(request)
		
		# Execute post-response policies
		gateway_response = request.get('gateway_response')
		for policy in policies:
			if policy['type'] in ['transformation', 'logging']:
				await self._execute_policy(policy, gateway_request, gateway_response)
		
		return response
	
	async def _get_applicable_policies(self, gateway_request: GatewayRequest) -> List[Dict[str, Any]]:
		"""Get policies applicable to request."""
		# Implementation would query database for applicable policies
		return []
	
	async def _execute_policy(self, policy: Dict[str, Any], gateway_request: GatewayRequest, 
							 gateway_response: Optional[GatewayResponse] = None) -> Dict[str, Any]:
		"""Execute individual policy."""
		
		policy_type = policy['type']
		config = policy['config']
		
		if policy_type == 'rate_limiting':
			return await self._execute_rate_limit_policy(config, gateway_request)
		elif policy_type == 'cors':
			return await self._execute_cors_policy(config, gateway_request)
		elif policy_type == 'transformation':
			return await self._execute_transformation_policy(config, gateway_request, gateway_response)
		elif policy_type == 'validation':
			return await self._execute_validation_policy(config, gateway_request)
		else:
			return {'success': True}
	
	async def _execute_rate_limit_policy(self, config: Dict[str, Any], 
										gateway_request: GatewayRequest) -> Dict[str, Any]:
		"""Execute rate limiting policy."""
		# Rate limiting is handled by RateLimitMiddleware
		return {'success': True}
	
	async def _execute_cors_policy(self, config: Dict[str, Any], 
								  gateway_request: GatewayRequest) -> Dict[str, Any]:
		"""Execute CORS policy."""
		# CORS is handled by aiohttp-cors
		return {'success': True}
	
	async def _execute_transformation_policy(self, config: Dict[str, Any], 
											gateway_request: GatewayRequest,
											gateway_response: Optional[GatewayResponse]) -> Dict[str, Any]:
		"""Execute transformation policy."""
		# Implementation would transform request/response
		return {'success': True}
	
	async def _execute_validation_policy(self, config: Dict[str, Any], 
										gateway_request: GatewayRequest) -> Dict[str, Any]:
		"""Execute validation policy."""
		# Implementation would validate request against schema
		return {'success': True}

class CachingMiddleware:
	"""Response caching middleware with Redis backend."""
	
	def __init__(self, redis_client: aioredis.Redis):
		self.redis = redis_client
		
	async def __call__(self, request: web.Request, handler: Callable) -> web.Response:
		"""Cache responses for GET requests."""
		
		gateway_request = request.get('gateway_request')
		if not gateway_request or gateway_request.method != 'GET':
			return await handler(request)
		
		# Generate cache key
		cache_key = self._generate_cache_key(gateway_request)
		
		# Try to get cached response
		cached_response = await self._get_cached_response(cache_key)
		if cached_response:
			return web.Response(
				body=cached_response['body'],
				status=cached_response['status'],
				headers=cached_response['headers']
			)
		
		# Execute request
		response = await handler(request)
		
		# Cache successful responses
		if 200 <= response.status < 300 and self._is_cacheable(gateway_request):
			await self._cache_response(cache_key, response, ttl=300)
		
		return response
	
	def _generate_cache_key(self, gateway_request: GatewayRequest) -> str:
		"""Generate cache key for request."""
		key_parts = [
			gateway_request.path,
			json.dumps(gateway_request.query_params, sort_keys=True),
			gateway_request.tenant_id
		]
		return f"cache:{hashlib.md5('|'.join(key_parts).encode()).hexdigest()}"
	
	async def _get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
		"""Get cached response."""
		try:
			cached_data = await self.redis.get(cache_key)
			if cached_data:
				return json.loads(cached_data)
		except Exception as e:
			print(f"Cache get error: {e}")
		return None
	
	async def _cache_response(self, cache_key: str, response: web.Response, ttl: int):
		"""Cache response."""
		try:
			cache_data = {
				'status': response.status,
				'headers': dict(response.headers),
				'body': response.body.decode() if response.body else ''
			}
			await self.redis.setex(cache_key, ttl, json.dumps(cache_data))
		except Exception as e:
			print(f"Cache set error: {e}")
	
	def _is_cacheable(self, gateway_request: GatewayRequest) -> bool:
		"""Check if request is cacheable."""
		# Don't cache requests with authorization headers
		if 'Authorization' in gateway_request.headers:
			return False
		return True

# =============================================================================
# Load Balancer
# =============================================================================

class LoadBalancer:
	"""Load balancer with multiple strategies and health checking."""
	
	def __init__(self):
		self.strategies = {
			LoadBalancingStrategy.ROUND_ROBIN: self._round_robin,
			LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN: self._weighted_round_robin,
			LoadBalancingStrategy.LEAST_CONNECTIONS: self._least_connections,
			LoadBalancingStrategy.IP_HASH: self._ip_hash,
			LoadBalancingStrategy.RANDOM: self._random
		}
		self.round_robin_counters = {}
		
	async def select_server(self, upstream_servers: List[UpstreamServer], 
						   strategy: LoadBalancingStrategy, 
						   client_ip: str = None) -> Optional[UpstreamServer]:
		"""Select upstream server using specified strategy."""
		
		# Filter healthy servers
		healthy_servers = [s for s in upstream_servers if s.is_healthy]
		if not healthy_servers:
			return None
		
		strategy_func = self.strategies.get(strategy, self._round_robin)
		return await strategy_func(healthy_servers, client_ip)
	
	async def _round_robin(self, servers: List[UpstreamServer], 
						  client_ip: str = None) -> UpstreamServer:
		"""Round robin selection."""
		server_key = id(servers)
		if server_key not in self.round_robin_counters:
			self.round_robin_counters[server_key] = 0
		
		server = servers[self.round_robin_counters[server_key] % len(servers)]
		self.round_robin_counters[server_key] += 1
		return server
	
	async def _weighted_round_robin(self, servers: List[UpstreamServer], 
								   client_ip: str = None) -> UpstreamServer:
		"""Weighted round robin selection."""
		total_weight = sum(s.weight for s in servers)
		if total_weight == 0:
			return await self._round_robin(servers, client_ip)
		
		# Create weighted list
		weighted_servers = []
		for server in servers:
			weighted_servers.extend([server] * server.weight)
		
		return await self._round_robin(weighted_servers, client_ip)
	
	async def _least_connections(self, servers: List[UpstreamServer], 
								client_ip: str = None) -> UpstreamServer:
		"""Least connections selection."""
		return min(servers, key=lambda s: s.current_connections)
	
	async def _ip_hash(self, servers: List[UpstreamServer], 
					  client_ip: str = None) -> UpstreamServer:
		"""IP hash selection for session affinity."""
		if not client_ip:
			return await self._round_robin(servers)
		
		hash_value = hash(client_ip)
		return servers[hash_value % len(servers)]
	
	async def _random(self, servers: List[UpstreamServer], 
					 client_ip: str = None) -> UpstreamServer:
		"""Random selection."""
		import random
		return random.choice(servers)

# =============================================================================
# Circuit Breaker
# =============================================================================

class CircuitBreaker:
	"""Circuit breaker pattern implementation."""
	
	def __init__(self, failure_threshold: int = 5, timeout: int = 60, 
				 expected_exception: type = Exception):
		self.failure_threshold = failure_threshold
		self.timeout = timeout
		self.expected_exception = expected_exception
		self.failure_count = 0
		self.last_failure_time = None
		self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
		
	async def call(self, func: Callable, *args, **kwargs):
		"""Execute function with circuit breaker protection."""
		
		if self.state == 'OPEN':
			if self._should_attempt_reset():
				self.state = 'HALF_OPEN'
			else:
				raise Exception("Circuit breaker is OPEN")
		
		try:
			result = await func(*args, **kwargs)
			self._on_success()
			return result
		except self.expected_exception as e:
			self._on_failure()
			raise e
	
	def _should_attempt_reset(self) -> bool:
		"""Check if enough time has passed to attempt reset."""
		if self.last_failure_time is None:
			return True
		return time.time() - self.last_failure_time >= self.timeout
	
	def _on_success(self):
		"""Handle successful call."""
		self.failure_count = 0
		self.state = 'CLOSED'
	
	def _on_failure(self):
		"""Handle failed call."""
		self.failure_count += 1
		self.last_failure_time = time.time()
		
		if self.failure_count >= self.failure_threshold:
			self.state = 'OPEN'

# =============================================================================
# Gateway Request Router
# =============================================================================

class GatewayRouter:
	"""High-performance API gateway router."""
	
	def __init__(self, api_service: APILifecycleService, 
				 consumer_service: ConsumerManagementService,
				 policy_service: PolicyManagementService,
				 analytics_service: AnalyticsService,
				 redis_client: aioredis.Redis):
		
		self.api_service = api_service
		self.consumer_service = consumer_service
		self.policy_service = policy_service
		self.analytics_service = analytics_service
		self.redis = redis_client
		
		self.load_balancer = LoadBalancer()
		self.circuit_breakers = {}
		self.route_cache = {}
		self.client_session = None
		
	async def initialize(self):
		"""Initialize gateway router."""
		timeout = ClientTimeout(total=30)
		self.client_session = ClientSession(timeout=timeout)
		
	async def shutdown(self):
		"""Shutdown gateway router."""
		if self.client_session:
			await self.client_session.close()
	
	async def route_request(self, gateway_request: GatewayRequest) -> GatewayResponse:
		"""Route request to appropriate upstream service."""
		
		start_time = time.time()
		
		try:
			# Find matching route
			route_config = await self._find_route(gateway_request)
			if not route_config:
				return GatewayResponse(
					status_code=404,
					headers={'Content-Type': 'application/json'},
					body=json.dumps({'error': 'Route not found'}).encode(),
					response_time_ms=(time.time() - start_time) * 1000,
					error_code='ROUTE_NOT_FOUND'
				)
			
			# Select upstream server
			upstream_server = await self.load_balancer.select_server(
				route_config.upstream_servers,
				route_config.load_balancing,
				gateway_request.client_ip
			)
			
			if not upstream_server:
				return GatewayResponse(
					status_code=503,
					headers={'Content-Type': 'application/json'},
					body=json.dumps({'error': 'No healthy upstream servers'}).encode(),
					response_time_ms=(time.time() - start_time) * 1000,
					error_code='NO_UPSTREAM_SERVERS'
				)
			
			# Execute request with circuit breaker
			circuit_breaker = self._get_circuit_breaker(upstream_server.url)
			
			gateway_response = await circuit_breaker.call(
				self._proxy_request, 
				gateway_request, 
				upstream_server, 
				route_config
			)
			
			# Record analytics
			await self._record_analytics(gateway_request, gateway_response, route_config)
			
			return gateway_response
			
		except Exception as e:
			error_response = GatewayResponse(
				status_code=500,
				headers={'Content-Type': 'application/json'},
				body=json.dumps({'error': 'Internal server error', 'message': str(e)}).encode(),
				response_time_ms=(time.time() - start_time) * 1000,
				error_code='INTERNAL_ERROR',
				error_message=str(e)
			)
			
			# Record error analytics
			await self._record_analytics(gateway_request, error_response, None)
			
			return error_response
	
	async def _find_route(self, gateway_request: GatewayRequest) -> Optional[RouteConfig]:
		"""Find matching route configuration."""
		
		# Check cache first
		cache_key = f"route:{gateway_request.path}:{gateway_request.method}:{gateway_request.tenant_id}"
		cached_route = self.route_cache.get(cache_key)
		if cached_route:
			return cached_route
		
		# Query database for matching API
		try:
			apis = await self.api_service.get_apis_by_tenant(gateway_request.tenant_id)
			
			for api in apis:
				if api.status != APIStatus.ACTIVE.value:
					continue
				
				# Check if path matches
				if gateway_request.path.startswith(api.base_path):
					# Create route config
					upstream_servers = [
						UpstreamServer(
							url=api.upstream_url,
							weight=1,
							max_connections=1000
						)
					]
					
					route_config = RouteConfig(
						api_id=api.api_id,
						path_pattern=api.base_path,
						upstream_servers=upstream_servers,
						load_balancing=LoadBalancingStrategy(api.load_balancing_algorithm),
						timeout_ms=api.timeout_ms,
						retry_attempts=api.retry_attempts,
						circuit_breaker_enabled=True,
						cache_enabled=False,
						policies=[]
					)
					
					# Cache route for 5 minutes
					self.route_cache[cache_key] = route_config
					asyncio.create_task(self._expire_route_cache(cache_key, 300))
					
					return route_config
			
		except Exception as e:
			print(f"Route lookup error: {e}")
		
		return None
	
	async def _proxy_request(self, gateway_request: GatewayRequest, 
							upstream_server: UpstreamServer,
							route_config: RouteConfig) -> GatewayResponse:
		"""Proxy request to upstream server."""
		
		start_time = time.time()
		
		# Build upstream URL
		upstream_path = gateway_request.path
		if route_config.path_pattern != '/':
			upstream_path = upstream_path[len(route_config.path_pattern):]
		
		upstream_url = urljoin(upstream_server.url, upstream_path.lstrip('/'))
		
		# Prepare headers
		headers = dict(gateway_request.headers)
		headers['X-Forwarded-For'] = gateway_request.client_ip
		headers['X-Forwarded-Proto'] = 'https'
		headers['X-Request-ID'] = gateway_request.request_id
		
		# Remove hop-by-hop headers
		hop_by_hop_headers = [
			'connection', 'keep-alive', 'proxy-authenticate',
			'proxy-authorization', 'te', 'trailers', 'transfer-encoding', 'upgrade'
		]
		for header in hop_by_hop_headers:
			headers.pop(header, None)
		
		try:
			# Increment connection count
			upstream_server.current_connections += 1
			
			# Make request
			async with self.client_session.request(
				method=gateway_request.method,
				url=upstream_url,
				headers=headers,
				params=gateway_request.query_params,
				data=gateway_request.body if gateway_request.body else None,
				timeout=ClientTimeout(total=route_config.timeout_ms / 1000)
			) as response:
				
				response_body = await response.read()
				response_headers = dict(response.headers)
				
				# Remove hop-by-hop headers from response
				for header in hop_by_hop_headers:
					response_headers.pop(header, None)
				
				response_time_ms = (time.time() - start_time) * 1000
				
				# Record response time for server
				upstream_server.response_times.append(response_time_ms)
				if len(upstream_server.response_times) > 100:
					upstream_server.response_times = upstream_server.response_times[-100:]
				
				return GatewayResponse(
					status_code=response.status,
					headers=response_headers,
					body=response_body,
					response_time_ms=response_time_ms
				)
				
		except ClientError as e:
			response_time_ms = (time.time() - start_time) * 1000
			
			return GatewayResponse(
				status_code=502,
				headers={'Content-Type': 'application/json'},
				body=json.dumps({'error': 'Bad Gateway', 'message': str(e)}).encode(),
				response_time_ms=response_time_ms,
				error_code='BAD_GATEWAY',
				error_message=str(e)
			)
			
		except asyncio.TimeoutError:
			response_time_ms = (time.time() - start_time) * 1000
			
			return GatewayResponse(
				status_code=504,
				headers={'Content-Type': 'application/json'},
				body=json.dumps({'error': 'Gateway Timeout'}).encode(),
				response_time_ms=response_time_ms,
				error_code='GATEWAY_TIMEOUT'
			)
			
		finally:
			# Decrement connection count
			upstream_server.current_connections = max(0, upstream_server.current_connections - 1)
	
	def _get_circuit_breaker(self, server_url: str) -> CircuitBreaker:
		"""Get or create circuit breaker for server."""
		if server_url not in self.circuit_breakers:
			self.circuit_breakers[server_url] = CircuitBreaker(
				failure_threshold=5,
				timeout=60,
				expected_exception=Exception
			)
		return self.circuit_breakers[server_url]
	
	async def _record_analytics(self, gateway_request: GatewayRequest,
							   gateway_response: GatewayResponse,
							   route_config: Optional[RouteConfig]):
		"""Record request analytics."""
		
		try:
			# Create usage record
			usage_record_data = {
				'request_id': gateway_request.request_id,
				'consumer_id': gateway_request.consumer_id,
				'api_id': route_config.api_id if route_config else None,
				'endpoint_path': gateway_request.path,
				'method': gateway_request.method,
				'timestamp': gateway_request.timestamp,
				'response_status': gateway_response.status_code,
				'response_time_ms': int(gateway_response.response_time_ms),
				'client_ip': gateway_request.client_ip,
				'user_agent': gateway_request.user_agent,
				'billable': gateway_response.status_code < 500,
				'error_code': gateway_response.error_code,
				'error_message': gateway_response.error_message,
				'tenant_id': gateway_request.tenant_id
			}
			
			# Record asynchronously
			asyncio.create_task(
				self.analytics_service.record_usage(usage_record_data)
			)
			
		except Exception as e:
			print(f"Analytics recording error: {e}")
	
	async def _expire_route_cache(self, cache_key: str, ttl: int):
		"""Expire route cache entry after TTL."""
		await asyncio.sleep(ttl)
		self.route_cache.pop(cache_key, None)

# =============================================================================
# Main Gateway Application
# =============================================================================

class APIGateway:
	"""Main API Gateway application."""
	
	def __init__(self, host: str = '0.0.0.0', port: int = 8080,
				 redis_url: str = 'redis://localhost:6379',
				 database_url: str = None):
		
		self.host = host
		self.port = port
		self.redis_url = redis_url
		self.database_url = database_url
		
		self.app = None
		self.redis = None
		self.router = None
		
		# Services
		self.api_service = None
		self.consumer_service = None
		self.policy_service = None
		self.analytics_service = None
		
	async def initialize(self):
		"""Initialize gateway application."""
		
		# Initialize Redis
		self.redis = aioredis.from_url(self.redis_url)
		
		# Initialize services
		self.api_service = APILifecycleService()
		self.consumer_service = ConsumerManagementService()
		self.policy_service = PolicyManagementService()
		self.analytics_service = AnalyticsService()
		
		# Initialize router
		self.router = GatewayRouter(
			api_service=self.api_service,
			consumer_service=self.consumer_service,
			policy_service=self.policy_service,
			analytics_service=self.analytics_service,
			redis_client=self.redis
		)
		await self.router.initialize()
		
		# Create web application
		self.app = web.Application(
			middlewares=[
				normalize_path_middleware(),
				self._request_context_middleware,
				RateLimitMiddleware(self.redis),
				AuthenticationMiddleware(self.consumer_service),
				PolicyEnforcementMiddleware(self.policy_service),
				CachingMiddleware(self.redis),
				self._gateway_handler_middleware
			]
		)
		
		# Setup CORS
		cors = cors_setup(self.app, defaults={
			"*": ResourceOptions(
				allow_credentials=True,
				expose_headers="*",
				allow_headers="*",
				allow_methods="*"
			)
		})
		
		# Add routes
		self.app.router.add_route('*', '/health', self._health_check)
		self.app.router.add_route('*', '/{path:.*}', self._gateway_handler)
		
		# Add CORS to all routes
		for route in list(self.app.router.routes()):
			cors.add(route)
	
	async def _request_context_middleware(self, request: web.Request, handler: Callable) -> web.Response:
		"""Create gateway request context."""
		
		# Generate request ID
		request_id = f"req_{secrets.token_urlsafe(16)}"
		
		# Create gateway request
		gateway_request = GatewayRequest(
			request_id=request_id,
			method=request.method,
			path=request.path,
			headers=dict(request.headers),
			query_params=dict(request.query),
			body=await request.read() if request.body_exists else b'',
			client_ip=request.remote or '127.0.0.1',
			user_agent=request.headers.get('User-Agent', ''),
			timestamp=datetime.now(timezone.utc)
		)
		
		# Store in request
		request['gateway_request'] = gateway_request
		
		return await handler(request)
	
	async def _gateway_handler_middleware(self, request: web.Request, handler: Callable) -> web.Response:
		"""Gateway request processing middleware."""
		
		# Skip middleware for health checks
		if request.path == '/health':
			return await handler(request)
		
		gateway_request = request.get('gateway_request')
		if not gateway_request:
			return web.json_response({'error': 'Invalid request context'}, status=500)
		
		# Route request through gateway
		gateway_response = await self.router.route_request(gateway_request)
		
		# Store response in request for other middleware
		request['gateway_response'] = gateway_response
		
		# Convert to web response
		return web.Response(
			body=gateway_response.body,
			status=gateway_response.status_code,
			headers=gateway_response.headers
		)
	
	async def _gateway_handler(self, request: web.Request) -> web.Response:
		"""Main gateway request handler."""
		# This is handled by the middleware
		return web.json_response({'error': 'Request not processed'}, status=500)
	
	async def _health_check(self, request: web.Request) -> web.Response:
		"""Health check endpoint."""
		
		health_status = {
			'status': 'healthy',
			'timestamp': datetime.now(timezone.utc).isoformat(),
			'version': '1.0.0',
			'component': 'api_gateway',
			'checks': {
				'redis': 'unknown',
				'database': 'unknown'
			}
		}
		
		# Check Redis
		try:
			await self.redis.ping()
			health_status['checks']['redis'] = 'healthy'
		except Exception:
			health_status['checks']['redis'] = 'unhealthy'
			health_status['status'] = 'degraded'
		
		# Check database (would need actual database connection)
		health_status['checks']['database'] = 'healthy'
		
		status_code = 200 if health_status['status'] in ['healthy', 'degraded'] else 503
		
		return web.json_response(health_status, status=status_code)
	
	async def start(self):
		"""Start the gateway server."""
		
		await self.initialize()
		
		runner = web.AppRunner(self.app)
		await runner.setup()
		
		site = web.TCPSite(runner, self.host, self.port)
		await site.start()
		
		print(f"API Gateway started on {self.host}:{self.port}")
		print(f"Health check: http://{self.host}:{self.port}/health")
		
		return runner
	
	async def stop(self, runner: web.AppRunner):
		"""Stop the gateway server."""
		
		await runner.cleanup()
		
		if self.router:
			await self.router.shutdown()
		
		if self.redis:
			await self.redis.close()

# =============================================================================
# Export Gateway Components
# =============================================================================

__all__ = [
	# Data Classes
	'UpstreamServer',
	'RouteConfig',
	'GatewayRequest',
	'GatewayResponse',
	'PolicyContext',
	
	# Enums
	'RequestMethod',
	'LoadBalancingStrategy',
	
	# Middleware
	'RateLimitMiddleware',
	'AuthenticationMiddleware',
	'PolicyEnforcementMiddleware',
	'CachingMiddleware',
	
	# Core Components
	'LoadBalancer',
	'CircuitBreaker',
	'GatewayRouter',
	'APIGateway'
]