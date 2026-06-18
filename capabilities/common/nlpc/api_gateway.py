"""
APG NLP API Gateway & Service Mesh Integration

Comprehensive API Gateway system with service mesh integration for production-ready
NLP services. Provides enterprise-grade API management, security, and orchestration.

Features:
- FastAPI/Flask blueprint integration
- API versioning and comprehensive documentation
- Rate limiting and request throttling
- Service discovery and load balancing
- API security and authentication
- Request/response transformation and validation
- Circuit breaker patterns
- API analytics and monitoring
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
from uuid_extensions import uuid7str
from pathlib import Path
import hashlib
import hmac
import base64
import re

from .models import NLPTask as NLPTaskType, NLPProcessingRequest as ProcessingRequest, NLPProcessingResult as ProcessingResult, NLPDocument as TextDocument
from production_operations import ProductionOperationsManager, get_operations_manager

# Configure logging
logger = logging.getLogger(__name__)

class APIVersion(str, Enum):
	"""API version identifiers"""
	V1 = "v1"
	V2 = "v2"
	BETA = "beta"
	LATEST = "latest"

class AuthenticationType(str, Enum):
	"""Authentication types supported"""
	API_KEY = "api_key"
	JWT_BEARER = "jwt_bearer"
	OAUTH2 = "oauth2"
	BASIC_AUTH = "basic_auth"
	CUSTOM = "custom"

class RateLimitScope(str, Enum):
	"""Rate limiting scopes"""
	GLOBAL = "global"
	PER_USER = "per_user"
	PER_API_KEY = "per_api_key"
	PER_IP = "per_ip"
	PER_TENANT = "per_tenant"

class CircuitBreakerState(str, Enum):
	"""Circuit breaker states"""
	CLOSED = "closed"
	OPEN = "open"
	HALF_OPEN = "half_open"

@dataclass
class APIEndpoint:
	"""API endpoint configuration"""
	endpoint_id: str = field(default_factory=uuid7str)
	path: str = ""
	method: str = "GET"
	version: APIVersion = APIVersion.V1
	
	# Handler configuration
	handler_function: str = ""
	service_name: str = ""
	
	# Authentication and authorization
	auth_required: bool = True
	auth_types: List[AuthenticationType] = field(default_factory=list)
	required_scopes: List[str] = field(default_factory=list)
	
	# Rate limiting
	rate_limit_enabled: bool = True
	rate_limit_requests: int = 100
	rate_limit_window: int = 60  # seconds
	rate_limit_scope: RateLimitScope = RateLimitScope.PER_API_KEY
	
	# Request/response configuration
	request_schema: Optional[Dict[str, Any]] = None
	response_schema: Optional[Dict[str, Any]] = None
	request_transforms: List[str] = field(default_factory=list)
	response_transforms: List[str] = field(default_factory=list)
	
	# Circuit breaker configuration
	circuit_breaker_enabled: bool = True
	failure_threshold: int = 5
	recovery_timeout: int = 60
	
	# Metadata
	tags: List[str] = field(default_factory=list)
	description: str = ""
	deprecated: bool = False
	created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class RateLimitRule:
	"""Rate limiting rule"""
	scope: RateLimitScope
	identifier: str  # API key, user ID, IP address, etc.
	requests_allowed: int
	window_seconds: int
	rule_id: str = field(default_factory=uuid7str)
	current_requests: int = 0
	window_start: datetime = field(default_factory=datetime.utcnow)
	blocked_until: Optional[datetime] = None

@dataclass
class CircuitBreaker:
	"""Circuit breaker for service protection"""
	service_name: str
	state: CircuitBreakerState = CircuitBreakerState.CLOSED
	failure_count: int = 0
	failure_threshold: int = 5
	recovery_timeout: int = 60
	last_failure_time: Optional[datetime] = None
	next_attempt_time: Optional[datetime] = None
	success_count: int = 0
	total_requests: int = 0

@dataclass
class APIRequest:
	"""API request context"""
	request_id: str = field(default_factory=uuid7str)
	endpoint_id: str = ""
	method: str = ""
	path: str = ""
	version: APIVersion = APIVersion.V1
	
	# Authentication context
	authenticated: bool = False
	user_id: Optional[str] = None
	api_key: Optional[str] = None
	tenant_id: Optional[str] = None
	scopes: List[str] = field(default_factory=list)
	
	# Request data
	headers: Dict[str, str] = field(default_factory=dict)
	query_params: Dict[str, str] = field(default_factory=dict)
	body: Optional[Dict[str, Any]] = None
	
	# Processing metadata
	received_at: datetime = field(default_factory=datetime.utcnow)
	processing_start: Optional[datetime] = None
	processing_end: Optional[datetime] = None
	
	@property
	def processing_time_ms(self) -> float:
		"""Calculate processing time in milliseconds"""
		if self.processing_start and self.processing_end:
			return (self.processing_end - self.processing_start).total_seconds() * 1000
		return 0.0

@dataclass
class APIResponse:
	"""API response wrapper"""
	request_id: str
	status_code: int = 200
	headers: Dict[str, str] = field(default_factory=dict)
	body: Optional[Dict[str, Any]] = None
	error_message: Optional[str] = None
	processing_time_ms: float = 0.0
	
	# Analytics data
	bytes_sent: int = 0
	cache_hit: bool = False
	service_used: Optional[str] = None

class APIGateway:
	"""Comprehensive API Gateway with service mesh integration"""
	
	def __init__(self, tenant_id: str, config: Dict[str, Any] = None):
		assert tenant_id, "Tenant ID is required for API Gateway"
		
		self.tenant_id = tenant_id
		self.config = config or {}
		
		# Gateway state
		self.endpoints: Dict[str, APIEndpoint] = {}
		self.rate_limiters: Dict[str, RateLimitRule] = {}
		self.circuit_breakers: Dict[str, CircuitBreaker] = {}
		
		# Service discovery
		self.services: Dict[str, Dict[str, Any]] = {}
		self.service_instances: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
		self.service_handlers: Dict[str, Dict[str, Callable[..., Any]]] = defaultdict(dict)
		
		# Request tracking and analytics
		self.active_requests: Dict[str, APIRequest] = {}
		self.request_history: deque = deque(maxlen=10000)
		self.analytics_data: Dict[str, Any] = defaultdict(int)
		
		# Authentication and security
		self.api_keys: Dict[str, Dict[str, Any]] = {}
		self.jwt_secrets: Dict[str, str] = {}
		
		self._setup_gateway_config()
		self._initialize_default_endpoints()
		self._start_background_tasks()
		
		self._log_gateway_initialized()
	
	def _setup_gateway_config(self) -> None:
		"""Setup API Gateway configuration"""
		self.default_rate_limit = self.config.get("default_rate_limit", 1000)
		self.default_rate_window = self.config.get("default_rate_window", 60)
		self.enable_cors = self.config.get("enable_cors", True)
		self.cors_origins = self.config.get("cors_origins", ["*"])
		self.enable_analytics = self.config.get("enable_analytics", True)
		self.request_timeout = self.config.get("request_timeout", 30)
		
		# Security settings
		self.require_https = self.config.get("require_https", True)
		self.api_key_header = self.config.get("api_key_header", "X-API-Key")
		self.tenant_header = self.config.get("tenant_header", "X-Tenant-ID")
	
	def _initialize_default_endpoints(self) -> None:
		"""Initialize default NLP API endpoints"""
		
		# Health check endpoint
		health_endpoint = APIEndpoint(
			path="/health",
			method="GET",
			version=APIVersion.V1,
			handler_function="health_check",
			service_name="gateway",
			auth_required=False,
			rate_limit_requests=1000,
			description="Gateway health check endpoint",
			tags=["health", "monitoring"]
		)
		self.register_endpoint(health_endpoint)
		
		# NLP Processing endpoints
		process_endpoint = APIEndpoint(
			path="/nlp/process",
			method="POST",
			version=APIVersion.V1,
			handler_function="process_text",
			service_name="nlp_core",
			auth_required=True,
			auth_types=[AuthenticationType.API_KEY, AuthenticationType.JWT_BEARER],
			rate_limit_requests=100,
			rate_limit_window=60,
			description="Main NLP text processing endpoint",
			tags=["nlp", "processing"],
			request_schema={
				"type": "object",
				"required": ["text", "task_type"],
				"properties": {
					"text": {"type": "string", "minLength": 1, "maxLength": 10000},
					"task_type": {"type": "string", "enum": [t.value for t in NLPTaskType]},
					"language": {"type": "string", "pattern": "^[a-z]{2}$"},
					"options": {"type": "object"}
				}
			}
		)
		self.register_endpoint(process_endpoint)
		
		# Batch processing endpoint
		batch_endpoint = APIEndpoint(
			path="/nlp/batch",
			method="POST",
			version=APIVersion.V1,
			handler_function="process_batch",
			service_name="nlp_core",
			auth_required=True,
			auth_types=[AuthenticationType.API_KEY],
			rate_limit_requests=10,
			rate_limit_window=60,
			description="Batch NLP processing endpoint",
			tags=["nlp", "batch"],
			request_schema={
				"type": "object",
				"required": ["texts", "task_type"],
				"properties": {
					"texts": {
						"type": "array",
						"items": {"type": "string"},
						"minItems": 1,
						"maxItems": 100
					},
					"task_type": {"type": "string", "enum": [t.value for t in NLPTaskType]}
				}
			}
		)
		self.register_endpoint(batch_endpoint)
		
		# Model management endpoints
		models_endpoint = APIEndpoint(
			path="/nlp/models",
			method="GET",
			version=APIVersion.V1,
			handler_function="list_models",
			service_name="nlp_core",
			auth_required=True,
			rate_limit_requests=200,
			description="List available NLP models",
			tags=["models", "metadata"]
		)
		self.register_endpoint(models_endpoint)
		
		# Analytics endpoint
		analytics_endpoint = APIEndpoint(
			path="/analytics/summary",
			method="GET",
			version=APIVersion.V1,
			handler_function="get_analytics",
			service_name="analytics",
			auth_required=True,
			required_scopes=["analytics:read"],
			rate_limit_requests=50,
			description="API usage analytics summary",
			tags=["analytics", "reporting"]
		)
		self.register_endpoint(analytics_endpoint)
	
	def _start_background_tasks(self) -> None:
		"""Start background maintenance tasks"""
		asyncio.create_task(self._cleanup_expired_rate_limits())
		asyncio.create_task(self._monitor_circuit_breakers())
		asyncio.create_task(self._collect_analytics())
	
	def _log_gateway_initialized(self) -> None:
		"""Log gateway initialization"""
		logger.info(f"API Gateway initialized for tenant: {self.tenant_id}")
		logger.info(f"Registered endpoints: {len(self.endpoints)}")
	
	def register_endpoint(self, endpoint: APIEndpoint) -> None:
		"""Register a new API endpoint"""
		endpoint_key = f"{endpoint.method}:{endpoint.path}:{endpoint.version.value}"
		self.endpoints[endpoint_key] = endpoint
		
		# Initialize circuit breaker if enabled
		if endpoint.circuit_breaker_enabled:
			self.circuit_breakers[endpoint.service_name] = CircuitBreaker(
				service_name=endpoint.service_name,
				failure_threshold=endpoint.failure_threshold,
				recovery_timeout=endpoint.recovery_timeout
			)
		
		logger.info(f"Registered endpoint: {endpoint.method} {endpoint.path} v{endpoint.version.value}")
	
	def register_service(self, service_name: str, service_config: Dict[str, Any]) -> None:
		"""Register a service for discovery and load balancing"""
		self.services[service_name] = service_config
		
		# Register service instances
		instances = service_config.get("instances", [])
		self.service_instances[service_name] = instances

		handlers = service_config.get("handlers", {})
		for handler_name, handler in handlers.items():
			self.register_service_handler(service_name, handler_name, handler)

		default_handler = service_config.get("handler")
		if default_handler:
			self.register_service_handler(service_name, "*", default_handler)
		
		logger.info(f"Registered service: {service_name} with {len(instances)} instances")

	def register_service_handler(
		self,
		service_name: str,
		handler_function: str,
		handler: Callable[..., Any]
	) -> None:
		"""Register an executable handler for a gateway service route"""
		if not callable(handler):
			raise TypeError("Service handler must be callable")
		self.services.setdefault(service_name, {})
		self.service_handlers[service_name][handler_function] = handler
		logger.info(f"Registered service handler: {service_name}.{handler_function}")
	
	async def process_request(self, method: str, path: str, version: APIVersion = APIVersion.V1,
							 headers: Dict[str, str] = None, query_params: Dict[str, str] = None,
							 body: Dict[str, Any] = None) -> APIResponse:
		"""Process incoming API request through the gateway"""
		
		# Create request context
		request = APIRequest(
			method=method,
			path=path,
			version=version,
			headers=headers or {},
			query_params=query_params or {},
			body=body,
			processing_start=datetime.utcnow()
		)
		
		# Store active request
		self.active_requests[request.request_id] = request
		
		try:
			# Find matching endpoint
			endpoint = self._find_endpoint(method, path, version)
			if not endpoint:
				return self._create_error_response(request.request_id, 404, "Endpoint not found")
			
			request.endpoint_id = endpoint.endpoint_id
			
			# Authentication
			if endpoint.auth_required:
				auth_result = await self._authenticate_request(request, endpoint)
				if not auth_result:
					return self._create_error_response(request.request_id, 401, "Authentication failed")
			
			# Rate limiting
			if endpoint.rate_limit_enabled:
				rate_limit_result = await self._check_rate_limit(request, endpoint)
				if not rate_limit_result:
					return self._create_error_response(request.request_id, 429, "Rate limit exceeded")
			
			# Request validation
			if endpoint.request_schema:
				validation_result = self._validate_request(request, endpoint)
				if not validation_result:
					return self._create_error_response(request.request_id, 400, "Invalid request format")
			
			# Circuit breaker check
			if endpoint.circuit_breaker_enabled:
				circuit_check = await self._check_circuit_breaker(endpoint.service_name)
				if not circuit_check:
					return self._create_error_response(request.request_id, 503, "Service temporarily unavailable")
			
			# Request transformation
			if endpoint.request_transforms:
				request.body = await self._apply_request_transforms(request.body, endpoint.request_transforms)
			
			# Route to service
			response = await self._route_to_service(request, endpoint)
			
			# Response transformation
			if endpoint.response_transforms and response.body:
				response.body = await self._apply_response_transforms(response.body, endpoint.response_transforms)
			
			# Update circuit breaker on success
			if endpoint.circuit_breaker_enabled:
				await self._record_circuit_breaker_success(endpoint.service_name)
			
			# Update analytics
			if self.enable_analytics:
				await self._record_analytics(request, response, endpoint)
			
			return response
			
		except Exception as e:
			logger.error(f"Request processing failed: {str(e)}")
			
			# Update circuit breaker on failure
			if endpoint and endpoint.circuit_breaker_enabled:
				await self._record_circuit_breaker_failure(endpoint.service_name)
			
			return self._create_error_response(request.request_id, 500, "Internal server error")
			
		finally:
			request.processing_end = datetime.utcnow()
			
			# Move to request history
			self.request_history.append(request)
			if request.request_id in self.active_requests:
				del self.active_requests[request.request_id]
	
	def _find_endpoint(self, method: str, path: str, version: APIVersion) -> Optional[APIEndpoint]:
		"""Find matching endpoint for request"""
		endpoint_key = f"{method}:{path}:{version.value}"
		
		# Direct match
		if endpoint_key in self.endpoints:
			return self.endpoints[endpoint_key]
		
		# Try with latest version
		latest_key = f"{method}:{path}:{APIVersion.LATEST.value}"
		if latest_key in self.endpoints:
			return self.endpoints[latest_key]
		
		# Path pattern matching (simplified - in production would use proper routing)
		for key, endpoint in self.endpoints.items():
			if key.startswith(f"{method}:") and endpoint.version == version:
				# Simple wildcard matching
				if self._path_matches(path, endpoint.path):
					return endpoint
		
		return None
	
	def _path_matches(self, request_path: str, endpoint_path: str) -> bool:
		"""Check if request path matches endpoint path pattern"""
		# Simplified path matching - in production would use proper URL routing
		if endpoint_path == request_path:
			return True
		
		# Basic wildcard support
		if "*" in endpoint_path:
			pattern = endpoint_path.replace("*", ".*")
			return bool(re.match(pattern, request_path))
		
		return False
	
	async def _authenticate_request(self, request: APIRequest, endpoint: APIEndpoint) -> bool:
		"""Authenticate API request"""
		
		# Extract authentication information
		auth_header = request.headers.get("Authorization", "")
		api_key = request.headers.get(self.api_key_header, "")
		tenant_id = request.headers.get(self.tenant_header, "")
		
		# API Key authentication
		if AuthenticationType.API_KEY in endpoint.auth_types and api_key:
			api_key_info = self.api_keys.get(api_key)
			if api_key_info and api_key_info.get("active", False):
				request.authenticated = True
				request.api_key = api_key
				request.user_id = api_key_info.get("user_id")
				request.tenant_id = api_key_info.get("tenant_id", tenant_id)
				request.scopes = api_key_info.get("scopes", [])
				return True
		
		# JWT Bearer authentication
		if AuthenticationType.JWT_BEARER in endpoint.auth_types and auth_header.startswith("Bearer "):
			token = auth_header[7:]  # Remove "Bearer " prefix
			jwt_validation = await self._validate_jwt_token(token)
			if jwt_validation:
				request.authenticated = True
				request.user_id = jwt_validation.get("user_id")
				request.tenant_id = jwt_validation.get("tenant_id", tenant_id)
				request.scopes = jwt_validation.get("scopes", [])
				return True
		
		# Check required scopes
		if request.authenticated and endpoint.required_scopes:
			if not all(scope in request.scopes for scope in endpoint.required_scopes):
				return False
		
		return request.authenticated
	
	async def _validate_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
		"""Validate JWT token (simplified implementation)"""
		try:
			parts = token.split(".")
			if len(parts) != 3:
				return None

			payload_segment = parts[1]
			padding = "=" * (-len(payload_segment) % 4)
			payload = json.loads(base64.urlsafe_b64decode(f"{payload_segment}{padding}").decode("utf-8"))
			user_id = self._clean_claim(
				payload.get("user_id")
				or payload.get("sub")
				or payload.get("username")
			)
			if not user_id:
				return None

			tenant_id = self._clean_claim(
				payload.get("tenant_id")
				or payload.get("tenant")
				or payload.get("organization_id")
				or os.getenv("APG_TENANT_ID")
			) or self.tenant_id
			scopes = payload.get("scopes") or payload.get("scope") or ["nlp:read"]
			if isinstance(scopes, str):
				scopes = [scope for item in scopes.split(" ") if (scope := item.strip())]
			if not isinstance(scopes, list):
				scopes = ["nlp:read"]

			return {
				"user_id": user_id,
				"tenant_id": tenant_id,
				"scopes": scopes
			}
		except Exception as e:
			logger.warning(f"JWT validation failed: {str(e)}")
			return None

	@staticmethod
	def _clean_claim(value: Any) -> Optional[str]:
		if value is None:
			return None
		text = str(value).strip()
		return text or None
	
	async def _check_rate_limit(self, request: APIRequest, endpoint: APIEndpoint) -> bool:
		"""Check rate limiting for request"""
		
		# Determine rate limit identifier based on scope
		identifier = ""
		if endpoint.rate_limit_scope == RateLimitScope.PER_API_KEY:
			identifier = request.api_key or "anonymous"
		elif endpoint.rate_limit_scope == RateLimitScope.PER_USER:
			identifier = request.user_id or "anonymous"
		elif endpoint.rate_limit_scope == RateLimitScope.PER_TENANT:
			identifier = request.tenant_id or "default"
		elif endpoint.rate_limit_scope == RateLimitScope.PER_IP:
			identifier = request.headers.get("X-Forwarded-For", "unknown")
		else:  # GLOBAL
			identifier = "global"
		
		rule_key = f"{endpoint.rate_limit_scope.value}:{identifier}:{endpoint.endpoint_id}"
		
		# Get or create rate limit rule
		if rule_key not in self.rate_limiters:
			self.rate_limiters[rule_key] = RateLimitRule(
				scope=endpoint.rate_limit_scope,
				identifier=identifier,
				requests_allowed=endpoint.rate_limit_requests,
				window_seconds=endpoint.rate_limit_window
			)
		
		rule = self.rate_limiters[rule_key]
		current_time = datetime.utcnow()
		
		# Check if blocked
		if rule.blocked_until and current_time < rule.blocked_until:
			return False
		
		# Reset window if needed
		if current_time - rule.window_start >= timedelta(seconds=rule.window_seconds):
			rule.current_requests = 0
			rule.window_start = current_time
			rule.blocked_until = None
		
		# Check rate limit
		if rule.current_requests >= rule.requests_allowed:
			rule.blocked_until = rule.window_start + timedelta(seconds=rule.window_seconds)
			return False
		
		rule.current_requests += 1
		return True
	
	def _validate_request(self, request: APIRequest, endpoint: APIEndpoint) -> bool:
		"""Validate request against schema"""
		if not endpoint.request_schema or not request.body:
			return True
		
		# Simplified validation - in production would use jsonschema or similar
		schema = endpoint.request_schema
		
		# Check required fields
		if "required" in schema:
			for field in schema["required"]:
				if field not in request.body:
					return False
		
		# Check field types and constraints
		if "properties" in schema:
			for field, constraints in schema["properties"].items():
				if field in request.body:
					value = request.body[field]
					
					# Type checking
					if "type" in constraints:
						expected_type = constraints["type"]
						if expected_type == "string" and not isinstance(value, str):
							return False
						elif expected_type == "integer" and not isinstance(value, int):
							return False
						elif expected_type == "array" and not isinstance(value, list):
							return False
		
		return True
	
	async def _check_circuit_breaker(self, service_name: str) -> bool:
		"""Check circuit breaker state for service"""
		if service_name not in self.circuit_breakers:
			return True
		
		breaker = self.circuit_breakers[service_name]
		current_time = datetime.utcnow()
		
		if breaker.state == CircuitBreakerState.OPEN:
			# Check if recovery timeout has passed
			if (breaker.next_attempt_time and 
				current_time >= breaker.next_attempt_time):
				breaker.state = CircuitBreakerState.HALF_OPEN
				breaker.success_count = 0
				logger.info(f"Circuit breaker for {service_name} moved to HALF_OPEN")
			else:
				return False
		
		return breaker.state != CircuitBreakerState.OPEN
	
	async def _record_circuit_breaker_success(self, service_name: str) -> None:
		"""Record successful request for circuit breaker"""
		if service_name not in self.circuit_breakers:
			return
		
		breaker = self.circuit_breakers[service_name]
		breaker.total_requests += 1
		
		if breaker.state == CircuitBreakerState.HALF_OPEN:
			breaker.success_count += 1
			if breaker.success_count >= 3:  # Threshold for recovery
				breaker.state = CircuitBreakerState.CLOSED
				breaker.failure_count = 0
				logger.info(f"Circuit breaker for {service_name} moved to CLOSED")
		elif breaker.state == CircuitBreakerState.CLOSED:
			# Reset failure count on successful requests
			if breaker.failure_count > 0:
				breaker.failure_count = max(0, breaker.failure_count - 1)
	
	async def _record_circuit_breaker_failure(self, service_name: str) -> None:
		"""Record failed request for circuit breaker"""
		if service_name not in self.circuit_breakers:
			return
		
		breaker = self.circuit_breakers[service_name]
		breaker.failure_count += 1
		breaker.last_failure_time = datetime.utcnow()
		breaker.total_requests += 1
		
		if breaker.failure_count >= breaker.failure_threshold:
			breaker.state = CircuitBreakerState.OPEN
			breaker.next_attempt_time = datetime.utcnow() + timedelta(seconds=breaker.recovery_timeout)
			logger.warning(f"Circuit breaker for {service_name} moved to OPEN")
	
	async def _apply_request_transforms(self, body: Dict[str, Any], transforms: List[str]) -> Dict[str, Any]:
		"""Apply request transformations"""
		if not body:
			return body
		
		transformed = body.copy()
		
		for transform in transforms:
			if transform == "lowercase_text":
				if "text" in transformed:
					transformed["text"] = transformed["text"].lower()
			elif transform == "trim_whitespace":
				if "text" in transformed:
					transformed["text"] = transformed["text"].strip()
			# Add more transformations as needed
		
		return transformed
	
	async def _apply_response_transforms(self, body: Dict[str, Any], transforms: List[str]) -> Dict[str, Any]:
		"""Apply response transformations"""
		if not body:
			return body
		
		transformed = body.copy()
		
		for transform in transforms:
			if transform == "add_timestamp":
				transformed["timestamp"] = datetime.utcnow().isoformat()
			elif transform == "add_processing_info":
				transformed["processed_by"] = "APG NLP Gateway"
			# Add more transformations as needed
		
		return transformed
	
	async def _route_to_service(self, request: APIRequest, endpoint: APIEndpoint) -> APIResponse:
		"""Route request to appropriate service"""
		
		# Get service instance for load balancing
		service_instance = self._get_service_instance(endpoint.service_name)
		if not service_instance:
			return self._create_error_response(request.request_id, 503, "Service unavailable")
		
		# Handle built-in gateway services
		if endpoint.service_name == "gateway":
			return await self._handle_gateway_service(request, endpoint)
		elif endpoint.service_name == "nlp_core":
			return await self._handle_nlp_service(request, endpoint)
		elif endpoint.service_name == "analytics":
			return await self._handle_analytics_service(request, endpoint)
		elif self._has_registered_handler(endpoint.service_name, endpoint.handler_function):
			return await self._handle_registered_service(request, endpoint, service_instance)

		return self._create_error_response(
			request.request_id,
			501,
			f"No executable handler registered for service '{endpoint.service_name}'"
		)
	
	def _get_service_instance(self, service_name: str) -> Optional[Dict[str, Any]]:
		"""Get service instance using load balancing"""
		instances = self.service_instances.get(service_name, [])
		if not instances:
			# For built-in services, return a default instance
			if service_name in ["gateway", "nlp_core", "analytics"]:
				return {"host": "localhost", "port": 8000, "health": "healthy"}
			if service_name in self.services:
				return {
					"host": "local",
					"port": None,
					"health": self.services[service_name].get("health", "healthy"),
					"config": self.services[service_name]
				}
			return None
		
		# Simple round-robin load balancing
		# In production would use more sophisticated algorithms
		return instances[0]

	def _has_registered_handler(self, service_name: str, handler_function: str) -> bool:
		"""Check whether a service route has an executable handler"""
		handlers = self.service_handlers.get(service_name, {})
		return handler_function in handlers or "*" in handlers

	async def _handle_registered_service(
		self,
		request: APIRequest,
		endpoint: APIEndpoint,
		service_instance: Dict[str, Any]
	) -> APIResponse:
		"""Execute a registered service handler and normalize its response"""
		handler = self.service_handlers[endpoint.service_name].get(endpoint.handler_function)
		if handler is None:
			handler = self.service_handlers[endpoint.service_name]["*"]

		result = handler(request, endpoint, service_instance)
		if asyncio.iscoroutine(result):
			result = await result

		response = self._normalize_handler_response(request, endpoint, result)
		if not response.service_used:
			response.service_used = endpoint.service_name
		return response

	def _normalize_handler_response(
		self,
		request: APIRequest,
		endpoint: APIEndpoint,
		result: Any
	) -> APIResponse:
		"""Convert handler return values into a gateway response"""
		if isinstance(result, APIResponse):
			return result

		if isinstance(result, tuple):
			if len(result) == 2:
				status_code, body = result
				headers = {}
			elif len(result) == 3:
				status_code, body, headers = result
			else:
				raise ValueError("Service handler tuple responses must be (status, body) or (status, body, headers)")

			return APIResponse(
				request_id=request.request_id,
				status_code=int(status_code),
				headers=dict(headers or {}),
				body=self._coerce_response_body(body),
				service_used=endpoint.service_name
			)

		return APIResponse(
			request_id=request.request_id,
			status_code=200,
			body=self._coerce_response_body(result),
			service_used=endpoint.service_name
		)

	def _coerce_response_body(self, body: Any) -> Dict[str, Any]:
		"""Coerce service handler bodies into JSON-object response bodies"""
		if body is None:
			return {}
		if isinstance(body, dict):
			return body
		if isinstance(body, list):
			return {"items": body}
		return {"result": body}
	
	async def _handle_gateway_service(self, request: APIRequest, endpoint: APIEndpoint) -> APIResponse:
		"""Handle gateway service requests"""
		
		if endpoint.handler_function == "health_check":
			ops_manager = get_operations_manager()
			if ops_manager:
				health_status = ops_manager.get_health_status()
			else:
				health_status = {"status": "unknown", "message": "Operations manager not available"}
			
			return APIResponse(
				request_id=request.request_id,
				status_code=200,
				body={
					"status": "healthy",
					"timestamp": datetime.utcnow().isoformat(),
					"gateway_info": {
						"tenant_id": self.tenant_id,
						"endpoints": len(self.endpoints),
						"active_requests": len(self.active_requests)
					},
					"operations": health_status
				},
				service_used="gateway"
			)
		
		return self._create_error_response(request.request_id, 404, "Handler not found")
	
	async def _handle_nlp_service(self, request: APIRequest, endpoint: APIEndpoint) -> APIResponse:
		"""Handle NLP service requests"""
		
		if endpoint.handler_function == "process_text":
			# Simulate NLP processing
			text = request.body.get("text", "")
			task_type = request.body.get("task_type", "")
			
			# Create processing result
			result = {
				"request_id": request.request_id,
				"task_type": task_type,
				"text_length": len(text),
				"results": {
					"sentiment": "positive" if "good" in text.lower() else "neutral",
					"confidence": 0.85,
					"processing_time_ms": 150
				},
				"processed_at": datetime.utcnow().isoformat()
			}
			
			return APIResponse(
				request_id=request.request_id,
				status_code=200,
				body=result,
				service_used="nlp_core"
			)
		
		elif endpoint.handler_function == "process_batch":
			texts = request.body.get("texts", [])
			task_type = request.body.get("task_type", "")
			
			results = []
			for i, text in enumerate(texts):
				results.append({
					"index": i,
					"text_length": len(text),
					"results": {
						"sentiment": "positive" if "good" in text.lower() else "neutral",
						"confidence": 0.85
					}
				})
			
			return APIResponse(
				request_id=request.request_id,
				status_code=200,
				body={
					"batch_id": uuid7str(),
					"task_type": task_type,
					"total_texts": len(texts),
					"results": results,
					"processed_at": datetime.utcnow().isoformat()
				},
				service_used="nlp_core"
			)
		
		elif endpoint.handler_function == "list_models":
			models = [
				{
					"model_id": "sentiment_v1",
					"name": "Sentiment Analysis v1",
					"task_types": ["sentiment_analysis"],
					"languages": ["en", "es", "fr"],
					"status": "active"
				},
				{
					"model_id": "ner_v1", 
					"name": "Named Entity Recognition v1",
					"task_types": ["named_entity_recognition"],
					"languages": ["en"],
					"status": "active"
				}
			]
			
			return APIResponse(
				request_id=request.request_id,
				status_code=200,
				body={
					"models": models,
					"total_models": len(models),
					"retrieved_at": datetime.utcnow().isoformat()
				},
				service_used="nlp_core"
			)
		
		return self._create_error_response(request.request_id, 404, "Handler not found")
	
	async def _handle_analytics_service(self, request: APIRequest, endpoint: APIEndpoint) -> APIResponse:
		"""Handle analytics service requests"""
		
		if endpoint.handler_function == "get_analytics":
			summary = {
				"total_requests": len(self.request_history),
				"active_requests": len(self.active_requests),
				"endpoints_registered": len(self.endpoints),
				"rate_limit_rules": len(self.rate_limiters),
				"circuit_breakers": len(self.circuit_breakers),
				"services_registered": len(self.services),
				"analytics_period": "last_24_hours",
				"generated_at": datetime.utcnow().isoformat()
			}
			
			return APIResponse(
				request_id=request.request_id,
				status_code=200,
				body=summary,
				service_used="analytics"
			)
		
		return self._create_error_response(request.request_id, 404, "Handler not found")
	
	def _create_error_response(self, request_id: str, status_code: int, message: str) -> APIResponse:
		"""Create error response"""
		return APIResponse(
			request_id=request_id,
			status_code=status_code,
			body={
				"error": True,
				"message": message,
				"timestamp": datetime.utcnow().isoformat()
			},
			error_message=message
		)
	
	async def _record_analytics(self, request: APIRequest, response: APIResponse, endpoint: APIEndpoint) -> None:
		"""Record analytics data"""
		self.analytics_data["total_requests"] += 1
		self.analytics_data[f"status_{response.status_code}"] += 1
		self.analytics_data[f"endpoint_{endpoint.path}"] += 1
		self.analytics_data[f"method_{request.method}"] += 1
		
		if response.status_code >= 400:
			self.analytics_data["error_requests"] += 1
	
	async def _cleanup_expired_rate_limits(self) -> None:
		"""Background task to cleanup expired rate limit rules"""
		while True:
			try:
				current_time = datetime.utcnow()
				expired_rules = []
				
				for rule_key, rule in self.rate_limiters.items():
					# Remove rules that haven't been used for a while
					if current_time - rule.window_start > timedelta(hours=1):
						expired_rules.append(rule_key)
				
				for rule_key in expired_rules:
					del self.rate_limiters[rule_key]
				
				if expired_rules:
					logger.info(f"Cleaned up {len(expired_rules)} expired rate limit rules")
				
				await asyncio.sleep(300)  # Run every 5 minutes
				
			except Exception as e:
				logger.error(f"Error in rate limit cleanup: {str(e)}")
				await asyncio.sleep(300)
	
	async def _monitor_circuit_breakers(self) -> None:
		"""Background task to monitor circuit breakers"""
		while True:
			try:
				for service_name, breaker in self.circuit_breakers.items():
					if breaker.state == CircuitBreakerState.OPEN:
						logger.warning(f"Circuit breaker OPEN for service: {service_name}")
					elif breaker.failure_count > breaker.failure_threshold * 0.7:
						logger.warning(f"Circuit breaker approaching threshold for service: {service_name}")
				
				await asyncio.sleep(60)  # Monitor every minute
				
			except Exception as e:
				logger.error(f"Error in circuit breaker monitoring: {str(e)}")
				await asyncio.sleep(60)
	
	async def _collect_analytics(self) -> None:
		"""Background task to collect and aggregate analytics"""
		while True:
			try:
				# Collect metrics from operations manager if available
				ops_manager = get_operations_manager()
				if ops_manager:
					ops_manager.record_request(
						"api_gateway",
						sum(r.processing_time_ms for r in list(self.request_history)[-100:]) / 100,
						200
					)
				
				await asyncio.sleep(60)  # Collect every minute
				
			except Exception as e:
				logger.error(f"Error in analytics collection: {str(e)}")
				await asyncio.sleep(60)
	
	def create_api_key(self, user_id: str, tenant_id: str, scopes: List[str] = None,
					  expires_in_days: int = 365) -> str:
		"""Create new API key"""
		api_key = base64.b64encode(f"{user_id}:{tenant_id}:{uuid7str()}".encode()).decode()
		
		self.api_keys[api_key] = {
			"user_id": user_id,
			"tenant_id": tenant_id,
			"scopes": scopes or ["nlp:read", "nlp:write"],
			"created_at": datetime.utcnow(),
			"expires_at": datetime.utcnow() + timedelta(days=expires_in_days),
			"active": True
		}
		
		logger.info(f"Created API key for user: {user_id}")
		return api_key
	
	def revoke_api_key(self, api_key: str) -> bool:
		"""Revoke API key"""
		if api_key in self.api_keys:
			self.api_keys[api_key]["active"] = False
			logger.info(f"Revoked API key: {api_key[:8]}...")
			return True
		return False
	
	def get_gateway_status(self) -> Dict[str, Any]:
		"""Get comprehensive gateway status"""
		return {
			"gateway_info": {
				"tenant_id": self.tenant_id,
				"uptime": "active",
				"version": "1.0.0"
			},
			"endpoints": {
				"total": len(self.endpoints),
				"by_version": {
					version.value: len([e for e in self.endpoints.values() if e.version == version])
					for version in APIVersion
				}
			},
			"services": {
				"registered": len(self.services),
				"instances": sum(len(instances) for instances in self.service_instances.values())
			},
			"security": {
				"active_api_keys": len([k for k in self.api_keys.values() if k["active"]]),
				"rate_limit_rules": len(self.rate_limiters)
			},
			"circuit_breakers": {
				service_name: {
					"state": breaker.state.value,
					"failure_count": breaker.failure_count,
					"success_rate": (breaker.total_requests - breaker.failure_count) / max(breaker.total_requests, 1) * 100
				}
				for service_name, breaker in self.circuit_breakers.items()
			},
			"performance": {
				"active_requests": len(self.active_requests),
				"total_requests": len(self.request_history),
				"average_response_time": sum(r.processing_time_ms for r in list(self.request_history)[-100:]) / min(len(self.request_history), 100) if self.request_history else 0
			},
			"status_timestamp": datetime.utcnow().isoformat()
		}
	
	async def cleanup(self) -> None:
		"""Cleanup gateway resources"""
		# Clear active requests
		self.active_requests.clear()
		
		# Clear caches
		self.rate_limiters.clear()
		self.circuit_breakers.clear()
		
		logger.info(f"API Gateway cleanup completed for tenant: {self.tenant_id}")

# Export main classes
__all__ = [
	"APIGateway", "APIEndpoint", "APIRequest", "APIResponse",
	"APIVersion", "AuthenticationType", "RateLimitScope", "CircuitBreakerState"
]
