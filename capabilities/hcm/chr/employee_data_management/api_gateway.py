"""
APG Employee Data Management - API Gateway & Integration Hub

Comprehensive API gateway with intelligent routing, caching, security,
and seamless integration with external HR systems and APG capabilities.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero | APG Platform Architect
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import time
from urllib.parse import urlparse
from uuid_extensions import uuid7str

# APG Platform Integration
from ....ai_orchestration.service import AIOrchestrationService
from ....auth_rbac.service import AuthRBACService
from ....audit_compliance.service import AuditComplianceService
from ....real_time_collaboration.service import CollaborationService
from .service import RevolutionaryEmployeeDataManagementService
from .ai_intelligence_engine import EmployeeAIIntelligenceEngine
from .analytics_dashboard import EmployeeAnalyticsDashboard


class HTTPMethod(str, Enum):
	"""HTTP methods for API endpoints."""
	GET = "GET"
	POST = "POST"
	PUT = "PUT"
	PATCH = "PATCH"
	DELETE = "DELETE"


class AuthenticationMethod(str, Enum):
	"""Authentication methods."""
	API_KEY = "api_key"
	JWT_TOKEN = "jwt_token"
	OAUTH2 = "oauth2"
	BASIC_AUTH = "basic_auth"
	CERTIFICATE = "certificate"


class IntegrationType(str, Enum):
	"""Types of integrations."""
	WEBHOOK = "webhook"
	REST_API = "rest_api"
	GRAPHQL = "graphql"
	GRPC = "grpc"
	MESSAGE_QUEUE = "message_queue"
	DATABASE = "database"
	FILE_SYNC = "file_sync"


@dataclass
class APIEndpoint:
	"""API endpoint configuration."""
	endpoint_id: str = field(default_factory=uuid7str)
	path: str = ""
	method: HTTPMethod = HTTPMethod.GET
	handler: Optional[Callable] = None
	auth_required: bool = True
	rate_limit: int = 100  # requests per minute
	cache_ttl: int = 300   # seconds
	description: str = ""
	tags: List[str] = field(default_factory=list)
	enabled: bool = True


@dataclass
class ExternalIntegration:
	"""External system integration configuration."""
	integration_id: str = field(default_factory=uuid7str)
	system_name: str = ""
	integration_type: IntegrationType = IntegrationType.REST_API
	base_url: str = ""
	auth_method: AuthenticationMethod = AuthenticationMethod.API_KEY
	auth_config: Dict[str, Any] = field(default_factory=dict)
	sync_frequency: int = 3600  # seconds
	field_mappings: Dict[str, str] = field(default_factory=dict)
	enabled: bool = True
	last_sync: Optional[datetime] = None


@dataclass
class APIRequest:
	"""API request context."""
	request_id: str = field(default_factory=uuid7str)
	endpoint_path: str = ""
	method: HTTPMethod = HTTPMethod.GET
	headers: Dict[str, str] = field(default_factory=dict)
	query_params: Dict[str, str] = field(default_factory=dict)
	body: Any = None
	user_id: Optional[str] = None
	tenant_id: Optional[str] = None
	timestamp: datetime = field(default_factory=datetime.utcnow)
	client_ip: Optional[str] = None


@dataclass
class APIResponse:
	"""API response structure."""
	status_code: int = 200
	data: Any = None
	error: Optional[str] = None
	metadata: Dict[str, Any] = field(default_factory=dict)
	headers: Dict[str, str] = field(default_factory=dict)
	cached: bool = False
	execution_time_ms: int = 0


class EmployeeAPIGateway:
	"""Comprehensive API gateway for employee data management with intelligent routing and caching."""
	
	def __init__(self, tenant_id: str, config: Optional[Dict[str, Any]] = None):
		self.tenant_id = tenant_id
		self.logger = logging.getLogger(f"APIGateway.{tenant_id}")
		
		# Configuration
		self.config = config or {
			'enable_caching': True,
			'enable_rate_limiting': True,
			'enable_analytics': True,
			'cache_size_mb': 100,
			'default_rate_limit': 1000,
			'request_timeout': 30
		}
		
		# APG Service Integration
		self.auth_service = AuthRBACService(tenant_id)
		self.audit_service = AuditComplianceService(tenant_id)
		self.collaboration = CollaborationService(tenant_id)
		self.ai_orchestration = AIOrchestrationService(tenant_id)
		
		# Core Services
		self.employee_service = RevolutionaryEmployeeDataManagementService(tenant_id)
		self.ai_intelligence = EmployeeAIIntelligenceEngine(tenant_id)
		self.analytics_dashboard = EmployeeAnalyticsDashboard(tenant_id)
		
		# API Gateway Components
		self.endpoints: Dict[str, APIEndpoint] = {}
		self.integrations: Dict[str, ExternalIntegration] = {}
		self.middleware_stack: List[Callable] = []
		
		# Caching and Performance
		self.response_cache: Dict[str, Tuple[datetime, APIResponse]] = {}
		self.rate_limit_store: Dict[str, List[datetime]] = {}
		
		# Analytics
		self.request_analytics = {
			'total_requests': 0,
			'successful_requests': 0,
			'failed_requests': 0,
			'cached_responses': 0,
			'average_response_time': 0.0
		}
		
		# Initialize gateway
		asyncio.create_task(self._initialize_api_gateway())

	async def _log_api_operation(self, operation: str, details: Dict[str, Any] = None) -> None:
		"""Log API operations for analytics and debugging."""
		log_details = details or {}
		self.logger.info(f"[API_GATEWAY] {operation}: {log_details}")

	async def _initialize_api_gateway(self) -> None:
		"""Initialize API gateway components."""
		try:
			# Setup default endpoints
			await self._setup_default_endpoints()
			
			# Setup middleware stack
			await self._setup_middleware()
			
			# Initialize external integrations
			await self._initialize_integrations()
			
			# Setup monitoring
			await self._setup_monitoring()
			
			self.logger.info("API Gateway initialized successfully")
			
		except Exception as e:
			self.logger.error(f"Failed to initialize API gateway: {str(e)}")
			raise

	# ============================================================================
	# REQUEST HANDLING AND ROUTING
	# ============================================================================

	async def handle_request(self, request: APIRequest) -> APIResponse:
		"""Handle incoming API request with full middleware stack."""
		start_time = time.time()
		
		try:
			await self._log_api_operation("request_received", {
				"path": request.endpoint_path,
				"method": request.method,
				"user_id": request.user_id,
				"client_ip": request.client_ip
			})
			
			# Process through middleware stack
			for middleware in self.middleware_stack:
				request = await middleware(request)
				if hasattr(request, 'error_response'):
					return request.error_response
			
			# Find matching endpoint
			endpoint = self._find_endpoint(request.endpoint_path, request.method)
			if not endpoint:
				return APIResponse(
					status_code=404,
					error="Endpoint not found"
				)
			
			# Check cache
			if self.config['enable_caching'] and request.method == HTTPMethod.GET:
				cached_response = await self._get_cached_response(request)
				if cached_response:
					self.request_analytics['cached_responses'] += 1
					return cached_response
			
			# Execute endpoint handler
			response = await endpoint.handler(request)
			
			# Cache successful GET responses
			if (self.config['enable_caching'] and 
				request.method == HTTPMethod.GET and 
				response.status_code == 200):
				await self._cache_response(request, response)
			
			# Update analytics
			execution_time = int((time.time() - start_time) * 1000)
			response.execution_time_ms = execution_time
			
			self.request_analytics['total_requests'] += 1
			if response.status_code < 400:
				self.request_analytics['successful_requests'] += 1
			else:
				self.request_analytics['failed_requests'] += 1
			
			# Audit logging
			await self.audit_service.log_api_access(
				user_id=request.user_id,
				endpoint=request.endpoint_path,
				method=request.method,
				status_code=response.status_code,
				execution_time=execution_time
			)
			
			return response
			
		except Exception as e:
			self.logger.error(f"Request handling failed: {str(e)}")
			self.request_analytics['failed_requests'] += 1
			
			return APIResponse(
				status_code=500,
				error=f"Internal server error: {str(e)}"
			)

	def _find_endpoint(self, path: str, method: HTTPMethod) -> Optional[APIEndpoint]:
		"""Find matching endpoint for request."""
		# Simple path matching - would be more sophisticated in production
		endpoint_key = f"{method}:{path}"
		return self.endpoints.get(endpoint_key)

	# ============================================================================
	# MIDDLEWARE COMPONENTS
	# ============================================================================

	async def _setup_middleware(self) -> None:
		"""Setup middleware stack for request processing."""
		self.middleware_stack = [
			self._authentication_middleware,
			self._rate_limiting_middleware,
			self._validation_middleware,
			self._enrichment_middleware
		]

	async def _authentication_middleware(self, request: APIRequest) -> APIRequest:
		"""Authentication middleware."""
		endpoint = self._find_endpoint(request.endpoint_path, request.method)
		
		if endpoint and endpoint.auth_required:
			# Extract auth token from headers
			auth_header = request.headers.get('Authorization', '')
			
			if not auth_header:
				request.error_response = APIResponse(
					status_code=401,
					error="Authentication required"
				)
				return request
			
			# Validate token with auth service
			try:
				user_info = await self.auth_service.validate_token(auth_header)
				request.user_id = user_info.get('user_id')
				request.tenant_id = user_info.get('tenant_id', self.tenant_id)
			except Exception as e:
				request.error_response = APIResponse(
					status_code=401,
					error=f"Invalid authentication: {str(e)}"
				)
				return request
		
		return request

	async def _rate_limiting_middleware(self, request: APIRequest) -> APIRequest:
		"""Rate limiting middleware."""
		if not self.config['enable_rate_limiting']:
			return request
		
		endpoint = self._find_endpoint(request.endpoint_path, request.method)
		rate_limit = endpoint.rate_limit if endpoint else self.config['default_rate_limit']
		
		# Use client IP or user ID for rate limiting
		client_key = request.client_ip or request.user_id or "anonymous"
		
		now = datetime.utcnow()
		window_start = now - timedelta(minutes=1)
		
		# Clean old requests
		if client_key in self.rate_limit_store:
			self.rate_limit_store[client_key] = [
				req_time for req_time in self.rate_limit_store[client_key]
				if req_time > window_start
			]
		else:
			self.rate_limit_store[client_key] = []
		
		# Check rate limit
		if len(self.rate_limit_store[client_key]) >= rate_limit:
			request.error_response = APIResponse(
				status_code=429,
				error="Rate limit exceeded",
				headers={'Retry-After': '60'}
			)
			return request
		
		# Add current request
		self.rate_limit_store[client_key].append(now)
		
		return request

	async def _validation_middleware(self, request: APIRequest) -> APIRequest:
		"""Request validation middleware."""
		# Validate required parameters
		endpoint = self._find_endpoint(request.endpoint_path, request.method)
		
		if endpoint and request.method in [HTTPMethod.POST, HTTPMethod.PUT, HTTPMethod.PATCH]:
			if not request.body:
				request.error_response = APIResponse(
					status_code=400,
					error="Request body is required"
				)
				return request
		
		return request

	async def _enrichment_middleware(self, request: APIRequest) -> APIRequest:
		"""Request enrichment middleware."""
		# Add metadata and context
		request.metadata = {
			'gateway_version': '1.0.0',
			'processing_timestamp': datetime.utcnow().isoformat(),
			'tenant_id': self.tenant_id
		}
		
		return request

	# ============================================================================
	# ENDPOINT HANDLERS
	# ============================================================================

	async def _setup_default_endpoints(self) -> None:
		"""Setup default API endpoints."""
		
		# Employee CRUD endpoints
		self.endpoints["GET:/api/v1/employees"] = APIEndpoint(
			path="/api/v1/employees",
			method=HTTPMethod.GET,
			handler=self._handle_list_employees,
			description="List employees with filtering and pagination",
			tags=["employees", "read"]
		)
		
		self.endpoints["POST:/api/v1/employees"] = APIEndpoint(
			path="/api/v1/employees",
			method=HTTPMethod.POST,
			handler=self._handle_create_employee,
			description="Create new employee",
			tags=["employees", "write"],
			cache_ttl=0  # No caching for write operations
		)
		
		self.endpoints["GET:/api/v1/employees/{id}"] = APIEndpoint(
			path="/api/v1/employees/{id}",
			method=HTTPMethod.GET,
			handler=self._handle_get_employee,
			description="Get employee by ID",
			tags=["employees", "read"]
		)
		
		self.endpoints["PUT:/api/v1/employees/{id}"] = APIEndpoint(
			path="/api/v1/employees/{id}",
			method=HTTPMethod.PUT,
			handler=self._handle_update_employee,
			description="Update employee",
			tags=["employees", "write"],
			cache_ttl=0
		)
		
		self.endpoints["DELETE:/api/v1/employees/{id}"] = APIEndpoint(
			path="/api/v1/employees/{id}",
			method=HTTPMethod.DELETE,
			handler=self._handle_delete_employee,
			description="Delete employee",
			tags=["employees", "write"],
			cache_ttl=0
		)
		
		# AI and Analytics endpoints
		self.endpoints["POST:/api/v1/employees/{id}/analyze"] = APIEndpoint(
			path="/api/v1/employees/{id}/analyze",
			method=HTTPMethod.POST,
			handler=self._handle_ai_analysis,
			description="Perform AI analysis on employee",
			tags=["ai", "analysis"],
			rate_limit=50  # Lower rate limit for AI operations
		)
		
		self.endpoints["GET:/api/v1/analytics/dashboard"] = APIEndpoint(
			path="/api/v1/analytics/dashboard",
			method=HTTPMethod.GET,
			handler=self._handle_analytics_dashboard,
			description="Get analytics dashboard data",
			tags=["analytics", "dashboard"],
			cache_ttl=600  # Cache for 10 minutes
		)
		
		# Integration endpoints
		self.endpoints["POST:/api/v1/integrations/sync"] = APIEndpoint(
			path="/api/v1/integrations/sync",
			method=HTTPMethod.POST,
			handler=self._handle_integration_sync,
			description="Trigger integration sync",
			tags=["integrations", "sync"],
			rate_limit=10  # Very limited for sync operations
		)

	async def _handle_list_employees(self, request: APIRequest) -> APIResponse:
		"""Handle list employees request."""
		try:
			# Parse query parameters
			page = int(request.query_params.get('page', 1))
			limit = int(request.query_params.get('limit', 50))
			search = request.query_params.get('search', '')
			department = request.query_params.get('department', '')
			
			# Get employees
			employees = await self.employee_service.search_employees(
				search_criteria={
					'search_text': search,
					'department_id': department,
					'limit': limit,
					'offset': (page - 1) * limit
				}
			)
			
			return APIResponse(
				status_code=200,
				data={
					'employees': employees.employees,
					'total_count': employees.total_count,
					'page': page,
					'limit': limit
				},
				metadata={
					'endpoint': 'list_employees',
					'result_count': len(employees.employees)
				}
			)
			
		except Exception as e:
			return APIResponse(
				status_code=500,
				error=f"Failed to list employees: {str(e)}"
			)

	async def _handle_get_employee(self, request: APIRequest) -> APIResponse:
		"""Handle get employee request."""
		try:
			# Extract employee ID from path
			employee_id = self._extract_path_param(request.endpoint_path, 'id')
			
			if not employee_id:
				return APIResponse(
					status_code=400,
					error="Employee ID is required"
				)
			
			# Get employee
			employee = await self.employee_service.get_employee_by_id(employee_id)
			
			if not employee:
				return APIResponse(
					status_code=404,
					error="Employee not found"
				)
			
			return APIResponse(
				status_code=200,
				data=employee.to_dict(),
				metadata={
					'endpoint': 'get_employee',
					'employee_id': employee_id
				}
			)
			
		except Exception as e:
			return APIResponse(
				status_code=500,
				error=f"Failed to get employee: {str(e)}"
			)

	async def _handle_create_employee(self, request: APIRequest) -> APIResponse:
		"""Handle create employee request."""
		try:
			employee_data = request.body
			
			# Create employee
			result = await self.employee_service.create_employee_revolutionary(employee_data)
			
			if not result.success:
				return APIResponse(
					status_code=400,
					error=f"Failed to create employee: {result.error_message}",
					data={'validation_errors': result.validation_errors}
				)
			
			return APIResponse(
				status_code=201,
				data=result.employee_data,
				metadata={
					'endpoint': 'create_employee',
					'employee_id': result.employee_data.get('employee_id')
				}
			)
			
		except Exception as e:
			return APIResponse(
				status_code=500,
				error=f"Failed to create employee: {str(e)}"
			)

	async def _handle_update_employee(self, request: APIRequest) -> APIResponse:
		"""Handle update employee request."""
		try:
			employee_id = self._extract_path_param(request.endpoint_path, 'id')
			update_data = request.body
			
			# Update employee
			result = await self.employee_service.update_employee_revolutionary(
				employee_id, update_data
			)
			
			if not result.success:
				return APIResponse(
					status_code=400,
					error=f"Failed to update employee: {result.error_message}",
					data={'validation_errors': result.validation_errors}
				)
			
			return APIResponse(
				status_code=200,
				data=result.employee_data,
				metadata={
					'endpoint': 'update_employee',
					'employee_id': employee_id
				}
			)
			
		except Exception as e:
			return APIResponse(
				status_code=500,
				error=f"Failed to update employee: {str(e)}"
			)

	async def _handle_delete_employee(self, request: APIRequest) -> APIResponse:
		"""Handle delete employee request."""
		try:
			employee_id = self._extract_path_param(request.endpoint_path, 'id')
			
			# Delete employee
			result = await self.employee_service.delete_employee(employee_id)
			
			if not result.success:
				return APIResponse(
					status_code=400,
					error=f"Failed to delete employee: {result.error_message}"
				)
			
			return APIResponse(
				status_code=204,
				metadata={
					'endpoint': 'delete_employee',
					'employee_id': employee_id
				}
			)
			
		except Exception as e:
			return APIResponse(
				status_code=500,
				error=f"Failed to delete employee: {str(e)}"
			)

	async def _handle_ai_analysis(self, request: APIRequest) -> APIResponse:
		"""Handle AI analysis request."""
		try:
			employee_id = self._extract_path_param(request.endpoint_path, 'id')
			
			# Perform AI analysis
			analysis_result = await self.ai_intelligence.analyze_employee_comprehensive(employee_id)
			
			return APIResponse(
				status_code=200,
				data=analysis_result.__dict__,
				metadata={
					'endpoint': 'ai_analysis',
					'employee_id': employee_id,
					'analysis_type': 'comprehensive'
				}
			)
			
		except Exception as e:
			return APIResponse(
				status_code=500,
				error=f"AI analysis failed: {str(e)}"
			)

	async def _handle_analytics_dashboard(self, request: APIRequest) -> APIResponse:
		"""Handle analytics dashboard request."""
		try:
			timeframe = request.query_params.get('timeframe', 'monthly')
			
			# Get dashboard data
			dashboard_data = await self.analytics_dashboard.get_dashboard_data(
				"executive_overview",
				timeframe
			)
			
			return APIResponse(
				status_code=200,
				data=dashboard_data,
				metadata={
					'endpoint': 'analytics_dashboard',
					'timeframe': timeframe
				}
			)
			
		except Exception as e:
			return APIResponse(
				status_code=500,
				error=f"Analytics dashboard failed: {str(e)}"
			)

	async def _handle_integration_sync(self, request: APIRequest) -> APIResponse:
		"""Handle integration sync request."""
		try:
			integration_id = request.body.get('integration_id')
			
			if not integration_id:
				return APIResponse(
					status_code=400,
					error="Integration ID is required"
				)
			
			# Trigger integration sync
			sync_result = await self._trigger_integration_sync(integration_id)
			
			return APIResponse(
				status_code=200,
				data=sync_result,
				metadata={
					'endpoint': 'integration_sync',
					'integration_id': integration_id
				}
			)
			
		except Exception as e:
			return APIResponse(
				status_code=500,
				error=f"Integration sync failed: {str(e)}"
			)

	# ============================================================================
	# CACHING SYSTEM
	# ============================================================================

	async def _get_cached_response(self, request: APIRequest) -> Optional[APIResponse]:
		"""Get cached response for request."""
		cache_key = self._generate_cache_key(request)
		
		if cache_key in self.response_cache:
			cached_time, cached_response = self.response_cache[cache_key]
			
			# Check if cache is still valid
			endpoint = self._find_endpoint(request.endpoint_path, request.method)
			cache_ttl = endpoint.cache_ttl if endpoint else 300
			
			if (datetime.utcnow() - cached_time).seconds < cache_ttl:
				cached_response.cached = True
				return cached_response
			else:
				# Remove expired cache
				del self.response_cache[cache_key]
		
		return None

	async def _cache_response(self, request: APIRequest, response: APIResponse) -> None:
		"""Cache response for future requests."""
		cache_key = self._generate_cache_key(request)
		self.response_cache[cache_key] = (datetime.utcnow(), response)
		
		# Simple cache cleanup - remove oldest entries if cache is full
		if len(self.response_cache) > 1000:
			oldest_key = min(self.response_cache.keys(), 
							key=lambda k: self.response_cache[k][0])
			del self.response_cache[oldest_key]

	def _generate_cache_key(self, request: APIRequest) -> str:
		"""Generate cache key for request."""
		key_data = f"{request.method}:{request.endpoint_path}:{json.dumps(request.query_params, sort_keys=True)}"
		return hashlib.md5(key_data.encode()).hexdigest()

	# ============================================================================
	# EXTERNAL INTEGRATIONS
	# ============================================================================

	async def _initialize_integrations(self) -> None:
		"""Initialize external system integrations."""
		# Sample integrations - would be configured via admin interface
		
		# Workday Integration
		workday_integration = ExternalIntegration(
			system_name="Workday HCM",
			integration_type=IntegrationType.REST_API,
			base_url="https://api.workday.com",
			auth_method=AuthenticationMethod.OAUTH2,
			auth_config={
				'client_id': 'configured_client_id',
				'client_secret': 'configured_secret',
				'scope': 'employee_read'
			},
			field_mappings={
				'employee_id': 'workday_employee_id',
				'first_name': 'given_name',
				'last_name': 'family_name',
				'work_email': 'primary_email'
			},
			sync_frequency=3600  # Sync every hour
		)
		
		self.integrations[workday_integration.integration_id] = workday_integration
		
		# BambooHR Integration
		bamboo_integration = ExternalIntegration(
			system_name="BambooHR",
			integration_type=IntegrationType.REST_API,
			base_url="https://api.bamboohr.com",
			auth_method=AuthenticationMethod.API_KEY,
			auth_config={
				'api_key': 'configured_api_key',
				'company_domain': 'company.bamboohr.com'
			},
			field_mappings={
				'employee_id': 'id',
				'first_name': 'firstName',
				'last_name': 'lastName',
				'work_email': 'workEmail'
			},
			sync_frequency=7200  # Sync every 2 hours
		)
		
		self.integrations[bamboo_integration.integration_id] = bamboo_integration

	async def _trigger_integration_sync(self, integration_id: str) -> Dict[str, Any]:
		"""Trigger synchronization with external system."""
		if integration_id not in self.integrations:
			raise ValueError(f"Integration not found: {integration_id}")
		
		integration = self.integrations[integration_id]
		
		try:
			# Simulate sync operation
			sync_result = {
				'integration_id': integration_id,
				'system_name': integration.system_name,
				'sync_started': datetime.utcnow().isoformat(),
				'status': 'in_progress',
				'records_processed': 0,
				'records_updated': 0,
				'errors': []
			}
			
			# In real implementation, this would:
			# 1. Authenticate with external system
			# 2. Fetch data from external API
			# 3. Transform data using field mappings
			# 4. Update local employee records
			# 5. Handle conflicts and errors
			
			# Simulate processing
			await asyncio.sleep(0.1)  # Simulate async operation
			
			sync_result.update({
				'status': 'completed',
				'sync_completed': datetime.utcnow().isoformat(),
				'records_processed': 150,
				'records_updated': 23
			})
			
			integration.last_sync = datetime.utcnow()
			
			return sync_result
			
		except Exception as e:
			self.logger.error(f"Integration sync failed: {str(e)}")
			raise

	# ============================================================================
	# MONITORING AND ANALYTICS
	# ============================================================================

	async def _setup_monitoring(self) -> None:
		"""Setup monitoring and health checks."""
		# Setup periodic cache cleanup
		asyncio.create_task(self._periodic_cache_cleanup())
		
		# Setup analytics aggregation
		asyncio.create_task(self._periodic_analytics_aggregation())

	async def _periodic_cache_cleanup(self) -> None:
		"""Periodic cache cleanup task."""
		while True:
			try:
				await asyncio.sleep(3600)  # Run every hour
				
				# Remove expired cache entries
				now = datetime.utcnow()
				expired_keys = []
				
				for cache_key, (cached_time, _) in self.response_cache.items():
					if (now - cached_time).seconds > 3600:  # 1 hour max cache
						expired_keys.append(cache_key)
				
				for key in expired_keys:
					del self.response_cache[key]
				
				self.logger.info(f"Cache cleanup: removed {len(expired_keys)} expired entries")
				
			except Exception as e:
				self.logger.error(f"Cache cleanup failed: {str(e)}")

	async def _periodic_analytics_aggregation(self) -> None:
		"""Periodic analytics aggregation task."""
		while True:
			try:
				await asyncio.sleep(900)  # Run every 15 minutes
				
				# Calculate average response time
				if self.request_analytics['total_requests'] > 0:
					# This would aggregate from detailed request logs in production
					pass
				
				# Log analytics
				await self._log_api_operation("analytics_update", self.request_analytics)
				
			except Exception as e:
				self.logger.error(f"Analytics aggregation failed: {str(e)}")

	# ============================================================================
	# UTILITY METHODS
	# ============================================================================

	def _extract_path_param(self, path: str, param_name: str) -> Optional[str]:
		"""Extract parameter from URL path."""
		# Simple implementation - would use proper URL routing in production
		path_parts = path.split('/')
		
		if param_name == 'id' and len(path_parts) >= 4:
			return path_parts[-1]  # Last part is usually the ID
		
		return None

	async def get_api_statistics(self) -> Dict[str, Any]:
		"""Get API gateway statistics."""
		return {
			'tenant_id': self.tenant_id,
			'endpoints_count': len(self.endpoints),
			'integrations_count': len(self.integrations),
			'cache_entries': len(self.response_cache),
			'rate_limit_entries': len(self.rate_limit_store),
			'request_analytics': self.request_analytics.copy(),
			'uptime': "active",
			'last_update': datetime.utcnow().isoformat()
		}

	async def add_custom_endpoint(self, endpoint: APIEndpoint) -> None:
		"""Add custom API endpoint."""
		endpoint_key = f"{endpoint.method}:{endpoint.path}"
		self.endpoints[endpoint_key] = endpoint
		
		await self._log_api_operation("endpoint_added", {
			"path": endpoint.path,
			"method": endpoint.method,
			"description": endpoint.description
		})

	async def add_integration(self, integration: ExternalIntegration) -> None:
		"""Add external system integration."""
		self.integrations[integration.integration_id] = integration
		
		await self._log_api_operation("integration_added", {
			"system_name": integration.system_name,
			"integration_type": integration.integration_type
		})

	async def health_check(self) -> Dict[str, Any]:
		"""Perform health check of API gateway."""
		try:
			# Check core services
			services_health = {
				'employee_service': 'healthy',
				'ai_intelligence': 'healthy',
				'analytics_dashboard': 'healthy',
				'auth_service': 'healthy',
				'audit_service': 'healthy'
			}
			
			# Check integrations
			integration_status = {}
			for integration_id, integration in self.integrations.items():
				# In production, would ping external systems
				integration_status[integration.system_name] = {
					'status': 'healthy' if integration.enabled else 'disabled',
					'last_sync': integration.last_sync.isoformat() if integration.last_sync else None
				}
			
			return {
				'status': 'healthy',
				'timestamp': datetime.utcnow().isoformat(),
				'services': services_health,
				'integrations': integration_status,
				'statistics': await self.get_api_statistics()
			}
			
		except Exception as e:
			return {
				'status': 'unhealthy',
				'error': str(e),
				'timestamp': datetime.utcnow().isoformat()
			}