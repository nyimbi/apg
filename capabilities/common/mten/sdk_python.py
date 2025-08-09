#!/usr/bin/env python3
"""
MTen Python SDK

Company: Datacraft
Copyright: © 2025
Author: Nyimbi Odero

Comprehensive Python SDK for Multi-Tenant Management (MTen) capability with async support,
type hints, and ergonomic API design following modern Python best practices.
"""

import asyncio
import json
from datetime import datetime, UTC
from typing import Dict, List, Any, Optional, Union, AsyncGenerator, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
from uuid_extensions import uuid7str

__version__ = "1.0.0"
__author__ = "Nyimbi Odero"
__email__ = "nyimbi@gmail.com"

T = TypeVar('T')


class TenantStatus(str, Enum):
	"""Tenant status enumeration"""
	ACTIVE = "active"
	SUSPENDED = "suspended"
	PENDING = "pending"
	ARCHIVED = "archived"


class TenantTier(str, Enum):
	"""Tenant tier enumeration"""
	FREE = "free"
	STANDARD = "standard"
	PREMIUM = "premium"
	ENTERPRISE = "enterprise"


class DeploymentStatus(str, Enum):
	"""Deployment status enumeration"""
	PENDING = "pending"
	IN_PROGRESS = "in_progress"
	COMPLETED = "completed"
	FAILED = "failed"
	ROLLED_BACK = "rolled_back"


@dataclass
class APIResponse(Generic[T]):
	"""Generic API response wrapper"""
	success: bool
	data: Optional[T] = None
	error: Optional[str] = None
	message: Optional[str] = None
	request_id: str = field(default_factory=uuid7str)
	timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class Tenant:
	"""Tenant data model"""
	id: str
	name: str
	display_name: str
	status: TenantStatus
	tier: TenantTier
	created_at: datetime
	updated_at: datetime
	configuration: Dict[str, Any] = field(default_factory=dict)
	metadata: Dict[str, Any] = field(default_factory=dict)
	resource_usage: Dict[str, Any] = field(default_factory=dict)
	
	@classmethod
	def from_dict(cls, data: Dict[str, Any]) -> 'Tenant':
		"""Create Tenant from dictionary"""
		return cls(
			id=data['id'],
			name=data['name'],
			display_name=data['display_name'],
			status=TenantStatus(data['status']),
			tier=TenantTier(data['tier']),
			created_at=datetime.fromisoformat(data['created_at']),
			updated_at=datetime.fromisoformat(data['updated_at']),
			configuration=data.get('configuration', {}),
			metadata=data.get('metadata', {}),
			resource_usage=data.get('resource_usage', {})
		)


@dataclass
class TenantTemplate:
	"""Tenant template data model"""
	id: str
	name: str
	display_name: str
	description: str
	category: str
	version: str
	configuration: Dict[str, Any]
	resource_requirements: Dict[str, Any]
	created_at: datetime
	is_public: bool = True
	tags: List[str] = field(default_factory=list)
	
	@classmethod
	def from_dict(cls, data: Dict[str, Any]) -> 'TenantTemplate':
		"""Create TenantTemplate from dictionary"""
		return cls(
			id=data['id'],
			name=data['name'],
			display_name=data['display_name'],
			description=data['description'],
			category=data['category'],
			version=data['version'],
			configuration=data['configuration'],
			resource_requirements=data['resource_requirements'],
			created_at=datetime.fromisoformat(data['created_at']),
			is_public=data.get('is_public', True),
			tags=data.get('tags', [])
		)


@dataclass
class DeploymentResult:
	"""Deployment result data model"""
	id: str
	tenant_id: str
	status: DeploymentStatus
	strategy: str
	version: str
	started_at: datetime
	completed_at: Optional[datetime] = None
	logs: List[str] = field(default_factory=list)
	rollback_available: bool = False
	
	@classmethod
	def from_dict(cls, data: Dict[str, Any]) -> 'DeploymentResult':
		"""Create DeploymentResult from dictionary"""
		return cls(
			id=data['id'],
			tenant_id=data['tenant_id'],
			status=DeploymentStatus(data['status']),
			strategy=data['strategy'],
			version=data['version'],
			started_at=datetime.fromisoformat(data['started_at']),
			completed_at=datetime.fromisoformat(data['completed_at']) if data.get('completed_at') else None,
			logs=data.get('logs', []),
			rollback_available=data.get('rollback_available', False)
		)


@dataclass
class AnalyticsMetrics:
	"""Analytics metrics data model"""
	tenant_id: str
	timestamp: datetime
	cpu_usage_percent: float
	memory_usage_mb: float
	storage_usage_gb: float
	request_count: int
	error_rate: float
	response_time_ms: float
	active_users: int
	
	@classmethod
	def from_dict(cls, data: Dict[str, Any]) -> 'AnalyticsMetrics':
		"""Create AnalyticsMetrics from dictionary"""
		return cls(
			tenant_id=data['tenant_id'],
			timestamp=datetime.fromisoformat(data['timestamp']),
			cpu_usage_percent=data['cpu_usage_percent'],
			memory_usage_mb=data['memory_usage_mb'],
			storage_usage_gb=data['storage_usage_gb'],
			request_count=data['request_count'],
			error_rate=data['error_rate'],
			response_time_ms=data['response_time_ms'],
			active_users=data['active_users']
		)


class MTenSDKError(Exception):
	"""Base exception for MTen SDK"""
	def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[Dict] = None):
		super().__init__(message)
		self.status_code = status_code
		self.response_data = response_data


class AuthenticationError(MTenSDKError):
	"""Authentication related errors"""
	pass


class ValidationError(MTenSDKError):
	"""Data validation errors"""
	pass


class NetworkError(MTenSDKError):
	"""Network related errors"""
	pass


class MTenClient:
	"""
	Asynchronous MTen API client with comprehensive functionality.
	
	Provides high-level interface for all Multi-Tenant Management operations
	including tenant lifecycle, templates, deployments, and analytics.
	
	Examples:
		Basic usage:
		```python
		async with MTenClient("https://api.example.com", api_key="key") as client:
			tenants = await client.list_tenants()
			tenant = await client.create_tenant("my-app", TenantTier.PREMIUM)
		```
		
		With custom configuration:
		```python
		client = MTenClient(
			base_url="https://api.example.com",
			api_key="key",
			timeout=30,
			retry_attempts=3
		)
		await client.initialize()
		```
	"""
	
	def __init__(
		self,
		base_url: str,
		api_key: str,
		timeout: int = 30,
		retry_attempts: int = 3,
		retry_delay: float = 1.0,
		verify_ssl: bool = True,
		user_agent: str = f"MTen-Python-SDK/{__version__}"
	):
		"""
		Initialize MTen client.
		
		Args:
			base_url: Base URL for the MTen API
			api_key: API key for authentication
			timeout: Request timeout in seconds
			retry_attempts: Number of retry attempts for failed requests
			retry_delay: Delay between retry attempts in seconds
			verify_ssl: Whether to verify SSL certificates
			user_agent: User agent string for requests
		"""
		self.base_url = base_url.rstrip('/')
		self.api_key = api_key
		self.timeout = timeout
		self.retry_attempts = retry_attempts
		self.retry_delay = retry_delay
		self.verify_ssl = verify_ssl
		self.user_agent = user_agent
		
		self._session: Optional[aiohttp.ClientSession] = None
		self._initialized = False
	
	async def __aenter__(self) -> 'MTenClient':
		"""Async context manager entry"""
		await self.initialize()
		return self
	
	async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
		"""Async context manager exit"""
		await self.close()
	
	async def initialize(self) -> None:
		"""Initialize the HTTP session"""
		if self._session is None:
			connector = aiohttp.TCPConnector(verify_ssl=self.verify_ssl)
			timeout = aiohttp.ClientTimeout(total=self.timeout)
			
			self._session = aiohttp.ClientSession(
				connector=connector,
				timeout=timeout,
				headers={
					'Authorization': f'Bearer {self.api_key}',
					'User-Agent': self.user_agent,
					'Content-Type': 'application/json',
					'Accept': 'application/json'
				}
			)
			self._initialized = True
	
	async def close(self) -> None:
		"""Close the HTTP session"""
		if self._session:
			await self._session.close()
			self._session = None
			self._initialized = False
	
	async def _request(
		self,
		method: str,
		endpoint: str,
		data: Optional[Dict[str, Any]] = None,
		params: Optional[Dict[str, Any]] = None
	) -> Dict[str, Any]:
		"""Make HTTP request with retry logic"""
		if not self._initialized:
			await self.initialize()
		
		url = f"{self.base_url}/api/v1{endpoint}"
		
		for attempt in range(self.retry_attempts + 1):
			try:
				async with self._session.request(
					method,
					url,
					json=data,
					params=params
				) as response:
					response_data = await response.json() if response.content_type == 'application/json' else {}
					
					if response.status >= 400:
						if response.status == 401:
							raise AuthenticationError(
								"Invalid API key or authentication failed",
								response.status,
								response_data
							)
						elif response.status == 422:
							raise ValidationError(
								response_data.get('message', 'Validation error'),
								response.status,
								response_data
							)
						else:
							raise MTenSDKError(
								response_data.get('message', f'HTTP {response.status}'),
								response.status,
								response_data
							)
					
					return response_data
					
			except (aiohttp.ClientError, asyncio.TimeoutError) as e:
				if attempt == self.retry_attempts:
					raise NetworkError(f"Network error after {self.retry_attempts} retries: {str(e)}")
				
				await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
		
		raise MTenSDKError("Unexpected error in request processing")
	
	# Tenant Management Methods
	
	async def list_tenants(
		self,
		status: Optional[TenantStatus] = None,
		tier: Optional[TenantTier] = None,
		limit: int = 100,
		offset: int = 0
	) -> APIResponse[List[Tenant]]:
		"""
		List tenants with optional filtering.
		
		Args:
			status: Filter by tenant status
			tier: Filter by tenant tier  
			limit: Maximum number of results (1-1000)
			offset: Number of results to skip
			
		Returns:
			API response containing list of tenants
		"""
		params = {'limit': limit, 'offset': offset}
		if status:
			params['status'] = status.value
		if tier:
			params['tier'] = tier.value
		
		response = await self._request('GET', '/tenants', params=params)
		
		tenants = [Tenant.from_dict(t) for t in response.get('data', [])]
		return APIResponse(success=True, data=tenants)
	
	async def get_tenant(self, tenant_id: str) -> APIResponse[Tenant]:
		"""
		Get tenant by ID.
		
		Args:
			tenant_id: Unique tenant identifier
			
		Returns:
			API response containing tenant data
		"""
		response = await self._request('GET', f'/tenants/{tenant_id}')
		
		tenant = Tenant.from_dict(response['data'])
		return APIResponse(success=True, data=tenant)
	
	async def create_tenant(
		self,
		name: str,
		tier: TenantTier,
		display_name: Optional[str] = None,
		template_id: Optional[str] = None,
		configuration: Optional[Dict[str, Any]] = None,
		metadata: Optional[Dict[str, Any]] = None
	) -> APIResponse[Tenant]:
		"""
		Create new tenant.
		
		Args:
			name: Unique tenant name
			tier: Tenant tier level
			display_name: Human-readable display name
			template_id: Optional template to use for configuration
			configuration: Tenant-specific configuration
			metadata: Additional metadata
			
		Returns:
			API response containing created tenant
		"""
		data = {
			'name': name,
			'tier': tier.value,
			'display_name': display_name or name,
			'configuration': configuration or {},
			'metadata': metadata or {}
		}
		
		if template_id:
			data['template_id'] = template_id
		
		response = await self._request('POST', '/tenants', data=data)
		
		tenant = Tenant.from_dict(response['data'])
		return APIResponse(success=True, data=tenant, message="Tenant created successfully")
	
	async def update_tenant(
		self,
		tenant_id: str,
		display_name: Optional[str] = None,
		tier: Optional[TenantTier] = None,
		configuration: Optional[Dict[str, Any]] = None,
		metadata: Optional[Dict[str, Any]] = None
	) -> APIResponse[Tenant]:
		"""
		Update existing tenant.
		
		Args:
			tenant_id: Tenant to update
			display_name: New display name
			tier: New tier level
			configuration: Configuration updates
			metadata: Metadata updates
			
		Returns:
			API response containing updated tenant
		"""
		data = {}
		if display_name is not None:
			data['display_name'] = display_name
		if tier is not None:
			data['tier'] = tier.value
		if configuration is not None:
			data['configuration'] = configuration
		if metadata is not None:
			data['metadata'] = metadata
		
		response = await self._request('PATCH', f'/tenants/{tenant_id}', data=data)
		
		tenant = Tenant.from_dict(response['data'])
		return APIResponse(success=True, data=tenant, message="Tenant updated successfully")
	
	async def delete_tenant(self, tenant_id: str, force: bool = False) -> APIResponse[bool]:
		"""
		Delete tenant.
		
		Args:
			tenant_id: Tenant to delete
			force: Force deletion even if tenant has active resources
			
		Returns:
			API response indicating success
		"""
		params = {'force': 'true'} if force else {}
		
		await self._request('DELETE', f'/tenants/{tenant_id}', params=params)
		
		return APIResponse(success=True, data=True, message="Tenant deleted successfully")
	
	# Template Management Methods
	
	async def list_templates(
		self,
		category: Optional[str] = None,
		public_only: bool = True,
		limit: int = 50
	) -> APIResponse[List[TenantTemplate]]:
		"""
		List available tenant templates.
		
		Args:
			category: Filter by template category
			public_only: Only return public templates
			limit: Maximum number of results
			
		Returns:
			API response containing list of templates
		"""
		params = {'limit': limit, 'public_only': public_only}
		if category:
			params['category'] = category
		
		response = await self._request('GET', '/templates', params=params)
		
		templates = [TenantTemplate.from_dict(t) for t in response.get('data', [])]
		return APIResponse(success=True, data=templates)
	
	async def get_template(self, template_id: str) -> APIResponse[TenantTemplate]:
		"""
		Get template by ID.
		
		Args:
			template_id: Template identifier
			
		Returns:
			API response containing template data
		"""
		response = await self._request('GET', f'/templates/{template_id}')
		
		template = TenantTemplate.from_dict(response['data'])
		return APIResponse(success=True, data=template)
	
	async def create_template(
		self,
		name: str,
		display_name: str,
		description: str,
		category: str,
		configuration: Dict[str, Any],
		resource_requirements: Optional[Dict[str, Any]] = None,
		tags: Optional[List[str]] = None,
		is_public: bool = False
	) -> APIResponse[TenantTemplate]:
		"""
		Create new tenant template.
		
		Args:
			name: Unique template name
			display_name: Human-readable name
			description: Template description
			category: Template category
			configuration: Template configuration
			resource_requirements: Resource requirements
			tags: Template tags for discovery
			is_public: Whether template is publicly available
			
		Returns:
			API response containing created template
		"""
		data = {
			'name': name,
			'display_name': display_name,
			'description': description,
			'category': category,
			'configuration': configuration,
			'resource_requirements': resource_requirements or {},
			'tags': tags or [],
			'is_public': is_public
		}
		
		response = await self._request('POST', '/templates', data=data)
		
		template = TenantTemplate.from_dict(response['data'])
		return APIResponse(success=True, data=template, message="Template created successfully")
	
	# Deployment Methods
	
	async def deploy_tenant(
		self,
		tenant_id: str,
		version: Optional[str] = None,
		strategy: str = "rolling"
	) -> APIResponse[DeploymentResult]:
		"""
		Deploy tenant with specified strategy.
		
		Args:
			tenant_id: Tenant to deploy
			version: Version to deploy (defaults to latest)
			strategy: Deployment strategy (rolling, blue_green, canary)
			
		Returns:
			API response containing deployment result
		"""
		data = {
			'tenant_id': tenant_id,
			'strategy': strategy
		}
		
		if version:
			data['version'] = version
		
		response = await self._request('POST', '/deployments', data=data)
		
		deployment = DeploymentResult.from_dict(response['data'])
		return APIResponse(success=True, data=deployment, message="Deployment initiated")
	
	async def get_deployment_status(self, deployment_id: str) -> APIResponse[DeploymentResult]:
		"""
		Get deployment status.
		
		Args:
			deployment_id: Deployment identifier
			
		Returns:
			API response containing deployment status
		"""
		response = await self._request('GET', f'/deployments/{deployment_id}')
		
		deployment = DeploymentResult.from_dict(response['data'])
		return APIResponse(success=True, data=deployment)
	
	async def rollback_deployment(
		self,
		deployment_id: str,
		target_version: Optional[str] = None
	) -> APIResponse[DeploymentResult]:
		"""
		Rollback deployment to previous version.
		
		Args:
			deployment_id: Deployment to rollback
			target_version: Specific version to rollback to
			
		Returns:
			API response containing rollback result
		"""
		data = {}
		if target_version:
			data['target_version'] = target_version
		
		response = await self._request('POST', f'/deployments/{deployment_id}/rollback', data=data)
		
		deployment = DeploymentResult.from_dict(response['data'])
		return APIResponse(success=True, data=deployment, message="Rollback initiated")
	
	# Analytics Methods
	
	async def get_tenant_metrics(
		self,
		tenant_id: str,
		start_time: Optional[datetime] = None,
		end_time: Optional[datetime] = None,
		interval: str = "1h"
	) -> APIResponse[List[AnalyticsMetrics]]:
		"""
		Get tenant analytics metrics.
		
		Args:
			tenant_id: Tenant to get metrics for
			start_time: Start of time range
			end_time: End of time range
			interval: Aggregation interval (1m, 5m, 1h, 1d)
			
		Returns:
			API response containing metrics data
		"""
		params = {'interval': interval}
		
		if start_time:
			params['start_time'] = start_time.isoformat()
		if end_time:
			params['end_time'] = end_time.isoformat()
		
		response = await self._request(
			'GET', 
			f'/tenants/{tenant_id}/metrics',
			params=params
		)
		
		metrics = [AnalyticsMetrics.from_dict(m) for m in response.get('data', [])]
		return APIResponse(success=True, data=metrics)
	
	async def get_tenant_health_score(self, tenant_id: str) -> APIResponse[float]:
		"""
		Get tenant health score.
		
		Args:
			tenant_id: Tenant identifier
			
		Returns:
			API response containing health score (0.0-1.0)
		"""
		response = await self._request('GET', f'/tenants/{tenant_id}/health')
		
		health_score = response['data']['health_score']
		return APIResponse(success=True, data=health_score)
	
	# Stream Methods for Real-time Updates
	
	async def stream_tenant_events(
		self,
		tenant_ids: Optional[List[str]] = None
	) -> AsyncGenerator[Dict[str, Any], None]:
		"""
		Stream real-time tenant events.
		
		Args:
			tenant_ids: Specific tenants to monitor (all if None)
			
		Yields:
			Tenant event data
		"""
		params = {}
		if tenant_ids:
			params['tenant_ids'] = ','.join(tenant_ids)
		
		url = f"{self.base_url}/api/v1/tenants/stream"
		
		async with self._session.get(url, params=params) as response:
			if response.status != 200:
				raise MTenSDKError(f"Stream connection failed: {response.status}")
			
			async for line in response.content:
				if line:
					try:
						event_data = json.loads(line.decode('utf-8'))
						yield event_data
					except json.JSONDecodeError:
						continue  # Skip malformed lines
	
	async def stream_deployment_logs(
		self,
		deployment_id: str
	) -> AsyncGenerator[str, None]:
		"""
		Stream real-time deployment logs.
		
		Args:
			deployment_id: Deployment to stream logs for
			
		Yields:
			Log lines
		"""
		url = f"{self.base_url}/api/v1/deployments/{deployment_id}/logs/stream"
		
		async with self._session.get(url) as response:
			if response.status != 200:
				raise MTenSDKError(f"Log stream connection failed: {response.status}")
			
			async for line in response.content:
				if line:
					yield line.decode('utf-8').strip()
	
	# Utility Methods
	
	async def ping(self) -> APIResponse[Dict[str, Any]]:
		"""
		Ping the API to check connectivity and authentication.
		
		Returns:
			API response with server information
		"""
		response = await self._request('GET', '/ping')
		
		return APIResponse(success=True, data=response['data'])
	
	async def get_api_info(self) -> APIResponse[Dict[str, Any]]:
		"""
		Get API version and capability information.
		
		Returns:
			API response with API information
		"""
		response = await self._request('GET', '/info')
		
		return APIResponse(success=True, data=response['data'])


# High-level convenience functions

async def create_mten_client(
	base_url: str,
	api_key: str,
	**kwargs
) -> MTenClient:
	"""
	Create and initialize MTen client.
	
	Args:
		base_url: MTen API base URL
		api_key: API authentication key
		**kwargs: Additional client configuration
		
	Returns:
		Initialized MTen client
	"""
	client = MTenClient(base_url, api_key, **kwargs)
	await client.initialize()
	return client


async def quick_tenant_setup(
	client: MTenClient,
	name: str,
	tier: TenantTier,
	template_name: Optional[str] = None
) -> Tenant:
	"""
	Quick tenant setup with sensible defaults.
	
	Args:
		client: Initialized MTen client
		name: Tenant name
		tier: Tenant tier
		template_name: Optional template name
		
	Returns:
		Created tenant
	"""
	template_id = None
	
	if template_name:
		templates_response = await client.list_templates()
		if templates_response.success:
			template = next(
				(t for t in templates_response.data if t.name == template_name),
				None
			)
			if template:
				template_id = template.id
	
	tenant_response = await client.create_tenant(
		name=name,
		tier=tier,
		template_id=template_id
	)
	
	if not tenant_response.success:
		raise MTenSDKError(f"Failed to create tenant: {tenant_response.error}")
	
	return tenant_response.data


# Export public API
__all__ = [
	'MTenClient',
	'MTenSDKError',
	'AuthenticationError',
	'ValidationError',
	'NetworkError',
	'TenantStatus',
	'TenantTier',
	'DeploymentStatus',
	'Tenant',
	'TenantTemplate',
	'DeploymentResult',
	'AnalyticsMetrics',
	'APIResponse',
	'create_mten_client',
	'quick_tenant_setup'
]