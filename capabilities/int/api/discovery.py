"""
APG Integration API Management - Service Discovery and Integration

Service discovery, capability registration, and APG platform integration
for seamless inter-capability communication and orchestration.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import urlparse, urljoin

import aiohttp
import aioredis
from pydantic import BaseModel, Field, validator

from .models import AMAPI, AMEndpoint, APIStatus, ProtocolType
from .service import APILifecycleService

# =============================================================================
# Service Discovery Models
# =============================================================================

class ServiceHealth(str, Enum):
	"""Service health status."""
	HEALTHY = "healthy"
	DEGRADED = "degraded"
	UNHEALTHY = "unhealthy"
	UNKNOWN = "unknown"

class CapabilityType(str, Enum):
	"""APG capability types."""
	FOUNDATION = "foundation"
	CORE_BUSINESS = "core_business"
	CROSS_FUNCTIONAL = "cross_functional"
	INDUSTRY_VERTICAL = "industry_vertical"
	EMERGING_TECH = "emerging_tech"

@dataclass
class ServiceEndpoint:
	"""Service endpoint information."""
	url: str
	protocol: str
	health_check_path: str = "/health"
	timeout_ms: int = 5000
	priority: int = 100
	weight: int = 1
	metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DiscoveredService:
	"""Discovered service information."""
	service_id: str
	service_name: str
	capability_id: str
	service_type: str
	version: str
	endpoints: List[ServiceEndpoint]
	health_status: ServiceHealth = ServiceHealth.UNKNOWN
	last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
	metadata: Dict[str, Any] = field(default_factory=dict)
	tags: List[str] = field(default_factory=list)

class APGCapabilityInfo(BaseModel):
	"""APG capability registration information."""
	
	capability_id: str = Field(..., description="Unique capability identifier")
	capability_name: str = Field(..., description="Human-readable capability name")
	capability_type: CapabilityType = Field(..., description="Capability category")
	version: str = Field(..., description="Capability version")
	description: Optional[str] = Field(None, description="Capability description")
	
	# Service endpoints
	base_url: str = Field(..., description="Base URL for the capability")
	api_endpoints: Dict[str, str] = Field(default_factory=dict, description="API endpoint paths")
	health_endpoint: str = Field("/health", description="Health check endpoint")
	metrics_endpoint: Optional[str] = Field("/metrics", description="Metrics endpoint")
	
	# API specifications
	openapi_url: Optional[str] = Field(None, description="OpenAPI specification URL")
	graphql_url: Optional[str] = Field(None, description="GraphQL endpoint URL")
	websocket_url: Optional[str] = Field(None, description="WebSocket endpoint URL")
	
	# Dependencies and relationships
	dependencies: List[str] = Field(default_factory=list, description="Required capability dependencies")
	provides: List[str] = Field(default_factory=list, description="Services this capability provides")
	event_patterns: List[str] = Field(default_factory=list, description="Event patterns this capability handles")
	
	# Configuration
	multi_tenant: bool = Field(True, description="Supports multi-tenancy")
	auto_scaling: bool = Field(False, description="Supports auto-scaling")
	load_balancing: bool = Field(True, description="Supports load balancing")
	
	# Metadata
	tags: List[str] = Field(default_factory=list, description="Service tags")
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
	
	@validator('capability_id')
	def validate_capability_id(cls, v):
		if not v or len(v.strip()) == 0:
			raise ValueError('Capability ID cannot be empty')
		return v.strip().lower()

class APIDiscoveryInfo(BaseModel):
	"""API discovery information for external services."""
	
	api_id: str = Field(..., description="API identifier")
	api_name: str = Field(..., description="API name")
	service_name: str = Field(..., description="Service name")
	base_url: str = Field(..., description="API base URL")
	
	# API specification
	protocol: ProtocolType = Field(ProtocolType.REST, description="API protocol")
	openapi_spec_url: Optional[str] = Field(None, description="OpenAPI specification URL")
	version: str = Field("1.0.0", description="API version")
	
	# Discovery metadata
	discovered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	discovery_method: str = Field("manual", description="How the API was discovered")
	
	# Health and status
	health_check_url: Optional[str] = Field(None, description="Health check URL")
	status: str = Field("unknown", description="API status")
	
	# Security
	auth_required: bool = Field(True, description="Authentication required")
	auth_methods: List[str] = Field(default_factory=list, description="Supported auth methods")
	
	# Categorization
	category: Optional[str] = Field(None, description="API category")
	tags: List[str] = Field(default_factory=list, description="API tags")

# =============================================================================
# Service Discovery Engine
# =============================================================================

class ServiceDiscovery:
	"""Service discovery engine for APG capabilities."""
	
	def __init__(self, redis_client: aioredis.Redis, 
				 api_service: APILifecycleService,
				 capability_id: str = "integration_api_management"):
		
		self.redis = redis_client
		self.api_service = api_service
		self.capability_id = capability_id
		
		# Registry storage
		self.service_registry = {}
		self.capability_registry = {}
		self.api_registry = {}
		
		# Discovery configuration
		self.discovery_interval = 30  # seconds
		self.health_check_interval = 60  # seconds
		self.service_ttl = 300  # seconds
		
		# Discovery tasks
		self.discovery_tasks = []
		self.health_check_tasks = []
		
		# Event callbacks
		self.service_added_callbacks = []
		self.service_removed_callbacks = []
		self.service_health_changed_callbacks = []
	
	async def initialize(self):
		"""Initialize service discovery."""
		
		# Load existing registrations from Redis
		await self._load_registry_from_redis()
		
		# Start discovery and health check tasks
		self.discovery_tasks = [
			asyncio.create_task(self._discovery_loop()),
			asyncio.create_task(self._health_check_loop()),
			asyncio.create_task(self._registry_cleanup_loop())
		]
	
	async def shutdown(self):
		"""Shutdown service discovery."""
		
		# Cancel all tasks
		for task in self.discovery_tasks + self.health_check_tasks:
			task.cancel()
		
		await asyncio.gather(
			*self.discovery_tasks, 
			*self.health_check_tasks, 
			return_exceptions=True
		)
	
	# =============================================================================
	# Capability Registration
	# =============================================================================
	
	async def register_capability(self, capability_info: APGCapabilityInfo) -> bool:
		"""Register an APG capability."""
		
		try:
			# Store in local registry
			self.capability_registry[capability_info.capability_id] = capability_info
			
			# Store in Redis
			redis_key = f"apg:capabilities:{capability_info.capability_id}"
			await self.redis.setex(
				redis_key, 
				self.service_ttl, 
				capability_info.json()
			)
			
			# Register API endpoints if present
			await self._register_capability_apis(capability_info)
			
			# Trigger callbacks
			await self._trigger_service_added_callbacks(capability_info.capability_id, capability_info.dict())
			
			return True
			
		except Exception as e:
			print(f"Error registering capability {capability_info.capability_id}: {e}")
			return False
	
	async def unregister_capability(self, capability_id: str) -> bool:
		"""Unregister an APG capability."""
		
		try:
			# Remove from local registry
			capability_info = self.capability_registry.pop(capability_id, None)
			
			# Remove from Redis
			redis_key = f"apg:capabilities:{capability_id}"
			await self.redis.delete(redis_key)
			
			# Unregister API endpoints
			if capability_info:
				await self._unregister_capability_apis(capability_info)
			
			# Trigger callbacks
			await self._trigger_service_removed_callbacks(capability_id)
			
			return True
			
		except Exception as e:
			print(f"Error unregistering capability {capability_id}: {e}")
			return False
	
	async def get_capability(self, capability_id: str) -> Optional[APGCapabilityInfo]:
		"""Get capability information."""
		
		# Try local registry first
		if capability_id in self.capability_registry:
			return self.capability_registry[capability_id]
		
		# Try Redis
		redis_key = f"apg:capabilities:{capability_id}"
		data = await self.redis.get(redis_key)
		
		if data:
			try:
				capability_info = APGCapabilityInfo.parse_raw(data)
				self.capability_registry[capability_id] = capability_info
				return capability_info
			except Exception as e:
				print(f"Error parsing capability data for {capability_id}: {e}")
		
		return None
	
	async def list_capabilities(self, capability_type: Optional[CapabilityType] = None) -> List[APGCapabilityInfo]:
		"""List all registered capabilities."""
		
		capabilities = []
		
		# Get all capabilities from Redis
		pattern = "apg:capabilities:*"
		async for key in self.redis.scan_iter(match=pattern):
			data = await self.redis.get(key)
			if data:
				try:
					capability_info = APGCapabilityInfo.parse_raw(data)
					
					# Filter by type if specified
					if capability_type is None or capability_info.capability_type == capability_type:
						capabilities.append(capability_info)
						
				except Exception as e:
					print(f"Error parsing capability data from {key}: {e}")
		
		return capabilities
	
	# =============================================================================
	# API Discovery
	# =============================================================================
	
	async def discover_api(self, api_discovery_info: APIDiscoveryInfo) -> bool:
		"""Discover and register an external API."""
		
		try:
			# Validate API accessibility
			if not await self._validate_api_accessibility(api_discovery_info):
				return False
			
			# Download API specification if available
			api_spec = None
			if api_discovery_info.openapi_spec_url:
				api_spec = await self._download_api_specification(api_discovery_info.openapi_spec_url)
			
			# Register API in API Management
			api_config = await self._create_api_config_from_discovery(api_discovery_info, api_spec)
			
			api_id = await self.api_service.register_api(
				config=api_config,
				tenant_id="discovered",
				created_by="discovery_engine"
			)
			
			# Store discovery information
			self.api_registry[api_id] = api_discovery_info
			
			# Store in Redis
			redis_key = f"apg:discovered_apis:{api_id}"
			await self.redis.setex(
				redis_key,
				self.service_ttl,
				api_discovery_info.json()
			)
			
			return True
			
		except Exception as e:
			print(f"Error discovering API {api_discovery_info.api_name}: {e}")
			return False
	
	async def auto_discover_apis(self, base_urls: List[str]) -> List[APIDiscoveryInfo]:
		"""Automatically discover APIs from base URLs."""
		
		discovered_apis = []
		
		for base_url in base_urls:
			try:
				# Common API discovery patterns
				discovery_patterns = [
					"/api/v1",
					"/api/v2", 
					"/v1",
					"/v2",
					"/graphql",
					"/api",
					"/.well-known/api"
				]
				
				async with aiohttp.ClientSession() as session:
					for pattern in discovery_patterns:
						test_url = urljoin(base_url, pattern)
						
						try:
							async with session.get(test_url, timeout=aiohttp.ClientTimeout(total=5)) as response:
								if response.status == 200:
									# Try to discover API specification
									api_info = await self._analyze_api_endpoint(test_url, response)
									if api_info:
										discovered_apis.append(api_info)
										
						except Exception:
							continue  # Try next pattern
						
			except Exception as e:
				print(f"Error auto-discovering APIs from {base_url}: {e}")
		
		return discovered_apis
	
	# =============================================================================
	# Service Health Monitoring
	# =============================================================================
	
	async def check_service_health(self, service_id: str) -> ServiceHealth:
		"""Check health of a specific service."""
		
		capability_info = await self.get_capability(service_id)
		if not capability_info:
			return ServiceHealth.UNKNOWN
		
		try:
			health_url = urljoin(capability_info.base_url, capability_info.health_endpoint)
			
			async with aiohttp.ClientSession() as session:
				async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
					if response.status == 200:
						health_data = await response.json()
						
						# Parse health response
						status = health_data.get('status', 'unknown').lower()
						
						if status in ['healthy', 'ok', 'up']:
							return ServiceHealth.HEALTHY
						elif status in ['degraded', 'warning']:
							return ServiceHealth.DEGRADED
						else:
							return ServiceHealth.UNHEALTHY
					else:
						return ServiceHealth.UNHEALTHY
						
		except Exception as e:
			print(f"Health check failed for {service_id}: {e}")
			return ServiceHealth.UNHEALTHY
	
	async def get_service_health_summary(self) -> Dict[str, ServiceHealth]:
		"""Get health summary for all services."""
		
		health_summary = {}
		
		for capability_id in self.capability_registry.keys():
			health_status = await self.check_service_health(capability_id)
			health_summary[capability_id] = health_status
		
		return health_summary
	
	# =============================================================================
	# Service Resolution
	# =============================================================================
	
	async def resolve_service(self, service_name: str) -> Optional[str]:
		"""Resolve service name to service URL."""
		
		# Try exact capability ID match
		capability_info = await self.get_capability(service_name)
		if capability_info:
			return capability_info.base_url
		
		# Try service name search
		for capability_info in self.capability_registry.values():
			if capability_info.capability_name.lower() == service_name.lower():
				return capability_info.base_url
			
			# Check provides list
			if service_name in capability_info.provides:
				return capability_info.base_url
		
		return None
	
	async def resolve_api_endpoint(self, capability_id: str, endpoint_name: str) -> Optional[str]:
		"""Resolve API endpoint URL."""
		
		capability_info = await self.get_capability(capability_id)
		if not capability_info:
			return None
		
		endpoint_path = capability_info.api_endpoints.get(endpoint_name)
		if endpoint_path:
			return urljoin(capability_info.base_url, endpoint_path)
		
		return None
	
	async def find_services_by_tag(self, tag: str) -> List[APGCapabilityInfo]:
		"""Find services by tag."""
		
		matching_services = []
		
		for capability_info in self.capability_registry.values():
			if tag in capability_info.tags:
				matching_services.append(capability_info)
		
		return matching_services
	
	async def find_services_by_dependency(self, dependency: str) -> List[APGCapabilityInfo]:
		"""Find services that depend on a specific capability."""
		
		dependent_services = []
		
		for capability_info in self.capability_registry.values():
			if dependency in capability_info.dependencies:
				dependent_services.append(capability_info)
		
		return dependent_services
	
	# =============================================================================
	# Event Callbacks
	# =============================================================================
	
	def add_service_added_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
		"""Add callback for service addition events."""
		self.service_added_callbacks.append(callback)
	
	def add_service_removed_callback(self, callback: Callable[[str], None]):
		"""Add callback for service removal events."""
		self.service_removed_callbacks.append(callback)
	
	def add_service_health_changed_callback(self, callback: Callable[[str, ServiceHealth], None]):
		"""Add callback for service health change events."""
		self.service_health_changed_callbacks.append(callback)
	
	async def _trigger_service_added_callbacks(self, service_id: str, service_data: Dict[str, Any]):
		"""Trigger service added callbacks."""
		for callback in self.service_added_callbacks:
			try:
				await callback(service_id, service_data)
			except Exception as e:
				print(f"Error in service added callback: {e}")
	
	async def _trigger_service_removed_callbacks(self, service_id: str):
		"""Trigger service removed callbacks."""
		for callback in self.service_removed_callbacks:
			try:
				await callback(service_id)
			except Exception as e:
				print(f"Error in service removed callback: {e}")
	
	async def _trigger_service_health_changed_callbacks(self, service_id: str, health_status: ServiceHealth):
		"""Trigger service health changed callbacks."""
		for callback in self.service_health_changed_callbacks:
			try:
				await callback(service_id, health_status)
			except Exception as e:
				print(f"Error in service health changed callback: {e}")
	
	# =============================================================================
	# Internal Helper Methods
	# =============================================================================
	
	async def _load_registry_from_redis(self):
		"""Load existing registry from Redis."""
		
		try:
			# Load capabilities
			pattern = "apg:capabilities:*"
			async for key in self.redis.scan_iter(match=pattern):
				data = await self.redis.get(key)
				if data:
					try:
						capability_info = APGCapabilityInfo.parse_raw(data)
						self.capability_registry[capability_info.capability_id] = capability_info
					except Exception as e:
						print(f"Error loading capability from {key}: {e}")
			
			# Load discovered APIs
			pattern = "apg:discovered_apis:*"
			async for key in self.redis.scan_iter(match=pattern):
				data = await self.redis.get(key)
				if data:
					try:
						api_info = APIDiscoveryInfo.parse_raw(data)
						self.api_registry[api_info.api_id] = api_info
					except Exception as e:
						print(f"Error loading API from {key}: {e}")
						
		except Exception as e:
			print(f"Error loading registry from Redis: {e}")
	
	async def _discovery_loop(self):
		"""Main discovery loop."""
		
		while True:
			try:
				# Refresh capability registrations
				await self._refresh_capability_registrations()
				
				# Auto-discover new APIs
				await self._auto_discover_new_apis()
				
				await asyncio.sleep(self.discovery_interval)
				
			except asyncio.CancelledError:
				break
			except Exception as e:
				print(f"Error in discovery loop: {e}")
				await asyncio.sleep(self.discovery_interval)
	
	async def _health_check_loop(self):
		"""Health check loop."""
		
		while True:
			try:
				# Check health of all registered capabilities
				for capability_id in list(self.capability_registry.keys()):
					health_status = await self.check_service_health(capability_id)
					
					# Update health status and trigger callbacks if changed
					capability_info = self.capability_registry.get(capability_id)
					if capability_info:
						# Store previous health for comparison
						previous_health = getattr(capability_info, '_last_health', ServiceHealth.UNKNOWN)
						
						if health_status != previous_health:
							capability_info._last_health = health_status
							await self._trigger_service_health_changed_callbacks(capability_id, health_status)
				
				await asyncio.sleep(self.health_check_interval)
				
			except asyncio.CancelledError:
				break
			except Exception as e:
				print(f"Error in health check loop: {e}")
				await asyncio.sleep(self.health_check_interval)
	
	async def _registry_cleanup_loop(self):
		"""Registry cleanup loop to remove stale entries."""
		
		while True:
			try:
				# Cleanup stale capabilities
				current_time = time.time()
				stale_capabilities = []
				
				for capability_id, capability_info in self.capability_registry.items():
					# Check if capability is still alive
					health_status = await self.check_service_health(capability_id)
					
					if health_status == ServiceHealth.UNHEALTHY:
						# Mark for removal if unhealthy for too long
						if not hasattr(capability_info, '_unhealthy_since'):
							capability_info._unhealthy_since = current_time
						elif current_time - capability_info._unhealthy_since > 300:  # 5 minutes
							stale_capabilities.append(capability_id)
					else:
						# Reset unhealthy timer
						if hasattr(capability_info, '_unhealthy_since'):
							delattr(capability_info, '_unhealthy_since')
				
				# Remove stale capabilities
				for capability_id in stale_capabilities:
					await self.unregister_capability(capability_id)
				
				await asyncio.sleep(120)  # Cleanup every 2 minutes
				
			except asyncio.CancelledError:
				break
			except Exception as e:
				print(f"Error in registry cleanup loop: {e}")
				await asyncio.sleep(120)
	
	async def _refresh_capability_registrations(self):
		"""Refresh capability registrations in Redis."""
		
		for capability_id, capability_info in self.capability_registry.items():
			try:
				redis_key = f"apg:capabilities:{capability_id}"
				await self.redis.setex(redis_key, self.service_ttl, capability_info.json())
			except Exception as e:
				print(f"Error refreshing registration for {capability_id}: {e}")
	
	async def _auto_discover_new_apis(self):
		"""Auto-discover new APIs from known capability endpoints."""
		
		for capability_info in self.capability_registry.values():
			try:
				# Check if capability has API discovery endpoint
				discovery_url = urljoin(capability_info.base_url, "/api/discovery")
				
				async with aiohttp.ClientSession() as session:
					try:
						async with session.get(discovery_url, timeout=aiohttp.ClientTimeout(total=5)) as response:
							if response.status == 200:
								discovery_data = await response.json()
								
								# Process discovered APIs
								if isinstance(discovery_data, dict) and 'apis' in discovery_data:
									for api_data in discovery_data['apis']:
										api_info = APIDiscoveryInfo(**api_data)
										await self.discover_api(api_info)
								
					except aiohttp.ClientError:
						continue  # Discovery endpoint not available
						
			except Exception as e:
				print(f"Error in auto-discovery for {capability_info.capability_id}: {e}")
	
	async def _register_capability_apis(self, capability_info: APGCapabilityInfo):
		"""Register API endpoints for a capability."""
		
		try:
			for endpoint_name, endpoint_path in capability_info.api_endpoints.items():
				# Create API discovery info
				api_info = APIDiscoveryInfo(
					api_id=f"{capability_info.capability_id}_{endpoint_name}",
					api_name=f"{capability_info.capability_name} {endpoint_name.title()}",
					service_name=capability_info.capability_name,
					base_url=urljoin(capability_info.base_url, endpoint_path),
					discovery_method="capability_registration",
					category=capability_info.capability_type.value,
					tags=capability_info.tags
				)
				
				await self.discover_api(api_info)
				
		except Exception as e:
			print(f"Error registering APIs for capability {capability_info.capability_id}: {e}")
	
	async def _unregister_capability_apis(self, capability_info: APGCapabilityInfo):
		"""Unregister API endpoints for a capability."""
		
		try:
			# Remove APIs associated with this capability
			apis_to_remove = []
			
			for api_id, api_info in self.api_registry.items():
				if api_info.service_name == capability_info.capability_name:
					apis_to_remove.append(api_id)
			
			for api_id in apis_to_remove:
				self.api_registry.pop(api_id, None)
				
				# Remove from Redis
				redis_key = f"apg:discovered_apis:{api_id}"
				await self.redis.delete(redis_key)
				
				# Deregister from API Management
				await self.api_service.deregister_api(api_id, "discovered", "discovery_engine")
				
		except Exception as e:
			print(f"Error unregistering APIs for capability {capability_info.capability_id}: {e}")
	
	async def _validate_api_accessibility(self, api_info: APIDiscoveryInfo) -> bool:
		"""Validate that an API is accessible."""
		
		try:
			async with aiohttp.ClientSession() as session:
				async with session.get(api_info.base_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
					return response.status < 500
		except Exception:
			return False
	
	async def _download_api_specification(self, spec_url: str) -> Optional[Dict[str, Any]]:
		"""Download API specification from URL."""
		
		try:
			async with aiohttp.ClientSession() as session:
				async with session.get(spec_url, timeout=aiohttp.ClientTimeout(total=30)) as response:
					if response.status == 200:
						content_type = response.headers.get('content-type', '').lower()
						
						if 'json' in content_type:
							return await response.json()
						elif 'yaml' in content_type or 'yml' in content_type:
							import yaml
							text = await response.text()
							return yaml.safe_load(text)
		except Exception as e:
			print(f"Error downloading API specification from {spec_url}: {e}")
		
		return None
	
	async def _create_api_config_from_discovery(self, api_info: APIDiscoveryInfo, 
											   api_spec: Optional[Dict[str, Any]]) -> Any:
		"""Create API configuration from discovery information."""
		
		from .models import APIConfig
		
		# Extract information from OpenAPI spec if available
		api_title = api_info.api_name
		api_description = None
		
		if api_spec:
			info = api_spec.get('info', {})
			api_title = info.get('title', api_info.api_name)
			api_description = info.get('description')
		
		return APIConfig(
			api_name=api_info.api_name.lower().replace(' ', '_'),
			api_title=api_title,
			api_description=api_description,
			version=api_info.version,
			protocol_type=api_info.protocol,
			base_path="/",
			upstream_url=api_info.base_url,
			is_public=not api_info.auth_required,
			openapi_spec=api_spec,
			category=api_info.category,
			tags=api_info.tags
		)
	
	async def _analyze_api_endpoint(self, url: str, response: aiohttp.ClientResponse) -> Optional[APIDiscoveryInfo]:
		"""Analyze an API endpoint and extract discovery information."""
		
		try:
			# Try to determine API type from response
			content_type = response.headers.get('content-type', '').lower()
			
			if 'json' in content_type:
				data = await response.json()
				
				# Check for OpenAPI/Swagger
				if 'swagger' in data or 'openapi' in data:
					info = data.get('info', {})
					return APIDiscoveryInfo(
						api_id=f"discovered_{int(time.time())}",
						api_name=info.get('title', 'Discovered API'),
						service_name=info.get('title', 'Unknown Service'),
						base_url=url,
						version=info.get('version', '1.0.0'),
						openapi_spec_url=url,
						discovery_method="auto_discovery"
					)
				
				# Check for GraphQL
				elif 'data' in data or 'errors' in data:
					return APIDiscoveryInfo(
						api_id=f"discovered_graphql_{int(time.time())}",
						api_name="GraphQL API",
						service_name="GraphQL Service",
						base_url=url,
						protocol=ProtocolType.GRAPHQL,
						discovery_method="auto_discovery"
					)
		
		except Exception:
			pass
		
		return None

# =============================================================================
# Export Discovery Components
# =============================================================================

__all__ = [
	# Enums
	'ServiceHealth',
	'CapabilityType',
	
	# Data Classes
	'ServiceEndpoint',
	'DiscoveredService',
	
	# Models
	'APGCapabilityInfo',
	'APIDiscoveryInfo',
	
	# Core Component
	'ServiceDiscovery'
]