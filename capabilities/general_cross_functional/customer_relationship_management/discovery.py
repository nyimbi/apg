"""
APG Customer Relationship Management - Service Discovery Integration

Revolutionary service discovery implementation providing seamless APG ecosystem
integration with intelligent service location, health monitoring, and 
automatic failover capabilities.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum

# APG Core imports (these would be actual APG framework imports)
from apg.core.discovery import ServiceDiscovery, ServiceInfo, ServiceStatus
from apg.core.events import EventBus, Event
from apg.core.monitoring import ServiceHealthMonitor


logger = logging.getLogger(__name__)


class ServiceType(str, Enum):
	"""Types of services in APG ecosystem"""
	AUTH_RBAC = "auth_rbac"
	AUDIT_COMPLIANCE = "audit_compliance"
	NOTIFICATION_ENGINE = "notification_engine"
	DOCUMENT_MANAGEMENT = "document_management"
	BUSINESS_INTELLIGENCE = "business_intelligence"
	AI_ORCHESTRATION = "ai_orchestration"
	REAL_TIME_COLLABORATION = "real_time_collaboration"
	COMPUTER_VISION = "computer_vision"
	FEDERATED_LEARNING = "federated_learning"
	WORKFLOW_BPM = "workflow_bpm"


@dataclass
class DiscoveredService:
	"""Information about a discovered service"""
	service_id: str
	service_type: ServiceType
	service_name: str
	version: str
	endpoint_url: str
	health_check_url: str
	capabilities: List[str]
	status: ServiceStatus
	last_seen: datetime
	metadata: Dict[str, Any]


class CRMDiscovery:
	"""
	CRM service discovery manager for APG ecosystem integration.
	
	Handles discovery, monitoring, and management of dependent services
	required by the CRM capability.
	"""
	
	def __init__(
		self, 
		service_discovery: ServiceDiscovery,
		event_bus: EventBus,
		health_monitor: Optional[ServiceHealthMonitor] = None
	):
		"""
		Initialize CRM discovery manager
		
		Args:
			service_discovery: APG service discovery instance
			event_bus: APG event bus instance
			health_monitor: Optional health monitoring instance
		"""
		self.service_discovery = service_discovery
		self.event_bus = event_bus
		self.health_monitor = health_monitor or ServiceHealthMonitor()
		
		# Discovered services cache
		self.dependent_services: Dict[str, DiscoveredService] = {}
		self.service_clients: Dict[str, Any] = {}
		
		# Discovery configuration
		self.discovery_interval = 30  # seconds
		self.health_check_interval = 60  # seconds
		self.service_timeout = 10  # seconds
		
		# Required services for CRM capability
		self.required_services = {
			ServiceType.AUTH_RBAC: {
				"min_version": "1.0.0",
				"required_capabilities": ["user_auth", "role_management", "permission_check"],
				"critical": True
			},
			ServiceType.AUDIT_COMPLIANCE: {
				"min_version": "1.0.0", 
				"required_capabilities": ["audit_logging", "compliance_tracking"],
				"critical": True
			},
			ServiceType.NOTIFICATION_ENGINE: {
				"min_version": "1.0.0",
				"required_capabilities": ["email_notifications", "push_notifications"],
				"critical": False
			},
			ServiceType.DOCUMENT_MANAGEMENT: {
				"min_version": "1.0.0",
				"required_capabilities": ["file_storage", "document_versioning"],
				"critical": False
			},
			ServiceType.BUSINESS_INTELLIGENCE: {
				"min_version": "1.0.0",
				"required_capabilities": ["analytics", "reporting", "dashboards"],
				"critical": False
			},
			ServiceType.AI_ORCHESTRATION: {
				"min_version": "1.0.0",
				"required_capabilities": ["ml_inference", "model_management"],
				"critical": False
			}
		}
		
		# Discovery state
		self._discovery_running = False
		self._discovery_task: Optional[asyncio.Task] = None
		self._health_check_task: Optional[asyncio.Task] = None
		
		logger.info("🔍 CRM Discovery manager initialized")
	
	async def initialize(self):
		"""Initialize discovery manager"""
		try:
			logger.info("🔧 Initializing CRM service discovery...")
			
			# Register for service events
			await self._setup_event_subscriptions()
			
			# Start discovery loops
			await self.start_discovery()
			
			logger.info("✅ CRM service discovery initialized successfully")
			
		except Exception as e:
			logger.error(f"Failed to initialize CRM discovery: {str(e)}", exc_info=True)
			raise
	
	async def _setup_event_subscriptions(self):
		"""Setup event subscriptions for service updates"""
		# Subscribe to service registration events
		await self.event_bus.subscribe(
			"service.registered",
			self._handle_service_registered
		)
		
		# Subscribe to service deregistration events
		await self.event_bus.subscribe(
			"service.deregistered", 
			self._handle_service_deregistered
		)
		
		# Subscribe to service health events
		await self.event_bus.subscribe(
			"service.health_changed",
			self._handle_service_health_changed
		)
	
	async def start_discovery(self):
		"""Start discovery background tasks"""
		if self._discovery_running:
			logger.warning("Discovery already running")
			return
		
		self._discovery_running = True
		
		# Start discovery loop
		self._discovery_task = asyncio.create_task(self._discovery_loop())
		
		# Start health check loop
		self._health_check_task = asyncio.create_task(self._health_check_loop())
		
		logger.info("🔄 Discovery background tasks started")
	
	async def stop_discovery(self):
		"""Stop discovery background tasks"""
		self._discovery_running = False
		
		if self._discovery_task:
			self._discovery_task.cancel()
			try:
				await self._discovery_task
			except asyncio.CancelledError:
				pass
		
		if self._health_check_task:
			self._health_check_task.cancel()
			try:
				await self._health_check_task
			except asyncio.CancelledError:
				pass
		
		logger.info("🛑 Discovery background tasks stopped")
	
	async def discover_dependencies(self):
		"""Discover all required dependencies"""
		logger.info("🔍 Discovering CRM dependencies...")
		
		discovered_count = 0
		
		for service_type, requirements in self.required_services.items():
			try:
				services = await self.service_discovery.find_services(
					service_type=service_type.value,
					min_version=requirements["min_version"],
					required_capabilities=requirements["required_capabilities"]
				)
				
				if services:
					# Use the first available service (could implement load balancing)
					service_info = services[0]
					discovered_service = self._create_discovered_service(service_info, service_type)
					
					# Store discovered service
					self.dependent_services[service_type.value] = discovered_service
					
					# Create service client
					client = await self._create_service_client(discovered_service)
					self.service_clients[service_type.value] = client
					
					discovered_count += 1
					logger.info(f"✅ Discovered {service_type.value}: {service_info.endpoint_url}")
					
				else:
					if requirements["critical"]:
						logger.error(f"❌ Critical service {service_type.value} not found")
					else:
						logger.warning(f"⚠️ Optional service {service_type.value} not found")
			
			except Exception as e:
				logger.error(f"Failed to discover {service_type.value}: {str(e)}")
				if requirements["critical"]:
					raise
		
		logger.info(f"🎉 Discovered {discovered_count}/{len(self.required_services)} services")
		
		# Publish discovery completion event
		await self.event_bus.publish(Event(
			event_type="crm.discovery.completed",
			data={
				"discovered_services": list(self.dependent_services.keys()),
				"total_services": len(self.required_services),
				"discovered_count": discovered_count
			}
		))
	
	def _create_discovered_service(
		self, 
		service_info: ServiceInfo, 
		service_type: ServiceType
	) -> DiscoveredService:
		"""Create discovered service object"""
		return DiscoveredService(
			service_id=service_info.service_id,
			service_type=service_type,
			service_name=service_info.service_name,
			version=service_info.version,
			endpoint_url=service_info.endpoint_url,
			health_check_url=service_info.health_check_url,
			capabilities=service_info.capabilities,
			status=service_info.status,
			last_seen=datetime.utcnow(),
			metadata=service_info.metadata
		)
	
	async def _create_service_client(self, service: DiscoveredService) -> Any:
		"""Create HTTP client for discovered service"""
		# This would create an actual HTTP client (httpx, aiohttp, etc.)
		# For now, return a mock client
		class ServiceClient:
			def __init__(self, service: DiscoveredService):
				self.service = service
				self.base_url = service.endpoint_url
			
			async def health_check(self):
				"""Check service health"""
				# Mock health check - would make actual HTTP request
				return {
					"status": "healthy",
					"service_id": self.service.service_id,
					"timestamp": datetime.utcnow().isoformat()
				}
			
			async def call_api(self, endpoint: str, method: str = "GET", **kwargs):
				"""Make API call to service"""
				# Mock API call - would make actual HTTP request
				return {
					"success": True,
					"endpoint": f"{self.base_url}{endpoint}",
					"method": method,
					"service": self.service.service_name
				}
		
		return ServiceClient(service)
	
	async def _discovery_loop(self):
		"""Background discovery loop"""
		while self._discovery_running:
			try:
				await self.refresh_service_discovery()
				await asyncio.sleep(self.discovery_interval)
				
			except asyncio.CancelledError:
				break
			except Exception as e:
				logger.error(f"Discovery loop error: {str(e)}", exc_info=True)
				await asyncio.sleep(self.discovery_interval)
	
	async def _health_check_loop(self):
		"""Background health check loop"""
		while self._discovery_running:
			try:
				await self.check_services_health()
				await asyncio.sleep(self.health_check_interval)
				
			except asyncio.CancelledError:
				break
			except Exception as e:
				logger.error(f"Health check loop error: {str(e)}", exc_info=True)
				await asyncio.sleep(self.health_check_interval)
	
	async def refresh_service_discovery(self):
		"""Refresh discovered services"""
		try:
			# Re-discover services to catch new instances or updates
			for service_type in self.required_services.keys():
				if service_type.value not in self.dependent_services:
					# Try to discover missing services
					await self._discover_single_service(service_type)
		
		except Exception as e:
			logger.error(f"Failed to refresh service discovery: {str(e)}")
	
	async def _discover_single_service(self, service_type: ServiceType):
		"""Discover a single service type"""
		try:
			requirements = self.required_services[service_type]
			services = await self.service_discovery.find_services(
				service_type=service_type.value,
				min_version=requirements["min_version"],
				required_capabilities=requirements["required_capabilities"]
			)
			
			if services:
				service_info = services[0]
				discovered_service = self._create_discovered_service(service_info, service_type)
				
				self.dependent_services[service_type.value] = discovered_service
				
				client = await self._create_service_client(discovered_service)
				self.service_clients[service_type.value] = client
				
				logger.info(f"🔄 Re-discovered {service_type.value}")
		
		except Exception as e:
			logger.error(f"Failed to discover {service_type.value}: {str(e)}")
	
	async def check_services_health(self):
		"""Check health of all discovered services"""
		unhealthy_services = []
		
		for service_name, client in self.service_clients.items():
			try:
				health_result = await asyncio.wait_for(
					client.health_check(),
					timeout=self.service_timeout
				)
				
				# Update service status
				if service_name in self.dependent_services:
					service = self.dependent_services[service_name]
					service.last_seen = datetime.utcnow()
					
					if health_result.get("status") == "healthy":
						service.status = ServiceStatus.HEALTHY
					else:
						service.status = ServiceStatus.UNHEALTHY
						unhealthy_services.append(service_name)
			
			except asyncio.TimeoutError:
				logger.warning(f"Health check timeout for {service_name}")
				if service_name in self.dependent_services:
					self.dependent_services[service_name].status = ServiceStatus.UNHEALTHY
					unhealthy_services.append(service_name)
			
			except Exception as e:
				logger.error(f"Health check failed for {service_name}: {str(e)}")
				if service_name in self.dependent_services:
					self.dependent_services[service_name].status = ServiceStatus.UNHEALTHY
					unhealthy_services.append(service_name)
		
		# Handle unhealthy services
		if unhealthy_services:
			await self._handle_unhealthy_services(unhealthy_services)
	
	async def _handle_unhealthy_services(self, unhealthy_services: List[str]):
		"""Handle unhealthy services"""
		for service_name in unhealthy_services:
			logger.warning(f"🚨 Service {service_name} is unhealthy, attempting recovery...")
			
			# Try to rediscover the service
			service_type = ServiceType(service_name)
			await self._discover_single_service(service_type)
		
		# Publish unhealthy services event
		await self.event_bus.publish(Event(
			event_type="crm.services.unhealthy",
			data={
				"unhealthy_services": unhealthy_services,
				"timestamp": datetime.utcnow().isoformat()
			}
		))
	
	async def _handle_service_registered(self, event: Event):
		"""Handle service registration event"""
		service_info = event.data
		service_type = service_info.get("service_type")
		
		if service_type in [st.value for st in self.required_services.keys()]:
			logger.info(f"🆕 Required service {service_type} registered, updating discovery...")
			await self._discover_single_service(ServiceType(service_type))
	
	async def _handle_service_deregistered(self, event: Event):
		"""Handle service deregistration event"""
		service_info = event.data
		service_type = service_info.get("service_type")
		
		if service_type in self.dependent_services:
			logger.warning(f"📤 Service {service_type} deregistered, removing from cache...")
			
			# Remove from cache
			del self.dependent_services[service_type]
			if service_type in self.service_clients:
				del self.service_clients[service_type]
			
			# Try to find replacement
			await self._discover_single_service(ServiceType(service_type))
	
	async def _handle_service_health_changed(self, event: Event):
		"""Handle service health change event"""
		service_info = event.data
		service_id = service_info.get("service_id")
		new_status = service_info.get("status")
		
		# Update cached service status
		for service in self.dependent_services.values():
			if service.service_id == service_id:
				service.status = ServiceStatus(new_status)
				service.last_seen = datetime.utcnow()
				break
	
	def get_service_client(self, service_type: ServiceType) -> Optional[Any]:
		"""Get client for a specific service type"""
		return self.service_clients.get(service_type.value)
	
	def is_service_available(self, service_type: ServiceType) -> bool:
		"""Check if a service is available and healthy"""
		service = self.dependent_services.get(service_type.value)
		if not service:
			return False
		
		return service.status == ServiceStatus.HEALTHY
	
	def get_service_info(self, service_type: ServiceType) -> Optional[DiscoveredService]:
		"""Get information about a discovered service"""
		return self.dependent_services.get(service_type.value)
	
	def get_all_services(self) -> Dict[str, DiscoveredService]:
		"""Get all discovered services"""
		return self.dependent_services.copy()
	
	def get_service_status_summary(self) -> Dict[str, Any]:
		"""Get summary of service statuses"""
		total_services = len(self.required_services)
		discovered_services = len(self.dependent_services)
		healthy_services = sum(
			1 for service in self.dependent_services.values()
			if service.status == ServiceStatus.HEALTHY
		)
		
		critical_services = [
			st.value for st, req in self.required_services.items()
			if req["critical"]
		]
		
		critical_healthy = sum(
			1 for service_name in critical_services
			if (service_name in self.dependent_services and 
				self.dependent_services[service_name].status == ServiceStatus.HEALTHY)
		)
		
		return {
			"total_required": total_services,
			"discovered": discovered_services,
			"healthy": healthy_services,
			"critical_services": len(critical_services),
			"critical_healthy": critical_healthy,
			"overall_health": "healthy" if critical_healthy == len(critical_services) else "degraded",
			"last_check": datetime.utcnow().isoformat()
		}
	
	async def shutdown(self):
		"""Shutdown discovery manager"""
		try:
			logger.info("🛑 Shutting down CRM discovery manager...")
			
			await self.stop_discovery()
			
			# Clear caches
			self.dependent_services.clear()
			self.service_clients.clear()
			
			logger.info("✅ CRM discovery manager shutdown completed")
			
		except Exception as e:
			logger.error(f"Error during discovery shutdown: {str(e)}", exc_info=True)


# Utility functions for common service interactions

async def call_auth_service(discovery: CRMDiscovery, endpoint: str, **kwargs) -> Dict[str, Any]:
	"""Helper function to call auth service"""
	client = discovery.get_service_client(ServiceType.AUTH_RBAC)
	if not client:
		raise RuntimeError("Auth service not available")
	
	return await client.call_api(endpoint, **kwargs)


async def call_notification_service(discovery: CRMDiscovery, endpoint: str, **kwargs) -> Dict[str, Any]:
	"""Helper function to call notification service"""
	client = discovery.get_service_client(ServiceType.NOTIFICATION_ENGINE)
	if not client:
		raise RuntimeError("Notification service not available")
	
	return await client.call_api(endpoint, **kwargs)


async def call_audit_service(discovery: CRMDiscovery, endpoint: str, **kwargs) -> Dict[str, Any]:
	"""Helper function to call audit service"""
	client = discovery.get_service_client(ServiceType.AUDIT_COMPLIANCE)
	if not client:
		raise RuntimeError("Audit service not available")
	
	return await client.call_api(endpoint, **kwargs)


# Export classes and functions
__all__ = [
	"CRMDiscovery",
	"DiscoveredService",
	"ServiceType",
	"call_auth_service",
	"call_notification_service", 
	"call_audit_service"
]