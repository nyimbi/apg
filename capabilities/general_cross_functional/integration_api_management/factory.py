"""
APG Integration API Management - Capability Factory

Capability factory for initializing and orchestrating all Integration API Management
components with APG platform integration and discovery capabilities.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

import aioredis
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .config import APIManagementSettings, create_configuration, get_database_url, get_redis_url
from .models import Base
from .service import (
	APILifecycleService, ConsumerManagementService,
	PolicyManagementService, AnalyticsService
)
from .gateway import APIGateway
from .monitoring import MetricsCollector, HealthMonitor, AlertManager
from .discovery import ServiceDiscovery, APGCapabilityInfo, CapabilityType
from .integration import APGIntegrationManager, APGEvent, EventType
from .runner import GatewayApplication

# =============================================================================
# Integration API Management Capability Factory
# =============================================================================

class IntegrationAPIManagementCapability:
	"""Main capability class for Integration API Management."""
	
	def __init__(self, config: Optional[APIManagementSettings] = None):
		self.config = config or create_configuration()
		self.logger = logging.getLogger(__name__)
		
		# Core components
		self.database_engine = None
		self.redis_client = None
		self.session_factory = None
		
		# Services
		self.api_service = None
		self.consumer_service = None
		self.policy_service = None
		self.analytics_service = None
		
		# Gateway components
		self.gateway = None
		self.gateway_app = None
		
		# Monitoring components
		self.metrics_collector = None
		self.health_monitor = None
		self.alert_manager = None
		
		# APG Integration components
		self.service_discovery = None
		self.integration_manager = None
		
		# Capability metadata
		self.capability_info = APGCapabilityInfo(
			capability_id="integration_api_management",
			capability_name="Integration API Management",
			capability_type=CapabilityType.FOUNDATION,
			version="1.0.0",
			description="Comprehensive API gateway and management platform for secure, scalable integration",
			base_url=f"http://{self.config.gateway.host}:{self.config.gateway.port}",
			api_endpoints={
				"admin": "/admin",
				"api": "/api/v1",
				"analytics": "/analytics",
				"developer": "/developer",
				"gateway": "/gateway"
			},
			health_endpoint="/health",
			metrics_endpoint="/metrics",
			openapi_url="/api/v1/openapi.json",
			dependencies=["capability_registry", "event_streaming_bus"],
			provides=[
				"api_gateway", "api_lifecycle_management", "consumer_management",
				"analytics_monitoring", "developer_portal", "service_discovery"
			],
			event_patterns=[
				"api.*", "consumer.*", "policy.*", "gateway.*", "health.*"
			],
			multi_tenant=True,
			auto_scaling=True,
			load_balancing=True,
			tags=["foundation", "api", "gateway", "management", "integration"],
			metadata={
				"max_rps": 100000,
				"supported_protocols": ["REST", "GraphQL", "gRPC", "WebSocket"],
				"auth_methods": ["API_KEY", "JWT", "OAuth2", "mTLS"],
				"deployment_strategy": "blue_green"
			}
		)
		
		# Initialization state
		self.initialized = False
		self.running = False
	
	async def initialize(self, register_with_apg: bool = True) -> bool:
		"""Initialize the capability and all its components."""
		
		if self.initialized:
			return True
		
		try:
			self.logger.info("Initializing Integration API Management capability...")
			
			# Initialize database
			await self._initialize_database()
			
			# Initialize Redis
			await self._initialize_redis()
			
			# Initialize core services
			await self._initialize_services()
			
			# Initialize monitoring
			await self._initialize_monitoring()
			
			# Initialize APG integration
			if register_with_apg:
				await self._initialize_apg_integration()
			
			# Initialize gateway
			await self._initialize_gateway()
			
			self.initialized = True
			self.logger.info("Integration API Management capability initialized successfully")
			
			return True
			
		except Exception as e:
			self.logger.error(f"Failed to initialize capability: {e}")
			await self.cleanup()
			return False
	
	async def start(self, register_with_apg: bool = True) -> bool:
		"""Start the capability and all its services."""
		
		if not self.initialized:
			if not await self.initialize(register_with_apg):
				return False
		
		try:
			self.logger.info("Starting Integration API Management capability...")
			
			# Start monitoring services
			if self.health_monitor:
				await self.health_monitor.start_monitoring()
			
			# Start APG integration
			if self.integration_manager and register_with_apg:
				await self._start_apg_integration()
			
			# Start gateway application
			if self.gateway_app:
				# This would be done in a separate task/process
				self.logger.info("Gateway application ready to start")
			
			self.running = True
			
			# Publish capability started event
			if self.integration_manager and register_with_apg:
				await self.integration_manager.publish_event(APGEvent(
					event_type=EventType.CAPABILITY_REGISTERED,
					source_capability=self.capability_info.capability_id,
					payload=self.capability_info.dict()
				))
			
			self.logger.info("Integration API Management capability started successfully")
			return True
			
		except Exception as e:
			self.logger.error(f"Failed to start capability: {e}")
			return False
	
	async def stop(self) -> bool:
		"""Stop the capability and all its services."""
		
		if not self.running:
			return True
		
		try:
			self.logger.info("Stopping Integration API Management capability...")
			
			# Publish capability stopping event
			if self.integration_manager:
				await self.integration_manager.publish_event(APGEvent(
					event_type=EventType.CAPABILITY_UNREGISTERED,
					source_capability=self.capability_info.capability_id,
					payload={"reason": "graceful_shutdown"}
				))
			
			# Stop APG integration
			if self.integration_manager:
				await self.integration_manager.shutdown()
			
			# Stop monitoring
			if self.health_monitor:
				await self.health_monitor.stop_monitoring()
			
			# Stop service discovery
			if self.service_discovery:
				await self.service_discovery.shutdown()
			
			self.running = False
			self.logger.info("Integration API Management capability stopped successfully")
			
			return True
			
		except Exception as e:
			self.logger.error(f"Error stopping capability: {e}")
			return False
	
	async def cleanup(self):
		"""Cleanup all resources."""
		
		try:
			# Stop if running
			if self.running:
				await self.stop()
			
			# Close Redis connection
			if self.redis_client:
				await self.redis_client.close()
			
			# Close database connections
			if self.database_engine:
				self.database_engine.dispose()
			
			self.initialized = False
			
		except Exception as e:
			self.logger.error(f"Error during cleanup: {e}")
	
	# =============================================================================
	# Health and Status
	# =============================================================================
	
	async def get_health_status(self) -> Dict[str, Any]:
		"""Get comprehensive health status."""
		
		if not self.initialized:
			return {
				"status": "uninitialized",
				"timestamp": datetime.now(timezone.utc).isoformat()
			}
		
		health_status = {
			"status": "healthy",
			"timestamp": datetime.now(timezone.utc).isoformat(),
			"capability": self.capability_info.dict(),
			"components": {}
		}
		
		try:
			# Check database health
			if self.database_engine:
				try:
					with self.database_engine.connect() as conn:
						conn.execute("SELECT 1")
					health_status["components"]["database"] = "healthy"
				except Exception as e:
					health_status["components"]["database"] = f"unhealthy: {e}"
					health_status["status"] = "degraded"
			
			# Check Redis health
			if self.redis_client:
				try:
					await self.redis_client.ping()
					health_status["components"]["redis"] = "healthy"
				except Exception as e:
					health_status["components"]["redis"] = f"unhealthy: {e}"
					health_status["status"] = "degraded"
			
			# Get monitoring health
			if self.health_monitor:
				monitor_health = await self.health_monitor.get_health_report()
				health_status["components"]["monitoring"] = {
					"status": monitor_health.overall_status.value,
					"checks": len(monitor_health.checks),
					"alerts": len(monitor_health.alerts)
				}
				
				if monitor_health.overall_status.value != "healthy":
					health_status["status"] = "degraded"
			
			# Get service discovery status
			if self.service_discovery:
				capabilities = await self.service_discovery.list_capabilities()
				health_status["components"]["service_discovery"] = {
					"status": "healthy",
					"registered_capabilities": len(capabilities)
				}
			
		except Exception as e:
			health_status["status"] = "unhealthy"
			health_status["error"] = str(e)
		
		return health_status
	
	async def get_metrics_summary(self) -> Dict[str, Any]:
		"""Get metrics summary."""
		
		if not self.metrics_collector:
			return {}
		
		return await self.metrics_collector.get_metrics_summary()
	
	async def get_capability_info(self) -> APGCapabilityInfo:
		"""Get capability information."""
		return self.capability_info
	
	# =============================================================================
	# APG Platform Integration
	# =============================================================================
	
	async def register_with_apg_platform(self) -> bool:
		"""Register this capability with the APG platform."""
		
		if not self.service_discovery:
			return False
		
		try:
			return await self.service_discovery.register_capability(self.capability_info)
		except Exception as e:
			self.logger.error(f"Failed to register with APG platform: {e}")
			return False
	
	async def discover_apg_capabilities(self) -> List[APGCapabilityInfo]:
		"""Discover other APG capabilities."""
		
		if not self.service_discovery:
			return []
		
		return await self.service_discovery.list_capabilities()
	
	async def resolve_apg_service(self, service_name: str) -> Optional[str]:
		"""Resolve APG service URL."""
		
		if not self.service_discovery:
			return None
		
		return await self.service_discovery.resolve_service(service_name)
	
	# =============================================================================
	# Service Access Methods
	# =============================================================================
	
	def get_api_service(self) -> Optional[APILifecycleService]:
		"""Get API lifecycle service."""
		return self.api_service
	
	def get_consumer_service(self) -> Optional[ConsumerManagementService]:
		"""Get consumer management service."""
		return self.consumer_service
	
	def get_policy_service(self) -> Optional[PolicyManagementService]:
		"""Get policy management service."""
		return self.policy_service
	
	def get_analytics_service(self) -> Optional[AnalyticsService]:
		"""Get analytics service."""
		return self.analytics_service
	
	def get_service_discovery(self) -> Optional[ServiceDiscovery]:
		"""Get service discovery."""
		return self.service_discovery
	
	def get_integration_manager(self) -> Optional[APGIntegrationManager]:
		"""Get APG integration manager."""
		return self.integration_manager
	
	def get_metrics_collector(self) -> Optional[MetricsCollector]:
		"""Get metrics collector."""
		return self.metrics_collector
	
	def get_health_monitor(self) -> Optional[HealthMonitor]:
		"""Get health monitor."""
		return self.health_monitor
	
	# =============================================================================
	# Internal Initialization Methods
	# =============================================================================
	
	async def _initialize_database(self):
		"""Initialize database connection."""
		
		self.database_engine = create_engine(
			get_database_url(self.config),
			pool_size=self.config.database.pool_size,
			max_overflow=self.config.database.max_overflow,
			pool_timeout=self.config.database.pool_timeout,
			pool_recycle=self.config.database.pool_recycle,
			echo=self.config.database.echo
		)
		
		# Create tables
		Base.metadata.create_all(self.database_engine)
		
		# Create session factory
		self.session_factory = sessionmaker(bind=self.database_engine)
		
		self.logger.info("Database initialized")
	
	async def _initialize_redis(self):
		"""Initialize Redis connection."""
		
		self.redis_client = aioredis.from_url(get_redis_url(self.config))
		await self.redis_client.ping()
		
		self.logger.info("Redis initialized")
	
	async def _initialize_services(self):
		"""Initialize core services."""
		
		self.api_service = APILifecycleService()
		self.consumer_service = ConsumerManagementService()
		self.policy_service = PolicyManagementService()
		self.analytics_service = AnalyticsService()
		
		self.logger.info("Core services initialized")
	
	async def _initialize_monitoring(self):
		"""Initialize monitoring components."""
		
		self.metrics_collector = MetricsCollector(self.redis_client)
		self.health_monitor = HealthMonitor(
			self.redis_client, 
			self.analytics_service, 
			self.metrics_collector
		)
		self.alert_manager = AlertManager(self.redis_client)
		
		self.logger.info("Monitoring components initialized")
	
	async def _initialize_apg_integration(self):
		"""Initialize APG platform integration."""
		
		# Initialize service discovery
		self.service_discovery = ServiceDiscovery(
			self.redis_client,
			self.api_service,
			self.capability_info.capability_id
		)
		await self.service_discovery.initialize()
		
		# Initialize integration manager
		self.integration_manager = APGIntegrationManager(
			self.redis_client,
			self.service_discovery,
			self.api_service,
			self.consumer_service,
			self.analytics_service,
			self.metrics_collector,
			self.health_monitor
		)
		await self.integration_manager.initialize()
		
		self.logger.info("APG integration initialized")
	
	async def _initialize_gateway(self):
		"""Initialize gateway components."""
		
		# Create gateway application
		self.gateway_app = GatewayApplication(self.config)
		
		self.logger.info("Gateway components initialized")
	
	async def _start_apg_integration(self):
		"""Start APG platform integration."""
		
		# Register with APG platform
		await self.register_with_apg_platform()
		
		# Set up default event handlers
		self._setup_default_event_handlers()
		
		self.logger.info("APG integration started")
	
	def _setup_default_event_handlers(self):
		"""Setup default event handlers."""
		
		if not self.integration_manager:
			return
		
		# API lifecycle events
		async def on_api_registered(event: APGEvent):
			self.logger.info(f"API registered: {event.payload.get('api_name')}")
		
		async def on_api_deregistered(event: APGEvent):
			self.logger.info(f"API deregistered: {event.payload.get('api_id')}")
		
		# Consumer events
		async def on_consumer_registered(event: APGEvent):
			self.logger.info(f"Consumer registered: {event.payload.get('consumer_name')}")
		
		# Health events
		async def on_health_changed(event: APGEvent):
			capability_id = event.payload.get('capability_id')
			health_status = event.payload.get('health_status')
			self.logger.warning(f"Health changed for {capability_id}: {health_status}")
		
		# Register event handlers
		self.integration_manager.add_event_handler(EventType.API_REGISTERED, on_api_registered)
		self.integration_manager.add_event_handler(EventType.API_DEREGISTERED, on_api_deregistered)
		self.integration_manager.add_event_handler(EventType.CONSUMER_REGISTERED, on_consumer_registered)
		self.integration_manager.add_event_handler(EventType.CAPABILITY_HEALTH_CHANGED, on_health_changed)

# =============================================================================
# Capability Factory Functions
# =============================================================================

async def create_integration_api_management_capability(
	config: Optional[APIManagementSettings] = None,
	register_with_apg: bool = True
) -> IntegrationAPIManagementCapability:
	"""Factory function to create and initialize the capability."""
	
	capability = IntegrationAPIManagementCapability(config)
	
	if not await capability.initialize(register_with_apg):
		raise RuntimeError("Failed to initialize Integration API Management capability")
	
	return capability

async def create_standalone_capability(config: Optional[APIManagementSettings] = None) -> IntegrationAPIManagementCapability:
	"""Create capability without APG platform integration."""
	
	return await create_integration_api_management_capability(config, register_with_apg=False)

def get_capability_metadata() -> Dict[str, Any]:
	"""Get capability metadata without initialization."""
	
	return {
		"capability_id": "integration_api_management",
		"capability_name": "Integration API Management", 
		"version": "1.0.0",
		"category": "foundation",
		"description": "Comprehensive API gateway and management platform",
		"provides": [
			"api_gateway", "api_lifecycle_management", "consumer_management",
			"analytics_monitoring", "developer_portal", "service_discovery"
		],
		"dependencies": ["capability_registry", "event_streaming_bus"],
		"endpoints": {
			"health": "/health",
			"admin": "/admin",
			"api": "/api/v1",
			"analytics": "/analytics",
			"developer": "/developer",
			"gateway": "/gateway"
		}
	}

# =============================================================================
# Export Factory Components
# =============================================================================

__all__ = [
	'IntegrationAPIManagementCapability',
	'create_integration_api_management_capability',
	'create_standalone_capability',
	'get_capability_metadata'
]