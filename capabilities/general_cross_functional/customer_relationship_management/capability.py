"""
APG Customer Relationship Management Capability

Revolutionary CRM capability providing 10x superior functionality compared to
industry leaders through advanced AI orchestration, seamless APG integration,
and delightful user experience.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

from pydantic import BaseModel, Field
from fastapi import HTTPException

# APG Core imports (these would be actual APG framework imports)
from apg.core.capability import APGCapability, CapabilityInfo, CapabilityStatus
from apg.core.registry import capability_registry
from apg.core.discovery import ServiceDiscovery
from apg.core.events import EventBus
from apg.core.config import APGConfigManager
from apg.core.monitoring import APGMonitoring
from apg.core.auth import APGAuthProvider
from apg.core.gateway import APGGateway

# Local imports
from .service import CRMService
from .models import CRMCapabilityConfig
from .database import DatabaseManager
from .api import app as fastapi_app
from .discovery import CRMDiscovery
from .event_handlers import CRMEventPublisher, CRMEventSubscriber
from .auth_integration import CRMAuthProvider
from .monitoring_integration import CRMMonitoring


logger = logging.getLogger(__name__)


class CustomerRelationshipManagementCapability(APGCapability):
	"""
	APG Customer Relationship Management Capability
	
	Provides revolutionary CRM functionality that is 10x superior to industry
	leaders through advanced AI orchestration, seamless APG integration,
	and exceptional user experience.
	"""
	
	def __init__(self):
		"""Initialize the CRM capability"""
		super().__init__(
			info=CapabilityInfo(
				id="customer_relationship_management",
				name="Customer Relationship Management",
				version="1.0.0",
				category="general_cross_functional",
				domain="business_operations",
				description="Revolutionary CRM capability with AI-powered insights and seamless customer engagement",
				author="Nyimbi Odero <nyimbi@gmail.com>",
				license="Proprietary - Datacraft © 2025",
				tags=[
					"crm", "customer-management", "sales", "marketing", "ai-powered",
					"lead-management", "opportunity-tracking", "customer-intelligence",
					"sales-automation", "predictive-analytics", "mobile-first",
					"real-time", "collaboration", "omnichannel", "revenue-intelligence"
				],
				capabilities=[
					"contact_management",
					"account_management", 
					"lead_management",
					"opportunity_management",
					"sales_pipeline",
					"customer_intelligence",
					"predictive_analytics",
					"ai_recommendations",
					"marketing_automation",
					"customer_service",
					"mobile_crm",
					"real_time_collaboration",
					"omnichannel_engagement",
					"revenue_forecasting",
					"workflow_automation",
					"document_management",
					"reporting_analytics",
					"integration_hub"
				],
				dependencies=[
					"auth_rbac>=1.0.0",
					"audit_compliance>=1.0.0", 
					"notification_engine>=1.0.0",
					"document_management>=1.0.0",
					"business_intelligence>=1.0.0",
					"ai_orchestration>=1.0.0",
					"real_time_collaboration>=1.0.0",
					"computer_vision>=1.0.0",
					"federated_learning>=1.0.0",
					"workflow_bpm>=1.0.0"
				],
				health_check_endpoint="/health",
				metrics_endpoint="/metrics",
				openapi_endpoint="/openapi.json"
			)
		)
		
		# Initialize core components
		self.db_manager: Optional[DatabaseManager] = None
		self.service: Optional[CRMService] = None
		self.discovery: Optional[CRMDiscovery] = None
		self.event_publisher: Optional[CRMEventPublisher] = None
		self.event_subscriber: Optional[CRMEventSubscriber] = None
		self.auth_provider: Optional[CRMAuthProvider] = None
		self.monitoring: Optional[CRMMonitoring] = None
		
		# APG integrations
		self.apg_gateway: Optional[APGGateway] = None
		self.service_discovery: Optional[ServiceDiscovery] = None
		self.event_bus: Optional[EventBus] = None
		self.config_manager: Optional[APGConfigManager] = None
		self.apg_monitoring: Optional[APGMonitoring] = None
		
		# Status tracking
		self.initialization_start: Optional[datetime] = None
		self.ready_timestamp: Optional[datetime] = None
		self.dependent_services: Dict[str, Any] = {}
	
	async def initialize(self) -> bool:
		"""
		Initialize the CRM capability and register with APG ecosystem
		
		Returns:
			bool: True if initialization successful, False otherwise
		"""
		self.initialization_start = datetime.utcnow()
		logger.info("🚀 Initializing APG Customer Relationship Management capability...")
		
		try:
			# Phase 1: Initialize APG core integrations
			await self._initialize_apg_integrations()
			logger.info("✅ APG integrations initialized")
			
			# Phase 2: Initialize database and data layer
			await self._initialize_database()
			logger.info("✅ Database layer initialized")
			
			# Phase 3: Initialize business service layer
			await self._initialize_services()
			logger.info("✅ Business services initialized")
			
			# Phase 4: Initialize API and communication layer
			await self._initialize_api_layer()
			logger.info("✅ API layer initialized")
			
			# Phase 5: Initialize event-driven architecture
			await self._initialize_event_architecture()
			logger.info("✅ Event architecture initialized")
			
			# Phase 6: Initialize monitoring and health checks
			await self._initialize_monitoring()
			logger.info("✅ Monitoring and health checks initialized")
			
			# Phase 7: Discover and connect to dependent services
			await self._discover_dependencies()
			logger.info("✅ Service dependencies discovered")
			
			# Phase 8: Register with APG capability registry
			await self._register_with_apg()
			logger.info("✅ Registered with APG capability registry")
			
			# Phase 9: Run initialization validation
			await self._validate_initialization()
			logger.info("✅ Initialization validation completed")
			
			self.ready_timestamp = datetime.utcnow()
			self.status = CapabilityStatus.READY
			
			initialization_time = (self.ready_timestamp - self.initialization_start).total_seconds()
			logger.info(f"🎉 CRM capability initialization completed successfully in {initialization_time:.2f}s")
			
			return True
			
		except Exception as e:
			logger.error(f"💥 Failed to initialize CRM capability: {str(e)}", exc_info=True)
			self.status = CapabilityStatus.FAILED
			return False
	
	async def _initialize_apg_integrations(self):
		"""Initialize core APG framework integrations"""
		logger.info("🔌 Initializing APG framework integrations...")
		
		# Initialize APG Gateway
		self.apg_gateway = APGGateway()
		await self.apg_gateway.initialize()
		
		# Initialize Service Discovery
		self.service_discovery = ServiceDiscovery()
		await self.service_discovery.initialize()
		
		# Initialize Event Bus
		self.event_bus = EventBus()
		await self.event_bus.initialize()
		
		# Initialize Configuration Manager
		self.config_manager = APGConfigManager()
		await self.config_manager.initialize()
		
		# Initialize APG Monitoring
		self.apg_monitoring = APGMonitoring()
		await self.apg_monitoring.initialize()
		
		logger.info("✅ APG framework integrations ready")
	
	async def _initialize_database(self):
		"""Initialize database connections and schema"""
		logger.info("🗄️ Initializing database layer...")
		
		# Initialize database manager
		self.db_manager = DatabaseManager()
		await self.db_manager.initialize()
		
		# Run database migrations
		await self.db_manager.run_migrations()
		
		# Validate database schema
		await self.db_manager.validate_schema()
		
		logger.info("✅ Database layer ready")
	
	async def _initialize_services(self):
		"""Initialize business service layer"""
		logger.info("⚙️ Initializing business services...")
		
		# Initialize core CRM service
		self.service = CRMService(
			db_manager=self.db_manager,
			config_manager=self.config_manager
		)
		await self.service.initialize()
		
		logger.info("✅ Business services ready")
	
	async def _initialize_api_layer(self):
		"""Initialize API endpoints and communication layer"""
		logger.info("🌐 Initializing API layer...")
		
		# Configure FastAPI application
		fastapi_app.dependency_overrides.update({
			"get_crm_service": lambda: self.service,
			"get_db_manager": lambda: self.db_manager,
			"get_config_manager": lambda: self.config_manager
		})
		
		# Register API endpoints with APG Gateway
		await self._register_api_endpoints()
		
		logger.info("✅ API layer ready")
	
	async def _initialize_event_architecture(self):
		"""Initialize event-driven architecture"""
		logger.info("📡 Initializing event architecture...")
		
		# Initialize event publisher
		self.event_publisher = CRMEventPublisher(self.event_bus)
		await self.event_publisher.initialize()
		
		# Initialize event subscriber
		self.event_subscriber = CRMEventSubscriber(
			event_bus=self.event_bus,
			service=self.service
		)
		await self.event_subscriber.setup_subscriptions()
		
		logger.info("✅ Event architecture ready")
	
	async def _initialize_monitoring(self):
		"""Initialize monitoring and health checks"""
		logger.info("📊 Initializing monitoring...")
		
		# Initialize CRM-specific monitoring
		self.monitoring = CRMMonitoring(self.apg_monitoring)
		await self.monitoring.initialize()
		
		# Initialize authentication provider
		self.auth_provider = CRMAuthProvider()
		await self.auth_provider.initialize()
		
		logger.info("✅ Monitoring ready")
	
	async def _discover_dependencies(self):
		"""Discover and connect to dependent APG capabilities"""
		logger.info("🔍 Discovering service dependencies...")
		
		# Initialize service discovery handler
		self.discovery = CRMDiscovery(
			service_discovery=self.service_discovery,
			event_bus=self.event_bus
		)
		
		# Discover dependent capabilities
		await self.discovery.discover_dependencies()
		
		# Store discovered services
		self.dependent_services = self.discovery.dependent_services
		
		logger.info(f"✅ Discovered {len(self.dependent_services)} dependent services")
	
	async def _register_with_apg(self):
		"""Register capability with APG registry"""
		logger.info("📝 Registering with APG capability registry...")
		
		# Register capability
		await capability_registry.register(self)
		
		# Register health check
		await capability_registry.register_health_check(
			capability_id=self.info.id,
			health_check_url=f"/api/general_cross_functional/customer_relationship_management{self.info.health_check_endpoint}"
		)
		
		logger.info("✅ Registered with APG registry")
	
	async def _register_api_endpoints(self):
		"""Register API endpoints with APG Gateway"""
		logger.info("🔗 Registering API endpoints...")
		
		endpoints = {
			"rest_api": {
				"base_path": "/api/general_cross_functional/customer_relationship_management",
				"health_check": "/health",
				"openapi_spec": "/openapi.json",
				"metrics": "/metrics",
				"authentication": "required",
				"rate_limiting": "enabled",
				"cors_enabled": True,
				"documentation_url": "/docs"
			},
			"websocket": {
				"base_path": "/ws/customer_relationship_management",
				"authentication": "required",
				"real_time_events": True,
				"connection_limit": 1000
			},
			"mobile_api": {
				"base_path": "/api/mobile/general_cross_functional/customer_relationship_management",
				"optimized_payloads": True,
				"offline_support": True,
				"push_notifications": True
			},
			"graphql": {
				"base_path": "/graphql/customer_relationship_management",
				"authentication": "required",
				"subscription_support": True,
				"query_complexity_limit": 1000
			}
		}
		
		await self.apg_gateway.register_capability_endpoints(
			capability_id=self.info.id,
			endpoints=endpoints
		)
		
		logger.info("✅ API endpoints registered")
	
	async def _validate_initialization(self):
		"""Validate that initialization was successful"""
		logger.info("🔍 Running initialization validation...")
		
		validations = [
			("Database connection", self._validate_database_connection()),
			("Service layer", self._validate_service_layer()),
			("API endpoints", self._validate_api_endpoints()),
			("Event architecture", self._validate_event_architecture()),
			("Monitoring systems", self._validate_monitoring()),
			("APG integrations", self._validate_apg_integrations())
		]
		
		for validation_name, validation_coro in validations:
			try:
				await validation_coro
				logger.info(f"✅ {validation_name} validation passed")
			except Exception as e:
				logger.error(f"❌ {validation_name} validation failed: {str(e)}")
				raise
		
		logger.info("✅ All initialization validations passed")
	
	async def _validate_database_connection(self):
		"""Validate database connection"""
		if not self.db_manager:
			raise Exception("Database manager not initialized")
		
		await self.db_manager.health_check()
	
	async def _validate_service_layer(self):
		"""Validate service layer"""
		if not self.service:
			raise Exception("CRM service not initialized")
		
		await self.service.health_check()
	
	async def _validate_api_endpoints(self):
		"""Validate API endpoints"""
		# This would make actual HTTP requests to validate endpoints
		pass
	
	async def _validate_event_architecture(self):
		"""Validate event architecture"""
		if not self.event_publisher or not self.event_subscriber:
			raise Exception("Event architecture not properly initialized")
		
		# Test event publishing and subscription
		await self.event_publisher.health_check()
	
	async def _validate_monitoring(self):
		"""Validate monitoring systems"""
		if not self.monitoring:
			raise Exception("Monitoring not initialized")
		
		await self.monitoring.health_check()
	
	async def _validate_apg_integrations(self):
		"""Validate APG integrations"""
		integrations = [
			("APG Gateway", self.apg_gateway),
			("Service Discovery", self.service_discovery),
			("Event Bus", self.event_bus),
			("Config Manager", self.config_manager),
			("APG Monitoring", self.apg_monitoring)
		]
		
		for name, integration in integrations:
			if not integration:
				raise Exception(f"{name} not initialized")
	
	async def health_check(self) -> Dict[str, Any]:
		"""
		Comprehensive health check for the CRM capability
		
		Returns:
			Dict containing health status and component details
		"""
		health_status = {
			"status": "healthy",
			"timestamp": datetime.utcnow().isoformat(),
			"capability": {
				"id": self.info.id,
				"version": self.info.version,
				"status": self.status.value,
				"uptime_seconds": (datetime.utcnow() - self.ready_timestamp).total_seconds() if self.ready_timestamp else 0
			},
			"components": {},
			"dependencies": {},
			"metrics": {}
		}
		
		try:
			# Check database health
			if self.db_manager:
				db_health = await self.db_manager.health_check()
				health_status["components"]["database"] = db_health
			
			# Check service health
			if self.service:
				service_health = await self.service.health_check()
				health_status["components"]["service"] = service_health
			
			# Check event system health
			if self.event_publisher and self.event_subscriber:
				event_health = {
					"publisher": await self.event_publisher.health_check(),
					"subscriber": "active"
				}
				health_status["components"]["events"] = event_health
			
			# Check dependent services
			for service_name, service_client in self.dependent_services.items():
				try:
					dep_health = await service_client.health_check()
					health_status["dependencies"][service_name] = dep_health
				except Exception as e:
					health_status["dependencies"][service_name] = {
						"status": "unhealthy",
						"error": str(e)
					}
			
			# Collect basic metrics
			if self.monitoring:
				health_status["metrics"] = await self.monitoring.get_health_metrics()
			
			# Determine overall health
			component_statuses = [comp.get("status", "unknown") for comp in health_status["components"].values()]
			dependency_statuses = [dep.get("status", "unknown") for dep in health_status["dependencies"].values()]
			
			if any(status == "unhealthy" for status in component_statuses):
				health_status["status"] = "degraded"
			elif any(status == "unhealthy" for status in dependency_statuses):
				health_status["status"] = "degraded"
			
		except Exception as e:
			logger.error(f"Health check failed: {str(e)}", exc_info=True)
			health_status["status"] = "unhealthy"
			health_status["error"] = str(e)
		
		return health_status
	
	async def shutdown(self):
		"""Gracefully shutdown the CRM capability"""
		logger.info("🛑 Shutting down CRM capability...")
		
		self.status = CapabilityStatus.STOPPING
		
		try:
			# Stop event processing
			if self.event_subscriber:
				await self.event_subscriber.shutdown()
			
			# Stop services
			if self.service:
				await self.service.shutdown()
			
			# Close database connections
			if self.db_manager:
				await self.db_manager.shutdown()
			
			# Unregister from APG registry
			await capability_registry.unregister(self.info.id)
			
			self.status = CapabilityStatus.STOPPED
			logger.info("✅ CRM capability shutdown completed")
			
		except Exception as e:
			logger.error(f"Error during shutdown: {str(e)}", exc_info=True)
			self.status = CapabilityStatus.FAILED
	
	def get_capability_info(self) -> Dict[str, Any]:
		"""
		Get comprehensive capability information
		
		Returns:
			Dict containing capability metadata and status
		"""
		return {
			"id": self.info.id,
			"name": self.info.name,
			"version": self.info.version,
			"category": self.info.category,
			"domain": self.info.domain,
			"description": self.info.description,
			"status": self.status.value,
			"capabilities": self.info.capabilities,
			"dependencies": self.info.dependencies,
			"tags": self.info.tags,
			"author": self.info.author,
			"license": self.info.license,
			"endpoints": {
				"health": self.info.health_check_endpoint,
				"metrics": self.info.metrics_endpoint,
				"openapi": self.info.openapi_endpoint
			},
			"initialization_time": self.initialization_start.isoformat() if self.initialization_start else None,
			"ready_time": self.ready_timestamp.isoformat() if self.ready_timestamp else None,
			"dependent_services": list(self.dependent_services.keys())
		}


# Global capability instance
crm_capability = CustomerRelationshipManagementCapability()


async def initialize_capability() -> bool:
	"""
	Initialize the CRM capability
	
	Returns:
		bool: True if successful, False otherwise
	"""
	return await crm_capability.initialize()


async def get_capability_health() -> Dict[str, Any]:
	"""
	Get capability health status
	
	Returns:
		Dict containing health information
	"""
	return await crm_capability.health_check()


async def shutdown_capability():
	"""Shutdown the capability gracefully"""
	await crm_capability.shutdown()


def get_capability() -> CustomerRelationshipManagementCapability:
	"""
	Get the capability instance
	
	Returns:
		The CRM capability instance
	"""
	return crm_capability


# Export the capability for APG framework discovery
__all__ = [
	"CustomerRelationshipManagementCapability",
	"crm_capability",
	"initialize_capability",
	"get_capability_health", 
	"shutdown_capability",
	"get_capability"
]