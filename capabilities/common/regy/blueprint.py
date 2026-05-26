#!/usr/bin/env python3
"""
Registry (regy) - APG Blueprint Integration
==========================================

Flask blueprint integration with APG composition engine registration,
menu integration, and comprehensive capability management.

Author: APG Platform Team
Copyright: © 2025 Datacraft
Website: www.datacraft.co.ke
"""

import asyncio
import logging
from asyncio import events
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

from flask import Blueprint, current_app, g
from flask_appbuilder import AppBuilder, SQLA

from .api import registry_bp, api
from .service import ServiceRegistryService
from .views import (
	ServiceRegistryView, ServiceDiscoveryView,
	ServiceHealthView, ServiceAnalyticsView
)
from .models import ServiceRegistration, ServiceStatus

# APG Integration Imports
try:
	from ..composition.registry import CapabilityRegistry
	from ..auth.service import AuthService
	from ..moni.service import MonitoringService
	from ..audl.service import AuditLoggingService
	APG_COMPOSITION_AVAILABLE = True
except ImportError:
	# Fallback for development
	CapabilityRegistry = None
	AuthService = None
	MonitoringService = None
	AuditLoggingService = None
	APG_COMPOSITION_AVAILABLE = False

# Global registry service instance
_registry_service: Optional[ServiceRegistryService] = None


def _get_logger(app=None):
	"""Return the active Flask logger or a module logger outside app context."""
	if app is not None:
		return app.logger
	try:
		return current_app.logger
	except RuntimeError:
		return logging.getLogger(__name__)


def _has_apg_composition() -> bool:
	"""Treat patched test doubles as available composition integrations."""
	return CapabilityRegistry is not None


def _run_async(coroutine):
	"""Execute a coroutine without depending on Flask's async extra."""
	if events._get_running_loop() is None:
		return asyncio.run(coroutine)

	previous_loop = events._get_running_loop()
	loop = asyncio.new_event_loop()
	try:
		events._set_running_loop(None)
		return loop.run_until_complete(coroutine)
	finally:
		pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
		for task in pending:
			task.cancel()
		if pending:
			loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
		loop.close()
		events._set_running_loop(previous_loop)

def get_registry_service(tenant_id: str = "default") -> ServiceRegistryService:
	"""Get or create registry service instance."""
	global _registry_service
	if not _registry_service:
		_registry_service = ServiceRegistryService(tenant_id)
	return _registry_service

async def initialize_registry_service(tenant_id: str = "default") -> ServiceRegistryService:
	"""Initialize registry service with APG integrations."""
	service = get_registry_service(tenant_id)
	if not service.initialized:
		await service.initialize()
	return service

def init_app(app, db: SQLA = None):
	"""
	Initialize Registry capability with Flask-AppBuilder application.
	
	Args:
		app: Flask-AppBuilder application instance
		db: SQLAlchemy database instance (optional)
	"""
	
	# Register API blueprint
	app.register_blueprint(registry_bp)
	
	# Register Flask-AppBuilder views
	if hasattr(app, 'appbuilder'):
		appbuilder = app.appbuilder
		
		# Add views to AppBuilder
		appbuilder.add_view(
			ServiceRegistryView,
			"Service Registry",
			icon="fa-network-wired",
			category="Integration",
			category_icon="fa-cogs"
		)
		
		appbuilder.add_view(
			ServiceDiscoveryView,
			"Service Discovery", 
			icon="fa-search",
			category="Integration"
		)
		
		appbuilder.add_view(
			ServiceHealthView,
			"Service Health",
			icon="fa-heartbeat", 
			category="Monitoring",
			category_icon="fa-chart-line"
		)
		
		appbuilder.add_view(
			ServiceAnalyticsView,
			"Service Analytics",
			icon="fa-analytics",
			category="Analytics",
			category_icon="fa-chart-bar"
		)
		
		# Add API documentation link
		appbuilder.add_link(
			"Registry API Docs",
			href="/api/regy/v1/",
			icon="fa-book",
			category="Documentation"
		)
	
	async def setup_registry():
		"""Setup registry service on application startup."""
		try:
			# Get tenant ID from configuration or default
			tenant_id = app.config.get('APG_DEFAULT_TENANT_ID', 'default')
			
			# Initialize registry service
			await initialize_registry_service(tenant_id)
			
			# Register with APG composition engine
			await register_with_apg_composition(tenant_id)
			
			app.logger.info(f"Registry capability initialized successfully for tenant: {tenant_id}")
			
		except Exception as e:
			app.logger.error(f"Failed to initialize registry capability: {str(e)}")
			# Don't fail app startup, but log the error

	# Initialize registry on first request.
	if hasattr(app, "before_first_request"):
		@app.before_first_request
		def setup_registry_before_first_request():
			_run_async(setup_registry())
	else:
		startup_state = app.extensions.setdefault(
			"regy_startup_state",
			{"completed": False, "in_progress": False},
		)

		@app.before_request
		def setup_registry_on_first_request():
			"""Flask 3 compatibility path for one-time startup initialization."""
			if startup_state["completed"] or startup_state["in_progress"]:
				return None
			startup_state["in_progress"] = True
			try:
				_run_async(setup_registry())
			finally:
				startup_state["completed"] = True
				startup_state["in_progress"] = False
			return None
	
	# Add health check endpoint
	@app.route('/health/registry')
	def registry_health_check():
		"""Registry capability health check endpoint.""" 
		try:
			service = get_registry_service()
			
			if not service.initialized:
				return {
					'status': 'initializing',
					'message': 'Registry service is initializing'
				}, 503
			
			# Get basic statistics
			stats = _run_async(service.get_registry_statistics())
			
			return {
				'status': 'healthy',
				'capability': 'registry',
				'version': '1.0.0',
				'tenant_id': service.tenant_id,
				'uptime_seconds': stats['registry_info']['uptime_seconds'],
				'total_services': stats['service_statistics']['total_services'],
				'healthy_services': stats['service_statistics']['healthy_services'],
				'performance': stats['performance_counters']
			}
			
		except Exception as e:
			return {
				'status': 'unhealthy',
				'capability': 'registry',
				'error': str(e)
			}, 500
	
	# Add metrics endpoint for APG monitoring
	@app.route('/metrics/registry')
	def registry_metrics():
		"""Registry capability metrics endpoint."""
		try:
			service = get_registry_service()
			
			if not service.initialized:
				return app.response_class(
					'{"error":"Registry not initialized"}',
					status=503,
					content_type='application/json; charset=utf-8'
				)
			
			stats = _run_async(service.get_registry_statistics())
			
			# Convert to Prometheus-style metrics
			metrics = [
				f"registry_total_services {stats['service_statistics']['total_services']}",
				f"registry_healthy_services {stats['service_statistics']['healthy_services']}",
				f"registry_degraded_services {stats['service_statistics']['degraded_services']}",
				f"registry_unhealthy_services {stats['service_statistics']['unhealthy_services']}",
				f"registry_total_registrations {stats['performance_counters']['total_registrations']}",
				f"registry_total_discoveries {stats['performance_counters']['total_discoveries']}",
				f"registry_total_health_checks {stats['performance_counters']['total_health_checks']}",
				f"registry_cache_hit_rate {stats['performance_counters']['cache_hit_rate']}",
				f"registry_uptime_seconds {stats['registry_info']['uptime_seconds']}"
			]
			
			return app.response_class(
				'\n'.join(metrics),
				status=200,
				content_type='text/plain; charset=utf-8'
			)
			
		except Exception as e:
			return app.response_class(
				f"# Error: {str(e)}",
				status=500,
				content_type='text/plain; charset=utf-8'
			)

async def register_with_apg_composition(tenant_id: str = "default") -> bool:
	"""
	Register Registry capability with APG Composition Engine.
	
	Args:
		tenant_id: APG tenant identifier
		
	Returns:
		bool: Registration success status
	"""
	
	logger = _get_logger()

	if not _has_apg_composition():
		logger.warning("APG Composition Engine not available - skipping registration")
		return False
	
	try:
		# Get capability registry
		capability_registry = CapabilityRegistry(tenant_id)
		await capability_registry.initialize()
		
		# Define capability metadata
		capability_metadata = {
			"id": "regy",
			"name": "Registry (regy)",
			"version": "1.0.0",
			"description": "API/Service Registry with intelligent discovery and health monitoring",
			"type": "integration", 
			"category": "service_management",
			"status": "active",
			
			# APG Dependencies
			"dependencies": [
				{
					"capability_id": "auth",
					"version": ">=1.0.0",
					"required": True,
					"integration_points": ["service_authentication", "rbac_policies"]
				},
				{
					"capability_id": "conf",
					"version": ">=1.0.0", 
					"required": True,
					"integration_points": ["dynamic_configuration", "service_config"]
				},
				{
					"capability_id": "moni",
					"version": ">=1.0.0",
					"required": True,
					"integration_points": ["service_metrics", "health_checks"]
				},
				{
					"capability_id": "audl",
					"version": ">=1.0.0",
					"required": True,
					"integration_points": ["registration_events", "discovery_logs"]
				},
				{
					"capability_id": "apig",
					"version": ">=1.0.0",
					"required": False,
					"integration_points": ["gateway_integration", "routing_updates"]
				}
			],
			
			# Services Provided
			"provides": [
				{
					"service_id": "service_discovery",
					"name": "Intelligent Service Discovery",
					"description": "AI-powered service discovery with health-aware routing",
					"endpoints": ["/api/regy/v1/discovery/search"],
					"capabilities": ["intelligent_ranking", "predictive_filtering", "health_awareness"]
				},
				{
					"service_id": "service_registration", 
					"name": "Dynamic Service Registration",
					"description": "Zero-downtime service registration with dependency mapping",
					"endpoints": ["/api/regy/v1/services"],
					"capabilities": ["zero_downtime", "dependency_mapping", "validation"]
				},
				{
					"service_id": "health_monitoring",
					"name": "ML-Powered Health Monitoring", 
					"description": "Advanced health monitoring with anomaly detection",
					"endpoints": ["/api/regy/v1/health"],
					"capabilities": ["anomaly_detection", "predictive_analysis", "adaptive_monitoring"]
				},
				{
					"service_id": "circuit_breaking",
					"name": "Intelligent Circuit Breaking",
					"description": "AI-optimized circuit breaker management",
					"endpoints": ["/api/regy/v1/health"],
					"capabilities": ["adaptive_thresholds", "pattern_recognition", "intelligent_recovery"]
				},
				{
					"service_id": "api_versioning",
					"name": "API Version Management",
					"description": "Semantic versioning with migration assistance",
					"endpoints": ["/api/regy/v1/services"],
					"capabilities": ["semantic_versioning", "migration_assistance", "compatibility_checking"]
				}
			],
			
			# UI Integration
			"ui_integration": {
				"menu_items": [
					{
						"name": "Service Registry",
						"url": "/serviceregistryview/list/",
						"icon": "fa-network-wired",
						"category": "Integration",
						"permissions": ["registry:list_services"]
					},
					{
						"name": "Service Discovery", 
						"url": "/discovery/search/",
						"icon": "fa-search",
						"category": "Integration",
						"permissions": ["registry:discover_services"]
					},
					{
						"name": "Service Health",
						"url": "/health/dashboard/",
						"icon": "fa-heartbeat",
						"category": "Monitoring", 
						"permissions": ["registry:view_health"]
					},
					{
						"name": "Service Analytics",
						"url": "/analytics/dashboard/",
						"icon": "fa-analytics",
						"category": "Analytics",
						"permissions": ["registry:view_analytics"]
					}
				],
				"dashboard_widgets": [
					{
						"id": "service_health_overview",
						"name": "Service Health Overview",
						"type": "chart",
						"size": "medium",
						"data_source": "/api/regy/v1/health",
						"refresh_interval": 30
					},
					{
						"id": "registry_statistics",
						"name": "Registry Statistics",
						"type": "metrics",
						"size": "small", 
						"data_source": "/metrics/registry",
						"refresh_interval": 60
					}
				]
			},
			
			# API Endpoints
			"api_endpoints": {
				"base_url": "/api/regy/v1",
				"documentation_url": "/api/regy/v1/",
				"health_check": "/health/registry",
				"metrics": "/metrics/registry",
				"endpoints": [
					{
						"path": "/services",
						"methods": ["GET", "POST"],
						"description": "Service registration and listing",
						"permissions": ["registry:list_services", "registry:register_service"]
					},
					{
						"path": "/services/{service_id}",
						"methods": ["GET", "PUT", "DELETE"], 
						"description": "Individual service management",
						"permissions": ["registry:get_service", "registry:update_service", "registry:deregister_service"]
					},
					{
						"path": "/discovery/search",
						"methods": ["POST"],
						"description": "Intelligent service discovery", 
						"permissions": ["registry:discover_services"]
					},
					{
						"path": "/health/services/{service_id}",
						"methods": ["GET", "PUT"],
						"description": "Service health monitoring",
						"permissions": ["registry:view_health", "registry:update_health"]
					}
				]
			},
			
			# Configuration Schema
			"configuration": {
				"cache_ttl_seconds": {
					"type": "integer",
					"default": 300,
					"description": "Cache TTL for discovery results"
				},
				"anomaly_detection": {
					"type": "boolean", 
					"default": False,
					"description": "Enable ML-powered anomaly detection"
				},
				"predictive_scaling": {
					"type": "boolean",
					"default": False,
					"description": "Enable AI-powered predictive scaling"
				},
				"health_check_interval": {
					"type": "integer",
					"default": 30,
					"description": "Default health check interval in seconds"
				}
			},
			
			# Permissions
			"permissions": [
				{
					"name": "registry:list_services",
					"description": "List registered services"
				},
				{
					"name": "registry:register_service",
					"description": "Register new services"
				},
				{
					"name": "registry:get_service", 
					"description": "View service details"
				},
				{
					"name": "registry:update_service",
					"description": "Update service configuration"
				},
				{
					"name": "registry:deregister_service",
					"description": "Deregister services"
				},
				{
					"name": "registry:discover_services",
					"description": "Discover services"
				},
				{
					"name": "registry:view_health",
					"description": "View service health status"
				},
				{
					"name": "registry:update_health",
					"description": "Update service health status"
				},
				{
					"name": "registry:view_metrics",
					"description": "View service metrics"
				},
				{
					"name": "registry:view_analytics",
					"description": "View service analytics"
				},
				{
					"name": "registry:view_events",
					"description": "View registry events"
				},
				{
					"name": "registry:view_statistics",
					"description": "View registry statistics"
				},
				{
					"name": "registry:trigger_health_check",
					"description": "Trigger manual health checks"
				}
			],
			
			# Metadata
			"created_at": datetime.now(timezone.utc).isoformat(),
			"created_by": "system",
			"tenant_id": tenant_id
		}
		
		# Register capability
		success = await capability_registry.register_capability(capability_metadata)
		
		if success:
			logger.info("Registry capability registered successfully with APG Composition Engine")
			return True
		else:
			logger.error("Failed to register Registry capability with APG Composition Engine")
			return False
			
	except Exception as e:
		logger.error(f"Error registering Registry capability: {str(e)}")
		return False

async def validate_apg_dependencies(tenant_id: str = "default") -> Dict[str, bool]:
	"""
	Validate that all APG dependencies are available and functional.
	
	Args:
		tenant_id: APG tenant identifier
		
	Returns:
		Dict[str, bool]: Dependency validation results
	"""
	
	validation_results = {
		"auth": False,
		"conf": False, 
		"moni": False,
		"audl": False,
		"composition": _has_apg_composition()
	}
	
	try:
		# Validate auth service
		if AuthService:
			auth_service = AuthService(tenant_id)
			await auth_service.initialize()
			validation_results["auth"] = True
	except Exception as e:
		_get_logger().warning(f"Auth service validation failed: {str(e)}")
	
	try:
		# Validate monitoring service  
		if MonitoringService:
			monitoring_service = MonitoringService(tenant_id)
			await monitoring_service.initialize()
			validation_results["moni"] = True
	except Exception as e:
		_get_logger().warning(f"Monitoring service validation failed: {str(e)}")
	
	try:
		# Validate audit service
		if AuditLoggingService:
			audit_service = AuditLoggingService(tenant_id)
			await audit_service.initialize()
			validation_results["audl"] = True
	except Exception as e:
		_get_logger().warning(f"Audit service validation failed: {str(e)}")
	
	return validation_results

def configure_default_data(app, tenant_id: str = "default"):
	"""
	Configure default services and health checks for the registry.
	
	Args:
		app: Flask application instance
		tenant_id: APG tenant identifier
	"""
	
	async def setup_default_services():
		"""Setup default services for demonstration and testing."""
		try:
			service = await initialize_registry_service(tenant_id)
			
			# Check if we already have services (avoid duplicates)
			if len(service.services) > 0:
				return
			
			# Register default services
			default_services = [
				{
					"name": "apg-auth-service",
					"display_name": "APG Authentication Service",
					"description": "Core authentication and authorization service",
					"service_type": "auth_service",
					"namespace": "apg-core",
					"environment": "production",
					"base_path": "/api/auth/v1",
					"instances": [
						{
							"instance_name": "auth-primary",
							"host": "auth.apg.datacraft.co.ke",
							"port": 443,
							"base_url": "https://auth.apg.datacraft.co.ke",
							"weight": 100,
							"tenant_id": tenant_id,
							"registered_by": "apg-auth-service"
						}
					],
					"discovery_enabled": True,
					"health_check_enabled": True,
					"circuit_breaker_enabled": True,
					"tags": ["apg-core", "authentication", "production"]
				},
				{
					"name": "apg-config-service", 
					"display_name": "APG Configuration Service",
					"description": "Dynamic configuration management service",
					"service_type": "microservice",
					"namespace": "apg-core", 
					"environment": "production",
					"base_path": "/api/config/v1",
					"instances": [
						{
							"instance_name": "config-primary",
							"host": "config.apg.datacraft.co.ke",
							"port": 443,
							"base_url": "https://config.apg.datacraft.co.ke",
							"weight": 100,
							"tenant_id": tenant_id,
							"registered_by": "apg-config-service"
						}
					],
					"discovery_enabled": True,
					"health_check_enabled": True,
					"circuit_breaker_enabled": True,
					"tags": ["apg-core", "configuration", "production"]
				}
			]
			
			# Register each default service
			for service_data in default_services:
				try:
					await service.register_service(service_data, "system")
					app.logger.info(f"Registered default service: {service_data['name']}")
				except Exception as e:
					app.logger.warning(f"Failed to register default service {service_data['name']}: {str(e)}")
			
			app.logger.info("Default services setup completed")
			
		except Exception as e:
			app.logger.error(f"Failed to setup default services: {str(e)}")

	if hasattr(app, "before_first_request"):
		@app.before_first_request
		def setup_default_services_before_first_request():
			_run_async(setup_default_services())
	else:
		startup_state = app.extensions.setdefault(
			"regy_default_data_state",
			{"completed": False, "in_progress": False},
		)

		@app.before_request
		def setup_default_services_on_first_request():
			if startup_state["completed"] or startup_state["in_progress"]:
				return None
			startup_state["in_progress"] = True
			try:
				_run_async(setup_default_services())
			finally:
				startup_state["completed"] = True
				startup_state["in_progress"] = False
			return None

# Export initialization function
__all__ = [
	'init_app', 'get_registry_service', 'initialize_registry_service',
	'register_with_apg_composition', 'validate_apg_dependencies', 'configure_default_data'
]
