#!/usr/bin/env python3

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache
"""
Registry (regy) - APG Service Implementation
===========================================

Core service registry business logic with intelligent service discovery,
ML-powered health monitoring, and seamless APG ecosystem integration.

Author: APG Platform Team
Copyright: © 2025 Datacraft
Website: www.datacraft.co.ke
"""

import asyncio
import time
import json
import hashlib
import random
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from collections import defaultdict
import logging

from .models import (
	ServiceRegistration, ServiceInstance, ServiceDiscoveryQuery, ServiceDiscoveryResult,
	ServiceHealthStatus, ServiceEvent, ServiceMetrics, HealthCheck, CircuitBreakerConfig,
	ServiceVersion, ServiceStatus, ServiceType, HealthCheckType, CircuitBreakerState,
	LoadBalanceStrategy, ProtocolType, ValidatedVersion
)

# APG Integration Imports - Integration with existing capabilities
try:
	from ..auth.service import AuthService
	from ..conf.service import ConfigurationService  
	from ..moni.service import MonitoringService
	from ..audl.service import AuditLoggingService
	APG_INTEGRATION_AVAILABLE = True
except ImportError:
	# Fallback for development/testing without full APG platform
	AuthService = None
	ConfigurationService = None
	MonitoringService = None
	AuditLoggingService = None
	APG_INTEGRATION_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

class _ServiceMetricsStore(defaultdict):
	"""Metrics store that supports dict-style and legacy append-style writes."""

	def __init__(self):
		super().__init__(list)

	def append(self, metric: ServiceMetrics) -> None:
		self[metric.service_id].append(metric)

class ServiceRegistryService:
	"""
	Core service registry with intelligent features and APG integration.
	
	Provides comprehensive service registration, discovery, health monitoring,
	and circuit breaking with ML-powered optimization and predictive analytics.
	"""
	
	def __init__(self, tenant_id: str, config: Optional[Dict[str, Any]] = None):
		"""
		Initialize service registry with APG integration.
		
		Args:
			tenant_id: APG tenant identifier for multi-tenancy
			config: Optional configuration dictionary
		"""
		assert tenant_id, "tenant_id is required for APG multi-tenancy"
		
		self.tenant_id = tenant_id
		self.config = config or {}
		self.initialized = False
		
		# Core Data Stores - In production, these would be database-backed
		self.services: Dict[str, ServiceRegistration] = {}
		self.service_health: Dict[str, ServiceHealthStatus] = {}
		self.service_metrics: Dict[str, List[ServiceMetrics]] = _ServiceMetricsStore()
		self.service_events: List[ServiceEvent] = []
		
		# Caching and Performance
		self.discovery_cache: Dict[str, Tuple[ServiceDiscoveryResult, datetime]] = {}
		self.health_cache: Dict[str, Tuple[ServiceHealthStatus, datetime]] = {}
		self.cache_ttl_seconds = self.config.get('cache_ttl_seconds', 300)  # 5 minutes
		
		# ML/AI Features State
		self.ml_models_loaded = False
		self.anomaly_detection_enabled = self.config.get('anomaly_detection', False)
		self.predictive_scaling_enabled = self.config.get('predictive_scaling', False)
		
		# APG Service Integrations
		self.auth_service: Optional[AuthService] = None
		self.config_service: Optional[ConfigurationService] = None
		self.monitoring_service: Optional[MonitoringService] = None
		self.audit_service: Optional[AuditLoggingService] = None
		
		# Performance Counters
		self.total_registrations = 0
		self.total_discoveries = 0
		self.total_health_checks = 0
		self.cache_hits = 0
		self.cache_misses = 0
		self.start_time = datetime.now(timezone.utc)
		self.startup_time = self.start_time
		self.cache_manager = self.discovery_cache
		self.health_monitor = None
		self.circuit_breaker_manager = None
		
		self._log_initialization_start()
		
	async def initialize(self) -> None:
		"""Initialize registry with APG service integrations."""
		assert not self.initialized, "Service registry already initialized"
		
		try:
			await self._initialize_apg_integrations()
			await self._load_ml_models()
			await self._setup_health_monitoring()
			await self._initialize_circuit_breakers()
			
			self.initialized = True
			self._log_initialization_complete()
			
		except Exception as e:
			self._log_initialization_error(str(e))
			raise
	
	async def _initialize_apg_integrations(self) -> None:
		"""Initialize APG service integrations."""
		if not APG_INTEGRATION_AVAILABLE:
			self._log_apg_integration_unavailable()
			return
		
		try:
			# Initialize auth service for service authentication
			if AuthService:
				self.auth_service = AuthService(self.tenant_id)
				await self.auth_service.initialize()
				self._log_apg_auth_initialized()
			
			# Initialize configuration service for dynamic config
			if ConfigurationService:
				self.config_service = ConfigurationService(self.tenant_id)
				await self.config_service.initialize()
				self._log_apg_config_initialized()
			
			# Initialize monitoring service for metrics
			if MonitoringService:
				self.monitoring_service = MonitoringService(self.tenant_id)
				await self.monitoring_service.initialize()
				self._log_apg_monitoring_initialized()
			
			# Initialize audit service for compliance
			if AuditLoggingService:
				self.audit_service = AuditLoggingService(self.tenant_id)
				await self.audit_service.initialize()
				self._log_apg_audit_initialized()
				
		except Exception as e:
			self._log_apg_integration_error(str(e))
			# Continue without APG integration in development
	
	async def _load_ml_models(self) -> None:
		"""Load ML models for intelligent features."""
		try:
			# In production, this would load actual ML models
			# For now, simulate model loading
			await asyncio.sleep(0.1)  # Simulate model loading time
			
			self.ml_models_loaded = True
			self._log_ml_models_loaded()
			
		except Exception as e:
			self._log_ml_models_error(str(e))
			self.ml_models_loaded = False
	
	async def _setup_health_monitoring(self) -> None:
		"""Setup background health monitoring."""
		# Start background health monitoring task
		self.health_monitor = asyncio.create_task(self._health_monitoring_loop())
		self._log_health_monitoring_started()
	
	async def _initialize_circuit_breakers(self) -> None:
		"""Initialize circuit breaker management."""
		# Start background circuit breaker management
		self.circuit_breaker_manager = asyncio.create_task(self._circuit_breaker_management_loop())
		self._log_circuit_breaker_initialized()
	
	async def register_service(
		self,
		service_data: Dict[str, Any],
		created_by: str
	) -> ServiceRegistration:
		"""
		Register a new service with intelligent features.
		
		Args:
			service_data: Service registration data
			created_by: User registering the service
			
		Returns:
			ServiceRegistration: Registered service object
		"""
		assert self.initialized, "Service registry not initialized"
		assert service_data.get('name'), "Service name is required"
		assert created_by, "created_by is required for audit trail"
		
		# Add APG tenant context
		service_data['tenant_id'] = self.tenant_id
		service_data['created_by'] = created_by
		service_data['last_modified_by'] = created_by
		
		# Create service registration
		service = ServiceRegistration(**service_data)
		for instance in service.instances:
			instance.service_id = service.id
			instance.tenant_id = self.tenant_id
		
		# Validate service registration
		await self._validate_service_registration(service)
		
		# Store service
		self.services[service.id] = service
		self.total_registrations += 1
		
		# Initialize health monitoring
		await self._initialize_service_health(service)
		
		# Create audit event
		await self._create_service_event(
			event_type="service_registered",
			service_id=service.id,
			message=f"Service '{service.name}' registered successfully",
			severity="info",
			triggered_by="registry_service"
		)
		
		# Integrate with APG monitoring
		if self.monitoring_service:
			await self._report_service_metric(service.id, "registration", 1)
		
		self._log_service_registered(service.name, service.id)
		return service
	
	async def deregister_service(
		self,
		service_id: str,
		deregistered_by: str
	) -> bool:
		"""
		Deregister a service and clean up resources.
		
		Args:
			service_id: Service identifier
			deregistered_by: User deregistering the service
			
		Returns:
			bool: Success status
		"""
		assert self.initialized, "Service registry not initialized"
		assert service_id, "service_id is required"
		assert deregistered_by, "deregistered_by is required for audit trail"
		
		if service_id not in self.services:
			self._log_service_not_found(service_id)
			return False
		
		service = self.services[service_id]
		
		# Validate deregistration permissions
		await self._validate_deregistration_permissions(service, deregistered_by)
		
		# Remove from stores
		del self.services[service_id]
		self.service_health.pop(service_id, None)
		self.service_metrics.pop(service_id, None)
		
		# Clear caches
		self._clear_discovery_cache()
		
		# Create audit event
		await self._create_service_event(
			event_type="service_deregistered",
			service_id=service_id,
			message=f"Service '{service.name}' deregistered",
			severity="info",
			triggered_by="registry_service"
		)
		
		self._log_service_deregistered(service.name, service_id)
		return True
	
	async def discover_services(
		self,
		query: ServiceDiscoveryQuery
	) -> ServiceDiscoveryResult:
		"""
		Intelligent service discovery with ML-powered ranking.
		
		Args:
			query: Service discovery query parameters
			
		Returns:
			ServiceDiscoveryResult: Matching services with intelligent ranking
		"""
		assert self.initialized, "Service registry not initialized"
		if query.tenant_id != self.tenant_id:
			return ServiceDiscoveryResult(
				total_count=0,
				returned_count=0,
				query_time_ms=0.0,
				services=[],
				tenant_id=query.tenant_id
			)
		
		start_time = time.perf_counter()
		
		# Check cache first
		cache_key = self._generate_cache_key(query)
		cached_result = self._get_cached_discovery_result(cache_key)
		if cached_result:
			self._log_discovery_cache_hit(cache_key)
			return cached_result
		
		# Filter services based on query
		matching_services = await self._filter_services(query)
		
		# Apply intelligent ranking if enabled
		if query.intelligent_ranking and self.ml_models_loaded:
			matching_services = await self._rank_services_intelligently(matching_services, query)
		
		# Apply pagination
		total_count = len(matching_services)
		paginated_services = matching_services[query.offset:query.offset + query.limit]
		
		# Build result
		query_time_ms = (time.perf_counter() - start_time) * 1000
		result = ServiceDiscoveryResult(
			total_count=total_count,
			returned_count=len(paginated_services),
			query_time_ms=query_time_ms,
			services=paginated_services,
			tenant_id=self.tenant_id
		)
		
		# Cache result
		self._cache_discovery_result(cache_key, result)
		
		self.total_discoveries += 1
		self._log_service_discovery_completed(total_count, query_time_ms)
		
		return result
	
	async def get_service_health(
		self,
		service_id: str
	) -> Optional[ServiceHealthStatus]:
		"""
		Get comprehensive service health status.
		
		Args:
			service_id: Service identifier
			
		Returns:
			ServiceHealthStatus: Service health information or None
		"""
		assert self.initialized, "Service registry not initialized"
		assert service_id, "service_id is required"
		
		if service_id not in self.services:
			return None
		
		# Check cache
		cached_health = self._get_cached_health_status(service_id)
		if cached_health:
			return cached_health
		
		# Get fresh health status
		health_status = await self._compute_service_health(service_id)
		
		# Cache result
		self._cache_health_status(service_id, health_status)
		
		return health_status
	
	async def update_service_health(
		self,
		service_id: str,
		health_data: Dict[str, Any]
	) -> bool:
		"""
		Update service health information.
		
		Args:
			service_id: Service identifier
			health_data: Health status data
			
		Returns:
			bool: Success status
		"""
		assert self.initialized, "Service registry not initialized"
		assert service_id, "service_id is required"
		
		if service_id not in self.services:
			return False
		
		service = self.services[service_id]
		
		# Update health data
		health_data['tenant_id'] = self.tenant_id
		health_data['service_id'] = service_id
		
		health_status = ServiceHealthStatus(**health_data)
		self.service_health[service_id] = health_status
		
		# Clear health cache
		self._clear_health_cache(service_id)
		
		# Check for status changes
		await self._check_health_status_changes(service, health_status)
		
		self.total_health_checks += 1
		return True
	
	async def get_service_metrics(
		self,
		service_id: str,
		time_range_hours: int = 24
	) -> List[ServiceMetrics]:
		"""
		Get service performance metrics.
		
		Args:
			service_id: Service identifier
			time_range_hours: Time range for metrics
			
		Returns:
			List[ServiceMetrics]: Service metrics
		"""
		assert self.initialized, "Service registry not initialized"
		assert service_id, "service_id is required"
		
		if service_id not in self.services:
			return []
		
		# Filter metrics by time range
		cutoff_time = datetime.now(timezone.utc) - timedelta(hours=time_range_hours)
		metrics = [
			m for m in self.service_metrics.get(service_id, [])
			if m.timestamp >= cutoff_time
		]
		
		return sorted(metrics, key=lambda x: x.timestamp, reverse=True)
	
	async def _validate_service_registration(self, service: ServiceRegistration) -> None:
		"""Validate service registration data."""
		# Check for duplicate names in same namespace
		for existing_service in self.services.values():
			if (existing_service.name == service.name and 
				existing_service.namespace == service.namespace and
				existing_service.tenant_id == service.tenant_id):
				raise ValueError(f"Service '{service.name}' is already registered in namespace '{service.namespace}'")
		
		# Validate service instances
		for instance in service.instances:
			assert instance.tenant_id == self.tenant_id, "Instance tenant mismatch"
			await self._validate_service_instance(instance)
		
		# Use APG auth for additional validation if available
		if self.auth_service:
			await self._validate_service_permissions(service)
	
	async def _validate_service_instance(self, instance: ServiceInstance) -> None:
		"""Validate service instance configuration."""
		# Check port availability (simplified)
		if instance.port < 1 or instance.port > 65535:
			raise ValueError(f"Invalid port number: {instance.port}")
		
		# Validate health checks
		for health_check in instance.health_checks:
			if health_check.type == HealthCheckType.HTTP and not health_check.url:
				raise ValueError("HTTP health check requires URL")
	
	async def _validate_service_permissions(self, service: ServiceRegistration) -> None:
		"""Validate service registration permissions using APG auth."""
		try:
			# Check if user has permission to register services
			has_permission = await self.auth_service.check_permission(
				service.created_by,
				"service_registry:register",
				{"namespace": service.namespace}
			)
			
			if not has_permission:
				raise PermissionError(f"User '{service.created_by}' does not have permission to register services")
				
		except Exception as e:
			self._log_permission_validation_error(str(e))
			# In development, continue without strict permission checking
	
	async def _validate_deregistration_permissions(
		self,
		service: ServiceRegistration,
		deregistered_by: str
	) -> None:
		"""Validate service deregistration permissions."""
		if self.auth_service:
			has_permission = await self.auth_service.check_permission(
				deregistered_by,
				"service_registry:deregister",
				{"service_id": service.id}
			)
			
			if not has_permission:
				raise PermissionError(f"User '{deregistered_by}' does not have permission to deregister service")
	
	async def _initialize_service_health(self, service: ServiceRegistration) -> None:
		"""Initialize health monitoring for a new service."""
		for instance in service.instances:
			health_status = ServiceHealthStatus(
				service_id=service.id,
				instance_id=instance.id,
				overall_status=ServiceStatus.STARTING,
				health_score=0.5,  # Initial neutral score
				status_message="Service starting",
				response_time_ms=0.0,
				cpu_usage_percent=0.0,
				memory_usage_percent=0.0,
				active_connections=0,
				circuit_breaker_state=CircuitBreakerState.CLOSED,
				failure_count=0,
				tenant_id=self.tenant_id
			)
			
			self.service_health[f"{service.id}:{instance.id}"] = health_status
	
	async def _filter_services(
		self,
		query: ServiceDiscoveryQuery
	) -> List[ServiceRegistration]:
		"""Filter services based on query parameters."""
		matching_services = []
		
		for service in self.services.values():
			# Tenant isolation
			if service.tenant_id != query.tenant_id:
				continue
			
			# Name filter
			if query.service_name and query.service_name not in service.name:
				continue
			
			# Type filter
			if query.service_type and service.service_type != query.service_type:
				continue
			
			# Namespace filter
			if query.namespace and service.namespace != query.namespace:
				continue
			
			# Environment filter
			if query.environment and service.environment != query.environment:
				continue
			
			# Status filter
			if query.status and service.status != query.status:
				continue
			
			# Health filter
			if query.healthy_only:
				health_status = await self.get_service_health(service.id)
				if health_status and health_status.overall_status in [
					ServiceStatus.UNHEALTHY,
					ServiceStatus.CRITICAL,
					ServiceStatus.STOPPED
				]:
					continue
			
			# Health score filter
			if query.min_health_score > 0.0:
				health_status = await self.get_service_health(service.id)
				if not health_status or health_status.health_score < query.min_health_score:
					continue
			
			# Tag filter
			if query.tags:
				if not all(tag in service.tags for tag in query.tags):
					continue
			
			# Label filter
			if query.labels:
				if not all(service.labels.get(k) == v for k, v in query.labels.items()):
					continue
			
			matching_services.append(service)
		
		return matching_services
	
	async def _rank_services_intelligently(
		self,
		services: List[ServiceRegistration],
		query: ServiceDiscoveryQuery
	) -> List[ServiceRegistration]:
		"""Apply ML-powered intelligent ranking to services."""
		if not self.ml_models_loaded:
			return services
		
		# Simplified ML ranking - in production, this would use actual ML models
		scored_services = []
		
		for service in services:
			score = 1.0  # Base score
			
			# Health score weight (40%)
			health_status = await self.get_service_health(service.id)
			if health_status:
				score *= (0.6 + 0.4 * health_status.health_score)
			
			# Performance weight (30%)
			if service.average_response_time > 0:
				# Lower response time = higher score
				response_score = max(0.1, 1.0 - (service.average_response_time / 1000.0))
				score *= (0.7 + 0.3 * response_score)
			
			# Availability weight (20%)
			score *= (0.8 + 0.2 * (service.uptime_percentage / 100.0))
			
			# Load weight (10%)
			total_instances = len(service.instances)
			if total_instances > 0:
				avg_connections = sum(i.current_connections for i in service.instances) / total_instances
				max_connections = max((i.max_connections or 100) for i in service.instances)
				load_factor = 1.0 - (avg_connections / max_connections) if max_connections > 0 else 1.0
				score *= (0.9 + 0.1 * load_factor)
			
			scored_services.append((service, score))
		
		# Sort by score descending
		scored_services.sort(key=lambda x: x[1], reverse=True)
		return [service for service, _ in scored_services]
	
	async def _compute_service_health(self, service_id: str) -> ServiceHealthStatus:
		"""Compute comprehensive service health status."""
		service = self.services[service_id]
		
		# Start with neutral health
		overall_health_score = 0.5
		status_message = "Computing health status"
		overall_status = ServiceStatus.UNKNOWN
		
		# Aggregate instance health
		instance_health_scores = []
		
		for instance in service.instances:
			instance_key = f"{service_id}:{instance.id}"
			if instance_key in self.service_health:
				instance_health = self.service_health[instance_key]
				instance_health_scores.append(instance_health.health_score)
		
		if instance_health_scores:
			overall_health_score = sum(instance_health_scores) / len(instance_health_scores)
			
			# Determine overall status based on health score
			if overall_health_score >= 0.8:
				overall_status = ServiceStatus.HEALTHY
				status_message = "Service is healthy"
			elif overall_health_score >= 0.5:
				overall_status = ServiceStatus.DEGRADED  
				status_message = "Service is degraded"
			elif overall_health_score >= 0.3:
				overall_status = ServiceStatus.UNHEALTHY
				status_message = "Service is unhealthy"
			else:
				overall_status = ServiceStatus.CRITICAL
				status_message = "Service is critical"
		
		return ServiceHealthStatus(
			service_id=service_id,
			instance_id="aggregate",
			overall_status=overall_status,
			health_score=overall_health_score,
			status_message=status_message,
			response_time_ms=service.average_response_time,
			cpu_usage_percent=50.0,  # Placeholder
			memory_usage_percent=60.0,  # Placeholder
			active_connections=sum(i.current_connections for i in service.instances),
			circuit_breaker_state=CircuitBreakerState.CLOSED,  # Placeholder
			failure_count=service.total_errors,
			tenant_id=self.tenant_id
		)
	
	async def _check_health_status_changes(
		self,
		service: ServiceRegistration,
		health_status: ServiceHealthStatus
	) -> None:
		"""Check for health status changes and trigger events."""
		previous_health = self.service_health.get(f"{service.id}:aggregate")
		
		if previous_health and previous_health.overall_status != health_status.overall_status:
			# Status changed - create event
			await self._create_service_event(
				event_type="health_status_changed",
				service_id=service.id,
				message=f"Service health changed from {previous_health.overall_status} to {health_status.overall_status}",
				severity="warning" if health_status.overall_status in [ServiceStatus.UNHEALTHY, ServiceStatus.CRITICAL] else "info",
				triggered_by="health_monitor",
				previous_state=previous_health.overall_status,
				new_state=health_status.overall_status
			)
	
	async def _create_service_event(
		self,
		event_type: str,
		service_id: str,
		message: str,
		severity: str,
		triggered_by: str,
		instance_id: Optional[str] = None,
		previous_state: Optional[str] = None,
		new_state: Optional[str] = None
	) -> ServiceEvent:
		"""Create and store service event."""
		event = ServiceEvent(
			event_type=event_type,
			service_id=service_id,
			instance_id=instance_id,
			severity=severity,
			message=message,
			previous_state=previous_state,
			new_state=new_state,
			triggered_by=triggered_by,
			tenant_id=self.tenant_id,
			created_by=triggered_by
		)
		
		self.service_events.append(event)
		
		# Integrate with APG audit logging
		if self.audit_service:
			await self.audit_service.log_event(
				event_type=event_type,
				resource_id=service_id,
				message=message,
				severity=severity,
				user_id=triggered_by,
				metadata={
					"instance_id": instance_id,
					"previous_state": previous_state,
					"new_state": new_state
				}
			)
		
		return event
	
	async def _report_service_metric(
		self,
		service_id: str,
		metric_name: str,
		value: Union[int, float]
	) -> None:
		"""Report service metric to APG monitoring."""
		if self.monitoring_service:
			await self.monitoring_service.record_metric(
				metric_name=f"registry.service.{metric_name}",
				value=value,
				tags={
					"service_id": service_id,
					"tenant_id": self.tenant_id
				}
			)
	
	async def _health_monitoring_loop(self) -> None:
		"""Background health monitoring loop."""
		while True:
			try:
				await self._perform_health_checks()
				await asyncio.sleep(30)  # Check every 30 seconds
			except Exception as e:
				self._log_health_monitoring_error(str(e))
				await asyncio.sleep(60)  # Back off on error
	
	async def _perform_health_checks(self) -> None:
		"""Perform health checks on all registered services."""
		for service_id, service in self.services.items():
			try:
				# Update health status for each service
				health_status = await self._compute_service_health(service_id)
				self.service_health[f"{service_id}:aggregate"] = health_status
				
				# Clear cache to force fresh data
				self._clear_health_cache(service_id)
				
			except Exception as e:
				self._log_health_check_error(service_id, str(e))
	
	async def _circuit_breaker_management_loop(self) -> None:
		"""Background circuit breaker management loop."""
		while True:
			try:
				await self._manage_circuit_breakers()
				await asyncio.sleep(10)  # Check every 10 seconds
			except Exception as e:
				self._log_circuit_breaker_error(str(e))
				await asyncio.sleep(30)  # Back off on error
	
	async def _manage_circuit_breakers(self) -> None:
		"""Manage circuit breaker states for all services."""
		for service in self.services.values():
			for instance in service.instances:
				for cb_config in instance.circuit_breakers:
					await self._update_circuit_breaker_state(cb_config, service, instance)
	
	async def _update_circuit_breaker_state(
		self,
		cb_config: CircuitBreakerConfig,
		service: ServiceRegistration,
		instance: ServiceInstance
	) -> None:
		"""Update circuit breaker state based on current conditions."""
		current_time = datetime.now(timezone.utc)
		
		# Get current health status
		health_key = f"{service.id}:{instance.id}"
		health_status = self.service_health.get(health_key)
		
		if not health_status:
			return
		
		# Update circuit breaker state based on health
		if cb_config.state == CircuitBreakerState.CLOSED:
			# Check if we should open
			if health_status.failure_count >= cb_config.failure_threshold:
				cb_config.state = CircuitBreakerState.OPEN
				cb_config.last_failure_time = current_time
				await self._create_service_event(
					event_type="circuit_breaker_opened",
					service_id=service.id,
					instance_id=instance.id,
					message=f"Circuit breaker opened due to {health_status.failure_count} failures",
					severity="warning",
					triggered_by="circuit_breaker"
				)
		
		elif cb_config.state == CircuitBreakerState.OPEN:
			# Check if we should try half-open
			if (cb_config.last_failure_time and
				current_time - cb_config.last_failure_time > timedelta(seconds=cb_config.timeout_seconds)):
				cb_config.state = CircuitBreakerState.HALF_OPEN
				await self._create_service_event(
					event_type="circuit_breaker_half_open",
					service_id=service.id,
					instance_id=instance.id,
					message="Circuit breaker moved to half-open state",
					severity="info",
					triggered_by="circuit_breaker"
				)
		
		elif cb_config.state == CircuitBreakerState.HALF_OPEN:
			# Check if we should close or open
			if health_status.overall_status == ServiceStatus.HEALTHY:
				cb_config.state = CircuitBreakerState.CLOSED
				cb_config.failed_requests = 0
				await self._create_service_event(
					event_type="circuit_breaker_closed",
					service_id=service.id,
					instance_id=instance.id,
					message="Circuit breaker closed - service recovered",
					severity="info",
					triggered_by="circuit_breaker"
				)
			elif health_status.failure_count > 0:
				cb_config.state = CircuitBreakerState.OPEN
				cb_config.last_failure_time = current_time
	
	def _generate_cache_key(self, query: ServiceDiscoveryQuery) -> str:
		"""Generate cache key for discovery query."""
		query_dict = query.model_dump()
		query_str = json.dumps(query_dict, sort_keys=True)
		return hashlib.md5(query_str.encode()).hexdigest()
	
	def _get_cached_discovery_result(self, cache_key: str) -> Optional[ServiceDiscoveryResult]:
		"""Get cached discovery result if valid."""
		if cache_key in self.discovery_cache:
			result, cached_time = self.discovery_cache[cache_key]
			if datetime.now(timezone.utc) - cached_time < timedelta(seconds=self.cache_ttl_seconds):
				self.cache_hits += 1
				result.cached_result = True
				return result
			else:
				del self.discovery_cache[cache_key]
		self.cache_misses += 1
		return None
	
	def _cache_discovery_result(self, cache_key: str, result: ServiceDiscoveryResult) -> None:
		"""Cache discovery result."""
		self.discovery_cache[cache_key] = (result, datetime.now(timezone.utc))
	
	def _clear_discovery_cache(self) -> None:
		"""Clear all discovery cache entries."""
		self.discovery_cache.clear()
	
	def _get_cached_health_status(self, service_id: str) -> Optional[ServiceHealthStatus]:
		"""Get cached health status if valid."""
		cache_key = f"health:{service_id}"
		if cache_key in self.health_cache:
			health_status, cached_time = self.health_cache[cache_key]
			if datetime.now(timezone.utc) - cached_time < timedelta(seconds=60):  # 1 minute TTL for health
				self.cache_hits += 1
				return health_status
			else:
				del self.health_cache[cache_key]
		self.cache_misses += 1
		return None
	
	def _cache_health_status(self, service_id: str, health_status: ServiceHealthStatus) -> None:
		"""Cache health status."""
		cache_key = f"health:{service_id}"
		self.health_cache[cache_key] = (health_status, datetime.now(timezone.utc))
	
	def _clear_health_cache(self, service_id: str) -> None:
		"""Clear health cache for specific service."""
		cache_key = f"health:{service_id}"
		self.health_cache.pop(cache_key, None)
	
	async def get_registry_statistics(self) -> Dict[str, Any]:
		"""Get comprehensive registry statistics."""
		uptime_seconds = (datetime.now(timezone.utc) - self.start_time).total_seconds()
		
		# Service statistics
		service_stats = {
			"total_services": len(self.services),
			"services_by_type": defaultdict(int),
			"services_by_status": defaultdict(int),
			"services_by_environment": defaultdict(int),
			"total_instances": 0,
			"healthy_services": 0,
			"degraded_services": 0,
			"unhealthy_services": 0
		}
		
		for service in self.services.values():
			service_stats["services_by_type"][service.service_type] += 1
			service_stats["services_by_status"][service.status] += 1
			service_stats["services_by_environment"][service.environment] += 1
			service_stats["total_instances"] += len(service.instances)
			
			# Get health status
			health_status = await self.get_service_health(service.id)
			if health_status:
				if health_status.overall_status == ServiceStatus.HEALTHY:
					service_stats["healthy_services"] += 1
				elif health_status.overall_status == ServiceStatus.DEGRADED:
					service_stats["degraded_services"] += 1
				elif health_status.overall_status in [ServiceStatus.UNHEALTHY, ServiceStatus.CRITICAL]:
					service_stats["unhealthy_services"] += 1
		
		return {
			"registry_info": {
				"tenant_id": self.tenant_id,
				"uptime_seconds": uptime_seconds,
				"initialized": self.initialized,
				"ml_models_loaded": self.ml_models_loaded,
				"apg_integration_available": APG_INTEGRATION_AVAILABLE
			},
			"service_statistics": dict(service_stats),
			"performance_counters": {
				"total_registrations": self.total_registrations,
				"total_discoveries": self.total_discoveries,
				"total_health_checks": self.total_health_checks,
				"cache_hit_rate": self._calculate_cache_hit_rate(),
				"discovery_cache_size": len(self.discovery_cache),
				"health_cache_size": len(self.health_cache)
			},
			"events_summary": {
				"total_events": len(self.service_events),
				"recent_events": len([e for e in self.service_events 
									if e.timestamp > datetime.now(timezone.utc) - timedelta(hours=1)])
			}
		}
	
	def _calculate_cache_hit_rate(self) -> float:
		"""Calculate cache hit rate from observed cache lookups."""
		total_cache_lookups = self.cache_hits + self.cache_misses
		if total_cache_lookups == 0:
			return 0.0
		return self.cache_hits / total_cache_lookups
	
	# Logging Methods - APG CLAUDE.md standard
	def _log_initialization_start(self) -> None:
		"""Log registry initialization start."""
		logger.info(f"Registry [tenant:{self.tenant_id}] initialization starting")
	
	def _log_initialization_complete(self) -> None:
		"""Log registry initialization completion."""
		logger.info(f"Registry [tenant:{self.tenant_id}] initialization completed successfully")
	
	def _log_initialization_error(self, error: str) -> None:
		"""Log registry initialization error."""
		logger.error(f"Registry [tenant:{self.tenant_id}] initialization failed: {error}")
	
	def _log_apg_integration_unavailable(self) -> None:
		"""Log APG integration unavailable."""
		logger.warning(f"Registry [tenant:{self.tenant_id}] APG integration not available - running in standalone mode")
	
	def _log_apg_auth_initialized(self) -> None:
		"""Log APG auth service initialized."""
		logger.info(f"Registry [tenant:{self.tenant_id}] APG auth service integration initialized")
	
	def _log_apg_config_initialized(self) -> None:
		"""Log APG config service initialized."""
		logger.info(f"Registry [tenant:{self.tenant_id}] APG configuration service integration initialized")
	
	def _log_apg_monitoring_initialized(self) -> None:
		"""Log APG monitoring service initialized.""" 
		logger.info(f"Registry [tenant:{self.tenant_id}] APG monitoring service integration initialized")
	
	def _log_apg_audit_initialized(self) -> None:
		"""Log APG audit service initialized."""
		logger.info(f"Registry [tenant:{self.tenant_id}] APG audit service integration initialized")
	
	def _log_apg_integration_error(self, error: str) -> None:
		"""Log APG integration error."""
		logger.warning(f"Registry [tenant:{self.tenant_id}] APG integration error: {error} - continuing without full integration")
	
	def _log_ml_models_loaded(self) -> None:
		"""Log ML models loaded."""
		logger.info(f"Registry [tenant:{self.tenant_id}] ML models loaded successfully")
	
	def _log_ml_models_error(self, error: str) -> None:
		"""Log ML models loading error."""
		logger.warning(f"Registry [tenant:{self.tenant_id}] ML models loading failed: {error} - intelligent features disabled")
	
	def _log_health_monitoring_started(self) -> None:
		"""Log health monitoring started."""
		logger.info(f"Registry [tenant:{self.tenant_id}] health monitoring background task started")
	
	def _log_circuit_breaker_initialized(self) -> None:
		"""Log circuit breaker initialized."""
		logger.info(f"Registry [tenant:{self.tenant_id}] circuit breaker management initialized")
	
	def _log_service_registered(self, service_name: str, service_id: str) -> None:
		"""Log service registration."""
		logger.info(f"Registry [tenant:{self.tenant_id}] service '{service_name}' registered with ID {service_id}")
	
	def _log_service_deregistered(self, service_name: str, service_id: str) -> None:
		"""Log service deregistration."""
		logger.info(f"Registry [tenant:{self.tenant_id}] service '{service_name}' deregistered (ID: {service_id})")
	
	def _log_service_not_found(self, service_id: str) -> None:
		"""Log service not found."""
		logger.warning(f"Registry [tenant:{self.tenant_id}] service not found: {service_id}")
	
	def _log_service_discovery_completed(self, count: int, query_time_ms: float) -> None:
		"""Log service discovery completion."""
		logger.info(f"Registry [tenant:{self.tenant_id}] discovery completed: {count} services found in {query_time_ms:.2f}ms")
	
	def _log_discovery_cache_hit(self, cache_key: str) -> None:
		"""Log discovery cache hit."""
		logger.debug(f"Registry [tenant:{self.tenant_id}] discovery cache hit for key: {cache_key}")
	
	def _log_permission_validation_error(self, error: str) -> None:
		"""Log permission validation error."""
		logger.warning(f"Registry [tenant:{self.tenant_id}] permission validation error: {error}")
	
	def _log_health_monitoring_error(self, error: str) -> None:
		"""Log health monitoring error."""
		logger.error(f"Registry [tenant:{self.tenant_id}] health monitoring error: {error}")
	
	def _log_health_check_error(self, service_id: str, error: str) -> None:
		"""Log health check error."""
		logger.error(f"Registry [tenant:{self.tenant_id}] health check error for service {service_id}: {error}")
	
	def _log_circuit_breaker_error(self, error: str) -> None:
		"""Log circuit breaker management error."""
		logger.error(f"Registry [tenant:{self.tenant_id}] circuit breaker management error: {error}")

	# ── Extended methods ───────────────────────────────────────────────────────

	async def service_register(
		self,
		service_data: dict[str, Any],
		created_by: str,
	) -> dict[str, Any]:
		"""Spec alias for register_service; returns dict."""
		result = await self.register_service(service_data, created_by)
		return result.model_dump() if hasattr(result, "model_dump") else vars(result)

	async def service_deregister(self, service_id: str, deregistered_by: str) -> dict[str, Any]:
		"""Spec alias for deregister_service."""
		success = await self.deregister_service(service_id, deregistered_by)
		return {"service_id": service_id, "deregistered": success, "deregistered_by": deregistered_by}

	async def health_check_service(self, service_id: str) -> dict[str, Any]:
		"""Return health status dict for a service."""
		health = await self.get_service_health(service_id)
		if health is None:
			return {"service_id": service_id, "status": "not_found"}
		return health.model_dump() if hasattr(health, "model_dump") else vars(health)

	async def service_discover(self, query: Any) -> dict[str, Any]:
		"""Spec alias for discover_services."""
		result = await self.discover_services(query)
		return result.model_dump() if hasattr(result, "model_dump") else vars(result)

	async def metadata_update(
		self,
		service_id: str,
		metadata: dict[str, Any],
		updated_by: str,
	) -> dict[str, Any]:
		"""Update arbitrary metadata on a registered service."""
		assert self.initialized, "Service registry not initialized"
		if service_id not in self.services:
			return {"service_id": service_id, "updated": False}
		service = self.services[service_id]
		service.labels.update(metadata)
		service.last_modified_by = updated_by
		self._clear_discovery_cache()
		await self._create_service_event(
			event_type="metadata_updated",
			service_id=service_id,
			message=f"Metadata updated by {updated_by}",
			severity="info",
			triggered_by="registry_service",
		)
		return {"service_id": service_id, "updated": True, "keys": list(metadata.keys())}

	async def tag_add(
		self,
		service_id: str,
		tags: list[str],
		updated_by: str,
	) -> dict[str, Any]:
		"""Add tags to a registered service."""
		assert self.initialized, "Service registry not initialized"
		if service_id not in self.services:
			return {"service_id": service_id, "updated": False}
		service = self.services[service_id]
		service.tags = list(set(service.tags) | set(tags))
		service.last_modified_by = updated_by
		self._clear_discovery_cache()
		return {"service_id": service_id, "tags": service.tags}

	async def version_track(
		self,
		service_id: str,
		version: str,
		updated_by: str,
	) -> dict[str, Any]:
		"""Record a version string against a service."""
		assert self.initialized, "Service registry not initialized"
		if service_id not in self.services:
			return {"service_id": service_id, "tracked": False}
		service = self.services[service_id]
		service.version = version  # type: ignore[attr-defined]
		service.last_modified_by = updated_by
		self._clear_discovery_cache()
		await self._create_service_event(
			event_type="version_tracked",
			service_id=service_id,
			message=f"Version set to {version} by {updated_by}",
			severity="info",
			triggered_by="registry_service",
		)
		return {"service_id": service_id, "version": version}

	async def dependency_graph(self, service_id: str) -> dict[str, Any]:
		"""Return a dependency graph rooted at service_id (label-based)."""
		assert self.initialized, "Service registry not initialized"
		if service_id not in self.services:
			return {"service_id": service_id, "dependencies": []}
		service = self.services[service_id]
		deps = service.labels.get("dependencies", [])
		if isinstance(deps, str):
			deps = [d.strip() for d in deps.split(",") if d.strip()]
		return {"service_id": service_id, "service_name": service.name, "dependencies": deps}

	async def capability_search(
		self,
		capability: str,
		environment: str | None = None,
	) -> list[dict[str, Any]]:
		"""Find services that advertise a given capability tag."""
		assert self.initialized, "Service registry not initialized"
		results: list[dict[str, Any]] = []
		for service in self.services.values():
			if service.tenant_id != self.tenant_id:
				continue
			if environment and service.environment != environment:
				continue
			if capability in service.tags or capability in service.labels.get("capabilities", []):
				results.append({"id": service.id, "name": service.name, "namespace": service.namespace, "environment": service.environment})
		return results

	async def registry_export(self) -> dict[str, Any]:
		"""Export all services and events for the tenant."""
		assert self.initialized, "Service registry not initialized"
		services: list[dict[str, Any]] = []
		for service in self.services.values():
			if service.tenant_id == self.tenant_id:
				services.append({
					"id": service.id,
					"name": service.name,
					"namespace": service.namespace,
					"status": service.status,
					"tags": service.tags,
				})
		return {
			"tenant_id": self.tenant_id,
			"service_count": len(services),
			"services": services,
			"event_count": len(self.service_events),
			"exported_at": datetime.now(timezone.utc).isoformat(),
		}

	async def registry_import(
		self,
		services_data: list[dict[str, Any]],
		imported_by: str,
	) -> dict[str, Any]:
		"""Bulk-import services from an export payload."""
		imported: list[str] = []
		skipped: list[str] = []
		for svc in services_data:
			svc["tenant_id"] = self.tenant_id
			svc["created_by"] = imported_by
			svc["last_modified_by"] = imported_by
			name = svc.get("name", "")
			ns = svc.get("namespace", "default")
			already_exists = any(
				s.name == name and s.namespace == ns and s.tenant_id == self.tenant_id
				for s in self.services.values()
			)
			if already_exists:
				skipped.append(name)
			else:
				try:
					result = await self.register_service(svc, imported_by)
					imported.append(result.id if hasattr(result, "id") else str(result))
				except Exception:
					skipped.append(name)
		return {"imported": imported, "skipped": skipped, "imported_by": imported_by}

	async def change_notify(
		self,
		service_id: str,
		change_type: str,
		changed_by: str,
		payload: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Record a change notification event for a service."""
		assert self.initialized, "Service registry not initialized"
		event = await self._create_service_event(
			event_type=f"change_notification:{change_type}",
			service_id=service_id,
			message=f"Change '{change_type}' notified by {changed_by}",
			severity="info",
			triggered_by=changed_by,
		)
		return event.model_dump() if hasattr(event, "model_dump") else vars(event)

	async def registry_analytics(self) -> dict[str, Any]:
		"""Alias for get_registry_statistics."""
		return await self.get_registry_statistics()

	async def federation_registry(
		self,
		remote_tenant_id: str,
		remote_services: list[dict[str, Any]],
		federated_by: str,
	) -> dict[str, Any]:
		"""Federate services from a remote tenant into read-only local entries."""
		federated: list[str] = []
		for svc in remote_services:
			svc_copy = dict(svc)
			svc_copy["tenant_id"] = self.tenant_id
			svc_copy["namespace"] = f"federated:{remote_tenant_id}"
			svc_copy["tags"] = list(set(svc_copy.get("tags", [])) | {"federated", f"remote_tenant:{remote_tenant_id}"})
			svc_copy["created_by"] = federated_by
			svc_copy["last_modified_by"] = federated_by
			try:
				result = await self.register_service(svc_copy, federated_by)
				federated.append(result.id if hasattr(result, "id") else str(result))
			except Exception as _exc:
				_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)
		return {"remote_tenant_id": remote_tenant_id, "federated_count": len(federated), "federated_ids": federated}

	async def namespace_manage(
		self,
		namespace: str,
		action: str = "list",
		updated_by: str = "system",
	) -> dict[str, Any]:
		"""List or create a namespace (action: 'list' | 'create' | 'delete')."""
		assert self.initialized, "Service registry not initialized"
		services_in_ns = [s for s in self.services.values() if s.tenant_id == self.tenant_id and s.namespace == namespace]
		if action == "delete":
			if services_in_ns:
				raise ValueError(f"namespace_not_empty:{namespace}")
			return {"namespace": namespace, "action": "deleted", "service_count": 0}
		return {
			"namespace": namespace,
			"action": action,
			"service_count": len(services_in_ns),
			"services": [{"id": s.id, "name": s.name, "status": s.status} for s in services_in_ns],
		}
