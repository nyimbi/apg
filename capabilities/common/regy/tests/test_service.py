#!/usr/bin/env python3
"""
Registry (regy) - Service Test Suite
====================================

Comprehensive tests for ServiceRegistryService business logic with
edge cases, error conditions, and APG integration scenarios.

Author: APG Platform Team
Copyright: © 2025 Datacraft
"""

import asyncio
import pytest
import pytest_asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any
from unittest.mock import Mock, patch, AsyncMock

from ..service import ServiceRegistryService
from ..models import (
	ServiceRegistration, ServiceInstance, ServiceDiscoveryQuery, ServiceDiscoveryResult,
	ServiceHealthStatus, ServiceEvent, ServiceMetrics, ServiceStatus, ServiceType,
	HealthCheckType, CircuitBreakerState, LoadBalanceStrategy, HealthCheck, CircuitBreakerConfig,
	ServiceVersion, ServiceEndpoint
)

from . import TEST_TENANT_ID, TEST_USER_ID, TEST_SERVICE_NAME

class TestServiceRegistryService:
	"""Test ServiceRegistryService core functionality."""
	
	@pytest_asyncio.fixture
	async def service(self):
		"""Create a fresh service instance for testing."""
		svc = ServiceRegistryService(TEST_TENANT_ID)
		await svc.initialize()
		return svc
	
	async def test_service_initialization(self):
		"""Test service initialization process."""
		service = ServiceRegistryService(TEST_TENANT_ID)
		
		# Should not be initialized yet
		assert not service.initialized
		assert service.tenant_id == TEST_TENANT_ID
		assert len(service.services) == 0
		assert len(service.service_health) == 0
		assert len(service.service_metrics) == 0
		assert len(service.service_events) == 0
		
		# Initialize service
		await service.initialize()
		
		# Should be initialized now
		assert service.initialized
		assert service.startup_time is not None
		assert service.cache_manager is not None
		assert service.circuit_breaker_manager is not None
		assert service.health_monitor is not None
	
	async def test_service_registration_basic(self, service):
		"""Test basic service registration."""
		service_data = {
			"name": TEST_SERVICE_NAME,
			"display_name": "Test Service",
			"description": "A test service",
			"service_type": ServiceType.REST_API,
			"namespace": "test",
			"environment": "development",
			"base_path": "/api/v1",
			"tags": ["test", "api"]
		}
		
		registered_service = await service.register_service(service_data, TEST_USER_ID)
		
		# Validate registration
		assert registered_service.name == TEST_SERVICE_NAME
		assert registered_service.display_name == "Test Service"
		assert registered_service.service_type == ServiceType.REST_API
		assert registered_service.namespace == "test"
		assert registered_service.environment == "development"
		assert registered_service.created_by == TEST_USER_ID
		assert registered_service.tenant_id == TEST_TENANT_ID
		assert len(registered_service.tags) == 2
		
		# Should be in services dict
		assert registered_service.id in service.services
		assert service.services[registered_service.id] == registered_service
		
		# Should have generated an event
		events = [e for e in service.service_events if e.service_id == registered_service.id]
		assert len(events) >= 1
		assert any(e.event_type == "service_registered" for e in events)
	
	async def test_service_registration_with_instances(self, service):
		"""Test service registration with instances."""
		service_data = {
			"name": "service-with-instances",
			"display_name": "Service with Instances",
			"service_type": ServiceType.MICROSERVICE,
			"namespace": "prod",
			"environment": "production",
			"instances": [{
				"service_id": "will-be-set",
				"instance_name": "primary",
				"host": "service.example.com",
				"port": 8080,
				"base_url": "http://service.example.com:8080",
				"weight": 100,
				"tenant_id": TEST_TENANT_ID,
				"registered_by": TEST_USER_ID
			}]
		}
		
		registered_service = await service.register_service(service_data, TEST_USER_ID)
		
		# Should have instances
		assert len(registered_service.instances) == 1
		instance = registered_service.instances[0]
		assert instance.service_id == registered_service.id
		assert instance.instance_name == "primary"
		assert instance.host == "service.example.com"
		assert instance.port == 8080
		assert instance.weight == 100
	
	async def test_service_registration_duplicate_name(self, service):
		"""Test registration fails for duplicate service names."""
		service_data = {
			"name": "duplicate-service",
			"display_name": "First Service",
			"service_type": ServiceType.REST_API,
			"namespace": "test",
			"environment": "development"
		}
		
		# Register first service
		first_service = await service.register_service(service_data, TEST_USER_ID)
		assert first_service is not None
		
		# Try to register duplicate
		with pytest.raises(ValueError) as exc_info:
			await service.register_service(service_data, TEST_USER_ID)
		
		assert "already registered" in str(exc_info.value).lower()
	
	async def test_service_deregistration(self, service):
		"""Test service deregistration."""
		# Register a service first
		service_data = {
			"name": "temp-service",
			"display_name": "Temporary Service",
			"service_type": ServiceType.REST_API,
			"namespace": "test",
			"environment": "development"
		}
		
		registered_service = await service.register_service(service_data, TEST_USER_ID)
		service_id = registered_service.id
		
		# Verify it's registered
		assert service_id in service.services
		
		# Deregister it
		success = await service.deregister_service(service_id, TEST_USER_ID)
		assert success is True
		
		# Should no longer be in services
		assert service_id not in service.services
		
		# Should have generated an event
		events = [e for e in service.service_events if e.service_id == service_id]
		deregister_events = [e for e in events if e.event_type == "service_deregistered"]
		assert len(deregister_events) >= 1
	
	async def test_service_deregistration_nonexistent(self, service):
		"""Test deregistration of non-existent service."""
		success = await service.deregister_service("non-existent-id", TEST_USER_ID)
		assert success is False
	
	async def test_service_discovery_basic(self, service):
		"""Test basic service discovery."""
		# Register multiple services
		services_data = [
			{
				"name": "api-service",
				"display_name": "API Service",
				"service_type": ServiceType.REST_API,
				"namespace": "prod",
				"environment": "production",
				"tags": ["api", "production"]
			},
			{
				"name": "web-service", 
				"display_name": "Web Service",
				"service_type": ServiceType.WEB_SERVICE,
				"namespace": "prod",
				"environment": "production",
				"tags": ["web", "production"]
			},
			{
				"name": "test-service",
				"display_name": "Test Service", 
				"service_type": ServiceType.REST_API,
				"namespace": "test",
				"environment": "development",
				"tags": ["test", "development"]
			}
		]
		
		for svc_data in services_data:
			await service.register_service(svc_data, TEST_USER_ID)
		
		# Test discovery by service type
		query = ServiceDiscoveryQuery(
			service_type=ServiceType.REST_API,
			tenant_id=TEST_TENANT_ID
		)
		result = await service.discover_services(query)
		
		assert result.total_count == 2
		assert result.returned_count == 2
		assert len(result.services) == 2
		assert all(svc.service_type == ServiceType.REST_API for svc in result.services)
		
		# Test discovery by environment
		query = ServiceDiscoveryQuery(
			environment="production",
			tenant_id=TEST_TENANT_ID
		)
		result = await service.discover_services(query)
		
		assert result.total_count == 2
		assert result.returned_count == 2
		prod_services = result.services
		assert all(svc.environment == "production" for svc in prod_services)
		
		# Test discovery by namespace
		query = ServiceDiscoveryQuery(
			namespace="test",
			tenant_id=TEST_TENANT_ID
		)
		result = await service.discover_services(query)
		
		assert result.total_count == 1
		assert result.returned_count == 1
		assert result.services[0].namespace == "test"
	
	async def test_service_discovery_with_tags(self, service):
		"""Test service discovery with tag filtering."""
		# Register services with different tags
		service_data = {
			"name": "tagged-service",
			"display_name": "Tagged Service",
			"service_type": ServiceType.MICROSERVICE,
			"namespace": "prod",
			"environment": "production",
			"tags": ["critical", "api", "user-facing", "high-availability"]
		}
		
		registered_service = await service.register_service(service_data, TEST_USER_ID)
		
		# Discovery with single tag
		query = ServiceDiscoveryQuery(
			tags=["critical"],
			tenant_id=TEST_TENANT_ID
		)
		result = await service.discover_services(query)
		
		assert result.total_count == 1
		assert "critical" in result.services[0].tags
		
		# Discovery with multiple tags (all must match)
		query = ServiceDiscoveryQuery(
			tags=["critical", "api"],
			tenant_id=TEST_TENANT_ID
		)
		result = await service.discover_services(query)
		
		assert result.total_count == 1
		service_tags = result.services[0].tags
		assert "critical" in service_tags
		assert "api" in service_tags
		
		# Discovery with non-matching tag
		query = ServiceDiscoveryQuery(
			tags=["non-existent-tag"],
			tenant_id=TEST_TENANT_ID
		)
		result = await service.discover_services(query)
		
		assert result.total_count == 0
	
	async def test_service_discovery_pagination(self, service):
		"""Test service discovery pagination."""
		# Register multiple services
		for i in range(25):
			service_data = {
				"name": f"service-{i:02d}",
				"display_name": f"Service {i}",
				"service_type": ServiceType.MICROSERVICE,
				"namespace": "test",
				"environment": "development"
			}
			await service.register_service(service_data, TEST_USER_ID)
		
		# Test first page
		query = ServiceDiscoveryQuery(
			namespace="test",
			limit=10,
			offset=0,
			tenant_id=TEST_TENANT_ID
		)
		result = await service.discover_services(query)
		
		assert result.total_count == 25
		assert result.returned_count == 10
		assert len(result.services) == 10
		
		# Test second page
		query = ServiceDiscoveryQuery(
			namespace="test",
			limit=10,
			offset=10,
			tenant_id=TEST_TENANT_ID
		)
		result = await service.discover_services(query)
		
		assert result.total_count == 25
		assert result.returned_count == 10
		assert len(result.services) == 10
		
		# Test last page
		query = ServiceDiscoveryQuery(
			namespace="test",
			limit=10,
			offset=20,
			tenant_id=TEST_TENANT_ID
		)
		result = await service.discover_services(query)
		
		assert result.total_count == 25
		assert result.returned_count == 5
		assert len(result.services) == 5
	
	async def test_service_health_monitoring(self, service):
		"""Test service health monitoring."""
		# Register a service with health checks
		service_data = {
			"name": "health-test-service",
			"display_name": "Health Test Service",
			"service_type": ServiceType.REST_API,
			"namespace": "prod",
			"environment": "production",
			"instances": [{
				"service_id": "will-be-set",
				"instance_name": "primary",
				"host": "health.example.com",
				"port": 8080,
				"base_url": "http://health.example.com:8080",
				"health_checks": [{
					"name": "HTTP Health Check",
					"type": HealthCheckType.HTTP,
					"url": "http://health.example.com:8080/health",
					"tenant_id": TEST_TENANT_ID,
					"created_by": TEST_USER_ID
				}],
				"tenant_id": TEST_TENANT_ID,
				"registered_by": TEST_USER_ID
			}]
		}
		
		registered_service = await service.register_service(service_data, TEST_USER_ID)
		service_id = registered_service.id
		
		# Compute health status
		health_status = await service._compute_service_health(service_id)
		
		assert health_status is not None
		assert health_status.service_id == service_id
		assert 0.0 <= health_status.health_score <= 1.0
		assert health_status.tenant_id == TEST_TENANT_ID
		
		# Get health status through public method
		retrieved_health = await service.get_service_health(service_id)
		assert retrieved_health is not None
		assert retrieved_health.service_id == service_id
	
	async def test_service_health_nonexistent(self, service):
		"""Test getting health for non-existent service."""
		health = await service.get_service_health("non-existent-service")
		assert health is None
	
	async def test_service_metrics_collection(self, service):
		"""Test service metrics collection."""
		# Register a service
		service_data = {
			"name": "metrics-service",
			"display_name": "Metrics Service",
			"service_type": ServiceType.REST_API,
			"namespace": "prod",
			"environment": "production"
		}
		
		registered_service = await service.register_service(service_data, TEST_USER_ID)
		service_id = registered_service.id
		
		# Simulate some metrics
		metrics = ServiceMetrics(
			service_id=service_id,
			metric_type="performance",
			request_count=1000,
			error_count=25,
			response_time_p50=45.2,
			response_time_p95=120.5,
			cpu_usage_avg=55.7,
			memory_usage_avg=67.3,
			tenant_id=TEST_TENANT_ID
		)
		
		service.service_metrics.append(metrics)
		
		# Get metrics
		retrieved_metrics = await service.get_service_metrics(service_id, 1)
		assert len(retrieved_metrics) == 1
		assert retrieved_metrics[0].service_id == service_id
		assert retrieved_metrics[0].request_count == 1000
		assert retrieved_metrics[0].error_count == 25
	
	async def test_registry_statistics(self, service):
		"""Test registry statistics calculation."""
		# Register services with different statuses
		services_data = [
			{
				"name": "healthy-service",
				"display_name": "Healthy Service",
				"service_type": ServiceType.REST_API,
				"namespace": "prod",
				"environment": "production"
			},
			{
				"name": "degraded-service",
				"display_name": "Degraded Service",
				"service_type": ServiceType.MICROSERVICE,
				"namespace": "prod",
				"environment": "production"
			}
		]
		
		for svc_data in services_data:
			await service.register_service(svc_data, TEST_USER_ID)
		
		# Get statistics
		stats = await service.get_registry_statistics()
		
		# Validate structure
		assert 'registry_info' in stats
		assert 'service_statistics' in stats
		assert 'performance_counters' in stats
		
		# Registry info
		registry_info = stats['registry_info']
		assert 'tenant_id' in registry_info
		assert 'uptime_seconds' in registry_info
		assert 'initialized' in registry_info
		assert registry_info['tenant_id'] == TEST_TENANT_ID
		assert registry_info['initialized'] is True
		
		# Service statistics
		service_stats = stats['service_statistics']
		assert 'total_services' in service_stats
		assert service_stats['total_services'] == 2
		assert 'healthy_services' in service_stats
		assert 'degraded_services' in service_stats
		assert 'unhealthy_services' in service_stats
		
		# Performance counters
		perf_counters = stats['performance_counters']
		assert 'total_registrations' in perf_counters
		assert 'total_discoveries' in perf_counters
		assert 'total_health_checks' in perf_counters
		assert 'cache_hit_rate' in perf_counters
	
	async def test_intelligent_ranking(self, service):
		"""Test intelligent ranking in discovery."""
		# Register services
		services_data = [
			{
				"name": "fast-service",
				"display_name": "Fast Service",
				"service_type": ServiceType.REST_API,
				"namespace": "prod",
				"environment": "production",
				"total_requests": 10000,
				"total_errors": 10,
				"average_response_time": 25.5,
				"uptime_percentage": 99.9
			},
			{
				"name": "slow-service",
				"display_name": "Slow Service", 
				"service_type": ServiceType.REST_API,
				"namespace": "prod",
				"environment": "production",
				"total_requests": 5000,
				"total_errors": 150,
				"average_response_time": 450.0,
				"uptime_percentage": 95.2
			}
		]
		
		for svc_data in services_data:
			await service.register_service(svc_data, TEST_USER_ID)
		
		# Enable intelligent ranking
		query = ServiceDiscoveryQuery(
			service_type=ServiceType.REST_API,
			intelligent_ranking=True,
			tenant_id=TEST_TENANT_ID
		)
		result = await service.discover_services(query)
		
		assert result.total_count == 2
		# Fast service should rank higher (better performance)
		assert result.services[0].name == "fast-service"
		assert result.services[1].name == "slow-service"
	
	async def test_circuit_breaker_integration(self, service):
		"""Test circuit breaker integration."""
		# Register service with circuit breaker
		service_data = {
			"name": "cb-service",
			"display_name": "Circuit Breaker Service",
			"service_type": ServiceType.REST_API,
			"namespace": "prod",
			"environment": "production",
			"instances": [{
				"service_id": "will-be-set",
				"instance_name": "primary",
				"host": "cb.example.com",
				"port": 8080,
				"base_url": "http://cb.example.com:8080",
				"circuit_breakers": [{
					"name": "Main Circuit Breaker",
					"failure_threshold": 5,
					"success_threshold": 3,
					"timeout_seconds": 60,
					"tenant_id": TEST_TENANT_ID,
					"created_by": TEST_USER_ID
				}],
				"tenant_id": TEST_TENANT_ID,
				"registered_by": TEST_USER_ID
			}]
		}
		
		registered_service = await service.register_service(service_data, TEST_USER_ID)
		
		# Validate circuit breaker configuration
		assert len(registered_service.instances) == 1
		instance = registered_service.instances[0]
		assert len(instance.circuit_breakers) == 1
		
		cb = instance.circuit_breakers[0]
		assert cb.name == "Main Circuit Breaker"
		assert cb.failure_threshold == 5
		assert cb.success_threshold == 3
		assert cb.timeout_seconds == 60
		assert cb.state == CircuitBreakerState.CLOSED  # Default state
	
	async def test_event_logging(self, service):
		"""Test event logging functionality."""
		initial_event_count = len(service.service_events)
		
		# Register a service
		service_data = {
			"name": "event-test-service",
			"display_name": "Event Test Service",
			"service_type": ServiceType.REST_API,
			"namespace": "test",
			"environment": "development"
		}
		
		registered_service = await service.register_service(service_data, TEST_USER_ID)
		
		# Should have generated events
		assert len(service.service_events) > initial_event_count
		
		# Find registration event
		reg_events = [e for e in service.service_events 
					 if e.service_id == registered_service.id and e.event_type == "service_registered"]
		assert len(reg_events) >= 1
		
		reg_event = reg_events[0]
		assert reg_event.service_id == registered_service.id
		assert reg_event.event_type == "service_registered"
		assert reg_event.severity == "info"
		assert reg_event.triggered_by == "registry_service"
		assert reg_event.tenant_id == TEST_TENANT_ID
		
		# Deregister and check for deregistration event
		await service.deregister_service(registered_service.id, TEST_USER_ID)
		
		dereg_events = [e for e in service.service_events 
					   if e.service_id == registered_service.id and e.event_type == "service_deregistered"]
		assert len(dereg_events) >= 1
	
	async def test_cache_functionality(self, service):
		"""Test caching functionality."""
		# Register a service
		service_data = {
			"name": "cached-service",
			"display_name": "Cached Service",
			"service_type": ServiceType.REST_API,
			"namespace": "prod",
			"environment": "production"
		}
		
		registered_service = await service.register_service(service_data, TEST_USER_ID)
		
		# First discovery call
		query = ServiceDiscoveryQuery(
			service_name="cached-service",
			tenant_id=TEST_TENANT_ID
		)
		result1 = await service.discover_services(query)
		
		# Second discovery call (should potentially use cache)
		result2 = await service.discover_services(query)
		
		# Results should be consistent
		assert result1.total_count == result2.total_count
		assert result1.returned_count == result2.returned_count
		assert len(result1.services) == len(result2.services)
		
		if result1.services and result2.services:
			assert result1.services[0].id == result2.services[0].id
	
	async def test_ml_features_placeholder(self, service):
		"""Test ML features are properly integrated."""
		# Test that ML models can be loaded (placeholder)
		assert hasattr(service, 'ml_models_loaded')
		
		# In a real implementation, this would test:
		# - Anomaly detection
		# - Predictive scaling
		# - Intelligent routing optimization
		# - Health prediction
		
		# For now, just verify the flags exist
		service_data = {
			"name": "ml-service",
			"display_name": "ML Service",
			"service_type": ServiceType.AI_SERVICE,
			"namespace": "ml",
			"environment": "production",
			"predictive_scaling": True,
			"intelligent_routing": True,
			"anomaly_detection": True
		}
		
		registered_service = await service.register_service(service_data, TEST_USER_ID)
		
		assert registered_service.predictive_scaling is True
		assert registered_service.intelligent_routing is True
		assert registered_service.anomaly_detection is True
	
	async def test_tenant_isolation(self, service):
		"""Test tenant isolation works correctly."""
		# Create service for different tenant
		other_tenant_service = ServiceRegistryService("other-tenant")
		await other_tenant_service.initialize()
		
		# Register service in first tenant
		service_data = {
			"name": "tenant-test-service",
			"display_name": "Tenant Test Service",
			"service_type": ServiceType.REST_API,
			"namespace": "test",
			"environment": "development"
		}
		
		service1_registered = await service.register_service(service_data, TEST_USER_ID)
		
		# Try to discover from second tenant
		query = ServiceDiscoveryQuery(
			service_name="tenant-test-service",
			tenant_id="other-tenant"
		)
		result = await other_tenant_service.discover_services(query)
		
		# Should not find the service from the other tenant
		assert result.total_count == 0
		assert result.returned_count == 0
		assert len(result.services) == 0
		
		# Service should exist in first tenant
		query_tenant1 = ServiceDiscoveryQuery(
			service_name="tenant-test-service",
			tenant_id=TEST_TENANT_ID
		)
		result_tenant1 = await service.discover_services(query_tenant1)
		
		assert result_tenant1.total_count == 1
		assert result_tenant1.services[0].id == service1_registered.id
	
	async def test_error_handling(self, service):
		"""Test error handling in various scenarios."""
		# Test invalid service data
		with pytest.raises(Exception):  # Could be ValueError or ValidationError
			await service.register_service({}, TEST_USER_ID)
		
		# Test invalid discovery query
		with pytest.raises(Exception):
			invalid_query = ServiceDiscoveryQuery(
				limit=-1,  # Invalid limit
				tenant_id=TEST_TENANT_ID
			)
			await service.discover_services(invalid_query)
		
		# Test operations on non-existent services
		non_existent_id = "non-existent-service-id"
		
		health = await service.get_service_health(non_existent_id)
		assert health is None
		
		metrics = await service.get_service_metrics(non_existent_id, 1)
		assert len(metrics) == 0
		
		deregister_result = await service.deregister_service(non_existent_id, TEST_USER_ID)
		assert deregister_result is False

class TestServiceRegistryServiceIntegration:
	"""Test integration scenarios with mocked APG components."""
	
	@pytest_asyncio.fixture
	async def service_with_mocks(self):
		"""Create service instance with mocked APG integrations."""
		with patch('capabilities.common.regy.service.AuthService') as mock_auth, \
			 patch('capabilities.common.regy.service.MonitoringService') as mock_monitoring, \
			 patch('capabilities.common.regy.service.AuditLoggingService') as mock_audit:
			
			# Setup mocks
			mock_auth.return_value.initialize = AsyncMock()
			mock_monitoring.return_value.initialize = AsyncMock()
			mock_audit.return_value.initialize = AsyncMock()
			
			svc = ServiceRegistryService(TEST_TENANT_ID)
			await svc.initialize()
			return svc
	
	async def test_apg_auth_integration(self, service_with_mocks):
		"""Test integration with APG auth service."""
		# Register service with auth requirements
		service_data = {
			"name": "auth-service",
			"display_name": "Authenticated Service",
			"service_type": ServiceType.REST_API,
			"namespace": "secure",
			"environment": "production",
			"authentication_required": True,
			"authorization_policies": ["admin-only", "read-write"]
		}
		
		registered_service = await service_with_mocks.register_service(service_data, TEST_USER_ID)
		
		assert registered_service.authentication_required is True
		assert len(registered_service.authorization_policies) == 2
		assert "admin-only" in registered_service.authorization_policies
	
	async def test_apg_monitoring_integration(self, service_with_mocks):
		"""Test integration with APG monitoring service."""
		# This would test metrics collection and reporting
		# For now, just verify service can be created with monitoring enabled
		service_data = {
			"name": "monitored-service",
			"display_name": "Monitored Service",
			"service_type": ServiceType.REST_API,
			"namespace": "prod",
			"environment": "production"
		}
		
		registered_service = await service_with_mocks.register_service(service_data, TEST_USER_ID)
		
		# Get registry statistics (which would integrate with monitoring)
		stats = await service_with_mocks.get_registry_statistics()
		assert 'performance_counters' in stats
		assert stats['performance_counters']['total_registrations'] >= 1
	
	async def test_apg_audit_integration(self, service_with_mocks):
		"""Test integration with APG audit service."""
		# Register and deregister service
		service_data = {
			"name": "audited-service",
			"display_name": "Audited Service",
			"service_type": ServiceType.REST_API,
			"namespace": "audit",
			"environment": "production"
		}
		
		registered_service = await service_with_mocks.register_service(service_data, TEST_USER_ID)
		await service_with_mocks.deregister_service(registered_service.id, TEST_USER_ID)
		
		# Verify audit events were created
		audit_events = [e for e in service_with_mocks.service_events 
					   if e.service_id == registered_service.id]
		assert len(audit_events) >= 2  # Registration and deregistration
		
		# Check event details
		reg_event = next(e for e in audit_events if e.event_type == "service_registered")
		assert reg_event.triggered_by == "registry_service"
		assert reg_event.tenant_id == TEST_TENANT_ID
		
		dereg_event = next(e for e in audit_events if e.event_type == "service_deregistered")
		assert dereg_event.severity in ["info", "warning"]

if __name__ == "__main__":
	pytest.main([__file__, "-v"])
