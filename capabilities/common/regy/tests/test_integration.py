#!/usr/bin/env python3
"""
Registry (regy) - Integration Test Suite
========================================

Comprehensive integration tests for APG capability integration,
end-to-end workflows, and cross-component functionality.

Author: APG Platform Team
Copyright: © 2025 Datacraft
"""

import asyncio
import pytest
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any
from unittest.mock import Mock, patch, AsyncMock, MagicMock

from ..service import ServiceRegistryService
from ..api import registry_bp, api
from ..blueprint import init_app, get_registry_service, register_with_apg_composition
from ..models import (
	ServiceRegistration, ServiceDiscoveryQuery, ServiceHealthStatus,
	ServiceType, ServiceStatus, LoadBalanceStrategy
)

from . import TEST_TENANT_ID, TEST_USER_ID, TEST_SERVICE_NAME

class TestAPGCapabilityIntegration:
	"""Test integration with APG platform components."""
	
	@pytest.fixture
	def mock_capability_registry(self):
		"""Mock APG capability registry."""
		mock_registry = Mock()
		mock_registry.initialize = AsyncMock()
		mock_registry.register_capability = AsyncMock(return_value=True)
		return mock_registry
	
	@pytest.fixture  
	def mock_apg_services(self):
		"""Mock APG service dependencies."""
		mocks = {
			'auth': Mock(),
			'monitoring': Mock(),
			'audit': Mock(),
			'config': Mock()
		}
		
		for service_mock in mocks.values():
			service_mock.initialize = AsyncMock()
		
		return mocks
	
	@patch('capabilities.common.regy.blueprint.CapabilityRegistry')
	async def test_apg_composition_registration(self, mock_registry_class, mock_capability_registry):
		"""Test registration with APG composition engine."""
		mock_registry_class.return_value = mock_capability_registry
		
		# Test registration
		success = await register_with_apg_composition(TEST_TENANT_ID)
		
		assert success is True
		mock_capability_registry.initialize.assert_called_once()
		mock_capability_registry.register_capability.assert_called_once()
		
		# Verify capability metadata
		call_args = mock_capability_registry.register_capability.call_args[0][0]
		assert call_args['id'] == 'regy'
		assert call_args['name'] == 'Registry (regy)'
		assert call_args['version'] == '1.0.0'
		assert call_args['tenant_id'] == TEST_TENANT_ID
		
		# Check dependencies
		dependencies = call_args['dependencies']
		dependency_ids = [dep['capability_id'] for dep in dependencies]
		assert 'auth' in dependency_ids
		assert 'conf' in dependency_ids
		assert 'moni' in dependency_ids
		assert 'audl' in dependency_ids
		
		# Check provided services
		provides = call_args['provides']
		service_ids = [service['service_id'] for service in provides]
		assert 'service_discovery' in service_ids
		assert 'service_registration' in service_ids
		assert 'health_monitoring' in service_ids
	
	@patch('capabilities.common.regy.blueprint.AuthService')
	@patch('capabilities.common.regy.blueprint.MonitoringService') 
	@patch('capabilities.common.regy.blueprint.AuditLoggingService')
	async def test_apg_service_integration(self, mock_audit_service, mock_monitoring_service, mock_auth_service, mock_apg_services):
		"""Test integration with APG core services."""
		# Setup mocks
		mock_auth_service.return_value = mock_apg_services['auth']
		mock_monitoring_service.return_value = mock_apg_services['monitoring']  
		mock_audit_service.return_value = mock_apg_services['audit']
		
		# Create and initialize service
		service = ServiceRegistryService(TEST_TENANT_ID)
		await service.initialize()
		
		# Verify service initialized successfully
		assert service.initialized is True
		assert service.tenant_id == TEST_TENANT_ID
	
	async def test_end_to_end_service_lifecycle(self):
		"""Test complete service lifecycle from registration to deregistration."""
		# Initialize service
		service = ServiceRegistryService(TEST_TENANT_ID)
		await service.initialize()
		
		# 1. Service Registration
		service_data = {
			"name": "e2e-test-service",
			"display_name": "End-to-End Test Service",
			"description": "Service for testing complete lifecycle",
			"service_type": ServiceType.REST_API,
			"namespace": "test",
			"environment": "integration",
			"base_path": "/api/v1",
			"instances": [{
				"service_id": "will-be-set",
				"instance_name": "primary",
				"host": "e2e.example.com",
				"port": 8080,
				"base_url": "http://e2e.example.com:8080",
				"weight": 100,
				"tenant_id": TEST_TENANT_ID,
				"registered_by": TEST_USER_ID
			}],
			"discovery_enabled": True,
			"health_check_enabled": True,
			"circuit_breaker_enabled": True,
			"predictive_scaling": True,
			"intelligent_routing": True,
			"tags": ["e2e", "test", "lifecycle"]
		}
		
		registered_service = await service.register_service(service_data, TEST_USER_ID)
		service_id = registered_service.id
		
		# Verify registration
		assert registered_service.name == "e2e-test-service"
		assert registered_service.tenant_id == TEST_TENANT_ID
		assert len(registered_service.instances) == 1
		assert len(registered_service.tags) == 3
		
		# 2. Service Discovery
		discovery_query = ServiceDiscoveryQuery(
			service_name="e2e-test-service",
			tenant_id=TEST_TENANT_ID,
			include_instances=True,
			include_health=True
		)
		discovery_result = await service.discover_services(discovery_query)
		
		# Verify discovery
		assert discovery_result.total_count == 1
		assert discovery_result.returned_count == 1
		assert len(discovery_result.services) == 1
		assert discovery_result.services[0].id == service_id
		
		# 3. Health Monitoring
		health_status = await service._compute_service_health(service_id)
		assert health_status is not None
		assert health_status.service_id == service_id
		assert 0.0 <= health_status.health_score <= 1.0
		
		# 4. Metrics Collection
		metrics = await service.get_service_metrics(service_id, 1)
		# Metrics might be empty initially, but call should succeed
		assert isinstance(metrics, list)
		
		# 5. Registry Statistics
		stats = await service.get_registry_statistics()
		assert stats['service_statistics']['total_services'] >= 1
		assert stats['registry_info']['tenant_id'] == TEST_TENANT_ID
		
		# 6. Service Deregistration
		deregister_success = await service.deregister_service(service_id, TEST_USER_ID)
		assert deregister_success is True
		
		# Verify deregistration
		assert service_id not in service.services
		
		# Discovery should return empty results
		discovery_after_dereg = await service.discover_services(discovery_query)
		assert discovery_after_dereg.total_count == 0
	
	async def test_multi_tenant_isolation(self):
		"""Test that tenant isolation works correctly across all operations."""
		# Create services for different tenants
		tenant1_service = ServiceRegistryService("tenant-1")
		tenant2_service = ServiceRegistryService("tenant-2")
		
		await tenant1_service.initialize()
		await tenant2_service.initialize()
		
		# Register services in different tenants with same name
		service_data = {
			"name": "shared-service-name",
			"display_name": "Tenant Service",
			"service_type": ServiceType.MICROSERVICE,
			"namespace": "multi-tenant-test",
			"environment": "integration"
		}
		
		service1 = await tenant1_service.register_service(service_data, "tenant1-user")
		service2 = await tenant2_service.register_service(service_data, "tenant2-user")
		
		# Verify services have different IDs
		assert service1.id != service2.id
		assert service1.tenant_id == "tenant-1"
		assert service2.tenant_id == "tenant-2"
		
		# Discovery from tenant 1 should only find tenant 1 service
		query1 = ServiceDiscoveryQuery(
			service_name="shared-service-name",
			tenant_id="tenant-1"
		)
		result1 = await tenant1_service.discover_services(query1)
		
		assert result1.total_count == 1
		assert result1.services[0].id == service1.id
		assert result1.services[0].tenant_id == "tenant-1"
		
		# Discovery from tenant 2 should only find tenant 2 service
		query2 = ServiceDiscoveryQuery(
			service_name="shared-service-name", 
			tenant_id="tenant-2"
		)
		result2 = await tenant2_service.discover_services(query2)
		
		assert result2.total_count == 1
		assert result2.services[0].id == service2.id
		assert result2.services[0].tenant_id == "tenant-2"
		
		# Cross-tenant discovery should return empty
		cross_query = ServiceDiscoveryQuery(
			service_name="shared-service-name",
			tenant_id="tenant-1"  # Looking for tenant-1 services
		)
		cross_result = await tenant2_service.discover_services(cross_query)
		assert cross_result.total_count == 0
	
	async def test_intelligent_features_integration(self):
		"""Test integration of ML/AI features."""
		service = ServiceRegistryService(TEST_TENANT_ID)
		await service.initialize()
		
		# Register services with different performance characteristics
		services_data = [
			{
				"name": "high-performance-service",
				"display_name": "High Performance Service",
				"service_type": ServiceType.REST_API,
				"namespace": "prod",
				"environment": "production",
				"predictive_scaling": True,
				"intelligent_routing": True,
				"anomaly_detection": True,
				"total_requests": 100000,
				"total_errors": 10,
				"average_response_time": 25.5,
				"uptime_percentage": 99.99
			},
			{
				"name": "standard-service",
				"display_name": "Standard Service", 
				"service_type": ServiceType.REST_API,
				"namespace": "prod",
				"environment": "production",
				"predictive_scaling": False,
				"intelligent_routing": False,
				"anomaly_detection": False,
				"total_requests": 50000,
				"total_errors": 250,
				"average_response_time": 125.0,
				"uptime_percentage": 98.5
			}
		]
		
		registered_services = []
		for svc_data in services_data:
			registered_service = await service.register_service(svc_data, TEST_USER_ID)
			registered_services.append(registered_service)
		
		# Test intelligent ranking
		query = ServiceDiscoveryQuery(
			service_type=ServiceType.REST_API,
			environment="production",
			intelligent_ranking=True,
			tenant_id=TEST_TENANT_ID
		)
		result = await service.discover_services(query)
		
		assert result.total_count == 2
		# High-performance service should rank first
		assert result.services[0].name == "high-performance-service"
		assert result.services[1].name == "standard-service"
		
		# Verify ML features are enabled
		high_perf_service = result.services[0]
		assert high_perf_service.predictive_scaling is True
		assert high_perf_service.intelligent_routing is True
		assert high_perf_service.anomaly_detection is True
	
	async def test_circuit_breaker_integration(self):
		"""Test circuit breaker integration across service lifecycle."""
		service = ServiceRegistryService(TEST_TENANT_ID)
		await service.initialize()
		
		# Register service with circuit breaker
		service_data = {
			"name": "cb-integration-service",
			"display_name": "Circuit Breaker Integration Service",
			"service_type": ServiceType.MICROSERVICE,
			"namespace": "reliability",
			"environment": "production",
			"instances": [{
				"service_id": "will-be-set",
				"instance_name": "cb-instance",
				"host": "cb.reliability.com",
				"port": 8080,
				"base_url": "http://cb.reliability.com:8080",
				"circuit_breakers": [{
					"name": "Integration Circuit Breaker",
					"failure_threshold": 5,
					"success_threshold": 3,
					"timeout_seconds": 60,
					"adaptive_thresholds": True,
					"pattern_recognition": True,
					"intelligent_recovery": True,
					"tenant_id": TEST_TENANT_ID,
					"created_by": TEST_USER_ID
				}],
				"tenant_id": TEST_TENANT_ID,
				"registered_by": TEST_USER_ID
			}]
		}
		
		registered_service = await service.register_service(service_data, TEST_USER_ID)
		service_id = registered_service.id
		
		# Verify circuit breaker configuration
		instance = registered_service.instances[0]
		cb = instance.circuit_breakers[0]
		assert cb.adaptive_thresholds is True
		assert cb.pattern_recognition is True
		assert cb.intelligent_recovery is True
		
		# Test health monitoring with circuit breaker
		health_status = await service._compute_service_health(service_id)
		assert health_status is not None
		# Circuit breaker state should be included in health
		assert hasattr(health_status, 'circuit_breaker_state')
		
		# Discovery should include circuit breaker information
		query = ServiceDiscoveryQuery(
			service_name="cb-integration-service",
			tenant_id=TEST_TENANT_ID,
			include_instances=True
		)
		result = await service.discover_services(query)
		
		assert result.total_count == 1
		discovered_service = result.services[0]
		discovered_instance = discovered_service.instances[0]
		assert len(discovered_instance.circuit_breakers) == 1
	
	async def test_performance_under_load(self):
		"""Test system performance under simulated load."""
		service = ServiceRegistryService(TEST_TENANT_ID)
		await service.initialize()
		
		# Register multiple services
		registration_tasks = []
		for i in range(50):
			service_data = {
				"name": f"load-test-service-{i:02d}",
				"display_name": f"Load Test Service {i}",
				"service_type": ServiceType.MICROSERVICE,
				"namespace": "load-test",
				"environment": "test",
				"tags": [f"batch-{i // 10}", "load-test"]
			}
			task = service.register_service(service_data, f"user-{i}")
			registration_tasks.append(task)
		
		# Execute registrations concurrently
		registered_services = await asyncio.gather(*registration_tasks)
		assert len(registered_services) == 50
		
		# Test concurrent discoveries
		discovery_tasks = []
		for i in range(20):
			query = ServiceDiscoveryQuery(
				namespace="load-test",
				service_type=ServiceType.MICROSERVICE,
				limit=25,
				offset=i,
				tenant_id=TEST_TENANT_ID
			)
			task = service.discover_services(query)
			discovery_tasks.append(task)
		
		discovery_results = await asyncio.gather(*discovery_tasks)
		
		# Verify all discoveries succeeded
		assert len(discovery_results) == 20
		for result in discovery_results:
			assert result.total_count == 50  # All found the same total
		
		# Test health monitoring for all services
		health_tasks = []
		for registered_service in registered_services[:10]:  # Test first 10
			task = service._compute_service_health(registered_service.id)
			health_tasks.append(task)
		
		health_results = await asyncio.gather(*health_tasks)
		
		# All health checks should succeed
		assert len(health_results) == 10
		for health in health_results:
			assert health is not None
			assert 0.0 <= health.health_score <= 1.0
		
		# Verify registry statistics
		stats = await service.get_registry_statistics()
		assert stats['service_statistics']['total_services'] == 50
		assert stats['performance_counters']['total_registrations'] == 50

class TestFlaskIntegration:
	"""Test Flask-AppBuilder blueprint integration."""
	
	@pytest.fixture
	def app(self):
		"""Create Flask application for testing."""
		from flask import Flask
		app = Flask(__name__)
		app.config['TESTING'] = True
		app.config['WTF_CSRF_ENABLED'] = False
		app.config['APG_DEFAULT_TENANT_ID'] = TEST_TENANT_ID
		
		return app
	
	@pytest.fixture
	def app_with_blueprint(self, app):
		"""Flask app with registry blueprint initialized."""
		with patch('capabilities.common.regy.blueprint.register_with_apg_composition', new=AsyncMock(return_value=True)):
			init_app(app)
		return app
	
	@pytest.fixture
	def client(self, app_with_blueprint):
		"""Test client for Flask app."""
		return app_with_blueprint.test_client()
	
	def test_blueprint_initialization(self, app_with_blueprint):
		"""Test that blueprint is properly initialized."""
		# Check that blueprint is registered
		assert any(bp.name == 'registry' for bp in app_with_blueprint.iter_blueprints())
		
		# Check routes exist
		with app_with_blueprint.test_request_context():
			rules = [rule.rule for rule in app_with_blueprint.url_map.iter_rules()]
			assert any('/api/regy/v1/' in rule for rule in rules)
	
	@patch('capabilities.common.regy.blueprint.get_registry_service')
	async def test_health_check_endpoint(self, mock_get_service, client):
		"""Test Flask health check endpoint."""
		# Setup mock
		mock_service = Mock()
		mock_service.initialized = True
		mock_service.tenant_id = TEST_TENANT_ID
		mock_service.get_registry_statistics = AsyncMock(return_value={
			'registry_info': {'uptime_seconds': 3600},
			'service_statistics': {
				'total_services': 5,
				'healthy_services': 4,
				'degraded_services': 1,
				'unhealthy_services': 0
			},
			'performance_counters': {
				'total_registrations': 8,
				'total_discoveries': 150,
				'cache_hit_rate': 0.85
			}
		})
		mock_get_service.return_value = mock_service
		
		# Test health check
		response = client.get('/health/registry')
		
		assert response.status_code == 200
		data = response.get_json()
		
		assert data['status'] == 'healthy'
		assert data['capability'] == 'registry'
		assert data['tenant_id'] == TEST_TENANT_ID
		assert data['total_services'] == 5
		assert data['healthy_services'] == 4
	
	@patch('capabilities.common.regy.blueprint.get_registry_service')
	async def test_metrics_endpoint(self, mock_get_service, client):
		"""Test Flask metrics endpoint."""
		# Setup mock
		mock_service = Mock()
		mock_service.initialized = True
		mock_service.get_registry_statistics = AsyncMock(return_value={
			'service_statistics': {
				'total_services': 10,
				'healthy_services': 8,
				'degraded_services': 1,
				'unhealthy_services': 1
			},
			'performance_counters': {
				'total_registrations': 15,
				'total_discoveries': 300,
				'total_health_checks': 1200,
				'cache_hit_rate': 0.92
			},
			'registry_info': {'uptime_seconds': 7200}
		})
		mock_get_service.return_value = mock_service
		
		# Test metrics endpoint
		response = client.get('/metrics/registry')
		
		assert response.status_code == 200
		assert response.content_type == 'text/plain; charset=utf-8'
		
		data = response.get_data(as_text=True)
		
		# Check Prometheus format
		assert 'registry_total_services 10' in data
		assert 'registry_healthy_services 8' in data
		assert 'registry_total_registrations 15' in data
		assert 'registry_uptime_seconds 7200' in data

class TestErrorHandlingAndResilience:
	"""Test error handling and system resilience."""
	
	async def test_service_initialization_failure(self):
		"""Test handling of service initialization failures."""
		# Create service but don't initialize
		service = ServiceRegistryService(TEST_TENANT_ID)
		
		# Operations should handle uninitialized state gracefully
		with pytest.raises(Exception):
			await service.register_service({}, TEST_USER_ID)
	
	async def test_invalid_data_handling(self):
		"""Test handling of invalid input data."""
		service = ServiceRegistryService(TEST_TENANT_ID)
		await service.initialize()
		
		# Invalid service data
		with pytest.raises(Exception):
			await service.register_service({"name": ""}, TEST_USER_ID)  # Empty name
		
		with pytest.raises(Exception):
			await service.register_service({"invalid": "data"}, TEST_USER_ID)  # Missing required fields
		
		# Invalid discovery query
		with pytest.raises(Exception):
			query = ServiceDiscoveryQuery(
				limit=-1,  # Invalid limit
				tenant_id=TEST_TENANT_ID
			)
			await service.discover_services(query)
	
	async def test_resource_cleanup(self):
		"""Test proper resource cleanup on errors."""
		service = ServiceRegistryService(TEST_TENANT_ID)
		await service.initialize()
		
		initial_service_count = len(service.services)
		
		# Try to register invalid service
		try:
			await service.register_service({"name": ""}, TEST_USER_ID)
		except Exception:
			pass  # Expected to fail
		
		# Service count should not have changed
		assert len(service.services) == initial_service_count
	
	async def test_concurrent_operations_consistency(self):
		"""Test data consistency under concurrent operations."""
		service = ServiceRegistryService(TEST_TENANT_ID)
		await service.initialize()
		
		# Concurrent registrations and deregistrations
		async def register_and_deregister(index):
			service_data = {
				"name": f"concurrent-service-{index}",
				"display_name": f"Concurrent Service {index}",
				"service_type": ServiceType.MICROSERVICE,
				"namespace": "concurrent",
				"environment": "test"
			}
			
			registered = await service.register_service(service_data, f"user-{index}")
			await asyncio.sleep(0.01)  # Small delay
			await service.deregister_service(registered.id, f"user-{index}")
			return registered.id
		
		# Run multiple concurrent operations
		tasks = [register_and_deregister(i) for i in range(10)]
		completed_ids = await asyncio.gather(*tasks)
		
		# All operations should have completed successfully
		assert len(completed_ids) == 10
		assert len(set(completed_ids)) == 10  # All unique
		
		# No services should remain registered
		assert len(service.services) == 0

if __name__ == "__main__":
	pytest.main([__file__, "-v"])
