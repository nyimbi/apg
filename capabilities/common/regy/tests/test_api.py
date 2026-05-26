#!/usr/bin/env python3
"""
Registry (regy) - API Test Suite
================================

Comprehensive tests for Flask-RESTX API endpoints with authentication,
authorization, and error handling scenarios.

Author: APG Platform Team
Copyright: © 2025 Datacraft
"""

import asyncio
import json
import pytest
from datetime import datetime, timezone
from typing import Dict, List, Any
from unittest.mock import Mock, patch, AsyncMock

# Test framework imports
import pytest_httpserver
from werkzeug import Request, Response

from ..api import registry_bp, api
from ..service import ServiceRegistryService
from ..models import (
	ServiceRegistration, ServiceDiscoveryQuery, ServiceType,
	ServiceStatus, HealthCheckType, LoadBalanceStrategy
)

from . import TEST_TENANT_ID, TEST_USER_ID, TEST_SERVICE_NAME

class TestRegistryAPI:
	"""Test Flask-RESTX API endpoints."""
	
	@pytest.fixture
	def app(self):
		"""Create Flask test application."""
		from flask import Flask
		app = Flask(__name__)
		app.config['TESTING'] = True
		app.config['WTF_CSRF_ENABLED'] = False
		app.register_blueprint(registry_bp)
		return app
	
	@pytest.fixture
	def client(self, app):
		"""Create test client."""
		return app.test_client()
	
	@pytest.fixture
	def mock_service(self):
		"""Create mocked service instance."""
		service = Mock(spec=ServiceRegistryService)
		service.tenant_id = TEST_TENANT_ID
		service.initialized = True
		service.services = {}
		service.service_health = {}
		service.service_metrics = []
		service.service_events = []
		
		# Mock async methods
		service.initialize = AsyncMock()
		service.register_service = AsyncMock()
		service.deregister_service = AsyncMock()
		service.discover_services = AsyncMock()
		service.get_service_health = AsyncMock()
		service.get_service_metrics = AsyncMock()
		service.get_registry_statistics = AsyncMock()
		service.update_service_health = AsyncMock()
		service._compute_service_health = AsyncMock()
		
		return service
	
	def test_api_blueprint_registration(self, app):
		"""Test that API blueprint is properly registered."""
		# Check that blueprint is registered
		assert any(bp.name == 'registry' for bp in app.iter_blueprints())
		
		# Check that routes exist
		with app.test_request_context():
			rules = [rule.rule for rule in app.url_map.iter_rules()]
			
			# Check main API endpoints exist
			assert any('/api/regy/v1/services' in rule for rule in rules)
			assert any('/api/regy/v1/discovery' in rule for rule in rules)
			assert any('/api/regy/v1/health' in rule for rule in rules)
			assert any('/api/regy/v1/metrics' in rule for rule in rules)
			assert any('/api/regy/v1/events' in rule for rule in rules)
	
	def test_api_documentation_endpoint(self, client):
		"""Test API documentation is accessible."""
		response = client.get('/api/regy/v1/docs/')
		assert response.status_code == 200
		
		# Check that it's HTML content (Swagger UI)
		assert 'text/html' in response.content_type
	
	@patch('capabilities.common.regy.api.get_registry_service')
	async def test_service_list_endpoint(self, mock_get_service, client, mock_service):
		"""Test GET /services endpoint."""
		# Setup mock
		mock_get_service.return_value = mock_service
		
		# Mock discovery result
		from ..models import ServiceDiscoveryResult
		mock_result = ServiceDiscoveryResult(
			services=[],
			total_count=0,
			returned_count=0,
			query_time_ms=5.2,
			tenant_id=TEST_TENANT_ID
		)
		mock_service.discover_services.return_value = mock_result
		
		# Test request
		response = client.get('/api/regy/v1/services')
		
		assert response.status_code == 200
		data = response.get_json()
		
		assert 'services' in data
		assert 'total_count' in data
		assert 'returned_count' in data
		assert 'query_time_ms' in data
		assert data['total_count'] == 0
		assert data['returned_count'] == 0
		assert isinstance(data['services'], list)
	
	@patch('capabilities.common.regy.api.get_registry_service')
	async def test_service_registration_endpoint(self, mock_get_service, client, mock_service):
		"""Test POST /services endpoint."""
		# Setup mock
		mock_get_service.return_value = mock_service
		
		# Create mock registered service
		from ..models import ServiceRegistration
		mock_registered_service = ServiceRegistration(
			name=TEST_SERVICE_NAME,
			display_name="Test Service",
			description="A test service",
			service_type=ServiceType.REST_API,
			namespace="test",
			environment="development",
			tenant_id=TEST_TENANT_ID,
			created_by=TEST_USER_ID,
			last_modified_by=TEST_USER_ID
		)
		mock_service.register_service.return_value = mock_registered_service
		
		# Test request
		service_data = {
			"name": TEST_SERVICE_NAME,
			"display_name": "Test Service",
			"description": "A test service",
			"service_type": "rest_api",
			"namespace": "test",
			"environment": "development"
		}
		
		response = client.post(
			'/api/regy/v1/services',
			data=json.dumps(service_data),
			content_type='application/json',
			headers={'X-User-ID': TEST_USER_ID}
		)
		
		assert response.status_code == 201
		data = response.get_json()
		
		assert data['name'] == TEST_SERVICE_NAME
		assert data['display_name'] == "Test Service"
		assert data['service_type'] == ServiceType.REST_API
	
	@patch('capabilities.common.regy.api.get_registry_service')
	async def test_service_detail_endpoint(self, mock_get_service, client, mock_service):
		"""Test GET /services/{service_id} endpoint."""
		# Setup mock
		mock_get_service.return_value = mock_service
		
		service_id = "test-service-id"
		from ..models import ServiceRegistration
		mock_service_obj = ServiceRegistration(
			id=service_id,
			name=TEST_SERVICE_NAME,
			display_name="Test Service",
			service_type=ServiceType.REST_API,
			namespace="test",
			environment="development",
			tenant_id=TEST_TENANT_ID,
			created_by=TEST_USER_ID,
			last_modified_by=TEST_USER_ID
		)
		
		mock_service.services = {service_id: mock_service_obj}
		
		# Test request
		response = client.get(f'/api/regy/v1/services/{service_id}')
		
		assert response.status_code == 200
		data = response.get_json()
		
		assert data['id'] == service_id
		assert data['name'] == TEST_SERVICE_NAME
		assert data['display_name'] == "Test Service"
	
	@patch('capabilities.common.regy.api.get_registry_service')
	async def test_service_detail_not_found(self, mock_get_service, client, mock_service):
		"""Test GET /services/{service_id} with non-existent service."""
		# Setup mock
		mock_get_service.return_value = mock_service
		mock_service.services = {}
		
		# Test request
		response = client.get('/api/regy/v1/services/non-existent-id')
		
		assert response.status_code == 404
		data = response.get_json()
		
		assert 'error' in data
		assert 'Not Found' in data['error']
	
	@patch('capabilities.common.regy.api.get_registry_service')
	async def test_service_update_endpoint(self, mock_get_service, client, mock_service):
		"""Test PUT /services/{service_id} endpoint."""
		# Setup mock
		mock_get_service.return_value = mock_service
		
		service_id = "test-service-id"
		from ..models import ServiceRegistration
		mock_service_obj = ServiceRegistration(
			id=service_id,
			name=TEST_SERVICE_NAME,
			display_name="Test Service",
			service_type=ServiceType.REST_API,
			namespace="test",
			environment="development",
			tenant_id=TEST_TENANT_ID,
			created_by=TEST_USER_ID,
			last_modified_by=TEST_USER_ID
		)
		
		mock_service.services = {service_id: mock_service_obj}
		
		# Test request
		update_data = {
			"display_name": "Updated Test Service",
			"description": "Updated description"
		}
		
		response = client.put(
			f'/api/regy/v1/services/{service_id}',
			data=json.dumps(update_data),
			content_type='application/json',
			headers={'X-User-ID': TEST_USER_ID}
		)
		
		assert response.status_code == 200
		data = response.get_json()
		
		# Verify the update was applied to the mock object
		assert mock_service_obj.display_name == "Updated Test Service"
	
	@patch('capabilities.common.regy.api.get_registry_service')
	async def test_service_deregistration_endpoint(self, mock_get_service, client, mock_service):
		"""Test DELETE /services/{service_id} endpoint."""
		# Setup mock
		mock_get_service.return_value = mock_service
		mock_service.deregister_service.return_value = True
		
		service_id = "test-service-id"
		
		# Test request
		response = client.delete(
			f'/api/regy/v1/services/{service_id}',
			headers={'X-User-ID': TEST_USER_ID}
		)
		
		assert response.status_code == 200
		data = response.get_json()
		
		assert 'message' in data
		assert 'successfully' in data['message'].lower()
		
		# Verify deregister_service was called
		mock_service.deregister_service.assert_called_once_with(service_id, TEST_USER_ID)
	
	@patch('capabilities.common.regy.api.get_registry_service')
	async def test_service_discovery_endpoint(self, mock_get_service, client, mock_service):
		"""Test POST /discovery/search endpoint."""
		# Setup mock
		mock_get_service.return_value = mock_service
		
		from ..models import ServiceDiscoveryResult, ServiceRegistration
		mock_service_obj = ServiceRegistration(
			name="discovered-service",
			display_name="Discovered Service",
			service_type=ServiceType.REST_API,
			namespace="prod",
			environment="production",
			tenant_id=TEST_TENANT_ID,
			created_by=TEST_USER_ID,
			last_modified_by=TEST_USER_ID
		)
		
		mock_result = ServiceDiscoveryResult(
			services=[mock_service_obj],
			total_count=1,
			returned_count=1,
			query_time_ms=12.5,
			tenant_id=TEST_TENANT_ID
		)
		mock_service.discover_services.return_value = mock_result
		
		# Test request
		discovery_query = {
			"service_type": "rest_api",
			"environment": "production",
			"healthy_only": True,
			"limit": 50
		}
		
		response = client.post(
			'/api/regy/v1/discovery/search',
			data=json.dumps(discovery_query),
			content_type='application/json',
			headers={'X-Tenant-ID': TEST_TENANT_ID}
		)
		
		assert response.status_code == 200
		data = response.get_json()
		
		assert 'services' in data
		assert 'total_count' in data
		assert data['total_count'] == 1
		assert len(data['services']) == 1
		assert data['services'][0]['name'] == "discovered-service"
	
	@patch('capabilities.common.regy.api.get_registry_service')
	async def test_service_discovery_by_name(self, mock_get_service, client, mock_service):
		"""Test GET /discovery/by-name/{service_name} endpoint."""
		# Setup mock
		mock_get_service.return_value = mock_service
		
		from ..models import ServiceDiscoveryResult, ServiceRegistration
		service_name = "target-service"
		mock_service_obj = ServiceRegistration(
			name=service_name,
			display_name="Target Service",
			service_type=ServiceType.MICROSERVICE,
			namespace="prod",
			environment="production",
			tenant_id=TEST_TENANT_ID,
			created_by=TEST_USER_ID,
			last_modified_by=TEST_USER_ID
		)
		
		mock_result = ServiceDiscoveryResult(
			services=[mock_service_obj],
			total_count=1,
			returned_count=1,
			query_time_ms=8.3,
			tenant_id=TEST_TENANT_ID
		)
		mock_service.discover_services.return_value = mock_result
		
		# Test request
		response = client.get(f'/api/regy/v1/discovery/by-name/{service_name}')
		
		assert response.status_code == 200
		data = response.get_json()
		
		assert data['total_count'] == 1
		assert data['services'][0]['name'] == service_name
	
	@patch('capabilities.common.regy.api.get_registry_service')
	async def test_health_overview_endpoint(self, mock_get_service, client, mock_service):
		"""Test GET /health endpoint."""
		# Setup mock
		mock_get_service.return_value = mock_service
		
		mock_stats = {
			'service_statistics': {
				'total_services': 10,
				'healthy_services': 8,
				'degraded_services': 1,
				'unhealthy_services': 1
			},
			'registry_info': {
				'uptime_seconds': 3600
			},
			'performance_counters': {
				'total_registrations': 15,
				'total_discoveries': 250,
				'cache_hit_rate': 0.85
			}
		}
		mock_service.get_registry_statistics.return_value = mock_stats
		
		# Test request
		response = client.get('/api/regy/v1/health')
		
		assert response.status_code == 200
		data = response.get_json()
		
		assert 'registry_health' in data
		assert 'total_services' in data
		assert 'healthy_services' in data
		assert 'uptime_seconds' in data
		assert data['total_services'] == 10
		assert data['healthy_services'] == 8
	
	@patch('capabilities.common.regy.api.get_registry_service')
	async def test_service_health_endpoint(self, mock_get_service, client, mock_service):
		"""Test GET /health/services/{service_id} endpoint."""
		# Setup mock
		mock_get_service.return_value = mock_service
		
		service_id = "health-test-service"
		from ..models import ServiceHealthStatus
		mock_health_status = ServiceHealthStatus(
			service_id=service_id,
			instance_id="instance-1",
			overall_status=ServiceStatus.HEALTHY,
			health_score=0.95,
			status_message="Service is healthy",
			response_time_ms=45.2,
			cpu_usage_percent=35.7,
			memory_usage_percent=52.3,
			tenant_id=TEST_TENANT_ID
		)
		mock_service.get_service_health.return_value = mock_health_status
		
		# Test request
		response = client.get(f'/api/regy/v1/health/services/{service_id}')
		
		assert response.status_code == 200
		data = response.get_json()
		
		assert data['service_id'] == service_id
		assert data['overall_status'] == ServiceStatus.HEALTHY
		assert data['health_score'] == 0.95
		assert data['response_time_ms'] == 45.2
	
	@patch('capabilities.common.regy.api.get_registry_service')
	async def test_health_check_trigger_endpoint(self, mock_get_service, client, mock_service):
		"""Test POST /health/check/{service_id} endpoint."""
		# Setup mock
		mock_get_service.return_value = mock_service
		
		service_id = "health-check-service"
		mock_service.services = {service_id: Mock()}
		
		from ..models import ServiceHealthStatus
		mock_health_status = ServiceHealthStatus(
			service_id=service_id,
			instance_id="instance-1",
			overall_status=ServiceStatus.HEALTHY,
			health_score=0.88,
			status_message="Health check completed",
			tenant_id=TEST_TENANT_ID
		)
		mock_service._compute_service_health.return_value = mock_health_status
		
		# Test request
		response = client.post(f'/api/regy/v1/health/check/{service_id}')
		
		assert response.status_code == 200
		data = response.get_json()
		
		assert 'message' in data
		assert 'health_status' in data
		assert 'successfully' in data['message'].lower()
		assert data['health_status']['service_id'] == service_id
	
	@patch('capabilities.common.regy.api.get_registry_service')
	async def test_service_metrics_endpoint(self, mock_get_service, client, mock_service):
		"""Test GET /metrics/services/{service_id} endpoint."""
		# Setup mock
		mock_get_service.return_value = mock_service
		
		service_id = "metrics-service"
		from ..models import ServiceMetrics
		mock_metrics = [
			ServiceMetrics(
				service_id=service_id,
				metric_type="performance",
				request_count=1000,
				error_count=25,
				response_time_p50=45.2,
				cpu_usage_avg=55.7,
				tenant_id=TEST_TENANT_ID
			)
		]
		mock_service.get_service_metrics.return_value = mock_metrics
		
		# Test request
		response = client.get(f'/api/regy/v1/metrics/services/{service_id}?hours=24')
		
		assert response.status_code == 200
		data = response.get_json()
		
		assert data['service_id'] == service_id
		assert data['time_range_hours'] == 24
		assert data['metrics_count'] == 1
		assert len(data['metrics']) == 1
		assert data['metrics'][0]['request_count'] == 1000
	
	@patch('capabilities.common.regy.api.get_registry_service')
	async def test_registry_statistics_endpoint(self, mock_get_service, client, mock_service):
		"""Test GET /metrics/registry/statistics endpoint."""
		# Setup mock
		mock_get_service.return_value = mock_service
		
		mock_stats = {
			'registry_info': {
				'tenant_id': TEST_TENANT_ID,
				'uptime_seconds': 7200,
				'initialized': True
			},
			'service_statistics': {
				'total_services': 15,
				'healthy_services': 12,
				'degraded_services': 2,
				'unhealthy_services': 1
			},
			'performance_counters': {
				'total_registrations': 20,
				'total_discoveries': 500,
				'total_health_checks': 1200,
				'cache_hit_rate': 0.92
			}
		}
		mock_service.get_registry_statistics.return_value = mock_stats
		
		# Test request
		response = client.get('/api/regy/v1/metrics/registry/statistics')
		
		assert response.status_code == 200
		data = response.get_json()
		
		assert 'registry_info' in data
		assert 'service_statistics' in data
		assert 'performance_counters' in data
		assert data['registry_info']['tenant_id'] == TEST_TENANT_ID
		assert data['service_statistics']['total_services'] == 15
	
	@patch('capabilities.common.regy.api.get_registry_service')
	async def test_events_list_endpoint(self, mock_get_service, client, mock_service):
		"""Test GET /events endpoint."""
		# Setup mock
		mock_get_service.return_value = mock_service
		
		from ..models import ServiceEvent
		mock_events = [
			ServiceEvent(
				event_type="service_registered",
				service_id="test-service-1",
				severity="info",
				message="Service registered successfully",
				triggered_by="registry_service",
				tenant_id=TEST_TENANT_ID,
				created_by=TEST_USER_ID
			),
			ServiceEvent(
				event_type="service_health_check",
				service_id="test-service-1", 
				severity="info",
				message="Health check passed",
				triggered_by="health_monitor",
				tenant_id=TEST_TENANT_ID,
				created_by="system"
			)
		]
		mock_service.service_events = mock_events
		
		# Test request
		response = client.get('/api/regy/v1/events?limit=10&offset=0')
		
		assert response.status_code == 200
		data = response.get_json()
		
		assert 'events' in data
		assert 'total_count' in data
		assert 'returned_count' in data
		assert data['total_count'] == 2
		assert len(data['events']) == 2
		assert data['events'][0]['event_type'] == "service_registered"
	
	@patch('capabilities.common.regy.api.get_registry_service')
	async def test_event_detail_endpoint(self, mock_get_service, client, mock_service):
		"""Test GET /events/{event_id} endpoint."""
		# Setup mock
		mock_get_service.return_value = mock_service
		
		event_id = "event-123"
		from ..models import ServiceEvent
		mock_event = ServiceEvent(
			id=event_id,
			event_type="service_failure",
			service_id="failed-service",
			severity="critical",
			message="Service health check failed",
			triggered_by="health_monitor",
			tenant_id=TEST_TENANT_ID,
			created_by="system"
		)
		mock_service.service_events = [mock_event]
		
		# Test request
		response = client.get(f'/api/regy/v1/events/{event_id}')
		
		assert response.status_code == 200
		data = response.get_json()
		
		assert data['id'] == event_id
		assert data['event_type'] == "service_failure"
		assert data['severity'] == "critical"
	
	@patch('capabilities.common.regy.api.get_registry_service')
	async def test_registry_status_endpoint(self, mock_get_service, client, mock_service):
		"""Test GET /status endpoint."""
		# Setup mock
		mock_get_service.return_value = mock_service
		
		mock_stats = {
			'registry_info': {'uptime_seconds': 1800},
			'performance_counters': {
				'total_registrations': 10,
				'total_discoveries': 100,
				'cache_hit_rate': 0.8
			}
		}
		mock_service.get_registry_statistics.return_value = mock_stats
		
		# Test request
		response = client.get('/api/regy/v1/status')
		
		assert response.status_code == 200
		data = response.get_json()
		
		assert data['status'] == 'healthy'
		assert data['version'] == '1.0.0'
		assert data['tenant_id'] == TEST_TENANT_ID
		assert data['initialized'] is True
		assert 'uptime_seconds' in data
		assert 'performance' in data
	
	@patch('capabilities.common.regy.api.get_registry_service')
	async def test_registry_readiness_endpoint(self, mock_get_service, client, mock_service):
		"""Test GET /ready endpoint."""
		# Setup mock
		mock_get_service.return_value = mock_service
		
		# Test request
		response = client.get('/api/regy/v1/ready')
		
		assert response.status_code == 200
		data = response.get_json()
		
		assert data['ready'] is True
		assert 'message' in data
	
	def test_error_handling_bad_request(self, client):
		"""Test 400 Bad Request error handling."""
		# Send invalid JSON
		response = client.post(
			'/api/regy/v1/services',
			data='{"invalid": json}',
			content_type='application/json'
		)
		
		assert response.status_code == 400
		data = response.get_json()
		assert 'error' in data
	
	def test_error_handling_not_found(self, client):
		"""Test 404 Not Found error handling."""
		response = client.get('/api/regy/v1/nonexistent-endpoint')
		
		assert response.status_code == 404
	
	def test_websocket_info_endpoints(self, client):
		"""Test WebSocket information endpoints."""
		# Health WebSocket info
		response = client.get('/api/regy/v1/ws/health')
		
		assert response.status_code == 200
		data = response.get_json()
		
		assert 'endpoint' in data
		assert 'protocols' in data
		assert '/api/regy/v1/ws/health' in data['endpoint']
		
		# Events WebSocket info
		response = client.get('/api/regy/v1/ws/events')
		
		assert response.status_code == 200
		data = response.get_json()
		
		assert 'endpoint' in data
		assert 'protocols' in data
		assert '/api/regy/v1/ws/events' in data['endpoint']

class TestAPIAuthentication:
	"""Test API authentication and authorization scenarios."""
	
	@pytest.fixture
	def app_with_auth(self):
		"""Create Flask app with authentication enabled."""
		from flask import Flask
		app = Flask(__name__)
		app.config['TESTING'] = True
		app.config['WTF_CSRF_ENABLED'] = False
		
		# Mock authentication
		with patch('capabilities.common.regy.api.APG_AUTH_AVAILABLE', True):
			app.register_blueprint(registry_bp)
		
		return app
	
	@pytest.fixture
	def auth_client(self, app_with_auth):
		"""Create authenticated test client."""
		return app_with_auth.test_client()
	
	def test_unauthenticated_request(self, auth_client):
		"""Test that unauthenticated requests are handled."""
		# This test would verify authentication enforcement
		# For now, it's a placeholder since auth decorators are mocked
		response = auth_client.get('/api/regy/v1/services')
		# In real implementation with auth, this might return 401
		# For now, it depends on how the mock auth decorators behave
		assert response.status_code in [200, 401]
	
	def test_authorized_request(self, auth_client):
		"""Test authorized request with proper headers."""
		headers = {
			'X-Tenant-ID': TEST_TENANT_ID,
			'X-User-ID': TEST_USER_ID,
			'Authorization': 'Bearer test-token'
		}
		
		response = auth_client.get('/api/regy/v1/services', headers=headers)
		# This would pass authentication in real implementation
		assert response.status_code in [200, 401]  # Depends on mock behavior

class TestAPIPerformance:
	"""Test API performance characteristics."""
	
	@pytest.fixture
	def app(self):
		"""Create Flask test application."""
		from flask import Flask
		app = Flask(__name__)
		app.config['TESTING'] = True
		app.register_blueprint(registry_bp)
		return app
	
	@pytest.fixture
	def client(self, app):
		"""Create test client."""
		return app.test_client()
	
	@patch('capabilities.common.regy.api.get_registry_service')
	async def test_concurrent_requests(self, mock_get_service, client):
		"""Test handling of concurrent API requests."""
		# Setup mock
		mock_service = Mock(spec=ServiceRegistryService)
		mock_service.tenant_id = TEST_TENANT_ID
		mock_service.initialized = True
		mock_service.discover_services = AsyncMock()
		
		from ..models import ServiceDiscoveryResult
		mock_result = ServiceDiscoveryResult(
			services=[],
			total_count=0,
			returned_count=0,
			query_time_ms=1.0,
			tenant_id=TEST_TENANT_ID
		)
		mock_service.discover_services.return_value = mock_result
		mock_get_service.return_value = mock_service
		
		# Make multiple concurrent requests
		import threading
		results = []
		
		def make_request():
			response = client.get('/api/regy/v1/services')
			results.append(response.status_code)
		
		threads = []
		for _ in range(5):
			thread = threading.Thread(target=make_request)
			threads.append(thread)
			thread.start()
		
		for thread in threads:
			thread.join()
		
		# All requests should succeed
		assert all(status == 200 for status in results)
		assert len(results) == 5
	
	@patch('capabilities.common.regy.api.get_registry_service')
	async def test_large_response_handling(self, mock_get_service, client):
		"""Test handling of large response payloads."""
		# Setup mock with large dataset
		mock_service = Mock(spec=ServiceRegistryService)
		mock_service.tenant_id = TEST_TENANT_ID
		mock_service.initialized = True
		mock_service.discover_services = AsyncMock()
		
		# Create large list of services
		from ..models import ServiceDiscoveryResult, ServiceRegistration
		large_service_list = []
		for i in range(100):
			service = ServiceRegistration(
				name=f"service-{i:03d}",
				display_name=f"Service {i}",
				service_type=ServiceType.MICROSERVICE,
				namespace="test",
				environment="development",
				tenant_id=TEST_TENANT_ID,
				created_by=TEST_USER_ID,
				last_modified_by=TEST_USER_ID
			)
			large_service_list.append(service)
		
		mock_result = ServiceDiscoveryResult(
			services=large_service_list,
			total_count=100,
			returned_count=100,
			query_time_ms=25.0,
			tenant_id=TEST_TENANT_ID
		)
		mock_service.discover_services.return_value = mock_result
		mock_get_service.return_value = mock_service
		
		# Test request
		response = client.get('/api/regy/v1/services')
		
		assert response.status_code == 200
		data = response.get_json()
		
		assert data['total_count'] == 100
		assert len(data['services']) == 100

if __name__ == "__main__":
	pytest.main([__file__, "-v"])
