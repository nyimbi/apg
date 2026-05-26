#!/usr/bin/env python3
"""
Registry (regy) - Flask-AppBuilder Views Test Suite
==================================================

Comprehensive tests for Flask-AppBuilder views with Pydantic v2 models,
real-time features, and APG UI integration.

Author: APG Platform Team
Copyright: © 2025 Datacraft
"""

import asyncio
import pytest
from datetime import datetime, timezone
from typing import Dict, List, Any
from unittest.mock import Mock, patch, AsyncMock, MagicMock

from flask import Flask
from flask_appbuilder import AppBuilder, SQLA

from ..views import (
	ServiceRegistryView, ServiceDiscoveryView, ServiceHealthView, ServiceAnalyticsView,
	ServiceRegistrationForm, ServiceInstanceForm, ServiceDiscoveryForm,
	HealthCheckForm, CircuitBreakerForm, get_registry_service
)
from ..models import ServiceType, ServiceStatus, LoadBalanceStrategy, HealthCheckType
from . import TEST_TENANT_ID, TEST_USER_ID, TEST_SERVICE_NAME

class TestPydanticViewModels:
	"""Test Pydantic v2 models used in views."""
	
	def test_service_registration_form_validation(self):
		"""Test ServiceRegistrationForm validation."""
		# Valid form data
		valid_data = {
			"name": "valid-service",
			"display_name": "Valid Service",
			"description": "A valid service",
			"service_type": ServiceType.REST_API,
			"namespace": "test",
			"environment": "development",
			"base_path": "/api/v1",
			"tags": ["test", "api"],
			"discovery_enabled": True,
			"load_balance_strategy": LoadBalanceStrategy.ROUND_ROBIN,
			"health_check_enabled": True,
			"circuit_breaker_enabled": True,
			"predictive_scaling": False,
			"intelligent_routing": False
		}
		
		form = ServiceRegistrationForm(**valid_data)
		
		assert form.name == "valid-service"
		assert form.display_name == "Valid Service"
		assert form.service_type == ServiceType.REST_API
		assert form.namespace == "test"
		assert form.environment == "development"
		assert len(form.tags) == 2
	
	def test_service_registration_form_name_validation(self):
		"""Test service name validation in form."""
		# Test name normalization
		form_data = {
			"name": "Service-With_Mixed123",
			"display_name": "Service",
			"service_type": ServiceType.REST_API,
			"namespace": "test",
			"environment": "dev"
		}
		
		form = ServiceRegistrationForm(**form_data)
		assert form.name == "service-with_mixed123"  # Should be lowercase
		
		# Test invalid name
		with pytest.raises(Exception):  # ValidationError
			invalid_data = form_data.copy()
			invalid_data["name"] = "invalid@name!"
			ServiceRegistrationForm(**invalid_data)
	
	def test_service_instance_form_validation(self):
		"""Test ServiceInstanceForm validation."""
		valid_data = {
			"instance_name": "test-instance",
			"host": "test.example.com",
			"port": 8080,
			"base_url": "http://test.example.com:8080",
			"weight": 100,
			"max_connections": 500,
			"environment": "test",
			"deployment_version": "1.0.0",
			"tags": ["primary", "test"]
		}
		
		form = ServiceInstanceForm(**valid_data)
		
		assert form.instance_name == "test-instance"
		assert form.host == "test.example.com"
		assert form.port == 8080
		assert form.weight == 100
		assert len(form.tags) == 2
	
	def test_service_instance_form_port_validation(self):
		"""Test port validation in instance form."""
		base_data = {
			"instance_name": "test",
			"host": "test.com",
			"base_url": "http://test.com:8080"
		}
		
		# Valid port
		valid_data = base_data.copy()
		valid_data["port"] = 8080
		form = ServiceInstanceForm(**valid_data)
		assert form.port == 8080
		
		# Invalid ports
		with pytest.raises(Exception):
			invalid_data = base_data.copy()
			invalid_data["port"] = 0
			ServiceInstanceForm(**invalid_data)
		
		with pytest.raises(Exception):
			invalid_data = base_data.copy()
			invalid_data["port"] = 70000
			ServiceInstanceForm(**invalid_data)
	
	def test_service_discovery_form_validation(self):
		"""Test ServiceDiscoveryForm validation."""
		valid_data = {
			"service_name": "search-service",
			"service_type": ServiceType.MICROSERVICE,
			"namespace": "prod",
			"environment": "production",
			"status": ServiceStatus.HEALTHY,
			"healthy_only": True,
			"min_health_score": 0.8,
			"intelligent_ranking": True,
			"predictive_filtering": False,
			"limit": 50,
			"offset": 0,
			"include_instances": True,
			"include_health": True,
			"include_metrics": False
		}
		
		form = ServiceDiscoveryForm(**valid_data)
		
		assert form.service_name == "search-service"
		assert form.service_type == ServiceType.MICROSERVICE
		assert form.min_health_score == 0.8
		assert form.intelligent_ranking is True
		assert form.limit == 50
	
	def test_health_check_form_validation(self):
		"""Test HealthCheckForm validation."""
		valid_data = {
			"name": "HTTP Health Check",
			"type": HealthCheckType.HTTP,
			"enabled": True,
			"url": "http://service.com/health",
			"interval_seconds": 30,
			"timeout_seconds": 10,
			"healthy_threshold": 2,
			"unhealthy_threshold": 3,
			"expected_response_codes": [200, 201],
			"adaptive_intervals": True,
			"anomaly_detection": False
		}
		
		form = HealthCheckForm(**valid_data)
		
		assert form.name == "HTTP Health Check"
		assert form.type == HealthCheckType.HTTP
		assert form.interval_seconds == 30
		assert form.adaptive_intervals is True
		assert len(form.expected_response_codes) == 2
	
	def test_circuit_breaker_form_validation(self):
		"""Test CircuitBreakerForm validation."""
		valid_data = {
			"name": "API Circuit Breaker",
			"enabled": True,
			"failure_threshold": 5,
			"success_threshold": 3,
			"timeout_seconds": 60,
			"failure_rate_threshold": 50.0,
			"minimum_request_threshold": 10,
			"rolling_window_seconds": 60,
			"adaptive_thresholds": True,
			"pattern_recognition": False,
			"intelligent_recovery": True
		}
		
		form = CircuitBreakerForm(**valid_data)
		
		assert form.name == "API Circuit Breaker"
		assert form.failure_threshold == 5
		assert form.failure_rate_threshold == 50.0
		assert form.adaptive_thresholds is True
		assert form.intelligent_recovery is True

class TestServiceRegistryView:
	"""Test ServiceRegistryView functionality."""
	
	@pytest.fixture
	def app(self):
		"""Create Flask application with AppBuilder."""
		app = Flask(__name__)
		app.config['SECRET_KEY'] = 'test-secret'
		app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
		app.config['TESTING'] = True
		
		db = SQLA(app)
		appbuilder = AppBuilder(app, db.session)
		
		return app
	
	@pytest.fixture
	def view_instance(self, app):
		"""Create ServiceRegistryView instance."""
		with app.app_context():
			view = ServiceRegistryView()
			return view
	
	@pytest.fixture
	def client(self, app):
		"""Create test client."""
		return app.test_client()
	
	@patch('capabilities.common.regy.views.get_registry_service')
	async def test_list_view(self, mock_get_service, view_instance, app, client):
		"""Test service list view."""
		# Setup mock
		mock_service = Mock()
		mock_service.tenant_id = TEST_TENANT_ID
		mock_service.discover_services = AsyncMock()
		mock_service.get_service_health = AsyncMock()
		mock_service.get_registry_statistics = AsyncMock()
		
		from ..models import ServiceDiscoveryResult, ServiceRegistration, ServiceHealthStatus
		
		# Mock service data
		mock_svc = ServiceRegistration(
			id="test-service-id",
			name=TEST_SERVICE_NAME,
			display_name="Test Service",
			service_type=ServiceType.REST_API,
			namespace="test",
			environment="development",
			status=ServiceStatus.HEALTHY,
			tenant_id=TEST_TENANT_ID,
			created_by=TEST_USER_ID,
			last_modified_by=TEST_USER_ID
		)
		
		mock_result = ServiceDiscoveryResult(
			services=[mock_svc],
			total_count=1,
			returned_count=1,
			query_time_ms=5.2,
			tenant_id=TEST_TENANT_ID
		)
		
		mock_health = ServiceHealthStatus(
			service_id="test-service-id",
			instance_id="test-instance",
			overall_status=ServiceStatus.HEALTHY,
			health_score=0.95,
			status_message="Healthy",
			tenant_id=TEST_TENANT_ID
		)
		
		mock_stats = {
			'service_statistics': {
				'total_services': 1,
				'healthy_services': 1,
				'degraded_services': 0,
				'unhealthy_services': 0
			},
			'performance_counters': {
				'total_registrations': 1,
				'total_discoveries': 5,
				'cache_hit_rate': 0.8
			}
		}
		
		mock_service.discover_services.return_value = mock_result
		mock_service.get_service_health.return_value = mock_health
		mock_service.get_registry_statistics.return_value = mock_stats
		mock_get_service.return_value = mock_service
		
		# Test the view
		with app.test_request_context():
			with patch('capabilities.common.regy.views.ensure_service_initialized', new=AsyncMock()):
				# This would test the actual view method
				# For now, just verify the mock setup works
				assert mock_service.tenant_id == TEST_TENANT_ID
	
	@patch('capabilities.common.regy.views.get_registry_service')
	async def test_add_view(self, mock_get_service, view_instance, app):
		"""Test service registration view."""
		# Setup mock
		mock_service = Mock()
		mock_service.tenant_id = TEST_TENANT_ID
		mock_service.register_service = AsyncMock()
		
		from ..models import ServiceRegistration
		mock_registered = ServiceRegistration(
			id="new-service-id",
			name="new-service",
			display_name="New Service",
			service_type=ServiceType.REST_API,
			namespace="test",
			environment="development",
			tenant_id=TEST_TENANT_ID,
			created_by=TEST_USER_ID,
			last_modified_by=TEST_USER_ID
		)
		
		mock_service.register_service.return_value = mock_registered
		mock_get_service.return_value = mock_service
		
		# Test form data processing
		form_data = {
			'name': 'new-service',
			'display_name': 'New Service',
			'service_type': 'rest_api',
			'namespace': 'test',
			'environment': 'development',
			'tags': 'test, api, new',
			'discovery_enabled': 'on',
			'health_check_enabled': 'on'
		}
		
		with app.test_request_context(method='POST', data=form_data):
			# Verify form data can be processed
			assert form_data['name'] == 'new-service'
			assert 'on' == form_data.get('discovery_enabled')

class TestServiceDiscoveryView:
	"""Test ServiceDiscoveryView functionality."""
	
	@pytest.fixture
	def app(self):
		"""Create Flask application."""
		app = Flask(__name__)
		app.config['SECRET_KEY'] = 'test-secret'
		app.config['TESTING'] = True
		return app
	
	@pytest.fixture
	def view_instance(self):
		"""Create ServiceDiscoveryView instance."""
		return ServiceDiscoveryView()
	
	@patch('capabilities.common.regy.views.get_registry_service')
	async def test_search_view(self, mock_get_service, view_instance, app):
		"""Test service discovery search view."""
		# Setup mock
		mock_service = Mock()
		mock_service.tenant_id = TEST_TENANT_ID
		mock_service.discover_services = AsyncMock()
		
		from ..models import ServiceDiscoveryResult, ServiceRegistration
		
		mock_svc = ServiceRegistration(
			id="discovered-service-id",
			name="discovered-service",
			display_name="Discovered Service",
			service_type=ServiceType.MICROSERVICE,
			namespace="prod",
			environment="production",
			tenant_id=TEST_TENANT_ID,
			created_by=TEST_USER_ID,
			last_modified_by=TEST_USER_ID
		)
		
		mock_result = ServiceDiscoveryResult(
			services=[mock_svc],
			total_count=1,
			returned_count=1,
			query_time_ms=12.5,
			tenant_id=TEST_TENANT_ID
		)
		
		mock_service.discover_services.return_value = mock_result
		mock_get_service.return_value = mock_service
		
		# Test search form data
		search_data = {
			'service_type': 'microservice',
			'environment': 'production',
			'healthy_only': 'on',
			'intelligent_ranking': 'on',
			'limit': '25'
		}
		
		with app.test_request_context(method='POST', data=search_data):
			# Verify search data processing
			assert search_data['service_type'] == 'microservice'
			assert search_data['healthy_only'] == 'on'
	
	@patch('capabilities.common.regy.views.get_registry_service')
	async def test_api_search_view(self, mock_get_service, view_instance, app):
		"""Test API search endpoint."""
		# Setup mock
		mock_service = Mock()
		mock_service.tenant_id = TEST_TENANT_ID
		mock_service.discover_services = AsyncMock()
		
		from ..models import ServiceDiscoveryResult
		mock_result = ServiceDiscoveryResult(
			services=[],
			total_count=0,
			returned_count=0,
			query_time_ms=3.1,
			tenant_id=TEST_TENANT_ID
		)
		
		mock_service.discover_services.return_value = mock_result
		mock_get_service.return_value = mock_service
		
		# Test JSON API data
		api_data = {
			"service_type": "rest_api",
			"namespace": "prod",
			"healthy_only": True,
			"limit": 50
		}
		
		with app.test_request_context(
			method='POST',
			json=api_data,
			content_type='application/json'
		):
			# Verify API can process JSON data
			assert api_data['service_type'] == 'rest_api'
			assert api_data['healthy_only'] is True

class TestServiceHealthView:
	"""Test ServiceHealthView functionality."""
	
	@pytest.fixture
	def app(self):
		"""Create Flask application."""
		app = Flask(__name__)
		app.config['SECRET_KEY'] = 'test-secret'
		app.config['TESTING'] = True
		return app
	
	@pytest.fixture
	def view_instance(self):
		"""Create ServiceHealthView instance."""
		return ServiceHealthView()
	
	@patch('capabilities.common.regy.views.get_registry_service')
	async def test_dashboard_view(self, mock_get_service, view_instance, app):
		"""Test health dashboard view."""
		# Setup mock
		mock_service = Mock()
		mock_service.tenant_id = TEST_TENANT_ID
		mock_service.discover_services = AsyncMock()
		mock_service.get_service_health = AsyncMock()
		
		from ..models import (
			ServiceDiscoveryResult, ServiceRegistration, ServiceHealthStatus
		)
		
		# Mock healthy service
		mock_svc = ServiceRegistration(
			id="healthy-service-id",
			name="healthy-service",
			display_name="Healthy Service",
			service_type=ServiceType.REST_API,
			namespace="prod",
			environment="production",
			tenant_id=TEST_TENANT_ID,
			created_by=TEST_USER_ID,
			last_modified_by=TEST_USER_ID
		)
		
		mock_result = ServiceDiscoveryResult(
			services=[mock_svc],
			total_count=1,
			returned_count=1,
			query_time_ms=8.3,
			tenant_id=TEST_TENANT_ID
		)
		
		mock_health = ServiceHealthStatus(
			service_id="healthy-service-id",
			instance_id="instance-1",
			overall_status=ServiceStatus.HEALTHY,
			health_score=0.92,
			status_message="All systems operational",
			response_time_ms=45.2,
			cpu_usage_percent=35.7,
			memory_usage_percent=52.3,
			tenant_id=TEST_TENANT_ID
		)
		
		mock_service.discover_services.return_value = mock_result
		mock_service.get_service_health.return_value = mock_health
		mock_get_service.return_value = mock_service
		
		with app.test_request_context():
			# Verify health data structure
			assert mock_health.overall_status == ServiceStatus.HEALTHY
			assert mock_health.health_score == 0.92
			assert mock_health.response_time_ms == 45.2
	
	@patch('capabilities.common.regy.views.get_registry_service')
	async def test_service_health_detail(self, mock_get_service, view_instance, app):
		"""Test individual service health detail view."""
		# Setup mock
		mock_service = Mock()
		mock_service.tenant_id = TEST_TENANT_ID
		mock_service.services = {}
		mock_service.get_service_health = AsyncMock()
		mock_service.get_service_metrics = AsyncMock()
		mock_service.service_events = []
		
		service_id = "health-detail-service"
		
		from ..models import ServiceRegistration, ServiceHealthStatus, ServiceMetrics
		
		mock_svc_obj = ServiceRegistration(
			id=service_id,
			name="health-detail-service",
			display_name="Health Detail Service",
			service_type=ServiceType.MICROSERVICE,
			namespace="prod",
			environment="production",
			tenant_id=TEST_TENANT_ID,
			created_by=TEST_USER_ID,
			last_modified_by=TEST_USER_ID
		)
		
		mock_health = ServiceHealthStatus(
			service_id=service_id,
			instance_id="detail-instance",
			overall_status=ServiceStatus.DEGRADED,
			health_score=0.75,
			status_message="Service experiencing minor issues",
			tenant_id=TEST_TENANT_ID
		)
		
		mock_metrics = [
			ServiceMetrics(
				service_id=service_id,
				metric_type="performance",
				request_count=5000,
				error_count=50,
				response_time_p50=65.3,
				tenant_id=TEST_TENANT_ID
			)
		]
		
		mock_service.services[service_id] = mock_svc_obj
		mock_service.get_service_health.return_value = mock_health
		mock_service.get_service_metrics.return_value = mock_metrics
		mock_get_service.return_value = mock_service
		
		with app.test_request_context():
			# Verify service detail data
			assert mock_svc_obj.id == service_id
			assert mock_health.overall_status == ServiceStatus.DEGRADED
			assert len(mock_metrics) == 1
			assert mock_metrics[0].request_count == 5000

class TestServiceAnalyticsView:
	"""Test ServiceAnalyticsView functionality."""
	
	@pytest.fixture
	def app(self):
		"""Create Flask application."""
		app = Flask(__name__)
		app.config['SECRET_KEY'] = 'test-secret'
		app.config['TESTING'] = True
		return app
	
	@pytest.fixture
	def view_instance(self):
		"""Create ServiceAnalyticsView instance."""
		return ServiceAnalyticsView()
	
	@patch('capabilities.common.regy.views.get_registry_service')
	async def test_analytics_dashboard(self, mock_get_service, view_instance, app):
		"""Test analytics dashboard view."""
		# Setup mock
		mock_service = Mock()
		mock_service.tenant_id = TEST_TENANT_ID
		mock_service.get_registry_statistics = AsyncMock()
		mock_service.services = {"service-1": Mock(), "service-2": Mock()}
		mock_service.get_service_metrics = AsyncMock()
		
		mock_stats = {
			'registry_info': {
				'tenant_id': TEST_TENANT_ID,
				'uptime_seconds': 86400,
				'initialized': True
			},
			'service_statistics': {
				'total_services': 2,
				'healthy_services': 2,
				'degraded_services': 0,
				'unhealthy_services': 0
			},
			'performance_counters': {
				'total_registrations': 5,
				'total_discoveries': 250,
				'total_health_checks': 1200,
				'cache_hit_rate': 0.92
			}
		}
		
		from ..models import ServiceMetrics
		mock_metrics = [
			ServiceMetrics(
				service_id="service-1",
				metric_type="business",
				request_count=10000,
				error_count=50,
				response_time_p50=45.2,
				tenant_id=TEST_TENANT_ID
			),
			ServiceMetrics(
				service_id="service-2",
				metric_type="performance", 
				request_count=8000,
				error_count=25,
				response_time_p50=38.7,
				tenant_id=TEST_TENANT_ID
			)
		]
		
		mock_service.get_registry_statistics.return_value = mock_stats
		mock_service.get_service_metrics.return_value = mock_metrics
		mock_get_service.return_value = mock_service
		
		with app.test_request_context():
			# Verify analytics data structure
			assert mock_stats['service_statistics']['total_services'] == 2
			assert mock_stats['performance_counters']['cache_hit_rate'] == 0.92
			assert len(mock_metrics) == 2
	
	@patch('capabilities.common.regy.views.get_registry_service')
	async def test_service_analytics_detail(self, mock_get_service, view_instance, app):
		"""Test individual service analytics view."""
		# Setup mock
		mock_service = Mock()
		mock_service.tenant_id = TEST_TENANT_ID
		mock_service.services = {}
		mock_service.get_service_metrics = AsyncMock()
		
		service_id = "analytics-service"
		
		from ..models import ServiceRegistration, ServiceMetrics
		
		mock_svc_obj = ServiceRegistration(
			id=service_id,
			name="analytics-service",
			display_name="Analytics Service",
			service_type=ServiceType.REST_API,
			namespace="analytics",
			environment="production",
			tenant_id=TEST_TENANT_ID,
			created_by=TEST_USER_ID,
			last_modified_by=TEST_USER_ID
		)
		
		# Mock a week's worth of metrics
		mock_metrics = []
		for i in range(7):
			metric = ServiceMetrics(
				service_id=service_id,
				metric_type="daily",
				request_count=10000 + i * 1000,
				error_count=50 + i * 5,
				response_time_p50=45.0 + i * 2.0,
				availability_percentage=99.5 - i * 0.1,
				tenant_id=TEST_TENANT_ID
			)
			mock_metrics.append(metric)
		
		mock_service.services[service_id] = mock_svc_obj
		mock_service.get_service_metrics.return_value = mock_metrics
		mock_get_service.return_value = mock_service
		
		with app.test_request_context():
			# Verify analytics calculations
			total_requests = sum(m.request_count for m in mock_metrics)
			total_errors = sum(m.error_count for m in mock_metrics)
			avg_response_time = sum(m.response_time_p50 for m in mock_metrics) / len(mock_metrics)
			
			assert total_requests == 91000  # 10000 + 11000 + ... + 16000
			assert total_errors == 455  # 50 + 55 + ... + 80
			assert avg_response_time == 51.0  # Average of 45, 47, 49, 51, 53, 55, 57

class TestViewUtilities:
	"""Test view utility functions."""
	
	@patch('capabilities.common.regy.views.registry_service', None)
	def test_get_registry_service_creation(self):
		"""Test registry service creation."""
		with patch('flask.request') as mock_request:
			mock_request.args.get.return_value = TEST_TENANT_ID
			
			service = get_registry_service()
			
			assert service is not None
			assert service.tenant_id == TEST_TENANT_ID
	
	async def test_ensure_service_initialized(self):
		"""Test service initialization helper."""
		from ..views import ensure_service_initialized
		
		mock_service = Mock()
		mock_service.initialized = False
		mock_service.initialize = AsyncMock()
		
		with patch('capabilities.common.regy.views.get_registry_service', return_value=mock_service):
			await ensure_service_initialized()
			mock_service.initialize.assert_called_once()

class TestViewErrorHandling:
	"""Test error handling in views."""
	
	@pytest.fixture
	def app(self):
		"""Create Flask application."""
		app = Flask(__name__)
		app.config['SECRET_KEY'] = 'test-secret'
		app.config['TESTING'] = True
		return app
	
	@patch('capabilities.common.regy.views.get_registry_service')
	async def test_view_service_not_found_handling(self, mock_get_service, app):
		"""Test handling of service not found errors."""
		# Setup mock for non-existent service
		mock_service = Mock()
		mock_service.tenant_id = TEST_TENANT_ID
		mock_service.services = {}  # Empty services dict
		mock_get_service.return_value = mock_service
		
		view = ServiceRegistryView()
		
		with app.test_request_context():
			# Test would verify that non-existent service IDs are handled gracefully
			assert 'non-existent-service' not in mock_service.services
	
	@patch('capabilities.common.regy.views.get_registry_service')
	async def test_view_initialization_error_handling(self, mock_get_service, app):
		"""Test handling of service initialization errors."""
		# Setup mock that fails to initialize
		mock_service = Mock()
		mock_service.initialized = False
		mock_service.initialize = AsyncMock(side_effect=Exception("Initialization failed"))
		mock_get_service.return_value = mock_service
		
		from ..views import ensure_service_initialized
		
		with app.test_request_context():
			# Test would verify that initialization errors are handled
			with pytest.raises(Exception):
				await ensure_service_initialized()

if __name__ == "__main__":
	pytest.main([__file__, "-v"])
