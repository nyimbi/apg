#!/usr/bin/env python3
"""
Registry (regy) - Pytest Configuration
======================================

Pytest fixtures and configuration for Registry capability testing.

Author: APG Platform Team
Copyright: © 2025 Datacraft
"""

import asyncio
import inspect
import pytest
import sys
import types
import unittest.mock as mock
from typing import Dict, List, Any, Generator
from unittest.mock import Mock, patch, AsyncMock

if "pytest_httpserver" not in sys.modules:
	pytest_httpserver_stub = types.ModuleType("pytest_httpserver")

	class HTTPServer:
		"""Small pytest-httpserver stand-in for tests that only import it."""

		def __init__(self, *args, **kwargs):
			self.host = "127.0.0.1"
			self.port = 0

		def url_for(self, path: str = "/") -> str:
			return f"http://{self.host}:{self.port}{path}"

	pytest_httpserver_stub.HTTPServer = HTTPServer
	sys.modules["pytest_httpserver"] = pytest_httpserver_stub

try:
	import flask_appbuilder
except ImportError:
	flask_appbuilder = types.ModuleType("flask_appbuilder")
	flask_appbuilder.AppBuilder = lambda *args, **kwargs: types.SimpleNamespace()
	sys.modules["flask_appbuilder"] = flask_appbuilder

try:
	from werkzeug.local import LocalProxy
except ImportError:
	LocalProxy = None

_original_is_async_obj = mock._is_async_obj

def _safe_is_async_obj(obj):
	"""Avoid forcing Flask LocalProxy objects to resolve during patch setup."""
	if LocalProxy is not None and isinstance(obj, LocalProxy):
		return False
	return _original_is_async_obj(obj)

mock._is_async_obj = _safe_is_async_obj

if not hasattr(flask_appbuilder, "SQLA"):
	class SQLA:
		def __init__(self, app=None):
			self.app = app
			self.session = types.SimpleNamespace()

	flask_appbuilder.SQLA = SQLA

if hasattr(flask_appbuilder, "AppBuilder"):
	class TestAppBuilder:
		"""Minimal Flask-AppBuilder stand-in for REGY tests."""

		def __init__(self, app=None, session=None, **kwargs):
			self.app = app
			self.session = session
			self.views = []
			self.links = []
			if app is not None:
				app.appbuilder = self

		def add_view(self, view, *args, **kwargs):
			self.views.append((view, args, kwargs))
			return view

		def add_link(self, *args, **kwargs):
			self.links.append((args, kwargs))

	flask_appbuilder.AppBuilder = TestAppBuilder

# Test constants
TEST_TENANT_ID = "test-tenant"
TEST_USER_ID = "test-user"
TEST_SERVICE_NAME = "test-service"

@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
	"""Create an event loop for the test session."""
	loop = asyncio.new_event_loop()
	asyncio.set_event_loop(loop)
	yield loop
	loop.close()

@pytest.fixture
def mock_apg_dependencies():
	"""Mock APG service dependencies."""
	with patch('capabilities.common.regy.service.AuthService') as mock_auth, \
		 patch('capabilities.common.regy.service.MonitoringService') as mock_monitoring, \
		 patch('capabilities.common.regy.service.AuditLoggingService') as mock_audit, \
		 patch('capabilities.common.regy.service.ConfigurationService') as mock_config:
		
		# Setup async methods
		mock_auth.return_value.initialize = AsyncMock()
		mock_auth.return_value.validate_permission = AsyncMock(return_value=True)
		
		mock_monitoring.return_value.initialize = AsyncMock()
		mock_monitoring.return_value.record_metric = AsyncMock()
		
		mock_audit.return_value.initialize = AsyncMock()
		mock_audit.return_value.log_event = AsyncMock()
		
		mock_config.return_value.initialize = AsyncMock()
		mock_config.return_value.get_config = AsyncMock(return_value={})
		
		yield {
			'auth': mock_auth,
			'monitoring': mock_monitoring,
			'audit': mock_audit,
			'config': mock_config
		}

@pytest.fixture
def sample_service_data():
	"""Sample service data for testing."""
	return {
		"name": TEST_SERVICE_NAME,
		"display_name": "Test Service",
		"description": "A service for testing purposes",
		"service_type": "rest_api",
		"namespace": "test",
		"environment": "development",
		"base_path": "/api/v1",
		"tags": ["test", "api", "development"],
		"discovery_enabled": True,
		"health_check_enabled": True,
		"circuit_breaker_enabled": True
	}

@pytest.fixture
def sample_service_instance_data():
	"""Sample service instance data for testing."""
	return {
		"instance_name": "test-instance",
		"host": "test.example.com",
		"port": 8080,
		"base_url": "http://test.example.com:8080",
		"weight": 100,
		"max_connections": 500,
		"environment": "development",
		"tags": ["primary", "test"]
	}

@pytest.fixture
def sample_health_check_data():
	"""Sample health check data for testing."""
	return {
		"name": "HTTP Health Check",
		"type": "http",
		"enabled": True,
		"url": "http://test.example.com:8080/health",
		"interval_seconds": 30,
		"timeout_seconds": 10,
		"healthy_threshold": 2,
		"unhealthy_threshold": 3,
		"expected_response_codes": [200, 201],
		"adaptive_intervals": False,
		"anomaly_detection": False
	}

@pytest.fixture
def sample_circuit_breaker_data():
	"""Sample circuit breaker data for testing."""
	return {
		"name": "Test Circuit Breaker",
		"enabled": True,
		"failure_threshold": 5,
		"success_threshold": 3,
		"timeout_seconds": 60,
		"failure_rate_threshold": 50.0,
		"minimum_request_threshold": 10,
		"rolling_window_seconds": 60,
		"adaptive_thresholds": False,
		"pattern_recognition": False,
		"intelligent_recovery": False
	}

@pytest.fixture
def mock_flask_app():
	"""Mock Flask application for testing."""
	from flask import Flask
	app = Flask(__name__)
	app.config['TESTING'] = True
	app.config['WTF_CSRF_ENABLED'] = False
	app.config['APG_DEFAULT_TENANT_ID'] = TEST_TENANT_ID
	return app

@pytest.fixture
def mock_registry_service():
	"""Mock ServiceRegistryService for testing."""
	from ..service import ServiceRegistryService
	
	service = Mock(spec=ServiceRegistryService)
	service.tenant_id = TEST_TENANT_ID
	service.initialized = True
	service.ml_models_loaded = False
	service.services = {}
	service.service_health = {}
	service.service_metrics = []
	service.service_events = []
	service.startup_time = asyncio.get_event_loop().time()
	
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
	service._rank_services_intelligently = AsyncMock()
	
	return service

@pytest.fixture
def performance_test_services():
	"""Generate multiple services for performance testing."""
	services = []
	for i in range(100):
		service_data = {
			"name": f"perf-service-{i:03d}",
			"display_name": f"Performance Test Service {i}",
			"description": f"Service {i} for performance testing",
			"service_type": "microservice",
			"namespace": "performance",
			"environment": "test",
			"base_path": f"/api/v{i % 3 + 1}",
			"tags": [f"batch-{i // 20}", "performance", "test"],
			"total_requests": 1000 + i * 100,
			"total_errors": i % 10,
			"average_response_time": 50.0 + i * 2.5,
			"uptime_percentage": 95.0 + (i % 5)
		}
		services.append(service_data)
	return services

@pytest.fixture
def mock_websocket_manager():
	"""Mock WebSocket manager for real-time features."""
	manager = Mock()
	manager.broadcast_health_update = AsyncMock()
	manager.broadcast_service_event = AsyncMock()
	manager.add_client = Mock()
	manager.remove_client = Mock()
	manager.get_connected_clients = Mock(return_value=[])
	return manager

# Pytest configuration
def pytest_configure(config):
	"""Configure pytest with custom markers."""
	config.addinivalue_line(
		"markers", "integration: mark test as integration test"
	)
	config.addinivalue_line(
		"markers", "performance: mark test as performance test"  
	)
	config.addinivalue_line(
		"markers", "ml_features: mark test as requiring ML features"
	)
	config.addinivalue_line(
		"markers", "apg_integration: mark test as requiring APG integration"
	)

def pytest_pyfunc_call(pyfuncitem):
	"""Run coroutine results from async tests hidden behind patch wrappers."""
	if pyfuncitem.get_closest_marker("asyncio") is not None:
		return None
	if not getattr(pyfuncitem, "_regy_async_test", False):
		return None

	testfunction = pyfuncitem.obj
	fixture_names = pyfuncitem._fixtureinfo.argnames
	testargs = {
		name: pyfuncitem.funcargs[name]
		for name in fixture_names
		if name in pyfuncitem.funcargs
	}
	result = testfunction(**testargs)
	if inspect.isawaitable(result):
		loop = asyncio.get_event_loop_policy().new_event_loop()
		try:
			loop.run_until_complete(result)
		finally:
			pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
			for task in pending:
				task.cancel()
			if pending:
				loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
			loop.close()
		return True
	return None

def pytest_collection_modifyitems(config, items):
	"""Modify test collection to add markers based on test names."""
	for item in items:
		if (
			item.get_closest_marker("asyncio") is None
			and inspect.iscoroutinefunction(inspect.unwrap(getattr(item, "obj", None)))
		):
			setattr(item, "_regy_async_test", True)

		# Add integration marker to integration tests
		if "integration" in item.nodeid.lower():
			item.add_marker(pytest.mark.integration)
		
		# Add performance marker to performance tests
		if "performance" in item.nodeid.lower() or "load" in item.nodeid.lower():
			item.add_marker(pytest.mark.performance)
		
		# Add ML features marker to ML-related tests
		if any(keyword in item.nodeid.lower() for keyword in ["ml", "intelligent", "predictive", "anomaly"]):
			item.add_marker(pytest.mark.ml_features)
		
		# Add APG integration marker to APG-related tests
		if "apg" in item.nodeid.lower():
			item.add_marker(pytest.mark.apg_integration)

@pytest.fixture(autouse=True)
def setup_test_environment():
	"""Setup test environment for each test."""
	# Setup any global test state
	yield
	# Cleanup after each test
	pass

# Export test constants for use in test modules
__all__ = [
	'TEST_TENANT_ID',
	'TEST_USER_ID', 
	'TEST_SERVICE_NAME'
]
