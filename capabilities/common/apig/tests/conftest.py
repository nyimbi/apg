#!/usr/bin/env python3
"""
APG Intelligent Gateway (APIG) - Test Configuration

Pytest configuration and fixtures for comprehensive APIG testing.
Provides APG-compatible test setup with async support and APG service mocking.

Author: APG Platform Team
Copyright: © 2025 Datacraft
"""

import asyncio
import pytest
from datetime import datetime, timezone
from typing import Dict, Any, AsyncGenerator, Generator
from unittest.mock import Mock, AsyncMock

# Import APIG components - avoiding parent directory imports that cause issues
import sys
import os

# Add current directory to path for local imports
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
	sys.path.insert(0, current_dir)

# Only import our local APIG modules
try:
	from service import APGIntelligentGatewayService
	from models import (
		AgGatewayConfig, AgApiRoute, AgPolicy, AgUpstreamService,
		AgRateLimit, AgCacheConfig, AgHealthCheck, AgTrafficMetrics,
		AgSecurityEvent, AgSecurityPolicy, AgWasmModule, AgHttpRequest,
		AgHttpResponse, EnvironmentType, HttpMethod, PolicyType,
		LoadBalancingAlgorithm, ThreatLevel
	)
except ImportError as e:
	# If imports fail, set up minimal test environment
	print(f"Warning: Could not import APIG modules: {e}")
	class MockModel:
		pass
	
	APGIntelligentGatewayService = MockModel
	AgGatewayConfig = MockModel

# APG Test Configuration
APG_TEST_CONFIG = {
	'tenant_id': 'test-tenant-apig',
	'user_id': 'test-user-apig',
	'environment': 'testing',
	'enable_logging': True,
	'mock_apg_services': True
}

@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
	"""
	Create event loop for the entire test session.
	
	Following APG async patterns - no @pytest.mark.asyncio decorators needed.
	"""
	loop = asyncio.get_event_loop_policy().new_event_loop()
	yield loop
	loop.close()

@pytest.fixture
async def apig_service() -> AsyncGenerator[APGIntelligentGatewayService, None]:
	"""
	Create and initialize APIG service for testing.
	
	Returns:
		APGIntelligentGatewayService: Fully initialized service instance
	"""
	service = APGIntelligentGatewayService(
		tenant_id=APG_TEST_CONFIG['tenant_id'],
		user_id=APG_TEST_CONFIG['user_id'],
		config={'testing': True}
	)
	
	await service.initialize()
	
	yield service
	
	# Cleanup
	await service.shutdown()

@pytest.fixture
def mock_apg_auth_service() -> Mock:
	"""
	Mock APG auth_rbac service for testing.
	
	Returns:
		Mock: Configured mock auth service
	"""
	mock_auth = Mock()
	mock_auth.authenticate.return_value = {
		'authenticated': True,
		'user_id': APG_TEST_CONFIG['user_id'],
		'tenant_id': APG_TEST_CONFIG['tenant_id'],
		'permissions': ['gateway:admin', 'gateway:read', 'gateway:write']
	}
	mock_auth.authorize.return_value = True
	mock_auth.get_user_roles.return_value = ['admin', 'user']
	
	return mock_auth

@pytest.fixture
def mock_apg_monitoring_service() -> AsyncMock:
	"""
	Mock APG monitoring service for testing.
	
	Returns:
		AsyncMock: Configured mock monitoring service
	"""
	mock_monitoring = AsyncMock()
	mock_monitoring.collect_metrics.return_value = {
		'timestamp': datetime.now(timezone.utc),
		'metrics': {
			'requests_per_second': 1000,
			'response_time_p95': 5.2,
			'error_rate': 0.1
		}
	}
	mock_monitoring.create_alert.return_value = True
	mock_monitoring.get_health_status.return_value = 'healthy'
	
	return mock_monitoring

@pytest.fixture
def mock_apg_messaging_service() -> AsyncMock:
	"""
	Mock APG message queuing service for testing.
	
	Returns:
		AsyncMock: Configured mock messaging service
	"""
	mock_messaging = AsyncMock()
	mock_messaging.publish_event.return_value = True
	mock_messaging.subscribe_to_queue.return_value = AsyncMock()
	mock_messaging.get_queue_stats.return_value = {
		'messages_pending': 0,
		'consumers_active': 1
	}
	
	return mock_messaging

@pytest.fixture
def mock_apg_config_service() -> AsyncMock:
	"""
	Mock APG configuration service for testing.
	
	Returns:
		AsyncMock: Configured mock config service
	"""
	mock_config = AsyncMock()
	mock_config.get_configuration.return_value = {
		'gateway_defaults': {
			'max_connections': 10000,
			'timeout_ms': 30000
		},
		'service_discovery': {
			'enabled': True,
			'refresh_interval': 30
		}
	}
	mock_config.discover_services.return_value = [
		{
			'name': 'test-api-service',
			'url': 'http://api.test.local:8080',
			'health_check': '/health'
		}
	]
	
	return mock_config

@pytest.fixture
def mock_apg_ai_orchestration() -> AsyncMock:
	"""
	Mock APG AI orchestration service for testing.
	
	Returns:
		AsyncMock: Configured mock AI service
	"""
	mock_ai = AsyncMock()
	mock_ai.process_request.return_value = {
		'success': True,
		'generated_policy': {
			'name': 'AI Generated Test Policy',
			'type': 'security',
			'configuration': {'rate_limit': 1000},
			'conditions': ['request.ip not in blocked_ips']
		}
	}
	mock_ai.get_available_models.return_value = [
		'llama3.2:latest',
		'mistral:latest'
	]
	
	return mock_ai

@pytest.fixture
def sample_gateway_config() -> AgGatewayConfig:
	"""
	Sample gateway configuration for testing.
	
	Returns:
		AgGatewayConfig: Test gateway configuration
	"""
	return AgGatewayConfig(
		name='test-gateway',
		description='Test API Gateway for unit testing',
		environment=EnvironmentType.TESTING,
		listen_port=8080,
		max_connections=1000,
		tenant_id=APG_TEST_CONFIG['tenant_id'],
		created_by=APG_TEST_CONFIG['user_id']
	)

@pytest.fixture
def sample_upstream_service() -> AgUpstreamService:
	"""
	Sample upstream service for testing.
	
	Returns:
		AgUpstreamService: Test upstream service configuration
	"""
	return AgUpstreamService(
		name='test-api-service',
		base_url='https://api.test.example.com',
		weight=100,
		max_connections=50
	)

@pytest.fixture
def sample_api_route(sample_upstream_service: AgUpstreamService) -> AgApiRoute:
	"""
	Sample API route for testing.
	
	Args:
		sample_upstream_service: Upstream service fixture
		
	Returns:
		AgApiRoute: Test API route configuration
	"""
	return AgApiRoute(
		path='/api/v1/users',
		method=HttpMethod.GET,
		name='list-users',
		description='List all users',
		upstream_services=[sample_upstream_service],
		load_balancing_algorithm=LoadBalancingAlgorithm.ROUND_ROBIN,
		auth_required=True,
		tenant_id=APG_TEST_CONFIG['tenant_id'],
		created_by=APG_TEST_CONFIG['user_id']
	)

@pytest.fixture
def sample_rate_limit() -> AgRateLimit:
	"""
	Sample rate limit configuration for testing.
	
	Returns:
		AgRateLimit: Test rate limit configuration
	"""
	return AgRateLimit(
		requests_per_second=100,
		requests_per_minute=5000,
		burst_size=200,
		key_extractor='client_ip'
	)

@pytest.fixture
def sample_security_policy() -> AgSecurityPolicy:
	"""
	Sample security policy for testing.
	
	Returns:
		AgSecurityPolicy: Test security policy
	"""
	return AgSecurityPolicy(
		name='test-security-policy',
		description='Comprehensive security policy for testing',
		threat_detection_enabled=True,
		anomaly_detection_enabled=True,
		ddos_protection_enabled=True,
		bot_detection_enabled=True,
		tenant_id=APG_TEST_CONFIG['tenant_id'],
		created_by=APG_TEST_CONFIG['user_id']
	)

@pytest.fixture
def sample_wasm_module() -> AgWasmModule:
	"""
	Sample WASM module for testing.
	
	Returns:
		AgWasmModule: Test WASM module configuration
	"""
	return AgWasmModule(
		name='test-request-transformer',
		description='Transform incoming requests for testing',
		wasm_binary_path='/path/to/test-module.wasm',
		entry_point='process_request',
		memory_limit_mb=32,
		execution_timeout_ms=1000,
		tenant_id=APG_TEST_CONFIG['tenant_id'],
		created_by=APG_TEST_CONFIG['user_id']
	)

@pytest.fixture
def sample_http_request() -> AgHttpRequest:
	"""
	Sample HTTP request for testing.
	
	Returns:
		AgHttpRequest: Test HTTP request
	"""
	return AgHttpRequest(
		method=HttpMethod.GET,
		path='/api/v1/users',
		query_string='limit=10&offset=0',
		headers={
			'User-Agent': 'APIG-Test-Client/1.0',
			'Accept': 'application/json',
			'Authorization': 'Bearer test-token'
		},
		client_ip='192.168.1.100',
		tenant_id=APG_TEST_CONFIG['tenant_id'],
		user_id=APG_TEST_CONFIG['user_id']
	)

@pytest.fixture
def sample_security_event() -> AgSecurityEvent:
	"""
	Sample security event for testing.
	
	Returns:
		AgSecurityEvent: Test security event
	"""
	return AgSecurityEvent(
		gateway_id='test-gateway-id',
		event_type='suspicious_traffic',
		threat_level=ThreatLevel.MEDIUM,
		confidence=0.85,
		source_ip='192.168.1.200',
		user_agent='Suspicious-Bot/1.0',
		route_path='/api/v1/admin',
		attack_signature='sql_injection_attempt',
		action_taken='request_blocked',
		blocked=True,
		tenant_id=APG_TEST_CONFIG['tenant_id']
	)

@pytest.fixture
def sample_traffic_metrics() -> AgTrafficMetrics:
	"""
	Sample traffic metrics for testing.
	
	Returns:
		AgTrafficMetrics: Test traffic metrics
	"""
	return AgTrafficMetrics(
		gateway_id='test-gateway-id',
		timestamp=datetime.now(timezone.utc),
		request_count=1000,
		requests_per_second=50.5,
		response_time_p50=2.3,
		response_time_p95=8.7,
		response_time_p99=15.2,
		error_count=5,
		error_rate=0.5,
		bytes_sent=1024000,
		bytes_received=256000,
		active_connections=25,
		tenant_id=APG_TEST_CONFIG['tenant_id']
	)

# Test Utilities

def assert_valid_uuid(uuid_string: str) -> None:
	"""
	Assert that string is a valid UUID.
	
	Args:
		uuid_string: String to validate as UUID
		
	Raises:
		AssertionError: If string is not a valid UUID
	"""
	import uuid
	try:
		uuid.UUID(uuid_string)
	except ValueError:
		raise AssertionError(f"'{uuid_string}' is not a valid UUID")

def assert_recent_timestamp(timestamp: datetime, max_age_seconds: int = 60) -> None:
	"""
	Assert that timestamp is recent (within max_age_seconds).
	
	Args:
		timestamp: Timestamp to check
		max_age_seconds: Maximum age in seconds
		
	Raises:
		AssertionError: If timestamp is too old
	"""
	now = datetime.now(timezone.utc)
	age = (now - timestamp).total_seconds()
	assert age <= max_age_seconds, f"Timestamp is {age:.1f}s old (max: {max_age_seconds}s)"

async def wait_for_condition(condition_func, timeout_seconds: float = 5.0, check_interval: float = 0.1):
	"""
	Wait for async condition to become true.
	
	Args:
		condition_func: Async function that returns bool when condition is met
		timeout_seconds: Maximum time to wait
		check_interval: Time between checks
		
	Raises:
		TimeoutError: If condition not met within timeout
	"""
	import time
	
	start_time = time.time()
	while time.time() - start_time < timeout_seconds:
		if await condition_func():
			return
		await asyncio.sleep(check_interval)
	
	raise TimeoutError(f"Condition not met within {timeout_seconds} seconds")

# Test Markers for Different Test Categories

# Use these markers to categorize tests:
# @pytest.mark.unit - Unit tests
# @pytest.mark.integration - Integration tests  
# @pytest.mark.performance - Performance tests
# @pytest.mark.security - Security tests
# @pytest.mark.apg_integration - APG platform integration tests

pytest_plugins = []  # No additional plugins needed for APG testing

# APG Test Configuration
def pytest_configure(config):
	"""Configure pytest for APG testing."""
	config.addinivalue_line("markers", "unit: Unit tests")
	config.addinivalue_line("markers", "integration: Integration tests")
	config.addinivalue_line("markers", "performance: Performance tests")
	config.addinivalue_line("markers", "security: Security tests")
	config.addinivalue_line("markers", "apg_integration: APG platform integration tests")

def pytest_collection_modifyitems(config, items):
	"""Modify test collection for APG patterns."""
	# Add default markers based on test file names
	for item in items:
		if "integration" in item.nodeid:
			item.add_marker(pytest.mark.integration)
		elif "performance" in item.nodeid:
			item.add_marker(pytest.mark.performance)
		elif "security" in item.nodeid:
			item.add_marker(pytest.mark.security)
		elif "apg" in item.nodeid:
			item.add_marker(pytest.mark.apg_integration)
		else:
			item.add_marker(pytest.mark.unit)

# Export test utilities
__all__ = [
	'APG_TEST_CONFIG',
	'assert_valid_uuid',
	'assert_recent_timestamp', 
	'wait_for_condition'
]