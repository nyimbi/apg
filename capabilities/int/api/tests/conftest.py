"""
APG Integration API Management - Test Configuration

Pytest configuration and fixtures for comprehensive testing of the
Integration API Management capability.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import pytest
import pytest_asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from unittest.mock import AsyncMock, MagicMock

import aioredis
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from ..models import Base, AMAPI, AMConsumer, AMAPIKey, APIStatus, ConsumerStatus
from ..config import APIManagementSettings, Environment
from ..service import (
	APILifecycleService, ConsumerManagementService,
	PolicyManagementService, AnalyticsService
)
from ..discovery import ServiceDiscovery, APGCapabilityInfo, CapabilityType
from ..integration import APGIntegrationManager
from ..monitoring import MetricsCollector, HealthMonitor, AlertManager
from ..gateway import APIGateway, GatewayRouter
from ..factory import IntegrationAPIManagementCapability

# =============================================================================
# Test Configuration
# =============================================================================

@pytest.fixture(scope="session")
def event_loop():
	"""Create event loop for async tests."""
	loop = asyncio.get_event_loop_policy().new_event_loop()
	yield loop
	loop.close()

@pytest.fixture
def test_config():
	"""Test configuration settings."""
	return APIManagementSettings(
		environment=Environment.TESTING,
		debug=True,
		database=APIManagementSettings.DatabaseConfig(
			engine="sqlite",
			database=":memory:",
			echo=True
		),
		redis=APIManagementSettings.RedisConfig(
			host="localhost",
			port=6379,
			database=1  # Use test database
		),
		gateway=APIManagementSettings.GatewayConfig(
			host="127.0.0.1",
			port=8081,  # Test port
			workers=1
		),
		security=APIManagementSettings.SecurityConfig(
			jwt_secret_key="test-secret-key-32-characters-long"
		),
		monitoring=APIManagementSettings.MonitoringConfig(
			log_level="DEBUG",
			alerting_enabled=False
		)
	)

# =============================================================================
# Database Fixtures
# =============================================================================

@pytest.fixture
def test_engine(test_config):
	"""Create test database engine."""
	engine = create_engine(
		"sqlite:///:memory:",
		poolclass=StaticPool,
		connect_args={
			"check_same_thread": False,
		},
		echo=test_config.database.echo
	)
	
	# Create all tables
	Base.metadata.create_all(engine)
	
	yield engine
	
	# Cleanup
	Base.metadata.drop_all(engine)
	engine.dispose()

@pytest.fixture
def test_session(test_engine):
	"""Create test database session."""
	Session = sessionmaker(bind=test_engine)
	session = Session()
	
	yield session
	
	session.close()

@pytest_asyncio.fixture
async def redis_client(test_config):
	"""Create test Redis client."""
	client = aioredis.from_url(f"redis://{test_config.redis.host}:{test_config.redis.port}/{test_config.redis.database}")
	
	# Clear test database
	await client.flushdb()
	
	yield client
	
	await client.flushdb()
	await client.close()

# =============================================================================
# Service Fixtures
# =============================================================================

@pytest.fixture
def api_service(test_session):
	"""Create API lifecycle service."""
	service = APILifecycleService()
	service._session = test_session  # Inject test session
	return service

@pytest.fixture
def consumer_service(test_session):
	"""Create consumer management service."""
	service = ConsumerManagementService()
	service._session = test_session  # Inject test session
	return service

@pytest.fixture
def policy_service(test_session):
	"""Create policy management service."""
	service = PolicyManagementService()
	service._session = test_session  # Inject test session
	return service

@pytest.fixture
def analytics_service(test_session):
	"""Create analytics service."""
	service = AnalyticsService()
	service._session = test_session  # Inject test session
	return service

@pytest_asyncio.fixture
async def metrics_collector(redis_client):
	"""Create metrics collector."""
	return MetricsCollector(redis_client)

@pytest_asyncio.fixture
async def health_monitor(redis_client, analytics_service, metrics_collector):
	"""Create health monitor."""
	return HealthMonitor(redis_client, analytics_service, metrics_collector)

@pytest_asyncio.fixture
async def alert_manager(redis_client):
	"""Create alert manager."""
	return AlertManager(redis_client)

@pytest_asyncio.fixture
async def service_discovery(redis_client, api_service):
	"""Create service discovery."""
	discovery = ServiceDiscovery(redis_client, api_service)
	await discovery.initialize()
	
	yield discovery
	
	await discovery.shutdown()

@pytest_asyncio.fixture
async def integration_manager(redis_client, service_discovery, api_service, 
							 consumer_service, analytics_service, 
							 metrics_collector, health_monitor):
	"""Create APG integration manager."""
	manager = APGIntegrationManager(
		redis_client, service_discovery, api_service,
		consumer_service, analytics_service, metrics_collector, health_monitor
	)
	await manager.initialize()
	
	yield manager
	
	await manager.shutdown()

# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def sample_api_data():
	"""Sample API data for testing."""
	return {
		"api_name": "test_api",
		"api_title": "Test API",
		"api_description": "A test API for unit testing",
		"version": "1.0.0",
		"protocol_type": "rest",
		"base_path": "/test",
		"upstream_url": "http://localhost:8000",
		"is_public": False,
		"timeout_ms": 30000,
		"retry_attempts": 3,
		"auth_type": "api_key",
		"category": "testing",
		"tags": ["test", "api"]
	}

@pytest.fixture
def sample_consumer_data():
	"""Sample consumer data for testing."""
	return {
		"consumer_name": "test_consumer",
		"organization": "Test Organization",
		"contact_email": "test@example.com",
		"contact_name": "Test User",
		"status": "pending",
		"global_rate_limit": 1000,
		"portal_access": True
	}

@pytest.fixture
def sample_capability_info():
	"""Sample APG capability info for testing."""
	return APGCapabilityInfo(
		capability_id="test_capability",
		capability_name="Test Capability",
		capability_type=CapabilityType.CORE_BUSINESS,
		version="1.0.0",
		description="A test capability",
		base_url="http://localhost:8000",
		api_endpoints={
			"api": "/api/v1",
			"health": "/health"
		},
		dependencies=["integration_api_management"],
		provides=["test_service"],
		tags=["test", "capability"]
	)

@pytest.fixture
def sample_workflow_data():
	"""Sample workflow data for testing."""
	from ..integration import CrossCapabilityWorkflow, WorkflowStep, EventType
	
	return CrossCapabilityWorkflow(
		workflow_id="test_workflow",
		workflow_name="Test Workflow",
		description="A test workflow",
		trigger_events=[EventType.CAPABILITY_REGISTERED],
		steps=[
			WorkflowStep(
				step_id="step1",
				step_name="First Step",
				capability_id="test_capability",
				action="test_action",
				parameters={"param1": "value1"}
			),
			WorkflowStep(
				step_id="step2",
				step_name="Second Step",
				capability_id="test_capability",
				action="test_action2",
				parameters={"param2": "value2"}
			)
		]
	)

# =============================================================================
# Database Test Data
# =============================================================================

@pytest.fixture
def db_api(test_session, sample_api_data):
	"""Create test API in database."""
	api = AMAPI(
		api_id="test_api_123",
		tenant_id="test_tenant",
		created_by="test_user",
		**sample_api_data
	)
	test_session.add(api)
	test_session.commit()
	return api

@pytest.fixture
def db_consumer(test_session, sample_consumer_data):
	"""Create test consumer in database."""
	consumer = AMConsumer(
		consumer_id="test_consumer_123",
		tenant_id="test_tenant",
		created_by="test_user",
		**sample_consumer_data
	)
	test_session.add(consumer)
	test_session.commit()
	return consumer

@pytest.fixture
def db_api_key(test_session, db_consumer):
	"""Create test API key in database."""
	api_key = AMAPIKey(
		key_id="test_key_123",
		consumer_id=db_consumer.consumer_id,
		key_name="test_key",
		key_hash="hashed_key_value",
		key_prefix="tk_12345",
		scopes=["read", "write"],
		active=True,
		created_by="test_user"
	)
	test_session.add(api_key)
	test_session.commit()
	return api_key

# =============================================================================
# Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_http_session():
	"""Mock aiohttp ClientSession."""
	session = AsyncMock()
	
	# Mock successful response
	response = AsyncMock()
	response.status = 200
	response.json = AsyncMock(return_value={"status": "healthy"})
	response.text = AsyncMock(return_value="OK")
	response.headers = {"content-type": "application/json"}
	
	session.get = AsyncMock(return_value=response)
	session.post = AsyncMock(return_value=response)
	session.__aenter__ = AsyncMock(return_value=session)
	session.__aexit__ = AsyncMock(return_value=None)
	
	return session

@pytest.fixture
def mock_gateway_request():
	"""Mock gateway request."""
	from ..gateway import GatewayRequest
	
	return GatewayRequest(
		request_id="test_req_123",
		method="GET",
		path="/test/endpoint",
		headers={"Authorization": "Bearer test_token"},
		query_params={"param1": "value1"},
		body=b"",
		client_ip="127.0.0.1",
		user_agent="Test Agent",
		timestamp=datetime.now(timezone.utc),
		tenant_id="test_tenant"
	)

@pytest.fixture
def mock_upstream_server():
	"""Mock upstream server."""
	from ..gateway import UpstreamServer
	
	return UpstreamServer(
		url="http://localhost:8000",
		weight=1,
		max_connections=100,
		health_check_path="/health",
		is_healthy=True
	)

# =============================================================================
# Integration Test Fixtures
# =============================================================================

@pytest_asyncio.fixture
async def test_capability(test_config):
	"""Create test capability instance."""
	capability = IntegrationAPIManagementCapability(test_config)
	
	# Initialize without APG registration for testing
	await capability.initialize(register_with_apg=False)
	
	yield capability
	
	await capability.cleanup()

@pytest.fixture
def test_server_url(test_config):
	"""Test server URL."""
	return f"http://{test_config.gateway.host}:{test_config.gateway.port}"

# =============================================================================
# Performance Test Fixtures
# =============================================================================

@pytest.fixture
def performance_test_data():
	"""Generate performance test data."""
	apis = []
	consumers = []
	
	# Generate multiple APIs
	for i in range(100):
		apis.append({
			"api_name": f"perf_api_{i:03d}",
			"api_title": f"Performance Test API {i}",
			"version": "1.0.0",
			"protocol_type": "rest",
			"base_path": f"/perf{i:03d}",
			"upstream_url": f"http://upstream{i:03d}.local",
			"tenant_id": "perf_tenant",
			"created_by": "perf_test"
		})
	
	# Generate multiple consumers
	for i in range(50):
		consumers.append({
			"consumer_name": f"perf_consumer_{i:03d}",
			"contact_email": f"perf{i:03d}@test.com",
			"tenant_id": "perf_tenant",
			"created_by": "perf_test"
		})
	
	return {"apis": apis, "consumers": consumers}

# =============================================================================
# Cleanup Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def cleanup_after_test():
	"""Cleanup after each test."""
	yield
	
	# Any additional cleanup can be added here
	pass

# =============================================================================
# Test Markers
# =============================================================================

# Register custom pytest markers
pytest_plugins = []

def pytest_configure(config):
	"""Configure pytest with custom markers."""
	config.addinivalue_line(
		"markers", "unit: mark test as a unit test"
	)
	config.addinivalue_line(
		"markers", "integration: mark test as an integration test"
	)
	config.addinivalue_line(
		"markers", "performance: mark test as a performance test"
	)
	config.addinivalue_line(
		"markers", "e2e: mark test as an end-to-end test"
	)
	config.addinivalue_line(
		"markers", "slow: mark test as slow running"
	)

# =============================================================================
# Test Utilities
# =============================================================================

@pytest.fixture
def assert_eventually():
	"""Utility for testing eventual consistency."""
	async def _assert_eventually(condition_func, timeout=5.0, interval=0.1):
		"""Assert that condition becomes true within timeout."""
		import time
		start_time = time.time()
		
		while time.time() - start_time < timeout:
			if await condition_func():
				return True
			await asyncio.sleep(interval)
		
		# Final check
		return await condition_func()
	
	return _assert_eventually

@pytest.fixture
def test_metrics():
	"""Test metrics collection utilities."""
	def _create_test_metric(name, value, labels=None):
		from ..monitoring import Metric, MetricType
		
		return Metric(
			name=name,
			metric_type=MetricType.GAUGE,
			value=value,
			timestamp=datetime.now(timezone.utc),
			labels=labels or {},
			description=f"Test metric: {name}"
		)
	
	return _create_test_metric

# =============================================================================
# Export Test Fixtures
# =============================================================================

__all__ = [
	# Configuration
	'test_config',
	'test_server_url',
	
	# Database
	'test_engine',
	'test_session',
	'redis_client',
	
	# Services
	'api_service',
	'consumer_service',
	'policy_service',
	'analytics_service',
	'metrics_collector',
	'health_monitor',
	'alert_manager',
	'service_discovery',
	'integration_manager',
	
	# Test Data
	'sample_api_data',
	'sample_consumer_data',
	'sample_capability_info',
	'sample_workflow_data',
	'performance_test_data',
	
	# Database Objects
	'db_api',
	'db_consumer',
	'db_api_key',
	
	# Mocks
	'mock_http_session',
	'mock_gateway_request',
	'mock_upstream_server',
	
	# Integration
	'test_capability',
	
	# Utilities
	'assert_eventually',
	'test_metrics'
]