"""
APG API Service Mesh - Test Configuration

Pytest configuration and shared fixtures for comprehensive testing of the service mesh.
Provides database setup, Redis configuration, and common test utilities.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import pytest
import os
from typing import AsyncGenerator, Generator
from unittest.mock import Mock, AsyncMock

import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient

from .models import Base
from .service import ASMService
from .api import api_app
from .apg_integration import APGServiceMeshIntegration

# =============================================================================
# Test Configuration
# =============================================================================

# Test database URL (in-memory SQLite for fast tests)
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

# Test Redis configuration
TEST_REDIS_URL = os.getenv("TEST_REDIS_URL", "redis://localhost:6379/15")

# Test settings
TEST_SETTINGS = {
	"database_url": TEST_DATABASE_URL,
	"redis_url": TEST_REDIS_URL,
	"environment": "test",
	"debug": True,
	"testing": True
}

# =============================================================================
# Database Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
	"""Create an instance of the default event loop for the test session."""
	loop = asyncio.get_event_loop_policy().new_event_loop()
	yield loop
	loop.close()

@pytest.fixture(scope="session")
async def test_engine():
	"""Create test database engine with proper configuration."""
	engine = create_async_engine(
		TEST_DATABASE_URL,
		echo=False,  # Set to True for SQL debugging
		poolclass=StaticPool,
		connect_args={
			"check_same_thread": False,
		},
	)
	
	# Create all tables
	async with engine.begin() as conn:
		await conn.run_sync(Base.metadata.create_all)
	
	yield engine
	
	# Cleanup
	await engine.dispose()

@pytest.fixture
async def test_db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
	"""Create a test database session with transaction rollback."""
	async_session_maker = async_sessionmaker(
		test_engine,
		class_=AsyncSession,
		expire_on_commit=False,
		autoflush=False,
		autocommit=False,
	)
	
	async with async_session_maker() as session:
		# Begin a transaction
		trans = await session.begin()
		
		yield session
		
		# Rollback transaction to clean up
		await trans.rollback()

# =============================================================================
# Redis Fixtures
# =============================================================================

@pytest.fixture(scope="session")
async def test_redis_client():
	"""Create test Redis client."""
	client = redis.from_url(TEST_REDIS_URL)
	
	# Clear test database
	await client.flushdb()
	
	yield client
	
	# Cleanup
	await client.flushdb()
	await client.close()

@pytest.fixture
async def clean_redis(test_redis_client):
	"""Provide clean Redis instance for each test."""
	await test_redis_client.flushdb()
	yield test_redis_client
	await test_redis_client.flushdb()

# =============================================================================
# Service Fixtures
# =============================================================================

@pytest.fixture
async def asm_service(test_db_session, clean_redis) -> ASMService:
	"""Create ASM service instance for testing."""
	service = ASMService(test_db_session, clean_redis)
	yield service

@pytest.fixture
async def apg_integration(asm_service, clean_redis) -> APGServiceMeshIntegration:
	"""Create APG integration service for testing."""
	integration = APGServiceMeshIntegration(asm_service, clean_redis)
	yield integration

# =============================================================================
# API Fixtures
# =============================================================================

@pytest.fixture
def test_client() -> TestClient:
	"""Create FastAPI test client."""
	return TestClient(api_app)

@pytest.fixture
def mock_auth_headers() -> dict:
	"""Mock authentication headers for API tests."""
	return {
		"Authorization": "Bearer test_token",
		"X-Tenant-ID": "test_tenant",
		"Content-Type": "application/json"
	}

# =============================================================================
# Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_httpx_client():
	"""Mock httpx.AsyncClient for HTTP requests."""
	mock_client = AsyncMock()
	
	# Mock successful health check response
	mock_response = Mock()
	mock_response.status_code = 200
	mock_response.json.return_value = {"status": "healthy"}
	mock_client.get.return_value = mock_response
	
	return mock_client

@pytest.fixture
def mock_websocket():
	"""Mock WebSocket for testing real-time features."""
	mock_ws = Mock()
	mock_ws.accept = AsyncMock()
	mock_ws.send_text = AsyncMock()
	mock_ws.receive_text = AsyncMock()
	mock_ws.close = AsyncMock()
	
	return mock_ws

# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def sample_service_config():
	"""Sample service configuration for testing."""
	return {
		"service_name": "test-service",
		"service_version": "v1.0.0",
		"namespace": "test",
		"description": "Test service for validation",
		"environment": "test",
		"tags": ["test", "validation"],
		"metadata": {"test": True, "category": "testing"}
	}

@pytest.fixture
def sample_endpoints():
	"""Sample endpoints configuration for testing."""
	return [
		{
			"host": "localhost",
			"port": 8080,
			"protocol": "http",
			"path": "/api/v1",
			"weight": 100,
			"enabled": True,
			"health_check_path": "/health",
			"health_check_interval": 30,
			"health_check_timeout": 5,
			"tls_enabled": False
		},
		{
			"host": "localhost",
			"port": 8081,
			"protocol": "http",
			"path": "/api/v1",
			"weight": 100,
			"enabled": True,
			"health_check_path": "/health",
			"health_check_interval": 30,
			"health_check_timeout": 5,
			"tls_enabled": False
		}
	]

@pytest.fixture
def sample_route_config():
	"""Sample route configuration for testing."""
	return {
		"route_name": "test-route",
		"match_type": "prefix",
		"match_value": "/api/test",
		"destination_services": [
			{"service_id": "test-service", "weight": 100}
		],
		"timeout_ms": 30000,
		"retry_attempts": 3,
		"priority": 1000,
		"enabled": True
	}

@pytest.fixture
def sample_load_balancer_config():
	"""Sample load balancer configuration for testing."""
	return {
		"load_balancer_name": "test-lb",
		"algorithm": "round_robin",
		"session_affinity": False,
		"health_check_enabled": True,
		"health_check_interval": 30,
		"circuit_breaker_enabled": True,
		"failure_threshold": 5,
		"max_connections": 100,
		"enabled": True
	}

@pytest.fixture
def sample_policy_config():
	"""Sample policy configuration for testing."""
	return {
		"policy_name": "test-policy",
		"policy_type": "rate_limit",
		"configuration": {
			"requests_per_second": 100,
			"window_size_seconds": 60,
			"burst_size": 10
		},
		"enabled": True,
		"priority": 1000,
		"description": "Test rate limiting policy"
	}

# =============================================================================
# Multi-Service Test Fixtures
# =============================================================================

@pytest.fixture
async def registered_services(asm_service, sample_endpoints):
	"""Create multiple registered services for testing."""
	services = []
	
	service_configs = [
		{
			"service_name": "auth-service",
			"service_version": "v1.0.0",
			"namespace": "auth",
			"description": "Authentication service",
			"environment": "test",
			"tags": ["auth", "security"],
			"metadata": {"category": "security"}
		},
		{
			"service_name": "payment-service",
			"service_version": "v2.1.0",
			"namespace": "financial",
			"description": "Payment processing service",
			"environment": "test",
			"tags": ["payment", "financial"],
			"metadata": {"category": "financial"}
		},
		{
			"service_name": "notification-service",
			"service_version": "v1.5.0",
			"namespace": "communication",
			"description": "Notification service",
			"environment": "test",
			"tags": ["notification", "messaging"],
			"metadata": {"category": "communication"}
		}
	]
	
	for config in service_configs:
		service_id = await asm_service.register_service(
			service_config=config,
			endpoints=sample_endpoints,
			tenant_id="test_tenant",
			created_by="test_user"
		)
		services.append({
			"service_id": service_id,
			"config": config
		})
	
	return services

@pytest.fixture
async def service_with_routes(asm_service, registered_services):
	"""Create services with associated routes for testing."""
	routes = []
	
	route_configs = [
		{
			"route_name": "auth-route",
			"match_type": "prefix",
			"match_value": "/api/auth",
			"destination_services": [{"service_id": registered_services[0]["service_id"], "weight": 100}],
			"priority": 1000
		},
		{
			"route_name": "payment-route",
			"match_type": "prefix",
			"match_value": "/api/payments",
			"destination_services": [{"service_id": registered_services[1]["service_id"], "weight": 100}],
			"priority": 1000
		}
	]
	
	for config in route_configs:
		route_id = await asm_service.traffic_manager.create_route(
			route_config=config,
			tenant_id="test_tenant",
			created_by="test_user"
		)
		routes.append({
			"route_id": route_id,
			"config": config
		})
	
	return {
		"services": registered_services,
		"routes": routes
	}

# =============================================================================
# Performance Test Fixtures
# =============================================================================

@pytest.fixture
def performance_test_config():
	"""Configuration for performance tests."""
	return {
		"max_response_time_ms": 1000,
		"max_95th_percentile_ms": 2000,
		"min_throughput_rps": 10.0,
		"max_error_rate": 0.05,
		"concurrent_users": 50,
		"test_duration_seconds": 60
	}

@pytest.fixture
async def large_service_dataset(asm_service, sample_endpoints):
	"""Create large dataset of services for performance testing."""
	services = []
	
	for i in range(100):
		config = {
			"service_name": f"perf-service-{i:03d}",
			"service_version": "v1.0.0",
			"namespace": f"namespace-{i % 10}",
			"description": f"Performance test service {i}",
			"environment": "test",
			"tags": ["performance", "test", f"batch-{i // 10}"],
			"metadata": {"batch": i // 10, "index": i}
		}
		
		service_id = await asm_service.register_service(
			service_config=config,
			endpoints=sample_endpoints,
			tenant_id="test_tenant",
			created_by="perf_test_user"
		)
		
		services.append({
			"service_id": service_id,
			"config": config
		})
	
	return services

# =============================================================================
# Cleanup Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
async def cleanup_after_test(test_db_session, clean_redis):
	"""Automatically cleanup after each test."""
	yield
	
	# Additional cleanup if needed
	# This runs after each test automatically due to autouse=True
	pass

# =============================================================================
# Test Markers and Configuration
# =============================================================================

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
		"markers", "api: mark test as an API test"
	)
	config.addinivalue_line(
		"markers", "websocket: mark test as a WebSocket test"
	)
	config.addinivalue_line(
		"markers", "slow: mark test as slow running"
	)

def pytest_collection_modifyitems(config, items):
	"""Modify test collection to add markers automatically."""
	for item in items:
		# Add markers based on test file location or name patterns
		if "test_api" in str(item.fspath):
			item.add_marker(pytest.mark.api)
		if "test_performance" in item.name or "performance" in item.name:
			item.add_marker(pytest.mark.performance)
		if "test_integration" in item.name or "integration" in item.name:
			item.add_marker(pytest.mark.integration)
		if "websocket" in item.name:
			item.add_marker(pytest.mark.websocket)

# =============================================================================
# Test Utilities
# =============================================================================

@pytest.fixture
def test_utilities():
	"""Provide test utility functions."""
	
	class TestUtils:
		@staticmethod
		def assert_service_structure(service_data):
			"""Assert that service data has the correct structure."""
			required_fields = ["service_id", "service_name", "service_version", "status", "health_status"]
			for field in required_fields:
				assert field in service_data, f"Missing required field: {field}"
		
		@staticmethod
		def assert_endpoint_structure(endpoint_data):
			"""Assert that endpoint data has the correct structure."""
			required_fields = ["endpoint_id", "host", "port", "protocol", "enabled"]
			for field in required_fields:
				assert field in endpoint_data, f"Missing required field: {field}"
		
		@staticmethod
		def assert_route_structure(route_data):
			"""Assert that route data has the correct structure."""
			required_fields = ["route_id", "route_name", "match_type", "match_value", "destination_services"]
			for field in required_fields:
				assert field in route_data, f"Missing required field: {field}"
		
		@staticmethod
		def generate_test_id(prefix="test"):
			"""Generate a unique test ID."""
			from uuid_extensions import uuid7str
			return f"{prefix}_{uuid7str()[:8]}"
		
		@staticmethod
		def create_mock_request_data():
			"""Create mock request data for testing."""
			from datetime import datetime, timezone
			return {
				"start_time": datetime.now(timezone.utc),
				"method": "GET",
				"path": "/api/test",
				"headers": {"User-Agent": "test-client/1.0"},
				"client_ip": "192.168.1.100"
			}
		
		@staticmethod
		def create_mock_response_data(status_code=200):
			"""Create mock response data for testing."""
			from datetime import datetime, timezone
			return {
				"end_time": datetime.now(timezone.utc),
				"status_code": status_code,
				"headers": {"Content-Type": "application/json"},
				"response_size": 1024
			}
	
	return TestUtils

# =============================================================================
# Environment Configuration
# =============================================================================

@pytest.fixture(scope="session", autouse=True)
def configure_test_environment():
	"""Configure test environment variables."""
	os.environ.update({
		"ENVIRONMENT": "test",
		"DATABASE_URL": TEST_DATABASE_URL,
		"REDIS_URL": TEST_REDIS_URL,
		"LOG_LEVEL": "DEBUG",
		"TESTING": "true"
	})
	
	yield
	
	# Cleanup environment variables if needed
	test_env_vars = ["ENVIRONMENT", "DATABASE_URL", "REDIS_URL", "LOG_LEVEL", "TESTING"]
	for var in test_env_vars:
		if var in os.environ:
			del os.environ[var]