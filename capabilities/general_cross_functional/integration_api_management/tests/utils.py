"""
APG Integration API Management - Test Utilities

Utility functions and helpers for testing the Integration API Management capability.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable, Awaitable
from unittest.mock import AsyncMock, MagicMock

from ..models import AMAPI, AMConsumer, AMAPIKey, APIStatus, ConsumerStatus


# =============================================================================
# Test Data Generators
# =============================================================================

def generate_test_api_data(index: int = 0, tenant_id: str = "test_tenant") -> Dict[str, Any]:
	"""Generate test API data."""
	return {
		"api_name": f"test_api_{index:03d}",
		"api_title": f"Test API {index}",
		"api_description": f"Test API {index} for unit testing",
		"version": "1.0.0",
		"protocol_type": "rest",
		"base_path": f"/test{index:03d}",
		"upstream_url": f"http://test{index:03d}.local:8000",
		"is_public": False,
		"timeout_ms": 30000,
		"retry_attempts": 3,
		"auth_type": "api_key",
		"category": "testing",
		"tags": ["test", "api", f"test{index}"]
	}


def generate_test_consumer_data(index: int = 0, tenant_id: str = "test_tenant") -> Dict[str, Any]:
	"""Generate test consumer data."""
	return {
		"consumer_name": f"test_consumer_{index:03d}",
		"organization": f"Test Organization {index}",
		"contact_email": f"test{index:03d}@example.com",
		"contact_name": f"Test User {index}",
		"description": f"Test consumer {index} for unit testing",
		"status": "pending",
		"global_rate_limit": 1000 + index * 100,
		"portal_access": True,
		"webhook_url": f"https://test{index:03d}.webhook.com/events"
	}


def generate_test_workflow_data(index: int = 0) -> Dict[str, Any]:
	"""Generate test workflow data."""
	from ..integration import CrossCapabilityWorkflow, WorkflowStep, EventType
	
	return CrossCapabilityWorkflow(
		workflow_id=f"test_workflow_{index:03d}",
		workflow_name=f"Test Workflow {index}",
		description=f"Test workflow {index} for unit testing",
		trigger_events=[EventType.API_REGISTERED, EventType.CONSUMER_APPROVED],
		steps=[
			WorkflowStep(
				step_id=f"step1_{index}",
				step_name=f"First Step {index}",
				capability_id=f"test_capability_{index}",
				action="test_action",
				parameters={"param1": f"value{index}", "param2": index}
			),
			WorkflowStep(
				step_id=f"step2_{index}",
				step_name=f"Second Step {index}",
				capability_id=f"test_capability_{index}",
				action="test_action2",
				parameters={"param3": f"value{index + 1}", "param4": index * 2}
			)
		],
		timeout_seconds=300,
		max_retries=3,
		retry_delay_seconds=10
	)


def generate_bulk_test_data(count: int) -> Dict[str, List[Dict[str, Any]]]:
	"""Generate bulk test data for performance testing."""
	return {
		"apis": [generate_test_api_data(i) for i in range(count)],
		"consumers": [generate_test_consumer_data(i) for i in range(count)],
		"workflows": [generate_test_workflow_data(i) for i in range(count)]
	}


# =============================================================================
# Database Test Utilities
# =============================================================================

async def create_test_api(session, api_data: Dict[str, Any], tenant_id: str = "test_tenant") -> AMAPI:
	"""Create a test API in the database."""
	api = AMAPI(
		tenant_id=tenant_id,
		created_by="test_user",
		**api_data
	)
	session.add(api)
	session.commit()
	return api


async def create_test_consumer(session, consumer_data: Dict[str, Any], tenant_id: str = "test_tenant") -> AMConsumer:
	"""Create a test consumer in the database."""
	consumer = AMConsumer(
		tenant_id=tenant_id,
		created_by="test_user",
		**consumer_data
	)
	session.add(consumer)
	session.commit()
	return consumer


async def create_test_api_key(session, consumer: AMConsumer, key_name: str = "test_key") -> AMAPIKey:
	"""Create a test API key in the database."""
	api_key = AMAPIKey(
		consumer_id=consumer.consumer_id,
		key_name=key_name,
		key_hash="test_hash_value",
		key_prefix="tk_test",
		scopes=["read", "write"],
		active=True,
		created_by="test_user"
	)
	session.add(api_key)
	session.commit()
	return api_key


async def cleanup_test_data(session, tenant_id: str = "test_tenant"):
	"""Clean up test data from database."""
	# Delete in reverse dependency order
	session.query(AMAPIKey).filter(
		AMAPIKey.consumer.has(AMConsumer.tenant_id == tenant_id)
	).delete(synchronize_session=False)
	
	session.query(AMConsumer).filter_by(tenant_id=tenant_id).delete()
	session.query(AMAPI).filter_by(tenant_id=tenant_id).delete()
	session.commit()


# =============================================================================
# Mock Utilities
# =============================================================================

def create_mock_http_session(responses: List[Dict[str, Any]]) -> AsyncMock:
	"""Create a mock HTTP session with predefined responses."""
	session = AsyncMock()
	
	response_queue = responses.copy()
	
	async def mock_request(*args, **kwargs):
		if response_queue:
			response_data = response_queue.pop(0)
		else:
			response_data = {"status": 200, "json": {"result": "default"}}
		
		response = AsyncMock()
		response.status = response_data.get("status", 200)
		response.headers = response_data.get("headers", {"Content-Type": "application/json"})
		response.json = AsyncMock(return_value=response_data.get("json", {}))
		response.text = AsyncMock(return_value=response_data.get("text", ""))
		response.read = AsyncMock(return_value=response_data.get("body", b""))
		
		return response
	
	session.get = AsyncMock(side_effect=mock_request)
	session.post = AsyncMock(side_effect=mock_request)
	session.put = AsyncMock(side_effect=mock_request)
	session.delete = AsyncMock(side_effect=mock_request)
	session.__aenter__ = AsyncMock(return_value=session)
	session.__aexit__ = AsyncMock(return_value=None)
	
	return session


def create_mock_gateway_request(
	path: str = "/test",
	method: str = "GET",
	headers: Optional[Dict[str, str]] = None,
	tenant_id: str = "test_tenant"
) -> Any:
	"""Create a mock gateway request."""
	from ..gateway import GatewayRequest
	
	return GatewayRequest(
		request_id=f"mock_req_{int(time.time())}",
		method=method,
		path=path,
		headers=headers or {},
		query_params={},
		body=b"",
		client_ip="127.0.0.1",
		user_agent="Test Agent",
		timestamp=datetime.now(timezone.utc),
		tenant_id=tenant_id
	)


def create_mock_upstream_server(url: str = "http://localhost:8000", healthy: bool = True) -> Any:
	"""Create a mock upstream server."""
	from ..gateway import UpstreamServer
	
	return UpstreamServer(
		url=url,
		weight=1,
		max_connections=100,
		health_check_path="/health",
		is_healthy=healthy,
		current_connections=0,
		last_health_check=datetime.now(timezone.utc) if healthy else None
	)


# =============================================================================
# Assertion Utilities
# =============================================================================

async def assert_eventually(
	condition: Callable[[], Awaitable[bool]],
	timeout: float = 5.0,
	interval: float = 0.1,
	error_message: str = "Condition was not met within timeout"
) -> bool:
	"""Assert that a condition becomes true within a timeout period."""
	start_time = time.time()
	
	while time.time() - start_time < timeout:
		if await condition():
			return True
		await asyncio.sleep(interval)
	
	# Final check
	if await condition():
		return True
	
	raise AssertionError(error_message)


def assert_api_data_matches(api: AMAPI, expected_data: Dict[str, Any]):
	"""Assert that an API object matches expected data."""
	assert api.api_name == expected_data["api_name"]
	assert api.api_title == expected_data["api_title"]
	assert api.base_path == expected_data["base_path"]
	assert api.upstream_url == expected_data["upstream_url"]
	assert api.version == expected_data["version"]
	assert api.protocol_type == expected_data["protocol_type"]


def assert_consumer_data_matches(consumer: AMConsumer, expected_data: Dict[str, Any]):
	"""Assert that a consumer object matches expected data."""
	assert consumer.consumer_name == expected_data["consumer_name"]
	assert consumer.organization == expected_data["organization"]
	assert consumer.contact_email == expected_data["contact_email"]
	assert consumer.contact_name == expected_data["contact_name"]


def assert_metrics_recorded(metrics: List[Any], expected_count: int, metric_name: str):
	"""Assert that metrics were recorded correctly."""
	matching_metrics = [m for m in metrics if m.name == metric_name]
	assert len(matching_metrics) == expected_count, f"Expected {expected_count} metrics named '{metric_name}', got {len(matching_metrics)}"


# =============================================================================
# Performance Measurement Utilities
# =============================================================================

class PerformanceTimer:
	"""Context manager for measuring performance."""
	
	def __init__(self, name: str = "operation"):
		self.name = name
		self.start_time = None
		self.end_time = None
		self.duration = None
	
	def __enter__(self):
		self.start_time = time.time()
		return self
	
	def __exit__(self, exc_type, exc_val, exc_tb):
		self.end_time = time.time()
		self.duration = self.end_time - self.start_time
		print(f"{self.name} completed in {self.duration:.3f} seconds")


async def measure_async_performance(
	operation: Callable[[], Awaitable[Any]],
	name: str = "async operation",
	iterations: int = 1
) -> Dict[str, float]:
	"""Measure performance of an async operation."""
	durations = []
	
	for _ in range(iterations):
		start_time = time.time()
		await operation()
		end_time = time.time()
		durations.append(end_time - start_time)
	
	total_duration = sum(durations)
	avg_duration = total_duration / iterations
	min_duration = min(durations)
	max_duration = max(durations)
	throughput = iterations / total_duration
	
	print(f"{name} performance ({iterations} iterations):")
	print(f"  Total: {total_duration:.3f}s")
	print(f"  Average: {avg_duration:.3f}s")
	print(f"  Min: {min_duration:.3f}s")
	print(f"  Max: {max_duration:.3f}s")
	print(f"  Throughput: {throughput:.1f} ops/sec")
	
	return {
		"total_duration": total_duration,
		"avg_duration": avg_duration,
		"min_duration": min_duration,
		"max_duration": max_duration,
		"throughput": throughput
	}


async def measure_concurrent_performance(
	operation: Callable[[int], Awaitable[Any]],
	concurrency: int,
	name: str = "concurrent operation"
) -> Dict[str, float]:
	"""Measure performance of concurrent operations."""
	start_time = time.time()
	
	tasks = [operation(i) for i in range(concurrency)]
	results = await asyncio.gather(*tasks, return_exceptions=True)
	
	end_time = time.time()
	duration = end_time - start_time
	
	successful_results = [r for r in results if not isinstance(r, Exception)]
	failed_results = [r for r in results if isinstance(r, Exception)]
	
	throughput = len(successful_results) / duration
	success_rate = len(successful_results) / len(results)
	
	print(f"{name} concurrent performance ({concurrency} operations):")
	print(f"  Duration: {duration:.3f}s")
	print(f"  Successful: {len(successful_results)}")
	print(f"  Failed: {len(failed_results)}")
	print(f"  Success rate: {success_rate:.1%}")
	print(f"  Throughput: {throughput:.1f} ops/sec")
	
	return {
		"duration": duration,
		"successful_count": len(successful_results),
		"failed_count": len(failed_results),
		"success_rate": success_rate,
		"throughput": throughput
	}


# =============================================================================
# Redis Test Utilities
# =============================================================================

async def clear_redis_test_data(redis_client, pattern: str = "test:*"):
	"""Clear test data from Redis."""
	cursor = 0
	while True:
		cursor, keys = await redis_client.scan(cursor, match=pattern, count=100)
		if keys:
			await redis_client.delete(*keys)
		if cursor == 0:
			break


async def verify_redis_key_exists(redis_client, key: str, timeout: float = 1.0) -> bool:
	"""Verify that a Redis key exists within timeout."""
	start_time = time.time()
	
	while time.time() - start_time < timeout:
		if await redis_client.exists(key):
			return True
		await asyncio.sleep(0.1)
	
	return False


# =============================================================================
# Validation Utilities
# =============================================================================

def validate_api_config(config: Dict[str, Any]) -> List[str]:
	"""Validate API configuration and return list of errors."""
	errors = []
	
	required_fields = ["api_name", "api_title", "base_path", "upstream_url"]
	for field in required_fields:
		if field not in config or not config[field]:
			errors.append(f"Missing required field: {field}")
	
	if "base_path" in config and not config["base_path"].startswith("/"):
		errors.append("Base path must start with '/'")
	
	if "timeout_ms" in config and config["timeout_ms"] < 1000:
		errors.append("Timeout must be at least 1000ms")
	
	if "retry_attempts" in config and config["retry_attempts"] < 0:
		errors.append("Retry attempts cannot be negative")
	
	return errors


def validate_consumer_config(config: Dict[str, Any]) -> List[str]:
	"""Validate consumer configuration and return list of errors."""
	errors = []
	
	required_fields = ["consumer_name", "contact_email"]
	for field in required_fields:
		if field not in config or not config[field]:
			errors.append(f"Missing required field: {field}")
	
	if "contact_email" in config:
		import re
		email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
		if not re.match(email_pattern, config["contact_email"]):
			errors.append("Invalid email address format")
	
	if "global_rate_limit" in config and config["global_rate_limit"] <= 0:
		errors.append("Global rate limit must be positive")
	
	return errors


# =============================================================================
# Test Environment Utilities
# =============================================================================

def setup_test_environment() -> Dict[str, Any]:
	"""Set up test environment variables and configuration."""
	return {
		"database_url": "sqlite:///:memory:",
		"redis_url": "redis://localhost:6379/1",
		"debug": True,
		"testing": True,
		"log_level": "DEBUG"
	}


def teardown_test_environment():
	"""Clean up test environment."""
	# Clean up any global state or resources
	pass


# =============================================================================
# Health Check Utilities
# =============================================================================

async def create_mock_health_check(
	status: str = "healthy",
	response_time: int = 50,
	details: Optional[Dict[str, Any]] = None
) -> Callable[[], Awaitable[Dict[str, Any]]]:
	"""Create a mock health check function."""
	async def health_check():
		await asyncio.sleep(response_time / 1000.0)  # Simulate response time
		
		result = {
			"status": status,
			"timestamp": datetime.now(timezone.utc),
			"response_time_ms": response_time
		}
		
		if details:
			result["details"] = details
		
		if status == "unhealthy":
			result["error"] = "Simulated service failure"
		elif status == "degraded":
			result["warning"] = "Simulated service degradation"
		
		return result
	
	return health_check


# =============================================================================
# Export Utilities
# =============================================================================

__all__ = [
	# Data generators
	'generate_test_api_data',
	'generate_test_consumer_data', 
	'generate_test_workflow_data',
	'generate_bulk_test_data',
	
	# Database utilities
	'create_test_api',
	'create_test_consumer',
	'create_test_api_key',
	'cleanup_test_data',
	
	# Mock utilities
	'create_mock_http_session',
	'create_mock_gateway_request',
	'create_mock_upstream_server',
	
	# Assertion utilities
	'assert_eventually',
	'assert_api_data_matches',
	'assert_consumer_data_matches',
	'assert_metrics_recorded',
	
	# Performance utilities
	'PerformanceTimer',
	'measure_async_performance',
	'measure_concurrent_performance',
	
	# Redis utilities
	'clear_redis_test_data',
	'verify_redis_key_exists',
	
	# Validation utilities
	'validate_api_config',
	'validate_consumer_config',
	
	# Environment utilities
	'setup_test_environment',
	'teardown_test_environment',
	
	# Health check utilities
	'create_mock_health_check'
]