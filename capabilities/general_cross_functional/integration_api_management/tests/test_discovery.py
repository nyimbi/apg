"""
APG Integration API Management - Discovery Tests

Unit and integration tests for service discovery including capability registration,
health monitoring, and API auto-discovery functionality.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import pytest
import pytest_asyncio
import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from ..discovery import (
	ServiceDiscovery, APGCapabilityInfo, CapabilityType,
	ServiceRegistry, ServiceEndpoint, HealthStatus,
	ServiceDiscoveryEventBus, EventType
)
from ..models import AMAPI, APIStatus

# =============================================================================
# Service Discovery Tests
# =============================================================================

@pytest.mark.unit
class TestServiceDiscovery:
	"""Test service discovery core functionality."""
	
	@pytest_asyncio.fixture
	async def discovery(self, redis_client, api_service):
		"""Create service discovery instance."""
		discovery = ServiceDiscovery(redis_client, api_service)
		await discovery.initialize()
		
		yield discovery
		
		await discovery.shutdown()
	
	@pytest.mark.asyncio
	async def test_register_capability(self, discovery, sample_capability_info):
		"""Test capability registration."""
		success = await discovery.register_capability(sample_capability_info)
		assert success is True
		
		# Verify capability was registered
		capability = await discovery.get_capability(sample_capability_info.capability_id)
		assert capability is not None
		assert capability.capability_name == sample_capability_info.capability_name
		assert capability.capability_type == sample_capability_info.capability_type
	
	@pytest.mark.asyncio
	async def test_get_capability_not_found(self, discovery):
		"""Test getting non-existent capability."""
		capability = await discovery.get_capability("non_existent_capability")
		assert capability is None
	
	@pytest.mark.asyncio
	async def test_list_capabilities(self, discovery, sample_capability_info):
		"""Test listing all capabilities."""
		# Register test capability
		await discovery.register_capability(sample_capability_info)
		
		# List capabilities
		capabilities = await discovery.list_capabilities()
		assert len(capabilities) >= 1
		
		# Find our test capability
		test_cap = next((c for c in capabilities if c.capability_id == sample_capability_info.capability_id), None)
		assert test_cap is not None
		assert test_cap.capability_name == sample_capability_info.capability_name
	
	@pytest.mark.asyncio
	async def test_list_capabilities_by_type(self, discovery, sample_capability_info):
		"""Test listing capabilities by type."""
		# Register test capability
		await discovery.register_capability(sample_capability_info)
		
		# List by specific type
		capabilities = await discovery.list_capabilities_by_type(CapabilityType.CORE_BUSINESS)
		assert len(capabilities) >= 1
		
		# All should be of the requested type
		assert all(c.capability_type == CapabilityType.CORE_BUSINESS for c in capabilities)
	
	@pytest.mark.asyncio
	async def test_unregister_capability(self, discovery, sample_capability_info):
		"""Test capability unregistration."""
		# Register first
		await discovery.register_capability(sample_capability_info)
		
		# Verify it's there
		capability = await discovery.get_capability(sample_capability_info.capability_id)
		assert capability is not None
		
		# Unregister
		success = await discovery.unregister_capability(sample_capability_info.capability_id)
		assert success is True
		
		# Verify it's gone
		capability = await discovery.get_capability(sample_capability_info.capability_id)
		assert capability is None
	
	@pytest.mark.asyncio
	async def test_update_capability_health(self, discovery, sample_capability_info):
		"""Test updating capability health status."""
		# Register capability
		await discovery.register_capability(sample_capability_info)
		
		# Update health status
		health_data = {
			"status": "healthy",
			"last_check": datetime.now(timezone.utc),
			"response_time_ms": 50,
			"details": {"cpu": 45.2, "memory": 67.8}
		}
		
		success = await discovery.update_capability_health(
			sample_capability_info.capability_id,
			health_data
		)
		assert success is True
		
		# Verify health was updated
		capability = await discovery.get_capability(sample_capability_info.capability_id)
		assert capability.health_status == "healthy"

@pytest.mark.unit
class TestServiceRegistry:
	"""Test service registry functionality."""
	
	@pytest_asyncio.fixture
	async def registry(self, redis_client):
		"""Create service registry."""
		registry = ServiceRegistry(redis_client)
		await registry.initialize()
		
		yield registry
		
		await registry.cleanup()
	
	@pytest.mark.asyncio
	async def test_register_service_endpoint(self, registry):
		"""Test service endpoint registration."""
		endpoint = ServiceEndpoint(
			service_id="test_service",
			endpoint_id="test_endpoint",
			url="http://localhost:8000/api",
			health_check_url="http://localhost:8000/health",
			capabilities=["api_management"],
			tags=["test", "api"]
		)
		
		success = await registry.register_endpoint(endpoint)
		assert success is True
		
		# Verify endpoint was registered
		retrieved = await registry.get_endpoint("test_service", "test_endpoint")
		assert retrieved is not None
		assert retrieved.url == endpoint.url
		assert retrieved.capabilities == endpoint.capabilities
	
	@pytest.mark.asyncio
	async def test_list_service_endpoints(self, registry):
		"""Test listing service endpoints."""
		# Register multiple endpoints
		endpoints = [
			ServiceEndpoint(
				service_id="service1",
				endpoint_id="endpoint1",
				url="http://service1:8000/api",
				capabilities=["capability1"]
			),
			ServiceEndpoint(
				service_id="service1", 
				endpoint_id="endpoint2",
				url="http://service1:8001/api",
				capabilities=["capability2"]
			),
			ServiceEndpoint(
				service_id="service2",
				endpoint_id="endpoint1", 
				url="http://service2:8000/api",
				capabilities=["capability1"]
			)
		]
		
		for endpoint in endpoints:
			await registry.register_endpoint(endpoint)
		
		# List all endpoints for service1
		service1_endpoints = await registry.list_endpoints("service1")
		assert len(service1_endpoints) == 2
		
		# List all endpoints
		all_endpoints = await registry.list_all_endpoints()
		assert len(all_endpoints) >= 3
	
	@pytest.mark.asyncio
	async def test_update_endpoint_health(self, registry):
		"""Test updating endpoint health."""
		endpoint = ServiceEndpoint(
			service_id="health_test_service",
			endpoint_id="health_test_endpoint",
			url="http://localhost:8000/api",
			health_check_url="http://localhost:8000/health"
		)
		
		await registry.register_endpoint(endpoint)
		
		# Update health
		health_data = {
			"status": HealthStatus.HEALTHY,
			"last_check": datetime.now(timezone.utc),
			"response_time_ms": 75
		}
		
		success = await registry.update_endpoint_health(
			"health_test_service",
			"health_test_endpoint", 
			health_data
		)
		assert success is True
		
		# Verify health was updated
		retrieved = await registry.get_endpoint("health_test_service", "health_test_endpoint")
		assert retrieved.health_status == HealthStatus.HEALTHY
		assert retrieved.last_health_check is not None

# =============================================================================
# API Auto-Discovery Tests
# =============================================================================

@pytest.mark.unit
class TestAPIAutoDiscovery:
	"""Test API auto-discovery functionality."""
	
	@pytest.mark.asyncio
	async def test_discover_apis_from_capability(self, discovery, db_api):
		"""Test discovering APIs from a registered capability."""
		# Create capability with API endpoints
		capability_info = APGCapabilityInfo(
			capability_id="api_discovery_test",
			capability_name="API Discovery Test",
			capability_type=CapabilityType.GENERAL_CROSS_FUNCTIONAL,
			version="1.0.0",
			description="Test capability for API discovery",
			base_url="http://localhost:8000",
			api_endpoints={
				"api": "/api/v1",
				"health": "/health",
				"metrics": "/metrics"
			},
			dependencies=[],
			provides=["test_apis"],
			tags=["test", "discovery"]
		)
		
		# Register capability
		await discovery.register_capability(capability_info)
		
		# Discover APIs
		discovered_apis = await discovery.discover_apis_from_capability("api_discovery_test")
		
		assert len(discovered_apis) == 3
		assert any(api["path"] == "/api/v1" for api in discovered_apis)
		assert any(api["path"] == "/health" for api in discovered_apis)
		assert any(api["path"] == "/metrics" for api in discovered_apis)
	
	@pytest.mark.asyncio
	async def test_auto_register_discovered_apis(self, discovery, api_service):
		"""Test auto-registration of discovered APIs."""
		# Mock the API service
		api_service.register_api = AsyncMock(return_value="auto_api_123")
		
		# Create capability
		capability_info = APGCapabilityInfo(
			capability_id="auto_register_test",
			capability_name="Auto Register Test",
			capability_type=CapabilityType.CORE_BUSINESS,
			version="1.0.0",
			description="Test auto-registration",
			base_url="http://localhost:8000",
			api_endpoints={
				"users": "/api/users",
				"orders": "/api/orders"
			},
			auto_register_apis=True,
			dependencies=[],
			provides=["user_management", "order_management"],
			tags=["auto", "register"]
		)
		
		# Register capability with auto-registration
		success = await discovery.register_capability(capability_info)
		assert success is True
		
		# Verify APIs were auto-registered
		assert api_service.register_api.call_count == 2

# =============================================================================
# Service Discovery Event Bus Tests
# =============================================================================

@pytest.mark.unit
class TestServiceDiscoveryEventBus:
	"""Test service discovery event bus."""
	
	@pytest_asyncio.fixture
	async def event_bus(self, redis_client):
		"""Create event bus."""
		bus = ServiceDiscoveryEventBus(redis_client)
		await bus.initialize()
		
		yield bus
		
		await bus.cleanup()
	
	@pytest.mark.asyncio
	async def test_publish_and_subscribe_events(self, event_bus):
		"""Test publishing and subscribing to events."""
		received_events = []
		
		async def event_handler(event):
			received_events.append(event)
		
		# Subscribe to capability registration events
		await event_bus.subscribe(EventType.CAPABILITY_REGISTERED, event_handler)
		
		# Publish event
		event_data = {
			"capability_id": "test_capability",
			"capability_name": "Test Capability",
			"timestamp": datetime.now(timezone.utc)
		}
		
		await event_bus.publish(EventType.CAPABILITY_REGISTERED, event_data)
		
		# Wait for event processing
		await asyncio.sleep(0.1)
		
		# Verify event was received
		assert len(received_events) == 1
		assert received_events[0]["capability_id"] == "test_capability"
	
	@pytest.mark.asyncio
	async def test_multiple_subscribers(self, event_bus):
		"""Test multiple subscribers for same event type."""
		handler1_events = []
		handler2_events = []
		
		async def handler1(event):
			handler1_events.append(event)
		
		async def handler2(event):
			handler2_events.append(event)
		
		# Subscribe both handlers
		await event_bus.subscribe(EventType.CAPABILITY_HEALTH_CHANGED, handler1)
		await event_bus.subscribe(EventType.CAPABILITY_HEALTH_CHANGED, handler2)
		
		# Publish event
		event_data = {"capability_id": "test", "new_status": "unhealthy"}
		await event_bus.publish(EventType.CAPABILITY_HEALTH_CHANGED, event_data)
		
		# Wait for processing
		await asyncio.sleep(0.1)
		
		# Both handlers should receive the event
		assert len(handler1_events) == 1
		assert len(handler2_events) == 1
		assert handler1_events[0]["capability_id"] == "test"
		assert handler2_events[0]["capability_id"] == "test"

# =============================================================================
# Health Monitoring Tests
# =============================================================================

@pytest.mark.unit
class TestHealthMonitoring:
	"""Test health monitoring for discovered services."""
	
	@pytest.mark.asyncio
	async def test_health_check_healthy_service(self, discovery, mock_http_session):
		"""Test health check for healthy service."""
		# Register capability
		capability_info = APGCapabilityInfo(
			capability_id="healthy_service",
			capability_name="Healthy Service",
			capability_type=CapabilityType.CORE_BUSINESS,
			version="1.0.0",
			description="Healthy test service",
			base_url="http://localhost:8000",
			health_check_endpoint="/health",
			dependencies=[],
			provides=["test_service"],
			tags=["test"]
		)
		
		await discovery.register_capability(capability_info)
		
		# Mock healthy response
		mock_http_session.get.return_value.status = 200
		mock_http_session.get.return_value.json.return_value = {"status": "healthy"}
		
		with patch('aiohttp.ClientSession', return_value=mock_http_session):
			# Perform health check
			is_healthy = await discovery.check_capability_health("healthy_service")
			assert is_healthy is True
	
	@pytest.mark.asyncio
	async def test_health_check_unhealthy_service(self, discovery, mock_http_session):
		"""Test health check for unhealthy service."""
		# Register capability
		capability_info = APGCapabilityInfo(
			capability_id="unhealthy_service",
			capability_name="Unhealthy Service",
			capability_type=CapabilityType.CORE_BUSINESS,
			version="1.0.0",
			description="Unhealthy test service",
			base_url="http://localhost:8000",
			health_check_endpoint="/health",
			dependencies=[],
			provides=["test_service"],
			tags=["test"]
		)
		
		await discovery.register_capability(capability_info)
		
		# Mock unhealthy response
		mock_http_session.get.return_value.status = 503
		mock_http_session.get.return_value.json.return_value = {"status": "unhealthy"}
		
		with patch('aiohttp.ClientSession', return_value=mock_http_session):
			# Perform health check
			is_healthy = await discovery.check_capability_health("unhealthy_service")
			assert is_healthy is False
	
	@pytest.mark.asyncio
	async def test_periodic_health_monitoring(self, discovery, sample_capability_info):
		"""Test periodic health monitoring."""
		# Register capability
		await discovery.register_capability(sample_capability_info)
		
		# Mock health check method
		discovery.check_capability_health = AsyncMock(return_value=True)
		
		# Start health monitoring
		await discovery.start_health_monitoring(interval=0.1)  # 100ms for testing
		
		# Wait for a few checks
		await asyncio.sleep(0.3)
		
		# Stop monitoring
		await discovery.stop_health_monitoring()
		
		# Verify health checks were performed
		assert discovery.check_capability_health.call_count >= 2

# =============================================================================
# Discovery Integration Tests
# =============================================================================

@pytest.mark.integration
class TestDiscoveryIntegration:
	"""Test discovery integration scenarios."""
	
	@pytest.mark.asyncio
	async def test_full_capability_lifecycle(self, discovery, sample_capability_info):
		"""Test complete capability lifecycle."""
		# 1. Register capability
		success = await discovery.register_capability(sample_capability_info)
		assert success is True
		
		# 2. Verify registration
		capability = await discovery.get_capability(sample_capability_info.capability_id)
		assert capability is not None
		
		# 3. Update health
		health_data = {
			"status": "healthy",
			"last_check": datetime.now(timezone.utc),
			"response_time_ms": 25
		}
		await discovery.update_capability_health(sample_capability_info.capability_id, health_data)
		
		# 4. Verify health update
		updated_capability = await discovery.get_capability(sample_capability_info.capability_id)
		assert updated_capability.health_status == "healthy"
		
		# 5. List capabilities
		capabilities = await discovery.list_capabilities()
		assert any(c.capability_id == sample_capability_info.capability_id for c in capabilities)
		
		# 6. Unregister
		success = await discovery.unregister_capability(sample_capability_info.capability_id)
		assert success is True
		
		# 7. Verify unregistration
		capability = await discovery.get_capability(sample_capability_info.capability_id)
		assert capability is None
	
	@pytest.mark.asyncio
	async def test_dependency_resolution(self, discovery):
		"""Test capability dependency resolution."""
		# Register dependency
		dependency_info = APGCapabilityInfo(
			capability_id="dependency_service",
			capability_name="Dependency Service",
			capability_type=CapabilityType.GENERAL_CROSS_FUNCTIONAL,
			version="1.0.0",
			description="Dependency service",
			base_url="http://localhost:8001",
			dependencies=[],
			provides=["base_service"],
			tags=["dependency"]
		)
		
		await discovery.register_capability(dependency_info)
		
		# Register dependent capability
		dependent_info = APGCapabilityInfo(
			capability_id="dependent_service",
			capability_name="Dependent Service", 
			capability_type=CapabilityType.CORE_BUSINESS,
			version="1.0.0",
			description="Service that depends on others",
			base_url="http://localhost:8002",
			dependencies=["dependency_service"],
			provides=["advanced_service"],
			tags=["dependent"]
		)
		
		await discovery.register_capability(dependent_info)
		
		# Resolve dependencies
		dependencies = await discovery.resolve_dependencies("dependent_service")
		assert len(dependencies) == 1
		assert dependencies[0].capability_id == "dependency_service"
	
	@pytest.mark.asyncio
	async def test_service_mesh_discovery(self, discovery):
		"""Test service mesh style discovery."""
		# Register multiple interconnected services
		services = [
			APGCapabilityInfo(
				capability_id="gateway_service",
				capability_name="API Gateway",
				capability_type=CapabilityType.GENERAL_CROSS_FUNCTIONAL,
				version="1.0.0",
				base_url="http://gateway:8000",
				dependencies=[],
				provides=["api_routing", "load_balancing"],
				tags=["gateway", "infrastructure"]
			),
			APGCapabilityInfo(
				capability_id="user_service",
				capability_name="User Management",
				capability_type=CapabilityType.CORE_BUSINESS,
				version="1.0.0",
				base_url="http://users:8000",
				dependencies=["gateway_service"],
				provides=["user_management"],
				tags=["users", "business"]
			),
			APGCapabilityInfo(
				capability_id="order_service",
				capability_name="Order Management", 
				capability_type=CapabilityType.CORE_BUSINESS,
				version="1.0.0",
				base_url="http://orders:8000",
				dependencies=["gateway_service", "user_service"],
				provides=["order_management"],
				tags=["orders", "business"]
			)
		]
		
		# Register all services
		for service in services:
			await discovery.register_capability(service)
		
		# Test service discovery by type
		business_services = await discovery.list_capabilities_by_type(CapabilityType.CORE_BUSINESS)
		assert len(business_services) == 2
		
		infrastructure_services = await discovery.list_capabilities_by_type(CapabilityType.GENERAL_CROSS_FUNCTIONAL)
		assert len(infrastructure_services) == 1
		
		# Test dependency graph
		order_deps = await discovery.resolve_dependencies("order_service")
		assert len(order_deps) == 2
		assert any(dep.capability_id == "gateway_service" for dep in order_deps)
		assert any(dep.capability_id == "user_service" for dep in order_deps)

# =============================================================================
# Discovery Error Handling Tests
# =============================================================================

@pytest.mark.unit
class TestDiscoveryErrorHandling:
	"""Test discovery error handling scenarios."""
	
	@pytest.mark.asyncio
	async def test_register_duplicate_capability(self, discovery, sample_capability_info):
		"""Test handling duplicate capability registration."""
		# Register first time
		success1 = await discovery.register_capability(sample_capability_info)
		assert success1 is True
		
		# Try to register again - should update, not fail
		success2 = await discovery.register_capability(sample_capability_info)
		assert success2 is True
	
	@pytest.mark.asyncio
	async def test_unregister_nonexistent_capability(self, discovery):
		"""Test unregistering non-existent capability."""
		success = await discovery.unregister_capability("nonexistent_capability")
		assert success is False
	
	@pytest.mark.asyncio
	async def test_health_check_connection_error(self, discovery):
		"""Test health check with connection error."""
		# Register capability with unreachable endpoint
		capability_info = APGCapabilityInfo(
			capability_id="unreachable_service",
			capability_name="Unreachable Service",
			capability_type=CapabilityType.CORE_BUSINESS,
			version="1.0.0",
			base_url="http://unreachable:8000",
			health_check_endpoint="/health",
			dependencies=[],
			provides=["unreachable_service"],
			tags=["test"]
		)
		
		await discovery.register_capability(capability_info)
		
		# Health check should handle connection error gracefully
		is_healthy = await discovery.check_capability_health("unreachable_service")
		assert is_healthy is False
	
	@pytest.mark.asyncio
	async def test_circular_dependency_detection(self, discovery):
		"""Test detection of circular dependencies."""
		# Create circular dependency scenario
		service_a = APGCapabilityInfo(
			capability_id="service_a",
			capability_name="Service A",
			capability_type=CapabilityType.CORE_BUSINESS,
			version="1.0.0",
			base_url="http://service-a:8000",
			dependencies=["service_b"],  # Depends on B
			provides=["service_a_func"],
			tags=["test"]
		)
		
		service_b = APGCapabilityInfo(
			capability_id="service_b",
			capability_name="Service B",
			capability_type=CapabilityType.CORE_BUSINESS,
			version="1.0.0",
			base_url="http://service-b:8000", 
			dependencies=["service_a"],  # Depends on A (circular)
			provides=["service_b_func"],
			tags=["test"]
		)
		
		# Register services
		await discovery.register_capability(service_a)
		await discovery.register_capability(service_b)
		
		# Dependency resolution should detect circular dependency
		with pytest.raises(ValueError, match="Circular dependency detected"):
			await discovery.resolve_dependencies("service_a")

# =============================================================================
# Discovery Performance Tests
# =============================================================================

@pytest.mark.performance
class TestDiscoveryPerformance:
	"""Test discovery performance characteristics."""
	
	@pytest.mark.asyncio
	async def test_bulk_capability_registration(self, discovery):
		"""Test bulk capability registration performance."""
		import time
		
		# Generate test capabilities
		capabilities = []
		for i in range(100):
			capability = APGCapabilityInfo(
				capability_id=f"perf_capability_{i:03d}",
				capability_name=f"Performance Test Capability {i}",
				capability_type=CapabilityType.CORE_BUSINESS,
				version="1.0.0",
				base_url=f"http://perf-service-{i:03d}:8000",
				dependencies=[],
				provides=[f"perf_service_{i}"],
				tags=["performance", "test"]
			)
			capabilities.append(capability)
		
		# Measure registration time
		start_time = time.time()
		
		for capability in capabilities:
			await discovery.register_capability(capability)
		
		end_time = time.time()
		duration = end_time - start_time
		
		# Should register 100 capabilities in reasonable time
		assert duration < 10.0  # Less than 10 seconds
		
		# Calculate throughput
		throughput = len(capabilities) / duration
		assert throughput > 10.0  # At least 10 registrations per second
		
		print(f"Registered {len(capabilities)} capabilities in {duration:.2f}s (throughput: {throughput:.1f} cap/s)")
	
	@pytest.mark.asyncio
	async def test_capability_lookup_performance(self, discovery):
		"""Test capability lookup performance."""
		# Register test capabilities
		for i in range(50):
			capability = APGCapabilityInfo(
				capability_id=f"lookup_test_{i:03d}",
				capability_name=f"Lookup Test {i}",
				capability_type=CapabilityType.CORE_BUSINESS,
				version="1.0.0",
				base_url=f"http://lookup-{i:03d}:8000",
				dependencies=[],
				provides=[f"lookup_service_{i}"],
				tags=["lookup", "test"]
			)
			await discovery.register_capability(capability)
		
		import time
		
		# Measure lookup performance
		start_time = time.time()
		
		lookups = []
		for i in range(100):  # 100 lookups
			capability_id = f"lookup_test_{i % 50:03d}"
			capability = await discovery.get_capability(capability_id)
			lookups.append(capability)
		
		end_time = time.time()
		duration = end_time - start_time
		
		# Verify all lookups succeeded
		assert all(cap is not None for cap in lookups)
		
		# Should be fast
		assert duration < 2.0  # Less than 2 seconds for 100 lookups
		
		# Calculate lookup throughput
		throughput = len(lookups) / duration
		assert throughput > 50.0  # At least 50 lookups per second
		
		print(f"Performed {len(lookups)} lookups in {duration:.2f}s (throughput: {throughput:.1f} lookups/s)")