"""
APG Integration API Management - End-to-End Tests

Comprehensive end-to-end tests validating complete functionality across
all components of the Integration API Management capability.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import pytest
import pytest_asyncio
import asyncio
import json
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from ..factory import IntegrationAPIManagementCapability
from ..models import APIConfig, ConsumerConfig, APIKeyConfig, PolicyConfig, PolicyType
from ..discovery import APGCapabilityInfo, CapabilityType

# =============================================================================
# Full Capability E2E Tests
# =============================================================================

@pytest.mark.e2e
class TestFullCapabilityE2E:
	"""Test complete capability functionality end-to-end."""
	
	@pytest.mark.asyncio
	async def test_complete_api_lifecycle_e2e(self, test_capability):
		"""Test complete API lifecycle from registration to retirement."""
		# 1. Register new API
		api_config = APIConfig(
			api_name="e2e_test_api",
			api_title="E2E Test API",
			api_description="End-to-end test API",
			version="1.0.0",
			protocol_type="rest",
			base_path="/e2e",
			upstream_url="http://e2e-service:8000",
			timeout_ms=30000,
			retry_attempts=3,
			auth_type="api_key",
			category="testing",
			tags=["e2e", "test"]
		)
		
		api_id = await test_capability.api_service.register_api(
			config=api_config,
			tenant_id="e2e_tenant",
			created_by="e2e_user"
		)
		
		assert api_id is not None
		assert api_id.startswith("api_")
		
		# 2. Activate API
		activation_success = await test_capability.api_service.activate_api(
			api_id=api_id,
			tenant_id="e2e_tenant",
			activated_by="e2e_admin"
		)
		
		assert activation_success is True
		
		# 3. Register consumer
		consumer_config = ConsumerConfig(
			consumer_name="e2e_test_consumer",
			organization="E2E Test Organization",
			contact_email="e2e@test.com",
			contact_name="E2E Test User",
			global_rate_limit=1000,
			portal_access=True
		)
		
		consumer_id = await test_capability.consumer_service.register_consumer(
			config=consumer_config,
			tenant_id="e2e_tenant",
			created_by="e2e_user"
		)
		
		assert consumer_id is not None
		
		# 4. Approve consumer
		approval_success = await test_capability.consumer_service.approve_consumer(
			consumer_id=consumer_id,
			tenant_id="e2e_tenant",
			approved_by="e2e_admin"
		)
		
		assert approval_success is True
		
		# 5. Generate API key
		key_config = APIKeyConfig(
			key_name="e2e_primary_key",
			scopes=["read", "write"],
			allowed_apis=[api_id]
		)
		
		key_id, api_key = await test_capability.consumer_service.generate_api_key(
			config=key_config,
			tenant_id="e2e_tenant",
			created_by="e2e_user"
		)
		
		assert key_id is not None
		assert api_key is not None
		assert len(api_key) >= 32
		
		# 6. Create policy
		policy_config = PolicyConfig(
			policy_name="e2e_rate_limit",
			policy_type=PolicyType.RATE_LIMITING,
			config={
				"requests_per_minute": 100,
				"burst_size": 20
			},
			execution_order=100
		)
		
		policy_id = await test_capability.policy_service.create_policy(
			api_id=api_id,
			config=policy_config,
			tenant_id="e2e_tenant",
			created_by="e2e_admin"
		)
		
		assert policy_id is not None
		
		# 7. Validate API key
		validated_consumer_id = await test_capability.consumer_service.validate_api_key(
			api_key_hash=test_capability.consumer_service._hash_api_key(api_key),
			tenant_id="e2e_tenant"
		)
		
		assert validated_consumer_id == consumer_id
		
		# 8. Record usage
		usage_data = {
			"request_id": "e2e_req_123",
			"consumer_id": consumer_id,
			"api_id": api_id,
			"endpoint_path": "/e2e/test",
			"method": "GET",
			"timestamp": datetime.now(timezone.utc),
			"response_status": 200,
			"response_time_ms": 125,
			"client_ip": "127.0.0.1",
			"tenant_id": "e2e_tenant"
		}
		
		usage_success = await test_capability.analytics_service.record_usage(usage_data)
		assert usage_success is True
		
		# 9. Get analytics
		total_requests = await test_capability.analytics_service.get_total_requests(
			start_time=datetime.now(timezone.utc) - timedelta(minutes=1),
			end_time=datetime.now(timezone.utc) + timedelta(minutes=1),
			tenant_id="e2e_tenant"
		)
		
		assert total_requests >= 1
		
		# 10. Deprecate API
		deprecation_success = await test_capability.api_service.deprecate_api(
			api_id=api_id,
			migration_timeline="3 months",
			tenant_id="e2e_tenant",
			deprecated_by="e2e_admin"
		)
		
		assert deprecation_success is True
		
		# 11. Verify final state
		final_api = await test_capability.api_service.get_api(api_id, "e2e_tenant")
		assert final_api.status == "deprecated"
	
	@pytest.mark.asyncio
	async def test_gateway_request_processing_e2e(self, test_capability):
		"""Test complete gateway request processing pipeline."""
		# Set up test API and consumer
		api_config = APIConfig(
			api_name="gateway_test_api",
			api_title="Gateway Test API",
			version="1.0.0",
			base_path="/gateway-test",
			upstream_url="http://test-upstream:8000"
		)
		
		api_id = await test_capability.api_service.register_api(
			config=api_config,
			tenant_id="gateway_tenant",
			created_by="gateway_user"
		)
		
		await test_capability.api_service.activate_api(
			api_id=api_id,
			tenant_id="gateway_tenant",
			activated_by="gateway_admin"
		)
		
		# Register consumer and generate API key
		consumer_config = ConsumerConfig(
			consumer_name="gateway_consumer",
			contact_email="gateway@test.com"
		)
		
		consumer_id = await test_capability.consumer_service.register_consumer(
			config=consumer_config,
			tenant_id="gateway_tenant",
			created_by="gateway_user"
		)
		
		await test_capability.consumer_service.approve_consumer(
			consumer_id=consumer_id,
			tenant_id="gateway_tenant",
			approved_by="gateway_admin"
		)
		
		key_config = APIKeyConfig(key_name="gateway_key", allowed_apis=[api_id])
		key_id, api_key = await test_capability.consumer_service.generate_api_key(
			config=key_config,
			tenant_id="gateway_tenant",
			created_by="gateway_user"
		)
		
		# Mock upstream service response
		with patch('aiohttp.ClientSession') as mock_session:
			mock_response = AsyncMock()
			mock_response.status = 200
			mock_response.headers = {"Content-Type": "application/json"}
			mock_response.read = AsyncMock(return_value=b'{"result": "success", "data": "test"}')
			
			mock_session.return_value.__aenter__.return_value.request = AsyncMock(return_value=mock_response)
			mock_session.return_value.__aenter__.return_value.__aenter__ = AsyncMock(return_value=mock_response)
			mock_session.return_value.__aenter__.return_value.__aexit__ = AsyncMock(return_value=None)
			
			# Create gateway request
			from ..gateway import GatewayRequest
			
			request = GatewayRequest(
				request_id="gateway_e2e_123",
				method="GET",
				path="/gateway-test/users",
				headers={
					"X-API-Key": api_key,
					"Content-Type": "application/json"
				},
				query_params={"limit": "10"},
				body=b"",
				client_ip="127.0.0.1",
				user_agent="E2E Test Client",
				timestamp=datetime.now(timezone.utc),
				tenant_id="gateway_tenant"
			)
			
			# Process request through gateway
			if hasattr(test_capability, 'gateway') and test_capability.gateway:
				response = await test_capability.gateway.router.route_request(request)
				
				# Verify successful response
				assert response.status_code == 200
				assert b'{"result": "success"' in response.body
			
			# Verify usage was recorded
			await asyncio.sleep(0.1)  # Allow time for async usage recording
			
			total_requests = await test_capability.analytics_service.get_total_requests(
				start_time=datetime.now(timezone.utc) - timedelta(minutes=1),
				end_time=datetime.now(timezone.utc) + timedelta(minutes=1),
				tenant_id="gateway_tenant"
			)
			
			# Should have recorded the gateway request
			assert total_requests >= 1

@pytest.mark.e2e
class TestAPGIntegrationE2E:
	"""Test APG platform integration end-to-end."""
	
	@pytest.mark.asyncio
	async def test_capability_registration_and_discovery_e2e(self, test_capability):
		"""Test capability registration and service discovery end-to-end."""
		# Register this capability with APG platform
		capability_info = APGCapabilityInfo(
			capability_id="integration_api_management",
			capability_name="Integration API Management",
			capability_type=CapabilityType.GENERAL_CROSS_FUNCTIONAL,
			version="1.0.0",
			description="Enterprise API management and gateway",
			base_url="http://localhost:8080",
			api_endpoints={
				"api": "/api/v1",
				"health": "/health",
				"metrics": "/metrics"
			},
			dependencies=[],
			provides=[
				"api_lifecycle_management",
				"consumer_management", 
				"policy_enforcement",
				"analytics_collection",
				"service_discovery"
			],
			tags=["api", "gateway", "management"]
		)
		
		# Register with discovery service
		if hasattr(test_capability, 'discovery') and test_capability.discovery:
			registration_success = await test_capability.discovery.register_capability(capability_info)
			assert registration_success is True
			
			# Verify registration
			registered_capability = await test_capability.discovery.get_capability("integration_api_management")
			assert registered_capability is not None
			assert registered_capability.capability_name == "Integration API Management"
			
			# List all capabilities
			all_capabilities = await test_capability.discovery.list_capabilities()
			assert len(all_capabilities) >= 1
			assert any(cap.capability_id == "integration_api_management" for cap in all_capabilities)
	
	@pytest.mark.asyncio
	async def test_cross_capability_workflow_e2e(self, test_capability):
		"""Test cross-capability workflow execution end-to-end."""
		if not hasattr(test_capability, 'integration_manager') or not test_capability.integration_manager:
			pytest.skip("Integration manager not available")
		
		# Create test workflow
		from ..integration import CrossCapabilityWorkflow, WorkflowStep, EventType
		
		workflow = CrossCapabilityWorkflow(
			workflow_id="e2e_test_workflow",
			workflow_name="E2E Test Workflow",
			description="End-to-end test workflow",
			trigger_events=[EventType.API_REGISTERED],
			steps=[
				WorkflowStep(
					step_id="notify_discovery",
					step_name="Notify Service Discovery",
					capability_id="service_discovery",
					action="update_api_registry",
					parameters={"auto_register": True}
				),
				WorkflowStep(
					step_id="update_monitoring",
					step_name="Update Monitoring",
					capability_id="monitoring_service",
					action="add_api_monitoring",
					parameters={"enable_alerts": True}
				)
			]
		)
		
		# Register workflow
		workflow_id = await test_capability.integration_manager.register_workflow(workflow)
		assert workflow_id == "e2e_test_workflow"
		
		# Trigger workflow by publishing event
		await test_capability.integration_manager.event_bus.publish(
			EventType.API_REGISTERED,
			{
				"api_id": "test_api_123",
				"api_name": "test_api",
				"tenant_id": "e2e_tenant",
				"timestamp": datetime.now(timezone.utc)
			}
		)
		
		# Wait for workflow execution
		await asyncio.sleep(0.5)
		
		# Verify workflow was executed
		executions = await test_capability.integration_manager.list_workflow_executions(
			workflow_id="e2e_test_workflow",
			start_time=datetime.now(timezone.utc) - timedelta(minutes=1)
		)
		
		assert len(executions) >= 1
	
	@pytest.mark.asyncio
	async def test_health_monitoring_and_alerting_e2e(self, test_capability):
		"""Test health monitoring and alerting end-to-end."""
		if not hasattr(test_capability, 'health_monitor') or not test_capability.health_monitor:
			pytest.skip("Health monitor not available")
		
		# Register health check for API service
		async def api_service_health():
			# Simulate varying health states
			import random
			
			cpu_usage = random.uniform(30, 95)
			memory_usage = random.uniform(40, 90)
			
			if cpu_usage > 90 or memory_usage > 85:
				return {
					"status": "unhealthy",
					"cpu_usage": cpu_usage,
					"memory_usage": memory_usage,
					"error": "High resource usage"
				}
			elif cpu_usage > 80 or memory_usage > 75:
				return {
					"status": "degraded",
					"cpu_usage": cpu_usage,
					"memory_usage": memory_usage,
					"warning": "Elevated resource usage"
				}
			else:
				return {
					"status": "healthy",
					"cpu_usage": cpu_usage,
					"memory_usage": memory_usage
				}
		
		await test_capability.health_monitor.register_health_check(
			name="api_service",
			check_function=api_service_health,
			interval_seconds=30
		)
		
		# Perform health check
		health_result = await test_capability.health_monitor.perform_health_check("api_service")
		assert "status" in health_result
		assert health_result["status"] in ["healthy", "degraded", "unhealthy"]
		
		# Get system health
		system_health = await test_capability.health_monitor.get_system_health()
		assert "overall_status" in system_health
		assert "services" in system_health
		assert len(system_health["services"]) >= 1

# =============================================================================
# Multi-Tenant E2E Tests
# =============================================================================

@pytest.mark.e2e
class TestMultiTenantE2E:
	"""Test multi-tenant scenarios end-to-end."""
	
	@pytest.mark.asyncio
	async def test_tenant_isolation_e2e(self, test_capability):
		"""Test complete tenant isolation across all components."""
		tenants = ["tenant_a", "tenant_b", "tenant_c"]
		tenant_data = {}
		
		# Set up data for each tenant
		for i, tenant_id in enumerate(tenants):
			# Register API for tenant
			api_config = APIConfig(
				api_name=f"tenant_{i}_api",
				api_title=f"Tenant {i} API",
				version="1.0.0",
				base_path=f"/tenant{i}",
				upstream_url=f"http://tenant{i}-service:8000"
			)
			
			api_id = await test_capability.api_service.register_api(
				config=api_config,
				tenant_id=tenant_id,
				created_by=f"user_{i}"
			)
			
			await test_capability.api_service.activate_api(
				api_id=api_id,
				tenant_id=tenant_id,
				activated_by=f"admin_{i}"
			)
			
			# Register consumer for tenant
			consumer_config = ConsumerConfig(
				consumer_name=f"tenant_{i}_consumer",
				contact_email=f"tenant{i}@test.com"
			)
			
			consumer_id = await test_capability.consumer_service.register_consumer(
				config=consumer_config,
				tenant_id=tenant_id,
				created_by=f"user_{i}"
			)
			
			await test_capability.consumer_service.approve_consumer(
				consumer_id=consumer_id,
				tenant_id=tenant_id,
				approved_by=f"admin_{i}"
			)
			
			# Generate API key
			key_config = APIKeyConfig(key_name=f"tenant_{i}_key")
			key_id, api_key = await test_capability.consumer_service.generate_api_key(
				config=key_config,
				tenant_id=tenant_id,
				created_by=f"user_{i}"
			)
			
			# Record usage
			for j in range(5):
				usage_data = {
					"request_id": f"tenant_{i}_req_{j}",
					"consumer_id": consumer_id,
					"api_id": api_id,
					"endpoint_path": f"/tenant{i}/test",
					"method": "GET",
					"timestamp": datetime.now(timezone.utc),
					"response_status": 200,
					"response_time_ms": 100 + j * 10,
					"client_ip": f"192.168.{i}.100",
					"tenant_id": tenant_id
				}
				await test_capability.analytics_service.record_usage(usage_data)
			
			tenant_data[tenant_id] = {
				"api_id": api_id,
				"consumer_id": consumer_id,
				"key_id": key_id,
				"api_key": api_key
			}
		
		# Verify tenant isolation
		for tenant_id in tenants:
			# Each tenant should only see their own APIs
			tenant_apis = await test_capability.api_service.get_apis_by_tenant(tenant_id)
			assert len(tenant_apis) == 1
			assert tenant_apis[0].api_id == tenant_data[tenant_id]["api_id"]
			
			# Each tenant should only see their own usage data
			tenant_requests = await test_capability.analytics_service.get_total_requests(
				start_time=datetime.now(timezone.utc) - timedelta(minutes=1),
				end_time=datetime.now(timezone.utc) + timedelta(minutes=1),
				tenant_id=tenant_id
			)
			assert tenant_requests == 5  # Each tenant recorded 5 requests
			
			# API key should only work for the owning tenant
			validated_consumer = await test_capability.consumer_service.validate_api_key(
				api_key_hash=test_capability.consumer_service._hash_api_key(tenant_data[tenant_id]["api_key"]),
				tenant_id=tenant_id
			)
			assert validated_consumer == tenant_data[tenant_id]["consumer_id"]
			
			# API key should not work for other tenants
			for other_tenant in tenants:
				if other_tenant != tenant_id:
					invalid_validation = await test_capability.consumer_service.validate_api_key(
						api_key_hash=test_capability.consumer_service._hash_api_key(tenant_data[tenant_id]["api_key"]),
						tenant_id=other_tenant
					)
					assert invalid_validation is None

# =============================================================================
# Performance E2E Tests
# =============================================================================

@pytest.mark.e2e
@pytest.mark.performance
class TestPerformanceE2E:
	"""Test performance characteristics end-to-end."""
	
	@pytest.mark.asyncio
	async def test_high_throughput_api_management_e2e(self, test_capability):
		"""Test high-throughput API management operations."""
		import time
		
		# Register multiple APIs concurrently
		start_time = time.time()
		api_count = 100
		
		async def register_api(index):
			api_config = APIConfig(
				api_name=f"perf_api_{index:03d}",
				api_title=f"Performance API {index}",
				version="1.0.0",
				base_path=f"/perf{index:03d}",
				upstream_url=f"http://perf{index:03d}:8000"
			)
			
			return await test_capability.api_service.register_api(
				config=api_config,
				tenant_id="perf_tenant",
				created_by="perf_user"
			)
		
		# Register APIs concurrently
		tasks = [register_api(i) for i in range(api_count)]
		api_ids = await asyncio.gather(*tasks)
		
		registration_time = time.time() - start_time
		
		# Verify all APIs were registered
		assert len(api_ids) == api_count
		assert all(api_id is not None for api_id in api_ids)
		
		# Check registration throughput
		registration_throughput = api_count / registration_time
		assert registration_throughput > 10  # At least 10 APIs/second
		
		print(f"Registered {api_count} APIs in {registration_time:.2f}s (throughput: {registration_throughput:.1f} APIs/s)")
		
		# Activate all APIs concurrently
		start_time = time.time()
		
		activation_tasks = [
			test_capability.api_service.activate_api(api_id, "perf_tenant", "perf_admin")
			for api_id in api_ids
		]
		
		activation_results = await asyncio.gather(*activation_tasks)
		activation_time = time.time() - start_time
		
		# Verify all activations succeeded
		assert all(result is True for result in activation_results)
		
		activation_throughput = api_count / activation_time
		assert activation_throughput > 20  # At least 20 activations/second
		
		print(f"Activated {api_count} APIs in {activation_time:.2f}s (throughput: {activation_throughput:.1f} activations/s)")
	
	@pytest.mark.asyncio
	async def test_concurrent_request_processing_e2e(self, test_capability):
		"""Test concurrent request processing through gateway."""
		# Set up test API
		api_config = APIConfig(
			api_name="concurrent_test_api",
			api_title="Concurrent Test API",
			version="1.0.0",
			base_path="/concurrent",
			upstream_url="http://concurrent-service:8000"
		)
		
		api_id = await test_capability.api_service.register_api(
			config=api_config,
			tenant_id="concurrent_tenant",
			created_by="concurrent_user"
		)
		
		await test_capability.api_service.activate_api(
			api_id=api_id,
			tenant_id="concurrent_tenant",
			activated_by="concurrent_admin"
		)
		
		# Set up consumer and API key
		consumer_config = ConsumerConfig(
			consumer_name="concurrent_consumer",
			contact_email="concurrent@test.com"
		)
		
		consumer_id = await test_capability.consumer_service.register_consumer(
			config=consumer_config,
			tenant_id="concurrent_tenant",
			created_by="concurrent_user"
		)
		
		await test_capability.consumer_service.approve_consumer(
			consumer_id=consumer_id,
			tenant_id="concurrent_tenant",
			approved_by="concurrent_admin"
		)
		
		key_config = APIKeyConfig(key_name="concurrent_key")
		key_id, api_key = await test_capability.consumer_service.generate_api_key(
			config=key_config,
			tenant_id="concurrent_tenant",
			created_by="concurrent_user"
		)
		
		# Mock upstream responses
		with patch('aiohttp.ClientSession') as mock_session:
			mock_response = AsyncMock()
			mock_response.status = 200
			mock_response.headers = {"Content-Type": "application/json"}
			mock_response.read = AsyncMock(return_value=b'{"result": "success"}')
			
			mock_session.return_value.__aenter__.return_value.request = AsyncMock(return_value=mock_response)
			mock_session.return_value.__aenter__.return_value.__aenter__ = AsyncMock(return_value=mock_response)
			mock_session.return_value.__aenter__.return_value.__aexit__ = AsyncMock(return_value=None)
			
			# Process concurrent requests
			if hasattr(test_capability, 'gateway') and test_capability.gateway:
				import time
				
				start_time = time.time()
				request_count = 200
				
				async def process_request(index):
					from ..gateway import GatewayRequest
					
					request = GatewayRequest(
						request_id=f"concurrent_req_{index}",
						method="GET",
						path="/concurrent/test",
						headers={"X-API-Key": api_key},
						query_params={},
						body=b"",
						client_ip="127.0.0.1",
						user_agent="Concurrent Test",
						timestamp=datetime.now(timezone.utc),
						tenant_id="concurrent_tenant"
					)
					
					return await test_capability.gateway.router.route_request(request)
				
				# Process requests concurrently
				tasks = [process_request(i) for i in range(request_count)]
				responses = await asyncio.gather(*tasks)
				
				processing_time = time.time() - start_time
				
				# Verify all requests succeeded
				assert len(responses) == request_count
				assert all(response.status_code == 200 for response in responses)
				
				# Check throughput
				throughput = request_count / processing_time
				assert throughput > 50  # At least 50 requests/second
				
				print(f"Processed {request_count} requests in {processing_time:.2f}s (throughput: {throughput:.1f} req/s)")

# =============================================================================
# Error Recovery E2E Tests
# =============================================================================

@pytest.mark.e2e
class TestErrorRecoveryE2E:
	"""Test error recovery scenarios end-to-end."""
	
	@pytest.mark.asyncio
	async def test_database_connection_recovery_e2e(self, test_capability):
		"""Test recovery from database connection failures."""
		# Simulate database connection failure
		original_session = test_capability.api_service._session
		
		# Mock database failure
		test_capability.api_service._session = None
		
		# Attempt operation during failure
		api_config = APIConfig(
			api_name="recovery_test_api",
			api_title="Recovery Test API",
			version="1.0.0",
			base_path="/recovery",
			upstream_url="http://recovery-service:8000"
		)
		
		# Should handle gracefully during failure
		api_id = await test_capability.api_service.register_api(
			config=api_config,
			tenant_id="recovery_tenant",
			created_by="recovery_user"
		)
		
		# Should return None during database failure
		assert api_id is None
		
		# Restore database connection
		test_capability.api_service._session = original_session
		
		# Operation should work after recovery
		api_id = await test_capability.api_service.register_api(
			config=api_config,
			tenant_id="recovery_tenant",
			created_by="recovery_user"
		)
		
		assert api_id is not None
	
	@pytest.mark.asyncio
	async def test_upstream_service_failure_recovery_e2e(self, test_capability):
		"""Test recovery from upstream service failures."""
		# Set up test API
		api_config = APIConfig(
			api_name="upstream_failure_api",
			api_title="Upstream Failure API",
			version="1.0.0",
			base_path="/upstream-failure",
			upstream_url="http://failing-service:8000"
		)
		
		api_id = await test_capability.api_service.register_api(
			config=api_config,
			tenant_id="failure_tenant",
			created_by="failure_user"
		)
		
		await test_capability.api_service.activate_api(
			api_id=api_id,
			tenant_id="failure_tenant",
			activated_by="failure_admin"
		)
		
		# Set up consumer
		consumer_config = ConsumerConfig(
			consumer_name="failure_consumer",
			contact_email="failure@test.com"
		)
		
		consumer_id = await test_capability.consumer_service.register_consumer(
			config=consumer_config,
			tenant_id="failure_tenant",
			created_by="failure_user"
		)
		
		await test_capability.consumer_service.approve_consumer(
			consumer_id=consumer_id,
			tenant_id="failure_tenant",
			approved_by="failure_admin"
		)
		
		key_config = APIKeyConfig(key_name="failure_key")
		key_id, api_key = await test_capability.consumer_service.generate_api_key(
			config=key_config,
			tenant_id="failure_tenant",
			created_by="failure_user"
		)
		
		if hasattr(test_capability, 'gateway') and test_capability.gateway:
			# Simulate upstream failure, then recovery
			failure_count = 0
			
			def mock_failing_request(*args, **kwargs):
				nonlocal failure_count
				failure_count += 1
				
				response = AsyncMock()
				if failure_count <= 3:
					# First 3 requests fail
					response.status = 503
					response.headers = {"Content-Type": "application/json"}
					response.read = AsyncMock(return_value=b'{"error": "Service unavailable"}')
				else:
					# Subsequent requests succeed
					response.status = 200
					response.headers = {"Content-Type": "application/json"}
					response.read = AsyncMock(return_value=b'{"result": "recovered"}')
				
				return response
			
			with patch('aiohttp.ClientSession') as mock_session:
				mock_session.return_value.__aenter__.return_value.request = AsyncMock(side_effect=mock_failing_request)
				mock_session.return_value.__aenter__.return_value.__aenter__ = AsyncMock(return_value=mock_session.return_value.__aenter__.return_value)
				mock_session.return_value.__aenter__.return_value.__aexit__ = AsyncMock(return_value=None)
				
				from ..gateway import GatewayRequest
				
				# Make requests during failure and recovery
				responses = []
				for i in range(5):
					request = GatewayRequest(
						request_id=f"failure_req_{i}",
						method="GET",
						path="/upstream-failure/test",
						headers={"X-API-Key": api_key},
						query_params={},
						body=b"",
						client_ip="127.0.0.1",
						user_agent="Failure Test",
						timestamp=datetime.now(timezone.utc),
						tenant_id="failure_tenant"
					)
					
					response = await test_capability.gateway.router.route_request(request)
					responses.append(response)
				
				# First 3 should fail, last 2 should succeed
				assert responses[0].status_code == 503
				assert responses[1].status_code == 503
				assert responses[2].status_code == 503
				assert responses[3].status_code == 200
				assert responses[4].status_code == 200
				
				# Verify recovery response content
				assert b'recovered' in responses[4].body