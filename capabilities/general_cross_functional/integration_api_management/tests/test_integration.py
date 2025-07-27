"""
APG Integration API Management - Integration Tests

Unit and integration tests for APG platform integration including workflow
orchestration, event management, and cross-capability coordination.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import pytest
import pytest_asyncio
import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from ..integration import (
	APGIntegrationManager, CrossCapabilityWorkflow, WorkflowStep,
	WorkflowExecutionEngine, EventType, EventBus, PolicyManager,
	CapabilityInteractionLog, WorkflowExecution, WorkflowStatus
)
from ..discovery import APGCapabilityInfo, CapabilityType

# =============================================================================
# APG Integration Manager Tests
# =============================================================================

@pytest.mark.unit
class TestAPGIntegrationManager:
	"""Test APG integration manager functionality."""
	
	@pytest.mark.asyncio
	async def test_initialization(self, integration_manager):
		"""Test integration manager initialization."""
		assert integration_manager is not None
		assert integration_manager.is_initialized is True
		assert integration_manager.event_bus is not None
		assert integration_manager.workflow_engine is not None
	
	@pytest.mark.asyncio
	async def test_register_workflow(self, integration_manager, sample_workflow_data):
		"""Test workflow registration."""
		workflow_id = await integration_manager.register_workflow(sample_workflow_data)
		
		assert workflow_id is not None
		assert workflow_id == sample_workflow_data.workflow_id
		
		# Verify workflow was registered
		workflow = await integration_manager.get_workflow(workflow_id)
		assert workflow is not None
		assert workflow.workflow_name == sample_workflow_data.workflow_name
	
	@pytest.mark.asyncio
	async def test_execute_workflow(self, integration_manager, sample_workflow_data):
		"""Test workflow execution."""
		# Register workflow first
		await integration_manager.register_workflow(sample_workflow_data)
		
		# Execute workflow
		execution_id = await integration_manager.execute_workflow(
			workflow_id=sample_workflow_data.workflow_id,
			trigger_data={"source": "test", "event_id": "test_event_123"},
			context={"tenant_id": "test_tenant", "user_id": "test_user"}
		)
		
		assert execution_id is not None
		
		# Verify execution was created
		execution = await integration_manager.get_workflow_execution(execution_id)
		assert execution is not None
		assert execution.workflow_id == sample_workflow_data.workflow_id
		assert execution.status in [WorkflowStatus.RUNNING, WorkflowStatus.COMPLETED]
	
	@pytest.mark.asyncio
	async def test_list_active_workflows(self, integration_manager, sample_workflow_data):
		"""Test listing active workflows."""
		# Register workflow
		await integration_manager.register_workflow(sample_workflow_data)
		
		# List workflows
		workflows = await integration_manager.list_workflows(active_only=True)
		
		assert len(workflows) >= 1
		assert any(w.workflow_id == sample_workflow_data.workflow_id for w in workflows)
	
	@pytest.mark.asyncio
	async def test_capability_interaction_logging(self, integration_manager):
		"""Test capability interaction logging."""
		interaction_data = {
			"source_capability": "capability_a",
			"target_capability": "capability_b",
			"interaction_type": "api_call",
			"endpoint": "/api/v1/users",
			"method": "GET",
			"request_id": "req_123",
			"timestamp": datetime.now(timezone.utc),
			"duration_ms": 150,
			"status_code": 200,
			"success": True,
			"metadata": {"tenant_id": "test_tenant"}
		}
		
		log_id = await integration_manager.log_capability_interaction(interaction_data)
		assert log_id is not None
		
		# Verify log was created
		logs = await integration_manager.get_capability_interactions(
			source_capability="capability_a",
			target_capability="capability_b",
			start_time=datetime.now(timezone.utc) - timedelta(minutes=1),
			end_time=datetime.now(timezone.utc) + timedelta(minutes=1)
		)
		
		assert len(logs) >= 1
		assert any(log.request_id == "req_123" for log in logs)

@pytest.mark.unit
class TestWorkflowExecutionEngine:
	"""Test workflow execution engine."""
	
	@pytest_asyncio.fixture
	async def workflow_engine(self, redis_client, service_discovery, metrics_collector):
		"""Create workflow execution engine."""
		engine = WorkflowExecutionEngine(redis_client, service_discovery, metrics_collector)
		await engine.initialize()
		
		yield engine
		
		await engine.cleanup()
	
	@pytest.mark.asyncio
	async def test_simple_workflow_execution(self, workflow_engine, sample_workflow_data):
		"""Test simple workflow execution."""
		# Mock capability endpoints
		with patch('aiohttp.ClientSession') as mock_session:
			mock_response = AsyncMock()
			mock_response.status = 200
			mock_response.json = AsyncMock(return_value={"result": "success"})
			
			mock_session.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
			mock_session.return_value.__aenter__.return_value.__aenter__ = AsyncMock(return_value=mock_response)
			mock_session.return_value.__aenter__.return_value.__aexit__ = AsyncMock(return_value=None)
			
			# Execute workflow
			execution_id = await workflow_engine.execute_workflow(
				workflow=sample_workflow_data,
				trigger_data={"event": "test"},
				context={"tenant_id": "test_tenant"}
			)
			
			assert execution_id is not None
			
			# Wait for execution to complete
			await asyncio.sleep(0.5)
			
			# Verify execution completed
			execution = await workflow_engine.get_execution(execution_id)
			assert execution.status == WorkflowStatus.COMPLETED
	
	@pytest.mark.asyncio
	async def test_workflow_with_failure(self, workflow_engine, sample_workflow_data):
		"""Test workflow execution with step failure."""
		# Mock capability endpoint that fails
		with patch('aiohttp.ClientSession') as mock_session:
			mock_response = AsyncMock()
			mock_response.status = 500
			mock_response.json = AsyncMock(return_value={"error": "Internal server error"})
			
			mock_session.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
			mock_session.return_value.__aenter__.return_value.__aenter__ = AsyncMock(return_value=mock_response)
			mock_session.return_value.__aenter__.return_value.__aexit__ = AsyncMock(return_value=None)
			
			# Execute workflow
			execution_id = await workflow_engine.execute_workflow(
				workflow=sample_workflow_data,
				trigger_data={"event": "test"},
				context={"tenant_id": "test_tenant"}
			)
			
			# Wait for execution to fail
			await asyncio.sleep(0.5)
			
			# Verify execution failed
			execution = await workflow_engine.get_execution(execution_id)
			assert execution.status == WorkflowStatus.FAILED
			assert execution.error_message is not None
	
	@pytest.mark.asyncio
	async def test_workflow_retry_mechanism(self, workflow_engine):
		"""Test workflow step retry mechanism."""
		# Create workflow with retry configuration
		workflow = CrossCapabilityWorkflow(
			workflow_id="retry_test_workflow",
			workflow_name="Retry Test Workflow",
			description="Test workflow with retries",
			trigger_events=[EventType.CAPABILITY_REGISTERED],
			steps=[
				WorkflowStep(
					step_id="retry_step",
					step_name="Step with Retries",
					capability_id="test_capability",
					action="test_action",
					parameters={"param": "value"},
					retry_count=3,
					retry_delay_ms=100
				)
			]
		)
		
		failure_count = 0
		
		# Mock endpoint that fails first 2 times, succeeds on 3rd
		def mock_request(*args, **kwargs):
			nonlocal failure_count
			failure_count += 1
			
			response = AsyncMock()
			if failure_count < 3:
				response.status = 500
				response.json = AsyncMock(return_value={"error": "Temporary failure"})
			else:
				response.status = 200
				response.json = AsyncMock(return_value={"result": "success"})
			
			return response
		
		with patch('aiohttp.ClientSession') as mock_session:
			mock_session.return_value.__aenter__.return_value.post = AsyncMock(side_effect=mock_request)
			mock_session.return_value.__aenter__.return_value.__aenter__ = AsyncMock(return_value=mock_session.return_value.__aenter__.return_value)
			mock_session.return_value.__aenter__.return_value.__aexit__ = AsyncMock(return_value=None)
			
			# Execute workflow
			execution_id = await workflow_engine.execute_workflow(
				workflow=workflow,
				trigger_data={"event": "test"},
				context={"tenant_id": "test_tenant"}
			)
			
			# Wait for retries to complete
			await asyncio.sleep(1.0)
			
			# Verify execution succeeded after retries
			execution = await workflow_engine.get_execution(execution_id)
			assert execution.status == WorkflowStatus.COMPLETED
			assert failure_count == 3  # Failed twice, succeeded on third attempt

# =============================================================================
# Event Bus Tests
# =============================================================================

@pytest.mark.unit
class TestEventBus:
	"""Test event bus functionality."""
	
	@pytest_asyncio.fixture
	async def event_bus(self, redis_client):
		"""Create event bus."""
		bus = EventBus(redis_client)
		await bus.initialize()
		
		yield bus
		
		await bus.cleanup()
	
	@pytest.mark.asyncio
	async def test_publish_and_subscribe(self, event_bus):
		"""Test event publishing and subscription."""
		received_events = []
		
		async def event_handler(event_type, event_data):
			received_events.append((event_type, event_data))
		
		# Subscribe to events
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
		assert received_events[0][0] == EventType.CAPABILITY_REGISTERED
		assert received_events[0][1]["capability_id"] == "test_capability"
	
	@pytest.mark.asyncio
	async def test_event_filtering(self, event_bus):
		"""Test event filtering by criteria."""
		received_events = []
		
		async def filtered_handler(event_type, event_data):
			if event_data.get("tenant_id") == "target_tenant":
				received_events.append((event_type, event_data))
		
		# Subscribe with filtering logic
		await event_bus.subscribe(EventType.API_CALL_MADE, filtered_handler)
		
		# Publish events with different tenant IDs
		events = [
			{"api_id": "api1", "tenant_id": "target_tenant", "status": "success"},
			{"api_id": "api2", "tenant_id": "other_tenant", "status": "success"},
			{"api_id": "api3", "tenant_id": "target_tenant", "status": "error"}
		]
		
		for event_data in events:
			await event_bus.publish(EventType.API_CALL_MADE, event_data)
		
		# Wait for processing
		await asyncio.sleep(0.2)
		
		# Only events for target_tenant should be received
		assert len(received_events) == 2
		assert all(event[1]["tenant_id"] == "target_tenant" for event in received_events)
	
	@pytest.mark.asyncio
	async def test_event_replay(self, event_bus):
		"""Test event replay functionality."""
		# Publish some events
		for i in range(5):
			event_data = {
				"event_id": f"event_{i}",
				"timestamp": datetime.now(timezone.utc),
				"data": f"Event {i}"
			}
			await event_bus.publish(EventType.WORKFLOW_COMPLETED, event_data)
		
		# Subscribe to replayed events
		replayed_events = []
		
		async def replay_handler(event_type, event_data):
			replayed_events.append(event_data)
		
		# Replay events from the last minute
		start_time = datetime.now(timezone.utc) - timedelta(minutes=1)
		await event_bus.replay_events(
			EventType.WORKFLOW_COMPLETED,
			start_time,
			replay_handler
		)
		
		# All 5 events should be replayed
		assert len(replayed_events) == 5
		assert all("event_id" in event for event in replayed_events)

# =============================================================================
# Policy Manager Tests
# =============================================================================

@pytest.mark.unit
class TestPolicyManager:
	"""Test policy manager for cross-capability interactions."""
	
	@pytest_asyncio.fixture
	async def policy_manager(self, redis_client):
		"""Create policy manager."""
		manager = PolicyManager(redis_client)
		await manager.initialize()
		
		yield manager
		
		await manager.cleanup()
	
	@pytest.mark.asyncio
	async def test_create_interaction_policy(self, policy_manager):
		"""Test creating interaction policy."""
		policy_data = {
			"policy_name": "user_management_access",
			"source_capability": "order_management",
			"target_capability": "user_management",
			"allowed_operations": ["GET /users", "POST /users/validate"],
			"rate_limit": {
				"requests_per_minute": 1000,
				"burst_size": 100
			},
			"security_requirements": {
				"authentication": "required",
				"authorization": "rbac",
				"encryption": "tls"
			},
			"data_sharing_rules": {
				"allowed_fields": ["user_id", "email", "status"],
				"prohibited_fields": ["password", "ssn", "credit_card"]
			}
		}
		
		policy_id = await policy_manager.create_policy(policy_data)
		assert policy_id is not None
		
		# Verify policy was created
		policy = await policy_manager.get_policy(policy_id)
		assert policy is not None
		assert policy.policy_name == "user_management_access"
		assert policy.source_capability == "order_management"
	
	@pytest.mark.asyncio
	async def test_evaluate_interaction_policy(self, policy_manager):
		"""Test evaluating interaction against policies."""
		# Create policy
		policy_data = {
			"policy_name": "restricted_access",
			"source_capability": "external_service",
			"target_capability": "core_service",
			"allowed_operations": ["GET /public/*"],
			"rate_limit": {"requests_per_minute": 100}
		}
		
		await policy_manager.create_policy(policy_data)
		
		# Test allowed interaction
		interaction = {
			"source_capability": "external_service",
			"target_capability": "core_service",
			"operation": "GET /public/status",
			"timestamp": datetime.now(timezone.utc)
		}
		
		result = await policy_manager.evaluate_interaction(interaction)
		assert result.allowed is True
		assert result.policies_applied > 0
		
		# Test prohibited interaction
		prohibited_interaction = {
			"source_capability": "external_service",
			"target_capability": "core_service",
			"operation": "DELETE /admin/users",
			"timestamp": datetime.now(timezone.utc)
		}
		
		result = await policy_manager.evaluate_interaction(prohibited_interaction)
		assert result.allowed is False
		assert result.reason is not None
	
	@pytest.mark.asyncio
	async def test_policy_enforcement_logging(self, policy_manager):
		"""Test policy enforcement logging."""
		# Create policy
		policy_data = {
			"policy_name": "audit_policy",
			"source_capability": "*",
			"target_capability": "audit_service",
			"logging_required": True,
			"audit_level": "detailed"
		}
		
		await policy_manager.create_policy(policy_data)
		
		# Simulate interaction
		interaction = {
			"source_capability": "user_service",
			"target_capability": "audit_service",
			"operation": "POST /audit/events",
			"timestamp": datetime.now(timezone.utc),
			"user_id": "test_user",
			"tenant_id": "test_tenant"
		}
		
		result = await policy_manager.evaluate_interaction(interaction)
		assert result.allowed is True
		assert result.audit_required is True
		
		# Verify audit log was created
		audit_logs = await policy_manager.get_audit_logs(
			start_time=datetime.now(timezone.utc) - timedelta(minutes=1),
			end_time=datetime.now(timezone.utc) + timedelta(minutes=1)
		)
		
		assert len(audit_logs) >= 1

# =============================================================================
# Cross-Capability Integration Tests
# =============================================================================

@pytest.mark.integration
class TestCrossCapabilityIntegration:
	"""Test cross-capability integration scenarios."""
	
	@pytest.mark.asyncio
	async def test_api_management_discovery_integration(self, integration_manager, api_service):
		"""Test integration between API management and service discovery."""
		# Mock API registration triggering discovery update
		api_service.register_api = AsyncMock(return_value="api_123")
		
		# Register a workflow that responds to API registration
		workflow = CrossCapabilityWorkflow(
			workflow_id="api_discovery_sync",
			workflow_name="API Discovery Sync",
			description="Sync API registration with service discovery",
			trigger_events=[EventType.API_REGISTERED],
			steps=[
				WorkflowStep(
					step_id="update_discovery",
					step_name="Update Service Discovery",
					capability_id="service_discovery",
					action="register_api_endpoint",
					parameters={"auto_discovery": True}
				)
			]
		)
		
		await integration_manager.register_workflow(workflow)
		
		# Trigger API registration event
		await integration_manager.event_bus.publish(
			EventType.API_REGISTERED,
			{
				"api_id": "api_123",
				"api_name": "test_api",
				"base_path": "/test",
				"upstream_url": "http://test-service:8000",
				"tenant_id": "test_tenant"
			}
		)
		
		# Wait for workflow execution
		await asyncio.sleep(0.5)
		
		# Verify workflow was triggered
		executions = await integration_manager.list_workflow_executions(
			workflow_id="api_discovery_sync",
			start_time=datetime.now(timezone.utc) - timedelta(minutes=1)
		)
		
		assert len(executions) >= 1
		assert executions[0].status in [WorkflowStatus.COMPLETED, WorkflowStatus.RUNNING]
	
	@pytest.mark.asyncio
	async def test_multi_tenant_workflow_isolation(self, integration_manager, sample_workflow_data):
		"""Test workflow isolation between tenants."""
		# Register workflow
		await integration_manager.register_workflow(sample_workflow_data)
		
		# Execute workflow for tenant A
		execution_a = await integration_manager.execute_workflow(
			workflow_id=sample_workflow_data.workflow_id,
			trigger_data={"event": "test_a"},
			context={"tenant_id": "tenant_a", "user_id": "user_a"}
		)
		
		# Execute workflow for tenant B
		execution_b = await integration_manager.execute_workflow(
			workflow_id=sample_workflow_data.workflow_id,
			trigger_data={"event": "test_b"},
			context={"tenant_id": "tenant_b", "user_id": "user_b"}
		)
		
		# Verify executions are isolated
		assert execution_a != execution_b
		
		# Get executions for each tenant
		tenant_a_executions = await integration_manager.list_workflow_executions(
			workflow_id=sample_workflow_data.workflow_id,
			tenant_id="tenant_a"
		)
		
		tenant_b_executions = await integration_manager.list_workflow_executions(
			workflow_id=sample_workflow_data.workflow_id,
			tenant_id="tenant_b"
		)
		
		# Each tenant should only see their own executions
		assert len(tenant_a_executions) == 1
		assert len(tenant_b_executions) == 1
		assert tenant_a_executions[0].execution_id == execution_a
		assert tenant_b_executions[0].execution_id == execution_b
	
	@pytest.mark.asyncio
	async def test_capability_health_monitoring_integration(self, integration_manager, health_monitor):
		"""Test integration with capability health monitoring."""
		# Register health monitoring workflow
		health_workflow = CrossCapabilityWorkflow(
			workflow_id="health_alert_workflow",
			workflow_name="Health Alert Workflow",
			description="Respond to capability health changes",
			trigger_events=[EventType.CAPABILITY_HEALTH_CHANGED],
			steps=[
				WorkflowStep(
					step_id="check_dependencies",
					step_name="Check Dependent Capabilities",
					capability_id="service_discovery",
					action="check_dependents",
					parameters={"cascade_check": True}
				),
				WorkflowStep(
					step_id="send_alert",
					step_name="Send Health Alert",
					capability_id="notification_service",
					action="send_alert",
					parameters={"alert_type": "health_degraded"}
				)
			]
		)
		
		await integration_manager.register_workflow(health_workflow)
		
		# Simulate capability health change
		await integration_manager.event_bus.publish(
			EventType.CAPABILITY_HEALTH_CHANGED,
			{
				"capability_id": "critical_service",
				"old_status": "healthy",
				"new_status": "unhealthy",
				"timestamp": datetime.now(timezone.utc),
				"details": {"error": "Service unavailable"}
			}
		)
		
		# Wait for workflow execution
		await asyncio.sleep(0.5)
		
		# Verify health alert workflow was triggered
		executions = await integration_manager.list_workflow_executions(
			workflow_id="health_alert_workflow",
			start_time=datetime.now(timezone.utc) - timedelta(minutes=1)
		)
		
		assert len(executions) >= 1

# =============================================================================
# Integration Error Handling Tests
# =============================================================================

@pytest.mark.unit
class TestIntegrationErrorHandling:
	"""Test integration error handling scenarios."""
	
	@pytest.mark.asyncio
	async def test_workflow_timeout_handling(self, integration_manager):
		"""Test handling of workflow timeouts."""
		# Create workflow with short timeout
		timeout_workflow = CrossCapabilityWorkflow(
			workflow_id="timeout_test_workflow",
			workflow_name="Timeout Test Workflow",
			description="Workflow that times out",
			trigger_events=[EventType.CAPABILITY_REGISTERED],
			timeout_seconds=1,  # 1 second timeout
			steps=[
				WorkflowStep(
					step_id="slow_step",
					step_name="Slow Step",
					capability_id="slow_service",
					action="slow_action",
					parameters={"delay": 5}  # 5 second delay
				)
			]
		)
		
		await integration_manager.register_workflow(timeout_workflow)
		
		# Execute workflow
		execution_id = await integration_manager.execute_workflow(
			workflow_id="timeout_test_workflow",
			trigger_data={"event": "test"},
			context={"tenant_id": "test_tenant"}
		)
		
		# Wait for timeout
		await asyncio.sleep(2.0)
		
		# Verify execution timed out
		execution = await integration_manager.get_workflow_execution(execution_id)
		assert execution.status == WorkflowStatus.FAILED
		assert "timeout" in execution.error_message.lower()
	
	@pytest.mark.asyncio
	async def test_invalid_workflow_step_handling(self, integration_manager):
		"""Test handling of invalid workflow steps."""
		# Create workflow with invalid step
		invalid_workflow = CrossCapabilityWorkflow(
			workflow_id="invalid_test_workflow",
			workflow_name="Invalid Test Workflow",
			description="Workflow with invalid step",
			trigger_events=[EventType.CAPABILITY_REGISTERED],
			steps=[
				WorkflowStep(
					step_id="invalid_step",
					step_name="Invalid Step",
					capability_id="nonexistent_capability",
					action="nonexistent_action",
					parameters={}
				)
			]
		)
		
		await integration_manager.register_workflow(invalid_workflow)
		
		# Execute workflow
		execution_id = await integration_manager.execute_workflow(
			workflow_id="invalid_test_workflow",
			trigger_data={"event": "test"},
			context={"tenant_id": "test_tenant"}
		)
		
		# Wait for execution to fail
		await asyncio.sleep(0.5)
		
		# Verify execution failed
		execution = await integration_manager.get_workflow_execution(execution_id)
		assert execution.status == WorkflowStatus.FAILED
		assert execution.error_message is not None
	
	@pytest.mark.asyncio
	async def test_concurrent_workflow_execution_limits(self, integration_manager, sample_workflow_data):
		"""Test concurrent workflow execution limits."""
		# Register workflow with concurrency limit
		limited_workflow = sample_workflow_data
		limited_workflow.max_concurrent_executions = 2
		
		await integration_manager.register_workflow(limited_workflow)
		
		# Start multiple executions simultaneously
		executions = []
		for i in range(5):
			execution_id = await integration_manager.execute_workflow(
				workflow_id=sample_workflow_data.workflow_id,
				trigger_data={"event": f"test_{i}"},
				context={"tenant_id": "test_tenant"}
			)
			executions.append(execution_id)
		
		# Wait briefly for processing
		await asyncio.sleep(0.2)
		
		# Check execution statuses
		running_count = 0
		queued_count = 0
		
		for execution_id in executions:
			execution = await integration_manager.get_workflow_execution(execution_id)
			if execution.status == WorkflowStatus.RUNNING:
				running_count += 1
			elif execution.status == WorkflowStatus.QUEUED:
				queued_count += 1
		
		# Should respect concurrency limit
		assert running_count <= 2
		assert queued_count >= 3

# =============================================================================
# Integration Performance Tests
# =============================================================================

@pytest.mark.performance
class TestIntegrationPerformance:
	"""Test integration performance characteristics."""
	
	@pytest.mark.asyncio
	async def test_event_throughput(self, integration_manager):
		"""Test event publishing and processing throughput."""
		import time
		
		# Subscribe to events
		received_count = 0
		
		async def event_counter(event_type, event_data):
			nonlocal received_count
			received_count += 1
		
		await integration_manager.event_bus.subscribe(EventType.API_CALL_MADE, event_counter)
		
		# Publish many events
		start_time = time.time()
		event_count = 1000
		
		for i in range(event_count):
			await integration_manager.event_bus.publish(
				EventType.API_CALL_MADE,
				{
					"api_id": f"api_{i}",
					"consumer_id": f"consumer_{i % 10}",
					"timestamp": datetime.now(timezone.utc),
					"status": "success"
				}
			)
		
		# Wait for processing
		await asyncio.sleep(2.0)
		
		end_time = time.time()
		duration = end_time - start_time
		
		# Calculate throughput
		publish_throughput = event_count / duration
		processing_throughput = received_count / 2.0  # 2 seconds for processing
		
		assert publish_throughput > 100  # At least 100 events/second publish
		assert processing_throughput > 50  # At least 50 events/second processing
		
		print(f"Published {event_count} events in {duration:.2f}s (throughput: {publish_throughput:.1f} events/s)")
		print(f"Processed {received_count} events in 2.0s (throughput: {processing_throughput:.1f} events/s)")
	
	@pytest.mark.asyncio
	async def test_concurrent_workflow_execution_performance(self, integration_manager, sample_workflow_data):
		"""Test concurrent workflow execution performance."""
		import time
		
		# Register workflow
		await integration_manager.register_workflow(sample_workflow_data)
		
		# Execute many workflows concurrently
		start_time = time.time()
		execution_count = 50
		
		tasks = []
		for i in range(execution_count):
			task = integration_manager.execute_workflow(
				workflow_id=sample_workflow_data.workflow_id,
				trigger_data={"event": f"perf_test_{i}"},
				context={"tenant_id": "perf_tenant", "user_id": f"user_{i}"}
			)
			tasks.append(task)
		
		# Wait for all executions to start
		execution_ids = await asyncio.gather(*tasks)
		
		end_time = time.time()
		duration = end_time - start_time
		
		# Verify all executions were created
		assert len(execution_ids) == execution_count
		assert all(eid is not None for eid in execution_ids)
		
		# Calculate throughput
		throughput = execution_count / duration
		assert throughput > 10  # At least 10 executions/second
		
		print(f"Started {execution_count} workflow executions in {duration:.2f}s (throughput: {throughput:.1f} exec/s)")