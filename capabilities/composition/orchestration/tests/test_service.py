"""
APG Workflow Orchestration Service Tests

Comprehensive unit tests for the workflow orchestration service layer.
Tests service operations, business logic, error handling, and integrations.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
import json

from ..service import WorkflowOrchestrationService
from ..models import (
	Workflow, WorkflowInstance, TaskDefinition, TaskExecution,
	WorkflowStatus, TaskStatus, Priority, TaskType
)
from ..database import DatabaseManager

class TestWorkflowOrchestrationService:
	"""Test WorkflowOrchestrationService functionality."""
	
	@pytest.mark.asyncio
	async def test_service_initialization(self, database_manager, redis_client):
		"""Test service initialization."""
		service = WorkflowOrchestrationService(
			database_manager, redis_client, "test_tenant"
		)
		
		assert service.tenant_id == "test_tenant"
		assert service.database_manager == database_manager
		assert service.redis_client == redis_client
		assert service.executor is not None
		assert service.scheduler is not None
		assert service.state_manager is not None
	
	@pytest.mark.asyncio
	async def test_execute_workflow_success(self, workflow_service, sample_workflow):
		"""Test successful workflow execution."""
		# Mock workflow storage
		with patch.object(workflow_service.executor, 'execute_workflow') as mock_execute:
			mock_instance = WorkflowInstance(
				workflow_id=sample_workflow.id,
				input_data={"test": "data"},
				status=WorkflowStatus.RUNNING,
				tenant_id="test_tenant",
				started_by="test_user"
			)
			mock_execute.return_value = mock_instance
			
			# Execute workflow
			result = await workflow_service.execute_workflow(
				workflow_id=sample_workflow.id,
				input_data={"test": "data"},
				user_id="test_user"
			)
			
			assert result is not None
			assert result.workflow_id == sample_workflow.id
			assert result.status == WorkflowStatus.RUNNING
			assert result.input_data == {"test": "data"}
			mock_execute.assert_called_once()
	
	@pytest.mark.asyncio
	async def test_execute_workflow_with_overrides(self, workflow_service, sample_workflow):
		"""Test workflow execution with configuration overrides."""
		with patch.object(workflow_service.executor, 'execute_workflow') as mock_execute:
			mock_instance = WorkflowInstance(
				workflow_id=sample_workflow.id,
				input_data={"test": "data"},
				configuration_overrides={"max_concurrent_tasks": 10},
				status=WorkflowStatus.RUNNING,
				tenant_id="test_tenant",
				started_by="test_user"
			)
			mock_execute.return_value = mock_instance
			
			# Execute with overrides
			result = await workflow_service.execute_workflow(
				workflow_id=sample_workflow.id,
				input_data={"test": "data"},
				user_id="test_user",
				configuration_overrides={"max_concurrent_tasks": 10}
			)
			
			assert result.configuration_overrides == {"max_concurrent_tasks": 10}
	
	@pytest.mark.asyncio
	async def test_execute_workflow_with_priority(self, workflow_service, sample_workflow):
		"""Test workflow execution with custom priority."""
		with patch.object(workflow_service.executor, 'execute_workflow') as mock_execute:
			mock_instance = WorkflowInstance(
				workflow_id=sample_workflow.id,
				input_data={"test": "data"},
				priority=Priority.CRITICAL,
				status=WorkflowStatus.RUNNING,
				tenant_id="test_tenant",
				started_by="test_user"
			)
			mock_execute.return_value = mock_instance
			
			# Execute with custom priority
			result = await workflow_service.execute_workflow(
				workflow_id=sample_workflow.id,
				input_data={"test": "data"},
				user_id="test_user",
				priority=Priority.CRITICAL
			)
			
			assert result.priority == Priority.CRITICAL
	
	@pytest.mark.asyncio
	async def test_execute_workflow_scheduled(self, workflow_service, sample_workflow):
		"""Test scheduled workflow execution."""
		future_time = datetime.now(timezone.utc) + timedelta(hours=1)
		
		with patch.object(workflow_service.scheduler, 'schedule_workflow') as mock_schedule:
			mock_instance = WorkflowInstance(
				workflow_id=sample_workflow.id,
				input_data={"test": "data"},
				status=WorkflowStatus.SCHEDULED,
				tenant_id="test_tenant",
				started_by="test_user"
			)
			mock_schedule.return_value = mock_instance
			
			# Schedule workflow
			result = await workflow_service.execute_workflow(
				workflow_id=sample_workflow.id,
				input_data={"test": "data"},
				user_id="test_user",
				schedule_at=future_time
			)
			
			assert result.status == WorkflowStatus.SCHEDULED
			mock_schedule.assert_called_once()
	
	@pytest.mark.asyncio
	async def test_execute_workflow_error_handling(self, workflow_service, sample_workflow):
		"""Test workflow execution error handling."""
		with patch.object(workflow_service.executor, 'execute_workflow') as mock_execute:
			mock_execute.side_effect = Exception("Execution failed")
			
			# Test execution failure
			with pytest.raises(Exception, match="Execution failed"):
				await workflow_service.execute_workflow(
					workflow_id=sample_workflow.id,
					input_data={"test": "data"},
					user_id="test_user"
				)
	
	@pytest.mark.asyncio
	async def test_get_workflow_instance(self, workflow_service, sample_workflow_instance):
		"""Test getting workflow instance."""
		with patch.object(workflow_service.repositories['workflow_instance'], 'get_by_id') as mock_get:
			mock_get.return_value = sample_workflow_instance
			
			# Get instance
			result = await workflow_service.get_workflow_instance(
				sample_workflow_instance.id
			)
			
			assert result == sample_workflow_instance
			mock_get.assert_called_once_with(
				sample_workflow_instance.id,
				workflow_service.tenant_id
			)
	
	@pytest.mark.asyncio
	async def test_get_workflow_instance_with_tasks(self, workflow_service, sample_workflow_instance, sample_task_execution):
		"""Test getting workflow instance with task executions."""
		with patch.object(workflow_service.repositories['workflow_instance'], 'get_by_id') as mock_get_instance, \
			 patch.object(workflow_service.repositories['task_execution'], 'get_by_instance_id') as mock_get_tasks:
			
			mock_get_instance.return_value = sample_workflow_instance
			mock_get_tasks.return_value = [sample_task_execution]
			
			# Get instance with tasks
			result = await workflow_service.get_workflow_instance(
				sample_workflow_instance.id,
				include_tasks=True
			)
			
			assert result == sample_workflow_instance
			mock_get_instance.assert_called_once()
			mock_get_tasks.assert_called_once()
	
	@pytest.mark.asyncio
	async def test_get_workflow_instances(self, workflow_service, sample_workflow):
		"""Test getting workflow instances list."""
		mock_instances = [
			WorkflowInstance(
				workflow_id=sample_workflow.id,
				input_data={"test": f"data_{i}"},
				status=WorkflowStatus.COMPLETED if i % 2 == 0 else WorkflowStatus.RUNNING,
				tenant_id="test_tenant",
				started_by="test_user"
			)
			for i in range(3)
		]
		
		with patch.object(workflow_service.repositories['workflow_instance'], 'get_by_workflow_id') as mock_get:
			mock_get.return_value = mock_instances
			
			# Get instances
			result = await workflow_service.get_workflow_instances(
				sample_workflow.id,
				status=["completed", "running"],
				limit=10,
				offset=0
			)
			
			assert len(result) == 3
			assert all(inst.workflow_id == sample_workflow.id for inst in result)
	
	@pytest.mark.asyncio
	async def test_pause_workflow_instance(self, workflow_service, sample_workflow_instance):
		"""Test pausing workflow instance."""
		with patch.object(workflow_service.executor, 'pause_instance') as mock_pause, \
			 patch.object(workflow_service.repositories['workflow_instance'], 'update_status') as mock_update:
			
			mock_pause.return_value = True
			updated_instance = sample_workflow_instance.model_copy()
			updated_instance.status = WorkflowStatus.PAUSED
			mock_update.return_value = updated_instance
			
			# Pause instance
			result = await workflow_service.pause_workflow_instance(
				sample_workflow_instance.id,
				"test_user"
			)
			
			assert result is True
			mock_pause.assert_called_once()
			mock_update.assert_called_once()
	
	@pytest.mark.asyncio
	async def test_resume_workflow_instance(self, workflow_service, sample_workflow_instance):
		"""Test resuming workflow instance."""
		# Set initial state as paused
		paused_instance = sample_workflow_instance.model_copy()
		paused_instance.status = WorkflowStatus.PAUSED
		
		with patch.object(workflow_service.executor, 'resume_instance') as mock_resume, \
			 patch.object(workflow_service.repositories['workflow_instance'], 'update_status') as mock_update:
			
			mock_resume.return_value = True
			resumed_instance = paused_instance.model_copy()
			resumed_instance.status = WorkflowStatus.RUNNING
			mock_update.return_value = resumed_instance
			
			# Resume instance
			result = await workflow_service.resume_workflow_instance(
				sample_workflow_instance.id,
				"test_user"
			)
			
			assert result is True
			mock_resume.assert_called_once()
			mock_update.assert_called_once()
	
	@pytest.mark.asyncio
	async def test_stop_workflow_instance(self, workflow_service, sample_workflow_instance):
		"""Test stopping workflow instance."""
		with patch.object(workflow_service.executor, 'stop_instance') as mock_stop, \
			 patch.object(workflow_service.repositories['workflow_instance'], 'update_status') as mock_update:
			
			mock_stop.return_value = True
			stopped_instance = sample_workflow_instance.model_copy()
			stopped_instance.status = WorkflowStatus.CANCELLED
			mock_update.return_value = stopped_instance
			
			# Stop instance
			result = await workflow_service.stop_workflow_instance(
				sample_workflow_instance.id,
				"test_user",
				"User requested stop"
			)
			
			assert result is True
			mock_stop.assert_called_once_with(
				sample_workflow_instance.id,
				"test_user",
				"User requested stop"
			)
			mock_update.assert_called_once()
	
	@pytest.mark.asyncio
	async def test_get_instance_metrics(self, workflow_service, sample_workflow_instance):
		"""Test getting instance metrics."""
		mock_metrics = {
			"instance_id": sample_workflow_instance.id,
			"status": "running",
			"progress_percentage": 75,
			"tasks_completed": 3,
			"tasks_total": 5,
			"duration_seconds": 300,
			"estimated_remaining_seconds": 120
		}
		
		with patch.object(workflow_service.state_manager, 'get_instance_metrics') as mock_get_metrics:
			mock_get_metrics.return_value = mock_metrics
			
			# Get metrics
			result = await workflow_service.get_instance_metrics(
				sample_workflow_instance.id
			)
			
			assert result == mock_metrics
			assert result["progress_percentage"] == 75
			assert result["tasks_completed"] == 3
	
	@pytest.mark.asyncio
	async def test_update_task_assignment(self, workflow_service, sample_task_execution):
		"""Test updating task assignment."""
		with patch.object(workflow_service.repositories['task_execution'], 'update') as mock_update:
			updated_execution = sample_task_execution.model_copy()
			updated_execution.assigned_to = "new_user"
			mock_update.return_value = updated_execution
			
			# Update assignment
			result = await workflow_service.update_task_assignment(
				sample_task_execution.id,
				"new_user",
				"admin_user"
			)
			
			assert result is not None
			assert result.assigned_to == "new_user"
			mock_update.assert_called_once()
	
	@pytest.mark.asyncio
	async def test_transfer_task(self, workflow_service, sample_task_execution):
		"""Test transferring task between users."""
		transfer_data = {
			"from_user": "old_user",
			"to_user": "new_user",
			"reason": "Workload balancing",
			"notes": "Transfer due to availability"
		}
		
		with patch.object(workflow_service.repositories['task_execution'], 'get_by_id') as mock_get, \
			 patch.object(workflow_service.repositories['task_execution'], 'update') as mock_update, \
			 patch.object(workflow_service, '_log_task_transfer') as mock_log:
			
			# Setup current task state
			current_task = sample_task_execution.model_copy()
			current_task.assigned_to = "old_user"
			mock_get.return_value = current_task
			
			# Setup updated task state
			updated_task = current_task.model_copy()
			updated_task.assigned_to = "new_user"
			mock_update.return_value = updated_task
			
			# Transfer task
			result = await workflow_service.transfer_task(
				sample_task_execution.id,
				transfer_data,
				"admin_user"
			)
			
			assert result is not None
			assert result.assigned_to == "new_user"
			mock_get.assert_called_once()
			mock_update.assert_called_once()
			mock_log.assert_called_once()
	
	@pytest.mark.asyncio
	async def test_get_workflow_performance_analytics(self, workflow_service, sample_workflow):
		"""Test getting workflow performance analytics."""
		mock_analytics = {
			"workflow_id": sample_workflow.id,
			"total_executions": 100,
			"success_rate": 95.0,
			"average_duration_seconds": 300,
			"median_duration_seconds": 280,
			"p95_duration_seconds": 450,
			"failure_analysis": {
				"common_errors": ["timeout", "connection_error"],
				"failure_distribution": {"timeout": 3, "connection_error": 2}
			},
			"performance_trends": {
				"last_7_days": {"avg_duration": 290, "success_rate": 96.0},
				"last_30_days": {"avg_duration": 305, "success_rate": 94.5}
			}
		}
		
		with patch.object(workflow_service, '_calculate_performance_analytics') as mock_calc:
			mock_calc.return_value = mock_analytics
			
			# Get analytics
			result = await workflow_service.get_workflow_performance_analytics(
				sample_workflow.id,
				time_range_days=30
			)
			
			assert result == mock_analytics
			assert result["success_rate"] == 95.0
			assert result["total_executions"] == 100
	
	@pytest.mark.asyncio
	async def test_cross_capability_workflow_execution(self, workflow_service):
		"""Test cross-capability workflow execution."""
		# Create workflow with cross-capability tasks
		cross_capability_tasks = [
			TaskDefinition(
				id="auth_task",
				name="User Authentication",
				task_type=TaskType.INTEGRATION,
				configuration={
					"capability": "auth_rbac",
					"action": "authenticate_user",
					"parameters": {"user_id": "test_user"}
				}
			),
			TaskDefinition(
				id="audit_task",
				name="Log Activity",
				task_type=TaskType.INTEGRATION,
				configuration={
					"capability": "audit_compliance",
					"action": "log_activity",
					"parameters": {"activity": "workflow_execution"}
				},
				dependencies=["auth_task"]
			),
			TaskDefinition(
				id="notification_task",
				name="Send Notification",
				task_type=TaskType.NOTIFICATION,
				configuration={
					"capability": "notification_engine",
					"channel": "email",
					"template": "workflow_complete"
				},
				dependencies=["audit_task"]
			)
		]
		
		cross_capability_workflow = Workflow(
			name="Cross-Capability Workflow",
			description="Workflow integrating multiple APG capabilities",
			tasks=cross_capability_tasks,
			tenant_id="test_tenant",
			created_by="test_user",
			updated_by="test_user",
			tags=["cross_capability", "integration"]
		)
		
		with patch.object(workflow_service.executor, 'execute_workflow') as mock_execute:
			mock_instance = WorkflowInstance(
				workflow_id=cross_capability_workflow.id,
				input_data={"user_context": "test"},
				status=WorkflowStatus.RUNNING,
				tenant_id="test_tenant",
				started_by="test_user"
			)
			mock_execute.return_value = mock_instance
			
			# Execute cross-capability workflow
			result = await workflow_service.execute_workflow(
				workflow_id=cross_capability_workflow.id,
				input_data={"user_context": "test"},
				user_id="test_user"
			)
			
			assert result is not None
			assert result.workflow_id == cross_capability_workflow.id
			mock_execute.assert_called_once()

class TestServiceErrorHandling:
	"""Test service error handling and edge cases."""
	
	@pytest.mark.asyncio
	async def test_execute_nonexistent_workflow(self, workflow_service):
		"""Test executing non-existent workflow."""
		with patch.object(workflow_service.executor, 'execute_workflow') as mock_execute:
			mock_execute.side_effect = ValueError("Workflow not found")
			
			with pytest.raises(ValueError, match="Workflow not found"):
				await workflow_service.execute_workflow(
					workflow_id="nonexistent_id",
					input_data={},
					user_id="test_user"
				)
	
	@pytest.mark.asyncio
	async def test_pause_completed_instance(self, workflow_service, sample_workflow_instance):
		"""Test pausing already completed instance."""
		completed_instance = sample_workflow_instance.model_copy()
		completed_instance.status = WorkflowStatus.COMPLETED
		
		with patch.object(workflow_service.repositories['workflow_instance'], 'get_by_id') as mock_get:
			mock_get.return_value = completed_instance
			
			# Try to pause completed instance
			result = await workflow_service.pause_workflow_instance(
				sample_workflow_instance.id,
				"test_user"
			)
			
			assert result is False  # Should not allow pausing completed instance
	
	@pytest.mark.asyncio
	async def test_resume_non_paused_instance(self, workflow_service, sample_workflow_instance):
		"""Test resuming non-paused instance."""
		running_instance = sample_workflow_instance.model_copy()
		running_instance.status = WorkflowStatus.RUNNING
		
		with patch.object(workflow_service.repositories['workflow_instance'], 'get_by_id') as mock_get:
			mock_get.return_value = running_instance
			
			# Try to resume running instance
			result = await workflow_service.resume_workflow_instance(
				sample_workflow_instance.id,
				"test_user"
			)
			
			assert result is False  # Should not allow resuming non-paused instance
	
	@pytest.mark.asyncio
	async def test_invalid_task_transfer(self, workflow_service, sample_task_execution):
		"""Test invalid task transfer scenarios."""
		# Test transferring non-existent task
		with patch.object(workflow_service.repositories['task_execution'], 'get_by_id') as mock_get:
			mock_get.return_value = None
			
			transfer_data = {
				"from_user": "old_user",
				"to_user": "new_user",
				"reason": "Test"
			}
			
			with pytest.raises(ValueError, match="Task not found"):
				await workflow_service.transfer_task(
					"nonexistent_task_id",
					transfer_data,
					"admin_user"
				)
	
	@pytest.mark.asyncio
	async def test_service_health_check(self, workflow_service):
		"""Test service health check functionality."""
		with patch.object(workflow_service.database_manager, 'get_session') as mock_session, \
			 patch.object(workflow_service.redis_client, 'ping') as mock_ping:
			
			# Mock successful health check
			mock_session.return_value.__aenter__ = AsyncMock()
			mock_session.return_value.__aexit__ = AsyncMock()
			mock_ping.return_value = True
			
			health_status = await workflow_service.get_health_status()
			
			assert health_status["status"] == "healthy"
			assert health_status["database"] == "connected"
			assert health_status["redis"] == "connected"
			assert "timestamp" in health_status

class TestServicePerformance:
	"""Test service performance characteristics."""
	
	@pytest.mark.asyncio
	async def test_concurrent_workflow_executions(self, workflow_service, sample_workflow):
		"""Test concurrent workflow executions."""
		# Setup mock for multiple concurrent executions
		with patch.object(workflow_service.executor, 'execute_workflow') as mock_execute:
			mock_instances = [
				WorkflowInstance(
					workflow_id=sample_workflow.id,
					input_data={"test": f"data_{i}"},
					status=WorkflowStatus.RUNNING,
					tenant_id="test_tenant",
					started_by="test_user"
				)
				for i in range(5)
			]
			mock_execute.side_effect = mock_instances
			
			# Execute workflows concurrently
			async def execute_workflow(index: int):
				return await workflow_service.execute_workflow(
					workflow_id=sample_workflow.id,
					input_data={"test": f"data_{index}"},
					user_id="test_user"
				)
			
			tasks = [execute_workflow(i) for i in range(5)]
			results = await asyncio.gather(*tasks)
			
			assert len(results) == 5
			assert all(result.status == WorkflowStatus.RUNNING for result in results)
			assert mock_execute.call_count == 5
	
	@pytest.mark.asyncio
	async def test_large_workflow_handling(self, workflow_service):
		"""Test handling of workflows with many tasks."""
		# Create workflow with many tasks
		large_task_list = [
			TaskDefinition(
				id=f"task_{i}",
				name=f"Task {i}",
				task_type=TaskType.TASK,
				configuration={"step": i},
				dependencies=[f"task_{i-1}"] if i > 0 else []
			)
			for i in range(100)  # 100 tasks
		]
		
		large_workflow = Workflow(
			name="Large Workflow",
			description="Workflow with many tasks",
			tasks=large_task_list,
			tenant_id="test_tenant",
			created_by="test_user",
			updated_by="test_user"
		)
		
		with patch.object(workflow_service.executor, 'execute_workflow') as mock_execute:
			mock_instance = WorkflowInstance(
				workflow_id=large_workflow.id,
				input_data={},
				status=WorkflowStatus.RUNNING,
				tenant_id="test_tenant",
				started_by="test_user"
			)
			mock_execute.return_value = mock_instance
			
			# Execute large workflow
			result = await workflow_service.execute_workflow(
				workflow_id=large_workflow.id,
				input_data={},
				user_id="test_user"
			)
			
			assert result is not None
			assert result.workflow_id == large_workflow.id
			mock_execute.assert_called_once()
	
	@pytest.mark.asyncio
	async def test_memory_usage_optimization(self, workflow_service, sample_workflow):
		"""Test memory usage optimization during operations."""
		# Test that service properly manages memory during bulk operations
		with patch.object(workflow_service.repositories['workflow_instance'], 'get_by_workflow_id') as mock_get:
			# Mock large number of instances
			mock_instances = [
				WorkflowInstance(
					workflow_id=sample_workflow.id,
					input_data={"large_data": "x" * 1000},  # 1KB per instance
					status=WorkflowStatus.COMPLETED,
					tenant_id="test_tenant",
					started_by="test_user"
				)
				for _ in range(1000)  # 1000 instances
			]
			mock_get.return_value = mock_instances
			
			# Get instances with pagination
			result = await workflow_service.get_workflow_instances(
				sample_workflow.id,
				limit=50,  # Paginated to reduce memory usage
				offset=0
			)
			
			# Should limit results to avoid memory issues
			assert len(result) <= 50
			mock_get.assert_called_once()

class TestServiceIntegration:
	"""Test service integration with other components."""
	
	@pytest.mark.asyncio
	async def test_apg_capability_integration(self, workflow_service):
		"""Test integration with other APG capabilities."""
		# Mock APG capability calls
		with patch('requests.post') as mock_post:
			mock_post.return_value.status_code = 200
			mock_post.return_value.json.return_value = {"status": "success"}
			
			# Test auth_rbac integration
			auth_result = await workflow_service._call_apg_capability(
				"auth_rbac",
				"validate_permissions",
				{"user_id": "test_user", "resource": "workflow"}
			)
			
			assert auth_result["status"] == "success"
			mock_post.assert_called_once()
	
	@pytest.mark.asyncio
	async def test_audit_logging_integration(self, workflow_service, sample_workflow_instance):
		"""Test audit logging integration."""
		with patch.object(workflow_service, '_log_audit_event') as mock_audit:
			# Execute workflow to trigger audit logging
			with patch.object(workflow_service.executor, 'execute_workflow') as mock_execute:
				mock_execute.return_value = sample_workflow_instance
				
				await workflow_service.execute_workflow(
					workflow_id=sample_workflow_instance.workflow_id,
					input_data={"test": "data"},
					user_id="test_user"
				)
				
				# Verify audit logging was called
				mock_audit.assert_called()
	
	@pytest.mark.asyncio
	async def test_notification_integration(self, workflow_service, sample_workflow_instance):
		"""Test notification integration."""
		with patch.object(workflow_service, '_send_notification') as mock_notify:
			# Complete workflow instance to trigger notification
			with patch.object(workflow_service.repositories['workflow_instance'], 'update_status') as mock_update:
				completed_instance = sample_workflow_instance.model_copy()
				completed_instance.status = WorkflowStatus.COMPLETED
				mock_update.return_value = completed_instance
				
				await workflow_service._handle_instance_completion(sample_workflow_instance.id)
				
				# Verify notification was sent
				mock_notify.assert_called()
	
	@pytest.mark.asyncio
	async def test_real_time_collaboration_integration(self, workflow_service, sample_workflow_instance):
		"""Test real-time collaboration integration."""
		with patch.object(workflow_service, '_broadcast_instance_update') as mock_broadcast:
			# Update instance to trigger real-time broadcast
			await workflow_service._update_instance_status(
				sample_workflow_instance.id,
				WorkflowStatus.RUNNING,
				"test_user"
			)
			
			# Verify real-time update was broadcast
			mock_broadcast.assert_called_with(
				sample_workflow_instance.id,
				{"status": "running", "updated_by": "test_user"}
			)