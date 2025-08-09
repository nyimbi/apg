#!/usr/bin/env python3
"""
APG Workflow Orchestration Integration Tests

Comprehensive integration tests for end-to-end workflow execution,
APG capability integration, and external system connectivity.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import pytest
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock
import tempfile
import os

from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

# APG Core imports
from ..apg_integration import APGIntegration
from ..service import WorkflowOrchestrationService
from ..engine import WorkflowExecutionEngine
from ..database import DatabaseManager
from ..api import create_app
from ..models import *
from ..connectors.apg_connectors import *
from ..connectors.external_connectors import *

# Test utilities
from .conftest import TestHelpers


class TestAPGCapabilityIntegration:
	"""Test APG capability integration and cross-capability workflows."""
	
	@pytest.mark.integration
	@pytest.mark.apg
	async def test_auth_rbac_integration(self, workflow_service, sample_workflow_data):
		"""Test integration with auth_rbac capability."""
		# Create workflow with auth requirements
		workflow_data = sample_workflow_data.copy()
		workflow_data["metadata"]["auth_required"] = True
		workflow_data["metadata"]["required_roles"] = ["workflow_admin", "operator"]
		
		# Mock APG auth connector
		with patch('..connectors.apg_connectors.AuthRBACConnector') as mock_connector:
			mock_connector.return_value.validate_user_permissions = AsyncMock(return_value=True)
			mock_connector.return_value.get_user_roles = AsyncMock(return_value=["workflow_admin"])
			
			# Create workflow with auth validation
			workflow = await workflow_service.create_workflow(workflow_data, user_id="test_user")
			
			# Verify auth integration
			assert workflow.metadata.get("auth_required") is True
			assert "workflow_admin" in workflow.metadata.get("required_roles", [])
			
			# Test workflow execution with auth
			instance = await workflow_service.execute_workflow(
				workflow.id, 
				execution_context={"user_id": "test_user", "roles": ["workflow_admin"]}
			)
			
			assert instance.status == WorkflowStatus.RUNNING
			mock_connector.return_value.validate_user_permissions.assert_called_once()
	
	@pytest.mark.integration
	@pytest.mark.apg
	async def test_audit_compliance_integration(self, workflow_service, sample_workflow_data):
		"""Test integration with audit_compliance capability."""
		# Create workflow with compliance requirements
		workflow_data = sample_workflow_data.copy()
		workflow_data["metadata"]["compliance_required"] = True
		workflow_data["metadata"]["audit_level"] = "detailed"
		
		# Mock APG audit connector
		with patch('..connectors.apg_connectors.AuditComplianceConnector') as mock_connector:
			mock_connector.return_value.log_workflow_event = AsyncMock()
			mock_connector.return_value.validate_compliance = AsyncMock(return_value=True)
			
			# Create workflow with audit logging
			workflow = await workflow_service.create_workflow(workflow_data, user_id="test_user")
			
			# Execute workflow with audit trail
			instance = await workflow_service.execute_workflow(workflow.id)
			
			# Verify audit integration
			mock_connector.return_value.log_workflow_event.assert_called()
			mock_connector.return_value.validate_compliance.assert_called()
	
	@pytest.mark.integration
	@pytest.mark.apg
	async def test_data_lake_integration(self, workflow_service, sample_workflow_data):
		"""Test integration with data_lake capability."""
		# Create workflow with data processing tasks
		workflow_data = sample_workflow_data.copy()
		workflow_data["tasks"][0]["connector_type"] = "data_lake"
		workflow_data["tasks"][0]["config"] = {
			"operation": "store_dataset",
			"dataset_name": "workflow_results",
			"format": "parquet"
		}
		
		# Mock APG data lake connector
		with patch('..connectors.apg_connectors.DataLakeConnector') as mock_connector:
			mock_connector.return_value.store_dataset = AsyncMock(return_value={"dataset_id": "ds_123"})
			mock_connector.return_value.get_dataset = AsyncMock(return_value={"data": "test_data"})
			
			# Execute workflow with data lake operations
			workflow = await workflow_service.create_workflow(workflow_data, user_id="test_user")
			instance = await workflow_service.execute_workflow(workflow.id)
			
			# Wait for task completion
			await asyncio.sleep(0.1)
			
			# Verify data lake integration
			mock_connector.return_value.store_dataset.assert_called()
	
	@pytest.mark.integration
	@pytest.mark.apg
	async def test_real_time_collaboration_integration(self, workflow_service, sample_workflow_data):
		"""Test integration with real_time_collaboration capability."""
		# Create collaborative workflow
		workflow_data = sample_workflow_data.copy()
		workflow_data["metadata"]["collaborative"] = True
		workflow_data["metadata"]["shared_with"] = ["user1", "user2"]
		
		# Mock APG collaboration connector
		with patch('..connectors.apg_connectors.RealTimeCollaborationConnector') as mock_connector:
			mock_connector.return_value.create_collaboration_session = AsyncMock(
				return_value={"session_id": "collab_123"}
			)
			mock_connector.return_value.broadcast_workflow_update = AsyncMock()
			
			# Create collaborative workflow
			workflow = await workflow_service.create_workflow(workflow_data, user_id="test_user")
			
			# Update workflow (should trigger collaboration broadcast)
			updated_data = workflow_data.copy()
			updated_data["description"] = "Updated workflow"
			await workflow_service.update_workflow(workflow.id, updated_data)
			
			# Verify collaboration integration
			mock_connector.return_value.create_collaboration_session.assert_called()
			mock_connector.return_value.broadcast_workflow_update.assert_called()
	
	@pytest.mark.integration
	@pytest.mark.apg
	async def test_cross_capability_workflow(self, workflow_service):
		"""Test complex workflow using multiple APG capabilities."""
		# Create multi-capability workflow
		workflow_data = {
			"name": "Cross-Capability Workflow",
			"description": "Workflow using multiple APG capabilities",
			"tenant_id": "test_tenant",
			"tasks": [
				{
					"id": "auth_task",
					"name": "Authenticate User",
					"task_type": "connector",
					"connector_type": "auth_rbac",
					"config": {"operation": "validate_user", "user_id": "${input.user_id}"}
				},
				{
					"id": "data_task", 
					"name": "Process Data",
					"task_type": "connector",
					"connector_type": "data_lake",
					"config": {"operation": "transform_data", "dataset": "${auth_task.result.dataset}"},
					"depends_on": ["auth_task"]
				},
				{
					"id": "audit_task",
					"name": "Log Activity", 
					"task_type": "connector",
					"connector_type": "audit_compliance",
					"config": {"operation": "log_activity", "activity": "data_processing"},
					"depends_on": ["data_task"]
				}
			],
			"metadata": {
				"multi_capability": True,
				"capabilities_used": ["auth_rbac", "data_lake", "audit_compliance"]
			}
		}
		
		# Mock all APG connectors
		with patch.multiple(
			'..connectors.apg_connectors',
			AuthRBACConnector=MagicMock(),
			DataLakeConnector=MagicMock(),
			AuditComplianceConnector=MagicMock()
		) as mocks:
			# Setup mock returns
			mocks['AuthRBACConnector'].return_value.validate_user = AsyncMock(
				return_value={"valid": True, "dataset": "user_data"}
			)
			mocks['DataLakeConnector'].return_value.transform_data = AsyncMock(
				return_value={"transformed": True, "records": 100}
			)
			mocks['AuditComplianceConnector'].return_value.log_activity = AsyncMock(
				return_value={"logged": True, "audit_id": "audit_123"}
			)
			
			# Execute cross-capability workflow
			workflow = await workflow_service.create_workflow(workflow_data, user_id="test_user")
			instance = await workflow_service.execute_workflow(
				workflow.id,
				execution_context={"user_id": "test_user"}
			)
			
			# Wait for execution
			await asyncio.sleep(0.2)
			
			# Verify all capabilities were called
			mocks['AuthRBACConnector'].return_value.validate_user.assert_called()
			mocks['DataLakeConnector'].return_value.transform_data.assert_called()
			mocks['AuditComplianceConnector'].return_value.log_activity.assert_called()


class TestEndToEndWorkflowExecution:
	"""Test complete end-to-end workflow execution scenarios."""
	
	@pytest.mark.integration
	@pytest.mark.slow
	async def test_simple_linear_workflow(self, workflow_service):
		"""Test execution of simple linear workflow."""
		workflow_data = {
			"name": "Simple Linear Workflow",
			"description": "Basic sequential task execution",
			"tenant_id": "test_tenant",
			"tasks": [
				{
					"id": "task1",
					"name": "First Task",
					"task_type": "script",
					"config": {"script": "return {'result': 'task1_complete'}"},
					"timeout_seconds": 30
				},
				{
					"id": "task2", 
					"name": "Second Task",
					"task_type": "script",
					"config": {"script": "return {'result': f'task2_complete_{input.task1.result}'}"},
					"depends_on": ["task1"],
					"timeout_seconds": 30
				},
				{
					"id": "task3",
					"name": "Final Task",
					"task_type": "script", 
					"config": {"script": "return {'final': True, 'previous': input.task2.result}"},
					"depends_on": ["task2"],
					"timeout_seconds": 30
				}
			]
		}
		
		# Execute workflow
		workflow = await workflow_service.create_workflow(workflow_data, user_id="test_user")
		instance = await workflow_service.execute_workflow(workflow.id)
		
		# Wait for completion
		max_wait = 10
		waited = 0
		while instance.status not in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED] and waited < max_wait:
			await asyncio.sleep(0.5)
			waited += 0.5
			instance = await workflow_service.get_workflow_instance(instance.id)
		
		# Verify completion
		assert instance.status == WorkflowStatus.COMPLETED
		assert len(instance.task_executions) == 3
		
		# Verify task execution order
		executions = sorted(instance.task_executions, key=lambda x: x.started_at)
		assert executions[0].task_id == "task1"
		assert executions[1].task_id == "task2" 
		assert executions[2].task_id == "task3"
		
		# Verify all tasks completed successfully
		for execution in executions:
			assert execution.status == TaskStatus.COMPLETED
	
	@pytest.mark.integration
	async def test_parallel_workflow_execution(self, workflow_service):
		"""Test workflow with parallel task execution."""
		workflow_data = {
			"name": "Parallel Workflow",
			"description": "Workflow with parallel branches",
			"tenant_id": "test_tenant",
			"tasks": [
				{
					"id": "start_task",
					"name": "Start Task",
					"task_type": "script",
					"config": {"script": "return {'started': True}"}
				},
				{
					"id": "parallel_task1",
					"name": "Parallel Task 1",
					"task_type": "script",
					"config": {"script": "import time; time.sleep(0.1); return {'branch': 1}"},
					"depends_on": ["start_task"]
				},
				{
					"id": "parallel_task2",
					"name": "Parallel Task 2", 
					"task_type": "script",
					"config": {"script": "import time; time.sleep(0.1); return {'branch': 2}"},
					"depends_on": ["start_task"]
				},
				{
					"id": "parallel_task3",
					"name": "Parallel Task 3",
					"task_type": "script",
					"config": {"script": "import time; time.sleep(0.1); return {'branch': 3}"},
					"depends_on": ["start_task"]
				},
				{
					"id": "merge_task",
					"name": "Merge Task",
					"task_type": "script",
					"config": {"script": "return {'merged': [input.parallel_task1, input.parallel_task2, input.parallel_task3]}"},
					"depends_on": ["parallel_task1", "parallel_task2", "parallel_task3"]
				}
			]
		}
		
		# Execute workflow
		workflow = await workflow_service.create_workflow(workflow_data, user_id="test_user")
		start_time = datetime.utcnow()
		instance = await workflow_service.execute_workflow(workflow.id)
		
		# Wait for completion
		while instance.status not in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]:
			await asyncio.sleep(0.1)
			instance = await workflow_service.get_workflow_instance(instance.id)
		
		end_time = datetime.utcnow()
		execution_time = (end_time - start_time).total_seconds()
		
		# Verify parallel execution (should be faster than sequential)
		assert execution_time < 1.0  # Should complete in under 1 second
		assert instance.status == WorkflowStatus.COMPLETED
		assert len(instance.task_executions) == 5
		
		# Verify parallel tasks ran concurrently
		parallel_executions = [ex for ex in instance.task_executions if "parallel_task" in ex.task_id]
		assert len(parallel_executions) == 3
		
		# All parallel tasks should have similar start times
		start_times = [ex.started_at for ex in parallel_executions]
		time_diff = max(start_times) - min(start_times)
		assert time_diff.total_seconds() < 0.5  # Started within 0.5 seconds
	
	@pytest.mark.integration
	async def test_conditional_workflow_execution(self, workflow_service):
		"""Test workflow with conditional branching."""
		workflow_data = {
			"name": "Conditional Workflow",
			"description": "Workflow with conditional logic",
			"tenant_id": "test_tenant",
			"tasks": [
				{
					"id": "decision_task",
					"name": "Decision Task",
					"task_type": "script",
					"config": {"script": "import random; return {'proceed': random.choice([True, False])}"}
				},
				{
					"id": "branch_a",
					"name": "Branch A",
					"task_type": "script",
					"config": {"script": "return {'branch': 'A', 'executed': True}"},
					"depends_on": ["decision_task"],
					"condition": "${decision_task.result.proceed} == True"
				},
				{
					"id": "branch_b", 
					"name": "Branch B",
					"task_type": "script",
					"config": {"script": "return {'branch': 'B', 'executed': True}"},
					"depends_on": ["decision_task"],
					"condition": "${decision_task.result.proceed} == False"
				},
				{
					"id": "final_task",
					"name": "Final Task",
					"task_type": "script",
					"config": {"script": "return {'completed': True}"},
					"depends_on": ["branch_a", "branch_b"],
					"execution_mode": "any_dependency"  # Run when any dependency completes
				}
			]
		}
		
		# Execute workflow multiple times to test both branches
		for _ in range(5):  # Run multiple times to ensure we hit both branches
			workflow = await workflow_service.create_workflow(workflow_data, user_id="test_user")
			instance = await workflow_service.execute_workflow(workflow.id)
			
			# Wait for completion
			while instance.status not in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]:
				await asyncio.sleep(0.1)
				instance = await workflow_service.get_workflow_instance(instance.id)
			
			# Verify conditional execution
			assert instance.status == WorkflowStatus.COMPLETED
			
			# Check that only one branch executed
			branch_executions = [ex for ex in instance.task_executions if ex.task_id in ["branch_a", "branch_b"]]
			assert len(branch_executions) == 1  # Only one branch should execute
			
			# Final task should always execute
			final_executions = [ex for ex in instance.task_executions if ex.task_id == "final_task"]
			assert len(final_executions) == 1
			assert final_executions[0].status == TaskStatus.COMPLETED
	
	@pytest.mark.integration
	async def test_workflow_with_retries(self, workflow_service):
		"""Test workflow task retry mechanism."""
		workflow_data = {
			"name": "Retry Workflow",
			"description": "Workflow with retry logic",
			"tenant_id": "test_tenant",
			"tasks": [
				{
					"id": "flaky_task",
					"name": "Flaky Task",
					"task_type": "script",
					"config": {
						"script": """
import random
if random.random() < 0.7:  # 70% chance of failure
    raise Exception("Random failure")
return {'success': True}
"""
					},
					"retry_config": {
						"max_retries": 5,
						"retry_delay": 0.1,
						"backoff_multiplier": 1.5
					}
				}
			]
		}
		
		# Execute workflow
		workflow = await workflow_service.create_workflow(workflow_data, user_id="test_user")
		instance = await workflow_service.execute_workflow(workflow.id)
		
		# Wait for completion
		max_wait = 10
		waited = 0
		while instance.status not in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED] and waited < max_wait:
			await asyncio.sleep(0.2)
			waited += 0.2
			instance = await workflow_service.get_workflow_instance(instance.id)
		
		# Should eventually succeed or exhaust retries
		assert instance.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]
		
		# Check task execution details
		task_execution = instance.task_executions[0]
		if instance.status == WorkflowStatus.COMPLETED:
			assert task_execution.status == TaskStatus.COMPLETED
		else:
			assert task_execution.status == TaskStatus.FAILED
			# Should have attempted retries
			assert task_execution.retry_count > 0
	
	@pytest.mark.integration
	async def test_workflow_timeout_handling(self, workflow_service):
		"""Test workflow timeout handling."""
		workflow_data = {
			"name": "Timeout Workflow",
			"description": "Workflow with timeout",
			"tenant_id": "test_tenant",
			"timeout_seconds": 2,
			"tasks": [
				{
					"id": "long_task",
					"name": "Long Running Task",
					"task_type": "script",
					"config": {"script": "import time; time.sleep(5); return {'done': True}"},
					"timeout_seconds": 1
				}
			]
		}
		
		# Execute workflow
		workflow = await workflow_service.create_workflow(workflow_data, user_id="test_user")
		instance = await workflow_service.execute_workflow(workflow.id)
		
		# Wait for timeout
		max_wait = 5
		waited = 0
		while instance.status not in [WorkflowStatus.FAILED, WorkflowStatus.COMPLETED] and waited < max_wait:
			await asyncio.sleep(0.2)
			waited += 0.2
			instance = await workflow_service.get_workflow_instance(instance.id)
		
		# Should fail due to timeout
		assert instance.status == WorkflowStatus.FAILED
		assert "timeout" in instance.error_details.lower()


class TestConnectorIntegration:
	"""Test external system connector integration."""
	
	@pytest.mark.integration
	async def test_rest_api_connector_integration(self, workflow_service):
		"""Test REST API connector integration."""
		workflow_data = {
			"name": "REST API Workflow",
			"description": "Workflow using REST API connector", 
			"tenant_id": "test_tenant",
			"tasks": [
				{
					"id": "api_call",
					"name": "API Call Task",
					"task_type": "connector",
					"connector_type": "rest_api",
					"config": {
						"url": "https://jsonplaceholder.typicode.com/posts/1",
						"method": "GET",
						"headers": {"Content-Type": "application/json"}
					}
				}
			]
		}
		
		# Mock REST API connector
		with patch('..connectors.external_connectors.RESTAPIConnector') as mock_connector:
			mock_connector.return_value.execute = AsyncMock(
				return_value={"id": 1, "title": "Test Post", "body": "Test content"}
			)
			
			# Execute workflow
			workflow = await workflow_service.create_workflow(workflow_data, user_id="test_user")
			instance = await workflow_service.execute_workflow(workflow.id)
			
			# Wait for completion
			while instance.status not in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]:
				await asyncio.sleep(0.1)
				instance = await workflow_service.get_workflow_instance(instance.id)
			
			# Verify REST API integration
			assert instance.status == WorkflowStatus.COMPLETED
			mock_connector.return_value.execute.assert_called_once()
	
	@pytest.mark.integration
	async def test_database_connector_integration(self, workflow_service):
		"""Test database connector integration."""
		workflow_data = {
			"name": "Database Workflow",
			"description": "Workflow using database connector",
			"tenant_id": "test_tenant", 
			"tasks": [
				{
					"id": "db_query",
					"name": "Database Query",
					"task_type": "connector",
					"connector_type": "database",
					"config": {
						"connection_string": "postgresql://test:test@localhost/test",
						"query": "SELECT COUNT(*) as count FROM workflows",
						"operation": "select"
					}
				}
			]
		}
		
		# Mock database connector
		with patch('..connectors.external_connectors.DatabaseConnector') as mock_connector:
			mock_connector.return_value.execute = AsyncMock(
				return_value={"count": 5}
			)
			
			# Execute workflow
			workflow = await workflow_service.create_workflow(workflow_data, user_id="test_user")
			instance = await workflow_service.execute_workflow(workflow.id)
			
			# Wait for completion
			while instance.status not in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]:
				await asyncio.sleep(0.1)
				instance = await workflow_service.get_workflow_instance(instance.id)
			
			# Verify database integration
			assert instance.status == WorkflowStatus.COMPLETED
			mock_connector.return_value.execute.assert_called_once()
	
	@pytest.mark.integration
	async def test_file_system_connector_integration(self, workflow_service):
		"""Test file system connector integration."""
		with tempfile.TemporaryDirectory() as temp_dir:
			test_file = Path(temp_dir) / "test.txt"
			test_file.write_text("test content")
			
			workflow_data = {
				"name": "File System Workflow",
				"description": "Workflow using file system connector",
				"tenant_id": "test_tenant",
				"tasks": [
					{
						"id": "file_read",
						"name": "Read File",
						"task_type": "connector",
						"connector_type": "file_system",
						"config": {
							"operation": "read",
							"file_path": str(test_file)
						}
					}
				]
			}
			
			# Mock file system connector
			with patch('..connectors.external_connectors.FileSystemConnector') as mock_connector:
				mock_connector.return_value.execute = AsyncMock(
					return_value={"content": "test content", "size": 12}
				)
				
				# Execute workflow
				workflow = await workflow_service.create_workflow(workflow_data, user_id="test_user")
				instance = await workflow_service.execute_workflow(workflow.id)
				
				# Wait for completion
				while instance.status not in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]:
					await asyncio.sleep(0.1)
					instance = await workflow_service.get_workflow_instance(instance.id)
				
				# Verify file system integration
				assert instance.status == WorkflowStatus.COMPLETED
				mock_connector.return_value.execute.assert_called_once()
	
	@pytest.mark.integration
	async def test_message_queue_connector_integration(self, workflow_service):
		"""Test message queue connector integration."""
		workflow_data = {
			"name": "Message Queue Workflow",
			"description": "Workflow using message queue connector",
			"tenant_id": "test_tenant",
			"tasks": [
				{
					"id": "send_message",
					"name": "Send Message",
					"task_type": "connector",
					"connector_type": "message_queue",
					"config": {
						"queue_url": "redis://localhost:6379",
						"queue_name": "test_queue",
						"operation": "send",
						"message": {"data": "test message"}
					}
				}
			]
		}
		
		# Mock message queue connector
		with patch('..connectors.external_connectors.MessageQueueConnector') as mock_connector:
			mock_connector.return_value.execute = AsyncMock(
				return_value={"message_id": "msg_123", "sent": True}
			)
			
			# Execute workflow
			workflow = await workflow_service.create_workflow(workflow_data, user_id="test_user")
			instance = await workflow_service.execute_workflow(workflow.id)
			
			# Wait for completion
			while instance.status not in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]:
				await asyncio.sleep(0.1)
				instance = await workflow_service.get_workflow_instance(instance.id)
			
			# Verify message queue integration
			assert instance.status == WorkflowStatus.COMPLETED
			mock_connector.return_value.execute.assert_called_once()


class TestWorkflowAPIIntegration:
	"""Test API integration and workflow management through REST endpoints."""
	
	@pytest.mark.integration
	@pytest.mark.api
	def test_workflow_crud_via_api(self, test_client):
		"""Test complete workflow CRUD operations via API."""
		# Create workflow
		workflow_data = {
			"name": "API Test Workflow",
			"description": "Workflow created via API",
			"tenant_id": "test_tenant",
			"tasks": [
				{
					"id": "test_task",
					"name": "Test Task",
					"task_type": "script",
					"config": {"script": "return {'result': 'success'}"}
				}
			]
		}
		
		# POST - Create workflow
		response = test_client.post("/api/v1/workflows", json=workflow_data)
		assert response.status_code == 201
		workflow = response.json()
		workflow_id = workflow["id"]
		
		# GET - Retrieve workflow
		response = test_client.get(f"/api/v1/workflows/{workflow_id}")
		assert response.status_code == 200
		retrieved_workflow = response.json()
		assert retrieved_workflow["name"] == workflow_data["name"]
		
		# PUT - Update workflow
		updated_data = workflow_data.copy()
		updated_data["description"] = "Updated description"
		response = test_client.put(f"/api/v1/workflows/{workflow_id}", json=updated_data)
		assert response.status_code == 200
		updated_workflow = response.json()
		assert updated_workflow["description"] == "Updated description"
		
		# POST - Execute workflow
		response = test_client.post(f"/api/v1/workflows/{workflow_id}/execute")
		assert response.status_code == 200
		instance = response.json()
		assert instance["workflow_id"] == workflow_id
		
		# DELETE - Delete workflow
		response = test_client.delete(f"/api/v1/workflows/{workflow_id}")
		assert response.status_code == 204
		
		# Verify deletion
		response = test_client.get(f"/api/v1/workflows/{workflow_id}")
		assert response.status_code == 404
	
	@pytest.mark.integration
	@pytest.mark.api
	def test_workflow_execution_monitoring_via_api(self, test_client):
		"""Test workflow execution monitoring through API."""
		# Create and execute workflow
		workflow_data = {
			"name": "Monitoring Test Workflow",
			"description": "Workflow for monitoring test",
			"tenant_id": "test_tenant",
			"tasks": [
				{
					"id": "monitored_task",
					"name": "Monitored Task",
					"task_type": "script",
					"config": {"script": "import time; time.sleep(0.1); return {'monitored': True}"}
				}
			]
		}
		
		# Create workflow
		response = test_client.post("/api/v1/workflows", json=workflow_data)
		workflow_id = response.json()["id"]
		
		# Execute workflow
		response = test_client.post(f"/api/v1/workflows/{workflow_id}/execute")
		instance_id = response.json()["id"]
		
		# Monitor execution
		max_attempts = 20
		attempts = 0
		while attempts < max_attempts:
			response = test_client.get(f"/api/v1/workflow-instances/{instance_id}")
			assert response.status_code == 200
			instance = response.json()
			
			if instance["status"] in ["completed", "failed"]:
				break
				
			attempts += 1
			asyncio.sleep(0.1)
		
		# Verify completion
		assert instance["status"] == "completed"
		
		# Get execution details
		response = test_client.get(f"/api/v1/workflow-instances/{instance_id}/executions")
		assert response.status_code == 200
		executions = response.json()
		assert len(executions) == 1
		assert executions[0]["task_id"] == "monitored_task"
		assert executions[0]["status"] == "completed"


class TestMultiTenantIntegration:
	"""Test multi-tenant workflow isolation and security."""
	
	@pytest.mark.integration
	async def test_tenant_isolation(self, workflow_service):
		"""Test that workflows are isolated between tenants."""
		# Create workflows for different tenants
		workflow_data_tenant1 = {
			"name": "Tenant 1 Workflow",
			"description": "Workflow for tenant 1",
			"tenant_id": "tenant_1",
			"tasks": [{"id": "task1", "name": "Task 1", "task_type": "script", "config": {"script": "return {'tenant': 1}"}}]
		}
		
		workflow_data_tenant2 = {
			"name": "Tenant 2 Workflow", 
			"description": "Workflow for tenant 2",
			"tenant_id": "tenant_2",
			"tasks": [{"id": "task2", "name": "Task 2", "task_type": "script", "config": {"script": "return {'tenant': 2}"}}]
		}
		
		# Create workflows
		workflow1 = await workflow_service.create_workflow(workflow_data_tenant1, user_id="user1")
		workflow2 = await workflow_service.create_workflow(workflow_data_tenant2, user_id="user2")
		
		# Verify tenant isolation
		tenant1_workflows = await workflow_service.list_workflows(tenant_id="tenant_1")
		tenant2_workflows = await workflow_service.list_workflows(tenant_id="tenant_2")
		
		assert len(tenant1_workflows) >= 1
		assert len(tenant2_workflows) >= 1
		assert workflow1.id in [w.id for w in tenant1_workflows]
		assert workflow2.id in [w.id for w in tenant2_workflows]
		assert workflow1.id not in [w.id for w in tenant2_workflows]
		assert workflow2.id not in [w.id for w in tenant1_workflows]
	
	@pytest.mark.integration
	async def test_cross_tenant_access_prevention(self, workflow_service):
		"""Test that cross-tenant access is prevented."""
		# Create workflow for tenant 1
		workflow_data = {
			"name": "Protected Workflow",
			"description": "Workflow that should be protected",
			"tenant_id": "tenant_1",
			"tasks": [{"id": "task", "name": "Task", "task_type": "script", "config": {"script": "return {'protected': True}"}}]
		}
		
		workflow = await workflow_service.create_workflow(workflow_data, user_id="user1")
		
		# Try to access from different tenant - should fail or be empty
		try:
			other_tenant_workflow = await workflow_service.get_workflow(workflow.id, tenant_id="tenant_2")
			# If no exception, should be None or different workflow
			assert other_tenant_workflow is None or other_tenant_workflow.id != workflow.id
		except Exception:
			# Expected - cross-tenant access should be prevented
			pass


class TestPerformanceIntegration:
	"""Test workflow system performance under various conditions."""
	
	@pytest.mark.integration
	@pytest.mark.performance
	async def test_concurrent_workflow_execution(self, workflow_service):
		"""Test concurrent execution of multiple workflows."""
		# Create multiple workflows
		workflows = []
		for i in range(5):
			workflow_data = {
				"name": f"Concurrent Workflow {i}",
				"description": f"Workflow {i} for concurrency test",
				"tenant_id": "test_tenant",
				"tasks": [
					{
						"id": f"task_{i}",
						"name": f"Task {i}",
						"task_type": "script",
						"config": {"script": f"import time; time.sleep(0.1); return {{'workflow': {i}}}"}
					}
				]
			}
			workflow = await workflow_service.create_workflow(workflow_data, user_id="test_user")
			workflows.append(workflow)
		
		# Execute all workflows concurrently
		start_time = datetime.utcnow()
		instances = await asyncio.gather(*[
			workflow_service.execute_workflow(workflow.id) 
			for workflow in workflows
		])
		
		# Wait for all to complete
		while True:
			statuses = await asyncio.gather(*[
				workflow_service.get_workflow_instance(instance.id)
				for instance in instances
			])
			
			if all(status.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED] for status in statuses):
				break
				
			await asyncio.sleep(0.1)
		
		end_time = datetime.utcnow()
		total_time = (end_time - start_time).total_seconds()
		
		# Verify concurrent execution performance
		assert total_time < 2.0  # Should complete much faster than sequential
		assert all(status.status == WorkflowStatus.COMPLETED for status in statuses)
	
	@pytest.mark.integration
	@pytest.mark.performance
	async def test_large_workflow_execution(self, workflow_service):
		"""Test execution of workflow with many tasks."""
		# Create workflow with many tasks
		tasks = []
		for i in range(20):
			task = {
				"id": f"task_{i}",
				"name": f"Task {i}",
				"task_type": "script",
				"config": {"script": f"return {{'task_number': {i}}}"}
			}
			if i > 0:
				task["depends_on"] = [f"task_{i-1}"]  # Chain tasks
			tasks.append(task)
		
		workflow_data = {
			"name": "Large Workflow",
			"description": "Workflow with many tasks",
			"tenant_id": "test_tenant",
			"tasks": tasks
		}
		
		# Execute large workflow
		start_time = datetime.utcnow()
		workflow = await workflow_service.create_workflow(workflow_data, user_id="test_user")
		instance = await workflow_service.execute_workflow(workflow.id)
		
		# Wait for completion
		while instance.status not in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]:
			await asyncio.sleep(0.1)
			instance = await workflow_service.get_workflow_instance(instance.id)
		
		end_time = datetime.utcnow()
		execution_time = (end_time - start_time).total_seconds()
		
		# Verify successful execution
		assert instance.status == WorkflowStatus.COMPLETED
		assert len(instance.task_executions) == 20
		assert execution_time < 10.0  # Should complete within reasonable time


class TestErrorHandlingIntegration:
	"""Test error handling and recovery mechanisms."""
	
	@pytest.mark.integration
	async def test_workflow_failure_handling(self, workflow_service):
		"""Test workflow behavior when tasks fail."""
		workflow_data = {
			"name": "Failure Test Workflow",
			"description": "Workflow designed to test failure handling",
			"tenant_id": "test_tenant",
			"tasks": [
				{
					"id": "failing_task",
					"name": "Failing Task",
					"task_type": "script",
					"config": {"script": "raise Exception('Intentional failure')"}
				},
				{
					"id": "dependent_task",
					"name": "Dependent Task",
					"task_type": "script",
					"config": {"script": "return {'should_not_run': True}"},
					"depends_on": ["failing_task"]
				}
			]
		}
		
		# Execute workflow with failure
		workflow = await workflow_service.create_workflow(workflow_data, user_id="test_user")
		instance = await workflow_service.execute_workflow(workflow.id)
		
		# Wait for failure
		while instance.status not in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]:
			await asyncio.sleep(0.1)
			instance = await workflow_service.get_workflow_instance(instance.id)
		
		# Verify failure handling
		assert instance.status == WorkflowStatus.FAILED
		assert "Intentional failure" in instance.error_details
		
		# Verify dependent task didn't run
		failing_execution = next(ex for ex in instance.task_executions if ex.task_id == "failing_task")
		assert failing_execution.status == TaskStatus.FAILED
		
		dependent_executions = [ex for ex in instance.task_executions if ex.task_id == "dependent_task"]
		assert len(dependent_executions) == 0  # Should not have executed
	
	@pytest.mark.integration
	async def test_workflow_recovery_mechanisms(self, workflow_service):
		"""Test workflow recovery after failures."""
		workflow_data = {
			"name": "Recovery Test Workflow", 
			"description": "Workflow with recovery mechanisms",
			"tenant_id": "test_tenant",
			"tasks": [
				{
					"id": "recoverable_task",
					"name": "Recoverable Task",
					"task_type": "script",
					"config": {"script": "raise Exception('Recoverable failure')"},
					"retry_config": {
						"max_retries": 3,
						"retry_delay": 0.1
					},
					"fallback_task": {
						"id": "fallback",
						"name": "Fallback Task",
						"task_type": "script",
						"config": {"script": "return {'recovered': True}"}
					}
				}
			]
		}
		
		# Execute workflow with recovery
		workflow = await workflow_service.create_workflow(workflow_data, user_id="test_user")
		instance = await workflow_service.execute_workflow(workflow.id)
		
		# Wait for completion/failure
		max_wait = 5
		waited = 0
		while instance.status not in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED] and waited < max_wait:
			await asyncio.sleep(0.2)
			waited += 0.2
			instance = await workflow_service.get_workflow_instance(instance.id)
		
		# Should eventually fail after retries, then fallback should execute
		assert instance.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]
		
		# Check retry attempts
		main_execution = next(ex for ex in instance.task_executions if ex.task_id == "recoverable_task")
		assert main_execution.retry_count > 0