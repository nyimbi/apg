"""
APG Workflow Orchestration API Tests

Comprehensive unit tests for FastAPI REST endpoints, authentication, 
rate limiting, and API functionality.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
import json

from fastapi.testclient import TestClient
from fastapi import HTTPException, status
import httpx

from ..api import (
	create_api_app, APIResponse, WorkflowCreateRequest, 
	WorkflowUpdateRequest, WorkflowExecuteRequest, PaginationParams,
	get_current_user, require_permission, RateLimitMiddleware
)
from ..models import (
	Workflow, WorkflowInstance, TaskDefinition, TaskExecution,
	WorkflowStatus, TaskStatus, Priority, TaskType
)

class TestAPIApplication:
	"""Test API application creation and configuration."""
	
	@pytest.mark.asyncio
	async def test_create_api_app(self, database_manager, redis_client):
		"""Test API application creation."""
		app = create_api_app(database_manager, redis_client, "test_tenant")
		
		assert app is not None
		assert app.title == "APG Workflow Orchestration API"
		assert app.version == "1.0.0"
		assert hasattr(app.state, 'workflow_service')
		assert hasattr(app.state, 'workflow_manager')
	
	def test_api_app_middleware(self, database_manager, redis_client):
		"""Test API middleware configuration."""
		app = create_api_app(database_manager, redis_client, "test_tenant")
		
		# Check middleware is configured
		middleware_types = [type(middleware.cls).__name__ for middleware in app.user_middleware]
		assert 'CORSMiddleware' in middleware_types
		assert 'GZipMiddleware' in middleware_types
		assert 'RateLimitMiddleware' in middleware_types
	
	def test_api_documentation_endpoints(self, database_manager, redis_client):
		"""Test API documentation endpoints."""
		app = create_api_app(database_manager, redis_client, "test_tenant")
		client = TestClient(app)
		
		# Test OpenAPI documentation
		response = client.get("/docs")
		assert response.status_code == 200
		
		# Test ReDoc documentation
		response = client.get("/redoc")
		assert response.status_code == 200
		
		# Test OpenAPI JSON
		response = client.get("/openapi.json")
		assert response.status_code == 200
		assert response.headers["content-type"] == "application/json"

class TestHealthEndpoints:
	"""Test health check endpoints."""
	
	def test_basic_health_check(self, database_manager, redis_client):
		"""Test basic health check endpoint."""
		app = create_api_app(database_manager, redis_client, "test_tenant")
		client = TestClient(app)
		
		response = client.get("/health")
		assert response.status_code == 200
		
		data = response.json()
		assert data["success"] is True
		assert data["message"] == "API is healthy"
		assert "data" in data
		assert data["data"]["status"] == "healthy"
	
	def test_detailed_health_check(self, database_manager, redis_client):
		"""Test detailed health check endpoint."""
		app = create_api_app(database_manager, redis_client, "test_tenant")
		client = TestClient(app)
		
		# Mock authentication
		with patch('__main__.get_current_user') as mock_auth:
			mock_auth.return_value = {"user_id": "test_user", "tenant_id": "test_tenant"}
			
			response = client.get(
				"/health/detailed",
				headers={"Authorization": "Bearer test_token"}
			)
			assert response.status_code == 200
			
			data = response.json()
			assert data["success"] is True
			assert "data" in data
			assert "database" in data["data"]
			assert "redis" in data["data"]
			assert "services" in data["data"]

class TestWorkflowEndpoints:
	"""Test workflow CRUD endpoints."""
	
	def test_create_workflow_success(self, database_manager, redis_client, sample_workflow):
		"""Test successful workflow creation."""
		app = create_api_app(database_manager, redis_client, "test_tenant")
		client = TestClient(app)
		
		workflow_data = {
			"name": sample_workflow.name,
			"description": sample_workflow.description,
			"tasks": [task.model_dump() for task in sample_workflow.tasks],
			"configuration": sample_workflow.configuration,
			"priority": sample_workflow.priority.value,
			"tags": sample_workflow.tags
		}
		
		with patch('__main__.get_current_user') as mock_auth, \
			 patch.object(app.state.workflow_manager, 'create_workflow') as mock_create:
			
			mock_auth.return_value = {
				"user_id": "test_user",
				"tenant_id": "test_tenant",
				"permissions": ["workflows.write"]
			}
			mock_create.return_value = sample_workflow
			
			response = client.post(
				"/workflows",
				json=workflow_data,
				headers={"Authorization": "Bearer test_token"}
			)
			
			assert response.status_code == 200
			data = response.json()
			assert data["success"] is True
			assert data["message"] == "Workflow created successfully"
			assert "data" in data
			mock_create.assert_called_once()
	
	def test_create_workflow_validation_error(self, database_manager, redis_client):
		"""Test workflow creation with validation errors."""
		app = create_api_app(database_manager, redis_client, "test_tenant")
		client = TestClient(app)
		
		invalid_workflow_data = {
			"name": "",  # Empty name should fail validation
			"tasks": []  # Empty tasks should fail validation
		}
		
		with patch('__main__.get_current_user') as mock_auth:
			mock_auth.return_value = {
				"user_id": "test_user",
				"tenant_id": "test_tenant",
				"permissions": ["workflows.write"]
			}
			
			response = client.post(
				"/workflows",
				json=invalid_workflow_data,
				headers={"Authorization": "Bearer test_token"}
			)
			
			assert response.status_code == 422  # Validation error
	
	def test_get_workflow_success(self, database_manager, redis_client, sample_workflow):
		"""Test successful workflow retrieval."""
		app = create_api_app(database_manager, redis_client, "test_tenant")
		client = TestClient(app)
		
		with patch('__main__.get_current_user') as mock_auth, \
			 patch.object(app.state.workflow_manager, 'get_workflow') as mock_get:
			
			mock_auth.return_value = {
				"user_id": "test_user",
				"tenant_id": "test_tenant",
				"permissions": ["workflows.read"]
			}
			mock_get.return_value = sample_workflow
			
			response = client.get(
				f"/workflows/{sample_workflow.id}",
				headers={"Authorization": "Bearer test_token"}
			)
			
			assert response.status_code == 200
			data = response.json()
			assert data["success"] is True
			assert data["message"] == "Workflow retrieved successfully"
			assert data["data"]["name"] == sample_workflow.name
	
	def test_get_workflow_not_found(self, database_manager, redis_client):
		"""Test workflow retrieval when workflow doesn't exist."""
		app = create_api_app(database_manager, redis_client, "test_tenant")
		client = TestClient(app)
		
		with patch('__main__.get_current_user') as mock_auth, \
			 patch.object(app.state.workflow_manager, 'get_workflow') as mock_get:
			
			mock_auth.return_value = {
				"user_id": "test_user",
				"tenant_id": "test_tenant",
				"permissions": ["workflows.read"]
			}
			mock_get.return_value = None
			
			response = client.get(
				"/workflows/nonexistent_id",
				headers={"Authorization": "Bearer test_token"}
			)
			
			assert response.status_code == 404
	
	def test_update_workflow_success(self, database_manager, redis_client, sample_workflow):
		"""Test successful workflow update."""
		app = create_api_app(database_manager, redis_client, "test_tenant")
		client = TestClient(app)
		
		update_data = {
			"name": "Updated Workflow Name",
			"description": "Updated description",
			"priority": "high"
		}
		
		updated_workflow = sample_workflow.model_copy()
		updated_workflow.name = update_data["name"]
		updated_workflow.description = update_data["description"]
		updated_workflow.priority = Priority.HIGH
		
		with patch('__main__.get_current_user') as mock_auth, \
			 patch.object(app.state.workflow_manager, 'update_workflow') as mock_update:
			
			mock_auth.return_value = {
				"user_id": "test_user",
				"tenant_id": "test_tenant",
				"permissions": ["workflows.write"]
			}
			mock_update.return_value = updated_workflow
			
			response = client.put(
				f"/workflows/{sample_workflow.id}",
				json=update_data,
				headers={"Authorization": "Bearer test_token"}
			)
			
			assert response.status_code == 200
			data = response.json()
			assert data["success"] is True
			assert data["data"]["name"] == "Updated Workflow Name"
	
	def test_delete_workflow_success(self, database_manager, redis_client, sample_workflow):
		"""Test successful workflow deletion."""
		app = create_api_app(database_manager, redis_client, "test_tenant")
		client = TestClient(app)
		
		with patch('__main__.get_current_user') as mock_auth, \
			 patch.object(app.state.workflow_manager, 'delete_workflow') as mock_delete:
			
			mock_auth.return_value = {
				"user_id": "test_user",
				"tenant_id": "test_tenant",
				"permissions": ["workflows.delete"]
			}
			mock_delete.return_value = True
			
			response = client.delete(
				f"/workflows/{sample_workflow.id}",
				headers={"Authorization": "Bearer test_token"}
			)
			
			assert response.status_code == 200
			data = response.json()
			assert data["success"] is True
			assert data["data"]["deleted"] is True
	
	def test_search_workflows(self, database_manager, redis_client):
		"""Test workflow search functionality."""
		app = create_api_app(database_manager, redis_client, "test_tenant")
		client = TestClient(app)
		
		# Create sample workflows for search results
		sample_workflows = [
			Workflow(
				name=f"Test Workflow {i}",
				tasks=[TaskDefinition(name=f"Task {i}", task_type=TaskType.TASK)],
				tenant_id="test_tenant",
				created_by="test_user",
				updated_by="test_user",
				priority=Priority.MEDIUM if i % 2 == 0 else Priority.HIGH,
				tags=[f"tag_{i}"]
			)
			for i in range(3)
		]
		
		with patch('__main__.get_current_user') as mock_auth, \
			 patch.object(app.state.workflow_manager, 'search_workflows') as mock_search:
			
			mock_auth.return_value = {
				"user_id": "test_user",
				"tenant_id": "test_tenant",
				"permissions": ["workflows.read"]
			}
			mock_search.return_value = sample_workflows
			
			response = client.get(
				"/workflows?name_pattern=Test Workflow%&priority=high&limit=10",
				headers={"Authorization": "Bearer test_token"}
			)
			
			assert response.status_code == 200
			data = response.json()
			assert data["success"] is True
			assert len(data["data"]) == 3
			assert data["metadata"]["count"] == 3
	
	def test_clone_workflow(self, database_manager, redis_client, sample_workflow):
		"""Test workflow cloning."""
		app = create_api_app(database_manager, redis_client, "test_tenant")
		client = TestClient(app)
		
		cloned_workflow = sample_workflow.model_copy()
		cloned_workflow.id = "cloned_workflow_id"
		cloned_workflow.name = "Cloned Workflow"
		
		clone_request = {
			"new_name": "Cloned Workflow",
			"modifications": {"priority": "high"}
		}
		
		with patch('__main__.get_current_user') as mock_auth, \
			 patch.object(app.state.workflow_manager, 'clone_workflow') as mock_clone:
			
			mock_auth.return_value = {
				"user_id": "test_user",
				"tenant_id": "test_tenant",
				"permissions": ["workflows.write"]
			}
			mock_clone.return_value = cloned_workflow
			
			response = client.post(
				f"/workflows/{sample_workflow.id}/clone",
				json=clone_request,
				headers={"Authorization": "Bearer test_token"}
			)
			
			assert response.status_code == 200
			data = response.json()
			assert data["success"] is True
			assert data["data"]["name"] == "Cloned Workflow"
			assert data["metadata"]["original_workflow_id"] == sample_workflow.id
	
	def test_get_workflow_statistics(self, database_manager, redis_client, sample_workflow):
		"""Test workflow statistics retrieval."""
		app = create_api_app(database_manager, redis_client, "test_tenant")
		client = TestClient(app)
		
		mock_statistics = {
			"total_executions": 100,
			"successful_executions": 95,
			"failed_executions": 5,
			"average_duration_seconds": 300.0,
			"success_rate": 95.0,
			"active_instances": 3
		}
		
		with patch('__main__.get_current_user') as mock_auth, \
			 patch.object(app.state.workflow_manager, 'get_workflow_statistics') as mock_stats:
			
			mock_auth.return_value = {
				"user_id": "test_user",
				"tenant_id": "test_tenant",
				"permissions": ["workflows.read"]
			}
			mock_stats.return_value = mock_statistics
			
			response = client.get(
				f"/workflows/{sample_workflow.id}/statistics?time_range_days=30",
				headers={"Authorization": "Bearer test_token"}
			)
			
			assert response.status_code == 200
			data = response.json()
			assert data["success"] is True
			assert data["data"]["total_executions"] == 100
			assert data["data"]["success_rate"] == 95.0

class TestWorkflowExecutionEndpoints:
	"""Test workflow execution endpoints."""
	
	def test_execute_workflow_success(self, database_manager, redis_client, sample_workflow, sample_workflow_instance):
		"""Test successful workflow execution."""
		app = create_api_app(database_manager, redis_client, "test_tenant")
		client = TestClient(app)
		
		execute_request = {
			"input_data": {"test": "data"},
			"configuration_overrides": {"max_concurrent_tasks": 10},
			"priority": "high",
			"tags": ["execution_test"]
		}
		
		with patch('__main__.get_current_user') as mock_auth, \
			 patch.object(app.state.workflow_service, 'execute_workflow') as mock_execute:
			
			mock_auth.return_value = {
				"user_id": "test_user",
				"tenant_id": "test_tenant",
				"permissions": ["workflows.execute"]
			}
			mock_execute.return_value = sample_workflow_instance
			
			response = client.post(
				f"/workflows/{sample_workflow.id}/execute",
				json=execute_request,
				headers={"Authorization": "Bearer test_token"}
			)
			
			assert response.status_code == 200
			data = response.json()
			assert data["success"] is True
			assert data["message"] == "Workflow execution started"
			assert "instance_id" in data["metadata"]
			mock_execute.assert_called_once()
	
	def test_get_workflow_instances(self, database_manager, redis_client, sample_workflow):
		"""Test getting workflow instances."""
		app = create_api_app(database_manager, redis_client, "test_tenant")
		client = TestClient(app)
		
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
		
		with patch('__main__.get_current_user') as mock_auth, \
			 patch.object(app.state.workflow_service, 'get_workflow_instances') as mock_get_instances:
			
			mock_auth.return_value = {
				"user_id": "test_user",
				"tenant_id": "test_tenant",
				"permissions": ["workflows.read"]
			}
			mock_get_instances.return_value = mock_instances
			
			response = client.get(
				f"/workflows/{sample_workflow.id}/instances?status=completed&status=running&limit=10",
				headers={"Authorization": "Bearer test_token"}
			)
			
			assert response.status_code == 200
			data = response.json()
			assert data["success"] is True
			assert len(data["data"]) == 3
			assert data["metadata"]["count"] == 3
	
	def test_get_workflow_instance(self, database_manager, redis_client, sample_workflow, sample_workflow_instance):
		"""Test getting specific workflow instance."""
		app = create_api_app(database_manager, redis_client, "test_tenant")
		client = TestClient(app)
		
		with patch('__main__.get_current_user') as mock_auth, \
			 patch.object(app.state.workflow_service, 'get_workflow_instance') as mock_get_instance:
			
			mock_auth.return_value = {
				"user_id": "test_user",
				"tenant_id": "test_tenant",
				"permissions": ["workflows.read"]
			}
			mock_get_instance.return_value = sample_workflow_instance
			
			response = client.get(
				f"/workflows/{sample_workflow.id}/instances/{sample_workflow_instance.id}?include_tasks=true",
				headers={"Authorization": "Bearer test_token"}
			)
			
			assert response.status_code == 200
			data = response.json()
			assert data["success"] is True
			assert data["data"]["id"] == sample_workflow_instance.id
	
	def test_pause_workflow_instance(self, database_manager, redis_client, sample_workflow, sample_workflow_instance):
		"""Test pausing workflow instance."""
		app = create_api_app(database_manager, redis_client, "test_tenant")
		client = TestClient(app)
		
		with patch('__main__.get_current_user') as mock_auth, \
			 patch.object(app.state.workflow_service, 'pause_workflow_instance') as mock_pause:
			
			mock_auth.return_value = {
				"user_id": "test_user",
				"tenant_id": "test_tenant",
				"permissions": ["workflows.execute"]
			}
			mock_pause.return_value = True
			
			response = client.post(
				f"/workflows/{sample_workflow.id}/instances/{sample_workflow_instance.id}/pause",
				headers={"Authorization": "Bearer test_token"}
			)
			
			assert response.status_code == 200
			data = response.json()
			assert data["success"] is True
			assert data["data"]["paused"] is True
	
	def test_resume_workflow_instance(self, database_manager, redis_client, sample_workflow, sample_workflow_instance):
		"""Test resuming workflow instance."""
		app = create_api_app(database_manager, redis_client, "test_tenant")
		client = TestClient(app)
		
		with patch('__main__.get_current_user') as mock_auth, \
			 patch.object(app.state.workflow_service, 'resume_workflow_instance') as mock_resume:
			
			mock_auth.return_value = {
				"user_id": "test_user",
				"tenant_id": "test_tenant",
				"permissions": ["workflows.execute"]
			}
			mock_resume.return_value = True
			
			response = client.post(
				f"/workflows/{sample_workflow.id}/instances/{sample_workflow_instance.id}/resume",
				headers={"Authorization": "Bearer test_token"}
			)
			
			assert response.status_code == 200
			data = response.json()
			assert data["success"] is True
			assert data["data"]["resumed"] is True
	
	def test_stop_workflow_instance(self, database_manager, redis_client, sample_workflow, sample_workflow_instance):
		"""Test stopping workflow instance."""
		app = create_api_app(database_manager, redis_client, "test_tenant")
		client = TestClient(app)
		
		stop_request = {"reason": "User requested stop"}
		
		with patch('__main__.get_current_user') as mock_auth, \
			 patch.object(app.state.workflow_service, 'stop_workflow_instance') as mock_stop:
			
			mock_auth.return_value = {
				"user_id": "test_user",
				"tenant_id": "test_tenant",
				"permissions": ["workflows.execute"]
			}
			mock_stop.return_value = True
			
			response = client.post(
				f"/workflows/{sample_workflow.id}/instances/{sample_workflow_instance.id}/stop",
				json=stop_request,
				headers={"Authorization": "Bearer test_token"}
			)
			
			assert response.status_code == 200
			data = response.json()
			assert data["success"] is True
			assert data["data"]["stopped"] is True

class TestVersionControlEndpoints:
	"""Test version control endpoints."""
	
	def test_list_workflow_versions(self, database_manager, redis_client, sample_workflow):
		"""Test listing workflow versions."""
		app = create_api_app(database_manager, redis_client, "test_tenant")
		client = TestClient(app)
		
		mock_versions = [
			{
				"id": f"version_{i}",
				"workflow_id": sample_workflow.id,
				"version_number": f"1.{i}.0",
				"title": f"Version {i}",
				"created_at": datetime.now(timezone.utc).isoformat()
			}
			for i in range(3)
		]
		
		with patch('__main__.get_current_user') as mock_auth, \
			 patch.object(app.state.version_manager, 'list_versions') as mock_list:
			
			mock_auth.return_value = {
				"user_id": "test_user",
				"tenant_id": "test_tenant",
				"permissions": ["workflows.read"]
			}
			mock_list.return_value = mock_versions
			
			response = client.get(
				f"/workflows/{sample_workflow.id}/versions?branch_name=main&limit=50",
				headers={"Authorization": "Bearer test_token"}
			)
			
			assert response.status_code == 200
			data = response.json()
			assert data["success"] is True
			assert len(data["data"]) == 3
	
	def test_get_workflow_version(self, database_manager, redis_client, sample_workflow):
		"""Test getting specific workflow version."""
		app = create_api_app(database_manager, redis_client, "test_tenant")
		client = TestClient(app)
		
		mock_version = {
			"id": "version_123",
			"workflow_id": sample_workflow.id,
			"version_number": "1.2.0",
			"title": "Version 1.2.0",
			"created_at": datetime.now(timezone.utc).isoformat()
		}
		
		with patch('__main__.get_current_user') as mock_auth, \
			 patch.object(app.state.version_manager, 'get_version') as mock_get:
			
			mock_auth.return_value = {
				"user_id": "test_user",
				"tenant_id": "test_tenant",
				"permissions": ["workflows.read"]
			}
			mock_get.return_value = mock_version
			
			response = client.get(
				f"/workflows/{sample_workflow.id}/versions/version_123",
				headers={"Authorization": "Bearer test_token"}
			)
			
			assert response.status_code == 200
			data = response.json()
			assert data["success"] is True
			assert data["data"]["version_number"] == "1.2.0"
	
	def test_compare_workflow_versions(self, database_manager, redis_client, sample_workflow):
		"""Test comparing workflow versions."""
		app = create_api_app(database_manager, redis_client, "test_tenant")
		client = TestClient(app)
		
		mock_comparison = {
			"from_version": "version_1",
			"to_version": "version_2",
			"changes": [
				{"type": "task_added", "task_id": "new_task", "description": "Added new task"}
			],
			"added_tasks": ["new_task"],
			"modified_tasks": [],
			"deleted_tasks": [],
			"compatibility_score": 85.0,
			"breaking_changes": False,
			"summary": "1 task(s) added"
		}
		
		with patch('__main__.get_current_user') as mock_auth, \
			 patch.object(app.state.version_manager, 'compare_versions') as mock_compare:
			
			mock_auth.return_value = {
				"user_id": "test_user",
				"tenant_id": "test_tenant",
				"permissions": ["workflows.read"]
			}
			mock_compare.return_value = mock_comparison
			
			response = client.post(
				f"/workflows/{sample_workflow.id}/versions/version_1/compare/version_2",
				headers={"Authorization": "Bearer test_token"}
			)
			
			assert response.status_code == 200
			data = response.json()
			assert data["success"] is True
			assert data["data"]["compatibility_score"] == 85.0
			assert data["data"]["breaking_changes"] is False

class TestAuthentication:
	"""Test authentication and authorization."""
	
	def test_missing_authentication(self, database_manager, redis_client):
		"""Test endpoints without authentication."""
		app = create_api_app(database_manager, redis_client, "test_tenant")
		client = TestClient(app)
		
		# Try to access protected endpoint without token
		response = client.get("/workflows")
		assert response.status_code == 403  # Forbidden
	
	def test_invalid_token(self, database_manager, redis_client):
		"""Test endpoints with invalid token."""
		app = create_api_app(database_manager, redis_client, "test_tenant")
		client = TestClient(app)
		
		with patch('__main__.get_current_user') as mock_auth:
			mock_auth.side_effect = HTTPException(status_code=401, detail="Invalid token")
			
			response = client.get(
				"/workflows",
				headers={"Authorization": "Bearer invalid_token"}
			)
			assert response.status_code == 401
	
	def test_insufficient_permissions(self, database_manager, redis_client):
		"""Test endpoints with insufficient permissions."""
		app = create_api_app(database_manager, redis_client, "test_tenant")
		client = TestClient(app)
		
		with patch('__main__.get_current_user') as mock_auth:
			mock_auth.return_value = {
				"user_id": "test_user",
				"tenant_id": "test_tenant",
				"permissions": ["workflows.read"]  # Missing workflows.write
			}
			
			response = client.post(
				"/workflows",
				json={"name": "Test", "tasks": []},
				headers={"Authorization": "Bearer test_token"}
			)
			assert response.status_code == 403  # Forbidden

class TestRateLimiting:
	"""Test rate limiting functionality."""
	
	@pytest.mark.asyncio
	async def test_rate_limit_middleware(self, redis_client):
		"""Test rate limit middleware functionality."""
		from fastapi import FastAPI, Request
		
		app = FastAPI()
		middleware = RateLimitMiddleware(app, redis_client, default_rate_limit=5)
		
		# Mock request
		request = MagicMock(spec=Request)
		request.state = MagicMock()
		request.state.user_id = "test_user"
		
		# Mock call_next
		async def mock_call_next(request):
			from fastapi import Response
			return Response("OK", status_code=200)
		
		# Test within rate limit
		for i in range(5):
			response = await middleware.dispatch(request, mock_call_next)
			assert response.status_code == 200
		
		# Test rate limit exceeded
		with patch.object(redis_client, 'get') as mock_get, \
			 patch.object(redis_client, 'incr') as mock_incr:
			
			mock_get.return_value = "5"  # Already at limit
			
			response = await middleware.dispatch(request, mock_call_next)
			assert response.status_code == 429
	
	def test_rate_limit_headers(self, database_manager, redis_client):
		"""Test rate limit headers in responses."""
		app = create_api_app(database_manager, redis_client, "test_tenant")
		client = TestClient(app)
		
		response = client.get("/health")
		assert response.status_code == 200
		
		# Check rate limit headers are present
		assert "X-RateLimit-Limit" in response.headers
		assert "X-RateLimit-Remaining" in response.headers
		assert "X-RateLimit-Reset" in response.headers

class TestErrorHandling:
	"""Test error handling and exception responses."""
	
	def test_http_exception_handler(self, database_manager, redis_client):
		"""Test HTTP exception handling."""
		app = create_api_app(database_manager, redis_client, "test_tenant")
		client = TestClient(app)
		
		with patch('__main__.get_current_user') as mock_auth, \
			 patch.object(app.state.workflow_manager, 'get_workflow') as mock_get:
			
			mock_auth.return_value = {
				"user_id": "test_user",
				"tenant_id": "test_tenant",
				"permissions": ["workflows.read"]
			}
			mock_get.return_value = None  # Workflow not found
			
			response = client.get(
				"/workflows/nonexistent_id",
				headers={"Authorization": "Bearer test_token"}
			)
			
			assert response.status_code == 404
			data = response.json()
			assert data["success"] is False
			assert "not found" in data["message"].lower()
	
	def test_validation_error_handling(self, database_manager, redis_client):
		"""Test validation error handling."""
		app = create_api_app(database_manager, redis_client, "test_tenant")
		client = TestClient(app)
		
		invalid_data = {
			"name": "",  # Invalid: empty name
			"tasks": []  # Invalid: empty tasks
		}
		
		with patch('__main__.get_current_user') as mock_auth:
			mock_auth.return_value = {
				"user_id": "test_user",
				"tenant_id": "test_tenant",
				"permissions": ["workflows.write"]
			}
			
			response = client.post(
				"/workflows",
				json=invalid_data,
				headers={"Authorization": "Bearer test_token"}
			)
			
			assert response.status_code == 422
			data = response.json()
			assert "detail" in data  # FastAPI validation error format
	
	def test_general_exception_handler(self, database_manager, redis_client):
		"""Test general exception handling."""
		app = create_api_app(database_manager, redis_client, "test_tenant")
		client = TestClient(app)
		
		with patch('__main__.get_current_user') as mock_auth, \
			 patch.object(app.state.workflow_manager, 'create_workflow') as mock_create:
			
			mock_auth.return_value = {
				"user_id": "test_user",
				"tenant_id": "test_tenant",
				"permissions": ["workflows.write"]
			}
			mock_create.side_effect = Exception("Unexpected error")
			
			response = client.post(
				"/workflows",
				json={"name": "Test", "tasks": [{"name": "Task", "task_type": "task"}]},
				headers={"Authorization": "Bearer test_token"}
			)
			
			assert response.status_code == 500
			data = response.json()
			assert data["success"] is False
			assert data["message"] == "Internal server error"

class TestAPIModels:
	"""Test API request/response models."""
	
	def test_api_response_model(self):
		"""Test APIResponse model."""
		response = APIResponse(
			success=True,
			message="Operation completed",
			data={"result": "success"},
			metadata={"execution_time": 1.5}
		)
		
		assert response.success is True
		assert response.message == "Operation completed"
		assert response.data["result"] == "success"
		assert response.metadata["execution_time"] == 1.5
		assert isinstance(response.timestamp, datetime)
		assert response.errors == []
	
	def test_workflow_create_request_model(self):
		"""Test WorkflowCreateRequest model."""
		request_data = {
			"name": "Test Workflow",
			"description": "Test description",
			"tasks": [{"name": "Task 1", "type": "task"}],
			"configuration": {"max_concurrent": 5},
			"priority": "high",
			"tags": ["test", "api"],
			"sla_hours": 24.0
		}
		
		request = WorkflowCreateRequest(**request_data)
		
		assert request.name == "Test Workflow"
		assert request.description == "Test description"
		assert len(request.tasks) == 1
		assert request.priority == Priority.HIGH
		assert request.sla_hours == 24.0
		assert "test" in request.tags
	
	def test_workflow_execute_request_model(self):
		"""Test WorkflowExecuteRequest model."""
		request_data = {
			"input_data": {"param1": "value1"},
			"configuration_overrides": {"timeout": 3600},
			"priority": "critical",
			"tags": ["urgent"],
			"schedule_at": datetime.now(timezone.utc) + timedelta(hours=1)
		}
		
		request = WorkflowExecuteRequest(**request_data)
		
		assert request.input_data["param1"] == "value1"
		assert request.configuration_overrides["timeout"] == 3600
		assert request.priority == Priority.CRITICAL
		assert "urgent" in request.tags
		assert isinstance(request.schedule_at, datetime)
	
	def test_pagination_params_model(self):
		"""Test PaginationParams model."""
		# Test valid parameters
		params = PaginationParams(offset=10, limit=50)
		assert params.offset == 10
		assert params.limit == 50
		
		# Test default values
		default_params = PaginationParams()
		assert default_params.offset == 0
		assert default_params.limit == 100
		
		# Test validation
		with pytest.raises(ValueError):
			PaginationParams(offset=-1)  # Negative offset
		
		with pytest.raises(ValueError):
			PaginationParams(limit=2000)  # Limit too high