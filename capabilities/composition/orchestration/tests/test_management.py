"""
APG Workflow Orchestration Management Tests

Comprehensive unit tests for workflow management, version control, 
and deployment management functionality.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
import json

from ..management import (
	WorkflowManager, WorkflowOperations, WorkflowValidationLevel,
	WorkflowSearchFilter, WorkflowValidationResult, WorkflowStatistics
)
from ..management.version_control import (
	VersionManager, WorkflowVersionControl, WorkflowVersion,
	VersionComparison, MergeResult, MergeConflict, VersionStatus, MergeStrategy
)
from ..management.deployment_manager import (
	DeploymentManager, DeploymentEnvironment, DeploymentPlan,
	DeploymentStrategy, DeploymentStatus
)
from ..models import (
	Workflow, WorkflowInstance, TaskDefinition, TaskExecution,
	WorkflowStatus, TaskStatus, Priority, TaskType
)

class TestWorkflowManager:
	"""Test WorkflowManager functionality."""
	
	@pytest.mark.asyncio
	async def test_workflow_manager_initialization(self, database_manager, redis_client):
		"""Test workflow manager initialization."""
		manager = WorkflowManager(database_manager, redis_client, "test_tenant")
		
		assert manager.tenant_id == "test_tenant"
		assert manager.operations is not None
		assert isinstance(manager.operations, WorkflowOperations)
	
	@pytest.mark.asyncio
	async def test_create_workflow_success(self, workflow_manager, test_helpers):
		"""Test successful workflow creation."""
		workflow_data = test_helpers.create_test_workflow_data()
		
		with patch.object(workflow_manager.operations, 'create_workflow') as mock_create:
			expected_workflow = Workflow(**workflow_data)
			mock_create.return_value = expected_workflow
			
			result = await workflow_manager.create_workflow(
				workflow_data, "test_user", WorkflowValidationLevel.STANDARD
			)
			
			assert result is not None
			test_helpers.assert_workflow_fields(result, workflow_data)
			mock_create.assert_called_once_with(
				workflow_data, "test_user", WorkflowValidationLevel.STANDARD
			)
	
	@pytest.mark.asyncio
	async def test_create_workflow_validation_failure(self, workflow_manager):
		"""Test workflow creation with validation failure."""
		invalid_workflow_data = {
			"name": "",  # Invalid: empty name
			"tasks": [],  # Invalid: empty tasks
			"tenant_id": "test_tenant",
			"created_by": "test_user",
			"updated_by": "test_user"
		}
		
		with patch.object(workflow_manager.operations, 'create_workflow') as mock_create:
			mock_create.side_effect = ValueError("Workflow validation failed")
			
			with pytest.raises(ValueError, match="Workflow validation failed"):
				await workflow_manager.create_workflow(
					invalid_workflow_data, "test_user"
				)
	
	@pytest.mark.asyncio
	async def test_get_workflow_success(self, workflow_manager, sample_workflow):
		"""Test successful workflow retrieval."""
		with patch.object(workflow_manager.operations, 'get_workflow') as mock_get:
			mock_get.return_value = sample_workflow
			
			result = await workflow_manager.get_workflow(
				sample_workflow.id, "test_user"
			)
			
			assert result == sample_workflow
			mock_get.assert_called_once_with(
				sample_workflow.id, "test_user", False
			)
	
	@pytest.mark.asyncio
	async def test_get_workflow_with_instances(self, workflow_manager, sample_workflow):
		"""Test workflow retrieval with instances."""
		with patch.object(workflow_manager.operations, 'get_workflow') as mock_get:
			mock_get.return_value = sample_workflow
			
			result = await workflow_manager.get_workflow(
				sample_workflow.id, "test_user", include_instances=True
			)
			
			assert result == sample_workflow
			mock_get.assert_called_once_with(
				sample_workflow.id, "test_user", True
			)
	
	@pytest.mark.asyncio
	async def test_get_workflow_not_found(self, workflow_manager):
		"""Test workflow retrieval when not found."""
		with patch.object(workflow_manager.operations, 'get_workflow') as mock_get:
			mock_get.return_value = None
			
			result = await workflow_manager.get_workflow(
				"nonexistent_id", "test_user"
			)
			
			assert result is None
	
	@pytest.mark.asyncio
	async def test_update_workflow_success(self, workflow_manager, sample_workflow):
		"""Test successful workflow update."""
		updates = {
			"name": "Updated Workflow Name",
			"description": "Updated description",
			"priority": Priority.HIGH.value
		}
		
		updated_workflow = sample_workflow.model_copy()
		updated_workflow.name = updates["name"]
		updated_workflow.description = updates["description"]
		updated_workflow.priority = Priority.HIGH
		
		with patch.object(workflow_manager.operations, 'update_workflow') as mock_update:
			mock_update.return_value = updated_workflow
			
			result = await workflow_manager.update_workflow(
				sample_workflow.id, updates, "test_user"
			)
			
			assert result.name == "Updated Workflow Name"
			assert result.priority == Priority.HIGH
			mock_update.assert_called_once()
	
	@pytest.mark.asyncio
	async def test_delete_workflow_success(self, workflow_manager, sample_workflow):
		"""Test successful workflow deletion."""
		with patch.object(workflow_manager.operations, 'delete_workflow') as mock_delete:
			mock_delete.return_value = True
			
			result = await workflow_manager.delete_workflow(
				sample_workflow.id, "test_user", hard_delete=False
			)
			
			assert result is True
			mock_delete.assert_called_once_with(
				sample_workflow.id, "test_user", False
			)
	
	@pytest.mark.asyncio
	async def test_search_workflows(self, workflow_manager):
		"""Test workflow search functionality."""
		search_filters = WorkflowSearchFilter(
			name_pattern="Test%",
			priority=[Priority.HIGH],
			tags=["test"],
			limit=10
		)
		
		mock_workflows = [
			Workflow(
				name=f"Test Workflow {i}",
				tasks=[TaskDefinition(name=f"Task {i}", task_type=TaskType.TASK)],
				priority=Priority.HIGH,
				tenant_id="test_tenant",
				created_by="test_user",
				updated_by="test_user",
				tags=["test"]
			)
			for i in range(3)
		]
		
		with patch.object(workflow_manager.operations, 'search_workflows') as mock_search:
			mock_search.return_value = mock_workflows
			
			result = await workflow_manager.search_workflows(search_filters, "test_user")
			
			assert len(result) == 3
			assert all(workflow.priority == Priority.HIGH for workflow in result)
			assert all("test" in workflow.tags for workflow in result)
			mock_search.assert_called_once_with(search_filters, "test_user")
	
	@pytest.mark.asyncio
	async def test_clone_workflow_success(self, workflow_manager, sample_workflow):
		"""Test successful workflow cloning."""
		cloned_workflow = sample_workflow.model_copy()
		cloned_workflow.id = "cloned_workflow_id"
		cloned_workflow.name = "Cloned Workflow"
		
		modifications = {"priority": Priority.CRITICAL.value}
		
		with patch.object(workflow_manager.operations, 'clone_workflow') as mock_clone:
			mock_clone.return_value = cloned_workflow
			
			result = await workflow_manager.clone_workflow(
				sample_workflow.id, "Cloned Workflow", "test_user", modifications
			)
			
			assert result.name == "Cloned Workflow"
			assert result.id != sample_workflow.id
			mock_clone.assert_called_once_with(
				sample_workflow.id, "Cloned Workflow", "test_user", modifications
			)
	
	@pytest.mark.asyncio
	async def test_get_workflow_statistics(self, workflow_manager, sample_workflow):
		"""Test workflow statistics retrieval."""
		mock_statistics = WorkflowStatistics(
			total_executions=100,
			successful_executions=95,
			failed_executions=5,
			average_duration_seconds=300.0,
			success_rate=95.0,
			active_instances=3,
			total_tasks=150,
			completed_tasks=140,
			failed_tasks=10
		)
		
		with patch.object(workflow_manager.operations, 'get_workflow_statistics') as mock_stats:
			mock_stats.return_value = mock_statistics
			
			result = await workflow_manager.get_workflow_statistics(
				sample_workflow.id, "test_user", 30
			)
			
			assert result.total_executions == 100
			assert result.success_rate == 95.0
			assert result.active_instances == 3
			mock_stats.assert_called_once_with(
				sample_workflow.id, "test_user", 30
			)

class TestWorkflowOperations:
	"""Test WorkflowOperations functionality."""
	
	@pytest.mark.asyncio
	async def test_workflow_validation_basic(self, database_manager, redis_client):
		"""Test basic workflow validation."""
		operations = WorkflowOperations(database_manager, redis_client, "test_tenant")
		
		# Valid workflow
		valid_workflow = Workflow(
			name="Valid Workflow",
			tasks=[TaskDefinition(name="Valid Task", task_type=TaskType.TASK)],
			tenant_id="test_tenant",
			created_by="test_user",
			updated_by="test_user"
		)
		
		validation_result = await operations._validate_workflow(
			valid_workflow, WorkflowValidationLevel.BASIC
		)
		
		assert validation_result.is_valid is True
		assert len(validation_result.errors) == 0
		assert validation_result.score > 0
	
	@pytest.mark.asyncio
	async def test_workflow_validation_errors(self, database_manager, redis_client):
		"""Test workflow validation with errors."""
		operations = WorkflowOperations(database_manager, redis_client, "test_tenant")
		
		# Invalid workflow
		invalid_workflow = Workflow(
			name="",  # Empty name
			tasks=[],  # Empty tasks
			tenant_id="test_tenant",
			created_by="test_user",
			updated_by="test_user"
		)
		
		validation_result = await operations._validate_workflow(
			invalid_workflow, WorkflowValidationLevel.BASIC
		)
		
		assert validation_result.is_valid is False
		assert len(validation_result.errors) > 0
		assert "name is required" in " ".join(validation_result.errors).lower()
		assert "at least one task" in " ".join(validation_result.errors).lower()
		assert validation_result.score == 0
	
	@pytest.mark.asyncio
	async def test_workflow_validation_circular_dependencies(self, database_manager, redis_client):
		"""Test workflow validation for circular dependencies."""
		operations = WorkflowOperations(database_manager, redis_client, "test_tenant")
		
		# Workflow with circular dependencies
		task1 = TaskDefinition(
			id="task_1",
			name="Task 1",
			task_type=TaskType.TASK,
			dependencies=["task_2"]  # Depends on task_2
		)
		task2 = TaskDefinition(
			id="task_2",
			name="Task 2",
			task_type=TaskType.TASK,
			dependencies=["task_1"]  # Depends on task_1 (circular!)
		)
		
		circular_workflow = Workflow(
			name="Circular Workflow",
			tasks=[task1, task2],
			tenant_id="test_tenant",
			created_by="test_user",
			updated_by="test_user"
		)
		
		validation_result = await operations._validate_workflow(
			circular_workflow, WorkflowValidationLevel.STANDARD
		)
		
		assert validation_result.is_valid is False
		assert "circular dependencies" in " ".join(validation_result.errors).lower()
	
	@pytest.mark.asyncio
	async def test_workflow_validation_enterprise_level(self, database_manager, redis_client):
		"""Test enterprise-level workflow validation."""
		operations = WorkflowOperations(database_manager, redis_client, "test_tenant")
		
		# Enterprise workflow with metadata
		enterprise_workflow = Workflow(
			name="Enterprise Workflow",
			tasks=[
				TaskDefinition(
					name="Compliance Task",
					task_type=TaskType.INTEGRATION,
					configuration={"service": "compliance_check"},
					estimated_duration=3600  # 1 hour
				)
			],
			tenant_id="test_tenant",
			created_by="test_user",
			updated_by="test_user",
			metadata={"compliance_level": "high"}
		)
		
		validation_result = await operations._validate_workflow(
			enterprise_workflow, WorkflowValidationLevel.ENTERPRISE
		)
		
		assert validation_result.is_valid is True
		assert validation_result.validation_level == WorkflowValidationLevel.ENTERPRISE
		assert validation_result.details["estimated_duration"] > 0
	
	@pytest.mark.asyncio
	async def test_workflow_caching(self, database_manager, redis_client, sample_workflow):
		"""Test workflow caching functionality."""
		operations = WorkflowOperations(database_manager, redis_client, "test_tenant")
		
		# Mock Redis operations
		with patch.object(operations.redis_client, 'setex') as mock_setex, \
			 patch.object(operations.redis_client, 'get') as mock_get:
			
			# Test caching workflow
			await operations._cache_workflow(sample_workflow)
			mock_setex.assert_called_once()
			
			# Test retrieving cached workflow
			mock_get.return_value = sample_workflow.model_dump_json()
			cached_workflow = await operations._get_cached_workflow(sample_workflow.id)
			
			assert cached_workflow is not None
			assert cached_workflow.id == sample_workflow.id
			assert cached_workflow.name == sample_workflow.name
	
	@pytest.mark.asyncio
	async def test_operation_logging(self, database_manager, redis_client, sample_workflow):
		"""Test operation logging functionality."""
		operations = WorkflowOperations(database_manager, redis_client, "test_tenant")
		
		from ..management.workflow_manager import WorkflowOperationType
		
		with patch.object(operations.redis_client, 'lpush') as mock_lpush:
			# Log an operation
			await operations._log_operation(
				WorkflowOperationType.CREATE,
				sample_workflow.id,
				"test_user",
				{"workflow_name": sample_workflow.name}
			)
			
			# Verify operation was logged
			assert len(operations.operation_history) == 1
			logged_operation = operations.operation_history[0]
			assert logged_operation["operation"] == "create"
			assert logged_operation["workflow_id"] == sample_workflow.id
			assert logged_operation["user_id"] == "test_user"
			mock_lpush.assert_called_once()

class TestVersionManager:
	"""Test VersionManager functionality."""
	
	@pytest.mark.asyncio
	async def test_version_manager_initialization(self, database_manager, redis_client):
		"""Test version manager initialization."""
		manager = VersionManager(database_manager, redis_client, "test_tenant")
		
		assert manager.tenant_id == "test_tenant"
		assert manager.version_control is not None
		assert isinstance(manager.version_control, WorkflowVersionControl)
	
	@pytest.mark.asyncio
	async def test_create_version(self, version_manager, sample_workflow):
		"""Test version creation."""
		with patch.object(version_manager.version_control, 'create_version') as mock_create:
			mock_version = WorkflowVersion(
				workflow_id=sample_workflow.id,
				version_number="1.0.0",
				title="Initial Version",
				description="First version of the workflow",
				checksum="abc123",
				workflow_definition=sample_workflow.model_dump(),
				created_by="test_user",
				tenant_id="test_tenant"
			)
			mock_create.return_value = mock_version
			
			result = await version_manager.create_version(
				sample_workflow, "1.0.0", "Initial Version", "First version", "main"
			)
			
			assert result.version_number == "1.0.0"
			assert result.title == "Initial Version"
			assert result.workflow_id == sample_workflow.id
			mock_create.assert_called_once()
	
	@pytest.mark.asyncio
	async def test_get_version(self, version_manager):
		"""Test version retrieval."""
		version_id = "version_123"
		mock_version = WorkflowVersion(
			id=version_id,
			workflow_id="workflow_123",
			version_number="1.2.0",
			title="Version 1.2.0",
			checksum="def456",
			workflow_definition={"name": "Test"},
			created_by="test_user",
			tenant_id="test_tenant"
		)
		
		with patch.object(version_manager.version_control, 'get_version') as mock_get:
			mock_get.return_value = mock_version
			
			result = await version_manager.get_version(version_id)
			
			assert result == mock_version
			assert result.id == version_id
			mock_get.assert_called_once_with(version_id)
	
	@pytest.mark.asyncio
	async def test_list_versions(self, version_manager):
		"""Test version listing."""
		workflow_id = "workflow_123"
		mock_versions = [
			WorkflowVersion(
				id=f"version_{i}",
				workflow_id=workflow_id,
				version_number=f"1.{i}.0",
				title=f"Version 1.{i}.0",
				checksum=f"hash_{i}",
				workflow_definition={"name": f"Test {i}"},
				created_by="test_user",
				tenant_id="test_tenant"
			)
			for i in range(3)
		]
		
		with patch.object(version_manager.version_control, 'list_versions') as mock_list:
			mock_list.return_value = mock_versions
			
			result = await version_manager.list_versions(
				workflow_id, branch_name="main", limit=10
			)
			
			assert len(result) == 3
			assert all(v.workflow_id == workflow_id for v in result)
			mock_list.assert_called_once_with(workflow_id, "main", None, 10)
	
	@pytest.mark.asyncio
	async def test_compare_versions(self, version_manager):
		"""Test version comparison."""
		from_version = "version_1"
		to_version = "version_2"
		
		mock_comparison = VersionComparison(
			from_version=from_version,
			to_version=to_version,
			changes=[
				{"type": "task_added", "task_id": "new_task", "description": "Added new task"}
			],
			added_tasks=["new_task"],
			modified_tasks=[],
			deleted_tasks=[],
			compatibility_score=85.0,
			breaking_changes=False,
			summary="1 task(s) added"
		)
		
		with patch.object(version_manager.version_control, 'compare_versions') as mock_compare:
			mock_compare.return_value = mock_comparison
			
			result = await version_manager.compare_versions(from_version, to_version)
			
			assert result.from_version == from_version
			assert result.to_version == to_version
			assert result.compatibility_score == 85.0
			assert result.breaking_changes is False
			assert len(result.added_tasks) == 1
			mock_compare.assert_called_once_with(from_version, to_version)
	
	@pytest.mark.asyncio
	async def test_create_branch(self, version_manager):
		"""Test branch creation."""
		workflow_id = "workflow_123"
		branch_name = "feature_branch"
		from_version_id = "version_main"
		
		mock_branch_version = WorkflowVersion(
			workflow_id=workflow_id,
			version_number="1.0.0",
			branch_name=branch_name,
			title=f"Branch {branch_name}",
			checksum="branch_hash",
			workflow_definition={"name": "Branched Workflow"},
			parent_version_id=from_version_id,
			created_by="test_user",
			tenant_id="test_tenant"
		)
		
		with patch.object(version_manager.version_control, 'create_branch') as mock_create_branch:
			mock_create_branch.return_value = mock_branch_version
			
			result = await version_manager.create_branch(
				workflow_id, branch_name, from_version_id, "test_user", "Feature branch"
			)
			
			assert result.branch_name == branch_name
			assert result.parent_version_id == from_version_id
			assert result.workflow_id == workflow_id
			mock_create_branch.assert_called_once()
	
	@pytest.mark.asyncio
	async def test_merge_versions(self, version_manager):
		"""Test version merging."""
		base_version_id = "base_version"
		merge_version_id = "merge_version"
		target_branch = "main"
		
		mock_merge_result = MergeResult(
			success=True,
			merged_version_id="merged_version_123",
			conflicts=[],
			auto_resolved=2,
			manual_required=0,
			summary="Successfully merged 2 conflicts automatically"
		)
		
		with patch.object(version_manager.version_control, 'merge_versions') as mock_merge:
			mock_merge.return_value = mock_merge_result
			
			result = await version_manager.merge_versions(
				base_version_id, merge_version_id, target_branch,
				MergeStrategy.AUTO, "test_user", "Merge feature branch"
			)
			
			assert result.success is True
			assert result.merged_version_id == "merged_version_123"
			assert result.auto_resolved == 2
			assert result.manual_required == 0
			mock_merge.assert_called_once()
	
	@pytest.mark.asyncio
	async def test_tag_version(self, version_manager):
		"""Test version tagging."""
		version_id = "version_123"
		tag_name = "v1.0.0-release"
		
		with patch.object(version_manager.version_control, 'tag_version') as mock_tag:
			mock_tag.return_value = True
			
			result = await version_manager.tag_version(
				version_id, tag_name, "Release version", "test_user"
			)
			
			assert result is True
			mock_tag.assert_called_once_with(
				version_id, tag_name, "Release version", "test_user"
			)
	
	@pytest.mark.asyncio
	async def test_restore_version(self, version_manager, sample_workflow):
		"""Test version restoration."""
		version_id = "version_to_restore"
		
		with patch.object(version_manager.version_control, 'restore_version') as mock_restore:
			mock_restore.return_value = sample_workflow
			
			result = await version_manager.restore_version(version_id, "test_user")
			
			assert result == sample_workflow
			mock_restore.assert_called_once_with(version_id, "test_user")

class TestDeploymentManager:
	"""Test DeploymentManager functionality."""
	
	@pytest.mark.asyncio
	async def test_deployment_manager_initialization(self, database_manager, redis_client, version_manager):
		"""Test deployment manager initialization."""
		manager = DeploymentManager(
			database_manager, redis_client, version_manager, "test_tenant"
		)
		
		assert manager.tenant_id == "test_tenant"
		assert manager.database_manager == database_manager
		assert manager.redis_client == redis_client
		assert manager.version_manager == version_manager
	
	@pytest.mark.asyncio
	async def test_create_deployment_environment(self, database_manager, redis_client, version_manager):
		"""Test deployment environment creation."""
		manager = DeploymentManager(
			database_manager, redis_client, version_manager, "test_tenant"
		)
		
		env_config = {
			"name": "production",
			"description": "Production environment",
			"configuration": {
				"instance_count": 3,
				"auto_scaling": True,
				"health_check_interval": 30
			},
			"resource_limits": {
				"cpu": "4 cores",
				"memory": "8GB",
				"storage": "100GB"
			}
		}
		
		with patch.object(manager, '_create_environment') as mock_create:
			mock_environment = DeploymentEnvironment(
				id="env_123",
				name="production",
				tenant_id="test_tenant",
				created_by="test_user",
				**env_config
			)
			mock_create.return_value = mock_environment
			
			result = await manager.create_deployment_environment(
				env_config, "test_user"
			)
			
			assert result.name == "production"
			assert result.configuration["instance_count"] == 3
			mock_create.assert_called_once()
	
	@pytest.mark.asyncio
	async def test_create_deployment_plan(self, database_manager, redis_client, version_manager):
		"""Test deployment plan creation."""
		manager = DeploymentManager(
			database_manager, redis_client, version_manager, "test_tenant"
		)
		
		plan_config = {
			"name": "Production Deployment",
			"workflow_version_id": "version_123",
			"target_environment_id": "env_prod",
			"strategy": DeploymentStrategy.BLUE_GREEN,
			"configuration": {
				"rollback_threshold": 0.1,
				"health_check_timeout": 300,
				"pre_deployment_checks": True
			}
		}
		
		with patch.object(manager, '_create_deployment_plan') as mock_create_plan:
			mock_plan = DeploymentPlan(
				id="plan_123",
				tenant_id="test_tenant",
				created_by="test_user",
				**plan_config
			)
			mock_create_plan.return_value = mock_plan
			
			result = await manager.create_deployment_plan(
				plan_config, "test_user"
			)
			
			assert result.name == "Production Deployment"
			assert result.strategy == DeploymentStrategy.BLUE_GREEN
			assert result.workflow_version_id == "version_123"
			mock_create_plan.assert_called_once()
	
	@pytest.mark.asyncio
	async def test_execute_deployment_blue_green(self, database_manager, redis_client, version_manager):
		"""Test blue-green deployment execution."""
		manager = DeploymentManager(
			database_manager, redis_client, version_manager, "test_tenant"
		)
		
		deployment_plan = DeploymentPlan(
			id="plan_123",
			name="Blue-Green Deployment",
			workflow_version_id="version_123",
			target_environment_id="env_prod",
			strategy=DeploymentStrategy.BLUE_GREEN,
			tenant_id="test_tenant",
			created_by="test_user"
		)
		
		with patch.object(manager, '_execute_blue_green_deployment') as mock_execute:
			mock_execution = {
				"deployment_id": "deploy_123",
				"status": "in_progress",
				"blue_environment": "prod_blue",
				"green_environment": "prod_green",
				"current_phase": "deploying_green"
			}
			mock_execute.return_value = mock_execution
			
			result = await manager.execute_deployment(
				deployment_plan, "test_user"
			)
			
			assert result["status"] == "in_progress"
			assert result["strategy"] == "blue_green"
			assert result["current_phase"] == "deploying_green"
			mock_execute.assert_called_once()
	
	@pytest.mark.asyncio
	async def test_execute_deployment_rolling(self, database_manager, redis_client, version_manager):
		"""Test rolling deployment execution."""
		manager = DeploymentManager(
			database_manager, redis_client, version_manager, "test_tenant"
		)
		
		deployment_plan = DeploymentPlan(
			id="plan_456",
			name="Rolling Deployment",
			workflow_version_id="version_456",
			target_environment_id="env_staging",
			strategy=DeploymentStrategy.ROLLING,
			configuration={
				"batch_size": 2,
				"batch_delay": 60
			},
			tenant_id="test_tenant",
			created_by="test_user"
		)
		
		with patch.object(manager, '_execute_rolling_deployment') as mock_execute:
			mock_execution = {
				"deployment_id": "deploy_456",
				"status": "in_progress",
				"total_instances": 10,
				"updated_instances": 4,
				"current_batch": 3,
				"remaining_batches": 2
			}
			mock_execute.return_value = mock_execution
			
			result = await manager.execute_deployment(
				deployment_plan, "test_user"
			)
			
			assert result["status"] == "in_progress"
			assert result["total_instances"] == 10
			assert result["updated_instances"] == 4
			assert result["current_batch"] == 3
			mock_execute.assert_called_once()
	
	@pytest.mark.asyncio
	async def test_deployment_health_monitoring(self, database_manager, redis_client, version_manager):
		"""Test deployment health monitoring."""
		manager = DeploymentManager(
			database_manager, redis_client, version_manager, "test_tenant"
		)
		
		deployment_id = "deploy_123"
		
		with patch.object(manager, '_monitor_deployment_health') as mock_monitor:
			mock_health_data = {
				"deployment_id": deployment_id,
				"overall_health": "healthy",
				"instance_health": {
					"healthy": 8,
					"unhealthy": 0,
					"unknown": 2
				},
				"performance_metrics": {
					"response_time_ms": 150,
					"error_rate": 0.001,
					"throughput_rps": 1000
				},
				"last_check": datetime.now(timezone.utc).isoformat()
			}
			mock_monitor.return_value = mock_health_data
			
			result = await manager.get_deployment_health(deployment_id)
			
			assert result["overall_health"] == "healthy"
			assert result["instance_health"]["healthy"] == 8
			assert result["performance_metrics"]["error_rate"] == 0.001
			mock_monitor.assert_called_once_with(deployment_id)
	
	@pytest.mark.asyncio
	async def test_deployment_rollback(self, database_manager, redis_client, version_manager):
		"""Test deployment rollback functionality."""
		manager = DeploymentManager(
			database_manager, redis_client, version_manager, "test_tenant"
		)
		
		deployment_id = "deploy_123"
		rollback_reason = "High error rate detected"
		
		with patch.object(manager, '_execute_rollback') as mock_rollback:
			mock_rollback_result = {
				"deployment_id": deployment_id,
				"rollback_status": "in_progress",
				"previous_version": "version_122",
				"rollback_strategy": "immediate",
				"estimated_completion": datetime.now(timezone.utc) + timedelta(minutes=5)
			}
			mock_rollback.return_value = mock_rollback_result
			
			result = await manager.rollback_deployment(
				deployment_id, rollback_reason, "test_user"
			)
			
			assert result["rollback_status"] == "in_progress"
			assert result["previous_version"] == "version_122"
			assert result["rollback_strategy"] == "immediate"
			mock_rollback.assert_called_once_with(
				deployment_id, rollback_reason, "test_user"
			)
	
	@pytest.mark.asyncio
	async def test_deployment_analytics(self, database_manager, redis_client, version_manager):
		"""Test deployment analytics and reporting."""
		manager = DeploymentManager(
			database_manager, redis_client, version_manager, "test_tenant"
		)
		
		environment_id = "env_prod"
		time_range_days = 30
		
		with patch.object(manager, '_generate_deployment_analytics') as mock_analytics:
			mock_analytics_data = {
				"environment_id": environment_id,
				"time_range_days": time_range_days,
				"total_deployments": 45,
				"successful_deployments": 42,
				"failed_deployments": 3,
				"success_rate": 93.3,
				"average_deployment_time": 420,  # seconds
				"rollback_rate": 6.7,
				"deployment_frequency": {
					"per_day": 1.5,
					"per_week": 10.5
				},
				"failure_analysis": {
					"infrastructure_failures": 1,
					"application_failures": 2,
					"configuration_errors": 0
				}
			}
			mock_analytics.return_value = mock_analytics_data
			
			result = await manager.get_deployment_analytics(
				environment_id, time_range_days
			)
			
			assert result["total_deployments"] == 45
			assert result["success_rate"] == 93.3
			assert result["rollback_rate"] == 6.7
			assert result["failure_analysis"]["application_failures"] == 2
			mock_analytics.assert_called_once_with(environment_id, time_range_days)

class TestManagementIntegration:
	"""Test integration between management components."""
	
	@pytest.mark.asyncio
	async def test_workflow_to_version_to_deployment_flow(
		self, workflow_manager, version_manager, database_manager, redis_client
	):
		"""Test complete flow from workflow creation to deployment."""
		deployment_manager = DeploymentManager(
			database_manager, redis_client, version_manager, "test_tenant"
		)
		
		# 1. Create workflow
		workflow_data = {
			"name": "End-to-End Test Workflow",
			"tasks": [
				{
					"name": "Test Task",
					"task_type": "task",
					"configuration": {"action": "test"}
				}
			],
			"tenant_id": "test_tenant",
			"created_by": "test_user",
			"updated_by": "test_user"
		}
		
		with patch.object(workflow_manager.operations, 'create_workflow') as mock_create_workflow:
			mock_workflow = Workflow(**workflow_data)
			mock_create_workflow.return_value = mock_workflow
			
			workflow = await workflow_manager.create_workflow(
				workflow_data, "test_user"
			)
			
			# 2. Create version
			with patch.object(version_manager.version_control, 'create_version') as mock_create_version:
				mock_version = WorkflowVersion(
					workflow_id=workflow.id,
					version_number="1.0.0",
					title="Initial Release",
					checksum="hash123",
					workflow_definition=workflow.model_dump(),
					created_by="test_user",
					tenant_id="test_tenant"
				)
				mock_create_version.return_value = mock_version
				
				version = await version_manager.create_version(
					workflow, "1.0.0", "Initial Release"
				)
				
				# 3. Create deployment plan
				plan_config = {
					"name": "Production Deployment",
					"workflow_version_id": version.id,
					"target_environment_id": "env_prod",
					"strategy": DeploymentStrategy.BLUE_GREEN
				}
				
				with patch.object(deployment_manager, '_create_deployment_plan') as mock_create_plan:
					mock_plan = DeploymentPlan(
						id="plan_123",
						tenant_id="test_tenant",
						created_by="test_user",
						**plan_config
					)
					mock_create_plan.return_value = mock_plan
					
					plan = await deployment_manager.create_deployment_plan(
						plan_config, "test_user"
					)
					
					# Verify the complete flow
					assert workflow.name == "End-to-End Test Workflow"
					assert version.workflow_id == workflow.id
					assert plan.workflow_version_id == version.id
					assert plan.strategy == DeploymentStrategy.BLUE_GREEN
	
	@pytest.mark.asyncio
	async def test_cross_component_error_handling(
		self, workflow_manager, version_manager, database_manager, redis_client
	):
		"""Test error handling across management components."""
		deployment_manager = DeploymentManager(
			database_manager, redis_client, version_manager, "test_tenant"
		)
		
		# Test workflow creation failure
		with patch.object(workflow_manager.operations, 'create_workflow') as mock_create:
			mock_create.side_effect = ValueError("Database connection failed")
			
			with pytest.raises(ValueError, match="Database connection failed"):
				await workflow_manager.create_workflow({}, "test_user")
		
		# Test version creation failure
		with patch.object(version_manager.version_control, 'create_version') as mock_create_version:
			mock_create_version.side_effect = Exception("Version storage failed")
			
			sample_workflow = Workflow(
				name="Test",
				tasks=[TaskDefinition(name="Task", task_type=TaskType.TASK)],
				tenant_id="test_tenant",
				created_by="test_user",
				updated_by="test_user"
			)
			
			with pytest.raises(Exception, match="Version storage failed"):
				await version_manager.create_version(
					sample_workflow, "1.0.0", "Test Version"
				)
		
		# Test deployment failure
		with patch.object(deployment_manager, '_create_deployment_plan') as mock_create_plan:
			mock_create_plan.side_effect = RuntimeError("Environment not available")
			
			with pytest.raises(RuntimeError, match="Environment not available"):
				await deployment_manager.create_deployment_plan({}, "test_user")