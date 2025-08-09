"""
APG Workflow Orchestration Database Tests

Comprehensive unit tests for database models, repositories, and database operations.
Tests SQLAlchemy async models, CRUD operations, and database integrity.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
from sqlalchemy import select, func, and_, or_, desc
from sqlalchemy.exc import IntegrityError
import json

from ..database import (
	DatabaseManager, create_repositories,
	CRWorkflow, CRWorkflowInstance, CRTaskExecution,
	CRWorkflowVersion, CRDeploymentEnvironment, CRDeploymentPlan,
	WorkflowRepository, WorkflowInstanceRepository, TaskExecutionRepository
)
from ..models import (
	Workflow, WorkflowInstance, TaskDefinition, TaskExecution,
	WorkflowStatus, TaskStatus, Priority, TaskType
)

class TestDatabaseManager:
	"""Test DatabaseManager functionality."""
	
	@pytest.mark.asyncio
	async def test_database_manager_initialization(self):
		"""Test database manager initialization."""
		manager = DatabaseManager("sqlite+aiosqlite:///:memory:")
		
		assert manager.database_url == "sqlite+aiosqlite:///:memory:"
		assert manager.pool_size == 20  # default
		assert manager.max_overflow == 0  # default
		assert manager.engine is not None
		assert manager.session_factory is not None
		
		await manager.close()
	
	@pytest.mark.asyncio
	async def test_session_context_manager(self, database_manager):
		"""Test session context manager."""
		async with database_manager.get_session() as session:
			assert session is not None
			assert hasattr(session, 'execute')
			assert hasattr(session, 'commit')
			assert hasattr(session, 'rollback')
	
	@pytest.mark.asyncio
	async def test_session_auto_cleanup(self, database_manager):
		"""Test session automatic cleanup."""
		session_id = None
		
		async with database_manager.get_session() as session:
			session_id = id(session)
			# Session should be active within context
			assert not session.is_closed
		
		# Session should be cleaned up after context
		# Note: We can't easily test session closure without implementation details

class TestCRWorkflowModel:
	"""Test CRWorkflow SQLAlchemy model."""
	
	@pytest.mark.asyncio
	async def test_create_workflow(self, db_session, sample_workflow):
		"""Test creating workflow in database."""
		db_workflow = CRWorkflow(
			id=sample_workflow.id,
			name=sample_workflow.name,
			description=sample_workflow.description,
			definition=sample_workflow.model_dump(),
			status=sample_workflow.status.value,
			priority=sample_workflow.priority.value,
			tenant_id=sample_workflow.tenant_id,
			created_by=sample_workflow.created_by,
			updated_by=sample_workflow.updated_by,
			created_at=sample_workflow.created_at,
			updated_at=sample_workflow.updated_at,
			tags=sample_workflow.tags,
			configuration=sample_workflow.configuration or {},
			metadata=sample_workflow.metadata or {}
		)
		
		db_session.add(db_workflow)
		await db_session.commit()
		await db_session.refresh(db_workflow)
		
		assert db_workflow.id == sample_workflow.id
		assert db_workflow.name == sample_workflow.name
		assert db_workflow.status == sample_workflow.status.value
		assert db_workflow.tenant_id == sample_workflow.tenant_id
		assert db_workflow.created_at is not None
	
	@pytest.mark.asyncio
	async def test_workflow_constraints(self, db_session, sample_workflow):
		"""Test workflow database constraints."""
		# Test unique constraint on name per tenant
		db_workflow1 = CRWorkflow(
			id=sample_workflow.id,
			name="Unique Workflow",
			definition=sample_workflow.model_dump(),
			status=sample_workflow.status.value,
			priority=sample_workflow.priority.value,
			tenant_id="tenant_1",
			created_by=sample_workflow.created_by,
			updated_by=sample_workflow.updated_by
		)
		
		db_workflow2 = CRWorkflow(
			id="different_id",
			name="Unique Workflow",  # Same name
			definition=sample_workflow.model_dump(),
			status=sample_workflow.status.value,
			priority=sample_workflow.priority.value,
			tenant_id="tenant_1",  # Same tenant - should fail
			created_by=sample_workflow.created_by,
			updated_by=sample_workflow.updated_by
		)
		
		db_session.add(db_workflow1)
		await db_session.commit()
		
		db_session.add(db_workflow2)
		with pytest.raises(IntegrityError):
			await db_session.commit()
	
	@pytest.mark.asyncio
	async def test_workflow_soft_delete(self, db_session, sample_workflow):
		"""Test workflow soft delete functionality."""
		db_workflow = CRWorkflow(
			id=sample_workflow.id,
			name=sample_workflow.name,
			definition=sample_workflow.model_dump(),
			status=sample_workflow.status.value,
			priority=sample_workflow.priority.value,
			tenant_id=sample_workflow.tenant_id,
			created_by=sample_workflow.created_by,
			updated_by=sample_workflow.updated_by
		)
		
		db_session.add(db_workflow)
		await db_session.commit()
		
		# Soft delete
		db_workflow.deleted_at = datetime.now(timezone.utc)
		db_workflow.deleted_by = "test_user"
		await db_session.commit()
		
		# Verify soft delete
		assert db_workflow.deleted_at is not None
		assert db_workflow.deleted_by == "test_user"
		
		# Query should exclude soft-deleted records
		result = await db_session.execute(
			select(CRWorkflow).where(
				and_(
					CRWorkflow.id == sample_workflow.id,
					CRWorkflow.deleted_at.is_(None)
				)
			)
		)
		assert result.scalar_one_or_none() is None

class TestCRWorkflowInstanceModel:
	"""Test CRWorkflowInstance SQLAlchemy model."""
	
	@pytest.mark.asyncio
	async def test_create_workflow_instance(self, db_session, sample_workflow, sample_workflow_instance):
		"""Test creating workflow instance in database."""
		# Create workflow first
		db_workflow = CRWorkflow(
			id=sample_workflow.id,
			name=sample_workflow.name,
			definition=sample_workflow.model_dump(),
			status=sample_workflow.status.value,
			priority=sample_workflow.priority.value,
			tenant_id=sample_workflow.tenant_id,
			created_by=sample_workflow.created_by,
			updated_by=sample_workflow.updated_by
		)
		db_session.add(db_workflow)
		await db_session.commit()
		
		# Create workflow instance
		db_instance = CRWorkflowInstance(
			id=sample_workflow_instance.id,
			workflow_id=sample_workflow.id,
			status=sample_workflow_instance.status.value,
			priority=sample_workflow_instance.priority.value,
			input_data=sample_workflow_instance.input_data,
			configuration_overrides=sample_workflow_instance.configuration_overrides,
			tenant_id=sample_workflow_instance.tenant_id,
			started_by=sample_workflow_instance.started_by,
			created_at=sample_workflow_instance.created_at,
			tags=sample_workflow_instance.tags
		)
		
		db_session.add(db_instance)
		await db_session.commit()
		await db_session.refresh(db_instance)
		
		assert db_instance.id == sample_workflow_instance.id
		assert db_instance.workflow_id == sample_workflow.id
		assert db_instance.status == sample_workflow_instance.status.value
		assert db_instance.input_data == sample_workflow_instance.input_data
	
	@pytest.mark.asyncio
	async def test_workflow_instance_relationships(self, db_session, sample_workflow, sample_workflow_instance):
		"""Test workflow instance foreign key relationships."""
		# Create workflow
		db_workflow = CRWorkflow(
			id=sample_workflow.id,
			name=sample_workflow.name,
			definition=sample_workflow.model_dump(),
			status=sample_workflow.status.value,
			priority=sample_workflow.priority.value,
			tenant_id=sample_workflow.tenant_id,
			created_by=sample_workflow.created_by,
			updated_by=sample_workflow.updated_by
		)
		db_session.add(db_workflow)
		await db_session.commit()
		
		# Create instance
		db_instance = CRWorkflowInstance(
			id=sample_workflow_instance.id,
			workflow_id=sample_workflow.id,
			status=sample_workflow_instance.status.value,
			tenant_id=sample_workflow_instance.tenant_id,
			started_by=sample_workflow_instance.started_by
		)
		db_session.add(db_instance)
		await db_session.commit()
		
		# Test relationship query
		result = await db_session.execute(
			select(CRWorkflowInstance, CRWorkflow)
			.join(CRWorkflow)
			.where(CRWorkflowInstance.id == sample_workflow_instance.id)
		)
		
		instance, workflow = result.first()
		assert instance.workflow_id == workflow.id
		assert workflow.name == sample_workflow.name
	
	@pytest.mark.asyncio
	async def test_instance_progress_tracking(self, db_session, sample_workflow_instance):
		"""Test instance progress tracking."""
		db_instance = CRWorkflowInstance(
			id=sample_workflow_instance.id,
			workflow_id="dummy_workflow_id",
			status=WorkflowStatus.RUNNING.value,
			progress_percentage=0,
			tenant_id=sample_workflow_instance.tenant_id,
			started_by=sample_workflow_instance.started_by,
			started_at=datetime.now(timezone.utc)
		)
		
		db_session.add(db_instance)
		await db_session.commit()
		
		# Update progress
		db_instance.progress_percentage = 50
		await db_session.commit()
		
		assert db_instance.progress_percentage == 50
		
		# Complete instance
		db_instance.status = WorkflowStatus.COMPLETED.value
		db_instance.progress_percentage = 100
		db_instance.completed_at = datetime.now(timezone.utc)
		await db_session.commit()
		
		assert db_instance.status == WorkflowStatus.COMPLETED.value
		assert db_instance.progress_percentage == 100
		assert db_instance.completed_at is not None

class TestCRTaskExecutionModel:
	"""Test CRTaskExecution SQLAlchemy model."""
	
	@pytest.mark.asyncio
	async def test_create_task_execution(self, db_session, sample_task_execution, sample_workflow_instance):
		"""Test creating task execution in database."""
		# Create instance first
		db_instance = CRWorkflowInstance(
			id=sample_workflow_instance.id,
			workflow_id="dummy_workflow_id",
			status=sample_workflow_instance.status.value,
			tenant_id=sample_workflow_instance.tenant_id,
			started_by=sample_workflow_instance.started_by
		)
		db_session.add(db_instance)
		await db_session.commit()
		
		# Create task execution
		db_execution = CRTaskExecution(
			id=sample_task_execution.id,
			instance_id=sample_workflow_instance.id,
			task_id=sample_task_execution.task_id,
			task_definition=sample_task_execution.task_definition.model_dump(),
			status=sample_task_execution.status.value,
			input_data=sample_task_execution.input_data,
			configuration=sample_task_execution.configuration,
			tenant_id=sample_task_execution.tenant_id,
			created_at=sample_task_execution.created_at
		)
		
		db_session.add(db_execution)
		await db_session.commit()
		await db_session.refresh(db_execution)
		
		assert db_execution.id == sample_task_execution.id
		assert db_execution.instance_id == sample_workflow_instance.id
		assert db_execution.task_id == sample_task_execution.task_id
		assert db_execution.status == sample_task_execution.status.value
	
	@pytest.mark.asyncio
	async def test_task_execution_timing(self, db_session, sample_task_execution):
		"""Test task execution timing fields."""
		start_time = datetime.now(timezone.utc)
		
		db_execution = CRTaskExecution(
			id=sample_task_execution.id,
			instance_id="dummy_instance_id",
			task_id=sample_task_execution.task_id,
			task_definition=sample_task_execution.task_definition.model_dump(),
			status=TaskStatus.RUNNING.value,
			tenant_id=sample_task_execution.tenant_id,
			started_at=start_time
		)
		
		db_session.add(db_execution)
		await db_session.commit()
		
		# Complete task
		end_time = start_time + timedelta(minutes=5)
		db_execution.status = TaskStatus.COMPLETED.value
		db_execution.completed_at = end_time
		db_execution.output_data = {"result": "success"}
		
		await db_session.commit()
		
		assert db_execution.started_at == start_time
		assert db_execution.completed_at == end_time
		assert db_execution.output_data == {"result": "success"}
	
	@pytest.mark.asyncio
	async def test_task_execution_retry_tracking(self, db_session, sample_task_execution):
		"""Test task execution retry tracking."""
		db_execution = CRTaskExecution(
			id=sample_task_execution.id,
			instance_id="dummy_instance_id",
			task_id=sample_task_execution.task_id,
			task_definition=sample_task_execution.task_definition.model_dump(),
			status=TaskStatus.FAILED.value,
			retry_count=0,
			tenant_id=sample_task_execution.tenant_id
		)
		
		db_session.add(db_execution)
		await db_session.commit()
		
		# Increment retry count
		db_execution.retry_count = 1
		db_execution.status = TaskStatus.RUNNING.value
		db_execution.error_details = None
		await db_session.commit()
		
		assert db_execution.retry_count == 1
		assert db_execution.status == TaskStatus.RUNNING.value
		
		# Fail again
		db_execution.retry_count = 2
		db_execution.status = TaskStatus.FAILED.value
		db_execution.error_details = {"error": "Connection timeout", "code": "TIMEOUT"}
		await db_session.commit()
		
		assert db_execution.retry_count == 2
		assert db_execution.error_details["error"] == "Connection timeout"

class TestWorkflowRepository:
	"""Test WorkflowRepository functionality."""
	
	@pytest.mark.asyncio
	async def test_create_workflow(self, database_manager, sample_workflow):
		"""Test repository create workflow."""
		repo = WorkflowRepository(database_manager)
		
		created_workflow = await repo.create(sample_workflow)
		
		assert created_workflow.id == sample_workflow.id
		assert created_workflow.name == sample_workflow.name
		assert created_workflow.tenant_id == sample_workflow.tenant_id
	
	@pytest.mark.asyncio
	async def test_get_workflow_by_id(self, database_manager, sample_workflow):
		"""Test repository get workflow by ID."""
		repo = WorkflowRepository(database_manager)
		
		# Create workflow
		await repo.create(sample_workflow)
		
		# Get workflow
		retrieved_workflow = await repo.get_by_id(sample_workflow.id, sample_workflow.tenant_id)
		
		assert retrieved_workflow is not None
		assert retrieved_workflow.id == sample_workflow.id
		assert retrieved_workflow.name == sample_workflow.name
	
	@pytest.mark.asyncio
	async def test_update_workflow(self, database_manager, sample_workflow):
		"""Test repository update workflow."""
		repo = WorkflowRepository(database_manager)
		
		# Create workflow
		await repo.create(sample_workflow)
		
		# Update workflow
		updates = {
			"name": "Updated Workflow Name",
			"description": "Updated description",
			"priority": Priority.HIGH.value
		}
		
		updated_workflow = await repo.update(sample_workflow.id, sample_workflow.tenant_id, updates)
		
		assert updated_workflow is not None
		assert updated_workflow.name == "Updated Workflow Name"
		assert updated_workflow.description == "Updated description"
	
	@pytest.mark.asyncio
	async def test_delete_workflow(self, database_manager, sample_workflow):
		"""Test repository delete workflow."""
		repo = WorkflowRepository(database_manager)
		
		# Create workflow
		await repo.create(sample_workflow)
		
		# Delete workflow (soft delete)
		success = await repo.delete(sample_workflow.id, sample_workflow.tenant_id, "test_user")
		
		assert success is True
		
		# Verify soft delete
		deleted_workflow = await repo.get_by_id(sample_workflow.id, sample_workflow.tenant_id)
		assert deleted_workflow is None  # Should not return soft-deleted
	
	@pytest.mark.asyncio
	async def test_search_workflows(self, database_manager):
		"""Test repository search workflows."""
		repo = WorkflowRepository(database_manager)
		
		# Create multiple workflows
		workflows = []
		for i in range(5):
			workflow_data = {
				"name": f"Test Workflow {i}",
				"description": f"Description {i}",
				"tasks": [
					{
						"name": f"Task {i}",
						"task_type": "task",
						"configuration": {"step": i}
					}
				],
				"priority": Priority.MEDIUM if i % 2 == 0 else Priority.HIGH,
				"tenant_id": "test_tenant",
				"created_by": "test_user",
				"updated_by": "test_user",
				"tags": [f"tag_{i}"] if i % 2 == 0 else ["common_tag"]
			}
			workflow = Workflow(**workflow_data)
			workflows.append(workflow)
			await repo.create(workflow)
		
		# Test search by name pattern
		results = await repo.search(
			tenant_id="test_tenant",
			name_pattern="Test Workflow%",
			limit=10
		)
		assert len(results) == 5
		
		# Test search by priority
		high_priority_results = await repo.search(
			tenant_id="test_tenant",
			priority=[Priority.HIGH.value],
			limit=10
		)
		assert len(high_priority_results) == 2  # Workflows 1 and 3
		
		# Test search by tags
		tag_results = await repo.search(
			tenant_id="test_tenant",
			tags=["common_tag"],
			limit=10
		)
		assert len(tag_results) == 2  # Workflows 1 and 3
	
	@pytest.mark.asyncio
	async def test_get_workflow_statistics(self, database_manager, sample_workflow):
		"""Test repository get workflow statistics."""
		repo = WorkflowRepository(database_manager)
		instance_repo = WorkflowInstanceRepository(database_manager)
		
		# Create workflow
		await repo.create(sample_workflow)
		
		# Create instances with different statuses
		for i, status in enumerate([WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.RUNNING]):
			instance_data = {
				"workflow_id": sample_workflow.id,
				"status": status,
				"input_data": {"test": f"data_{i}"},
				"tenant_id": sample_workflow.tenant_id,
				"started_by": "test_user"
			}
			if status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]:
				instance_data["started_at"] = datetime.now(timezone.utc) - timedelta(hours=1)
				instance_data["completed_at"] = datetime.now(timezone.utc)
			
			instance = WorkflowInstance(**instance_data)
			await instance_repo.create(instance)
		
		# Get statistics
		stats = await repo.get_statistics(sample_workflow.id, sample_workflow.tenant_id, days=30)
		
		assert stats["total_executions"] == 3
		assert stats["successful_executions"] == 1
		assert stats["failed_executions"] == 1
		assert stats["active_instances"] == 1

class TestWorkflowInstanceRepository:
	"""Test WorkflowInstanceRepository functionality."""
	
	@pytest.mark.asyncio
	async def test_create_instance(self, database_manager, sample_workflow, sample_workflow_instance):
		"""Test repository create workflow instance."""
		workflow_repo = WorkflowRepository(database_manager)
		instance_repo = WorkflowInstanceRepository(database_manager)
		
		# Create workflow first
		await workflow_repo.create(sample_workflow)
		
		# Create instance
		created_instance = await instance_repo.create(sample_workflow_instance)
		
		assert created_instance.id == sample_workflow_instance.id
		assert created_instance.workflow_id == sample_workflow.id
		assert created_instance.status == sample_workflow_instance.status
	
	@pytest.mark.asyncio
	async def test_update_instance_progress(self, database_manager, sample_workflow, sample_workflow_instance):
		"""Test repository update instance progress."""
		workflow_repo = WorkflowRepository(database_manager)
		instance_repo = WorkflowInstanceRepository(database_manager)
		
		# Create workflow and instance
		await workflow_repo.create(sample_workflow)
		await instance_repo.create(sample_workflow_instance)
		
		# Update progress
		updated_instance = await instance_repo.update_progress(
			sample_workflow_instance.id,
			sample_workflow_instance.tenant_id,
			progress_percentage=75
		)
		
		assert updated_instance is not None
		assert updated_instance.progress_percentage == 75
	
	@pytest.mark.asyncio
	async def test_get_instances_by_workflow(self, database_manager, sample_workflow):
		"""Test repository get instances by workflow ID."""
		workflow_repo = WorkflowRepository(database_manager)
		instance_repo = WorkflowInstanceRepository(database_manager)
		
		# Create workflow
		await workflow_repo.create(sample_workflow)
		
		# Create multiple instances
		instances = []
		for i in range(3):
			instance_data = {
				"workflow_id": sample_workflow.id,
				"input_data": {"test": f"data_{i}"},
				"tenant_id": sample_workflow.tenant_id,
				"started_by": "test_user"
			}
			instance = WorkflowInstance(**instance_data)
			instances.append(instance)
			await instance_repo.create(instance)
		
		# Get instances
		retrieved_instances = await instance_repo.get_by_workflow_id(
			sample_workflow.id,
			sample_workflow.tenant_id,
			limit=10
		)
		
		assert len(retrieved_instances) == 3
		assert all(inst.workflow_id == sample_workflow.id for inst in retrieved_instances)

class TestTaskExecutionRepository:
	"""Test TaskExecutionRepository functionality."""
	
	@pytest.mark.asyncio
	async def test_create_task_execution(self, database_manager, sample_workflow, sample_workflow_instance, sample_task_execution):
		"""Test repository create task execution."""
		workflow_repo = WorkflowRepository(database_manager)
		instance_repo = WorkflowInstanceRepository(database_manager)
		execution_repo = TaskExecutionRepository(database_manager)
		
		# Create workflow and instance
		await workflow_repo.create(sample_workflow)
		await instance_repo.create(sample_workflow_instance)
		
		# Create task execution
		created_execution = await execution_repo.create(sample_task_execution)
		
		assert created_execution.id == sample_task_execution.id
		assert created_execution.instance_id == sample_workflow_instance.id
		assert created_execution.task_id == sample_task_execution.task_id
	
	@pytest.mark.asyncio
	async def test_update_task_execution_status(self, database_manager, sample_workflow, sample_workflow_instance, sample_task_execution):
		"""Test repository update task execution status."""
		workflow_repo = WorkflowRepository(database_manager)
		instance_repo = WorkflowInstanceRepository(database_manager)
		execution_repo = TaskExecutionRepository(database_manager)
		
		# Create workflow, instance, and execution
		await workflow_repo.create(sample_workflow)
		await instance_repo.create(sample_workflow_instance)
		await execution_repo.create(sample_task_execution)
		
		# Update status
		updated_execution = await execution_repo.update_status(
			sample_task_execution.id,
			sample_task_execution.tenant_id,
			TaskStatus.COMPLETED,
			output_data={"result": "success"}
		)
		
		assert updated_execution is not None
		assert updated_execution.status == TaskStatus.COMPLETED
		assert updated_execution.output_data == {"result": "success"}
		assert updated_execution.completed_at is not None
	
	@pytest.mark.asyncio
	async def test_get_executions_by_instance(self, database_manager, sample_workflow, sample_workflow_instance):
		"""Test repository get executions by instance ID."""
		workflow_repo = WorkflowRepository(database_manager)
		instance_repo = WorkflowInstanceRepository(database_manager)
		execution_repo = TaskExecutionRepository(database_manager)
		
		# Create workflow and instance
		await workflow_repo.create(sample_workflow)
		await instance_repo.create(sample_workflow_instance)
		
		# Create multiple task executions
		executions = []
		for i, task in enumerate(sample_workflow.tasks):
			execution_data = {
				"instance_id": sample_workflow_instance.id,
				"task_id": task.id,
				"task_definition": task,
				"input_data": {"test": f"data_{i}"},
				"tenant_id": sample_workflow_instance.tenant_id
			}
			execution = TaskExecution(**execution_data)
			executions.append(execution)
			await execution_repo.create(execution)
		
		# Get executions
		retrieved_executions = await execution_repo.get_by_instance_id(
			sample_workflow_instance.id,
			sample_workflow_instance.tenant_id
		)
		
		assert len(retrieved_executions) == len(sample_workflow.tasks)
		assert all(exec.instance_id == sample_workflow_instance.id for exec in retrieved_executions)

class TestDatabaseIntegration:
	"""Test database integration and complex queries."""
	
	@pytest.mark.asyncio
	async def test_multi_tenant_isolation(self, database_manager, sample_workflow):
		"""Test multi-tenant data isolation."""
		repo = WorkflowRepository(database_manager)
		
		# Create workflows for different tenants
		workflow_tenant1 = sample_workflow
		workflow_tenant1.tenant_id = "tenant_1"
		
		workflow_tenant2_data = sample_workflow.model_dump()
		workflow_tenant2_data["tenant_id"] = "tenant_2"
		workflow_tenant2_data["id"] = "different_workflow_id"
		workflow_tenant2 = Workflow(**workflow_tenant2_data)
		
		await repo.create(workflow_tenant1)
		await repo.create(workflow_tenant2)
		
		# Test tenant isolation
		tenant1_workflows = await repo.search(tenant_id="tenant_1", limit=10)
		tenant2_workflows = await repo.search(tenant_id="tenant_2", limit=10)
		
		assert len(tenant1_workflows) == 1
		assert len(tenant2_workflows) == 1
		assert tenant1_workflows[0].tenant_id == "tenant_1"
		assert tenant2_workflows[0].tenant_id == "tenant_2"
	
	@pytest.mark.asyncio
	async def test_complex_workflow_queries(self, database_manager, complex_workflow):
		"""Test complex workflow queries."""
		repo = WorkflowRepository(database_manager)
		
		# Create complex workflow
		await repo.create(complex_workflow)
		
		# Query with multiple filters
		results = await repo.search(
			tenant_id=complex_workflow.tenant_id,
			name_pattern="Complex%",
			priority=[Priority.HIGH.value],
			tags=["integration"],
			limit=10
		)
		
		assert len(results) == 1
		assert results[0].name == complex_workflow.name
		assert results[0].priority == Priority.HIGH
		assert "integration" in results[0].tags
	
	@pytest.mark.asyncio
	async def test_concurrent_operations(self, database_manager, sample_workflow):
		"""Test concurrent database operations."""
		repo = WorkflowRepository(database_manager)
		
		# Create multiple workflows concurrently
		async def create_workflow(index: int):
			workflow_data = sample_workflow.model_dump()
			workflow_data["id"] = f"concurrent_workflow_{index}"
			workflow_data["name"] = f"Concurrent Workflow {index}"
			workflow = Workflow(**workflow_data)
			return await repo.create(workflow)
		
		# Run concurrent operations
		tasks = [create_workflow(i) for i in range(5)]
		results = await asyncio.gather(*tasks)
		
		assert len(results) == 5
		assert all(result is not None for result in results)
		
		# Verify all workflows were created
		all_workflows = await repo.search(
			tenant_id=sample_workflow.tenant_id,
			name_pattern="Concurrent Workflow%",
			limit=10
		)
		assert len(all_workflows) == 5