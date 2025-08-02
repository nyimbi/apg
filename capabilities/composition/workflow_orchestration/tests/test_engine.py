"""
APG Workflow Orchestration Engine Tests

Comprehensive unit tests for workflow execution engine, scheduler, and state manager.
Tests core workflow orchestration logic, task execution, and state management.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch, call
import json

from ..engine import WorkflowExecutor, TaskScheduler, StateManager
from ..engine.bpml_engine import BPMLEngine, BPMLParser
from ..models import (
	Workflow, WorkflowInstance, TaskDefinition, TaskExecution,
	WorkflowStatus, TaskStatus, Priority, TaskType
)

class TestWorkflowExecutor:
	"""Test WorkflowExecutor functionality."""
	
	@pytest.mark.asyncio
	async def test_executor_initialization(self, database_manager, redis_client):
		"""Test executor initialization."""
		executor = WorkflowExecutor(database_manager, redis_client, "test_tenant")
		
		assert executor.tenant_id == "test_tenant"
		assert executor.database_manager == database_manager
		assert executor.redis_client == redis_client
		assert executor.active_instances == {}
		assert executor.task_executors == {}
	
	@pytest.mark.asyncio
	async def test_execute_workflow_simple(self, workflow_executor, sample_workflow):
		"""Test simple workflow execution."""
		with patch.object(workflow_executor.repositories['workflow'], 'get_by_id') as mock_get_workflow, \
			 patch.object(workflow_executor.repositories['workflow_instance'], 'create') as mock_create_instance, \
			 patch.object(workflow_executor, '_start_execution_loop') as mock_start_loop:
			
			mock_get_workflow.return_value = sample_workflow
			mock_instance = WorkflowInstance(
				workflow_id=sample_workflow.id,
				input_data={"test": "data"},
				status=WorkflowStatus.RUNNING,
				tenant_id="test_tenant",
				started_by="test_user"
			)
			mock_create_instance.return_value = mock_instance
			
			# Execute workflow
			result = await workflow_executor.execute_workflow(
				workflow_id=sample_workflow.id,
				input_data={"test": "data"},
				user_id="test_user"
			)
			
			assert result is not None
			assert result.workflow_id == sample_workflow.id
			assert result.status == WorkflowStatus.RUNNING
			mock_get_workflow.assert_called_once()
			mock_create_instance.assert_called_once()
			mock_start_loop.assert_called_once()
	
	@pytest.mark.asyncio
	async def test_execute_workflow_with_validation(self, workflow_executor, sample_workflow):
		"""Test workflow execution with input validation."""
		# Mock workflow with input schema
		workflow_with_schema = sample_workflow.model_copy()
		workflow_with_schema.configuration = {
			"input_schema": {
				"type": "object",
				"properties": {
					"required_field": {"type": "string"}
				},
				"required": ["required_field"]
			}
		}
		
		with patch.object(workflow_executor.repositories['workflow'], 'get_by_id') as mock_get:
			mock_get.return_value = workflow_with_schema
			
			# Test invalid input
			with pytest.raises(ValueError, match="Input validation failed"):
				await workflow_executor.execute_workflow(
					workflow_id=sample_workflow.id,
					input_data={"invalid": "data"},  # Missing required_field
					user_id="test_user"
				)
	
	@pytest.mark.asyncio
	async def test_execute_workflow_not_found(self, workflow_executor):
		"""Test executing non-existent workflow."""
		with patch.object(workflow_executor.repositories['workflow'], 'get_by_id') as mock_get:
			mock_get.return_value = None
			
			with pytest.raises(ValueError, match="Workflow not found"):
				await workflow_executor.execute_workflow(
					workflow_id="nonexistent_id",
					input_data={},
					user_id="test_user"
				)
	
	@pytest.mark.asyncio
	async def test_execute_task_success(self, workflow_executor, sample_task_execution):
		"""Test successful task execution."""
		with patch.object(workflow_executor, '_get_task_executor') as mock_get_executor, \
			 patch.object(workflow_executor.repositories['task_execution'], 'update_status') as mock_update:
			
			# Mock task executor
			mock_executor = AsyncMock()
			mock_executor.execute.return_value = {"result": "success"}
			mock_get_executor.return_value = mock_executor
			
			# Mock status update
			completed_execution = sample_task_execution.model_copy()
			completed_execution.status = TaskStatus.COMPLETED
			completed_execution.output_data = {"result": "success"}
			mock_update.return_value = completed_execution
			
			# Execute task
			result = await workflow_executor.execute_task(sample_task_execution)
			
			assert result is not None
			assert result.status == TaskStatus.COMPLETED
			assert result.output_data == {"result": "success"}
			mock_executor.execute.assert_called_once()
			mock_update.assert_called_once()
	
	@pytest.mark.asyncio
	async def test_execute_task_failure(self, workflow_executor, sample_task_execution):
		"""Test task execution failure and retry logic."""
		with patch.object(workflow_executor, '_get_task_executor') as mock_get_executor, \
			 patch.object(workflow_executor.repositories['task_execution'], 'update_status') as mock_update:
			
			# Mock failing task executor
			mock_executor = AsyncMock()
			mock_executor.execute.side_effect = Exception("Task execution failed")
			mock_get_executor.return_value = mock_executor
			
			# Mock status updates
			failed_execution = sample_task_execution.model_copy()
			failed_execution.status = TaskStatus.FAILED
			failed_execution.error_details = {"error": "Task execution failed"}
			mock_update.return_value = failed_execution
			
			# Execute task (should fail)
			result = await workflow_executor.execute_task(sample_task_execution)
			
			assert result.status == TaskStatus.FAILED
			assert "Task execution failed" in str(result.error_details)
			mock_executor.execute.assert_called_once()
	
	@pytest.mark.asyncio
	async def test_execute_task_with_retries(self, workflow_executor, sample_task_execution):
		"""Test task execution with retry logic."""
		# Set up task with retries
		retry_task = sample_task_execution.model_copy()
		retry_task.task_definition.max_retries = 2
		
		with patch.object(workflow_executor, '_get_task_executor') as mock_get_executor, \
			 patch.object(workflow_executor.repositories['task_execution'], 'update_status') as mock_update, \
			 patch.object(workflow_executor, '_should_retry_task') as mock_should_retry:
			
			# Mock task executor that fails then succeeds
			mock_executor = AsyncMock()
			mock_executor.execute.side_effect = [
				Exception("Temporary failure"),  # First attempt fails
				{"result": "success"}  # Retry succeeds
			]
			mock_get_executor.return_value = mock_executor
			
			# Mock retry logic
			mock_should_retry.return_value = True
			
			# Mock status updates
			mock_update.side_effect = [
				retry_task,  # First failure
				retry_task.model_copy()  # Retry success
			]
			
			# Execute task with retry
			await workflow_executor.execute_task(retry_task)
			
			assert mock_executor.execute.call_count == 2  # Initial + 1 retry
			assert mock_update.call_count >= 2
	
	@pytest.mark.asyncio
	async def test_pause_instance(self, workflow_executor, sample_workflow_instance):
		"""Test pausing workflow instance."""
		# Add instance to active instances
		workflow_executor.active_instances[sample_workflow_instance.id] = {
			"status": "running",
			"tasks": {}
		}
		
		with patch.object(workflow_executor.repositories['workflow_instance'], 'update_status') as mock_update:
			paused_instance = sample_workflow_instance.model_copy()
			paused_instance.status = WorkflowStatus.PAUSED
			mock_update.return_value = paused_instance
			
			# Pause instance
			result = await workflow_executor.pause_instance(
				sample_workflow_instance.id,
				"test_user"
			)
			
			assert result is True
			assert workflow_executor.active_instances[sample_workflow_instance.id]["status"] == "paused"
			mock_update.assert_called_once()
	
	@pytest.mark.asyncio
	async def test_resume_instance(self, workflow_executor, sample_workflow_instance):
		"""Test resuming paused workflow instance."""
		# Add paused instance to active instances
		workflow_executor.active_instances[sample_workflow_instance.id] = {
			"status": "paused",
			"tasks": {}
		}
		
		with patch.object(workflow_executor.repositories['workflow_instance'], 'update_status') as mock_update, \
			 patch.object(workflow_executor, '_continue_execution') as mock_continue:
			
			resumed_instance = sample_workflow_instance.model_copy()
			resumed_instance.status = WorkflowStatus.RUNNING
			mock_update.return_value = resumed_instance
			
			# Resume instance
			result = await workflow_executor.resume_instance(
				sample_workflow_instance.id,
				"test_user"
			)
			
			assert result is True
			assert workflow_executor.active_instances[sample_workflow_instance.id]["status"] == "running"
			mock_update.assert_called_once()
			mock_continue.assert_called_once()
	
	@pytest.mark.asyncio
	async def test_stop_instance(self, workflow_executor, sample_workflow_instance):
		"""Test stopping workflow instance."""
		# Add instance to active instances
		workflow_executor.active_instances[sample_workflow_instance.id] = {
			"status": "running",
			"tasks": {"task_1": "running"}
		}
		
		with patch.object(workflow_executor.repositories['workflow_instance'], 'update_status') as mock_update, \
			 patch.object(workflow_executor, '_cancel_running_tasks') as mock_cancel:
			
			stopped_instance = sample_workflow_instance.model_copy()
			stopped_instance.status = WorkflowStatus.CANCELLED
			mock_update.return_value = stopped_instance
			
			# Stop instance
			result = await workflow_executor.stop_instance(
				sample_workflow_instance.id,
				"test_user",
				"User requested stop"
			)
			
			assert result is True
			assert sample_workflow_instance.id not in workflow_executor.active_instances
			mock_update.assert_called_once()
			mock_cancel.assert_called_once()
	
	@pytest.mark.asyncio
	async def test_workflow_compensation_logic(self, workflow_executor, complex_workflow):
		"""Test workflow compensation logic on failure."""
		# Create instance with some completed tasks
		instance = WorkflowInstance(
			workflow_id=complex_workflow.id,
			input_data={"test": "data"},
			status=WorkflowStatus.RUNNING,
			tenant_id="test_tenant",
			started_by="test_user"
		)
		
		# Mock completed tasks that need compensation
		completed_tasks = [
			TaskExecution(
				instance_id=instance.id,
				task_id="task_1",
				task_definition=complex_workflow.tasks[0],
				status=TaskStatus.COMPLETED,
				output_data={"created_resource": "resource_123"},
				tenant_id="test_tenant"
			)
		]
		
		with patch.object(workflow_executor.repositories['task_execution'], 'get_by_instance_id') as mock_get_tasks, \
			 patch.object(workflow_executor, '_execute_compensation') as mock_compensate:
			
			mock_get_tasks.return_value = completed_tasks
			
			# Trigger compensation
			await workflow_executor._handle_workflow_failure(instance.id, "Task failure")
			
			mock_get_tasks.assert_called_once()
			mock_compensate.assert_called_once()

class TestTaskScheduler:
	"""Test TaskScheduler functionality."""
	
	@pytest.mark.asyncio
	async def test_scheduler_initialization(self, database_manager, redis_client):
		"""Test scheduler initialization."""
		scheduler = TaskScheduler(database_manager, redis_client, "test_tenant")
		
		assert scheduler.tenant_id == "test_tenant"
		assert scheduler.database_manager == database_manager
		assert scheduler.redis_client == redis_client
		assert scheduler.task_queue == []
		assert scheduler.worker_nodes == {}
	
	@pytest.mark.asyncio
	async def test_schedule_task_immediate(self, task_scheduler, sample_task_execution):
		"""Test immediate task scheduling."""
		with patch.object(task_scheduler, '_add_to_queue') as mock_add_queue:
			# Schedule task immediately
			await task_scheduler.schedule_task(
				sample_task_execution,
				schedule_at=None,
				priority=Priority.HIGH
			)
			
			mock_add_queue.assert_called_once()
			# Verify task was added with correct priority
			args = mock_add_queue.call_args[0]
			assert args[0] == sample_task_execution
			assert args[1] == Priority.HIGH
	
	@pytest.mark.asyncio
	async def test_schedule_task_future(self, task_scheduler, sample_task_execution):
		"""Test future task scheduling."""
		future_time = datetime.now(timezone.utc) + timedelta(hours=1)
		
		with patch.object(task_scheduler, '_schedule_delayed_task') as mock_schedule_delayed:
			# Schedule task for future execution
			await task_scheduler.schedule_task(
				sample_task_execution,
				schedule_at=future_time,
				priority=Priority.MEDIUM
			)
			
			mock_schedule_delayed.assert_called_once_with(
				sample_task_execution,
				future_time,
				Priority.MEDIUM
			)
	
	@pytest.mark.asyncio
	async def test_schedule_workflow(self, task_scheduler, sample_workflow, sample_workflow_instance):
		"""Test workflow scheduling."""
		future_time = datetime.now(timezone.utc) + timedelta(minutes=30)
		
		with patch.object(task_scheduler.repositories['workflow_instance'], 'create') as mock_create, \
			 patch.object(task_scheduler, '_schedule_workflow_execution') as mock_schedule:
			
			scheduled_instance = sample_workflow_instance.model_copy()
			scheduled_instance.status = WorkflowStatus.SCHEDULED
			mock_create.return_value = scheduled_instance
			
			# Schedule workflow
			result = await task_scheduler.schedule_workflow(
				workflow_id=sample_workflow.id,
				input_data={"test": "data"},
				user_id="test_user",
				schedule_at=future_time
			)
			
			assert result.status == WorkflowStatus.SCHEDULED
			mock_create.assert_called_once()
			mock_schedule.assert_called_once()
	
	@pytest.mark.asyncio
	async def test_priority_queue_ordering(self, task_scheduler):
		"""Test priority-based task queue ordering."""
		# Create tasks with different priorities
		high_priority_task = TaskExecution(
			instance_id="instance_1",
			task_id="high_task",
			task_definition=TaskDefinition(name="High Priority", task_type=TaskType.TASK),
			tenant_id="test_tenant"
		)
		
		low_priority_task = TaskExecution(
			instance_id="instance_2",
			task_id="low_task",
			task_definition=TaskDefinition(name="Low Priority", task_type=TaskType.TASK),
			tenant_id="test_tenant"
		)
		
		# Add tasks in reverse priority order
		await task_scheduler._add_to_queue(low_priority_task, Priority.LOW)
		await task_scheduler._add_to_queue(high_priority_task, Priority.HIGH)
		
		# Get next task should return high priority first
		next_task = await task_scheduler._get_next_task()
		assert next_task.task_id == "high_task"
		
		# Second call should return low priority task
		next_task = await task_scheduler._get_next_task()
		assert next_task.task_id == "low_task"
	
	@pytest.mark.asyncio
	async def test_resource_allocation(self, task_scheduler, sample_task_execution):
		"""Test resource-based task allocation."""
		# Configure resource requirements
		resource_task = sample_task_execution.model_copy()
		resource_task.task_definition.configuration = {
			"resource_requirements": {
				"cpu": "2 cores",
				"memory": "4GB",
				"gpu": False
			}
		}
		
		# Mock available resources
		with patch.object(task_scheduler, '_get_available_resources') as mock_resources, \
			 patch.object(task_scheduler, '_allocate_resources') as mock_allocate:
			
			mock_resources.return_value = {
				"cpu": "8 cores",
				"memory": "16GB",
				"gpu": True
			}
			
			# Schedule resource-intensive task
			await task_scheduler.schedule_task(resource_task)
			
			mock_resources.assert_called_once()
			mock_allocate.assert_called_once()
	
	@pytest.mark.asyncio
	async def test_worker_node_management(self, task_scheduler):
		"""Test worker node registration and management."""
		worker_info = {
			"worker_id": "worker_001",
			"capabilities": ["python", "integration", "notification"],
			"resources": {"cpu": "4 cores", "memory": "8GB"},
			"max_concurrent_tasks": 5
		}
		
		# Register worker node
		await task_scheduler.register_worker_node(worker_info)
		
		assert "worker_001" in task_scheduler.worker_nodes
		assert task_scheduler.worker_nodes["worker_001"]["capabilities"] == worker_info["capabilities"]
	
	@pytest.mark.asyncio
	async def test_task_distribution(self, task_scheduler):
		"""Test task distribution across worker nodes."""
		# Register multiple worker nodes
		workers = [
			{
				"worker_id": f"worker_{i}",
				"capabilities": ["python", "general"],
				"resources": {"cpu": "2 cores", "memory": "4GB"},
				"max_concurrent_tasks": 3
			}
			for i in range(3)
		]
		
		for worker in workers:
			await task_scheduler.register_worker_node(worker)
		
		# Create multiple tasks
		tasks = [
			TaskExecution(
				instance_id=f"instance_{i}",
				task_id=f"task_{i}",
				task_definition=TaskDefinition(name=f"Task {i}", task_type=TaskType.TASK),
				tenant_id="test_tenant"
			)
			for i in range(6)
		]
		
		with patch.object(task_scheduler, '_assign_task_to_worker') as mock_assign:
			# Schedule all tasks
			for task in tasks:
				await task_scheduler.schedule_task(task)
			
			# Should distribute tasks across workers
			assert mock_assign.call_count == len(tasks)
	
	@pytest.mark.asyncio
	async def test_scheduler_performance_optimization(self, task_scheduler):
		"""Test scheduler performance optimization."""
		# Mock performance metrics
		performance_metrics = {
			"avg_task_duration": 120,  # seconds
			"queue_wait_time": 30,
			"worker_utilization": 0.75,
			"failure_rate": 0.05
		}
		
		with patch.object(task_scheduler, '_get_performance_metrics') as mock_metrics, \
			 patch.object(task_scheduler, '_optimize_scheduling') as mock_optimize:
			
			mock_metrics.return_value = performance_metrics
			
			# Trigger performance optimization
			await task_scheduler._analyze_and_optimize_performance()
			
			mock_metrics.assert_called_once()
			mock_optimize.assert_called_once_with(performance_metrics)

class TestStateManager:
	"""Test StateManager functionality."""
	
	@pytest.mark.asyncio
	async def test_state_manager_initialization(self, database_manager, redis_client):
		"""Test state manager initialization."""
		state_manager = StateManager(database_manager, redis_client, "test_tenant")
		
		assert state_manager.tenant_id == "test_tenant"
		assert state_manager.database_manager == database_manager
		assert state_manager.redis_client == redis_client
		assert state_manager.state_cache == {}
	
	@pytest.mark.asyncio
	async def test_create_instance_state(self, state_manager, sample_workflow_instance):
		"""Test creating instance state."""
		initial_state = {
			"status": "running",
			"progress": 0,
			"current_tasks": [],
			"completed_tasks": [],
			"failed_tasks": [],
			"variables": {}
		}
		
		with patch.object(state_manager, '_persist_state') as mock_persist:
			# Create instance state
			await state_manager.create_instance_state(
				sample_workflow_instance.id,
				initial_state
			)
			
			assert sample_workflow_instance.id in state_manager.state_cache
			assert state_manager.state_cache[sample_workflow_instance.id] == initial_state
			mock_persist.assert_called_once()
	
	@pytest.mark.asyncio
	async def test_update_instance_state(self, state_manager, sample_workflow_instance):
		"""Test updating instance state."""
		# Create initial state
		initial_state = {
			"status": "running",
			"progress": 0,
			"current_tasks": ["task_1"],
			"completed_tasks": [],
			"variables": {"var1": "value1"}
		}
		
		state_manager.state_cache[sample_workflow_instance.id] = initial_state
		
		# Update state
		updates = {
			"progress": 50,
			"completed_tasks": ["task_1"],
			"current_tasks": ["task_2"],
			"variables": {"var1": "updated_value", "var2": "value2"}
		}
		
		with patch.object(state_manager, '_persist_state') as mock_persist:
			await state_manager.update_instance_state(
				sample_workflow_instance.id,
				updates
			)
			
			updated_state = state_manager.state_cache[sample_workflow_instance.id]
			assert updated_state["progress"] == 50
			assert updated_state["completed_tasks"] == ["task_1"]
			assert updated_state["current_tasks"] == ["task_2"]
			assert updated_state["variables"]["var2"] == "value2"
			mock_persist.assert_called_once()
	
	@pytest.mark.asyncio
	async def test_get_instance_state(self, state_manager, sample_workflow_instance):
		"""Test getting instance state."""
		# Test cache hit
		cached_state = {"status": "running", "progress": 25}
		state_manager.state_cache[sample_workflow_instance.id] = cached_state
		
		state = await state_manager.get_instance_state(sample_workflow_instance.id)
		assert state == cached_state
		
		# Test cache miss with Redis retrieval
		del state_manager.state_cache[sample_workflow_instance.id]
		
		with patch.object(state_manager.redis_client, 'get') as mock_redis_get:
			redis_state = {"status": "paused", "progress": 75}
			mock_redis_get.return_value = json.dumps(redis_state)
			
			state = await state_manager.get_instance_state(sample_workflow_instance.id)
			assert state == redis_state
			assert state_manager.state_cache[sample_workflow_instance.id] == redis_state
	
	@pytest.mark.asyncio
	async def test_state_transitions(self, state_manager, sample_workflow_instance):
		"""Test state transitions and validation."""
		# Define valid state transitions
		valid_transitions = {
			"pending": ["running", "cancelled"],
			"running": ["paused", "completed", "failed", "cancelled"],
			"paused": ["running", "cancelled"],
			"completed": [],
			"failed": [],
			"cancelled": []
		}
		
		with patch.object(state_manager, '_get_valid_transitions') as mock_transitions:
			mock_transitions.return_value = valid_transitions
			
			# Test valid transition
			initial_state = {"status": "pending"}
			state_manager.state_cache[sample_workflow_instance.id] = initial_state
			
			# Valid transition: pending -> running
			await state_manager.transition_state(
				sample_workflow_instance.id,
				"running"
			)
			
			updated_state = state_manager.state_cache[sample_workflow_instance.id]
			assert updated_state["status"] == "running"
			
			# Test invalid transition
			with pytest.raises(ValueError, match="Invalid state transition"):
				await state_manager.transition_state(
					sample_workflow_instance.id,
					"pending"  # running -> pending is invalid
				)
	
	@pytest.mark.asyncio
	async def test_state_checkpoints(self, state_manager, sample_workflow_instance):
		"""Test state checkpoint creation and restoration."""
		current_state = {
			"status": "running",
			"progress": 60,
			"completed_tasks": ["task_1", "task_2"],
			"current_tasks": ["task_3"],
			"variables": {"checkpoint_data": "important"}
		}
		
		state_manager.state_cache[sample_workflow_instance.id] = current_state
		
		with patch.object(state_manager, '_create_checkpoint') as mock_checkpoint, \
			 patch.object(state_manager, '_persist_checkpoint') as mock_persist:
			
			# Create checkpoint
			checkpoint_id = await state_manager.create_checkpoint(
				sample_workflow_instance.id,
				"Before critical task"
			)
			
			assert checkpoint_id is not None
			mock_checkpoint.assert_called_once()
			mock_persist.assert_called_once()
	
	@pytest.mark.asyncio
	async def test_state_recovery(self, state_manager, sample_workflow_instance):
		"""Test state recovery from checkpoints."""
		checkpoint_state = {
			"status": "running",
			"progress": 40,
			"completed_tasks": ["task_1"],
			"current_tasks": ["task_2"],
			"variables": {"recovery_point": True}
		}
		
		with patch.object(state_manager, '_load_checkpoint') as mock_load, \
			 patch.object(state_manager, '_restore_state') as mock_restore:
			
			mock_load.return_value = checkpoint_state
			
			# Recover from checkpoint
			await state_manager.recover_from_checkpoint(
				sample_workflow_instance.id,
				"checkpoint_123"
			)
			
			mock_load.assert_called_once_with("checkpoint_123")
			mock_restore.assert_called_once_with(
				sample_workflow_instance.id,
				checkpoint_state
			)
	
	@pytest.mark.asyncio
	async def test_state_history_tracking(self, state_manager, sample_workflow_instance):
		"""Test state history tracking."""
		# Initialize state
		initial_state = {"status": "pending", "progress": 0}
		state_manager.state_cache[sample_workflow_instance.id] = initial_state
		
		with patch.object(state_manager, '_record_state_change') as mock_record:
			# Update state multiple times
			updates = [
				{"status": "running", "progress": 25},
				{"progress": 50},
				{"progress": 75},
				{"status": "completed", "progress": 100}
			]
			
			for update in updates:
				await state_manager.update_instance_state(
					sample_workflow_instance.id,
					update
				)
			
			# Verify state changes were recorded
			assert mock_record.call_count == len(updates)
	
	@pytest.mark.asyncio
	async def test_concurrent_state_updates(self, state_manager, sample_workflow_instance):
		"""Test concurrent state updates with locking."""
		initial_state = {"status": "running", "counter": 0}
		state_manager.state_cache[sample_workflow_instance.id] = initial_state
		
		with patch.object(state_manager, '_acquire_state_lock') as mock_lock, \
			 patch.object(state_manager, '_release_state_lock') as mock_unlock:
			
			mock_lock.return_value.__aenter__ = AsyncMock()
			mock_lock.return_value.__aexit__ = AsyncMock()
			
			# Simulate concurrent updates
			async def update_counter(increment: int):
				updates = {"counter": initial_state["counter"] + increment}
				await state_manager.update_instance_state(
					sample_workflow_instance.id,
					updates
				)
			
			# Run concurrent updates
			await asyncio.gather(
				update_counter(1),
				update_counter(2),
				update_counter(3)
			)
			
			# Verify locking was used
			assert mock_lock.call_count == 3

class TestBPMLEngine:
	"""Test BPML Engine functionality."""
	
	@pytest.mark.asyncio
	async def test_bpml_parser_initialization(self):
		"""Test BPML parser initialization."""
		parser = BPMLParser()
		
		assert parser is not None
		assert hasattr(parser, 'parse_xml')
		assert hasattr(parser, 'parse_json')
		assert hasattr(parser, 'validate_workflow')
	
	@pytest.mark.asyncio
	async def test_parse_bpml_xml(self):
		"""Test parsing BPML XML format."""
		bpml_xml = """
		<workflow xmlns="http://www.bpmn.org/bpmn" id="test_workflow" name="Test Workflow">
			<startEvent id="start" name="Start" />
			<task id="task1" name="Process Data">
				<documentation>Process incoming data</documentation>
			</task>
			<endEvent id="end" name="End" />
			<sequenceFlow id="flow1" sourceRef="start" targetRef="task1" />
			<sequenceFlow id="flow2" sourceRef="task1" targetRef="end" />
		</workflow>
		"""
		
		parser = BPMLParser()
		workflow_def = await parser.parse_xml(bpml_xml)
		
		assert workflow_def is not None
		assert workflow_def["name"] == "Test Workflow"
		assert len(workflow_def["tasks"]) == 1
		assert workflow_def["tasks"][0]["name"] == "Process Data"
	
	@pytest.mark.asyncio
	async def test_parse_bpml_json(self):
		"""Test parsing BPML JSON format."""
		bpml_json = {
			"id": "test_workflow",
			"name": "Test JSON Workflow",
			"tasks": [
				{
					"id": "task1",
					"name": "First Task",
					"type": "serviceTask",
					"configuration": {
						"service": "data_processor",
						"method": "process"
					}
				}
			],
			"flows": [
				{
					"id": "flow1",
					"from": "start",
					"to": "task1"
				}
			]
		}
		
		parser = BPMLParser()
		workflow_def = await parser.parse_json(bpml_json)
		
		assert workflow_def is not None
		assert workflow_def["name"] == "Test JSON Workflow"
		assert len(workflow_def["tasks"]) == 1
		assert workflow_def["tasks"][0]["configuration"]["service"] == "data_processor"
	
	@pytest.mark.asyncio
	async def test_bpml_gateway_support(self):
		"""Test BPML gateway support (exclusive, parallel, inclusive)."""
		bpml_with_gateways = {
			"id": "gateway_workflow",
			"name": "Workflow with Gateways",
			"tasks": [
				{"id": "task1", "name": "Initial Task", "type": "task"},
				{"id": "gateway1", "name": "Decision Gateway", "type": "exclusiveGateway"},
				{"id": "task2a", "name": "Path A", "type": "task"},
				{"id": "task2b", "name": "Path B", "type": "task"},
				{"id": "gateway2", "name": "Merge Gateway", "type": "parallelGateway"}
			],
			"flows": [
				{"from": "task1", "to": "gateway1"},
				{"from": "gateway1", "to": "task2a", "condition": "x > 10"},
				{"from": "gateway1", "to": "task2b", "condition": "x <= 10"},
				{"from": "task2a", "to": "gateway2"},
				{"from": "task2b", "to": "gateway2"}
			]
		}
		
		parser = BPMLParser()
		workflow_def = await parser.parse_json(bpml_with_gateways)
		
		assert workflow_def is not None
		assert len(workflow_def["tasks"]) == 5
		
		# Find gateways
		gateways = [task for task in workflow_def["tasks"] if "Gateway" in task.get("type", "")]
		assert len(gateways) == 2
	
	@pytest.mark.asyncio
	async def test_bpml_engine_execution(self, database_manager, redis_client):
		"""Test BPML engine workflow execution."""
		engine = BPMLEngine(database_manager, redis_client, "test_tenant")
		
		# Simple BPML workflow
		bpml_workflow = {
			"id": "simple_bpml",
			"name": "Simple BPML Workflow",
			"tasks": [
				{
					"id": "start",
					"name": "Start Event",
					"type": "startEvent"
				},
				{
					"id": "process_task",
					"name": "Process Data",
					"type": "serviceTask",
					"configuration": {
						"service": "data_service",
						"operation": "process"
					}
				},
				{
					"id": "end",
					"name": "End Event", 
					"type": "endEvent"
				}
			],
			"flows": [
				{"from": "start", "to": "process_task"},
				{"from": "process_task", "to": "end"}
			]
		}
		
		with patch.object(engine, '_execute_task') as mock_execute:
			mock_execute.return_value = {"status": "completed", "result": "processed"}
			
			# Execute BPML workflow
			result = await engine.execute_workflow(
				bpml_workflow,
				input_data={"test": "data"}
			)
			
			assert result is not None
			assert result["status"] in ["completed", "running"]
			mock_execute.assert_called()
	
	@pytest.mark.asyncio
	async def test_bpml_token_flow_control(self, database_manager, redis_client):
		"""Test BPML token-based flow control."""
		engine = BPMLEngine(database_manager, redis_client, "test_tenant")
		
		# Workflow with parallel execution
		parallel_workflow = {
			"id": "parallel_bpml",
			"name": "Parallel BPML Workflow",
			"tasks": [
				{"id": "start", "type": "startEvent"},
				{"id": "fork", "type": "parallelGateway"},
				{"id": "task_a", "type": "task", "name": "Task A"},
				{"id": "task_b", "type": "task", "name": "Task B"},
				{"id": "join", "type": "parallelGateway"},
				{"id": "end", "type": "endEvent"}
			],
			"flows": [
				{"from": "start", "to": "fork"},
				{"from": "fork", "to": "task_a"},
				{"from": "fork", "to": "task_b"},
				{"from": "task_a", "to": "join"},
				{"from": "task_b", "to": "join"},
				{"from": "join", "to": "end"}
			]
		}
		
		with patch.object(engine, '_manage_token_flow') as mock_token_flow, \
			 patch.object(engine, '_execute_parallel_tasks') as mock_parallel:
			
			mock_parallel.return_value = {"task_a": "completed", "task_b": "completed"}
			
			# Execute parallel workflow
			await engine.execute_workflow(parallel_workflow, {})
			
			# Verify token flow management was called
			mock_token_flow.assert_called()
			mock_parallel.assert_called()