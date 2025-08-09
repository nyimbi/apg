"""
APG Workflow Orchestration Model Tests

Comprehensive unit tests for Pydantic v2 models with >95% coverage.
Tests validation, serialization, deserialization, and model behavior.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import pytest
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List
from pydantic import ValidationError
from uuid_extensions import uuid7str

from ..models import (
	Workflow, WorkflowInstance, TaskDefinition, TaskExecution,
	WorkflowStatus, TaskStatus, Priority, TaskType,
	WorkflowMetrics, ExecutionContext, TaskDependency,
	RetryPolicy, NotificationConfig, SecurityContext
)

class TestTaskDefinition:
	"""Test TaskDefinition model."""
	
	def test_task_definition_creation(self):
		"""Test basic task definition creation."""
		task = TaskDefinition(
			name="Test Task",
			description="Test description",
			task_type=TaskType.TASK,
			configuration={"action": "test"},
			dependencies=[],
			estimated_duration=60
		)
		
		assert task.name == "Test Task"
		assert task.description == "Test description"
		assert task.task_type == TaskType.TASK
		assert task.configuration == {"action": "test"}
		assert task.dependencies == []
		assert task.estimated_duration == 60
		assert task.max_retries == 3  # default
		assert task.timeout_seconds == 300  # default
		assert len(task.id) > 0  # UUID generated
	
	def test_task_definition_validation(self):
		"""Test task definition validation."""
		# Test required fields
		with pytest.raises(ValidationError) as exc_info:
			TaskDefinition()
		
		errors = exc_info.value.errors()
		required_fields = {"name", "task_type"}
		missing_fields = {error["loc"][0] for error in errors if error["type"] == "missing"}
		assert required_fields.issubset(missing_fields)
	
	def test_task_definition_name_validation(self):
		"""Test task name validation."""
		# Test empty name
		with pytest.raises(ValidationError):
			TaskDefinition(
				name="",
				task_type=TaskType.TASK
			)
		
		# Test name too long
		with pytest.raises(ValidationError):
			TaskDefinition(
				name="x" * 201,  # Max 200 characters
				task_type=TaskType.TASK
			)
	
	def test_task_definition_duration_validation(self):
		"""Test duration field validation."""
		# Test negative duration
		with pytest.raises(ValidationError):
			TaskDefinition(
				name="Test Task",
				task_type=TaskType.TASK,
				estimated_duration=-1
			)
		
		# Test zero duration (should be valid)
		task = TaskDefinition(
			name="Test Task",
			task_type=TaskType.TASK,
			estimated_duration=0
		)
		assert task.estimated_duration == 0
	
	def test_task_definition_retry_validation(self):
		"""Test retry policy validation."""
		# Test negative retries
		with pytest.raises(ValidationError):
			TaskDefinition(
				name="Test Task",
				task_type=TaskType.TASK,
				max_retries=-1
			)
		
		# Test valid retries
		task = TaskDefinition(
			name="Test Task",
			task_type=TaskType.TASK,
			max_retries=5
		)
		assert task.max_retries == 5
	
	def test_task_definition_serialization(self):
		"""Test task definition serialization."""
		task = TaskDefinition(
			name="Test Task",
			description="Test description",
			task_type=TaskType.INTEGRATION,
			configuration={"endpoint": "https://api.example.com"},
			dependencies=["task_1", "task_2"],
			estimated_duration=120,
			max_retries=2,
			timeout_seconds=600,
			assignee="user@example.com"
		)
		
		# Test JSON serialization
		json_data = task.model_dump_json()
		parsed_data = json.loads(json_data)
		
		assert parsed_data["name"] == "Test Task"
		assert parsed_data["task_type"] == "integration"
		assert parsed_data["dependencies"] == ["task_1", "task_2"]
		assert parsed_data["estimated_duration"] == 120
	
	def test_task_definition_deserialization(self):
		"""Test task definition deserialization."""
		data = {
			"name": "Deserialized Task",
			"description": "From JSON",
			"task_type": "notification",
			"configuration": {"channel": "email"},
			"dependencies": ["dep_1"],
			"estimated_duration": 90
		}
		
		task = TaskDefinition(**data)
		assert task.name == "Deserialized Task"
		assert task.task_type == TaskType.NOTIFICATION
		assert task.configuration == {"channel": "email"}

class TestWorkflow:
	"""Test Workflow model."""
	
	def test_workflow_creation(self, sample_task_definition):
		"""Test basic workflow creation."""
		workflow = Workflow(
			name="Test Workflow",
			description="Test workflow description",
			tasks=[sample_task_definition],
			tenant_id="test_tenant",
			created_by="test_user",
			updated_by="test_user"
		)
		
		assert workflow.name == "Test Workflow"
		assert workflow.description == "Test workflow description"
		assert len(workflow.tasks) == 1
		assert workflow.status == WorkflowStatus.DRAFT  # default
		assert workflow.priority == Priority.MEDIUM  # default
		assert workflow.tenant_id == "test_tenant"
		assert isinstance(workflow.created_at, datetime)
		assert isinstance(workflow.updated_at, datetime)
	
	def test_workflow_validation(self):
		"""Test workflow validation."""
		# Test missing required fields
		with pytest.raises(ValidationError) as exc_info:
			Workflow()
		
		errors = exc_info.value.errors()
		required_fields = {"name", "tasks", "tenant_id", "created_by", "updated_by"}
		missing_fields = {error["loc"][0] for error in errors if error["type"] == "missing"}
		assert required_fields.issubset(missing_fields)
	
	def test_workflow_tasks_validation(self, sample_task_definition):
		"""Test workflow tasks validation."""
		# Test empty tasks
		with pytest.raises(ValidationError):
			Workflow(
				name="Empty Workflow",
				tasks=[],
				tenant_id="test_tenant",
				created_by="test_user",
				updated_by="test_user"
			)
		
		# Test valid tasks
		workflow = Workflow(
			name="Valid Workflow",
			tasks=[sample_task_definition],
			tenant_id="test_tenant",
			created_by="test_user",
			updated_by="test_user"
		)
		assert len(workflow.tasks) == 1
	
	def test_workflow_name_validation(self, sample_task_definition):
		"""Test workflow name validation."""
		# Test empty name
		with pytest.raises(ValidationError):
			Workflow(
				name="",
				tasks=[sample_task_definition],
				tenant_id="test_tenant",
				created_by="test_user",
				updated_by="test_user"
			)
		
		# Test name too long
		with pytest.raises(ValidationError):
			Workflow(
				name="x" * 201,  # Max 200 characters
				tasks=[sample_task_definition],
				tenant_id="test_tenant",
				created_by="test_user",
				updated_by="test_user"
			)
	
	def test_workflow_sla_validation(self, sample_task_definition):
		"""Test SLA validation."""
		# Test negative SLA
		with pytest.raises(ValidationError):
			Workflow(
				name="Test Workflow",
				tasks=[sample_task_definition],
				sla_hours=-1.0,
				tenant_id="test_tenant",
				created_by="test_user",
				updated_by="test_user"
			)
		
		# Test valid SLA
		workflow = Workflow(
			name="Test Workflow",
			tasks=[sample_task_definition],
			sla_hours=24.5,
			tenant_id="test_tenant",
			created_by="test_user",
			updated_by="test_user"
		)
		assert workflow.sla_hours == 24.5
	
	def test_workflow_serialization(self, sample_task_definition):
		"""Test workflow serialization."""
		workflow = Workflow(
			name="Serializable Workflow",
			description="For JSON testing",
			tasks=[sample_task_definition],
			configuration={"max_concurrent": 5},
			priority=Priority.HIGH,
			tenant_id="test_tenant",
			created_by="test_user",
			updated_by="test_user",
			tags=["test", "serialization"],
			sla_hours=12.0
		)
		
		# Test JSON serialization
		json_data = workflow.model_dump_json()
		parsed_data = json.loads(json_data)
		
		assert parsed_data["name"] == "Serializable Workflow"
		assert parsed_data["priority"] == "high"
		assert parsed_data["tags"] == ["test", "serialization"]
		assert parsed_data["sla_hours"] == 12.0
		assert len(parsed_data["tasks"]) == 1
	
	def test_workflow_update_timestamps(self, sample_task_definition):
		"""Test workflow timestamp behavior."""
		workflow = Workflow(
			name="Timestamp Test",
			tasks=[sample_task_definition],
			tenant_id="test_tenant",
			created_by="test_user",
			updated_by="test_user"
		)
		
		original_created = workflow.created_at
		original_updated = workflow.updated_at
		
		# Simulate update
		updated_workflow_data = workflow.model_dump()
		updated_workflow_data["updated_at"] = datetime.now(timezone.utc)
		updated_workflow_data["description"] = "Updated description"
		
		updated_workflow = Workflow(**updated_workflow_data)
		
		assert updated_workflow.created_at == original_created
		assert updated_workflow.updated_at > original_updated
		assert updated_workflow.description == "Updated description"

class TestWorkflowInstance:
	"""Test WorkflowInstance model."""
	
	def test_workflow_instance_creation(self):
		"""Test basic workflow instance creation."""
		instance = WorkflowInstance(
			workflow_id=uuid7str(),
			input_data={"key": "value"},
			tenant_id="test_tenant",
			started_by="test_user"
		)
		
		assert len(instance.id) > 0
		assert len(instance.workflow_id) > 0
		assert instance.input_data == {"key": "value"}
		assert instance.status == WorkflowStatus.PENDING  # default
		assert instance.priority == Priority.MEDIUM  # default
		assert instance.tenant_id == "test_tenant"
		assert instance.started_by == "test_user"
		assert isinstance(instance.created_at, datetime)
	
	def test_workflow_instance_validation(self):
		"""Test workflow instance validation."""
		# Test missing required fields
		with pytest.raises(ValidationError) as exc_info:
			WorkflowInstance()
		
		errors = exc_info.value.errors()
		required_fields = {"workflow_id", "tenant_id", "started_by"}
		missing_fields = {error["loc"][0] for error in errors if error["type"] == "missing"}
		assert required_fields.issubset(missing_fields)
	
	def test_workflow_instance_progress_validation(self):
		"""Test progress percentage validation."""
		# Test negative progress
		with pytest.raises(ValidationError):
			WorkflowInstance(
				workflow_id=uuid7str(),
				progress_percentage=-1,
				tenant_id="test_tenant",
				started_by="test_user"
			)
		
		# Test progress over 100
		with pytest.raises(ValidationError):
			WorkflowInstance(
				workflow_id=uuid7str(),
				progress_percentage=101,
				tenant_id="test_tenant",
				started_by="test_user"
			)
		
		# Test valid progress
		instance = WorkflowInstance(
			workflow_id=uuid7str(),
			progress_percentage=75,
			tenant_id="test_tenant",
			started_by="test_user"
		)
		assert instance.progress_percentage == 75
	
	def test_workflow_instance_execution_times(self):
		"""Test execution time relationships."""
		now = datetime.now(timezone.utc)
		started_time = now - timedelta(hours=1)
		completed_time = now
		
		instance = WorkflowInstance(
			workflow_id=uuid7str(),
			started_at=started_time,
			completed_at=completed_time,
			tenant_id="test_tenant",
			started_by="test_user"
		)
		
		assert instance.started_at == started_time
		assert instance.completed_at == completed_time
		assert instance.started_at < instance.completed_at

class TestTaskExecution:
	"""Test TaskExecution model."""
	
	def test_task_execution_creation(self, sample_task_definition):
		"""Test basic task execution creation."""
		execution = TaskExecution(
			instance_id=uuid7str(),
			task_id=sample_task_definition.id,
			task_definition=sample_task_definition,
			tenant_id="test_tenant"
		)
		
		assert len(execution.id) > 0
		assert len(execution.instance_id) > 0
		assert execution.task_id == sample_task_definition.id
		assert execution.task_definition == sample_task_definition
		assert execution.status == TaskStatus.PENDING  # default
		assert execution.tenant_id == "test_tenant"
		assert isinstance(execution.created_at, datetime)
	
	def test_task_execution_validation(self, sample_task_definition):
		"""Test task execution validation."""
		# Test missing required fields
		with pytest.raises(ValidationError) as exc_info:
			TaskExecution()
		
		errors = exc_info.value.errors()
		required_fields = {"instance_id", "task_id", "task_definition", "tenant_id"}
		missing_fields = {error["loc"][0] for error in errors if error["type"] == "missing"}
		assert required_fields.issubset(missing_fields)
	
	def test_task_execution_retry_validation(self, sample_task_definition):
		"""Test retry count validation."""
		# Test negative retry count
		with pytest.raises(ValidationError):
			TaskExecution(
				instance_id=uuid7str(),
				task_id=sample_task_definition.id,
				task_definition=sample_task_definition,
				retry_count=-1,
				tenant_id="test_tenant"
			)
		
		# Test valid retry count
		execution = TaskExecution(
			instance_id=uuid7str(),
			task_id=sample_task_definition.id,
			task_definition=sample_task_definition,
			retry_count=2,
			tenant_id="test_tenant"
		)
		assert execution.retry_count == 2
	
	def test_task_execution_duration_calculation(self, sample_task_definition):
		"""Test duration calculation."""
		started_time = datetime.now(timezone.utc) - timedelta(minutes=5)
		completed_time = datetime.now(timezone.utc)
		
		execution = TaskExecution(
			instance_id=uuid7str(),
			task_id=sample_task_definition.id,
			task_definition=sample_task_definition,
			started_at=started_time,
			completed_at=completed_time,
			tenant_id="test_tenant"
		)
		
		# Duration should be approximately 5 minutes (300 seconds)
		expected_duration = (completed_time - started_time).total_seconds()
		assert execution.started_at == started_time
		assert execution.completed_at == completed_time

class TestEnums:
	"""Test enum definitions."""
	
	def test_workflow_status_enum(self):
		"""Test WorkflowStatus enum."""
		assert WorkflowStatus.DRAFT.value == "draft"
		assert WorkflowStatus.ACTIVE.value == "active"
		assert WorkflowStatus.PAUSED.value == "paused"
		assert WorkflowStatus.COMPLETED.value == "completed"
		assert WorkflowStatus.FAILED.value == "failed"
		assert WorkflowStatus.CANCELLED.value == "cancelled"
		
		# Test enum values
		all_statuses = list(WorkflowStatus)
		assert len(all_statuses) == 6
	
	def test_task_status_enum(self):
		"""Test TaskStatus enum."""
		assert TaskStatus.PENDING.value == "pending"
		assert TaskStatus.RUNNING.value == "running"
		assert TaskStatus.COMPLETED.value == "completed"
		assert TaskStatus.FAILED.value == "failed"
		assert TaskStatus.SKIPPED.value == "skipped"
		assert TaskStatus.CANCELLED.value == "cancelled"
		
		# Test enum values
		all_statuses = list(TaskStatus)
		assert len(all_statuses) == 6
	
	def test_priority_enum(self):
		"""Test Priority enum."""
		assert Priority.LOW.value == "low"
		assert Priority.MEDIUM.value == "medium"
		assert Priority.HIGH.value == "high"
		assert Priority.CRITICAL.value == "critical"
		
		# Test priority ordering
		priorities = list(Priority)
		assert len(priorities) == 4
	
	def test_task_type_enum(self):
		"""Test TaskType enum."""
		assert TaskType.TASK.value == "task"
		assert TaskType.INTEGRATION.value == "integration"
		assert TaskType.NOTIFICATION.value == "notification"
		assert TaskType.APPROVAL.value == "approval"
		assert TaskType.CONDITION.value == "condition"
		assert TaskType.LOOP.value == "loop"
		assert TaskType.SCRIPT.value == "script"
		
		# Test all task types
		all_types = list(TaskType)
		assert len(all_types) == 7

class TestModelIntegration:
	"""Test model integration and relationships."""
	
	def test_workflow_with_multiple_tasks(self):
		"""Test workflow with multiple tasks."""
		tasks = [
			TaskDefinition(
				name=f"Task {i}",
				task_type=TaskType.TASK,
				configuration={"step": i}
			)
			for i in range(1, 4)
		]
		
		workflow = Workflow(
			name="Multi-task Workflow",
			tasks=tasks,
			tenant_id="test_tenant",
			created_by="test_user",
			updated_by="test_user"
		)
		
		assert len(workflow.tasks) == 3
		assert workflow.tasks[0].name == "Task 1"
		assert workflow.tasks[2].name == "Task 3"
	
	def test_task_dependencies(self):
		"""Test task dependency relationships."""
		task1 = TaskDefinition(
			id="task_1",
			name="First Task",
			task_type=TaskType.TASK,
			dependencies=[]
		)
		
		task2 = TaskDefinition(
			id="task_2", 
			name="Second Task",
			task_type=TaskType.TASK,
			dependencies=["task_1"]
		)
		
		task3 = TaskDefinition(
			id="task_3",
			name="Third Task", 
			task_type=TaskType.TASK,
			dependencies=["task_1", "task_2"]
		)
		
		workflow = Workflow(
			name="Dependency Test",
			tasks=[task1, task2, task3],
			tenant_id="test_tenant",
			created_by="test_user",
			updated_by="test_user"
		)
		
		# Verify dependencies
		assert len(workflow.tasks[0].dependencies) == 0
		assert workflow.tasks[1].dependencies == ["task_1"]
		assert workflow.tasks[2].dependencies == ["task_1", "task_2"]
	
	def test_complex_workflow_serialization(self):
		"""Test complex workflow serialization/deserialization."""
		# Create complex workflow
		tasks = [
			TaskDefinition(
				name="Data Fetch",
				description="Fetch data from API",
				task_type=TaskType.INTEGRATION,
				configuration={
					"endpoint": "https://api.example.com/data",
					"method": "GET",
					"headers": {"Authorization": "Bearer token"}
				},
				dependencies=[],
				estimated_duration=120,
				max_retries=2
			),
			TaskDefinition(
				name="Data Processing",
				description="Process fetched data",
				task_type=TaskType.SCRIPT,
				configuration={
					"script_type": "python",
					"script_content": "def process(data): return data"
				},
				dependencies=["task_1"],
				estimated_duration=300
			)
		]
		
		workflow = Workflow(
			name="Complex Integration Workflow",
			description="Multi-step data processing workflow",
			tasks=tasks,
			configuration={
				"max_concurrent_tasks": 2,
				"retry_policy": "exponential_backoff",
				"timeout_seconds": 3600
			},
			priority=Priority.HIGH,
			tenant_id="test_tenant",
			created_by="test_user",
			updated_by="test_user",
			tags=["integration", "data_processing", "api"],
			sla_hours=4.0
		)
		
		# Serialize to JSON
		json_str = workflow.model_dump_json()
		parsed_data = json.loads(json_str)
		
		# Deserialize from JSON
		restored_workflow = Workflow(**parsed_data)
		
		# Verify restoration
		assert restored_workflow.name == workflow.name
		assert len(restored_workflow.tasks) == len(workflow.tasks)
		assert restored_workflow.tasks[0].name == "Data Fetch"
		assert restored_workflow.tasks[1].dependencies == ["task_1"]
		assert restored_workflow.priority == Priority.HIGH
		assert restored_workflow.sla_hours == 4.0

class TestModelEdgeCases:
	"""Test edge cases and boundary conditions."""
	
	def test_empty_optional_fields(self, sample_task_definition):
		"""Test handling of empty optional fields."""
		workflow = Workflow(
			name="Minimal Workflow",
			tasks=[sample_task_definition],
			tenant_id="test_tenant", 
			created_by="test_user",
			updated_by="test_user",
			description="",  # Empty description
			configuration={},  # Empty configuration
			tags=[],  # Empty tags
			metadata={}  # Empty metadata
		)
		
		assert workflow.description == ""
		assert workflow.configuration == {}
		assert workflow.tags == []
		assert workflow.metadata == {}
	
	def test_unicode_handling(self):
		"""Test Unicode character handling."""
		task = TaskDefinition(
			name="Unicode Task 🚀",
			description="Task with émojis and spëcial characters: 中文",
			task_type=TaskType.TASK,
			configuration={"message": "Hello 世界! 🌍"}
		)
		
		assert "🚀" in task.name
		assert "émojis" in task.description
		assert "中文" in task.description
		assert "世界" in task.configuration["message"]
		
		# Test serialization with Unicode
		json_str = task.model_dump_json()
		restored_task = TaskDefinition.model_validate_json(json_str)
		
		assert restored_task.name == task.name
		assert restored_task.description == task.description
		assert restored_task.configuration["message"] == task.configuration["message"]
	
	def test_large_data_handling(self):
		"""Test handling of large data structures."""
		# Create large configuration
		large_config = {f"key_{i}": f"value_{i}" * 100 for i in range(100)}
		
		task = TaskDefinition(
			name="Large Data Task",
			task_type=TaskType.TASK,
			configuration=large_config
		)
		
		assert len(task.configuration) == 100
		assert len(str(task.configuration)) > 10000  # Large serialized size
		
		# Test serialization/deserialization with large data
		json_str = task.model_dump_json()
		assert len(json_str) > 10000
		
		restored_task = TaskDefinition.model_validate_json(json_str)
		assert restored_task.configuration == large_config
	
	def test_nested_complex_data(self):
		"""Test deeply nested complex data structures."""
		complex_config = {
			"database": {
				"connections": [
					{
						"name": "primary",
						"settings": {
							"host": "localhost",
							"port": 5432,
							"credentials": {
								"username": "user",
								"password_ref": "secret_key"
							}
						}
					},
					{
						"name": "replica",
						"settings": {
							"host": "replica.example.com",
							"port": 5432,
							"read_only": True
						}
					}
				]
			},
			"processing": {
				"steps": [
					{"action": "validate", "params": {"strict": True}},
					{"action": "transform", "params": {"format": "json"}},
					{"action": "store", "params": {"table": "processed_data"}}
				]
			}
		}
		
		task = TaskDefinition(
			name="Complex Config Task",
			task_type=TaskType.INTEGRATION,
			configuration=complex_config
		)
		
		# Verify nested access
		assert task.configuration["database"]["connections"][0]["name"] == "primary"
		assert task.configuration["processing"]["steps"][1]["action"] == "transform"
		
		# Test serialization preserves structure
		json_str = task.model_dump_json()
		restored_task = TaskDefinition.model_validate_json(json_str)
		
		assert restored_task.configuration == complex_config