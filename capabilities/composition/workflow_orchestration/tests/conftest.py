"""
APG Workflow Orchestration Test Configuration

Comprehensive pytest configuration and fixtures for testing workflow orchestration
system with APG integration, database setup, and testing utilities.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import pytest
import pytest_asyncio
from typing import Dict, List, Optional, Any, AsyncGenerator
from datetime import datetime, timezone
import json
import os
from pathlib import Path

# Testing imports
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import event
from sqlalchemy.pool import StaticPool
import fakeredis.aioredis

# Application imports
from ..models import (
	Workflow, WorkflowInstance, TaskDefinition, TaskExecution,
	WorkflowStatus, TaskStatus, Priority, TaskType
)
from ..database import DatabaseManager, create_repositories, Base
from ..service import WorkflowOrchestrationService
from ..management import WorkflowManager, VersionManager, DeploymentManager
from ..engine import WorkflowExecutor, TaskScheduler, StateManager
from ..connectors import (
	BaseConnector, RESTConnector, DatabaseConnector, 
	CloudConnector, MessageQueueConnector, FileConnector
)

# Test configuration
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"
TEST_REDIS_URL = "redis://localhost:6379/15"  # Use DB 15 for tests
TEST_TENANT_ID = "test_tenant"
TEST_USER_ID = "test_user"

@pytest.fixture(scope="session")
def event_loop():
	"""Create an instance of the default event loop for the test session."""
	loop = asyncio.get_event_loop_policy().new_event_loop()
	yield loop
	loop.close()

@pytest_asyncio.fixture(scope="function")
async def async_engine():
	"""Create test database engine."""
	engine = create_async_engine(
		TEST_DATABASE_URL,
		echo=False,
		poolclass=StaticPool,
		connect_args={
			"check_same_thread": False,
		}
	)
	
	# Create all tables
	async with engine.begin() as conn:
		await conn.run_sync(Base.metadata.create_all)
	
	yield engine
	
	# Cleanup
	async with engine.begin() as conn:
		await conn.run_sync(Base.metadata.drop_all)
	
	await engine.dispose()

@pytest_asyncio.fixture(scope="function")
async def db_session(async_engine):
	"""Create test database session."""
	async_session = async_sessionmaker(
		async_engine,
		class_=AsyncSession,
		expire_on_commit=False
	)
	
	async with async_session() as session:
		yield session

@pytest_asyncio.fixture(scope="function")
async def redis_client():
	"""Create test Redis client using fakeredis."""
	client = fakeredis.aioredis.from_url("redis://localhost", decode_responses=True)
	yield client
	await client.flushall()
	await client.close()

@pytest_asyncio.fixture(scope="function")
async def database_manager(async_engine):
	"""Create test database manager."""
	manager = DatabaseManager(TEST_DATABASE_URL, pool_size=5, max_overflow=10)
	# Override engine for testing
	manager.engine = async_engine
	manager.session_factory = async_sessionmaker(
		async_engine,
		class_=AsyncSession,
		expire_on_commit=False
	)
	yield manager
	await manager.close()

@pytest_asyncio.fixture(scope="function")
async def workflow_service(database_manager, redis_client):
	"""Create workflow orchestration service."""
	service = WorkflowOrchestrationService(
		database_manager, redis_client, TEST_TENANT_ID
	)
	yield service

@pytest_asyncio.fixture(scope="function") 
async def workflow_manager(database_manager, redis_client):
	"""Create workflow manager."""
	manager = WorkflowManager(database_manager, redis_client, TEST_TENANT_ID)
	yield manager

@pytest_asyncio.fixture(scope="function")
async def version_manager(database_manager, redis_client):
	"""Create version manager."""
	manager = VersionManager(database_manager, redis_client, TEST_TENANT_ID)
	yield manager

@pytest_asyncio.fixture(scope="function")
async def deployment_manager(database_manager, redis_client, version_manager):
	"""Create deployment manager."""
	manager = DeploymentManager(
		database_manager, redis_client, version_manager, TEST_TENANT_ID
	)
	yield manager

@pytest_asyncio.fixture(scope="function")
async def workflow_executor(database_manager, redis_client):
	"""Create workflow executor."""
	executor = WorkflowExecutor(database_manager, redis_client, TEST_TENANT_ID)
	yield executor

@pytest_asyncio.fixture(scope="function")
async def task_scheduler(database_manager, redis_client):
	"""Create task scheduler."""
	scheduler = TaskScheduler(database_manager, redis_client, TEST_TENANT_ID)
	yield scheduler

@pytest_asyncio.fixture(scope="function")
async def state_manager(database_manager, redis_client):
	"""Create state manager."""
	manager = StateManager(database_manager, redis_client, TEST_TENANT_ID)
	yield manager

# Sample data fixtures
@pytest.fixture
def sample_task_definition():
	"""Create sample task definition."""
	return TaskDefinition(
		name="Test Task",
		description="A test task for unit testing",
		task_type=TaskType.TASK,
		configuration={
			"action": "test_action",
			"parameters": {"test_param": "test_value"}
		},
		dependencies=[],
		estimated_duration=30,
		max_retries=3,
		timeout_seconds=300,
		assignee="test_user",
		metadata={"test_metadata": "value"}
	)

@pytest.fixture
def sample_workflow(sample_task_definition):
	"""Create sample workflow."""
	return Workflow(
		name="Test Workflow",
		description="A test workflow for unit testing",
		tasks=[sample_task_definition],
		configuration={
			"max_concurrent_tasks": 5,
			"retry_policy": "exponential_backoff"
		},
		priority=Priority.MEDIUM,
		tenant_id=TEST_TENANT_ID,
		created_by=TEST_USER_ID,
		updated_by=TEST_USER_ID,
		tags=["test", "unit_test"],
		metadata={"test_workflow": True}
	)

@pytest.fixture
def complex_workflow():
	"""Create complex workflow with multiple tasks and dependencies."""
	tasks = [
		TaskDefinition(
			id="task_1",
			name="Initial Task",
			description="First task in workflow",
			task_type=TaskType.TASK,
			configuration={"action": "initialize"},
			dependencies=[],
			estimated_duration=60
		),
		TaskDefinition(
			id="task_2", 
			name="Processing Task",
			description="Process data task",
			task_type=TaskType.INTEGRATION,
			configuration={
				"connector": "rest_api",
				"endpoint": "https://api.example.com/process"
			},
			dependencies=["task_1"],
			estimated_duration=120
		),
		TaskDefinition(
			id="task_3",
			name="Notification Task", 
			description="Send notification task",
			task_type=TaskType.NOTIFICATION,
			configuration={
				"channel": "email",
				"template": "process_complete"
			},
			dependencies=["task_2"],
			estimated_duration=30
		)
	]
	
	return Workflow(
		name="Complex Test Workflow",
		description="Multi-task workflow for comprehensive testing",
		tasks=tasks,
		configuration={
			"max_concurrent_tasks": 2,
			"retry_policy": "linear_backoff",
			"failure_threshold": 2
		},
		priority=Priority.HIGH,
		tenant_id=TEST_TENANT_ID,
		created_by=TEST_USER_ID,
		updated_by=TEST_USER_ID,
		tags=["complex", "integration", "test"],
		sla_hours=24.0
	)

@pytest.fixture
def sample_workflow_instance(sample_workflow):
	"""Create sample workflow instance."""
	return WorkflowInstance(
		workflow_id=sample_workflow.id,
		input_data={"test_input": "value"},
		configuration_overrides={"max_concurrent_tasks": 3},
		priority=Priority.MEDIUM,
		tenant_id=TEST_TENANT_ID,
		started_by=TEST_USER_ID,
		tags=["test_instance"]
	)

@pytest.fixture  
def sample_task_execution(sample_task_definition, sample_workflow_instance):
	"""Create sample task execution."""
	return TaskExecution(
		instance_id=sample_workflow_instance.id,
		task_id=sample_task_definition.id,
		task_definition=sample_task_definition,
		input_data={"task_input": "value"},
		configuration=sample_task_definition.configuration,
		assigned_to="test_worker",
		tenant_id=TEST_TENANT_ID
	)

# Connector fixtures
@pytest_asyncio.fixture(scope="function")
async def rest_connector():
	"""Create REST connector for testing."""
	connector = RESTConnector("test_rest", {
		"base_url": "https://httpbin.org",
		"timeout": 30,
		"retry_count": 3
	})
	yield connector

@pytest_asyncio.fixture(scope="function") 
async def database_connector(database_manager):
	"""Create database connector for testing."""
	connector = DatabaseConnector("test_db", {
		"connection_string": TEST_DATABASE_URL,
		"pool_size": 5
	})
	yield connector

# Mock data fixtures
@pytest.fixture
def mock_execution_data():
	"""Mock execution data for testing."""
	return {
		"workflow_executions": [
			{
				"id": "exec_1",
				"workflow_id": "workflow_1", 
				"status": "completed",
				"started_at": datetime.now(timezone.utc),
				"completed_at": datetime.now(timezone.utc),
				"duration_seconds": 300
			},
			{
				"id": "exec_2",
				"workflow_id": "workflow_1",
				"status": "failed", 
				"started_at": datetime.now(timezone.utc),
				"completed_at": datetime.now(timezone.utc),
				"duration_seconds": 150
			}
		],
		"task_executions": [
			{
				"id": "task_exec_1",
				"instance_id": "exec_1",
				"task_id": "task_1",
				"status": "completed",
				"started_at": datetime.now(timezone.utc),
				"completed_at": datetime.now(timezone.utc),
				"duration_seconds": 120
			}
		]
	}

# Test utilities
class TestHelpers:
	"""Test helper utilities."""
	
	@staticmethod
	def create_test_workflow_data(**kwargs) -> Dict[str, Any]:
		"""Create test workflow data with defaults."""
		default_data = {
			"name": "Test Workflow",
			"description": "Test workflow description",
			"tasks": [
				{
					"name": "Test Task",
					"description": "Test task description",
					"task_type": "task",
					"configuration": {"action": "test"},
					"dependencies": [],
					"estimated_duration": 60
				}
			],
			"configuration": {"max_concurrent_tasks": 5},
			"priority": "medium",
			"tags": ["test"],
			"tenant_id": TEST_TENANT_ID,
			"created_by": TEST_USER_ID,
			"updated_by": TEST_USER_ID
		}
		default_data.update(kwargs)
		return default_data
	
	@staticmethod
	def assert_workflow_fields(workflow: Workflow, expected_data: Dict[str, Any]):
		"""Assert workflow fields match expected data."""
		assert workflow.name == expected_data["name"]
		assert workflow.description == expected_data["description"]
		assert len(workflow.tasks) == len(expected_data["tasks"])
		assert workflow.priority.value == expected_data["priority"]
		assert workflow.tenant_id == expected_data["tenant_id"]
	
	@staticmethod
	def assert_datetime_recent(dt: datetime, tolerance_seconds: int = 10):
		"""Assert datetime is recent within tolerance."""
		now = datetime.now(timezone.utc)
		diff = abs((now - dt).total_seconds())
		assert diff <= tolerance_seconds, f"Datetime {dt} not within {tolerance_seconds}s of now"

@pytest.fixture
def test_helpers():
	"""Test helper utilities fixture."""
	return TestHelpers

# Environment setup
@pytest.fixture(autouse=True)
def setup_test_environment():
	"""Setup test environment variables."""
	original_env = os.environ.copy()
	
	# Set test environment variables
	os.environ.update({
		"TESTING": "true",
		"DATABASE_URL": TEST_DATABASE_URL,
		"REDIS_URL": TEST_REDIS_URL,
		"LOG_LEVEL": "DEBUG",
		"APG_TENANT_ID": TEST_TENANT_ID
	})
	
	yield
	
	# Restore original environment
	os.environ.clear()
	os.environ.update(original_env)

# Async test markers
pytestmark = pytest.mark.asyncio

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "test_data"
FIXTURES_DIR = Path(__file__).parent / "fixtures"

@pytest.fixture
def test_data_dir():
	"""Test data directory path."""
	return TEST_DATA_DIR

@pytest.fixture  
def fixtures_dir():
	"""Fixtures directory path."""
	return FIXTURES_DIR

# Performance testing fixtures
@pytest.fixture
def performance_config():
	"""Performance testing configuration."""
	return {
		"max_execution_time": 10.0,  # seconds
		"memory_limit_mb": 100,
		"concurrent_workflows": 10,
		"load_test_duration": 30  # seconds
	}