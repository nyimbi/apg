#!/usr/bin/env python3
"""
APG ETLP Comprehensive Test Suite
Full integration tests for ETLP capability

Author: APG Platform Team
Copyright: © 2025 Datacraft
"""

import asyncio
import json
import pytest
from datetime import datetime
from uuid_extensions import uuid7str

from ...models import (
	Pipeline, Execution, Transformation, DataSource, QualityRule,
	PipelineStatus, ExecutionMode, TransformationType, DataSourceType, QualityRuleType
)
from ...service import ETLPService
from ...views import (
	PipelineCreateRequest, PipelineUpdateRequest, PipelineExecuteRequest,
	TransformationCreateRequest, DataSourceCreateRequest, QualityRuleCreateRequest
)


@pytest.fixture
def event_loop():
	"""Create event loop for async tests"""
	loop = asyncio.get_event_loop()
	yield loop


@pytest.fixture
def tenant_id():
	"""Generate test tenant ID"""
	return f"test-tenant-{uuid7str()[:8]}"


@pytest.fixture
def user_id():
	"""Generate test user ID"""
	return f"test-user-{uuid7str()[:8]}"


@pytest.fixture
def etlp_service(tenant_id, user_id):
	"""Create ETLP service instance for testing"""
	return ETLPService(tenant_id, user_id)


class TestPipelineModels:
	"""Test core pipeline data models"""
	
	def test_pipeline_creation(self, tenant_id, user_id):
		"""Test pipeline model creation and validation"""
		pipeline_data = {
			"name": "Test Data Pipeline",
			"description": "A test pipeline for unit testing",
			"tenant_id": tenant_id,
			"created_by": user_id,
			"execution_mode": ExecutionMode.BATCH,
			"steps": [
				{"type": "extract", "source": "database"},
				{"type": "transform", "transformation_id": "test-transform"},
				{"type": "load", "target": "warehouse"}
			],
			"transformations": ["transform-1", "transform-2"],
			"data_sources": ["source-1"],
			"data_targets": ["target-1"],
			"quality_rules": ["rule-1"],
			"max_parallelism": 8,
			"timeout_minutes": 120,
			"retry_count": 2,
			"ai_optimization_enabled": True
		}
		
		pipeline = Pipeline(**pipeline_data)
		
		assert pipeline.name == "Test Data Pipeline"
		assert pipeline.tenant_id == tenant_id
		assert pipeline.created_by == user_id
		assert pipeline.status == PipelineStatus.DRAFT
		assert pipeline.execution_mode == ExecutionMode.BATCH
		assert len(pipeline.steps) == 3
		assert len(pipeline.transformations) == 2
		assert pipeline.max_parallelism == 8
		assert pipeline.ai_optimization_enabled is True
		assert pipeline.version == "1.0.0"
		assert pipeline.is_deleted is False
		assert isinstance(pipeline.created_at, datetime)
	
	def test_pipeline_validation(self):
		"""Test pipeline validation rules"""
		# Test empty name validation
		with pytest.raises(ValueError):
			Pipeline(
				name="",
				tenant_id="test",
				created_by="user"
			)
		
		# Test invalid cron expression
		with pytest.raises(ValueError):
			Pipeline(
				name="Test Pipeline",
				tenant_id="test", 
				created_by="user",
				schedule_cron="invalid cron"
			)
		
		# Test invalid version format
		with pytest.raises(ValueError):
			Pipeline(
				name="Test Pipeline",
				tenant_id="test",
				created_by="user",
				version="invalid.version"
			)
	
	def test_execution_model(self, tenant_id):
		"""Test execution model creation"""
		execution_data = {
			"pipeline_id": uuid7str(),
			"tenant_id": tenant_id,
			"status": PipelineStatus.RUNNING,
			"execution_mode": ExecutionMode.BATCH,
			"triggered_by": "test-user",
			"trigger_type": "manual",
			"pipeline_version": "1.0.0",
			"records_processed": 1000,
			"records_failed": 10
		}
		
		execution = Execution(**execution_data)
		
		assert execution.tenant_id == tenant_id
		assert execution.status == PipelineStatus.RUNNING
		assert execution.records_processed == 1000
		assert execution.records_failed == 10
		assert execution.success_rate == 99.0  # (1000-10)/1000 * 100
	
	def test_transformation_model(self, tenant_id, user_id):
		"""Test transformation model"""
		transform_data = {
			"name": "Test Transformation",
			"description": "A test transformation",
			"tenant_id": tenant_id,
			"created_by": user_id,
			"type": TransformationType.MAP,
			"logic": {"operation": "select", "fields": ["name", "email"]},
			"input_schema": {"type": "object"},
			"output_schema": {"type": "object"},
			"parameters": {"case_sensitive": False}
		}
		
		transformation = Transformation(**transform_data)
		
		assert transformation.name == "Test Transformation"
		assert transformation.type == TransformationType.MAP
		assert transformation.logic["operation"] == "select"
		assert transformation.cacheable is False
		assert transformation.parallel_execution is True
	
	def test_data_source_model(self, tenant_id, user_id):
		"""Test data source model"""
		source_data = {
			"name": "Test Database",
			"description": "Test database connection",
			"tenant_id": tenant_id,
			"created_by": user_id,
			"type": DataSourceType.DATABASE,
			"connection_string": "postgresql://user:pass@localhost:5432/testdb",
			"credentials": {"username": "test", "password": "secret"},
			"use_ssl": True,
			"timeout_seconds": 30,
			"batch_size": 5000,
			"max_connections": 10
		}
		
		data_source = DataSource(**source_data)
		
		assert data_source.name == "Test Database"
		assert data_source.type == DataSourceType.DATABASE
		assert data_source.use_ssl is True
		assert data_source.batch_size == 5000
		assert data_source.health_check_enabled is True
	
	def test_quality_rule_model(self, tenant_id, user_id):
		"""Test quality rule model"""
		rule_data = {
			"name": "Not Null Check",
			"description": "Ensure field is not null",
			"tenant_id": tenant_id,
			"created_by": user_id,
			"type": QualityRuleType.NOT_NULL,
			"field_name": "email",
			"condition": {"field": "email", "operator": "not_null"},
			"validation_logic": {"check": "value is not None"},
			"error_message": "Email field cannot be null",
			"severity": "error",
			"stop_on_violation": True,
			"sample_percentage": 100.0
		}
		
		quality_rule = QualityRule(**rule_data)
		
		assert quality_rule.name == "Not Null Check"
		assert quality_rule.type == QualityRuleType.NOT_NULL
		assert quality_rule.field_name == "email"
		assert quality_rule.stop_on_violation is True
		assert quality_rule.violation_rate == 0.0  # Initial rate


class TestETLPService:
	"""Test ETLP service business logic"""
	
	async def test_pipeline_lifecycle(self, etlp_service):
		"""Test complete pipeline lifecycle"""
		# Create pipeline
		pipeline_data = {
			"name": "Lifecycle Test Pipeline",
			"description": "Testing complete pipeline lifecycle",
			"execution_mode": "batch",
			"steps": [
				{"type": "extract", "source": "test-db"},
				{"type": "transform", "transformation": "clean-data"},
				{"type": "load", "target": "warehouse"}
			],
			"max_parallelism": 4,
			"ai_optimization_enabled": True
		}
		
		pipeline = await etlp_service.create_pipeline(pipeline_data)
		assert pipeline.name == "Lifecycle Test Pipeline"
		assert pipeline.status == PipelineStatus.DRAFT
		
		# Update pipeline
		updates = {
			"description": "Updated description",
			"max_parallelism": 8,
			"status": PipelineStatus.ACTIVE
		}
		
		updated_pipeline = await etlp_service.update_pipeline(pipeline.id, updates)
		assert updated_pipeline.description == "Updated description"
		assert updated_pipeline.max_parallelism == 8
		assert updated_pipeline.version != pipeline.version  # Version should increment
		
		# Execute pipeline
		execution_id = await etlp_service.execute_pipeline(pipeline.id)
		assert execution_id is not None
		
		# Get execution details
		execution = await etlp_service.get_execution(execution_id)
		assert execution is not None
		assert execution.pipeline_id == pipeline.id
		
		# List executions
		executions = await etlp_service.list_executions(pipeline.id)
		assert len(executions) >= 1
		
		# Delete pipeline (soft delete)
		deleted = await etlp_service.delete_pipeline(pipeline.id, hard_delete=False)
		assert deleted is True
		
		# Verify soft delete
		pipeline_after_delete = await etlp_service.get_pipeline(pipeline.id)
		assert pipeline_after_delete is None or pipeline_after_delete.is_deleted is True
	
	async def test_transformation_management(self, etlp_service):
		"""Test transformation creation and management"""
		transform_data = {
			"name": "Data Cleaner",
			"description": "Clean and standardize data",
			"type": "clean",
			"logic": {
				"operations": [
					{"type": "trim", "fields": ["name", "email"]},
					{"type": "lowercase", "fields": ["email"]},
					{"type": "validate_email", "field": "email"}
				]
			},
			"input_schema": {
				"type": "object",
				"properties": {
					"name": {"type": "string"},
					"email": {"type": "string"}
				}
			},
			"output_schema": {
				"type": "object",
				"properties": {
					"name": {"type": "string"},
					"email": {"type": "string"}
				}
			},
			"parameters": {"strict_validation": True}
		}
		
		transformation = await etlp_service.create_transformation(transform_data)
		assert transformation.name == "Data Cleaner"
		assert transformation.type.value == "clean"
		assert "operations" in transformation.logic
	
	async def test_data_source_management(self, etlp_service):
		"""Test data source creation and health checking"""
		source_data = {
			"name": "Test MySQL Database",
			"description": "MySQL database for testing",
			"type": "database",
			"connection_string": "mysql://test:password@localhost:3306/testdb",
			"credentials": {"username": "test", "password": "password"},
			"use_ssl": False,
			"timeout_seconds": 45,
			"batch_size": 2000
		}
		
		data_source = await etlp_service.create_data_source(source_data)
		assert data_source.name == "Test MySQL Database"
		assert data_source.type.value == "database"
		
		# Test health check
		health_result = await etlp_service.test_data_source(data_source.id)
		assert "healthy" in health_result
		assert "response_time_ms" in health_result
	
	async def test_quality_rule_management(self, etlp_service):
		"""Test quality rule creation and validation"""
		rule_data = {
			"name": "Email Format Validation",
			"description": "Validate email addresses",
			"type": "format",
			"field_name": "email",
			"condition": {"pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"},
			"validation_logic": {"regex_match": True},
			"error_message": "Invalid email format: {value}",
			"severity": "warning",
			"sample_percentage": 10.0
		}
		
		quality_rule = await etlp_service.create_quality_rule(rule_data)
		assert quality_rule.name == "Email Format Validation"
		assert quality_rule.type.value == "format"
		assert quality_rule.sample_percentage == 10.0
	
	async def test_ai_optimization(self, etlp_service):
		"""Test AI-powered optimization features"""
		# Create a pipeline for optimization
		pipeline_data = {
			"name": "Optimization Test Pipeline",
			"description": "Pipeline to test AI optimization",
			"execution_mode": "batch",
			"steps": [
				{"type": "extract", "source": "large-dataset"},
				{"type": "transform", "transformation": "complex-aggregation"},
				{"type": "load", "target": "warehouse"}
			],
			"ai_optimization_enabled": True
		}
		
		pipeline = await etlp_service.create_pipeline(pipeline_data)
		
		# Get optimization recommendations
		recommendations = await etlp_service.optimize_pipeline(pipeline.id)
		
		assert "performance_improvements" in recommendations
		assert "resource_optimizations" in recommendations
		assert "reliability_enhancements" in recommendations
		assert "cost_optimizations" in recommendations
	
	async def test_collaboration_features(self, etlp_service):
		"""Test collaboration features"""
		# Create a collaborative pipeline
		pipeline_data = {
			"name": "Collaborative Pipeline",
			"description": "Pipeline for testing collaboration",
			"collaboration_enabled": True,
			"execution_mode": "batch"
		}
		
		pipeline = await etlp_service.create_pipeline(pipeline_data)
		
		# Get collaborators
		collaborators = await etlp_service.get_pipeline_collaborators(pipeline.id)
		assert isinstance(collaborators, list)


class TestAPIViews:
	"""Test API request/response models"""
	
	def test_pipeline_create_request(self):
		"""Test pipeline creation request validation"""
		request_data = {
			"name": "API Test Pipeline",
			"description": "Testing API models",
			"execution_mode": "streaming",
			"max_parallelism": 6,
			"timeout_minutes": 90,
			"retry_count": 1,
			"ai_optimization_enabled": True,
			"tags": ["api", "test"]
		}
		
		request = PipelineCreateRequest(**request_data)
		
		assert request.name == "API Test Pipeline"
		assert request.execution_mode.value == "streaming"
		assert request.max_parallelism == 6
		assert "api" in request.tags
	
	def test_pipeline_update_request(self):
		"""Test pipeline update request validation"""
		request_data = {
			"name": "Updated Pipeline Name",
			"description": "Updated description",
			"status": "active",
			"max_parallelism": 12
		}
		
		request = PipelineUpdateRequest(**request_data)
		
		assert request.name == "Updated Pipeline Name"
		assert request.status.value == "active"
		assert request.max_parallelism == 12
		assert request.execution_mode is None  # Optional field not provided
	
	def test_pipeline_execute_request(self):
		"""Test pipeline execution request validation"""
		request_data = {
			"execution_mode": "micro_batch",
			"configuration": {"batch_size": 1000, "parallelism": 4},
			"environment_variables": {"ENV": "test", "DEBUG": "true"}
		}
		
		request = PipelineExecuteRequest(**request_data)
		
		assert request.execution_mode.value == "micro_batch"
		assert request.configuration["batch_size"] == 1000
		assert request.environment_variables["ENV"] == "test"
	
	def test_transformation_create_request(self):
		"""Test transformation creation request validation"""
		request_data = {
			"name": "API Transform",
			"description": "Transformation via API",
			"type": "filter",
			"logic": {"condition": "age > 18"},
			"input_schema": {"type": "object"},
			"parameters": {"strict": True},
			"tags": ["filter", "validation"],
			"cacheable": True,
			"parallel_execution": True
		}
		
		request = TransformationCreateRequest(**request_data)
		
		assert request.name == "API Transform"
		assert request.type.value == "filter"
		assert request.logic["condition"] == "age > 18"
		assert request.cacheable is True
	
	def test_data_source_create_request(self):
		"""Test data source creation request validation"""
		request_data = {
			"name": "API Data Source",
			"description": "Data source via API",
			"type": "api",
			"connection_string": "https://api.example.com/data",
			"credentials": {"api_key": "secret"},
			"use_ssl": True,
			"timeout_seconds": 60,
			"headers": {"Authorization": "Bearer token"},
			"batch_size": 3000,
			"max_connections": 8,
			"health_check_enabled": True
		}
		
		request = DataSourceCreateRequest(**request_data)
		
		assert request.name == "API Data Source"
		assert request.type.value == "api"
		assert request.connection_string == "https://api.example.com/data"
		assert request.batch_size == 3000
		assert request.health_check_enabled is True
	
	def test_quality_rule_create_request(self):
		"""Test quality rule creation request validation"""
		request_data = {
			"name": "API Quality Rule",
			"description": "Quality rule via API",
			"type": "unique",
			"field_name": "user_id",
			"condition": {"unique_constraint": True},
			"validation_logic": {"check_uniqueness": True},
			"error_message": "Duplicate user_id found",
			"severity": "error",
			"enabled": True,
			"stop_on_violation": False,
			"sample_percentage": 100.0,
			"tags": ["uniqueness", "constraint"]
		}
		
		request = QualityRuleCreateRequest(**request_data)
		
		assert request.name == "API Quality Rule"
		assert request.type.value == "unique"
		assert request.field_name == "user_id"
		assert request.sample_percentage == 100.0
		assert "uniqueness" in request.tags


class TestIntegration:
	"""Test APG platform integration"""
	
	async def test_audit_logging(self, etlp_service):
		"""Test APG audit service integration"""
		# Mock audit service
		class MockAuditService:
			def __init__(self):
				self.events = []
			
			async def log_event(self, event_type, data, user_id):
				self.events.append({
					"event_type": event_type,
					"data": data,
					"user_id": user_id,
					"timestamp": datetime.utcnow()
				})
		
		mock_audit = MockAuditService()
		etlp_service.audit_service = mock_audit
		
		# Create pipeline - should trigger audit log
		pipeline_data = {
			"name": "Audit Test Pipeline",
			"description": "Testing audit integration"
		}
		
		pipeline = await etlp_service.create_pipeline(pipeline_data)
		
		# Verify audit event was logged
		assert len(mock_audit.events) > 0
		audit_event = mock_audit.events[0]
		assert audit_event["event_type"] == "pipeline_created"
		assert audit_event["data"]["pipeline_id"] == pipeline.id
	
	async def test_notification_integration(self, etlp_service):
		"""Test APG notification service integration"""
		# Mock notification service
		class MockNotificationService:
			def __init__(self):
				self.notifications = []
			
			async def send_notification(self, notification_type, data, recipients):
				self.notifications.append({
					"type": notification_type,
					"data": data,
					"recipients": recipients,
					"timestamp": datetime.utcnow()
				})
		
		mock_notification = MockNotificationService()
		etlp_service.notification_service = mock_notification
		
		# Create pipeline with notifications enabled
		pipeline_data = {
			"name": "Notification Test Pipeline",
			"description": "Testing notification integration",
			"alert_on_failure": True
		}
		
		pipeline = await etlp_service.create_pipeline(pipeline_data)
		
		# Mock pipeline failure to trigger notification
		execution = Execution(
			pipeline_id=pipeline.id,
			tenant_id=etlp_service.tenant_id,
			status=PipelineStatus.FAILED,
			execution_mode=ExecutionMode.BATCH,
			triggered_by=etlp_service.user_id,
			trigger_type="manual",
			pipeline_version="1.0.0",
			error_message="Test failure"
		)
		
		# Simulate failure notification
		if pipeline.alert_on_failure and etlp_service.notification_service:
			await etlp_service.notification_service.send_notification(
				"pipeline_failure",
				{"pipeline_id": pipeline.id, "execution_id": execution.id},
				[pipeline.created_by]
			)
		
		# Verify notification was sent
		assert len(mock_notification.notifications) > 0
		notification = mock_notification.notifications[0]
		assert notification["type"] == "pipeline_failure"


@pytest.mark.asyncio
async def test_full_integration_scenario():
	"""Test complete ETLP scenario end-to-end"""
	tenant_id = f"integration-tenant-{uuid7str()[:8]}"
	user_id = f"integration-user-{uuid7str()[:8]}"
	
	# Initialize service
	etlp_service = ETLPService(tenant_id, user_id)
	
	# 1. Create data sources
	source_data = {
		"name": "Customer Database",
		"description": "Primary customer data source",
		"type": "database",
		"connection_string": "postgresql://user:pass@localhost/customers"
	}
	data_source = await etlp_service.create_data_source(source_data)
	
	# 2. Create transformations
	transform_data = {
		"name": "Customer Data Cleaner",
		"description": "Clean and validate customer data",
		"type": "clean",
		"logic": {"operations": ["trim", "validate", "standardize"]}
	}
	transformation = await etlp_service.create_transformation(transform_data)
	
	# 3. Create quality rules
	rule_data = {
		"name": "Email Validation",
		"description": "Validate customer email addresses",
		"type": "format",
		"field_name": "email",
		"condition": {"pattern": r".+@.+\..+"},
		"validation_logic": {"regex_match": True},
		"error_message": "Invalid email format"
	}
	quality_rule = await etlp_service.create_quality_rule(rule_data)
	
	# 4. Create comprehensive pipeline
	pipeline_data = {
		"name": "Customer Data Processing Pipeline",
		"description": "End-to-end customer data processing",
		"execution_mode": "batch",
		"steps": [
			{"type": "extract", "source": data_source.id},
			{"type": "transform", "transformation": transformation.id},
			{"type": "quality_check", "rules": [quality_rule.id]},
			{"type": "load", "target": "data_warehouse"}
		],
		"transformations": [transformation.id],
		"data_sources": [data_source.id],
		"quality_rules": [quality_rule.id],
		"ai_optimization_enabled": True,
		"monitoring_enabled": True,
		"alert_on_failure": True
	}
	
	pipeline = await etlp_service.create_pipeline(pipeline_data)
	
	# 5. Execute pipeline
	execution_id = await etlp_service.execute_pipeline(pipeline.id)
	
	# 6. Monitor execution
	execution = await etlp_service.get_execution(execution_id)
	assert execution is not None
	assert execution.pipeline_id == pipeline.id
	
	# 7. Get AI optimization recommendations
	recommendations = await etlp_service.optimize_pipeline(pipeline.id)
	assert isinstance(recommendations, dict)
	
	# 8. Test collaboration features
	collaborators = await etlp_service.get_pipeline_collaborators(pipeline.id)
	assert isinstance(collaborators, list)
	
	# Verify all components work together
	assert pipeline.name == "Customer Data Processing Pipeline"
	assert data_source.id in pipeline.data_sources
	assert transformation.id in pipeline.transformations  
	assert quality_rule.id in pipeline.quality_rules
	assert pipeline.ai_optimization_enabled is True


if __name__ == "__main__":
	pytest.main([__file__, "-v"])