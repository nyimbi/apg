#!/usr/bin/env python3
"""
APG ETLP Test Configuration
Pytest configuration and shared fixtures for ETLP tests

Author: APG Platform Team
Copyright: © 2025 Datacraft
"""

import asyncio
import pytest
from datetime import datetime
from uuid_extensions import uuid7str

from ...models import Pipeline, Execution, Transformation, DataSource, QualityRule
from ...service import ETLPService


@pytest.fixture(scope="session")
def event_loop():
	"""Create event loop for async tests"""
	loop = asyncio.get_event_loop()
	yield loop
	loop.close()


@pytest.fixture
def tenant_id():
	"""Generate unique tenant ID for each test"""
	return f"test-tenant-{uuid7str()[:8]}"


@pytest.fixture  
def user_id():
	"""Generate unique user ID for each test"""
	return f"test-user-{uuid7str()[:8]}"


@pytest.fixture
def admin_user_id():
	"""Generate admin user ID for tests requiring elevated permissions"""
	return f"admin-user-{uuid7str()[:8]}"


@pytest.fixture
def etlp_service(tenant_id, user_id):
	"""Create ETLP service instance for testing"""
	service = ETLPService(tenant_id, user_id)
	
	# Mock APG service dependencies for testing
	service.metadata_service = MockMetadataService()
	service.aicr_service = MockAICRService()
	service.auth_service = MockAuthService()
	service.audit_service = MockAuditService()
	service.notification_service = MockNotificationService()
	service.collaboration_service = MockCollaborationService()
	
	return service


@pytest.fixture
def sample_pipeline_data(tenant_id, user_id):
	"""Sample pipeline data for testing"""
	return {
		"name": "Test Pipeline",
		"description": "A sample pipeline for testing",
		"tenant_id": tenant_id,
		"created_by": user_id,
		"execution_mode": "batch",
		"steps": [
			{"type": "extract", "source": "test-source"},
			{"type": "transform", "transformation": "test-transform"},
			{"type": "load", "target": "test-target"}
		],
		"transformations": ["transform-1"],
		"data_sources": ["source-1"],
		"data_targets": ["target-1"],
		"quality_rules": ["rule-1"],
		"max_parallelism": 4,
		"timeout_minutes": 60,
		"retry_count": 3,
		"ai_optimization_enabled": True,
		"monitoring_enabled": True,
		"alert_on_failure": True,
		"tags": ["test", "sample"]
	}


@pytest.fixture
def sample_transformation_data(tenant_id, user_id):
	"""Sample transformation data for testing"""
	return {
		"name": "Test Transformation",
		"description": "A sample transformation for testing",
		"tenant_id": tenant_id,
		"created_by": user_id,
		"type": "map",
		"logic": {
			"operation": "select",
			"fields": ["id", "name", "email"],
			"conditions": [{"field": "active", "value": True}]
		},
		"input_schema": {
			"type": "object",
			"properties": {
				"id": {"type": "integer"},
				"name": {"type": "string"},
				"email": {"type": "string"},
				"active": {"type": "boolean"}
			}
		},
		"output_schema": {
			"type": "object",
			"properties": {
				"id": {"type": "integer"},
				"name": {"type": "string"},
				"email": {"type": "string"}
			}
		},
		"parameters": {"strict_validation": True},
		"tags": ["mapping", "filter"],
		"cacheable": True,
		"parallel_execution": True
	}


@pytest.fixture
def sample_data_source_data(tenant_id, user_id):
	"""Sample data source data for testing"""
	return {
		"name": "Test Database",
		"description": "A sample database connection for testing",
		"tenant_id": tenant_id,
		"created_by": user_id,
		"type": "database",
		"connection_string": "postgresql://test:password@localhost:5432/testdb",
		"credentials": {"username": "test", "password": "password"},
		"use_ssl": True,
		"timeout_seconds": 30,
		"settings": {"pool_size": 5, "max_overflow": 10},
		"headers": {},
		"batch_size": 1000,
		"max_connections": 5,
		"tags": ["testing", "postgresql"],
		"category": "database",
		"health_check_enabled": True
	}


@pytest.fixture
def sample_quality_rule_data(tenant_id, user_id):
	"""Sample quality rule data for testing"""
	return {
		"name": "Test Quality Rule",
		"description": "A sample quality rule for testing",
		"tenant_id": tenant_id,
		"created_by": user_id,
		"type": "not_null",
		"field_name": "email",
		"condition": {"field": "email", "operator": "not_null"},
		"severity": "error",
		"validation_logic": {"check": "value is not None and value.strip() != ''"},
		"error_message": "Email field cannot be null or empty",
		"suggested_fix": "Provide a valid email address",
		"enabled": True,
		"stop_on_violation": False,
		"sample_percentage": 100.0,
		"tags": ["validation", "email"],
		"category": "data_integrity"
	}


@pytest.fixture
def sample_execution_data(tenant_id):
	"""Sample execution data for testing"""
	return {
		"pipeline_id": uuid7str(),
		"tenant_id": tenant_id,
		"status": "running",
		"execution_mode": "batch",
		"triggered_by": "test-user",
		"trigger_type": "manual",
		"pipeline_version": "1.0.0",
		"configuration": {"batch_size": 1000},
		"environment": {"ENV": "test"},
		"started_at": datetime.utcnow(),
		"records_processed": 0,
		"records_failed": 0
	}


# Mock APG Service Classes for Testing

class MockMetadataService:
	"""Mock metadata service for testing"""
	
	def __init__(self):
		self.metadata = {}
	
	async def register_pipeline_metadata(self, pipeline):
		self.metadata[pipeline.id] = {
			"type": "pipeline",
			"name": pipeline.name,
			"description": pipeline.description,
			"tenant_id": pipeline.tenant_id
		}
	
	async def update_pipeline_metadata(self, pipeline):
		if pipeline.id in self.metadata:
			self.metadata[pipeline.id].update({
				"name": pipeline.name,
				"description": pipeline.description
			})
	
	async def remove_pipeline_metadata(self, pipeline):
		if pipeline.id in self.metadata:
			del self.metadata[pipeline.id]


class MockAICRService:
	"""Mock AI/CR service for testing"""
	
	async def optimize_pipeline(self, pipeline, performance_data):
		return {
			"performance_improvements": [
				{
					"type": "parallelization",
					"description": "Increase parallelism for better performance",
					"impact": "high",
					"estimated_improvement": "30% faster"
				}
			],
			"resource_optimizations": [
				{
					"type": "memory_optimization",
					"description": "Optimize memory usage",
					"impact": "medium",
					"estimated_savings": "20% less memory"
				}
			],
			"reliability_enhancements": [],
			"cost_optimizations": []
		}
	
	async def predict_execution_time(self, pipeline, data_size):
		# Simple mock prediction
		base_time = len(pipeline.steps) * 60  # 1 minute per step
		return base_time * (data_size / 1000)  # Scale with data size


class MockAuthService:
	"""Mock auth service for testing"""
	
	async def check_permission(self, user_id, permission, resource_id=None):
		# Mock permission check - allow all for testing
		return True
	
	async def validate_token(self, token):
		return {
			"user_id": "test-user",
			"tenant_id": "test-tenant",
			"permissions": ["*"]
		}


class MockAuditService:
	"""Mock audit service for testing"""
	
	def __init__(self):
		self.events = []
	
	async def log_event(self, event_type, data, user_id):
		self.events.append({
			"event_type": event_type,
			"data": data,
			"user_id": user_id,
			"timestamp": datetime.utcnow()
		})


class MockNotificationService:
	"""Mock notification service for testing"""
	
	def __init__(self):
		self.notifications = []
	
	async def send_notification(self, notification_type, data, recipients):
		self.notifications.append({
			"type": notification_type,
			"data": data,
			"recipients": recipients,
			"timestamp": datetime.utcnow()
		})


class MockCollaborationService:
	"""Mock collaboration service for testing"""
	
	def __init__(self):
		self.collaborations = {}
	
	async def get_pipeline_collaborators(self, pipeline_id):
		return self.collaborations.get(pipeline_id, [])
	
	async def add_collaborator(self, pipeline_id, user_id, role="viewer"):
		if pipeline_id not in self.collaborations:
			self.collaborations[pipeline_id] = []
		
		self.collaborations[pipeline_id].append({
			"user_id": user_id,
			"role": role,
			"added_at": datetime.utcnow()
		})


# Test Data Factories

@pytest.fixture
def pipeline_factory(sample_pipeline_data):
	"""Factory for creating pipeline test data"""
	def _create_pipeline(**overrides):
		data = sample_pipeline_data.copy()
		data.update(overrides)
		return Pipeline(**data)
	
	return _create_pipeline


@pytest.fixture
def transformation_factory(sample_transformation_data):
	"""Factory for creating transformation test data"""
	def _create_transformation(**overrides):
		data = sample_transformation_data.copy()
		data.update(overrides)
		return Transformation(**data)
	
	return _create_transformation


@pytest.fixture
def data_source_factory(sample_data_source_data):
	"""Factory for creating data source test data"""
	def _create_data_source(**overrides):
		data = sample_data_source_data.copy()
		data.update(overrides)
		return DataSource(**data)
	
	return _create_data_source


@pytest.fixture
def quality_rule_factory(sample_quality_rule_data):
	"""Factory for creating quality rule test data"""
	def _create_quality_rule(**overrides):
		data = sample_quality_rule_data.copy()
		data.update(overrides)
		return QualityRule(**data)
	
	return _create_quality_rule


@pytest.fixture
def execution_factory(sample_execution_data):
	"""Factory for creating execution test data"""
	def _create_execution(**overrides):
		data = sample_execution_data.copy()
		data.update(overrides)
		return Execution(**data)
	
	return _create_execution


# Performance Test Utilities

@pytest.fixture
def performance_monitor():
	"""Monitor for performance testing"""
	class PerformanceMonitor:
		def __init__(self):
			self.measurements = []
		
		def measure(self, operation_name):
			import time
			start_time = time.time()
			
			class MeasurementContext:
				def __enter__(self):
					return self
				
				def __exit__(self, exc_type, exc_val, exc_tb):
					end_time = time.time()
					duration = end_time - start_time
					self.parent.measurements.append({
						"operation": operation_name,
						"duration_seconds": duration,
						"timestamp": datetime.utcnow()
					})
			
			context = MeasurementContext()
			context.parent = self
			return context
		
		def get_average_duration(self, operation_name):
			measurements = [m for m in self.measurements if m["operation"] == operation_name]
			if not measurements:
				return 0
			return sum(m["duration_seconds"] for m in measurements) / len(measurements)
	
	return PerformanceMonitor()


# Pytest configuration
pytest_plugins = ["pytest_asyncio"]

def pytest_configure(config):
	"""Configure pytest settings"""
	config.addinivalue_line(
		"markers", 
		"slow: marks tests as slow (deselect with '-m \"not slow\"')"
	)
	config.addinivalue_line(
		"markers",
		"integration: marks tests as integration tests"
	)
	config.addinivalue_line(
		"markers",
		"unit: marks tests as unit tests"
	)