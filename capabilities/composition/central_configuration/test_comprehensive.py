"""
APG Central Configuration - Comprehensive Test Suite

Complete testing suite including unit tests, integration tests,
performance tests, and end-to-end testing for the Central Configuration capability.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import pytest
import asyncio
import json
import time
import uuid
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Any

# Testing frameworks
import pytest_asyncio
from faker import Faker
from pytest_benchmark import benchmark

# Database testing
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Flask testing
from flask import Flask
from flask_testing import TestCase

# Our modules
from .service import CentralConfigurationEngine
from .ai_engine import CentralConfigurationAI
from .ml_models import CentralConfigurationML
from .analytics.reporting_engine import AdvancedAnalyticsEngine
from .performance.auto_scaler import IntelligentAutoScaler
from .security.audit_engine import SecurityAuditEngine
from .deployment.multi_region_orchestrator import MultiRegionOrchestrator
from .integrations.enterprise_connectors import EnterpriseIntegrationManager
from .models import (
	CCConfiguration, CCConfigurationTemplate, CCConfigurationVersion,
	CCWorkspace, CCUser, CCAuditLog, SecurityLevel, ConfigurationStatus
)
from .blueprint import CentralConfigurationBlueprint


# ==================== Test Configuration ====================

fake = Faker()

@pytest.fixture(scope="session")
def event_loop():
	"""Create event loop for async testing."""
	loop = asyncio.new_event_loop()
	yield loop
	loop.close()


@pytest.fixture
async def test_database():
	"""Create test database."""
	engine = create_engine(
		"sqlite:///:memory:",
		poolclass=StaticPool,
		connect_args={"check_same_thread": False}
	)
	
	# Import and create all tables
	from .models import db
	db.metadata.create_all(engine)
	
	TestingSession = sessionmaker(bind=engine)
	session = TestingSession()
	
	yield session
	
	session.close()


@pytest.fixture
async def config_engine(test_database):
	"""Create configuration engine for testing."""
	engine = CentralConfigurationEngine()
	engine.db_session = test_database
	await engine.initialize()
	return engine


@pytest.fixture
async def sample_configuration(config_engine):
	"""Create sample configuration for testing."""
	config_data = {
		"database": {
			"host": "localhost",
			"port": 5432,
			"username": "testuser",
			"password": "testpass",
			"database": "testdb"
		}
	}
	
	config = await config_engine.create_configuration(
		name="Test Database Config",
		key_path="/app/database/postgres",
		value=config_data,
		security_level=SecurityLevel.MEDIUM,
		workspace_id="test-workspace"
	)
	
	return config


# ==================== Unit Tests ====================

class TestCentralConfigurationEngine:
	"""Unit tests for the core configuration engine."""
	
	@pytest.mark.asyncio
	async def test_create_configuration(self, config_engine):
		"""Test configuration creation."""
		config_data = {"key": "value", "nested": {"key": "value"}}
		
		config = await config_engine.create_configuration(
			name="Test Config",
			key_path="/test/config",
			value=config_data,
			security_level=SecurityLevel.LOW,
			workspace_id="test-workspace"
		)
		
		assert config is not None
		assert config.name == "Test Config"
		assert config.key_path == "/test/config"
		assert config.status == ConfigurationStatus.DRAFT
		assert config.security_level == SecurityLevel.LOW
	
	@pytest.mark.asyncio
	async def test_get_configuration(self, config_engine, sample_configuration):
		"""Test configuration retrieval."""
		config = await config_engine.get_configuration(sample_configuration.id)
		
		assert config is not None
		assert config.name == sample_configuration.name
		assert config.value["database"]["host"] == "localhost"
	
	@pytest.mark.asyncio
	async def test_update_configuration(self, config_engine, sample_configuration):
		"""Test configuration updates."""
		new_data = {"database": {"host": "production.db.com", "port": 5432}}
		
		updated_config = await config_engine.update_configuration(
			sample_configuration.id,
			value=new_data,
			change_summary="Updated database host"
		)
		
		assert updated_config.value["database"]["host"] == "production.db.com"
		assert updated_config.version > sample_configuration.version
	
	@pytest.mark.asyncio
	async def test_configuration_validation(self, config_engine):
		"""Test configuration validation."""
		# Valid configuration
		valid_config = {"host": "localhost", "port": 5432}
		validation_result = await config_engine.validate_configuration_data(valid_config)
		assert validation_result["valid"] is True
		
		# Invalid configuration (empty)
		invalid_config = {}
		validation_result = await config_engine.validate_configuration_data(invalid_config)
		assert validation_result["valid"] is False
		assert len(validation_result["errors"]) > 0
	
	@pytest.mark.asyncio
	async def test_configuration_search(self, config_engine, sample_configuration):
		"""Test configuration search functionality."""
		# Search by name
		results = await config_engine.search_configurations(query="Test Database")
		assert len(results) == 1
		assert results[0].id == sample_configuration.id
		
		# Search by key path
		results = await config_engine.search_configurations(key_pattern="/app/database/*")
		assert len(results) == 1
		
		# Search with no results
		results = await config_engine.search_configurations(query="nonexistent")
		assert len(results) == 0
	
	@pytest.mark.asyncio
	async def test_configuration_encryption(self, config_engine):
		"""Test configuration encryption/decryption."""
		sensitive_data = {"api_key": "super-secret-key", "password": "secret123"}
		
		config = await config_engine.create_configuration(
			name="Sensitive Config",
			key_path="/app/secrets",
			value=sensitive_data,
			security_level=SecurityLevel.HIGH,
			workspace_id="test-workspace",
			encrypt=True
		)
		
		# Verify data is encrypted in storage
		assert config.is_encrypted is True
		
		# Verify we can decrypt and retrieve original data
		decrypted_config = await config_engine.get_configuration(config.id)
		assert decrypted_config.value["api_key"] == "super-secret-key"


class TestAIEngine:
	"""Unit tests for the AI engine."""
	
	@pytest.mark.asyncio
	async def test_ai_initialization(self, config_engine):
		"""Test AI engine initialization."""
		ai_engine = CentralConfigurationAI(config_engine)
		await ai_engine.initialize()
		
		assert ai_engine.ollama_client is not None
		assert len(ai_engine.available_models) > 0
	
	@pytest.mark.asyncio
	async def test_configuration_optimization(self, config_engine, sample_configuration):
		"""Test AI configuration optimization."""
		ai_engine = CentralConfigurationAI(config_engine)
		await ai_engine.initialize()
		
		# Mock Ollama response
		with patch.object(ai_engine.ollama_client, 'generate') as mock_generate:
			mock_generate.return_value = {
				'response': json.dumps({
					"optimized_config": {
						"database": {
							"host": "localhost",
							"port": 5432,
							"connection_pool_size": 20,
							"timeout": 30
						}
					},
					"improvements": ["Added connection pooling", "Optimized timeout"]
				})
			}
			
			optimization = await ai_engine.optimize_configuration(sample_configuration.value)
			
			assert "optimized_config" in optimization
			assert "improvements" in optimization
			assert len(optimization["improvements"]) > 0
	
	@pytest.mark.asyncio
	async def test_natural_language_query(self, config_engine):
		"""Test natural language query processing."""
		ai_engine = CentralConfigurationAI(config_engine)
		await ai_engine.initialize()
		
		with patch.object(ai_engine.ollama_client, 'generate') as mock_generate:
			mock_generate.return_value = {
				'response': json.dumps({
					"intent": "search_configurations",
					"filters": {"key_pattern": "*database*"},
					"confidence": 0.95
				})
			}
			
			result = await ai_engine.process_natural_language_query(
				"Show me all database configurations"
			)
			
			assert result["intent"] == "search_configurations"
			assert result["confidence"] > 0.8
	
	@pytest.mark.asyncio
	async def test_anomaly_detection(self, config_engine):
		"""Test AI anomaly detection."""
		ai_engine = CentralConfigurationAI(config_engine)
		await ai_engine.initialize()
		
		# Create metrics data
		metrics_data = {
			"response_time": [100, 105, 98, 102, 500],  # Last value is anomalous
			"error_rate": [0.1, 0.15, 0.12, 0.11, 0.08],
			"cpu_usage": [45, 48, 46, 44, 47]
		}
		
		anomalies = await ai_engine.detect_anomalies(metrics_data)
		
		assert len(anomalies) > 0
		assert any(anomaly["metric"] == "response_time" for anomaly in anomalies)


class TestMLModels:
	"""Unit tests for machine learning models."""
	
	@pytest.mark.asyncio
	async def test_ml_initialization(self, config_engine):
		"""Test ML models initialization."""
		ml_models = CentralConfigurationML(config_engine)
		await ml_models.initialize()
		
		assert ml_models.anomaly_detector is not None
		assert ml_models.performance_predictor is not None
	
	@pytest.mark.asyncio
	async def test_anomaly_detection_model(self, config_engine):
		"""Test anomaly detection ML model."""
		ml_models = CentralConfigurationML(config_engine)
		await ml_models.initialize()
		
		# Create training data
		normal_data = [[i, i*2, i*0.5] for i in range(100, 200)]
		anomalous_data = [[1000, 2000, 500]]  # Clear outlier
		
		# Train model
		ml_models.anomaly_detector.fit(normal_data)
		
		# Test detection
		anomaly_scores = ml_models.anomaly_detector.decision_function(anomalous_data)
		assert anomaly_scores[0] < 0  # Negative score indicates anomaly
	
	@pytest.mark.asyncio
	async def test_performance_prediction(self, config_engine):
		"""Test performance prediction model."""
		ml_models = CentralConfigurationML(config_engine)
		await ml_models.initialize()
		
		# Mock historical data
		historical_data = []
		for i in range(100):
			historical_data.append({
				"timestamp": datetime.now(timezone.utc) - timedelta(hours=i),
				"cpu_usage": 50 + (i % 20),
				"memory_usage": 60 + (i % 15),
				"request_rate": 100 + (i % 50)
			})
		
		prediction = await ml_models.predict_performance_metrics(
			historical_data,
			prediction_horizon_minutes=60
		)
		
		assert "predicted_cpu_usage" in prediction
		assert "predicted_memory_usage" in prediction
		assert "confidence_score" in prediction
		assert 0 <= prediction["confidence_score"] <= 1


# ==================== Integration Tests ====================

class TestIntegrationFlows:
	"""Integration tests for complete workflows."""
	
	@pytest.mark.asyncio
	async def test_configuration_lifecycle(self, config_engine):
		"""Test complete configuration lifecycle."""
		# 1. Create configuration
		config_data = {"service": {"name": "test-api", "port": 8080}}
		config = await config_engine.create_configuration(
			name="API Service Config",
			key_path="/services/api",
			value=config_data,
			security_level=SecurityLevel.MEDIUM,
			workspace_id="test-workspace"
		)
		
		assert config.status == ConfigurationStatus.DRAFT
		assert config.version == 1
		
		# 2. Update configuration
		updated_data = {"service": {"name": "test-api", "port": 9090}}
		updated_config = await config_engine.update_configuration(
			config.id,
			value=updated_data,
			change_summary="Changed port"
		)
		
		assert updated_config.version == 2
		assert updated_config.value["service"]["port"] == 9090
		
		# 3. Activate configuration
		activated_config = await config_engine.update_configuration(
			config.id,
			status=ConfigurationStatus.ACTIVE
		)
		
		assert activated_config.status == ConfigurationStatus.ACTIVE
		
		# 4. Archive configuration
		archived_config = await config_engine.update_configuration(
			config.id,
			status=ConfigurationStatus.ARCHIVED
		)
		
		assert archived_config.status == ConfigurationStatus.ARCHIVED
	
	@pytest.mark.asyncio
	async def test_ai_ml_integration(self, config_engine):
		"""Test AI and ML integration workflow."""
		# Initialize AI and ML components
		ai_engine = CentralConfigurationAI(config_engine)
		ml_models = CentralConfigurationML(config_engine)
		
		await ai_engine.initialize()
		await ml_models.initialize()
		
		# Create configuration
		config_data = {"database": {"host": "localhost", "pool_size": 5}}
		config = await config_engine.create_configuration(
			name="DB Config",
			key_path="/db/primary",
			value=config_data,
			security_level=SecurityLevel.MEDIUM,
			workspace_id="test-workspace"
		)
		
		# AI optimization
		with patch.object(ai_engine.ollama_client, 'generate') as mock_generate:
			mock_generate.return_value = {
				'response': json.dumps({
					"optimized_config": {
						"database": {"host": "localhost", "pool_size": 20}
					},
					"improvements": ["Increased pool size"]
				})
			}
			
			optimization = await ai_engine.optimize_configuration(config.value)
			assert optimization["optimized_config"]["database"]["pool_size"] == 20
		
		# ML anomaly detection
		metrics = {"response_time": [100, 105, 1000]}  # Anomalous spike
		anomalies = await ml_models.detect_configuration_anomalies(config.id, metrics)
		assert len(anomalies) > 0
	
	@pytest.mark.asyncio
	async def test_multi_component_workflow(self, config_engine):
		"""Test workflow involving multiple components."""
		# Initialize all components
		ai_engine = CentralConfigurationAI(config_engine)
		analytics_engine = AdvancedAnalyticsEngine(config_engine)
		auto_scaler = IntelligentAutoScaler(config_engine)
		
		await ai_engine.initialize()
		
		# Create multiple configurations
		configs = []
		for i in range(5):
			config = await config_engine.create_configuration(
				name=f"Service {i} Config",
				key_path=f"/services/service-{i}",
				value={"port": 8000 + i, "replicas": 2},
				security_level=SecurityLevel.LOW,
				workspace_id="test-workspace"
			)
			configs.append(config)
		
		# Generate analytics report
		from .analytics.reporting_engine import ReportType, TimeRange
		report = await analytics_engine.generate_comprehensive_report(
			report_type=ReportType.USAGE_ANALYTICS,
			time_range=TimeRange.LAST_7D
		)
		
		assert report is not None
		assert len(report.sections) > 0
		
		# Test auto-scaling decisions
		scaling_metrics = {
			"cpu_usage": 85.0,
			"memory_usage": 78.0,
			"request_rate": 500.0
		}
		
		scaling_decision = await auto_scaler.make_scaling_decision(
			"service-0",
			scaling_metrics
		)
		
		assert scaling_decision is not None
		assert "direction" in scaling_decision


# ==================== Performance Tests ====================

class TestPerformance:
	"""Performance and load testing."""
	
	@pytest.mark.benchmark
	def test_configuration_creation_performance(self, benchmark, config_engine):
		"""Benchmark configuration creation performance."""
		
		async def create_config():
			return await config_engine.create_configuration(
				name=f"Perf Test {uuid.uuid4()}",
				key_path=f"/perf/test/{uuid.uuid4()}",
				value={"key": "value", "number": 42},
				security_level=SecurityLevel.LOW,
				workspace_id="perf-test"
			)
		
		def sync_create_config():
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			try:
				return loop.run_until_complete(create_config())
			finally:
				loop.close()
		
		result = benchmark(sync_create_config)
		assert result is not None
	
	@pytest.mark.asyncio
	async def test_bulk_configuration_operations(self, config_engine):
		"""Test bulk operations performance."""
		start_time = time.time()
		
		# Create 100 configurations
		tasks = []
		for i in range(100):
			task = config_engine.create_configuration(
				name=f"Bulk Config {i}",
				key_path=f"/bulk/test/{i}",
				value={"index": i, "data": f"test-data-{i}"},
				security_level=SecurityLevel.LOW,
				workspace_id="bulk-test"
			)
			tasks.append(task)
		
		configs = await asyncio.gather(*tasks)
		
		end_time = time.time()
		duration = end_time - start_time
		
		assert len(configs) == 100
		assert duration < 10.0  # Should complete within 10 seconds
		
		# Test bulk search
		search_start = time.time()
		results = await config_engine.search_configurations(query="Bulk Config")
		search_duration = time.time() - search_start
		
		assert len(results) == 100
		assert search_duration < 2.0  # Search should be fast
	
	@pytest.mark.asyncio
	async def test_concurrent_updates(self, config_engine, sample_configuration):
		"""Test concurrent configuration updates."""
		
		async def update_config(value_suffix):
			return await config_engine.update_configuration(
				sample_configuration.id,
				value={"concurrent_test": f"value-{value_suffix}"},
				change_summary=f"Concurrent update {value_suffix}"
			)
		
		# Launch 10 concurrent updates
		tasks = [update_config(i) for i in range(10)]
		results = await asyncio.gather(*tasks, return_exceptions=True)
		
		# At least some should succeed (exact behavior depends on locking strategy)
		successful_updates = [r for r in results if not isinstance(r, Exception)]
		assert len(successful_updates) > 0
		
		# Final configuration should have a version > 1
		final_config = await config_engine.get_configuration(sample_configuration.id)
		assert final_config.version > 1


# ==================== Security Tests ====================

class TestSecurity:
	"""Security-focused tests."""
	
	@pytest.mark.asyncio
	async def test_encryption_security(self, config_engine):
		"""Test encryption security measures."""
		sensitive_data = {
			"database_password": "super-secret-password",
			"api_key": "sk-1234567890abcdef",
			"private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBg..."
		}
		
		# Create encrypted configuration
		config = await config_engine.create_configuration(
			name="Encrypted Secrets",
			key_path="/secrets/production",
			value=sensitive_data,
			security_level=SecurityLevel.CRITICAL,
			workspace_id="secure-workspace",
			encrypt=True
		)
		
		assert config.is_encrypted is True
		
		# Verify we can retrieve and decrypt
		retrieved_config = await config_engine.get_configuration(config.id)
		assert retrieved_config.value["database_password"] == "super-secret-password"
		
		# Verify encryption key rotation works
		await config_engine.rotate_encryption_keys()
		
		# Should still be able to decrypt after rotation
		post_rotation_config = await config_engine.get_configuration(config.id)
		assert post_rotation_config.value["api_key"] == "sk-1234567890abcdef"
	
	@pytest.mark.asyncio
	async def test_access_control(self, config_engine):
		"""Test access control mechanisms."""
		# Create configurations with different security levels
		public_config = await config_engine.create_configuration(
			name="Public Config",
			key_path="/public/config",
			value={"public": "data"},
			security_level=SecurityLevel.PUBLIC,
			workspace_id="test-workspace"
		)
		
		critical_config = await config_engine.create_configuration(
			name="Critical Config",
			key_path="/critical/config",
			value={"sensitive": "data"},
			security_level=SecurityLevel.CRITICAL,
			workspace_id="test-workspace"
		)
		
		# Mock user contexts
		public_user_context = {"security_level": SecurityLevel.PUBLIC}
		admin_user_context = {"security_level": SecurityLevel.CRITICAL}
		
		# Public user should access public config but not critical
		public_accessible = await config_engine.check_access(
			public_config.id, public_user_context
		)
		critical_accessible = await config_engine.check_access(
			critical_config.id, public_user_context
		)
		
		assert public_accessible is True
		assert critical_accessible is False
		
		# Admin should access both
		public_admin_accessible = await config_engine.check_access(
			public_config.id, admin_user_context
		)
		critical_admin_accessible = await config_engine.check_access(
			critical_config.id, admin_user_context
		)
		
		assert public_admin_accessible is True
		assert critical_admin_accessible is True
	
	@pytest.mark.asyncio
	async def test_audit_logging(self, config_engine):
		"""Test comprehensive audit logging."""
		# Create configuration (should be logged)
		config = await config_engine.create_configuration(
			name="Audit Test Config",
			key_path="/audit/test",
			value={"test": "data"},
			security_level=SecurityLevel.MEDIUM,
			workspace_id="audit-workspace"
		)
		
		# Update configuration (should be logged)
		await config_engine.update_configuration(
			config.id,
			value={"test": "updated_data"},
			change_summary="Audit test update"
		)
		
		# Retrieve audit logs
		audit_logs = await config_engine.get_audit_logs(
			resource_type="configuration",
			resource_id=config.id
		)
		
		assert len(audit_logs) >= 2  # Create + Update
		assert any(log.action == "create" for log in audit_logs)
		assert any(log.action == "update" for log in audit_logs)
		
		# Verify audit log details
		create_log = next(log for log in audit_logs if log.action == "create")
		assert create_log.resource_type == "configuration"
		assert create_log.resource_id == config.id
		assert "test" in create_log.new_values


# ==================== End-to-End Tests ====================

class TestEndToEnd:
	"""End-to-end testing scenarios."""
	
	@pytest.mark.asyncio
	async def test_complete_deployment_workflow(self, config_engine):
		"""Test complete deployment workflow from creation to production."""
		# 1. Create development configuration
		dev_config = await config_engine.create_configuration(
			name="E2E Test API Config",
			key_path="/e2e/api/config",
			value={
				"environment": "development",
				"database_url": "localhost:5432",
				"debug": True,
				"log_level": "DEBUG"
			},
			security_level=SecurityLevel.LOW,
			workspace_id="e2e-workspace"
		)
		
		# 2. Test and validate configuration
		validation_result = await config_engine.validate_configuration_data(dev_config.value)
		assert validation_result["valid"] is True
		
		# 3. Clone for staging
		staging_config = await config_engine.create_configuration(
			name="E2E Test API Config - Staging",
			key_path="/e2e/api/config/staging",
			value={
				"environment": "staging",
				"database_url": "staging-db:5432",
				"debug": False,
				"log_level": "INFO"
			},
			security_level=SecurityLevel.MEDIUM,
			workspace_id="e2e-workspace"
		)
		
		# 4. Promote to production
		prod_config = await config_engine.create_configuration(
			name="E2E Test API Config - Production",
			key_path="/e2e/api/config/production",
			value={
				"environment": "production",
				"database_url": "prod-db:5432",
				"debug": False,
				"log_level": "WARN"
			},
			security_level=SecurityLevel.HIGH,
			workspace_id="e2e-workspace",
			encrypt=True
		)
		
		# 5. Activate production configuration
		await config_engine.update_configuration(
			prod_config.id,
			status=ConfigurationStatus.ACTIVE
		)
		
		# 6. Verify deployment
		active_config = await config_engine.get_configuration(prod_config.id)
		assert active_config.status == ConfigurationStatus.ACTIVE
		assert active_config.value["environment"] == "production"
		assert active_config.is_encrypted is True
	
	@pytest.mark.asyncio
	async def test_disaster_recovery_scenario(self, config_engine):
		"""Test disaster recovery and backup/restore workflows."""
		# Create critical configuration
		critical_config = await config_engine.create_configuration(
			name="DR Test Critical Config",
			key_path="/dr/critical",
			value={
				"primary_database": "primary-db:5432",
				"backup_database": "backup-db:5432",
				"failover_threshold": 30
			},
			security_level=SecurityLevel.CRITICAL,
			workspace_id="dr-workspace",
			encrypt=True
		)
		
		# Create backup
		backup_id = await config_engine.create_backup(
			backup_type="full",
			configurations=[critical_config.id]
		)
		
		assert backup_id is not None
		
		# Simulate disaster - corrupt configuration
		await config_engine.update_configuration(
			critical_config.id,
			value={"corrupted": True},
			change_summary="Simulated corruption"
		)
		
		# Restore from backup
		restore_result = await config_engine.restore_from_backup(
			backup_id,
			target_configurations=[critical_config.id]
		)
		
		assert restore_result["success"] is True
		
		# Verify restoration
		restored_config = await config_engine.get_configuration(critical_config.id)
		assert "primary_database" in restored_config.value
		assert "corrupted" not in restored_config.value


# ==================== Test Utilities ====================

class TestDataFactory:
	"""Factory for creating test data."""
	
	@staticmethod
	def create_sample_configuration_data():
		"""Create sample configuration data."""
		return {
			"service": {
				"name": fake.word(),
				"port": fake.port_number(),
				"environment": fake.random_element(["dev", "staging", "prod"]),
				"replicas": fake.random_int(1, 10)
			},
			"database": {
				"host": fake.hostname(),
				"port": 5432,
				"name": fake.word(),
				"ssl": fake.boolean()
			},
			"cache": {
				"type": "redis",
				"host": fake.hostname(),
				"port": 6379,
				"timeout": fake.random_int(1, 60)
			}
		}
	
	@staticmethod
	def create_sample_template_data():
		"""Create sample template data."""
		return {
			"name": f"{fake.word().title()} Template",
			"category": fake.random_element(["database", "api", "cache", "queue"]),
			"template_data": TestDataFactory.create_sample_configuration_data(),
			"variables": {
				"host": {"type": "string", "default": "localhost"},
				"port": {"type": "integer", "default": 8080},
				"environment": {"type": "enum", "values": ["dev", "staging", "prod"]}
			}
		}


# ==================== Test Runner Configuration ====================

pytest_plugins = [
	"pytest_asyncio",
	"pytest_benchmark",
	"pytest_mock"
]

def pytest_configure(config):
	"""Configure pytest with custom markers."""
	config.addinivalue_line(
		"markers", "benchmark: mark test as benchmark test"
	)
	config.addinivalue_line(
		"markers", "integration: mark test as integration test"
	)
	config.addinivalue_line(
		"markers", "security: mark test as security test"
	)
	config.addinivalue_line(
		"markers", "performance: mark test as performance test"
	)
	config.addinivalue_line(
		"markers", "e2e: mark test as end-to-end test"
	)


# ==================== Main Test Execution ====================

if __name__ == "__main__":
	"""Run tests directly."""
	pytest.main([
		__file__,
		"--verbose",
		"--tb=short",
		"--benchmark-only",
		"--benchmark-sort=mean"
	])