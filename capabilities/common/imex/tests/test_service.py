"""
Test APG Import/Export Service

Comprehensive tests for the ImportExportService business logic layer.
Tests async operations, APG integration, and performance characteristics.
"""

import asyncio
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, Mock, patch
from uuid_extensions import uuid7str

from ..service import ImportExportService, imex_service
from ..models import (
	ImportExportJob, JobExecution, JobStatus, JobType, ProcessingMetrics,
	DataQualityReport, Workflow, SchemaMapping, ValidationRule
)
from . import (
	generate_test_job_config, generate_test_data_sample, generate_test_schema,
	generate_test_workflow_config, TEST_CONFIG, AsyncTestCase,
	assert_valid_uuid, assert_metrics_structure, MockConnection,
	MockDataSource, MockAIService
)


class TestImportExportService(AsyncTestCase):
	"""Test ImportExportService functionality"""

	def setup_method(self):
		"""Setup test method"""
		super().setup_method()
		self.service = ImportExportService()

	async def test_service_initialization(self):
		"""Test service initialization"""
		service = ImportExportService()

		# Test initial state
		assert service.health_status == "healthy"
		assert isinstance(service.active_jobs, dict)
		assert isinstance(service.performance_metrics, dict)

		# Test initialization
		await service.initialize()

		assert service.health_status == "ready"
		assert service.ai_client is not None
		assert service.etlp_client is not None
		assert service.conn_client is not None

	async def test_create_job_valid(self):
		"""Test creating valid import/export job"""
		await self.service.initialize()

		job_config = generate_test_job_config()
		user_id = "test_user"

		job = await self.service.create_job(job_config, user_id)

		assert isinstance(job, ImportExportJob)
		assert job.name == job_config["name"]
		assert job.job_type == JobType(job_config["job_type"])
		assert job.created_by == user_id
		assert job.status == JobStatus.DRAFT
		assert_valid_uuid(job.id)

		# Check job is stored in active jobs
		assert job.id in self.service.active_jobs
		assert self.service.performance_metrics["jobs_created"] == 1

	async def test_create_job_invalid_config(self):
		"""Test creating job with invalid configuration"""
		await self.service.initialize()

		invalid_config = {
			"name": "",  # Invalid empty name
			"job_type": "invalid_type"
		}

		with pytest.raises(Exception):
			await self.service.create_job(invalid_config, "test_user")

	async def test_create_job_missing_user(self):
		"""Test creating job without user"""
		await self.service.initialize()

		job_config = generate_test_job_config()

		with pytest.raises(AssertionError):
			await self.service.create_job(job_config, None)

	async def test_execute_job_import(self):
		"""Test executing import job"""
		await self.service.initialize()

		# Create job
		job_config = generate_test_job_config("import")
		job = await self.service.create_job(job_config, "test_user")

		# Execute job
		execution = await self.service.execute_job(job.id)

		assert isinstance(execution, JobExecution)
		assert execution.job_id == job.id
		assert execution.status == JobStatus.COMPLETED
		assert execution.started_at is not None
		assert execution.completed_at is not None
		assert_metrics_structure(execution.metrics.dict())

		# Check job status updated
		assert job.status == JobStatus.COMPLETED
		assert job.last_run_at is not None
		assert execution.id in job.execution_history

	async def test_execute_job_export(self):
		"""Test executing export job"""
		await self.service.initialize()

		# Create job
		job_config = generate_test_job_config("export")
		job = await self.service.create_job(job_config, "test_user")

		# Execute job
		execution = await self.service.execute_job(job.id)

		assert isinstance(execution, JobExecution)
		assert execution.job_id == job.id
		assert execution.status == JobStatus.COMPLETED
		assert_metrics_structure(execution.metrics.dict())

	async def test_execute_job_migration(self):
		"""Test executing migration job"""
		await self.service.initialize()

		# Create job
		job_config = generate_test_job_config("migration")
		job = await self.service.create_job(job_config, "test_user")

		# Execute job
		execution = await self.service.execute_job(job.id)

		assert isinstance(execution, JobExecution)
		assert execution.status == JobStatus.COMPLETED
		assert_metrics_structure(execution.metrics.dict())

	async def test_execute_job_sync(self):
		"""Test executing sync job"""
		await self.service.initialize()

		# Create job
		job_config = generate_test_job_config("sync")
		job = await self.service.create_job(job_config, "test_user")

		# Execute job
		execution = await self.service.execute_job(job.id)

		assert isinstance(execution, JobExecution)
		assert execution.status == JobStatus.COMPLETED

	async def test_execute_job_not_found(self):
		"""Test executing non-existent job"""
		await self.service.initialize()

		fake_job_id = uuid7str()

		with pytest.raises(ValueError) as exc_info:
			await self.service.execute_job(fake_job_id)

		assert "Job not found" in str(exc_info.value)

	async def test_execute_job_invalid_status(self):
		"""Test executing job with invalid status"""
		await self.service.initialize()

		# Create job and set invalid status
		job_config = generate_test_job_config()
		job = await self.service.create_job(job_config, "test_user")
		job.status = JobStatus.RUNNING  # Already running

		with pytest.raises(ValueError) as exc_info:
			await self.service.execute_job(job.id)

		assert "cannot be executed" in str(exc_info.value)

	async def test_execute_job_with_config(self):
		"""Test executing job with custom execution config"""
		await self.service.initialize()

		job_config = generate_test_job_config()
		job = await self.service.create_job(job_config, "test_user")

		execution_config = {
			"priority": "high",
			"resource_limits": {"memory": "2GB", "cpu": "2"}
		}

		execution = await self.service.execute_job(job.id, execution_config)

		assert execution.execution_config == execution_config

	async def test_get_job_metrics(self):
		"""Test getting job metrics"""
		await self.service.initialize()

		# Create and execute job
		job_config = generate_test_job_config()
		job = await self.service.create_job(job_config, "test_user")

		# Start execution (but don't complete)
		job.status = JobStatus.RUNNING
		job.current_execution = JobExecution(
			job_id=job.id,
			execution_number=1,
			status=JobStatus.RUNNING
		)

		# Get metrics
		metrics = await self.service.get_job_metrics(job.id)

		assert isinstance(metrics, ProcessingMetrics)
		assert_metrics_structure(metrics.dict())

	async def test_get_job_metrics_not_found(self):
		"""Test getting metrics for non-existent job"""
		await self.service.initialize()

		fake_job_id = uuid7str()

		with pytest.raises(ValueError) as exc_info:
			await self.service.get_job_metrics(fake_job_id)

		assert "No active execution found" in str(exc_info.value)

	async def test_detect_schema_automatically(self):
		"""Test automatic schema detection"""
		await self.service.initialize()

		source_config = {
			"source_type": "file",
			"file_path": "/data/test.csv",
			"format": "csv"
		}

		from ..models import SourceConfig
		source = SourceConfig(**source_config)

		detected_schema = await self.service.detect_schema_automatically(source)

		assert isinstance(detected_schema, dict)
		assert "fields" in detected_schema
		assert isinstance(detected_schema["fields"], list)

	async def test_suggest_field_mappings(self):
		"""Test field mapping suggestions"""
		await self.service.initialize()

		source_schema = generate_test_schema()
		target_schema = {
			"fields": [
				{"name": "identifier", "type": "integer"},
				{"name": "full_name", "type": "string"},
				{"name": "contact_email", "type": "string"}
			]
		}

		suggestions = await self.service.suggest_field_mappings(source_schema, target_schema)

		assert isinstance(suggestions, list)
		assert len(suggestions) > 0
		for suggestion in suggestions:
			assert "source" in suggestion or "source_field" in suggestion
			assert "target" in suggestion or "target_field" in suggestion

	async def test_validate_data_quality(self):
		"""Test data quality validation"""
		await self.service.initialize()

		# Create job first
		job_config = generate_test_job_config()
		job = await self.service.create_job(job_config, "test_user")

		# Generate test data
		data_sample = generate_test_data_sample(100)

		# Validate quality
		quality_report = await self.service.validate_data_quality(job.id, data_sample)

		assert isinstance(quality_report, DataQualityReport)
		assert quality_report.job_id == job.id
		assert quality_report.total_records == 100
		assert quality_report.overall_quality_score >= 0.0
		assert quality_report.overall_quality_score <= 1.0
		assert isinstance(quality_report.validation_issues, dict)

	async def test_create_workflow(self):
		"""Test creating workflow"""
		await self.service.initialize()

		workflow_config = generate_test_workflow_config()
		user_id = "test_user"

		workflow = await self.service.create_workflow(workflow_config, user_id)

		assert isinstance(workflow, Workflow)
		assert workflow.name == workflow_config["name"]
		assert workflow.created_by == user_id
		assert len(workflow.steps) == len(workflow_config["steps"])
		assert_valid_uuid(workflow.id)

	async def test_execute_workflow(self):
		"""Test executing workflow"""
		await self.service.initialize()

		workflow_config = generate_test_workflow_config()
		workflow = await self.service.create_workflow(workflow_config, "test_user")

		execution_id = await self.service.execute_workflow(workflow)

		assert_valid_uuid(execution_id)
		assert workflow.status == JobStatus.COMPLETED
		assert workflow.last_execution_id == execution_id
		assert execution_id in workflow.execution_history

	async def test_get_system_performance_metrics(self):
		"""Test getting system performance metrics"""
		await self.service.initialize()

		metrics = await self.service.get_system_performance_metrics()

		assert isinstance(metrics, dict)
		assert "system_status" in metrics
		assert "uptime_seconds" in metrics
		assert "active_jobs_count" in metrics
		assert "total_jobs_created" in metrics
		assert "success_rate" in metrics

		assert metrics["system_status"] == self.service.health_status
		assert isinstance(metrics["uptime_seconds"], (int, float))
		assert isinstance(metrics["success_rate"], (int, float))

	async def test_optimize_job_performance(self):
		"""Test job performance optimization"""
		await self.service.initialize()

		job_config = generate_test_job_config()
		job = await self.service.create_job(job_config, "test_user")

		optimization_plan = await self.service.optimize_job_performance(job.id)

		assert isinstance(optimization_plan, dict)
		# Mock implementation returns recommendations
		assert "recommendations" in optimization_plan

	async def test_health_check(self):
		"""Test service health check"""
		await self.service.initialize()

		health_data = await self.service.health_check()

		assert isinstance(health_data, dict)
		assert "service" in health_data
		assert "status" in health_data
		assert "timestamp" in health_data
		assert "version" in health_data
		assert "components" in health_data
		assert "active_jobs" in health_data
		assert "performance_metrics" in health_data

		assert health_data["service"] == "imex"
		assert health_data["status"] in ["healthy", "degraded"]


class TestServiceValidation:
	"""Test service validation methods"""

	async def test_validate_job_configuration(self):
		"""Test job configuration validation"""
		service = ImportExportService()
		await service.initialize()

		job_config = generate_test_job_config()
		job = ImportExportJob(**job_config)

		# Should not raise exception for valid config
		await service._validate_job_configuration(job)

	async def test_validate_schema_mapping(self):
		"""Test schema mapping validation"""
		service = ImportExportService()
		await service.initialize()

		mapping_config = {
			"name": "Test Mapping",
			"field_mappings": [
				{
					"source_field": "id",
					"target_field": "identifier"
				}
			],
			"created_by": "test_user"
		}

		mapping = SchemaMapping(**mapping_config)

		# Should not raise exception for valid mapping
		await service._validate_schema_mapping(mapping)


class TestServicePerformance:
	"""Test service performance characteristics"""

	async def test_concurrent_job_creation(self):
		"""Test concurrent job creation"""
		service = ImportExportService()
		await service.initialize()

		async def create_job(index):
			job_config = generate_test_job_config()
			job_config["name"] = f"Concurrent Job {index}"
			return await service.create_job(job_config, f"user_{index}")

		# Create 10 jobs concurrently
		tasks = [create_job(i) for i in range(10)]
		jobs = await asyncio.gather(*tasks, return_exceptions=True)


		assert len(jobs) == 10
		assert len(service.active_jobs) == 10
		assert service.performance_metrics["jobs_created"] == 10

		# Verify all jobs have unique IDs
		job_ids = [job.id for job in jobs]
		assert len(set(job_ids)) == 10

	async def test_concurrent_job_execution(self):
		"""Test concurrent job execution"""
		service = ImportExportService()
		await service.initialize()

		# Create multiple jobs
		jobs = []
		for i in range(5):
			job_config = generate_test_job_config()
			job_config["name"] = f"Parallel Job {i}"
			job = await service.create_job(job_config, f"user_{i}")
			jobs.append(job)

		# Execute jobs concurrently
		tasks = [service.execute_job(job.id) for job in jobs]
		executions = await asyncio.gather(*tasks, return_exceptions=True)


		assert len(executions) == 5
		assert all(exec.status == JobStatus.COMPLETED for exec in executions)
		assert service.performance_metrics["jobs_executed"] == 5
		assert service.performance_metrics["jobs_completed"] == 5

	async def test_large_data_processing(self):
		"""Test processing large data samples"""
		service = ImportExportService()
		await service.initialize()

		# Create job with large data sample
		job_config = generate_test_job_config()
		job = await service.create_job(job_config, "test_user")

		# Generate large data sample
		large_data_sample = generate_test_data_sample(10000)

		# Validate data quality (should handle large dataset)
		quality_report = await service.validate_data_quality(job.id, large_data_sample)

		assert quality_report.total_records == 10000
		assert isinstance(quality_report.overall_quality_score, float)

	async def test_memory_efficiency(self):
		"""Test memory efficiency with streaming operations"""
		service = ImportExportService()
		await service.initialize()

		# Mock streaming data source
		mock_source = MockDataSource(generate_test_data_sample(1000))

		# Test streaming in chunks
		total_records = 0
		async for batch in service._stream_data_batches(mock_source, 100):
			assert len(batch) <= 100
			total_records += len(batch)

		assert total_records == 1000


class TestServiceIntegration:
	"""Test service integration with APG capabilities"""

	async def test_ai_client_integration(self):
		"""Test AI client integration"""
		service = ImportExportService()
		await service.initialize()

		assert service.ai_client is not None

		# Test AI-powered schema detection
		source_config = {
			"source_type": "file",
			"file_path": "/data/test.csv",
			"format": "csv"
		}

		from ..models import SourceConfig
		source = SourceConfig(**source_config)

		schema = await service.detect_schema_automatically(source)
		assert isinstance(schema, dict)

	async def test_etlp_client_integration(self):
		"""Test ETLP client integration"""
		service = ImportExportService()
		await service.initialize()

		assert service.etlp_client is not None

	async def test_conn_client_integration(self):
		"""Test connection client integration"""
		service = ImportExportService()
		await service.initialize()

		assert service.conn_client is not None

	async def test_audit_client_integration(self):
		"""Test audit client integration"""
		service = ImportExportService()
		await service.initialize()

		assert service.audit_client is not None

	async def test_notification_client_integration(self):
		"""Test notification client integration"""
		service = ImportExportService()
		await service.initialize()

		assert service.notification_client is not None


class TestServiceErrorHandling:
	"""Test service error handling"""

	async def test_service_initialization_failure(self):
		"""Test service initialization failure handling"""
		service = ImportExportService()

		# Mock a dependency failure
		with patch.object(service, '_initialize_apg_clients', side_effect=Exception("Mock failure")):
			with pytest.raises(RuntimeError) as exc_info:
				await service.initialize()

			assert "Service initialization failed" in str(exc_info.value)
			assert service.health_status == "failed"

	async def test_job_execution_failure(self):
		"""Test job execution failure handling"""
		service = ImportExportService()
		await service.initialize()

		job_config = generate_test_job_config()
		job = await service.create_job(job_config, "test_user")

		# Mock execution failure
		with patch.object(service, '_execute_import_job', side_effect=Exception("Mock execution failure")):
			with pytest.raises(RuntimeError) as exc_info:
				await service.execute_job(job.id)

			assert "Job execution failed" in str(exc_info.value)
			assert job.status == JobStatus.FAILED
			assert service.performance_metrics["jobs_failed"] == 1

	async def test_schema_detection_failure(self):
		"""Test schema detection failure handling"""
		service = ImportExportService()
		await service.initialize()

		source_config = {
			"source_type": "file",
			"file_path": "/nonexistent/file.csv",
			"format": "csv"
		}

		from ..models import SourceConfig
		source = SourceConfig(**source_config)

		# Mock detection failure
		with patch.object(service, '_initialize_data_source', side_effect=Exception("File not found")):
			with pytest.raises(RuntimeError) as exc_info:
				await service.detect_schema_automatically(source)

			assert "Schema detection failed" in str(exc_info.value)


class TestServiceSingleton:
	"""Test service singleton behavior"""

	def test_service_singleton(self):
		"""Test that imex_service is properly initialized"""
		assert imex_service is not None
		assert isinstance(imex_service, ImportExportService)
		assert imex_service.health_status == "healthy"

	async def test_service_singleton_initialization(self):
		"""Test singleton service initialization"""
		# Initialize the singleton service
		await imex_service.initialize()

		assert imex_service.health_status == "ready"
		assert imex_service.ai_client is not None
		assert imex_service.performance_metrics["jobs_created"] >= 0
