"""
Test APG Import/Export Models

Comprehensive tests for Pydantic v2 data models with APG integration.
Tests validation, serialization, and business logic.
"""

import pytest
from datetime import datetime, timezone
from pydantic import ValidationError
from uuid_extensions import uuid7str

from ..models import (
	ImportExportJob, JobExecution, JobStatus, JobType, ProcessingMetrics,
	DataQualityReport, Workflow, WorkflowStep, SchemaMapping, FieldMapping,
	ValidationRule, TransformationStep, SourceConfig, TargetConfig,
	DataFormat, SourceType, CompressionType, ValidationLevel, ProcessingPriority,
	ErrorHandlingStrategy, ConnectionTemplate, MonitoringAlert
)
from . import (
	generate_test_job_config, generate_test_data_sample, generate_test_schema,
	TEST_CONFIG, assert_valid_uuid, assert_valid_timestamp
)


class TestImportExportJob:
	"""Test ImportExportJob model"""

	def test_job_creation_valid(self):
		"""Test valid job creation"""
		job_config = generate_test_job_config()
		job = ImportExportJob(**job_config)

		assert job.name == job_config["name"]
		assert job.job_type == JobType(job_config["job_type"])
		assert job.status == JobStatus.DRAFT
		assert_valid_uuid(job.id)
		assert job.tenant_id == TEST_CONFIG["test_tenant_id"]
		assert isinstance(job.created_at, datetime)
		assert isinstance(job.updated_at, datetime)

	def test_job_creation_invalid_tenant(self):
		"""Test job creation with invalid tenant"""
		job_config = generate_test_job_config()
		job_config["tenant_id"] = ""  # Empty tenant ID

		with pytest.raises(ValidationError) as exc_info:
			ImportExportJob(**job_config)

		assert "Tenant ID cannot be empty" in str(exc_info.value)

	def test_job_creation_missing_required_fields(self):
		"""Test job creation with missing required fields"""
		job_config = generate_test_job_config()
		del job_config["name"]  # Remove required field

		with pytest.raises(ValidationError) as exc_info:
			ImportExportJob(**job_config)

		assert "Field required" in str(exc_info.value)

	def test_job_serialization(self):
		"""Test job serialization to dict"""
		job_config = generate_test_job_config()
		job = ImportExportJob(**job_config)

		job_dict = job.model_dump()

		assert isinstance(job_dict, dict)
		assert job_dict["name"] == job.name
		assert job_dict["job_type"] == job.job_type.value
		assert job_dict["status"] == job.status.value
		assert "id" in job_dict
		assert "created_at" in job_dict

	def test_job_deserialization(self):
		"""Test job deserialization from dict"""
		job_config = generate_test_job_config()
		original_job = ImportExportJob(**job_config)
		job_dict = original_job.model_dump()

		restored_job = ImportExportJob.model_validate(job_dict)

		assert restored_job.name == original_job.name
		assert restored_job.job_type == original_job.job_type
		assert restored_job.id == original_job.id

	def test_job_validation_level_enum(self):
		"""Test validation level enum handling"""
		job_config = generate_test_job_config()
		job_config["validation_level"] = "strict"

		job = ImportExportJob(**job_config)
		assert job.validation_level == ValidationLevel.STRICT

	def test_job_priority_enum(self):
		"""Test priority enum handling"""
		job_config = generate_test_job_config()
		job_config["priority"] = "high"

		job = ImportExportJob(**job_config)
		assert job.priority == ProcessingPriority.HIGH

	def test_job_error_handling_enum(self):
		"""Test error handling enum"""
		job_config = generate_test_job_config()
		job_config["error_handling"] = "fail_fast"

		job = ImportExportJob(**job_config)
		assert job.error_handling == ErrorHandlingStrategy.FAIL_FAST

	def test_job_updated_at_auto_update(self):
		"""Test that updated_at is automatically set"""
		job_config = generate_test_job_config()
		job = ImportExportJob(**job_config)

		original_updated_at = job.updated_at

		# Simulate update
		job.description = "Updated description"
		job_dict = job.model_dump()
		updated_job = ImportExportJob.model_validate(job_dict)

		assert updated_job.updated_at >= original_updated_at


class TestSourceConfig:
	"""Test SourceConfig model"""

	def test_source_config_creation(self):
		"""Test valid source config creation"""
		config = {
			"source_type": "file",
			"file_path": "/data/test.csv",
			"format": "csv",
			"has_header": True,
			"chunk_size": 5000
		}

		source = SourceConfig(**config)

		assert source.source_type == SourceType.FILE
		assert source.format == DataFormat.CSV
		assert source.chunk_size == 5000
		assert source.has_header is True

	def test_source_config_validation_chunk_size(self):
		"""Test chunk size validation"""
		config = {
			"source_type": "file",
			"file_path": "/data/test.csv",
			"format": "csv",
			"chunk_size": -100  # Invalid negative value
		}

		with pytest.raises(ValidationError) as exc_info:
			SourceConfig(**config)

		assert "Value must be positive" in str(exc_info.value)

	def test_source_config_compression(self):
		"""Test compression type handling"""
		config = {
			"source_type": "file",
			"file_path": "/data/test.csv.gz",
			"format": "csv",
			"compression": "gzip"
		}

		source = SourceConfig(**config)
		assert source.compression == CompressionType.GZIP

	def test_source_config_custom_options(self):
		"""Test custom options handling"""
		config = {
			"source_type": "api",
			"api_config": {"url": "https://api.example.com"},
			"format": "json",
			"custom_options": {"rate_limit": 100, "auth_token": "secret"}
		}

		source = SourceConfig(**config)
		assert source.custom_options["rate_limit"] == 100
		assert source.custom_options["auth_token"] == "secret"


class TestTargetConfig:
	"""Test TargetConfig model"""

	def test_target_config_creation(self):
		"""Test valid target config creation"""
		config = {
			"target_type": "database",
			"connection_id": "postgres_main",
			"format": "parquet",
			"batch_size": 2000,
			"overwrite_existing": True
		}

		target = TargetConfig(**config)

		assert target.target_type == SourceType.DATABASE
		assert target.format == DataFormat.PARQUET
		assert target.batch_size == 2000
		assert target.overwrite_existing is True

	def test_target_config_validation_batch_size(self):
		"""Test batch size validation"""
		config = {
			"target_type": "file",
			"file_path": "/output/data.json",
			"format": "json",
			"batch_size": 0  # Invalid zero value
		}

		with pytest.raises(ValidationError) as exc_info:
			TargetConfig(**config)

		assert "Value must be positive" in str(exc_info.value)


class TestSchemaMapping:
	"""Test SchemaMapping model"""

	def test_schema_mapping_creation(self):
		"""Test valid schema mapping creation"""
		field_mappings = [
			{
				"source_field": "customer_id",
				"target_field": "id",
				"data_type": "integer",
				"nullable": False
			},
			{
				"source_field": "customer_name",
				"target_field": "name",
				"data_type": "string",
				"transformation": "strip().title()"
			}
		]

		mapping_config = {
			"name": "Customer Mapping",
			"description": "Maps customer data fields",
			"field_mappings": field_mappings,
			"auto_map_similar_fields": True,
			"created_by": "test_user"
		}

		mapping = SchemaMapping(**mapping_config)

		assert mapping.name == "Customer Mapping"
		assert len(mapping.field_mappings) == 2
		assert mapping.auto_map_similar_fields is True
		assert_valid_uuid(mapping.id)

	def test_field_mapping_creation(self):
		"""Test field mapping creation"""
		field_config = {
			"source_field": "email_address",
			"target_field": "email",
			"data_type": "string",
			"validation_rules": ["email_format", "not_empty"]
		}

		field_mapping = FieldMapping(**field_config)

		assert field_mapping.source_field == "email_address"
		assert field_mapping.target_field == "email"
		assert len(field_mapping.validation_rules) == 2


class TestValidationRule:
	"""Test ValidationRule model"""

	def test_validation_rule_creation(self):
		"""Test valid validation rule creation"""
		rule_config = {
			"name": "Email Format Check",
			"description": "Validates email format",
			"rule_type": "pattern",
			"field_name": "email",
			"parameters": {"pattern": r"^[^@]+@[^@]+\.[^@]+$"},
			"error_message": "Invalid email format",
			"severity": "error"
		}

		rule = ValidationRule(**rule_config)

		assert rule.name == "Email Format Check"
		assert rule.rule_type == "pattern"
		assert rule.field_name == "email"
		assert rule.severity == "error"
		assert_valid_uuid(rule.id)

	def test_validation_rule_defaults(self):
		"""Test validation rule default values"""
		rule_config = {
			"name": "Required Field",
			"rule_type": "required",
			"field_name": "id",
			"error_message": "ID is required"
		}

		rule = ValidationRule(**rule_config)

		assert rule.severity == "error"  # Default value
		assert rule.enabled is True  # Default value
		assert isinstance(rule.parameters, dict)


class TestProcessingMetrics:
	"""Test ProcessingMetrics model"""

	def test_metrics_creation(self):
		"""Test valid metrics creation"""
		metrics_config = {
			"records_processed": 10000,
			"records_successful": 9800,
			"records_failed": 200,
			"processing_time_seconds": 45.5,
			"throughput_records_per_second": 219.78,
			"memory_usage_mb": 512.5,
			"cpu_usage_percent": 75.2
		}

		metrics = ProcessingMetrics(**metrics_config)

		assert metrics.records_processed == 10000
		assert metrics.records_successful == 9800
		assert metrics.records_failed == 200
		assert metrics.throughput_records_per_second == 219.78
		assert isinstance(metrics.last_updated, datetime)

	def test_metrics_validation_negative_values(self):
		"""Test metrics validation for negative values"""
		metrics_config = {
			"records_processed": 1000,
			"processing_time_seconds": -10.0  # Invalid negative value
		}

		with pytest.raises(ValidationError) as exc_info:
			ProcessingMetrics(**metrics_config)

		assert "Value must be non-negative" in str(exc_info.value)

	def test_metrics_defaults(self):
		"""Test metrics default values"""
		metrics = ProcessingMetrics()

		assert metrics.records_processed == 0
		assert metrics.records_successful == 0
		assert metrics.records_failed == 0
		assert metrics.processing_time_seconds == 0.0
		assert metrics.throughput_records_per_second == 0.0
		assert isinstance(metrics.error_summary, dict)
		assert isinstance(metrics.validation_summary, dict)


class TestJobExecution:
	"""Test JobExecution model"""

	def test_execution_creation(self):
		"""Test valid execution creation"""
		execution_config = {
			"job_id": uuid7str(),
			"execution_number": 1,
			"status": "running",
			"started_at": datetime.now(timezone.utc)
		}

		execution = JobExecution(**execution_config)

		assert execution.job_id == execution_config["job_id"]
		assert execution.execution_number == 1
		assert execution.status == JobStatus.RUNNING
		assert_valid_uuid(execution.id)
		assert isinstance(execution.metrics, ProcessingMetrics)

	def test_execution_status_enum(self):
		"""Test execution status enum handling"""
		execution_config = {
			"job_id": uuid7str(),
			"execution_number": 1,
			"status": "completed"
		}

		execution = JobExecution(**execution_config)
		assert execution.status == JobStatus.COMPLETED

	def test_execution_with_metrics(self):
		"""Test execution with custom metrics"""
		metrics_data = {
			"records_processed": 5000,
			"records_successful": 4950,
			"records_failed": 50
		}

		execution_config = {
			"job_id": uuid7str(),
			"execution_number": 1,
			"metrics": metrics_data
		}

		execution = JobExecution(**execution_config)
		assert execution.metrics.records_processed == 5000
		assert execution.metrics.records_successful == 4950
		assert execution.metrics.records_failed == 50


class TestWorkflow:
	"""Test Workflow model"""

	def test_workflow_creation(self):
		"""Test valid workflow creation"""
		steps = [
			{
				"name": "Extract Data",
				"step_type": "import",
				"configuration": {"source": "database"},
				"dependencies": []
			},
			{
				"name": "Transform Data",
				"step_type": "transform",
				"configuration": {"script": "clean_data.py"},
				"dependencies": ["Extract Data"]
			}
		]

		workflow_config = {
			"name": "Data Processing Workflow",
			"description": "Complete data processing pipeline",
			"tenant_id": TEST_CONFIG["test_tenant_id"],
			"steps": steps,
			"parallel_execution": False,
			"created_by": "test_user"
		}

		workflow = Workflow(**workflow_config)

		assert workflow.name == "Data Processing Workflow"
		assert len(workflow.steps) == 2
		assert workflow.parallel_execution is False
		assert workflow.status == JobStatus.DRAFT
		assert_valid_uuid(workflow.id)

	def test_workflow_step_creation(self):
		"""Test workflow step creation"""
		step_config = {
			"name": "Data Validation",
			"description": "Validate imported data",
			"step_type": "validate",
			"configuration": {"rules": ["completeness", "accuracy"]},
			"dependencies": ["import_step"],
			"timeout_minutes": 30
		}

		step = WorkflowStep(**step_config)

		assert step.name == "Data Validation"
		assert step.step_type == "validate"
		assert step.timeout_minutes == 30
		assert_valid_uuid(step.id)


class TestDataQualityReport:
	"""Test DataQualityReport model"""

	def test_quality_report_creation(self):
		"""Test valid quality report creation"""
		report_config = {
			"job_id": uuid7str(),
			"execution_id": uuid7str(),
			"total_records": 10000,
			"valid_records": 9500,
			"invalid_records": 500,
			"completeness_score": 0.95,
			"consistency_score": 0.88,
			"accuracy_score": 0.92,
			"overall_quality_score": 0.916
		}

		report = DataQualityReport(**report_config)

		assert report.total_records == 10000
		assert report.valid_records == 9500
		assert report.invalid_records == 500
		assert report.overall_quality_score == 0.916
		assert_valid_uuid(report.id)
		assert isinstance(report.generated_at, datetime)

	def test_quality_report_score_validation(self):
		"""Test quality score validation"""
		report_config = {
			"job_id": uuid7str(),
			"execution_id": uuid7str(),
			"total_records": 1000,
			"valid_records": 900,
			"invalid_records": 100,
			"completeness_score": -0.5,  # Invalid negative score
			"consistency_score": 0.8,
			"accuracy_score": 0.9,
			"overall_quality_score": 0.8
		}

		with pytest.raises(ValidationError) as exc_info:
			DataQualityReport(**report_config)

		assert "Value must be non-negative" in str(exc_info.value)


class TestConnectionTemplate:
	"""Test ConnectionTemplate model"""

	def test_template_creation(self):
		"""Test valid template creation"""
		template_config = {
			"name": "PostgreSQL CSV Import",
			"description": "Standard template for PostgreSQL CSV imports",
			"tenant_id": TEST_CONFIG["test_tenant_id"],
			"category": "database",
			"source_template": {
				"source_type": "file",
				"format": "csv",
				"has_header": True
			},
			"target_template": {
				"target_type": "database",
				"format": "sql"
			},
			"created_by": "admin"
		}

		template = ConnectionTemplate(**template_config)

		assert template.name == "PostgreSQL CSV Import"
		assert template.category == "database"
		assert template.usage_count == 0
		assert_valid_uuid(template.id)


class TestMonitoringAlert:
	"""Test MonitoringAlert model"""

	def test_alert_creation(self):
		"""Test valid alert creation"""
		alert_config = {
			"name": "High Error Rate Alert",
			"description": "Alert when error rate exceeds 5%",
			"tenant_id": TEST_CONFIG["test_tenant_id"],
			"metric_name": "error_rate",
			"threshold_value": 5.0,
			"comparison_operator": "gt",
			"evaluation_window_minutes": 10,
			"notification_channels": ["email", "slack"],
			"created_by": "admin"
		}

		alert = MonitoringAlert(**alert_config)

		assert alert.name == "High Error Rate Alert"
		assert alert.threshold_value == 5.0
		assert alert.comparison_operator == "gt"
		assert alert.enabled is True
		assert_valid_uuid(alert.id)


class TestModelRegistration:
	"""Test model registry functionality"""

	def test_model_registry_completeness(self):
		"""Test that all models are registered"""
		from ..models import model_registry

		expected_models = [
			"ImportExportJob", "JobExecution", "SourceConfig", "TargetConfig",
			"SchemaMapping", "ValidationRule", "TransformationStep",
			"ProcessingMetrics", "DataQualityReport", "Workflow", "WorkflowStep",
			"ConnectionTemplate", "MonitoringAlert"
		]

		for model_name in expected_models:
			assert model_name in model_registry, f"Model {model_name} not in registry"

	def test_model_registry_classes(self):
		"""Test that registry contains actual model classes"""
		from ..models import model_registry

		for model_name, model_class in model_registry.items():
			assert hasattr(model_class, "model_validate")
			assert hasattr(model_class, "model_dump")


class TestModelIntegration:
	"""Test model integration and relationships"""

	def test_job_with_execution(self):
		"""Test job with execution relationship"""
		job_config = generate_test_job_config()
		job = ImportExportJob(**job_config)

		execution_config = {
			"job_id": job.id,
			"execution_number": 1,
			"status": "running"
		}
		execution = JobExecution(**execution_config)

		assert execution.job_id == job.id
		assert execution.execution_number == 1

	def test_job_with_schema_mapping(self):
		"""Test job with schema mapping relationship"""
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

		job_config = generate_test_job_config()
		job_config["schema_mapping"] = mapping
		job = ImportExportJob(**job_config)

		assert job.schema_mapping.name == "Test Mapping"
		assert len(job.schema_mapping.field_mappings) == 1