"""
Unit Tests for AICR Models
===========================

© 2025 Datacraft - www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>

Comprehensive unit tests for all AICR Pydantic models ensuring validation,
serialization, and business logic correctness with 100% coverage.
"""

import pytest
from datetime import datetime, timedelta
from typing import Any, Dict, List
from uuid import UUID

from pydantic import ValidationError
from uuid_extensions import uuid7str

from ..models import (
	AICRCapabilityBase,
	AICRModel,
	AICRInferenceRequest,
	AICRInferenceResponse,
	AICRPipeline,
	AICRMetric,
	ModelType,
	InferenceStatus,
	PipelineStatus,
	MetricType
)


class TestAICRCapabilityBase:
	"""Test cases for AICRCapabilityBase model."""

	def test_capability_base_creation(self):
		"""Test creating a basic capability instance."""
		capability = AICRCapabilityBase(
			name="test_capability",
			description="Test capability description"
		)

		assert capability.name == "test_capability"
		assert capability.description == "Test capability description"
		assert isinstance(capability.capability_id, str)
		assert len(capability.capability_id) > 0
		assert isinstance(capability.created_at, datetime)
		assert isinstance(capability.updated_at, datetime)
		assert capability.version == "1.0.0"
		assert capability.is_active == True
		assert capability.tags == []
		assert capability.metadata == {}

	def test_capability_base_validation(self):
		"""Test validation rules for capability base."""
		# Test required fields
		with pytest.raises(ValidationError) as exc_info:
			AICRCapabilityBase()

		errors = exc_info.value.errors()
		required_fields = {error['loc'][0] for error in errors if error['type'] == 'missing'}
		assert 'name' in required_fields
		assert 'description' in required_fields

	def test_capability_base_with_optional_fields(self):
		"""Test capability base with all optional fields."""
		metadata = {"key1": "value1", "key2": 42}
		tags = ["test", "capability"]

		capability = AICRCapabilityBase(
			name="test_capability",
			description="Test description",
			version="2.0.0",
			is_active=False,
			tags=tags,
			metadata=metadata
		)

		assert capability.version == "2.0.0"
		assert capability.is_active == False
		assert capability.tags == tags
		assert capability.metadata == metadata

	def test_capability_base_serialization(self):
		"""Test serialization and deserialization."""
		capability = AICRCapabilityBase(
			name="test_capability",
			description="Test description"
		)

		# Test model_dump
		data = capability.model_dump()
		assert isinstance(data, dict)
		assert data['name'] == "test_capability"
		assert data['description'] == "Test description"
		assert 'capability_id' in data
		assert 'created_at' in data
		assert 'updated_at' in data

		# Test round-trip serialization
		restored = AICRCapabilityBase.model_validate(data)
		assert restored.name == capability.name
		assert restored.description == capability.description
		assert restored.capability_id == capability.capability_id


class TestAICRModel:
	"""Test cases for AICRModel."""

	def test_model_creation_minimal(self):
		"""Test creating a model with minimal required fields."""
		model = AICRModel(
			name="test_model",
			description="Test model description",
			model_type=ModelType.CLASSIFICATION,
			framework="pytorch"
		)

		assert model.name == "test_model"
		assert model.description == "Test model description"
		assert model.model_type == ModelType.CLASSIFICATION
		assert model.framework == "pytorch"
		assert isinstance(model.model_id, str)
		assert model.version == "1.0.0"
		assert model.status == "inactive"
		assert model.input_schema == {}
		assert model.output_schema == {}
		assert model.configuration == {}
		assert model.performance_metrics == {}
		assert model.file_path is None
		assert model.deployment_count == 0
		assert model.last_inference is None

	def test_model_creation_complete(self):
		"""Test creating a model with all fields."""
		input_schema = {"type": "object", "properties": {"data": {"type": "array"}}}
		output_schema = {"type": "object", "properties": {"predictions": {"type": "array"}}}
		configuration = {"batch_size": 32, "device": "cuda"}
		metrics = {"accuracy": 0.95, "f1_score": 0.92}

		model = AICRModel(
			name="complete_model",
			description="Complete model with all fields",
			model_type=ModelType.REGRESSION,
			framework="tensorflow",
			version="2.1.0",
			status="active",
			input_schema=input_schema,
			output_schema=output_schema,
			configuration=configuration,
			performance_metrics=metrics,
			file_path="/models/complete_model.h5",
			deployment_count=5
		)

		assert model.name == "complete_model"
		assert model.model_type == ModelType.REGRESSION
		assert model.framework == "tensorflow"
		assert model.version == "2.1.0"
		assert model.status == "active"
		assert model.input_schema == input_schema
		assert model.output_schema == output_schema
		assert model.configuration == configuration
		assert model.performance_metrics == metrics
		assert model.file_path == "/models/complete_model.h5"
		assert model.deployment_count == 5

	def test_model_validation_errors(self):
		"""Test model validation errors."""
		# Test missing required fields
		with pytest.raises(ValidationError) as exc_info:
			AICRModel()

		errors = exc_info.value.errors()
		required_fields = {error['loc'][0] for error in errors if error['type'] == 'missing'}
		assert 'name' in required_fields
		assert 'description' in required_fields
		assert 'model_type' in required_fields
		assert 'framework' in required_fields

		# Test invalid model type
		with pytest.raises(ValidationError):
			AICRModel(
				name="test",
				description="test",
				model_type="invalid_type",
				framework="pytorch"
			)

	def test_model_serialization(self):
		"""Test model serialization and deserialization."""
		model = AICRModel(
			name="test_model",
			description="Test model",
			model_type=ModelType.CLASSIFICATION,
			framework="pytorch",
			configuration={"lr": 0.001}
		)

		# Test serialization
		data = model.model_dump()
		assert isinstance(data, dict)
		assert data['name'] == "test_model"
		assert data['model_type'] == "classification"
		assert data['framework'] == "pytorch"
		assert data['configuration'] == {"lr": 0.001}

		# Test deserialization
		restored = AICRModel.model_validate(data)
		assert restored.name == model.name
		assert restored.model_type == model.model_type
		assert restored.framework == model.framework
		assert restored.configuration == model.configuration


class TestAICRInferenceRequest:
	"""Test cases for AICRInferenceRequest."""

	def test_inference_request_creation(self):
		"""Test creating an inference request."""
		input_data = {"features": [1, 2, 3, 4, 5]}
		parameters = {"temperature": 0.7}

		request = AICRInferenceRequest(
			model_id="test_model_id",
			input_data=input_data,
			parameters=parameters,
			output_format="json"
		)

		assert request.model_id == "test_model_id"
		assert request.input_data == input_data
		assert request.parameters == parameters
		assert request.output_format == "json"
		assert isinstance(request.request_id, str)
		assert isinstance(request.timestamp, datetime)
		assert request.priority == "normal"
		assert request.timeout_seconds == 30
		assert request.metadata == {}

	def test_inference_request_validation(self):
		"""Test inference request validation."""
		# Test missing required fields
		with pytest.raises(ValidationError) as exc_info:
			AICRInferenceRequest()

		errors = exc_info.value.errors()
		required_fields = {error['loc'][0] for error in errors if error['type'] == 'missing'}
		assert 'model_id' in required_fields
		assert 'input_data' in required_fields

		# Test valid request
		request = AICRInferenceRequest(
			model_id="test_model",
			input_data={"data": [1, 2, 3]}
		)
		assert request.model_id == "test_model"
		assert request.input_data == {"data": [1, 2, 3]}

	def test_inference_request_serialization(self):
		"""Test inference request serialization."""
		request = AICRInferenceRequest(
			model_id="test_model",
			input_data={"data": [1, 2, 3]},
			priority="high",
			timeout_seconds=60
		)

		data = request.model_dump()
		assert data['model_id'] == "test_model"
		assert data['input_data'] == {"data": [1, 2, 3]}
		assert data['priority'] == "high"
		assert data['timeout_seconds'] == 60

		restored = AICRInferenceRequest.model_validate(data)
		assert restored.model_id == request.model_id
		assert restored.input_data == request.input_data
		assert restored.priority == request.priority


class TestAICRInferenceResponse:
	"""Test cases for AICRInferenceResponse."""

	def test_inference_response_creation(self):
		"""Test creating an inference response."""
		predictions = {"class": "cat", "confidence": 0.95}
		confidence_scores = [0.05, 0.95]
		metadata = {"model_version": "1.0", "preprocessing_time": 10}

		response = AICRInferenceResponse(
			request_id="test_request_id",
			model_id="test_model_id",
			status=InferenceStatus.COMPLETED,
			predictions=predictions,
			confidence_scores=confidence_scores,
			processing_time_ms=150.5,
			metadata=metadata
		)

		assert response.request_id == "test_request_id"
		assert response.model_id == "test_model_id"
		assert response.status == InferenceStatus.COMPLETED
		assert response.predictions == predictions
		assert response.confidence_scores == confidence_scores
		assert response.processing_time_ms == 150.5
		assert response.metadata == metadata
		assert isinstance(response.response_id, str)
		assert isinstance(response.timestamp, datetime)
		assert response.error_message is None

	def test_inference_response_validation(self):
		"""Test inference response validation."""
		# Test missing required fields
		with pytest.raises(ValidationError) as exc_info:
			AICRInferenceResponse()

		errors = exc_info.value.errors()
		required_fields = {error['loc'][0] for error in errors if error['type'] == 'missing'}
		assert 'request_id' in required_fields
		assert 'model_id' in required_fields
		assert 'status' in required_fields

		# Test valid minimal response
		response = AICRInferenceResponse(
			request_id="test_request",
			model_id="test_model",
			status=InferenceStatus.PENDING
		)
		assert response.request_id == "test_request"
		assert response.model_id == "test_model"
		assert response.status == InferenceStatus.PENDING

	def test_inference_response_with_error(self):
		"""Test inference response with error status."""
		response = AICRInferenceResponse(
			request_id="error_request",
			model_id="test_model",
			status=InferenceStatus.FAILED,
			error_message="Model not found"
		)

		assert response.status == InferenceStatus.FAILED
		assert response.error_message == "Model not found"
		assert response.predictions is None


class TestAICRPipeline:
	"""Test cases for AICRPipeline."""

	def test_pipeline_creation(self):
		"""Test creating a pipeline."""
		stages = ["data_loading", "preprocessing", "training", "evaluation"]
		configuration = {"epochs": 100, "batch_size": 32}
		data_sources = ["dataset1", "dataset2"]

		pipeline = AICRPipeline(
			name="test_pipeline",
			description="Test ML pipeline",
			pipeline_type="training",
			stages=stages,
			configuration=configuration,
			data_sources=data_sources
		)

		assert pipeline.name == "test_pipeline"
		assert pipeline.description == "Test ML pipeline"
		assert pipeline.pipeline_type == "training"
		assert pipeline.stages == stages
		assert pipeline.configuration == configuration
		assert pipeline.data_sources == data_sources
		assert isinstance(pipeline.pipeline_id, str)
		assert pipeline.status == PipelineStatus.PENDING
		assert pipeline.execution_count == 0
		assert pipeline.success_rate == 0.0
		assert pipeline.last_execution is None
		assert pipeline.schedule is None

	def test_pipeline_validation(self):
		"""Test pipeline validation."""
		# Test missing required fields
		with pytest.raises(ValidationError) as exc_info:
			AICRPipeline()

		errors = exc_info.value.errors()
		required_fields = {error['loc'][0] for error in errors if error['type'] == 'missing'}
		assert 'name' in required_fields
		assert 'description' in required_fields
		assert 'pipeline_type' in required_fields
		assert 'stages' in required_fields

		# Test empty stages validation
		with pytest.raises(ValidationError):
			AICRPipeline(
				name="test",
				description="test",
				pipeline_type="training",
				stages=[]
			)

	def test_pipeline_with_schedule(self):
		"""Test pipeline with execution schedule."""
		pipeline = AICRPipeline(
			name="scheduled_pipeline",
			description="Pipeline with schedule",
			pipeline_type="batch_inference",
			stages=["load_data", "predict", "save_results"],
			schedule="0 0 * * *"  # Daily at midnight
		)

		assert pipeline.schedule == "0 0 * * *"
		assert pipeline.pipeline_type == "batch_inference"

	def test_pipeline_execution_tracking(self):
		"""Test pipeline execution tracking fields."""
		pipeline = AICRPipeline(
			name="tracking_pipeline",
			description="Pipeline with execution tracking",
			pipeline_type="evaluation",
			stages=["evaluate"],
			execution_count=10,
			success_rate=85.0,
			last_execution=datetime.utcnow()
		)

		assert pipeline.execution_count == 10
		assert pipeline.success_rate == 85.0
		assert isinstance(pipeline.last_execution, datetime)


class TestAICRMetric:
	"""Test cases for AICRMetric."""

	def test_metric_creation(self):
		"""Test creating a metric."""
		labels = {"component": "inference_engine", "model": "test_model"}

		metric = AICRMetric(
			metric_name="inference_latency",
			metric_type=MetricType.HISTOGRAM,
			value=125.5,
			labels=labels,
			unit="milliseconds",
			source_component="inference_engine"
		)

		assert metric.metric_name == "inference_latency"
		assert metric.metric_type == MetricType.HISTOGRAM
		assert metric.value == 125.5
		assert metric.labels == labels
		assert metric.unit == "milliseconds"
		assert metric.source_component == "inference_engine"
		assert isinstance(metric.metric_id, str)
		assert isinstance(metric.timestamp, datetime)
		assert metric.description == ""
		assert metric.metadata == {}

	def test_metric_validation(self):
		"""Test metric validation."""
		# Test missing required fields
		with pytest.raises(ValidationError) as exc_info:
			AICRMetric()

		errors = exc_info.value.errors()
		required_fields = {error['loc'][0] for error in errors if error['type'] == 'missing'}
		assert 'metric_name' in required_fields
		assert 'metric_type' in required_fields
		assert 'value' in required_fields
		assert 'source_component' in required_fields

		# Test valid minimal metric
		metric = AICRMetric(
			metric_name="cpu_usage",
			metric_type=MetricType.GAUGE,
			value=75.2,
			source_component="system"
		)
		assert metric.metric_name == "cpu_usage"
		assert metric.value == 75.2

	def test_metric_with_metadata(self):
		"""Test metric with additional metadata."""
		metadata = {
			"collection_method": "prometheus",
			"resolution": "1s",
			"aggregation": "avg"
		}

		metric = AICRMetric(
			metric_name="memory_usage",
			metric_type=MetricType.GAUGE,
			value=1024.0,
			source_component="system",
			unit="MB",
			description="System memory usage",
			metadata=metadata
		)

		assert metric.unit == "MB"
		assert metric.description == "System memory usage"
		assert metric.metadata == metadata

	def test_metric_serialization(self):
		"""Test metric serialization."""
		metric = AICRMetric(
			metric_name="test_metric",
			metric_type=MetricType.COUNTER,
			value=42.0,
			source_component="test_component",
			labels={"env": "test"}
		)

		data = metric.model_dump()
		assert data['metric_name'] == "test_metric"
		assert data['metric_type'] == "counter"
		assert data['value'] == 42.0
		assert data['source_component'] == "test_component"
		assert data['labels'] == {"env": "test"}

		restored = AICRMetric.model_validate(data)
		assert restored.metric_name == metric.metric_name
		assert restored.metric_type == metric.metric_type
		assert restored.value == metric.value


class TestEnumerations:
	"""Test cases for enumeration types."""

	def test_model_type_enum(self):
		"""Test ModelType enumeration."""
		assert ModelType.CLASSIFICATION == "classification"
		assert ModelType.REGRESSION == "regression"
		assert ModelType.CLUSTERING == "clustering"
		assert ModelType.ANOMALY_DETECTION == "anomaly_detection"
		assert ModelType.TIME_SERIES == "time_series"
		assert ModelType.NLP == "nlp"
		assert ModelType.COMPUTER_VISION == "computer_vision"
		assert ModelType.RECOMMENDATION == "recommendation"
		assert ModelType.REINFORCEMENT_LEARNING == "reinforcement_learning"

		# Test all values are unique
		values = [item.value for item in ModelType]
		assert len(values) == len(set(values))

	def test_inference_status_enum(self):
		"""Test InferenceStatus enumeration."""
		assert InferenceStatus.PENDING == "pending"
		assert InferenceStatus.RUNNING == "running"
		assert InferenceStatus.COMPLETED == "completed"
		assert InferenceStatus.FAILED == "failed"
		assert InferenceStatus.CANCELLED == "cancelled"

		# Test all values are unique
		values = [item.value for item in InferenceStatus]
		assert len(values) == len(set(values))

	def test_pipeline_status_enum(self):
		"""Test PipelineStatus enumeration."""
		assert PipelineStatus.PENDING == "pending"
		assert PipelineStatus.RUNNING == "running"
		assert PipelineStatus.COMPLETED == "completed"
		assert PipelineStatus.FAILED == "failed"
		assert PipelineStatus.PAUSED == "paused"
		assert PipelineStatus.CANCELLED == "cancelled"

		# Test all values are unique
		values = [item.value for item in PipelineStatus]
		assert len(values) == len(set(values))

	def test_metric_type_enum(self):
		"""Test MetricType enumeration."""
		assert MetricType.COUNTER == "counter"
		assert MetricType.GAUGE == "gauge"
		assert MetricType.HISTOGRAM == "histogram"
		assert MetricType.SUMMARY == "summary"

		# Test all values are unique
		values = [item.value for item in MetricType]
		assert len(values) == len(set(values))


class TestModelInteractions:
	"""Test cases for model interactions and business logic."""

	def test_model_lifecycle(self):
		"""Test model lifecycle status changes."""
		model = AICRModel(
			name="lifecycle_model",
			description="Model for testing lifecycle",
			model_type=ModelType.CLASSIFICATION,
			framework="pytorch"
		)

		# Initial state
		assert model.status == "inactive"
		assert model.deployment_count == 0
		assert model.last_inference is None

		# Simulate deployment
		model.status = "active"
		model.deployment_count = 1

		assert model.status == "active"
		assert model.deployment_count == 1

	def test_pipeline_execution_flow(self):
		"""Test pipeline execution status flow."""
		pipeline = AICRPipeline(
			name="execution_pipeline",
			description="Pipeline for testing execution",
			pipeline_type="training",
			stages=["preprocess", "train", "evaluate"]
		)

		# Initial state
		assert pipeline.status == PipelineStatus.PENDING
		assert pipeline.execution_count == 0
		assert pipeline.success_rate == 0.0

		# Simulate execution
		pipeline.status = PipelineStatus.RUNNING
		pipeline.execution_count = 1

		# Simulate completion
		pipeline.status = PipelineStatus.COMPLETED
		pipeline.success_rate = 100.0
		pipeline.last_execution = datetime.utcnow()

		assert pipeline.status == PipelineStatus.COMPLETED
		assert pipeline.success_rate == 100.0
		assert isinstance(pipeline.last_execution, datetime)

	def test_inference_request_response_matching(self):
		"""Test matching inference requests with responses."""
		request = AICRInferenceRequest(
			model_id="test_model",
			input_data={"data": [1, 2, 3]}
		)

		response = AICRInferenceResponse(
			request_id=request.request_id,
			model_id=request.model_id,
			status=InferenceStatus.COMPLETED,
			predictions={"result": "success"}
		)

		# Test that request and response are properly linked
		assert response.request_id == request.request_id
		assert response.model_id == request.model_id

		# Test timestamps (response should be after request)
		assert response.timestamp >= request.timestamp

	def test_metric_aggregation_compatibility(self):
		"""Test that metrics can be properly aggregated."""
		metrics = []

		# Create multiple metrics with same name but different values
		for i in range(5):
			metric = AICRMetric(
				metric_name="cpu_usage",
				metric_type=MetricType.GAUGE,
				value=float(70 + i),
				source_component="system",
				labels={"instance": f"server-{i}"}
			)
			metrics.append(metric)

		# Test that all metrics have the same name and type
		metric_names = {m.metric_name for m in metrics}
		metric_types = {m.metric_type for m in metrics}

		assert len(metric_names) == 1
		assert len(metric_types) == 1
		assert list(metric_names)[0] == "cpu_usage"
		assert list(metric_types)[0] == MetricType.GAUGE

		# Test that values are different (representing different measurements)
		values = [m.value for m in metrics]
		assert len(set(values)) == 5  # All values should be unique


if __name__ == "__main__":
	pytest.main([__file__])