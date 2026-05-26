"""
Pytest Configuration and Fixtures for AICR Tests
==================================================

© 2025 Datacraft - www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>

Shared pytest configuration, fixtures, and utilities for comprehensive
testing of the AI Core Framework capability with proper async support,
mocking infrastructure, and test data generation.
"""

import pytest
import asyncio
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, AsyncGenerator
from unittest.mock import Mock, AsyncMock, patch

import numpy as np

from ..models import (
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
from ..service import AICoreService
from ..monitoring import AIMonitoringSystem, MetricsCollector, AlertManager, PerformanceAnalyzer
from ..ml_pipeline import MLPipelineFramework
from ..security import SecurityManager


# Pytest configuration
pytest_plugins = ['pytest_asyncio']


@pytest.fixture(scope="session")
def event_loop():
	"""Create an instance of the default event loop for the test session."""
	loop = asyncio.get_event_loop_policy().new_event_loop()
	yield loop
	loop.close()


@pytest.fixture
def temp_directory():
	"""Create a temporary directory for test files."""
	temp_dir = tempfile.mkdtemp()
	yield Path(temp_dir)
	shutil.rmtree(temp_dir)


@pytest.fixture
def sample_model_data():
	"""Sample model data for testing."""
	return {
		"name": "test_classification_model",
		"description": "A test classification model for unit testing",
		"model_type": "classification",
		"framework": "pytorch",
		"version": "1.0.0",
		"input_schema": {
			"type": "object",
			"properties": {
				"features": {
					"type": "array",
					"items": {"type": "number"}
				}
			}
		},
		"output_schema": {
			"type": "object",
			"properties": {
				"predictions": {
					"type": "array",
					"items": {"type": "number"}
				},
				"confidence": {
					"type": "number"
				}
			}
		},
		"configuration": {
			"batch_size": 32,
			"device": "cpu",
			"num_classes": 10
		},
		"performance_metrics": {
			"accuracy": 0.95,
			"precision": 0.93,
			"recall": 0.94,
			"f1_score": 0.935
		}
	}


@pytest.fixture
def sample_model(sample_model_data):
	"""Create a sample AICRModel instance."""
	return AICRModel(**sample_model_data)


@pytest.fixture
def sample_inference_request():
	"""Sample inference request for testing."""
	return AICRInferenceRequest(
		model_id="test_model_id",
		input_data={
			"features": [1.0, 2.0, 3.0, 4.0, 5.0]
		},
		parameters={
			"temperature": 0.7,
			"top_k": 5
		},
		output_format="json",
		priority="normal",
		timeout_seconds=30
	)


@pytest.fixture
def sample_inference_response(sample_inference_request):
	"""Sample inference response for testing."""
	return AICRInferenceResponse(
		request_id=sample_inference_request.request_id,
		model_id=sample_inference_request.model_id,
		status=InferenceStatus.COMPLETED,
		predictions={
			"class": "cat",
			"confidence": 0.95,
			"probabilities": [0.05, 0.95]
		},
		confidence_scores=[0.05, 0.95],
		processing_time_ms=125.5,
		metadata={
			"model_version": "1.0.0",
			"framework": "pytorch",
			"device": "cpu"
		}
	)


@pytest.fixture
def sample_pipeline_data():
	"""Sample pipeline data for testing."""
	return {
		"name": "test_training_pipeline",
		"description": "A test training pipeline for unit testing",
		"pipeline_type": "training",
		"stages": [
			"data_loading",
			"data_preprocessing",
			"feature_engineering",
			"model_training",
			"model_evaluation",
			"model_deployment"
		],
		"configuration": {
			"epochs": 100,
			"batch_size": 32,
			"learning_rate": 0.001,
			"validation_split": 0.2
		},
		"data_sources": [
			"training_dataset_v1",
			"validation_dataset_v1"
		],
		"schedule": "0 2 * * 0"  # Weekly at 2 AM on Sunday
	}


@pytest.fixture
def sample_pipeline(sample_pipeline_data):
	"""Create a sample AICRPipeline instance."""
	return AICRPipeline(**sample_pipeline_data)


@pytest.fixture
def sample_metrics():
	"""Generate sample metrics for testing."""
	metrics = []
	base_time = datetime.utcnow()

	metric_configs = [
		("cpu_usage", MetricType.GAUGE, "system"),
		("memory_usage", MetricType.GAUGE, "system"),
		("inference_latency", MetricType.HISTOGRAM, "inference_engine"),
		("request_count", MetricType.COUNTER, "api_server"),
		("error_rate", MetricType.GAUGE, "api_server")
	]

	for i in range(100):
		for metric_name, metric_type, component in metric_configs:
			# Generate realistic values with some noise
			if metric_name == "cpu_usage":
				value = 70.0 + np.random.normal(0, 5)
			elif metric_name == "memory_usage":
				value = 60.0 + np.random.normal(0, 8)
			elif metric_name == "inference_latency":
				value = 150.0 + np.random.exponential(20)
			elif metric_name == "request_count":
				value = float(i * 10 + np.random.poisson(5))
			else:  # error_rate
				value = 0.05 + np.random.exponential(0.02)

			metric = AICRMetric(
				metric_name=metric_name,
				metric_type=metric_type,
				value=max(0, value),  # Ensure non-negative
				source_component=component,
				timestamp=base_time - timedelta(minutes=i),
				labels={"environment": "test", "instance": f"test-{i % 3}"}
			)
			metrics.append(metric)

	return metrics


@pytest.fixture
async def mock_ai_service():
	"""Create a mock AI service for testing."""
	service = Mock(spec=AICoreService)
	service.service_id = "mock_service_id"
	service._initialized = True
	service.models = {}
	service.deployment_registry = {}
	service.inference_engines = {}

	# Setup async methods
	service.initialize = AsyncMock()
	service.register_model = AsyncMock()
	service.get_model = AsyncMock()
	service.list_models = AsyncMock()
	service.update_model = AsyncMock()
	service.delete_model = AsyncMock()
	service.deploy_model = AsyncMock()
	service.undeploy_model = AsyncMock()
	service.run_inference = AsyncMock()
	service.run_batch_inference = AsyncMock()
	service.cleanup = AsyncMock()

	return service


@pytest.fixture
async def mock_monitoring_system():
	"""Create a mock monitoring system for testing."""
	system = Mock(spec=AIMonitoringSystem)
	system.system_id = "mock_monitoring_id"
	system._initialized = True
	system.metrics_collector = Mock(spec=MetricsCollector)
	system.alert_manager = Mock(spec=AlertManager)
	system.performance_analyzer = Mock(spec=PerformanceAnalyzer)

	# Setup async methods
	system.initialize = AsyncMock()
	system.register_ai_component = AsyncMock()
	system.get_system_health = AsyncMock()
	system.get_performance_summary = AsyncMock()
	system.create_dashboard = AsyncMock()

	# Setup metrics collector methods
	system.metrics_collector.initialize = AsyncMock()
	system.metrics_collector.register_metric = AsyncMock()
	system.metrics_collector.collect_metric = AsyncMock()
	system.metrics_collector.get_metrics = AsyncMock()

	# Setup alert manager methods
	system.alert_manager.initialize = AsyncMock()
	system.alert_manager.create_alert = AsyncMock()
	system.alert_manager.update_alert = AsyncMock()

	# Setup performance analyzer methods
	system.performance_analyzer.initialize = AsyncMock()
	system.performance_analyzer.create_baseline = AsyncMock()
	system.performance_analyzer.detect_anomalies = AsyncMock()
	system.performance_analyzer.perform_trend_analysis = AsyncMock()

	return system


@pytest.fixture
async def mock_ml_pipeline_framework():
	"""Create a mock ML pipeline framework for testing."""
	framework = Mock(spec=MLPipelineFramework)
	framework.framework_id = "mock_framework_id"
	framework._initialized = True
	framework.orchestrator = Mock()
	framework.pipeline_templates = {}
	framework.execution_history = []

	# Setup async methods
	framework.initialize = AsyncMock()
	framework.create_pipeline_from_template = AsyncMock()
	framework.execute_pipeline = AsyncMock()
	framework.get_execution_status = AsyncMock()
	framework.get_pipeline_metrics = AsyncMock()

	# Setup orchestrator methods
	framework.orchestrator.pipelines = {}
	framework.orchestrator.executions = {}
	framework.orchestrator.register_pipeline = AsyncMock()
	framework.orchestrator.execute_pipeline = AsyncMock()

	return framework


@pytest.fixture
async def mock_security_manager():
	"""Create a mock security manager for testing."""
	manager = Mock(spec=SecurityManager)
	manager.manager_id = "mock_security_id"
	manager._initialized = True

	# Setup async methods
	manager.initialize = AsyncMock()
	manager.generate_jwt_token = AsyncMock()
	manager.validate_jwt_token = AsyncMock()
	manager.encrypt_data = AsyncMock()
	manager.decrypt_data = AsyncMock()
	manager.hash_password = AsyncMock()
	manager.verify_password = AsyncMock()

	return manager


@pytest.fixture
def mock_inference_engine():
	"""Create a mock inference engine for testing."""
	engine = Mock()
	engine.engine_id = "mock_engine_id"
	engine.framework = "pytorch"
	engine._initialized = True

	# Setup async methods
	engine.initialize = AsyncMock()
	engine.deploy_model = AsyncMock()
	engine.undeploy_model = AsyncMock()
	engine.run_inference = AsyncMock()
	engine.run_batch_inference = AsyncMock()
	engine.get_model_info = AsyncMock()
	engine.cleanup = AsyncMock()

	# Setup default return values
	engine.deploy_model.return_value = {"success": True, "endpoint": "test_endpoint"}
	engine.undeploy_model.return_value = {"success": True}
	engine.run_inference.return_value = {
		"predictions": {"class": "test", "confidence": 0.9},
		"processing_time_ms": 100.0
	}
	engine.run_batch_inference.return_value = [
		{"predictions": {"class": "test1"}, "processing_time_ms": 95.0},
		{"predictions": {"class": "test2"}, "processing_time_ms": 105.0}
	]

	return engine


@pytest.fixture
def mock_database_session():
	"""Create a mock database session for testing."""
	session = Mock()
	session.add = Mock()
	session.commit = Mock()
	session.rollback = Mock()
	session.query = Mock()
	session.close = Mock()

	# Setup query builder methods
	query_mock = Mock()
	query_mock.filter = Mock(return_value=query_mock)
	query_mock.order_by = Mock(return_value=query_mock)
	query_mock.limit = Mock(return_value=query_mock)
	query_mock.offset = Mock(return_value=query_mock)
	query_mock.all = Mock(return_value=[])
	query_mock.first = Mock(return_value=None)
	query_mock.count = Mock(return_value=0)
	session.query.return_value = query_mock

	return session


class TestDataGenerator:
	"""Utility class for generating test data."""

	@staticmethod
	def generate_time_series_data(
		metric_name: str,
		hours: int = 24,
		base_value: float = 50.0,
		noise_level: float = 5.0,
		trend: Optional[float] = None,
		seasonal_period: Optional[int] = None
	) -> List[AICRMetric]:
		"""Generate time series metric data for testing.

		Args:
			metric_name: Name of the metric
			hours: Number of hours of data to generate
			base_value: Base value around which to generate data
			noise_level: Standard deviation of noise to add
			trend: Linear trend to add (value per hour)
			seasonal_period: Period for seasonal pattern (in hours)

		Returns:
			List[AICRMetric]: Generated metrics
		"""
		metrics = []
		base_time = datetime.utcnow()

		for i in range(hours * 60):  # Generate minute-by-minute data
			timestamp = base_time - timedelta(minutes=i)

			# Start with base value
			value = base_value

			# Add trend
			if trend:
				value += trend * (i / 60.0)  # Convert minutes to hours

			# Add seasonal pattern
			if seasonal_period:
				seasonal_component = 10 * np.sin(2 * np.pi * i / (seasonal_period * 60))
				value += seasonal_component

			# Add noise
			value += np.random.normal(0, noise_level)

			# Ensure non-negative for certain metrics
			if metric_name in ["cpu_usage", "memory_usage", "disk_usage"]:
				value = max(0, min(100, value))  # Clamp to 0-100%
			else:
				value = max(0, value)  # Ensure non-negative

			metric = AICRMetric(
				metric_name=metric_name,
				metric_type=MetricType.GAUGE,
				value=value,
				source_component="test_generator",
				timestamp=timestamp,
				labels={"generator": "test", "series": "time_series"}
			)
			metrics.append(metric)

		return metrics

	@staticmethod
	def generate_anomalous_data(
		metric_name: str,
		normal_count: int = 100,
		anomaly_count: int = 5,
		normal_mean: float = 50.0,
		normal_std: float = 5.0,
		anomaly_factor: float = 3.0
	) -> List[AICRMetric]:
		"""Generate metric data with anomalies for testing.

		Args:
			metric_name: Name of the metric
			normal_count: Number of normal data points
			anomaly_count: Number of anomalous data points
			normal_mean: Mean of normal distribution
			normal_std: Standard deviation of normal distribution
			anomaly_factor: Factor by which anomalies deviate from normal

		Returns:
			List[AICRMetric]: Generated metrics with anomalies
		"""
		metrics = []
		base_time = datetime.utcnow()

		# Generate normal data
		for i in range(normal_count):
			value = np.random.normal(normal_mean, normal_std)
			timestamp = base_time - timedelta(minutes=i)

			metric = AICRMetric(
				metric_name=metric_name,
				metric_type=MetricType.GAUGE,
				value=max(0, value),
				source_component="test_generator",
				timestamp=timestamp,
				labels={"type": "normal"}
			)
			metrics.append(metric)

		# Generate anomalous data
		anomaly_indices = np.random.choice(normal_count, anomaly_count, replace=False)
		for idx in anomaly_indices:
			# Replace normal value with anomaly
			anomaly_value = normal_mean + anomaly_factor * normal_std * np.random.choice([-1, 1])
			metrics[idx].value = max(0, anomaly_value)
			metrics[idx].labels["type"] = "anomaly"

		return metrics

	@staticmethod
	def generate_model_performance_data(
		num_models: int = 10,
		metrics_per_model: int = 5
	) -> List[Dict[str, Any]]:
		"""Generate model performance data for testing.

		Args:
			num_models: Number of models to generate data for
			metrics_per_model: Number of metric types per model

		Returns:
			List[Dict[str, Any]]: Generated model performance data
		"""
		models_data = []

		metric_types = ["accuracy", "precision", "recall", "f1_score", "latency_ms"]

		for i in range(num_models):
			model_data = {
				"model_id": f"model_{i:03d}",
				"model_name": f"test_model_{i}",
				"model_type": np.random.choice(["classification", "regression", "clustering"]),
				"framework": np.random.choice(["pytorch", "tensorflow", "sklearn"]),
				"performance_metrics": {}
			}

			for j, metric_type in enumerate(metric_types[:metrics_per_model]):
				if metric_type == "latency_ms":
					# Latency in milliseconds (50-500ms)
					value = 50 + np.random.exponential(100)
				else:
					# Performance metrics (0.7-0.99)
					value = 0.7 + 0.29 * np.random.beta(2, 1)

				model_data["performance_metrics"][metric_type] = value

			models_data.append(model_data)

		return models_data


@pytest.fixture
def test_data_generator():
	"""Provide the TestDataGenerator class for test use."""
	return TestDataGenerator


# Custom pytest markers
def pytest_configure(config):
	"""Configure custom pytest markers."""
	config.addinivalue_line(
		"markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
	)
	config.addinivalue_line(
		"markers", "integration: marks tests as integration tests"
	)
	config.addinivalue_line(
		"markers", "unit: marks tests as unit tests"
	)
	config.addinivalue_line(
		"markers", "performance: marks tests as performance tests"
	)
	config.addinivalue_line(
		"markers", "security: marks tests as security tests"
	)


# Test utilities
class AsyncContextManager:
	"""Utility for creating async context managers in tests."""

	def __init__(self, mock_obj):
		self.mock_obj = mock_obj

	async def __aenter__(self):
		return self.mock_obj

	async def __aexit__(self, exc_type, exc_val, exc_tb):
		pass


def create_async_context_manager(mock_obj):
	"""Create an async context manager from a mock object."""
	return AsyncContextManager(mock_obj)


def assert_model_equality(model1: AICRModel, model2: AICRModel, ignore_timestamps: bool = True):
	"""Assert that two models are equal, optionally ignoring timestamps."""
	assert model1.model_id == model2.model_id
	assert model1.name == model2.name
	assert model1.description == model2.description
	assert model1.model_type == model2.model_type
	assert model1.framework == model2.framework
	assert model1.version == model2.version
	assert model1.status == model2.status

	if not ignore_timestamps:
		assert model1.created_at == model2.created_at
		assert model1.updated_at == model2.updated_at


def assert_metric_validity(metric: AICRMetric):
	"""Assert that a metric is valid and well-formed."""
	assert isinstance(metric.metric_id, str)
	assert len(metric.metric_id) > 0
	assert isinstance(metric.metric_name, str)
	assert len(metric.metric_name) > 0
	assert isinstance(metric.metric_type, MetricType)
	assert isinstance(metric.value, (int, float))
	assert isinstance(metric.timestamp, datetime)
	assert isinstance(metric.source_component, str)
	assert len(metric.source_component) > 0


def assert_inference_response_validity(response: AICRInferenceResponse):
	"""Assert that an inference response is valid and well-formed."""
	assert isinstance(response.response_id, str)
	assert len(response.response_id) > 0
	assert isinstance(response.request_id, str)
	assert len(response.request_id) > 0
	assert isinstance(response.model_id, str)
	assert len(response.model_id) > 0
	assert isinstance(response.status, InferenceStatus)
	assert isinstance(response.timestamp, datetime)

	if response.status == InferenceStatus.COMPLETED:
		assert response.predictions is not None
		assert response.processing_time_ms is not None
		assert response.processing_time_ms > 0
	elif response.status == InferenceStatus.FAILED:
		assert response.error_message is not None
		assert len(response.error_message) > 0