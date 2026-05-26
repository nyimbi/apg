"""
Integration Tests for AICR Capability
======================================

© 2025 Datacraft - www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>

Comprehensive integration tests for the AI Core Framework capability
covering end-to-end workflows, component interactions, system integration,
and real-world usage scenarios with full operational validation.
"""

import pytest
import asyncio
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, AsyncMock, patch
import numpy as np

from ..service import AICoreService
from ..monitoring import ai_monitoring_system
from ..ml_pipeline import ml_pipeline_framework
from ..model_marketplace import model_marketplace
from ..websocket import websocket_server
from ..security import SecurityManager
from ..models import (
	AICRModel,
	AICRInferenceRequest,
	AICRInferenceResponse,
	AICRPipeline,
	ModelType,
	InferenceStatus,
	PipelineStatus
)


@pytest.mark.integration
class TestAICRServiceIntegration:
	"""Integration tests for the complete AICR service stack."""

	@pytest.fixture
	async def integrated_service(self):
		"""Create a fully integrated AI service with all components."""
		service = AICoreService()

		# Initialize with real components (mocked where necessary)
		with patch.object(service.security_manager, 'initialize', new_callable=AsyncMock), \
			 patch.object(service.monitoring, 'initialize', new_callable=AsyncMock), \
			 patch.object(service, '_initialize_inference_engines', new_callable=AsyncMock), \
			 patch.object(service, '_start_background_tasks', new_callable=AsyncMock):

			await service.initialize()

		# Setup mock inference engines
		mock_pytorch_engine = Mock()
		mock_pytorch_engine.deploy_model = AsyncMock(return_value={"success": True, "endpoint": "pytorch_endpoint"})
		mock_pytorch_engine.undeploy_model = AsyncMock(return_value={"success": True})
		mock_pytorch_engine.run_inference = AsyncMock()
		mock_pytorch_engine.run_batch_inference = AsyncMock()

		mock_tensorflow_engine = Mock()
		mock_tensorflow_engine.deploy_model = AsyncMock(return_value={"success": True, "endpoint": "tf_endpoint"})
		mock_tensorflow_engine.undeploy_model = AsyncMock(return_value={"success": True})
		mock_tensorflow_engine.run_inference = AsyncMock()
		mock_tensorflow_engine.run_batch_inference = AsyncMock()

		service.inference_engines["pytorch"] = mock_pytorch_engine
		service.inference_engines["tensorflow"] = mock_tensorflow_engine

		return service

	@pytest.mark.asyncio
	async def test_complete_model_lifecycle(self, integrated_service):
		"""Test complete model lifecycle from registration to deletion."""
		service = integrated_service

		# Step 1: Register model
		model_data = {
			"name": "integration_test_model",
			"description": "Model for integration testing",
			"model_type": "classification",
			"framework": "pytorch",
			"version": "1.0.0",
			"file_path": "/models/integration_test.pth"
		}

		model = await service.register_model(model_data)
		assert model.name == "integration_test_model"
		assert model.model_id in service.models

		# Step 2: Deploy model
		deployment_result = await service.deploy_model(model.model_id)
		assert deployment_result["success"] == True
		assert model.model_id in service.deployment_registry

		# Verify model status updated
		deployed_model = service.models[model.model_id]
		assert deployed_model.status == "deployed"
		assert deployed_model.deployment_count == 1

		# Step 3: Run inference
		mock_inference_result = {
			"predictions": {"class": "cat", "confidence": 0.95},
			"processing_time_ms": 150.0
		}
		service.inference_engines["pytorch"].run_inference.return_value = mock_inference_result

		inference_request = AICRInferenceRequest(
			model_id=model.model_id,
			input_data={"image": "base64_encoded_image"}
		)

		inference_response = await service.run_inference(inference_request)
		assert inference_response.status == InferenceStatus.COMPLETED
		assert inference_response.predictions is not None

		# Verify last_inference updated
		updated_model = service.models[model.model_id]
		assert updated_model.last_inference is not None

		# Step 4: Update model
		update_data = {
			"description": "Updated integration test model",
			"version": "1.1.0"
		}

		updated_model = await service.update_model(model.model_id, update_data)
		assert updated_model.description == "Updated integration test model"
		assert updated_model.version == "1.1.0"

		# Step 5: Undeploy model
		undeploy_result = await service.undeploy_model(model.model_id)
		assert undeploy_result["success"] == True
		assert model.model_id not in service.deployment_registry

		# Step 6: Delete model
		delete_success = await service.delete_model(model.model_id)
		assert delete_success == True
		assert model.model_id not in service.models

	@pytest.mark.asyncio
	async def test_multi_framework_deployment(self, integrated_service):
		"""Test deploying models across different frameworks."""
		service = integrated_service

		# Register models for different frameworks
		pytorch_model_data = {
			"name": "pytorch_model",
			"description": "PyTorch model",
			"model_type": "classification",
			"framework": "pytorch"
		}

		tensorflow_model_data = {
			"name": "tensorflow_model",
			"description": "TensorFlow model",
			"model_type": "regression",
			"framework": "tensorflow"
		}

		pytorch_model = await service.register_model(pytorch_model_data)
		tensorflow_model = await service.register_model(tensorflow_model_data)

		# Deploy both models
		pytorch_deployment = await service.deploy_model(pytorch_model.model_id)
		tensorflow_deployment = await service.deploy_model(tensorflow_model.model_id)

		assert pytorch_deployment["success"] == True
		assert tensorflow_deployment["success"] == True

		# Verify both models are deployed
		assert len(service.deployment_registry) == 2
		assert pytorch_model.model_id in service.deployment_registry
		assert tensorflow_model.model_id in service.deployment_registry

		# Test inference on both models
		service.inference_engines["pytorch"].run_inference.return_value = {
			"predictions": {"class": "dog"}, "processing_time_ms": 120.0
		}
		service.inference_engines["tensorflow"].run_inference.return_value = {
			"predictions": {"value": 42.5}, "processing_time_ms": 180.0
		}

		pytorch_request = AICRInferenceRequest(
			model_id=pytorch_model.model_id,
			input_data={"features": [1, 2, 3]}
		)

		tensorflow_request = AICRInferenceRequest(
			model_id=tensorflow_model.model_id,
			input_data={"features": [4, 5, 6]}
		)

		pytorch_response = await service.run_inference(pytorch_request)
		tensorflow_response = await service.run_inference(tensorflow_request)

		assert pytorch_response.status == InferenceStatus.COMPLETED
		assert tensorflow_response.status == InferenceStatus.COMPLETED
		assert pytorch_response.predictions["class"] == "dog"
		assert tensorflow_response.predictions["value"] == 42.5

	@pytest.mark.asyncio
	async def test_batch_inference_integration(self, integrated_service):
		"""Test batch inference with multiple inputs."""
		service = integrated_service

		# Register and deploy model
		model_data = {
			"name": "batch_test_model",
			"description": "Model for batch testing",
			"model_type": "classification",
			"framework": "pytorch"
		}

		model = await service.register_model(model_data)
		await service.deploy_model(model.model_id)

		# Setup batch inference mock
		batch_results = [
			{"predictions": {"class": "cat"}, "processing_time_ms": 100},
			{"predictions": {"class": "dog"}, "processing_time_ms": 110},
			{"predictions": {"class": "bird"}, "processing_time_ms": 95}
		]
		service.inference_engines["pytorch"].run_batch_inference.return_value = batch_results

		# Prepare batch data
		batch_data = [
			{"image": "cat_image"},
			{"image": "dog_image"},
			{"image": "bird_image"}
		]

		# Run batch inference
		batch_responses = await service.run_batch_inference(model.model_id, batch_data)

		assert len(batch_responses) == 3
		for i, response in enumerate(batch_responses):
			assert response.status == InferenceStatus.COMPLETED
			assert response.model_id == model.model_id
			assert response.predictions is not None

	@pytest.mark.asyncio
	async def test_concurrent_operations(self, integrated_service):
		"""Test concurrent model operations."""
		service = integrated_service

		# Register multiple models concurrently
		model_tasks = []
		for i in range(5):
			model_data = {
				"name": f"concurrent_model_{i}",
				"description": f"Concurrent model {i}",
				"model_type": "classification",
				"framework": "pytorch"
			}
			task = service.register_model(model_data)
			model_tasks.append(task)

		models = await asyncio.gather(*model_tasks)
		assert len(models) == 5
		assert len(service.models) == 5

		# Deploy all models concurrently
		deployment_tasks = []
		for model in models:
			task = service.deploy_model(model.model_id)
			deployment_tasks.append(task)

		deployment_results = await asyncio.gather(*deployment_tasks)
		assert all(result["success"] for result in deployment_results)
		assert len(service.deployment_registry) == 5

		# Run concurrent inference
		service.inference_engines["pytorch"].run_inference.return_value = {
			"predictions": {"class": "test"}, "processing_time_ms": 100.0
		}

		inference_tasks = []
		for model in models:
			request = AICRInferenceRequest(
				model_id=model.model_id,
				input_data={"data": [1, 2, 3]}
			)
			task = service.run_inference(request)
			inference_tasks.append(task)

		inference_responses = await asyncio.gather(*inference_tasks)
		assert len(inference_responses) == 5
		assert all(resp.status == InferenceStatus.COMPLETED for resp in inference_responses)


@pytest.mark.integration
class TestMonitoringIntegration:
	"""Integration tests for monitoring system with other components."""

	@pytest.fixture
	async def monitored_system(self):
		"""Create integrated system with monitoring enabled."""
		# Initialize AI service with monitoring
		ai_service = AICoreService()

		with patch.object(ai_service.security_manager, 'initialize', new_callable=AsyncMock), \
			 patch.object(ai_service, '_initialize_inference_engines', new_callable=AsyncMock), \
			 patch.object(ai_service, '_start_background_tasks', new_callable=AsyncMock):

			# Use real monitoring system
			await ai_service.initialize()

		# Initialize monitoring system
		with patch.object(ai_monitoring_system.security_manager, 'initialize', new_callable=AsyncMock), \
			 patch.object(ai_monitoring_system, '_create_default_dashboards', new_callable=AsyncMock), \
			 patch.object(ai_monitoring_system, '_start_monitoring_tasks', new_callable=AsyncMock):

			await ai_monitoring_system.initialize()

		return ai_service, ai_monitoring_system

	@pytest.mark.asyncio
	async def test_model_lifecycle_monitoring(self, monitored_system):
		"""Test monitoring of complete model lifecycle."""
		ai_service, monitoring = monitored_system

		# Register component for monitoring
		component_config = {
			"metrics": [
				{
					"name": "model_registrations",
					"type": "counter",
					"collector": lambda: len(ai_service.models),
					"interval": 5
				}
			],
			"alerts": [
				{
					"alert_name": "High Model Count",
					"description": "Too many models registered",
					"severity": "medium",
					"condition": "model_count > 10",
					"threshold": 10.0,
					"metric_name": "model_registrations"
				}
			]
		}

		component_id = await monitoring.register_ai_component("ai_service", component_config)
		assert isinstance(component_id, str)

		# Register models and verify metrics collection
		for i in range(3):
			model_data = {
				"name": f"monitored_model_{i}",
				"description": f"Model {i} for monitoring",
				"model_type": "classification",
				"framework": "pytorch"
			}
			await ai_service.register_model(model_data)

		# Collect metrics manually
		await monitoring.metrics_collector.collect_metric(
			metric_name="model_registrations",
			value=len(ai_service.models),
			source_component="ai_service"
		)

		# Get system health
		health_data = await monitoring.get_system_health()
		assert "system_id" in health_data
		assert "overall_health_score" in health_data

	@pytest.mark.asyncio
	async def test_inference_performance_monitoring(self, monitored_system):
		"""Test monitoring of inference performance."""
		ai_service, monitoring = monitored_system

		# Setup mock inference engine
		mock_engine = Mock()
		mock_engine.deploy_model = AsyncMock(return_value={"success": True})
		mock_engine.run_inference = AsyncMock()
		ai_service.inference_engines["pytorch"] = mock_engine

		# Register and deploy model
		model_data = {
			"name": "perf_test_model",
			"description": "Model for performance testing",
			"model_type": "classification",
			"framework": "pytorch"
		}

		model = await ai_service.register_model(model_data)
		await ai_service.deploy_model(model.model_id)

		# Run multiple inference requests and collect metrics
		latencies = []
		for i in range(10):
			latency = 100 + np.random.normal(0, 20)  # Simulate varying latency
			latencies.append(latency)

			mock_engine.run_inference.return_value = {
				"predictions": {"class": f"class_{i}"},
				"processing_time_ms": latency
			}

			request = AICRInferenceRequest(
				model_id=model.model_id,
				input_data={"data": [i, i+1, i+2]}
			)

			response = await ai_service.run_inference(request)

			# Collect inference latency metric
			await monitoring.metrics_collector.collect_metric(
				metric_name="inference_latency",
				value=latency,
				source_component="ai_service",
				labels={"model_id": model.model_id}
			)

		# Analyze performance metrics
		metrics = await monitoring.metrics_collector.get_metrics(
			metric_names=["inference_latency"]
		)

		assert len(metrics) == 10
		avg_latency = np.mean([m.value for m in metrics])
		assert 80 < avg_latency < 120  # Should be around 100ms

	@pytest.mark.asyncio
	async def test_alert_integration(self, monitored_system):
		"""Test alert integration with system events."""
		ai_service, monitoring = monitored_system

		# Create alert for high error rate
		alert_config = {
			"alert_name": "High Error Rate",
			"description": "Inference error rate is too high",
			"severity": "high",
			"condition": "error_rate > 0.1",
			"threshold": 0.1,
			"metric_name": "inference_error_rate",
			"notification_channels": ["log"]
		}

		alert_id = await monitoring.alert_manager.create_alert(alert_config)

		# Simulate high error rate
		total_requests = 20
		failed_requests = 5
		error_rate = failed_requests / total_requests

		await monitoring.metrics_collector.collect_metric(
			metric_name="inference_error_rate",
			value=error_rate,
			source_component="ai_service"
		)

		# Mock alert evaluation
		with patch.object(monitoring.alert_manager, '_get_current_metric_value', new_callable=AsyncMock) as mock_metric:
			mock_metric.return_value = error_rate

			with patch.object(monitoring.alert_manager, '_evaluate_condition_expression', new_callable=AsyncMock) as mock_condition:
				mock_condition.return_value = True  # Error rate exceeds threshold

				with patch.object(monitoring.alert_manager, '_trigger_alert', new_callable=AsyncMock) as mock_trigger:
					alert = monitoring.alert_manager.alerts[alert_id]
					await monitoring.alert_manager._evaluate_alert_condition(alert)

					# Verify alert was triggered
					mock_trigger.assert_called_once()


@pytest.mark.integration
class TestMLPipelineIntegration:
	"""Integration tests for ML pipeline framework with AI service."""

	@pytest.fixture
	async def pipeline_system(self):
		"""Create integrated pipeline system."""
		# Initialize ML pipeline framework
		with patch.object(ml_pipeline_framework.security_manager, 'initialize', new_callable=AsyncMock), \
			 patch.object(ml_pipeline_framework.metrics_collector, 'initialize', new_callable=AsyncMock), \
			 patch.object(ml_pipeline_framework.optimizer, 'initialize', new_callable=AsyncMock), \
			 patch.object(ml_pipeline_framework.automl_engine, 'initialize', new_callable=AsyncMock), \
			 patch.object(ml_pipeline_framework, '_load_default_templates', new_callable=AsyncMock):

			await ml_pipeline_framework.initialize()

		return ml_pipeline_framework

	@pytest.mark.asyncio
	async def test_pipeline_creation_and_execution(self, pipeline_system):
		"""Test creating and executing ML pipelines."""
		framework = pipeline_system

		# Create pipeline from template
		pipeline_config = {
			"name": "integration_test_pipeline",
			"description": "Pipeline for integration testing",
			"training_config": {
				"model_type": "classification",
				"algorithm": "random_forest",
				"metrics": ["accuracy", "f1_score"]
			},
			"data_sources": ["test_dataset"]
		}

		# Mock template
		from ..ml_pipeline import MLPipeline, PipelineStageConfig, ModelTrainingConfig, AutoMLConfiguration
		from ..ml_pipeline import PipelineStage

		mock_template = MLPipeline(
			pipeline_name="test_template",
			description="Test template",
			stages=[
				PipelineStageConfig(
					stage_name=PipelineStage.DATA_INGESTION,
					stage_order=1
				),
				PipelineStageConfig(
					stage_name=PipelineStage.MODEL_TRAINING,
					stage_order=2
				)
			],
			training_config=ModelTrainingConfig(
				model_type="classification",
				algorithm="auto",
				metrics=["accuracy"]
			),
			automl_config=AutoMLConfiguration()
		)

		framework.pipeline_templates["classification"] = mock_template

		pipeline = await framework.create_pipeline_from_template("classification", pipeline_config)
		assert pipeline.pipeline_name == "integration_test_pipeline"
		assert pipeline.pipeline_id in framework.orchestrator.pipelines

		# Mock pipeline execution
		with patch.object(framework.orchestrator, '_execute_pipeline_async', new_callable=AsyncMock):
			execution_id = await framework.execute_pipeline(pipeline.pipeline_id)
			assert isinstance(execution_id, str)

	@pytest.mark.asyncio
	async def test_automl_integration(self, pipeline_system):
		"""Test AutoML integration with pipelines."""
		framework = pipeline_system

		# Mock AutoML engine optimization
		mock_automl_result = {
			"feature_engineering": {
				"transformations_applied": ["polynomial", "interaction"],
				"feature_count_improvement": 20
			},
			"model_selection": {
				"best_algorithm": "gradient_boosting",
				"algorithms_evaluated": 5,
				"best_score": 0.95
			}
		}

		framework.automl_engine.optimize_pipeline = AsyncMock(return_value=mock_automl_result)

		# Create pipeline with AutoML enabled
		from ..ml_pipeline import MLPipeline, PipelineStageConfig, ModelTrainingConfig, AutoMLConfiguration
		from ..ml_pipeline import PipelineStage

		pipeline = MLPipeline(
			pipeline_name="automl_test_pipeline",
			description="Pipeline with AutoML",
			stages=[
				PipelineStageConfig(
					stage_name=PipelineStage.DATA_INGESTION,
					stage_order=1
				)
			],
			training_config=ModelTrainingConfig(
				model_type="classification",
				algorithm="auto",
				metrics=["accuracy"]
			),
			automl_config=AutoMLConfiguration(
				auto_feature_engineering=True,
				auto_model_selection=True,
				time_budget_minutes=30
			)
		)

		pipeline_id = await framework.orchestrator.register_pipeline(pipeline)

		# Mock execution with AutoML
		from ..ml_pipeline import PipelineExecution
		mock_execution = PipelineExecution(
			pipeline_id=pipeline_id,
			execution_number=1
		)

		automl_result = await framework.automl_engine.optimize_pipeline(
			pipeline, mock_execution, {"test_data": "mock_data"}
		)

		assert "feature_engineering" in automl_result
		assert "model_selection" in automl_result
		assert automl_result["model_selection"]["best_score"] == 0.95


@pytest.mark.integration
class TestEndToEndWorkflows:
	"""End-to-end integration tests for complete workflows."""

	@pytest.fixture
	async def full_stack(self):
		"""Create complete AICR stack for testing."""
		# Initialize AI service
		ai_service = AICoreService()
		with patch.object(ai_service.security_manager, 'initialize', new_callable=AsyncMock), \
			 patch.object(ai_service, '_initialize_inference_engines', new_callable=AsyncMock), \
			 patch.object(ai_service, '_start_background_tasks', new_callable=AsyncMock):
			await ai_service.initialize()

		# Initialize monitoring
		with patch.object(ai_monitoring_system.security_manager, 'initialize', new_callable=AsyncMock), \
			 patch.object(ai_monitoring_system, '_create_default_dashboards', new_callable=AsyncMock), \
			 patch.object(ai_monitoring_system, '_start_monitoring_tasks', new_callable=AsyncMock):
			await ai_monitoring_system.initialize()

		# Initialize ML pipeline framework
		with patch.object(ml_pipeline_framework.security_manager, 'initialize', new_callable=AsyncMock), \
			 patch.object(ml_pipeline_framework.metrics_collector, 'initialize', new_callable=AsyncMock), \
			 patch.object(ml_pipeline_framework.optimizer, 'initialize', new_callable=AsyncMock), \
			 patch.object(ml_pipeline_framework.automl_engine, 'initialize', new_callable=AsyncMock), \
			 patch.object(ml_pipeline_framework, '_load_default_templates', new_callable=AsyncMock):
			await ml_pipeline_framework.initialize()

		# Setup mock inference engine
		mock_engine = Mock()
		mock_engine.deploy_model = AsyncMock(return_value={"success": True})
		mock_engine.undeploy_model = AsyncMock(return_value={"success": True})
		mock_engine.run_inference = AsyncMock()
		ai_service.inference_engines["pytorch"] = mock_engine

		return {
			"ai_service": ai_service,
			"monitoring": ai_monitoring_system,
			"ml_pipeline": ml_pipeline_framework,
			"mock_engine": mock_engine
		}

	@pytest.mark.asyncio
	async def test_ml_model_development_workflow(self, full_stack):
		"""Test complete ML model development workflow."""
		ai_service = full_stack["ai_service"]
		monitoring = full_stack["monitoring"]
		ml_pipeline = full_stack["ml_pipeline"]
		mock_engine = full_stack["mock_engine"]

		# Step 1: Create and execute training pipeline
		from ..ml_pipeline import MLPipeline, PipelineStageConfig, ModelTrainingConfig
		from ..ml_pipeline import PipelineStage

		training_pipeline = MLPipeline(
			pipeline_name="dev_workflow_pipeline",
			description="Development workflow pipeline",
			stages=[
				PipelineStageConfig(
					stage_name=PipelineStage.DATA_INGESTION,
					stage_order=1
				),
				PipelineStageConfig(
					stage_name=PipelineStage.MODEL_TRAINING,
					stage_order=2
				)
			],
			training_config=ModelTrainingConfig(
				model_type="classification",
				algorithm="random_forest",
				metrics=["accuracy", "f1_score"]
			)
		)

		pipeline_id = await ml_pipeline.orchestrator.register_pipeline(training_pipeline)

		# Mock successful pipeline execution
		from ..ml_pipeline import PipelineExecution
		mock_execution = PipelineExecution(
			pipeline_id=pipeline_id,
			execution_number=1,
			status="completed"
		)
		ml_pipeline.orchestrator.executions[f"exec_{pipeline_id}"] = mock_execution

		# Step 2: Register trained model
		model_data = {
			"name": "workflow_trained_model",
			"description": "Model from development workflow",
			"model_type": "classification",
			"framework": "pytorch",
			"performance_metrics": {
				"accuracy": 0.95,
				"f1_score": 0.93
			}
		}

		model = await ai_service.register_model(model_data)

		# Step 3: Deploy model for inference
		await ai_service.deploy_model(model.model_id)

		# Step 4: Run test inference
		mock_engine.run_inference.return_value = {
			"predictions": {"class": "positive", "confidence": 0.89},
			"processing_time_ms": 125.0
		}

		test_request = AICRInferenceRequest(
			model_id=model.model_id,
			input_data={"features": [0.5, 0.3, 0.8, 0.2]}
		)

		test_response = await ai_service.run_inference(test_request)
		assert test_response.status == InferenceStatus.COMPLETED

		# Step 5: Monitor performance
		await monitoring.metrics_collector.collect_metric(
			metric_name="model_accuracy",
			value=0.95,
			source_component="workflow_test",
			labels={"model_id": model.model_id}
		)

		await monitoring.metrics_collector.collect_metric(
			metric_name="inference_latency",
			value=125.0,
			source_component="workflow_test",
			labels={"model_id": model.model_id}
		)

		# Step 6: Get performance summary
		performance_summary = await monitoring.get_performance_summary()
		assert "performance_statistics" in performance_summary

		# Verify end-to-end workflow
		assert len(ai_service.models) == 1
		assert len(ai_service.deployment_registry) == 1
		assert model.model_id in ai_service.deployment_registry

	@pytest.mark.asyncio
	async def test_production_deployment_workflow(self, full_stack):
		"""Test production deployment workflow with monitoring."""
		ai_service = full_stack["ai_service"]
		monitoring = full_stack["monitoring"]
		mock_engine = full_stack["mock_engine"]

		# Step 1: Register production model
		production_model_data = {
			"name": "production_model_v1",
			"description": "Production-ready model",
			"model_type": "classification",
			"framework": "pytorch",
			"version": "1.0.0",
			"performance_metrics": {
				"accuracy": 0.97,
				"precision": 0.96,
				"recall": 0.95
			}
		}

		model = await ai_service.register_model(production_model_data)

		# Step 2: Setup production monitoring
		component_config = {
			"metrics": [
				{
					"name": "production_inference_count",
					"type": "counter",
					"collector": lambda: 0,  # Would be real counter in production
					"interval": 10
				}
			],
			"alerts": [
				{
					"alert_name": "Production Model Down",
					"description": "Production model is not responding",
					"severity": "critical",
					"condition": "response_time > 1000",
					"threshold": 1000.0,
					"metric_name": "inference_latency"
				}
			]
		}

		await monitoring.register_ai_component("production_service", component_config)

		# Step 3: Deploy to production
		production_deployment = await ai_service.deploy_model(
			model.model_id,
			{"environment": "production", "replicas": 3}
		)
		assert production_deployment["success"] == True

		# Step 4: Simulate production traffic
		inference_count = 0
		total_latency = 0

		for i in range(10):
			latency = 150 + np.random.normal(0, 30)  # Realistic latency variation
			total_latency += latency
			inference_count += 1

			mock_engine.run_inference.return_value = {
				"predictions": {"class": f"class_{i % 3}", "confidence": 0.9 + np.random.normal(0, 0.05)},
				"processing_time_ms": latency
			}

			request = AICRInferenceRequest(
				model_id=model.model_id,
				input_data={"data": np.random.rand(10).tolist()}
			)

			response = await ai_service.run_inference(request)
			assert response.status == InferenceStatus.COMPLETED

			# Collect production metrics
			await monitoring.metrics_collector.collect_metric(
				metric_name="production_inference_count",
				value=inference_count,
				source_component="production_service"
			)

			await monitoring.metrics_collector.collect_metric(
				metric_name="inference_latency",
				value=latency,
				source_component="production_service",
				labels={"model_id": model.model_id}
			)

		# Step 5: Verify production metrics
		metrics = await monitoring.metrics_collector.get_metrics(
			metric_names=["inference_latency", "production_inference_count"]
		)

		latency_metrics = [m for m in metrics if m.metric_name == "inference_latency"]
		count_metrics = [m for m in metrics if m.metric_name == "production_inference_count"]

		assert len(latency_metrics) == 10
		assert len(count_metrics) == 10

		avg_latency = np.mean([m.value for m in latency_metrics])
		assert 100 < avg_latency < 200  # Should be reasonable

		# Step 6: Get system health
		health_data = await monitoring.get_system_health()
		assert health_data["monitoring_status"] == "active"

	@pytest.mark.asyncio
	async def test_model_ab_testing_workflow(self, full_stack):
		"""Test A/B testing workflow with multiple model versions."""
		ai_service = full_stack["ai_service"]
		monitoring = full_stack["monitoring"]
		mock_engine = full_stack["mock_engine"]

		# Step 1: Register two model versions
		model_a_data = {
			"name": "ab_test_model_a",
			"description": "Model A for A/B testing",
			"model_type": "classification",
			"framework": "pytorch",
			"version": "1.0.0"
		}

		model_b_data = {
			"name": "ab_test_model_b",
			"description": "Model B for A/B testing",
			"model_type": "classification",
			"framework": "pytorch",
			"version": "2.0.0"
		}

		model_a = await ai_service.register_model(model_a_data)
		model_b = await ai_service.register_model(model_b_data)

		# Step 2: Deploy both models
		await ai_service.deploy_model(model_a.model_id)
		await ai_service.deploy_model(model_b.model_id)

		# Step 3: Simulate A/B test traffic
		model_a_results = []
		model_b_results = []

		for i in range(20):
			# Alternate between models
			if i % 2 == 0:
				# Model A - slightly lower performance
				confidence = 0.85 + np.random.normal(0, 0.05)
				latency = 150 + np.random.normal(0, 20)
				model_id = model_a.model_id
				model_a_results.append(confidence)
			else:
				# Model B - slightly better performance
				confidence = 0.90 + np.random.normal(0, 0.05)
				latency = 140 + np.random.normal(0, 15)
				model_id = model_b.model_id
				model_b_results.append(confidence)

			mock_engine.run_inference.return_value = {
				"predictions": {"class": "test", "confidence": confidence},
				"processing_time_ms": latency
			}

			request = AICRInferenceRequest(
				model_id=model_id,
				input_data={"data": [i, i+1, i+2]}
			)

			response = await ai_service.run_inference(request)

			# Collect A/B test metrics
			await monitoring.metrics_collector.collect_metric(
				metric_name="model_confidence",
				value=confidence,
				source_component="ab_test",
				labels={"model_id": model_id, "version": "A" if i % 2 == 0 else "B"}
			)

			await monitoring.metrics_collector.collect_metric(
				metric_name="model_latency",
				value=latency,
				source_component="ab_test",
				labels={"model_id": model_id, "version": "A" if i % 2 == 0 else "B"}
			)

		# Step 4: Analyze A/B test results
		metrics = await monitoring.metrics_collector.get_metrics(
			metric_names=["model_confidence", "model_latency"]
		)

		model_a_confidence = [
			m.value for m in metrics
			if m.metric_name == "model_confidence" and m.labels.get("version") == "A"
		]
		model_b_confidence = [
			m.value for m in metrics
			if m.metric_name == "model_confidence" and m.labels.get("version") == "B"
		]

		avg_confidence_a = np.mean(model_a_confidence)
		avg_confidence_b = np.mean(model_b_confidence)

		# Model B should perform better
		assert avg_confidence_b > avg_confidence_a
		assert len(model_a_confidence) == 10
		assert len(model_b_confidence) == 10


@pytest.mark.integration
class TestSystemResilience:
	"""Integration tests for system resilience and failure handling."""

	@pytest.mark.asyncio
	async def test_partial_system_failure_resilience(self):
		"""Test system resilience to partial component failures."""
		# Initialize AI service
		ai_service = AICoreService()

		# Simulate monitoring initialization failure
		with patch.object(ai_service.security_manager, 'initialize', new_callable=AsyncMock), \
			 patch.object(ai_service.monitoring, 'initialize', side_effect=Exception("Monitoring failed")), \
			 patch.object(ai_service, '_initialize_inference_engines', new_callable=AsyncMock), \
			 patch.object(ai_service, '_start_background_tasks', new_callable=AsyncMock):

			# Service should handle monitoring failure gracefully
			with pytest.raises(Exception):
				await ai_service.initialize()

		# Test with monitoring bypassed
		with patch.object(ai_service.security_manager, 'initialize', new_callable=AsyncMock), \
			 patch.object(ai_service.monitoring, 'initialize', new_callable=AsyncMock), \
			 patch.object(ai_service, '_initialize_inference_engines', new_callable=AsyncMock), \
			 patch.object(ai_service, '_start_background_tasks', new_callable=AsyncMock):

			await ai_service.initialize()

		# Test model operations still work
		model_data = {
			"name": "resilience_test_model",
			"description": "Model for resilience testing",
			"model_type": "classification",
			"framework": "pytorch"
		}

		model = await ai_service.register_model(model_data)
		assert model.name == "resilience_test_model"

	@pytest.mark.asyncio
	async def test_inference_engine_failure_handling(self):
		"""Test handling of inference engine failures."""
		ai_service = AICoreService()

		with patch.object(ai_service.security_manager, 'initialize', new_callable=AsyncMock), \
			 patch.object(ai_service.monitoring, 'initialize', new_callable=AsyncMock), \
			 patch.object(ai_service, '_initialize_inference_engines', new_callable=AsyncMock), \
			 patch.object(ai_service, '_start_background_tasks', new_callable=AsyncMock):

			await ai_service.initialize()

		# Register model
		model_data = {
			"name": "failure_test_model",
			"description": "Model for failure testing",
			"model_type": "classification",
			"framework": "pytorch"
		}

		model = await ai_service.register_model(model_data)

		# Setup failing inference engine
		mock_engine = Mock()
		mock_engine.deploy_model = AsyncMock(side_effect=Exception("Engine deployment failed"))
		ai_service.inference_engines["pytorch"] = mock_engine

		# Deployment should fail gracefully
		with pytest.raises(Exception):
			await ai_service.deploy_model(model.model_id)

		# Model should remain in system but not deployed
		assert model.model_id in ai_service.models
		assert model.model_id not in ai_service.deployment_registry

	@pytest.mark.asyncio
	async def test_concurrent_failure_recovery(self):
		"""Test recovery from concurrent operation failures."""
		ai_service = AICoreService()

		with patch.object(ai_service.security_manager, 'initialize', new_callable=AsyncMock), \
			 patch.object(ai_service.monitoring, 'initialize', new_callable=AsyncMock), \
			 patch.object(ai_service, '_initialize_inference_engines', new_callable=AsyncMock), \
			 patch.object(ai_service, '_start_background_tasks', new_callable=AsyncMock):

			await ai_service.initialize()

		# Setup mock engine that fails randomly
		call_count = 0

		async def failing_deployment(*args, **kwargs):
			nonlocal call_count
			call_count += 1
			if call_count % 3 == 0:  # Fail every third call
				raise Exception(f"Random failure {call_count}")
			return {"success": True}

		mock_engine = Mock()
		mock_engine.deploy_model = failing_deployment
		ai_service.inference_engines["pytorch"] = mock_engine

		# Register multiple models
		models = []
		for i in range(5):
			model_data = {
				"name": f"concurrent_failure_model_{i}",
				"description": f"Model {i} for concurrent failure testing",
				"model_type": "classification",
				"framework": "pytorch"
			}
			model = await ai_service.register_model(model_data)
			models.append(model)

		# Attempt concurrent deployments
		deployment_tasks = []
		for model in models:
			task = ai_service.deploy_model(model.model_id)
			deployment_tasks.append(task)

		# Some should succeed, some should fail
		results = await asyncio.gather(*deployment_tasks, return_exceptions=True)

		successes = [r for r in results if not isinstance(r, Exception)]
		failures = [r for r in results if isinstance(r, Exception)]

		# Should have both successes and failures
		assert len(successes) > 0
		assert len(failures) > 0

		# System should remain stable
		assert len(ai_service.models) == 5


if __name__ == "__main__":
	pytest.main([__file__, "-v"])