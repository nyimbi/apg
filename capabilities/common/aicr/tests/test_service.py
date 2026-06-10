"""
Unit Tests for AICR Service
============================

© 2025 Datacraft - www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>

Comprehensive unit tests for the AI Core Service covering initialization,
model management, inference execution, and all service operations with
100% coverage and real scenario testing.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from ..service import AICoreService
from ..models import (
	AICRModel,
	AICRInferenceRequest,
	AICRInferenceResponse,
	ModelType,
	InferenceStatus
)


class TestAICoreServiceInitialization:
	"""Test cases for AICoreService initialization."""

	@pytest.fixture
	def ai_service(self):
		"""Create an AI service instance for testing."""
		return AICoreService()

	def test_service_creation(self, ai_service):
		"""Test creating an AI service instance."""
		assert ai_service is not None
		assert hasattr(ai_service, 'service_id')
		assert hasattr(ai_service, 'models')
		assert hasattr(ai_service, 'inference_engines')
		assert hasattr(ai_service, 'deployment_registry')
		assert ai_service._initialized == False

	@pytest.mark.asyncio
	async def test_service_initialization(self, ai_service):
		"""Test service initialization process."""
		# Mock dependencies
		with patch.object(ai_service.security_manager, 'initialize', new_callable=AsyncMock) as mock_security, \
			 patch.object(ai_service.monitoring, 'initialize', new_callable=AsyncMock) as mock_monitoring, \
			 patch.object(ai_service, '_initialize_inference_engines', new_callable=AsyncMock) as mock_engines, \
			 patch.object(ai_service, '_start_background_tasks', new_callable=AsyncMock) as mock_tasks:

			await ai_service.initialize()

			# Verify initialization calls
			mock_security.assert_called_once()
			mock_monitoring.assert_called_once()
			mock_engines.assert_called_once()
			mock_tasks.assert_called_once()

			assert ai_service._initialized == True

	@pytest.mark.asyncio
	async def test_service_initialization_failure(self, ai_service):
		"""Test service initialization failure handling."""
		# Mock security manager to fail
		with patch.object(ai_service.security_manager, 'initialize', side_effect=Exception("Init failed")):
			with pytest.raises(Exception) as exc_info:
				await ai_service.initialize()

			assert "Init failed" in str(exc_info.value)
			assert ai_service._initialized == False

	@pytest.mark.asyncio
	async def test_service_cleanup(self, ai_service):
		"""Test service cleanup process."""
		# Initialize first
		with patch.object(ai_service.security_manager, 'initialize', new_callable=AsyncMock), \
			 patch.object(ai_service.monitoring, 'initialize', new_callable=AsyncMock), \
			 patch.object(ai_service, '_initialize_inference_engines', new_callable=AsyncMock), \
			 patch.object(ai_service, '_start_background_tasks', new_callable=AsyncMock):

			await ai_service.initialize()

		# Test cleanup
		with patch.object(ai_service, '_cleanup_background_tasks', new_callable=AsyncMock) as mock_cleanup:
			await ai_service.cleanup()
			mock_cleanup.assert_called_once()


class TestModelManagement:
	"""Test cases for model management operations."""

	@pytest.fixture
	async def initialized_service(self):
		"""Create and initialize an AI service for testing."""
		service = AICoreService()

		# Mock all dependencies
		with patch.object(service.security_manager, 'initialize', new_callable=AsyncMock), \
			 patch.object(service.monitoring, 'initialize', new_callable=AsyncMock), \
			 patch.object(service, '_initialize_inference_engines', new_callable=AsyncMock), \
			 patch.object(service, '_start_background_tasks', new_callable=AsyncMock):

			await service.initialize()

		return service

	@pytest.mark.asyncio
	async def test_register_model(self, initialized_service):
		"""Test registering a new model."""
		model_data = {
			"name": "test_model",
			"description": "Test model for unit testing",
			"model_type": "classification",
			"framework": "pytorch",
			"version": "1.0.0"
		}

		model = await initialized_service.register_model(model_data)

		assert isinstance(model, AICRModel)
		assert model.name == "test_model"
		assert model.model_type == ModelType.CLASSIFICATION
		assert model.framework == "pytorch"
		assert model.model_id in initialized_service.models

		# Verify model is stored
		stored_model = initialized_service.models[model.model_id]
		assert stored_model.name == model.name

	@pytest.mark.asyncio
	async def test_register_model_validation_error(self, initialized_service):
		"""Test registering a model with invalid data."""
		invalid_model_data = {
			"name": "",  # Invalid empty name
			"description": "Test model",
			"model_type": "invalid_type",  # Invalid type
			"framework": "pytorch"
		}

		with pytest.raises(Exception):
			await initialized_service.register_model(invalid_model_data)

	@pytest.mark.asyncio
	async def test_get_model(self, initialized_service):
		"""Test retrieving a model by ID."""
		# Register a model first
		model_data = {
			"name": "get_test_model",
			"description": "Model for get testing",
			"model_type": "regression",
			"framework": "sklearn"
		}

		registered_model = await initialized_service.register_model(model_data)

		# Test getting the model
		retrieved_model = await initialized_service.get_model(registered_model.model_id)

		assert retrieved_model is not None
		assert retrieved_model.model_id == registered_model.model_id
		assert retrieved_model.name == "get_test_model"

		# Test getting non-existent model
		non_existent_model = await initialized_service.get_model("non_existent_id")
		assert non_existent_model is None

	@pytest.mark.asyncio
	async def test_list_models(self, initialized_service):
		"""Test listing models with filters."""
		# Register multiple models
		models_data = [
			{
				"name": "model_1",
				"description": "First model",
				"model_type": "classification",
				"framework": "pytorch"
			},
			{
				"name": "model_2",
				"description": "Second model",
				"model_type": "regression",
				"framework": "tensorflow"
			},
			{
				"name": "model_3",
				"description": "Third model",
				"model_type": "classification",
				"framework": "pytorch"
			}
		]

		for model_data in models_data:
			await initialized_service.register_model(model_data)

		# Test listing all models
		all_models = await initialized_service.list_models()
		assert len(all_models) == 3

		# Test filtering by model type
		classification_models = await initialized_service.list_models(model_type="classification")
		assert len(classification_models) == 2
		assert all(m.model_type == ModelType.CLASSIFICATION for m in classification_models)

		# Test filtering by framework
		pytorch_models = await initialized_service.list_models(framework="pytorch")
		assert len(pytorch_models) == 2
		assert all(m.framework == "pytorch" for m in pytorch_models)

		# Test filtering with limit
		limited_models = await initialized_service.list_models(limit=2)
		assert len(limited_models) == 2

	@pytest.mark.asyncio
	async def test_update_model(self, initialized_service):
		"""Test updating an existing model."""
		# Register a model
		model_data = {
			"name": "update_test_model",
			"description": "Model for update testing",
			"model_type": "classification",
			"framework": "pytorch"
		}

		model = await initialized_service.register_model(model_data)

		# Update the model
		update_data = {
			"description": "Updated description",
			"version": "2.0.0",
			"status": "active"
		}

		updated_model = await initialized_service.update_model(model.model_id, update_data)

		assert updated_model is not None
		assert updated_model.description == "Updated description"
		assert updated_model.version == "2.0.0"
		assert updated_model.status == "active"

		# Verify the stored model is updated
		stored_model = initialized_service.models[model.model_id]
		assert stored_model.description == "Updated description"

	@pytest.mark.asyncio
	async def test_delete_model(self, initialized_service):
		"""Test deleting a model."""
		# Register a model
		model_data = {
			"name": "delete_test_model",
			"description": "Model for delete testing",
			"model_type": "clustering",
			"framework": "sklearn"
		}

		model = await initialized_service.register_model(model_data)
		model_id = model.model_id

		# Verify model exists
		assert model_id in initialized_service.models

		# Delete the model
		success = await initialized_service.delete_model(model_id)

		assert success == True
		assert model_id not in initialized_service.models

		# Test deleting non-existent model
		success = await initialized_service.delete_model("non_existent_id")
		assert success == False


class TestModelDeployment:
	"""Test cases for model deployment operations."""

	@pytest.fixture
	async def service_with_model(self):
		"""Create service with a registered model."""
		service = AICoreService()

		# Mock initialization
		with patch.object(service.security_manager, 'initialize', new_callable=AsyncMock), \
			 patch.object(service.monitoring, 'initialize', new_callable=AsyncMock), \
			 patch.object(service, '_initialize_inference_engines', new_callable=AsyncMock), \
			 patch.object(service, '_start_background_tasks', new_callable=AsyncMock):

			await service.initialize()

		# Register a test model
		model_data = {
			"name": "deployment_test_model",
			"description": "Model for deployment testing",
			"model_type": "classification",
			"framework": "pytorch",
			"file_path": "/models/test_model.pth"
		}

		model = await service.register_model(model_data)
		return service, model

	@pytest.mark.asyncio
	async def test_deploy_model_success(self, service_with_model):
		"""Test successful model deployment."""
		service, model = service_with_model

		# Mock inference engine deployment
		mock_engine = Mock()
		mock_engine.deploy_model = AsyncMock(return_value={"success": True, "endpoint": "test_endpoint"})
		service.inference_engines["pytorch"] = mock_engine

		deployment_config = {"gpu_enabled": True, "batch_size": 32}
		result = await service.deploy_model(model.model_id, deployment_config)

		assert result["success"] == True
		assert "endpoint" in result

		# Verify model status updated
		updated_model = service.models[model.model_id]
		assert updated_model.status == "deployed"
		assert updated_model.deployment_count == 1

		# Verify deployment registry
		assert model.model_id in service.deployment_registry

	@pytest.mark.asyncio
	async def test_deploy_nonexistent_model(self, service_with_model):
		"""Test deploying a non-existent model."""
		service, _ = service_with_model

		with pytest.raises(ValueError) as exc_info:
			await service.deploy_model("non_existent_id")

		assert "Model not found" in str(exc_info.value)

	@pytest.mark.asyncio
	async def test_deploy_model_engine_failure(self, service_with_model):
		"""Test model deployment with engine failure."""
		service, model = service_with_model

		# Mock inference engine deployment failure
		mock_engine = Mock()
		mock_engine.deploy_model = AsyncMock(side_effect=Exception("Deployment failed"))
		service.inference_engines["pytorch"] = mock_engine

		with pytest.raises(Exception) as exc_info:
			await service.deploy_model(model.model_id)

		assert "Deployment failed" in str(exc_info.value)

		# Verify model status not changed
		updated_model = service.models[model.model_id]
		assert updated_model.status != "deployed"

	@pytest.mark.asyncio
	async def test_undeploy_model(self, service_with_model):
		"""Test undeploying a model."""
		service, model = service_with_model

		# First deploy the model
		mock_engine = Mock()
		mock_engine.deploy_model = AsyncMock(return_value={"success": True})
		mock_engine.undeploy_model = AsyncMock(return_value={"success": True})
		service.inference_engines["pytorch"] = mock_engine

		await service.deploy_model(model.model_id)

		# Now undeploy
		result = await service.undeploy_model(model.model_id)

		assert result["success"] == True

		# Verify model status updated
		updated_model = service.models[model.model_id]
		assert updated_model.status != "deployed"

		# Verify removed from deployment registry
		assert model.model_id not in service.deployment_registry


class TestInferenceExecution:
	"""Test cases for inference execution."""

	@pytest.fixture
	async def service_with_deployed_model(self):
		"""Create service with a deployed model."""
		service = AICoreService()

		# Mock initialization
		with patch.object(service.security_manager, 'initialize', new_callable=AsyncMock), \
			 patch.object(service.monitoring, 'initialize', new_callable=AsyncMock), \
			 patch.object(service, '_initialize_inference_engines', new_callable=AsyncMock), \
			 patch.object(service, '_start_background_tasks', new_callable=AsyncMock):

			await service.initialize()

		# Register and deploy a model
		model_data = {
			"name": "inference_test_model",
			"description": "Model for inference testing",
			"model_type": "classification",
			"framework": "pytorch"
		}

		model = await service.register_model(model_data)

		# Mock engine for deployment and inference
		mock_engine = Mock()
		mock_engine.deploy_model = AsyncMock(return_value={"success": True})
		mock_engine.run_inference = AsyncMock(return_value={
			"predictions": {"class": "cat", "confidence": 0.95},
			"processing_time_ms": 150.5
		})
		service.inference_engines["pytorch"] = mock_engine

		await service.deploy_model(model.model_id)

		return service, model

	@pytest.mark.asyncio
	async def test_run_inference_success(self, service_with_deployed_model):
		"""Test successful inference execution."""
		service, model = service_with_deployed_model

		# Create inference request
		request = AICRInferenceRequest(
			model_id=model.model_id,
			input_data={"image": "base64_encoded_image"},
			parameters={"temperature": 0.7}
		)

		# Run inference
		response = await service.run_inference(request)

		assert isinstance(response, AICRInferenceResponse)
		assert response.request_id == request.request_id
		assert response.model_id == model.model_id
		assert response.status == InferenceStatus.COMPLETED
		assert response.predictions is not None
		assert response.processing_time_ms > 0

		# Verify model last_inference updated
		updated_model = service.models[model.model_id]
		assert updated_model.last_inference is not None

	@pytest.mark.asyncio
	async def test_run_inference_undeployed_model(self, service_with_deployed_model):
		"""Test inference on undeployed model."""
		service, model = service_with_deployed_model

		# Undeploy the model first
		await service.undeploy_model(model.model_id)

		# Create inference request
		request = AICRInferenceRequest(
			model_id=model.model_id,
			input_data={"data": [1, 2, 3]}
		)

		# Run inference should fail
		response = await service.run_inference(request)

		assert response.status == InferenceStatus.FAILED
		assert "not deployed" in response.error_message

	@pytest.mark.asyncio
	async def test_run_inference_engine_failure(self, service_with_deployed_model):
		"""Test inference with engine failure."""
		service, model = service_with_deployed_model

		# Mock engine to fail
		mock_engine = service.inference_engines["pytorch"]
		mock_engine.run_inference = AsyncMock(side_effect=Exception("Inference failed"))

		# Create inference request
		request = AICRInferenceRequest(
			model_id=model.model_id,
			input_data={"data": [1, 2, 3]}
		)

		# Run inference
		response = await service.run_inference(request)

		assert response.status == InferenceStatus.FAILED
		assert "Inference failed" in response.error_message

	@pytest.mark.asyncio
	async def test_run_inference_timeout(self, service_with_deployed_model):
		"""Test inference timeout handling."""
		service, model = service_with_deployed_model

		# Mock engine to timeout
		async def slow_inference(*args, **kwargs):
			await asyncio.sleep(2)  # Simulate slow inference
			return {"predictions": {"result": "slow"}}

		mock_engine = service.inference_engines["pytorch"]
		mock_engine.run_inference = slow_inference

		# Create inference request with short timeout
		request = AICRInferenceRequest(
			model_id=model.model_id,
			input_data={"data": [1, 2, 3]},
			timeout_seconds=1
		)

		# Run inference
		response = await service.run_inference(request)

		assert response.status == InferenceStatus.FAILED
		assert "timeout" in response.error_message.lower()


class TestBatchInference:
	"""Test cases for batch inference operations."""

	@pytest.fixture
	async def service_for_batch(self):
		"""Create service for batch inference testing."""
		service = AICoreService()

		# Mock initialization
		with patch.object(service.security_manager, 'initialize', new_callable=AsyncMock), \
			 patch.object(service.monitoring, 'initialize', new_callable=AsyncMock), \
			 patch.object(service, '_initialize_inference_engines', new_callable=AsyncMock), \
			 patch.object(service, '_start_background_tasks', new_callable=AsyncMock):

			await service.initialize()

		# Register and deploy a model
		model_data = {
			"name": "batch_test_model",
			"description": "Model for batch testing",
			"model_type": "classification",
			"framework": "pytorch"
		}

		model = await service.register_model(model_data)

		# Mock engine
		mock_engine = Mock()
		mock_engine.deploy_model = AsyncMock(return_value={"success": True})
		mock_engine.run_batch_inference = AsyncMock(return_value=[
			{"predictions": {"class": "cat"}, "processing_time_ms": 100},
			{"predictions": {"class": "dog"}, "processing_time_ms": 110},
			{"predictions": {"class": "bird"}, "processing_time_ms": 95}
		])
		service.inference_engines["pytorch"] = mock_engine

		await service.deploy_model(model.model_id)

		return service, model

	@pytest.mark.asyncio
	async def test_run_batch_inference(self, service_for_batch):
		"""Test batch inference execution."""
		service, model = service_for_batch

		# Create batch data
		batch_data = [
			{"image": "cat_image"},
			{"image": "dog_image"},
			{"image": "bird_image"}
		]

		# Run batch inference
		results = await service.run_batch_inference(model.model_id, batch_data)

		assert len(results) == 3

		for result in results:
			assert isinstance(result, AICRInferenceResponse)
			assert result.model_id == model.model_id
			assert result.status == InferenceStatus.COMPLETED
			assert result.predictions is not None

	@pytest.mark.asyncio
	async def test_run_batch_inference_partial_failure(self, service_for_batch):
		"""Test batch inference with partial failures."""
		service, model = service_for_batch

		# Mock engine to fail on second item
		def batch_inference_with_failure(*args, **kwargs):
			return [
				{"predictions": {"class": "cat"}, "processing_time_ms": 100},
				{"error": "Processing failed"},
				{"predictions": {"class": "bird"}, "processing_time_ms": 95}
			]

		mock_engine = service.inference_engines["pytorch"]
		mock_engine.run_batch_inference = AsyncMock(side_effect=batch_inference_with_failure)

		batch_data = [
			{"image": "cat_image"},
			{"image": "invalid_image"},
			{"image": "bird_image"}
		]

		results = await service.run_batch_inference(model.model_id, batch_data)

		assert len(results) == 3
		assert results[0].status == InferenceStatus.COMPLETED
		assert results[1].status == InferenceStatus.FAILED
		assert results[2].status == InferenceStatus.COMPLETED


class TestServiceMonitoring:
	"""Test cases for service monitoring and metrics."""

	@pytest.fixture
	async def monitored_service(self):
		"""Create service with monitoring enabled."""
		service = AICoreService()

		# Mock monitoring
		service.monitoring = Mock()
		service.monitoring.initialize = AsyncMock()
		service.monitoring.record_metric = AsyncMock()
		service.monitoring.record_event = AsyncMock()

		with patch.object(service.security_manager, 'initialize', new_callable=AsyncMock), \
			 patch.object(service, '_initialize_inference_engines', new_callable=AsyncMock), \
			 patch.object(service, '_start_background_tasks', new_callable=AsyncMock):

			await service.initialize()

		return service

	@pytest.mark.asyncio
	async def test_service_metrics_recording(self, monitored_service):
		"""Test that service operations record metrics."""
		# Register a model
		model_data = {
			"name": "metrics_test_model",
			"description": "Model for metrics testing",
			"model_type": "classification",
			"framework": "pytorch"
		}

		await monitored_service.register_model(model_data)

		# Verify metrics were recorded
		monitored_service.monitoring.record_event.assert_called()

		# Check that model registration event was recorded
		calls = monitored_service.monitoring.record_event.call_args_list
		event_types = [call[0][0] for call in calls]
		assert "model_registered" in event_types

	@pytest.mark.asyncio
	async def test_inference_metrics_recording(self, monitored_service):
		"""Test that inference operations record metrics."""
		# Register and deploy a model
		model_data = {
			"name": "inference_metrics_model",
			"description": "Model for inference metrics testing",
			"model_type": "classification",
			"framework": "pytorch"
		}

		model = await monitored_service.register_model(model_data)

		# Mock engine
		mock_engine = Mock()
		mock_engine.deploy_model = AsyncMock(return_value={"success": True})
		mock_engine.run_inference = AsyncMock(return_value={
			"predictions": {"class": "test"},
			"processing_time_ms": 200
		})
		monitored_service.inference_engines["pytorch"] = mock_engine

		await monitored_service.deploy_model(model.model_id)

		# Create and run inference
		request = AICRInferenceRequest(
			model_id=model.model_id,
			input_data={"data": [1, 2, 3]}
		)

		await monitored_service.run_inference(request)

		# Verify inference metrics were recorded
		metric_calls = monitored_service.monitoring.record_metric.call_args_list
		metric_names = [call[0][0] for call in metric_calls]

		assert "inference_latency" in metric_names
		assert "inference_count" in metric_names


class TestServiceErrorHandling:
	"""Test cases for service error handling and resilience."""

	@pytest.fixture
	def error_service(self):
		"""Create service for error testing."""
		return AICoreService()

	@pytest.mark.asyncio
	async def test_operation_before_initialization(self, error_service):
		"""Test operations before service initialization."""
		# Try to register model before initialization
		model_data = {
			"name": "test_model",
			"description": "Test model",
			"model_type": "classification",
			"framework": "pytorch"
		}

		with pytest.raises(RuntimeError) as exc_info:
			await error_service.register_model(model_data)

		assert "not initialized" in str(exc_info.value)

	@pytest.mark.asyncio
	async def test_concurrent_operations(self, error_service):
		"""Test concurrent service operations."""
		# Initialize service
		with patch.object(error_service.security_manager, 'initialize', new_callable=AsyncMock), \
			 patch.object(error_service.monitoring, 'initialize', new_callable=AsyncMock), \
			 patch.object(error_service, '_initialize_inference_engines', new_callable=AsyncMock), \
			 patch.object(error_service, '_start_background_tasks', new_callable=AsyncMock):

			await error_service.initialize()

		# Create multiple concurrent model registrations
		model_tasks = []
		for i in range(5):
			model_data = {
				"name": f"concurrent_model_{i}",
				"description": f"Concurrent model {i}",
				"model_type": "classification",
				"framework": "pytorch"
			}
			task = error_service.register_model(model_data)
			model_tasks.append(task)

		# Wait for all registrations to complete
		models = await asyncio.gather(*model_tasks, return_exceptions=True)


		# Verify all models were registered
		assert len(models) == 5
		assert len(error_service.models) == 5

		# Verify all models have unique IDs
		model_ids = [model.model_id for model in models]
		assert len(set(model_ids)) == 5


if __name__ == "__main__":
	pytest.main([__file__])