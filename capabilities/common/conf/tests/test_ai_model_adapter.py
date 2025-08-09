"""
APG Configuration Management - AI Model Adapter Tests
Comprehensive tests for AI model configuration management integration with common/ai.

© 2025 Datacraft - www.datacraft.co.ke  
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

from ..ai_model_adapter import AIModelConfigurationAdapter, get_ai_model_adapter
from ..models import (
	AIModelConfiguration, MLPipelineConfiguration, NLPServiceConfiguration,
	AIModelFramework, AIModelType, AIModelState, ModelProvider, CloudProvider
)


class TestAIModelConfigurationAdapter:
	"""Test suite for AI Model Configuration Adapter"""
	
	@pytest.fixture
	async def adapter(self) -> AIModelConfigurationAdapter:
		"""Create adapter instance for testing"""
		return AIModelConfigurationAdapter(tenant_id="test_tenant")
	
	@pytest.fixture
	def sample_ollama_config(self) -> Dict[str, Any]:
		"""Sample Ollama model configuration"""
		return {
			"name": "test-llama-chat",
			"display_name": "Test Llama Chat Model",
			"description": "Test language model for conversational AI",
			"framework": AIModelFramework.OLLAMA,
			"model_type": AIModelType.TEXT_GENERATION,
			"provider": ModelProvider.OLLAMA,
			"provider_model_name": "llama3.2:latest",
			"cloud_provider": CloudProvider.ON_PREMISES,
			"deployment_target": "testing",
			"model_parameters": {
				"temperature": 0.7,
				"max_tokens": 2048,
				"top_p": 0.9
			},
			"runtime_config": {
				"batch_size": 1,
				"max_concurrent_requests": 5
			},
			"resource_requirements": {
				"cpu_cores": 2,
				"memory_gb": 4,
				"gpu_required": False
			},
			"supported_tasks": [AIModelType.TEXT_GENERATION],
			"supported_languages": ["en"],
			"tags": {"environment": "testing", "use_case": "chat"},
			"created_by": "test@datacraft.co.ke"
		}
	
	@pytest.fixture
	def sample_transformers_config(self) -> Dict[str, Any]:
		"""Sample Transformers model configuration"""
		return {
			"name": "test-bert-sentiment",
			"display_name": "Test BERT Sentiment Analyzer",
			"description": "Test BERT model for sentiment analysis",
			"framework": AIModelFramework.TRANSFORMERS,
			"model_type": AIModelType.SENTIMENT_ANALYSIS,
			"provider": ModelProvider.TRANSFORMERS,
			"provider_model_name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
			"cloud_provider": CloudProvider.AWS,
			"deployment_target": "testing",
			"model_parameters": {
				"max_length": 512,
				"truncation": True,
				"padding": True
			},
			"runtime_config": {
				"batch_size": 16,
				"use_gpu": False
			},
			"resource_requirements": {
				"cpu_cores": 1,
				"memory_gb": 2,
				"gpu_required": False
			},
			"supported_tasks": [AIModelType.SENTIMENT_ANALYSIS],
			"supported_languages": ["en"],
			"tags": {"environment": "testing", "use_case": "sentiment"},
			"created_by": "test@datacraft.co.ke"
		}
	
	async def test_adapter_initialization(self, adapter: AIModelConfigurationAdapter):
		"""Test adapter initialization"""
		assert adapter.tenant_id == "test_tenant"
		assert isinstance(adapter.ai_model_configs, dict)
		assert isinstance(adapter.ml_pipeline_configs, dict)
		assert isinstance(adapter.nlp_service_configs, dict)
		assert adapter._config_manager is None
		assert adapter._nlp_service is None
		assert adapter._gitops_manager is None
	
	async def test_adapter_initialization_with_invalid_tenant(self):
		"""Test adapter initialization with invalid tenant ID"""
		with pytest.raises(AssertionError, match="tenant_id is required"):
			AIModelConfigurationAdapter("")
		
		with pytest.raises(AssertionError, match="tenant_id must be string"):
			AIModelConfigurationAdapter(123)
	
	async def test_register_ai_model_configuration(
		self,
		adapter: AIModelConfigurationAdapter,
		sample_ollama_config: Dict[str, Any]
	):
		"""Test registering AI model configuration"""
		# Mock configuration manager
		mock_config_manager = AsyncMock()
		adapter.set_config_manager(mock_config_manager)
		
		# Register model
		model_id = await adapter.register_ai_model_configuration(sample_ollama_config)
		
		# Verify registration
		assert model_id in adapter.ai_model_configs
		config = adapter.ai_model_configs[model_id]
		assert config.name == "test-llama-chat"
		assert config.framework == AIModelFramework.OLLAMA
		assert config.model_type == AIModelType.TEXT_GENERATION
		assert config.state == AIModelState.CONFIGURED
		
		# Verify CM resource creation was called
		mock_config_manager.create_resource.assert_called_once()
	
	async def test_register_ai_model_configuration_without_data(
		self,
		adapter: AIModelConfigurationAdapter
	):
		"""Test registering AI model configuration without data"""
		with pytest.raises(AssertionError, match="Model configuration data is required"):
			await adapter.register_ai_model_configuration({})
	
	async def test_convert_to_cm_resource(
		self,
		adapter: AIModelConfigurationAdapter,
		sample_transformers_config: Dict[str, Any]
	):
		"""Test converting AI model config to CM resource"""
		# Create AI model configuration
		ai_model_config = AIModelConfiguration(
			tenant_id="test_tenant",
			**sample_transformers_config
		)
		
		# Convert to CM resource
		cm_resource = await adapter._convert_to_cm_resource(ai_model_config)
		
		# Verify CM resource properties
		assert cm_resource.tenant_id == "test_tenant"
		assert cm_resource.name == "test-bert-sentiment"
		assert cm_resource.resource_type.value == "ai_model"
		assert cm_resource.cloud_provider == CloudProvider.AWS
		assert "ai_model_type" in cm_resource.tags
		assert "framework" in cm_resource.tags
		assert "provider" in cm_resource.tags
	
	async def test_deploy_ai_model_configuration(
		self,
		adapter: AIModelConfigurationAdapter,
		sample_ollama_config: Dict[str, Any]
	):
		"""Test deploying AI model configuration"""
		# Register model first
		model_id = await adapter.register_ai_model_configuration(sample_ollama_config)
		
		# Mock dependencies
		mock_nlp_service = AsyncMock()
		mock_gitops_manager = AsyncMock()
		mock_gitops_manager.create_deployment_plan.return_value = "deployment_123"
		
		adapter.set_nlp_service(mock_nlp_service)
		adapter.set_gitops_manager(mock_gitops_manager)
		
		# Deploy model
		deployment_id = await adapter.deploy_ai_model_configuration(model_id)
		
		# Verify deployment
		config = adapter.ai_model_configs[model_id]
		assert config.state == AIModelState.READY
		assert config.deployed_at is not None
		assert deployment_id == "deployment_123"
		
		# Verify GitOps deployment was called
		mock_gitops_manager.create_deployment_plan.assert_called_once()
	
	async def test_deploy_nonexistent_model(self, adapter: AIModelConfigurationAdapter):
		"""Test deploying nonexistent model configuration"""
		with pytest.raises(ValueError, match="AI model configuration not found"):
			await adapter.deploy_ai_model_configuration("nonexistent_id")
	
	@patch('capabilities.common.conf.ai_model_adapter.logger')
	async def test_deploy_ai_model_with_error(
		self,
		mock_logger,
		adapter: AIModelConfigurationAdapter,
		sample_ollama_config: Dict[str, Any]
	):
		"""Test deploying AI model configuration with error"""
		# Register model first
		model_id = await adapter.register_ai_model_configuration(sample_ollama_config)
		
		# Mock GitOps manager to raise error
		mock_gitops_manager = AsyncMock()
		mock_gitops_manager.create_deployment_plan.side_effect = Exception("Deployment failed")
		adapter.set_gitops_manager(mock_gitops_manager)
		
		# Attempt deployment
		with pytest.raises(Exception, match="Deployment failed"):
			await adapter.deploy_ai_model_configuration(model_id)
		
		# Verify error state
		config = adapter.ai_model_configs[model_id]
		assert config.state == AIModelState.FAILED
	
	async def test_nlp_service_integration(
		self,
		adapter: AIModelConfigurationAdapter,
		sample_transformers_config: Dict[str, Any]
	):
		"""Test NLP service integration"""
		# Create AI model configuration
		ai_model_config = AIModelConfiguration(
			tenant_id="test_tenant",
			**sample_transformers_config
		)
		
		# Mock NLP service
		mock_nlp_service = AsyncMock()
		adapter.set_nlp_service(mock_nlp_service)
		
		# Mock registration method
		adapter._register_model_with_nlp_service = AsyncMock()
		
		# Test integration
		await adapter._integrate_with_nlp_service(ai_model_config)
		
		# Verify registration was called with correct data
		adapter._register_model_with_nlp_service.assert_called_once()
		call_args = adapter._register_model_with_nlp_service.call_args[0][0]
		assert call_args["name"] == "Test BERT Sentiment Analyzer"
		assert call_args["provider"] == "transformers"
		assert "sentiment_analysis" in call_args["supported_tasks"]
	
	async def test_provider_mapping(self, adapter: AIModelConfigurationAdapter):
		"""Test provider mapping for NLP service"""
		assert adapter._map_to_nlp_provider(ModelProvider.OLLAMA) == "ollama"
		assert adapter._map_to_nlp_provider(ModelProvider.TRANSFORMERS) == "transformers"
		assert adapter._map_to_nlp_provider(ModelProvider.SPACY) == "spacy"
		assert adapter._map_to_nlp_provider(ModelProvider.OPENAI) == "openai"
		assert adapter._map_to_nlp_provider(ModelProvider.CUSTOM) == "custom"
	
	async def test_task_mapping(self, adapter: AIModelConfigurationAdapter):
		"""Test task type mapping for NLP service"""
		assert adapter._map_to_nlp_task(AIModelType.TEXT_GENERATION) == "text_generation"
		assert adapter._map_to_nlp_task(AIModelType.SENTIMENT_ANALYSIS) == "sentiment_analysis"
		assert adapter._map_to_nlp_task(AIModelType.NAMED_ENTITY_RECOGNITION) == "named_entity_recognition"
		assert adapter._map_to_nlp_task(AIModelType.TEXT_CLASSIFICATION) == "text_classification"
	
	async def test_generate_deployment_manifest(
		self,
		adapter: AIModelConfigurationAdapter,
		sample_ollama_config: Dict[str, Any]
	):
		"""Test deployment manifest generation"""
		# Create AI model configuration
		ai_model_config = AIModelConfiguration(
			tenant_id="test_tenant",
			**sample_ollama_config
		)
		
		# Generate manifest
		manifest = await adapter._generate_deployment_manifest(ai_model_config, {})
		
		# Verify manifest structure
		assert manifest["apiVersion"] == "apg.datacraft.co.ke/v1"
		assert manifest["kind"] == "AIModelDeployment"
		assert manifest["metadata"]["name"] == "test-llama-chat"
		assert manifest["spec"]["model"]["framework"] == "ollama"
		assert manifest["spec"]["model"]["type"] == "text_generation"
		assert "apg.datacraft.co.ke/tenant-id" in manifest["metadata"]["annotations"]
	
	async def test_create_ml_pipeline_configuration(
		self,
		adapter: AIModelConfigurationAdapter,
		sample_ollama_config: Dict[str, Any],
		sample_transformers_config: Dict[str, Any]
	):
		"""Test creating ML pipeline configuration"""
		# Register models first
		model1_id = await adapter.register_ai_model_configuration(sample_ollama_config)
		model2_id = await adapter.register_ai_model_configuration(sample_transformers_config)
		
		# Mock configuration manager
		mock_config_manager = AsyncMock()
		adapter.set_config_manager(mock_config_manager)
		
		# Create pipeline configuration
		pipeline_config_data = {
			"name": "test-ml-pipeline",
			"description": "Test ML pipeline with multiple models",
			"version": "1.0.0",
			"models": [model1_id, model2_id],
			"execution_mode": "sequential",
			"preprocessing_steps": ["tokenization", "normalization"],
			"postprocessing_steps": ["aggregation"],
			"input_schema": {"type": "text", "format": "string"},
			"output_schema": {"type": "analysis", "format": "json"},
			"pipeline_config": {"timeout": 300, "retry_count": 3},
			"parallelism": 2,
			"timeout_seconds": 600,
			"resource_requirements": {"cpu_cores": 4, "memory_gb": 8},
			"cloud_provider": CloudProvider.AWS,
			"deployment_target": "testing",
			"tags": {"environment": "testing", "type": "pipeline"},
			"created_by": "test@datacraft.co.ke"
		}
		
		# Create pipeline
		pipeline_id = await adapter.create_ml_pipeline_configuration(pipeline_config_data)
		
		# Verify pipeline creation
		assert pipeline_id in adapter.ml_pipeline_configs
		pipeline = adapter.ml_pipeline_configs[pipeline_id]
		assert pipeline.name == "test-ml-pipeline"
		assert len(pipeline.models) == 2
		assert model1_id in pipeline.models
		assert model2_id in pipeline.models
		
		# Verify CM resource creation was called
		mock_config_manager.create_resource.assert_called_once()
	
	async def test_create_ml_pipeline_with_invalid_model(
		self,
		adapter: AIModelConfigurationAdapter
	):
		"""Test creating ML pipeline with invalid model reference"""
		pipeline_config_data = {
			"name": "test-invalid-pipeline",
			"models": ["nonexistent_model_id"],
			"execution_mode": "sequential",
			"created_by": "test@datacraft.co.ke"
		}
		
		with pytest.raises(ValueError, match="Referenced AI model not found"):
			await adapter.create_ml_pipeline_configuration(pipeline_config_data)
	
	async def test_create_nlp_service_configuration(
		self,
		adapter: AIModelConfigurationAdapter,
		sample_transformers_config: Dict[str, Any]
	):
		"""Test creating NLP service configuration"""
		# Register model first
		model_id = await adapter.register_ai_model_configuration(sample_transformers_config)
		
		# Mock configuration manager
		mock_config_manager = AsyncMock()
		adapter.set_config_manager(mock_config_manager)
		
		# Create NLP service configuration
		service_config_data = {
			"name": "test-nlp-service",
			"description": "Test NLP service configuration",
			"service_type": "multi_model",
			"registered_models": [model_id],
			"service_config": {
				"max_concurrent_requests": 100,
				"request_timeout": 30,
				"batch_processing": True
			},
			"api_config": {
				"version": "v1",
				"authentication": "api_key",
				"rate_limiting": {"requests_per_minute": 1000}
			},
			"deployment_config": {
				"replicas": 3,
				"load_balancing": "round_robin",
				"health_checks": True
			},
			"cloud_provider": CloudProvider.AWS,
			"deployment_target": "testing",
			"tags": {"environment": "testing", "service": "nlp"},
			"created_by": "test@datacraft.co.ke"
		}
		
		# Create service
		service_id = await adapter.create_nlp_service_configuration(service_config_data)
		
		# Verify service creation
		assert service_id in adapter.nlp_service_configs
		service = adapter.nlp_service_configs[service_id]
		assert service.name == "test-nlp-service"
		assert model_id in service.registered_models
		
		# Verify CM resource creation was called
		mock_config_manager.create_resource.assert_called_once()
	
	async def test_get_ai_model_configuration(
		self,
		adapter: AIModelConfigurationAdapter,
		sample_ollama_config: Dict[str, Any]
	):
		"""Test retrieving AI model configuration"""
		# Register model first
		model_id = await adapter.register_ai_model_configuration(sample_ollama_config)
		
		# Retrieve configuration
		config = await adapter.get_ai_model_configuration(model_id)
		
		# Verify retrieval
		assert config.id == model_id
		assert config.name == "test-llama-chat"
		assert config.framework == AIModelFramework.OLLAMA
	
	async def test_get_nonexistent_ai_model_configuration(
		self,
		adapter: AIModelConfigurationAdapter
	):
		"""Test retrieving nonexistent AI model configuration"""
		with pytest.raises(ValueError, match="AI model configuration not found"):
			await adapter.get_ai_model_configuration("nonexistent_id")
	
	async def test_list_ai_model_configurations(
		self,
		adapter: AIModelConfigurationAdapter,
		sample_ollama_config: Dict[str, Any],
		sample_transformers_config: Dict[str, Any]
	):
		"""Test listing AI model configurations"""
		# Register models
		await adapter.register_ai_model_configuration(sample_ollama_config)
		await adapter.register_ai_model_configuration(sample_transformers_config)
		
		# List all configurations
		all_configs = await adapter.list_ai_model_configurations()
		assert len(all_configs) == 2
		
		# List with framework filter
		ollama_configs = await adapter.list_ai_model_configurations(
			filters={"framework": AIModelFramework.OLLAMA}
		)
		assert len(ollama_configs) == 1
		assert ollama_configs[0].framework == AIModelFramework.OLLAMA
		
		# List with model type filter
		sentiment_configs = await adapter.list_ai_model_configurations(
			filters={"model_type": AIModelType.SENTIMENT_ANALYSIS}
		)
		assert len(sentiment_configs) == 1
		assert sentiment_configs[0].model_type == AIModelType.SENTIMENT_ANALYSIS
	
	async def test_update_ai_model_state(
		self,
		adapter: AIModelConfigurationAdapter,
		sample_ollama_config: Dict[str, Any]
	):
		"""Test updating AI model state"""
		# Register model first
		model_id = await adapter.register_ai_model_configuration(sample_ollama_config)
		
		# Update state
		await adapter.update_ai_model_state(model_id, AIModelState.LOADING)
		
		# Verify state update
		config = adapter.ai_model_configs[model_id]
		assert config.state == AIModelState.LOADING
		assert config.updated_at is not None
	
	async def test_get_configuration_summary(
		self,
		adapter: AIModelConfigurationAdapter,
		sample_ollama_config: Dict[str, Any],
		sample_transformers_config: Dict[str, Any]
	):
		"""Test getting configuration summary"""
		# Register models
		await adapter.register_ai_model_configuration(sample_ollama_config)
		await adapter.register_ai_model_configuration(sample_transformers_config)
		
		# Create pipeline
		model_ids = list(adapter.ai_model_configs.keys())
		pipeline_config_data = {
			"name": "test-pipeline",
			"models": model_ids,
			"execution_mode": "sequential",
			"created_by": "test@datacraft.co.ke"
		}
		await adapter.create_ml_pipeline_configuration(pipeline_config_data)
		
		# Get summary
		summary = await adapter.get_configuration_summary()
		
		# Verify summary
		assert summary["totals"]["ai_models"] == 2
		assert summary["totals"]["ml_pipelines"] == 1
		assert summary["totals"]["nlp_services"] == 0
		assert "configured" in summary["ai_model_states"]
		assert "ollama" in summary["frameworks"]
		assert "transformers" in summary["frameworks"]
		assert summary["integration_status"]["config_manager_available"] is False
	
	async def test_dependency_injection(self, adapter: AIModelConfigurationAdapter):
		"""Test dependency injection methods"""
		mock_config_manager = Mock()
		mock_nlp_service = Mock()
		mock_gitops_manager = Mock()
		
		# Test dependency injection
		adapter.set_config_manager(mock_config_manager)
		adapter.set_nlp_service(mock_nlp_service)
		adapter.set_gitops_manager(mock_gitops_manager)
		
		# Verify dependencies were set
		assert adapter._config_manager == mock_config_manager
		assert adapter._nlp_service == mock_nlp_service
		assert adapter._gitops_manager == mock_gitops_manager


class TestGetAIModelAdapter:
	"""Test suite for get_ai_model_adapter factory function"""
	
	async def test_get_ai_model_adapter(self):
		"""Test getting AI model adapter instance"""
		adapter = await get_ai_model_adapter("test_tenant")
		
		# Verify adapter creation
		assert isinstance(adapter, AIModelConfigurationAdapter)
		assert adapter.tenant_id == "test_tenant"
		
		# Verify sample configurations were initialized
		configs = await adapter.list_ai_model_configurations()
		assert len(configs) >= 2  # Should have sample configurations
		
		# Check for expected sample models
		model_names = [config.name for config in configs]
		assert "llama-3.2-chat" in model_names
		assert "bert-sentiment-analyzer" in model_names


class TestAIModelConfigurationIntegration:
	"""Integration tests for AI model configuration with mock common/ai services"""
	
	async def test_end_to_end_ai_model_workflow(self):
		"""Test complete AI model configuration workflow"""
		# Create adapter
		adapter = await get_ai_model_adapter("integration_test")
		
		# Mock dependencies
		mock_config_manager = AsyncMock()
		mock_nlp_service = AsyncMock()
		mock_gitops_manager = AsyncMock()
		mock_gitops_manager.create_deployment_plan.return_value = "deployment_456"
		
		adapter.set_config_manager(mock_config_manager)
		adapter.set_nlp_service(mock_nlp_service)
		adapter.set_gitops_manager(mock_gitops_manager)
		
		# Get sample configuration
		configs = await adapter.list_ai_model_configurations()
		model_config = configs[0]
		model_id = model_config.id
		
		# Test deployment
		deployment_id = await adapter.deploy_ai_model_configuration(model_id)
		
		# Verify deployment
		assert deployment_id == "deployment_456"
		updated_config = await adapter.get_ai_model_configuration(model_id)
		assert updated_config.state == AIModelState.READY
		assert updated_config.deployed_at is not None
		
		# Test ML pipeline creation
		pipeline_config_data = {
			"name": "integration-test-pipeline",
			"description": "Integration test ML pipeline",
			"models": [model_id],
			"execution_mode": "sequential",
			"created_by": "integration@datacraft.co.ke"
		}
		
		pipeline_id = await adapter.create_ml_pipeline_configuration(pipeline_config_data)
		pipeline = adapter.ml_pipeline_configs[pipeline_id]
		assert pipeline.name == "integration-test-pipeline"
		assert model_id in pipeline.models
		
		# Test NLP service configuration
		service_config_data = {
			"name": "integration-test-nlp-service",
			"description": "Integration test NLP service",
			"registered_models": [model_id],
			"created_by": "integration@datacraft.co.ke"
		}
		
		service_id = await adapter.create_nlp_service_configuration(service_config_data)
		service = adapter.nlp_service_configs[service_id]
		assert service.name == "integration-test-nlp-service"
		assert model_id in service.registered_models
		
		# Get final summary
		summary = await adapter.get_configuration_summary()
		assert summary["totals"]["ai_models"] >= 2  # Sample + any registered models
		assert summary["totals"]["ml_pipelines"] == 1
		assert summary["totals"]["nlp_services"] == 1