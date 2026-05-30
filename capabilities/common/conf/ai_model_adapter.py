"""
APG Configuration Management - AI Model Configuration Adapter
Production AI model configuration management integrating with common/nlpc services.

This adapter enables the configuration management system to store, version, 
deploy and manage AI models through the established GitOps workflows.

© 2025 Datacraft - www.datacraft.co.ke  
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

from .models import (
	AIModelConfiguration, MLPipelineConfiguration, NLPServiceConfiguration,
	AIModelFramework, AIModelType, AIModelState, ModelProvider, CloudProvider,
	CMResource, ResourceType, ConfigurationDSL, ResourceState
)

# Logging setup following APG patterns
logger = logging.getLogger(__name__)


class AIModelConfigurationAdapter:
	"""
	AI Model Configuration Adapter for integrating AI models with APG Configuration Management.
	
	This adapter bridges AI model configurations with the universal configuration
	management system, enabling GitOps workflows for AI model deployment.
	"""
	
	def __init__(self, tenant_id: str):
		"""Initialize AI model configuration adapter"""
		assert tenant_id, "tenant_id is required for multi-tenancy"
		assert isinstance(tenant_id, str), "tenant_id must be string"
		
		self.tenant_id = tenant_id
		self.ai_model_configs: Dict[str, AIModelConfiguration] = {}
		self.ml_pipeline_configs: Dict[str, MLPipelineConfiguration] = {}
		self.nlp_service_configs: Dict[str, NLPServiceConfiguration] = {}
		
		# Integration with configuration management
		self._config_manager = None  # Will be injected by APG
		self._nlp_service = None     # Will be injected by common/nlpc
		self._gitops_manager = None  # Will be injected by GitOps
		
		self._log_adapter_initialized()
	
	def _log_adapter_initialized(self) -> None:
		"""Log adapter initialization"""
		logger.info(f"AI Model Configuration Adapter initialized for tenant: {self.tenant_id}")
	
	async def register_ai_model_configuration(
		self,
		model_config_data: Dict[str, Any]
	) -> str:
		"""
		Register AI model configuration for infrastructure management.
		
		This method creates an AI model configuration that can be managed
		through the universal configuration management system.
		"""
		assert model_config_data, "Model configuration data is required"
		
		# Create AI model configuration
		ai_model_config = AIModelConfiguration(
			tenant_id=self.tenant_id,
			**model_config_data
		)
		
		# Store configuration
		self.ai_model_configs[ai_model_config.id] = ai_model_config
		
		# Convert to universal configuration resource
		cm_resource = await self._convert_to_cm_resource(ai_model_config)
		
		# Register with configuration management system
		if self._config_manager:
			await self._config_manager.create_resource(cm_resource)
		
		self._log_ai_model_registered(ai_model_config.id, ai_model_config.name)
		
		return ai_model_config.id
	
	def _log_ai_model_registered(self, model_id: str, name: str) -> None:
		"""Log AI model registration"""
		logger.info(f"AI model configuration registered: {model_id} ({name})")
	
	async def _convert_to_cm_resource(self, ai_model_config: AIModelConfiguration) -> CMResource:
		"""Convert AI model configuration to universal configuration resource"""
		
		# Convert AI model config to configuration DSL
		config_dsl = ai_model_config.to_configuration_dsl()
		
		# Create CM resource
		cm_resource = CMResource(
			tenant_id=self.tenant_id,
			name=ai_model_config.name,
			display_name=ai_model_config.display_name,
			description=ai_model_config.description,
			resource_type=ResourceType.AI_MODEL,
			cloud_provider=ai_model_config.cloud_provider,
			configuration=config_dsl,
			created_by=ai_model_config.created_by,
			tags={
				**ai_model_config.tags,
				"ai_model_type": ai_model_config.model_type.value,
				"framework": ai_model_config.framework.value,
				"provider": ai_model_config.provider.value
			}
		)
		
		return cm_resource
	
	async def deploy_ai_model_configuration(
		self,
		model_config_id: str,
		deployment_options: Optional[Dict[str, Any]] = None
	) -> str:
		"""
		Deploy AI model configuration through GitOps workflows.
		
		This method deploys the AI model configuration using the established
		GitOps deployment orchestration system.
		"""
		assert model_config_id, "Model configuration ID is required"
		
		if model_config_id not in self.ai_model_configs:
			raise ValueError(f"AI model configuration not found: {model_config_id}")
		
		ai_model_config = self.ai_model_configs[model_config_id]
		options = deployment_options or {}
		
		# Update model state
		ai_model_config.state = AIModelState.LOADING
		
		try:
			# Integrate with NLP service if configured
			if ai_model_config.nlp_service_integration and self._nlp_service:
				await self._integrate_with_nlp_service(ai_model_config)
			
			# Deploy through GitOps if configured
			deployment_id = None
			if self._gitops_manager:
				deployment_id = await self._deploy_through_gitops(ai_model_config, options)
			
			# Update state and deployment info
			ai_model_config.state = AIModelState.READY
			ai_model_config.deployed_at = datetime.utcnow()
			
			self._log_ai_model_deployed(model_config_id, deployment_id)
			
			return deployment_id or f"direct_{model_config_id}"
			
		except Exception as e:
			ai_model_config.state = AIModelState.FAILED
			self._log_ai_model_deployment_error(model_config_id, str(e))
			raise
	
	def _log_ai_model_deployed(self, model_id: str, deployment_id: Optional[str]) -> None:
		"""Log AI model deployment"""
		logger.info(f"AI model deployed: {model_id} (deployment: {deployment_id})")
	
	def _log_ai_model_deployment_error(self, model_id: str, error: str) -> None:
		"""Log AI model deployment error"""
		logger.error(f"AI model deployment failed: {model_id} - {error}")
	
	async def _integrate_with_nlp_service(self, ai_model_config: AIModelConfiguration) -> None:
		"""Integrate AI model with common/nlpc NLP service"""
		assert self._nlp_service, "NLP service integration not available"
		
		# Create model registration data for NLP service
		model_registration_data = {
			"name": ai_model_config.display_name or ai_model_config.name,
			"model_key": ai_model_config.provider_model_name,
			"provider": self._map_to_nlp_provider(ai_model_config.provider),
			"provider_model_name": ai_model_config.provider_model_name,
			"supported_tasks": [self._map_to_nlp_task(task) for task in ai_model_config.supported_tasks],
			"supported_languages": ai_model_config.supported_languages,
			"model_config": ai_model_config.model_parameters,
			"runtime_config": ai_model_config.runtime_config
		}
		
		# Register model with NLP service
		try:
			# This would call the actual NLP service integration
			# For now, we'll simulate the registration
			await self._register_model_with_nlp_service(model_registration_data)
			
			self._log_nlp_integration_success(ai_model_config.id)
			
		except Exception as e:
			self._log_nlp_integration_error(ai_model_config.id, str(e))
			raise
	
	def _map_to_nlp_provider(self, provider: ModelProvider) -> str:
		"""Map configuration provider to NLP service provider"""
		provider_mapping = {
			ModelProvider.OLLAMA: "ollama",
			ModelProvider.TRANSFORMERS: "transformers", 
			ModelProvider.SPACY: "spacy",
			ModelProvider.OPENAI: "openai",
			ModelProvider.CUSTOM: "custom",
			ModelProvider.LOCAL: "local"
		}
		return provider_mapping.get(provider, "custom")
	
	def _map_to_nlp_task(self, task: AIModelType) -> str:
		"""Map configuration task type to NLP service task type"""
		task_mapping = {
			AIModelType.TEXT_GENERATION: "text_generation",
			AIModelType.SENTIMENT_ANALYSIS: "sentiment_analysis",
			AIModelType.NAMED_ENTITY_RECOGNITION: "named_entity_recognition",
			AIModelType.TEXT_CLASSIFICATION: "text_classification",
			AIModelType.QUESTION_ANSWERING: "question_answering",
			AIModelType.TEXT_SUMMARIZATION: "text_summarization",
			AIModelType.TRANSLATION: "translation",
			AIModelType.EMBEDDING: "embedding"
		}
		return task_mapping.get(task, "custom")
	
	async def _register_model_with_nlp_service(self, registration_data: Dict[str, Any]) -> None:
		"""Register model with NLP service"""
		# This would integrate with the actual NLP service
		# For now, we'll log the registration attempt
		self._log_nlp_service_registration(registration_data["name"])
	
	def _log_nlp_service_registration(self, model_name: str) -> None:
		"""Log NLP service registration"""
		logger.info(f"Registering model with NLP service: {model_name}")
	
	def _log_nlp_integration_success(self, model_id: str) -> None:
		"""Log successful NLP integration"""
		logger.info(f"AI model integrated with NLP service: {model_id}")
	
	def _log_nlp_integration_error(self, model_id: str, error: str) -> None:
		"""Log NLP integration error"""
		logger.error(f"NLP service integration failed for model {model_id}: {error}")
	
	async def _deploy_through_gitops(
		self,
		ai_model_config: AIModelConfiguration,
		options: Dict[str, Any]
	) -> str:
		"""Deploy AI model configuration through GitOps workflows"""
		
		# Generate deployment manifest
		deployment_manifest = await self._generate_deployment_manifest(ai_model_config, options)
		
		# Create deployment plan
		deployment_plan = {
			"resource_id": ai_model_config.id,
			"resource_type": "ai_model",
			"deployment_strategy": options.get("strategy", "rolling_update"),
			"target_environment": ai_model_config.deployment_target,
			"manifest": deployment_manifest,
			"health_checks": ai_model_config.health_check_config,
			"rollback_enabled": options.get("enable_rollback", True)
		}
		
		# Execute through GitOps manager
		if self._gitops_manager:
			deployment_id = await self._gitops_manager.create_deployment_plan(deployment_plan)
			return deployment_id
		
		return f"manifest_{ai_model_config.id}"
	
	async def _generate_deployment_manifest(
		self,
		ai_model_config: AIModelConfiguration,
		options: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Generate Kubernetes-style deployment manifest for AI model"""
		
		manifest = {
			"apiVersion": "apg.datacraft.co.ke/v1",
			"kind": "AIModelDeployment",
			"metadata": {
				"name": ai_model_config.name,
				"namespace": options.get("namespace", "default"),
				"labels": {
					"app": ai_model_config.name,
					"framework": ai_model_config.framework.value,
					"model-type": ai_model_config.model_type.value,
					"provider": ai_model_config.provider.value
				},
				"annotations": {
					"apg.datacraft.co.ke/tenant-id": self.tenant_id,
					"apg.datacraft.co.ke/config-id": ai_model_config.id,
					"apg.datacraft.co.ke/deployment-target": ai_model_config.deployment_target
				}
			},
			"spec": {
				"model": {
					"framework": ai_model_config.framework.value,
					"type": ai_model_config.model_type.value,
					"provider": ai_model_config.provider.value,
					"providerModelName": ai_model_config.provider_model_name,
					"modelPath": ai_model_config.model_path,
					"configuration": ai_model_config.model_parameters
				},
				"runtime": ai_model_config.runtime_config,
				"resources": ai_model_config.resource_requirements,
				"scaling": ai_model_config.scaling_config,
				"integration": {
					"nlpService": ai_model_config.nlp_service_integration,
					"supportedTasks": [task.value for task in ai_model_config.supported_tasks],
					"supportedLanguages": ai_model_config.supported_languages
				},
				"monitoring": {
					"enabled": ai_model_config.monitoring_enabled,
					"healthCheck": ai_model_config.health_check_config,
					"metrics": ai_model_config.performance_metrics
				}
			}
		}
		
		return manifest
	
	async def create_ml_pipeline_configuration(
		self,
		pipeline_config_data: Dict[str, Any]
	) -> str:
		"""Create ML pipeline configuration that orchestrates multiple AI models"""
		assert pipeline_config_data, "Pipeline configuration data is required"
		
		# Create ML pipeline configuration
		ml_pipeline_config = MLPipelineConfiguration(
			tenant_id=self.tenant_id,
			**pipeline_config_data
		)
		
		# Validate that referenced models exist
		for model_id in ml_pipeline_config.models:
			if model_id not in self.ai_model_configs:
				raise ValueError(f"Referenced AI model not found: {model_id}")
		
		# Store configuration
		self.ml_pipeline_configs[ml_pipeline_config.id] = ml_pipeline_config
		
		# Convert to universal configuration resource
		cm_resource = await self._convert_pipeline_to_cm_resource(ml_pipeline_config)
		
		# Register with configuration management system
		if self._config_manager:
			await self._config_manager.create_resource(cm_resource)
		
		self._log_pipeline_created(ml_pipeline_config.id, ml_pipeline_config.name)
		
		return ml_pipeline_config.id
	
	def _log_pipeline_created(self, pipeline_id: str, name: str) -> None:
		"""Log ML pipeline creation"""
		logger.info(f"ML pipeline configuration created: {pipeline_id} ({name})")
	
	async def _convert_pipeline_to_cm_resource(self, pipeline_config: MLPipelineConfiguration) -> CMResource:
		"""Convert ML pipeline configuration to CM resource"""
		
		# Create configuration DSL for pipeline
		config_dsl = ConfigurationDSL(
			kind="MLPipeline",
			metadata={
				"name": pipeline_config.name,
				"version": pipeline_config.version,
				"execution_mode": pipeline_config.execution_mode
			},
			spec={
				"models": pipeline_config.models,
				"preprocessing": pipeline_config.preprocessing_steps,
				"postprocessing": pipeline_config.postprocessing_steps,
				"input_schema": pipeline_config.input_schema,
				"output_schema": pipeline_config.output_schema,
				"configuration": pipeline_config.pipeline_config,
				"execution": {
					"mode": pipeline_config.execution_mode,
					"parallelism": pipeline_config.parallelism,
					"timeout": pipeline_config.timeout_seconds
				},
				"resources": pipeline_config.resource_requirements,
				"deployment": {
					"target": pipeline_config.deployment_target,
					"cloud_provider": pipeline_config.cloud_provider
				}
			}
		)
		
		# Create CM resource
		cm_resource = CMResource(
			tenant_id=self.tenant_id,
			name=pipeline_config.name,
			description=pipeline_config.description,
			resource_type=ResourceType.ML_PIPELINE,
			cloud_provider=pipeline_config.cloud_provider,
			configuration=config_dsl,
			created_by=pipeline_config.created_by,
			tags={
				**pipeline_config.tags,
				"execution_mode": pipeline_config.execution_mode,
				"model_count": str(len(pipeline_config.models))
			}
		)
		
		return cm_resource
	
	async def create_nlp_service_configuration(
		self,
		service_config_data: Dict[str, Any]
	) -> str:
		"""Create NLP service configuration for common/nlpc integration"""
		assert service_config_data, "Service configuration data is required"
		
		# Create NLP service configuration
		nlp_service_config = NLPServiceConfiguration(
			tenant_id=self.tenant_id,
			**service_config_data
		)
		
		# Validate that referenced models exist
		for model_id in nlp_service_config.registered_models:
			if model_id not in self.ai_model_configs:
				raise ValueError(f"Referenced AI model not found: {model_id}")
		
		# Store configuration
		self.nlp_service_configs[nlp_service_config.id] = nlp_service_config
		
		# Convert to universal configuration resource
		cm_resource = await self._convert_nlp_service_to_cm_resource(nlp_service_config)
		
		# Register with configuration management system
		if self._config_manager:
			await self._config_manager.create_resource(cm_resource)
		
		self._log_nlp_service_created(nlp_service_config.id, nlp_service_config.name)
		
		return nlp_service_config.id
	
	def _log_nlp_service_created(self, service_id: str, name: str) -> None:
		"""Log NLP service configuration creation"""
		logger.info(f"NLP service configuration created: {service_id} ({name})")
	
	async def _convert_nlp_service_to_cm_resource(self, service_config: NLPServiceConfiguration) -> CMResource:
		"""Convert NLP service configuration to CM resource"""
		
		# Use the built-in configuration DSL conversion
		config_dsl = service_config.to_configuration_dsl()
		
		# Create CM resource
		cm_resource = CMResource(
			tenant_id=self.tenant_id,
			name=service_config.name,
			description=service_config.description,
			resource_type=ResourceType.NLP_SERVICE,
			cloud_provider=service_config.cloud_provider,
			configuration=config_dsl,
			created_by=service_config.created_by,
			tags={
				**service_config.tags,
				"service_type": "nlp_processing",
				"model_count": str(len(service_config.registered_models))
			}
		)
		
		return cm_resource
	
	async def get_ai_model_configuration(self, model_config_id: str) -> AIModelConfiguration:
		"""Get AI model configuration by ID"""
		assert model_config_id, "Model configuration ID is required"
		
		if model_config_id not in self.ai_model_configs:
			raise ValueError(f"AI model configuration not found: {model_config_id}")
		
		return self.ai_model_configs[model_config_id]
	
	async def list_ai_model_configurations(
		self,
		filters: Optional[Dict[str, Any]] = None
	) -> List[AIModelConfiguration]:
		"""List AI model configurations with optional filters"""
		
		configs = list(self.ai_model_configs.values())
		
		if not filters:
			return configs
		
		# Apply filters
		filtered_configs = []
		for config in configs:
			match = True
			
			if "framework" in filters and config.framework != filters["framework"]:
				match = False
			if "model_type" in filters and config.model_type != filters["model_type"]:
				match = False
			if "provider" in filters and config.provider != filters["provider"]:
				match = False
			if "state" in filters and config.state != filters["state"]:
				match = False
			
			if match:
				filtered_configs.append(config)
		
		return filtered_configs
	
	async def update_ai_model_state(self, model_config_id: str, new_state: AIModelState) -> None:
		"""Update AI model configuration state"""
		assert model_config_id, "Model configuration ID is required"
		assert new_state, "New state is required"
		
		if model_config_id not in self.ai_model_configs:
			raise ValueError(f"AI model configuration not found: {model_config_id}")
		
		config = self.ai_model_configs[model_config_id]
		old_state = config.state
		config.state = new_state
		config.updated_at = datetime.utcnow()
		
		self._log_state_change(model_config_id, old_state, new_state)
	
	def _log_state_change(self, model_id: str, old_state: AIModelState, new_state: AIModelState) -> None:
		"""Log model state change"""
		logger.info(f"AI model state changed: {model_id} ({old_state} -> {new_state})")
	
	async def get_configuration_summary(self) -> Dict[str, Any]:
		"""Get summary of all AI model configurations"""
		
		total_models = len(self.ai_model_configs)
		total_pipelines = len(self.ml_pipeline_configs)
		total_services = len(self.nlp_service_configs)
		
		# Model states distribution
		state_counts = {}
		for config in self.ai_model_configs.values():
			state = config.state.value
			state_counts[state] = state_counts.get(state, 0) + 1
		
		# Framework distribution
		framework_counts = {}
		for config in self.ai_model_configs.values():
			framework = config.framework.value
			framework_counts[framework] = framework_counts.get(framework, 0) + 1
		
		# Provider distribution
		provider_counts = {}
		for config in self.ai_model_configs.values():
			provider = config.provider.value
			provider_counts[provider] = provider_counts.get(provider, 0) + 1
		
		return {
			"timestamp": datetime.utcnow().isoformat(),
			"totals": {
				"ai_models": total_models,
				"ml_pipelines": total_pipelines,
				"nlp_services": total_services
			},
			"ai_model_states": state_counts,
			"frameworks": framework_counts,
			"providers": provider_counts,
			"integration_status": {
				"config_manager_available": self._config_manager is not None,
				"nlp_service_available": self._nlp_service is not None,
				"gitops_manager_available": self._gitops_manager is not None
			}
		}
	
	# Dependency injection methods for APG integration
	
	def set_config_manager(self, config_manager) -> None:
		"""Inject configuration manager dependency"""
		self._config_manager = config_manager
		logger.info("Configuration manager dependency injected")
	
	def set_nlp_service(self, nlp_service) -> None:
		"""Inject NLP service dependency"""
		self._nlp_service = nlp_service
		logger.info("NLP service dependency injected")
	
	def set_gitops_manager(self, gitops_manager) -> None:
		"""Inject GitOps manager dependency"""
		self._gitops_manager = gitops_manager
		logger.info("GitOps manager dependency injected")


async def get_ai_model_adapter(tenant_id: str) -> AIModelConfigurationAdapter:
	"""Get AI model configuration adapter instance"""
	
	adapter = AIModelConfigurationAdapter(tenant_id)
	
	# Initialize with sample configurations for testing
	await adapter._initialize_sample_configurations()
	
	return adapter


# Helper methods for the AIModelConfigurationAdapter
async def _initialize_sample_configurations(self):
	"""Initialize sample AI model configurations"""
	
	# Sample Ollama model configuration
	ollama_model_config = {
		"name": "llama-3.2-chat",
		"display_name": "Llama 3.2 Chat Model",
		"description": "Advanced language model for conversational AI",
		"framework": AIModelFramework.OLLAMA,
		"model_type": AIModelType.TEXT_GENERATION,
		"provider": ModelProvider.OLLAMA,
		"provider_model_name": "llama3.2:latest",
		"cloud_provider": CloudProvider.ON_PREMISES,
		"deployment_target": "development",
		"model_parameters": {
			"temperature": 0.7,
			"max_tokens": 2048,
			"top_p": 0.9
		},
		"runtime_config": {
			"batch_size": 1,
			"max_concurrent_requests": 10
		},
		"resource_requirements": {
			"cpu_cores": 4,
			"memory_gb": 8,
			"gpu_required": False
		},
		"supported_tasks": [AIModelType.TEXT_GENERATION, AIModelType.QUESTION_ANSWERING],
		"supported_languages": ["en", "es", "fr"],
		"tags": {
			"environment": "development",
			"use_case": "chat"
		},
		"created_by": "system@datacraft.co.ke"
	}
	
	await self.register_ai_model_configuration(ollama_model_config)
	
	# Sample Transformers model configuration
	transformers_model_config = {
		"name": "bert-sentiment-analyzer",
		"display_name": "BERT Sentiment Analyzer",
		"description": "Fine-tuned BERT model for sentiment analysis",
		"framework": AIModelFramework.TRANSFORMERS,
		"model_type": AIModelType.SENTIMENT_ANALYSIS,
		"provider": ModelProvider.TRANSFORMERS,
		"provider_model_name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
		"cloud_provider": CloudProvider.AWS,
		"deployment_target": "production",
		"model_parameters": {
			"max_length": 512,
			"truncation": True,
			"padding": True
		},
		"runtime_config": {
			"batch_size": 32,
			"use_gpu": True
		},
		"resource_requirements": {
			"cpu_cores": 2,
			"memory_gb": 4,
			"gpu_memory_gb": 6,
			"gpu_required": True
		},
		"supported_tasks": [AIModelType.SENTIMENT_ANALYSIS, AIModelType.TEXT_CLASSIFICATION],
		"supported_languages": ["en"],
		"tags": {
			"environment": "production",
			"use_case": "sentiment_analysis"
		},
		"created_by": "ml_engineer@datacraft.co.ke"
	}
	
	await self.register_ai_model_configuration(transformers_model_config)


# Attach the method to the class
AIModelConfigurationAdapter._initialize_sample_configurations = _initialize_sample_configurations


__all__ = [
	"AIModelConfigurationAdapter",
	"get_ai_model_adapter"
]