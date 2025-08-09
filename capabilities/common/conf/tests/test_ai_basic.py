"""
Basic AI Model Adapter Tests - Without complex dependencies
Testing core functionality of AI model configuration management.

© 2025 Datacraft - www.datacraft.co.ke  
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import sys
import os

# Add the path to the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

try:
	from capabilities.common.conf.ai_model_adapter import AIModelConfigurationAdapter
	from capabilities.common.conf.models import (
		AIModelConfiguration, AIModelFramework, AIModelType, 
		AIModelState, ModelProvider, CloudProvider
	)
except ImportError:
	# If the full import fails, try relative imports
	import os
	sys.path.insert(0, os.path.dirname(__file__))
	sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
	
	# Mock the complex dependencies to test just the basic adapter
	from unittest.mock import Mock
	sys.modules['capabilities.common.conf.gitops_integration'] = Mock()
	sys.modules['capabilities.common.conf.automated_testing'] = Mock()
	sys.modules['capabilities.common.conf.deployment_orchestration'] = Mock()
	sys.modules['capabilities.common.conf.security_integration'] = Mock()
	
	from capabilities.common.conf.ai_model_adapter import AIModelConfigurationAdapter
	from capabilities.common.conf.models import (
		AIModelConfiguration, AIModelFramework, AIModelType, 
		AIModelState, ModelProvider, CloudProvider
	)


async def test_basic_ai_model_adapter():
	"""Test basic AI model adapter functionality"""
	print("Testing AI Model Configuration Adapter...")
	
	# Create adapter
	adapter = AIModelConfigurationAdapter(tenant_id="test_tenant")
	print("✓ Adapter initialized successfully")
	
	# Test basic configuration
	sample_config = {
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
	
	# Register model configuration
	model_id = await adapter.register_ai_model_configuration(sample_config)
	print(f"✓ Model registered successfully: {model_id}")
	
	# Verify configuration storage
	config = await adapter.get_ai_model_configuration(model_id)
	assert config.name == "test-llama-chat"
	assert config.framework == AIModelFramework.OLLAMA
	assert config.state == AIModelState.CONFIGURED
	print("✓ Configuration retrieved and verified")
	
	# Test configuration listing
	configs = await adapter.list_ai_model_configurations()
	assert len(configs) == 1
	assert configs[0].id == model_id
	print("✓ Configuration listing works")
	
	# Test filtering
	ollama_configs = await adapter.list_ai_model_configurations(
		filters={"framework": AIModelFramework.OLLAMA}
	)
	assert len(ollama_configs) == 1
	print("✓ Configuration filtering works")
	
	# Test state updates
	await adapter.update_ai_model_state(model_id, AIModelState.LOADING)
	updated_config = await adapter.get_ai_model_configuration(model_id)
	assert updated_config.state == AIModelState.LOADING
	print("✓ State updates work")
	
	# Test summary
	summary = await adapter.get_configuration_summary()
	assert summary["totals"]["ai_models"] == 1
	assert "loading" in summary["ai_model_states"]
	assert "ollama" in summary["frameworks"]
	print("✓ Configuration summary works")
	
	print("\n🎉 All basic tests passed!")
	return True


async def test_ai_model_creation():
	"""Test direct AI model configuration creation"""
	print("Testing direct AI model configuration creation...")
	
	# Create AI model configuration directly
	ai_model_config = AIModelConfiguration(
		tenant_id="test_tenant",
		name="direct-test-model",
		display_name="Direct Test Model",
		description="Direct test model creation",
		framework=AIModelFramework.TRANSFORMERS,
		model_type=AIModelType.SENTIMENT_ANALYSIS,
		provider=ModelProvider.TRANSFORMERS,
		provider_model_name="cardiffnlp/twitter-roberta-base-sentiment-latest",
		cloud_provider=CloudProvider.AWS,
		deployment_target="testing",
		model_parameters={
			"max_length": 512,
			"truncation": True,
			"padding": True
		},
		runtime_config={
			"batch_size": 16,
			"use_gpu": False
		},
		resource_requirements={
			"cpu_cores": 1,
			"memory_gb": 2,
			"gpu_required": False
		},
		supported_tasks=[AIModelType.SENTIMENT_ANALYSIS],
		supported_languages=["en"],
		tags={"environment": "testing", "use_case": "sentiment"},
		created_by="test@datacraft.co.ke"
	)
	
	print(f"✓ AI model configuration created: {ai_model_config.id}")
	print(f"  Name: {ai_model_config.name}")
	print(f"  Framework: {ai_model_config.framework}")
	print(f"  Type: {ai_model_config.model_type}")
	print(f"  State: {ai_model_config.state}")
	
	# Test configuration DSL conversion
	config_dsl = ai_model_config.to_configuration_dsl()
	print("✓ Configuration DSL conversion works")
	print(f"  Kind: {config_dsl.kind}")
	print(f"  Metadata keys: {list(config_dsl.metadata.keys())}")
	
	print("\n🎉 Direct model creation test passed!")
	return True


async def test_provider_mappings():
	"""Test provider and task mappings"""
	print("Testing provider and task mappings...")
	
	adapter = AIModelConfigurationAdapter(tenant_id="test_tenant")
	
	# Test provider mappings
	assert adapter._map_to_nlp_provider(ModelProvider.OLLAMA) == "ollama"
	assert adapter._map_to_nlp_provider(ModelProvider.TRANSFORMERS) == "transformers"
	assert adapter._map_to_nlp_provider(ModelProvider.SPACY) == "spacy"
	assert adapter._map_to_nlp_provider(ModelProvider.OPENAI) == "openai"
	print("✓ Provider mappings work")
	
	# Test task mappings
	assert adapter._map_to_nlp_task(AIModelType.TEXT_GENERATION) == "text_generation"
	assert adapter._map_to_nlp_task(AIModelType.SENTIMENT_ANALYSIS) == "sentiment_analysis"
	assert adapter._map_to_nlp_task(AIModelType.NAMED_ENTITY_RECOGNITION) == "named_entity_recognition"
	assert adapter._map_to_nlp_task(AIModelType.TEXT_CLASSIFICATION) == "text_classification"
	print("✓ Task mappings work")
	
	print("\n🎉 Mapping tests passed!")
	return True


async def main():
	"""Run all basic tests"""
	print("=" * 60)
	print("APG Configuration Management - AI Model Adapter Basic Tests")
	print("=" * 60)
	
	try:
		await test_basic_ai_model_adapter()
		await test_ai_model_creation()
		await test_provider_mappings()
		
		print("\n" + "=" * 60)
		print("🎉 ALL BASIC TESTS PASSED SUCCESSFULLY!")
		print("=" * 60)
		
	except Exception as e:
		print(f"\n❌ Test failed with error: {e}")
		import traceback
		traceback.print_exc()
		return False
	
	return True


if __name__ == "__main__":
	success = asyncio.run(main())
	sys.exit(0 if success else 1)