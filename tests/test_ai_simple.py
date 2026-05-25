"""
Simple AI Model Configuration Test
Testing core AI model configuration management without complex dependencies.

© 2025 Datacraft - www.datacraft.co.ke  
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Mock complex dependencies before importing
from unittest.mock import Mock
sys.modules['capabilities.common.conf.gitops_integration'] = Mock()
sys.modules['capabilities.common.conf.automated_testing'] = Mock() 
sys.modules['capabilities.common.conf.deployment_orchestration'] = Mock()
sys.modules['capabilities.common.conf.security_integration'] = Mock()

# Now import what we need
from capabilities.common.conf.models import (
	AIModelConfiguration, AIModelFramework, AIModelType, 
	AIModelState, ModelProvider, CloudProvider
)


def test_ai_model_configuration_creation():
	"""Test AI model configuration creation"""
	print("Testing AI Model Configuration creation...")
	
	# Test basic configuration creation
	config = AIModelConfiguration(
		tenant_id="test_tenant",
		name="test-llama-model",
		display_name="Test Llama Model",
		description="Test language model",
		framework=AIModelFramework.OLLAMA,
		model_type=AIModelType.TEXT_GENERATION,
		provider=ModelProvider.OLLAMA,
		provider_model_name="llama3.2:latest",
		cloud_provider=CloudProvider.ON_PREMISES,
		deployment_target="testing",
		model_parameters={
			"temperature": 0.7,
			"max_tokens": 2048
		},
		runtime_config={
			"batch_size": 1
		},
		resource_requirements={
			"cpu_cores": 2,
			"memory_gb": 4
		},
		supported_tasks=[AIModelType.TEXT_GENERATION],
		supported_languages=["en"],
		created_by="test@datacraft.co.ke"
	)
	
	print(f"✓ AI model configuration created successfully")
	print(f"  ID: {config.id}")
	print(f"  Name: {config.name}")
	print(f"  Framework: {config.framework}")
	print(f"  Type: {config.model_type}")
	print(f"  State: {config.state}")
	
	# Test configuration DSL conversion
	config_dsl = config.to_configuration_dsl()
	print(f"✓ Configuration DSL conversion successful")
	print(f"  Kind: {config_dsl.kind}")
	print(f"  Metadata keys: {list(config_dsl.metadata.keys())}")
	print(f"  Spec keys: {list(config_dsl.spec.keys())}")
	
	return config


def test_ai_model_enums():
	"""Test AI model enums work correctly"""
	print("\nTesting AI model enums...")
	
	# Test frameworks
	assert AIModelFramework.OLLAMA == "ollama"
	assert AIModelFramework.TRANSFORMERS == "transformers"
	assert AIModelFramework.SPACY == "spacy"
	print("✓ AI model frameworks work")
	
	# Test model types
	assert AIModelType.TEXT_GENERATION == "text_generation"
	assert AIModelType.SENTIMENT_ANALYSIS == "sentiment_analysis"
	assert AIModelType.NAMED_ENTITY_RECOGNITION == "named_entity_recognition"
	print("✓ AI model types work")
	
	# Test providers
	assert ModelProvider.OLLAMA == "ollama"
	assert ModelProvider.TRANSFORMERS == "transformers"
	assert ModelProvider.OPENAI == "openai"
	print("✓ Model providers work")
	
	# Test states
	assert AIModelState.CONFIGURED == "configured"
	assert AIModelState.LOADING == "loading"
	assert AIModelState.READY == "ready"
	assert AIModelState.FAILED == "failed"
	print("✓ AI model states work")


def test_different_ai_models():
	"""Test creating different types of AI models"""
	print("\nTesting different AI model configurations...")
	
	# Test Transformers sentiment model
	sentiment_config = AIModelConfiguration(
		tenant_id="test_tenant",
		name="bert-sentiment-analyzer",
		display_name="BERT Sentiment Analyzer",
		description="BERT model for sentiment analysis",
		framework=AIModelFramework.TRANSFORMERS,
		model_type=AIModelType.SENTIMENT_ANALYSIS,
		provider=ModelProvider.TRANSFORMERS,
		provider_model_name="cardiffnlp/twitter-roberta-base-sentiment-latest",
		cloud_provider=CloudProvider.AWS,
		deployment_target="production",
		model_parameters={
			"max_length": 512,
			"truncation": True,
			"padding": True
		},
		runtime_config={
			"batch_size": 32,
			"use_gpu": False
		},
		resource_requirements={
			"cpu_cores": 2,
			"memory_gb": 4,
			"gpu_required": False
		},
		supported_tasks=[AIModelType.SENTIMENT_ANALYSIS],
		supported_languages=["en"],
		created_by="test@datacraft.co.ke"
	)
	
	print(f"✓ BERT sentiment model created: {sentiment_config.id}")
	
	# Test spaCy NER model
	ner_config = AIModelConfiguration(
		tenant_id="test_tenant",
		name="spacy-ner-model",
		display_name="spaCy NER Model",
		description="spaCy model for named entity recognition",
		framework=AIModelFramework.SPACY,
		model_type=AIModelType.NAMED_ENTITY_RECOGNITION,
		provider=ModelProvider.SPACY,
		provider_model_name="en_core_web_lg",
		cloud_provider=CloudProvider.ON_PREMISES,
		deployment_target="development",
		model_parameters={
			"disable": ["parser", "tagger"],
			"enable": ["ner"]
		},
		runtime_config={
			"batch_size": 64
		},
		resource_requirements={
			"cpu_cores": 1,
			"memory_gb": 2
		},
		supported_tasks=[AIModelType.NAMED_ENTITY_RECOGNITION],
		supported_languages=["en"],
		created_by="test@datacraft.co.ke"
	)
	
	print(f"✓ spaCy NER model created: {ner_config.id}")
	
	return [sentiment_config, ner_config]


def main():
	"""Run all tests"""
	print("=" * 60)
	print("APG Configuration Management - AI Model Configuration Tests")
	print("=" * 60)
	
	try:
		# Test enum definitions
		test_ai_model_enums()
		
		# Test basic configuration creation
		config = test_ai_model_configuration_creation()
		
		# Test different model types
		configs = test_different_ai_models()
		
		print("\n" + "=" * 60)
		print("🎉 ALL AI MODEL CONFIGURATION TESTS PASSED!")
		print(f"✓ Created {len(configs) + 1} different AI model configurations")
		print("✓ All enums and data structures work correctly")
		print("✓ Configuration DSL conversion works")
		print("=" * 60)
		
		return True
		
	except Exception as e:
		print(f"\n❌ Test failed with error: {e}")
		import traceback
		traceback.print_exc()
		return False


if __name__ == "__main__":
	success = main()
	sys.exit(0 if success else 1)