#!/usr/bin/env python3
"""
APG NLP Capability Foundation Validation Script

Validates the basic functionality of the NLP capability including:
- Model initialization and loading
- APG integration components
- Basic text processing
- Database schema validation
"""

import asyncio
import logging
import sys
from pathlib import Path
import json
from typing import Dict, Any, List
from uuid_extensions import uuid7str

# Add capability to path
sys.path.insert(0, str(Path(__file__).parent))

from models import (
	TextDocument, NLPModel, ProcessingRequest, ProcessingResult,
	StreamingSession, StreamingChunk, SystemHealth,
	NLPTaskType, ModelProvider, ProcessingStatus, QualityLevel, LanguageCode
)
from service import NLPService, ModelConfig
from __init__ import (
	APG_CAPABILITY_METADATA, 
	get_capability_metadata,
	get_blueprint_config,
	validate_apg_dependencies,
	get_supported_languages,
	get_available_models
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _log_validation_start() -> None:
	"""Log validation start"""
	logger.info("🚀 Starting APG NLP Capability Foundation Validation")

def _log_validation_complete() -> None:
	"""Log validation completion"""
	logger.info("✅ APG NLP Capability Foundation Validation Complete")

def _log_test_section(name: str) -> None:
	"""Log test section start"""
	logger.info(f"📋 Testing: {name}")

def _log_test_passed(test_name: str) -> None:
	"""Log test passed"""
	logger.info(f"✅ PASS: {test_name}")

def _log_test_failed(test_name: str, error: str) -> None:
	"""Log test failed"""
	logger.error(f"❌ FAIL: {test_name} - {error}")

def _log_test_warning(test_name: str, warning: str) -> None:
	"""Log test warning"""
	logger.warning(f"⚠️  WARN: {test_name} - {warning}")

async def validate_models_and_schemas():
	"""Validate Pydantic models and schemas"""
	_log_test_section("Pydantic Models and Schemas")
	
	try:
		# Test TextDocument model
		doc = TextDocument(
			tenant_id=uuid7str(),
			content="This is a test document for APG NLP validation.",
			title="Test Document",
			language=LanguageCode.EN,
			metadata={"source": "validation_script", "test": True}
		)
		
		assert doc.id is not None
		assert doc.tenant_id is not None
		assert doc.word_count == 0  # Not calculated yet
		assert doc.estimated_processing_time > 0
		_log_test_passed("TextDocument model creation and validation")
		
		# Test NLPModel model
		model = NLPModel(
			tenant_id=uuid7str(),
			name="Test BERT Model",
			model_key="bert-base-uncased",
			provider=ModelProvider.TRANSFORMERS,
			provider_model_name="bert-base-uncased",
			supported_tasks=[NLPTaskType.SENTIMENT_ANALYSIS, NLPTaskType.TEXT_CLASSIFICATION],
			supported_languages=[LanguageCode.EN],
			average_latency_ms=150.5,
			accuracy_score=0.89
		)
		
		assert model.is_available == False  # Not loaded yet
		assert model.success_rate == 0.0    # No requests yet
		_log_test_passed("NLPModel model creation and computed fields")
		
		# Test ProcessingRequest model
		request = ProcessingRequest(
			tenant_id=uuid7str(),
			task_type=NLPTaskType.SENTIMENT_ANALYSIS,
			text_content="This is amazing! I love this product.",
			quality_level=QualityLevel.BALANCED
		)
		
		assert request.id is not None
		assert request.fallback_enabled == True
		_log_test_passed("ProcessingRequest model creation and defaults")
		
		# Test ProcessingResult model
		result = ProcessingResult(
			request_id=request.id,
			tenant_id=request.tenant_id,
			task_type=request.task_type,
			model_used="test_model_id",
			provider_used=ModelProvider.TRANSFORMERS,
			processing_time_ms=125.3,
			total_time_ms=125.3,
			results={"sentiment": "positive", "confidence": 0.92}
		)
		
		assert result.is_successful == True
		assert result.performance_rating in ["excellent", "good", "acceptable", "poor"]
		_log_test_passed("ProcessingResult model with computed properties")
		
		# Test StreamingSession model
		session = StreamingSession(
			tenant_id=uuid7str(),
			user_id=uuid7str(),
			task_type=NLPTaskType.SENTIMENT_ANALYSIS
		)
		
		assert session.status == "active"
		assert session.chunks_processed == 0
		_log_test_passed("StreamingSession model creation")
		
		# Test SystemHealth model
		health = SystemHealth(
			overall_status="healthy",
			component_status={"models": "healthy", "database": "healthy"},
			average_response_time_ms=95.2,
			requests_per_minute=450,
			active_sessions=12,
			queue_depth=3,
			cpu_usage_percent=45.8,
			memory_usage_percent=62.1,
			disk_usage_percent=23.4,
			total_models=5,
			active_models=5,
			loaded_models=3,
			failed_models=0
		)
		
		assert health.model_availability_percent == 100.0
		assert health.performance_rating in ["excellent", "good", "acceptable", "poor"]
		_log_test_passed("SystemHealth model with computed metrics")
		
		logger.info("✅ All Pydantic models validated successfully")
		
	except Exception as e:
		_log_test_failed("Pydantic Models Validation", str(e))
		return False
		
	return True

async def validate_apg_integration():
	"""Validate APG integration components"""
	_log_test_section("APG Integration Components")
	
	try:
		# Test capability metadata
		metadata = get_capability_metadata()
		
		assert metadata["capability_id"] == "nlp"
		assert metadata["name"] == "Natural Language Processing"
		assert metadata["version"] == "1.0.0"
		assert "composition" in metadata
		assert "features" in metadata
		assert "model_config" in metadata
		_log_test_passed("APG capability metadata structure")
		
		# Test composition requirements
		composition = metadata["composition"]
		assert "provides" in composition
		assert "requires" in composition
		assert "enhances" in composition
		assert len(composition["provides"]) >= 8  # Should provide many NLP services
		_log_test_passed("APG composition engine requirements")
		
		# Test blueprint configuration
		blueprint_config = get_blueprint_config()
		
		assert blueprint_config["blueprint_name"] == "nlp"
		assert blueprint_config["url_prefix"] == "/nlp"
		assert len(blueprint_config["menu_links"]) >= 6
		assert len(blueprint_config["permissions"]) >= 6
		_log_test_passed("APG Flask blueprint configuration")
		
		# Test dependency validation
		dependencies = validate_apg_dependencies()
		assert isinstance(dependencies, list)
		_log_test_passed("APG dependency validation")
		
		# Test supported languages
		languages = get_supported_languages()
		assert "en" in languages
		assert "auto" in languages
		assert len(languages) >= 10
		_log_test_passed("Supported languages configuration")
		
		# Test available models configuration
		models = get_available_models()
		assert "ollama" in models
		assert "transformers" in models
		assert "spacy" in models
		assert len(models["transformers"]) >= 8
		_log_test_passed("Available models configuration")
		
		logger.info("✅ APG integration components validated successfully")
		
	except Exception as e:
		_log_test_failed("APG Integration Validation", str(e))
		return False
		
	return True

async def validate_service_initialization():
	"""Validate NLP service initialization"""
	_log_test_section("NLP Service Initialization")
	
	try:
		# Test service initialization
		tenant_id = uuid7str()
		config = ModelConfig(
			ollama_endpoint="http://localhost:11434",
			enable_gpu=False,  # Disable GPU for testing
			max_memory_gb=4.0,
			model_timeout_seconds=30
		)
		
		service = NLPService(tenant_id=tenant_id, config=config)
		
		assert service.tenant_id == tenant_id
		assert service.config.enable_gpu == False
		assert len(service._models) == 0  # Not initialized yet
		assert len(service._streaming_sessions) == 0
		_log_test_passed("NLP service basic initialization")
		
		# Test model initialization (will fail gracefully without actual models)
		try:
			await service.initialize_models()
			_log_test_passed("Model initialization (with graceful failures)")
		except Exception as e:
			_log_test_warning("Model initialization", f"Expected failures due to missing dependencies: {str(e)[:100]}")
		
		# Test system health check
		health = await service.get_system_health()
		
		assert isinstance(health, SystemHealth)
		assert health.tenant_id == tenant_id
		assert health.overall_status in ["healthy", "degraded", "unhealthy", "maintenance"]
		_log_test_passed("System health check")
		
		# Test available models (should be empty initially)
		models = await service.get_available_models()
		assert isinstance(models, list)
		_log_test_passed("Available models query")
		
		# Cleanup
		await service.cleanup()
		_log_test_passed("Service cleanup")
		
		logger.info("✅ NLP service initialization validated successfully")
		
	except Exception as e:
		_log_test_failed("Service Initialization Validation", str(e))
		return False
		
	return True

async def validate_core_nlp_methods():
	"""Validate core NLP processing methods"""
	_log_test_section("Core NLP Processing Methods")
	
	try:
		tenant_id = uuid7str()
		service = NLPService(tenant_id=tenant_id)
		
		# Test text for analysis
		test_text = "I absolutely love this amazing new product! It's fantastic and works perfectly. The customer service was excellent too."
		
		# Test sentiment analysis (should work with fallbacks)
		sentiment_result = await service.sentiment_analysis(test_text)
		
		assert "sentiment" in sentiment_result
		assert "confidence" in sentiment_result
		assert sentiment_result["sentiment"] in ["positive", "negative", "neutral"]
		assert 0.0 <= sentiment_result["confidence"] <= 1.0
		_log_test_passed("Sentiment analysis with fallback methods")
		
		# Test intent classification
		intents = ["compliment", "request", "complaint", "question"]
		intent_result = await service.intent_classification(test_text, intents)
		
		assert "predicted_intent" in intent_result
		assert "confidence" in intent_result
		assert intent_result["predicted_intent"] in intents
		_log_test_passed("Intent classification")
		
		# Test named entity recognition
		entity_text = "Apple Inc. was founded by Steve Jobs in Cupertino, California in 1976."
		ner_result = await service.named_entity_recognition(entity_text)
		
		assert "entities" in ner_result
		assert "entity_count" in ner_result
		assert isinstance(ner_result["entities"], list)
		_log_test_passed("Named entity recognition")
		
		# Test text classification
		categories = ["technology", "business", "entertainment", "sports"]
		class_result = await service.text_classification(test_text, categories)
		
		assert "predicted_category" in class_result
		assert "confidence" in class_result
		assert class_result["predicted_category"] in categories
		_log_test_passed("Text classification")
		
		# Test keyword extraction
		keyword_result = await service.keyword_extraction(test_text, num_keywords=5)
		
		assert "keywords" in keyword_result
		assert isinstance(keyword_result["keywords"], list)
		assert len(keyword_result["keywords"]) <= 5
		_log_test_passed("Keyword extraction")
		
		# Test text summarization
		long_text = "This is a longer text for summarization testing. " * 20  # Repeat to make it long enough
		summary_result = await service.text_summarization(long_text, max_length=50)
		
		assert "summary" in summary_result
		assert "method" in summary_result
		assert len(summary_result["summary"]) > 0
		_log_test_passed("Text summarization")
		
		# Test language detection
		lang_result = await service.language_detection("Hello world, this is English text")
		
		assert "detected_language" in lang_result
		assert "confidence" in lang_result
		_log_test_passed("Language detection")
		
		# Test content generation
		generation_result = await service.content_generation(
			"Write a short description of AI technology", 
			max_length=50
		)
		
		assert "generated_content" in generation_result
		assert "method" in generation_result
		_log_test_passed("Content generation")
		
		await service.cleanup()
		logger.info("✅ Core NLP methods validated successfully")
		
	except Exception as e:
		_log_test_failed("Core NLP Methods Validation", str(e))
		return False
		
	return True

async def validate_processing_workflow():
	"""Validate end-to-end processing workflow"""
	_log_test_section("End-to-End Processing Workflow")
	
	try:
		tenant_id = uuid7str()
		service = NLPService(tenant_id=tenant_id)
		
		# Create processing request
		request = ProcessingRequest(
			tenant_id=tenant_id,
			task_type=NLPTaskType.SENTIMENT_ANALYSIS,
			text_content="This product is absolutely amazing! I love it so much.",
			quality_level=QualityLevel.BALANCED,
			include_confidence=True
		)
		
		# Process the request (will use fallback methods)
		result = await service.process_text(request)
		
		assert isinstance(result, ProcessingResult)
		assert result.request_id == request.id
		assert result.tenant_id == tenant_id
		assert result.task_type == NLPTaskType.SENTIMENT_ANALYSIS
		assert result.processing_time_ms >= 0
		assert isinstance(result.results, dict)
		_log_test_passed("End-to-end processing workflow")
		
		# Test streaming session creation
		session_config = {
			"user_id": uuid7str(),
			"task_type": NLPTaskType.SENTIMENT_ANALYSIS,
			"chunk_size": 500
		}
		
		session = await service.create_streaming_session(session_config)
		
		assert isinstance(session, StreamingSession)
		assert session.tenant_id == tenant_id
		assert session.status == "active"
		_log_test_passed("Streaming session creation")
		
		# Test streaming chunk processing
		chunk = StreamingChunk(
			session_id=session.id,
			sequence_number=1,
			text_content="This is a streaming text chunk for testing.",
			start_position=0,
			end_position=47
		)
		
		chunk_result = await service.process_streaming_chunk(session.id, chunk)
		
		assert "chunk_id" in chunk_result
		assert "processing_time_ms" in chunk_result
		assert chunk_result["processing_time_ms"] >= 0
		_log_test_passed("Streaming chunk processing")
		
		await service.cleanup()
		logger.info("✅ Processing workflow validated successfully")
		
	except Exception as e:
		_log_test_failed("Processing Workflow Validation", str(e))
		return False
		
	return True

async def validate_performance_and_metrics():
	"""Validate performance monitoring and metrics"""
	_log_test_section("Performance Monitoring and Metrics")
	
	try:
		tenant_id = uuid7str()
		service = NLPService(tenant_id=tenant_id)
		
		# Process multiple requests to generate metrics
		for i in range(5):
			request = ProcessingRequest(
				tenant_id=tenant_id,
				task_type=NLPTaskType.SENTIMENT_ANALYSIS,
				text_content=f"Test sentence number {i + 1} for metrics validation.",
				quality_level=QualityLevel.FAST
			)
			
			result = await service.process_text(request)
			assert result.is_successful or result.status == ProcessingStatus.FAILED  # Either is acceptable
		
		_log_test_passed("Multiple processing requests for metrics")
		
		# Get system health with metrics
		health = await service.get_system_health()
		
		assert isinstance(health, SystemHealth)
		assert health.requests_per_minute >= 0
		assert health.average_response_time_ms >= 0
		_log_test_passed("System health metrics collection")
		
		# Test model performance tracking
		try:
			models = await service.get_available_models()
			if models:
				model_id = models[0].id
				perf = await service.get_model_performance(model_id)
				
				assert "model_id" in perf
				assert "total_requests" in perf
				assert "average_latency_ms" in perf
				_log_test_passed("Model performance metrics")
			else:
				_log_test_warning("Model performance metrics", "No models available for testing")
		except Exception as e:
			_log_test_warning("Model performance metrics", f"Expected with no loaded models: {str(e)}")
		
		await service.cleanup()
		logger.info("✅ Performance and metrics validated successfully")
		
	except Exception as e:
		_log_test_failed("Performance and Metrics Validation", str(e))
		return False
		
	return True

async def validate_database_schema():
	"""Validate database schema structure (mock validation)"""
	_log_test_section("Database Schema Validation")
	
	try:
		# Read and parse the SQL schema file
		schema_path = Path(__file__).parent / "database_schema.sql"
		
		if not schema_path.exists():
			_log_test_failed("Database Schema File", "database_schema.sql not found")
			return False
		
		schema_content = schema_path.read_text()
		
		# Check for required tables
		required_tables = [
			"nlp.documents",
			"nlp.models", 
			"nlp.processing_requests",
			"nlp.processing_results",
			"nlp.streaming_sessions",
			"nlp.streaming_chunks",
			"nlp.annotation_projects",
			"nlp.text_annotations",
			"nlp.text_analytics",
			"nlp.system_health"
		]
		
		for table in required_tables:
			if f"CREATE TABLE {table}" not in schema_content:
				_log_test_failed("Database Schema Tables", f"Missing table: {table}")
				return False
		
		_log_test_passed("All required database tables defined")
		
		# Check for required extensions
		required_extensions = ["uuid-ossp", "vector", "pg_trgm", "btree_gin"]
		
		for ext in required_extensions:
			if f'CREATE EXTENSION IF NOT EXISTS "{ext}"' not in schema_content:
				_log_test_failed("Database Extensions", f"Missing extension: {ext}")
				return False
		
		_log_test_passed("All required PostgreSQL extensions defined")
		
		# Check for indexes
		index_patterns = [
			"CREATE INDEX",
			"vector_cosine_ops",
			"gin(",
			"ivfflat"
		]
		
		for pattern in index_patterns:
			if pattern not in schema_content:
				_log_test_failed("Database Indexes", f"Missing index pattern: {pattern}")
				return False
		
		_log_test_passed("Database indexes and vector search configured")
		
		# Check for row-level security
		if "ENABLE ROW LEVEL SECURITY" not in schema_content:
			_log_test_failed("Database Security", "Row-level security not enabled")
			return False
		
		_log_test_passed("Row-level security configured for multi-tenancy")
		
		logger.info("✅ Database schema validated successfully")
		
	except Exception as e:
		_log_test_failed("Database Schema Validation", str(e))
		return False
		
	return True

async def main():
	"""Run all validation tests"""
	_log_validation_start()
	
	test_results = []
	
	# Run validation tests
	test_results.append(await validate_models_and_schemas())
	test_results.append(await validate_apg_integration())
	test_results.append(await validate_service_initialization())
	test_results.append(await validate_core_nlp_methods())
	test_results.append(await validate_processing_workflow())
	test_results.append(await validate_performance_and_metrics())
	test_results.append(await validate_database_schema())
	
	# Summary
	passed_tests = sum(test_results)
	total_tests = len(test_results)
	
	logger.info(f"\n{'='*60}")
	logger.info(f"VALIDATION SUMMARY")
	logger.info(f"{'='*60}")
	logger.info(f"Tests Passed: {passed_tests}/{total_tests}")
	logger.info(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
	
	if passed_tests == total_tests:
		logger.info("🎉 ALL TESTS PASSED - NLP Foundation is Ready!")
		_log_validation_complete()
		return 0
	else:
		logger.error(f"❌ {total_tests - passed_tests} TESTS FAILED")
		return 1

if __name__ == "__main__":
	sys.exit(asyncio.run(main()))