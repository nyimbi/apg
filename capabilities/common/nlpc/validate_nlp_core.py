#!/usr/bin/env python3
"""
APG NLP Capability Core Validation Script

Validates the core APG integration and model structures without requiring
heavy ML dependencies like PyTorch, Transformers, etc.
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
	logger.info("🚀 Starting APG NLP Core Validation")

def _log_validation_complete() -> None:
	"""Log validation completion"""
	logger.info("✅ APG NLP Core Validation Complete")

def _log_test_section(name: str) -> None:
	"""Log test section start"""
	logger.info(f"📋 Testing: {name}")

def _log_test_passed(test_name: str) -> None:
	"""Log test passed"""
	logger.info(f"✅ PASS: {test_name}")

def _log_test_failed(test_name: str, error: str) -> None:
	"""Log test failed"""
	logger.error(f"❌ FAIL: {test_name} - {error}")

async def validate_models_and_schemas():
	"""Validate Pydantic models and schemas"""
	_log_test_section("Pydantic Models and Schemas")
	
	try:
		# Test TextDocument model
		doc = TextDocument(
			tenant_id=uuid7str(),
			content="This is a test document for APG NLP validation. It contains enough text to test various NLP capabilities including sentiment analysis, entity recognition, and text classification. The document should be long enough to generate meaningful processing metrics and demonstrate the system's ability to handle real-world text processing tasks.",
			title="APG NLP Test Document",
			language=LanguageCode.EN,
			metadata={"source": "validation_script", "test": True, "priority": "high"}
		)
		
		assert doc.id is not None
		assert doc.tenant_id is not None
		assert doc.estimated_processing_time > 0
		_log_test_passed("TextDocument model creation and validation")
		
		# Test computed fields and properties
		assert doc.estimated_processing_time > 0.1  # Should be reasonable for this length
		_log_test_passed("TextDocument computed properties")
		
		# Test NLPModel model
		model = NLPModel(
			tenant_id=uuid7str(),
			name="Test BERT Base Model",
			model_key="bert-base-uncased",
			provider=ModelProvider.TRANSFORMERS,
			provider_model_name="bert-base-uncased",
			supported_tasks=[
				NLPTaskType.SENTIMENT_ANALYSIS, 
				NLPTaskType.TEXT_CLASSIFICATION,
				NLPTaskType.NAMED_ENTITY_RECOGNITION
			],
			supported_languages=[LanguageCode.EN, LanguageCode.ES],
			average_latency_ms=150.5,
			accuracy_score=0.89,
			total_requests=1000,
			successful_requests=950,
			failed_requests=50
		)
		
		assert model.is_available == False  # Not loaded yet
		assert model.success_rate == 95.0    # 950/1000 requests successful
		assert 90 <= model.success_rate <= 100  # Reasonable success rate
		_log_test_passed("NLPModel model creation and computed fields")
		
		# Test ProcessingRequest model
		request = ProcessingRequest(
			tenant_id=uuid7str(),
			task_type=NLPTaskType.SENTIMENT_ANALYSIS,
			text_content="This is absolutely amazing! I love this product so much. The quality is outstanding and the customer service was excellent.",
			quality_level=QualityLevel.BALANCED,
			include_confidence=True,
			include_explanations=False,
			timeout_seconds=120
		)
		
		assert request.id is not None
		assert request.fallback_enabled == True
		assert request.priority == "normal"
		assert request.output_format == "json"
		_log_test_passed("ProcessingRequest model creation and defaults")
		
		# Test field validation
		try:
			invalid_request = ProcessingRequest(
				tenant_id=uuid7str(),
				task_type=NLPTaskType.SENTIMENT_ANALYSIS,
				# Missing both text_content and document_id should fail
				quality_level=QualityLevel.BALANCED
			)
			assert False, "Should have failed validation"
		except ValueError:
			_log_test_passed("ProcessingRequest field validation")
		
		# Test ProcessingResult model
		result = ProcessingResult(
			request_id=request.id,
			tenant_id=request.tenant_id,
			task_type=request.task_type,
			model_used="test_model_id",
			provider_used=ModelProvider.TRANSFORMERS,
			processing_time_ms=125.3,
			total_time_ms=135.8,
			results={
				"sentiment": "positive", 
				"confidence": 0.92,
				"scores": {"positive": 0.92, "negative": 0.05, "neutral": 0.03}
			},
			confidence_score=0.92,
			quality_score=0.88
		)
		
		assert result.is_successful == True
		assert result.performance_rating in ["excellent", "good", "acceptable", "poor"]
		assert result.performance_rating == "acceptable"  # Should be "acceptable" for 125ms
		_log_test_passed("ProcessingResult model with computed properties")
		
		# Test StreamingSession model
		session = StreamingSession(
			tenant_id=uuid7str(),
			user_id=uuid7str(),
			task_type=NLPTaskType.SENTIMENT_ANALYSIS,
			chunk_size=800,
			overlap_size=80,
			aggregation_window_ms=3000
		)
		
		assert session.status == "active"
		assert session.chunks_processed == 0
		assert session.is_connected == True
		assert session.chunk_size == 800
		_log_test_passed("StreamingSession model creation")
		
		# Test StreamingChunk model
		chunk = StreamingChunk(
			session_id=session.id,
			sequence_number=1,
			text_content="This is the first streaming chunk of text for real-time processing.",
			start_position=0,
			end_position=68
		)
		
		assert chunk.status == ProcessingStatus.PENDING
		assert chunk.processing_time_ms is None
		assert chunk.processed_at is None
		_log_test_passed("StreamingChunk model creation")
		
		# Test SystemHealth model
		health = SystemHealth(
			overall_status="healthy",
			component_status={
				"models": "healthy", 
				"database": "healthy",
				"streaming": "healthy",
				"cache": "degraded"
			},
			average_response_time_ms=95.2,
			requests_per_minute=450,
			active_sessions=12,
			queue_depth=3,
			cpu_usage_percent=45.8,
			memory_usage_percent=62.1,
			disk_usage_percent=23.4,
			total_models=8,
			active_models=7,
			loaded_models=5,
			failed_models=1
		)
		
		assert health.model_availability_percent == 87.5  # 7/8 active models
		assert health.performance_rating == "excellent"  # Good performance metrics
		_log_test_passed("SystemHealth model with computed metrics")
		
		logger.info("✅ All Pydantic models validated successfully")
		
	except Exception as e:
		import traceback
		_log_test_failed("Pydantic Models Validation", f"{str(e)}\n{traceback.format_exc()}")
		return False
		
	return True

async def validate_apg_integration():
	"""Validate APG integration components"""
	_log_test_section("APG Integration Components")
	
	try:
		# Test capability metadata structure
		metadata = get_capability_metadata()
		
		assert metadata["capability_id"] == "nlp"
		assert metadata["name"] == "Natural Language Processing"
		assert metadata["version"] == "1.0.0"
		assert metadata["category"] == "common"
		assert metadata["author"] == "Datacraft"
		_log_test_passed("APG capability basic metadata")
		
		# Test composition engine integration
		composition = metadata["composition"]
		assert "provides" in composition
		assert "requires" in composition
		assert "enhances" in composition
		assert "optional" in composition
		
		# Verify provides services
		provides = composition["provides"]
		expected_services = [
			"text_processing", "sentiment_analysis", "entity_extraction",
			"text_classification", "language_detection", "text_summarization",
			"streaming_nlp", "collaborative_annotation"
		]
		for service in expected_services:
			assert service in provides, f"Missing service: {service}"
		_log_test_passed("APG composition engine services")
		
		# Verify requires dependencies
		requires = composition["requires"]
		expected_deps = ["ai_orchestration", "auth_rbac", "audit_compliance", "document_management"]
		for dep in expected_deps:
			assert dep in requires, f"Missing dependency: {dep}"
		_log_test_passed("APG composition engine dependencies")
		
		# Test feature configuration
		features = metadata["features"]
		assert "multi_model_orchestration" in features
		assert "real_time_streaming" in features
		assert "collaborative_workbench" in features
		assert "enterprise_compliance" in features
		assert "domain_adaptation" in features
		
		# Verify feature details
		streaming_feature = features["real_time_streaming"]
		assert streaming_feature["latency_target"] == "< 100ms"
		assert streaming_feature["throughput_target"] == "10K+ docs/minute"
		assert streaming_feature["websocket_support"] == True
		_log_test_passed("APG feature configuration")
		
		# Test model configuration
		model_config = metadata["model_config"]
		assert "ollama_integration" in model_config
		assert "transformers_integration" in model_config
		assert "spacy_integration" in model_config
		
		ollama_config = model_config["ollama_integration"]
		assert ollama_config["enabled"] == True
		assert len(ollama_config["supported_models"]) >= 8
		assert "llama3.2:latest" in ollama_config["supported_models"]
		_log_test_passed("APG model configuration")
		
		# Test performance targets
		performance = metadata["performance"]
		assert "latency_targets" in performance
		assert "throughput_targets" in performance
		assert "resource_targets" in performance
		
		latency_targets = performance["latency_targets"]
		assert "text_processing" in latency_targets
		assert "sentiment_analysis" in latency_targets
		assert latency_targets["streaming_chunk"] == "< 25ms"
		_log_test_passed("APG performance targets")
		
		# Test blueprint configuration
		blueprint_config = get_blueprint_config()
		
		assert blueprint_config["blueprint_name"] == "nlp"
		assert blueprint_config["url_prefix"] == "/nlp"
		assert blueprint_config["template_folder"] == "templates"
		assert blueprint_config["static_folder"] == "static"
		
		# Verify menu links
		menu_links = blueprint_config["menu_links"]
		assert len(menu_links) >= 6
		
		menu_hrefs = [link["href"] for link in menu_links]
		expected_links = ["/nlp/dashboard", "/nlp/process", "/nlp/models", "/nlp/streaming"]
		for link in expected_links:
			assert link in menu_hrefs, f"Missing menu link: {link}"
		_log_test_passed("APG Flask blueprint configuration")
		
		# Test permissions
		permissions = blueprint_config["permissions"]
		assert len(permissions) >= 6
		
		permission_names = [perm["name"] for perm in permissions]
		expected_perms = ["nlp_view", "nlp_process", "nlp_manage_models", "nlp_streaming", "nlp_admin"]
		for perm in expected_perms:
			assert perm in permission_names, f"Missing permission: {perm}"
		_log_test_passed("APG permission configuration")
		
		# Test utility functions
		dependencies = validate_apg_dependencies()
		assert isinstance(dependencies, list)
		_log_test_passed("APG dependency validation function")
		
		languages = get_supported_languages()
		assert isinstance(languages, list)
		assert "en" in languages
		assert "auto" in languages
		assert len(languages) >= 12
		_log_test_passed("Supported languages function")
		
		models = get_available_models()
		assert isinstance(models, dict)
		assert "ollama" in models
		assert "transformers" in models
		assert "spacy" in models
		assert len(models["transformers"]) >= 8
		_log_test_passed("Available models function")
		
		logger.info("✅ APG integration components validated successfully")
		
	except Exception as e:
		_log_test_failed("APG Integration Validation", str(e))
		return False
		
	return True

async def validate_database_schema():
	"""Validate database schema structure"""
	_log_test_section("Database Schema Structure")
	
	try:
		# Read and validate the SQL schema file
		schema_path = Path(__file__).parent / "database_schema.sql"
		
		if not schema_path.exists():
			_log_test_failed("Database Schema File", "database_schema.sql not found")
			return False
		
		schema_content = schema_path.read_text()
		
		# Check for PostgreSQL extensions
		required_extensions = ["uuid-ossp", "vector", "pg_trgm", "btree_gin"]
		for ext in required_extensions:
			if f'CREATE EXTENSION IF NOT EXISTS "{ext}"' not in schema_content:
				_log_test_failed("PostgreSQL Extensions", f"Missing extension: {ext}")
				return False
		_log_test_passed("PostgreSQL extensions configuration")
		
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
			"nlp.model_training_configs",
			"nlp.system_health"
		]
		
		for table in required_tables:
			if f"CREATE TABLE {table}" not in schema_content:
				_log_test_failed("Database Tables", f"Missing table: {table}")
				return False
		_log_test_passed("All required database tables defined")
		
		# Check for vector support
		vector_features = [
			"content_embedding vector(1536)",
			"title_embedding vector(384)",
			"result_embedding vector(1536)",
			"chunk_embedding vector(384)",
			"annotation_embedding vector(384)"
		]
		
		for feature in vector_features:
			if feature not in schema_content:
				_log_test_failed("Vector Support", f"Missing vector feature: {feature}")
				return False
		_log_test_passed("Vector embeddings and similarity search configured")
		
		# Check for performance indexes
		index_patterns = [
			"CREATE INDEX",
			"vector_cosine_ops",
			"gin(",
			"ivfflat",
			"idx_documents_tenant_id",
			"idx_models_provider",
			"idx_processing_results_confidence_score"
		]
		
		for pattern in index_patterns:
			if pattern not in schema_content:
				_log_test_failed("Performance Indexes", f"Missing index pattern: {pattern}")
				return False
		_log_test_passed("Performance indexes configured")
		
		# Check for multi-tenancy
		tenancy_features = [
			"tenant_id UUID NOT NULL",
			"ENABLE ROW LEVEL SECURITY",
			"nlp_documents_tenant_isolation",
			"current_setting('apg.current_tenant_id')"
		]
		
		for feature in tenancy_features:
			if feature not in schema_content:
				_log_test_failed("Multi-Tenancy", f"Missing tenancy feature: {feature}")
				return False
		_log_test_passed("Multi-tenancy and row-level security configured")
		
		# Check for audit trails
		audit_features = [
			"created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP",
			"updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP",
			"created_by UUID",
			"is_deleted BOOLEAN DEFAULT FALSE",
			"update_updated_at_column()"
		]
		
		for feature in audit_features:
			if feature not in schema_content:
				_log_test_failed("Audit Trails", f"Missing audit feature: {feature}")
				return False
		_log_test_passed("Audit trails and data lifecycle management configured")
		
		# Check for materialized views
		if "CREATE MATERIALIZED VIEW nlp.model_performance_summary" not in schema_content:
			_log_test_failed("Performance Views", "Missing performance summary view")
			return False
		_log_test_passed("Performance analytics views configured")
		
		logger.info("✅ Database schema validated successfully")
		
	except Exception as e:
		_log_test_failed("Database Schema Validation", str(e))
		return False
		
	return True

async def validate_model_enums_and_types():
	"""Validate enums and type definitions"""
	_log_test_section("Model Enums and Type Definitions")
	
	try:
		# Test NLP task types
		task_types = [
			NLPTaskType.SENTIMENT_ANALYSIS,
			NLPTaskType.ENTITY_EXTRACTION,
			NLPTaskType.TEXT_CLASSIFICATION,
			NLPTaskType.TEXT_SUMMARIZATION,
			NLPTaskType.LANGUAGE_DETECTION,
			NLPTaskType.QUESTION_ANSWERING,
			NLPTaskType.NAMED_ENTITY_RECOGNITION,
			NLPTaskType.TOPIC_MODELING
		]
		
		assert len(task_types) >= 8
		assert all(isinstance(task.value, str) for task in task_types)
		_log_test_passed("NLP task type enumerations")
		
		# Test model providers
		providers = [
			ModelProvider.OLLAMA,
			ModelProvider.TRANSFORMERS,
			ModelProvider.SPACY,
			ModelProvider.NLTK,
			ModelProvider.CUSTOM
		]
		
		assert len(providers) == 5
		assert all(isinstance(provider.value, str) for provider in providers)
		_log_test_passed("Model provider enumerations")
		
		# Test processing status
		statuses = [
			ProcessingStatus.PENDING,
			ProcessingStatus.PROCESSING,
			ProcessingStatus.COMPLETED,
			ProcessingStatus.FAILED,
			ProcessingStatus.CANCELLED
		]
		
		assert len(statuses) == 5
		assert all(isinstance(status.value, str) for status in statuses)
		_log_test_passed("Processing status enumerations")
		
		# Test quality levels
		quality_levels = [
			QualityLevel.FAST,
			QualityLevel.BALANCED,
			QualityLevel.ACCURATE,
			QualityLevel.BEST
		]
		
		assert len(quality_levels) == 4
		assert all(isinstance(level.value, str) for level in quality_levels)
		_log_test_passed("Quality level enumerations")
		
		# Test language codes
		languages = [
			LanguageCode.AUTO,
			LanguageCode.EN,
			LanguageCode.ES,
			LanguageCode.FR,
			LanguageCode.DE,
			LanguageCode.ZH,
			LanguageCode.JA,
			LanguageCode.AR,
			LanguageCode.HI
		]
		
		assert len(languages) >= 9
		assert all(isinstance(lang.value, str) for lang in languages)
		assert LanguageCode.AUTO.value == "auto"
		assert LanguageCode.EN.value == "en"
		_log_test_passed("Language code enumerations")
		
		logger.info("✅ Model enums and types validated successfully")
		
	except Exception as e:
		_log_test_failed("Model Enums and Types Validation", str(e))
		return False
		
	return True

async def validate_model_relationships():
	"""Validate model relationships and constraints"""
	_log_test_section("Model Relationships and Constraints")
	
	try:
		tenant_id = uuid7str()
		
		# Create related models
		document = TextDocument(
			tenant_id=tenant_id,
			content="Sample document content for relationship testing.",
			title="Relationship Test Document"
		)
		
		model = NLPModel(
			tenant_id=tenant_id,
			name="Test Model",
			model_key="test-model-v1",
			provider=ModelProvider.TRANSFORMERS,
			provider_model_name="test/model",
			supported_tasks=[NLPTaskType.SENTIMENT_ANALYSIS],
			supported_languages=[LanguageCode.EN]
		)
		
		# Create processing request referencing document
		request = ProcessingRequest(
			tenant_id=tenant_id,
			document_id=document.id,  # Reference to document
			task_type=NLPTaskType.SENTIMENT_ANALYSIS,
			preferred_model=model.id  # Reference to model
		)
		
		# Create processing result referencing request
		result = ProcessingResult(
			request_id=request.id,
			tenant_id=tenant_id,
			task_type=request.task_type,
			model_used=model.id,
			provider_used=model.provider,
			processing_time_ms=150.0,
			total_time_ms=150.0,
			results={"sentiment": "neutral", "confidence": 0.7}
		)
		
		# Verify relationships
		assert result.request_id == request.id
		assert result.model_used == model.id
		assert request.document_id == document.id
		assert request.preferred_model == model.id
		assert result.tenant_id == request.tenant_id == document.tenant_id == model.tenant_id
		_log_test_passed("Model ID relationships and references")
		
		# Test streaming relationships
		session = StreamingSession(
			tenant_id=tenant_id,
			user_id=uuid7str(),
			task_type=NLPTaskType.SENTIMENT_ANALYSIS,
			model_id=model.id
		)
		
		chunk = StreamingChunk(
			session_id=session.id,
			sequence_number=1,
			text_content="Streaming chunk content",
			start_position=0,
			end_position=22
		)
		
		assert chunk.session_id == session.id
		assert session.model_id == model.id
		_log_test_passed("Streaming model relationships")
		
		# Test annotation project relationships
		from models import AnnotationProject, TextAnnotation
		
		project = AnnotationProject(
			tenant_id=tenant_id,
			name="Test Annotation Project",
			annotation_type=NLPTaskType.NAMED_ENTITY_RECOGNITION,
			annotation_schema={"entities": ["PERSON", "ORG", "LOC"]},
			team_members=[uuid7str(), uuid7str()],
			project_manager=uuid7str(),
			created_by=uuid7str()
		)
		
		annotation = TextAnnotation(
			project_id=project.id,
			document_id=document.id,
			annotator_id=project.team_members[0],
			start_position=0,
			end_position=10,
			annotated_text="Sample tex",
			annotation_value={"entity": "MISC", "confidence": 0.9}
		)
		
		assert annotation.project_id == project.id
		assert annotation.document_id == document.id
		assert annotation.annotator_id in project.team_members
		_log_test_passed("Annotation project relationships")
		
		logger.info("✅ Model relationships validated successfully")
		
	except Exception as e:
		_log_test_failed("Model Relationships Validation", str(e))
		return False
		
	return True

async def main():
	"""Run all validation tests"""
	_log_validation_start()
	
	test_results = []
	
	# Run core validation tests
	test_results.append(await validate_models_and_schemas())
	test_results.append(await validate_apg_integration()) 
	test_results.append(await validate_database_schema())
	test_results.append(await validate_model_enums_and_types())
	test_results.append(await validate_model_relationships())
	
	# Summary
	passed_tests = sum(test_results)
	total_tests = len(test_results)
	
	logger.info(f"\n{'='*60}")
	logger.info(f"APG NLP CORE VALIDATION SUMMARY")
	logger.info(f"{'='*60}")
	logger.info(f"Tests Passed: {passed_tests}/{total_tests}")
	logger.info(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
	
	if passed_tests == total_tests:
		logger.info("🎉 ALL CORE TESTS PASSED - NLP Foundation is Ready!")
		logger.info("📦 Models: ✅ Validated")
		logger.info("🔗 APG Integration: ✅ Validated") 
		logger.info("🗃️  Database Schema: ✅ Validated")
		logger.info("📝 Enums & Types: ✅ Validated")
		logger.info("🔀 Relationships: ✅ Validated")
		_log_validation_complete()
		return 0
	else:
		logger.error(f"❌ {total_tests - passed_tests} TESTS FAILED")
		logger.error("Please review the failed tests and fix issues before proceeding.")
		return 1

if __name__ == "__main__":
	sys.exit(asyncio.run(main()))