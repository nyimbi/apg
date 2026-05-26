"""
Comprehensive unit tests for NLPC Pydantic models.

Tests all model validation, serialization, deserialization, and business logic
following APG testing standards with real objects and comprehensive coverage.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List
from uuid_extensions import uuid7str
from pydantic import ValidationError
import json

from ...models import (
	NLPDocument, ProcessingRequest, ProcessingResult, ContextSession,
	NLPTask, ProcessingStatus, LanguageCode, PriorityLevel, ModelType,
	ModelConfiguration, ProcessingRecord, AnnotationProject, SystemHealth
)

class TestNLPDocument:
	"""Test NLPDocument model validation and functionality"""
	
	def test_document_creation_with_required_fields(self):
		"""Test creating document with only required fields"""
		loop = asyncio.get_event_loop()
		
		document = NLPDocument(
			content="This is a test document for NLP processing.",
			tenant_id="test-tenant"
		)
		
		assert document.content == "This is a test document for NLP processing."
		assert document.tenant_id == "test-tenant"
		assert document.document_id is not None
		assert len(document.document_id) > 0
		assert document.language is None  # Optional field
		assert isinstance(document.metadata, dict)
		assert len(document.processing_history) == 0
		assert document.created_at is not None
		assert document.updated_at is not None
	
	def test_document_creation_with_all_fields(self):
		"""Test creating document with all fields populated"""
		loop = asyncio.get_event_loop()
		
		test_metadata = {"source": "api", "user_id": "user123"}
		test_history = [
			ProcessingRecord(
				task=NLPTask.SENTIMENT_ANALYSIS,
				status=ProcessingStatus.COMPLETED,
				model_used="test-model",
				results={"sentiment": "positive"},
				processing_time=125.5
			)
		]
		
		document = NLPDocument(
			content="Test document with all fields populated.",
			language=LanguageCode.EN,
			metadata=test_metadata,
			processing_history=test_history,
			tenant_id="test-tenant"
		)
		
		assert document.language == LanguageCode.EN
		assert document.metadata == test_metadata
		assert len(document.processing_history) == 1
		assert document.processing_history[0].task == NLPTask.SENTIMENT_ANALYSIS
	
	def test_document_validation_empty_content(self):
		"""Test validation fails for empty content"""
		loop = asyncio.get_event_loop()
		
		with pytest.raises(ValidationError) as exc_info:
			NLPDocument(
				content="",
				tenant_id="test-tenant"
			)
		
		errors = exc_info.value.errors()
		assert any("content cannot be empty" in str(error) for error in errors)
	
	def test_document_validation_whitespace_content(self):
		"""Test validation fails for whitespace-only content"""
		loop = asyncio.get_event_loop()
		
		with pytest.raises(ValidationError) as exc_info:
			NLPDocument(
				content="   \n\t   ",
				tenant_id="test-tenant"
			)
		
		errors = exc_info.value.errors()
		assert any("content cannot be empty" in str(error) for error in errors)
	
	def test_document_validation_missing_tenant_id(self):
		"""Test validation fails for missing tenant_id"""
		loop = asyncio.get_event_loop()
		
		with pytest.raises(ValidationError) as exc_info:
			NLPDocument(
				content="Test content"
			)
		
		errors = exc_info.value.errors()
		assert any("tenant_id" in str(error) for error in errors)
	
	def test_document_serialization(self):
		"""Test document serialization to dict"""
		loop = asyncio.get_event_loop()
		
		document = NLPDocument(
			content="Test serialization content.",
			language=LanguageCode.EN,
			metadata={"test": "data"},
			tenant_id="test-tenant"
		)
		
		data = document.model_dump()
		
		assert isinstance(data, dict)
		assert data["content"] == "Test serialization content."
		assert data["language"] == "en"
		assert data["metadata"] == {"test": "data"}
		assert data["tenant_id"] == "test-tenant"
		assert "document_id" in data
		assert "created_at" in data
		assert "updated_at" in data
	
	def test_document_json_serialization(self):
		"""Test document JSON serialization"""
		loop = asyncio.get_event_loop()
		
		document = NLPDocument(
			content="Test JSON serialization.",
			tenant_id="test-tenant"
		)
		
		json_str = document.model_dump_json()
		assert isinstance(json_str, str)
		
		# Parse back to verify
		data = json.loads(json_str)
		assert data["content"] == "Test JSON serialization."
		assert data["tenant_id"] == "test-tenant"
	
	def test_document_deserialization(self):
		"""Test document deserialization from dict"""
		loop = asyncio.get_event_loop()
		
		data = {
			"document_id": uuid7str(),
			"content": "Test deserialization content.",
			"language": "en",
			"metadata": {"source": "test"},
			"processing_history": [],
			"tenant_id": "test-tenant",
			"created_at": datetime.utcnow().isoformat(),
			"updated_at": datetime.utcnow().isoformat()
		}
		
		document = NLPDocument.model_validate(data)
		
		assert document.content == "Test deserialization content."
		assert document.language == LanguageCode.EN
		assert document.metadata == {"source": "test"}
		assert document.tenant_id == "test-tenant"

class TestProcessingRequest:
	"""Test ProcessingRequest model validation and functionality"""
	
	def test_request_creation_single_task(self):
		"""Test creating request with single task"""
		loop = asyncio.get_event_loop()
		
		request = ProcessingRequest(
			document_id=uuid7str(),
			tasks=[NLPTask.SENTIMENT_ANALYSIS],
			tenant_id="test-tenant"
		)
		
		assert len(request.tasks) == 1
		assert request.tasks[0] == NLPTask.SENTIMENT_ANALYSIS
		assert request.priority == PriorityLevel.MEDIUM  # Default
		assert isinstance(request.options, dict)
		assert request.request_id is not None
	
	def test_request_creation_multiple_tasks(self):
		"""Test creating request with multiple tasks"""
		loop = asyncio.get_event_loop()
		
		tasks = [
			NLPTask.SENTIMENT_ANALYSIS,
			NLPTask.NAMED_ENTITY_RECOGNITION,
			NLPTask.KEYWORD_EXTRACTION
		]
		
		request = ProcessingRequest(
			document_id=uuid7str(),
			tasks=tasks,
			priority=PriorityLevel.HIGH,
			tenant_id="test-tenant"
		)
		
		assert len(request.tasks) == 3
		assert request.priority == PriorityLevel.HIGH
		assert NLPTask.SENTIMENT_ANALYSIS in request.tasks
		assert NLPTask.NAMED_ENTITY_RECOGNITION in request.tasks
		assert NLPTask.KEYWORD_EXTRACTION in request.tasks
	
	def test_request_validation_empty_tasks(self):
		"""Test validation fails for empty tasks list"""
		loop = asyncio.get_event_loop()
		
		with pytest.raises(ValidationError) as exc_info:
			ProcessingRequest(
				document_id=uuid7str(),
				tasks=[],
				tenant_id="test-tenant"
			)
		
		errors = exc_info.value.errors()
		assert any("at least one task" in str(error) for error in errors)
	
	def test_request_with_options(self):
		"""Test request with processing options"""
		loop = asyncio.get_event_loop()
		
		options = {
			"model_preference": "spacy",
			"confidence_threshold": 0.8,
			"max_results": 10
		}
		
		request = ProcessingRequest(
			document_id=uuid7str(),
			tasks=[NLPTask.TEXT_CLASSIFICATION],
			options=options,
			tenant_id="test-tenant"
		)
		
		assert request.options == options
		assert request.options["model_preference"] == "spacy"
		assert request.options["confidence_threshold"] == 0.8

class TestProcessingResult:
	"""Test ProcessingResult model validation and functionality"""
	
	def test_result_creation_successful(self):
		"""Test creating successful processing result"""
		loop = asyncio.get_event_loop()
		
		request_id = uuid7str()
		results = {
			"sentiment": "positive",
			"confidence": 0.89,
			"entities": [{"text": "Apple", "label": "ORG"}]
		}
		
		result = ProcessingResult(
			request_id=request_id,
			task=NLPTask.SENTIMENT_ANALYSIS,
			status=ProcessingStatus.COMPLETED,
			results=results,
			processing_time=125.5,
			confidence_score=0.89,
			model_used="test-model",
			tenant_id="test-tenant"
		)
		
		assert result.request_id == request_id
		assert result.task == NLPTask.SENTIMENT_ANALYSIS
		assert result.status == ProcessingStatus.COMPLETED
		assert result.results == results
		assert result.processing_time == 125.5
		assert result.confidence_score == 0.89
		assert result.model_used == "test-model"
	
	def test_result_creation_failed(self):
		"""Test creating failed processing result"""
		loop = asyncio.get_event_loop()
		
		result = ProcessingResult(
			request_id=uuid7str(),
			task=NLPTask.TEXT_SUMMARIZATION,
			status=ProcessingStatus.FAILED,
			error_message="Model timeout",
			tenant_id="test-tenant"
		)
		
		assert result.status == ProcessingStatus.FAILED
		assert result.error_message == "Model timeout"
		assert result.results == {}  # Default empty dict
		assert result.confidence_score is None
	
	def test_result_validation_negative_processing_time(self):
		"""Test validation fails for negative processing time"""
		loop = asyncio.get_event_loop()
		
		with pytest.raises(ValidationError) as exc_info:
			ProcessingResult(
				request_id=uuid7str(),
				task=NLPTask.SENTIMENT_ANALYSIS,
				status=ProcessingStatus.COMPLETED,
				processing_time=-10.5,
				tenant_id="test-tenant"
			)
		
		errors = exc_info.value.errors()
		assert any("greater than or equal to 0" in str(error) for error in errors)
	
	def test_result_validation_invalid_confidence(self):
		"""Test validation fails for invalid confidence score"""
		loop = asyncio.get_event_loop()
		
		with pytest.raises(ValidationError) as exc_info:
			ProcessingResult(
				request_id=uuid7str(),
				task=NLPTask.SENTIMENT_ANALYSIS,
				status=ProcessingStatus.COMPLETED,
				confidence_score=1.5,  # > 1.0
				tenant_id="test-tenant"
			)
		
		errors = exc_info.value.errors()
		assert any("less than or equal to 1" in str(error) for error in errors)

class TestContextSession:
	"""Test ContextSession model validation and functionality"""
	
	def test_session_creation_basic(self):
		"""Test creating basic context session"""
		loop = asyncio.get_event_loop()
		
		session = ContextSession(
			tenant_id="test-tenant",
			user_id="user123"
		)
		
		assert session.tenant_id == "test-tenant"
		assert session.user_id == "user123"
		assert session.session_id is not None
		assert session.max_context_length == 10000  # Default
		assert session.memory_retention_hours == 24  # Default
		assert isinstance(session.context_data, list)
		assert len(session.context_data) == 0
	
	def test_session_creation_with_options(self):
		"""Test creating session with custom options"""
		loop = asyncio.get_event_loop()
		
		session_metadata = {"project": "test", "department": "research"}
		
		session = ContextSession(
			tenant_id="test-tenant",
			user_id="user123",
			max_context_length=5000,
			memory_retention_hours=12,
			session_metadata=session_metadata
		)
		
		assert session.max_context_length == 5000
		assert session.memory_retention_hours == 12
		assert session.session_metadata == session_metadata
	
	def test_session_validation_invalid_context_length(self):
		"""Test validation fails for invalid context length"""
		loop = asyncio.get_event_loop()
		
		with pytest.raises(ValidationError) as exc_info:
			ContextSession(
				tenant_id="test-tenant",
				user_id="user123",
				max_context_length=0  # Invalid
			)
		
		errors = exc_info.value.errors()
		assert any("greater than 0" in str(error) for error in errors)
	
	def test_session_validation_invalid_retention_hours(self):
		"""Test validation fails for invalid retention hours"""
		loop = asyncio.get_event_loop()
		
		with pytest.raises(ValidationError) as exc_info:
			ContextSession(
				tenant_id="test-tenant",
				user_id="user123",
				memory_retention_hours=-1  # Invalid
			)
		
		errors = exc_info.value.errors()
		assert any("greater than 0" in str(error) for error in errors)

class TestEnumModels:
	"""Test enum model validation and usage"""
	
	def test_nlp_task_enum_values(self):
		"""Test all NLP task enum values are valid"""
		loop = asyncio.get_event_loop()
		
		expected_tasks = [
			'tokenization',
			'sentence_segmentation', 
			'part_of_speech_tagging',
			'named_entity_recognition',
			'dependency_parsing',
			'sentiment_analysis',
			'text_classification',
			'text_summarization',
			'keyword_extraction',
			'language_detection',
			'question_answering',
			'text_generation'
		]
		
		for task_value in expected_tasks:
			task = NLPTask(task_value)
			assert task.value == task_value
	
	def test_processing_status_enum_values(self):
		"""Test all processing status enum values are valid"""
		loop = asyncio.get_event_loop()
		
		expected_statuses = [
			'pending',
			'processing',
			'completed',
			'failed',
			'cancelled'
		]
		
		for status_value in expected_statuses:
			status = ProcessingStatus(status_value)
			assert status.value == status_value
	
	def test_language_code_enum_values(self):
		"""Test key language code enum values are valid"""
		loop = asyncio.get_event_loop()
		
		key_languages = ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'zh', 'ja', 'ar']
		
		for lang_value in key_languages:
			language = LanguageCode(lang_value)
			assert language.value == lang_value
	
	def test_priority_level_enum_values(self):
		"""Test all priority level enum values are valid"""
		loop = asyncio.get_event_loop()
		
		expected_priorities = ['low', 'medium', 'high', 'critical']
		
		for priority_value in expected_priorities:
			priority = PriorityLevel(priority_value)
			assert priority.value == priority_value

class TestModelConfiguration:
	"""Test ModelConfiguration model validation"""
	
	def test_model_config_creation(self):
		"""Test creating model configuration"""
		loop = asyncio.get_event_loop()
		
		config = ModelConfiguration(
			model_type=ModelType.SPACY,
			language=LanguageCode.EN,
			configuration={"model_name": "en_core_web_sm"},
			tenant_id="test-tenant"
		)
		
		assert config.model_type == ModelType.SPACY
		assert config.language == LanguageCode.EN
		assert config.configuration["model_name"] == "en_core_web_sm"
		assert config.is_active == True  # Default
	
	def test_model_config_validation(self):
		"""Test model configuration validation"""
		loop = asyncio.get_event_loop()
		
		# Test invalid configuration type
		with pytest.raises(ValidationError):
			ModelConfiguration(
				model_type=ModelType.SPACY,
				configuration="invalid",  # Should be dict
				tenant_id="test-tenant"
			)

class TestSystemHealth:
	"""Test SystemHealth model validation"""
	
	def test_system_health_creation(self):
		"""Test creating system health status"""
		loop = asyncio.get_event_loop()
		
		components = {
			"nlp_models": {"status": "healthy", "loaded": 5},
			"cache_system": {"status": "healthy", "hit_rate": 0.85},
			"database": {"status": "healthy", "connections": 10}
		}
		
		health = SystemHealth(
			overall_status="healthy",
			components=components,
			tenant_id="test-tenant"
		)
		
		assert health.overall_status == "healthy"
		assert health.components == components
		assert health.last_check is not None
		assert health.tenant_id == "test-tenant"
	
	def test_system_health_with_issues(self):
		"""Test system health with degraded status"""
		loop = asyncio.get_event_loop()
		
		components = {
			"nlp_models": {"status": "degraded", "loaded": 3, "failed": 2},
			"cache_system": {"status": "healthy", "hit_rate": 0.85}
		}
		
		health = SystemHealth(
			overall_status="degraded",
			components=components,
			error_messages=["Two models failed to load"],
			tenant_id="test-tenant"
		)
		
		assert health.overall_status == "degraded"
		assert len(health.error_messages) == 1
		assert "Two models failed to load" in health.error_messages

class TestModelIntegration:
	"""Test model integration and relationships"""
	
	def test_document_processing_workflow(self):
		"""Test complete document processing workflow with models"""
		loop = asyncio.get_event_loop()
		
		# Create document
		document = NLPDocument(
			content="Apple Inc. reported strong quarterly earnings.",
			language=LanguageCode.EN,
			tenant_id="test-tenant"
		)
		
		# Create processing request
		request = ProcessingRequest(
			document_id=document.document_id,
			tasks=[NLPTask.SENTIMENT_ANALYSIS, NLPTask.NAMED_ENTITY_RECOGNITION],
			priority=PriorityLevel.HIGH,
			tenant_id="test-tenant"
		)
		
		# Create processing result
		result = ProcessingResult(
			request_id=request.request_id,
			task=NLPTask.SENTIMENT_ANALYSIS,
			status=ProcessingStatus.COMPLETED,
			results={
				"sentiment": "positive",
				"confidence": 0.92,
				"entities": [{"text": "Apple Inc.", "label": "ORG", "confidence": 0.95}]
			},
			processing_time=87.3,
			confidence_score=0.92,
			model_used="spacy_en_core_web_sm",
			tenant_id="test-tenant"
		)
		
		# Verify relationships
		assert request.document_id == document.document_id
		assert result.request_id == request.request_id
		assert document.tenant_id == request.tenant_id == result.tenant_id
		
		# Verify data integrity
		assert len(request.tasks) == 2
		assert result.confidence_score == result.results["confidence"]
		assert result.status == ProcessingStatus.COMPLETED
	
	def test_context_session_workflow(self):
		"""Test context session workflow with documents"""
		loop = asyncio.get_event_loop()
		
		# Create context session
		session = ContextSession(
			tenant_id="test-tenant",
			user_id="analyst123",
			max_context_length=5000,
			memory_retention_hours=12
		)
		
		# Create multiple documents for context
		documents = [
			NLPDocument(
				content="First document about Apple Inc.",
				tenant_id="test-tenant"
			),
			NLPDocument(
				content="Second document about Apple's products.",
				tenant_id="test-tenant"
			)
		]
		
		# Simulate adding to context
		context_data = []
		for doc in documents:
			context_entry = {
				"document_id": doc.document_id,
				"content_preview": doc.content[:100],
				"timestamp": datetime.utcnow().isoformat()
			}
			context_data.append(context_entry)
		
		# Update session with context
		session.context_data = context_data
		
		assert len(session.context_data) == 2
		assert session.context_data[0]["document_id"] == documents[0].document_id
		assert session.tenant_id == documents[0].tenant_id