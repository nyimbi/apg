"""
APG NLP Models Test Suite

Comprehensive tests for Pydantic v2 models with validation, serialization,
and APG integration patterns.

Uses modern pytest-asyncio patterns without decorators as per APG standards.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from typing import Dict, Any
from uuid_extensions import uuid7str

from ..models import (
	TextDocument, NLPModel, ProcessingRequest, ProcessingResult,
	StreamingSession, StreamingChunk, AnnotationProject, TextAnnotation,
	SystemHealth, ModelTrainingConfig, TextAnalytics,
	NLPTaskType, ModelProvider, ProcessingStatus, QualityLevel, LanguageCode
)
from . import TEST_CONFIG, TEST_TEXTS


class TestTextDocument:
	"""Test TextDocument model validation and functionality"""
	
	def test_create_valid_document(self):
		"""Test creating valid text document"""
		doc = TextDocument(
			tenant_id=TEST_CONFIG['test_tenant_id'],
			content=TEST_TEXTS['positive_sentiment'],
			title="Test Document",
			language=LanguageCode.EN
		)
		
		assert doc.tenant_id == TEST_CONFIG['test_tenant_id']
		assert doc.content == TEST_TEXTS['positive_sentiment']
		assert doc.title == "Test Document"
		assert doc.language == LanguageCode.EN
		assert doc.id is not None
		assert isinstance(doc.created_at, datetime)
	
	def test_document_validation_empty_content(self):
		"""Test document validation fails with empty content"""
		with pytest.raises(ValueError, match="Content cannot be empty"):
			TextDocument(
				tenant_id=TEST_CONFIG['test_tenant_id'],
				content="",
				title="Empty Document"
			)
	
	def test_document_validation_whitespace_content(self):
		"""Test document validation fails with whitespace-only content"""
		with pytest.raises(ValueError, match="Content cannot be empty"):
			TextDocument(
				tenant_id=TEST_CONFIG['test_tenant_id'],
				content="   \n\t  ",
				title="Whitespace Document"
			)
	
	def test_document_validation_content_too_long(self):
		"""Test document validation fails with content too long"""
		long_content = "x" * (10_000_001)  # Exceeds 10MB limit
		
		with pytest.raises(ValueError, match="Content exceeds maximum length"):
			TextDocument(
				tenant_id=TEST_CONFIG['test_tenant_id'],
				content=long_content,
				title="Too Long Document"
			)
	
	def test_estimated_processing_time(self):
		"""Test estimated processing time calculation"""
		short_doc = TextDocument(
			tenant_id=TEST_CONFIG['test_tenant_id'],
			content="Short text",
			title="Short Document"
		)
		
		long_doc = TextDocument(
			tenant_id=TEST_CONFIG['test_tenant_id'],
			content=TEST_TEXTS['long_text'],
			title="Long Document"
		)
		
		assert short_doc.estimated_processing_time > 0
		assert long_doc.estimated_processing_time > short_doc.estimated_processing_time
	
	def test_document_serialization(self):
		"""Test document model serialization"""
		doc = TextDocument(
			tenant_id=TEST_CONFIG['test_tenant_id'],
			content=TEST_TEXTS['positive_sentiment'],
			title="Serialization Test",
			metadata={"source": "test", "version": 1}
		)
		
		serialized = doc.model_dump()
		
		assert serialized['tenant_id'] == TEST_CONFIG['test_tenant_id']
		assert serialized['content'] == TEST_TEXTS['positive_sentiment']
		assert serialized['title'] == "Serialization Test"
		assert serialized['metadata']['source'] == "test"
		assert 'id' in serialized
		assert 'created_at' in serialized


class TestNLPModel:
	"""Test NLPModel validation and functionality"""
	
	def test_create_valid_model(self):
		"""Test creating valid NLP model"""
		model = NLPModel(
			tenant_id=TEST_CONFIG['test_tenant_id'],
			name="BERT Base",
			model_key="bert-base-uncased",
			provider=ModelProvider.TRANSFORMERS,
			provider_model_name="bert-base-uncased",
			supported_tasks=[NLPTaskType.SENTIMENT_ANALYSIS, NLPTaskType.TEXT_CLASSIFICATION],
			supported_languages=[LanguageCode.EN]
		)
		
		assert model.tenant_id == TEST_CONFIG['test_tenant_id']
		assert model.name == "BERT Base"
		assert model.provider == ModelProvider.TRANSFORMERS
		assert NLPTaskType.SENTIMENT_ANALYSIS in model.supported_tasks
		assert LanguageCode.EN in model.supported_languages
		assert model.is_active is True
		assert model.health_status == "unknown"
	
	def test_model_success_rate_calculation(self):
		"""Test model success rate calculation"""
		model = NLPModel(
			tenant_id=TEST_CONFIG['test_tenant_id'],
			name="Test Model",
			model_key="test-model",
			provider=ModelProvider.OLLAMA,
			provider_model_name="test-model",
			supported_tasks=[NLPTaskType.TEXT_GENERATION],
			supported_languages=[LanguageCode.EN],
			successful_requests=90,
			failed_requests=10
		)
		
		assert model.success_rate == 90.0
	
	def test_model_success_rate_no_requests(self):
		"""Test model success rate with no requests"""
		model = NLPModel(
			tenant_id=TEST_CONFIG['test_tenant_id'],
			name="New Model",
			model_key="new-model",
			provider=ModelProvider.SPACY,
			provider_model_name="en_core_web_sm",
			supported_tasks=[NLPTaskType.NAMED_ENTITY_RECOGNITION],
			supported_languages=[LanguageCode.EN]
		)
		
		assert model.success_rate == 0.0
	
	def test_model_availability_check(self):
		"""Test model availability logic"""
		available_model = NLPModel(
			tenant_id=TEST_CONFIG['test_tenant_id'],
			name="Available Model",
			model_key="available-model",
			provider=ModelProvider.TRANSFORMERS,
			provider_model_name="available-model",
			supported_tasks=[NLPTaskType.SENTIMENT_ANALYSIS],
			supported_languages=[LanguageCode.EN],
			is_active=True,
			is_loaded=True,
			health_status="healthy"
		)
		
		unavailable_model = NLPModel(
			tenant_id=TEST_CONFIG['test_tenant_id'],
			name="Unavailable Model",
			model_key="unavailable-model",
			provider=ModelProvider.TRANSFORMERS,
			provider_model_name="unavailable-model",
			supported_tasks=[NLPTaskType.SENTIMENT_ANALYSIS],
			supported_languages=[LanguageCode.EN],
			is_active=False,
			is_loaded=False,
			health_status="unhealthy"
		)
		
		assert available_model.is_available is True
		assert unavailable_model.is_available is False


class TestProcessingRequest:
	"""Test ProcessingRequest validation and functionality"""
	
	def test_create_valid_request_with_text(self):
		"""Test creating valid processing request with text content"""
		request = ProcessingRequest(
			tenant_id=TEST_CONFIG['test_tenant_id'],
			user_id=TEST_CONFIG['test_user_id'],
			task_type=NLPTaskType.SENTIMENT_ANALYSIS,
			text_content=TEST_TEXTS['positive_sentiment'],
			quality_level=QualityLevel.BALANCED
		)
		
		assert request.tenant_id == TEST_CONFIG['test_tenant_id']
		assert request.task_type == NLPTaskType.SENTIMENT_ANALYSIS
		assert request.text_content == TEST_TEXTS['positive_sentiment']
		assert request.quality_level == QualityLevel.BALANCED
		assert request.fallback_enabled is True
	
	def test_create_valid_request_with_document_id(self):
		"""Test creating valid processing request with document ID"""
		request = ProcessingRequest(
			tenant_id=TEST_CONFIG['test_tenant_id'],
			user_id=TEST_CONFIG['test_user_id'],
			task_type=NLPTaskType.ENTITY_EXTRACTION,
			document_id=uuid7str()
		)
		
		assert request.document_id is not None
		assert request.text_content is None
		assert request.task_type == NLPTaskType.ENTITY_EXTRACTION
	
	def test_request_validation_no_content_or_document(self):
		"""Test request validation fails without content or document ID"""
		with pytest.raises(ValueError, match="Either text_content or document_id must be provided"):
			ProcessingRequest(
				tenant_id=TEST_CONFIG['test_tenant_id'],
				user_id=TEST_CONFIG['test_user_id'],
				task_type=NLPTaskType.SENTIMENT_ANALYSIS
			)
	
	def test_request_validation_text_too_long(self):
		"""Test request validation fails with text too long"""
		long_text = "x" * (1_000_001)  # Exceeds 1MB limit
		
		with pytest.raises(ValueError, match="Direct text content exceeds 1MB limit"):
			ProcessingRequest(
				tenant_id=TEST_CONFIG['test_tenant_id'],
				user_id=TEST_CONFIG['test_user_id'],
				task_type=NLPTaskType.SENTIMENT_ANALYSIS,
				text_content=long_text
			)


class TestProcessingResult:
	"""Test ProcessingResult validation and functionality"""
	
	def test_create_successful_result(self):
		"""Test creating successful processing result"""
		request_id = uuid7str()
		
		result = ProcessingResult(
			request_id=request_id,
			tenant_id=TEST_CONFIG['test_tenant_id'],
			task_type=NLPTaskType.SENTIMENT_ANALYSIS,
			model_used="bert-base",
			provider_used=ModelProvider.TRANSFORMERS,
			processing_time_ms=45.2,
			total_time_ms=52.3,
			confidence_score=0.89,
			results={"sentiment": "positive", "confidence": 0.89},
			status=ProcessingStatus.COMPLETED
		)
		
		assert result.request_id == request_id
		assert result.task_type == NLPTaskType.SENTIMENT_ANALYSIS
		assert result.processing_time_ms == 45.2
		assert result.confidence_score == 0.89
		assert result.is_successful is True
		assert result.performance_rating == "excellent"  # <50ms
	
	def test_create_failed_result(self):
		"""Test creating failed processing result"""
		result = ProcessingResult(
			request_id=uuid7str(),
			tenant_id=TEST_CONFIG['test_tenant_id'],
			task_type=NLPTaskType.SENTIMENT_ANALYSIS,
			model_used="unknown",
			provider_used=ModelProvider.CUSTOM,
			processing_time_ms=1000.0,
			total_time_ms=1000.0,
			confidence_score=0.0,
			results={},
			status=ProcessingStatus.FAILED,
			error_message="Model unavailable"
		)
		
		assert result.is_successful is False
		assert result.status == ProcessingStatus.FAILED
		assert result.error_message == "Model unavailable"
		assert result.performance_rating == "poor"  # >500ms
	
	def test_performance_rating_categories(self):
		"""Test performance rating categorization"""
		excellent_result = ProcessingResult(
			request_id=uuid7str(),
			tenant_id=TEST_CONFIG['test_tenant_id'],
			task_type=NLPTaskType.SENTIMENT_ANALYSIS,
			model_used="fast-model",
			provider_used=ModelProvider.SPACY,
			processing_time_ms=25.0,
			total_time_ms=25.0,
			confidence_score=0.95,
			results={}
		)
		
		good_result = ProcessingResult(
			request_id=uuid7str(),
			tenant_id=TEST_CONFIG['test_tenant_id'],
			task_type=NLPTaskType.SENTIMENT_ANALYSIS,
			model_used="medium-model",
			provider_used=ModelProvider.TRANSFORMERS,
			processing_time_ms=75.0,
			total_time_ms=75.0,
			confidence_score=0.85,
			results={}
		)
		
		acceptable_result = ProcessingResult(
			request_id=uuid7str(),
			tenant_id=TEST_CONFIG['test_tenant_id'],
			task_type=NLPTaskType.SENTIMENT_ANALYSIS,
			model_used="slow-model",
			provider_used=ModelProvider.OLLAMA,
			processing_time_ms=250.0,
			total_time_ms=250.0,
			confidence_score=0.75,
			results={}
		)
		
		poor_result = ProcessingResult(
			request_id=uuid7str(),
			tenant_id=TEST_CONFIG['test_tenant_id'],
			task_type=NLPTaskType.SENTIMENT_ANALYSIS,
			model_used="very-slow-model",
			provider_used=ModelProvider.CUSTOM,
			processing_time_ms=750.0,
			total_time_ms=750.0,
			confidence_score=0.65,
			results={}
		)
		
		assert excellent_result.performance_rating == "excellent"
		assert good_result.performance_rating == "good"
		assert acceptable_result.performance_rating == "acceptable"
		assert poor_result.performance_rating == "poor"


class TestStreamingSession:
	"""Test StreamingSession validation and functionality"""
	
	def test_create_valid_streaming_session(self):
		"""Test creating valid streaming session"""
		session = StreamingSession(
			tenant_id=TEST_CONFIG['test_tenant_id'],
			user_id=TEST_CONFIG['test_user_id'],
			task_type=NLPTaskType.SENTIMENT_ANALYSIS,
			chunk_size=1000,
			overlap_size=100
		)
		
		assert session.tenant_id == TEST_CONFIG['test_tenant_id']
		assert session.user_id == TEST_CONFIG['test_user_id']
		assert session.task_type == NLPTaskType.SENTIMENT_ANALYSIS
		assert session.chunk_size == 1000
		assert session.overlap_size == 100
		assert session.status == "active"
		assert session.is_connected is True
		assert session.chunks_processed == 0
	
	def test_streaming_session_chunk_size_validation(self):
		"""Test streaming session chunk size validation"""
		with pytest.raises(ValueError):
			StreamingSession(
				tenant_id=TEST_CONFIG['test_tenant_id'],
				user_id=TEST_CONFIG['test_user_id'],
				task_type=NLPTaskType.SENTIMENT_ANALYSIS,
				chunk_size=50  # Below minimum of 100
			)
		
		with pytest.raises(ValueError):
			StreamingSession(
				tenant_id=TEST_CONFIG['test_tenant_id'],
				user_id=TEST_CONFIG['test_user_id'],
				task_type=NLPTaskType.SENTIMENT_ANALYSIS,
				chunk_size=15000  # Above maximum of 10000
			)


class TestStreamingChunk:
	"""Test StreamingChunk validation and functionality"""
	
	def test_create_valid_streaming_chunk(self):
		"""Test creating valid streaming chunk"""
		session_id = uuid7str()
		
		chunk = StreamingChunk(
			session_id=session_id,
			sequence_number=1,
			text_content=TEST_TEXTS['positive_sentiment'],
			start_position=0,
			end_position=len(TEST_TEXTS['positive_sentiment'])
		)
		
		assert chunk.session_id == session_id
		assert chunk.sequence_number == 1
		assert chunk.text_content == TEST_TEXTS['positive_sentiment']
		assert chunk.status == ProcessingStatus.PENDING
		assert chunk.results is None
	
	def test_streaming_chunk_validation_empty_content(self):
		"""Test streaming chunk validation fails with empty content"""
		with pytest.raises(ValueError):
			StreamingChunk(
				session_id=uuid7str(),
				sequence_number=1,
				text_content="",  # Empty content
				start_position=0,
				end_position=0
			)


class TestAnnotationProject:
	"""Test AnnotationProject validation and functionality"""
	
	def test_create_valid_annotation_project(self):
		"""Test creating valid annotation project"""
		project = AnnotationProject(
			tenant_id=TEST_CONFIG['test_tenant_id'],
			name="Sentiment Analysis Project",
			description="Analyze customer feedback sentiment",
			annotation_type=NLPTaskType.SENTIMENT_ANALYSIS,
			team_members=[TEST_CONFIG['test_user_id']],
			project_manager=TEST_CONFIG['test_user_id'],
			annotation_schema={"labels": ["positive", "negative", "neutral"]}
		)
		
		assert project.name == "Sentiment Analysis Project"
		assert project.annotation_type == NLPTaskType.SENTIMENT_ANALYSIS
		assert TEST_CONFIG['test_user_id'] in project.team_members
		assert project.project_manager == TEST_CONFIG['test_user_id']
		assert project.status == "planning"
		assert project.consensus_threshold == 0.8
	
	def test_annotation_project_completion_percentage(self):
		"""Test annotation project completion percentage calculation"""
		project = AnnotationProject(
			tenant_id=TEST_CONFIG['test_tenant_id'],
			name="Test Project",
			annotation_type=NLPTaskType.SENTIMENT_ANALYSIS,
			team_members=[TEST_CONFIG['test_user_id']],
			project_manager=TEST_CONFIG['test_user_id'],
			annotation_schema={"labels": ["positive", "negative"]},
			document_count=100,
			completed_annotations=25
		)
		
		assert project.completion_percentage == 25.0
	
	def test_annotation_project_no_documents(self):
		"""Test annotation project with no documents"""
		project = AnnotationProject(
			tenant_id=TEST_CONFIG['test_tenant_id'],
			name="Empty Project",
			annotation_type=NLPTaskType.SENTIMENT_ANALYSIS,
			team_members=[TEST_CONFIG['test_user_id']],
			project_manager=TEST_CONFIG['test_user_id'],
			annotation_schema={"labels": ["positive", "negative"]},
			document_count=0,
			completed_annotations=0
		)
		
		assert project.completion_percentage == 0.0


class TestSystemHealth:
	"""Test SystemHealth validation and functionality"""
	
	def test_create_healthy_system_status(self):
		"""Test creating healthy system status"""
		health = SystemHealth(
			tenant_id=TEST_CONFIG['test_tenant_id'],
			overall_status="healthy",
			component_status={"model_bert": "healthy", "model_spacy": "healthy"},
			average_response_time_ms=35.2,
			requests_per_minute=150,
			active_sessions=5,
			queue_depth=0,
			cpu_usage_percent=45.0,
			memory_usage_percent=60.0,
			disk_usage_percent=30.0,
			total_models=6,
			active_models=6,
			loaded_models=6,
			failed_models=0
		)
		
		assert health.overall_status == "healthy"
		assert health.model_availability_percent == 100.0
		assert health.performance_rating == "excellent"
	
	def test_system_health_performance_ratings(self):
		"""Test system health performance rating logic"""
		excellent_health = SystemHealth(
			tenant_id=TEST_CONFIG['test_tenant_id'],
			overall_status="healthy",
			component_status={},
			average_response_time_ms=50.0,  # <100ms
			requests_per_minute=100,
			active_sessions=2,
			queue_depth=0,
			cpu_usage_percent=50.0,  # <70%
			memory_usage_percent=60.0,  # <80%
			disk_usage_percent=40.0,
			total_models=4,
			active_models=4,
			loaded_models=4,
			failed_models=0
		)
		
		poor_health = SystemHealth(
			tenant_id=TEST_CONFIG['test_tenant_id'],
			overall_status="degraded",
			component_status={},
			average_response_time_ms=600.0,  # >500ms
			requests_per_minute=50,
			active_sessions=10,
			queue_depth=20,
			cpu_usage_percent=98.0,  # >95%
			memory_usage_percent=97.0,  # >95%
			disk_usage_percent=85.0,
			total_models=4,
			active_models=2,
			loaded_models=2,
			failed_models=2
		)
		
		assert excellent_health.performance_rating == "excellent"
		assert poor_health.performance_rating == "poor"
		assert poor_health.model_availability_percent == 50.0


class TestModelConfiguration:
	"""Test model configuration validation"""
	
	def test_model_config_dict_validation(self):
		"""Test Pydantic ConfigDict validation"""
		# Test that extra fields are forbidden
		with pytest.raises(ValueError):
			TextDocument(
				tenant_id=TEST_CONFIG['test_tenant_id'],
				content=TEST_TEXTS['positive_sentiment'],
				extra_field="not allowed"  # Should raise validation error
			)
	
	def test_field_aliases_and_validation(self):
		"""Test field aliases and validation by name"""
		doc_data = {
			'tenant_id': TEST_CONFIG['test_tenant_id'],
			'content': TEST_TEXTS['positive_sentiment'],
			'title': 'Test Document'
		}
		
		# Should work with exact field names
		doc = TextDocument(**doc_data)
		assert doc.tenant_id == TEST_CONFIG['test_tenant_id']
	
	def test_string_strip_whitespace(self):
		"""Test automatic string whitespace stripping"""
		doc = TextDocument(
			tenant_id="  " + TEST_CONFIG['test_tenant_id'] + "  ",
			content="  " + TEST_TEXTS['positive_sentiment'] + "  ",
			title="  Test Document  "
		)
		
		# Whitespace should be stripped automatically
		assert doc.tenant_id == TEST_CONFIG['test_tenant_id']
		assert not doc.title.startswith(" ")
		assert not doc.title.endswith(" ")


async def test_async_model_operations():
	"""Test async operations with models (following APG async patterns)"""
	loop = asyncio.get_event_loop()
	
	# Create document asynchronously
	doc = await loop.run_in_executor(None, lambda: TextDocument(
		tenant_id=TEST_CONFIG['test_tenant_id'],
		content=TEST_TEXTS['positive_sentiment'],
		title="Async Test Document"
	))
	
	assert doc.title == "Async Test Document"
	assert doc.tenant_id == TEST_CONFIG['test_tenant_id']