"""
Test configuration and fixtures for NLPC capability testing.

Following APG testing patterns with pytest fixtures, real objects,
and comprehensive async test support without decorators.
"""

import asyncio
import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import AsyncMock, MagicMock
import json
import logging
from datetime import datetime, timedelta
from uuid_extensions import uuid7str

# Import NLPC components
from ..service import NLPCService
from ..models import (
	NLPDocument, ProcessingRequest, ProcessingResult, ContextSession,
	NLPTask, ProcessingStatus, LanguageCode, PriorityLevel, ModelType
)

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== Test Configuration =====

@pytest.fixture(scope="session")
def event_loop():
	"""Create an instance of the default event loop for the test session."""
	loop = asyncio.get_event_loop_policy().new_event_loop()
	yield loop
	loop.close()

@pytest.fixture
def test_config() -> Dict[str, Any]:
	"""Test configuration for NLPC services"""
	return {
		'tenant_id': 'test-tenant',
		'cache_enabled': True,
		'performance_monitoring': True,
		'model_warming': False,  # Disable for faster tests
		'max_cache_size': 100,
		'cache_ttl': 300,
		'security': {
			'rbac_enabled': True,
			'audit_enabled': True,
			'data_classification': True
		},
		'models': {
			'spacy_enabled': True,
			'nltk_enabled': True,
			'transformers_enabled': False,  # Skip heavy models in tests
			'textblob_enabled': True
		}
	}

@pytest.fixture
def temp_directory():
	"""Create temporary directory for test files"""
	temp_dir = tempfile.mkdtemp(prefix='nlpc_test_')
	yield Path(temp_dir)
	shutil.rmtree(temp_dir, ignore_errors=True)

# ===== Service Fixtures =====

@pytest.fixture
async def nlpc_service(test_config):
	"""Create NLPC service instance for testing"""
	service = NLPCService(tenant_id=test_config['tenant_id'])
	
	# Initialize with test configuration
	await service.initialize_nlp_models(test_config['models'])
	await service.initialize_performance_system(test_config)
	
	yield service
	
	# Cleanup
	try:
		await service._cleanup_resources()
	except Exception as e:
		logger.warning(f"Service cleanup error: {str(e)}")

@pytest.fixture
async def mock_nlpc_service(test_config):
	"""Create mock NLPC service for testing without heavy dependencies"""
	service = MagicMock(spec=NLPCService)
	service.tenant_id = test_config['tenant_id']
	service.initialized = True
	
	# Mock async methods
	service.secure_process_document = AsyncMock()
	service.process_with_performance_optimization = AsyncMock()
	service.create_context_session = AsyncMock()
	service.orchestrate_nlp_pipeline = AsyncMock()
	service.get_performance_analytics = AsyncMock()
	service._check_service_health = AsyncMock()
	service._get_available_models = AsyncMock()
	service._warm_model = AsyncMock()
	
	# Configure mock return values
	service._check_service_health.return_value = {
		'status': 'healthy',
		'models_loaded': 3,
		'cache_enabled': True,
		'timestamp': datetime.utcnow().isoformat()
	}
	
	service._get_available_models.return_value = [
		{'name': 'spacy_en_core_web_sm', 'provider': 'spacy', 'loaded': True},
		{'name': 'nltk_punkt', 'provider': 'nltk', 'loaded': True},
		{'name': 'textblob_sentiment', 'provider': 'textblob', 'loaded': True}
	]
	
	yield service

# ===== Data Fixtures =====

@pytest.fixture
def sample_documents() -> List[NLPDocument]:
	"""Sample documents for testing various NLP tasks"""
	return [
		NLPDocument(
			content="Apple Inc. is planning to release new products this year. The company's CEO, Tim Cook, announced this during the quarterly earnings call. Investors are optimistic about the future prospects.",
			language=LanguageCode.EN,
			metadata={'source': 'test', 'category': 'business'},
			tenant_id='test-tenant'
		),
		NLPDocument(
			content="I absolutely love this new smartphone! The camera quality is outstanding and the battery life exceeded my expectations. Highly recommend it to anyone looking for a great device.",
			language=LanguageCode.EN,
			metadata={'source': 'test', 'category': 'review'},
			tenant_id='test-tenant'
		),
		NLPDocument(
			content="The Federal Reserve announced today that it will raise interest rates by 0.25 percentage points to combat inflation. This decision affects mortgage rates, credit cards, and business loans.",
			language=LanguageCode.EN,
			metadata={'source': 'test', 'category': 'finance'},
			tenant_id='test-tenant'
		),
		NLPDocument(
			content="Esta es una oración en español para probar la detección de idiomas y el procesamiento multilingüe del sistema de procesamiento de lenguaje natural.",
			language=LanguageCode.ES,
			metadata={'source': 'test', 'category': 'multilingual'},
			tenant_id='test-tenant'
		),
		NLPDocument(
			content="Ceci est un texte en français pour tester les capacités multilingues du système de traitement du langage naturel.",
			language=LanguageCode.FR,
			metadata={'source': 'test', 'category': 'multilingual'},
			tenant_id='test-tenant'
		)
	]

@pytest.fixture
def sample_processing_requests() -> List[ProcessingRequest]:
	"""Sample processing requests for testing"""
	base_tenant = 'test-tenant'
	
	return [
		ProcessingRequest(
			document_id=uuid7str(),
			tasks=[NLPTask.SENTIMENT_ANALYSIS],
			priority=PriorityLevel.MEDIUM,
			options={'model_preference': 'auto'},
			tenant_id=base_tenant
		),
		ProcessingRequest(
			document_id=uuid7str(),
			tasks=[NLPTask.NAMED_ENTITY_RECOGNITION],
			priority=PriorityLevel.HIGH,
			options={'extract_relationships': True},
			tenant_id=base_tenant
		),
		ProcessingRequest(
			document_id=uuid7str(),
			tasks=[NLPTask.TEXT_CLASSIFICATION, NLPTask.KEYWORD_EXTRACTION],
			priority=PriorityLevel.LOW,
			options={'categories': ['business', 'technology', 'finance']},
			tenant_id=base_tenant
		),
		ProcessingRequest(
			document_id=uuid7str(),
			tasks=[NLPTask.LANGUAGE_DETECTION],
			priority=PriorityLevel.MEDIUM,
			options={'confidence_threshold': 0.8},
			tenant_id=base_tenant
		),
		ProcessingRequest(
			document_id=uuid7str(),
			tasks=[NLPTask.TEXT_SUMMARIZATION],
			priority=PriorityLevel.HIGH,
			options={'max_sentences': 3, 'extractive': True},
			tenant_id=base_tenant
		)
	]

@pytest.fixture
def sample_security_context() -> Dict[str, Any]:
	"""Sample security context for testing"""
	return {
		'user_id': 'test-user-123',
		'tenant_id': 'test-tenant',
		'user_roles': ['nlp_user', 'nlp_analyst'],
		'permissions': [
			'nlp_process_text',
			'nlp_view_results',
			'nlp_manage_sessions'
		],
		'request_id': uuid7str(),
		'timestamp': datetime.utcnow().isoformat(),
		'data_classification': 'internal'
	}

@pytest.fixture
def sample_context_sessions() -> List[ContextSession]:
	"""Sample context sessions for testing"""
	return [
		ContextSession(
			session_id=uuid7str(),
			tenant_id='test-tenant',
			user_id='test-user-123',
			max_context_length=10000,
			memory_retention_hours=24,
			session_metadata={'project': 'test-analysis'}
		),
		ContextSession(
			session_id=uuid7str(),
			tenant_id='test-tenant',
			user_id='test-user-456',
			max_context_length=5000,
			memory_retention_hours=12,
			session_metadata={'project': 'batch-processing'}
		)
	]

# ===== Mock Fixtures =====

@pytest.fixture
def mock_spacy_model():
	"""Mock spaCy model for testing without actual model loading"""
	mock_doc = MagicMock()
	mock_doc.text = "Test document"
	mock_doc.lang_ = "en"
	mock_doc.sents = [MagicMock()]
	mock_doc.ents = [MagicMock()]
	
	mock_model = MagicMock()
	mock_model.__call__.return_value = mock_doc
	mock_model.meta = {'name': 'en_core_web_sm', 'version': '3.4.0'}
	
	return mock_model

@pytest.fixture
def mock_nltk_data():
	"""Mock NLTK data and models"""
	return {
		'punkt_tokenizer': MagicMock(),
		'stopwords': {'english': set(['the', 'a', 'an', 'and', 'or', 'but'])},
		'wordnet_lemmatizer': MagicMock(),
		'pos_tagger': MagicMock()
	}

@pytest.fixture
def mock_textblob():
	"""Mock TextBlob for testing sentiment analysis"""
	mock_blob = MagicMock()
	mock_blob.sentiment.polarity = 0.5
	mock_blob.sentiment.subjectivity = 0.6
	mock_blob.noun_phrases = ['test phrase', 'another phrase']
	mock_blob.tags = [('test', 'NN'), ('phrase', 'NN')]
	
	mock_textblob_class = MagicMock()
	mock_textblob_class.return_value = mock_blob
	
	return mock_textblob_class

# ===== Performance Testing Fixtures =====

@pytest.fixture
def performance_test_data():
	"""Data for performance testing"""
	return {
		'small_text': "This is a small test document with minimal content for basic performance testing.",
		'medium_text': " ".join([
			"This is a medium-sized test document designed to evaluate processing performance",
			"with moderate amounts of text content. It contains multiple sentences and",
			"various linguistic features to test different NLP capabilities comprehensively."
		] * 10),
		'large_text': " ".join([
			"This is a large test document created specifically for performance benchmarking",
			"and stress testing of the natural language processing system. It contains",
			"extensive text content with complex sentence structures, multiple paragraphs,",
			"and diverse linguistic patterns to thoroughly evaluate system performance",
			"under high-load conditions and ensure scalability requirements are met."
		] * 100),
		'expected_latencies': {
			'small': 50,    # ms
			'medium': 200,  # ms  
			'large': 1000   # ms
		}
	}

# ===== Assertion Helpers =====

def assert_processing_result_valid(result: ProcessingResult):
	"""Assert that a processing result is valid"""
	assert result is not None
	assert hasattr(result, 'request_id')
	assert hasattr(result, 'status')
	assert hasattr(result, 'results')
	assert result.status in [status.value for status in ProcessingStatus]
	assert result.processing_time >= 0
	assert isinstance(result.results, dict)

def assert_document_valid(document: NLPDocument):
	"""Assert that a document is valid"""
	assert document is not None
	assert hasattr(document, 'document_id')
	assert hasattr(document, 'content')
	assert hasattr(document, 'tenant_id')
	assert len(document.content.strip()) > 0
	assert document.tenant_id is not None

def assert_security_context_valid(context: Dict[str, Any]):
	"""Assert that security context is valid"""
	required_fields = ['user_id', 'tenant_id', 'user_roles', 'request_id']
	for field in required_fields:
		assert field in context
		assert context[field] is not None

# ===== Test Utilities =====

def create_test_document(content: str, language: Optional[LanguageCode] = None) -> NLPDocument:
	"""Create a test document with given content"""
	return NLPDocument(
		content=content,
		language=language or LanguageCode.EN,
		metadata={'source': 'test_utility'},
		tenant_id='test-tenant'
	)

def create_test_request(tasks: List[NLPTask], document_id: str = None) -> ProcessingRequest:
	"""Create a test processing request"""
	return ProcessingRequest(
		document_id=document_id or uuid7str(),
		tasks=tasks,
		priority=PriorityLevel.MEDIUM,
		options={},
		tenant_id='test-tenant'
	)

async def wait_for_async_operation(operation, timeout: float = 5.0):
	"""Wait for an async operation with timeout"""
	try:
		return await asyncio.wait_for(operation, timeout=timeout)
	except asyncio.TimeoutError:
		pytest.fail(f"Operation timed out after {timeout} seconds")

# ===== Cleanup =====

@pytest.fixture(autouse=True)
def cleanup_test_resources():
	"""Automatically cleanup test resources after each test"""
	yield
	# Cleanup logic would go here
	# For now, we rely on garbage collection and fixture cleanup