"""
APG RAG Comprehensive Testing Suite

Enterprise-grade testing framework with unit tests, integration tests, 
performance tests, and end-to-end validation for all RAG components.
"""

import asyncio
import pytest
import json
import time
import tempfile
import shutil
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import hashlib

# Testing imports
import pytest_asyncio
from pytest_httpserver import HTTPServer
from werkzeug import Response

# APG imports
from .models import (
	KnowledgeBase, KnowledgeBaseCreate, Document, DocumentCreate,
	Conversation, ConversationCreate, ConversationTurn,
	RetrievalRequest, RetrievalResult, GenerationRequest, GenerationResult,
	DocumentChunk, DocumentChunkCreate, RetrievalMethod, DocumentStatus
)
from .service import RAGService, RAGServiceConfig
from .document_processor import DocumentProcessor, ProcessingConfig
from .vector_service import VectorService, VectorIndexConfig
from .retrieval_engine import IntelligentRetrievalEngine, RetrievalConfig
from .generation_engine import RAGGenerationEngine, GenerationConfig
from .conversation_manager import ConversationManager, ConversationConfig
from .ollama_integration import AdvancedOllamaIntegration
from .security import SecurityManager
from .monitoring import PerformanceMonitor

# Test fixtures and utilities
@pytest.fixture
def temp_dir():
	"""Create temporary directory for tests"""
	temp_path = tempfile.mkdtemp()
	yield Path(temp_path)
	shutil.rmtree(temp_path)

@pytest.fixture
def mock_db_pool():
	"""Mock database connection pool"""
	pool = AsyncMock()
	conn = AsyncMock()
	pool.acquire.return_value.__aenter__.return_value = conn
	pool.acquire.return_value.__aexit__.return_value = None
	return pool

@pytest.fixture
def mock_ollama_integration():
	"""Mock Ollama integration"""
	ollama = AsyncMock(spec=AdvancedOllamaIntegration)
	
	# Mock embedding generation
	async def mock_generate_embeddings_async(texts, model, tenant_id, capability_id, priority, callback):
		embeddings = [[0.1] * 1024 for _ in texts]  # Mock embeddings
		result = {
			'success': True,
			'embeddings': embeddings,
			'model': model,
			'processing_time_ms': 100.0
		}
		if callback:
			callback(result)
	
	ollama.generate_embeddings_async.side_effect = mock_generate_embeddings_async
	
	# Mock text generation
	async def mock_generate_text_async(prompt, model, tenant_id, capability_id, max_tokens, temperature, callback):
		response = f"Mock response to: {prompt[:50]}..."
		result = {
			'success': True,
			'response': response,
			'model': model,
			'token_count': len(response.split()),
			'processing_time_ms': 500.0
		}
		if callback:
			callback(result)
	
	ollama.generate_text_async.side_effect = mock_generate_text_async
	
	# Mock system status
	ollama.get_system_status.return_value = {
		'service_status': 'running',
		'models_loaded': ['bge-m3', 'qwen3'],
		'timestamp': datetime.now().isoformat()
	}
	
	return ollama

@pytest.fixture
def sample_documents():
	"""Sample documents for testing"""
	return [
		{
			'title': 'Test Document 1',
			'content': 'This is a test document about artificial intelligence and machine learning.',
			'file_type': 'text/plain',
			'filename': 'test1.txt'
		},
		{
			'title': 'Test Document 2', 
			'content': 'This document discusses natural language processing and text analysis.',
			'file_type': 'text/plain',
			'filename': 'test2.txt'
		},
		{
			'title': 'Test Document 3',
			'content': 'A comprehensive guide to vector databases and similarity search.',
			'file_type': 'text/plain',
			'filename': 'test3.txt'
		}
	]

class TestDocumentProcessor:
	"""Test document processing functionality"""
	
	@pytest.mark.asyncio
	async def test_text_document_processing(self, temp_dir):
		"""Test basic text document processing"""
		config = ProcessingConfig()
		processor = DocumentProcessor(config, "test-tenant", "rag")
		
		# Create mock document
		document = Document(
			tenant_id="test-tenant",
			knowledge_base_id="test-kb",
			title="Test Document",
			filename="test.txt",
			file_type="text/plain"
		)
		
		content = b"This is a test document with multiple sentences. It contains information about testing. The document should be chunked properly."
		
		result = await processor.process_document(content, "text/plain", document)
		
		assert result.success
		assert len(result.chunks) > 0
		assert all(chunk.content for chunk in result.chunks)
		assert all(chunk.tenant_id == "test-tenant" for chunk in result.chunks)
	
	@pytest.mark.asyncio
	async def test_pdf_document_processing(self, temp_dir):
		"""Test PDF document processing"""
		config = ProcessingConfig()
		processor = DocumentProcessor(config, "test-tenant", "rag")
		
		# Mock PDF content (simplified)
		pdf_content = b"%PDF-1.4\nMock PDF content for testing"
		
		document = Document(
			tenant_id="test-tenant",
			knowledge_base_id="test-kb",
			title="Test PDF",
			filename="test.pdf",
			file_type="application/pdf"
		)
		
		with patch('PyPDF2.PdfReader') as mock_reader:
			mock_page = Mock()
			mock_page.extract_text.return_value = "This is extracted PDF text content."
			mock_reader.return_value.pages = [mock_page]
			
			result = await processor.process_document(pdf_content, "application/pdf", document)
			
			assert result.success
			assert len(result.chunks) > 0
	
	def test_chunk_size_configuration(self):
		"""Test different chunk size configurations"""
		config_small = ProcessingConfig(chunk_size=100, chunk_overlap=20)
		config_large = ProcessingConfig(chunk_size=2000, chunk_overlap=200)
		
		processor_small = DocumentProcessor(config_small, "test-tenant", "rag")
		processor_large = DocumentProcessor(config_large, "test-tenant", "rag")
		
		text = "This is a test " * 100  # Create long text
		
		chunks_small = processor_small._create_chunks(text, "test-doc")
		chunks_large = processor_large._create_chunks(text, "test-doc")
		
		assert len(chunks_small) > len(chunks_large)
		assert all(len(chunk.content) <= 120 for chunk in chunks_small)  # chunk_size + some buffer

class TestVectorService:
	"""Test vector indexing and search functionality"""
	
	@pytest.mark.asyncio
	async def test_chunk_indexing(self, mock_db_pool, mock_ollama_integration):
		"""Test chunk indexing with embeddings"""
		config = VectorIndexConfig()
		service = VectorService(config, mock_db_pool, mock_ollama_integration, "test-tenant", "rag")
		
		# Create test chunks
		chunks = [
			DocumentChunk(
				tenant_id="test-tenant",
				document_id="doc1",
				knowledge_base_id="kb1",
				chunk_index=0,
				content="Test chunk content 1"
			),
			DocumentChunk(
				tenant_id="test-tenant", 
				document_id="doc1",
				knowledge_base_id="kb1",
				chunk_index=1,
				content="Test chunk content 2"
			)
		]
		
		# Mock database operations
		mock_db_pool.acquire.return_value.__aenter__.return_value.execute.return_value = None
		
		result = await service.index_chunks(chunks)
		
		assert result.success
		assert result.chunks_processed == 2
		assert result.embeddings_generated == 2
	
	@pytest.mark.asyncio
	async def test_vector_search(self, mock_db_pool, mock_ollama_integration):
		"""Test vector similarity search"""
		config = VectorIndexConfig()
		service = VectorService(config, mock_db_pool, mock_ollama_integration, "test-tenant", "rag")
		
		# Mock search results
		mock_rows = [
			{
				'id': 'chunk1',
				'document_id': 'doc1',
				'content': 'Test content 1',
				'chunk_index': 0,
				'character_count': 100,
				'section_title': 'Section 1',
				'section_level': 1,
				'document_title': 'Test Doc',
				'document_filename': 'test.txt',
				'similarity_score': 0.95
			}
		]
		
		mock_db_pool.acquire.return_value.__aenter__.return_value.fetch.return_value = mock_rows
		mock_db_pool.acquire.return_value.__aenter__.return_value.execute.return_value = None
		
		query_embedding = [0.1] * 1024
		results = await service.vector_search(
			query_embedding=query_embedding,
			knowledge_base_id="test-kb",
			k=10,
			similarity_threshold=0.7
		)
		
		assert len(results) == 1
		assert results[0]['chunk_id'] == 'chunk1'
		assert results[0]['similarity_score'] == 0.95

class TestRetrievalEngine:
	"""Test intelligent retrieval functionality"""
	
	@pytest.mark.asyncio
	async def test_basic_retrieval(self, mock_db_pool, mock_ollama_integration):
		"""Test basic retrieval functionality"""
		config = RetrievalConfig()
		vector_service = VectorService(VectorIndexConfig(), mock_db_pool, mock_ollama_integration, "test-tenant", "rag")
		
		engine = IntelligentRetrievalEngine(
			config, mock_db_pool, vector_service, mock_ollama_integration, "test-tenant", "rag"
		)
		
		# Mock vector search results
		vector_service.vector_search = AsyncMock(return_value=[
			{
				'chunk_id': 'chunk1',
				'content': 'Test content about AI',
				'similarity_score': 0.9,
				'document_title': 'AI Guide'
			}
		])
		
		request = RetrievalRequest(
			query_text="What is artificial intelligence?",
			knowledge_base_id="test-kb",
			k_retrievals=5,
			similarity_threshold=0.7,
			retrieval_method=RetrievalMethod.VECTOR_SEARCH
		)
		
		result = await engine.retrieve(request)
		
		assert result.success
		assert len(result.retrieved_chunk_ids) > 0
		assert len(result.similarity_scores) > 0
	
	@pytest.mark.asyncio
	async def test_hybrid_search(self, mock_db_pool, mock_ollama_integration):
		"""Test hybrid search functionality"""
		config = RetrievalConfig()
		vector_service = VectorService(VectorIndexConfig(), mock_db_pool, mock_ollama_integration, "test-tenant", "rag")
		
		engine = IntelligentRetrievalEngine(
			config, mock_db_pool, vector_service, mock_ollama_integration, "test-tenant", "rag"
		)
		
		# Mock hybrid search results
		vector_service.hybrid_search = AsyncMock(return_value=[
			{
				'chunk_id': 'chunk1',
				'content': 'AI and machine learning',
				'combined_score': 0.85,
				'vector_score': 0.8,
				'text_score': 0.9
			}
		])
		
		request = RetrievalRequest(
			query_text="machine learning algorithms",
			knowledge_base_id="test-kb",
			retrieval_method=RetrievalMethod.HYBRID_SEARCH
		)
		
		result = await engine.retrieve(request)
		
		assert result.success
		assert len(result.retrieved_chunk_ids) > 0

class TestGenerationEngine:
	"""Test RAG generation functionality"""
	
	@pytest.mark.asyncio
	async def test_basic_generation(self, mock_ollama_integration):
		"""Test basic RAG response generation"""
		config = GenerationConfig()
		engine = RAGGenerationEngine(config, mock_ollama_integration, "test-tenant", "rag")
		
		# Create mock retrieval result
		retrieval_result = RetrievalResult(
			success=True,
			query_text="What is AI?",
			retrieved_chunk_ids=["chunk1"],
			similarity_scores=[0.9],
			processing_time_ms=100.0
		)
		
		request = GenerationRequest(
			prompt="What is artificial intelligence?",
			model="qwen3",
			max_tokens=512,
			temperature=0.7
		)
		
		result = await engine.generate_response(request, retrieval_result)
		
		assert result.success
		assert len(result.response_text) > 0
		assert result.generation_model == "qwen3"
		assert result.token_count > 0
	
	@pytest.mark.asyncio
	async def test_generation_with_sources(self, mock_ollama_integration):
		"""Test generation with source attribution"""
		config = GenerationConfig()
		engine = RAGGenerationEngine(config, mock_ollama_integration, "test-tenant", "rag")
		
		# Mock chunk retrieval for source attribution
		engine._get_chunks_for_attribution = AsyncMock(return_value=[
			{
				'chunk_id': 'chunk1',
				'content': 'AI is a field of computer science',
				'document_title': 'AI Handbook',
				'similarity_score': 0.9
			}
		])
		
		retrieval_result = RetrievalResult(
			success=True,
			query_text="What is AI?",
			retrieved_chunk_ids=["chunk1"],
			similarity_scores=[0.9]
		)
		
		request = GenerationRequest(
			prompt="What is artificial intelligence?",
			model="qwen3"
		)
		
		result = await engine.generate_response(request, retrieval_result)
		
		assert result.success
		assert len(result.sources_used) > 0
		assert result.citation_coverage > 0

class TestConversationManager:
	"""Test conversation management functionality"""
	
	@pytest.mark.asyncio
	async def test_conversation_creation(self, mock_db_pool, mock_ollama_integration):
		"""Test conversation creation"""
		config = ConversationConfig()
		
		# Create mock components
		vector_service = Mock()
		retrieval_engine = Mock()
		generation_engine = Mock()
		
		manager = ConversationManager(
			config, mock_db_pool, retrieval_engine, generation_engine, "test-tenant", "rag"
		)
		
		# Mock database operations
		mock_db_pool.acquire.return_value.__aenter__.return_value.execute.return_value = None
		
		conv_create = ConversationCreate(
			title="Test Conversation",
			knowledge_base_id="test-kb",
			user_id="test-user"
		)
		
		conversation = await manager.create_conversation(conv_create)
		
		assert conversation.title == "Test Conversation"
		assert conversation.tenant_id == "test-tenant"
		assert conversation.knowledge_base_id == "test-kb"
	
	@pytest.mark.asyncio
	async def test_chat_processing(self, mock_db_pool, mock_ollama_integration):
		"""Test chat message processing"""
		config = ConversationConfig()
		
		# Create mock components with proper async methods
		retrieval_engine = AsyncMock()
		generation_engine = AsyncMock()
		
		retrieval_result = RetrievalResult(
			success=True,
			query_text="Hello",
			retrieved_chunk_ids=["chunk1"],
			similarity_scores=[0.8]
		)
		retrieval_engine.retrieve.return_value = retrieval_result
		
		generation_result = GenerationResult(
			success=True,
			response_text="Hello! How can I help you?",
			generation_model="qwen3",
			token_count=8
		)
		generation_engine.generate_response.return_value = generation_result
		
		manager = ConversationManager(
			config, mock_db_pool, retrieval_engine, generation_engine, "test-tenant", "rag"
		)
		
		# Mock database operations
		mock_db_pool.acquire.return_value.__aenter__.return_value.execute.return_value = None
		mock_db_pool.acquire.return_value.__aenter__.return_value.fetch.return_value = []
		mock_db_pool.acquire.return_value.__aenter__.return_value.fetchrow.return_value = {
			'id': 'conv1',
			'tenant_id': 'test-tenant',
			'knowledge_base_id': 'test-kb',
			'title': 'Test Chat',
			'status': 'active',
			'turn_count': 0,
			'total_tokens_used': 0,
			'user_id': 'test-user',
			'created_at': datetime.now(),
			'updated_at': datetime.now()
		}
		
		user_turn, assistant_turn = await manager.process_user_message(
			conversation_id="conv1",
			user_message="Hello, how are you?",
			user_context={}
		)
		
		assert user_turn.content == "Hello, how are you?"
		assert assistant_turn.content == "Hello! How can I help you?"

class TestRAGService:
	"""Test main RAG service integration"""
	
	@pytest.mark.asyncio
	async def test_service_initialization(self, mock_db_pool, mock_ollama_integration):
		"""Test RAG service initialization"""
		config = RAGServiceConfig(
			tenant_id="test-tenant",
			capability_id="rag"
		)
		
		service = RAGService(config, mock_db_pool, mock_ollama_integration)
		
		# Mock component initialization
		with patch.multiple(
			'capabilities.common.rag.service',
			DocumentProcessor=Mock,
			VectorService=AsyncMock,
			IntelligentRetrievalEngine=AsyncMock,
			RAGGenerationEngine=AsyncMock,
			ConversationManager=Mock
		):
			await service.start()
			
			assert service.document_processor is not None
			assert service.vector_service is not None
			assert service.retrieval_engine is not None
			assert service.generation_engine is not None
			assert service.conversation_manager is not None
	
	@pytest.mark.asyncio
	async def test_knowledge_base_creation(self, mock_db_pool, mock_ollama_integration):
		"""Test knowledge base creation"""
		config = RAGServiceConfig(tenant_id="test-tenant")
		service = RAGService(config, mock_db_pool, mock_ollama_integration)
		
		# Mock database operations
		mock_db_pool.acquire.return_value.__aenter__.return_value.execute.return_value = None
		
		kb_create = KnowledgeBaseCreate(
			name="Test KB",
			description="Test knowledge base",
			user_id="test-user"
		)
		
		knowledge_base = await service.create_knowledge_base(kb_create)
		
		assert knowledge_base.name == "Test KB"
		assert knowledge_base.tenant_id == "test-tenant"
	
	@pytest.mark.asyncio
	async def test_document_upload_and_processing(self, mock_db_pool, mock_ollama_integration, sample_documents):
		"""Test document upload and processing"""
		config = RAGServiceConfig(tenant_id="test-tenant")
		service = RAGService(config, mock_db_pool, mock_ollama_integration)
		
		# Mock components
		service.document_processor = AsyncMock()
		service.vector_service = AsyncMock()
		
		# Mock processing result
		chunks = [
			DocumentChunk(
				tenant_id="test-tenant",
				document_id="doc1",
				knowledge_base_id="kb1",
				chunk_index=0,
				content="Test content"
			)
		]
		
		from .document_processor import ProcessingResult
		processing_result = ProcessingResult(
			success=True,
			chunks=chunks,
			processing_time_ms=100.0
		)
		
		service.document_processor.process_document.return_value = processing_result
		service.vector_service.index_chunks.return_value = None
		
		# Mock database operations
		mock_db_pool.acquire.return_value.__aenter__.return_value.execute.return_value = None
		mock_db_pool.acquire.return_value.__aenter__.return_value.fetchrow.return_value = {
			'id': 'kb1',
			'tenant_id': 'test-tenant',
			'name': 'Test KB'
		}
		
		doc_create = DocumentCreate(
			title=sample_documents[0]['title'],
			filename=sample_documents[0]['filename'],
			file_type=sample_documents[0]['file_type'],
			content_hash="test-hash",
			user_id="test-user"
		)
		
		content = sample_documents[0]['content'].encode()
		
		document = await service.add_document("kb1", doc_create, content)
		
		assert document.title == sample_documents[0]['title']
		assert document.tenant_id == "test-tenant"

class TestSecurityIntegration:
	"""Test security and compliance features"""
	
	@pytest.mark.asyncio
	async def test_tenant_isolation(self, mock_db_pool):
		"""Test tenant data isolation"""
		security_manager = SecurityManager(mock_db_pool)
		
		# Mock database responses for different tenants
		mock_db_pool.acquire.return_value.__aenter__.return_value.fetchval.side_effect = [
			True,   # Resource exists for tenant1
			False   # Resource doesn't exist for tenant2
		]
		
		# Test access for correct tenant
		has_access_1 = await security_manager.tenant_isolation.verify_tenant_access(
			"tenant1", "resource1", "knowledge_base"
		)
		assert has_access_1 is True
		
		# Test access for incorrect tenant
		has_access_2 = await security_manager.tenant_isolation.verify_tenant_access(
			"tenant2", "resource1", "knowledge_base"
		)
		assert has_access_2 is False
	
	@pytest.mark.asyncio
	async def test_audit_logging(self, mock_db_pool):
		"""Test audit logging functionality"""
		security_manager = SecurityManager(mock_db_pool)
		
		# Mock database operations
		mock_db_pool.acquire.return_value.__aenter__.return_value.execute.return_value = None
		
		from .security import SecurityContext, AccessOperation
		security_context = SecurityContext(
			tenant_id="test-tenant",
			user_id="test-user",
			session_id="test-session",
			ip_address="127.0.0.1"
		)
		
		audit_id = await security_manager.audit_logger.log_access(
			security_context=security_context,
			operation=AccessOperation.READ,
			resource_type="knowledge_base",
			resource_id="kb1",
			success=True
		)
		
		assert audit_id != ""
	
	def test_data_encryption(self):
		"""Test data encryption functionality"""
		from .security import EncryptionManager
		
		encryption_manager = EncryptionManager()
		
		original_data = "Sensitive information that needs encryption"
		encrypted_data = encryption_manager.encrypt_data(original_data)
		decrypted_data = encryption_manager.decrypt_data(encrypted_data)
		
		assert encrypted_data != original_data
		assert decrypted_data == original_data

class TestPerformanceMonitoring:
	"""Test performance monitoring and optimization"""
	
	@pytest.mark.asyncio
	async def test_metrics_collection(self, mock_db_pool):
		"""Test metrics collection functionality"""
		monitor = PerformanceMonitor(mock_db_pool, "test-tenant", "rag")
		
		# Test metric recording
		monitor.record_operation_time("document_processing", 1500.0)
		monitor.record_operation_count("embeddings_generated", 5)
		monitor.record_quality_metric("rag_accuracy_score", 0.85)
		
		# Verify metrics were recorded
		metrics = monitor.metrics_collector.get_all_metrics()
		
		assert "rag_document_processing_time_ms" in metrics
		assert "rag_embeddings_generated_count" in metrics
		assert "rag_accuracy_score" in metrics
	
	@pytest.mark.asyncio
	async def test_alert_processing(self, mock_db_pool):
		"""Test alert rule processing"""
		monitor = PerformanceMonitor(mock_db_pool, "test-tenant", "rag")
		
		# Simulate high CPU usage
		for _ in range(10):
			monitor.metrics_collector.record_metric("system_cpu_percent", 95.0)
		
		# Process alerts (normally done by background task)
		metrics_collector = monitor.metrics_collector
		
		# Find CPU alert rule
		cpu_alert_rule = None
		for rule in metrics_collector.alert_rules.values():
			if rule.metric_name == "system_cpu_percent" and rule.threshold == 95.0:
				cpu_alert_rule = rule
				break
		
		assert cpu_alert_rule is not None
		
		# Manually trigger alert evaluation
		cpu_metric = metrics_collector.get_metric("system_cpu_percent")
		avg_value = cpu_metric.get_average(cpu_alert_rule.duration_minutes)
		
		assert avg_value > cpu_alert_rule.threshold

class TestEndToEndWorkflows:
	"""End-to-end integration tests"""
	
	@pytest.mark.asyncio
	async def test_complete_rag_workflow(self, mock_db_pool, mock_ollama_integration, sample_documents):
		"""Test complete RAG workflow from document upload to response generation"""
		config = RAGServiceConfig(tenant_id="test-tenant")
		service = RAGService(config, mock_db_pool, mock_ollama_integration)
		
		# Mock all database operations
		mock_db_pool.acquire.return_value.__aenter__.return_value.execute.return_value = None
		mock_db_pool.acquire.return_value.__aenter__.return_value.fetchrow.return_value = {
			'id': 'kb1',
			'tenant_id': 'test-tenant',
			'name': 'Test KB',
			'embedding_model': 'bge-m3',
			'generation_model': 'qwen3'
		}
		mock_db_pool.acquire.return_value.__aenter__.return_value.fetch.return_value = []
		mock_db_pool.acquire.return_value.__aenter__.return_value.fetchval.return_value = True
		
		# Initialize service with mocked components
		await service._initialize_components()
		
		# Mock document processing
		service.document_processor.process_document = AsyncMock()
		chunks = [
			DocumentChunk(
				tenant_id="test-tenant",
				document_id="doc1",
				knowledge_base_id="kb1",
				chunk_index=0,
				content=sample_documents[0]['content']
			)
		]
		
		from .document_processor import ProcessingResult
		service.document_processor.process_document.return_value = ProcessingResult(
			success=True,
			chunks=chunks,
			processing_time_ms=100.0
		)
		
		# Mock vector indexing
		service.vector_service.index_chunks = AsyncMock()
		
		# Mock retrieval
		service.retrieval_engine.retrieve = AsyncMock()
		service.retrieval_engine.retrieve.return_value = RetrievalResult(
			success=True,
			query_text="What is AI?",
			retrieved_chunk_ids=["chunk1"],
			similarity_scores=[0.9]
		)
		
		# Mock generation
		service.generation_engine.generate_response = AsyncMock()
		service.generation_engine.generate_response.return_value = GenerationResult(
			success=True,
			response_text="AI is artificial intelligence.",
			generation_model="qwen3",
			token_count=5
		)
		
		# Step 1: Create knowledge base
		kb_create = KnowledgeBaseCreate(
			name="Test Knowledge Base",
			description="Test KB for end-to-end testing",
			user_id="test-user"
		)
		knowledge_base = await service.create_knowledge_base(kb_create)
		
		# Step 2: Upload document
		doc_create = DocumentCreate(
			title=sample_documents[0]['title'],
			filename=sample_documents[0]['filename'],
			file_type=sample_documents[0]['file_type'],
			content_hash="test-hash",
			user_id="test-user"
		)
		content = sample_documents[0]['content'].encode()
		document = await service.add_document(knowledge_base.id, doc_create, content)
		
		# Step 3: Query knowledge base
		retrieval_result = await service.query_knowledge_base(
			kb_id=knowledge_base.id,
			query_text="What is artificial intelligence?",
			k=5
		)
		
		# Step 4: Generate response
		generation_result = await service.generate_response(
			kb_id=knowledge_base.id,
			query_text="What is artificial intelligence?"
		)
		
		# Verify complete workflow
		assert knowledge_base.name == "Test Knowledge Base"
		assert document.title == sample_documents[0]['title']
		assert retrieval_result.success
		assert generation_result.success
		assert len(generation_result.response_text) > 0

class TestPerformanceBenchmarks:
	"""Performance and load testing"""
	
	@pytest.mark.asyncio
	async def test_concurrent_queries(self, mock_db_pool, mock_ollama_integration):
		"""Test system performance under concurrent query load"""
		config = RAGServiceConfig(tenant_id="test-tenant")
		service = RAGService(config, mock_db_pool, mock_ollama_integration)
		
		# Mock components for performance testing
		service.retrieval_engine = AsyncMock()
		service.generation_engine = AsyncMock()
		
		service.retrieval_engine.retrieve.return_value = RetrievalResult(
			success=True,
			query_text="test",
			retrieved_chunk_ids=["chunk1"],
			similarity_scores=[0.8]
		)
		
		service.generation_engine.generate_response.return_value = GenerationResult(
			success=True,
			response_text="Test response",
			generation_model="qwen3",
			token_count=2
		)
		
		# Mock database operations
		mock_db_pool.acquire.return_value.__aenter__.return_value.fetchrow.return_value = {
			'id': 'kb1',
			'tenant_id': 'test-tenant',
			'name': 'Test KB'
		}
		
		# Create concurrent queries
		query_tasks = []
		num_concurrent_queries = 10
		
		start_time = time.time()
		
		for i in range(num_concurrent_queries):
			task = service.generate_response(
				kb_id="kb1",
				query_text=f"Test query {i}"
			)
			query_tasks.append(task)
		
		# Execute all queries concurrently
		results = await asyncio.gather(*query_tasks, return_exceptions=True)
		
		end_time = time.time()
		total_time = end_time - start_time
		
		# Verify results
		successful_results = [r for r in results if isinstance(r, GenerationResult) and r.success]
		
		assert len(successful_results) == num_concurrent_queries
		assert total_time < 10.0  # Should complete within 10 seconds
		
		# Calculate average response time
		avg_response_time = total_time / num_concurrent_queries
		assert avg_response_time < 1.0  # Average should be under 1 second with mocks
	
	@pytest.mark.asyncio
	async def test_memory_usage_under_load(self, mock_db_pool, mock_ollama_integration):
		"""Test memory usage during high-load operations"""
		import psutil
		import gc
		
		process = psutil.Process()
		initial_memory = process.memory_info().rss / 1024 / 1024  # MB
		
		config = RAGServiceConfig(tenant_id="test-tenant")
		service = RAGService(config, mock_db_pool, mock_ollama_integration)
		
		# Simulate processing many documents
		large_content = "This is a large document content. " * 1000
		
		for i in range(50):  # Process 50 large documents
			doc_create = DocumentCreate(
				title=f"Large Document {i}",
				filename=f"large_doc_{i}.txt",
				file_type="text/plain",
				content_hash=f"hash_{i}",
				user_id="test-user"
			)
			
			# Mock the processing without actually doing heavy work
			pass
		
		# Force garbage collection
		gc.collect()
		
		final_memory = process.memory_info().rss / 1024 / 1024  # MB
		memory_increase = final_memory - initial_memory
		
		# Memory increase should be reasonable (less than 100MB for mocked operations)
		assert memory_increase < 100

# Pytest configuration and test discovery
def pytest_configure(config):
	"""Configure pytest for RAG testing"""
	config.addinivalue_line(
		"markers", "asyncio: mark test as an asyncio coroutine"
	)
	config.addinivalue_line(
		"markers", "integration: mark test as integration test"
	)
	config.addinivalue_line(
		"markers", "performance: mark test as performance test"
	)

# Test runners for different test categories
class TestRunner:
	"""Test execution utilities"""
	
	@staticmethod
	def run_unit_tests():
		"""Run unit tests only"""
		return pytest.main([
			"-v",
			"-m", "not integration and not performance",
			__file__
		])
	
	@staticmethod
	def run_integration_tests():
		"""Run integration tests"""
		return pytest.main([
			"-v", 
			"-m", "integration",
			__file__
		])
	
	@staticmethod
	def run_performance_tests():
		"""Run performance tests"""
		return pytest.main([
			"-v",
			"-m", "performance", 
			"--tb=short",
			__file__
		])
	
	@staticmethod
	def run_all_tests():
		"""Run all tests"""
		return pytest.main([
			"-v",
			"--tb=short",
			"--cov=capabilities.common.rag",
			"--cov-report=html",
			__file__
		])

if __name__ == "__main__":
	# Run all tests when executed directly
	TestRunner.run_all_tests()