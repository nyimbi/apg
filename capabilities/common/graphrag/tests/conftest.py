"""
APG GraphRAG Capability - Test Configuration and Fixtures

Shared test fixtures, utilities, and configuration for comprehensive GraphRAG testing.

Author: Datacraft (nyimbi@gmail.com)
Copyright: © 2025 Datacraft
Website: www.datacraft.co.ke
"""

import pytest
import asyncio
import tempfile
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Generator
from unittest.mock import AsyncMock, MagicMock, patch
from uuid_extensions import uuid7str

# Import GraphRAG components
from ..database import GraphRAGDatabaseService
from ..service import GraphRAGService
from ..ollama_integration import OllamaClient, OllamaConfig, EmbeddingResult, GenerationResult
from ..hybrid_retrieval import HybridRetrievalEngine
from ..reasoning_engine import ReasoningEngine
from ..incremental_updates import IncrementalUpdateEngine
from ..collaborative_curation import CollaborativeCurationEngine
from ..contextual_intelligence import ContextualIntelligenceEngine
from ..visualization import GraphVisualizationEngine, VisualizationConfig
from ..views import (
	KnowledgeGraph, GraphEntity, GraphRelationship, DocumentSource,
	GraphRAGQuery, GraphRAGResponse, KnowledgeGraphRequest,
	DocumentProcessingRequest, QueryContext, RetrievalConfig, ReasoningConfig
)


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
	"""Create an instance of the default event loop for the test session"""
	loop = asyncio.get_event_loop_policy().new_event_loop()
	yield loop
	loop.close()


# ============================================================================
# MOCK CONFIGURATIONS
# ============================================================================

@pytest.fixture
def mock_ollama_config():
	"""Mock Ollama configuration for testing"""
	return OllamaConfig(
		base_url="http://localhost:11434",
		embedding_model="bge-m3",
		generation_models=["qwen3", "deepseek-r1"],
		max_concurrent_requests=5,
		embedding_dimensions=1024,
		max_context_length=8000
	)


@pytest.fixture
def mock_embedding_result():
	"""Mock embedding result for testing"""
	return EmbeddingResult(
		embeddings=[0.1] * 1024,  # 1024 dimensional embedding
		model_used="bge-m3",
		input_tokens=10,
		processing_time_ms=150.0,
		cache_hit=False
	)


@pytest.fixture
def mock_generation_result():
	"""Mock generation result for testing"""
	return GenerationResult(
		generated_text="This is a test response from the GraphRAG system.",
		model_used="qwen3",
		input_tokens=50,
		output_tokens=12,
		processing_time_ms=800.0,
		confidence_score=0.87,
		finish_reason="completed",
		cache_hit=False
	)


# ============================================================================
# DATABASE FIXTURES
# ============================================================================

@pytest.fixture
async def mock_db_service():
	"""Mock database service for testing"""
	db_service = AsyncMock(spec=GraphRAGDatabaseService)
	
	# Mock initialization
	db_service.initialize = AsyncMock()
	db_service.cleanup = AsyncMock()
	
	# Mock knowledge graph operations
	db_service.create_knowledge_graph = AsyncMock()
	db_service.get_knowledge_graph = AsyncMock()
	db_service.list_knowledge_graphs = AsyncMock(return_value=[])
	db_service.delete_knowledge_graph = AsyncMock(return_value=True)
	
	# Mock entity operations
	db_service.create_entity = AsyncMock()
	db_service.get_entity = AsyncMock()
	db_service.list_entities = AsyncMock(return_value=[])
	db_service.update_entity = AsyncMock()
	db_service.delete_entity = AsyncMock(return_value=True)
	
	# Mock relationship operations
	db_service.create_relationship = AsyncMock()
	db_service.get_relationship = AsyncMock()
	db_service.list_relationships = AsyncMock(return_value=[])
	db_service.delete_relationship = AsyncMock(return_value=True)
	
	# Mock query operations
	db_service.save_query_history = AsyncMock()
	db_service.get_query_history = AsyncMock(return_value=[])
	
	return db_service


# ============================================================================
# OLLAMA CLIENT FIXTURES
# ============================================================================

@pytest.fixture
async def mock_ollama_client(mock_ollama_config, mock_embedding_result, mock_generation_result):
	"""Mock Ollama client for testing"""
	client = AsyncMock(spec=OllamaClient)
	client.config = mock_ollama_config
	
	# Mock initialization
	client.initialize = AsyncMock()
	client.cleanup = AsyncMock()
	
	# Mock embedding operations
	client.generate_embedding = AsyncMock(return_value=mock_embedding_result)
	client.generate_batch_embeddings = AsyncMock(return_value=[mock_embedding_result])
	
	# Mock generation operations
	client.generate_graphrag_response = AsyncMock(return_value=mock_generation_result)
	client.generate_with_model_selection = AsyncMock(return_value=mock_generation_result)
	
	# Mock specialized methods
	client.generate_entity_mentions = AsyncMock(return_value=[])
	client.generate_source_attribution = AsyncMock(return_value=[])
	client.calculate_response_quality = AsyncMock()
	
	# Mock performance stats
	client.get_performance_stats = MagicMock(return_value={
		"embedding_generation": {"average_ms": 150, "count": 10},
		"text_generation": {"average_ms": 800, "count": 5}
	})
	
	return client


# ============================================================================
# SAMPLE DATA FIXTURES
# ============================================================================

@pytest.fixture
def sample_knowledge_graph():
	"""Sample knowledge graph for testing"""
	return KnowledgeGraph(
		knowledge_graph_id=uuid7str(),
		tenant_id="test_tenant",
		name="Test Knowledge Graph",
		description="A test knowledge graph for unit testing",
		domain="testing",
		entity_count=10,
		relationship_count=15,
		document_count=5,
		avg_entity_confidence=0.85,
		status="active",
		created_at=datetime.utcnow(),
		last_updated=datetime.utcnow(),
		metadata={"test": True}
	)


@pytest.fixture
def sample_graph_entity():
	"""Sample graph entity for testing"""
	return GraphEntity(
		canonical_entity_id=uuid7str(),
		tenant_id="test_tenant",
		knowledge_graph_id=uuid7str(),
		canonical_name="Test Entity",
		entity_type="person",
		aliases=["Test Person", "TP"],
		properties={"occupation": "tester", "age": 30},
		confidence_score=0.9,
		embeddings=[0.1] * 1024,
		created_at=datetime.utcnow(),
		updated_at=datetime.utcnow()
	)


@pytest.fixture
def sample_graph_relationship():
	"""Sample graph relationship for testing"""
	return GraphRelationship(
		canonical_relationship_id=uuid7str(),
		tenant_id="test_tenant",
		knowledge_graph_id=uuid7str(),
		source_entity_id=uuid7str(),
		target_entity_id=uuid7str(),
		relationship_type="works_with",
		strength=0.8,
		properties={"since": "2023"},
		confidence_score=0.85,
		created_at=datetime.utcnow(),
		updated_at=datetime.utcnow()
	)


@pytest.fixture
def sample_document_source():
	"""Sample document source for testing"""
	return DocumentSource(
		document_id=uuid7str(),
		tenant_id="test_tenant",
		knowledge_graph_id=uuid7str(),
		title="Test Document",
		content_preview="This is a test document for GraphRAG processing...",
		source_type="text",
		source_url="https://example.com/test-doc",
		processing_status="completed",
		entity_count=5,
		relationship_count=8,
		processing_metadata={"processing_time_ms": 2500},
		processed_at=datetime.utcnow(),
		created_at=datetime.utcnow()
	)


@pytest.fixture
def sample_graphrag_query():
	"""Sample GraphRAG query for testing"""
	return GraphRAGQuery(
		query_id=uuid7str(),
		tenant_id="test_tenant",
		knowledge_graph_id=uuid7str(),
		query_text="What is the relationship between entity A and entity B?",
		query_type="factual",
		query_embedding=[0.1] * 1024,
		context=QueryContext(
			user_id="test_user",
			session_id=uuid7str(),
			conversation_history=[],
			domain_context={"domain": "testing"},
			temporal_context={"timeframe": "current"}
		),
		retrieval_config=RetrievalConfig(),
		reasoning_config=ReasoningConfig(),
		explanation_level="detailed",
		max_hops=3,
		status="pending"
	)


@pytest.fixture
def sample_graphrag_response():
	"""Sample GraphRAG response for testing"""
	return GraphRAGResponse(
		query_id=uuid7str(),
		tenant_id="test_tenant",
		knowledge_graph_id=uuid7str(),
		answer="Entity A works with Entity B in a collaborative relationship since 2023.",
		confidence_score=0.87,
		processing_time_ms=1200.0,
		entities_used=[],
		relationships_used=[],
		reasoning_chain=None,
		evidence=[],
		entity_mentions=[],
		source_attribution=[],
		quality_indicators=None,
		metadata={"test_response": True},
		status="completed"
	)


# ============================================================================
# SERVICE FIXTURES
# ============================================================================

@pytest.fixture
async def mock_hybrid_retrieval(mock_db_service, mock_ollama_client):
	"""Mock hybrid retrieval engine for testing"""
	engine = AsyncMock(spec=HybridRetrievalEngine)
	engine.db_service = mock_db_service
	engine.ollama_client = mock_ollama_client
	
	engine.search_hybrid = AsyncMock(return_value=[])
	engine.search_vector_similarity = AsyncMock(return_value=[])
	engine.search_graph_traversal = AsyncMock(return_value=[])
	engine.rank_and_fuse_results = AsyncMock(return_value=[])
	
	return engine


@pytest.fixture
async def mock_reasoning_engine(mock_db_service, mock_ollama_client):
	"""Mock reasoning engine for testing"""
	engine = AsyncMock(spec=ReasoningEngine)
	engine.db_service = mock_db_service
	engine.ollama_client = mock_ollama_client
	
	engine.perform_multi_hop_reasoning = AsyncMock()
	engine.generate_reasoning_chain = AsyncMock()
	engine.validate_reasoning = AsyncMock(return_value=True)
	
	return engine


@pytest.fixture
async def mock_incremental_updates(mock_db_service, mock_ollama_client):
	"""Mock incremental update engine for testing"""
	engine = AsyncMock(spec=IncrementalUpdateEngine)
	engine.db_service = mock_db_service
	engine.ollama_client = mock_ollama_client
	
	engine.process_incremental_update = AsyncMock()
	engine.process_batch_updates = AsyncMock(return_value=[])
	engine.detect_and_merge_duplicate_entities = AsyncMock()
	
	return engine


@pytest.fixture
async def mock_collaborative_curation(mock_db_service, mock_ollama_client):
	"""Mock collaborative curation engine for testing"""
	engine = AsyncMock(spec=CollaborativeCurationEngine)
	engine.db_service = mock_db_service
	engine.ollama_client = mock_ollama_client
	
	engine.submit_curation_suggestion = AsyncMock()
	engine.review_curation_suggestion = AsyncMock()
	engine.get_curation_analytics = AsyncMock(return_value={})
	
	return engine


@pytest.fixture
async def mock_contextual_intelligence(mock_db_service, mock_ollama_client):
	"""Mock contextual intelligence engine for testing"""
	engine = AsyncMock(spec=ContextualIntelligenceEngine)
	engine.db_service = mock_db_service
	engine.ollama_client = mock_ollama_client
	
	engine.analyze_contextual_intelligence = AsyncMock()
	engine.perform_adaptive_learning = AsyncMock()
	engine.optimize_query_contextually = AsyncMock()
	engine.detect_semantic_drift = AsyncMock(return_value={"drift_detected": False})
	
	return engine


@pytest.fixture
async def mock_visualization_engine(mock_db_service):
	"""Mock visualization engine for testing"""
	engine = AsyncMock(spec=GraphVisualizationEngine)
	engine.db_service = mock_db_service
	engine.config = VisualizationConfig()
	
	engine.generate_graph_visualization = AsyncMock()
	engine.generate_subgraph_visualization = AsyncMock()
	engine.generate_temporal_visualization = AsyncMock()
	engine.export_visualization = AsyncMock()
	
	return engine


@pytest.fixture
async def mock_graphrag_service(
	mock_db_service, 
	mock_ollama_client,
	mock_hybrid_retrieval,
	mock_reasoning_engine,
	mock_incremental_updates,
	mock_collaborative_curation,
	mock_contextual_intelligence
):
	"""Mock GraphRAG service with all dependencies"""
	service = AsyncMock(spec=GraphRAGService)
	
	# Set dependencies
	service.db_service = mock_db_service
	service.ollama_client = mock_ollama_client
	service.hybrid_retrieval = mock_hybrid_retrieval
	service.reasoning_engine = mock_reasoning_engine
	service.incremental_updates = mock_incremental_updates
	service.collaborative_curation = mock_collaborative_curation
	service.contextual_intelligence = mock_contextual_intelligence
	
	# Mock core methods
	service.create_knowledge_graph = AsyncMock()
	service.process_document = AsyncMock()
	service.process_query = AsyncMock()
	service.get_analytics_overview = AsyncMock(return_value={})
	service.get_performance_analytics = AsyncMock(return_value={})
	service.get_knowledge_graph_statistics = AsyncMock(return_value={})
	
	return service


# ============================================================================
# REQUEST/RESPONSE FIXTURES
# ============================================================================

@pytest.fixture
def sample_knowledge_graph_request():
	"""Sample knowledge graph creation request"""
	return KnowledgeGraphRequest(
		tenant_id="test_tenant",
		name="Test Knowledge Graph",
		description="A test knowledge graph",
		domain="testing",
		metadata={"test": True}
	)


@pytest.fixture
def sample_document_processing_request():
	"""Sample document processing request"""
	return DocumentProcessingRequest(
		tenant_id="test_tenant",
		knowledge_graph_id=uuid7str(),
		title="Test Document",
		content="This is test content for document processing. It contains entities like John Doe and organizations like Acme Corp.",
		source_url="https://example.com/test",
		source_type="text",
		processing_options={"extract_entities": True, "extract_relationships": True}
	)


# ============================================================================
# TEST UTILITIES
# ============================================================================

@pytest.fixture
def test_tenant_id():
	"""Standard test tenant ID"""
	return "test_tenant"


@pytest.fixture
def test_user_id():
	"""Standard test user ID"""
	return "test_user"


@pytest.fixture
def test_session_id():
	"""Standard test session ID"""
	return uuid7str()


class TestDataGenerator:
	"""Utility class for generating test data"""
	
	@staticmethod
	def create_entities(count: int, knowledge_graph_id: str, tenant_id: str = "test_tenant") -> List[GraphEntity]:
		"""Generate multiple test entities"""
		entities = []
		entity_types = ["person", "organization", "location", "concept"]
		
		for i in range(count):
			entity = GraphEntity(
				canonical_entity_id=uuid7str(),
				tenant_id=tenant_id,
				knowledge_graph_id=knowledge_graph_id,
				canonical_name=f"Test Entity {i}",
				entity_type=entity_types[i % len(entity_types)],
				aliases=[f"Entity {i}", f"E{i}"],
				properties={"index": i, "test": True},
				confidence_score=0.8 + (i % 3) * 0.1,
				embeddings=[0.1 + i * 0.01] * 1024,
				created_at=datetime.utcnow() - timedelta(days=i),
				updated_at=datetime.utcnow()
			)
			entities.append(entity)
		
		return entities
	
	@staticmethod
	def create_relationships(
		count: int, 
		entity_ids: List[str], 
		knowledge_graph_id: str, 
		tenant_id: str = "test_tenant"
	) -> List[GraphRelationship]:
		"""Generate multiple test relationships"""
		relationships = []
		relationship_types = ["works_with", "located_in", "part_of", "related_to"]
		
		for i in range(count):
			if len(entity_ids) >= 2:
				source_id = entity_ids[i % len(entity_ids)]
				target_id = entity_ids[(i + 1) % len(entity_ids)]
				
				relationship = GraphRelationship(
					canonical_relationship_id=uuid7str(),
					tenant_id=tenant_id,
					knowledge_graph_id=knowledge_graph_id,
					source_entity_id=source_id,
					target_entity_id=target_id,
					relationship_type=relationship_types[i % len(relationship_types)],
					strength=0.7 + (i % 4) * 0.1,
					properties={"index": i, "test": True},
					confidence_score=0.75 + (i % 4) * 0.05,
					created_at=datetime.utcnow() - timedelta(days=i),
					updated_at=datetime.utcnow()
				)
				relationships.append(relationship)
		
		return relationships


@pytest.fixture
def test_data_generator():
	"""Test data generator utility"""
	return TestDataGenerator


# ============================================================================
# TEMPORARY RESOURCES
# ============================================================================

@pytest.fixture
def temp_directory():
	"""Temporary directory for test files"""
	with tempfile.TemporaryDirectory() as temp_dir:
		yield temp_dir


@pytest.fixture
def temp_file():
	"""Temporary file for testing"""
	with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
		temp_file.write("Test file content for GraphRAG testing")
		temp_file.flush()
		yield temp_file.name
	
	# Cleanup
	if os.path.exists(temp_file.name):
		os.unlink(temp_file.name)


# ============================================================================
# PERFORMANCE TEST FIXTURES
# ============================================================================

@pytest.fixture
def performance_test_config():
	"""Configuration for performance tests"""
	return {
		"max_execution_time_ms": 5000,
		"max_memory_usage_mb": 500,
		"concurrent_requests": 10,
		"test_iterations": 100
	}


# ============================================================================
# INTEGRATION TEST FIXTURES
# ============================================================================

@pytest.fixture
def integration_test_data():
	"""Data for integration tests"""
	knowledge_graph_id = uuid7str()
	tenant_id = "integration_test_tenant"
	
	# Generate comprehensive test data
	entities = TestDataGenerator.create_entities(20, knowledge_graph_id, tenant_id)
	entity_ids = [e.canonical_entity_id for e in entities]
	relationships = TestDataGenerator.create_relationships(30, entity_ids, knowledge_graph_id, tenant_id)
	
	return {
		"knowledge_graph_id": knowledge_graph_id,
		"tenant_id": tenant_id,
		"entities": entities,
		"relationships": relationships,
		"entity_ids": entity_ids
	}


# ============================================================================
# CLEANUP FIXTURES
# ============================================================================

@pytest.fixture(autouse=True)
async def cleanup_after_test():
	"""Automatic cleanup after each test"""
	yield
	# Cleanup code can be added here if needed
	# For now, mocks don't require cleanup


__all__ = [
	'mock_ollama_config',
	'mock_embedding_result', 
	'mock_generation_result',
	'mock_db_service',
	'mock_ollama_client',
	'sample_knowledge_graph',
	'sample_graph_entity',
	'sample_graph_relationship',
	'sample_document_source',
	'sample_graphrag_query',
	'sample_graphrag_response',
	'mock_hybrid_retrieval',
	'mock_reasoning_engine',
	'mock_incremental_updates',
	'mock_collaborative_curation',
	'mock_contextual_intelligence',
	'mock_visualization_engine',
	'mock_graphrag_service',
	'test_data_generator',
	'integration_test_data',
	'performance_test_config'
]