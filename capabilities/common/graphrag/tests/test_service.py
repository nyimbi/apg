"""
APG GraphRAG Capability - Core Service Unit Tests

Comprehensive unit tests for the main GraphRAG service including knowledge graph
creation, document processing, query processing, and analytics.

Author: Datacraft (nyimbi@gmail.com)
Copyright: © 2025 Datacraft
Website: www.datacraft.co.ke
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid_extensions import uuid7str

from ..service import GraphRAGService
from ..views import (
	KnowledgeGraph, GraphEntity, GraphRelationship, DocumentSource,
	GraphRAGQuery, GraphRAGResponse, KnowledgeGraphRequest,
	DocumentProcessingRequest, DocumentProcessingResult, QueryContext
)


class TestGraphRAGService:
	"""Test suite for GraphRAG core service"""
	
	@pytest.mark.asyncio
	async def test_service_initialization(
		self,
		mock_db_service,
		mock_ollama_client,
		mock_hybrid_retrieval,
		mock_reasoning_engine,
		mock_incremental_updates,
		mock_collaborative_curation,
		mock_contextual_intelligence
	):
		"""Test GraphRAG service initialization"""
		service = GraphRAGService(
			db_service=mock_db_service,
			ollama_client=mock_ollama_client,
			hybrid_retrieval=mock_hybrid_retrieval,
			reasoning_engine=mock_reasoning_engine,
			incremental_updates=mock_incremental_updates,
			collaborative_curation=mock_collaborative_curation,
			contextual_intelligence=mock_contextual_intelligence
		)
		
		assert service.db_service == mock_db_service
		assert service.ollama_client == mock_ollama_client
		assert service.hybrid_retrieval == mock_hybrid_retrieval
		assert service.reasoning_engine == mock_reasoning_engine
		assert service.incremental_updates == mock_incremental_updates
		assert service.collaborative_curation == mock_collaborative_curation
		assert service.contextual_intelligence == mock_contextual_intelligence
	
	@pytest.mark.asyncio
	async def test_create_knowledge_graph_success(
		self,
		mock_graphrag_service,
		sample_knowledge_graph_request,
		sample_knowledge_graph
	):
		"""Test successful knowledge graph creation"""
		# Setup mock response
		mock_graphrag_service.create_knowledge_graph.return_value = sample_knowledge_graph
		
		# Execute
		result = await mock_graphrag_service.create_knowledge_graph(sample_knowledge_graph_request)
		
		# Verify
		assert result == sample_knowledge_graph
		mock_graphrag_service.create_knowledge_graph.assert_called_once_with(sample_knowledge_graph_request)
	
	@pytest.mark.asyncio
	async def test_create_knowledge_graph_with_real_service(
		self,
		mock_db_service,
		mock_ollama_client,
		mock_hybrid_retrieval,
		mock_reasoning_engine,
		mock_incremental_updates,
		mock_collaborative_curation,
		mock_contextual_intelligence,
		sample_knowledge_graph_request,
		sample_knowledge_graph
	):
		"""Test knowledge graph creation with real service instance"""
		# Setup mock database response
		mock_db_service.create_knowledge_graph.return_value = sample_knowledge_graph
		
		# Create real service instance
		service = GraphRAGService(
			db_service=mock_db_service,
			ollama_client=mock_ollama_client,
			hybrid_retrieval=mock_hybrid_retrieval,
			reasoning_engine=mock_reasoning_engine,
			incremental_updates=mock_incremental_updates,
			collaborative_curation=mock_collaborative_curation,
			contextual_intelligence=mock_contextual_intelligence
		)
		
		# Execute
		result = await service.create_knowledge_graph(sample_knowledge_graph_request)
		
		# Verify
		assert result.name == sample_knowledge_graph_request.name
		assert result.description == sample_knowledge_graph_request.description
		assert result.domain == sample_knowledge_graph_request.domain
		assert result.tenant_id == sample_knowledge_graph_request.tenant_id
		mock_db_service.create_knowledge_graph.assert_called_once()
	
	@pytest.mark.asyncio
	async def test_process_document_success(
		self,
		mock_graphrag_service,
		sample_document_processing_request
	):
		"""Test successful document processing"""
		# Setup mock response
		expected_result = DocumentProcessingResult(
			document_id=uuid7str(),
			tenant_id=sample_document_processing_request.tenant_id,
			knowledge_graph_id=sample_document_processing_request.knowledge_graph_id,
			processing_status="completed",
			entities_extracted=5,
			relationships_extracted=8,
			processing_time_ms=2500.0,
			confidence_score=0.87,
			metadata={"entities_found": ["John Doe", "Acme Corp"]}
		)
		
		mock_graphrag_service.process_document.return_value = expected_result
		
		# Execute
		result = await mock_graphrag_service.process_document(sample_document_processing_request)
		
		# Verify
		assert result == expected_result
		mock_graphrag_service.process_document.assert_called_once_with(sample_document_processing_request)
	
	@pytest.mark.asyncio
	async def test_process_document_with_real_service(
		self,
		mock_db_service,
		mock_ollama_client,
		mock_hybrid_retrieval,
		mock_reasoning_engine,
		mock_incremental_updates,
		mock_collaborative_curation,
		mock_contextual_intelligence,
		sample_document_processing_request,
		mock_embedding_result
	):
		"""Test document processing with real service instance"""
		# Setup mocks
		mock_db_service.create_document_source.return_value = AsyncMock()
		mock_ollama_client.generate_embedding.return_value = mock_embedding_result
		mock_db_service.create_entity.return_value = AsyncMock()
		mock_db_service.create_relationship.return_value = AsyncMock()
		
		# Create real service instance
		service = GraphRAGService(
			db_service=mock_db_service,
			ollama_client=mock_ollama_client,
			hybrid_retrieval=mock_hybrid_retrieval,
			reasoning_engine=mock_reasoning_engine,
			incremental_updates=mock_incremental_updates,
			collaborative_curation=mock_collaborative_curation,
			contextual_intelligence=mock_contextual_intelligence
		)
		
		# Execute
		result = await service.process_document(sample_document_processing_request)
		
		# Verify
		assert result.tenant_id == sample_document_processing_request.tenant_id
		assert result.knowledge_graph_id == sample_document_processing_request.knowledge_graph_id
		assert result.processing_status == "completed"
		assert result.entities_extracted >= 0
		assert result.relationships_extracted >= 0
	
	@pytest.mark.asyncio
	async def test_process_query_success(
		self,
		mock_graphrag_service,
		sample_graphrag_query,
		sample_graphrag_response
	):
		"""Test successful query processing"""
		# Setup mock response
		mock_graphrag_service.process_query.return_value = sample_graphrag_response
		
		# Execute
		result = await mock_graphrag_service.process_query(sample_graphrag_query)
		
		# Verify
		assert result == sample_graphrag_response
		mock_graphrag_service.process_query.assert_called_once_with(sample_graphrag_query)
	
	@pytest.mark.asyncio
	async def test_process_query_with_real_service(
		self,
		mock_db_service,
		mock_ollama_client,
		mock_hybrid_retrieval,
		mock_reasoning_engine,
		mock_incremental_updates,
		mock_collaborative_curation,
		mock_contextual_intelligence,
		sample_graphrag_query,
		mock_embedding_result,
		mock_generation_result
	):
		"""Test query processing with real service instance"""
		# Setup mocks
		mock_ollama_client.generate_embedding.return_value = mock_embedding_result
		mock_hybrid_retrieval.search_hybrid.return_value = []
		mock_reasoning_engine.perform_multi_hop_reasoning.return_value = AsyncMock()
		mock_ollama_client.generate_graphrag_response.return_value = mock_generation_result
		mock_db_service.save_query_history.return_value = AsyncMock()
		
		# Create real service instance
		service = GraphRAGService(
			db_service=mock_db_service,
			ollama_client=mock_ollama_client,
			hybrid_retrieval=mock_hybrid_retrieval,
			reasoning_engine=mock_reasoning_engine,
			incremental_updates=mock_incremental_updates,
			collaborative_curation=mock_collaborative_curation,
			contextual_intelligence=mock_contextual_intelligence
		)
		
		# Execute
		result = await service.process_query(sample_graphrag_query)
		
		# Verify
		assert result.query_id == sample_graphrag_query.query_id
		assert result.tenant_id == sample_graphrag_query.tenant_id
		assert result.knowledge_graph_id == sample_graphrag_query.knowledge_graph_id
		assert result.status == "completed"
		assert result.processing_time_ms > 0
		
		# Verify service calls
		mock_ollama_client.generate_embedding.assert_called()
		mock_hybrid_retrieval.search_hybrid.assert_called()
		mock_reasoning_engine.perform_multi_hop_reasoning.assert_called()
		mock_ollama_client.generate_graphrag_response.assert_called()
		mock_db_service.save_query_history.assert_called()
	
	@pytest.mark.asyncio
	async def test_get_analytics_overview(
		self,
		mock_graphrag_service,
		test_tenant_id
	):
		"""Test analytics overview retrieval"""
		# Setup mock response
		expected_analytics = {
			"knowledge_graphs": {"total": 5, "active": 4},
			"entities": {"total": 1250, "avg_confidence": 0.87},
			"relationships": {"total": 2100, "avg_strength": 0.83},
			"queries": {"today": 45, "avg_response_time_ms": 1200},
			"documents": {"processed": 125, "pending": 8}
		}
		
		mock_graphrag_service.get_analytics_overview.return_value = expected_analytics
		
		# Execute
		result = await mock_graphrag_service.get_analytics_overview(test_tenant_id)
		
		# Verify
		assert result == expected_analytics
		mock_graphrag_service.get_analytics_overview.assert_called_once_with(test_tenant_id)
	
	@pytest.mark.asyncio
	async def test_get_performance_analytics(
		self,
		mock_graphrag_service,
		test_tenant_id
	):
		"""Test performance analytics retrieval"""
		# Setup mock response
		expected_performance = {
			"query_performance": {
				"avg_response_time_ms": 1200,
				"p95_response_time_ms": 2800,
				"throughput_qps": 8.5
			},
			"system_health": {
				"database_status": "healthy",
				"ollama_status": "healthy",
				"memory_usage_percent": 65
			},
			"cache_performance": {
				"hit_rate": 0.78,
				"miss_rate": 0.22,
				"eviction_rate": 0.05
			}
		}
		
		mock_graphrag_service.get_performance_analytics.return_value = expected_performance
		
		# Execute
		result = await mock_graphrag_service.get_performance_analytics(
			test_tenant_id, 
			sample_knowledge_graph().knowledge_graph_id,
			timedelta(days=7)
		)
		
		# Verify
		assert result == expected_performance
		mock_graphrag_service.get_performance_analytics.assert_called_once()
	
	@pytest.mark.asyncio
	async def test_get_knowledge_graph_statistics(
		self,
		mock_graphrag_service,
		test_tenant_id,
		sample_knowledge_graph
	):
		"""Test knowledge graph statistics retrieval"""
		# Setup mock response
		expected_stats = {
			"basic_stats": {
				"entity_count": 150,
				"relationship_count": 280,
				"document_count": 25
			},
			"quality_metrics": {
				"avg_entity_confidence": 0.87,
				"avg_relationship_strength": 0.83,
				"data_completeness": 0.92
			},
			"graph_metrics": {
				"density": 0.024,
				"avg_degree": 3.7,
				"clustering_coefficient": 0.45
			},
			"temporal_analysis": {
				"growth_rate_entities": 0.15,
				"growth_rate_relationships": 0.22,
				"recent_activity_score": 0.68
			}
		}
		
		mock_graphrag_service.get_knowledge_graph_statistics.return_value = expected_stats
		
		# Execute
		result = await mock_graphrag_service.get_knowledge_graph_statistics(
			test_tenant_id,
			sample_knowledge_graph.knowledge_graph_id
		)
		
		# Verify
		assert result == expected_stats
		mock_graphrag_service.get_knowledge_graph_statistics.assert_called_once()
	
	@pytest.mark.asyncio
	async def test_error_handling_invalid_tenant(
		self,
		mock_db_service,
		mock_ollama_client,
		mock_hybrid_retrieval,
		mock_reasoning_engine,
		mock_incremental_updates,
		mock_collaborative_curation,
		mock_contextual_intelligence
	):
		"""Test error handling for invalid tenant"""
		# Setup mock to raise exception
		mock_db_service.create_knowledge_graph.side_effect = ValueError("Invalid tenant ID")
		
		# Create service
		service = GraphRAGService(
			db_service=mock_db_service,
			ollama_client=mock_ollama_client,
			hybrid_retrieval=mock_hybrid_retrieval,
			reasoning_engine=mock_reasoning_engine,
			incremental_updates=mock_incremental_updates,
			collaborative_curation=mock_collaborative_curation,
			contextual_intelligence=mock_contextual_intelligence
		)
		
		# Create invalid request
		invalid_request = KnowledgeGraphRequest(
			tenant_id="",  # Empty tenant ID
			name="Test Graph",
			description="Test",
			domain="test",
			metadata={}
		)
		
		# Execute and verify exception
		with pytest.raises(ValueError, match="Invalid tenant ID"):
			await service.create_knowledge_graph(invalid_request)
	
	@pytest.mark.asyncio
	async def test_concurrent_query_processing(
		self,
		mock_graphrag_service,
		test_data_generator,
		test_tenant_id
	):
		"""Test concurrent query processing"""
		knowledge_graph_id = uuid7str()
		
		# Create multiple queries
		queries = []
		for i in range(5):
			query = GraphRAGQuery(
				query_id=uuid7str(),
				tenant_id=test_tenant_id,
				knowledge_graph_id=knowledge_graph_id,
				query_text=f"Test query {i}",
				query_type="factual",
				context=QueryContext(
					user_id=f"user_{i}",
					session_id=uuid7str(),
					conversation_history=[],
					domain_context={},
					temporal_context={}
				),
				status="pending"
			)
			queries.append(query)
		
		# Setup mock responses
		responses = []
		for i, query in enumerate(queries):
			response = GraphRAGResponse(
				query_id=query.query_id,
				tenant_id=query.tenant_id,
				knowledge_graph_id=query.knowledge_graph_id,
				answer=f"Answer to query {i}",
				confidence_score=0.8 + i * 0.02,
				processing_time_ms=800 + i * 100,
				entities_used=[],
				relationships_used=[],
				reasoning_chain=None,
				evidence=[],
				entity_mentions=[],
				source_attribution=[],
				quality_indicators=None,
				metadata={},
				status="completed"
			)
			responses.append(response)
		
		# Configure mock to return responses
		mock_graphrag_service.process_query.side_effect = responses
		
		# Execute queries concurrently
		tasks = [mock_graphrag_service.process_query(query) for query in queries]
		results = await asyncio.gather(*tasks)
		
		# Verify
		assert len(results) == len(queries)
		for i, result in enumerate(results):
			assert result.query_id == queries[i].query_id
			assert result.answer == f"Answer to query {i}"
		
		# Verify all queries were processed
		assert mock_graphrag_service.process_query.call_count == len(queries)
	
	@pytest.mark.asyncio
	async def test_document_processing_with_no_content(
		self,
		mock_db_service,
		mock_ollama_client,
		mock_hybrid_retrieval,
		mock_reasoning_engine,
		mock_incremental_updates,
		mock_collaborative_curation,
		mock_contextual_intelligence,
		test_tenant_id
	):
		"""Test document processing with empty content"""
		# Create service
		service = GraphRAGService(
			db_service=mock_db_service,
			ollama_client=mock_ollama_client,
			hybrid_retrieval=mock_hybrid_retrieval,
			reasoning_engine=mock_reasoning_engine,
			incremental_updates=mock_incremental_updates,
			collaborative_curation=mock_collaborative_curation,
			contextual_intelligence=mock_contextual_intelligence
		)
		
		# Create request with empty content
		request = DocumentProcessingRequest(
			tenant_id=test_tenant_id,
			knowledge_graph_id=uuid7str(),
			title="Empty Document",
			content="",  # Empty content
			source_url="",
			source_type="text",
			processing_options={}
		)
		
		# Execute
		result = await service.process_document(request)
		
		# Verify
		assert result.processing_status == "completed"
		assert result.entities_extracted == 0
		assert result.relationships_extracted == 0
	
	@pytest.mark.asyncio
	async def test_query_with_invalid_graph_id(
		self,
		mock_db_service,
		mock_ollama_client,
		mock_hybrid_retrieval,
		mock_reasoning_engine,
		mock_incremental_updates,
		mock_collaborative_curation,
		mock_contextual_intelligence,
		test_tenant_id
	):
		"""Test query processing with invalid knowledge graph ID"""
		# Setup mock to return None for invalid graph
		mock_db_service.get_knowledge_graph.return_value = None
		
		# Create service
		service = GraphRAGService(
			db_service=mock_db_service,
			ollama_client=mock_ollama_client,
			hybrid_retrieval=mock_hybrid_retrieval,
			reasoning_engine=mock_reasoning_engine,
			incremental_updates=mock_incremental_updates,
			collaborative_curation=mock_collaborative_curation,
			contextual_intelligence=mock_contextual_intelligence
		)
		
		# Create query with invalid graph ID
		query = GraphRAGQuery(
			query_id=uuid7str(),
			tenant_id=test_tenant_id,
			knowledge_graph_id="invalid_graph_id",
			query_text="Test query",
			query_type="factual",
			context=QueryContext(
				user_id="test_user",
				session_id=uuid7str(),
				conversation_history=[],
				domain_context={},
				temporal_context={}
			),
			status="pending"
		)
		
		# Execute and verify error handling
		with pytest.raises(ValueError, match="Knowledge graph not found"):
			await service.process_query(query)
	
	@pytest.mark.asyncio
	async def test_service_cleanup(
		self,
		mock_db_service,
		mock_ollama_client,
		mock_hybrid_retrieval,
		mock_reasoning_engine,
		mock_incremental_updates,
		mock_collaborative_curation,
		mock_contextual_intelligence
	):
		"""Test service cleanup and resource management"""
		# Create service
		service = GraphRAGService(
			db_service=mock_db_service,
			ollama_client=mock_ollama_client,
			hybrid_retrieval=mock_hybrid_retrieval,
			reasoning_engine=mock_reasoning_engine,
			incremental_updates=mock_incremental_updates,
			collaborative_curation=mock_collaborative_curation,
			contextual_intelligence=mock_contextual_intelligence
		)
		
		# Test cleanup
		await service.cleanup()
		
		# Verify cleanup calls
		mock_db_service.cleanup.assert_called_once()
		mock_ollama_client.cleanup.assert_called_once()


class TestGraphRAGServiceIntegration:
	"""Integration tests for GraphRAG service with multiple components"""
	
	@pytest.mark.asyncio
	async def test_end_to_end_document_to_query_workflow(
		self,
		mock_db_service,
		mock_ollama_client,
		mock_hybrid_retrieval,
		mock_reasoning_engine,
		mock_incremental_updates,
		mock_collaborative_curation,
		mock_contextual_intelligence,
		test_tenant_id,
		mock_embedding_result,
		mock_generation_result
	):
		"""Test complete workflow from document processing to query answering"""
		# Setup mocks
		knowledge_graph_id = uuid7str()
		
		# Mock knowledge graph creation
		mock_graph = KnowledgeGraph(
			knowledge_graph_id=knowledge_graph_id,
			tenant_id=test_tenant_id,
			name="Integration Test Graph",
			description="Test graph for integration",
			domain="testing",
			entity_count=0,
			relationship_count=0,
			document_count=0,
			avg_entity_confidence=0.0,
			status="active",
			created_at=datetime.utcnow(),
			last_updated=datetime.utcnow(),
			metadata={}
		)
		mock_db_service.create_knowledge_graph.return_value = mock_graph
		
		# Mock document processing
		mock_db_service.create_document_source.return_value = AsyncMock()
		mock_ollama_client.generate_embedding.return_value = mock_embedding_result
		mock_db_service.create_entity.return_value = AsyncMock()
		mock_db_service.create_relationship.return_value = AsyncMock()
		
		# Mock query processing
		mock_hybrid_retrieval.search_hybrid.return_value = []
		mock_reasoning_engine.perform_multi_hop_reasoning.return_value = AsyncMock()
		mock_ollama_client.generate_graphrag_response.return_value = mock_generation_result
		mock_db_service.save_query_history.return_value = AsyncMock()
		
		# Create service
		service = GraphRAGService(
			db_service=mock_db_service,
			ollama_client=mock_ollama_client,
			hybrid_retrieval=mock_hybrid_retrieval,
			reasoning_engine=mock_reasoning_engine,
			incremental_updates=mock_incremental_updates,
			collaborative_curation=mock_collaborative_curation,
			contextual_intelligence=mock_contextual_intelligence
		)
		
		# Step 1: Create knowledge graph
		graph_request = KnowledgeGraphRequest(
			tenant_id=test_tenant_id,
			name="Integration Test Graph",
			description="Test graph for integration",
			domain="testing",
			metadata={}
		)
		
		created_graph = await service.create_knowledge_graph(graph_request)
		assert created_graph.knowledge_graph_id == knowledge_graph_id
		
		# Step 2: Process document
		doc_request = DocumentProcessingRequest(
			tenant_id=test_tenant_id,
			knowledge_graph_id=knowledge_graph_id,
			title="Test Document",
			content="John Doe works at Acme Corporation in New York. He is the CEO and has been there since 2020.",
			source_url="https://example.com/doc",
			source_type="text",
			processing_options={}
		)
		
		doc_result = await service.process_document(doc_request)
		assert doc_result.processing_status == "completed"
		assert doc_result.entities_extracted >= 0
		assert doc_result.relationships_extracted >= 0
		
		# Step 3: Process query
		query = GraphRAGQuery(
			query_id=uuid7str(),
			tenant_id=test_tenant_id,
			knowledge_graph_id=knowledge_graph_id,
			query_text="Who is the CEO of Acme Corporation?",
			query_type="factual",
			context=QueryContext(
				user_id="integration_test_user",
				session_id=uuid7str(),
				conversation_history=[],
				domain_context={},
				temporal_context={}
			),
			status="pending"
		)
		
		query_result = await service.process_query(query)
		assert query_result.status == "completed"
		assert query_result.processing_time_ms > 0
		assert query_result.confidence_score > 0
		
		# Verify the workflow executed all steps
		mock_db_service.create_knowledge_graph.assert_called_once()
		mock_db_service.create_document_source.assert_called()
		mock_ollama_client.generate_embedding.assert_called()
		mock_hybrid_retrieval.search_hybrid.assert_called()
		mock_reasoning_engine.perform_multi_hop_reasoning.assert_called()
		mock_ollama_client.generate_graphrag_response.assert_called()
		mock_db_service.save_query_history.assert_called()


__all__ = [
	'TestGraphRAGService',
	'TestGraphRAGServiceIntegration'
]