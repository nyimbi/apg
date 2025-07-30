"""
APG Document Content Management - Service Tests

Comprehensive tests for the core document management service including
all 10 revolutionary capabilities.

Copyright © 2025 Datacraft
Author: Nyimbi Odero <nyimbi@gmail.com>
Website: www.datacraft.co.ke
"""

import asyncio
import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from ..service import DocumentManagementService
from ..models import (
	DCMDocument, DCMDocumentType, DCMContentFormat,
	DCMIntelligentProcessing, DCMSemanticSearch, DCMContentIntelligence,
	DCMGenerativeAI, DCMPredictiveAnalytics, DCMRetentionPolicy
)


class TestDocumentManagementService:
	"""Test suite for DocumentManagementService"""
	
	@pytest.fixture
	def service(self):
		"""Create service instance for testing"""
		return DocumentManagementService()
	
	@pytest.fixture
	def sample_document_data(self):
		"""Sample document data for testing"""
		return {
			'name': 'Test Document',
			'title': 'Test Document Title',
			'description': 'Test document description',
			'document_type': DCMDocumentType.TEXT_DOCUMENT,
			'content_format': DCMContentFormat.PDF,
			'file_name': 'test.pdf',
			'file_size': 1024,
			'mime_type': 'application/pdf',
			'keywords': ['test', 'document'],
			'categories': ['testing']
		}
	
	@pytest.fixture
	def sample_document(self, sample_document_data):
		"""Create sample document for testing"""
		return DCMDocument(
			id='test-doc-123',
			tenant_id='test-tenant',
			created_by='test-user',
			updated_by='test-user',
			**sample_document_data
		)


class TestDocumentCreation:
	"""Test document creation with AI processing"""
	
	@pytest.mark.asyncio
	async def test_create_document_with_ai_processing(self, service, sample_document_data):
		"""Test document creation with full AI processing"""
		file_path = '/tmp/test.pdf'
		user_id = 'test-user'
		tenant_id = 'test-tenant'
		
		# Mock the AI processing components
		service.idp_processor.process_document = AsyncMock(return_value=MagicMock(
			extracted_data={'text_content': 'Sample document content'},
			confidence_score=0.95
		))
		
		service.classification_engine.classify_document = AsyncMock(return_value=MagicMock(
			ai_classification={'document_type': {'primary_type': 'contract', 'confidence': 0.9}},
			content_summary='Test document summary',
			related_concepts=['contract', 'legal', 'agreement']
		))
		
		service.predictive_engine.generate_predictions = AsyncMock(return_value=MagicMock(
			content_value_score=0.8,
			business_impact_score=0.7
		))
		
		# Test document creation
		document = await service.create_document(
			sample_document_data, file_path, user_id, tenant_id, process_ai=True
		)
		
		# Assertions
		assert document.name == 'Test Document'
		assert document.tenant_id == tenant_id
		assert document.created_by == user_id
		assert service.idp_processor.process_document.called
		assert service.classification_engine.classify_document.called
		assert service.predictive_engine.generate_predictions.called
	
	@pytest.mark.asyncio
	async def test_create_document_without_ai_processing(self, service, sample_document_data):
		"""Test document creation without AI processing"""
		file_path = '/tmp/test.pdf'
		user_id = 'test-user'
		tenant_id = 'test-tenant'
		
		# Test document creation without AI
		document = await service.create_document(
			sample_document_data, file_path, user_id, tenant_id, process_ai=False
		)
		
		# Assertions
		assert document.name == 'Test Document'
		assert document.tenant_id == tenant_id
		assert document.created_by == user_id


class TestSemanticSearch:
	"""Test semantic search functionality"""
	
	@pytest.mark.asyncio
	async def test_semantic_search_with_results(self, service):
		"""Test semantic search with matching results"""
		query = 'contract agreements legal documents'
		user_id = 'test-user'
		tenant_id = 'test-tenant'
		search_options = {}
		
		# Mock search engine
		service.search_engine.search_documents = AsyncMock(return_value=MagicMock(
			matching_documents=['doc1', 'doc2', 'doc3'],
			semantic_similarity_scores=[0.95, 0.87, 0.76],
			intent_classification={'intent': 'find_legal_documents', 'confidence': 0.92},
			search_time_ms=245,
			confidence_score=0.88
		))
		
		# Perform search
		result = await service.search_documents(query, user_id, tenant_id, search_options)
		
		# Assertions
		assert len(result.matching_documents) == 3
		assert result.intent_classification['intent'] == 'find_legal_documents'
		assert result.confidence_score == 0.88
		assert service.search_engine.search_documents.called
	
	@pytest.mark.asyncio
	async def test_semantic_search_no_results(self, service):
		"""Test semantic search with no matching results"""
		query = 'nonexistent content xyz'
		user_id = 'test-user'
		tenant_id = 'test-tenant'
		search_options = {}
		
		# Mock search engine with no results
		service.search_engine.search_documents = AsyncMock(return_value=MagicMock(
			matching_documents=[],
			semantic_similarity_scores=[],
			intent_classification={'intent': 'general_search', 'confidence': 0.3},
			search_time_ms=123,
			confidence_score=0.1
		))
		
		# Perform search
		result = await service.search_documents(query, user_id, tenant_id, search_options)
		
		# Assertions
		assert len(result.matching_documents) == 0
		assert result.confidence_score == 0.1


class TestGenerativeAI:
	"""Test generative AI interaction capabilities"""
	
	@pytest.mark.asyncio
	async def test_genai_summarization(self, service, sample_document):
		"""Test document summarization using generative AI"""
		user_prompt = 'Summarize this document'
		interaction_type = 'summarize'
		user_id = 'test-user'
		
		# Mock GenAI engine
		service.genai_engine.process_interaction = AsyncMock(return_value=MagicMock(
			interaction_type='summarize',
			user_prompt=user_prompt,
			genai_response='This document is a comprehensive test document covering various aspects...',
			confidence_score=0.91,
			processing_time_ms=1245,
			model_version='apg-genai-v1'
		))
		
		# Test interaction
		result = await service.interact_with_content(
			sample_document.id, user_prompt, interaction_type, user_id, [], {}
		)
		
		# Assertions
		assert result.interaction_type == 'summarize'
		assert 'comprehensive test document' in result.genai_response
		assert result.confidence_score == 0.91
		assert service.genai_engine.process_interaction.called
	
	@pytest.mark.asyncio
	async def test_genai_question_answering(self, service, sample_document):
		"""Test question answering using generative AI"""
		user_prompt = 'What is this document about?'
		interaction_type = 'qa'
		user_id = 'test-user'
		
		# Mock GenAI engine
		service.genai_engine.process_interaction = AsyncMock(return_value=MagicMock(
			interaction_type='qa',
			user_prompt=user_prompt,
			genai_response='This document is about testing the document management system capabilities.',
			confidence_score=0.87,
			processing_time_ms=892,
			model_version='apg-genai-v1'
		))
		
		# Test interaction
		result = await service.interact_with_content(
			sample_document.id, user_prompt, interaction_type, user_id, [], {}
		)
		
		# Assertions
		assert result.interaction_type == 'qa'
		assert 'testing the document management system' in result.genai_response
		assert result.confidence_score == 0.87


class TestRetentionEngine:
	"""Test smart retention and disposition capabilities"""
	
	@pytest.mark.asyncio
	async def test_retention_policy_application(self, service):
		"""Test applying retention policy to documents"""
		policy_id = 'test-policy-123'
		document_ids = ['doc1', 'doc2', 'doc3']
		
		# Mock retention engine
		service.retention_engine.apply_retention_policy = AsyncMock(return_value={
			'policy_id': policy_id,
			'documents_processed': 3,
			'actions_taken': [
				{'document_id': 'doc1', 'action': 'archive', 'status': 'completed'},
				{'document_id': 'doc2', 'action': 'archive', 'status': 'completed'},
				{'document_id': 'doc3', 'action': 'review', 'status': 'completed'}
			],
			'errors': [],
			'summary': {'success_rate': 1.0, 'total_documents': 3}
		})
		
		# Test policy application
		result = await service.apply_retention_policy(policy_id, document_ids)
		
		# Assertions
		assert result['policy_id'] == policy_id
		assert result['documents_processed'] == 3
		assert len(result['actions_taken']) == 3
		assert result['summary']['success_rate'] == 1.0


class TestBlockchainProvenance:
	"""Test blockchain document provenance capabilities"""
	
	@pytest.mark.asyncio
	async def test_document_provenance_verification(self, service):
		"""Test blockchain provenance verification"""
		document_id = 'test-doc-123'
		
		# Mock blockchain client
		service.apg_blockchain_client = MagicMock()
		service.apg_blockchain_client.verify_document = AsyncMock(return_value={
			'document_id': document_id,
			'verified': True,
			'transaction_hash': '0x1234567890abcdef',
			'block_number': 12345,
			'timestamp': datetime.utcnow().isoformat(),
			'integrity_status': 'verified'
		})
		
		# Test verification
		result = await service.verify_document_provenance(document_id)
		
		# Assertions
		assert result['document_id'] == document_id
		assert result['verified'] is True
		assert result['integrity_status'] == 'verified'
		assert service.apg_blockchain_client.verify_document.called
	
	@pytest.mark.asyncio
	async def test_document_provenance_no_blockchain(self, service):
		"""Test provenance verification without blockchain client"""
		document_id = 'test-doc-123'
		
		# No blockchain client configured
		service.apg_blockchain_client = None
		
		# Test verification should raise error
		with pytest.raises(ValueError, match="Blockchain client not configured"):
			await service.verify_document_provenance(document_id)


class TestPredictiveAnalytics:
	"""Test predictive analytics capabilities"""
	
	@pytest.mark.asyncio
	async def test_content_value_prediction(self, service, sample_document):
		"""Test content value and risk prediction"""
		content_intelligence = MagicMock()
		prediction_types = ['value', 'risk', 'usage', 'lifecycle']
		
		# Mock predictive engine
		service.predictive_engine.generate_predictions = AsyncMock(return_value=MagicMock(
			content_value_score=0.85,
			business_impact_score=0.78,
			risk_probability={'compliance_violation': 0.12, 'security_breach': 0.08},
			obsolescence_probability=0.25,
			prediction_confidence=0.82
		))
		
		# Test prediction generation
		result = await service.predictive_engine.generate_predictions(
			sample_document, content_intelligence, prediction_types
		)
		
		# Assertions
		assert result.content_value_score == 0.85
		assert result.business_impact_score == 0.78
		assert result.prediction_confidence == 0.82
		assert service.predictive_engine.generate_predictions.called


class TestOCRCapabilities:
	"""Test OCR (Optical Character Recognition) capabilities"""
	
	@pytest.mark.asyncio
	async def test_document_ocr_processing(self, service, sample_document):
		"""Test OCR processing for a document"""
		
		# Mock OCR service
		service.ocr_service.process_document_ocr = AsyncMock(return_value={
			'document_id': sample_document.id,
			'total_pages': 1,
			'combined_text': 'This is extracted text from the document.',
			'combined_confidence': 0.95,
			'processing_summary': {
				'total_words': 8,
				'total_characters': 45,
				'total_lines': 1,
				'average_confidence': 0.95,
				'languages_detected': ['eng'],
				'total_processing_time_ms': 2340
			}
		})
		
		# Test OCR processing
		result = await service.process_document_ocr(
			document_id=sample_document.id,
			file_path='/tmp/test.pdf',
			user_id='test-user',
			tenant_id='test-tenant'
		)
		
		# Assertions
		assert result['document_id'] == sample_document.id
		assert 'combined_text' in result
		assert result['combined_confidence'] == 0.95
		assert service.ocr_service.process_document_ocr.called
	
	@pytest.mark.asyncio
	async def test_batch_ocr_processing(self, service):
		"""Test batch OCR processing"""
		
		document_ids = ['doc1', 'doc2', 'doc3']
		
		# Mock OCR processing for individual documents
		service.process_document_ocr = AsyncMock(return_value={
			'document_id': 'doc1',
			'combined_text': 'Extracted text content',
			'combined_confidence': 0.9
		})
		
		# Test batch processing
		result = await service.batch_ocr_processing(
			document_ids=document_ids,
			user_id='test-user',
			tenant_id='test-tenant',
			batch_name='Test Batch'
		)
		
		# Assertions
		assert result['batch_name'] == 'Test Batch'
		assert result['total_documents'] == 3
		assert result['processed_documents'] == 3
	
	@pytest.mark.asyncio
	async def test_get_supported_ocr_languages(self, service):
		"""Test retrieving supported OCR languages"""
		
		# Mock supported languages
		service.ocr_service.ocr_engine.get_supported_languages = AsyncMock(return_value=[
			'eng', 'fra', 'deu', 'spa', 'ita'
		])
		
		# Test language retrieval
		languages = await service.get_supported_ocr_languages()
		
		# Assertions
		assert 'eng' in languages
		assert 'fra' in languages
		assert len(languages) == 5
	
	@pytest.mark.asyncio
	async def test_ocr_configuration_update(self, service):
		"""Test OCR configuration updates"""
		
		config_data = {
			'tesseract_cmd': 'tesseract',
			'languages': ['eng', 'fra'],
			'dpi': 300,
			'enable_preprocessing': True
		}
		
		# Test configuration update
		result = await service.update_ocr_configuration(
			config_name='test_config',
			config_data=config_data,
			user_id='test-user',
			tenant_id='test-tenant'
		)
		
		# Assertions
		assert result['config_name'] == 'test_config'
		assert result['status'] == 'updated'
		assert 'updated_at' in result


class TestComprehensiveAnalytics:
	"""Test comprehensive analytics capabilities"""
	
	@pytest.mark.asyncio
	async def test_comprehensive_analytics_retrieval(self, service):
		"""Test comprehensive analytics across all capabilities"""
		# Mock analytics from all engines
		service.idp_processor.get_processing_analytics = AsyncMock(return_value={
			'documents_processed': 1250,
			'accuracy_rate': 0.987,
			'processing_time_avg': 2.3
		})
		
		service.search_engine.get_search_analytics = AsyncMock(return_value={
			'searches_performed': 5670,
			'average_response_time': 245,
			'success_rate': 0.94
		})
		
		service.classification_engine.get_classification_analytics = AsyncMock(return_value={
			'classifications_performed': 1150,
			'average_confidence': 0.89,
			'accuracy_score': 0.92
		})
		
		# Test analytics retrieval
		analytics = await service.get_comprehensive_analytics()
		
		# Assertions
		assert 'service_statistics' in analytics
		assert 'idp_analytics' in analytics
		assert 'search_analytics' in analytics
		assert 'classification_analytics' in analytics
		assert 'ocr_analytics' in analytics
		assert analytics['idp_analytics']['documents_processed'] == 1250
		assert analytics['search_analytics']['searches_performed'] == 5670
	
	@pytest.mark.asyncio
	async def test_health_check(self, service):
		"""Test comprehensive health check"""
		# Test health check
		health_status = await service.health_check()
		
		# Assertions
		assert 'overall_status' in health_status
		assert 'services' in health_status
		assert 'apg_integrations' in health_status
		assert 'statistics' in health_status
		assert health_status['overall_status'] == 'healthy'
		assert 'idp_processor' in health_status['services']
		assert 'search_engine' in health_status['services']


class TestIntegrationScenarios:
	"""Integration tests for complex document management scenarios"""
	
	@pytest.mark.asyncio
	async def test_complete_document_lifecycle(self, service, sample_document_data):
		"""Test complete document lifecycle from creation to retention"""
		file_path = '/tmp/test.pdf'
		user_id = 'test-user'
		tenant_id = 'test-tenant'
		
		# Mock all required components
		service.idp_processor.process_document = AsyncMock(return_value=MagicMock(
			extracted_data={'text_content': 'Contract content'},
			confidence_score=0.95
		))
		
		service.classification_engine.classify_document = AsyncMock(return_value=MagicMock(
			ai_classification={'document_type': {'primary_type': 'contract'}},
			content_summary='Legal contract document',
			related_concepts=['contract', 'legal'],
			compliance_flags=['SOX_APPLICABLE']
		))
		
		service.retention_engine.analyze_document_retention = AsyncMock(return_value={
			'retention_recommendation': {
				'action': 'archive',
				'retention_days': 2555,
				'confidence': 0.9
			}
		})
		
		# 1. Create document with AI processing
		document = await service.create_document(
			sample_document_data, file_path, user_id, tenant_id, process_ai=True
		)
		
		# 2. Perform semantic search
		search_service_mock = AsyncMock(return_value=MagicMock(
			matching_documents=[document.id],
			confidence_score=0.9
		))
		service.search_engine.search_documents = search_service_mock
		
		search_result = await service.search_documents(
			'legal contract', user_id, tenant_id, {}
		)
		
		# 3. GenAI interaction
		service.genai_engine.process_interaction = AsyncMock(return_value=MagicMock(
			genai_response='This is a legal contract document.',
			confidence_score=0.88
		))
		
		genai_result = await service.interact_with_content(
			document.id, 'What type of document is this?', 'qa', user_id, [], {}
		)
		
		# Assertions for complete lifecycle
		assert document.name == 'Test Document'
		assert len(search_result.matching_documents) > 0
		assert 'legal contract' in genai_result.genai_response
		
		# Verify all services were called
		assert service.idp_processor.process_document.called
		assert service.classification_engine.classify_document.called
		assert service.retention_engine.analyze_document_retention.called
		assert service.search_engine.search_documents.called
		assert service.genai_engine.process_interaction.called


if __name__ == '__main__':
	pytest.main([__file__])