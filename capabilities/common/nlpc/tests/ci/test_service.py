"""
Comprehensive unit tests for NLPC service functionality.

Tests all service methods, NLP processing functions, performance optimization,
and integration points following APG testing standards.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List
from uuid_extensions import uuid7str
from unittest.mock import AsyncMock, MagicMock, patch
import json
import time

from ...service import NLPCService
from ...models import (
	NLPDocument, ProcessingRequest, ProcessingResult, ContextSession,
	NLPTask, ProcessingStatus, LanguageCode, PriorityLevel, ModelType
)
from ..conftest import (
	assert_processing_result_valid, assert_document_valid, assert_security_context_valid,
	create_test_document, create_test_request, wait_for_async_operation
)

class TestNLPCServiceInitialization:
	"""Test NLPC service initialization and setup"""
	
	async def test_service_creation(self, test_config):
		"""Test basic service creation"""
		service = NLPCService(tenant_id=test_config['tenant_id'])
		
		assert service.tenant_id == test_config['tenant_id']
		assert service.initialized == False
		assert service.performance_cache is not None
		assert service.context_sessions == {}
	
	async def test_service_initialization(self, test_config):
		"""Test service initialization with models"""
		service = NLPCService(tenant_id=test_config['tenant_id'])
		
		# Mock heavy model loading for tests
		with patch.object(service, '_load_spacy_models', new_callable=AsyncMock) as mock_spacy, \
			 patch.object(service, '_load_nltk_models', new_callable=AsyncMock) as mock_nltk:
			
			await service.initialize_nlp_models(test_config['models'])
			
			assert service.initialized == True
			mock_spacy.assert_called_once()
			mock_nltk.assert_called_once()
	
	async def test_service_health_check(self, nlpc_service):
		"""Test service health check functionality"""
		health = await nlpc_service._check_service_health()
		
		assert isinstance(health, dict)
		assert 'status' in health
		assert 'models_loaded' in health
		assert 'cache_enabled' in health
		assert 'timestamp' in health
		
		# Health status should be one of expected values
		assert health['status'] in ['healthy', 'degraded', 'unhealthy']
	
	async def test_service_available_models(self, nlpc_service):
		"""Test getting available models"""
		models = await nlpc_service._get_available_models()
		
		assert isinstance(models, list)
		for model in models:
			assert 'name' in model
			assert 'provider' in model
			assert 'loaded' in model

class TestTextProcessingPipeline:
	"""Test core text processing pipeline functionality"""
	
	async def test_intelligent_preprocess_text_basic(self, nlpc_service):
		"""Test basic text preprocessing"""
		text = "This is a test document for preprocessing!"
		
		result = await nlpc_service.intelligent_preprocess_text(text)
		
		assert isinstance(result, dict)
		assert 'cleaned_text' in result
		assert 'detected_language' in result
		assert 'preprocessing_steps' in result
		assert len(result['cleaned_text']) > 0
	
	async def test_intelligent_preprocess_text_multilingual(self, nlpc_service):
		"""Test preprocessing with multilingual content"""
		texts = [
			("Hello world, this is a test!", LanguageCode.EN),
			("Hola mundo, esto es una prueba!", LanguageCode.ES),
			("Bonjour le monde, ceci est un test!", LanguageCode.FR)
		]
		
		for text, expected_lang in texts:
			result = await nlpc_service.intelligent_preprocess_text(text)
			
			assert result['detected_language'] == expected_lang.value
			assert len(result['cleaned_text']) > 0
	
	async def test_enhanced_language_detection(self, nlpc_service):
		"""Test enhanced language detection with multiple algorithms"""
		test_cases = [
			("This is clearly an English text with proper grammar.", "en"),
			("Este es claramente un texto en español con gramática adecuada.", "es"),
			("Ceci est clairement un texte français avec une grammaire appropriée.", "fr"),
			("Das ist eindeutig ein deutscher Text mit angemessener Grammatik.", "de")
		]
		
		for text, expected_lang in test_cases:
			result = await nlpc_service._enhanced_language_detection(text)
			
			assert isinstance(result, dict)
			assert 'language' in result
			assert 'confidence' in result
			assert 'algorithms_used' in result
			assert result['language'] == expected_lang
			assert 0 <= result['confidence'] <= 1
	
	async def test_custom_multilingual_tokenization(self, nlpc_service):
		"""Test custom tokenization for multiple languages"""
		test_cases = [
			("Hello world! How are you?", LanguageCode.EN),
			("¡Hola mundo! ¿Cómo estás?", LanguageCode.ES),
			("Bonjour monde! Comment allez-vous?", LanguageCode.FR)
		]
		
		for text, language in test_cases:
			result = await nlpc_service._custom_multilingual_tokenization(
				text, language, {"preserve_punctuation": True}
			)
			
			assert isinstance(result, dict)
			assert 'tokens' in result
			assert 'sentence_boundaries' in result
			assert 'tokenization_method' in result
			assert len(result['tokens']) > 0
	
	async def test_intelligent_text_chunking(self, nlpc_service, performance_test_data):
		"""Test intelligent text chunking for large documents"""
		large_text = performance_test_data['large_text']
		
		result = await nlpc_service._intelligent_text_chunking(
			large_text, LanguageCode.EN, {"chunk_size": 1000, "overlap": 100}
		)
		
		assert isinstance(result, dict)
		assert 'chunks' in result
		assert 'chunk_metadata' in result
		assert len(result['chunks']) > 1
		
		# Verify chunks have proper overlap
		chunks = result['chunks']
		for i in range(len(chunks) - 1):
			current_chunk = chunks[i]['text']
			next_chunk = chunks[i + 1]['text']
			
			# Should have some overlap
			overlap_found = any(
				word in next_chunk for word in current_chunk.split()[-10:]
			)
			assert overlap_found or i == len(chunks) - 2  # Last chunk might not overlap

class TestModelIntegration:
	"""Test multi-framework model integration"""
	
	async def test_intelligent_model_selection(self, nlpc_service, sample_documents):
		"""Test intelligent model selection for different tasks"""
		document = sample_documents[0]  # Business document
		
		for task in [NLPTask.SENTIMENT_ANALYSIS, NLPTask.NAMED_ENTITY_RECOGNITION, NLPTask.TEXT_CLASSIFICATION]:
			result = await nlpc_service.intelligent_model_selection(
				task, document.content, document.language, {"accuracy": "high"}
			)
			
			assert isinstance(result, dict)
			assert 'selected_model' in result
			assert 'model_provider' in result
			assert 'selection_reasoning' in result
			assert 'confidence' in result
	
	async def test_adaptive_model_switching(self, nlpc_service, sample_documents, sample_processing_requests):
		"""Test adaptive model switching based on performance"""
		document = sample_documents[0]
		request = sample_processing_requests[0]
		
		# Simulate performance feedback
		performance_feedback = {
			'current_model': 'spacy_en_core_web_sm',
			'latency_ms': 150.0,
			'accuracy_score': 0.85,
			'error_rate': 0.02,
			'memory_usage': 512
		}
		
		result = await nlpc_service.adaptive_model_switching(
			document, request, performance_feedback
		)
		
		assert isinstance(result, dict)
		assert 'switch_recommended' in result
		assert 'target_model' in result
		assert 'reasoning' in result
		
		if result['switch_recommended']:
			assert result['target_model'] != performance_feedback['current_model']

class TestContextAwareProcessing:
	"""Test context-aware processing engine"""
	
	async def test_create_context_session(self, nlpc_service):
		"""Test creating context session"""
		tenant_id = "test-tenant"
		session_config = {
			"max_context_length": 5000,
			"memory_retention_hours": 12,
			"enable_summarization": True
		}
		
		session = await nlpc_service.create_context_session(tenant_id, session_config)
		
		assert isinstance(session, ContextSession)
		assert session.tenant_id == tenant_id
		assert session.max_context_length == 5000
		assert session.memory_retention_hours == 12
		assert session.session_id in nlpc_service.context_sessions
	
	async def test_process_with_context(self, nlpc_service, sample_documents, sample_processing_requests):
		"""Test processing with context session"""
		# Create context session first
		session = await nlpc_service.create_context_session("test-tenant")
		
		document = sample_documents[0]
		request = sample_processing_requests[0]
		
		result = await nlpc_service.process_with_context(
			document, request, session.session_id
		)
		
		assert_processing_result_valid(result)
		assert hasattr(result, 'context_used')
		assert result.context_used == True
	
	async def test_context_memory_management(self, nlpc_service):
		"""Test context memory management and cleanup"""
		session = await nlpc_service.create_context_session(
			"test-tenant", 
			{"max_context_length": 1000, "memory_retention_hours": 1}
		)
		
		# Add content to context
		large_content = "This is test content. " * 100  # 2000+ chars
		await nlpc_service._add_to_context(session.session_id, {
			"content": large_content,
			"timestamp": datetime.utcnow().isoformat()
		})
		
		# Check memory management
		context_data = await nlpc_service._get_context_data(session.session_id)
		
		# Should have triggered memory management due to size limit
		assert len(str(context_data)) <= session.max_context_length * 1.2  # Allow some overhead

class TestSecurityIntegration:
	"""Test APG security integration"""
	
	async def test_secure_process_document(self, nlpc_service, sample_documents, sample_processing_requests, sample_security_context):
		"""Test secure document processing with RBAC"""
		document = sample_documents[0]
		request = sample_processing_requests[0]
		
		result = await nlpc_service.secure_process_document(
			document, request, sample_security_context
		)
		
		assert_processing_result_valid(result)
		assert hasattr(result, 'security_applied')
		assert result.security_applied == True
	
	async def test_validate_rbac_permissions(self, nlpc_service, sample_documents):
		"""Test RBAC permission validation"""
		user_roles = ['nlp_user', 'nlp_analyst']
		requested_tasks = [NLPTask.SENTIMENT_ANALYSIS, NLPTask.NAMED_ENTITY_RECOGNITION]
		document = sample_documents[0]
		
		result = await nlpc_service._validate_rbac_permissions(
			user_roles, requested_tasks, document
		)
		
		assert isinstance(result, dict)
		assert 'allowed_tasks' in result
		assert 'denied_tasks' in result
		assert 'security_classification' in result
	
	async def test_classify_document_sensitivity(self, nlpc_service, sample_documents, sample_security_context):
		"""Test document sensitivity classification"""
		# Test with potentially sensitive content
		sensitive_document = create_test_document(
			"SSN: 123-45-6789, Credit Card: 4111-1111-1111-1111, Email: john.doe@company.com"
		)
		
		result = await nlpc_service._classify_document_sensitivity(
			sensitive_document, sample_security_context
		)
		
		assert isinstance(result, dict)
		assert 'classification' in result
		assert 'pii_detected' in result
		assert 'sensitive_entities' in result
		assert result['classification'] in ['public', 'internal', 'confidential', 'restricted']
	
	async def test_audit_compliance_logging(self, nlpc_service, sample_documents, sample_processing_requests, sample_security_context):
		"""Test audit compliance logging"""
		document = sample_documents[0]
		request = sample_processing_requests[0]
		
		with patch.object(nlpc_service, '_log_audit_event') as mock_audit:
			await nlpc_service.secure_process_document(
				document, request, sample_security_context
			)
			
			# Should have logged audit events
			assert mock_audit.call_count >= 1
			
			# Check audit log structure
			call_args = mock_audit.call_args_list[0][0]
			audit_event = call_args[0]
			
			assert 'event_type' in audit_event
			assert 'user_id' in audit_event
			assert 'tenant_id' in audit_event
			assert 'timestamp' in audit_event
			assert 'resource_accessed' in audit_event

class TestAdvancedNLPFeatures:
	"""Test advanced NLP-specific features"""
	
	async def test_orchestrate_nlp_pipeline(self, nlpc_service, sample_documents, sample_security_context):
		"""Test NLP pipeline orchestration"""
		documents = sample_documents[:3]
		
		pipeline_config = {
			'tasks': [NLPTask.SENTIMENT_ANALYSIS, NLPTask.KEYWORD_EXTRACTION],
			'priority': PriorityLevel.MEDIUM,
			'options': {'batch_optimization': True},
			'parallel_processing': True,
			'max_workers': 2
		}
		
		results = await nlpc_service.orchestrate_nlp_pipeline(
			documents, pipeline_config, sample_security_context
		)
		
		assert isinstance(results, list)
		assert len(results) == len(documents)
		
		for result in results:
			assert 'document_id' in result
			assert 'task_results' in result
			assert 'processing_time' in result
			assert 'status' in result
	
	async def test_create_model_ensemble(self, nlpc_service, sample_documents, sample_security_context):
		"""Test model ensemble creation and execution"""
		documents = sample_documents[:2]
		
		ensemble_config = {
			'models': ['spacy_en_core_web_sm', 'textblob_sentiment', 'nltk_vader'],
			'voting_strategy': 'weighted_average',
			'confidence_threshold': 0.7,
			'task_type': NLPTask.SENTIMENT_ANALYSIS
		}
		
		result = await nlpc_service.create_model_ensemble(
			documents, ensemble_config, sample_security_context
		)
		
		assert isinstance(result, dict)
		assert 'ensemble_id' in result
		assert 'models_loaded' in result
		assert 'voting_strategy' in result
		assert result['ensemble_id'] is not None
	
	async def test_execute_ensemble_processing(self, nlpc_service, sample_documents, sample_security_context):
		"""Test ensemble processing execution"""
		# First create ensemble
		ensemble_result = await nlpc_service.create_model_ensemble(
			sample_documents[:1], 
			{
				'models': ['spacy_en_core_web_sm', 'textblob_sentiment'],
				'voting_strategy': 'majority_vote',
				'task_type': NLPTask.SENTIMENT_ANALYSIS
			}, 
			sample_security_context
		)
		
		document = sample_documents[1]  # Positive review document
		
		result = await nlpc_service.execute_ensemble_processing(
			ensemble_result['ensemble_id'],
			document,
			NLPTask.SENTIMENT_ANALYSIS,
			sample_security_context
		)
		
		assert isinstance(result, dict)
		assert 'ensemble_result' in result
		assert 'individual_results' in result
		assert 'confidence_score' in result
		assert 'voting_details' in result
	
	async def test_optimize_nlp_workflow(self, nlpc_service):
		"""Test NLP workflow optimization"""
		# Sample workflow history with performance data
		workflow_history = [
			{
				'task': NLPTask.SENTIMENT_ANALYSIS.value,
				'model_used': 'spacy_en_core_web_sm',
				'processing_time': 145.5,
				'accuracy': 0.89,
				'memory_usage': 256
			},
			{
				'task': NLPTask.SENTIMENT_ANALYSIS.value,
				'model_used': 'textblob_sentiment',
				'processing_time': 89.2,
				'accuracy': 0.85,
				'memory_usage': 128
			},
			{
				'task': NLPTask.NAMED_ENTITY_RECOGNITION.value,
				'model_used': 'spacy_en_core_web_sm',
				'processing_time': 156.8,
				'accuracy': 0.92,
				'memory_usage': 256
			}
		]
		
		performance_targets = {
			'max_latency_ms': 100.0,
			'min_accuracy': 0.85,
			'max_memory_mb': 200
		}
		
		result = await nlpc_service.optimize_nlp_workflow(
			workflow_history, performance_targets
		)
		
		assert isinstance(result, dict)
		assert 'optimization_recommendations' in result
		assert 'projected_improvements' in result
		assert 'implementation_priority' in result

class TestPerformanceOptimization:
	"""Test performance optimization features"""
	
	async def test_initialize_performance_system(self, nlpc_service):
		"""Test performance system initialization"""
		performance_config = {
			'cache_enabled': True,
			'cache_size': 100,
			'cache_ttl': 300,
			'model_warming': True,
			'adaptive_optimization': True
		}
		
		await nlpc_service.initialize_performance_system(performance_config)
		
		assert nlpc_service.performance_cache is not None
		assert nlpc_service.cache_config['enabled'] == True
		assert nlpc_service.cache_config['max_size'] == 100
		assert nlpc_service.cache_config['ttl'] == 300
	
	async def test_process_with_performance_optimization(self, nlpc_service, sample_documents, sample_processing_requests, sample_security_context):
		"""Test processing with performance optimization"""
		document = sample_documents[0]
		request = sample_processing_requests[0]
		
		result = await nlpc_service.process_with_performance_optimization(
			document, request, sample_security_context
		)
		
		assert_processing_result_valid(result)
		assert hasattr(result, 'cache_used')
		assert hasattr(result, 'optimization_applied')
		assert hasattr(result, 'performance_metrics')
	
	async def test_intelligent_cache_decision(self, nlpc_service):
		"""Test intelligent caching decision logic"""
		cache_key = "test_document_sentiment_analysis"
		result_data = {
			"sentiment": "positive",
			"confidence": 0.89,
			"processing_time": 125.5
		}
		tasks = [NLPTask.SENTIMENT_ANALYSIS]
		
		decision = await nlpc_service._intelligent_cache_decision(
			cache_key, result_data, tasks
		)
		
		assert isinstance(decision, dict)
		assert 'should_cache' in decision
		assert 'cache_ttl' in decision
		assert 'reasoning' in decision
	
	async def test_model_warming(self, nlpc_service):
		"""Test model warming functionality"""
		models_to_warm = ['tokenization', 'sentiment_analysis', 'language_detection']
		
		for model_name in models_to_warm:
			result = await nlpc_service._warm_model(model_name, 'en')
			
			assert isinstance(result, dict)
			assert 'model_name' in result
			assert 'warming_time' in result
			assert 'status' in result
			assert result['status'] in ['success', 'already_loaded', 'failed']
	
	async def test_get_performance_analytics(self, nlpc_service):
		"""Test performance analytics retrieval"""
		# Test different time ranges
		time_ranges = [1, 24, 168]  # 1 hour, 24 hours, 1 week
		
		for hours in time_ranges:
			analytics = await nlpc_service.get_performance_analytics(hours)
			
			assert isinstance(analytics, dict)
			assert 'time_range_hours' in analytics
			assert 'performance' in analytics
			assert 'cache' in analytics
			assert 'models' in analytics
			assert 'requests' in analytics
			
			assert analytics['time_range_hours'] == hours
			
			# Verify analytics structure
			performance = analytics['performance']
			assert 'total_requests' in performance
			assert 'average_processing_time_ms' in performance
			assert 'success_rate' in performance
			
			cache = analytics['cache']
			assert 'hit_rate' in cache
			assert 'total_requests' in cache
			assert 'cache_size' in cache

class TestErrorHandlingAndEdgeCases:
	"""Test error handling and edge cases"""
	
	async def test_processing_empty_document(self, nlpc_service, sample_security_context):
		"""Test processing with empty/invalid document"""
		# This should be caught by model validation, but test service handling
		try:
			# Create minimal request
			request = ProcessingRequest(
				document_id=uuid7str(),
				tasks=[NLPTask.SENTIMENT_ANALYSIS],
				tenant_id="test-tenant"
			)
			
			# Test with mock empty document (service should handle gracefully)
			with patch.object(nlpc_service, '_validate_document_content') as mock_validate:
				mock_validate.side_effect = ValueError("Empty document content")
				
				result = await nlpc_service.secure_process_document(
					None, request, sample_security_context
				)
				
				# Should return error result, not raise exception
				assert result.status == ProcessingStatus.FAILED
				assert "Empty document content" in result.error_message
				
		except Exception as e:
			# Acceptable if service raises controlled exception
			assert "Empty" in str(e) or "Invalid" in str(e)
	
	async def test_processing_timeout_handling(self, nlpc_service, sample_documents, sample_processing_requests, sample_security_context):
		"""Test processing timeout handling"""
		document = sample_documents[0]
		request = sample_processing_requests[0]
		
		# Mock a timeout scenario
		with patch.object(nlpc_service, '_execute_nlp_task') as mock_execute:
			mock_execute.side_effect = asyncio.TimeoutError("Processing timeout")
			
			result = await nlpc_service.secure_process_document(
				document, request, sample_security_context
			)
			
			assert result.status == ProcessingStatus.FAILED
			assert "timeout" in result.error_message.lower()
	
	async def test_invalid_language_handling(self, nlpc_service):
		"""Test handling of unsupported languages"""
		# Test with unsupported language
		text = "Some text in unsupported language"
		
		result = await nlpc_service._enhanced_language_detection(text)
		
		assert isinstance(result, dict)
		assert 'language' in result
		
		# Should default to fallback or return 'unknown'
		assert result['language'] in ['unknown', 'en']  # en as fallback
	
	async def test_model_loading_failure_handling(self, nlpc_service):
		"""Test handling of model loading failures"""
		with patch.object(nlpc_service, '_load_spacy_models') as mock_load:
			mock_load.side_effect = Exception("Model loading failed")
			
			try:
				await nlpc_service.initialize_nlp_models({'spacy_enabled': True})
				
				# Should handle gracefully and set appropriate status
				health = await nlpc_service._check_service_health()
				assert health['status'] in ['degraded', 'unhealthy']
				
			except Exception as e:
				# Acceptable if initialization fails with clear error
				assert "Model loading failed" in str(e)

class TestAsyncOperations:
	"""Test async operations and concurrency"""
	
	async def test_concurrent_processing(self, nlpc_service, sample_documents, sample_security_context):
		"""Test concurrent document processing"""
		documents = sample_documents[:3]
		
		# Create concurrent processing tasks
		tasks = []
		for doc in documents:
			request = ProcessingRequest(
				document_id=doc.document_id,
				tasks=[NLPTask.SENTIMENT_ANALYSIS],
				tenant_id=doc.tenant_id
			)
			
			task = nlpc_service.secure_process_document(
				doc, request, sample_security_context
			)
			tasks.append(task)
		
		# Execute concurrently
		results = await asyncio.gather(*tasks, return_exceptions=True)
		
		# Verify all completed successfully
		for result in results:
			if isinstance(result, Exception):
				pytest.fail(f"Concurrent processing failed: {result}")
			
			assert_processing_result_valid(result)
	
	async def test_async_context_session_management(self, nlpc_service):
		"""Test async context session management"""
		# Create multiple sessions concurrently
		session_tasks = []
		for i in range(3):
			task = nlpc_service.create_context_session(
				f"tenant-{i}", 
				{"max_context_length": 1000 * (i + 1)}
			)
			session_tasks.append(task)
		
		sessions = await asyncio.gather(*session_tasks)
		
		assert len(sessions) == 3
		for i, session in enumerate(sessions):
			assert session.tenant_id == f"tenant-{i}"
			assert session.max_context_length == 1000 * (i + 1)
	
	async def test_performance_under_load(self, nlpc_service, performance_test_data, sample_security_context):
		"""Test performance under simulated load"""
		# Create multiple concurrent requests
		num_requests = 10
		document = create_test_document(performance_test_data['medium_text'])
		
		start_time = time.time()
		
		tasks = []
		for _ in range(num_requests):
			request = ProcessingRequest(
				document_id=uuid7str(),
				tasks=[NLPTask.SENTIMENT_ANALYSIS],
				tenant_id="test-tenant"
			)
			
			task = nlpc_service.secure_process_document(
				document, request, sample_security_context
			)
			tasks.append(task)
		
		results = await asyncio.gather(*tasks, return_exceptions=True)
		
		total_time = time.time() - start_time
		
		# Verify performance metrics
		successful_results = [r for r in results if not isinstance(r, Exception)]
		assert len(successful_results) == num_requests
		
		# Should handle concurrent load reasonably
		avg_time_per_request = total_time / num_requests
		assert avg_time_per_request < 5.0  # Should be under 5 seconds per request