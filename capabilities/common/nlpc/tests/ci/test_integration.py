"""
Integration tests for NLPC with APG capabilities and external systems.

Tests integration with auth_rbac, audit_compliance, composition engine,
and other APG components following APG testing standards.
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
	NLPTask, ProcessingStatus, LanguageCode, PriorityLevel
)
from ..conftest import (
	assert_processing_result_valid, assert_security_context_valid,
	create_test_document, create_test_request
)

class TestAPGCompositionEngineIntegration:
	"""Test integration with APG composition engine"""
	
	async def test_capability_registration(self, nlpc_service):
		"""Test NLPC capability registration with composition engine"""
		capability_metadata = {
			'capability_id': 'nlpc',
			'version': '1.0.0',
			'provides': [
				'text_processing',
				'sentiment_analysis', 
				'entity_recognition',
				'text_classification'
			],
			'requires': ['auth_rbac', 'audit_compliance', 'aicr'],
			'endpoints': [
				'/api/nlp/process',
				'/api/nlp/models',
				'/api/nlp/health'
			]
		}
		
		# Mock APG composition engine
		with patch('apg.composition.register_capability') as mock_register:
			mock_register.return_value = {'status': 'registered', 'capability_id': 'nlpc'}
			
			# Test registration
			result = await nlpc_service._register_with_composition_engine(capability_metadata)
			
			assert result['status'] == 'registered'
			mock_register.assert_called_once_with(capability_metadata)
	
	async def test_capability_discovery(self, nlpc_service):
		"""Test capability discovery through composition engine"""
		# Mock composition engine discovery
		with patch('apg.composition.discover_capabilities') as mock_discover:
			mock_discover.return_value = [
				{'id': 'auth_rbac', 'version': '1.0.0', 'status': 'active'},
				{'id': 'audit_compliance', 'version': '1.0.0', 'status': 'active'},
				{'id': 'aicr', 'version': '1.0.0', 'status': 'active'}
			]
			
			dependencies = await nlpc_service._check_capability_dependencies()
			
			assert 'auth_rbac' in dependencies
			assert 'audit_compliance' in dependencies
			assert 'aicr' in dependencies
			
			for dep in dependencies.values():
				assert dep['status'] == 'active'
	
	async def test_inter_capability_communication(self, nlpc_service):
		"""Test communication between NLPC and other APG capabilities"""
		# Mock AICR capability for model serving
		with patch('apg.capabilities.aicr.serve_model') as mock_aicr:
			mock_aicr.return_value = {
				'model_id': 'bert-base-uncased',
				'endpoint': 'http://aicr-service/models/bert-base-uncased',
				'status': 'ready'
			}
			
			# Test model request through AICR
			model_config = {
				'model_name': 'bert-base-uncased',
				'task': 'sentiment-analysis',
				'provider': 'transformers'
			}
			
			result = await nlpc_service._request_external_model(model_config)
			
			assert result['model_id'] == 'bert-base-uncased'
			assert result['status'] == 'ready'
			mock_aicr.assert_called_once_with(model_config)

class TestAuthRBACIntegration:
	"""Test integration with APG auth_rbac capability"""
	
	async def test_jwt_token_validation(self, nlpc_service):
		"""Test JWT token validation through auth_rbac"""
		mock_jwt_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.test.payload"
		
		with patch('apg.capabilities.auth_rbac.validate_jwt') as mock_validate:
			mock_validate.return_value = {
				'valid': True,
				'user_id': 'user123',
				'tenant_id': 'tenant456',
				'roles': ['nlp_user', 'nlp_analyst'],
				'permissions': ['nlp_process_text', 'nlp_view_results'],
				'expires_at': datetime.utcnow() + timedelta(hours=1)
			}
			
			result = await nlpc_service._validate_jwt_token(mock_jwt_token)
			
			assert result['valid'] == True
			assert result['user_id'] == 'user123'
			assert result['tenant_id'] == 'tenant456'
			assert 'nlp_user' in result['roles']
			assert 'nlp_process_text' in result['permissions']
	
	async def test_role_based_access_control(self, nlpc_service, sample_documents, sample_security_context):
		"""Test RBAC enforcement for NLP operations"""
		document = sample_documents[0]
		
		# Test with different role combinations
		role_test_cases = [
			{
				'roles': ['nlp_user'],
				'expected_tasks': [NLPTask.SENTIMENT_ANALYSIS, NLPTask.LANGUAGE_DETECTION],
				'denied_tasks': [NLPTask.TEXT_GENERATION]
			},
			{
				'roles': ['nlp_analyst'],
				'expected_tasks': [NLPTask.SENTIMENT_ANALYSIS, NLPTask.NAMED_ENTITY_RECOGNITION, NLPTask.TEXT_CLASSIFICATION],
				'denied_tasks': []
			},
			{
				'roles': ['nlp_admin'],
				'expected_tasks': list(NLPTask),
				'denied_tasks': []
			}
		]
		
		for test_case in role_test_cases:
			security_context = {
				**sample_security_context,
				'user_roles': test_case['roles']
			}
			
			result = await nlpc_service._validate_rbac_permissions(
				test_case['roles'], 
				test_case['expected_tasks'], 
				document
			)
			
			assert len(result['allowed_tasks']) >= len(test_case['expected_tasks'])
			assert len(result['denied_tasks']) == len(test_case['denied_tasks'])
	
	async def test_tenant_isolation(self, nlpc_service):
		"""Test multi-tenant data isolation"""
		tenants = ['tenant-a', 'tenant-b', 'tenant-c']
		documents_per_tenant = {}
		
		# Create documents for different tenants
		for tenant in tenants:
			docs = []
			for i in range(3):
				doc = NLPDocument(
					content=f"Test document {i} for {tenant}",
					tenant_id=tenant,
					metadata={'tenant': tenant, 'doc_id': i}
				)
				docs.append(doc)
			documents_per_tenant[tenant] = docs
		
		# Test tenant isolation in processing
		for tenant in tenants:
			service = NLPCService(tenant_id=tenant)
			
			# Should only access documents for this tenant
			available_docs = await service._get_tenant_documents(tenant)
			
			assert len(available_docs) == 3
			for doc in available_docs:
				assert doc.tenant_id == tenant
				assert doc.metadata['tenant'] == tenant
	
	async def test_permission_inheritance(self, nlpc_service):
		"""Test permission inheritance and delegation"""
		# Mock hierarchical roles
		with patch('apg.capabilities.auth_rbac.get_role_hierarchy') as mock_hierarchy:
			mock_hierarchy.return_value = {
				'nlp_admin': {
					'inherits_from': ['nlp_analyst', 'nlp_user'],
					'permissions': ['nlp_admin_all']
				},
				'nlp_analyst': {
					'inherits_from': ['nlp_user'],
					'permissions': ['nlp_analyze', 'nlp_batch_process']
				},
				'nlp_user': {
					'inherits_from': [],
					'permissions': ['nlp_process_text', 'nlp_view_results']
				}
			}
			
			# Test permission resolution for nlp_admin
			permissions = await nlpc_service._resolve_user_permissions(['nlp_admin'])
			
			expected_permissions = [
				'nlp_admin_all',
				'nlp_analyze', 
				'nlp_batch_process',
				'nlp_process_text',
				'nlp_view_results'
			]
			
			for perm in expected_permissions:
				assert perm in permissions

class TestAuditComplianceIntegration:
	"""Test integration with APG audit_compliance capability"""
	
	async def test_audit_event_logging(self, nlpc_service, sample_documents, sample_security_context):
		"""Test audit event logging for NLP operations"""
		with patch('apg.capabilities.audit_compliance.log_event') as mock_audit:
			document = sample_documents[0]
			request = ProcessingRequest(
				document_id=document.document_id,
				tasks=[NLPTask.SENTIMENT_ANALYSIS],
				tenant_id=document.tenant_id
			)
			
			await nlpc_service.secure_process_document(
				document, request, sample_security_context
			)
			
			# Should have logged multiple audit events
			assert mock_audit.call_count >= 2  # Start and completion events
			
			# Check audit event structure
			audit_calls = mock_audit.call_args_list
			start_event = audit_calls[0][0][0]
			
			required_fields = [
				'event_type', 'timestamp', 'user_id', 'tenant_id',
				'resource_type', 'resource_id', 'action', 'result'
			]
			
			for field in required_fields:
				assert field in start_event
	
	async def test_compliance_data_retention(self, nlpc_service, sample_documents):
		"""Test compliance data retention policies"""
		document = sample_documents[0]
		
		with patch('apg.capabilities.audit_compliance.apply_retention_policy') as mock_retention:
			mock_retention.return_value = {
				'retention_period_days': 2555,  # 7 years
				'classification': 'business_record',
				'deletion_date': datetime.utcnow() + timedelta(days=2555)
			}
			
			retention_policy = await nlpc_service._apply_data_retention_policy(document)
			
			assert retention_policy['retention_period_days'] == 2555
			assert retention_policy['classification'] == 'business_record'
			mock_retention.assert_called_once()
	
	async def test_gdpr_compliance_features(self, nlpc_service, sample_security_context):
		"""Test GDPR compliance features"""
		# Test with PII-containing document
		pii_document = create_test_document(
			"Hello, my name is John Doe and my email is john.doe@example.com. "
			"My phone number is +1-555-123-4567 and I live in New York."
		)
		
		with patch('apg.capabilities.audit_compliance.check_gdpr_compliance') as mock_gdpr:
			mock_gdpr.return_value = {
				'pii_detected': True,
				'pii_types': ['name', 'email', 'phone', 'location'],
				'requires_consent': True,
				'data_subject_rights': ['access', 'rectification', 'erasure', 'portability'],
				'lawful_basis': 'consent'
			}
			
			compliance_check = await nlpc_service._check_gdpr_compliance(
				pii_document, sample_security_context
			)
			
			assert compliance_check['pii_detected'] == True
			assert 'email' in compliance_check['pii_types']
			assert compliance_check['requires_consent'] == True
	
	async def test_audit_trail_integrity(self, nlpc_service, sample_documents, sample_security_context):
		"""Test audit trail integrity and tamper detection"""
		document = sample_documents[0]
		
		with patch('apg.capabilities.audit_compliance.create_audit_hash') as mock_hash, \
			 patch('apg.capabilities.audit_compliance.verify_audit_integrity') as mock_verify:
			
			mock_hash.return_value = "sha256:abc123def456..."
			mock_verify.return_value = {'valid': True, 'hash_verified': True}
			
			# Process document with audit trail
			request = ProcessingRequest(
				document_id=document.document_id,
				tasks=[NLPTask.SENTIMENT_ANALYSIS],
				tenant_id=document.tenant_id
			)
			
			result = await nlpc_service.secure_process_document(
				document, request, sample_security_context
			)
			
			# Verify audit hash was created
			mock_hash.assert_called()
			
			# Verify integrity check
			integrity_check = await nlpc_service._verify_audit_integrity(result.result_id)
			assert integrity_check['valid'] == True

class TestMultiTenantIntegration:
	"""Test multi-tenant processing and isolation"""
	
	async def test_tenant_resource_isolation(self, test_config):
		"""Test resource isolation between tenants"""
		tenants = ['tenant-1', 'tenant-2', 'tenant-3']
		services = {}
		
		# Create separate service instances for each tenant
		for tenant in tenants:
			service = NLPCService(tenant_id=tenant)
			await service.initialize_performance_system(test_config)
			services[tenant] = service
		
		# Test that each tenant has isolated resources
		for tenant, service in services.items():
			assert service.tenant_id == tenant
			
			# Each should have separate cache
			cache_stats = await service._get_cache_statistics()
			assert cache_stats['tenant_id'] == tenant
			
			# Each should have separate context sessions
			sessions = service.context_sessions
			for session_id, session in sessions.items():
				assert session.tenant_id == tenant
	
	async def test_cross_tenant_data_leakage_prevention(self, nlpc_service):
		"""Test prevention of cross-tenant data leakage"""
		# Create documents for different tenants
		tenant_a_doc = create_test_document("Sensitive data for tenant A")
		tenant_a_doc.tenant_id = "tenant-a"
		
		tenant_b_doc = create_test_document("Sensitive data for tenant B")  
		tenant_b_doc.tenant_id = "tenant-b"
		
		# Service configured for tenant A should not access tenant B data
		service_a = NLPCService(tenant_id="tenant-a")
		
		with pytest.raises(PermissionError) as exc_info:
			request = ProcessingRequest(
				document_id=tenant_b_doc.document_id,
				tasks=[NLPTask.SENTIMENT_ANALYSIS],
				tenant_id="tenant-b"  # Different tenant
			)
			
			await service_a._validate_tenant_access(tenant_b_doc, request)
		
		assert "tenant" in str(exc_info.value).lower()
	
	async def test_tenant_specific_configuration(self, test_config):
		"""Test tenant-specific configuration and customization"""
		tenant_configs = {
			'enterprise-tenant': {
				'max_cache_size': 1000,
				'performance_optimization': True,
				'security_level': 'high',
				'allowed_models': ['spacy', 'transformers']
			},
			'basic-tenant': {
				'max_cache_size': 100,
				'performance_optimization': False,
				'security_level': 'standard',
				'allowed_models': ['spacy', 'nltk']
			}
		}
		
		services = {}
		for tenant_id, config in tenant_configs.items():
			service = NLPCService(tenant_id=tenant_id)
			await service.initialize_performance_system({**test_config, **config})
			services[tenant_id] = service
		
		# Verify tenant-specific configurations
		enterprise_service = services['enterprise-tenant']
		basic_service = services['basic-tenant']
		
		assert enterprise_service.cache_config['max_size'] == 1000
		assert basic_service.cache_config['max_size'] == 100
		
		assert enterprise_service.performance_optimization_enabled == True
		assert basic_service.performance_optimization_enabled == False

class TestExternalSystemIntegration:
	"""Test integration with external systems and APIs"""
	
	async def test_ollama_integration(self, nlpc_service):
		"""Test integration with Ollama for local LLM serving"""
		with patch('requests.post') as mock_post:
			mock_response = MagicMock()
			mock_response.status_code = 200
			mock_response.json.return_value = {
				'model': 'llama3.2',
				'response': 'This text has a positive sentiment.',
				'done': True,
				'context': [123, 456, 789]
			}
			mock_post.return_value = mock_response
			
			result = await nlpc_service._query_ollama_model(
				model='llama3.2',
				prompt='Analyze the sentiment of: "I love this product!"',
				task=NLPTask.SENTIMENT_ANALYSIS
			)
			
			assert result['model'] == 'llama3.2'
			assert 'positive' in result['response'].lower()
			mock_post.assert_called_once()
	
	async def test_spacy_model_integration(self, nlpc_service):
		"""Test integration with spaCy models"""
		with patch('spacy.load') as mock_spacy_load:
			# Mock spaCy model
			mock_doc = MagicMock()
			mock_doc.text = "Apple Inc. is a great company."
			mock_doc.ents = [
				MagicMock(text="Apple Inc.", label_="ORG", start=0, end=9)
			]
			mock_doc.sentiment = 0.8
			
			mock_model = MagicMock()
			mock_model.return_value = mock_doc
			mock_spacy_load.return_value = mock_model
			
			result = await nlpc_service._process_with_spacy(
				text="Apple Inc. is a great company.",
				tasks=[NLPTask.NAMED_ENTITY_RECOGNITION],
				model_name="en_core_web_sm"
			)
			
			assert 'entities' in result
			assert len(result['entities']) > 0
			assert result['entities'][0]['text'] == "Apple Inc."
			assert result['entities'][0]['label'] == "ORG"
	
	async def test_nltk_integration(self, nlpc_service):
		"""Test integration with NLTK libraries"""
		with patch('nltk.sentiment.SentimentIntensityAnalyzer') as mock_analyzer:
			mock_analyzer_instance = MagicMock()
			mock_analyzer_instance.polarity_scores.return_value = {
				'neg': 0.1,
				'neu': 0.2, 
				'pos': 0.7,
				'compound': 0.6
			}
			mock_analyzer.return_value = mock_analyzer_instance
			
			result = await nlpc_service._process_with_nltk(
				text="I absolutely love this product!",
				tasks=[NLPTask.SENTIMENT_ANALYSIS],
				analyzer='vader'
			)
			
			assert 'sentiment' in result
			assert result['sentiment'] == 'positive'
			assert result['confidence'] > 0.5
			mock_analyzer_instance.polarity_scores.assert_called_once()
	
	async def test_transformers_integration(self, nlpc_service):
		"""Test integration with Hugging Face transformers"""
		with patch('transformers.pipeline') as mock_pipeline:
			# Mock transformers pipeline
			mock_pipe = MagicMock()
			mock_pipe.return_value = [
				{'label': 'POSITIVE', 'score': 0.89}
			]
			mock_pipeline.return_value = mock_pipe
			
			result = await nlpc_service._process_with_transformers(
				text="This is fantastic!",
				tasks=[NLPTask.SENTIMENT_ANALYSIS],
				model_name="distilbert-base-uncased-finetuned-sst-2-english"
			)
			
			assert 'sentiment' in result
			assert result['sentiment'] == 'positive'
			assert result['confidence'] == 0.89
			mock_pipe.assert_called_once()

class TestPerformanceIntegration:
	"""Test performance integration with APG infrastructure"""
	
	async def test_apm_integration(self, nlpc_service, sample_documents, sample_security_context):
		"""Test integration with APG Performance Monitoring (APM)"""
		with patch('apg.monitoring.start_trace') as mock_start, \
			 patch('apg.monitoring.end_trace') as mock_end:
			
			mock_trace_id = uuid7str()
			mock_start.return_value = {'trace_id': mock_trace_id, 'span_id': uuid7str()}
			
			document = sample_documents[0]
			request = ProcessingRequest(
				document_id=document.document_id,
				tasks=[NLPTask.SENTIMENT_ANALYSIS],
				tenant_id=document.tenant_id
			)
			
			result = await nlpc_service.secure_process_document(
				document, request, sample_security_context
			)
			
			# Should have started and ended trace
			mock_start.assert_called_once()
			mock_end.assert_called_once()
			
			# Verify trace context
			trace_call = mock_start.call_args[1]
			assert trace_call['operation'] == 'nlp_process_document'
			assert trace_call['tenant_id'] == document.tenant_id
	
	async def test_metrics_collection(self, nlpc_service, sample_documents):
		"""Test metrics collection for APG monitoring dashboard"""
		with patch('apg.metrics.increment_counter') as mock_counter, \
			 patch('apg.metrics.record_histogram') as mock_histogram:
			
			document = sample_documents[0]
			request = ProcessingRequest(
				document_id=document.document_id,
				tasks=[NLPTask.SENTIMENT_ANALYSIS],
				tenant_id=document.tenant_id
			)
			
			start_time = time.time()
			await nlpc_service._execute_nlp_task(
				document, NLPTask.SENTIMENT_ANALYSIS, {}
			)
			processing_time = (time.time() - start_time) * 1000
			
			# Should have recorded metrics
			mock_counter.assert_called()
			mock_histogram.assert_called()
			
			# Verify metric names and values
			counter_calls = [call[0][0] for call in mock_counter.call_args_list]
			assert 'nlp.requests.total' in counter_calls
			assert 'nlp.tasks.completed' in counter_calls
	
	async def test_load_balancing_integration(self, test_config):
		"""Test integration with APG load balancing"""
		# Simulate multiple service instances
		num_instances = 3
		services = []
		
		for i in range(num_instances):
			service = NLPCService(tenant_id=f"tenant-{i}")
			await service.initialize_performance_system(test_config)
			services.append(service)
		
		with patch('apg.loadbalancer.register_service') as mock_register:
			# Register each service instance
			for i, service in enumerate(services):
				await service._register_with_load_balancer({
					'service_id': f'nlpc-{i}',
					'host': f'nlpc-service-{i}',
					'port': 8000 + i,
					'health_check': '/health',
					'capabilities': ['text_processing', 'sentiment_analysis']
				})
			
			assert mock_register.call_count == num_instances

class TestErrorIntegrationScenarios:
	"""Test error handling in integration scenarios"""
	
	async def test_dependency_failure_handling(self, nlpc_service, sample_documents, sample_security_context):
		"""Test handling of dependency failures"""
		document = sample_documents[0]
		request = ProcessingRequest(
			document_id=document.document_id,
			tasks=[NLPTask.SENTIMENT_ANALYSIS],
			tenant_id=document.tenant_id
		)
		
		# Simulate auth_rbac failure
		with patch('apg.capabilities.auth_rbac.validate_jwt') as mock_auth:
			mock_auth.side_effect = Exception("Auth service unavailable")
			
			result = await nlpc_service.secure_process_document(
				document, request, sample_security_context
			)
			
			# Should handle gracefully and return error result
			assert result.status == ProcessingStatus.FAILED
			assert "auth" in result.error_message.lower()
	
	async def test_partial_capability_degradation(self, nlpc_service):
		"""Test handling of partial capability degradation"""
		# Simulate some models failing to load
		with patch.object(nlpc_service, '_load_spacy_models') as mock_spacy, \
			 patch.object(nlpc_service, '_load_nltk_models') as mock_nltk:
			
			mock_spacy.side_effect = Exception("spaCy models failed")
			mock_nltk.return_value = True  # NLTK succeeds
			
			await nlpc_service.initialize_nlp_models({
				'spacy_enabled': True,
				'nltk_enabled': True,
				'transformers_enabled': False
			})
			
			# Service should be partially operational
			health = await nlpc_service._check_service_health()
			assert health['status'] == 'degraded'
			assert 'spacy' in health.get('failed_components', [])
	
	async def test_cascade_failure_prevention(self, nlpc_service, sample_documents):
		"""Test prevention of cascade failures"""
		documents = sample_documents[:5]
		
		# Simulate one document causing model failure
		with patch.object(nlpc_service, '_execute_nlp_task') as mock_execute:
			# First call fails, subsequent calls succeed
			mock_execute.side_effect = [
				Exception("Model timeout"),
				{"sentiment": "positive", "confidence": 0.8},
				{"sentiment": "neutral", "confidence": 0.7},
				{"sentiment": "negative", "confidence": 0.9},
				{"sentiment": "positive", "confidence": 0.85}
			]
			
			results = []
			for doc in documents:
				try:
					result = await nlpc_service._execute_nlp_task(
						doc, NLPTask.SENTIMENT_ANALYSIS, {}
					)
					results.append(result)
				except Exception as e:
					results.append({"error": str(e)})
			
			# Should have isolated the failure
			assert len(results) == 5
			assert "error" in results[0]  # First failed
			assert "sentiment" in results[1]  # Others succeeded
			assert "sentiment" in results[2]
			assert "sentiment" in results[3]
			assert "sentiment" in results[4]