"""
Security and compliance tests for NLPC capability.

Tests PII detection, encryption, audit trails, RBAC enforcement,
and compliance framework validation following APG security standards.
"""

import pytest
import asyncio
import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List
from uuid_extensions import uuid7str
from unittest.mock import AsyncMock, MagicMock, patch
import secrets

from ...service import NLPCService
from ...models import (
	NLPDocument, ProcessingRequest, ProcessingResult, ContextSession,
	NLPTask, ProcessingStatus, LanguageCode, PriorityLevel
)
from ..conftest import (
	assert_processing_result_valid, assert_security_context_valid,
	create_test_document, create_test_request
)

class TestPIIDetectionAndMasking:
	"""Test PII detection and data masking functionality"""
	
	async def test_pii_detection_basic(self, nlpc_service):
		"""Test basic PII detection in text content"""
		pii_test_cases = [
			{
				'text': "My email is john.doe@company.com and phone is 555-123-4567",
				'expected_pii': ['email', 'phone'],
				'expected_entities': ['john.doe@company.com', '555-123-4567']
			},
			{
				'text': "SSN: 123-45-6789, Credit Card: 4111-1111-1111-1111",
				'expected_pii': ['ssn', 'credit_card'],
				'expected_entities': ['123-45-6789', '4111-1111-1111-1111']
			},
			{
				'text': "Please send the documents to 123 Main St, New York, NY 10001",
				'expected_pii': ['address'],
				'expected_entities': ['123 Main St, New York, NY 10001']
			}
		]
		
		for test_case in pii_test_cases:
			document = create_test_document(test_case['text'])
			security_context = {
				'user_id': 'test-user',
				'tenant_id': 'test-tenant',
				'user_roles': ['nlp_user'],
				'data_classification': 'restricted'
			}
			
			pii_result = await nlpc_service._classify_document_sensitivity(
				document, security_context
			)
			
			assert pii_result['pii_detected'] == True
			assert 'sensitive_entities' in pii_result
			
			detected_types = [entity['type'] for entity in pii_result['sensitive_entities']]
			for expected_type in test_case['expected_pii']:
				assert expected_type in detected_types
	
	async def test_pii_masking_options(self, nlpc_service):
		"""Test different PII masking strategies"""
		sensitive_text = "Contact John Doe at john.doe@company.com or call 555-123-4567"
		document = create_test_document(sensitive_text)
		
		masking_strategies = [
			'redact',      # Replace with [REDACTED]
			'tokenize',    # Replace with tokens like [EMAIL], [PHONE]
			'partial',     # Show only partial info: j***@***.com
			'hash',        # Replace with hash values
			'synthetic'    # Replace with synthetic but realistic data
		]
		
		for strategy in masking_strategies:
			masked_result = await nlpc_service._apply_pii_masking(
				document.content, 
				strategy,
				{'preserve_format': True}
			)
			
			assert isinstance(masked_result, dict)
			assert 'masked_text' in masked_result
			assert 'masking_map' in masked_result
			assert 'pii_locations' in masked_result
			
			masked_text = masked_result['masked_text']
			
			# Original PII should not be present in masked text
			assert 'john.doe@company.com' not in masked_text
			assert '555-123-4567' not in masked_text
			
			# But text should still be readable/processable
			assert 'John Doe' in masked_text or '[NAME]' in masked_text
			assert len(masked_text) > 20  # Should not be empty
	
	async def test_pii_detection_multilingual(self, nlpc_service):
		"""Test PII detection in multiple languages"""
		multilingual_pii_cases = [
			{
				'text': "Mi correo es juan.perez@empresa.com y teléfono 555-123-4567",
				'language': LanguageCode.ES,
				'expected_pii': ['email', 'phone']
			},
			{
				'text': "Mon email est jean.dupont@entreprise.fr et téléphone 01-23-45-67-89",
				'language': LanguageCode.FR,
				'expected_pii': ['email', 'phone']
			},
			{
				'text': "Meine E-Mail ist max.mustermann@firma.de und Telefon 030-12345678",
				'language': LanguageCode.DE,
				'expected_pii': ['email', 'phone']
			}
		]
		
		for test_case in multilingual_pii_cases:
			document = create_test_document(test_case['text'])
			document.language = test_case['language']
			
			security_context = {'user_id': 'test', 'tenant_id': 'test', 'user_roles': []}
			
			pii_result = await nlpc_service._classify_document_sensitivity(
				document, security_context
			)
			
			assert pii_result['pii_detected'] == True
			
			detected_types = [entity['type'] for entity in pii_result['sensitive_entities']]
			for expected_type in test_case['expected_pii']:
				assert expected_type in detected_types
	
	async def test_pii_false_positive_handling(self, nlpc_service):
		"""Test handling of PII false positives"""
		false_positive_cases = [
			"The ISBN number is 978-0-123456-78-9",  # ISBN, not SSN
			"Please call extension 123-45-6789",      # Extension, not SSN
			"The model number is ABC-123-DEF-456",    # Product code, not sensitive
			"Order ID: 4111-1111-1111-ABCD"          # Order ID, not credit card
		]
		
		for text in false_positive_cases:
			document = create_test_document(text)
			security_context = {'user_id': 'test', 'tenant_id': 'test', 'user_roles': []}
			
			pii_result = await nlpc_service._classify_document_sensitivity(
				document, security_context
			)
			
			# Should have sophisticated detection to minimize false positives
			if pii_result['pii_detected']:
				# If detected, should have low confidence or be properly categorized
				for entity in pii_result['sensitive_entities']:
					assert entity['confidence'] < 0.8 or entity['type'] in ['identifier', 'reference']

class TestEncryptionAndDataProtection:
	"""Test encryption and data protection features"""
	
	async def test_text_encryption_at_rest(self, nlpc_service):
		"""Test text encryption for data at rest"""
		sensitive_text = "This is highly confidential business information."
		document = create_test_document(sensitive_text)
		
		# Test encryption
		encryption_key = secrets.token_bytes(32)  # 256-bit key
		
		encrypted_result = await nlpc_service._encrypt_document_content(
			document, encryption_key, {'algorithm': 'AES-256-GCM'}
		)
		
		assert isinstance(encrypted_result, dict)
		assert 'encrypted_content' in encrypted_result
		assert 'encryption_metadata' in encrypted_result
		assert 'integrity_hash' in encrypted_result
		
		# Original content should not be visible in encrypted form
		encrypted_content = encrypted_result['encrypted_content']
		assert sensitive_text not in encrypted_content
		assert isinstance(encrypted_content, (str, bytes))
		
		# Test decryption
		decrypted_content = await nlpc_service._decrypt_document_content(
			encrypted_result, encryption_key
		)
		
		assert decrypted_content == sensitive_text
	
	async def test_processing_with_encryption(self, nlpc_service):
		"""Test NLP processing with encrypted content"""
		sensitive_document = create_test_document(
			"Confidential: Employee John Doe (SSN: 123-45-6789) salary information."
		)
		
		request = create_test_request([NLPTask.NAMED_ENTITY_RECOGNITION])
		
		security_context = {
			'user_id': 'authorized-user',
			'tenant_id': 'enterprise-tenant',
			'user_roles': ['nlp_admin'],
			'encryption_required': True,
			'data_classification': 'confidential'
		}
		
		# Process with encryption enabled
		result = await nlpc_service.secure_process_document(
			sensitive_document, request, security_context
		)
		
		assert_processing_result_valid(result)
		assert hasattr(result, 'encryption_applied')
		assert result.encryption_applied == True
		
		# Results should be available but original sensitive content protected
		assert 'entities' in result.results
		# Should detect entities but with privacy protections
		entities = result.results['entities']
		for entity in entities:
			if entity.get('type') == 'ssn':
				assert '123-45-6789' not in str(entity)  # Should be masked/encrypted
	
	async def test_key_rotation_handling(self, nlpc_service):
		"""Test encryption key rotation procedures"""
		documents = []
		
		# Create documents with different encryption keys (simulating rotation)
		old_key = secrets.token_bytes(32)
		new_key = secrets.token_bytes(32)
		
		for i in range(5):
			doc = create_test_document(f"Document {i} with sensitive data.")
			
			# Encrypt with old key
			encrypted_doc = await nlpc_service._encrypt_document_content(
				doc, old_key, {'key_version': 1}
			)
			documents.append((doc, encrypted_doc, old_key))
		
		# Simulate key rotation
		key_rotation_result = await nlpc_service._rotate_encryption_keys(
			documents, old_key, new_key, {'key_version': 2}
		)
		
		assert key_rotation_result['rotated_count'] == len(documents)
		assert key_rotation_result['failed_count'] == 0
		
		# Verify documents can be decrypted with new key
		for doc, encrypted_doc, _ in documents:
			decrypted = await nlpc_service._decrypt_document_content(
				encrypted_doc, new_key
			)
			assert decrypted == doc.content
	
	async def test_homomorphic_encryption_operations(self, nlpc_service):
		"""Test homomorphic encryption for privacy-preserving processing"""
		# Note: This is a simplified test - real homomorphic encryption is complex
		sensitive_numbers = [100, 200, 300, 400, 500]
		
		# Simulate homomorphic encryption of numeric data
		encrypted_values = []
		for value in sensitive_numbers:
			encrypted_value = await nlpc_service._homomorphic_encrypt(
				value, {'scheme': 'CKKS', 'precision': 40}
			)
			encrypted_values.append(encrypted_value)
		
		# Perform operations on encrypted data
		encrypted_sum = await nlpc_service._homomorphic_sum(encrypted_values)
		encrypted_average = await nlpc_service._homomorphic_divide(
			encrypted_sum, len(encrypted_values)
		)
		
		# Decrypt result
		decrypted_average = await nlpc_service._homomorphic_decrypt(encrypted_average)
		
		expected_average = sum(sensitive_numbers) / len(sensitive_numbers)
		
		# Should be close to expected (homomorphic encryption has precision limitations)
		assert abs(decrypted_average - expected_average) < 1.0

class TestAuditTrailAndCompliance:
	"""Test audit trail generation and compliance features"""
	
	async def test_comprehensive_audit_logging(self, nlpc_service, sample_documents, sample_security_context):
		"""Test comprehensive audit trail logging"""
		document = sample_documents[0]
		request = create_test_request([NLPTask.SENTIMENT_ANALYSIS])
		
		with patch('apg.capabilities.audit_compliance.log_event') as mock_audit:
			result = await nlpc_service.secure_process_document(
				document, request, sample_security_context
			)
			
			# Should have logged multiple events
			assert mock_audit.call_count >= 3
			
			# Verify audit event types
			logged_events = [call[0][0] for call in mock_audit.call_args_list]
			event_types = [event['event_type'] for event in logged_events]
			
			expected_event_types = [
				'nlp_processing_started',
				'document_classified',
				'nlp_processing_completed'
			]
			
			for expected_type in expected_event_types:
				assert expected_type in event_types
			
			# Verify audit event structure
			first_event = logged_events[0]
			required_fields = [
				'event_type', 'timestamp', 'user_id', 'tenant_id',
				'resource_id', 'action', 'result', 'ip_address', 'user_agent'
			]
			
			for field in required_fields:
				assert field in first_event
	
	async def test_audit_trail_integrity(self, nlpc_service, sample_documents):
		"""Test audit trail integrity and tamper detection"""
		document = sample_documents[0]
		security_context = {'user_id': 'test', 'tenant_id': 'test', 'user_roles': []}
		
		with patch('apg.capabilities.audit_compliance.create_audit_hash') as mock_hash, \
			 patch('apg.capabilities.audit_compliance.verify_audit_chain') as mock_verify:
			
			# Mock hash creation
			test_hash = hashlib.sha256(b"test_audit_data").hexdigest()
			mock_hash.return_value = test_hash
			
			# Mock chain verification
			mock_verify.return_value = {
				'valid': True,
				'chain_length': 1,
				'hash_verified': True,
				'timestamp_verified': True
			}
			
			request = create_test_request([NLPTask.SENTIMENT_ANALYSIS])
			await nlpc_service.secure_process_document(
				document, request, security_context
			)
			
			# Verify audit hash was created
			mock_hash.assert_called()
			
			# Verify chain integrity
			chain_verification = await nlpc_service._verify_audit_chain('test-tenant')
			assert chain_verification['valid'] == True
			assert chain_verification['hash_verified'] == True
	
	async def test_gdpr_compliance_features(self, nlpc_service):
		"""Test GDPR compliance features and data subject rights"""
		# Test with EU user data
		eu_document = create_test_document(
			"Hello, my name is Marie Dubois from Paris, France. "
			"My email is marie.dubois@example.fr and I consent to data processing."
		)
		
		gdpr_context = {
			'user_id': 'eu-user-123',
			'tenant_id': 'eu-tenant',
			'user_roles': ['data_subject'],
			'jurisdiction': 'EU',
			'consent_provided': True,
			'lawful_basis': 'consent'
		}
		
		# Test GDPR compliance check
		compliance_result = await nlpc_service._check_gdpr_compliance(
			eu_document, gdpr_context
		)
		
		assert isinstance(compliance_result, dict)
		assert 'pii_detected' in compliance_result
		assert 'lawful_basis' in compliance_result
		assert 'data_subject_rights' in compliance_result
		assert 'retention_period' in compliance_result
		
		# Verify data subject rights
		rights = compliance_result['data_subject_rights']
		expected_rights = ['access', 'rectification', 'erasure', 'portability', 'restriction']
		for right in expected_rights:
			assert right in rights
		
		# Test right to erasure (data deletion)
		deletion_result = await nlpc_service._execute_data_subject_right(
			'erasure', gdpr_context, {'document_ids': [eu_document.document_id]}
		)
		
		assert deletion_result['right_exercised'] == 'erasure'
		assert deletion_result['status'] == 'completed'
		assert deletion_result['affected_records'] >= 1
	
	async def test_hipaa_compliance_features(self, nlpc_service):
		"""Test HIPAA compliance for healthcare data"""
		healthcare_document = create_test_document(
			"Patient John Smith, DOB 01/01/1980, MRN 123456, "
			"diagnosed with hypertension. Prescription: Lisinopril 10mg daily."
		)
		
		hipaa_context = {
			'user_id': 'healthcare-provider',
			'tenant_id': 'hospital-tenant',
			'user_roles': ['healthcare_professional'],
			'jurisdiction': 'US',
			'covered_entity': True,
			'business_associate': False
		}
		
		# Test PHI detection and classification
		phi_result = await nlpc_service._detect_phi(healthcare_document, hipaa_context)
		
		assert phi_result['phi_detected'] == True
		assert 'phi_categories' in phi_result
		
		expected_phi_categories = ['name', 'date_of_birth', 'medical_record_number', 'medical_condition']
		detected_categories = phi_result['phi_categories']
		
		for category in expected_phi_categories:
			assert category in detected_categories
		
		# Test minimum necessary rule compliance
		access_result = await nlpc_service._apply_minimum_necessary_rule(
			healthcare_document, hipaa_context, {'purpose': 'treatment'}
		)
		
		assert access_result['access_granted'] == True
		assert 'restricted_fields' in access_result
		assert 'audit_logged' in access_result
	
	async def test_sox_compliance_financial_data(self, nlpc_service):
		"""Test SOX compliance for financial documents"""
		financial_document = create_test_document(
			"Q4 2023 Revenue: $1,250,000. Net Income: $180,000. "
			"Accounts Receivable: $340,000. Reviewed by CFO Jane Smith."
		)
		
		sox_context = {
			'user_id': 'financial-analyst',
			'tenant_id': 'public-company',
			'user_roles': ['financial_user'],
			'document_type': 'financial_statement',
			'sox_section': '302',  # CEO/CFO certification
			'public_company': True
		}
		
		# Test financial data classification
		sox_result = await nlpc_service._apply_sox_controls(
			financial_document, sox_context
		)
		
		assert sox_result['sox_applicable'] == True
		assert 'financial_data_detected' in sox_result
		assert 'control_requirements' in sox_result
		
		# Verify required controls
		controls = sox_result['control_requirements']
		expected_controls = ['dual_approval', 'audit_trail', 'access_logging', 'data_retention']
		
		for control in expected_controls:
			assert control in controls
		
		# Test audit trail for financial processing
		assert sox_result['audit_trail_created'] == True
		assert 'retention_period_years' in sox_result
		assert sox_result['retention_period_years'] >= 7  # SOX requirement

class TestRBACSecurityEnforcement:
	"""Test Role-Based Access Control security enforcement"""
	
	async def test_role_based_task_restrictions(self, nlpc_service, sample_documents):
		"""Test task restrictions based on user roles"""
		document = sample_documents[0]
		
		role_test_cases = [
			{
				'roles': ['nlp_user'],
				'allowed_tasks': [NLPTask.SENTIMENT_ANALYSIS, NLPTask.LANGUAGE_DETECTION],
				'denied_tasks': [NLPTask.TEXT_GENERATION, NLPTask.QUESTION_ANSWERING]
			},
			{
				'roles': ['nlp_analyst'],
				'allowed_tasks': [NLPTask.SENTIMENT_ANALYSIS, NLPTask.NAMED_ENTITY_RECOGNITION, NLPTask.TEXT_CLASSIFICATION],
				'denied_tasks': [NLPTask.TEXT_GENERATION]
			},
			{
				'roles': ['nlp_admin'],
				'allowed_tasks': list(NLPTask),  # All tasks
				'denied_tasks': []
			}
		]
		
		for test_case in role_test_cases:
			# Test allowed tasks
			for task in test_case['allowed_tasks']:
				security_context = {
					'user_id': 'test-user',
					'tenant_id': 'test-tenant',
					'user_roles': test_case['roles'],
					'permissions': [f'nlp_{task.value}']
				}
				
				request = create_test_request([task])
				
				result = await nlpc_service.secure_process_document(
					document, request, security_context
				)
				
				# Should succeed
				assert_processing_result_valid(result)
				assert result.status != ProcessingStatus.FAILED
			
			# Test denied tasks
			for task in test_case['denied_tasks']:
				security_context = {
					'user_id': 'test-user',
					'tenant_id': 'test-tenant',
					'user_roles': test_case['roles'],
					'permissions': []  # No permissions for denied tasks
				}
				
				request = create_test_request([task])
				
				with pytest.raises(PermissionError):
					await nlpc_service._validate_rbac_permissions(
						test_case['roles'], [task], document
					)
	
	async def test_tenant_isolation_security(self, nlpc_service):
		"""Test security of tenant data isolation"""
		# Create documents for different tenants
		tenant_a_document = create_test_document("Confidential data for tenant A")
		tenant_a_document.tenant_id = "tenant-a"
		
		tenant_b_document = create_test_document("Confidential data for tenant B")
		tenant_b_document.tenant_id = "tenant-b"
		
		# User from tenant A should not access tenant B data
		tenant_a_context = {
			'user_id': 'user-a',
			'tenant_id': 'tenant-a',
			'user_roles': ['nlp_admin'],
			'permissions': ['nlp_all']
		}
		
		# Test cross-tenant access prevention
		with pytest.raises(PermissionError) as exc_info:
			request = create_test_request([NLPTask.SENTIMENT_ANALYSIS])
			request.tenant_id = "tenant-b"  # Different tenant
			
			await nlpc_service._validate_tenant_access(
				tenant_b_document, request, tenant_a_context
			)
		
		assert "tenant" in str(exc_info.value).lower()
		assert "access denied" in str(exc_info.value).lower()
	
	async def test_data_classification_access_control(self, nlpc_service):
		"""Test access control based on data classification levels"""
		classification_test_cases = [
			{
				'classification': 'public',
				'user_clearance': 'public',
				'access_granted': True
			},
			{
				'classification': 'internal',
				'user_clearance': 'public',
				'access_granted': False
			},
			{
				'classification': 'confidential',
				'user_clearance': 'internal',
				'access_granted': False
			},
			{
				'classification': 'confidential',
				'user_clearance': 'confidential',
				'access_granted': True
			},
			{
				'classification': 'restricted',
				'user_clearance': 'confidential',
				'access_granted': False
			},
			{
				'classification': 'restricted',
				'user_clearance': 'restricted',
				'access_granted': True
			}
		]
		
		for test_case in classification_test_cases:
			document = create_test_document("Classified document content")
			document.metadata['data_classification'] = test_case['classification']
			
			security_context = {
				'user_id': 'test-user',
				'tenant_id': 'test-tenant',
				'user_roles': ['nlp_user'],
				'security_clearance': test_case['user_clearance'],
				'data_classification': test_case['classification']
			}
			
			access_check = await nlpc_service._validate_data_classification_access(
				document, security_context
			)
			
			assert access_check['access_granted'] == test_case['access_granted']
			
			if not test_case['access_granted']:
				assert 'insufficient_clearance' in access_check['denial_reason']
	
	async def test_time_based_access_controls(self, nlpc_service, sample_documents):
		"""Test time-based access controls and session timeouts"""
		document = sample_documents[0]
		
		# Test with expired session
		expired_context = {
			'user_id': 'test-user',
			'tenant_id': 'test-tenant',
			'user_roles': ['nlp_user'],
			'session_start': datetime.utcnow() - timedelta(hours=25),  # Expired
			'session_timeout_hours': 24
		}
		
		with pytest.raises(PermissionError) as exc_info:
			await nlpc_service._validate_session_timeout(expired_context)
		
		assert "session expired" in str(exc_info.value).lower()
		
		# Test with valid session
		valid_context = {
			'user_id': 'test-user',
			'tenant_id': 'test-tenant',
			'user_roles': ['nlp_user'],
			'session_start': datetime.utcnow() - timedelta(hours=2),  # Valid
			'session_timeout_hours': 24
		}
		
		# Should not raise exception
		await nlpc_service._validate_session_timeout(valid_context)
		
		# Test business hours restrictions
		business_hours_context = {
			'user_id': 'restricted-user',
			'tenant_id': 'test-tenant',
			'user_roles': ['nlp_user'],
			'access_restrictions': {
				'business_hours_only': True,
				'timezone': 'UTC',
				'business_start': 9,  # 9 AM
				'business_end': 17    # 5 PM
			}
		}
		
		# Mock current time to outside business hours
		with patch('datetime.datetime') as mock_datetime:
			mock_datetime.utcnow.return_value = datetime(2023, 1, 1, 22, 0, 0)  # 10 PM
			
			access_result = await nlpc_service._validate_business_hours_access(
				business_hours_context
			)
			
			assert access_result['access_granted'] == False
			assert 'outside_business_hours' in access_result['denial_reason']

class TestSecurityIncidentDetection:
	"""Test security incident detection and response"""
	
	async def test_anomaly_detection(self, nlpc_service):
		"""Test detection of anomalous processing patterns"""
		# Simulate normal usage pattern
		normal_requests = []
		for i in range(10):
			document = create_test_document(f"Normal document {i}")
			request = create_test_request([NLPTask.SENTIMENT_ANALYSIS])
			normal_requests.append((document, request))
		
		# Process normal requests to establish baseline
		security_context = {'user_id': 'normal-user', 'tenant_id': 'test', 'user_roles': []}
		
		for document, request in normal_requests:
			await nlpc_service.secure_process_document(document, request, security_context)
		
		# Simulate anomalous activity
		anomalous_activities = [
			{
				'type': 'volume_spike',
				'description': 'Sudden spike in processing requests',
				'requests_per_minute': 100  # Unusual volume
			},
			{
				'type': 'unusual_tasks',
				'description': 'Requesting sensitive tasks not normally used',
				'tasks': [NLPTask.TEXT_GENERATION, NLPTask.QUESTION_ANSWERING]
			},
			{
				'type': 'off_hours_access',
				'description': 'Processing requests outside normal hours',
				'timestamp': datetime(2023, 1, 1, 3, 0, 0)  # 3 AM
			}
		]
		
		for anomaly in anomalous_activities:
			anomaly_detected = await nlpc_service._detect_security_anomaly(
				anomaly, security_context
			)
			
			assert anomaly_detected['anomaly_detected'] == True
			assert anomaly_detected['anomaly_type'] == anomaly['type']
			assert 'risk_score' in anomaly_detected
			assert anomaly_detected['risk_score'] > 0.5  # High risk
	
	async def test_brute_force_detection(self, nlpc_service):
		"""Test detection of brute force attacks"""
		# Simulate repeated failed authentication attempts
		failed_attempts = []
		for i in range(10):
			attempt = {
				'user_id': 'attacker',
				'tenant_id': 'target-tenant',
				'timestamp': datetime.utcnow() - timedelta(minutes=i),
				'success': False,
				'error_type': 'authentication_failed'
			}
			failed_attempts.append(attempt)
		
		brute_force_detection = await nlpc_service._detect_brute_force_attack(
			failed_attempts, {'time_window_minutes': 15, 'failure_threshold': 5}
		)
		
		assert brute_force_detection['attack_detected'] == True
		assert brute_force_detection['attack_type'] == 'brute_force'
		assert brute_force_detection['failed_attempts'] == 10
		assert 'source_ip' in brute_force_detection
		assert 'recommended_action' in brute_force_detection
		
		# Should recommend blocking
		assert brute_force_detection['recommended_action'] == 'block_ip'
	
	async def test_data_exfiltration_detection(self, nlpc_service):
		"""Test detection of potential data exfiltration"""
		# Simulate suspicious data access patterns
		suspicious_access = {
			'user_id': 'suspicious-user',
			'tenant_id': 'test-tenant',
			'documents_accessed': 500,  # Unusually high
			'time_period_minutes': 30,  # Short time frame
			'data_categories': ['confidential', 'restricted'],
			'download_requests': 50,
			'export_attempts': 10
		}
		
		exfiltration_check = await nlpc_service._detect_data_exfiltration(
			suspicious_access, {'max_documents_per_hour': 100}
		)
		
		assert exfiltration_check['potential_exfiltration'] == True
		assert 'risk_indicators' in exfiltration_check
		assert 'volume_anomaly' in exfiltration_check['risk_indicators']
		assert 'sensitive_data_access' in exfiltration_check['risk_indicators']
		assert exfiltration_check['risk_score'] > 0.8  # High risk
	
	async def test_incident_response_automation(self, nlpc_service):
		"""Test automated incident response"""
		# Simulate high-risk security incident
		incident = {
			'incident_id': uuid7str(),
			'type': 'data_breach_suspected',
			'severity': 'critical',
			'affected_users': ['user-123', 'user-456'],
			'affected_tenants': ['tenant-abc'],
			'detection_time': datetime.utcnow(),
			'indicators': [
				'unauthorized_access_attempt',
				'data_exfiltration_suspected',
				'privilege_escalation_detected'
			]
		}
		
		response_actions = await nlpc_service._execute_incident_response(incident)
		
		assert isinstance(response_actions, dict)
		assert 'immediate_actions' in response_actions
		assert 'containment_actions' in response_actions
		assert 'investigation_actions' in response_actions
		assert 'notification_actions' in response_actions
		
		# Verify immediate actions
		immediate_actions = response_actions['immediate_actions']
		expected_immediate = ['disable_affected_accounts', 'revoke_sessions', 'alert_security_team']
		
		for action in expected_immediate:
			assert action in immediate_actions
		
		# Verify containment actions
		containment_actions = response_actions['containment_actions']
		expected_containment = ['isolate_affected_systems', 'preserve_evidence', 'block_suspicious_ips']
		
		for action in expected_containment:
			assert action in containment_actions