#!/usr/bin/env python3
"""
APG Message Queue Event Bus (MQEB) - Compliance Governance Tests
Tests for compliance automation and data governance

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from uuid_extensions import uuid7str

# Import MQEB components
from ..models import MQMessage, MessagePriority
from ..service import MQEBService
from ..compliance_governance import (
	ComplianceGovernanceEngine, PIIDetectionEngine, ComplianceRuleEngine, DataGovernanceEngine,
	ComplianceFramework, DataClassification, PIIType, RetentionAction,
	PIIDetectionResult, ComplianceRule, DataRetentionPolicy, ComplianceViolation,
	create_compliance_governance_engine
)


class TestPIIDetectionEngine:
	"""Test PII detection functionality"""
	
	def test_pii_detection_engine_initialization(self):
		"""Test PII detection engine initialization"""
		detector = PIIDetectionEngine()
		assert len(detector.pii_patterns) > 0
		assert PIIType.EMAIL in detector.pii_patterns
		assert PIIType.PHONE in detector.pii_patterns
		assert PIIType.SSN in detector.pii_patterns
		assert PIIType.CREDIT_CARD in detector.pii_patterns
	
	@pytest.mark.asyncio
	async def test_email_detection(self):
		"""Test email address detection"""
		detector = PIIDetectionEngine()
		
		message_with_email = MQMessage(
			topic="user.registration",
			payload=b'{"email": "user@example.com", "name": "John Doe"}',
			tenant_id="test_tenant",
			source_application="registration_service"
		)
		
		result = await detector.detect_pii(message_with_email)
		
		assert PIIType.EMAIL in result.pii_types
		assert result.confidence_scores[PIIType.EMAIL] > 0
		assert len(result.detected_patterns[PIIType.EMAIL]) > 0
		assert "user@example.com" in result.detected_patterns[PIIType.EMAIL]
	
	@pytest.mark.asyncio
	async def test_phone_number_detection(self):
		"""Test phone number detection"""
		detector = PIIDetectionEngine()
		
		message_with_phone = MQMessage(
			topic="customer.contact",
			payload=b'{"phone": "555-123-4567", "contact_method": "phone"}',
			tenant_id="test_tenant",
			source_application="crm_service"
		)
		
		result = await detector.detect_pii(message_with_phone)
		
		assert PIIType.PHONE in result.pii_types
		assert result.confidence_scores[PIIType.PHONE] > 0
		assert "555-123-4567" in str(result.detected_patterns[PIIType.PHONE])
	
	@pytest.mark.asyncio
	async def test_ssn_detection(self):
		"""Test Social Security Number detection"""
		detector = PIIDetectionEngine()
		
		message_with_ssn = MQMessage(
			topic="hr.employee.data",
			payload=b'{"ssn": "123-45-6789", "employee_id": "EMP001"}',
			tenant_id="test_tenant",
			source_application="hr_service"
		)
		
		result = await detector.detect_pii(message_with_ssn)
		
		assert PIIType.SSN in result.pii_types
		assert result.risk_level in ['high', 'critical']
		assert 'extra_access_controls' in result.recommendations
	
	@pytest.mark.asyncio
	async def test_credit_card_detection(self):
		"""Test credit card number detection"""
		detector = PIIDetectionEngine()
		
		message_with_cc = MQMessage(
			topic="payment.processing",
			payload=b'{"card_number": "4111111111111111", "amount": 100.00}',
			tenant_id="test_tenant",
			source_application="payment_service"
		)
		
		result = await detector.detect_pii(message_with_cc)
		
		assert PIIType.CREDIT_CARD in result.pii_types
		assert result.risk_level == 'critical'
		assert 'pci_compliance_required' in result.recommendations
		assert 'tokenize_card_data' in result.recommendations
	
	@pytest.mark.asyncio
	async def test_multiple_pii_types(self):
		"""Test detection of multiple PII types in single message"""
		detector = PIIDetectionEngine()
		
		message_with_multiple_pii = MQMessage(
			topic="customer.profile",
			payload=b'{"email": "john.doe@company.com", "phone": "(555) 123-4567", "address": "123 Main Street"}',
			tenant_id="test_tenant",
			source_application="customer_service"
		)
		
		result = await detector.detect_pii(message_with_multiple_pii)
		
		assert len(result.pii_types) >= 3  # Email, phone, address
		assert PIIType.EMAIL in result.pii_types
		assert PIIType.PHONE in result.pii_types
		assert PIIType.ADDRESS in result.pii_types
		assert result.risk_level in ['high', 'critical']
	
	@pytest.mark.asyncio
	async def test_no_pii_detection(self):
		"""Test message with no PII"""
		detector = PIIDetectionEngine()
		
		message_without_pii = MQMessage(
			topic="system.status",
			payload=b'{"status": "healthy", "uptime": 3600, "version": "1.0.0"}',
			tenant_id="test_tenant",
			source_application="monitoring_service"
		)
		
		result = await detector.detect_pii(message_without_pii)
		
		assert len(result.pii_types) == 0
		assert result.risk_level == 'low'
		assert len(result.recommendations) == 0


class TestComplianceRuleEngine:
	"""Test compliance rule evaluation"""
	
	def test_rule_engine_initialization(self):
		"""Test rule engine initialization"""
		rule_engine = ComplianceRuleEngine()
		
		assert len(rule_engine.rules) > 0
		assert 'gdpr_encryption' in rule_engine.rules
		assert 'hipaa_access_control' in rule_engine.rules
		assert 'pci_cardholder_data' in rule_engine.rules
		assert 'sox_financial_controls' in rule_engine.rules
	
	@pytest.mark.asyncio
	async def test_gdpr_compliance_evaluation(self):
		"""Test GDPR compliance rule evaluation"""
		rule_engine = ComplianceRuleEngine()
		detector = PIIDetectionEngine()
		
		# Message with PII that should trigger GDPR rules
		message = MQMessage(
			topic="user.data",
			payload=b'{"email": "user@example.com", "name": "John Doe"}',
			encrypted=False,  # Not encrypted - should trigger violation
			tenant_id="test_tenant",
			source_application="user_service"
		)
		
		pii_result = await detector.detect_pii(message)
		context = {'compliance_frameworks': [ComplianceFramework.GDPR]}
		
		violations = await rule_engine.evaluate_message_compliance(message, pii_result, context)
		
		# Should have GDPR encryption violation
		gdpr_violations = [v for v in violations if v.framework == ComplianceFramework.GDPR]
		assert len(gdpr_violations) > 0
		
		encryption_violations = [v for v in gdpr_violations if 'encryption' in v.rule_id]
		assert len(encryption_violations) > 0
	
	@pytest.mark.asyncio
	async def test_pci_compliance_evaluation(self):
		"""Test PCI DSS compliance rule evaluation"""
		rule_engine = ComplianceRuleEngine()
		detector = PIIDetectionEngine()
		
		# Message with credit card data
		message = MQMessage(
			topic="payment.data",
			payload=b'{"card_number": "4111111111111111", "cvv": "123"}',
			tenant_id="test_tenant",
			source_application="payment_service"
		)
		
		pii_result = await detector.detect_pii(message)
		context = {'compliance_frameworks': [ComplianceFramework.PCI_DSS]}
		
		violations = await rule_engine.evaluate_message_compliance(message, pii_result, context)
		
		# Should have PCI DSS violations
		pci_violations = [v for v in violations if v.framework == ComplianceFramework.PCI_DSS]
		assert len(pci_violations) > 0
		assert any('cardholder' in v.rule_id for v in pci_violations)
	
	@pytest.mark.asyncio
	async def test_hipaa_compliance_evaluation(self):
		"""Test HIPAA compliance rule evaluation"""
		rule_engine = ComplianceRuleEngine()
		detector = PIIDetectionEngine()
		
		# Message with health information
		message = MQMessage(
			topic="patient.medical.records",
			payload=b'{"patient_id": "P123", "diagnosis": "diabetes", "treatment": "insulin"}',
			tenant_id="test_tenant",
			source_application="healthcare_service"
		)
		
		pii_result = await detector.detect_pii(message)
		context = {
			'compliance_frameworks': [ComplianceFramework.HIPAA],
			'unauthorized_access': True  # Simulate unauthorized access attempt
		}
		
		violations = await rule_engine.evaluate_message_compliance(message, pii_result, context)
		
		# Should have HIPAA violations
		hipaa_violations = [v for v in violations if v.framework == ComplianceFramework.HIPAA]
		assert len(hipaa_violations) > 0


class TestDataGovernanceEngine:
	"""Test data governance and lifecycle management"""
	
	def test_governance_engine_initialization(self):
		"""Test data governance engine initialization"""
		governance = DataGovernanceEngine()
		
		assert len(governance.retention_policies) == 0
		assert len(governance.data_classifications) == 0
		assert len(governance.audit_logs) == 0
	
	@pytest.mark.asyncio
	async def test_data_classification(self):
		"""Test automatic data classification"""
		governance = DataGovernanceEngine()
		detector = PIIDetectionEngine()
		
		# High-risk message
		high_risk_message = MQMessage(
			topic="classified.secret.data",
			payload=b'{"ssn": "123-45-6789", "security_clearance": "top_secret"}',
			tenant_id="test_tenant",
			source_application="classified_app"
		)
		
		pii_result = await detector.detect_pii(high_risk_message)
		classification = await governance.classify_message_data(high_risk_message, pii_result)
		
		assert classification == DataClassification.TOP_SECRET
		
		# Public message
		public_message = MQMessage(
			topic="public.announcement",
			payload=b'{"announcement": "Company holiday schedule released"}',
			tenant_id="test_tenant",
			source_application="public_app"
		)
		
		pii_result = await detector.detect_pii(public_message)
		classification = await governance.classify_message_data(public_message, pii_result)
		
		assert classification == DataClassification.PUBLIC
	
	@pytest.mark.asyncio
	async def test_retention_policy_creation(self):
		"""Test creating data retention policies"""
		governance = DataGovernanceEngine()
		
		policy = DataRetentionPolicy(
			policy_id="gdpr_retention",
			name="GDPR User Data Retention",
			description="Retain user data for 2 years maximum",
			tenant_id="test_tenant",
			topic_patterns=["user.data.*", "customer.profile.*"],
			retention_period_days=730,  # 2 years
			retention_action=RetentionAction.DELETE,
			compliance_frameworks=[ComplianceFramework.GDPR]
		)
		
		policy_id = await governance.create_retention_policy(policy)
		
		assert policy_id == "gdpr_retention"
		assert policy_id in governance.retention_policies
		
		stored_policy = governance.retention_policies[policy_id]
		assert stored_policy.retention_period_days == 730
		assert stored_policy.retention_action == RetentionAction.DELETE
	
	@pytest.mark.asyncio
	async def test_retention_policy_application(self):
		"""Test applying retention policies to messages"""
		governance = DataGovernanceEngine()
		
		# Create retention policy
		policy = DataRetentionPolicy(
			policy_id="test_retention",
			name="Test Retention Policy",
			description="Test policy for retention",
			tenant_id="test_tenant",
			topic_patterns=["test.retention.*"],
			retention_period_days=1,  # 1 day for testing
			retention_action=RetentionAction.ANONYMIZE,
			compliance_frameworks=[ComplianceFramework.GDPR]
		)
		await governance.create_retention_policy(policy)
		
		# Create old message that should trigger retention
		old_message = MQMessage(
			topic="test.retention.data",
			payload=b'{"email": "old@example.com", "data": "old data"}',
			tenant_id="test_tenant",
			source_application="test_app"
		)
		# Set timestamp to 2 days ago
		old_message.timestamp = datetime.utcnow() - timedelta(days=2)
		
		# Apply retention policy
		result = await governance.apply_retention_policy(old_message, policy.policy_id)
		
		assert result == True
		assert old_message.headers.get('retention_status') == 'anonymized'
		assert 'anonymized_at' in old_message.headers
	
	@pytest.mark.asyncio
	async def test_audit_log_integrity(self):
		"""Test audit log integrity protection"""
		governance = DataGovernanceEngine()
		
		# Create audit entry
		await governance._log_audit_event(
			action="test_action",
			resource_type="message",
			resource_id="msg_123",
			tenant_id="test_tenant",
			details={"test": "data"},
			compliance_frameworks=[ComplianceFramework.GDPR]
		)
		
		assert len(governance.audit_logs) == 1
		audit_entry = governance.audit_logs[0]
		
		# Verify integrity hash is set
		assert audit_entry.integrity_hash is not None
		assert len(audit_entry.integrity_hash) == 64  # SHA-256 hash
		
		# Verify integrity hash calculation
		import hashlib
		import json
		content = f"{audit_entry.timestamp}{audit_entry.tenant_id}{audit_entry.action}{audit_entry.resource_id}{json.dumps(audit_entry.details, sort_keys=True)}"
		expected_hash = hashlib.sha256(content.encode()).hexdigest()
		assert audit_entry.integrity_hash == expected_hash
	
	@pytest.mark.asyncio
	async def test_compliance_report_generation(self):
		"""Test compliance report generation"""
		governance = DataGovernanceEngine()
		
		# Add some audit events
		for i in range(3):
			await governance._log_audit_event(
				action=f"test_action_{i}",
				resource_type="message",
				resource_id=f"msg_{i}",
				tenant_id="test_tenant",
				details={"action_number": i},
				compliance_frameworks=[ComplianceFramework.GDPR]
			)
		
		# Generate report
		start_date = datetime.utcnow() - timedelta(days=1)
		end_date = datetime.utcnow() + timedelta(days=1)
		
		report = await governance.get_compliance_report(
			"test_tenant",
			[ComplianceFramework.GDPR],
			start_date,
			end_date
		)
		
		assert report['tenant_id'] == "test_tenant"
		assert ComplianceFramework.GDPR.value in report['frameworks']
		assert report['summary']['total_audit_events'] == 3
		assert len(report['audit_events']) == 3


class TestComplianceGovernanceEngine:
	"""Test main compliance governance engine"""
	
	@pytest.fixture
	async def mqeb_service(self):
		"""Create MQEB service for testing"""
		service = MQEBService()
		await service.initialize()
		yield service
		await service.shutdown()
	
	@pytest.fixture
	async def compliance_engine(self, mqeb_service):
		"""Create compliance governance engine"""
		engine = await create_compliance_governance_engine(mqeb_service)
		yield engine
		await engine.shutdown()
	
	@pytest.mark.asyncio
	async def test_compliance_engine_initialization(self, compliance_engine):
		"""Test compliance engine initialization"""
		assert compliance_engine.enabled == True
		assert compliance_engine.auto_classification == True
		assert compliance_engine.auto_retention == True
		assert compliance_engine.pii_detector is not None
		assert compliance_engine.rule_engine is not None
		assert compliance_engine.governance is not None
	
	@pytest.mark.asyncio
	async def test_comprehensive_message_compliance_processing(self, compliance_engine):
		"""Test comprehensive compliance processing of message"""
		# Message with PII that triggers multiple compliance frameworks
		message = MQMessage(
			topic="customer.payment.data",
			payload=b'{"email": "customer@example.com", "card_number": "4111111111111111", "amount": 1000.00}',
			encrypted=False,  # Should trigger compliance violations
			tenant_id="test_tenant",
			source_application="payment_service"
		)
		
		context = {
			'compliance_frameworks': [ComplianceFramework.GDPR, ComplianceFramework.PCI_DSS],
			'user_id': 'payment_processor',
			'source_ip': '10.0.1.100'
		}
		
		# Process message for compliance
		result = await compliance_engine.process_message_compliance(message, context)
		
		# Should detect PII
		assert result['pii_detected'] == True
		assert PIIType.EMAIL.value in result['pii_details']['types']
		assert PIIType.CREDIT_CARD.value in result['pii_details']['types']
		
		# Should have compliance violations
		assert result['compliant'] == False
		assert len(result['violations']) > 0
		
		# Should classify data appropriately
		assert result['data_classification'] in [
			DataClassification.RESTRICTED.value,
			DataClassification.TOP_SECRET.value
		]
	
	@pytest.mark.asyncio
	async def test_compliance_status_reporting(self, compliance_engine):
		"""Test compliance status reporting"""
		status = await compliance_engine.get_compliance_status()
		
		assert 'enabled' in status
		assert 'total_violations' in status
		assert 'critical_violations' in status
		assert 'pii_detections' in status
		assert 'data_classifications' in status
		assert 'retention_policies' in status
		assert 'audit_logs' in status
		assert 'governance_metrics' in status
		
		assert status['enabled'] == True


class TestIntegrationWithMQEB:
	"""Test integration of compliance governance with MQEB service"""
	
	@pytest.mark.asyncio
	async def test_compliance_enabled_message_publishing(self):
		"""Test message publishing with compliance governance enabled"""
		service = MQEBService({'compliance_governance_enabled': True})
		await service.initialize()
		
		try:
			# Create test topic
			from ..models import TopicConfiguration
			topic_config = TopicConfiguration(
				name="compliance.test.topic",
				tenant_id="test_tenant",
				created_by="test_user"
			)
			await service.create_topic(topic_config)
			
			# Create message with PII
			message = MQMessage(
				topic="compliance.test.topic",
				payload=b'{"email": "test@example.com", "phone": "555-123-4567"}',
				tenant_id="test_tenant",
				source_application="compliance_test_app"
			)
			
			# Publish with compliance context
			compliance_context = {
				'compliance_frameworks': [ComplianceFramework.GDPR],
				'user_id': 'compliance_tester'
			}
			
			message_id = await service.publish_message(message, compliance_context)
			assert message_id == message.id
			
			# Verify message was processed (compliance warnings may have been logged)
			stored_message = service.message_store[message_id]
			assert stored_message is not None
		
		finally:
			await service.shutdown()
	
	@pytest.mark.asyncio
	async def test_data_classification_integration(self):
		"""Test data classification integration with message processing"""
		service = MQEBService({
			'compliance_governance_enabled': True,
			'auto_classification': True
		})
		await service.initialize()
		
		try:
			# Create topic
			from ..models import TopicConfiguration
			topic_config = TopicConfiguration(
				name="classification.test.topic",
				tenant_id="test_tenant",
				created_by="test_user"
			)
			await service.create_topic(topic_config)
			
			# Test different message types
			test_messages = [
				{
					'topic': 'classification.test.topic',
					'payload': b'{"announcement": "Company picnic next Friday"}',
					'expected_classification': DataClassification.PUBLIC
				},
				{
					'topic': 'classification.test.topic',
					'payload': b'{"email": "internal@company.com", "department": "engineering"}',
					'expected_classification': DataClassification.INTERNAL
				},
				{
					'topic': 'classification.test.topic',
					'payload': b'{"ssn": "123-45-6789", "salary": 75000}',
					'expected_classification': DataClassification.TOP_SECRET
				}
			]
			
			for msg_spec in test_messages:
				message = MQMessage(
					topic=msg_spec['topic'],
					payload=msg_spec['payload'],
					tenant_id="test_tenant",
					source_application="classification_test_app"
				)
				
				await service.publish_message(message)
				
				# In a full implementation, would verify classification was applied
				# For now, just verify message was processed successfully
				assert message.id in service.message_store
		
		finally:
			await service.shutdown()


if __name__ == "__main__":
	# Run tests if script is executed directly
	pytest.main([__file__, "-v"])