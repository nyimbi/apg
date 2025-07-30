#!/usr/bin/env python3
"""
Unit Tests for APG Notification Security Engine

Tests for security functionality including data validation, encryption,
compliance checking, audit logging, and privacy protection.

Author: Nyimbi Odero
Company: Datacraft
Copyright: © 2025
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

from ..security_engine import (
    SecurityEngine, create_security_engine, EncryptionManager,
    AuditLogger, ConsentManager, DataSubjectRightsManager,
    ComplianceChecker, SecurityValidator, ComplianceRegime,
    AuditEventType, LocationPrivacyLevel
)
from ..models import *
from .fixtures import *
from .utils import *

class TestSecurityValidator:
    """Test security validation utilities"""
    
    def test_email_validation_valid(self):
        """Test valid email validation"""
        validator = SecurityValidator()
        
        valid_emails = [
            'test@example.com',
            'user.name@domain.org',
            'firstname+lastname@company.co.uk',
            'test123@sub.domain.com'
        ]
        
        for email in valid_emails:
            assert validator.validate_email(email), f"Email should be valid: {email}"
    
    def test_email_validation_invalid(self):
        """Test invalid email validation"""
        validator = SecurityValidator()
        
        invalid_emails = [
            'not-an-email',
            '@domain.com',
            'test@',
            'test..test@domain.com',
            'test@domain',
            '<script>alert("xss")</script>@domain.com',
            'javascript:alert("xss")@domain.com'
        ]
        
        for email in invalid_emails:
            assert not validator.validate_email(email), f"Email should be invalid: {email}"
    
    def test_phone_validation_valid(self):
        """Test valid phone number validation"""
        validator = SecurityValidator()
        
        valid_phones = [
            '+1-555-123-4567',
            '+44 20 7946 0958',
            '+33 1 42 86 83 26',
            '1-555-123-4567',
            '(555) 123-4567'
        ]
        
        for phone in valid_phones:
            assert validator.validate_phone_number(phone), f"Phone should be valid: {phone}"
    
    def test_phone_validation_invalid(self):
        """Test invalid phone number validation"""
        validator = SecurityValidator()
        
        invalid_phones = [
            'not-a-phone',
            '123',
            '+1-555-DROP-TABLE',
            'javascript:alert("xss")'
        ]
        
        for phone in invalid_phones:
            assert not validator.validate_phone_number(phone), f"Phone should be invalid: {phone}"
    
    def test_input_sanitization(self):
        """Test input sanitization"""
        validator = SecurityValidator()
        
        test_cases = [
            ('Hello world', 'Hello world'),  # Safe input unchanged
            ('<script>alert("xss")</script>', 'alert("xss")'),  # Script tags removed
            ('Test &amp; Co', 'Test &amp;amp; Co'),  # HTML entities encoded
            ('javascript:alert("bad")', 'alert("bad")'),  # JavaScript protocol removed
            ('<iframe src="evil"></iframe>', ''),  # Iframe removed
        ]
        
        for input_data, expected in test_cases:
            result = validator.sanitize_input(input_data)
            # Check that dangerous content is removed/neutralized
            assert '<script' not in result.lower()
            assert 'javascript:' not in result.lower()
            assert '<iframe' not in result.lower()
    
    def test_secure_token_generation(self):
        """Test secure token generation"""
        validator = SecurityValidator()
        
        # Generate multiple tokens
        tokens = [validator.generate_secure_token() for _ in range(10)]
        
        # All tokens should be unique
        assert len(set(tokens)) == 10
        
        # Tokens should be of expected length (URL-safe base64)
        for token in tokens:
            assert len(token) >= 32
            assert all(c.isalnum() or c in '-_' for c in token)

class TestEncryptionManager:
    """Test encryption manager functionality"""
    
    def test_encryption_decryption(self):
        """Test basic encryption and decryption"""
        manager = EncryptionManager()
        
        test_data = "Sensitive information that needs encryption"
        
        # Encrypt data
        encrypted = manager.encrypt_data(test_data, context="test")
        
        assert encrypted is not None
        assert 'encrypted_data' in encrypted
        assert 'encryption_method' in encrypted
        assert 'encrypted_at' in encrypted
        assert 'key_id' in encrypted
        
        # Decrypt data
        decrypted = manager.decrypt_data(encrypted)
        assert decrypted == test_data
    
    def test_encryption_different_contexts(self):
        """Test encryption with different contexts"""
        manager = EncryptionManager()
        
        test_data = "Test data"
        
        # Encrypt with different contexts
        encrypted1 = manager.encrypt_data(test_data, context="context1")
        encrypted2 = manager.encrypt_data(test_data, context="context2")
        
        # Encrypted data should be different even for same input
        assert encrypted1['encrypted_data'] != encrypted2['encrypted_data']
        
        # But both should decrypt to same original data
        assert manager.decrypt_data(encrypted1) == test_data
        assert manager.decrypt_data(encrypted2) == test_data
    
    def test_key_rotation(self):
        """Test encryption key rotation"""
        manager = EncryptionManager()
        
        # Get original key ID
        original_key_id = manager._get_current_key_id()
        
        # Rotate keys
        rotation_result = manager.rotate_keys()
        
        assert 'old_key_id' in rotation_result
        assert 'new_key_id' in rotation_result
        assert 'rotated_at' in rotation_result
        assert rotation_result['old_key_id'] == original_key_id
        assert rotation_result['new_key_id'] != original_key_id
        
        # New key ID should be different
        new_key_id = manager._get_current_key_id()
        assert new_key_id == rotation_result['new_key_id']
        assert new_key_id != original_key_id

class TestAuditLogger:
    """Test audit logging functionality"""
    
    @pytest.mark.asyncio
    async def test_log_audit_event(self, test_config):
        """Test logging audit events"""
        logger = AuditLogger(test_config['test_tenant_id'])
        
        # Log an event
        event_id = await logger.log_event(
            event_type=AuditEventType.DATA_ACCESS,
            resource_type="user_profile",
            resource_id="user-123",
            action="profile_viewed",
            user_id="admin-user",
            details={'fields_accessed': ['email', 'name']},
            ip_address="192.168.1.1"
        )
        
        assert event_id is not None
        assert event_id.startswith('audit_')
        assert len(logger.audit_events) == 1
        
        # Verify event details
        event = logger.audit_events[0]
        assert event.event_type == AuditEventType.DATA_ACCESS
        assert event.resource_type == "user_profile"
        assert event.resource_id == "user-123"
        assert event.action == "profile_viewed"
        assert event.user_id == "admin-user"
        assert event.ip_address == "192.168.1.1"
    
    @pytest.mark.asyncio
    async def test_audit_event_risk_scoring(self, test_config):
        """Test audit event risk scoring"""
        logger = AuditLogger(test_config['test_tenant_id'])
        
        # Log high-risk event
        await logger.log_event(
            event_type=AuditEventType.DATA_DELETION,
            resource_type="personal_data",
            resource_id="sensitive-data-123",
            action="bulk_delete",
            details={'bulk_operation': True, 'records_affected': 1000}
        )
        
        # Log low-risk event
        await logger.log_event(
            event_type=AuditEventType.DATA_ACCESS,
            resource_type="public_data",
            resource_id="public-123",
            action="view"
        )
        
        assert len(logger.audit_events) == 2
        
        # High-risk event should have higher risk score
        high_risk_event = logger.audit_events[0]
        low_risk_event = logger.audit_events[1]
        
        assert high_risk_event.risk_score > low_risk_event.risk_score
        assert high_risk_event.risk_score > 0.7  # Should be high risk
    
    @pytest.mark.asyncio
    async def test_query_audit_events(self, test_config):
        """Test querying audit events"""
        logger = AuditLogger(test_config['test_tenant_id'])
        
        # Log multiple events
        base_time = datetime.utcnow()
        
        events_data = [
            (AuditEventType.DATA_ACCESS, "user-1", 0.2),
            (AuditEventType.DATA_MODIFICATION, "user-2", 0.5),
            (AuditEventType.DATA_DELETION, "user-1", 0.8),
            (AuditEventType.SECURITY_VIOLATION, "user-3", 0.9)
        ]
        
        for i, (event_type, user_id, risk_score) in enumerate(events_data):
            await logger.log_event(
                event_type=event_type,
                resource_type="test_resource",
                resource_id=f"resource-{i}",
                action="test_action",
                user_id=user_id
            )
            # Manually set risk score for testing
            logger.audit_events[-1].risk_score = risk_score
        
        # Query all events
        all_events = await logger.query_audit_events(
            start_date=base_time - timedelta(minutes=1),
            end_date=base_time + timedelta(minutes=1)
        )
        assert len(all_events) == 4
        
        # Query by event type
        access_events = await logger.query_audit_events(
            start_date=base_time - timedelta(minutes=1),
            end_date=base_time + timedelta(minutes=1),
            event_types=[AuditEventType.DATA_ACCESS]
        )
        assert len(access_events) == 1
        
        # Query by user
        user1_events = await logger.query_audit_events(
            start_date=base_time - timedelta(minutes=1),
            end_date=base_time + timedelta(minutes=1),
            user_id="user-1"
        )
        assert len(user1_events) == 2
        
        # Query by risk score
        high_risk_events = await logger.query_audit_events(
            start_date=base_time - timedelta(minutes=1),
            end_date=base_time + timedelta(minutes=1),
            min_risk_score=0.7
        )
        assert len(high_risk_events) == 2

class TestConsentManager:
    """Test consent management functionality"""
    
    @pytest.mark.asyncio
    async def test_record_consent(self, test_config):
        """Test recording user consent"""
        manager = ConsentManager(test_config['test_tenant_id'])
        
        # Record consent
        consent_id = await manager.record_consent(
            user_id="user-123",
            purpose="marketing_communications",
            granted=True,
            consent_method="explicit",
            data_categories=["email", "preferences"],
            processing_purposes=["marketing", "analytics"]
        )
        
        assert consent_id is not None
        assert consent_id.startswith('consent_')
        assert consent_id in manager.consent_records
        
        # Verify consent record
        consent = manager.consent_records[consent_id]
        assert consent.user_id == "user-123"
        assert consent.purpose == "marketing_communications"
        assert consent.granted is True
        assert consent.consent_method == "explicit"
        assert "email" in consent.data_categories
        assert "marketing" in consent.processing_purposes
    
    @pytest.mark.asyncio
    async def test_withdraw_consent(self, test_config):
        """Test withdrawing consent"""
        manager = ConsentManager(test_config['test_tenant_id'])
        
        # Record consent first
        consent_id = await manager.record_consent(
            user_id="user-123",
            purpose="marketing_communications",
            granted=True
        )
        
        # Withdraw consent
        success = await manager.withdraw_consent("user-123", consent_id)
        assert success is True
        
        # Verify consent was withdrawn
        consent = manager.consent_records[consent_id]
        assert consent.granted is False
        assert consent.withdrawal_date is not None
    
    @pytest.mark.asyncio
    async def test_check_consent(self, test_config):
        """Test checking consent status"""
        manager = ConsentManager(test_config['test_tenant_id'])
        
        # Record consent
        await manager.record_consent(
            user_id="user-123",
            purpose="marketing_communications",
            granted=True
        )
        
        # Check valid consent
        has_consent, consent_id = await manager.check_consent(
            user_id="user-123",
            purpose="marketing_communications"
        )
        assert has_consent is True
        assert consent_id is not None
        
        # Check non-existent consent
        has_consent, consent_id = await manager.check_consent(
            user_id="user-123",
            purpose="sms_marketing"
        )
        assert has_consent is False
        assert consent_id is None
    
    @pytest.mark.asyncio
    async def test_consent_expiry(self, test_config):
        """Test consent expiry handling"""
        manager = ConsentManager(test_config['test_tenant_id'])
        
        # Record consent with expiry
        consent_id = await manager.record_consent(
            user_id="user-123",
            purpose="marketing_communications",
            granted=True,
            expiry_days=1  # Expires in 1 day
        )
        
        # Should be valid now
        has_consent, _ = await manager.check_consent(
            user_id="user-123",
            purpose="marketing_communications"
        )
        assert has_consent is True
        
        # Manually expire consent for testing
        consent = manager.consent_records[consent_id]
        consent.expiry_date = datetime.utcnow() - timedelta(hours=1)
        
        # Should be expired now
        has_consent, result = await manager.check_consent(
            user_id="user-123",
            purpose="marketing_communications",
            check_expiry=True
        )
        assert has_consent is False
        assert result == "expired"
    
    @pytest.mark.asyncio
    async def test_get_user_consents(self, test_config):
        """Test getting all user consents"""
        manager = ConsentManager(test_config['test_tenant_id'])
        
        # Record multiple consents for user
        purposes = ["marketing_communications", "analytics", "personalization"]
        
        for purpose in purposes:
            await manager.record_consent(
                user_id="user-123",
                purpose=purpose,
                granted=True
            )
        
        # Get all consents for user
        user_consents = await manager.get_user_consents("user-123")
        
        assert len(user_consents) == 3
        consent_purposes = [c.purpose for c in user_consents]
        for purpose in purposes:
            assert purpose in consent_purposes

class TestDataSubjectRightsManager:
    """Test data subject rights management"""
    
    @pytest.mark.asyncio
    async def test_submit_access_request(self, test_config):
        """Test submitting data access request"""
        manager = DataSubjectRightsManager(test_config['test_tenant_id'])
        
        # Submit access request
        request_id = await manager.submit_request(
            user_id="user-123",
            request_type="access",
            details={'data_categories': ['profile', 'communications']},
            verification_method="email"
        )
        
        assert request_id is not None
        assert request_id.startswith('dsr_')
        assert request_id in manager.requests
        
        # Verify request details
        request = manager.requests[request_id]
        assert request.user_id == "user-123"
        assert request.request_type == "access"
        assert request.status == "pending"
        assert request.verification_method == "email"
    
    @pytest.mark.asyncio
    async def test_process_access_request(self, test_config):
        """Test processing data access request"""
        manager = DataSubjectRightsManager(test_config['test_tenant_id'])
        
        # Submit and process access request
        request_id = await manager.submit_request(
            user_id="user-123",
            request_type="access"
        )
        
        # Process the request
        result = await manager.process_access_request(request_id)
        
        assert result is not None
        assert 'request_id' in result
        assert 'user_id' in result
        assert 'data' in result
        assert 'completed_at' in result
        
        # Verify request status updated
        request = manager.requests[request_id]
        assert request.status == "completed"
        assert request.completed_at is not None
    
    @pytest.mark.asyncio
    async def test_process_erasure_request(self, test_config):
        """Test processing data erasure request"""
        manager = DataSubjectRightsManager(test_config['test_tenant_id'])
        
        # Submit and process erasure request
        request_id = await manager.submit_request(
            user_id="user-123",
            request_type="erasure"
        )
        
        # Process the request
        result = await manager.process_erasure_request(request_id)
        
        assert result is not None
        assert 'request_id' in result
        assert 'user_id' in result
        assert 'anonymized_records' in result
        assert 'completed_at' in result
        
        # Verify request status updated
        request = manager.requests[request_id]
        assert request.status == "completed"

class TestComplianceChecker:
    """Test compliance checking functionality"""
    
    @pytest.mark.asyncio
    async def test_gdpr_compliance_check(self, test_config):
        """Test GDPR compliance checking"""
        checker = ComplianceChecker(test_config['test_tenant_id'])
        
        # Test compliant data processing
        compliant_result = await checker.check_compliance(
            action="store",
            resource_type="personal_data",
            data={
                'id': 'data-123',
                'encrypted': True,
                'consent_obtained': True,
                'cross_border': False
            }
        )
        
        assert compliant_result['compliant'] is True
        assert len(compliant_result['violations']) == 0
        
        # Test non-compliant data processing
        non_compliant_result = await checker.check_compliance(
            action="store",
            resource_type="personal_data",
            data={
                'id': 'data-456',
                'encrypted': False,  # Violation: encryption required
                'consent_obtained': False,  # Violation: consent required
                'cross_border': True  # Violation: cross-border not allowed
            }
        )
        
        assert non_compliant_result['compliant'] is False
        assert len(non_compliant_result['violations']) > 0
    
    @pytest.mark.asyncio
    async def test_data_minimization_check(self, test_config):
        """Test data minimization compliance"""
        checker = ComplianceChecker(test_config['test_tenant_id'])
        
        # Test excessive data collection
        result = await checker.check_compliance(
            action="collect",
            resource_type="user_data",
            data={
                'id': 'collection-123',
                'fields': [f'field_{i}' for i in range(25)]  # Too many fields
            }
        )
        
        # Should generate warning about data minimization
        assert any('minimization' in warning.lower() for warning in result['warnings'])
    
    @pytest.mark.asyncio
    async def test_retention_compliance_check(self, test_config):
        """Test data retention compliance"""
        checker = ComplianceChecker(test_config['test_tenant_id'])
        
        # Test retention period exceeding policy
        result = await checker.check_compliance(
            action="retain",
            resource_type="user_data",
            data={
                'id': 'retention-123',
                'retention_days': 3000  # Exceeds 7-year policy (2555 days)
            }
        )
        
        # Should generate warning about retention period
        assert any('retention' in warning.lower() for warning in result['warnings'])

class TestSecurityEngine:
    """Test main security engine functionality"""
    
    @pytest.mark.asyncio
    async def test_engine_initialization(self, test_config):
        """Test security engine initialization"""
        engine = create_security_engine(test_config['test_tenant_id'])
        
        assert engine.tenant_id == test_config['test_tenant_id']
        assert hasattr(engine, 'encryption_manager')
        assert hasattr(engine, 'audit_logger')
        assert hasattr(engine, 'consent_manager')
        assert hasattr(engine, 'rights_manager')
        assert hasattr(engine, 'compliance_checker')
        assert hasattr(engine, 'validator')
    
    @pytest.mark.asyncio
    async def test_validate_and_secure_data(self, test_config, sample_security_data):
        """Test data validation and security"""
        engine = create_security_engine(test_config['test_tenant_id'])
        
        # Test with valid data
        result = await engine.validate_and_secure_data(
            data=sample_security_data['valid_data'],
            operation="store",
            user_context={'user_id': 'test-user'}
        )
        
        assert result is not None
        assert 'data' in result
        assert 'compliance' in result
        assert 'security_applied' in result
        assert result['compliance']['compliant'] is True
        assert result['security_applied'] is True
    
    @pytest.mark.asyncio
    async def test_validate_malicious_data(self, test_config, sample_security_data):
        """Test handling of malicious data"""
        engine = create_security_engine(test_config['test_tenant_id'])
        
        # Malicious data should be sanitized
        result = await engine.validate_and_secure_data(
            data=sample_security_data['malicious_data'],
            operation="store",
            user_context={'user_id': 'test-user'}
        )
        
        # Data should be sanitized
        secured_data = result['data']
        assert '<script>' not in str(secured_data)
        assert 'DROP TABLE' not in str(secured_data)
    
    @pytest.mark.asyncio
    async def test_authenticate_session(self, test_config):
        """Test session authentication"""
        engine = create_security_engine(test_config['test_tenant_id'])
        
        # Test valid session authentication
        session_token = "valid_session_token_12345678901234567890"
        session_data = await engine.authenticate_session(
            session_token=session_token,
            ip_address="192.168.1.1",
            user_agent="Test Browser"
        )
        
        assert session_data is not None
        assert session_data['authenticated'] is True
        assert session_data['session_id'] == session_token
        assert session_data['ip_address'] == "192.168.1.1"
        
        # Session should be tracked
        assert session_token in engine.active_sessions
    
    @pytest.mark.asyncio
    async def test_suspicious_activity_detection(self, test_config):
        """Test suspicious activity detection"""
        engine = create_security_engine(test_config['test_tenant_id'])
        
        # Simulate multiple failed attempts from same IP
        ip_address = "192.168.1.100"
        
        # Record multiple failed attempts
        for _ in range(6):
            await engine._record_failed_attempt(ip_address)
        
        # Next authentication attempt should be blocked
        with pytest.raises(ValueError, match="Suspicious activity detected"):
            await engine.authenticate_session(
                session_token="short_token",  # Will fail validation
                ip_address=ip_address
            )
    
    @pytest.mark.asyncio
    async def test_compliance_report_generation(self, test_config):
        """Test compliance report generation"""
        engine = create_security_engine(test_config['test_tenant_id'])
        
        # Generate some audit events first
        for i in range(10):
            await engine.audit_logger.log_event(
                event_type=AuditEventType.DATA_ACCESS,
                resource_type="test_resource",
                resource_id=f"resource-{i}",
                action="test_action",
                user_id="test-user"
            )
        
        # Generate compliance report
        report = await engine.generate_compliance_report(
            start_date=datetime.utcnow() - timedelta(hours=1),
            end_date=datetime.utcnow()
        )
        
        assert report is not None
        assert 'report_id' in report
        assert 'tenant_id' in report
        assert 'summary' in report
        assert 'event_breakdown' in report
        assert 'recommendations' in report
        
        # Verify summary contains expected metrics
        summary = report['summary']
        assert 'total_events' in summary
        assert 'compliance_score' in summary
        assert summary['total_events'] == 10
    
    @pytest.mark.asyncio
    async def test_security_status(self, test_config):
        """Test security status reporting"""
        engine = create_security_engine(test_config['test_tenant_id'])
        
        # Create some session and audit data
        await engine.authenticate_session(
            session_token="test_session_12345678901234567890",
            ip_address="192.168.1.1"
        )
        
        await engine.consent_manager.record_consent(
            user_id="test-user",
            purpose="test_purpose",
            granted=True
        )
        
        # Get security status
        status = await engine.get_security_status()
        
        assert status is not None
        assert 'tenant_id' in status
        assert 'threat_level' in status
        assert 'active_sessions' in status
        assert 'security_policies' in status
        assert 'status' in status
        
        assert status['tenant_id'] == test_config['test_tenant_id']
        assert status['active_sessions'] >= 1
        assert status['consent_records'] >= 1
        assert status['status'] == 'operational'

class TestSecurityEngineIntegration:
    """Integration tests for security engine"""
    
    @pytest.mark.asyncio
    async def test_notification_service_integration(
        self,
        notification_service,
        sample_notification_request,
        sample_user_profile,
        sample_notification_template
    ):
        """Test security integration with notification service"""
        
        # Setup security engine in notification service
        security_engine = create_security_engine(TEST_CONFIG['test_tenant_id'])
        notification_service.security_engine = security_engine
        
        # Setup test data
        notification_service.user_profiles[sample_user_profile.user_id] = sample_user_profile
        notification_service.templates[sample_notification_template.id] = sample_notification_template
        
        # Send notification (should trigger security validation)
        result = await notification_service.send_notification(sample_notification_request)
        
        # Verify security validation was performed
        assert result is not None
        
        # Check that audit events were logged
        audit_events = security_engine.audit_logger.audit_events
        assert len(audit_events) > 0
    
    @pytest.mark.asyncio
    async def test_gdpr_compliance_workflow(self, test_config):
        """Test complete GDPR compliance workflow"""
        engine = create_security_engine(test_config['test_tenant_id'])
        
        user_id = "gdpr-test-user"
        
        # 1. Record consent
        consent_id = await engine.consent_manager.record_consent(
            user_id=user_id,
            purpose="marketing_communications",
            granted=True,
            data_categories=["email", "preferences"]
        )
        
        # 2. Process data with consent
        validation_result = await engine.validate_and_secure_data(
            data={
                'user_id': user_id,
                'email': 'gdpr@test.com',
                'consent_obtained': True
            },
            operation="store"
        )
        assert validation_result['compliance']['compliant'] is True
        
        # 3. Submit data access request
        access_request_id = await engine.rights_manager.submit_request(
            user_id=user_id,
            request_type="access"
        )
        
        # 4. Process access request
        access_result = await engine.rights_manager.process_access_request(access_request_id)
        assert access_result['user_id'] == user_id
        
        # 5. Submit erasure request
        erasure_request_id = await engine.rights_manager.submit_request(
            user_id=user_id,
            request_type="erasure"
        )
        
        # 6. Process erasure request
        erasure_result = await engine.rights_manager.process_erasure_request(erasure_request_id)
        assert erasure_result['user_id'] == user_id
        
        # 7. Generate compliance report
        report = await engine.generate_compliance_report(
            start_date=datetime.utcnow() - timedelta(hours=1),
            end_date=datetime.utcnow()
        )
        
        # Verify complete workflow was audited
        assert len(engine.audit_logger.audit_events) > 0
        assert report['summary']['total_events'] > 0

class TestSecurityEnginePerformance:
    """Performance tests for security engine"""
    
    @pytest.mark.asyncio
    async def test_high_volume_audit_logging(self, test_config):
        """Test high-volume audit logging performance"""
        engine = create_security_engine(test_config['test_tenant_id'])
        
        # Log high volume of audit events
        event_count = 1000
        
        with TestTimer() as timer:
            tasks = []
            for i in range(event_count):
                task = engine.audit_logger.log_event(
                    event_type=AuditEventType.DATA_ACCESS,
                    resource_type="test_resource",
                    resource_id=f"resource-{i}",
                    action="performance_test",
                    user_id=f"user-{i % 100}"
                )
                tasks.append(task)
            
            await asyncio.gather(*tasks)
        
        # Performance assertions
        assert timer.elapsed < 5.0  # Should complete within 5 seconds
        assert len(engine.audit_logger.audit_events) == event_count
        
        throughput = event_count / timer.elapsed
        assert throughput > 200  # At least 200 events per second
        
        print(f"Audit logging performance: {event_count} events in {timer.elapsed:.2f} seconds")
        print(f"Throughput: {throughput:.2f} events/second")
    
    @pytest.mark.asyncio
    async def test_concurrent_encryption_operations(self, test_config):
        """Test concurrent encryption/decryption performance"""
        engine = create_security_engine(test_config['test_tenant_id'])
        
        # Prepare test data
        test_data = ["Sensitive data item " + str(i) for i in range(100)]
        
        # Concurrent encryption
        with TestTimer() as encrypt_timer:
            encrypt_tasks = [
                asyncio.create_task(asyncio.to_thread(
                    engine.encryption_manager.encrypt_data, data, f"context-{i}"
                )) for i, data in enumerate(test_data)
            ]
            encrypted_results = await asyncio.gather(*encrypt_tasks)
        
        # Concurrent decryption
        with TestTimer() as decrypt_timer:
            decrypt_tasks = [
                asyncio.create_task(asyncio.to_thread(
                    engine.encryption_manager.decrypt_data, encrypted
                )) for encrypted in encrypted_results
            ]
            decrypted_results = await asyncio.gather(*decrypt_tasks)
        
        # Verify results
        assert len(encrypted_results) == len(test_data)
        assert len(decrypted_results) == len(test_data)
        assert decrypted_results == test_data
        
        # Performance assertions
        assert encrypt_timer.elapsed < 5.0
        assert decrypt_timer.elapsed < 5.0
        
        print(f"Encryption performance: {len(test_data)} items in {encrypt_timer.elapsed:.2f} seconds")
        print(f"Decryption performance: {len(test_data)} items in {decrypt_timer.elapsed:.2f} seconds")

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])