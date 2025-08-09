#!/usr/bin/env python3
"""
APG Key Management - Model Tests
Comprehensive test suite for Pydantic models and validation

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any
from uuid_extensions import uuid7str

from ..models import (
	KeyAlgorithm, KeyUsage, KeyState, SecurityLevel, ComplianceFramework,
	KeyMetadata, KeyPolicy, KeySpec, Key, KeyOperation, SecurityThreat, 
	AuditEvent, HSMConfiguration, CloudKeyStore, KeyUsageStats,
	create_key_spec_async, validate_key_size, validate_tenant_id
)


class TestKeyAlgorithm:
	"""Test KeyAlgorithm enum"""
	
	def test_symmetric_algorithms(self):
		"""Test symmetric algorithm enumeration"""
		assert KeyAlgorithm.AES_128 == "AES-128"
		assert KeyAlgorithm.AES_256 == "AES-256"
		assert KeyAlgorithm.CHACHA20_POLY1305 == "ChaCha20-Poly1305"
	
	def test_asymmetric_algorithms(self):
		"""Test asymmetric algorithm enumeration"""
		assert KeyAlgorithm.RSA_2048 == "RSA-2048"
		assert KeyAlgorithm.RSA_4096 == "RSA-4096"
		assert KeyAlgorithm.ECDSA_P256 == "ECDSA-P256"
		assert KeyAlgorithm.ECDSA_P384 == "ECDSA-P384"
		assert KeyAlgorithm.ED25519 == "Ed25519"
	
	def test_post_quantum_algorithms(self):
		"""Test post-quantum algorithm enumeration"""
		assert KeyAlgorithm.KYBER_512 == "Kyber-512"
		assert KeyAlgorithm.KYBER_768 == "Kyber-768"
		assert KeyAlgorithm.KYBER_1024 == "Kyber-1024"
		assert KeyAlgorithm.DILITHIUM_2 == "Dilithium-2"
		assert KeyAlgorithm.DILITHIUM_3 == "Dilithium-3"
		assert KeyAlgorithm.DILITHIUM_5 == "Dilithium-5"
		assert KeyAlgorithm.FALCON_512 == "Falcon-512"
		assert KeyAlgorithm.FALCON_1024 == "Falcon-1024"


class TestKeyUsage:
	"""Test KeyUsage enum"""
	
	def test_usage_values(self):
		"""Test key usage enumeration"""
		assert KeyUsage.ENCRYPT == "encrypt"
		assert KeyUsage.DECRYPT == "decrypt"
		assert KeyUsage.SIGN == "sign"
		assert KeyUsage.VERIFY == "verify"
		assert KeyUsage.KEY_WRAP == "key_wrap"
		assert KeyUsage.KEY_UNWRAP == "key_unwrap"
		assert KeyUsage.DERIVE == "derive"
		assert KeyUsage.MAC == "mac"


class TestValidationFunctions:
	"""Test validation functions"""
	
	def test_validate_key_size_valid(self):
		"""Test valid key sizes"""
		assert validate_key_size(KeyAlgorithm.AES_128, 128) == 128
		assert validate_key_size(KeyAlgorithm.AES_256, 256) == 256
		assert validate_key_size(KeyAlgorithm.RSA_2048, 2048) == 2048
		assert validate_key_size(KeyAlgorithm.RSA_4096, 4096) == 4096
		assert validate_key_size(KeyAlgorithm.ECDSA_P256, 256) == 256
		assert validate_key_size(KeyAlgorithm.ECDSA_P384, 384) == 384
	
	def test_validate_key_size_invalid(self):
		"""Test invalid key sizes"""
		with pytest.raises(ValueError):
			validate_key_size(KeyAlgorithm.AES_128, 256)
		
		with pytest.raises(ValueError):
			validate_key_size(KeyAlgorithm.RSA_2048, 1024)
		
		with pytest.raises(ValueError):
			validate_key_size(KeyAlgorithm.ECDSA_P256, 384)
	
	def test_validate_tenant_id_valid(self):
		"""Test valid tenant IDs"""
		assert validate_tenant_id("tenant_123") == "tenant_123"
		assert validate_tenant_id("prod-system") == "prod-system"
		assert validate_tenant_id("test123") == "test123"
	
	def test_validate_tenant_id_invalid(self):
		"""Test invalid tenant IDs"""
		with pytest.raises(ValueError):
			validate_tenant_id("")  # Empty
		
		with pytest.raises(ValueError):
			validate_tenant_id("ab")  # Too short
		
		with pytest.raises(ValueError):
			validate_tenant_id("tenant@123")  # Invalid characters


class TestKeyMetadata:
	"""Test KeyMetadata model"""
	
	def test_key_metadata_valid(self):
		"""Test valid key metadata"""
		metadata = KeyMetadata(
			name="Test Key",
			description="A test key for unit testing",
			tags={"environment": "test", "project": "keym"},
			cost_center="engineering",
			project_id="keym-project",
			environment="test",
			owner="test@datacraft.co.ke"
		)
		
		assert metadata.name == "Test Key"
		assert metadata.description == "A test key for unit testing"
		assert metadata.tags["environment"] == "test"
		assert metadata.cost_center == "engineering"
		assert metadata.owner == "test@datacraft.co.ke"
	
	def test_key_metadata_required_only(self):
		"""Test key metadata with only required fields"""
		metadata = KeyMetadata(name="Minimal Key")
		
		assert metadata.name == "Minimal Key"
		assert metadata.description is None
		assert metadata.tags == {}
		assert metadata.cost_center is None
	
	def test_key_metadata_name_validation(self):
		"""Test key metadata name validation"""
		# Test maximum length
		long_name = "x" * 256
		with pytest.raises(ValueError):
			KeyMetadata(name=long_name)


class TestKeyPolicy:
	"""Test KeyPolicy model"""
	
	def test_key_policy_defaults(self):
		"""Test key policy with default values"""
		policy = KeyPolicy()
		
		assert policy.allowed_users == []
		assert policy.allowed_roles == []
		assert policy.allowed_applications == []
		assert policy.usage_restrictions == []
		assert policy.ip_whitelist == []
		assert policy.time_restrictions == {}
		assert policy.geographic_restrictions == []
		assert policy.auto_rotate is True
		assert policy.rotation_interval_days == 90
		assert policy.max_usage_count is None
		assert policy.expiry_date is None
		assert policy.compliance_frameworks == []
		assert policy.require_mfa is True
		assert policy.require_approval is False
		assert policy.min_security_level == SecurityLevel.INTERNAL
		assert policy.require_hsm is False
		assert policy.allow_export is False
	
	def test_key_policy_validation(self):
		"""Test key policy validation"""
		# Test rotation interval bounds
		policy = KeyPolicy(rotation_interval_days=1)
		assert policy.rotation_interval_days == 1
		
		policy = KeyPolicy(rotation_interval_days=3650)
		assert policy.rotation_interval_days == 3650
		
		# Test invalid rotation intervals
		with pytest.raises(ValueError):
			KeyPolicy(rotation_interval_days=0)
		
		with pytest.raises(ValueError):
			KeyPolicy(rotation_interval_days=3651)
	
	def test_key_policy_complex(self):
		"""Test complex key policy configuration"""
		expiry = datetime.utcnow() + timedelta(days=365)
		
		policy = KeyPolicy(
			allowed_users=["admin@company.com", "user@company.com"],
			allowed_roles=["admin", "key_user"],
			allowed_applications=["api-gateway", "database"],
			usage_restrictions=[KeyUsage.ENCRYPT, KeyUsage.DECRYPT],
			ip_whitelist=["10.0.0.0/8", "192.168.1.0/24"],
			time_restrictions={"allowed_hours": [9, 10, 11, 14, 15, 16]},
			geographic_restrictions=["US", "UK"],
			auto_rotate=True,
			rotation_interval_days=60,
			max_usage_count=100000,
			expiry_date=expiry,
			compliance_frameworks=[ComplianceFramework.GDPR, ComplianceFramework.HIPAA],
			require_mfa=True,
			require_approval=True,
			min_security_level=SecurityLevel.CONFIDENTIAL,
			require_hsm=True,
			allow_export=False
		)
		
		assert len(policy.allowed_users) == 2
		assert len(policy.allowed_roles) == 2
		assert len(policy.usage_restrictions) == 2
		assert policy.rotation_interval_days == 60
		assert policy.max_usage_count == 100000
		assert policy.expiry_date == expiry
		assert ComplianceFramework.GDPR in policy.compliance_frameworks
		assert policy.min_security_level == SecurityLevel.CONFIDENTIAL


class TestKeySpec:
	"""Test KeySpec model"""
	
	def test_key_spec_valid(self):
		"""Test valid key specification"""
		metadata = KeyMetadata(name="Test Key")
		policy = KeyPolicy()
		
		spec = KeySpec(
			tenant_id="test_tenant",
			algorithm=KeyAlgorithm.AES_256,
			key_size=256,
			usage=[KeyUsage.ENCRYPT, KeyUsage.DECRYPT],
			metadata=metadata,
			policy=policy,
			created_by="test@datacraft.co.ke"
		)
		
		assert spec.tenant_id == "test_tenant"
		assert spec.algorithm == KeyAlgorithm.AES_256
		assert spec.key_size == 256
		assert len(spec.usage) == 2
		assert KeyUsage.ENCRYPT in spec.usage
		assert spec.metadata.name == "Test Key"
		assert spec.state == KeyState.PENDING
		assert spec.security_level == SecurityLevel.INTERNAL
		assert spec.created_by == "test@datacraft.co.ke"
	
	def test_key_spec_key_size_validation(self):
		"""Test key size validation against algorithm"""
		metadata = KeyMetadata(name="Test Key")
		policy = KeyPolicy()
		
		# Valid combination
		spec = KeySpec(
			tenant_id="test_tenant",
			algorithm=KeyAlgorithm.AES_256,
			key_size=256,
			usage=[KeyUsage.ENCRYPT],
			metadata=metadata,
			policy=policy,
			created_by="test@datacraft.co.ke"
		)
		assert spec.key_size == 256
		
		# Invalid combination
		with pytest.raises(ValueError):
			KeySpec(
				tenant_id="test_tenant", 
				algorithm=KeyAlgorithm.AES_128,
				key_size=256,  # Wrong size for AES-128
				usage=[KeyUsage.ENCRYPT],
				metadata=metadata,
				policy=policy,
				created_by="test@datacraft.co.ke"
			)
	
	def test_key_spec_tenant_validation(self):
		"""Test tenant ID validation"""
		metadata = KeyMetadata(name="Test Key")
		policy = KeyPolicy()
		
		# Valid tenant ID
		spec = KeySpec(
			tenant_id="valid_tenant",
			algorithm=KeyAlgorithm.AES_256,
			key_size=256,
			usage=[KeyUsage.ENCRYPT],
			metadata=metadata,
			policy=policy,
			created_by="test@datacraft.co.ke"
		)
		assert spec.tenant_id == "valid_tenant"
		
		# Invalid tenant ID
		with pytest.raises(ValueError):
			KeySpec(
				tenant_id="ab",  # Too short
				algorithm=KeyAlgorithm.AES_256,
				key_size=256,
				usage=[KeyUsage.ENCRYPT],
				metadata=metadata,
				policy=policy,
				created_by="test@datacraft.co.ke"
			)


class TestKey:
	"""Test Key model"""
	
	def test_key_valid(self):
		"""Test valid key model"""
		metadata = KeyMetadata(name="Test Key")
		policy = KeyPolicy()
		
		spec = KeySpec(
			tenant_id="test_tenant",
			algorithm=KeyAlgorithm.AES_256,
			key_size=256,
			usage=[KeyUsage.ENCRYPT, KeyUsage.DECRYPT],
			metadata=metadata,
			policy=policy,
			created_by="test@datacraft.co.ke"
		)
		
		key_material = b"test_key_material_32_bytes_long"
		checksum = "abcd1234"
		
		key = Key(
			spec=spec,
			key_material=key_material,
			key_checksum=checksum,
			usage_count=0
		)
		
		assert key.spec.tenant_id == "test_tenant"
		assert key.key_material == key_material
		assert key.key_checksum == checksum
		assert key.usage_count == 0
		assert key.last_used is None
		assert key.previous_versions == []
		assert key.next_rotation is None
		assert key.backup_status == "pending"
	
	def test_key_with_hsm(self):
		"""Test key with HSM configuration"""
		metadata = KeyMetadata(name="HSM Test Key")
		policy = KeyPolicy(require_hsm=True)
		
		spec = KeySpec(
			tenant_id="test_tenant",
			algorithm=KeyAlgorithm.RSA_2048,
			key_size=2048,
			usage=[KeyUsage.SIGN, KeyUsage.VERIFY],
			metadata=metadata,
			policy=policy,
			created_by="test@datacraft.co.ke"
		)
		
		key = Key(
			spec=spec,
			hsm_key_id="hsm_key_123",
			hsm_session_id="session_456"
		)
		
		assert key.hsm_key_id == "hsm_key_123"
		assert key.hsm_session_id == "session_456"
		assert key.spec.policy.require_hsm is True


class TestKeyOperation:
	"""Test KeyOperation model"""
	
	def test_key_operation_valid(self):
		"""Test valid key operation"""
		test_data = b"test_data_to_encrypt"
		
		operation = KeyOperation(
			key_id="test_key_123",
			operation_type="encrypt",
			data=test_data,
			parameters={"algorithm": "AES-GCM"},
			user_id="test@datacraft.co.ke",
			application_id="test-app",
			request_ip="192.168.1.100",
			session_id="session_123"
		)
		
		assert operation.key_id == "test_key_123"
		assert operation.operation_type == "encrypt"
		assert operation.data == test_data
		assert operation.parameters["algorithm"] == "AES-GCM"
		assert operation.user_id == "test@datacraft.co.ke"
		assert operation.success is False  # Default
		assert operation.result_data is None
		assert operation.error_message is None
	
	def test_key_operation_completed(self):
		"""Test completed key operation"""
		operation = KeyOperation(
			key_id="test_key_123",
			operation_type="decrypt",
			user_id="test@datacraft.co.ke"
		)
		
		# Simulate completion
		operation.completed_at = datetime.utcnow()
		operation.success = True
		operation.result_data = b"decrypted_result"
		
		assert operation.success is True
		assert operation.result_data == b"decrypted_result"
		assert operation.completed_at is not None
		assert operation.error_message is None


class TestSecurityThreat:
	"""Test SecurityThreat model"""
	
	def test_security_threat_valid(self):
		"""Test valid security threat"""
		threat = SecurityThreat(
			tenant_id="test_tenant",
			threat_type="brute_force_attack",
			severity="high",
			confidence=0.89,
			affected_keys=["key_001", "key_002"],
			source_ip="192.168.1.100",
			user_id="suspicious@example.com",
			detection_method="ml_anomaly_detection",
			indicators={
				"failed_attempts": 50,
				"time_window_minutes": 5,
				"geographic_anomaly": True
			}
		)
		
		assert threat.tenant_id == "test_tenant"
		assert threat.threat_type == "brute_force_attack"
		assert threat.severity == "high"
		assert threat.confidence == 0.89
		assert len(threat.affected_keys) == 2
		assert threat.source_ip == "192.168.1.100"
		assert threat.detection_method == "ml_anomaly_detection"
		assert threat.status == "new"  # Default
		assert threat.indicators["failed_attempts"] == 50
	
	def test_security_threat_confidence_validation(self):
		"""Test security threat confidence validation"""
		# Valid confidence values
		threat = SecurityThreat(
			tenant_id="test_tenant",
			threat_type="test_threat",
			severity="medium",
			confidence=0.75,
			detection_method="test"
		)
		assert threat.confidence == 0.75
		
		# Invalid confidence values
		with pytest.raises(ValueError):
			SecurityThreat(
				tenant_id="test_tenant",
				threat_type="test_threat", 
				severity="medium",
				confidence=1.5,  # > 1.0
				detection_method="test"
			)
		
		with pytest.raises(ValueError):
			SecurityThreat(
				tenant_id="test_tenant",
				threat_type="test_threat",
				severity="medium", 
				confidence=-0.1,  # < 0.0
				detection_method="test"
			)


class TestAuditEvent:
	"""Test AuditEvent model"""
	
	def test_audit_event_valid(self):
		"""Test valid audit event"""
		event = AuditEvent(
			tenant_id="test_tenant",
			event_type="key_created",
			resource_type="key",
			resource_id="key_123",
			user_id="admin@company.com",
			application_id="key-manager",
			session_id="session_456",
			source_ip="10.0.1.100",
			user_agent="KeyManager/1.0",
			request_id="req_789",
			action="create_key",
			outcome="success",
			details={
				"algorithm": "AES-256",
				"key_size": 256,
				"usage": ["encrypt", "decrypt"]
			}
		)
		
		assert event.tenant_id == "test_tenant"
		assert event.event_type == "key_created"
		assert event.resource_type == "key"
		assert event.resource_id == "key_123"
		assert event.user_id == "admin@company.com"
		assert event.action == "create_key"
		assert event.outcome == "success"
		assert event.details["algorithm"] == "AES-256"
		assert event.retention_period_days == 2555  # Default ~7 years


class TestKeyUsageStats:
	"""Test KeyUsageStats model"""
	
	def test_key_usage_stats_valid(self):
		"""Test valid key usage statistics"""
		stats = KeyUsageStats(
			key_id="key_123",
			tenant_id="test_tenant",
			total_operations=1000,
			encrypt_operations=600,
			decrypt_operations=400,
			sign_operations=0,
			verify_operations=0,
			daily_operations={"2025-01-01": 100, "2025-01-02": 150},
			monthly_operations={"2025-01": 1000},
			average_latency_ms=25.5,
			success_rate=0.998,
			unique_users=5,
			unique_applications=2,
			first_used=datetime.utcnow() - timedelta(days=30),
			last_used=datetime.utcnow() - timedelta(minutes=5)
		)
		
		assert stats.key_id == "key_123"
		assert stats.tenant_id == "test_tenant"
		assert stats.total_operations == 1000
		assert stats.encrypt_operations == 600
		assert stats.decrypt_operations == 400
		assert stats.average_latency_ms == 25.5
		assert stats.success_rate == 0.998
		assert stats.unique_users == 5
		assert stats.daily_operations["2025-01-01"] == 100
	
	def test_key_usage_stats_validation(self):
		"""Test key usage statistics validation"""
		# Valid success rate
		stats = KeyUsageStats(
			key_id="key_123",
			tenant_id="test_tenant",
			success_rate=0.95
		)
		assert stats.success_rate == 0.95
		
		# Invalid success rate
		with pytest.raises(ValueError):
			KeyUsageStats(
				key_id="key_123",
				tenant_id="test_tenant",
				success_rate=1.5  # > 1.0
			)


class TestAsyncFactories:
	"""Test async factory functions"""
	
	@pytest.mark.asyncio
	async def test_create_key_spec_async(self):
		"""Test async key specification creation"""
		spec = await create_key_spec_async(
			tenant_id="test_tenant",
			algorithm=KeyAlgorithm.AES_256,
			usage=[KeyUsage.ENCRYPT, KeyUsage.DECRYPT],
			name="Test Async Key",
			created_by="test@datacraft.co.ke",
			description="A key created asynchronously",
			tags={"test": "true"},
			auto_rotate=False,
			rotation_interval_days=180
		)
		
		assert spec.tenant_id == "test_tenant"
		assert spec.algorithm == KeyAlgorithm.AES_256
		assert spec.key_size == 256  # Default for AES-256
		assert len(spec.usage) == 2
		assert spec.metadata.name == "Test Async Key"
		assert spec.metadata.description == "A key created asynchronously"
		assert spec.metadata.tags["test"] == "true"
		assert spec.policy.auto_rotate is False
		assert spec.policy.rotation_interval_days == 180
		assert spec.created_by == "test@datacraft.co.ke"
	
	@pytest.mark.asyncio
	async def test_create_key_spec_async_defaults(self):
		"""Test async key spec creation with defaults"""
		spec = await create_key_spec_async(
			tenant_id="test_tenant",
			algorithm=KeyAlgorithm.RSA_2048,
			usage=[KeyUsage.SIGN],
			name="Minimal Key",
			created_by="test@datacraft.co.ke"
		)
		
		assert spec.algorithm == KeyAlgorithm.RSA_2048
		assert spec.key_size == 2048  # Default for RSA-2048
		assert spec.metadata.name == "Minimal Key"
		assert spec.metadata.description is None
		assert spec.policy.auto_rotate is True  # Default
		assert spec.policy.rotation_interval_days == 90  # Default


if __name__ == "__main__":
	pytest.main([__file__])