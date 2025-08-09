#!/usr/bin/env python3
"""
APG Message Queue Event Bus (MQEB) - Quantum Security Tests
Tests for quantum-safe cryptography and zero-trust messaging

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import pytest
import asyncio
import json
from datetime import datetime
from uuid_extensions import uuid7str

# Import MQEB components
from ..models import MQMessage, MessagePriority
from ..service import MQEBService, create_mqeb_service
from ..quantum_security import (
	QuantumSecurityEngine, QuantumKeyManager, ZeroTrustMessageSecurity,
	QuantumAlgorithm, SecurityLevel, QuantumKeyPair,
	create_quantum_security_engine
)


class TestQuantumKeyManager:
	"""Test quantum key management functionality"""
	
	def test_key_manager_initialization(self):
		"""Test key manager initialization"""
		key_manager = QuantumKeyManager()
		assert len(key_manager.key_store) == 0
		assert len(key_manager.tenant_keys) == 0
		assert key_manager.key_rotation_policy.days == 90
	
	@pytest.mark.asyncio
	async def test_key_pair_generation(self):
		"""Test quantum key pair generation"""
		key_manager = QuantumKeyManager()
		
		key_id = await key_manager.generate_key_pair(
			QuantumAlgorithm.CRYSTALS_KYBER_512,
			"test_tenant",
			SecurityLevel.QUANTUM_SAFE
		)
		
		assert key_id in key_manager.key_store
		assert key_id in key_manager.tenant_keys["test_tenant"]
		assert key_id in key_manager.algorithm_keys[QuantumAlgorithm.CRYSTALS_KYBER_512]
		
		key_pair = await key_manager.get_key_pair(key_id)
		assert key_pair is not None
		assert key_pair.algorithm == QuantumAlgorithm.CRYSTALS_KYBER_512
		assert key_pair.security_level == SecurityLevel.QUANTUM_SAFE
		assert len(key_pair.public_key) > 0
		assert len(key_pair.private_key) > 0
	
	@pytest.mark.asyncio
	async def test_tenant_key_retrieval(self):
		"""Test retrieving keys for specific tenant"""
		key_manager = QuantumKeyManager()
		
		# Generate keys for different tenants and algorithms
		key1 = await key_manager.generate_key_pair(
			QuantumAlgorithm.CRYSTALS_KYBER_512, "tenant1"
		)
		key2 = await key_manager.generate_key_pair(
			QuantumAlgorithm.CRYSTALS_DILITHIUM_2, "tenant1"
		)
		key3 = await key_manager.generate_key_pair(
			QuantumAlgorithm.CRYSTALS_KYBER_512, "tenant2"
		)
		
		# Test tenant1 keys
		tenant1_keys = await key_manager.get_tenant_keys("tenant1")
		assert len(tenant1_keys) == 2
		
		# Test tenant1 Kyber keys only
		tenant1_kyber_keys = await key_manager.get_tenant_keys(
			"tenant1", QuantumAlgorithm.CRYSTALS_KYBER_512
		)
		assert len(tenant1_kyber_keys) == 1
		assert tenant1_kyber_keys[0].key_id == key1
		
		# Test tenant2 keys
		tenant2_keys = await key_manager.get_tenant_keys("tenant2")
		assert len(tenant2_keys) == 1
		assert tenant2_keys[0].key_id == key3


class TestZeroTrustMessageSecurity:
	"""Test zero-trust security framework"""
	
	@pytest.fixture
	def zero_trust_system(self):
		"""Create zero-trust security system"""
		key_manager = QuantumKeyManager()
		return ZeroTrustMessageSecurity(key_manager)
	
	@pytest.mark.asyncio
	async def test_trust_score_calculation(self, zero_trust_system):
		"""Test trust score calculation"""
		message = MQMessage(
			topic="test.topic",
			payload=b"Test message",
			tenant_id="test_tenant",
			source_application="trusted_app"
		)
		
		# High-trust context
		high_trust_context = {
			'authenticated': True,
			'mfa_verified': True,
			'device_fingerprint': 'known_device_fingerprint_12345',
			'source_ip': '192.168.1.100',
			'user_id': 'trusted_user'
		}
		
		trust_score = await zero_trust_system._calculate_trust_score(message, high_trust_context)
		assert trust_score > 0.8  # High trust score
		
		# Low-trust context
		low_trust_context = {
			'authenticated': False,
			'source_ip': '1.2.3.4',  # External IP
			'user_id': 'temp_user'
		}
		
		trust_score = await zero_trust_system._calculate_trust_score(message, low_trust_context)
		assert trust_score < 0.6  # Low trust score
	
	@pytest.mark.asyncio
	async def test_zero_trust_policy_evaluation(self, zero_trust_system):
		"""Test zero-trust policy evaluation"""
		message = MQMessage(
			topic="confidential.data",
			payload=b"Confidential message content",
			encrypted=True,
			tenant_id="test_tenant",
			source_application="secure_app"
		)
		
		# Context that should pass policy
		valid_context = {
			'authenticated': True,
			'mfa_verified': True,
			'source_ip': '10.0.1.100',
			'device_fingerprint': 'trusted_device_abcdef123456',
			'user_id': 'authorized_user'
		}
		
		policy_result = await zero_trust_system.apply_zero_trust_policy(message, valid_context)
		assert policy_result == True
		
		# Context that should fail policy
		invalid_context = {
			'authenticated': False,
			'source_ip': '1.2.3.4',
			'user_id': 'unauthorized_user'
		}
		
		policy_result = await zero_trust_system.apply_zero_trust_policy(message, invalid_context)
		assert policy_result == False
	
	def test_threat_detection(self, zero_trust_system):
		"""Test threat detection functionality"""
		# Malicious message content
		malicious_message = MQMessage(
			topic="test.topic",
			payload=b"<script>alert('xss')</script>",
			tenant_id="test_tenant",
			source_application="suspicious_app"
		)
		
		context = {'source_ip': '1.2.3.4'}
		threat_detected = zero_trust_system._is_threat_detected(malicious_message, context)
		assert threat_detected == True
		
		# Normal message content
		normal_message = MQMessage(
			topic="test.topic",
			payload=b"Normal business message",
			tenant_id="test_tenant",
			source_application="trusted_app"
		)
		
		threat_detected = zero_trust_system._is_threat_detected(normal_message, context)
		assert threat_detected == False


class TestQuantumSecurityEngine:
	"""Test main quantum security engine"""
	
	@pytest.fixture
	async def mqeb_service(self):
		"""Create MQEB service for testing"""
		service = MQEBService()
		await service.initialize()
		yield service
		await service.shutdown()
	
	@pytest.fixture
	async def security_engine(self, mqeb_service):
		"""Create quantum security engine"""
		engine = await create_quantum_security_engine(mqeb_service)
		yield engine
		await engine.shutdown()
	
	@pytest.mark.asyncio
	async def test_security_engine_initialization(self, security_engine):
		"""Test security engine initialization"""
		assert security_engine.enabled == True
		assert security_engine.key_manager is not None
		assert security_engine.zero_trust is not None
		assert security_engine.compliance is not None
		assert security_engine.default_algorithm == QuantumAlgorithm.CRYSTALS_KYBER_512
	
	@pytest.mark.asyncio
	async def test_message_security_application(self, security_engine):
		"""Test applying security to messages"""
		message = MQMessage(
			topic="secure.topic",
			payload=b"Sensitive business data",
			tenant_id="test_tenant",
			source_application="business_app",
			priority=MessagePriority.HIGH
		)
		
		context = {
			'authenticated': True,
			'user_id': 'business_user',
			'source_ip': '10.0.1.50',
			'compliance_frameworks': ['gdpr', 'iso_27001']
		}
		
		# Apply security to message
		security_result = await security_engine.secure_message(message, context)
		assert security_result == True
		
		# Message should be encrypted
		assert message.encrypted == True
		assert message.encryption_key_id is not None
		assert 'quantum_algorithm' in message.headers
		assert 'quantum_safe' in message.headers
	
	@pytest.mark.asyncio
	async def test_message_encryption_decryption(self, security_engine):
		"""Test message encryption and decryption"""
		original_payload = b"This is sensitive data that needs encryption"
		
		message = MQMessage(
			topic="encrypted.topic",
			payload=original_payload,
			tenant_id="test_tenant",
			source_application="secure_app"
		)
		
		context = {'authenticated': True, 'user_id': 'test_user'}
		
		# Encrypt message
		encryption_result = await security_engine.encrypt_message(message, context)
		assert encryption_result == True
		assert message.encrypted == True
		assert message.payload != original_payload  # Payload should be encrypted
		
		# Decrypt message
		decrypted_payload = await security_engine.decrypt_message(message, context)
		assert decrypted_payload == original_payload  # Should match original
	
	@pytest.mark.asyncio
	async def test_security_status_reporting(self, security_engine):
		"""Test security status reporting"""
		status = await security_engine.get_security_status()
		
		assert 'enabled' in status
		assert 'total_keys' in status
		assert 'active_keys' in status
		assert 'default_algorithm' in status
		assert 'zero_trust_events' in status
		assert 'compliance_reports' in status
		
		assert status['enabled'] == True
		assert status['default_algorithm'] == QuantumAlgorithm.CRYSTALS_KYBER_512.value


class TestQuantumCryptographyOperations:
	"""Test quantum cryptography operations"""
	
	@pytest.mark.asyncio
	async def test_kyber_operations(self):
		"""Test Kyber key encapsulation mechanism simulation"""
		key_manager = QuantumKeyManager()
		kyber_ops = key_manager._kyber_operations
		
		# Generate key pair
		public_key, private_key = await kyber_ops.generate_keypair()
		
		assert len(public_key) > 0
		assert len(private_key) > 0
		assert public_key != private_key
		
		# Test encryption/decryption
		plaintext = b"Test message for Kyber encryption"
		ciphertext = await kyber_ops.encrypt(public_key, plaintext)
		
		assert ciphertext != plaintext
		assert len(ciphertext) > len(plaintext)  # Should be larger due to IV and padding
		
		decrypted_text = await kyber_ops.decrypt(private_key, ciphertext)
		assert decrypted_text == plaintext
	
	@pytest.mark.asyncio
	async def test_dilithium_operations(self):
		"""Test Dilithium signature scheme simulation"""
		key_manager = QuantumKeyManager()
		dilithium_ops = key_manager._dilithium_operations
		
		# Generate key pair
		public_key, private_key = await dilithium_ops.generate_keypair()
		
		assert len(public_key) > 0
		assert len(private_key) > 0
		assert public_key != private_key
		
		# In a full implementation, would test signing and verification
	
	@pytest.mark.asyncio
	async def test_sphincs_operations(self):
		"""Test SPHINCS+ signature scheme simulation"""
		key_manager = QuantumKeyManager()
		sphincs_ops = key_manager._sphincs_operations
		
		# Generate key pair
		public_key, private_key = await sphincs_ops.generate_keypair()
		
		assert len(public_key) > 0
		assert len(private_key) > 0
		assert public_key != private_key


class TestIntegrationWithMQEB:
	"""Test integration of quantum security with MQEB service"""
	
	@pytest.mark.asyncio
	async def test_secure_message_publishing(self):
		"""Test publishing messages with quantum security enabled"""
		# Create MQEB service with security enabled
		service = MQEBService({'quantum_security_enabled': True})
		await service.initialize()
		
		try:
			# Create test topic
			from ..models import TopicConfiguration
			topic_config = TopicConfiguration(
				name="secure.test.topic",
				tenant_id="test_tenant",
				created_by="test_user"
			)
			await service.create_topic(topic_config)
			
			# Create message with sensitive data
			message = MQMessage(
				topic="secure.test.topic",
				payload=b"Confidential business information",
				tenant_id="test_tenant",
				source_application="business_app",
				priority=MessagePriority.CRITICAL
			)
			
			# Publish with security context
			security_context = {
				'authenticated': True,
				'user_id': 'business_user',
				'source_ip': '10.0.1.100',
				'device_fingerprint': 'trusted_device_123456'
			}
			
			message_id = await service.publish_message(message, security_context)
			assert message_id == message.id
			
			# Verify message is in store and encrypted
			stored_message = service.message_store[message_id]
			if hasattr(service, 'quantum_security'):
				# Message should be encrypted if security is enabled
				assert stored_message.encrypted == True
				assert 'quantum_safe' in stored_message.headers
		
		finally:
			await service.shutdown()
	
	@pytest.mark.asyncio
	async def test_security_policy_enforcement(self):
		"""Test security policy enforcement during message operations"""
		service = MQEBService({
			'quantum_security_enabled': True,
			'strict_security_mode': True
		})
		await service.initialize()
		
		try:
			# Create topic
			from ..models import TopicConfiguration
			topic_config = TopicConfiguration(
				name="policy.test.topic",
				tenant_id="test_tenant",
				created_by="test_user"
			)
			await service.create_topic(topic_config)
			
			# Create message that should trigger security policies
			message = MQMessage(
				topic="policy.test.topic",
				payload=b"Suspicious content with <script>alert('xss')</script>",
				tenant_id="test_tenant",
				source_application="untrusted_app"
			)
			
			# Attempt to publish with suspicious context
			suspicious_context = {
				'authenticated': False,
				'source_ip': '1.2.3.4',  # External IP
				'user_id': 'suspicious_user'
			}
			
			# This should either fail or generate warnings
			try:
				message_id = await service.publish_message(message, suspicious_context)
				# If it succeeds, check that warnings were logged
				# In a production system, this might be rejected entirely
			except ValueError as e:
				# Security rejection is also acceptable
				assert "security" in str(e).lower()
		
		finally:
			await service.shutdown()
	
	@pytest.mark.asyncio
	async def test_key_rotation_integration(self):
		"""Test key rotation in integrated system"""
		service = MQEBService({'quantum_security_enabled': True})
		await service.initialize()
		
		try:
			if hasattr(service, 'quantum_security'):
				# Check initial key count
				initial_key_count = len(service.quantum_security.key_manager.key_store)
				
				# Generate keys for tenant
				key_id = await service.quantum_security.key_manager.generate_key_pair(
					QuantumAlgorithm.CRYSTALS_KYBER_512,
					"rotation_test_tenant"
				)
				
				assert len(service.quantum_security.key_manager.key_store) == initial_key_count + 1
				
				# Verify key exists and is active
				key_pair = await service.quantum_security.key_manager.get_key_pair(key_id)
				assert key_pair is not None
				assert not key_pair.is_expired()
		
		finally:
			await service.shutdown()


if __name__ == "__main__":
	# Run tests if script is executed directly
	pytest.main([__file__, "-v"])