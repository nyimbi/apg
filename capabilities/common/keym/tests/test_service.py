#!/usr/bin/env python3
"""
APG Key Management - Service Tests
Comprehensive test suite for key management service

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch

from ..service import KeyManagementService, create_key_management_service
from ..models import (
	KeyAlgorithm, KeyUsage, KeyState, SecurityLevel,
	KeyMetadata, KeyPolicy, KeySpec, Key, KeyOperation, 
	create_key_spec_async
)


@pytest.fixture
async def key_service():
	"""Fixture for key management service"""
	service = KeyManagementService()
	await service.initialize({
		'tenant_id': 'test_tenant',
		'test_mode': True
	})
	return service


@pytest.fixture
async def sample_key_spec():
	"""Fixture for sample key specification"""
	return await create_key_spec_async(
		tenant_id="test_tenant",
		algorithm=KeyAlgorithm.AES_256,
		usage=[KeyUsage.ENCRYPT, KeyUsage.DECRYPT],
		name="Test Key",
		created_by="test@datacraft.co.ke",
		description="A key for testing"
	)


class TestKeyManagementService:
	"""Test KeyManagementService class"""
	
	@pytest.mark.asyncio
	async def test_service_initialization(self):
		"""Test service initialization"""
		service = KeyManagementService()
		assert not service.is_initialized
		
		config = {'tenant_id': 'test_tenant'}
		await service.initialize(config)
		
		assert service.is_initialized
		assert service.config == config
		assert isinstance(service.keys, dict)
		assert isinstance(service.usage_stats, dict)
		assert isinstance(service.threats, dict)
		assert isinstance(service.audit_events, list)
	
	@pytest.mark.asyncio
	async def test_factory_function(self):
		"""Test service factory function"""
		service = await create_key_management_service()
		assert isinstance(service, KeyManagementService)
		assert service.is_initialized
	
	@pytest.mark.asyncio
	async def test_service_initialization_assertions(self):
		"""Test service initialization assertions"""
		service = KeyManagementService()
		
		with pytest.raises(AssertionError):
			await service.initialize("invalid_config")  # Not a dict
	
	@pytest.mark.asyncio
	async def test_create_key_success(self, key_service, sample_key_spec):
		"""Test successful key creation"""
		user_id = "test@datacraft.co.ke"
		
		key = await key_service.create_key(sample_key_spec, user_id)
		
		assert isinstance(key, Key)
		assert key.spec.id == sample_key_spec.id
		assert key.spec.tenant_id == "test_tenant"
		assert key.spec.algorithm == KeyAlgorithm.AES_256
		assert key.spec.state == KeyState.ACTIVE
		assert key.key_material is not None
		assert key.key_checksum is not None
		assert key.usage_count == 0
		
		# Check key is stored
		assert sample_key_spec.id in key_service.keys
		assert sample_key_spec.id in key_service.usage_stats
	
	@pytest.mark.asyncio
	async def test_create_key_not_initialized(self, sample_key_spec):
		"""Test key creation without initialization"""
		service = KeyManagementService()
		
		with pytest.raises(AssertionError, match="Service not initialized"):
			await service.create_key(sample_key_spec, "test@datacraft.co.ke")
	
	@pytest.mark.asyncio
	async def test_create_key_missing_spec(self, key_service):
		"""Test key creation with missing spec"""
		with pytest.raises(AssertionError, match="Key specification required"):
			await key_service.create_key(None, "test@datacraft.co.ke")
	
	@pytest.mark.asyncio
	async def test_create_asymmetric_key(self, key_service):
		"""Test asymmetric key creation"""
		spec = await create_key_spec_async(
			tenant_id="test_tenant",
			algorithm=KeyAlgorithm.RSA_2048,
			usage=[KeyUsage.SIGN, KeyUsage.VERIFY],
			name="RSA Test Key",
			created_by="test@datacraft.co.ke"
		)
		
		key = await key_service.create_key(spec, "test@datacraft.co.ke")
		
		assert key.spec.algorithm == KeyAlgorithm.RSA_2048
		assert key.public_key is not None  # Should have public key for asymmetric
		assert key.key_material is not None
	
	@pytest.mark.asyncio
	async def test_create_key_with_auto_rotation(self, key_service):
		"""Test key creation with auto-rotation"""
		spec = await create_key_spec_async(
			tenant_id="test_tenant",
			algorithm=KeyAlgorithm.AES_256,
			usage=[KeyUsage.ENCRYPT, KeyUsage.DECRYPT],
			name="Auto-Rotate Key",
			created_by="test@datacraft.co.ke",
			auto_rotate=True,
			rotation_interval_days=30
		)
		
		key = await key_service.create_key(spec, "test@datacraft.co.ke")
		
		assert key.next_rotation is not None
		assert key.next_rotation > datetime.utcnow()
	
	@pytest.mark.asyncio
	async def test_retrieve_key_success(self, key_service, sample_key_spec):
		"""Test successful key retrieval"""
		# Create key first
		created_key = await key_service.create_key(sample_key_spec, "test@datacraft.co.ke")
		
		# Retrieve key
		retrieved_key = await key_service.retrieve_key(sample_key_spec.id, "test@datacraft.co.ke")
		
		assert retrieved_key is not None
		assert retrieved_key.spec.id == created_key.spec.id
		assert retrieved_key.key_material is None  # Should not include material by default
		assert retrieved_key.hsm_key_id is None
	
	@pytest.mark.asyncio
	async def test_retrieve_key_with_material(self, key_service, sample_key_spec):
		"""Test key retrieval with material"""
		# Create key first
		await key_service.create_key(sample_key_spec, "test@datacraft.co.ke")
		
		# Retrieve key with material
		retrieved_key = await key_service.retrieve_key(
			sample_key_spec.id, 
			"test@datacraft.co.ke",
			include_material=True
		)
		
		assert retrieved_key is not None
		assert retrieved_key.key_material is not None
	
	@pytest.mark.asyncio
	async def test_retrieve_key_not_found(self, key_service):
		"""Test retrieving non-existent key"""
		result = await key_service.retrieve_key("nonexistent_key", "test@datacraft.co.ke")
		assert result is None
	
	@pytest.mark.asyncio
	async def test_retrieve_key_not_initialized(self):
		"""Test key retrieval without initialization"""
		service = KeyManagementService()
		
		with pytest.raises(AssertionError, match="Service not initialized"):
			await service.retrieve_key("key_id", "user_id")
	
	@pytest.mark.asyncio
	async def test_rotate_key_success(self, key_service, sample_key_spec):
		"""Test successful key rotation"""
		# Create key first
		original_key = await key_service.create_key(sample_key_spec, "test@datacraft.co.ke")
		original_checksum = original_key.key_checksum
		
		# Rotate key
		rotated_key = await key_service.rotate_key(sample_key_spec.id, "test@datacraft.co.ke")
		
		assert rotated_key.spec.id == sample_key_spec.id
		assert rotated_key.key_checksum != original_checksum  # Different key material
		assert len(rotated_key.previous_versions) == 1  # Should have one previous version
		assert rotated_key.spec.updated_at > original_key.spec.updated_at
	
	@pytest.mark.asyncio
	async def test_rotate_key_not_found(self, key_service):
		"""Test rotating non-existent key"""
		with pytest.raises(ValueError, match="Key not found"):
			await key_service.rotate_key("nonexistent_key", "test@datacraft.co.ke")
	
	@pytest.mark.asyncio
	async def test_delete_key_success(self, key_service, sample_key_spec):
		"""Test successful key deletion"""
		# Create key first
		await key_service.create_key(sample_key_spec, "test@datacraft.co.ke")
		
		# Delete key
		result = await key_service.delete_key(sample_key_spec.id, "test@datacraft.co.ke")
		
		assert result is True
		assert sample_key_spec.id not in key_service.keys
		assert sample_key_spec.id not in key_service.usage_stats
	
	@pytest.mark.asyncio
	async def test_delete_key_soft_delete(self, key_service, sample_key_spec):
		"""Test soft key deletion"""
		# Create key first
		await key_service.create_key(sample_key_spec, "test@datacraft.co.ke")
		
		# Soft delete key
		result = await key_service.delete_key(
			sample_key_spec.id, 
			"test@datacraft.co.ke",
			secure_delete=False
		)
		
		assert result is True
		# Key should still exist but be deactivated
		assert sample_key_spec.id in key_service.keys
		key = key_service.keys[sample_key_spec.id]
		assert key.spec.state == KeyState.DEACTIVATED
	
	@pytest.mark.asyncio
	async def test_delete_key_not_found(self, key_service):
		"""Test deleting non-existent key"""
		result = await key_service.delete_key("nonexistent_key", "test@datacraft.co.ke")
		assert result is False
	
	@pytest.mark.asyncio
	async def test_encrypt_data_success(self, key_service, sample_key_spec):
		"""Test successful data encryption"""
		# Create key first
		await key_service.create_key(sample_key_spec, "test@datacraft.co.ke")
		
		# Encrypt data
		test_data = b"Hello, World!"
		encrypted_data = await key_service.encrypt_data(
			sample_key_spec.id,
			test_data,
			"test@datacraft.co.ke"
		)
		
		assert isinstance(encrypted_data, bytes)
		assert len(encrypted_data) > len(test_data)  # Should be larger due to IV + tag
		
		# Check usage stats updated
		stats = key_service.usage_stats[sample_key_spec.id]
		assert stats.total_operations == 1
		assert stats.encrypt_operations == 1
	
	@pytest.mark.asyncio
	async def test_encrypt_data_key_not_found(self, key_service):
		"""Test encryption with non-existent key"""
		with pytest.raises(ValueError, match="Key not found"):
			await key_service.encrypt_data("nonexistent_key", b"data", "test@datacraft.co.ke")
	
	@pytest.mark.asyncio
	async def test_encrypt_data_invalid_usage(self, key_service):
		"""Test encryption with key that doesn't allow encryption"""
		# Create sign-only key
		spec = await create_key_spec_async(
			tenant_id="test_tenant",
			algorithm=KeyAlgorithm.RSA_2048,
			usage=[KeyUsage.SIGN],  # No ENCRYPT
			name="Sign Only Key",
			created_by="test@datacraft.co.ke"
		)
		
		await key_service.create_key(spec, "test@datacraft.co.ke")
		
		with pytest.raises(ValueError, match="Key not authorized for encryption"):
			await key_service.encrypt_data(spec.id, b"data", "test@datacraft.co.ke")
	
	@pytest.mark.asyncio
	async def test_decrypt_data_success(self, key_service, sample_key_spec):
		"""Test successful data decryption"""
		# Create key first
		await key_service.create_key(sample_key_spec, "test@datacraft.co.ke")
		
		# Encrypt then decrypt data
		test_data = b"Hello, World!"
		encrypted_data = await key_service.encrypt_data(
			sample_key_spec.id,
			test_data,
			"test@datacraft.co.ke"
		)
		
		decrypted_data = await key_service.decrypt_data(
			sample_key_spec.id,
			encrypted_data,
			"test@datacraft.co.ke"
		)
		
		assert decrypted_data == test_data
		
		# Check usage stats
		stats = key_service.usage_stats[sample_key_spec.id]
		assert stats.total_operations == 2
		assert stats.encrypt_operations == 1
		assert stats.decrypt_operations == 1
	
	@pytest.mark.asyncio
	async def test_decrypt_data_invalid_usage(self, key_service):
		"""Test decryption with key that doesn't allow decryption"""
		# Create sign-only key
		spec = await create_key_spec_async(
			tenant_id="test_tenant",
			algorithm=KeyAlgorithm.RSA_2048,
			usage=[KeyUsage.SIGN],  # No DECRYPT
			name="Sign Only Key",
			created_by="test@datacraft.co.ke"
		)
		
		await key_service.create_key(spec, "test@datacraft.co.ke")
		
		with pytest.raises(ValueError, match="Key not authorized for decryption"):
			await key_service.decrypt_data(spec.id, b"encrypted_data", "test@datacraft.co.ke")
	
	@pytest.mark.asyncio
	async def test_list_keys_success(self, key_service):
		"""Test successful key listing"""
		# Create multiple keys
		specs = []
		for i in range(3):
			spec = await create_key_spec_async(
				tenant_id="test_tenant",
				algorithm=KeyAlgorithm.AES_256,
				usage=[KeyUsage.ENCRYPT, KeyUsage.DECRYPT],
				name=f"Test Key {i}",
				created_by="test@datacraft.co.ke"
			)
			specs.append(spec)
			await key_service.create_key(spec, "test@datacraft.co.ke")
		
		# List keys
		keys = await key_service.list_keys("test_tenant", "test@datacraft.co.ke")
		
		assert len(keys) == 3
		for key in keys:
			assert key.spec.tenant_id == "test_tenant"
			assert key.key_material is None  # Should not include material
			assert key.hsm_key_id is None
	
	@pytest.mark.asyncio
	async def test_list_keys_with_filters(self, key_service):
		"""Test key listing with filters"""
		# Create keys with different algorithms
		aes_spec = await create_key_spec_async(
			tenant_id="test_tenant",
			algorithm=KeyAlgorithm.AES_256,
			usage=[KeyUsage.ENCRYPT],
			name="AES Key",
			created_by="test@datacraft.co.ke"
		)
		await key_service.create_key(aes_spec, "test@datacraft.co.ke")
		
		rsa_spec = await create_key_spec_async(
			tenant_id="test_tenant", 
			algorithm=KeyAlgorithm.RSA_2048,
			usage=[KeyUsage.SIGN],
			name="RSA Key",
			created_by="test@datacraft.co.ke"
		)
		await key_service.create_key(rsa_spec, "test@datacraft.co.ke")
		
		# List only AES keys
		filters = {'algorithm': KeyAlgorithm.AES_256}
		keys = await key_service.list_keys("test_tenant", "test@datacraft.co.ke", filters)
		
		assert len(keys) == 1
		assert keys[0].spec.algorithm == KeyAlgorithm.AES_256
		
		# List only RSA keys
		filters = {'algorithm': KeyAlgorithm.RSA_2048}
		keys = await key_service.list_keys("test_tenant", "test@datacraft.co.ke", filters)
		
		assert len(keys) == 1
		assert keys[0].spec.algorithm == KeyAlgorithm.RSA_2048
	
	@pytest.mark.asyncio
	async def test_get_key_usage_stats(self, key_service, sample_key_spec):
		"""Test getting key usage statistics"""
		# Create key and perform operations
		await key_service.create_key(sample_key_spec, "test@datacraft.co.ke")
		
		test_data = b"test data"
		await key_service.encrypt_data(sample_key_spec.id, test_data, "test@datacraft.co.ke")
		
		# Get stats
		stats = await key_service.get_key_usage_stats(sample_key_spec.id, "test@datacraft.co.ke")
		
		assert stats is not None
		assert stats.key_id == sample_key_spec.id
		assert stats.total_operations == 1
		assert stats.encrypt_operations == 1
		assert stats.decrypt_operations == 0
	
	@pytest.mark.asyncio
	async def test_get_key_usage_stats_not_found(self, key_service):
		"""Test getting stats for non-existent key"""
		stats = await key_service.get_key_usage_stats("nonexistent_key", "test@datacraft.co.ke")
		assert stats is None
	
	@pytest.mark.asyncio
	async def test_get_audit_events(self, key_service, sample_key_spec):
		"""Test getting audit events"""
		# Create key to generate audit events
		await key_service.create_key(sample_key_spec, "test@datacraft.co.ke")
		
		# Get audit events
		events = await key_service.get_audit_events("test_tenant", "test@datacraft.co.ke")
		
		assert isinstance(events, list)
		assert len(events) > 0  # Should have creation event
		
		# Check event structure
		event = events[0]
		assert event.tenant_id == "test_tenant"
		assert event.event_type == "key_created"
		assert event.resource_id == sample_key_spec.id
		assert event.user_id == "test@datacraft.co.ke"
	
	@pytest.mark.asyncio
	async def test_get_audit_events_with_filters(self, key_service, sample_key_spec):
		"""Test getting audit events with filters"""
		# Create key
		await key_service.create_key(sample_key_spec, "test@datacraft.co.ke")
		
		# Encrypt data
		await key_service.encrypt_data(sample_key_spec.id, b"test", "test@datacraft.co.ke")
		
		# Get filtered events
		filters = {'event_type': 'data_encrypted'}
		events = await key_service.get_audit_events("test_tenant", "test@datacraft.co.ke", filters)
		
		assert len(events) >= 1
		for event in events:
			assert event.event_type == "data_encrypted"
	
	@pytest.mark.asyncio
	async def test_get_service_health(self, key_service):
		"""Test getting service health"""
		health = await key_service.get_service_health()
		
		assert isinstance(health, dict)
		assert health['status'] == 'healthy'
		assert health['initialized'] is True
		assert 'total_keys' in health
		assert 'active_keys' in health
		assert 'total_operations' in health
		assert 'threats_detected' in health
		assert 'audit_events' in health
		assert 'timestamp' in health
	
	@pytest.mark.asyncio
	async def test_validate_tenant_access(self, key_service):
		"""Test tenant access validation"""
		# Should not raise exception for valid tenant
		result = await key_service._validate_tenant_access("test_tenant", "test@datacraft.co.ke")
		assert result is True
		
		# Should raise assertion for empty tenant
		with pytest.raises(AssertionError, match="Tenant ID required"):
			await key_service._validate_tenant_access("", "test@datacraft.co.ke")
	
	@pytest.mark.asyncio
	async def test_check_key_permissions(self, key_service):
		"""Test key permission checking"""
		# Should not raise exception with valid parameters
		result = await key_service._check_key_permissions("key_id", "user_id", "read_key")
		assert result is True
		
		# Should raise assertion for missing key ID
		with pytest.raises(AssertionError, match="Key ID required"):
			await key_service._check_key_permissions("", "user_id", "read_key")
		
		# Should raise assertion for missing operation
		with pytest.raises(AssertionError, match="Operation required"):
			await key_service._check_key_permissions("key_id", "user_id", "")
	
	@pytest.mark.asyncio
	async def test_threat_detection(self, key_service):
		"""Test security threat detection"""
		operation = KeyOperation(
			key_id="test_key",
			operation_type="encrypt",
			user_id="test@datacraft.co.ke",
			request_ip="192.168.1.100"
		)
		
		threats = await key_service._detect_security_threats(operation)
		
		assert isinstance(threats, list)
		# In this test implementation, should return empty list
		# In production, would integrate with actual security client
	
	@pytest.mark.asyncio
	async def test_symmetric_key_generation(self, key_service):
		"""Test symmetric key generation"""
		key_material = await key_service._generate_symmetric_key(KeyAlgorithm.AES_256, 256)
		
		assert isinstance(key_material, bytes)
		assert len(key_material) == 32  # 256 bits = 32 bytes
	
	@pytest.mark.asyncio
	async def test_asymmetric_key_generation(self, key_service):
		"""Test asymmetric key generation"""
		private_key, public_key = await key_service._generate_asymmetric_key_pair(
			KeyAlgorithm.RSA_2048, 
			2048
		)
		
		assert isinstance(private_key, bytes)
		assert isinstance(public_key, bytes)
		assert len(private_key) > 0
		assert len(public_key) > 0
		assert b'BEGIN PRIVATE KEY' in private_key
		assert b'BEGIN PUBLIC KEY' in public_key
	
	@pytest.mark.asyncio
	async def test_key_material_encryption_decryption(self, key_service):
		"""Test key material encryption and decryption"""
		tenant_id = "test_tenant"
		original_material = b"test_key_material_32_bytes_long"
		
		# Encrypt
		encrypted_material = await key_service._encrypt_key_material(original_material, tenant_id)
		assert isinstance(encrypted_material, bytes)
		assert len(encrypted_material) > len(original_material)
		
		# Decrypt
		decrypted_material = await key_service._decrypt_key_material(encrypted_material, tenant_id)
		assert decrypted_material == original_material
	
	@pytest.mark.asyncio
	async def test_error_handling_runtime_error(self, key_service):
		"""Test error handling for runtime errors"""
		# Mock a method to raise an exception
		with patch.object(key_service, '_generate_symmetric_key', side_effect=Exception("Test error")):
			spec = await create_key_spec_async(
				tenant_id="test_tenant",
				algorithm=KeyAlgorithm.AES_256,
				usage=[KeyUsage.ENCRYPT],
				name="Error Test Key",
				created_by="test@datacraft.co.ke"
			)
			
			with pytest.raises(RuntimeError, match="Key creation failed"):
				await key_service.create_key(spec, "test@datacraft.co.ke")


class TestServiceIntegration:
	"""Test service integration scenarios"""
	
	@pytest.mark.asyncio
	async def test_full_key_lifecycle(self, key_service):
		"""Test complete key lifecycle"""
		# 1. Create key
		spec = await create_key_spec_async(
			tenant_id="test_tenant",
			algorithm=KeyAlgorithm.AES_256,
			usage=[KeyUsage.ENCRYPT, KeyUsage.DECRYPT],
			name="Lifecycle Test Key",
			created_by="test@datacraft.co.ke"
		)
		
		key = await key_service.create_key(spec, "test@datacraft.co.ke")
		assert key.spec.state == KeyState.ACTIVE
		
		# 2. Use key for encryption/decryption
		test_data = b"Lifecycle test data"
		encrypted = await key_service.encrypt_data(key.spec.id, test_data, "test@datacraft.co.ke")
		decrypted = await key_service.decrypt_data(key.spec.id, encrypted, "test@datacraft.co.ke")
		assert decrypted == test_data
		
		# 3. Check usage statistics
		stats = await key_service.get_key_usage_stats(key.spec.id, "test@datacraft.co.ke")
		assert stats.total_operations == 2
		assert stats.encrypt_operations == 1
		assert stats.decrypt_operations == 1
		
		# 4. Rotate key
		rotated_key = await key_service.rotate_key(key.spec.id, "test@datacraft.co.ke")
		assert len(rotated_key.previous_versions) == 1
		
		# 5. Delete key
		deleted = await key_service.delete_key(key.spec.id, "test@datacraft.co.ke")
		assert deleted is True
	
	@pytest.mark.asyncio
	async def test_concurrent_operations(self, key_service):
		"""Test concurrent key operations"""
		# Create key
		spec = await create_key_spec_async(
			tenant_id="test_tenant",
			algorithm=KeyAlgorithm.AES_256,
			usage=[KeyUsage.ENCRYPT, KeyUsage.DECRYPT],
			name="Concurrent Test Key",
			created_by="test@datacraft.co.ke"
		)
		
		key = await key_service.create_key(spec, "test@datacraft.co.ke")
		
		# Perform concurrent encryptions
		test_data_list = [f"test data {i}".encode() for i in range(10)]
		
		async def encrypt_data(data):
			return await key_service.encrypt_data(key.spec.id, data, "test@datacraft.co.ke")
		
		# Run encryptions concurrently
		encrypted_results = await asyncio.gather(
			*[encrypt_data(data) for data in test_data_list]
		, return_exceptions=True)
		
		assert len(encrypted_results) == 10
		for result in encrypted_results:
			assert isinstance(result, bytes)
		
		# Check usage statistics
		stats = await key_service.get_key_usage_stats(key.spec.id, "test@datacraft.co.ke")
		assert stats.total_operations == 10
		assert stats.encrypt_operations == 10


if __name__ == "__main__":
	pytest.main([__file__])