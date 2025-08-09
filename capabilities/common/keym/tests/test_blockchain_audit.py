#!/usr/bin/env python3
"""
APG Key Management - Blockchain Audit Tests
Comprehensive tests for blockchain-based immutable audit trail system

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa

from ..blockchain_audit import (
	BlockchainAuditLogger, AuditEvent, AuditBlock,
	AuditEventType, BlockchainType,
	log_key_operation, log_hsm_operation, log_policy_change, log_security_incident,
	create_blockchain_audit_system
)
from ..service import KeyManagementService


@pytest.fixture
async def mock_service():
	"""Mock key management service"""
	service = Mock(spec=KeyManagementService)
	service._db_pool = Mock()
	service.check_database_health = AsyncMock(return_value=True)
	service.check_cache_health = AsyncMock(return_value=True)
	return service


@pytest.fixture
def blockchain_config():
	"""Blockchain configuration for testing"""
	return {
		'type': 'private',
		'block_size': 5,  # Small for testing
		'block_interval': 1,  # Short interval for testing
		'difficulty': 2,  # Easy difficulty for testing
		'network_url': '',
		'contract_address': '',
		'private_key': '',
		'signing_key_path': ''
	}


@pytest.fixture
async def audit_logger(mock_service, blockchain_config):
	"""Blockchain audit logger instance"""
	logger = BlockchainAuditLogger(mock_service, blockchain_config)
	await logger.initialize()
	yield logger
	await logger.shutdown()


class TestAuditEvent:
	"""Test AuditEvent class"""
	
	def test_audit_event_creation(self):
		"""Test audit event creation and hash calculation"""
		event = AuditEvent(
			event_type=AuditEventType.KEY_CREATED,
			tenant_id="test-tenant",
			user_id="test-user",
			resource_id="test-key-123",
			resource_type="cryptographic_key",
			action="create_key",
			result="success",
			context={"algorithm": "AES-256"}
		)
		
		# Check event properties
		assert event.event_id is not None
		assert event.timestamp is not None
		assert event.event_type == AuditEventType.KEY_CREATED
		assert event.tenant_id == "test-tenant"
		assert event.user_id == "test-user"
		assert event.resource_id == "test-key-123"
		assert event.action == "create_key"
		assert event.result == "success"
		assert event.context["algorithm"] == "AES-256"
		
		# Check hash calculation
		assert event.hash_value is not None
		assert len(event.hash_value) == 64  # SHA-256 hex
		
		# Hash should be deterministic
		recalculated_hash = event.calculate_hash()
		assert event.hash_value == recalculated_hash
	
	def test_audit_event_serialization(self):
		"""Test audit event to/from dict conversion"""
		original_event = AuditEvent(
			event_type=AuditEventType.ENCRYPTION_OPERATION,
			tenant_id="test-tenant",
			user_id="test-user",
			resource_id="test-key-456",
			action="encrypt_data",
			result="success"
		)
		
		# Convert to dict
		event_dict = original_event.to_dict()
		
		# Check dict structure
		assert event_dict['event_id'] == original_event.event_id
		assert event_dict['event_type'] == original_event.event_type.value
		assert event_dict['tenant_id'] == original_event.tenant_id
		assert 'timestamp' in event_dict
		
		# Convert back from dict
		restored_event = AuditEvent.from_dict(event_dict)
		
		# Check restored event
		assert restored_event.event_id == original_event.event_id
		assert restored_event.event_type == original_event.event_type
		assert restored_event.tenant_id == original_event.tenant_id
		assert restored_event.hash_value == original_event.hash_value


class TestAuditBlock:
	"""Test AuditBlock class"""
	
	def test_audit_block_creation(self):
		"""Test audit block creation and validation"""
		# Create test events
		events = [
			AuditEvent(
				event_type=AuditEventType.KEY_CREATED,
				tenant_id="test-tenant",
				user_id="test-user",
				resource_id=f"test-key-{i}",
				action="create_key"
			) for i in range(3)
		]
		
		# Create block
		block = AuditBlock(
			block_number=1,
			previous_block_hash="0" * 64,
			events=events
		)
		
		# Check block properties
		assert block.block_id is not None
		assert block.block_number == 1
		assert block.previous_block_hash == "0" * 64
		assert len(block.events) == 3
		
		# Check merkle root calculation
		block.merkle_root = block.calculate_merkle_root()
		assert block.merkle_root is not None
		
		# Check block validation
		assert block.is_valid()
	
	def test_block_event_chaining(self):
		"""Test event chaining within block"""
		block = AuditBlock(block_number=1)
		
		# Add events one by one
		for i in range(3):
			event = AuditEvent(
				event_type=AuditEventType.KEY_ACCESSED,
				resource_id=f"key-{i}",
				action="access"
			)
			block.add_event(event)
		
		# Check event chaining
		for i in range(1, len(block.events)):
			current_event = block.events[i]
			previous_event = block.events[i-1]
			assert current_event.previous_hash == previous_event.hash_value
		
		# Check block integrity
		assert block.is_valid()
	
	def test_block_serialization(self):
		"""Test block serialization"""
		# Create block with events
		events = [
			AuditEvent(
				event_type=AuditEventType.KEY_DELETED,
				resource_id="test-key",
				action="delete"
			)
		]
		
		block = AuditBlock(
			block_number=5,
			events=events
		)
		block.merkle_root = block.calculate_merkle_root()
		
		# Serialize to dict
		block_dict = block.to_dict()
		
		# Check serialization
		assert block_dict['block_id'] == block.block_id
		assert block_dict['block_number'] == 5
		assert len(block_dict['events']) == 1
		assert block_dict['merkle_root'] == block.merkle_root


class TestBlockchainAuditLogger:
	"""Test BlockchainAuditLogger class"""
	
	@pytest.mark.asyncio
	async def test_logger_initialization(self, mock_service, blockchain_config):
		"""Test audit logger initialization"""
		logger = BlockchainAuditLogger(mock_service, blockchain_config)
		await logger.initialize()
		
		# Check initialization
		assert logger._is_running is True
		assert len(logger.blockchain) == 1  # Genesis block
		assert logger.blockchain[0].block_number == 0
		
		await logger.shutdown()
	
	@pytest.mark.asyncio
	async def test_genesis_block_creation(self, audit_logger):
		"""Test genesis block creation"""
		genesis_block = audit_logger.blockchain[0]
		
		# Check genesis block properties
		assert genesis_block.block_number == 0
		assert genesis_block.previous_block_hash == "0" * 64
		assert len(genesis_block.events) == 1
		
		# Check genesis event
		genesis_event = genesis_block.events[0]
		assert genesis_event.event_type == AuditEventType.ADMIN_ACTION
		assert genesis_event.action == "blockchain_initialized"
		assert genesis_event.context['genesis_block'] is True
	
	@pytest.mark.asyncio
	async def test_log_audit_event(self, audit_logger):
		"""Test logging audit events"""
		# Create test event
		event = AuditEvent(
			event_type=AuditEventType.KEY_CREATED,
			tenant_id="test-tenant",
			user_id="test-user",
			resource_id="test-key-123",
			action="create_key",
			result="success"
		)
		
		# Log event
		event_id = await audit_logger.log_audit_event(event)
		
		# Check event was logged
		assert event_id == event.event_id
		assert len(audit_logger.pending_events) == 1
		assert audit_logger.pending_events[0].event_id == event_id
		
		# Check event chaining
		if len(audit_logger.blockchain) > 0:
			last_block = audit_logger.blockchain[-1]
			if last_block.events:
				assert event.previous_hash == last_block.events[-1].hash_value
	
	@pytest.mark.asyncio
	async def test_block_creation(self, audit_logger):
		"""Test automatic block creation"""
		# Add events to reach block size threshold
		for i in range(audit_logger.block_size):
			event = AuditEvent(
				event_type=AuditEventType.KEY_ACCESSED,
				resource_id=f"key-{i}",
				action="access"
			)
			await audit_logger.log_audit_event(event)
		
		# Wait for block creation
		await asyncio.sleep(0.1)
		
		# Check block was created
		assert len(audit_logger.blockchain) == 2  # Genesis + new block
		assert len(audit_logger.pending_events) == 0  # Events moved to block
		
		new_block = audit_logger.blockchain[1]
		assert new_block.block_number == 1
		assert len(new_block.events) == audit_logger.block_size
		assert new_block.is_valid()
	
	@pytest.mark.asyncio
	async def test_blockchain_integrity_verification(self, audit_logger):
		"""Test blockchain integrity verification"""
		# Add multiple blocks
		for block_num in range(3):
			for event_num in range(audit_logger.block_size):
				event = AuditEvent(
					event_type=AuditEventType.ENCRYPTION_OPERATION,
					resource_id=f"key-{block_num}-{event_num}",
					action="encrypt"
				)
				await audit_logger.log_audit_event(event)
			
			# Wait for block creation
			await asyncio.sleep(0.1)
		
		# Verify blockchain integrity
		integrity_results = await audit_logger.verify_blockchain_integrity()
		
		# Check results
		assert integrity_results['valid'] is True
		assert integrity_results['total_blocks'] == 4  # Genesis + 3 blocks
		assert len(integrity_results['invalid_blocks']) == 0
		assert len(integrity_results['chain_breaks']) == 0
	
	@pytest.mark.asyncio
	async def test_audit_trail_filtering(self, audit_logger):
		"""Test audit trail retrieval with filtering"""
		# Add events for different users/resources
		test_events = [
			AuditEvent(
				event_type=AuditEventType.KEY_CREATED,
				user_id="user1",
				resource_id="key1",
				action="create"
			),
			AuditEvent(
				event_type=AuditEventType.KEY_ACCESSED,
				user_id="user2",
				resource_id="key1",
				action="access"
			),
			AuditEvent(
				event_type=AuditEventType.KEY_DELETED,
				user_id="user1",
				resource_id="key2",
				action="delete"
			)
		]
		
		for event in test_events:
			await audit_logger.log_audit_event(event)
		
		# Test filtering by user_id
		user1_events = await audit_logger.get_audit_trail(user_id="user1")
		assert len(user1_events) == 2
		assert all(event.user_id == "user1" for event in user1_events)
		
		# Test filtering by resource_id
		key1_events = await audit_logger.get_audit_trail(resource_id="key1")
		assert len(key1_events) == 2
		assert all(event.resource_id == "key1" for event in key1_events)
	
	@pytest.mark.asyncio
	async def test_merkle_proof_generation(self, audit_logger):
		"""Test Merkle proof generation for events"""
		# Add events to create a block
		events = []
		for i in range(audit_logger.block_size):
			event = AuditEvent(
				event_type=AuditEventType.HSM_OPERATION,
				resource_id=f"hsm-{i}",
				action="operation"
			)
			events.append(event)
			await audit_logger.log_audit_event(event)
		
		# Wait for block creation
		await asyncio.sleep(0.1)
		
		# Get merkle proof for first event
		target_event = events[0]
		proof = await audit_logger.get_merkle_proof(target_event.event_id)
		
		# Check proof structure
		assert proof is not None
		assert proof['event_id'] == target_event.event_id
		assert 'merkle_root' in proof
		assert 'proof' in proof
		assert 'leaf_index' in proof
	
	@pytest.mark.asyncio
	async def test_event_integrity_verification(self, audit_logger):
		"""Test individual event integrity verification"""
		# Add test event
		event = AuditEvent(
			event_type=AuditEventType.POLICY_CHANGE,
			resource_id="policy-123",
			action="update"
		)
		event_id = await audit_logger.log_audit_event(event)
		
		# Force block creation
		for i in range(audit_logger.block_size - 1):
			await audit_logger.log_audit_event(AuditEvent(action="filler"))
		await asyncio.sleep(0.1)
		
		# Verify event integrity
		verification_result = await audit_logger.verify_event_integrity(event_id)
		
		# Check verification results
		assert verification_result['valid'] is True
		assert verification_result['event_id'] == event_id
		assert verification_result['hash_valid'] is True
		assert verification_result['merkle_valid'] is True


class TestHelperFunctions:
	"""Test helper functions for common audit operations"""
	
	@pytest.mark.asyncio
	async def test_log_key_operation(self, audit_logger):
		"""Test key operation logging helper"""
		event_id = await log_key_operation(
			audit_logger,
			operation="encrypt",
			key_id="test-key-123",
			tenant_id="test-tenant",
			user_id="test-user",
			algorithm="AES-256",
			data_size=1024
		)
		
		assert event_id is not None
		assert len(audit_logger.pending_events) == 1
		
		logged_event = audit_logger.pending_events[0]
		assert logged_event.event_type == AuditEventType.KEY_ACCESSED
		assert logged_event.resource_id == "test-key-123"
		assert logged_event.action == "encrypt"
		assert logged_event.context["algorithm"] == "AES-256"
		assert logged_event.context["data_size"] == 1024
	
	@pytest.mark.asyncio
	async def test_log_hsm_operation(self, audit_logger):
		"""Test HSM operation logging helper"""
		event_id = await log_hsm_operation(
			audit_logger,
			hsm_id="luna-hsm-1",
			operation="generate_key",
			tenant_id="test-tenant",
			user_id="test-user",
			key_type="RSA-2048"
		)
		
		assert event_id is not None
		logged_event = audit_logger.pending_events[0]
		assert logged_event.event_type == AuditEventType.HSM_OPERATION
		assert logged_event.resource_id == "luna-hsm-1"
		assert logged_event.action == "generate_key"
		assert logged_event.context["key_type"] == "RSA-2048"
	
	@pytest.mark.asyncio
	async def test_log_policy_change(self, audit_logger):
		"""Test policy change logging helper"""
		old_policy = {"max_key_age": 365}
		new_policy = {"max_key_age": 180}
		
		event_id = await log_policy_change(
			audit_logger,
			policy_id="security-policy-1",
			change_type="update",
			tenant_id="test-tenant",
			user_id="admin-user",
			old_policy=old_policy,
			new_policy=new_policy
		)
		
		assert event_id is not None
		logged_event = audit_logger.pending_events[0]
		assert logged_event.event_type == AuditEventType.POLICY_CHANGE
		assert logged_event.resource_id == "security-policy-1"
		assert logged_event.action == "update"
		assert logged_event.context["old_policy"] == old_policy
		assert logged_event.context["new_policy"] == new_policy
	
	@pytest.mark.asyncio
	async def test_log_security_incident(self, audit_logger):
		"""Test security incident logging helper"""
		event_id = await log_security_incident(
			audit_logger,
			incident_type="unauthorized_access",
			severity="high",
			tenant_id="test-tenant",
			description="Multiple failed authentication attempts",
			source_ip="192.168.1.100",
			attempt_count=5
		)
		
		assert event_id is not None
		logged_event = audit_logger.pending_events[0]
		assert logged_event.event_type == AuditEventType.SECURITY_INCIDENT
		assert logged_event.action == "unauthorized_access"
		assert logged_event.result == "high"
		assert logged_event.context["description"] == "Multiple failed authentication attempts"
		assert logged_event.context["source_ip"] == "192.168.1.100"
		assert logged_event.context["attempt_count"] == 5


class TestBlockchainIntegration:
	"""Test blockchain network integration"""
	
	@pytest.mark.asyncio
	async def test_ethereum_configuration(self, mock_service):
		"""Test Ethereum blockchain configuration"""
		ethereum_config = {
			'type': 'ethereum',
			'network_url': 'http://localhost:8545',
			'contract_address': '0x1234567890123456789012345678901234567890',
			'private_key': '0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890',
			'poa_enabled': True
		}
		
		with patch('web3.Web3') as mock_web3:
			# Mock Web3 instance
			mock_w3 = Mock()
			mock_web3.return_value = mock_w3
			
			logger = BlockchainAuditLogger(mock_service, ethereum_config)
			await logger.initialize()
			
			# Check Ethereum configuration
			assert logger.blockchain_type == BlockchainType.ETHEREUM
			assert logger.network_url == 'http://localhost:8545'
			
			await logger.shutdown()
	
	@pytest.mark.asyncio
	async def test_private_blockchain_mode(self, mock_service):
		"""Test private blockchain mode"""
		private_config = {
			'type': 'private',
			'block_size': 10,
			'block_interval': 60,
			'difficulty': 3
		}
		
		logger = BlockchainAuditLogger(mock_service, private_config)
		await logger.initialize()
		
		# Check private blockchain configuration
		assert logger.blockchain_type == BlockchainType.PRIVATE_BLOCKCHAIN
		assert logger.block_size == 10
		assert logger.block_interval == 60
		
		await logger.shutdown()


class TestErrorHandling:
	"""Test error handling and edge cases"""
	
	@pytest.mark.asyncio
	async def test_invalid_event_verification(self, audit_logger):
		"""Test verification of invalid events"""
		# Create event with tampered hash
		event = AuditEvent(
			event_type=AuditEventType.KEY_CREATED,
			resource_id="test-key"
		)
		original_hash = event.hash_value
		event.hash_value = "invalid_hash"
		
		await audit_logger.log_audit_event(event)
		
		# Force block creation
		for i in range(audit_logger.block_size - 1):
			await audit_logger.log_audit_event(AuditEvent(action="filler"))
		await asyncio.sleep(0.1)
		
		# Verify blockchain integrity (should detect tampering)
		integrity_results = await audit_logger.verify_blockchain_integrity()
		assert integrity_results['valid'] is False
		assert len(integrity_results['invalid_events']) > 0
	
	@pytest.mark.asyncio
	async def test_broken_chain_detection(self, audit_logger):
		"""Test detection of broken chains"""
		# Add some valid events
		for i in range(audit_logger.block_size * 2):
			event = AuditEvent(resource_id=f"key-{i}")
			await audit_logger.log_audit_event(event)
		
		await asyncio.sleep(0.2)  # Wait for blocks to be created
		
		# Tamper with a block's previous hash
		if len(audit_logger.blockchain) > 1:
			audit_logger.blockchain[1].previous_block_hash = "tampered_hash"
		
		# Verify integrity
		integrity_results = await audit_logger.verify_blockchain_integrity()
		assert integrity_results['valid'] is False
		assert len(integrity_results['chain_breaks']) > 0
	
	@pytest.mark.asyncio
	async def test_missing_event_proof(self, audit_logger):
		"""Test handling of missing event proofs"""
		# Try to get proof for non-existent event
		proof = await audit_logger.get_merkle_proof("non-existent-event-id")
		assert proof is None
		
		# Try to verify non-existent event
		verification = await audit_logger.verify_event_integrity("non-existent-event-id")
		assert verification['valid'] is False
		assert 'error' in verification


@pytest.mark.asyncio
async def test_factory_function(mock_service):
	"""Test blockchain audit system factory function"""
	config = {
		'type': 'private',
		'block_size': 50,
		'difficulty': 2
	}
	
	audit_logger = await create_blockchain_audit_system(mock_service, config)
	
	# Check factory function result
	assert isinstance(audit_logger, BlockchainAuditLogger)
	assert audit_logger.blockchain_type == BlockchainType.PRIVATE_BLOCKCHAIN
	assert audit_logger.block_size == 50
	assert len(audit_logger.blockchain) == 1  # Genesis block
	
	await audit_logger.shutdown()


if __name__ == "__main__":
	pytest.main([__file__, "-v"])