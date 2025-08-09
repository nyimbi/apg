#!/usr/bin/env python3
"""
APG Key Management - Blockchain Integration & Immutable Audit
Blockchain-based immutable audit trails for comprehensive security compliance

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
import hashlib
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import aiohttp
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from uuid_extensions import uuid7str
import merkletools
from eth_account import Account
from web3 import Web3, AsyncWeb3
from web3.middleware import geth_poa_middleware

from .service import KeyManagementService


class BlockchainType(str, Enum):
	"""Supported blockchain types"""
	ETHEREUM = "ethereum"
	HYPERLEDGER_FABRIC = "hyperledger_fabric"
	PRIVATE_BLOCKCHAIN = "private"
	CONSORTIUM = "consortium"


class AuditEventType(str, Enum):
	"""Audit event types"""
	KEY_CREATED = "key_created"
	KEY_ACCESSED = "key_accessed"
	KEY_ROTATED = "key_rotated"
	KEY_DELETED = "key_deleted"
	ENCRYPTION_OPERATION = "encryption_operation"
	DECRYPTION_OPERATION = "decryption_operation"
	HSM_OPERATION = "hsm_operation"
	POLICY_CHANGE = "policy_change"
	USER_ACCESS = "user_access"
	ADMIN_ACTION = "admin_action"
	COMPLIANCE_CHECK = "compliance_check"
	SECURITY_INCIDENT = "security_incident"


@dataclass
class AuditEvent:
	"""Immutable audit event"""
	event_id: str = field(default_factory=uuid7str)
	timestamp: datetime = field(default_factory=datetime.utcnow)
	event_type: AuditEventType = AuditEventType.KEY_ACCESSED
	tenant_id: str = ""
	user_id: str = ""
	resource_id: str = ""
	resource_type: str = ""
	action: str = ""
	result: str = ""
	ip_address: str = ""
	user_agent: str = ""
	session_id: str = ""
	
	# Context data
	context: Dict[str, Any] = field(default_factory=dict)
	metadata: Dict[str, Any] = field(default_factory=dict)
	
	# Integrity fields
	hash_value: str = ""
	previous_hash: str = ""
	merkle_root: str = ""
	digital_signature: str = ""
	
	def __post_init__(self):
		"""Calculate hash value after initialization"""
		if not self.hash_value:
			self.hash_value = self.calculate_hash()
	
	def calculate_hash(self) -> str:
		"""Calculate SHA-256 hash of the event"""
		# Create deterministic representation
		event_data = {
			'event_id': self.event_id,
			'timestamp': self.timestamp.isoformat(),
			'event_type': self.event_type.value,
			'tenant_id': self.tenant_id,
			'user_id': self.user_id,
			'resource_id': self.resource_id,
			'resource_type': self.resource_type,
			'action': self.action,
			'result': self.result,
			'ip_address': self.ip_address,
			'user_agent': self.user_agent,
			'session_id': self.session_id,
			'context': self.context,
			'metadata': self.metadata,
			'previous_hash': self.previous_hash
		}
		
		# Sort keys for deterministic JSON
		event_json = json.dumps(event_data, sort_keys=True, separators=(',', ':'))
		return hashlib.sha256(event_json.encode('utf-8')).hexdigest()
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert to dictionary"""
		data = asdict(self)
		data['timestamp'] = self.timestamp.isoformat()
		data['event_type'] = self.event_type.value
		return data
	
	@classmethod
	def from_dict(cls, data: Dict[str, Any]) -> 'AuditEvent':
		"""Create from dictionary"""
		data = data.copy()
		data['timestamp'] = datetime.fromisoformat(data['timestamp'])
		data['event_type'] = AuditEventType(data['event_type'])
		return cls(**data)


@dataclass
class AuditBlock:
	"""Blockchain audit block"""
	block_id: str = field(default_factory=uuid7str)
	block_number: int = 0
	timestamp: datetime = field(default_factory=datetime.utcnow)
	previous_block_hash: str = ""
	merkle_root: str = ""
	events: List[AuditEvent] = field(default_factory=list)
	
	# Blockchain specific
	nonce: int = 0
	difficulty: int = 4
	hash_value: str = ""
	digital_signature: str = ""
	validator: str = ""
	
	def __post_init__(self):
		"""Calculate block hash after initialization"""
		if not self.hash_value:
			self.hash_value = self.calculate_hash()
	
	def calculate_hash(self) -> str:
		"""Calculate block hash"""
		block_data = {
			'block_id': self.block_id,
			'block_number': self.block_number,
			'timestamp': self.timestamp.isoformat(),
			'previous_block_hash': self.previous_block_hash,
			'merkle_root': self.merkle_root,
			'events_hash': self.calculate_events_hash(),
			'nonce': self.nonce
		}
		
		block_json = json.dumps(block_data, sort_keys=True, separators=(',', ':'))
		return hashlib.sha256(block_json.encode('utf-8')).hexdigest()
	
	def calculate_events_hash(self) -> str:
		"""Calculate hash of all events in block"""
		if not self.events:
			return ""
		
		event_hashes = [event.hash_value for event in self.events]
		combined_hash = hashlib.sha256(''.join(event_hashes).encode('utf-8')).hexdigest()
		return combined_hash
	
	def calculate_merkle_root(self) -> str:
		"""Calculate Merkle root of events"""
		if not self.events:
			return ""
		
		merkle_tree = merkletools.MerkleTools()
		for event in self.events:
			merkle_tree.add_leaf(event.hash_value)
		
		merkle_tree.make_tree()
		return merkle_tree.get_merkle_root() or ""
	
	def add_event(self, event: AuditEvent):
		"""Add event to block"""
		if self.events:
			event.previous_hash = self.events[-1].hash_value
		
		self.events.append(event)
		self.merkle_root = self.calculate_merkle_root()
		self.hash_value = self.calculate_hash()
	
	def is_valid(self) -> bool:
		"""Validate block integrity"""
		# Check hash
		expected_hash = self.calculate_hash()
		if self.hash_value != expected_hash:
			return False
		
		# Check merkle root
		expected_merkle = self.calculate_merkle_root()
		if self.merkle_root != expected_merkle:
			return False
		
		# Check events integrity
		for i, event in enumerate(self.events):
			if event.hash_value != event.calculate_hash():
				return False
			
			if i > 0 and event.previous_hash != self.events[i-1].hash_value:
				return False
		
		return True
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert to dictionary"""
		return {
			'block_id': self.block_id,
			'block_number': self.block_number,
			'timestamp': self.timestamp.isoformat(),
			'previous_block_hash': self.previous_block_hash,
			'merkle_root': self.merkle_root,
			'events': [event.to_dict() for event in self.events],
			'nonce': self.nonce,
			'difficulty': self.difficulty,
			'hash_value': self.hash_value,
			'digital_signature': self.digital_signature,
			'validator': self.validator
		}


class BlockchainAuditLogger:
	"""Blockchain-based immutable audit logger"""
	
	def __init__(self, service: KeyManagementService, blockchain_config: Dict[str, Any] = None):
		self.service = service
		self.config = blockchain_config or {}
		
		# Blockchain configuration
		self.blockchain_type = BlockchainType(self.config.get('type', 'private'))
		self.network_url = self.config.get('network_url', 'http://localhost:8545')
		self.contract_address = self.config.get('contract_address', '')
		self.private_key = self.config.get('private_key', '')
		
		# Local blockchain state
		self.blockchain: List[AuditBlock] = []
		self.pending_events: List[AuditEvent] = []
		self.block_size = self.config.get('block_size', 100)
		self.block_interval = self.config.get('block_interval', 300)  # 5 minutes
		
		# Ethereum integration
		self.w3: Optional[Web3] = None
		self.account: Optional[Account] = None
		
		# Background tasks
		self._block_creation_task: Optional[asyncio.Task] = None
		self._sync_task: Optional[asyncio.Task] = None
		self._is_running = False
		
		# Digital signing
		self._signing_key: Optional[rsa.RSAPrivateKey] = None
		self._init_signing_key()
	
	def _init_signing_key(self):
		"""Initialize digital signing key"""
		try:
			if self.config.get('signing_key_path'):
				with open(self.config['signing_key_path'], 'rb') as key_file:
					self._signing_key = load_pem_private_key(key_file.read(), password=None)
			else:
				# Generate new RSA key for signing
				self._signing_key = rsa.generate_private_key(
					public_exponent=65537,
					key_size=2048
				)
			
			logging.info("Digital signing key initialized")
		except Exception as e:
			logging.error(f"Failed to initialize signing key: {e}")
	
	async def initialize(self):
		"""Initialize blockchain audit system"""
		logging.info("Initializing blockchain audit system...")
		
		# Initialize Ethereum connection if configured
		if self.blockchain_type == BlockchainType.ETHEREUM and self.network_url:
			try:
				self.w3 = Web3(Web3.HTTPProvider(self.network_url))
				
				# Add PoA middleware if needed
				if self.config.get('poa_enabled', False):
					self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
				
				# Initialize account if private key provided
				if self.private_key:
					self.account = Account.from_key(self.private_key)
				
				logging.info("Ethereum blockchain connection established")
			except Exception as e:
				logging.error(f"Failed to connect to Ethereum: {e}")
		
		# Load existing blockchain
		await self._load_blockchain()
		
		# Create genesis block if blockchain is empty
		if not self.blockchain:
			await self._create_genesis_block()
		
		# Start background tasks
		self._is_running = True
		self._block_creation_task = asyncio.create_task(self._block_creation_loop())
		self._sync_task = asyncio.create_task(self._sync_with_network())
		
		logging.info("Blockchain audit system initialized successfully")
	
	async def shutdown(self):
		"""Shutdown blockchain audit system"""
		logging.info("Shutting down blockchain audit system...")
		
		self._is_running = False
		
		if self._block_creation_task:
			self._block_creation_task.cancel()
		
		if self._sync_task:
			self._sync_task.cancel()
		
		# Save final state
		await self._save_blockchain()
		
		logging.info("Blockchain audit system shut down")
	
	async def log_audit_event(self, event: AuditEvent) -> str:
		"""Log audit event to blockchain"""
		# Set chain linkage
		if self.pending_events:
			event.previous_hash = self.pending_events[-1].hash_value
		elif self.blockchain:
			# Link to last event in latest block
			last_block = self.blockchain[-1]
			if last_block.events:
				event.previous_hash = last_block.events[-1].hash_value
		
		# Recalculate hash with previous hash
		event.hash_value = event.calculate_hash()
		
		# Digital signature
		if self._signing_key:
			event.digital_signature = self._sign_event(event)
		
		# Add to pending events
		self.pending_events.append(event)
		
		# Trigger block creation if threshold reached
		if len(self.pending_events) >= self.block_size:
			await self._create_block()
		
		logging.debug(f"Audit event logged: {event.event_id}")
		return event.event_id
	
	def _sign_event(self, event: AuditEvent) -> str:
		"""Digitally sign audit event"""
		try:
			event_bytes = event.hash_value.encode('utf-8')
			signature = self._signing_key.sign(
				event_bytes,
				padding.PSS(
					mgf=padding.MGF1(hashes.SHA256()),
					salt_length=padding.PSS.MAX_LENGTH
				),
				hashes.SHA256()
			)
			return signature.hex()
		except Exception as e:
			logging.error(f"Failed to sign event: {e}")
			return ""
	
	async def _create_genesis_block(self):
		"""Create genesis block"""
		genesis_event = AuditEvent(
			event_type=AuditEventType.ADMIN_ACTION,
			tenant_id="system",
			user_id="system",
			action="blockchain_initialized",
			result="success",
			context={'genesis_block': True}
		)
		
		genesis_block = AuditBlock(
			block_number=0,
			previous_block_hash="0" * 64,
			events=[genesis_event]
		)
		
		genesis_block.merkle_root = genesis_block.calculate_merkle_root()
		genesis_block.hash_value = genesis_block.calculate_hash()
		
		# Digital signature
		if self._signing_key:
			genesis_block.digital_signature = self._sign_block(genesis_block)
		
		self.blockchain.append(genesis_block)
		logging.info("Genesis block created")
	
	async def _create_block(self):
		"""Create new block with pending events"""
		if not self.pending_events:
			return
		
		previous_block = self.blockchain[-1] if self.blockchain else None
		previous_hash = previous_block.hash_value if previous_block else "0" * 64
		
		new_block = AuditBlock(
			block_number=len(self.blockchain),
			previous_block_hash=previous_hash,
			events=self.pending_events.copy()
		)
		
		# Calculate merkle root
		new_block.merkle_root = new_block.calculate_merkle_root()
		
		# Proof of work (simplified)
		await self._mine_block(new_block)
		
		# Digital signature
		if self._signing_key:
			new_block.digital_signature = self._sign_block(new_block)
		
		# Add to blockchain
		self.blockchain.append(new_block)
		
		# Clear pending events
		self.pending_events.clear()
		
		# Sync with network
		if self.blockchain_type == BlockchainType.ETHEREUM:
			await self._sync_block_to_ethereum(new_block)
		
		logging.info(f"New block created: {new_block.block_id} with {len(new_block.events)} events")
	
	async def _mine_block(self, block: AuditBlock):
		"""Simple proof-of-work mining"""
		target = "0" * block.difficulty
		
		while not block.hash_value.startswith(target):
			block.nonce += 1
			block.hash_value = block.calculate_hash()
			
			# Yield control to prevent blocking
			if block.nonce % 1000 == 0:
				await asyncio.sleep(0.001)
		
		logging.debug(f"Block mined with nonce: {block.nonce}")
	
	def _sign_block(self, block: AuditBlock) -> str:
		"""Digitally sign block"""
		try:
			block_bytes = block.hash_value.encode('utf-8')
			signature = self._signing_key.sign(
				block_bytes,
				padding.PSS(
					mgf=padding.MGF1(hashes.SHA256()),
					salt_length=padding.PSS.MAX_LENGTH
				),
				hashes.SHA256()
			)
			return signature.hex()
		except Exception as e:
			logging.error(f"Failed to sign block: {e}")
			return ""
	
	async def _block_creation_loop(self):
		"""Background task for periodic block creation"""
		while self._is_running:
			try:
				await asyncio.sleep(self.block_interval)
				
				if self.pending_events:
					await self._create_block()
				
			except asyncio.CancelledError:
				break
			except Exception as e:
				logging.error(f"Error in block creation loop: {e}")
				await asyncio.sleep(60)
	
	async def _sync_with_network(self):
		"""Background task for network synchronization"""
		while self._is_running:
			try:
				if self.blockchain_type == BlockchainType.ETHEREUM and self.w3:
					await self._sync_with_ethereum()
				
				await asyncio.sleep(300)  # Sync every 5 minutes
				
			except asyncio.CancelledError:
				break
			except Exception as e:
				logging.error(f"Error in network sync: {e}")
				await asyncio.sleep(60)
	
	async def _sync_block_to_ethereum(self, block: AuditBlock):
		"""Sync block to Ethereum blockchain"""
		if not self.w3 or not self.account:
			return
		
		try:
			# Create transaction data
			block_data = json.dumps(block.to_dict(), separators=(',', ':'))
			
			# Build transaction
			transaction = {
				'to': self.contract_address,
				'data': self.w3.to_hex(block_data.encode('utf-8')),
				'gas': 2000000,
				'gasPrice': self.w3.to_wei('20', 'gwei'),
				'nonce': self.w3.eth.get_transaction_count(self.account.address),
			}
			
			# Sign and send transaction
			signed_txn = self.w3.eth.account.sign_transaction(transaction, self.private_key)
			tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
			
			logging.info(f"Block synced to Ethereum: {tx_hash.hex()}")
			
		except Exception as e:
			logging.error(f"Failed to sync block to Ethereum: {e}")
	
	async def _sync_with_ethereum(self):
		"""Sync local blockchain with Ethereum"""
		# Implementation would depend on smart contract design
		# For now, just log the sync attempt
		logging.debug("Syncing with Ethereum network...")
	
	async def verify_blockchain_integrity(self) -> Dict[str, Any]:
		"""Verify entire blockchain integrity"""
		results = {
			'valid': True,
			'total_blocks': len(self.blockchain),
			'total_events': sum(len(block.events) for block in self.blockchain),
			'invalid_blocks': [],
			'invalid_events': [],
			'chain_breaks': []
		}
		
		for i, block in enumerate(self.blockchain):
			# Verify block integrity
			if not block.is_valid():
				results['valid'] = False
				results['invalid_blocks'].append({
					'block_number': block.block_number,
					'block_id': block.block_id,
					'issues': ['hash_mismatch', 'merkle_root_mismatch']
				})
			
			# Verify chain linkage
			if i > 0:
				previous_block = self.blockchain[i-1]
				if block.previous_block_hash != previous_block.hash_value:
					results['valid'] = False
					results['chain_breaks'].append({
						'block_number': block.block_number,
						'expected_hash': previous_block.hash_value,
						'actual_hash': block.previous_block_hash
					})
			
			# Verify events in block
			for j, event in enumerate(block.events):
				if event.hash_value != event.calculate_hash():
					results['valid'] = False
					results['invalid_events'].append({
						'block_number': block.block_number,
						'event_id': event.event_id,
						'event_index': j
					})
		
		logging.info(f"Blockchain integrity check: {'VALID' if results['valid'] else 'INVALID'}")
		return results
	
	async def get_audit_trail(self, resource_id: str = None, user_id: str = None, 
							 start_date: datetime = None, end_date: datetime = None) -> List[AuditEvent]:
		"""Get audit trail with filtering"""
		events = []
		
		for block in self.blockchain:
			for event in block.events:
				# Apply filters
				if resource_id and event.resource_id != resource_id:
					continue
				if user_id and event.user_id != user_id:
					continue
				if start_date and event.timestamp < start_date:
					continue
				if end_date and event.timestamp > end_date:
					continue
				
				events.append(event)
		
		# Sort by timestamp
		events.sort(key=lambda x: x.timestamp)
		return events
	
	async def get_merkle_proof(self, event_id: str) -> Optional[Dict[str, Any]]:
		"""Get Merkle proof for specific event"""
		for block in self.blockchain:
			for i, event in enumerate(block.events):
				if event.event_id == event_id:
					# Build merkle tree
					merkle_tree = merkletools.MerkleTools()
					for block_event in block.events:
						merkle_tree.add_leaf(block_event.hash_value)
					
					merkle_tree.make_tree()
					proof = merkle_tree.get_proof(i)
					
					return {
						'event_id': event_id,
						'block_id': block.block_id,
						'block_number': block.block_number,
						'merkle_root': block.merkle_root,
						'proof': proof,
						'leaf_index': i
					}
		
		return None
	
	async def verify_event_integrity(self, event_id: str) -> Dict[str, Any]:
		"""Verify specific event integrity with blockchain proof"""
		# Find event
		target_event = None
		target_block = None
		
		for block in self.blockchain:
			for event in block.events:
				if event.event_id == event_id:
					target_event = event
					target_block = block
					break
			if target_event:
				break
		
		if not target_event or not target_block:
			return {'valid': False, 'error': 'Event not found'}
		
		# Get merkle proof
		merkle_proof = await self.get_merkle_proof(event_id)
		
		# Verify event hash
		expected_hash = target_event.calculate_hash()
		hash_valid = target_event.hash_value == expected_hash
		
		# Verify merkle proof
		if merkle_proof:
			merkle_tree = merkletools.MerkleTools()
			merkle_valid = merkle_tree.validate_proof(
				merkle_proof['proof'],
				target_event.hash_value,
				merkle_proof['merkle_root']
			)
		else:
			merkle_valid = False
		
		return {
			'valid': hash_valid and merkle_valid,
			'event_id': event_id,
			'hash_valid': hash_valid,
			'merkle_valid': merkle_valid,
			'block_id': target_block.block_id,
			'block_number': target_block.block_number,
			'merkle_proof': merkle_proof
		}
	
	async def _load_blockchain(self):
		"""Load blockchain from persistent storage"""
		# Implementation would load from database or file system
		logging.info("Loading blockchain from storage...")
	
	async def _save_blockchain(self):
		"""Save blockchain to persistent storage"""
		# Implementation would save to database or file system
		logging.info("Saving blockchain to storage...")


# Helper functions for common audit events
async def log_key_operation(audit_logger: BlockchainAuditLogger, operation: str, 
						   key_id: str, tenant_id: str, user_id: str, 
						   result: str = "success", **context):
	"""Log key operation audit event"""
	event = AuditEvent(
		event_type=AuditEventType.KEY_ACCESSED,
		tenant_id=tenant_id,
		user_id=user_id,
		resource_id=key_id,
		resource_type="cryptographic_key",
		action=operation,
		result=result,
		context=context
	)
	
	return await audit_logger.log_audit_event(event)


async def log_hsm_operation(audit_logger: BlockchainAuditLogger, hsm_id: str,
						   operation: str, tenant_id: str, user_id: str,
						   result: str = "success", **context):
	"""Log HSM operation audit event"""
	event = AuditEvent(
		event_type=AuditEventType.HSM_OPERATION,
		tenant_id=tenant_id,
		user_id=user_id,
		resource_id=hsm_id,
		resource_type="hsm",
		action=operation,
		result=result,
		context=context
	)
	
	return await audit_logger.log_audit_event(event)


async def log_policy_change(audit_logger: BlockchainAuditLogger, policy_id: str,
						   change_type: str, tenant_id: str, user_id: str,
						   old_policy: Dict[str, Any] = None, 
						   new_policy: Dict[str, Any] = None):
	"""Log policy change audit event"""
	event = AuditEvent(
		event_type=AuditEventType.POLICY_CHANGE,
		tenant_id=tenant_id,
		user_id=user_id,
		resource_id=policy_id,
		resource_type="security_policy",
		action=change_type,
		result="success",
		context={
			'old_policy': old_policy,
			'new_policy': new_policy
		}
	)
	
	return await audit_logger.log_audit_event(event)


async def log_security_incident(audit_logger: BlockchainAuditLogger, incident_type: str,
							   severity: str, tenant_id: str, description: str,
							   user_id: str = "", **context):
	"""Log security incident audit event"""
	event = AuditEvent(
		event_type=AuditEventType.SECURITY_INCIDENT,
		tenant_id=tenant_id,
		user_id=user_id,
		action=incident_type,
		result=severity,
		context={
			'description': description,
			'severity': severity,
			**context
		}
	)
	
	return await audit_logger.log_audit_event(event)


# Factory function
async def create_blockchain_audit_system(service: KeyManagementService, 
										config: Dict[str, Any] = None) -> BlockchainAuditLogger:
	"""Create and initialize blockchain audit system"""
	default_config = {
		'type': 'private',
		'block_size': 100,
		'block_interval': 300,
		'difficulty': 4,
		'network_url': '',
		'contract_address': '',
		'private_key': '',
		'signing_key_path': '',
		'poa_enabled': False
	}
	
	# Merge with provided config
	if config:
		default_config.update(config)
	
	# Create audit logger
	audit_logger = BlockchainAuditLogger(service, default_config)
	
	# Initialize
	await audit_logger.initialize()
	
	return audit_logger


# Export main components
__all__ = [
	'BlockchainAuditLogger', 'AuditEvent', 'AuditBlock',
	'AuditEventType', 'BlockchainType',
	'log_key_operation', 'log_hsm_operation', 'log_policy_change', 'log_security_incident',
	'create_blockchain_audit_system'
]