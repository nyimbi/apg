#!/usr/bin/env python3
"""
Blockchain-Based Digital Twin Provenance and Security
====================================================

Comprehensive blockchain-based security system for digital twins providing:
- Immutable audit trails for all twin data and changes
- Smart contracts for automated compliance verification
- Cryptographic signatures for data integrity
- Decentralized identity management
- Supply chain provenance tracking
"""

import asyncio
import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import base64
import hmac
from pathlib import Path

# Cryptographic imports
try:
	from cryptography.hazmat.primitives import hashes, serialization
	from cryptography.hazmat.primitives.asymmetric import rsa, padding
	from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
	from cryptography.fernet import Fernet
	import cryptography.exceptions
except ImportError:
	print("Warning: cryptography not available. Install with: pip install cryptography")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("blockchain_security")

class TransactionType(Enum):
	"""Types of blockchain transactions"""
	TWIN_CREATION = "twin_creation"
	DATA_UPDATE = "data_update"
	STATE_CHANGE = "state_change"
	ACCESS_GRANT = "access_grant"
	ACCESS_REVOKE = "access_revoke"
	COMPLIANCE_CHECK = "compliance_check"
	AUDIT_LOG = "audit_log"
	PROVENANCE_RECORD = "provenance_record"

class ComplianceStatus(Enum):
	"""Compliance verification status"""
	COMPLIANT = "compliant"
	NON_COMPLIANT = "non_compliant"
	PENDING = "pending"
	EXEMPTED = "exempted"
	UNKNOWN = "unknown"

class AccessLevel(Enum):
	"""Access levels for digital twins"""
	READ_ONLY = "read_only"
	READ_WRITE = "read_write"
	ADMIN = "admin"
	OWNER = "owner"
	AUDIT_ONLY = "audit_only"

@dataclass
class DigitalSignature:
	"""Digital signature for data integrity"""
	signature: str
	public_key: str
	algorithm: str
	timestamp: datetime
	signer_id: str
	
	def to_dict(self) -> Dict[str, Any]:
		return {
			'signature': self.signature,
			'public_key': self.public_key,
			'algorithm': self.algorithm,
			'timestamp': self.timestamp.isoformat(),
			'signer_id': self.signer_id
		}

@dataclass
class BlockchainTransaction:
	"""Individual blockchain transaction"""
	transaction_id: str
	timestamp: datetime
	transaction_type: TransactionType
	twin_id: str
	previous_hash: str
	data_hash: str
	payload: Dict[str, Any]
	digital_signature: DigitalSignature
	nonce: int
	validator_signatures: List[DigitalSignature]
	
	def calculate_hash(self) -> str:
		"""Calculate hash of transaction"""
		data = {
			'transaction_id': self.transaction_id,
			'timestamp': self.timestamp.isoformat(),
			'type': self.transaction_type.value,
			'twin_id': self.twin_id,
			'previous_hash': self.previous_hash,
			'data_hash': self.data_hash,
			'payload': self.payload,
			'nonce': self.nonce
		}
		return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
	
	def to_dict(self) -> Dict[str, Any]:
		return {
			'transaction_id': self.transaction_id,
			'timestamp': self.timestamp.isoformat(),
			'transaction_type': self.transaction_type.value,
			'twin_id': self.twin_id,
			'previous_hash': self.previous_hash,
			'data_hash': self.data_hash,
			'payload': self.payload,
			'digital_signature': self.digital_signature.to_dict(),
			'nonce': self.nonce,
			'validator_signatures': [sig.to_dict() for sig in self.validator_signatures],
			'transaction_hash': self.calculate_hash()
		}

@dataclass
class ProvenanceRecord:
	"""Provenance record for supply chain tracking"""
	record_id: str
	twin_id: str
	timestamp: datetime
	event_type: str
	location: str
	participant: str
	previous_record_hash: Optional[str]
	metadata: Dict[str, Any]
	certifications: List[str]
	digital_signature: DigitalSignature
	
	def calculate_hash(self) -> str:
		"""Calculate hash of provenance record"""
		data = {
			'record_id': self.record_id,
			'twin_id': self.twin_id,
			'timestamp': self.timestamp.isoformat(),
			'event_type': self.event_type,
			'location': self.location,
			'participant': self.participant,
			'previous_record_hash': self.previous_record_hash,
			'metadata': self.metadata,
			'certifications': self.certifications
		}
		return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

@dataclass
class SmartContract:
	"""Smart contract for automated compliance"""
	contract_id: str
	name: str
	description: str
	version: str
	rules: List[Dict[str, Any]]
	conditions: List[Dict[str, Any]]
	actions: List[Dict[str, Any]]
	creator: str
	created_at: datetime
	is_active: bool
	execution_count: int
	
	def to_dict(self) -> Dict[str, Any]:
		return {
			'contract_id': self.contract_id,
			'name': self.name,
			'description': self.description,
			'version': self.version,
			'rules': self.rules,
			'conditions': self.conditions,
			'actions': self.actions,
			'creator': self.creator,
			'created_at': self.created_at.isoformat(),
			'is_active': self.is_active,
			'execution_count': self.execution_count
		}

class CryptographicManager:
	"""Manages cryptographic operations for blockchain security"""
	
	def __init__(self):
		self.key_pairs: Dict[str, Tuple[str, str]] = {}  # user_id -> (private_key, public_key)
		self.symmetric_keys: Dict[str, bytes] = {}  # session keys
		
	def generate_key_pair(self, user_id: str) -> Tuple[str, str]:
		"""Generate RSA key pair for user"""
		private_key = rsa.generate_private_key(
			public_exponent=65537,
			key_size=2048
		)
		
		# Serialize private key
		private_pem = private_key.private_bytes(
			encoding=serialization.Encoding.PEM,
			format=serialization.PrivateFormat.PKCS8,
			encryption_algorithm=serialization.NoEncryption()
		).decode('utf-8')
		
		# Serialize public key
		public_key = private_key.public_key()
		public_pem = public_key.public_bytes(
			encoding=serialization.Encoding.PEM,
			format=serialization.PublicFormat.SubjectPublicKeyInfo
		).decode('utf-8')
		
		self.key_pairs[user_id] = (private_pem, public_pem)
		return private_pem, public_pem
	
	def sign_data(self, data: str, user_id: str) -> str:
		"""Create digital signature for data"""
		if user_id not in self.key_pairs:
			raise ValueError(f"No key pair found for user {user_id}")
		
		private_key_pem, _ = self.key_pairs[user_id]
		private_key = serialization.load_pem_private_key(
			private_key_pem.encode('utf-8'),
			password=None
		)
		
		signature = private_key.sign(
			data.encode('utf-8'),
			padding.PSS(
				mgf=padding.MGF1(hashes.SHA256()),
				salt_length=padding.PSS.MAX_LENGTH
			),
			hashes.SHA256()
		)
		
		return base64.b64encode(signature).decode('utf-8')
	
	def verify_signature(self, data: str, signature: str, public_key_pem: str) -> bool:
		"""Verify digital signature"""
		try:
			public_key = serialization.load_pem_public_key(public_key_pem.encode('utf-8'))
			signature_bytes = base64.b64decode(signature.encode('utf-8'))
			
			public_key.verify(
				signature_bytes,
				data.encode('utf-8'),
				padding.PSS(
					mgf=padding.MGF1(hashes.SHA256()),
					salt_length=padding.PSS.MAX_LENGTH
				),
				hashes.SHA256()
			)
			return True
		except Exception as e:
			logger.error(f"Signature verification failed: {e}")
			return False
	
	def encrypt_data(self, data: str, user_id: str) -> str:
		"""Encrypt data using user's public key"""
		if user_id not in self.key_pairs:
			raise ValueError(f"No key pair found for user {user_id}")
		
		_, public_key_pem = self.key_pairs[user_id]
		public_key = serialization.load_pem_public_key(public_key_pem.encode('utf-8'))
		
		encrypted = public_key.encrypt(
			data.encode('utf-8'),
			padding.OAEP(
				mgf=padding.MGF1(algorithm=hashes.SHA256()),
				algorithm=hashes.SHA256(),
				label=None
			)
		)
		
		return base64.b64encode(encrypted).decode('utf-8')
	
	def decrypt_data(self, encrypted_data: str, user_id: str) -> str:
		"""Decrypt data using user's private key"""
		if user_id not in self.key_pairs:
			raise ValueError(f"No key pair found for user {user_id}")
		
		private_key_pem, _ = self.key_pairs[user_id]
		private_key = serialization.load_pem_private_key(
			private_key_pem.encode('utf-8'),
			password=None
		)
		
		encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
		decrypted = private_key.decrypt(
			encrypted_bytes,
			padding.OAEP(
				mgf=padding.MGF1(algorithm=hashes.SHA256()),
				algorithm=hashes.SHA256(),
				label=None
			)
		)
		
		return decrypted.decode('utf-8')

class SmartContractEngine:
	"""Executes smart contracts for automated compliance"""
	
	def __init__(self):
		self.contracts: Dict[str, SmartContract] = {}
		self.execution_history: List[Dict[str, Any]] = []
	
	def register_contract(self, contract: SmartContract):
		"""Register a new smart contract"""
		self.contracts[contract.contract_id] = contract
		logger.info(f"Smart contract {contract.name} registered with ID {contract.contract_id}")
	
	async def execute_contract(self, contract_id: str, twin_data: Dict[str, Any], 
							  context: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Execute smart contract rules and return actions"""
		if contract_id not in self.contracts:
			raise ValueError(f"Contract {contract_id} not found")
		
		contract = self.contracts[contract_id]
		if not contract.is_active:
			return []
		
		execution_result = {
			'contract_id': contract_id,
			'timestamp': datetime.utcnow(),
			'twin_id': context.get('twin_id'),
			'triggered_rules': [],
			'executed_actions': [],
			'compliance_status': ComplianceStatus.UNKNOWN
		}
		
		# Evaluate conditions
		triggered_rules = []
		for rule in contract.rules:
			if self._evaluate_rule(rule, twin_data, context):
				triggered_rules.append(rule)
				execution_result['triggered_rules'].append(rule['rule_id'])
		
		# Execute actions for triggered rules
		executed_actions = []
		for rule in triggered_rules:
			actions = rule.get('actions', [])
			for action in actions:
				result = await self._execute_action(action, twin_data, context)
				executed_actions.append(result)
				execution_result['executed_actions'].append(result)
		
		# Determine compliance status
		compliance_status = self._determine_compliance_status(triggered_rules, executed_actions)
		execution_result['compliance_status'] = compliance_status
		
		# Update contract execution count
		contract.execution_count += 1
		
		# Record execution
		self.execution_history.append(execution_result)
		
		logger.info(f"Contract {contract.name} executed: {len(triggered_rules)} rules triggered, {len(executed_actions)} actions executed")
		
		return executed_actions
	
	def _evaluate_rule(self, rule: Dict[str, Any], twin_data: Dict[str, Any], 
					   context: Dict[str, Any]) -> bool:
		"""Evaluate if a rule condition is met"""
		conditions = rule.get('conditions', [])
		
		for condition in conditions:
			condition_type = condition.get('type')
			field = condition.get('field')
			operator = condition.get('operator')
			value = condition.get('value')
			
			# Get field value from twin data
			field_value = self._get_nested_value(twin_data, field)
			
			# Evaluate condition
			if not self._evaluate_condition(field_value, operator, value):
				return False
		
		return True
	
	def _evaluate_condition(self, field_value: Any, operator: str, expected_value: Any) -> bool:
		"""Evaluate individual condition"""
		try:
			if operator == 'equals':
				return field_value == expected_value
			elif operator == 'not_equals':
				return field_value != expected_value
			elif operator == 'greater_than':
				return float(field_value) > float(expected_value)
			elif operator == 'less_than':
				return float(field_value) < float(expected_value)
			elif operator == 'greater_than_or_equal':
				return float(field_value) >= float(expected_value)
			elif operator == 'less_than_or_equal':
				return float(field_value) <= float(expected_value)
			elif operator == 'contains':
				return expected_value in str(field_value)
			elif operator == 'not_contains':
				return expected_value not in str(field_value)
			elif operator == 'exists':
				return field_value is not None
			elif operator == 'not_exists':
				return field_value is None
			else:
				logger.warning(f"Unknown operator: {operator}")
				return False
		except Exception as e:
			logger.error(f"Error evaluating condition: {e}")
			return False
	
	def _get_nested_value(self, data: Dict[str, Any], field_path: str) -> Any:
		"""Get value from nested dictionary using dot notation"""
		parts = field_path.split('.')
		value = data
		
		for part in parts:
			if isinstance(value, dict) and part in value:
				value = value[part]
			else:
				return None
		
		return value
	
	async def _execute_action(self, action: Dict[str, Any], twin_data: Dict[str, Any], 
							 context: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute a smart contract action"""
		action_type = action.get('type')
		
		result = {
			'action_id': action.get('action_id'),
			'action_type': action_type,
			'timestamp': datetime.utcnow(),
			'status': 'success',
			'message': '',
			'data': {}
		}
		
		try:
			if action_type == 'alert':
				await self._send_alert(action, twin_data, context)
				result['message'] = f"Alert sent: {action.get('message')}"
				
			elif action_type == 'update_field':
				field = action.get('field')
				value = action.get('value')
				# In a real implementation, this would update the twin
				result['message'] = f"Updated {field} to {value}"
				result['data'] = {'field': field, 'new_value': value}
				
			elif action_type == 'restrict_access':
				user_id = action.get('user_id')
				# In a real implementation, this would update access controls
				result['message'] = f"Access restricted for user {user_id}"
				result['data'] = {'restricted_user': user_id}
				
			elif action_type == 'generate_report':
				report_type = action.get('report_type')
				# In a real implementation, this would generate a compliance report
				result['message'] = f"Generated {report_type} report"
				result['data'] = {'report_type': report_type, 'generated_at': datetime.utcnow().isoformat()}
				
			elif action_type == 'backup_data':
				# In a real implementation, this would trigger data backup
				result['message'] = "Data backup initiated"
				result['data'] = {'backup_initiated': True}
				
			else:
				result['status'] = 'error'
				result['message'] = f"Unknown action type: {action_type}"
				
		except Exception as e:
			result['status'] = 'error'
			result['message'] = str(e)
			logger.error(f"Error executing action {action_type}: {e}")
		
		return result
	
	async def _send_alert(self, action: Dict[str, Any], twin_data: Dict[str, Any], 
						  context: Dict[str, Any]):
		"""Send compliance alert"""
		# In a real implementation, this would send alerts via email, SMS, etc.
		logger.warning(f"COMPLIANCE ALERT: {action.get('message')} for twin {context.get('twin_id')}")
	
	def _determine_compliance_status(self, triggered_rules: List[Dict[str, Any]], 
									executed_actions: List[Dict[str, Any]]) -> ComplianceStatus:
		"""Determine overall compliance status"""
		if not triggered_rules:
			return ComplianceStatus.COMPLIANT
		
		# Check if any rule indicates non-compliance
		for rule in triggered_rules:
			if rule.get('compliance_impact') == 'violation':
				return ComplianceStatus.NON_COMPLIANT
		
		# Check if all corrective actions succeeded
		failed_actions = [action for action in executed_actions if action.get('status') != 'success']
		if failed_actions:
			return ComplianceStatus.PENDING
		
		return ComplianceStatus.COMPLIANT

class ProvenanceTracker:
	"""Tracks provenance and supply chain history"""
	
	def __init__(self, crypto_manager: CryptographicManager):
		self.crypto_manager = crypto_manager
		self.provenance_chains: Dict[str, List[ProvenanceRecord]] = {}
	
	def create_provenance_record(self, twin_id: str, event_type: str, location: str,
								participant: str, metadata: Dict[str, Any],
								certifications: List[str], signer_id: str) -> ProvenanceRecord:
		"""Create new provenance record"""
		
		# Get previous record hash if exists
		previous_hash = None
		if twin_id in self.provenance_chains and self.provenance_chains[twin_id]:
			previous_hash = self.provenance_chains[twin_id][-1].calculate_hash()
		
		record_id = f"prov_{uuid.uuid4().hex[:12]}"
		
		# Create digital signature
		record_data = {
			'record_id': record_id,
			'twin_id': twin_id,
			'event_type': event_type,
			'location': location,
			'participant': participant,
			'metadata': metadata,
			'certifications': certifications
		}
		
		signature_data = json.dumps(record_data, sort_keys=True)
		signature = self.crypto_manager.sign_data(signature_data, signer_id)
		_, public_key = self.crypto_manager.key_pairs[signer_id]
		
		digital_signature = DigitalSignature(
			signature=signature,
			public_key=public_key,
			algorithm="RSA-PSS-SHA256",
			timestamp=datetime.utcnow(),
			signer_id=signer_id
		)
		
		record = ProvenanceRecord(
			record_id=record_id,
			twin_id=twin_id,
			timestamp=datetime.utcnow(),
			event_type=event_type,
			location=location,
			participant=participant,
			previous_record_hash=previous_hash,
			metadata=metadata,
			certifications=certifications,
			digital_signature=digital_signature
		)
		
		# Add to chain
		if twin_id not in self.provenance_chains:
			self.provenance_chains[twin_id] = []
		self.provenance_chains[twin_id].append(record)
		
		logger.info(f"Provenance record {record_id} created for twin {twin_id}")
		return record
	
	def verify_provenance_chain(self, twin_id: str) -> Dict[str, Any]:
		"""Verify integrity of provenance chain"""
		if twin_id not in self.provenance_chains:
			return {'valid': False, 'error': 'No provenance chain found'}
		
		chain = self.provenance_chains[twin_id]
		verification_result = {
			'valid': True,
			'record_count': len(chain),
			'verified_records': 0,
			'invalid_records': [],
			'broken_links': []
		}
		
		previous_hash = None
		for i, record in enumerate(chain):
			# Verify digital signature
			record_data = {
				'record_id': record.record_id,
				'twin_id': record.twin_id,
				'event_type': record.event_type,
				'location': record.location,
				'participant': record.participant,
				'metadata': record.metadata,
				'certifications': record.certifications
			}
			signature_data = json.dumps(record_data, sort_keys=True)
			
			if not self.crypto_manager.verify_signature(
				signature_data, 
				record.digital_signature.signature, 
				record.digital_signature.public_key
			):
				verification_result['valid'] = False
				verification_result['invalid_records'].append(i)
				continue
			
			# Verify chain link
			if record.previous_record_hash != previous_hash:
				verification_result['valid'] = False
				verification_result['broken_links'].append(i)
				continue
			
			verification_result['verified_records'] += 1
			previous_hash = record.calculate_hash()
		
		return verification_result
	
	def get_provenance_history(self, twin_id: str) -> List[Dict[str, Any]]:
		"""Get complete provenance history for twin"""
		if twin_id not in self.provenance_chains:
			return []
		
		return [
			{
				'record_id': record.record_id,
				'timestamp': record.timestamp.isoformat(),
				'event_type': record.event_type,
				'location': record.location,
				'participant': record.participant,
				'metadata': record.metadata,
				'certifications': record.certifications,
				'hash': record.calculate_hash()
			}
			for record in self.provenance_chains[twin_id]
		]

class BlockchainSecurityEngine:
	"""Main blockchain security engine for digital twins"""
	
	def __init__(self, config: Dict[str, Any] = None):
		self.config = config or {}
		self.crypto_manager = CryptographicManager()
		self.smart_contract_engine = SmartContractEngine()
		self.provenance_tracker = ProvenanceTracker(self.crypto_manager)
		
		# Blockchain storage
		self.transactions: Dict[str, BlockchainTransaction] = {}
		self.transaction_chains: Dict[str, List[str]] = {}  # twin_id -> [transaction_ids]
		
		# Access control
		self.access_controls: Dict[str, Dict[str, AccessLevel]] = {}  # twin_id -> {user_id: access_level}
		
		# Initialize default smart contracts
		self._create_default_smart_contracts()
		
		logger.info("Blockchain Security Engine initialized")
	
	def register_user(self, user_id: str) -> Tuple[str, str]:
		"""Register new user and generate key pair"""
		private_key, public_key = self.crypto_manager.generate_key_pair(user_id)
		logger.info(f"User {user_id} registered with new key pair")
		return private_key, public_key
	
	async def create_twin_transaction(self, twin_id: str, transaction_type: TransactionType,
									 payload: Dict[str, Any], signer_id: str) -> str:
		"""Create new blockchain transaction for digital twin"""
		
		# Get previous transaction hash
		previous_hash = "0" * 64  # Genesis hash
		if twin_id in self.transaction_chains and self.transaction_chains[twin_id]:
			previous_tx_id = self.transaction_chains[twin_id][-1]
			previous_hash = self.transactions[previous_tx_id].calculate_hash()
		
		# Calculate data hash
		data_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
		
		# Create digital signature
		transaction_data = {
			'twin_id': twin_id,
			'type': transaction_type.value,
			'data_hash': data_hash,
			'timestamp': datetime.utcnow().isoformat()
		}
		signature_data = json.dumps(transaction_data, sort_keys=True)
		signature = self.crypto_manager.sign_data(signature_data, signer_id)
		_, public_key = self.crypto_manager.key_pairs[signer_id]
		
		digital_signature = DigitalSignature(
			signature=signature,
			public_key=public_key,
			algorithm="RSA-PSS-SHA256",
			timestamp=datetime.utcnow(),
			signer_id=signer_id
		)
		
		# Create transaction
		transaction_id = f"tx_{uuid.uuid4().hex[:16]}"
		transaction = BlockchainTransaction(
			transaction_id=transaction_id,
			timestamp=datetime.utcnow(),
			transaction_type=transaction_type,
			twin_id=twin_id,
			previous_hash=previous_hash,
			data_hash=data_hash,
			payload=payload,
			digital_signature=digital_signature,
			nonce=int(time.time() * 1000000) % 1000000,  # Simple nonce
			validator_signatures=[]  # Would be populated by network validators
		)
		
		# Store transaction
		self.transactions[transaction_id] = transaction
		
		# Add to chain
		if twin_id not in self.transaction_chains:
			self.transaction_chains[twin_id] = []
		self.transaction_chains[twin_id].append(transaction_id)
		
		# Execute relevant smart contracts
		await self._execute_smart_contracts_for_transaction(transaction)
		
		logger.info(f"Transaction {transaction_id} created for twin {twin_id}")
		return transaction_id
	
	async def _execute_smart_contracts_for_transaction(self, transaction: BlockchainTransaction):
		"""Execute smart contracts triggered by transaction"""
		context = {
			'twin_id': transaction.twin_id,
			'transaction_type': transaction.transaction_type.value,
			'timestamp': transaction.timestamp,
			'signer_id': transaction.digital_signature.signer_id
		}
		
		# Execute all active contracts
		for contract in self.smart_contract_engine.contracts.values():
			if contract.is_active:
				try:
					await self.smart_contract_engine.execute_contract(
						contract.contract_id,
						transaction.payload,
						context
					)
				except Exception as e:
					logger.error(f"Error executing contract {contract.name}: {e}")
	
	def verify_transaction_chain(self, twin_id: str) -> Dict[str, Any]:
		"""Verify integrity of transaction chain for twin"""
		if twin_id not in self.transaction_chains:
			return {'valid': False, 'error': 'No transaction chain found'}
		
		chain = self.transaction_chains[twin_id]
		verification_result = {
			'valid': True,
			'transaction_count': len(chain),
			'verified_transactions': 0,
			'invalid_transactions': [],
			'broken_links': []
		}
		
		previous_hash = "0" * 64
		for i, tx_id in enumerate(chain):
			transaction = self.transactions[tx_id]
			
			# Verify digital signature
			transaction_data = {
				'twin_id': transaction.twin_id,
				'type': transaction.transaction_type.value,
				'data_hash': transaction.data_hash,
				'timestamp': transaction.timestamp.isoformat()
			}
			signature_data = json.dumps(transaction_data, sort_keys=True)
			
			if not self.crypto_manager.verify_signature(
				signature_data,
				transaction.digital_signature.signature,
				transaction.digital_signature.public_key
			):
				verification_result['valid'] = False
				verification_result['invalid_transactions'].append(i)
				continue
			
			# Verify chain link
			if transaction.previous_hash != previous_hash:
				verification_result['valid'] = False
				verification_result['broken_links'].append(i)
				continue
			
			verification_result['verified_transactions'] += 1
			previous_hash = transaction.calculate_hash()
		
		return verification_result
	
	def grant_access(self, twin_id: str, user_id: str, access_level: AccessLevel, granter_id: str):
		"""Grant access to digital twin"""
		if twin_id not in self.access_controls:
			self.access_controls[twin_id] = {}
		
		self.access_controls[twin_id][user_id] = access_level
		
		# Create audit transaction
		asyncio.create_task(self.create_twin_transaction(
			twin_id=twin_id,
			transaction_type=TransactionType.ACCESS_GRANT,
			payload={
				'granted_to': user_id,
				'access_level': access_level.value,
				'granted_by': granter_id
			},
			signer_id=granter_id
		))
		
		logger.info(f"Access granted to {user_id} for twin {twin_id}: {access_level.value}")
	
	def revoke_access(self, twin_id: str, user_id: str, revoker_id: str):
		"""Revoke access to digital twin"""
		if twin_id in self.access_controls and user_id in self.access_controls[twin_id]:
			old_access = self.access_controls[twin_id][user_id]
			del self.access_controls[twin_id][user_id]
			
			# Create audit transaction
			asyncio.create_task(self.create_twin_transaction(
				twin_id=twin_id,
				transaction_type=TransactionType.ACCESS_REVOKE,
				payload={
					'revoked_from': user_id,
					'previous_access_level': old_access.value,
					'revoked_by': revoker_id
				},
				signer_id=revoker_id
			))
			
			logger.info(f"Access revoked from {user_id} for twin {twin_id}")
	
	def check_access(self, twin_id: str, user_id: str, required_level: AccessLevel) -> bool:
		"""Check if user has required access level"""
		if twin_id not in self.access_controls or user_id not in self.access_controls[twin_id]:
			return False
		
		user_level = self.access_controls[twin_id][user_id]
		
		# Define access level hierarchy
		level_hierarchy = {
			AccessLevel.READ_ONLY: 1,
			AccessLevel.AUDIT_ONLY: 2,
			AccessLevel.READ_WRITE: 3,
			AccessLevel.ADMIN: 4,
			AccessLevel.OWNER: 5
		}
		
		return level_hierarchy.get(user_level, 0) >= level_hierarchy.get(required_level, 0)
	
	def _create_default_smart_contracts(self):
		"""Create default smart contracts for common compliance scenarios"""
		
		# Temperature monitoring contract
		temp_contract = SmartContract(
			contract_id="temp_monitor_001",
			name="Temperature Monitoring",
			description="Monitors temperature thresholds and triggers alerts",
			version="1.0",
			rules=[
				{
					'rule_id': 'temp_high',
					'conditions': [
						{
							'type': 'threshold',
							'field': 'current_state.temperature',
							'operator': 'greater_than',
							'value': 80
						}
					],
					'actions': [
						{
							'action_id': 'temp_alert',
							'type': 'alert',
							'message': 'High temperature detected',
							'severity': 'warning'
						}
					],
					'compliance_impact': 'violation'
				}
			],
			conditions=[],
			actions=[],
			creator="system",
			created_at=datetime.utcnow(),
			is_active=True,
			execution_count=0
		)
		
		# Data integrity contract
		integrity_contract = SmartContract(
			contract_id="data_integrity_001",
			name="Data Integrity Monitoring",
			description="Ensures data integrity and triggers backup on anomalies",
			version="1.0",
			rules=[
				{
					'rule_id': 'data_anomaly',
					'conditions': [
						{
							'type': 'anomaly',
							'field': 'metadata_.anomaly_detected',
							'operator': 'equals',
							'value': True
						}
					],
					'actions': [
						{
							'action_id': 'backup_data',
							'type': 'backup_data',
							'immediate': True
						},
						{
							'action_id': 'integrity_alert',
							'type': 'alert',
							'message': 'Data anomaly detected - backup initiated',
							'severity': 'high'
						}
					],
					'compliance_impact': 'requires_attention'
				}
			],
			conditions=[],
			actions=[],
			creator="system",
			created_at=datetime.utcnow(),
			is_active=True,
			execution_count=0
		)
		
		self.smart_contract_engine.register_contract(temp_contract)
		self.smart_contract_engine.register_contract(integrity_contract)
	
	async def get_security_report(self, twin_id: str) -> Dict[str, Any]:
		"""Generate comprehensive security report for twin"""
		
		# Transaction chain verification
		tx_verification = self.verify_transaction_chain(twin_id)
		
		# Provenance chain verification
		prov_verification = self.provenance_tracker.verify_provenance_chain(twin_id)
		
		# Access control summary
		access_summary = {}
		if twin_id in self.access_controls:
			access_summary = {
				user_id: level.value
				for user_id, level in self.access_controls[twin_id].items()
			}
		
		# Smart contract execution history
		contract_history = [
			execution for execution in self.smart_contract_engine.execution_history
			if execution.get('twin_id') == twin_id
		]
		
		return {
			'twin_id': twin_id,
			'report_timestamp': datetime.utcnow().isoformat(),
			'transaction_chain': {
				'valid': tx_verification['valid'],
				'transaction_count': tx_verification['transaction_count'],
				'verified_count': tx_verification['verified_transactions'],
				'issues': {
					'invalid_transactions': len(tx_verification.get('invalid_transactions', [])),
					'broken_links': len(tx_verification.get('broken_links', []))
				}
			},
			'provenance_chain': {
				'valid': prov_verification['valid'],
				'record_count': prov_verification['record_count'],
				'verified_count': prov_verification['verified_records'],
				'issues': {
					'invalid_records': len(prov_verification.get('invalid_records', [])),
					'broken_links': len(prov_verification.get('broken_links', []))
				}
			},
			'access_control': {
				'total_users': len(access_summary),
				'users': access_summary
			},
			'smart_contracts': {
				'execution_count': len(contract_history),
				'recent_executions': contract_history[-10:]  # Last 10 executions
			},
			'overall_security_score': self._calculate_security_score(
				tx_verification, prov_verification, access_summary, contract_history
			)
		}
	
	def _calculate_security_score(self, tx_verification: Dict, prov_verification: Dict,
								 access_summary: Dict, contract_history: List) -> float:
		"""Calculate overall security score (0-100)"""
		score = 100.0
		
		# Transaction chain integrity (40% weight)
		if not tx_verification['valid']:
			score -= 40
		elif tx_verification['verified_transactions'] < tx_verification['transaction_count']:
			penalty = (1 - tx_verification['verified_transactions'] / tx_verification['transaction_count']) * 40
			score -= penalty
		
		# Provenance chain integrity (30% weight)  
		if not prov_verification['valid']:
			score -= 30
		elif prov_verification['verified_records'] < prov_verification['record_count']:
			penalty = (1 - prov_verification['verified_records'] / prov_verification['record_count']) * 30
			score -= penalty
		
		# Access control (20% weight)
		if not access_summary:
			score -= 10  # No access control is a moderate issue
		
		# Smart contract compliance (10% weight)
		if contract_history:
			non_compliant_executions = [
				ex for ex in contract_history[-20:]  # Check last 20 executions
				if ex.get('compliance_status') == ComplianceStatus.NON_COMPLIANT.value
			]
			if non_compliant_executions:
				penalty = (len(non_compliant_executions) / min(20, len(contract_history))) * 10
				score -= penalty
		
		return max(0.0, min(100.0, score))

# Test and example usage
async def test_blockchain_security():
	"""Test the blockchain security system"""
	
	# Initialize security engine
	security_engine = BlockchainSecurityEngine()
	
	# Register users
	print("Registering users...")
	admin_private, admin_public = security_engine.register_user("admin_001")
	user_private, user_public = security_engine.register_user("user_001")
	
	print(f"Admin public key: {admin_public[:50]}...")
	print(f"User public key: {user_public[:50]}...")
	
	# Create digital twin and grant access
	twin_id = "twin_industrial_001"
	security_engine.grant_access(twin_id, "admin_001", AccessLevel.OWNER, "admin_001")
	security_engine.grant_access(twin_id, "user_001", AccessLevel.READ_WRITE, "admin_001")
	
	# Create some blockchain transactions
	print(f"\nCreating blockchain transactions for {twin_id}...")
	
	# Twin creation
	await security_engine.create_twin_transaction(
		twin_id=twin_id,
		transaction_type=TransactionType.TWIN_CREATION,
		payload={
			'name': 'Industrial Machine 001',
			'type': 'manufacturing_equipment',
			'location': 'Factory Floor A',
			'initial_state': {
				'temperature': 25.0,
				'pressure': 10.0,
				'status': 'operational'
			}
		},
		signer_id="admin_001"
	)
	
	# Data updates
	await security_engine.create_twin_transaction(
		twin_id=twin_id,
		transaction_type=TransactionType.DATA_UPDATE,
		payload={
			'current_state': {
				'temperature': 75.0,
				'pressure': 12.0,
				'status': 'operational'
			},
			'update_reason': 'scheduled_monitoring'
		},
		signer_id="user_001"
	)
	
	# High temperature update (should trigger smart contract)
	await security_engine.create_twin_transaction(
		twin_id=twin_id,
		transaction_type=TransactionType.DATA_UPDATE,
		payload={
			'current_state': {
				'temperature': 85.0,  # Above threshold
				'pressure': 12.5,
				'status': 'operational'
			},
			'update_reason': 'anomaly_detected'
		},
		signer_id="user_001"
	)
	
	# Create provenance records
	print("\nCreating provenance records...")
	
	security_engine.provenance_tracker.create_provenance_record(
		twin_id=twin_id,
		event_type="manufacturing",
		location="Factory A - Line 1",
		participant="Manufacturer Corp",
		metadata={
			'batch_number': 'B2024001',
			'quality_check': 'passed',
			'operator': 'John Doe'
		},
		certifications=['ISO9001', 'CE'],
		signer_id="admin_001"
	)
	
	security_engine.provenance_tracker.create_provenance_record(
		twin_id=twin_id,
		event_type="quality_inspection",
		location="Quality Lab",
		participant="QA Inspector",
		metadata={
			'inspection_result': 'passed',
			'test_criteria': ['dimensional', 'material', 'functional'],
			'inspector': 'Jane Smith'
		},
		certifications=['ISO17025'],
		signer_id="admin_001"
	)
	
	# Verify chains
	print("\nVerifying blockchain integrity...")
	tx_verification = security_engine.verify_transaction_chain(twin_id)
	print(f"Transaction chain valid: {tx_verification['valid']}")
	print(f"Verified transactions: {tx_verification['verified_transactions']}/{tx_verification['transaction_count']}")
	
	prov_verification = security_engine.provenance_tracker.verify_provenance_chain(twin_id)
	print(f"Provenance chain valid: {prov_verification['valid']}")
	print(f"Verified records: {prov_verification['verified_records']}/{prov_verification['record_count']}")
	
	# Generate security report
	print("\nGenerating security report...")
	security_report = await security_engine.get_security_report(twin_id)
	
	print(f"\nSecurity Report for {twin_id}:")
	print(f"Overall Security Score: {security_report['overall_security_score']:.1f}/100")
	print(f"Transaction Chain: {security_report['transaction_chain']['verified_count']}/{security_report['transaction_chain']['transaction_count']} verified")
	print(f"Provenance Chain: {security_report['provenance_chain']['verified_count']}/{security_report['provenance_chain']['record_count']} verified")
	print(f"Access Control: {security_report['access_control']['total_users']} users")
	print(f"Smart Contract Executions: {security_report['smart_contracts']['execution_count']}")
	
	# Test access control
	print("\nTesting access control...")
	admin_access = security_engine.check_access(twin_id, "admin_001", AccessLevel.ADMIN)
	user_access = security_engine.check_access(twin_id, "user_001", AccessLevel.ADMIN)
	print(f"Admin has admin access: {admin_access}")
	print(f"User has admin access: {user_access}")

if __name__ == "__main__":
	asyncio.run(test_blockchain_security())