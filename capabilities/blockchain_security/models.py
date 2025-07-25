"""
Blockchain Security & Trust Models

Database models for blockchain integration, decentralized identity,
smart contracts, and cryptocurrency operations with security and compliance.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from sqlalchemy import Column, String, Text, Integer, Float, Boolean, DateTime, JSON, ForeignKey, LargeBinary
from sqlalchemy.orm import relationship
from uuid_extensions import uuid7str
import hashlib
import json

from ..auth_rbac.models import BaseMixin, AuditMixin, Model


def uuid7str():
	"""Generate UUID7 string for consistent ID generation"""
	from uuid_extensions import uuid7
	return str(uuid7())


class BSBlockchainNetwork(Model, AuditMixin, BaseMixin):
	"""
	Blockchain network configuration and monitoring.
	
	Stores configuration for different blockchain networks with
	health monitoring, performance metrics, and connection management.
	"""
	__tablename__ = 'bs_blockchain_network'
	
	# Identity
	network_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Network Identity
	name = Column(String(100), nullable=False)
	network_key = Column(String(50), nullable=False, unique=True)  # ethereum, polygon, bitcoin, etc.
	network_type = Column(String(50), nullable=False, index=True)  # mainnet, testnet, private
	chain_id = Column(Integer, nullable=True, index=True)  # EVM chain ID
	
	# Network Configuration
	rpc_endpoints = Column(JSON, nullable=False)  # List of RPC endpoints
	websocket_endpoints = Column(JSON, default=list)  # WebSocket endpoints
	explorer_urls = Column(JSON, default=list)  # Block explorer URLs
	
	# Consensus Configuration
	consensus_mechanism = Column(String(50), nullable=False)  # proof_of_stake, proof_of_work, etc.
	block_time_seconds = Column(Integer, default=15)  # Average block time
	confirmation_blocks = Column(Integer, default=12)  # Blocks for confirmation
	
	# Network Status
	is_active = Column(Boolean, default=True, index=True)
	health_status = Column(String(20), default='unknown', index=True)  # healthy, degraded, unhealthy
	last_health_check = Column(DateTime, nullable=True)
	current_block_number = Column(Integer, nullable=True)
	
	# Performance Metrics
	average_response_time = Column(Float, default=0.0)  # milliseconds
	transaction_throughput = Column(Float, default=0.0)  # TPS
	network_congestion = Column(Float, default=0.0)  # 0-100 congestion level
	
	# Gas/Fee Configuration
	supports_gas = Column(Boolean, default=True)
	gas_unit = Column(String(20), default='gwei')  # gwei, sat, etc.
	average_gas_price = Column(Float, default=0.0)
	max_gas_price = Column(Float, nullable=True)  # Maximum allowed gas price
	
	# Usage Statistics
	total_transactions = Column(Integer, default=0)
	successful_transactions = Column(Integer, default=0)
	failed_transactions = Column(Integer, default=0)
	total_gas_used = Column(Integer, default=0)
	total_fees_paid = Column(Float, default=0.0)
	
	# Relationships
	audit_anchors = relationship("BSAuditAnchor", back_populates="network")
	transactions = relationship("BSTransaction", back_populates="network")
	smart_contracts = relationship("BSSmartContract", back_populates="network")
	
	def __repr__(self):
		return f"<BSBlockchainNetwork {self.name} ({self.network_key})>"
	
	def is_healthy(self) -> bool:
		"""Check if network is healthy and available"""
		return (self.is_active and 
				self.health_status in ['healthy', 'degraded'] and
				self.current_block_number is not None)
	
	def calculate_success_rate(self) -> float:
		"""Calculate transaction success rate"""
		total = self.successful_transactions + self.failed_transactions
		if total == 0:
			return 0.0
		return (self.successful_transactions / total) * 100
	
	def get_primary_rpc_endpoint(self) -> Optional[str]:
		"""Get the primary RPC endpoint"""
		if self.rpc_endpoints and len(self.rpc_endpoints) > 0:
			return self.rpc_endpoints[0]
		return None
	
	def update_performance_metrics(self, response_time: float, block_number: int = None,
								   gas_price: float = None) -> None:
		"""Update network performance metrics"""
		# Update average response time (exponential moving average)
		if self.average_response_time == 0:
			self.average_response_time = response_time
		else:
			alpha = 0.1
			self.average_response_time = alpha * response_time + (1 - alpha) * self.average_response_time
		
		if block_number:
			self.current_block_number = block_number
		
		if gas_price:
			if self.average_gas_price == 0:
				self.average_gas_price = gas_price
			else:
				self.average_gas_price = alpha * gas_price + (1 - alpha) * self.average_gas_price


class BSAuditAnchor(Model, AuditMixin, BaseMixin):
	"""
	Blockchain audit event anchoring for tamper-proof logging.
	
	Links audit events to blockchain transactions for immutable
	verification and tamper-proof audit trails.
	"""
	__tablename__ = 'bs_audit_anchor'
	
	# Identity
	anchor_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	network_id = Column(String(36), ForeignKey('bs_blockchain_network.network_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Audit Event Reference
	audit_log_ids = Column(JSON, nullable=False)  # List of audit log IDs being anchored
	event_count = Column(Integer, nullable=False)
	event_hash = Column(String(64), nullable=False, index=True)  # Hash of events being anchored
	
	# Blockchain Transaction
	transaction_hash = Column(String(66), nullable=True, index=True)  # Blockchain transaction hash
	block_number = Column(Integer, nullable=True, index=True)
	block_hash = Column(String(66), nullable=True)
	transaction_index = Column(Integer, nullable=True)
	
	# Merkle Tree Structure
	merkle_root = Column(String(64), nullable=False, index=True)  # Merkle root of events
	merkle_tree_data = Column(JSON, nullable=True)  # Merkle tree structure for proofs
	proof_data = Column(JSON, nullable=True)  # Individual merkle proofs
	
	# Anchoring Status
	status = Column(String(20), default='pending', index=True)  # pending, submitted, confirmed, failed
	submitted_at = Column(DateTime, nullable=True)
	confirmed_at = Column(DateTime, nullable=True)
	confirmation_blocks = Column(Integer, default=0)
	
	# Gas and Cost Tracking
	gas_used = Column(Integer, nullable=True)
	gas_price = Column(Float, nullable=True)
	transaction_fee = Column(Float, nullable=True)
	
	# Verification
	verification_attempts = Column(Integer, default=0)
	last_verification = Column(DateTime, nullable=True)
	verification_status = Column(String(20), default='unverified')  # verified, unverified, failed
	
	# Error Handling
	error_message = Column(Text, nullable=True)
	retry_count = Column(Integer, default=0)
	max_retries = Column(Integer, default=3)
	
	# Relationships
	network = relationship("BSBlockchainNetwork", back_populates="audit_anchors")
	
	def __repr__(self):
		return f"<BSAuditAnchor {self.anchor_id} ({self.event_count} events, {self.status})>"
	
	def is_confirmed(self) -> bool:
		"""Check if anchor is confirmed on blockchain"""
		return (self.status == 'confirmed' and 
				self.transaction_hash is not None and
				self.confirmation_blocks >= self.network.confirmation_blocks)
	
	def calculate_merkle_root(self, event_hashes: List[str]) -> str:
		"""Calculate Merkle root from event hashes"""
		if not event_hashes:
			return ""
		
		# Simple Merkle tree implementation
		level = event_hashes[:]
		
		while len(level) > 1:
			next_level = []
			for i in range(0, len(level), 2):
				left = level[i]
				right = level[i + 1] if i + 1 < len(level) else level[i]
				
				# Combine hashes and hash again
				combined = left + right
				next_hash = hashlib.sha256(combined.encode()).hexdigest()
				next_level.append(next_hash)
			
			level = next_level
		
		return level[0]
	
	def generate_proof(self, target_hash: str) -> Optional[List[str]]:
		"""Generate Merkle proof for a specific event hash"""
		if not self.merkle_tree_data or target_hash not in self.merkle_tree_data.get('leaves', []):
			return None
		
		# Simplified proof generation - in practice would need full tree traversal
		proof = []
		# This would implement the actual Merkle proof generation algorithm
		return proof
	
	def verify_integrity(self) -> bool:
		"""Verify the integrity of the anchor"""
		if not self.is_confirmed():
			return False
		
		# Verify Merkle root matches calculated root
		if self.merkle_tree_data and 'leaves' in self.merkle_tree_data:
			calculated_root = self.calculate_merkle_root(self.merkle_tree_data['leaves'])
			return calculated_root == self.merkle_root
		
		return True
	
	def update_confirmation_status(self, current_block: int) -> None:
		"""Update confirmation status based on current block"""
		if self.block_number and current_block:
			self.confirmation_blocks = current_block - self.block_number
			
			if self.confirmation_blocks >= self.network.confirmation_blocks:
				self.status = 'confirmed'
				if not self.confirmed_at:
					self.confirmed_at = datetime.utcnow()


class BSDecentralizedIdentity(Model, AuditMixin, BaseMixin):
	"""
	Decentralized identity (DID) management and verification.
	
	Stores decentralized identities with verifiable credentials
	and blockchain-based identity verification.
	"""
	__tablename__ = 'bs_decentralized_identity'
	
	# Identity
	identity_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# DID Configuration
	did = Column(String(500), unique=True, nullable=False, index=True)  # Decentralized identifier
	did_method = Column(String(50), nullable=False)  # did:ethr, did:key, etc.
	did_document = Column(JSON, nullable=False)  # DID document with keys and services
	
	# Identity Subject
	subject_id = Column(String(36), nullable=True, index=True)  # Associated user/entity ID
	subject_type = Column(String(50), default='user')  # user, organization, service, device
	
	# Cryptographic Keys
	public_key = Column(Text, nullable=False)  # Primary public key
	key_type = Column(String(50), default='secp256k1')  # Key algorithm
	verification_methods = Column(JSON, default=list)  # Verification method references
	
	# Identity Status
	is_active = Column(Boolean, default=True, index=True)
	is_verified = Column(Boolean, default=False, index=True)
	verification_level = Column(String(20), default='basic')  # basic, enhanced, premium
	
	# Blockchain Registration
	registration_transaction = Column(String(66), nullable=True)  # Transaction hash
	registration_block = Column(Integer, nullable=True)
	registration_network = Column(String(50), nullable=True)
	
	# Recovery Configuration
	recovery_methods = Column(JSON, default=list)  # Recovery method configurations
	social_recovery_guardians = Column(JSON, default=list)  # Guardian addresses/DIDs
	recovery_threshold = Column(Integer, default=1)  # Minimum guardians for recovery
	
	# Usage Statistics
	credentials_issued = Column(Integer, default=0)
	credentials_verified = Column(Integer, default=0)
	last_activity = Column(DateTime, nullable=True, index=True)
	
	# Relationships
	credentials = relationship("BSVerifiableCredential", back_populates="identity", cascade="all, delete-orphan")
	
	def __repr__(self):
		return f"<BSDecentralizedIdentity {self.did}>"
	
	def is_valid(self) -> bool:
		"""Check if identity is valid and active"""
		return (self.is_active and 
				self.public_key is not None and
				self.did_document is not None)
	
	def add_verification_method(self, method_id: str, method_type: str, 
								public_key: str) -> None:
		"""Add verification method to DID document"""
		if not self.verification_methods:
			self.verification_methods = []
		
		method = {
			'id': f"{self.did}#{method_id}",
			'type': method_type,
			'controller': self.did,
			'publicKeyBase58': public_key
		}
		
		self.verification_methods.append(method)
		
		# Update DID document
		if 'verificationMethod' not in self.did_document:
			self.did_document['verificationMethod'] = []
		
		self.did_document['verificationMethod'].append(method)
	
	def add_service_endpoint(self, service_id: str, service_type: str, 
							 endpoint: str) -> None:
		"""Add service endpoint to DID document"""
		service = {
			'id': f"{self.did}#{service_id}",
			'type': service_type,
			'serviceEndpoint': endpoint
		}
		
		if 'service' not in self.did_document:
			self.did_document['service'] = []
		
		self.did_document['service'].append(service)
	
	def update_activity(self) -> None:
		"""Update last activity timestamp"""
		self.last_activity = datetime.utcnow()
	
	def can_recover(self, guardian_signatures: int) -> bool:
		"""Check if identity can be recovered with given guardian signatures"""
		return (guardian_signatures >= self.recovery_threshold and
				len(self.social_recovery_guardians) > 0)


class BSVerifiableCredential(Model, AuditMixin, BaseMixin):
	"""
	Verifiable credentials with cryptographic proofs.
	
	Stores verifiable credentials that can be cryptographically
	verified and linked to decentralized identities.
	"""
	__tablename__ = 'bs_verifiable_credential'
	
	# Identity
	credential_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	identity_id = Column(String(36), ForeignKey('bs_decentralized_identity.identity_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Credential Identity
	credential_type = Column(String(100), nullable=False, index=True)  # EmploymentCredential, etc.
	credential_schema = Column(String(500), nullable=True)  # Schema URL
	credential_context = Column(JSON, default=list)  # JSON-LD context
	
	# Issuance Information
	issuer_did = Column(String(500), nullable=False, index=True)  # Issuer's DID
	issuer_name = Column(String(200), nullable=True)
	issued_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
	expires_at = Column(DateTime, nullable=True, index=True)
	
	# Credential Subject and Claims
	subject_did = Column(String(500), nullable=False, index=True)  # Subject's DID
	credential_subject = Column(JSON, nullable=False)  # Claims about the subject
	evidence = Column(JSON, nullable=True)  # Supporting evidence
	
	# Cryptographic Proof
	proof_type = Column(String(50), nullable=False)  # Ed25519Signature2018, etc.
	proof_value = Column(Text, nullable=False)  # Cryptographic signature
	proof_purpose = Column(String(50), default='assertionMethod')
	verification_method = Column(String(500), nullable=False)  # Key used for signing
	
	# Credential Status
	status = Column(String(20), default='active', index=True)  # active, revoked, suspended, expired
	revocation_reason = Column(String(200), nullable=True)
	revoked_at = Column(DateTime, nullable=True, index=True)
	revoked_by = Column(String(500), nullable=True)  # Revoker's DID
	
	# Privacy and Selective Disclosure
	selective_disclosure = Column(Boolean, default=False)
	disclosed_attributes = Column(JSON, nullable=True)  # Attributes that can be selectively disclosed
	zero_knowledge_proofs = Column(JSON, nullable=True)  # ZK proof data
	
	# Usage Tracking
	verification_count = Column(Integer, default=0)
	last_verified = Column(DateTime, nullable=True, index=True)
	presentation_count = Column(Integer, default=0)  # Times presented/used
	
	# Blockchain Anchoring
	anchor_transaction = Column(String(66), nullable=True)  # Transaction hash if anchored
	anchor_block = Column(Integer, nullable=True)
	merkle_proof = Column(JSON, nullable=True)  # Merkle proof for batch anchoring
	
	# Relationships
	identity = relationship("BSDecentralizedIdentity", back_populates="credentials")
	
	def __repr__(self):
		return f"<BSVerifiableCredential {self.credential_type} for {self.subject_did}>"
	
	def is_valid(self) -> bool:
		"""Check if credential is valid (not expired or revoked)"""
		now = datetime.utcnow()
		
		if self.status != 'active':
			return False
		
		if self.expires_at and now > self.expires_at:
			return False
		
		return True
	
	def is_expired(self) -> bool:
		"""Check if credential has expired"""
		if not self.expires_at:
			return False
		return datetime.utcnow() > self.expires_at
	
	def revoke(self, reason: str, revoker_did: str) -> None:
		"""Revoke the credential"""
		self.status = 'revoked'
		self.revocation_reason = reason
		self.revoked_at = datetime.utcnow()
		self.revoked_by = revoker_did
	
	def verify_signature(self, public_key: str) -> bool:
		"""Verify credential signature (placeholder implementation)"""
		# This would implement actual cryptographic signature verification
		# using the appropriate algorithm based on proof_type
		return True  # Placeholder
	
	def create_presentation(self, disclosed_attrs: List[str] = None) -> Dict[str, Any]:
		"""Create a verifiable presentation of this credential"""
		presentation = {
			'@context': ['https://www.w3.org/2018/credentials/v1'],
			'type': ['VerifiablePresentation'],
			'verifiableCredential': [{
				'@context': self.credential_context,
				'type': ['VerifiableCredential', self.credential_type],
				'issuer': self.issuer_did,
				'issuanceDate': self.issued_at.isoformat(),
				'credentialSubject': self.credential_subject
			}]
		}
		
		if self.expires_at:
			presentation['verifiableCredential'][0]['expirationDate'] = self.expires_at.isoformat()
		
		# Add selective disclosure if specified
		if disclosed_attrs and self.selective_disclosure:
			filtered_subject = {
				attr: self.credential_subject[attr] 
				for attr in disclosed_attrs 
				if attr in self.credential_subject
			}
			presentation['verifiableCredential'][0]['credentialSubject'] = filtered_subject
		
		self.presentation_count += 1
		return presentation
	
	def update_verification_stats(self) -> None:
		"""Update verification statistics"""
		self.verification_count += 1
		self.last_verified = datetime.utcnow()


class BSSmartContract(Model, AuditMixin, BaseMixin):
	"""
	Smart contract deployment and management.
	
	Tracks deployed smart contracts with metadata, ABI,
	interaction history, and security analysis.
	"""
	__tablename__ = 'bs_smart_contract'
	
	# Identity
	contract_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	network_id = Column(String(36), ForeignKey('bs_blockchain_network.network_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Contract Identity
	name = Column(String(200), nullable=False)
	description = Column(Text, nullable=True)
	contract_type = Column(String(50), nullable=False, index=True)  # access_control, token, defi, etc.
	version = Column(String(20), default='1.0.0')
	
	# Deployment Information
	contract_address = Column(String(42), nullable=False, index=True)  # Ethereum address format
	deployment_transaction = Column(String(66), nullable=False, index=True)
	deployment_block = Column(Integer, nullable=False, index=True)
	deployed_by = Column(String(36), nullable=True, index=True)  # User ID
	deployed_at = Column(DateTime, nullable=False, default=datetime.utcnow)
	
	# Contract Code and Interface
	source_code = Column(Text, nullable=True)  # Solidity source code
	bytecode = Column(Text, nullable=False)  # Compiled bytecode
	abi = Column(JSON, nullable=False)  # Application Binary Interface
	compiler_version = Column(String(50), nullable=True)
	
	# Contract Configuration
	constructor_params = Column(JSON, nullable=True)  # Constructor parameters used
	is_upgradeable = Column(Boolean, default=False)
	proxy_pattern = Column(String(50), nullable=True)  # transparent, uups, beacon, etc.
	admin_addresses = Column(JSON, default=list)  # Admin/owner addresses
	
	# Security Analysis
	security_score = Column(Float, default=0.0)  # 0-100 security rating
	vulnerability_count = Column(Integer, default=0)
	last_security_scan = Column(DateTime, nullable=True)
	security_findings = Column(JSON, default=list)  # Security analysis results
	
	# Usage Statistics
	total_transactions = Column(Integer, default=0)
	total_gas_used = Column(Integer, default=0)
	unique_users = Column(Integer, default=0)
	last_interaction = Column(DateTime, nullable=True, index=True)
	
	# Financial Metrics
	total_value_locked = Column(Float, default=0.0)  # TVL for DeFi contracts
	total_fees_collected = Column(Float, default=0.0)
	transaction_volume = Column(Float, default=0.0)
	
	# Contract Status
	is_active = Column(Boolean, default=True, index=True)
	is_paused = Column(Boolean, default=False)
	is_verified = Column(Boolean, default=False, index=True)  # Source code verified
	
	# Monitoring and Alerts
	monitoring_enabled = Column(Boolean, default=True)
	alert_conditions = Column(JSON, default=list)  # Alert conditions
	last_alert = Column(DateTime, nullable=True)
	
	# Relationships
	network = relationship("BSBlockchainNetwork", back_populates="smart_contracts")
	transactions = relationship("BSTransaction", back_populates="smart_contract")
	
	def __repr__(self):
		return f"<BSSmartContract {self.name} at {self.contract_address}>"
	
	def is_operational(self) -> bool:
		"""Check if contract is operational"""
		return (self.is_active and 
				not self.is_paused and
				self.contract_address is not None)
	
	def get_function_signatures(self) -> List[str]:
		"""Get function signatures from ABI"""
		signatures = []
		
		for item in self.abi:
			if item.get('type') == 'function':
				name = item.get('name', '')
				inputs = item.get('inputs', [])
				input_types = [inp.get('type', '') for inp in inputs]
				signature = f"{name}({','.join(input_types)})"
				signatures.append(signature)
		
		return signatures
	
	def calculate_security_score(self) -> None:
		"""Calculate security score based on various factors"""
		base_score = 100.0
		
		# Deduct points for vulnerabilities
		base_score -= min(50.0, self.vulnerability_count * 10)
		
		# Deduct points if not verified
		if not self.is_verified:
			base_score -= 20.0
		
		# Deduct points if no recent security scan
		if self.last_security_scan:
			days_since_scan = (datetime.utcnow() - self.last_security_scan).days
			if days_since_scan > 90:  # More than 3 months
				base_score -= 15.0
		else:
			base_score -= 30.0  # Never scanned
		
		# Add points for upgradeable contracts (if properly implemented)
		if self.is_upgradeable and self.proxy_pattern:
			base_score += 5.0
		
		self.security_score = max(0.0, min(100.0, base_score))
	
	def update_usage_stats(self, gas_used: int, user_address: str, 
						   transaction_value: float = 0.0) -> None:
		"""Update contract usage statistics"""
		self.total_transactions += 1
		self.total_gas_used += gas_used
		self.last_interaction = datetime.utcnow()
		self.transaction_volume += transaction_value
		
		# Note: In a real implementation, you'd track unique users properly
		# This is a simplified version
	
	def add_security_finding(self, severity: str, description: str, 
							 recommendation: str = None) -> None:
		"""Add security finding"""
		finding = {
			'id': uuid7str(),
			'severity': severity,
			'description': description,
			'recommendation': recommendation,
			'detected_at': datetime.utcnow().isoformat()
		}
		
		if not self.security_findings:
			self.security_findings = []
		
		self.security_findings.append(finding)
		
		if severity in ['high', 'critical']:
			self.vulnerability_count += 1
		
		# Recalculate security score
		self.calculate_security_score()


class BSTransaction(Model, AuditMixin, BaseMixin):
	"""
	Blockchain transaction tracking and analysis.
	
	Records blockchain transactions with detailed metadata,
	performance metrics, and business context.
	"""
	__tablename__ = 'bs_transaction'
	
	# Identity
	transaction_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	network_id = Column(String(36), ForeignKey('bs_blockchain_network.network_id'), nullable=False, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Transaction Identity
	transaction_hash = Column(String(66), unique=True, nullable=False, index=True)
	block_number = Column(Integer, nullable=True, index=True)
	block_hash = Column(String(66), nullable=True)
	transaction_index = Column(Integer, nullable=True)
	
	# Transaction Details
	from_address = Column(String(42), nullable=False, index=True)
	to_address = Column(String(42), nullable=True, index=True)
	value = Column(String(100), default='0')  # Wei amount as string for precision
	gas_limit = Column(Integer, nullable=False)
	gas_used = Column(Integer, nullable=True)
	gas_price = Column(String(100), nullable=False)  # Wei as string
	nonce = Column(Integer, nullable=False)
	
	# Transaction Type and Purpose
	transaction_type = Column(String(50), nullable=False, index=True)  # transfer, contract_call, deployment
	business_purpose = Column(String(100), nullable=True, index=True)  # audit_anchor, payment, etc.
	smart_contract_id = Column(String(36), ForeignKey('bs_smart_contract.contract_id'), nullable=True, index=True)
	
	# Contract Interaction
	input_data = Column(Text, nullable=True)  # Transaction input data
	function_signature = Column(String(100), nullable=True)  # Called function
	function_parameters = Column(JSON, nullable=True)  # Decoded parameters
	
	# Transaction Status
	status = Column(String(20), default='pending', index=True)  # pending, confirmed, failed
	confirmation_blocks = Column(Integer, default=0)
	confirmed_at = Column(DateTime, nullable=True, index=True)
	
	# Business Context
	initiated_by = Column(String(36), nullable=True, index=True)  # User ID
	business_reference = Column(String(100), nullable=True, index=True)  # Business ref ID
	metadata = Column(JSON, default=dict)  # Additional business metadata
	
	# Performance Metrics
	submission_time = Column(DateTime, nullable=False, default=datetime.utcnow)
	confirmation_time = Column(DateTime, nullable=True)
	total_confirmation_duration = Column(Float, nullable=True)  # seconds
	
	# Cost Analysis
	transaction_fee = Column(Float, nullable=True)  # Calculated fee in network currency
	usd_fee = Column(Float, nullable=True)  # Fee in USD at time of transaction
	fee_efficiency_score = Column(Float, nullable=True)  # How efficient the fee was
	
	# Error Handling
	error_message = Column(Text, nullable=True)
	revert_reason = Column(String(500), nullable=True)  # Smart contract revert reason
	retry_count = Column(Integer, default=0)
	
	# Relationships
	network = relationship("BSBlockchainNetwork", back_populates="transactions")
	smart_contract = relationship("BSSmartContract", back_populates="transactions")
	
	def __repr__(self):
		return f"<BSTransaction {self.transaction_hash} ({self.status})>"
	
	def is_confirmed(self) -> bool:
		"""Check if transaction is confirmed"""
		return (self.status == 'confirmed' and 
				self.confirmation_blocks >= self.network.confirmation_blocks)
	
	def is_successful(self) -> bool:
		"""Check if transaction was successful"""
		return self.status == 'confirmed' and not self.error_message
	
	def calculate_confirmation_duration(self) -> None:
		"""Calculate confirmation duration"""
		if self.confirmation_time and self.submission_time:
			duration = (self.confirmation_time - self.submission_time).total_seconds()
			self.total_confirmation_duration = duration
	
	def calculate_transaction_fee(self, native_token_price_usd: float = None) -> None:
		"""Calculate transaction fee in network currency and USD"""
		if self.gas_used and self.gas_price:
			# Convert gas_price from string to int (Wei)
			gas_price_wei = int(self.gas_price)
			total_wei = self.gas_used * gas_price_wei
			
			# Convert to network currency (e.g., ETH)
			if self.network.network_key in ['ethereum', 'polygon']:
				self.transaction_fee = total_wei / 1e18  # Wei to ETH/MATIC
			elif self.network.network_key == 'bitcoin':
				self.transaction_fee = total_wei / 1e8   # Satoshi to BTC
			else:
				self.transaction_fee = total_wei / 1e18  # Default
			
			# Calculate USD fee if price provided
			if native_token_price_usd:
				self.usd_fee = self.transaction_fee * native_token_price_usd
	
	def update_confirmation_status(self, current_block: int) -> None:
		"""Update confirmation status based on current block"""
		if self.block_number and current_block:
			self.confirmation_blocks = current_block - self.block_number
			
			if self.confirmation_blocks >= self.network.confirmation_blocks and self.status == 'pending':
				self.status = 'confirmed'
				self.confirmed_at = datetime.utcnow()
				self.calculate_confirmation_duration()
	
	def decode_function_call(self) -> Dict[str, Any]:
		"""Decode function call from input data"""
		if not self.input_data or not self.smart_contract:
			return {}
		
		# This would implement actual ABI decoding
		# Simplified placeholder implementation
		decoded = {
			'function': self.function_signature,
			'parameters': self.function_parameters or {}
		}
		
		return decoded
	
	def calculate_fee_efficiency(self, network_average_gas_price: float) -> None:
		"""Calculate fee efficiency compared to network average"""
		if not self.gas_price or not network_average_gas_price:
			return
		
		gas_price_gwei = int(self.gas_price) / 1e9  # Convert Wei to Gwei
		efficiency_ratio = network_average_gas_price / gas_price_gwei
		
		# Efficiency score: >1 means overpaid, <1 means efficient
		if efficiency_ratio > 1.2:
			self.fee_efficiency_score = 100  # Very efficient
		elif efficiency_ratio > 1.0:
			self.fee_efficiency_score = 80   # Efficient
		elif efficiency_ratio > 0.8:
			self.fee_efficiency_score = 60   # Average
		elif efficiency_ratio > 0.6:
			self.fee_efficiency_score = 40   # Expensive
		else:
			self.fee_efficiency_score = 20   # Very expensive


class BSCryptocurrencyWallet(Model, AuditMixin, BaseMixin):
	"""
	Cryptocurrency wallet management and monitoring.
	
	Manages cryptocurrency wallets with balance tracking,
	transaction monitoring, and security configuration.
	"""
	__tablename__ = 'bs_cryptocurrency_wallet'
	
	# Identity
	wallet_id = Column(String(36), unique=True, nullable=False, default=uuid7str, index=True)
	tenant_id = Column(String(36), nullable=False, index=True)
	
	# Wallet Identity
	name = Column(String(200), nullable=False)
	wallet_type = Column(String(50), nullable=False, index=True)  # hot, cold, multisig, hardware
	purpose = Column(String(100), nullable=True)  # treasury, operational, personal, etc.
	
	# Wallet Addresses (multiple networks)
	addresses = Column(JSON, nullable=False)  # Network -> address mapping
	primary_network = Column(String(50), nullable=False, index=True)
	supported_networks = Column(JSON, default=list)
	
	# Multi-signature Configuration
	is_multisig = Column(Boolean, default=False)
	required_signatures = Column(Integer, default=1)
	total_signers = Column(Integer, default=1)
	signer_addresses = Column(JSON, default=list)
	
	# Security Configuration
	security_level = Column(String(20), default='standard')  # basic, standard, high, maximum
	encryption_enabled = Column(Boolean, default=True)
	backup_configured = Column(Boolean, default=False)
	recovery_configured = Column(Boolean, default=False)
	
	# Access Control
	authorized_users = Column(JSON, default=list)  # User IDs with access
	spending_limits = Column(JSON, default=dict)  # Per-user spending limits
	approval_required = Column(Boolean, default=True)
	time_locks = Column(JSON, default=dict)  # Time-locked transactions
	
	# Balance Information (cached)
	balances = Column(JSON, default=dict)  # Network -> token -> amount
	total_value_usd = Column(Float, default=0.0)
	last_balance_update = Column(DateTime, nullable=True, index=True)
	
	# Transaction Statistics
	total_transactions_sent = Column(Integer, default=0)
	total_transactions_received = Column(Integer, default=0)
	total_volume_sent = Column(Float, default=0.0)
	total_volume_received = Column(Float, default=0.0)
	
	# Wallet Status
	is_active = Column(Boolean, default=True, index=True)
	is_monitored = Column(Boolean, default=True)
	last_activity = Column(DateTime, nullable=True, index=True)
	
	# Risk Management
	risk_score = Column(Float, default=0.0)  # 0-100 risk rating
	suspicious_activity_detected = Column(Boolean, default=False)
	compliance_status = Column(String(20), default='compliant')  # compliant, flagged, blocked
	
	def __repr__(self):
		return f"<BSCryptocurrencyWallet {self.name} ({self.wallet_type})>"
	
	def get_address(self, network: str) -> Optional[str]:
		"""Get wallet address for specific network"""
		return self.addresses.get(network)
	
	def get_balance(self, network: str, token: str = 'native') -> float:
		"""Get balance for specific network and token"""
		network_balances = self.balances.get(network, {})
		return float(network_balances.get(token, 0.0))
	
	def update_balance(self, network: str, token: str, amount: float) -> None:
		"""Update balance for specific network and token"""
		if network not in self.balances:
			self.balances[network] = {}
		
		self.balances[network][token] = amount
		self.last_balance_update = datetime.utcnow()
	
	def calculate_total_value(self, token_prices: Dict[str, float]) -> None:
		"""Calculate total wallet value in USD"""
		total_value = 0.0
		
		for network, network_balances in self.balances.items():
			for token, amount in network_balances.items():
				price_key = f"{network}_{token}" if token != 'native' else network
				token_price = token_prices.get(price_key, 0.0)
				total_value += amount * token_price
		
		self.total_value_usd = total_value
	
	def requires_multisig_approval(self, transaction_amount: float) -> bool:
		"""Check if transaction requires multi-signature approval"""
		if not self.is_multisig:
			return False
		
		# Always require multisig for multisig wallets above certain threshold
		if transaction_amount > 1000:  # $1000 threshold
			return True
		
		return self.approval_required
	
	def is_spending_allowed(self, user_id: str, amount: float) -> bool:
		"""Check if user is allowed to spend given amount"""
		if user_id not in self.authorized_users:
			return False
		
		user_limits = self.spending_limits.get(user_id, {})
		daily_limit = user_limits.get('daily_limit')
		
		if daily_limit and amount > daily_limit:
			return False
		
		return True
	
	def update_transaction_stats(self, amount: float, is_outgoing: bool) -> None:
		"""Update transaction statistics"""
		if is_outgoing:
			self.total_transactions_sent += 1
			self.total_volume_sent += amount
		else:
			self.total_transactions_received += 1
			self.total_volume_received += amount
		
		self.last_activity = datetime.utcnow()
	
	def calculate_risk_score(self) -> None:
		"""Calculate wallet risk score based on various factors"""
		risk_score = 0.0
		
		# Base risk based on wallet type
		if self.wallet_type == 'hot':
			risk_score += 30
		elif self.wallet_type == 'cold':
			risk_score += 5
		elif self.wallet_type == 'multisig':
			risk_score += 10
		
		# Risk based on security configuration
		if not self.encryption_enabled:
			risk_score += 20
		
		if not self.backup_configured:
			risk_score += 15
		
		if not self.recovery_configured:
			risk_score += 10
		
		# Risk based on activity
		if self.suspicious_activity_detected:
			risk_score += 40
		
		if self.total_value_usd > 100000:  # High value wallet
			risk_score += 10
		
		# Risk based on compliance
		if self.compliance_status == 'flagged':
			risk_score += 25
		elif self.compliance_status == 'blocked':
			risk_score += 50
		
		self.risk_score = min(100.0, risk_score)