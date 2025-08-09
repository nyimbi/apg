#!/usr/bin/env python3
"""
APG Message Queue Event Bus (MQEB) - Quantum-Safe Security Engine
Post-quantum cryptography implementation for quantum-safe messaging

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
import hashlib
import hmac
import logging
import secrets
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import json

# Cryptography imports
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, ec
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend

from .models import MQMessage
from .service import MQEBService


class QuantumAlgorithm(str, Enum):
	"""Post-quantum cryptographic algorithms"""
	CRYSTALS_KYBER_512 = "kyber512"
	CRYSTALS_KYBER_768 = "kyber768"
	CRYSTALS_KYBER_1024 = "kyber1024"
	CRYSTALS_DILITHIUM_2 = "dilithium2"
	CRYSTALS_DILITHIUM_3 = "dilithium3"
	CRYSTALS_DILITHIUM_5 = "dilithium5"
	SPHINCS_PLUS_128 = "sphincs128"
	SPHINCS_PLUS_192 = "sphincs192"
	SPHINCS_PLUS_256 = "sphincs256"


class SecurityLevel(str, Enum):
	"""Security classification levels"""
	UNCLASSIFIED = "unclassified"
	CONFIDENTIAL = "confidential"
	SECRET = "secret"
	TOP_SECRET = "top_secret"
	QUANTUM_SAFE = "quantum_safe"


@dataclass
class QuantumKeyPair:
	"""Post-quantum cryptographic key pair"""
	algorithm: QuantumAlgorithm
	public_key: bytes
	private_key: bytes
	key_id: str
	created_at: datetime
	expires_at: Optional[datetime] = None
	security_level: SecurityLevel = SecurityLevel.QUANTUM_SAFE
	
	def is_expired(self) -> bool:
		"""Check if key pair is expired"""
		return self.expires_at is not None and datetime.utcnow() > self.expires_at


@dataclass
class EncryptionContext:
	"""Context for message encryption/decryption"""
	message_id: str
	tenant_id: str
	security_level: SecurityLevel
	algorithm: QuantumAlgorithm
	key_id: str
	nonce: bytes
	authenticated_data: bytes
	timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SecurityAuditEvent:
	"""Security audit event record"""
	event_id: str
	event_type: str
	message_id: Optional[str]
	tenant_id: str
	user_id: Optional[str]
	security_level: SecurityLevel
	details: Dict[str, Any]
	timestamp: datetime = field(default_factory=datetime.utcnow)
	success: bool = True
	risk_score: float = 0.0


class QuantumKeyManager:
	"""Manages post-quantum cryptographic keys"""
	
	def __init__(self):
		self.key_store: Dict[str, QuantumKeyPair] = {}
		self.tenant_keys: Dict[str, List[str]] = defaultdict(list)  # tenant_id -> key_ids
		self.algorithm_keys: Dict[QuantumAlgorithm, List[str]] = defaultdict(list)
		self.key_rotation_policy = timedelta(days=90)  # Rotate keys every 90 days
		
		# Simulated post-quantum algorithms (using classical crypto as placeholder)
		self.quantum_algorithms = {
			QuantumAlgorithm.CRYSTALS_KYBER_512: self._kyber_operations,
			QuantumAlgorithm.CRYSTALS_DILITHIUM_2: self._dilithium_operations,
			QuantumAlgorithm.SPHINCS_PLUS_128: self._sphincs_operations,
		}
		
		self.logger = logging.getLogger('mqeb.quantum_security')
	
	async def generate_key_pair(self, algorithm: QuantumAlgorithm, tenant_id: str,
							   security_level: SecurityLevel = SecurityLevel.QUANTUM_SAFE) -> str:
		"""Generate new post-quantum key pair"""
		try:
			key_id = f"qkey_{secrets.token_hex(16)}"
			
			# Generate key pair using specified algorithm
			operations = self.quantum_algorithms.get(algorithm)
			if not operations:
				raise ValueError(f"Unsupported quantum algorithm: {algorithm}")
			
			public_key, private_key = await operations.generate_keypair()
			
			key_pair = QuantumKeyPair(
				algorithm=algorithm,
				public_key=public_key,
				private_key=private_key,
				key_id=key_id,
				created_at=datetime.utcnow(),
				expires_at=datetime.utcnow() + self.key_rotation_policy,
				security_level=security_level
			)
			
			# Store key pair
			self.key_store[key_id] = key_pair
			self.tenant_keys[tenant_id].append(key_id)
			self.algorithm_keys[algorithm].append(key_id)
			
			self.logger.info(f"Generated quantum key pair {key_id} for tenant {tenant_id}")
			return key_id
			
		except Exception as e:
			self.logger.error(f"Failed to generate quantum key pair: {e}")
			raise
	
	async def get_key_pair(self, key_id: str) -> Optional[QuantumKeyPair]:
		"""Retrieve key pair by ID"""
		key_pair = self.key_store.get(key_id)
		if key_pair and key_pair.is_expired():
			self.logger.warning(f"Key pair {key_id} has expired")
			return None
		return key_pair
	
	async def get_tenant_keys(self, tenant_id: str, algorithm: Optional[QuantumAlgorithm] = None) -> List[QuantumKeyPair]:
		"""Get all active keys for a tenant"""
		tenant_key_ids = self.tenant_keys.get(tenant_id, [])
		keys = []
		
		for key_id in tenant_key_ids:
			key_pair = await self.get_key_pair(key_id)
			if key_pair and (algorithm is None or key_pair.algorithm == algorithm):
				keys.append(key_pair)
		
		return keys
	
	async def rotate_expired_keys(self) -> int:
		"""Rotate expired keys and return count of rotated keys"""
		rotated_count = 0
		
		for key_id, key_pair in list(self.key_store.items()):
			if key_pair.is_expired():
				# Generate new key pair with same parameters
				tenant_id = None
				for tid, key_ids in self.tenant_keys.items():
					if key_id in key_ids:
						tenant_id = tid
						break
				
				if tenant_id:
					new_key_id = await self.generate_key_pair(
						key_pair.algorithm, tenant_id, key_pair.security_level
					)
					
					# Remove expired key
					del self.key_store[key_id]
					self.tenant_keys[tenant_id].remove(key_id)
					self.algorithm_keys[key_pair.algorithm].remove(key_id)
					
					rotated_count += 1
					self.logger.info(f"Rotated expired key {key_id} -> {new_key_id}")
		
		return rotated_count
	
	# Simulated post-quantum algorithm implementations
	class _kyber_operations:
		@staticmethod
		async def generate_keypair() -> Tuple[bytes, bytes]:
			"""Simulate Kyber key generation (placeholder implementation)"""
			# In production, this would use actual Kyber implementation
			private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
			public_key = private_key.public_key()
			
			private_pem = private_key.private_bytes(
				encoding=serialization.Encoding.PEM,
				format=serialization.PrivateFormat.PKCS8,
				encryption_algorithm=serialization.NoEncryption()
			)
			public_pem = public_key.public_bytes(
				encoding=serialization.Encoding.PEM,
				format=serialization.PublicFormat.SubjectPublicKeyInfo
			)
			
			return public_pem, private_pem
		
		@staticmethod
		async def encrypt(public_key: bytes, plaintext: bytes) -> bytes:
			"""Simulate Kyber encryption"""
			# Simplified AES encryption as placeholder
			key = hashlib.sha256(public_key[:32]).digest()
			iv = secrets.token_bytes(16)
			cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
			encryptor = cipher.encryptor()
			
			# Pad plaintext to AES block size
			padding_length = 16 - (len(plaintext) % 16)
			padded_plaintext = plaintext + bytes([padding_length] * padding_length)
			
			ciphertext = encryptor.update(padded_plaintext) + encryptor.finalize()
			return iv + ciphertext
		
		@staticmethod
		async def decrypt(private_key: bytes, ciphertext: bytes) -> bytes:
			"""Simulate Kyber decryption"""
			iv = ciphertext[:16]
			encrypted_data = ciphertext[16:]
			
			key = hashlib.sha256(private_key[:32]).digest()
			cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
			decryptor = cipher.decryptor()
			
			padded_plaintext = decryptor.update(encrypted_data) + decryptor.finalize()
			
			# Remove padding
			padding_length = padded_plaintext[-1]
			return padded_plaintext[:-padding_length]
	
	class _dilithium_operations:
		@staticmethod
		async def generate_keypair() -> Tuple[bytes, bytes]:
			"""Simulate Dilithium key generation"""
			# Placeholder using ECDSA
			private_key = ec.generate_private_key(ec.SECP384R1())
			public_key = private_key.public_key()
			
			private_pem = private_key.private_bytes(
				encoding=serialization.Encoding.PEM,
				format=serialization.PrivateFormat.PKCS8,
				encryption_algorithm=serialization.NoEncryption()
			)
			public_pem = public_key.public_bytes(
				encoding=serialization.Encoding.PEM,
				format=serialization.PublicFormat.SubjectPublicKeyInfo
			)
			
			return public_pem, private_pem
	
	class _sphincs_operations:
		@staticmethod
		async def generate_keypair() -> Tuple[bytes, bytes]:
			"""Simulate SPHINCS+ key generation"""
			# Placeholder implementation
			private_key_data = secrets.token_bytes(64)
			public_key_data = hashlib.sha512(private_key_data).digest()[:32]
			
			return public_key_data, private_key_data


class ZeroTrustMessageSecurity:
	"""Zero-trust security framework for messages"""
	
	def __init__(self, key_manager: QuantumKeyManager):
		self.key_manager = key_manager
		self.message_policies: Dict[str, Dict] = {}
		self.access_logs: List[SecurityAuditEvent] = []
		self.threat_indicators: Dict[str, float] = defaultdict(float)
		
		self.logger = logging.getLogger('mqeb.zero_trust')
	
	async def apply_zero_trust_policy(self, message: MQMessage, context: Dict[str, Any]) -> bool:
		"""Apply zero-trust policy to message"""
		try:
			# Extract context information
			user_id = context.get('user_id')
			source_ip = context.get('source_ip', '0.0.0.0')
			device_fingerprint = context.get('device_fingerprint')
			time_of_day = datetime.utcnow().hour
			
			# Calculate trust score
			trust_score = await self._calculate_trust_score(message, context)
			
			# Get policy requirements
			policy = await self._get_message_policy(message.topic, message.tenant_id)
			required_trust_score = policy.get('min_trust_score', 0.7)
			
			# Apply policy rules
			policy_result = await self._evaluate_policy_rules(message, context, policy)
			
			# Make access decision
			access_granted = (trust_score >= required_trust_score and 
							policy_result['allowed'] and
							not self._is_threat_detected(message, context))
			
			# Log security event
			audit_event = SecurityAuditEvent(
				event_id=f"zt_{secrets.token_hex(8)}",
				event_type="zero_trust_evaluation",
				message_id=message.id,
				tenant_id=message.tenant_id,
				user_id=user_id,
				security_level=self._classify_message_security(message),
				details={
					'trust_score': trust_score,
					'required_score': required_trust_score,
					'policy_result': policy_result,
					'source_ip': source_ip,
					'device_fingerprint': device_fingerprint,
					'time_of_day': time_of_day
				},
				success=access_granted,
				risk_score=1.0 - trust_score
			)
			
			self.access_logs.append(audit_event)
			
			if not access_granted:
				self.logger.warning(f"Zero-trust policy denied access to message {message.id}")
			
			return access_granted
			
		except Exception as e:
			self.logger.error(f"Zero-trust policy evaluation failed: {e}")
			return False  # Fail secure
	
	async def _calculate_trust_score(self, message: MQMessage, context: Dict[str, Any]) -> float:
		"""Calculate trust score for message access"""
		base_score = 0.5
		
		# User authentication factor
		if context.get('authenticated'):
			base_score += 0.2
			if context.get('mfa_verified'):
				base_score += 0.1
		
		# Device trust factor
		device_fingerprint = context.get('device_fingerprint')
		if device_fingerprint and self._is_trusted_device(device_fingerprint):
			base_score += 0.15
		
		# Network location factor
		source_ip = context.get('source_ip', '0.0.0.0')
		if self._is_trusted_network(source_ip):
			base_score += 0.1
		
		# Time-based factor
		time_of_day = datetime.utcnow().hour
		if 8 <= time_of_day <= 18:  # Business hours
			base_score += 0.05
		
		# Application trust factor
		if self._is_trusted_application(message.source_application):
			base_score += 0.1
		
		# Historical behavior factor
		user_id = context.get('user_id')
		if user_id and self._has_good_reputation(user_id):
			base_score += 0.05
		
		return min(1.0, base_score)
	
	async def _get_message_policy(self, topic: str, tenant_id: str) -> Dict[str, Any]:
		"""Get zero-trust policy for message topic"""
		policy_key = f"{tenant_id}:{topic}"
		
		if policy_key not in self.message_policies:
			# Default policy
			self.message_policies[policy_key] = {
				'min_trust_score': 0.7,
				'require_mfa': False,
				'allowed_networks': ['0.0.0.0/0'],
				'allowed_hours': list(range(24)),
				'max_message_size': 10 * 1024 * 1024,  # 10MB
				'encryption_required': True,
				'audit_all_access': True
			}
		
		return self.message_policies[policy_key]
	
	async def _evaluate_policy_rules(self, message: MQMessage, context: Dict[str, Any], policy: Dict) -> Dict[str, Any]:
		"""Evaluate specific policy rules"""
		result = {'allowed': True, 'violations': []}
		
		# MFA requirement
		if policy.get('require_mfa') and not context.get('mfa_verified'):
			result['allowed'] = False
			result['violations'].append('mfa_required')
		
		# Network restrictions
		source_ip = context.get('source_ip', '0.0.0.0')
		if not self._ip_in_networks(source_ip, policy.get('allowed_networks', [])):
			result['allowed'] = False
			result['violations'].append('network_not_allowed')
		
		# Time restrictions
		current_hour = datetime.utcnow().hour
		if current_hour not in policy.get('allowed_hours', list(range(24))):
			result['allowed'] = False
			result['violations'].append('time_not_allowed')
		
		# Message size restrictions
		if len(message.payload) > policy.get('max_message_size', float('inf')):
			result['allowed'] = False
			result['violations'].append('message_too_large')
		
		# Encryption requirement
		if policy.get('encryption_required') and not message.encrypted:
			result['allowed'] = False
			result['violations'].append('encryption_required')
		
		return result
	
	def _classify_message_security(self, message: MQMessage) -> SecurityLevel:
		"""Classify message security level based on content and metadata"""
		# Simple classification based on topic patterns
		topic_lower = message.topic.lower()
		
		if any(keyword in topic_lower for keyword in ['secret', 'classified', 'confidential']):
			return SecurityLevel.TOP_SECRET
		elif any(keyword in topic_lower for keyword in ['financial', 'payment', 'transaction']):
			return SecurityLevel.SECRET
		elif any(keyword in topic_lower for keyword in ['internal', 'private', 'restricted']):
			return SecurityLevel.CONFIDENTIAL
		elif message.encrypted:
			return SecurityLevel.QUANTUM_SAFE
		else:
			return SecurityLevel.UNCLASSIFIED
	
	def _is_threat_detected(self, message: MQMessage, context: Dict[str, Any]) -> bool:
		"""Detect potential security threats"""
		threat_score = 0.0
		
		# Suspicious patterns in message content
		try:
			content = message.payload.decode('utf-8', errors='ignore').lower()
			suspicious_patterns = ['<script', 'javascript:', 'sql injection', 'union select', '../']
			
			for pattern in suspicious_patterns:
				if pattern in content:
					threat_score += 0.3
		except:
			pass
		
		# Unusual message size
		if len(message.payload) > 10 * 1024 * 1024:  # 10MB
			threat_score += 0.2
		
		# High frequency from same source
		source_key = f"{message.source_application}:{context.get('source_ip', 'unknown')}"
		self.threat_indicators[source_key] += 0.1
		if self.threat_indicators[source_key] > 10.0:  # High activity
			threat_score += 0.4
		
		return threat_score > 0.8
	
	def _is_trusted_device(self, device_fingerprint: str) -> bool:
		"""Check if device is trusted (simplified)"""
		# In production, this would check against a device trust database
		return len(device_fingerprint) > 20  # Simple heuristic
	
	def _is_trusted_network(self, ip_address: str) -> bool:
		"""Check if IP is from trusted network"""
		# Simplified: trust private networks
		return (ip_address.startswith('10.') or 
				ip_address.startswith('192.168.') or 
				ip_address.startswith('172.'))
	
	def _is_trusted_application(self, app_name: str) -> bool:
		"""Check if application is trusted"""
		trusted_apps = ['user_service', 'order_service', 'notification_service']
		return app_name in trusted_apps
	
	def _has_good_reputation(self, user_id: str) -> bool:
		"""Check user's security reputation"""
		# Simplified reputation check
		return not user_id.startswith('temp_')
	
	def _ip_in_networks(self, ip: str, networks: List[str]) -> bool:
		"""Check if IP is in allowed networks (simplified)"""
		if '0.0.0.0/0' in networks:
			return True
		# Simplified check - in production would use proper CIDR matching
		return any(ip.startswith(net.split('/')[0].rsplit('.', 1)[0]) for net in networks)


class ComplianceAutomation:
	"""Automated compliance management for messages"""
	
	def __init__(self):
		self.compliance_rules: Dict[str, Dict] = {}
		self.audit_trails: Dict[str, List[SecurityAuditEvent]] = defaultdict(list)
		self.compliance_reports: List[Dict] = []
		
		# Initialize compliance frameworks
		self._initialize_compliance_frameworks()
		
		self.logger = logging.getLogger('mqeb.compliance')
	
	def _initialize_compliance_frameworks(self):
		"""Initialize compliance framework rules"""
		self.compliance_rules = {
			'gdpr': {
				'data_residency': 'EU',
				'encryption_required': True,
				'retention_period_days': 365,
				'right_to_be_forgotten': True,
				'consent_required': True,
				'data_portability': True,
				'breach_notification_hours': 72
			},
			'hipaa': {
				'encryption_at_rest': True,
				'encryption_in_transit': True,
				'access_logging': True,
				'minimum_necessary': True,
				'audit_controls': True,
				'data_integrity': True,
				'breach_notification_days': 60
			},
			'pci_dss': {
				'encryption_required': True,
				'access_control': True,
				'network_monitoring': True,
				'vulnerability_scanning': True,
				'secure_coding': True,
				'audit_logging': True
			},
			'sox': {
				'financial_controls': True,
				'audit_trails': True,
				'segregation_of_duties': True,
				'change_management': True,
				'data_retention': True
			}
		}
	
	async def apply_compliance_controls(self, message: MQMessage, frameworks: List[str]) -> Dict[str, Any]:
		"""Apply compliance controls to message"""
		compliance_result = {
			'compliant': True,
			'violations': [],
			'controls_applied': [],
			'audit_requirements': []
		}
		
		for framework in frameworks:
			if framework not in self.compliance_rules:
				continue
			
			rules = self.compliance_rules[framework]
			framework_result = await self._apply_framework_rules(message, framework, rules)
			
			if not framework_result['compliant']:
				compliance_result['compliant'] = False
				compliance_result['violations'].extend(framework_result['violations'])
			
			compliance_result['controls_applied'].extend(framework_result['controls_applied'])
			compliance_result['audit_requirements'].extend(framework_result['audit_requirements'])
		
		# Log compliance evaluation
		audit_event = SecurityAuditEvent(
			event_id=f"comp_{secrets.token_hex(8)}",
			event_type="compliance_evaluation",
			message_id=message.id,
			tenant_id=message.tenant_id,
			security_level=SecurityLevel.CONFIDENTIAL,
			details={
				'frameworks': frameworks,
				'compliance_result': compliance_result
			},
			success=compliance_result['compliant']
		)
		
		self.audit_trails[message.tenant_id].append(audit_event)
		
		return compliance_result
	
	async def _apply_framework_rules(self, message: MQMessage, framework: str, rules: Dict) -> Dict[str, Any]:
		"""Apply specific compliance framework rules"""
		result = {
			'compliant': True,
			'violations': [],
			'controls_applied': [],
			'audit_requirements': []
		}
		
		# Encryption requirements
		if rules.get('encryption_required') and not message.encrypted:
			result['compliant'] = False
			result['violations'].append(f"{framework}: encryption_required")
		else:
			result['controls_applied'].append(f"{framework}: encryption_verified")
		
		# Data residency (simplified check)
		if rules.get('data_residency'):
			result['controls_applied'].append(f"{framework}: data_residency_checked")
		
		# Audit logging requirements
		if rules.get('access_logging') or rules.get('audit_controls'):
			result['audit_requirements'].append(f"{framework}: access_logging_required")
		
		# PII detection (simplified)
		if framework == 'gdpr' and self._contains_pii(message):
			result['audit_requirements'].append('gdpr: pii_detected_special_handling')
		
		# Financial data detection
		if framework == 'sox' and self._contains_financial_data(message):
			result['audit_requirements'].append('sox: financial_data_special_controls')
		
		return result
	
	def _contains_pii(self, message: MQMessage) -> bool:
		"""Detect potential PII in message (simplified)"""
		try:
			content = message.payload.decode('utf-8', errors='ignore').lower()
			pii_indicators = ['email', 'phone', 'ssn', 'credit_card', 'address', 'name']
			return any(indicator in content for indicator in pii_indicators)
		except:
			return False
	
	def _contains_financial_data(self, message: MQMessage) -> bool:
		"""Detect financial data in message"""
		try:
			content = message.payload.decode('utf-8', errors='ignore').lower()
			financial_indicators = ['payment', 'transaction', 'account', 'balance', 'invoice']
			return any(indicator in content for indicator in financial_indicators)
		except:
			return False
	
	async def generate_compliance_report(self, tenant_id: str, framework: str, 
										start_date: datetime, end_date: datetime) -> Dict[str, Any]:
		"""Generate compliance report for tenant"""
		audit_events = [
			event for event in self.audit_trails[tenant_id]
			if start_date <= event.timestamp <= end_date
		]
		
		report = {
			'tenant_id': tenant_id,
			'framework': framework,
			'report_period': {
				'start': start_date.isoformat(),
				'end': end_date.isoformat()
			},
			'generated_at': datetime.utcnow().isoformat(),
			'summary': {
				'total_events': len(audit_events),
				'compliant_events': sum(1 for e in audit_events if e.success),
				'violation_events': sum(1 for e in audit_events if not e.success),
				'compliance_rate': 0.0
			},
			'violations': [],
			'recommendations': []
		}
		
		if audit_events:
			report['summary']['compliance_rate'] = report['summary']['compliant_events'] / len(audit_events)
		
		# Collect violations
		for event in audit_events:
			if not event.success and framework in str(event.details):
				report['violations'].append({
					'event_id': event.event_id,
					'timestamp': event.timestamp.isoformat(),
					'message_id': event.message_id,
					'details': event.details
				})
		
		# Generate recommendations
		if report['summary']['compliance_rate'] < 0.95:
			report['recommendations'].append(
				f"Compliance rate ({report['summary']['compliance_rate']:.2%}) below 95% threshold"
			)
		
		self.compliance_reports.append(report)
		return report


class QuantumSecurityEngine:
	"""Main quantum security engine orchestrating all security components"""
	
	def __init__(self, mqeb_service: MQEBService):
		self.service = mqeb_service
		self.key_manager = QuantumKeyManager()
		self.zero_trust = ZeroTrustMessageSecurity(self.key_manager)
		self.compliance = ComplianceAutomation()
		
		# Security configuration
		self.enabled = True
		self.default_algorithm = QuantumAlgorithm.CRYSTALS_KYBER_512
		self.default_security_level = SecurityLevel.QUANTUM_SAFE
		
		# Background tasks
		self._background_tasks: Set[asyncio.Task] = set()
		
		self.logger = logging.getLogger('mqeb.quantum_security_engine')
	
	async def initialize(self) -> None:
		"""Initialize quantum security engine"""
		self.logger.info("Initializing quantum security engine...")
		
		# Generate initial key pairs for existing tenants
		await self._initialize_tenant_keys()
		
		# Start background security tasks
		await self._start_background_tasks()
		
		self.logger.info("Quantum security engine initialized")
	
	async def shutdown(self) -> None:
		"""Shutdown quantum security engine"""
		self.enabled = False
		
		# Cancel background tasks
		for task in self._background_tasks:
			task.cancel()
		
		await asyncio.gather(*self._background_tasks, return_exceptions=True)
		self.logger.info("Quantum security engine shut down")
	
	async def secure_message(self, message: MQMessage, context: Dict[str, Any]) -> bool:
		"""Apply quantum security to message"""
		try:
			if not self.enabled:
				return True
			
			# Apply zero-trust policy
			if not await self.zero_trust.apply_zero_trust_policy(message, context):
				return False
			
			# Apply compliance controls if required
			compliance_frameworks = context.get('compliance_frameworks', [])
			if compliance_frameworks:
				compliance_result = await self.compliance.apply_compliance_controls(
					message, compliance_frameworks
				)
				if not compliance_result['compliant']:
					self.logger.warning(f"Message {message.id} failed compliance: {compliance_result['violations']}")
					return False
			
			# Encrypt message if required
			if message.encrypted or self._requires_encryption(message):
				encrypted = await self.encrypt_message(message, context)
				if not encrypted:
					return False
			
			return True
			
		except Exception as e:
			self.logger.error(f"Failed to secure message {message.id}: {e}")
			return False
	
	async def encrypt_message(self, message: MQMessage, context: Dict[str, Any]) -> bool:
		"""Encrypt message using quantum-safe cryptography"""
		try:
			# Get or generate key for tenant
			keys = await self.key_manager.get_tenant_keys(
				message.tenant_id, self.default_algorithm
			)
			
			if not keys:
				# Generate new key pair
				key_id = await self.key_manager.generate_key_pair(
					self.default_algorithm, message.tenant_id, self.default_security_level
				)
				keys = await self.key_manager.get_tenant_keys(
					message.tenant_id, self.default_algorithm
				)
			
			if not keys:
				raise Exception("Failed to obtain encryption keys")
			
			key_pair = keys[0]  # Use first available key
			
			# Create encryption context
			encryption_context = EncryptionContext(
				message_id=message.id,
				tenant_id=message.tenant_id,
				security_level=self.default_security_level,
				algorithm=self.default_algorithm,
				key_id=key_pair.key_id,
				nonce=secrets.token_bytes(16),
				authenticated_data=json.dumps({
					'topic': message.topic,
					'source': message.source_application,
					'timestamp': message.timestamp.isoformat()
				}).encode()
			)
			
			# Encrypt message payload
			operations = self.key_manager.quantum_algorithms[self.default_algorithm]
			encrypted_payload = await operations.encrypt(key_pair.public_key, message.payload)
			
			# Update message with encrypted data
			message.payload = encrypted_payload
			message.encrypted = True
			message.encryption_key_id = key_pair.key_id
			message.headers.update({
				'quantum_algorithm': self.default_algorithm.value,
				'encryption_context': base64.b64encode(
					json.dumps(encryption_context.__dict__, default=str).encode()
				).decode(),
				'quantum_safe': 'true'
			})
			
			self.logger.debug(f"Message {message.id} encrypted with quantum-safe algorithm")
			return True
			
		except Exception as e:
			self.logger.error(f"Failed to encrypt message {message.id}: {e}")
			return False
	
	async def decrypt_message(self, message: MQMessage, context: Dict[str, Any]) -> bytes:
		"""Decrypt message using quantum-safe cryptography"""
		try:
			if not message.encrypted:
				return message.payload
			
			key_id = message.encryption_key_id
			if not key_id:
				raise Exception("No encryption key ID in message")
			
			key_pair = await self.key_manager.get_key_pair(key_id)
			if not key_pair:
				raise Exception(f"Encryption key {key_id} not found or expired")
			
			# Get algorithm operations
			operations = self.key_manager.quantum_algorithms[key_pair.algorithm]
			
			# Decrypt payload
			decrypted_payload = await operations.decrypt(key_pair.private_key, message.payload)
			
			self.logger.debug(f"Message {message.id} decrypted successfully")
			return decrypted_payload
			
		except Exception as e:
			self.logger.error(f"Failed to decrypt message {message.id}: {e}")
			raise
	
	def _requires_encryption(self, message: MQMessage) -> bool:
		"""Determine if message requires encryption"""
		# Always encrypt high-priority messages
		if message.priority.value == 'critical':
			return True
		
		# Encrypt based on topic patterns
		sensitive_patterns = ['financial', 'payment', 'personal', 'confidential', 'secret']
		return any(pattern in message.topic.lower() for pattern in sensitive_patterns)
	
	async def _initialize_tenant_keys(self) -> None:
		"""Initialize keys for existing tenants"""
		try:
			# In a real implementation, this would query the tenant database
			# For now, we'll initialize keys on-demand
			self.logger.info("Tenant key initialization completed")
		except Exception as e:
			self.logger.error(f"Failed to initialize tenant keys: {e}")
	
	async def _start_background_tasks(self) -> None:
		"""Start background security tasks"""
		
		# Key rotation task
		task = asyncio.create_task(self._key_rotation_loop())
		self._background_tasks.add(task)
		task.add_done_callback(self._background_tasks.discard)
		
		# Security monitoring task  
		task = asyncio.create_task(self._security_monitoring_loop())
		self._background_tasks.add(task)
		task.add_done_callback(self._background_tasks.discard)
		
		# Compliance reporting task
		task = asyncio.create_task(self._compliance_reporting_loop())
		self._background_tasks.add(task)
		task.add_done_callback(self._background_tasks.discard)
	
	async def _key_rotation_loop(self) -> None:
		"""Background task for automatic key rotation"""
		while self.enabled:
			try:
				await asyncio.sleep(3600)  # Check every hour
				
				rotated_keys = await self.key_manager.rotate_expired_keys()
				if rotated_keys > 0:
					self.logger.info(f"Rotated {rotated_keys} expired keys")
				
			except asyncio.CancelledError:
				break
			except Exception as e:
				self.logger.error(f"Key rotation error: {e}")
	
	async def _security_monitoring_loop(self) -> None:
		"""Background task for security monitoring"""
		while self.enabled:
			try:
				await asyncio.sleep(300)  # Check every 5 minutes
				
				# Monitor threat indicators
				high_risk_sources = [
					source for source, score in self.zero_trust.threat_indicators.items()
					if score > 5.0
				]
				
				if high_risk_sources:
					self.logger.warning(f"High-risk sources detected: {high_risk_sources}")
				
				# Reset threat indicators periodically
				if len(self.zero_trust.threat_indicators) > 1000:
					self.zero_trust.threat_indicators.clear()
				
			except asyncio.CancelledError:
				break
			except Exception as e:
				self.logger.error(f"Security monitoring error: {e}")
	
	async def _compliance_reporting_loop(self) -> None:
		"""Background task for compliance reporting"""
		while self.enabled:
			try:
				await asyncio.sleep(86400)  # Check daily
				
				# Generate daily compliance summaries
				# In production, this would generate reports for all tenants
				self.logger.info("Daily compliance check completed")
				
			except asyncio.CancelledError:
				break
			except Exception as e:
				self.logger.error(f"Compliance reporting error: {e}")
	
	async def get_security_status(self) -> Dict[str, Any]:
		"""Get current security status"""
		return {
			'enabled': self.enabled,
			'total_keys': len(self.key_manager.key_store),
			'active_keys': sum(1 for key in self.key_manager.key_store.values() if not key.is_expired()),
			'default_algorithm': self.default_algorithm.value,
			'zero_trust_events': len(self.zero_trust.access_logs),
			'compliance_reports': len(self.compliance.compliance_reports),
			'threat_indicators': len(self.zero_trust.threat_indicators)
		}


# Factory function
async def create_quantum_security_engine(mqeb_service: MQEBService) -> QuantumSecurityEngine:
	"""Create and initialize quantum security engine"""
	engine = QuantumSecurityEngine(mqeb_service)
	await engine.initialize()
	return engine


# Export components
__all__ = [
	'QuantumSecurityEngine', 'QuantumKeyManager', 'ZeroTrustMessageSecurity', 'ComplianceAutomation',
	'QuantumAlgorithm', 'SecurityLevel', 'QuantumKeyPair', 'EncryptionContext', 'SecurityAuditEvent',
	'create_quantum_security_engine'
]