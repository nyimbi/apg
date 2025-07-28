"""
APG Customer Relationship Management - Data Encryption & Privacy Controls

Enterprise-grade data encryption and privacy control system with field-level encryption,
data masking, tokenization, and comprehensive privacy policy enforcement.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import asyncio
import base64
import hashlib
import json
import logging
import os
import secrets
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from uuid_extensions import uuid7str
from pydantic import BaseModel, Field, ConfigDict, field_validator
import re

import asyncpg
import redis.asyncio as redis
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


logger = logging.getLogger(__name__)


# ================================
# Enums and Constants
# ================================

class EncryptionAlgorithm(Enum):
	AES_256_GCM = "aes_256_gcm"
	AES_256_CBC = "aes_256_cbc"
	FERNET = "fernet"
	RSA_2048 = "rsa_2048"
	RSA_4096 = "rsa_4096"


class DataClassification(Enum):
	PUBLIC = "public"
	INTERNAL = "internal"
	CONFIDENTIAL = "confidential"
	RESTRICTED = "restricted"
	TOP_SECRET = "top_secret"


class PrivacyLevel(Enum):
	NONE = "none"
	BASIC = "basic"
	ENHANCED = "enhanced"
	STRICT = "strict"
	MAXIMUM = "maximum"


class TokenType(Enum):
	FORMAT_PRESERVING = "format_preserving"
	NON_FORMAT_PRESERVING = "non_format_preserving"
	RANDOM = "random"
	DETERMINISTIC = "deterministic"
	REFERENCE = "reference"


class MaskingStrategy(Enum):
	FULL_MASK = "full_mask"
	PARTIAL_MASK = "partial_mask"
	FIRST_LAST = "first_last"
	MIDDLE_MASK = "middle_mask"
	EMAIL_MASK = "email_mask"
	PHONE_MASK = "phone_mask"
	CREDIT_CARD_MASK = "credit_card_mask"
	SSN_MASK = "ssn_mask"
	RANDOMIZE = "randomize"
	NULL_OUT = "null_out"


class PurposeOfProcessing(Enum):
	LEGITIMATE_INTEREST = "legitimate_interest"
	CONTRACT_PERFORMANCE = "contract_performance"
	LEGAL_OBLIGATION = "legal_obligation"
	VITAL_INTERESTS = "vital_interests"
	PUBLIC_TASK = "public_task"
	CONSENT = "consent"


class DataSubjectRight(Enum):
	ACCESS = "access"
	RECTIFICATION = "rectification"
	ERASURE = "erasure"
	RESTRICT_PROCESSING = "restrict_processing"
	DATA_PORTABILITY = "data_portability"
	OBJECT = "object"
	WITHDRAW_CONSENT = "withdraw_consent"


# ================================
# Pydantic Models
# ================================

class EncryptionKey(BaseModel):
	"""Encryption key model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	key_name: str
	algorithm: EncryptionAlgorithm
	key_purpose: str
	key_length: int
	encrypted_key: str  # Key encrypted with master key
	key_fingerprint: str
	is_active: bool = True
	is_default: bool = False
	created_at: datetime = Field(default_factory=datetime.utcnow)
	expires_at: Optional[datetime] = None
	rotation_scheduled_at: Optional[datetime] = None
	last_used_at: Optional[datetime] = None
	usage_count: int = 0
	allowed_operations: List[str] = Field(default_factory=list)
	key_hierarchy_level: int = 1
	parent_key_id: Optional[str] = None
	derived_keys: List[str] = Field(default_factory=list)
	metadata: Dict[str, Any] = Field(default_factory=dict)


class FieldEncryptionRule(BaseModel):
	"""Field-level encryption rule"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	rule_name: str
	table_name: str
	field_name: str
	data_classification: DataClassification
	encryption_algorithm: EncryptionAlgorithm
	encryption_key_id: str
	is_searchable: bool = False
	tokenization_enabled: bool = False
	token_type: Optional[TokenType] = None
	masking_strategy: Optional[MaskingStrategy] = None
	is_active: bool = True
	priority: int = 100
	conditions: Dict[str, Any] = Field(default_factory=dict)
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str
	metadata: Dict[str, Any] = Field(default_factory=dict)


class DataToken(BaseModel):
	"""Data tokenization model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	token_value: str
	original_value_hash: str
	token_type: TokenType
	field_name: str
	table_name: str
	data_classification: DataClassification
	format_preserved: bool = False
	reversible: bool = True
	created_at: datetime = Field(default_factory=datetime.utcnow)
	last_accessed_at: Optional[datetime] = None
	access_count: int = 0
	expires_at: Optional[datetime] = None
	metadata: Dict[str, Any] = Field(default_factory=dict)


class PrivacyPolicy(BaseModel):
	"""Privacy policy configuration"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	policy_name: str
	policy_version: str
	data_types: List[str] = Field(default_factory=list)
	purposes: List[PurposeOfProcessing] = Field(default_factory=list)
	retention_period_days: int
	geographic_restrictions: List[str] = Field(default_factory=list)
	consent_required: bool = True
	legitimate_interest_basis: Optional[str] = None
	data_minimization_rules: Dict[str, Any] = Field(default_factory=dict)
	automated_decision_making: bool = False
	profiling_enabled: bool = False
	third_party_sharing: bool = False
	international_transfers: bool = False
	is_active: bool = True
	effective_date: datetime = Field(default_factory=datetime.utcnow)
	expiry_date: Optional[datetime] = None
	approved_by: str
	metadata: Dict[str, Any] = Field(default_factory=dict)


class ConsentRecord(BaseModel):
	"""Data subject consent record"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	data_subject_id: str
	data_subject_email: str
	consent_type: str
	purposes: List[PurposeOfProcessing] = Field(default_factory=list)
	data_categories: List[str] = Field(default_factory=list)
	consent_given: bool
	consent_source: str
	consent_method: str
	consent_timestamp: datetime = Field(default_factory=datetime.utcnow)
	consent_expiry: Optional[datetime] = None
	withdrawal_timestamp: Optional[datetime] = None
	withdrawal_reason: Optional[str] = None
	lawful_basis: PurposeOfProcessing
	consent_evidence: Dict[str, Any] = Field(default_factory=dict)
	privacy_policy_version: str
	is_active: bool = True
	metadata: Dict[str, Any] = Field(default_factory=dict)


class DataSubjectRequest(BaseModel):
	"""Data subject rights request"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	request_type: DataSubjectRight
	data_subject_id: str
	data_subject_email: str
	request_description: str
	requested_at: datetime = Field(default_factory=datetime.utcnow)
	verification_method: str
	verification_status: str = "pending"
	verified_at: Optional[datetime] = None
	processing_status: str = "received"
	assigned_to: Optional[str] = None
	due_date: datetime
	completed_at: Optional[datetime] = None
	response_provided: bool = False
	response_data: Dict[str, Any] = Field(default_factory=dict)
	rejection_reason: Optional[str] = None
	escalated: bool = False
	escalated_to: Optional[str] = None
	external_reference: Optional[str] = None
	processing_log: List[Dict[str, Any]] = Field(default_factory=list)
	metadata: Dict[str, Any] = Field(default_factory=dict)


# ================================
# Data Encryption & Privacy Manager
# ================================

class DataEncryptionPrivacyManager:
	"""Enterprise data encryption and privacy control system"""
	
	def __init__(self, db_pool: asyncpg.Pool, redis_client: redis.Redis = None):
		self.db_pool = db_pool
		self.redis_client = redis_client
		self.master_key = self._load_or_generate_master_key()
		self.master_fernet = Fernet(self.master_key)
		self._encryption_keys = {}
		self._field_rules = {}
		self._token_cache = {}
		self._initialized = False
	
	def _load_or_generate_master_key(self) -> bytes:
		"""Load or generate master encryption key"""
		# In production, this should be managed by HSM or key management service
		master_key_path = os.getenv('MASTER_KEY_PATH', '/tmp/crm_master.key')
		
		try:
			if os.path.exists(master_key_path):
				with open(master_key_path, 'rb') as f:
					return f.read()
			else:
				# Generate new master key
				master_key = Fernet.generate_key()
				with open(master_key_path, 'wb') as f:
					f.write(master_key)
				return master_key
		except Exception as e:
			logger.warning(f"Could not load/save master key file: {str(e)}")
			# Fall back to generating key in memory
			return Fernet.generate_key()
	
	async def initialize(self):
		"""Initialize the encryption and privacy system"""
		try:
			if self._initialized:
				return
			
			logger.info("🔐 Initializing Data Encryption & Privacy Manager...")
			
			# Validate database connection
			async with self.db_pool.acquire() as conn:
				await conn.execute("SELECT 1")
			
			# Load encryption keys
			await self._load_encryption_keys()
			
			# Load field encryption rules
			await self._load_field_encryption_rules()
			
			# Initialize default keys if none exist
			await self._ensure_default_keys()
			
			# Start background tasks
			asyncio.create_task(self._key_rotation_task())
			asyncio.create_task(self._consent_expiry_task())
			
			self._initialized = True
			logger.info("✅ Data Encryption & Privacy Manager initialized successfully")
			
		except Exception as e:
			logger.error(f"❌ Failed to initialize encryption manager: {str(e)}")
			raise
	
	async def _load_encryption_keys(self):
		"""Load encryption keys from database"""
		try:
			async with self.db_pool.acquire() as conn:
				keys = await conn.fetch("""
					SELECT * FROM crm_encryption_keys 
					WHERE is_active = true
				""")
				
				for key_row in keys:
					key_data = dict(key_row)
					key_data['algorithm'] = EncryptionAlgorithm(key_row['algorithm'])
					key_data['allowed_operations'] = json.loads(key_row['allowed_operations'] or '[]')
					key_data['derived_keys'] = json.loads(key_row['derived_keys'] or '[]')
					key_data['metadata'] = json.loads(key_row['metadata'] or '{}')
					
					encryption_key = EncryptionKey(**key_data)
					self._encryption_keys[encryption_key.id] = encryption_key
				
				logger.info(f"🔑 Loaded {len(self._encryption_keys)} encryption keys")
				
		except Exception as e:
			logger.error(f"Error loading encryption keys: {str(e)}")
	
	async def _load_field_encryption_rules(self):
		"""Load field encryption rules from database"""
		try:
			async with self.db_pool.acquire() as conn:
				rules = await conn.fetch("""
					SELECT * FROM crm_field_encryption_rules 
					WHERE is_active = true
					ORDER BY priority DESC
				""")
				
				for rule_row in rules:
					rule_data = dict(rule_row)
					rule_data['data_classification'] = DataClassification(rule_row['data_classification'])
					rule_data['encryption_algorithm'] = EncryptionAlgorithm(rule_row['encryption_algorithm'])
					
					if rule_row['token_type']:
						rule_data['token_type'] = TokenType(rule_row['token_type'])
					if rule_row['masking_strategy']:
						rule_data['masking_strategy'] = MaskingStrategy(rule_row['masking_strategy'])
					
					rule_data['conditions'] = json.loads(rule_row['conditions'] or '{}')
					rule_data['metadata'] = json.loads(rule_row['metadata'] or '{}')
					
					rule = FieldEncryptionRule(**rule_data)
					rule_key = f"{rule.table_name}.{rule.field_name}"
					self._field_rules[rule_key] = rule
				
				logger.info(f"📋 Loaded {len(self._field_rules)} field encryption rules")
				
		except Exception as e:
			logger.error(f"Error loading field encryption rules: {str(e)}")
	
	async def _ensure_default_keys(self):
		"""Ensure default encryption keys exist"""
		try:
			if not self._encryption_keys:
				# Create default AES key
				await self.create_encryption_key(
					tenant_id="system",
					key_name="default_aes_key",
					algorithm=EncryptionAlgorithm.AES_256_GCM,
					key_purpose="general_encryption",
					is_default=True
				)
				
				# Create default tokenization key
				await self.create_encryption_key(
					tenant_id="system",
					key_name="default_tokenization_key",
					algorithm=EncryptionAlgorithm.FERNET,
					key_purpose="tokenization",
					is_default=False
				)
				
		except Exception as e:
			logger.error(f"Error ensuring default keys: {str(e)}")
	
	def _generate_key(self, algorithm: EncryptionAlgorithm, key_length: int) -> bytes:
		"""Generate encryption key based on algorithm"""
		if algorithm in [EncryptionAlgorithm.AES_256_GCM, EncryptionAlgorithm.AES_256_CBC]:
			return secrets.token_bytes(32)  # 256 bits
		elif algorithm == EncryptionAlgorithm.FERNET:
			return Fernet.generate_key()
		elif algorithm in [EncryptionAlgorithm.RSA_2048, EncryptionAlgorithm.RSA_4096]:
			key_size = 2048 if algorithm == EncryptionAlgorithm.RSA_2048 else 4096
			private_key = rsa.generate_private_key(
				public_exponent=65537,
				key_size=key_size
			)
			return private_key.private_bytes(
				encoding=serialization.Encoding.PEM,
				format=serialization.PrivateFormat.PKCS8,
				encryption_algorithm=serialization.NoEncryption()
			)
		else:
			raise ValueError(f"Unsupported algorithm: {algorithm}")
	
	def _calculate_key_fingerprint(self, key_data: bytes) -> str:
		"""Calculate key fingerprint"""
		return hashlib.sha256(key_data).hexdigest()[:16]
	
	async def create_encryption_key(
		self,
		tenant_id: str,
		key_name: str,
		algorithm: EncryptionAlgorithm,
		key_purpose: str,
		key_length: int = None,
		is_default: bool = False,
		expires_at: datetime = None,
		allowed_operations: List[str] = None
	) -> EncryptionKey:
		"""Create a new encryption key"""
		try:
			# Determine key length based on algorithm
			if key_length is None:
				key_length_map = {
					EncryptionAlgorithm.AES_256_GCM: 256,
					EncryptionAlgorithm.AES_256_CBC: 256,
					EncryptionAlgorithm.FERNET: 256,
					EncryptionAlgorithm.RSA_2048: 2048,
					EncryptionAlgorithm.RSA_4096: 4096,
				}
				key_length = key_length_map.get(algorithm, 256)
			
			# Generate the key
			raw_key = self._generate_key(algorithm, key_length)
			
			# Encrypt the key with master key
			encrypted_key = self.master_fernet.encrypt(raw_key).decode()
			
			# Calculate fingerprint
			fingerprint = self._calculate_key_fingerprint(raw_key)
			
			# Create key object
			encryption_key = EncryptionKey(
				tenant_id=tenant_id,
				key_name=key_name,
				algorithm=algorithm,
				key_purpose=key_purpose,
				key_length=key_length,
				encrypted_key=encrypted_key,
				key_fingerprint=fingerprint,
				is_default=is_default,
				expires_at=expires_at,
				allowed_operations=allowed_operations or ["encrypt", "decrypt"]
			)
			
			# Store in database
			async with self.db_pool.acquire() as conn:
				await conn.execute("""
					INSERT INTO crm_encryption_keys (
						id, tenant_id, key_name, algorithm, key_purpose,
						key_length, encrypted_key, key_fingerprint, is_active,
						is_default, expires_at, allowed_operations, metadata
					) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
				""",
				encryption_key.id, encryption_key.tenant_id, encryption_key.key_name,
				encryption_key.algorithm.value, encryption_key.key_purpose,
				encryption_key.key_length, encryption_key.encrypted_key,
				encryption_key.key_fingerprint, encryption_key.is_active,
				encryption_key.is_default, encryption_key.expires_at,
				json.dumps(encryption_key.allowed_operations), json.dumps(encryption_key.metadata))
			
			# Cache the key
			self._encryption_keys[encryption_key.id] = encryption_key
			
			logger.info(f"🔑 Created encryption key: {key_name} ({algorithm.value})")
			return encryption_key
			
		except Exception as e:
			logger.error(f"Error creating encryption key: {str(e)}")
			raise
	
	def _get_decrypted_key(self, key_id: str) -> bytes:
		"""Get decrypted key data"""
		if key_id not in self._encryption_keys:
			raise ValueError(f"Encryption key not found: {key_id}")
		
		encryption_key = self._encryption_keys[key_id]
		if not encryption_key.is_active:
			raise ValueError(f"Encryption key is inactive: {key_id}")
		
		# Decrypt the key
		encrypted_key_bytes = encryption_key.encrypted_key.encode()
		return self.master_fernet.decrypt(encrypted_key_bytes)
	
	async def encrypt_field_value(
		self,
		tenant_id: str,
		table_name: str,
		field_name: str,
		value: Any,
		encryption_key_id: str = None
	) -> str:
		"""Encrypt a field value"""
		try:
			if value is None:
				return None
			
			# Get encryption rule
			rule_key = f"{table_name}.{field_name}"
			rule = self._field_rules.get(rule_key)
			
			if not rule and not encryption_key_id:
				# No encryption rule found and no key specified
				return str(value)
			
			# Determine encryption key
			if encryption_key_id:
				key_id = encryption_key_id
			elif rule:
				key_id = rule.encryption_key_id
			else:
				# Use default key
				default_key = next(
					(k for k in self._encryption_keys.values() if k.is_default),
					None
				)
				if not default_key:
					raise ValueError("No default encryption key available")
				key_id = default_key.id
			
			# Get the key
			key_data = self._get_decrypted_key(key_id)
			encryption_key = self._encryption_keys[key_id]
			
			# Convert value to string
			value_str = str(value)
			
			# Encrypt based on algorithm
			if encryption_key.algorithm == EncryptionAlgorithm.FERNET:
				fernet = Fernet(key_data)
				encrypted_bytes = fernet.encrypt(value_str.encode())
				encrypted_value = base64.b64encode(encrypted_bytes).decode()
			
			elif encryption_key.algorithm == EncryptionAlgorithm.AES_256_GCM:
				# Generate random IV
				iv = secrets.token_bytes(12)  # 96-bit IV for GCM
				cipher = Cipher(algorithms.AES(key_data), modes.GCM(iv))
				encryptor = cipher.encryptor()
				
				ciphertext = encryptor.update(value_str.encode()) + encryptor.finalize()
				
				# Combine IV + tag + ciphertext
				encrypted_data = iv + encryptor.tag + ciphertext
				encrypted_value = base64.b64encode(encrypted_data).decode()
			
			else:
				raise ValueError(f"Unsupported encryption algorithm: {encryption_key.algorithm}")
			
			# Update key usage
			await self._update_key_usage(key_id)
			
			return encrypted_value
			
		except Exception as e:
			logger.error(f"Error encrypting field value: {str(e)}")
			raise
	
	async def decrypt_field_value(
		self,
		tenant_id: str,
		table_name: str,
		field_name: str,
		encrypted_value: str,
		encryption_key_id: str = None
	) -> Any:
		"""Decrypt a field value"""
		try:
			if encrypted_value is None:
				return None
			
			# Get encryption rule
			rule_key = f"{table_name}.{field_name}"
			rule = self._field_rules.get(rule_key)
			
			# Determine encryption key
			if encryption_key_id:
				key_id = encryption_key_id
			elif rule:
				key_id = rule.encryption_key_id
			else:
				# Try to find key by attempting decryption with all keys
				for key_id in self._encryption_keys.keys():
					try:
						return await self.decrypt_field_value(
							tenant_id, table_name, field_name, encrypted_value, key_id
						)
					except:
						continue
				raise ValueError("No suitable decryption key found")
			
			# Get the key
			key_data = self._get_decrypted_key(key_id)
			encryption_key = self._encryption_keys[key_id]
			
			# Decrypt based on algorithm
			if encryption_key.algorithm == EncryptionAlgorithm.FERNET:
				fernet = Fernet(key_data)
				encrypted_bytes = base64.b64decode(encrypted_value.encode())
				decrypted_bytes = fernet.decrypt(encrypted_bytes)
				decrypted_value = decrypted_bytes.decode()
			
			elif encryption_key.algorithm == EncryptionAlgorithm.AES_256_GCM:
				encrypted_data = base64.b64decode(encrypted_value.encode())
				
				# Extract IV, tag, and ciphertext
				iv = encrypted_data[:12]
				tag = encrypted_data[12:28]
				ciphertext = encrypted_data[28:]
				
				cipher = Cipher(algorithms.AES(key_data), modes.GCM(iv, tag))
				decryptor = cipher.decryptor()
				
				decrypted_bytes = decryptor.update(ciphertext) + decryptor.finalize()
				decrypted_value = decrypted_bytes.decode()
			
			else:
				raise ValueError(f"Unsupported decryption algorithm: {encryption_key.algorithm}")
			
			# Update key usage
			await self._update_key_usage(key_id)
			
			return decrypted_value
			
		except Exception as e:
			logger.error(f"Error decrypting field value: {str(e)}")
			raise
	
	async def _update_key_usage(self, key_id: str):
		"""Update key usage statistics"""
		try:
			async with self.db_pool.acquire() as conn:
				await conn.execute("""
					UPDATE crm_encryption_keys 
					SET usage_count = usage_count + 1, 
						last_used_at = NOW()
					WHERE id = $1
				""", key_id)
			
			# Update cached key
			if key_id in self._encryption_keys:
				self._encryption_keys[key_id].usage_count += 1
				self._encryption_keys[key_id].last_used_at = datetime.utcnow()
				
		except Exception as e:
			logger.error(f"Error updating key usage: {str(e)}")
	
	async def create_field_encryption_rule(
		self,
		tenant_id: str,
		rule_name: str,
		table_name: str,
		field_name: str,
		data_classification: DataClassification,
		encryption_algorithm: EncryptionAlgorithm,
		encryption_key_id: str,
		is_searchable: bool = False,
		tokenization_enabled: bool = False,
		token_type: TokenType = None,
		masking_strategy: MaskingStrategy = None,
		created_by: str = None
	) -> FieldEncryptionRule:
		"""Create field encryption rule"""
		try:
			rule = FieldEncryptionRule(
				tenant_id=tenant_id,
				rule_name=rule_name,
				table_name=table_name,
				field_name=field_name,
				data_classification=data_classification,
				encryption_algorithm=encryption_algorithm,
				encryption_key_id=encryption_key_id,
				is_searchable=is_searchable,
				tokenization_enabled=tokenization_enabled,
				token_type=token_type,
				masking_strategy=masking_strategy,
				created_by=created_by or "system"
			)
			
			# Store in database
			async with self.db_pool.acquire() as conn:
				await conn.execute("""
					INSERT INTO crm_field_encryption_rules (
						id, tenant_id, rule_name, table_name, field_name,
						data_classification, encryption_algorithm, encryption_key_id,
						is_searchable, tokenization_enabled, token_type,
						masking_strategy, is_active, priority, conditions,
						created_by, metadata
					) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
				""",
				rule.id, rule.tenant_id, rule.rule_name, rule.table_name,
				rule.field_name, rule.data_classification.value,
				rule.encryption_algorithm.value, rule.encryption_key_id,
				rule.is_searchable, rule.tokenization_enabled,
				rule.token_type.value if rule.token_type else None,
				rule.masking_strategy.value if rule.masking_strategy else None,
				rule.is_active, rule.priority, json.dumps(rule.conditions),
				rule.created_by, json.dumps(rule.metadata))
			
			# Cache the rule
			rule_key = f"{rule.table_name}.{rule.field_name}"
			self._field_rules[rule_key] = rule
			
			logger.info(f"📋 Created field encryption rule: {rule_name}")
			return rule
			
		except Exception as e:
			logger.error(f"Error creating field encryption rule: {str(e)}")
			raise
	
	def _generate_token(self, token_type: TokenType, original_value: str, field_name: str) -> str:
		"""Generate token for data tokenization"""
		if token_type == TokenType.RANDOM:
			return secrets.token_urlsafe(16)
		
		elif token_type == TokenType.DETERMINISTIC:
			# Generate deterministic token based on value and field
			hash_input = f"{original_value}:{field_name}".encode()
			return hashlib.sha256(hash_input).hexdigest()[:16]
		
		elif token_type == TokenType.FORMAT_PRESERVING:
			# Preserve format but replace with random values
			if re.match(r'^\d+$', original_value):  # All digits
				return ''.join(secrets.choice('0123456789') for _ in range(len(original_value)))
			elif re.match(r'^[A-Za-z]+$', original_value):  # All letters
				return ''.join(secrets.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz') for _ in range(len(original_value)))
			else:
				# Mixed format - preserve structure
				token = ""
				for char in original_value:
					if char.isdigit():
						token += secrets.choice('0123456789')
					elif char.isalpha():
						token += secrets.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
					else:
						token += char
				return token
		
		elif token_type == TokenType.REFERENCE:
			return f"REF_{secrets.token_hex(8).upper()}"
		
		else:
			return secrets.token_urlsafe(12)
	
	async def tokenize_value(
		self,
		tenant_id: str,
		table_name: str,
		field_name: str,
		original_value: str,
		token_type: TokenType = TokenType.RANDOM,
		reversible: bool = True
	) -> str:
		"""Tokenize a sensitive value"""
		try:
			if original_value is None:
				return None
			
			# Check if token already exists for this value
			value_hash = hashlib.sha256(f"{original_value}:{field_name}".encode()).hexdigest()
			
			async with self.db_pool.acquire() as conn:
				existing_token = await conn.fetchrow("""
					SELECT token_value FROM crm_data_tokens 
					WHERE tenant_id = $1 AND original_value_hash = $2 
					AND field_name = $3 AND table_name = $4
					AND expires_at > NOW()
				""", tenant_id, value_hash, field_name, table_name)
				
				if existing_token:
					return existing_token['token_value']
			
			# Generate new token
			token_value = self._generate_token(token_type, original_value, field_name)
			
			# Determine data classification
			rule_key = f"{table_name}.{field_name}"
			rule = self._field_rules.get(rule_key)
			data_classification = rule.data_classification if rule else DataClassification.INTERNAL
			
			# Create token record
			token = DataToken(
				tenant_id=tenant_id,
				token_value=token_value,
				original_value_hash=value_hash,
				token_type=token_type,
				field_name=field_name,
				table_name=table_name,
				data_classification=data_classification,
				format_preserved=token_type == TokenType.FORMAT_PRESERVING,
				reversible=reversible
			)
			
			# Store token
			await conn.execute("""
				INSERT INTO crm_data_tokens (
					id, tenant_id, token_value, original_value_hash,
					token_type, field_name, table_name, data_classification,
					format_preserved, reversible, metadata
				) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
			""",
			token.id, token.tenant_id, token.token_value,
			token.original_value_hash, token.token_type.value,
			token.field_name, token.table_name, token.data_classification.value,
			token.format_preserved, token.reversible, json.dumps(token.metadata))
			
			# Cache token if reversible
			if reversible:
				cache_key = f"token:{tenant_id}:{token_value}"
				self._token_cache[cache_key] = {
					'original_hash': value_hash,
					'field_name': field_name,
					'table_name': table_name
				}
			
			logger.debug(f"🎫 Tokenized value for {table_name}.{field_name}")
			return token_value
			
		except Exception as e:
			logger.error(f"Error tokenizing value: {str(e)}")
			raise
	
	def mask_value(self, value: str, strategy: MaskingStrategy) -> str:
		"""Apply data masking strategy"""
		if value is None:
			return None
		
		if strategy == MaskingStrategy.FULL_MASK:
			return '*' * len(value)
		
		elif strategy == MaskingStrategy.PARTIAL_MASK:
			if len(value) <= 4:
				return '*' * len(value)
			return value[:2] + '*' * (len(value) - 4) + value[-2:]
		
		elif strategy == MaskingStrategy.FIRST_LAST:
			if len(value) <= 2:
				return value
			return value[0] + '*' * (len(value) - 2) + value[-1]
		
		elif strategy == MaskingStrategy.MIDDLE_MASK:
			if len(value) <= 6:
				return value
			return value[:3] + '*' * (len(value) - 6) + value[-3:]
		
		elif strategy == MaskingStrategy.EMAIL_MASK:
			if '@' in value:
				local, domain = value.split('@', 1)
				masked_local = local[0] + '*' * (len(local) - 1) if len(local) > 1 else local
				return f"{masked_local}@{domain}"
			return self.mask_value(value, MaskingStrategy.PARTIAL_MASK)
		
		elif strategy == MaskingStrategy.PHONE_MASK:
			# Mask phone number, keep last 4 digits
			digits_only = re.sub(r'\D', '', value)
			if len(digits_only) >= 4:
				masked_digits = '*' * (len(digits_only) - 4) + digits_only[-4:]
				# Preserve original format structure
				result = value
				digit_index = 0
				for i, char in enumerate(value):
					if char.isdigit():
						if digit_index < len(masked_digits):
							result = result[:i] + masked_digits[digit_index] + result[i+1:]
							digit_index += 1
				return result
			return '*' * len(value)
		
		elif strategy == MaskingStrategy.CREDIT_CARD_MASK:
			# Mask credit card, show last 4 digits
			digits_only = re.sub(r'\D', '', value)
			if len(digits_only) >= 4:
				masked = '*' * (len(digits_only) - 4) + digits_only[-4:]
				# Preserve formatting
				result = ""
				mask_index = 0
				for char in value:
					if char.isdigit():
						result += masked[mask_index] if mask_index < len(masked) else char
						mask_index += 1
					else:
						result += char
				return result
			return '*' * len(value)
		
		elif strategy == MaskingStrategy.SSN_MASK:
			# Mask SSN, show last 4 digits
			digits_only = re.sub(r'\D', '', value)
			if len(digits_only) == 9:
				return f"***-**-{digits_only[-4:]}"
			return self.mask_value(value, MaskingStrategy.PARTIAL_MASK)
		
		elif strategy == MaskingStrategy.RANDOMIZE:
			# Generate random string of same length and character types
			result = ""
			for char in value:
				if char.isdigit():
					result += secrets.choice('0123456789')
				elif char.isalpha():
					if char.isupper():
						result += secrets.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
					else:
						result += secrets.choice('abcdefghijklmnopqrstuvwxyz')
				else:
					result += char
			return result
		
		elif strategy == MaskingStrategy.NULL_OUT:
			return None
		
		else:
			return value
	
	async def create_privacy_policy(
		self,
		tenant_id: str,
		policy_name: str,
		policy_version: str,
		data_types: List[str],
		purposes: List[PurposeOfProcessing],
		retention_period_days: int,
		approved_by: str,
		**kwargs
	) -> PrivacyPolicy:
		"""Create privacy policy"""
		try:
			policy = PrivacyPolicy(
				tenant_id=tenant_id,
				policy_name=policy_name,
				policy_version=policy_version,
				data_types=data_types,
				purposes=purposes,
				retention_period_days=retention_period_days,
				approved_by=approved_by,
				**kwargs
			)
			
			async with self.db_pool.acquire() as conn:
				await conn.execute("""
					INSERT INTO crm_privacy_policies (
						id, tenant_id, policy_name, policy_version, data_types,
						purposes, retention_period_days, geographic_restrictions,
						consent_required, legitimate_interest_basis, data_minimization_rules,
						automated_decision_making, profiling_enabled, third_party_sharing,
						international_transfers, is_active, effective_date, expiry_date,
						approved_by, metadata
					) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20)
				""",
				policy.id, policy.tenant_id, policy.policy_name, policy.policy_version,
				json.dumps(policy.data_types), json.dumps([p.value for p in policy.purposes]),
				policy.retention_period_days, json.dumps(policy.geographic_restrictions),
				policy.consent_required, policy.legitimate_interest_basis,
				json.dumps(policy.data_minimization_rules), policy.automated_decision_making,
				policy.profiling_enabled, policy.third_party_sharing, policy.international_transfers,
				policy.is_active, policy.effective_date, policy.expiry_date,
				policy.approved_by, json.dumps(policy.metadata))
			
			logger.info(f"📋 Created privacy policy: {policy_name} v{policy_version}")
			return policy
			
		except Exception as e:
			logger.error(f"Error creating privacy policy: {str(e)}")
			raise
	
	async def record_consent(
		self,
		tenant_id: str,
		data_subject_id: str,
		data_subject_email: str,
		consent_type: str,
		purposes: List[PurposeOfProcessing],
		data_categories: List[str],
		consent_given: bool,
		consent_source: str,
		consent_method: str,
		lawful_basis: PurposeOfProcessing,
		privacy_policy_version: str,
		consent_evidence: Dict[str, Any] = None
	) -> ConsentRecord:
		"""Record data subject consent"""
		try:
			consent = ConsentRecord(
				tenant_id=tenant_id,
				data_subject_id=data_subject_id,
				data_subject_email=data_subject_email,
				consent_type=consent_type,
				purposes=purposes,
				data_categories=data_categories,
				consent_given=consent_given,
				consent_source=consent_source,
				consent_method=consent_method,
				lawful_basis=lawful_basis,
				privacy_policy_version=privacy_policy_version,
				consent_evidence=consent_evidence or {}
			)
			
			async with self.db_pool.acquire() as conn:
				await conn.execute("""
					INSERT INTO crm_consent_records (
						id, tenant_id, data_subject_id, data_subject_email,
						consent_type, purposes, data_categories, consent_given,
						consent_source, consent_method, consent_timestamp,
						lawful_basis, consent_evidence, privacy_policy_version,
						is_active, metadata
					) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
				""",
				consent.id, consent.tenant_id, consent.data_subject_id,
				consent.data_subject_email, consent.consent_type,
				json.dumps([p.value for p in consent.purposes]),
				json.dumps(consent.data_categories), consent.consent_given,
				consent.consent_source, consent.consent_method,
				consent.consent_timestamp, consent.lawful_basis.value,
				json.dumps(consent.consent_evidence), consent.privacy_policy_version,
				consent.is_active, json.dumps(consent.metadata))
			
			logger.info(f"📝 Recorded consent: {data_subject_email} ({consent_type})")
			return consent
			
		except Exception as e:
			logger.error(f"Error recording consent: {str(e)}")
			raise
	
	async def create_data_subject_request(
		self,
		tenant_id: str,
		request_type: DataSubjectRight,
		data_subject_id: str,
		data_subject_email: str,
		request_description: str,
		verification_method: str
	) -> DataSubjectRequest:
		"""Create data subject rights request"""
		try:
			# Calculate due date (30 days for most rights, 72 hours for breach notification)
			due_days = 3 if request_type == DataSubjectRight.ERASURE else 30
			due_date = datetime.utcnow() + timedelta(days=due_days)
			
			request = DataSubjectRequest(
				tenant_id=tenant_id,
				request_type=request_type,
				data_subject_id=data_subject_id,
				data_subject_email=data_subject_email,
				request_description=request_description,
				verification_method=verification_method,
				due_date=due_date
			)
			
			async with self.db_pool.acquire() as conn:
				await conn.execute("""
					INSERT INTO crm_data_subject_requests (
						id, tenant_id, request_type, data_subject_id,
						data_subject_email, request_description, requested_at,
						verification_method, verification_status, processing_status,
						due_date, response_provided, escalated, processing_log, metadata
					) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
				""",
				request.id, request.tenant_id, request.request_type.value,
				request.data_subject_id, request.data_subject_email,
				request.request_description, request.requested_at,
				request.verification_method, request.verification_status,
				request.processing_status, request.due_date, request.response_provided,
				request.escalated, json.dumps(request.processing_log),
				json.dumps(request.metadata))
			
			logger.info(f"📨 Created data subject request: {request_type.value} for {data_subject_email}")
			return request
			
		except Exception as e:
			logger.error(f"Error creating data subject request: {str(e)}")
			raise
	
	async def _key_rotation_task(self):
		"""Background task for key rotation"""
		while True:
			try:
				await asyncio.sleep(86400)  # Check daily
				
				# Find keys that need rotation
				current_time = datetime.utcnow()
				
				for key_id, key in self._encryption_keys.items():
					if (key.rotation_scheduled_at and 
						key.rotation_scheduled_at <= current_time):
						
						await self._rotate_encryption_key(key_id)
				
			except Exception as e:
				logger.error(f"Error in key rotation task: {str(e)}")
	
	async def _rotate_encryption_key(self, key_id: str):
		"""Rotate an encryption key"""
		try:
			old_key = self._encryption_keys[key_id]
			
			# Create new key with same parameters
			new_key = await self.create_encryption_key(
				tenant_id=old_key.tenant_id,
				key_name=f"{old_key.key_name}_rotated_{int(datetime.utcnow().timestamp())}",
				algorithm=old_key.algorithm,
				key_purpose=old_key.key_purpose,
				key_length=old_key.key_length,
				is_default=old_key.is_default,
				allowed_operations=old_key.allowed_operations
			)
			
			# Update field encryption rules to use new key
			async with self.db_pool.acquire() as conn:
				await conn.execute("""
					UPDATE crm_field_encryption_rules 
					SET encryption_key_id = $1, updated_at = NOW()
					WHERE encryption_key_id = $2
				""", new_key.id, key_id)
			
			# Deactivate old key
			await conn.execute("""
				UPDATE crm_encryption_keys 
				SET is_active = false, rotation_scheduled_at = NULL
				WHERE id = $1
			""", key_id)
			
			# Update caches
			self._encryption_keys[key_id].is_active = False
			await self._load_field_encryption_rules()
			
			logger.info(f"🔄 Rotated encryption key: {old_key.key_name}")
			
		except Exception as e:
			logger.error(f"Error rotating key {key_id}: {str(e)}")
	
	async def _consent_expiry_task(self):
		"""Background task to handle consent expiry"""
		while True:
			try:
				await asyncio.sleep(3600)  # Check hourly
				
				# Find expired consents
				async with self.db_pool.acquire() as conn:
					expired_consents = await conn.fetch("""
						SELECT id, data_subject_id, data_subject_email 
						FROM crm_consent_records 
						WHERE consent_expiry < NOW() AND is_active = true
					""")
					
					for consent in expired_consents:
						# Mark consent as inactive
						await conn.execute("""
							UPDATE crm_consent_records 
							SET is_active = false 
							WHERE id = $1
						""", consent['id'])
						
						logger.info(f"⏰ Expired consent for {consent['data_subject_email']}")
				
			except Exception as e:
				logger.error(f"Error in consent expiry task: {str(e)}")
	
	async def health_check(self) -> Dict[str, Any]:
		"""Perform health check"""
		try:
			health_status = {
				'status': 'healthy',
				'timestamp': datetime.utcnow().isoformat(),
				'components': {}
			}
			
			# Check database connection
			try:
				async with self.db_pool.acquire() as conn:
					result = await conn.fetchrow("SELECT COUNT(*) as count FROM crm_encryption_keys")
				health_status['components']['database'] = 'healthy'
				health_status['components']['encryption_keys'] = result['count']
			except Exception as e:
				health_status['components']['database'] = f'unhealthy: {str(e)}'
				health_status['status'] = 'degraded'
			
			# Check master key
			try:
				test_data = b"health_check_test"
				encrypted_data = self.master_fernet.encrypt(test_data)
				decrypted_data = self.master_fernet.decrypt(encrypted_data)
				assert decrypted_data == test_data
				health_status['components']['master_key'] = 'healthy'
			except Exception as e:
				health_status['components']['master_key'] = f'unhealthy: {str(e)}'
				health_status['status'] = 'critical'
			
			# Check loaded keys and rules
			health_status['components']['loaded_keys'] = len(self._encryption_keys)
			health_status['components']['loaded_rules'] = len(self._field_rules)
			
			return health_status
			
		except Exception as e:
			return {
				'status': 'unhealthy',
				'error': str(e),
				'timestamp': datetime.utcnow().isoformat()
			}