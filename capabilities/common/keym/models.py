#!/usr/bin/env python3
"""
APG Key Management Models
Pydantic v2 models for quantum-safe key management following APG standards

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Annotated, Union
from uuid_extensions import uuid7str

from pydantic import BaseModel, Field, ConfigDict, AfterValidator, field_validator
from cryptography.hazmat.primitives.asymmetric import rsa, ec, padding
from cryptography.hazmat.primitives import hashes, serialization


class KeyAlgorithm(str, Enum):
	"""Supported cryptographic algorithms"""
	# Symmetric algorithms
	AES_128 = "AES-128"
	AES_256 = "AES-256"
	CHACHA20_POLY1305 = "ChaCha20-Poly1305"
	
	# Asymmetric algorithms (current)
	RSA_2048 = "RSA-2048"
	RSA_4096 = "RSA-4096"
	ECDSA_P256 = "ECDSA-P256"
	ECDSA_P384 = "ECDSA-P384"
	ED25519 = "Ed25519"
	
	# Post-quantum algorithms (future-ready)
	KYBER_512 = "Kyber-512"
	KYBER_768 = "Kyber-768"
	KYBER_1024 = "Kyber-1024"
	DILITHIUM_2 = "Dilithium-2"
	DILITHIUM_3 = "Dilithium-3"
	DILITHIUM_5 = "Dilithium-5"
	FALCON_512 = "Falcon-512"
	FALCON_1024 = "Falcon-1024"


class KeyUsage(str, Enum):
	"""Key usage purposes"""
	ENCRYPT = "encrypt"
	DECRYPT = "decrypt" 
	SIGN = "sign"
	VERIFY = "verify"
	KEY_WRAP = "key_wrap"
	KEY_UNWRAP = "key_unwrap"
	DERIVE = "derive"
	MAC = "mac"
	
	
class KeyState(str, Enum):
	"""Key lifecycle states"""
	PENDING = "pending"
	ACTIVE = "active"
	SUSPENDED = "suspended"
	DEACTIVATED = "deactivated"
	DESTROYED = "destroyed"
	ARCHIVED = "archived"
	COMPROMISED = "compromised"
	

class SecurityLevel(str, Enum):
	"""Security classification levels"""
	PUBLIC = "public"
	INTERNAL = "internal"
	CONFIDENTIAL = "confidential"
	RESTRICTED = "restricted"
	TOP_SECRET = "top_secret"


class ComplianceFramework(str, Enum):
	"""Supported compliance frameworks"""
	FIPS_140_2 = "FIPS_140_2"
	COMMON_CRITERIA = "Common_Criteria"
	GDPR = "GDPR"
	HIPAA = "HIPAA"
	PCI_DSS = "PCI_DSS"
	SOX = "SOX"
	ISO_27001 = "ISO_27001"


class HSMType(str, Enum):
	"""Hardware Security Module types"""
	SOFTWARE = "software"
	NETWORK_HSM = "network_hsm"
	PCIe_HSM = "pcie_hsm"
	USB_HSM = "usb_hsm"
	CLOUD_HSM = "cloud_hsm"
	TPM = "tpm"
	SECURE_ENCLAVE = "secure_enclave"


def validate_key_size(algorithm: KeyAlgorithm, key_size: int) -> int:
	"""Validate key size for given algorithm"""
	valid_sizes = {
		KeyAlgorithm.AES_128: [128],
		KeyAlgorithm.AES_256: [256],
		KeyAlgorithm.RSA_2048: [2048],
		KeyAlgorithm.RSA_4096: [4096],
		KeyAlgorithm.ECDSA_P256: [256],
		KeyAlgorithm.ECDSA_P384: [384],
		KeyAlgorithm.ED25519: [255],  # Fixed size for Ed25519
		KeyAlgorithm.KYBER_512: [512],
		KeyAlgorithm.KYBER_768: [768],
		KeyAlgorithm.KYBER_1024: [1024]
	}
	
	if algorithm in valid_sizes and key_size in valid_sizes[algorithm]:
		return key_size
	raise ValueError(f"Invalid key size {key_size} for algorithm {algorithm}")


def validate_tenant_id(tenant_id: str) -> str:
	"""Validate APG tenant identifier"""
	if not tenant_id or len(tenant_id) < 3:
		raise ValueError("Tenant ID must be at least 3 characters")
	if not tenant_id.replace('_', '').replace('-', '').isalnum():
		raise ValueError("Tenant ID must be alphanumeric with optional hyphens/underscores")
	return tenant_id


class KeyMetadata(BaseModel):
	"""Key metadata and attributes"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	name: str = Field(..., max_length=255, description="Human-readable key name")
	description: str | None = Field(default=None, max_length=1000, description="Key description")
	tags: Dict[str, str] = Field(default_factory=dict, description="Key tags and labels")
	cost_center: str | None = Field(default=None, max_length=100, description="Cost center for billing")
	project_id: str | None = Field(default=None, max_length=100, description="Associated project")
	environment: str | None = Field(default=None, max_length=50, description="Environment (dev/staging/prod)")
	owner: str | None = Field(default=None, max_length=255, description="Key owner/creator")


class KeyPolicy(BaseModel):
	"""Key usage and access policies"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	# Access control
	allowed_users: List[str] = Field(default_factory=list, description="Allowed user IDs")
	allowed_roles: List[str] = Field(default_factory=list, description="Allowed role names")
	allowed_applications: List[str] = Field(default_factory=list, description="Allowed application IDs")
	
	# Usage constraints
	usage_restrictions: List[KeyUsage] = Field(default_factory=list, description="Allowed key operations")
	ip_whitelist: List[str] = Field(default_factory=list, description="Allowed IP addresses/ranges")
	time_restrictions: Dict[str, Any] = Field(default_factory=dict, description="Time-based access restrictions")
	geographic_restrictions: List[str] = Field(default_factory=list, description="Geographic restrictions")
	
	# Lifecycle policies
	auto_rotate: bool = Field(default=True, description="Enable automatic key rotation")
	rotation_interval_days: int = Field(default=90, ge=1, le=3650, description="Rotation interval in days")
	max_usage_count: int | None = Field(default=None, ge=1, description="Maximum usage count before rotation")
	expiry_date: datetime | None = Field(default=None, description="Key expiration date")
	
	# Compliance requirements
	compliance_frameworks: List[ComplianceFramework] = Field(default_factory=list, description="Required compliance frameworks")
	require_mfa: bool = Field(default=True, description="Require multi-factor authentication")
	require_approval: bool = Field(default=False, description="Require approval for key operations")
	
	# Security policies
	min_security_level: SecurityLevel = Field(default=SecurityLevel.INTERNAL, description="Minimum security classification")
	require_hsm: bool = Field(default=False, description="Require hardware security module")
	allow_export: bool = Field(default=False, description="Allow key export")


class KeySpec(BaseModel):
	"""Key specification for creation"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Unique key identifier")
	tenant_id: Annotated[str, AfterValidator(validate_tenant_id)] = Field(..., description="APG tenant identifier")
	
	# Key properties
	algorithm: KeyAlgorithm = Field(..., description="Cryptographic algorithm")
	key_size: int = Field(..., gt=0, description="Key size in bits")
	usage: List[KeyUsage] = Field(..., min_length=1, description="Intended key usage")
	
	# Metadata
	metadata: KeyMetadata = Field(..., description="Key metadata and attributes")
	policy: KeyPolicy = Field(..., description="Key usage and access policies")
	
	# Security settings
	security_level: SecurityLevel = Field(default=SecurityLevel.INTERNAL, description="Security classification")
	hsm_type: HSMType = Field(default=HSMType.SOFTWARE, description="Hardware security module type")
	
	# Lifecycle
	state: KeyState = Field(default=KeyState.PENDING, description="Current key state")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
	created_by: str = Field(..., description="Creator user ID")
	
	@field_validator('key_size')
	@classmethod
	def validate_key_size_for_algorithm(cls, v: int, info) -> int:
		"""Validate key size matches algorithm requirements"""
		if 'algorithm' in info.data:
			return validate_key_size(info.data['algorithm'], v)
		return v


class Key(BaseModel):
	"""Complete key with cryptographic material"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	# Base specification
	spec: KeySpec = Field(..., description="Key specification")
	
	# Cryptographic material (encrypted at rest)
	key_material: bytes | None = Field(default=None, description="Encrypted key material")
	public_key: bytes | None = Field(default=None, description="Public key (for asymmetric keys)")
	key_checksum: str | None = Field(default=None, description="Key integrity checksum")
	
	# HSM information
	hsm_key_id: str | None = Field(default=None, description="HSM key identifier")
	hsm_session_id: str | None = Field(default=None, description="HSM session identifier")
	
	# Usage statistics
	usage_count: int = Field(default=0, ge=0, description="Number of times key has been used")
	last_used: datetime | None = Field(default=None, description="Last usage timestamp")
	
	# Rotation history
	previous_versions: List[str] = Field(default_factory=list, description="Previous key version IDs")
	next_rotation: datetime | None = Field(default=None, description="Scheduled rotation time")
	
	# Backup and recovery
	backup_status: str = Field(default="pending", description="Backup status")
	recovery_shares: List[str] = Field(default_factory=list, description="Secret sharing recovery shares")


class KeyOperation(BaseModel):
	"""Key operation request/response"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	operation_id: str = Field(default_factory=uuid7str, description="Operation identifier")
	key_id: str = Field(..., description="Target key identifier")
	operation_type: str = Field(..., description="Type of operation")
	
	# Request data
	data: bytes | None = Field(default=None, description="Operation input data")
	parameters: Dict[str, Any] = Field(default_factory=dict, description="Operation parameters")
	
	# Context
	user_id: str = Field(..., description="Requesting user ID")
	application_id: str | None = Field(default=None, description="Requesting application ID")
	request_ip: str | None = Field(default=None, description="Request IP address")
	session_id: str | None = Field(default=None, description="User session ID")
	
	# Timestamps
	requested_at: datetime = Field(default_factory=datetime.utcnow, description="Request timestamp")
	completed_at: datetime | None = Field(default=None, description="Completion timestamp")
	
	# Results
	success: bool = Field(default=False, description="Operation success status")
	result_data: bytes | None = Field(default=None, description="Operation result data")
	error_message: str | None = Field(default=None, description="Error message if failed")


class SecurityThreat(BaseModel):
	"""Security threat detection"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	threat_id: str = Field(default_factory=uuid7str, description="Threat identifier")
	tenant_id: str = Field(..., description="APG tenant identifier")
	
	# Threat details
	threat_type: str = Field(..., description="Type of security threat")
	severity: str = Field(..., description="Threat severity level")
	confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
	
	# Associated resources
	affected_keys: List[str] = Field(default_factory=list, description="Affected key IDs")
	source_ip: str | None = Field(default=None, description="Threat source IP")
	user_id: str | None = Field(default=None, description="Associated user ID")
	
	# Detection details
	detected_at: datetime = Field(default_factory=datetime.utcnow, description="Detection timestamp")
	detection_method: str = Field(..., description="Detection method used")
	indicators: Dict[str, Any] = Field(default_factory=dict, description="Threat indicators")
	
	# Response
	status: str = Field(default="new", description="Threat response status")
	response_actions: List[str] = Field(default_factory=list, description="Automated response actions")
	resolved_at: datetime | None = Field(default=None, description="Resolution timestamp")


class AuditEvent(BaseModel):
	"""Audit event for key operations"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	event_id: str = Field(default_factory=uuid7str, description="Event identifier")
	tenant_id: str = Field(..., description="APG tenant identifier")
	
	# Event details
	event_type: str = Field(..., description="Type of audit event")
	resource_type: str = Field(..., description="Resource type (key, policy, etc.)")
	resource_id: str = Field(..., description="Resource identifier")
	
	# Actor information
	user_id: str | None = Field(default=None, description="Acting user ID")
	application_id: str | None = Field(default=None, description="Acting application ID")
	session_id: str | None = Field(default=None, description="Session identifier")
	
	# Request context
	source_ip: str | None = Field(default=None, description="Source IP address")
	user_agent: str | None = Field(default=None, description="User agent string")
	request_id: str | None = Field(default=None, description="Request identifier")
	
	# Event data
	action: str = Field(..., description="Action performed")
	outcome: str = Field(..., description="Action outcome")
	details: Dict[str, Any] = Field(default_factory=dict, description="Additional event details")
	
	# Timestamps
	timestamp: datetime = Field(default_factory=datetime.utcnow, description="Event timestamp")
	
	# Compliance
	compliance_frameworks: List[ComplianceFramework] = Field(default_factory=list, description="Applicable compliance frameworks")
	retention_period_days: int = Field(default=2555, description="Audit retention period")


class HSMConfiguration(BaseModel):
	"""Hardware Security Module configuration"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	hsm_id: str = Field(default_factory=uuid7str, description="HSM identifier")
	tenant_id: str = Field(..., description="APG tenant identifier")
	
	# HSM details
	hsm_type: HSMType = Field(..., description="HSM type")
	vendor: str = Field(..., description="HSM vendor")
	model: str = Field(..., description="HSM model")
	serial_number: str | None = Field(default=None, description="HSM serial number")
	
	# Connection
	endpoint: str | None = Field(default=None, description="HSM endpoint URL")
	port: int | None = Field(default=None, ge=1, le=65535, description="HSM port")
	auth_method: str = Field(..., description="Authentication method")
	credentials: Dict[str, str] = Field(default_factory=dict, description="HSM credentials (encrypted)")
	
	# Configuration
	partition_name: str | None = Field(default=None, description="HSM partition name")
	slot_id: int | None = Field(default=None, ge=0, description="HSM slot ID")
	
	# Status
	status: str = Field(default="disconnected", description="HSM connection status")
	last_health_check: datetime | None = Field(default=None, description="Last health check")
	health_status: str = Field(default="unknown", description="HSM health status")
	
	# Capabilities
	supported_algorithms: List[KeyAlgorithm] = Field(default_factory=list, description="Supported algorithms")
	max_key_count: int | None = Field(default=None, ge=1, description="Maximum key count")
	current_key_count: int = Field(default=0, ge=0, description="Current key count")
	
	# Performance
	operations_per_second: int | None = Field(default=None, ge=1, description="Operations per second capacity")
	average_latency_ms: float | None = Field(default=None, ge=0, description="Average operation latency")


class CloudKeyStore(BaseModel):
	"""Cloud key store configuration"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	store_id: str = Field(default_factory=uuid7str, description="Key store identifier")
	tenant_id: str = Field(..., description="APG tenant identifier")
	
	# Cloud provider
	provider: str = Field(..., description="Cloud provider (aws, azure, gcp)")
	region: str = Field(..., description="Cloud region")
	service_name: str = Field(..., description="Cloud key service name")
	
	# Configuration
	key_vault_url: str | None = Field(default=None, description="Key vault URL")
	vault_name: str | None = Field(default=None, description="Vault name")
	resource_group: str | None = Field(default=None, description="Azure resource group")
	subscription_id: str | None = Field(default=None, description="Azure subscription ID")
	
	# Authentication
	auth_method: str = Field(..., description="Authentication method")
	credentials: Dict[str, str] = Field(default_factory=dict, description="Cloud credentials (encrypted)")
	
	# Status
	status: str = Field(default="disconnected", description="Connection status")
	last_sync: datetime | None = Field(default=None, description="Last synchronization")
	
	# Capabilities
	supported_algorithms: List[KeyAlgorithm] = Field(default_factory=list, description="Supported algorithms")
	supports_hsm: bool = Field(default=False, description="Supports hardware backing")


class KeyUsageStats(BaseModel):
	"""Key usage statistics"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	key_id: str = Field(..., description="Key identifier")
	tenant_id: str = Field(..., description="APG tenant identifier")
	
	# Usage counters
	total_operations: int = Field(default=0, ge=0, description="Total operations")
	encrypt_operations: int = Field(default=0, ge=0, description="Encryption operations")
	decrypt_operations: int = Field(default=0, ge=0, description="Decryption operations")
	sign_operations: int = Field(default=0, ge=0, description="Signing operations")
	verify_operations: int = Field(default=0, ge=0, description="Verification operations")
	
	# Time periods
	daily_operations: Dict[str, int] = Field(default_factory=dict, description="Daily operation counts")
	monthly_operations: Dict[str, int] = Field(default_factory=dict, description="Monthly operation counts")
	
	# Performance metrics
	average_latency_ms: float = Field(default=0.0, ge=0, description="Average operation latency")
	success_rate: float = Field(default=1.0, ge=0, le=1.0, description="Operation success rate")
	
	# User statistics
	unique_users: int = Field(default=0, ge=0, description="Number of unique users")
	unique_applications: int = Field(default=0, ge=0, description="Number of unique applications")
	
	# Timestamps
	first_used: datetime | None = Field(default=None, description="First usage timestamp")
	last_used: datetime | None = Field(default=None, description="Last usage timestamp")
	stats_generated_at: datetime = Field(default_factory=datetime.utcnow, description="Statistics generation time")


# Helper functions for async operations
async def _log_model_operation(operation: str, model_name: str, model_id: str) -> None:
	"""Log model operations for APG audit compliance"""
	print(f"[KEYM] {operation} {model_name} {model_id} at {datetime.utcnow()}")


async def create_key_spec_async(
	tenant_id: str,
	algorithm: KeyAlgorithm,
	usage: List[KeyUsage],
	name: str,
	created_by: str,
	**kwargs
) -> KeySpec:
	"""Async factory for creating key specifications"""
	await _log_model_operation("CREATE", "KeySpec", "pending")
	
	# Determine appropriate key size for algorithm
	default_sizes = {
		KeyAlgorithm.AES_128: 128,
		KeyAlgorithm.AES_256: 256,
		KeyAlgorithm.RSA_2048: 2048,
		KeyAlgorithm.RSA_4096: 4096,
		KeyAlgorithm.ECDSA_P256: 256,
		KeyAlgorithm.ECDSA_P384: 384,
		KeyAlgorithm.ED25519: 255,
		KeyAlgorithm.KYBER_512: 512,
		KeyAlgorithm.KYBER_768: 768,
		KeyAlgorithm.KYBER_1024: 1024
	}
	
	key_size = kwargs.get('key_size', default_sizes.get(algorithm, 256))
	
	metadata = KeyMetadata(
		name=name,
		description=kwargs.get('description'),
		tags=kwargs.get('tags', {}),
		owner=created_by
	)
	
	policy = KeyPolicy(
		usage_restrictions=usage,
		auto_rotate=kwargs.get('auto_rotate', True),
		rotation_interval_days=kwargs.get('rotation_interval_days', 90),
		require_mfa=kwargs.get('require_mfa', True)
	)
	
	spec = KeySpec(
		tenant_id=tenant_id,
		algorithm=algorithm,
		key_size=key_size,
		usage=usage,
		metadata=metadata,
		policy=policy,
		created_by=created_by,
		**{k: v for k, v in kwargs.items() if k in KeySpec.model_fields}
	)
	
	await _log_model_operation("CREATED", "KeySpec", spec.id)
	return spec


# Export all models for APG integration
__all__ = [
	"KeyAlgorithm", "KeyUsage", "KeyState", "SecurityLevel", "ComplianceFramework", "HSMType",
	"KeyMetadata", "KeyPolicy", "KeySpec", "Key", "KeyOperation", "SecurityThreat", "AuditEvent",
	"HSMConfiguration", "CloudKeyStore", "KeyUsageStats",
	"create_key_spec_async", "validate_key_size", "validate_tenant_id"
]