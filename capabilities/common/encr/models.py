"""
APG Encryption Services - Quantum-Safe Data Models

Revolutionary Pydantic v2 models for quantum-safe encryption with zero-knowledge
architecture, autonomous key management, and APG multi-tenant integration.

This module provides comprehensive data models for:
- Post-quantum cryptographic operations (CRYSTALS-Kyber, CRYSTALS-Dilithium)
- Zero-knowledge encryption architecture with privacy preservation
- Autonomous AI-driven key lifecycle management
- Multi-tenant isolation with shared threat intelligence
- APG capability integration patterns

APG Standards Compliance:
- Async Python with modern typing (str | None, list[str], dict[str, Any])
- Tabs for indentation (NEVER spaces)
- uuid7str for all ID fields
- Pydantic v2 with ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
- Comprehensive validation with Annotated[..., AfterValidator(...)]
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Annotated
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict, AfterValidator
from uuid_extensions import uuid7str


# APG Standards Configuration
MODEL_CONFIG = ConfigDict(
	extra='forbid', 
	validate_by_name=True, 
	validate_by_alias=True,
	use_enum_values=True,
	validate_assignment=True
)


# Quantum-Safe Algorithm Enums
class PostQuantumAlgorithm(str, Enum):
	"""NIST standardized post-quantum cryptographic algorithms"""
	CRYSTALS_KYBER_512 = "crystals-kyber-512"
	CRYSTALS_KYBER_768 = "crystals-kyber-768"  
	CRYSTALS_KYBER_1024 = "crystals-kyber-1024"
	CRYSTALS_DILITHIUM_2 = "crystals-dilithium-2"
	CRYSTALS_DILITHIUM_3 = "crystals-dilithium-3"
	CRYSTALS_DILITHIUM_5 = "crystals-dilithium-5"
	FALCON_512 = "falcon-512"
	FALCON_1024 = "falcon-1024"
	SPHINCS_PLUS_128S = "sphincs-plus-128s"
	SPHINCS_PLUS_256S = "sphincs-plus-256s"


class EncryptionMode(str, Enum):
	"""Encryption operation modes"""
	QUANTUM_SAFE = "quantum-safe"
	ZERO_KNOWLEDGE = "zero-knowledge"  
	HOMOMORPHIC = "homomorphic"
	NEUROMORPHIC = "neuromorphic"
	THRESHOLD = "threshold"
	HYBRID_CLASSICAL_QUANTUM = "hybrid-classical-quantum"


class KeyLifecycleState(str, Enum):
	"""AI-managed key lifecycle states"""
	GENERATING = "generating"
	ACTIVE = "active"
	ROTATION_SCHEDULED = "rotation-scheduled"
	ROTATING = "rotating"
	ESCROW = "escrow"
	DEPRECATED = "deprecated"
	DESTROYED = "destroyed"
	QUANTUM_UPGRADING = "quantum-upgrading"


class SecurityLevel(int, Enum):
	"""NIST post-quantum security levels"""
	LEVEL_1 = 1  # Classical security equivalent to AES-128
	LEVEL_2 = 2  # Classical security equivalent to SHA-256  
	LEVEL_3 = 3  # Classical security equivalent to AES-192
	LEVEL_4 = 4  # Classical security equivalent to SHA-384
	LEVEL_5 = 5  # Classical security equivalent to AES-256


class ThreatLevel(str, Enum):
	"""AI-assessed threat levels for adaptive encryption"""
	MINIMAL = "minimal"
	LOW = "low"
	MODERATE = "moderate"
	HIGH = "high"
	CRITICAL = "critical"
	QUANTUM_IMMINENT = "quantum-imminent"


class ComplianceFramework(str, Enum):
	"""Supported regulatory compliance frameworks"""
	GDPR = "gdpr"
	HIPAA = "hipaa"
	PCI_DSS = "pci-dss"
	SOX = "sox"
	ISO_27001 = "iso-27001"
	FIPS_140_2 = "fips-140-2"
	COMMON_CRITERIA = "common-criteria"
	NIST_CYBERSECURITY = "nist-cybersecurity"


# Validation Functions
def validate_tenant_id(tenant_id: str) -> str:
	"""Validate APG tenant identifier format"""
	if not tenant_id or len(tenant_id) < 8 or not tenant_id.isalnum():
		raise ValueError("Tenant ID must be alphanumeric and at least 8 characters")
	return tenant_id


def validate_entropy_quality(quality: float) -> float:
	"""Validate quantum entropy quality score"""
	if not 0.0 <= quality <= 1.0:
		raise ValueError("Entropy quality must be between 0.0 and 1.0")
	return quality


def validate_key_size(key_size: int) -> int:
	"""Validate cryptographic key size"""
	valid_sizes = {128, 192, 256, 384, 512, 1024, 2048, 3072, 4096}
	if key_size not in valid_sizes:
		raise ValueError(f"Key size must be one of: {valid_sizes}")
	return key_size


# Core Quantum-Safe Models

class QuantumEntropySource(BaseModel):
	"""Quantum entropy source for true randomness generation"""
	model_config = MODEL_CONFIG
	
	id: str = Field(default_factory=uuid7str, description="Entropy source identifier")
	tenant_id: Annotated[str, AfterValidator(validate_tenant_id)] = Field(..., description="APG tenant identifier")
	source_type: str = Field(..., description="Entropy source type (photonic, electronic, atmospheric, cosmic)")
	location: str = Field(..., description="Physical location of entropy source")
	quality_score: Annotated[float, AfterValidator(validate_entropy_quality)] = Field(..., description="Entropy quality (0.0-1.0)")
	last_harvest_at: datetime = Field(..., description="Last entropy collection timestamp")
	harvest_rate_mbps: float = Field(..., description="Entropy generation rate in Mbps")
	is_active: bool = Field(default=True, description="Source availability status")
	quantum_noise_level: float = Field(..., description="Quantum noise measurement")
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class PostQuantumKeyPair(BaseModel):
	"""Post-quantum cryptographic key pair"""
	model_config = MODEL_CONFIG
	
	id: str = Field(default_factory=uuid7str, description="Key pair identifier")
	tenant_id: Annotated[str, AfterValidator(validate_tenant_id)] = Field(..., description="APG tenant identifier")
	algorithm: PostQuantumAlgorithm = Field(..., description="Post-quantum algorithm used")
	security_level: SecurityLevel = Field(..., description="NIST security level")
	
	# CRYSTALS-Kyber KEM keys
	kyber_public_key: bytes = Field(..., description="Kyber public key for key encapsulation")
	kyber_secret_key: bytes = Field(..., description="Kyber secret key (encrypted at rest)")
	kyber_ciphertext: Optional[bytes] = Field(default=None, description="Encapsulated shared secret")
	
	# CRYSTALS-Dilithium signature keys  
	dilithium_public_key: bytes = Field(..., description="Dilithium public key for signatures")
	dilithium_secret_key: bytes = Field(..., description="Dilithium secret key (encrypted at rest)")
	
	# Key metadata
	key_size: Annotated[int, AfterValidator(validate_key_size)] = Field(..., description="Key size in bits")
	entropy_source_id: str = Field(..., description="Quantum entropy source used")
	generation_context: Dict[str, Any] = Field(default_factory=dict, description="Key generation context")
	
	# Lifecycle management
	state: KeyLifecycleState = Field(default=KeyLifecycleState.GENERATING, description="Current lifecycle state")
	autonomous_management: bool = Field(default=True, description="AI-managed lifecycle enabled")
	last_rotation: Optional[datetime] = Field(default=None, description="Last key rotation timestamp")
	next_rotation: Optional[datetime] = Field(default=None, description="Scheduled next rotation")
	rotation_frequency_days: int = Field(default=90, description="Rotation frequency in days")
	
	# Security and compliance
	zero_knowledge_protected: bool = Field(default=True, description="Zero-knowledge protection enabled")
	threshold_shares: Optional[int] = Field(default=None, description="Threshold cryptography shares")
	compliance_frameworks: List[ComplianceFramework] = Field(default_factory=list, description="Applicable compliance requirements")
	
	created_at: datetime = Field(default_factory=datetime.utcnow)
	expires_at: Optional[datetime] = Field(default=None)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class QuantumSafeSession(BaseModel):
	"""Quantum-safe cryptographic session"""
	model_config = MODEL_CONFIG
	
	id: str = Field(default_factory=uuid7str, description="Session identifier") 
	tenant_id: Annotated[str, AfterValidator(validate_tenant_id)] = Field(..., description="APG tenant identifier")
	user_id: str = Field(..., description="User identifier from APG auth")
	device_id: str = Field(..., description="Device identifier")
	
	# Session cryptography
	session_key: bytes = Field(..., description="Quantum-safe session key")
	key_pair_id: str = Field(..., description="Associated post-quantum key pair")
	encryption_mode: EncryptionMode = Field(..., description="Session encryption mode")
	
	# Zero-knowledge architecture
	client_key_share: bytes = Field(..., description="Client-side key share")
	server_key_share: bytes = Field(..., description="Server-side key share")
	threshold_required: int = Field(default=2, description="Threshold shares required for decryption")
	
	# Session security
	threat_level: ThreatLevel = Field(default=ThreatLevel.LOW, description="Current assessed threat level")
	adaptive_algorithm: PostQuantumAlgorithm = Field(..., description="Threat-adapted algorithm")
	quantum_safe_level: SecurityLevel = Field(..., description="Quantum safety level")
	
	# Session lifecycle
	is_active: bool = Field(default=True, description="Session activity status")
	last_activity: datetime = Field(default_factory=datetime.utcnow, description="Last session activity")
	session_timeout_minutes: int = Field(default=60, description="Session timeout in minutes")
	
	created_at: datetime = Field(default_factory=datetime.utcnow)
	expires_at: datetime = Field(..., description="Session expiration timestamp")


class ZeroKnowledgeProof(BaseModel):
	"""Zero-knowledge proof for privacy-preserving access control"""
	model_config = MODEL_CONFIG
	
	id: str = Field(default_factory=uuid7str, description="Proof identifier")
	tenant_id: Annotated[str, AfterValidator(validate_tenant_id)] = Field(..., description="APG tenant identifier")
	session_id: str = Field(..., description="Associated session identifier")
	
	# Proof components
	proof_data: bytes = Field(..., description="Zero-knowledge proof data")
	verification_key: bytes = Field(..., description="Verification key (public)")
	commitment: bytes = Field(..., description="Cryptographic commitment")
	challenge: bytes = Field(..., description="Challenge from verifier")
	response: bytes = Field(..., description="Prover response")
	
	# Proof metadata
	proof_system: str = Field(default="groth16", description="Zero-knowledge proof system used")
	circuit_hash: str = Field(..., description="Hash of the proving circuit")
	public_inputs: List[str] = Field(default_factory=list, description="Public inputs to the proof")
	
	# Verification status
	is_verified: bool = Field(default=False, description="Proof verification status")
	verified_at: Optional[datetime] = Field(default=None, description="Verification timestamp")
	verification_context: Dict[str, Any] = Field(default_factory=dict, description="Verification context")
	
	created_at: datetime = Field(default_factory=datetime.utcnow)
	expires_at: datetime = Field(..., description="Proof validity expiration")


class HomomorphicCiphertext(BaseModel):
	"""Ciphertext for homomorphic computation on encrypted data"""
	model_config = MODEL_CONFIG
	
	id: str = Field(default_factory=uuid7str, description="Ciphertext identifier")
	tenant_id: Annotated[str, AfterValidator(validate_tenant_id)] = Field(..., description="APG tenant identifier")
	session_id: str = Field(..., description="Associated session identifier")
	
	# Homomorphic encryption data
	ciphertext_data: bytes = Field(..., description="Encrypted data for homomorphic computation")
	scheme: str = Field(default="ckks", description="Homomorphic encryption scheme (BFV, BGV, CKKS)")
	parameters: Dict[str, Any] = Field(..., description="Encryption parameters")
	
	# Computation metadata
	computation_context: str = Field(..., description="Computation context identifier")
	data_type: str = Field(..., description="Original data type (integer, float, vector)")
	data_size: int = Field(..., description="Original data size in bytes")
	noise_level: float = Field(..., description="Current noise level")
	
	# Operations tracking
	operations_performed: List[str] = Field(default_factory=list, description="Homomorphic operations performed")
	operation_count: int = Field(default=0, description="Total number of operations")
	max_operations: int = Field(default=1000, description="Maximum allowed operations")
	
	created_at: datetime = Field(default_factory=datetime.utcnow)
	expires_at: datetime = Field(..., description="Ciphertext validity expiration")


class AutonomousKeyDecision(BaseModel):
	"""AI-driven autonomous key management decision"""
	model_config = MODEL_CONFIG
	
	id: str = Field(default_factory=uuid7str, description="Decision identifier")
	tenant_id: Annotated[str, AfterValidator(validate_tenant_id)] = Field(..., description="APG tenant identifier")
	key_pair_id: str = Field(..., description="Target key pair identifier")
	
	# Decision context
	decision_type: str = Field(..., description="Type of autonomous decision")
	confidence_score: float = Field(..., description="AI confidence in decision (0.0-1.0)")
	reasoning: Dict[str, Any] = Field(..., description="AI reasoning for decision")
	
	# Analysis inputs
	usage_patterns: Dict[str, Any] = Field(..., description="Key usage pattern analysis")
	security_assessment: Dict[str, Any] = Field(..., description="Security context assessment")
	threat_intelligence: Dict[str, Any] = Field(..., description="Current threat intelligence")
	compliance_requirements: List[ComplianceFramework] = Field(default_factory=list, description="Compliance requirements")
	
	# Recommended actions
	should_rotate: bool = Field(default=False, description="Recommend key rotation")
	should_backup: bool = Field(default=False, description="Recommend key backup")
	should_destroy: bool = Field(default=False, description="Recommend key destruction")
	should_upgrade_quantum: bool = Field(default=False, description="Recommend quantum-safe upgrade")
	
	# Action timing
	recommended_execution_time: datetime = Field(..., description="Recommended execution timestamp")
	priority_level: int = Field(default=5, description="Decision priority (1-10)")
	
	# Execution tracking
	is_executed: bool = Field(default=False, description="Decision execution status")
	executed_at: Optional[datetime] = Field(default=None, description="Execution timestamp")
	execution_result: Optional[Dict[str, Any]] = Field(default=None, description="Execution result")
	
	created_at: datetime = Field(default_factory=datetime.utcnow)


class CryptographicPolicy(BaseModel):
	"""AI-generated cryptographic policy based on data context and threats"""
	model_config = MODEL_CONFIG
	
	id: str = Field(default_factory=uuid7str, description="Policy identifier")
	tenant_id: Annotated[str, AfterValidator(validate_tenant_id)] = Field(..., description="APG tenant identifier")
	
	# Policy definition
	policy_name: str = Field(..., description="Human-readable policy name")
	policy_version: str = Field(default="1.0.0", description="Policy version")
	policy_description: str = Field(..., description="Policy description and purpose")
	
	# Algorithm selection
	required_algorithm: PostQuantumAlgorithm = Field(..., description="Required cryptographic algorithm")
	fallback_algorithms: List[PostQuantumAlgorithm] = Field(default_factory=list, description="Fallback algorithms")
	minimum_security_level: SecurityLevel = Field(..., description="Minimum required security level")
	quantum_safe_required: bool = Field(default=True, description="Quantum-safe algorithms mandatory")
	
	# Key management requirements
	key_rotation_interval_days: int = Field(..., description="Required key rotation interval")
	autonomous_management_required: bool = Field(default=True, description="AI management required")
	threshold_cryptography_shares: Optional[int] = Field(default=None, description="Threshold shares if required")
	
	# Compliance and regulatory requirements
	applicable_frameworks: List[ComplianceFramework] = Field(default_factory=list, description="Applicable compliance frameworks")
	data_residency_requirements: List[str] = Field(default_factory=list, description="Data residency constraints")
	retention_period_days: Optional[int] = Field(default=None, description="Data retention period")
	
	# Threat adaptation
	threat_adaptation_enabled: bool = Field(default=True, description="Real-time threat adaptation enabled")
	threat_response_sensitivity: float = Field(default=0.7, description="Threat response sensitivity (0.0-1.0)")
	quantum_threat_threshold: float = Field(default=0.8, description="Quantum threat activation threshold")
	
	# Performance requirements
	max_encryption_latency_ms: int = Field(default=100, description="Maximum allowed encryption latency")
	min_throughput_ops_per_sec: int = Field(default=1000, description="Minimum throughput requirement")
	
	# Audit and monitoring
	audit_level: str = Field(default="comprehensive", description="Required audit logging level")
	monitoring_enabled: bool = Field(default=True, description="Real-time monitoring enabled")
	
	# Policy lifecycle
	is_active: bool = Field(default=True, description="Policy active status")
	effective_from: datetime = Field(default_factory=datetime.utcnow, description="Policy effective date")
	effective_until: Optional[datetime] = Field(default=None, description="Policy expiration date")
	
	# AI policy generation metadata
	ai_generated: bool = Field(default=True, description="AI-generated policy flag")
	ai_confidence: float = Field(..., description="AI confidence in policy (0.0-1.0)")
	generation_context: Dict[str, Any] = Field(default_factory=dict, description="AI policy generation context")
	
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class ThreatIntelligence(BaseModel):
	"""Real-time threat intelligence for adaptive encryption"""
	model_config = MODEL_CONFIG
	
	id: str = Field(default_factory=uuid7str, description="Threat intelligence identifier")
	tenant_id: Annotated[str, AfterValidator(validate_tenant_id)] = Field(..., description="APG tenant identifier")
	
	# Threat assessment
	current_threat_level: ThreatLevel = Field(..., description="Overall assessed threat level")
	quantum_threat_probability: float = Field(..., description="Quantum computing threat probability (0.0-1.0)")
	nation_state_activity: bool = Field(default=False, description="Nation-state threat activity detected")
	
	# Threat indicators
	threat_sources: List[str] = Field(default_factory=list, description="Active threat sources")
	attack_vectors: List[str] = Field(default_factory=list, description="Observed attack vectors")
	targeted_algorithms: List[PostQuantumAlgorithm] = Field(default_factory=list, description="Algorithms under attack")
	
	# Intelligence sources
	intelligence_feeds: List[str] = Field(default_factory=list, description="Threat intelligence feed sources")
	last_feed_update: datetime = Field(..., description="Last threat feed update")
	confidence_score: float = Field(..., description="Intelligence confidence score (0.0-1.0)")
	
	# Adaptive recommendations
	recommended_algorithms: List[PostQuantumAlgorithm] = Field(default_factory=list, description="Recommended algorithms")
	recommended_security_level: SecurityLevel = Field(..., description="Recommended security level")
	immediate_action_required: bool = Field(default=False, description="Immediate action required flag")
	
	# Geospatial context
	threat_geography: List[str] = Field(default_factory=list, description="Geographic threat origins")
	affected_regions: List[str] = Field(default_factory=list, description="Affected geographic regions")
	
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	expires_at: datetime = Field(..., description="Intelligence validity expiration")


class EncryptionOperation(BaseModel):
	"""Comprehensive encryption operation record"""
	model_config = MODEL_CONFIG
	
	id: str = Field(default_factory=uuid7str, description="Operation identifier")
	tenant_id: Annotated[str, AfterValidator(validate_tenant_id)] = Field(..., description="APG tenant identifier")
	session_id: str = Field(..., description="Associated session identifier")
	
	# Operation details
	operation_type: str = Field(..., description="Type of encryption operation")
	encryption_mode: EncryptionMode = Field(..., description="Encryption mode used")
	algorithm_used: PostQuantumAlgorithm = Field(..., description="Cryptographic algorithm used")
	
	# Data context
	data_size_bytes: int = Field(..., description="Size of encrypted data")
	data_classification: str = Field(..., description="Data sensitivity classification")
	data_context: Dict[str, Any] = Field(default_factory=dict, description="Additional data context")
	
	# Performance metrics
	operation_latency_ms: float = Field(..., description="Operation latency in milliseconds")
	throughput_mbps: float = Field(..., description="Throughput in Mbps")
	cpu_usage_percent: float = Field(..., description="CPU usage percentage")
	memory_usage_mb: float = Field(..., description="Memory usage in MB")
	
	# Security context
	threat_level_at_operation: ThreatLevel = Field(..., description="Threat level during operation")
	security_level_achieved: SecurityLevel = Field(..., description="Security level achieved")
	zero_knowledge_proof_id: Optional[str] = Field(default=None, description="Associated ZK proof")
	
	# Quality assurance
	entropy_quality: Annotated[float, AfterValidator(validate_entropy_quality)] = Field(..., description="Entropy quality used")
	validation_passed: bool = Field(..., description="Operation validation result")
	error_details: Optional[str] = Field(default=None, description="Error details if failed")
	
	# Audit and compliance
	compliance_frameworks_met: List[ComplianceFramework] = Field(default_factory=list, description="Compliance frameworks satisfied")
	audit_trail_id: str = Field(..., description="Associated audit trail identifier")
	
	# Neuromorphic processing (if used)
	neuromorphic_processing_used: bool = Field(default=False, description="Neuromorphic processing utilized")
	neuromorphic_latency_ns: Optional[float] = Field(default=None, description="Neuromorphic processing latency")
	energy_consumption_pj: Optional[float] = Field(default=None, description="Energy consumption in picojoules")
	
	created_at: datetime = Field(default_factory=datetime.utcnow)
	completed_at: datetime = Field(..., description="Operation completion timestamp")


# APG Integration Models

class APGEncryptionContext(BaseModel):
	"""Comprehensive encryption context for APG capability integration"""
	model_config = MODEL_CONFIG
	
	id: str = Field(default_factory=uuid7str, description="Context identifier")
	tenant_id: Annotated[str, AfterValidator(validate_tenant_id)] = Field(..., description="APG tenant identifier")
	
	# APG capability context
	requesting_capability: str = Field(..., description="APG capability requesting encryption")
	capability_version: str = Field(..., description="Capability version")
	integration_context: Dict[str, Any] = Field(default_factory=dict, description="Integration-specific context")
	
	# Authentication context (from auth capability)
	user_context: Dict[str, Any] = Field(default_factory=dict, description="User authentication context")
	session_context: Dict[str, Any] = Field(default_factory=dict, description="Session context")
	rbac_context: Dict[str, Any] = Field(default_factory=dict, description="RBAC authorization context")
	
	# Security framework context (from secu capability)  
	security_assessment: Dict[str, Any] = Field(default_factory=dict, description="Security framework assessment")
	risk_score: float = Field(default=0.5, description="Current risk score from security framework")
	threat_context: Dict[str, Any] = Field(default_factory=dict, description="Threat intelligence context")
	
	# Audit context (from audl capability)
	audit_requirements: List[str] = Field(default_factory=list, description="Audit logging requirements")
	compliance_context: Dict[str, Any] = Field(default_factory=dict, description="Compliance audit context")
	
	# Performance context
	performance_requirements: Dict[str, Any] = Field(default_factory=dict, description="Performance requirements")
	latency_budget_ms: int = Field(default=100, description="Allocated latency budget")
	
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# Response Models

class QuantumSafeEncryptionResult(BaseModel):
	"""Result of quantum-safe encryption operation"""
	model_config = MODEL_CONFIG
	
	operation_id: str = Field(..., description="Unique operation identifier")
	encrypted_data: bytes = Field(..., description="Quantum-safe encrypted data")
	algorithm_used: PostQuantumAlgorithm = Field(..., description="Algorithm used for encryption")
	security_level: SecurityLevel = Field(..., description="Achieved security level")
	session_id: str = Field(..., description="Associated session identifier")
	zero_knowledge_proof_id: Optional[str] = Field(default=None, description="ZK proof if applicable")
	performance_metrics: Dict[str, Any] = Field(default_factory=dict, description="Operation performance data")
	compliance_evidence: List[ComplianceFramework] = Field(default_factory=list, description="Compliance frameworks met")
	created_at: datetime = Field(default_factory=datetime.utcnow)


class ZeroKnowledgeEncryptionResult(BaseModel):
	"""Result of zero-knowledge encryption operation"""
	model_config = MODEL_CONFIG
	
	operation_id: str = Field(..., description="Unique operation identifier")
	encrypted_data: bytes = Field(..., description="Zero-knowledge encrypted data")
	access_proof: ZeroKnowledgeProof = Field(..., description="Zero-knowledge access proof")
	threshold_shares: List[bytes] = Field(..., description="Threshold cryptography shares")
	privacy_guarantee_level: float = Field(..., description="Mathematical privacy guarantee level")
	session_id: str = Field(..., description="Associated session identifier")
	created_at: datetime = Field(default_factory=datetime.utcnow)


class HomomorphicEncryptionResult(BaseModel):
	"""Result of homomorphic encryption operation"""
	model_config = MODEL_CONFIG
	
	operation_id: str = Field(..., description="Unique operation identifier")
	homomorphic_ciphertext: HomomorphicCiphertext = Field(..., description="Homomorphic ciphertext data")
	computation_capability: List[str] = Field(..., description="Supported homomorphic operations")
	privacy_preservation_level: float = Field(..., description="Privacy preservation guarantee")
	performance_estimate: Dict[str, Any] = Field(default_factory=dict, description="Computation performance estimates")
	created_at: datetime = Field(default_factory=datetime.utcnow)


class AutonomousKeyManagementResult(BaseModel):
	"""Result of autonomous key management operation"""
	model_config = MODEL_CONFIG
	
	operation_id: str = Field(..., description="Unique operation identifier")
	decisions_made: List[AutonomousKeyDecision] = Field(..., description="AI-driven decisions made")
	keys_affected: List[str] = Field(..., description="Key pair IDs affected")
	actions_executed: List[str] = Field(..., description="Actions executed")
	ai_confidence: float = Field(..., description="Overall AI confidence in decisions")
	next_analysis_scheduled: datetime = Field(..., description="Next autonomous analysis timestamp")
	created_at: datetime = Field(default_factory=datetime.utcnow)


# Export all models for APG composition engine
__all__ = [
	# Core enums
	"PostQuantumAlgorithm",
	"EncryptionMode", 
	"KeyLifecycleState",
	"SecurityLevel",
	"ThreatLevel",
	"ComplianceFramework",
	
	# Core models
	"QuantumEntropySource",
	"PostQuantumKeyPair", 
	"QuantumSafeSession",
	"ZeroKnowledgeProof",
	"HomomorphicCiphertext",
	"AutonomousKeyDecision",
	"CryptographicPolicy",
	"ThreatIntelligence",
	"EncryptionOperation",
	
	# APG integration models
	"APGEncryptionContext",
	
	# Result models
	"QuantumSafeEncryptionResult",
	"ZeroKnowledgeEncryptionResult", 
	"HomomorphicEncryptionResult",
	"AutonomousKeyManagementResult"
]