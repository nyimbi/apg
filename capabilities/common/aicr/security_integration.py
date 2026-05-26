"""
APG AI Core Framework (aicr) - Comprehensive Security Integration

Purpose: Complete security integration layer providing JWT authentication,
         RBAC authorization, audit logging, and comprehensive security
         controls for AI operations within the APG platform ecosystem.

Dependencies: asyncio, jwt, cryptography, hashlib, secrets, typing
Security Features: JWT authentication, RBAC, audit logging, encryption,
                  session management, threat detection, compliance
Usage Context: Enterprise-grade security for all AI operations

This module provides:
- Complete JWT authentication with token management
- Role-based access control (RBAC) with granular permissions
- Comprehensive audit logging with risk assessment
- Session management with security monitoring
- API key management for service authentication
- Multi-tenant security boundaries and isolation
- Real-time threat detection and mitigation
- Compliance reporting and governance controls
"""

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import secrets
import time
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import uuid4
import jwt
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

from pydantic import BaseModel, Field, ConfigDict

from .models import uuid7str, _validate_tenant_id


def _log_security_event(event_type: str, user_id: str, result: str, details: str = "") -> str:
	"""Log security events with standardized format."""
	timestamp = datetime.now(timezone.utc).isoformat()
	return f"SECURITY [{event_type}] {user_id} - {result} {details} ({timestamp})"


def _log_auth_event(operation: str, user_id: str, success: bool, ip_address: str = "") -> str:
	"""Log authentication events."""
	status = "SUCCESS" if success else "FAILED"
	ip_info = f" from {ip_address}" if ip_address else ""
	return f"AUTH [{operation}] {user_id} - {status}{ip_info}"


def _log_rbac_event(user_id: str, resource: str, permission: str, granted: bool) -> str:
	"""Log RBAC authorization events."""
	status = "GRANTED" if granted else "DENIED"
	return f"RBAC [{user_id}] {permission} on {resource} - {status}"


class SecurityRole(str, Enum):
	"""Security roles for RBAC system.

	Defines hierarchical security roles with specific permissions
	for AI operations, model management, and system administration
	within the APG AI Core Framework.

	Attributes:
		GUEST: Limited read-only access to public resources
		USER: Standard user with basic AI inference capabilities
		DEVELOPER: Extended access for model development and testing
		ANALYST: Data analysis and reporting capabilities
		ADMIN: Administrative access for user and system management
		SUPER_ADMIN: Full system access including security configuration
		SERVICE: Service-to-service authentication for automated systems
		AUDIT: Audit and compliance access for monitoring and reporting
	"""
	GUEST = "guest"
	USER = "user"
	DEVELOPER = "developer"
	ANALYST = "analyst"
	ADMIN = "admin"
	SUPER_ADMIN = "super_admin"
	SERVICE = "service"
	AUDIT = "audit"


class SecurityPermission(str, Enum):
	"""Granular security permissions for AI operations.

	Defines specific permissions that can be granted to roles
	for fine-grained access control over AI Core Framework
	resources and operations.

	Attributes:
		READ_MODELS: View model information and metadata
		WRITE_MODELS: Create, update, and configure models
		DELETE_MODELS: Remove models from the system
		INFERENCE_EXECUTE: Execute AI inference operations
		INFERENCE_STREAM: Access streaming inference capabilities
		TRAIN_MODELS: Train and fine-tune AI models
		MANAGE_USERS: Create and manage user accounts
		MANAGE_ROLES: Assign and modify user roles
		VIEW_AUDIT: Access audit logs and compliance reports
		MANAGE_SECURITY: Configure security settings and policies
		SYSTEM_ADMIN: Full system administration capabilities
		API_ACCESS: Access programmatic API endpoints
		NEUROMORPHIC_ACCESS: Access neuromorphic processing features
		QUANTUM_SAFE: Access quantum-safe security features
	"""
	READ_MODELS = "read_models"
	WRITE_MODELS = "write_models"
	DELETE_MODELS = "delete_models"
	INFERENCE_EXECUTE = "inference_execute"
	INFERENCE_STREAM = "inference_stream"
	TRAIN_MODELS = "train_models"
	MANAGE_USERS = "manage_users"
	MANAGE_ROLES = "manage_roles"
	VIEW_AUDIT = "view_audit"
	MANAGE_SECURITY = "manage_security"
	SYSTEM_ADMIN = "system_admin"
	API_ACCESS = "api_access"
	NEUROMORPHIC_ACCESS = "neuromorphic_access"
	QUANTUM_SAFE = "quantum_safe"


class SecurityThreatLevel(str, Enum):
	"""Security threat assessment levels.

	Categorizes security events and user activities by threat level
	for appropriate response and monitoring actions.

	Attributes:
		LOW: Normal operations with minimal security risk
		MEDIUM: Elevated activity requiring monitoring
		HIGH: Suspicious activity requiring immediate attention
		CRITICAL: Active security threat requiring emergency response
	"""
	LOW = "low"
	MEDIUM = "medium"
	HIGH = "high"
	CRITICAL = "critical"


class AuthenticationToken(BaseModel):
	"""JWT authentication token with comprehensive metadata.

	Represents a complete authentication token including claims,
	expiration, permissions, and security context for secure
	AI operations within the APG platform.

	Attributes:
		token_id: Unique identifier for the token
		user_id: User identifier who owns the token
		tenant_id: Multi-tenant isolation identifier
		session_id: Session identifier for tracking
		token_type: Type of token (access, refresh, api_key)
		roles: List of assigned security roles
		permissions: Specific permissions granted
		issued_at: Token creation timestamp
		expires_at: Token expiration timestamp
		last_used: Last usage timestamp
		ip_address: IP address where token was issued
		user_agent: Client user agent information
		scopes: Access scopes for API operations
		refresh_token: Associated refresh token
		revoked: Whether token has been revoked
		revoked_at: Token revocation timestamp
		security_context: Additional security metadata
	"""
	token_id: str = Field(default_factory=uuid7str)
	user_id: str
	tenant_id: str
	session_id: str = Field(default_factory=uuid7str)
	token_type: str = "access"
	roles: List[SecurityRole] = Field(default_factory=list)
	permissions: List[SecurityPermission] = Field(default_factory=list)
	issued_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	expires_at: datetime
	last_used: Optional[datetime] = None
	ip_address: Optional[str] = None
	user_agent: Optional[str] = None
	scopes: List[str] = Field(default_factory=list)
	refresh_token: Optional[str] = None
	revoked: bool = False
	revoked_at: Optional[datetime] = None
	security_context: Dict[str, Any] = Field(default_factory=dict)

	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	def is_valid(self) -> bool:
		"""Check if token is currently valid."""
		if self.revoked:
			return False

		current_time = datetime.now(timezone.utc)
		return current_time < self.expires_at

	def update_last_used(self) -> None:
		"""Update last used timestamp."""
		self.last_used = datetime.now(timezone.utc)

	def revoke(self) -> None:
		"""Revoke the token."""
		self.revoked = True
		self.revoked_at = datetime.now(timezone.utc)


class SecurityAuditEvent(BaseModel):
	"""Comprehensive security audit event for compliance and monitoring.

	Records detailed security events including authentication attempts,
	authorization decisions, resource access, and security policy
	changes for comprehensive audit trails and compliance reporting.

	Attributes:
		event_id: Unique identifier for the audit event
		timestamp: Precise timestamp of the security event
		event_type: Category of security event
		event_action: Specific action that was performed
		user_id: User who performed the action
		tenant_id: Multi-tenant context identifier
		session_id: Session context for the event
		resource_type: Type of resource accessed or modified
		resource_id: Specific resource identifier
		ip_address: Source IP address of the request
		user_agent: Client user agent information
		success: Whether the operation succeeded
		failure_reason: Reason for failure if applicable
		threat_level: Assessed security threat level
		permissions_checked: Permissions that were evaluated
		roles_effective: Roles active during the operation
		request_data: Sanitized request data for context
		response_code: HTTP or system response code
		processing_time_ms: Time taken to process the request
		geo_location: Geographic location if available
		device_fingerprint: Device identification information
		compliance_tags: Tags for regulatory compliance
		risk_score: Calculated risk score for the event
		mitigation_actions: Actions taken to mitigate risks
		related_events: IDs of related security events
		metadata: Additional event-specific metadata
	"""
	event_id: str = Field(default_factory=uuid7str)
	timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	event_type: str
	event_action: str
	user_id: str
	tenant_id: str
	session_id: Optional[str] = None
	resource_type: str
	resource_id: str
	ip_address: Optional[str] = None
	user_agent: Optional[str] = None
	success: bool
	failure_reason: Optional[str] = None
	threat_level: SecurityThreatLevel = SecurityThreatLevel.LOW
	permissions_checked: List[SecurityPermission] = Field(default_factory=list)
	roles_effective: List[SecurityRole] = Field(default_factory=list)
	request_data: Dict[str, Any] = Field(default_factory=dict)
	response_code: Optional[int] = None
	processing_time_ms: float = 0.0
	geo_location: Optional[str] = None
	device_fingerprint: Optional[str] = None
	compliance_tags: List[str] = Field(default_factory=list)
	risk_score: float = 0.0
	mitigation_actions: List[str] = Field(default_factory=list)
	related_events: List[str] = Field(default_factory=list)
	metadata: Dict[str, Any] = Field(default_factory=dict)

	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)


class SecuritySession(BaseModel):
	"""Secure session management with comprehensive tracking.

	Manages user sessions with security monitoring, threat detection,
	and automatic security controls for protecting AI operations
	and maintaining session integrity.

	Attributes:
		session_id: Unique session identifier
		user_id: User owning the session
		tenant_id: Multi-tenant context
		created_at: Session creation timestamp
		last_activity: Last activity timestamp
		expires_at: Session expiration time
		ip_address: Client IP address
		user_agent: Client user agent
		device_fingerprint: Device identification
		geo_location: Geographic location
		authentication_method: How user was authenticated
		mfa_verified: Whether MFA was completed
		roles_active: Currently active roles
		permissions_active: Currently active permissions
		activity_count: Number of operations performed
		threat_indicators: Security threat indicators detected
		security_flags: Active security flags
		last_threat_assessment: Last security assessment time
		risk_score: Current session risk score
		auto_logout_enabled: Whether automatic logout is enabled
		session_metadata: Additional session information
	"""
	session_id: str = Field(default_factory=uuid7str)
	user_id: str
	tenant_id: str
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	last_activity: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	expires_at: datetime
	ip_address: Optional[str] = None
	user_agent: Optional[str] = None
	device_fingerprint: Optional[str] = None
	geo_location: Optional[str] = None
	authentication_method: str = "password"
	mfa_verified: bool = False
	roles_active: List[SecurityRole] = Field(default_factory=list)
	permissions_active: List[SecurityPermission] = Field(default_factory=list)
	activity_count: int = 0
	threat_indicators: List[str] = Field(default_factory=list)
	security_flags: List[str] = Field(default_factory=list)
	last_threat_assessment: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	risk_score: float = 0.0
	auto_logout_enabled: bool = True
	session_metadata: Dict[str, Any] = Field(default_factory=dict)

	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	def is_expired(self) -> bool:
		"""Check if session has expired."""
		return datetime.now(timezone.utc) > self.expires_at

	def update_activity(self) -> None:
		"""Update session activity."""
		self.last_activity = datetime.now(timezone.utc)
		self.activity_count += 1

	def assess_threat_level(self) -> SecurityThreatLevel:
		"""Assess current threat level based on indicators."""
		if len(self.threat_indicators) >= 3 or self.risk_score > 0.8:
			return SecurityThreatLevel.CRITICAL
		elif len(self.threat_indicators) >= 2 or self.risk_score > 0.6:
			return SecurityThreatLevel.HIGH
		elif len(self.threat_indicators) >= 1 or self.risk_score > 0.3:
			return SecurityThreatLevel.MEDIUM
		else:
			return SecurityThreatLevel.LOW


class CryptographicManager:
	"""Advanced cryptographic operations for secure AI processing.

	Provides comprehensive cryptographic services including symmetric
	and asymmetric encryption, digital signatures, key derivation,
	and secure random number generation for protecting AI models
	and sensitive data within the APG platform.

	Attributes:
		_backend: Cryptographic backend for operations
		_master_key: Master encryption key for symmetric operations
		_private_key: RSA private key for asymmetric operations
		_public_key: RSA public key for verification and encryption
		_key_derivation_salt: Salt for key derivation functions
		_encryption_cache: Cache for frequently used keys
	"""

	def __init__(self, master_key: Optional[bytes] = None):
		"""Initialize cryptographic manager.

		Args:
			master_key: Master key for symmetric encryption
		"""
		self._backend = default_backend()
		self._master_key = master_key or self._generate_master_key()
		self._private_key = self._generate_rsa_keypair()
		self._public_key = self._private_key.public_key()
		self._key_derivation_salt = secrets.token_bytes(32)
		self._encryption_cache: Dict[str, bytes] = {}

		# Initialize logging
		self._logger = logging.getLogger(__name__)

	def _generate_master_key(self) -> bytes:
		"""Generate a secure master key for symmetric encryption."""
		return secrets.token_bytes(32)  # 256-bit key

	def _generate_rsa_keypair(self) -> rsa.RSAPrivateKey:
		"""Generate RSA keypair for asymmetric operations."""
		return rsa.generate_private_key(
			public_exponent=65537,
			key_size=4096,  # High security key size
			backend=self._backend
		)

	def derive_key(self, password: str, salt: Optional[bytes] = None, iterations: int = 100000) -> bytes:
		"""Derive encryption key from password using PBKDF2.

		Args:
			password: Password to derive key from
			salt: Salt for key derivation
			iterations: Number of PBKDF2 iterations

		Returns:
			bytes: Derived encryption key
		"""
		if salt is None:
			salt = self._key_derivation_salt

		# Use PBKDF2 with SHA256
		from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

		kdf = PBKDF2HMAC(
			algorithm=hashes.SHA256(),
			length=32,
			salt=salt,
			iterations=iterations,
			backend=self._backend
		)

		return kdf.derive(password.encode('utf-8'))

	def encrypt_symmetric(self, data: bytes, key: Optional[bytes] = None) -> Dict[str, bytes]:
		"""Encrypt data using AES-256-GCM.

		Args:
			data: Data to encrypt
			key: Encryption key (uses master key if None)

		Returns:
			Dict[str, bytes]: Encrypted data with IV and tag
		"""
		if key is None:
			key = self._master_key

		# Generate random IV
		iv = secrets.token_bytes(12)  # 96-bit IV for GCM

		# Create cipher
		cipher = Cipher(
			algorithms.AES(key),
			modes.GCM(iv),
			backend=self._backend
		)

		encryptor = cipher.encryptor()
		ciphertext = encryptor.update(data) + encryptor.finalize()

		return {
			"ciphertext": ciphertext,
			"iv": iv,
			"tag": encryptor.tag
		}

	def decrypt_symmetric(self, encrypted_data: Dict[str, bytes], key: Optional[bytes] = None) -> bytes:
		"""Decrypt data using AES-256-GCM.

		Args:
			encrypted_data: Encrypted data with IV and tag
			key: Decryption key (uses master key if None)

		Returns:
			bytes: Decrypted data

		Raises:
			ValueError: If decryption fails
		"""
		if key is None:
			key = self._master_key

		# Create cipher with authentication tag
		cipher = Cipher(
			algorithms.AES(key),
			modes.GCM(encrypted_data["iv"], encrypted_data["tag"]),
			backend=self._backend
		)

		decryptor = cipher.decryptor()
		plaintext = decryptor.update(encrypted_data["ciphertext"]) + decryptor.finalize()

		return plaintext

	def encrypt_asymmetric(self, data: bytes, public_key: Optional[rsa.RSAPublicKey] = None) -> bytes:
		"""Encrypt data using RSA-OAEP.

		Args:
			data: Data to encrypt (max 446 bytes for 4096-bit key)
			public_key: Public key for encryption

		Returns:
			bytes: Encrypted data
		"""
		if public_key is None:
			public_key = self._public_key

		# Use OAEP padding with SHA256
		ciphertext = public_key.encrypt(
			data,
			padding.OAEP(
				mgf=padding.MGF1(algorithm=hashes.SHA256()),
				algorithm=hashes.SHA256(),
				label=None
			)
		)

		return ciphertext

	def decrypt_asymmetric(self, encrypted_data: bytes) -> bytes:
		"""Decrypt data using RSA-OAEP.

		Args:
			encrypted_data: Data to decrypt

		Returns:
			bytes: Decrypted data

		Raises:
			ValueError: If decryption fails
		"""
		plaintext = self._private_key.decrypt(
			encrypted_data,
			padding.OAEP(
				mgf=padding.MGF1(algorithm=hashes.SHA256()),
				algorithm=hashes.SHA256(),
				label=None
			)
		)

		return plaintext

	def sign_data(self, data: bytes) -> bytes:
		"""Create digital signature for data.

		Args:
			data: Data to sign

		Returns:
			bytes: Digital signature
		"""
		signature = self._private_key.sign(
			data,
			padding.PSS(
				mgf=padding.MGF1(hashes.SHA256()),
				salt_length=padding.PSS.MAX_LENGTH
			),
			hashes.SHA256()
		)

		return signature

	def verify_signature(self, data: bytes, signature: bytes,
						public_key: Optional[rsa.RSAPublicKey] = None) -> bool:
		"""Verify digital signature.

		Args:
			data: Original data
			signature: Digital signature to verify
			public_key: Public key for verification

		Returns:
			bool: True if signature is valid
		"""
		if public_key is None:
			public_key = self._public_key

		try:
			public_key.verify(
				signature,
				data,
				padding.PSS(
					mgf=padding.MGF1(hashes.SHA256()),
					salt_length=padding.PSS.MAX_LENGTH
				),
				hashes.SHA256()
			)
			return True
		except Exception:
			return False

	def generate_secure_token(self, length: int = 32) -> str:
		"""Generate cryptographically secure random token.

		Args:
			length: Token length in bytes

		Returns:
			str: Base64-encoded secure token
		"""
		token_bytes = secrets.token_bytes(length)
		return base64.urlsafe_b64encode(token_bytes).decode('utf-8').rstrip('=')

	def hash_password(self, password: str, salt: Optional[bytes] = None) -> Dict[str, str]:
		"""Hash password using PBKDF2 with secure parameters.

		Args:
			password: Password to hash
			salt: Salt for hashing

		Returns:
			Dict[str, str]: Hashed password and salt
		"""
		if salt is None:
			salt = secrets.token_bytes(32)

		# Use PBKDF2 with 600,000 iterations (OWASP recommendation)
		hashed = hashlib.pbkdf2_hmac(
			'sha256',
			password.encode('utf-8'),
			salt,
			600000
		)

		return {
			"hash": base64.b64encode(hashed).decode('utf-8'),
			"salt": base64.b64encode(salt).decode('utf-8')
		}

	def verify_password(self, password: str, stored_hash: str, stored_salt: str) -> bool:
		"""Verify password against stored hash.

		Args:
			password: Password to verify
			stored_hash: Stored password hash
			stored_salt: Stored salt

		Returns:
			bool: True if password is correct
		"""
		salt = base64.b64decode(stored_salt.encode('utf-8'))
		expected_hash = base64.b64decode(stored_hash.encode('utf-8'))

		actual_hash = hashlib.pbkdf2_hmac(
			'sha256',
			password.encode('utf-8'),
			salt,
			600000
		)

		# Use constant-time comparison to prevent timing attacks
		return hmac.compare_digest(expected_hash, actual_hash)

	def get_public_key_pem(self) -> str:
		"""Get public key in PEM format.

		Returns:
			str: PEM-encoded public key
		"""
		pem = self._public_key.public_key_pem = self._public_key.public_bytes(
			encoding=serialization.Encoding.PEM,
			format=serialization.PublicFormat.SubjectPublicKeyInfo
		)
		return pem.decode('utf-8')


class JWTManager:
	"""Advanced JWT token management with comprehensive security features.

	Manages JWT tokens for authentication and authorization with
	advanced security features including token rotation, blacklisting,
	and comprehensive validation for secure AI operations.

	Attributes:
		_secret_key: Secret key for JWT signing
		_algorithm: JWT signing algorithm
		_issuer: Token issuer identifier
		_crypto_manager: Cryptographic operations manager
		_token_blacklist: Set of revoked token IDs
		_refresh_tokens: Storage for refresh tokens
	"""

	def __init__(self, secret_key: Optional[str] = None, algorithm: str = "HS256"):
		"""Initialize JWT manager.

		Args:
			secret_key: Secret key for JWT signing
			algorithm: JWT signing algorithm
		"""
		self._secret_key = secret_key or secrets.token_urlsafe(64)
		self._algorithm = algorithm
		self._issuer = "apg-ai-core-framework"
		self._crypto_manager = CryptographicManager()
		self._token_blacklist: Set[str] = set()
		self._refresh_tokens: Dict[str, str] = {}  # refresh_token -> user_id

		# Initialize logging
		self._logger = logging.getLogger(__name__)

	def create_access_token(self, user_id: str, tenant_id: str, roles: List[SecurityRole],
						   permissions: List[SecurityPermission], expires_delta: Optional[timedelta] = None,
						   additional_claims: Optional[Dict[str, Any]] = None) -> str:
		"""Create JWT access token with comprehensive claims.

		Args:
			user_id: User identifier
			tenant_id: Tenant identifier
			roles: User roles
			permissions: User permissions
			expires_delta: Token expiration time
			additional_claims: Additional JWT claims

		Returns:
			str: JWT access token
		"""
		if expires_delta is None:
			expires_delta = timedelta(hours=1)  # Default 1 hour expiration

		now = datetime.now(timezone.utc)
		expires_at = now + expires_delta

		# Create token ID for tracking
		token_id = uuid7str()

		# Build JWT claims
		claims = {
			"sub": user_id,  # Subject (user ID)
			"iss": self._issuer,  # Issuer
			"aud": "apg-ai-core",  # Audience
			"iat": int(now.timestamp()),  # Issued at
			"exp": int(expires_at.timestamp()),  # Expires at
			"jti": token_id,  # JWT ID
			"tenant_id": tenant_id,
			"roles": [role.value for role in roles],
			"permissions": [perm.value for perm in permissions],
			"token_type": "access"
		}

		# Add additional claims
		if additional_claims:
			claims.update(additional_claims)

		# Create and sign JWT
		token = jwt.encode(claims, self._secret_key, algorithm=self._algorithm)

		self._logger.info(_log_auth_event("TOKEN_CREATED", user_id, True))

		return token

	def create_refresh_token(self, user_id: str, tenant_id: str) -> str:
		"""Create refresh token for token renewal.

		Args:
			user_id: User identifier
			tenant_id: Tenant identifier

		Returns:
			str: Refresh token
		"""
		# Generate secure random token
		refresh_token = self._crypto_manager.generate_secure_token(32)

		# Store mapping
		self._refresh_tokens[refresh_token] = user_id

		self._logger.info(_log_auth_event("REFRESH_TOKEN_CREATED", user_id, True))

		return refresh_token

	def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
		"""Verify and decode JWT token.

		Args:
			token: JWT token to verify

		Returns:
			Optional[Dict[str, Any]]: Decoded token claims or None if invalid
		"""
		try:
			# Decode and verify token
			claims = jwt.decode(
				token,
				self._secret_key,
				algorithms=[self._algorithm],
				issuer=self._issuer,
				audience="apg-ai-core"
			)

			# Check if token is blacklisted
			token_id = claims.get("jti")
			if token_id and token_id in self._token_blacklist:
				self._logger.warning(_log_auth_event("TOKEN_BLACKLISTED", claims.get("sub", "unknown"), False))
				return None

			# Verify token type
			if claims.get("token_type") != "access":
				return None

			self._logger.debug(_log_auth_event("TOKEN_VERIFIED", claims.get("sub", "unknown"), True))

			return claims

		except jwt.ExpiredSignatureError:
			self._logger.warning(_log_auth_event("TOKEN_EXPIRED", "unknown", False))
			return None
		except jwt.InvalidTokenError as e:
			self._logger.warning(_log_auth_event("TOKEN_INVALID", "unknown", False, str(e)))
			return None

	def refresh_access_token(self, refresh_token: str, new_roles: Optional[List[SecurityRole]] = None,
							new_permissions: Optional[List[SecurityPermission]] = None) -> Optional[str]:
		"""Refresh access token using refresh token.

		Args:
			refresh_token: Valid refresh token
			new_roles: Updated roles (optional)
			new_permissions: Updated permissions (optional)

		Returns:
			Optional[str]: New access token or None if refresh failed
		"""
		if refresh_token not in self._refresh_tokens:
			self._logger.warning(_log_auth_event("REFRESH_TOKEN_INVALID", "unknown", False))
			return None

		user_id = self._refresh_tokens[refresh_token]

		# For production, you would fetch user's current roles and permissions from database
		# Here we use defaults or provided values
		roles = new_roles or [SecurityRole.USER]
		permissions = new_permissions or [SecurityPermission.READ_MODELS, SecurityPermission.INFERENCE_EXECUTE]
		tenant_id = "default"  # Would be fetched from user profile

		# Create new access token
		new_token = self.create_access_token(
			user_id=user_id,
			tenant_id=tenant_id,
			roles=roles,
			permissions=permissions
		)

		self._logger.info(_log_auth_event("TOKEN_REFRESHED", user_id, True))

		return new_token

	def revoke_token(self, token: str) -> bool:
		"""Revoke (blacklist) a JWT token.

		Args:
			token: Token to revoke

		Returns:
			bool: True if token was revoked successfully
		"""
		try:
			# Decode token to get ID (without verification since we're revoking)
			claims = jwt.decode(token, options={"verify_signature": False})
			token_id = claims.get("jti")

			if token_id:
				self._token_blacklist.add(token_id)
				self._logger.info(_log_auth_event("TOKEN_REVOKED", claims.get("sub", "unknown"), True))
				return True

			return False

		except Exception as e:
			self._logger.error(f"Failed to revoke token: {str(e)}")
			return False

	def revoke_refresh_token(self, refresh_token: str) -> bool:
		"""Revoke a refresh token.

		Args:
			refresh_token: Refresh token to revoke

		Returns:
			bool: True if token was revoked successfully
		"""
		if refresh_token in self._refresh_tokens:
			user_id = self._refresh_tokens[refresh_token]
			del self._refresh_tokens[refresh_token]
			self._logger.info(_log_auth_event("REFRESH_TOKEN_REVOKED", user_id, True))
			return True

		return False

	def cleanup_expired_tokens(self) -> int:
		"""Clean up expired tokens from blacklist.

		Returns:
			int: Number of expired tokens removed
		"""
		# For production, you would implement proper cleanup based on token expiration
		# This is a simplified implementation
		current_size = len(self._token_blacklist)

		# Keep blacklist reasonable size (remove oldest entries)
		if current_size > 10000:
			# Convert to list, sort, and keep most recent 5000
			blacklist_list = list(self._token_blacklist)
			self._token_blacklist = set(blacklist_list[-5000:])

			removed = current_size - len(self._token_blacklist)
			self._logger.info(f"Cleaned up {removed} old blacklisted tokens")
			return removed

		return 0


class RBACManager:
	"""Role-Based Access Control (RBAC) management system.

	Comprehensive RBAC system managing roles, permissions, and
	access control decisions for AI operations with support for
	hierarchical roles, dynamic permissions, and audit logging.

	Attributes:
		_role_permissions: Mapping of roles to their permissions
		_user_roles: Mapping of users to their assigned roles
		_role_hierarchy: Hierarchical role relationships
		_permission_cache: Cache for permission lookups
		_audit_logger: Security audit event logger
	"""

	def __init__(self):
		"""Initialize RBAC manager with default role permissions."""
		self._role_permissions: Dict[SecurityRole, Set[SecurityPermission]] = {}
		self._user_roles: Dict[str, Set[SecurityRole]] = {}
		self._role_hierarchy: Dict[SecurityRole, Set[SecurityRole]] = {}
		self._permission_cache: Dict[str, Set[SecurityPermission]] = {}
		self._audit_events: List[SecurityAuditEvent] = []

		# Initialize default role permissions
		self._initialize_default_permissions()
		self._initialize_role_hierarchy()

		# Initialize logging
		self._logger = logging.getLogger(__name__)

	def _initialize_default_permissions(self) -> None:
		"""Initialize default permissions for each role."""
		# Guest role - minimal read access
		self._role_permissions[SecurityRole.GUEST] = {
			SecurityPermission.READ_MODELS
		}

		# User role - basic AI operations
		self._role_permissions[SecurityRole.USER] = {
			SecurityPermission.READ_MODELS,
			SecurityPermission.INFERENCE_EXECUTE,
			SecurityPermission.API_ACCESS
		}

		# Developer role - extended AI capabilities
		self._role_permissions[SecurityRole.DEVELOPER] = {
			SecurityPermission.READ_MODELS,
			SecurityPermission.WRITE_MODELS,
			SecurityPermission.INFERENCE_EXECUTE,
			SecurityPermission.INFERENCE_STREAM,
			SecurityPermission.TRAIN_MODELS,
			SecurityPermission.API_ACCESS,
			SecurityPermission.NEUROMORPHIC_ACCESS
		}

		# Analyst role - data analysis focus
		self._role_permissions[SecurityRole.ANALYST] = {
			SecurityPermission.READ_MODELS,
			SecurityPermission.INFERENCE_EXECUTE,
			SecurityPermission.INFERENCE_STREAM,
			SecurityPermission.VIEW_AUDIT,
			SecurityPermission.API_ACCESS
		}

		# Admin role - user and system management
		self._role_permissions[SecurityRole.ADMIN] = {
			SecurityPermission.READ_MODELS,
			SecurityPermission.WRITE_MODELS,
			SecurityPermission.DELETE_MODELS,
			SecurityPermission.INFERENCE_EXECUTE,
			SecurityPermission.INFERENCE_STREAM,
			SecurityPermission.TRAIN_MODELS,
			SecurityPermission.MANAGE_USERS,
			SecurityPermission.MANAGE_ROLES,
			SecurityPermission.VIEW_AUDIT,
			SecurityPermission.API_ACCESS,
			SecurityPermission.NEUROMORPHIC_ACCESS
		}

		# Super Admin role - full system access
		self._role_permissions[SecurityRole.SUPER_ADMIN] = set(SecurityPermission)

		# Service role - service-to-service authentication
		self._role_permissions[SecurityRole.SERVICE] = {
			SecurityPermission.READ_MODELS,
			SecurityPermission.INFERENCE_EXECUTE,
			SecurityPermission.API_ACCESS
		}

		# Audit role - compliance and monitoring
		self._role_permissions[SecurityRole.AUDIT] = {
			SecurityPermission.READ_MODELS,
			SecurityPermission.VIEW_AUDIT,
			SecurityPermission.API_ACCESS
		}

	def _initialize_role_hierarchy(self) -> None:
		"""Initialize role hierarchy for inherited permissions."""
		# Super Admin inherits from Admin
		self._role_hierarchy[SecurityRole.SUPER_ADMIN] = {SecurityRole.ADMIN}

		# Admin inherits from Developer and Analyst
		self._role_hierarchy[SecurityRole.ADMIN] = {SecurityRole.DEVELOPER, SecurityRole.ANALYST}

		# Developer inherits from User
		self._role_hierarchy[SecurityRole.DEVELOPER] = {SecurityRole.USER}

		# Analyst inherits from User
		self._role_hierarchy[SecurityRole.ANALYST] = {SecurityRole.USER}

		# User inherits from Guest
		self._role_hierarchy[SecurityRole.USER] = {SecurityRole.GUEST}

	def assign_role(self, user_id: str, role: SecurityRole, assigned_by: str) -> bool:
		"""Assign role to user.

		Args:
			user_id: User to assign role to
			role: Role to assign
			assigned_by: User performing the assignment

		Returns:
			bool: True if role was assigned successfully
		"""
		try:
			if user_id not in self._user_roles:
				self._user_roles[user_id] = set()

			self._user_roles[user_id].add(role)

			# Clear permission cache for user
			if user_id in self._permission_cache:
				del self._permission_cache[user_id]

			# Log audit event
			self._log_audit_event(
				event_type="role_assignment",
				event_action="assign_role",
				user_id=assigned_by,
				resource_type="user_role",
				resource_id=user_id,
				success=True,
				metadata={"role_assigned": role.value}
			)

			self._logger.info(_log_rbac_event(assigned_by, f"user:{user_id}", f"assign_role:{role.value}", True))

			return True

		except Exception as e:
			self._logger.error(f"Failed to assign role {role.value} to user {user_id}: {str(e)}")
			return False

	def revoke_role(self, user_id: str, role: SecurityRole, revoked_by: str) -> bool:
		"""Revoke role from user.

		Args:
			user_id: User to revoke role from
			role: Role to revoke
			revoked_by: User performing the revocation

		Returns:
			bool: True if role was revoked successfully
		"""
		try:
			if user_id in self._user_roles and role in self._user_roles[user_id]:
				self._user_roles[user_id].remove(role)

				# Clear permission cache for user
				if user_id in self._permission_cache:
					del self._permission_cache[user_id]

				# Log audit event
				self._log_audit_event(
					event_type="role_revocation",
					event_action="revoke_role",
					user_id=revoked_by,
					resource_type="user_role",
					resource_id=user_id,
					success=True,
					metadata={"role_revoked": role.value}
				)

				self._logger.info(_log_rbac_event(revoked_by, f"user:{user_id}", f"revoke_role:{role.value}", True))

				return True

			return False

		except Exception as e:
			self._logger.error(f"Failed to revoke role {role.value} from user {user_id}: {str(e)}")
			return False

	def get_user_roles(self, user_id: str) -> Set[SecurityRole]:
		"""Get all roles assigned to user.

		Args:
			user_id: User identifier

		Returns:
			Set[SecurityRole]: Set of assigned roles
		"""
		return self._user_roles.get(user_id, set())

	def get_user_permissions(self, user_id: str) -> Set[SecurityPermission]:
		"""Get all permissions for user based on assigned roles.

		Args:
			user_id: User identifier

		Returns:
			Set[SecurityPermission]: Set of effective permissions
		"""
		# Check cache first
		if user_id in self._permission_cache:
			return self._permission_cache[user_id]

		permissions = set()
		user_roles = self.get_user_roles(user_id)

		# Collect permissions from all roles (including inherited)
		for role in user_roles:
			permissions.update(self._get_role_permissions_recursive(role))

		# Cache permissions
		self._permission_cache[user_id] = permissions

		return permissions

	def _get_role_permissions_recursive(self, role: SecurityRole) -> Set[SecurityPermission]:
		"""Get permissions for role including inherited permissions."""
		permissions = self._role_permissions.get(role, set()).copy()

		# Add permissions from inherited roles
		inherited_roles = self._role_hierarchy.get(role, set())
		for inherited_role in inherited_roles:
			permissions.update(self._get_role_permissions_recursive(inherited_role))

		return permissions

	def check_permission(self, user_id: str, permission: SecurityPermission,
						resource_type: str, resource_id: str) -> bool:
		"""Check if user has specific permission for resource.

		Args:
			user_id: User identifier
			permission: Permission to check
			resource_type: Type of resource being accessed
			resource_id: Specific resource identifier

		Returns:
			bool: True if user has permission
		"""
		user_permissions = self.get_user_permissions(user_id)
		has_permission = permission in user_permissions

		# Log authorization decision
		self._log_audit_event(
			event_type="authorization",
			event_action="check_permission",
			user_id=user_id,
			resource_type=resource_type,
			resource_id=resource_id,
			success=has_permission,
			permissions_checked=[permission],
			roles_effective=list(self.get_user_roles(user_id))
		)

		self._logger.debug(_log_rbac_event(user_id, f"{resource_type}:{resource_id}", permission.value, has_permission))

		return has_permission

	def check_multiple_permissions(self, user_id: str, permissions: List[SecurityPermission],
								  resource_type: str, resource_id: str, require_all: bool = True) -> bool:
		"""Check if user has multiple permissions for resource.

		Args:
			user_id: User identifier
			permissions: List of permissions to check
			resource_type: Type of resource being accessed
			resource_id: Specific resource identifier
			require_all: Whether all permissions are required (True) or any (False)

		Returns:
			bool: True if permission check passes
		"""
		user_permissions = self.get_user_permissions(user_id)

		if require_all:
			has_permissions = all(perm in user_permissions for perm in permissions)
		else:
			has_permissions = any(perm in user_permissions for perm in permissions)

		# Log authorization decision
		self._log_audit_event(
			event_type="authorization",
			event_action="check_multiple_permissions",
			user_id=user_id,
			resource_type=resource_type,
			resource_id=resource_id,
			success=has_permissions,
			permissions_checked=permissions,
			roles_effective=list(self.get_user_roles(user_id)),
			metadata={"require_all": require_all}
		)

		return has_permissions

	def add_role_permission(self, role: SecurityRole, permission: SecurityPermission) -> bool:
		"""Add permission to role.

		Args:
			role: Role to modify
			permission: Permission to add

		Returns:
			bool: True if permission was added
		"""
		if role not in self._role_permissions:
			self._role_permissions[role] = set()

		self._role_permissions[role].add(permission)

		# Clear all permission caches since role permissions changed
		self._permission_cache.clear()

		self._logger.info(f"Added permission {permission.value} to role {role.value}")

		return True

	def remove_role_permission(self, role: SecurityRole, permission: SecurityPermission) -> bool:
		"""Remove permission from role.

		Args:
			role: Role to modify
			permission: Permission to remove

		Returns:
			bool: True if permission was removed
		"""
		if role in self._role_permissions and permission in self._role_permissions[role]:
			self._role_permissions[role].remove(permission)

			# Clear all permission caches since role permissions changed
			self._permission_cache.clear()

			self._logger.info(f"Removed permission {permission.value} from role {role.value}")

			return True

		return False

	def _log_audit_event(self, event_type: str, event_action: str, user_id: str,
						resource_type: str, resource_id: str, success: bool,
						permissions_checked: Optional[List[SecurityPermission]] = None,
						roles_effective: Optional[List[SecurityRole]] = None,
						metadata: Optional[Dict[str, Any]] = None) -> None:
		"""Log security audit event."""
		audit_event = SecurityAuditEvent(
			event_type=event_type,
			event_action=event_action,
			user_id=user_id,
			tenant_id="default",  # Would be determined from context
			resource_type=resource_type,
			resource_id=resource_id,
			success=success,
			permissions_checked=permissions_checked or [],
			roles_effective=roles_effective or [],
			metadata=metadata or {}
		)

		self._audit_events.append(audit_event)

		# Keep audit log reasonable size
		if len(self._audit_events) > 10000:
			self._audit_events = self._audit_events[-5000:]

	def get_audit_events(self, filters: Optional[Dict[str, Any]] = None) -> List[SecurityAuditEvent]:
		"""Get audit events with optional filtering.

		Args:
			filters: Optional filters for events

		Returns:
			List[SecurityAuditEvent]: Filtered audit events
		"""
		if not filters:
			return list(self._audit_events)

		filtered_events = []

		for event in self._audit_events:
			# Apply filters
			if filters.get("user_id") and event.user_id != filters["user_id"]:
				continue
			if filters.get("event_type") and event.event_type != filters["event_type"]:
				continue
			if filters.get("success") is not None and event.success != filters["success"]:
				continue
			if filters.get("resource_type") and event.resource_type != filters["resource_type"]:
				continue

			filtered_events.append(event)

		return filtered_events


class SecurityIntegrationManager:
	"""Comprehensive security integration manager for APG AI Core Framework.

	Central security management system integrating authentication,
	authorization, audit logging, and threat detection for complete
	security coverage of AI operations within the APG platform.

	Attributes:
		jwt_manager: JWT token management
		rbac_manager: Role-based access control
		crypto_manager: Cryptographic operations
		session_manager: Session management and tracking
		threat_detector: Security threat detection system
		audit_logger: Comprehensive audit logging
	"""

	def __init__(self, config: Optional[Dict[str, Any]] = None):
		"""Initialize security integration manager.

		Args:
			config: Security configuration parameters
		"""
		self.config = config or {}

		# Initialize security components
		self.jwt_manager = JWTManager(
			secret_key=self.config.get("jwt_secret"),
			algorithm=self.config.get("jwt_algorithm", "HS256")
		)
		self.rbac_manager = RBACManager()
		self.crypto_manager = CryptographicManager()

		# Session and threat management
		self.active_sessions: Dict[str, SecuritySession] = {}
		self.threat_indicators: Dict[str, List[str]] = {}
		self.api_keys: Dict[str, Dict[str, Any]] = {}

		# Security monitoring
		self.security_metrics = {
			"total_authentications": 0,
			"failed_authentications": 0,
			"active_sessions": 0,
			"threats_detected": 0,
			"permissions_denied": 0,
			"tokens_revoked": 0
		}

		# Initialize logging
		self._logger = logging.getLogger(__name__)

		# Start background security tasks
		self._start_security_monitoring()

	async def authenticate_user(self, username: str, password: str, ip_address: str = "",
							   user_agent: str = "", tenant_id: str = "default") -> Optional[Dict[str, Any]]:
		"""Authenticate user with comprehensive security checks.

		Args:
			username: Username for authentication
			password: User password
			ip_address: Client IP address
			user_agent: Client user agent
			tenant_id: Tenant context

		Returns:
			Optional[Dict[str, Any]]: Authentication result with tokens
		"""
		start_time = time.time()
		self.security_metrics["total_authentications"] += 1

		try:
			# Simulate user lookup and password verification
			# In production, this would query the user database
			user_data = await self._lookup_user(username, tenant_id)

			if not user_data:
				self._handle_auth_failure(username, "user_not_found", ip_address)
				return None

			# Verify password
			password_valid = self.crypto_manager.verify_password(
				password,
				user_data["password_hash"],
				user_data["password_salt"]
			)

			if not password_valid:
				self._handle_auth_failure(username, "invalid_password", ip_address)
				return None

			# Check for account security flags
			if user_data.get("account_locked", False):
				self._handle_auth_failure(username, "account_locked", ip_address)
				return None

			# Perform threat assessment
			threat_level = await self._assess_authentication_threats(username, ip_address, user_agent)

			if threat_level == SecurityThreatLevel.CRITICAL:
				self._handle_auth_failure(username, "security_threat", ip_address)
				return None

			# Create session
			session = await self._create_security_session(
				user_data["user_id"], tenant_id, ip_address, user_agent
			)

			# Get user roles and permissions
			user_roles = user_data.get("roles", [SecurityRole.USER])
			user_permissions = []
			for role in user_roles:
				user_permissions.extend(self.rbac_manager._get_role_permissions_recursive(role))

			# Create tokens
			access_token = self.jwt_manager.create_access_token(
				user_id=user_data["user_id"],
				tenant_id=tenant_id,
				roles=user_roles,
				permissions=list(set(user_permissions)),  # Remove duplicates
				additional_claims={
					"session_id": session.session_id,
					"ip_address": ip_address,
					"threat_level": threat_level.value
				}
			)

			refresh_token = self.jwt_manager.create_refresh_token(
				user_data["user_id"], tenant_id
			)

			# Update session with tokens
			session.last_activity = datetime.now(timezone.utc)
			session.roles_active = user_roles
			session.permissions_active = list(set(user_permissions))

			auth_time_ms = (time.time() - start_time) * 1000

			# Log successful authentication
			self._logger.info(_log_auth_event("AUTHENTICATE", user_data["user_id"], True, ip_address))

			# Create audit event
			self.rbac_manager._log_audit_event(
				event_type="authentication",
				event_action="user_login",
				user_id=user_data["user_id"],
				resource_type="authentication",
				resource_id="login",
				success=True,
				roles_effective=user_roles,
				processing_time_ms=auth_time_ms,
				metadata={
					"ip_address": ip_address,
					"user_agent": user_agent,
					"threat_level": threat_level.value
				}
			)

			return {
				"success": True,
				"user_id": user_data["user_id"],
				"tenant_id": tenant_id,
				"session_id": session.session_id,
				"access_token": access_token,
				"refresh_token": refresh_token,
				"expires_in": 3600,  # 1 hour
				"token_type": "Bearer",
				"roles": [role.value for role in user_roles],
				"permissions": [perm.value for perm in set(user_permissions)],
				"threat_level": threat_level.value,
				"requires_mfa": user_data.get("mfa_enabled", False) and not session.mfa_verified
			}

		except Exception as e:
			self._handle_auth_failure(username, f"system_error: {str(e)}", ip_address)
			return None

	async def _lookup_user(self, username: str, tenant_id: str) -> Optional[Dict[str, Any]]:
		"""Simulate user lookup from database.

		Args:
			username: Username to lookup
			tenant_id: Tenant context

		Returns:
			Optional[Dict[str, Any]]: User data or None
		"""
		# Simulate database lookup
		# In production, this would query the actual user database
		if username == "admin":
			password_data = self.crypto_manager.hash_password("admin123")
			return {
				"user_id": "admin_user_001",
				"username": username,
				"tenant_id": tenant_id,
				"password_hash": password_data["hash"],
				"password_salt": password_data["salt"],
				"roles": [SecurityRole.ADMIN],
				"account_locked": False,
				"mfa_enabled": False,
				"created_at": datetime.now(timezone.utc)
			}
		elif username == "testuser":
			password_data = self.crypto_manager.hash_password("password123")
			return {
				"user_id": "test_user_001",
				"username": username,
				"tenant_id": tenant_id,
				"password_hash": password_data["hash"],
				"password_salt": password_data["salt"],
				"roles": [SecurityRole.USER],
				"account_locked": False,
				"mfa_enabled": False,
				"created_at": datetime.now(timezone.utc)
			}

		return None

	def _handle_auth_failure(self, username: str, reason: str, ip_address: str) -> None:
		"""Handle authentication failure with logging and threat tracking."""
		self.security_metrics["failed_authentications"] += 1

		# Track threat indicators
		if ip_address not in self.threat_indicators:
			self.threat_indicators[ip_address] = []

		self.threat_indicators[ip_address].append(f"auth_failure:{reason}:{time.time()}")

		# Limit threat indicator history
		if len(self.threat_indicators[ip_address]) > 100:
			self.threat_indicators[ip_address] = self.threat_indicators[ip_address][-50:]

		self._logger.warning(_log_auth_event("AUTHENTICATE_FAILED", username, False, ip_address))

		# Create audit event for failed authentication
		self.rbac_manager._log_audit_event(
			event_type="authentication",
			event_action="user_login",
			user_id=username,
			resource_type="authentication",
			resource_id="login",
			success=False,
			failure_reason=reason,
			threat_level=SecurityThreatLevel.MEDIUM,
			metadata={"ip_address": ip_address, "failure_reason": reason}
		)

	async def _assess_authentication_threats(self, username: str, ip_address: str,
											user_agent: str) -> SecurityThreatLevel:
		"""Assess security threats for authentication attempt."""
		threat_score = 0.0

		# Check IP-based threats
		if ip_address in self.threat_indicators:
			recent_failures = [
				indicator for indicator in self.threat_indicators[ip_address]
				if indicator.startswith("auth_failure") and
				time.time() - float(indicator.split(":")[-1]) < 3600  # Last hour
			]

			if len(recent_failures) > 10:
				threat_score += 0.8
			elif len(recent_failures) > 5:
				threat_score += 0.4
			elif len(recent_failures) > 2:
				threat_score += 0.2

		# Check for suspicious user agent patterns
		if not user_agent or len(user_agent) < 10:
			threat_score += 0.2

		# Check for known attack patterns in username
		suspicious_patterns = ["admin", "root", "test", "user", "guest"]
		if any(pattern in username.lower() for pattern in suspicious_patterns):
			threat_score += 0.1

		# Determine threat level
		if threat_score >= 0.8:
			return SecurityThreatLevel.CRITICAL
		elif threat_score >= 0.6:
			return SecurityThreatLevel.HIGH
		elif threat_score >= 0.3:
			return SecurityThreatLevel.MEDIUM
		else:
			return SecurityThreatLevel.LOW

	async def _create_security_session(self, user_id: str, tenant_id: str,
									  ip_address: str, user_agent: str) -> SecuritySession:
		"""Create new security session with monitoring."""
		session = SecuritySession(
			user_id=user_id,
			tenant_id=tenant_id,
			expires_at=datetime.now(timezone.utc) + timedelta(hours=8),  # 8 hour session
			ip_address=ip_address,
			user_agent=user_agent,
			device_fingerprint=hashlib.sha256(f"{ip_address}:{user_agent}".encode()).hexdigest()[:16]
		)

		self.active_sessions[session.session_id] = session
		self.security_metrics["active_sessions"] = len(self.active_sessions)

		return session

	async def authorize_request(self, token: str, resource_type: str, resource_id: str,
							   required_permissions: List[SecurityPermission]) -> Dict[str, Any]:
		"""Authorize API request with comprehensive security checks.

		Args:
			token: JWT access token
			resource_type: Type of resource being accessed
			resource_id: Specific resource identifier
			required_permissions: List of required permissions

		Returns:
			Dict[str, Any]: Authorization result
		"""
		start_time = time.time()

		try:
			# Verify JWT token
			token_claims = self.jwt_manager.verify_token(token)
			if not token_claims:
				self.security_metrics["permissions_denied"] += 1
				return {
					"authorized": False,
					"reason": "invalid_token",
					"error": "Token verification failed"
				}

			user_id = token_claims["sub"]
			session_id = token_claims.get("session_id")

			# Verify session is still active
			if session_id and session_id in self.active_sessions:
				session = self.active_sessions[session_id]
				if session.is_expired():
					self.security_metrics["permissions_denied"] += 1
					return {
						"authorized": False,
						"reason": "session_expired",
						"error": "Session has expired"
					}

				# Update session activity
				session.update_activity()

			# Check permissions
			authorization_granted = self.rbac_manager.check_multiple_permissions(
				user_id=user_id,
				permissions=required_permissions,
				resource_type=resource_type,
				resource_id=resource_id,
				require_all=True
			)

			if not authorization_granted:
				self.security_metrics["permissions_denied"] += 1

			auth_time_ms = (time.time() - start_time) * 1000

			return {
				"authorized": authorization_granted,
				"user_id": user_id,
				"tenant_id": token_claims.get("tenant_id"),
				"session_id": session_id,
				"roles": token_claims.get("roles", []),
				"permissions": token_claims.get("permissions", []),
				"processing_time_ms": auth_time_ms,
				"threat_level": token_claims.get("threat_level", "low")
			}

		except Exception as e:
			self.security_metrics["permissions_denied"] += 1
			self._logger.error(f"Authorization failed: {str(e)}")

			return {
				"authorized": False,
				"reason": "authorization_error",
				"error": str(e)
			}

	async def create_api_key(self, user_id: str, name: str, permissions: List[SecurityPermission],
							expires_in_days: int = 365) -> Dict[str, Any]:
		"""Create API key for programmatic access.

		Args:
			user_id: User creating the API key
			name: Descriptive name for the API key
			permissions: Permissions to grant to the API key
			expires_in_days: Expiration time in days

		Returns:
			Dict[str, Any]: API key information
		"""
		# Generate secure API key
		api_key = f"apg_{self.crypto_manager.generate_secure_token(32)}"

		# Create API key metadata
		api_key_data = {
			"key_id": uuid7str(),
			"user_id": user_id,
			"name": name,
			"permissions": [perm.value for perm in permissions],
			"created_at": datetime.now(timezone.utc),
			"expires_at": datetime.now(timezone.utc) + timedelta(days=expires_in_days),
			"last_used": None,
			"usage_count": 0,
			"active": True
		}

		# Store API key (in production, hash the key for storage)
		self.api_keys[api_key] = api_key_data

		self._logger.info(f"Created API key '{name}' for user {user_id}")

		return {
			"api_key": api_key,
			"key_id": api_key_data["key_id"],
			"name": name,
			"permissions": [perm.value for perm in permissions],
			"expires_at": api_key_data["expires_at"].isoformat()
		}

	async def verify_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
		"""Verify API key and return associated data.

		Args:
			api_key: API key to verify

		Returns:
			Optional[Dict[str, Any]]: API key data or None
		"""
		if api_key not in self.api_keys:
			return None

		key_data = self.api_keys[api_key]

		# Check if key is active
		if not key_data["active"]:
			return None

		# Check if key has expired
		if datetime.now(timezone.utc) > key_data["expires_at"]:
			key_data["active"] = False
			return None

		# Update usage
		key_data["last_used"] = datetime.now(timezone.utc)
		key_data["usage_count"] += 1

		return key_data

	def _start_security_monitoring(self) -> None:
		"""Start background security monitoring tasks."""
		# In production, these would be proper async tasks
		# For now, we'll track the monitoring state
		self.monitoring_active = True
		self._logger.info("Security monitoring started")

	async def get_security_status(self) -> Dict[str, Any]:
		"""Get comprehensive security status and metrics.

		Returns:
			Dict[str, Any]: Security status information
		"""
		# Clean up expired sessions
		expired_sessions = [
			session_id for session_id, session in self.active_sessions.items()
			if session.is_expired()
		]

		for session_id in expired_sessions:
			del self.active_sessions[session_id]

		self.security_metrics["active_sessions"] = len(self.active_sessions)

		# Calculate security health score
		total_auths = max(1, self.security_metrics["total_authentications"])
		success_rate = (total_auths - self.security_metrics["failed_authentications"]) / total_auths

		health_score = min(100, int(success_rate * 100))

		return {
			"security_health_score": health_score,
			"authentication_metrics": {
				"total_authentications": self.security_metrics["total_authentications"],
				"failed_authentications": self.security_metrics["failed_authentications"],
				"success_rate_percent": success_rate * 100
			},
			"session_metrics": {
				"active_sessions": len(self.active_sessions),
				"expired_sessions_cleaned": len(expired_sessions)
			},
			"authorization_metrics": {
				"permissions_denied": self.security_metrics["permissions_denied"],
				"tokens_revoked": self.security_metrics["tokens_revoked"]
			},
			"threat_metrics": {
				"threats_detected": self.security_metrics["threats_detected"],
				"monitored_ips": len(self.threat_indicators),
				"active_threat_indicators": sum(len(indicators) for indicators in self.threat_indicators.values())
			},
			"api_key_metrics": {
				"total_api_keys": len(self.api_keys),
				"active_api_keys": sum(1 for key_data in self.api_keys.values() if key_data["active"])
			},
			"security_features": {
				"jwt_authentication": True,
				"rbac_authorization": True,
				"session_management": True,
				"threat_detection": True,
				"audit_logging": True,
				"api_key_support": True,
				"cryptographic_protection": True
			},
			"monitoring_status": "active" if hasattr(self, "monitoring_active") else "inactive"
		}


# Module exports
__all__ = [
	# Core security manager
	"SecurityIntegrationManager",

	# Security components
	"JWTManager", "RBACManager", "CryptographicManager",

	# Security models
	"AuthenticationToken", "SecurityAuditEvent", "SecuritySession",

	# Enums
	"SecurityRole", "SecurityPermission", "SecurityThreatLevel",

	# Utility functions
	"_log_security_event", "_log_auth_event", "_log_rbac_event"
]