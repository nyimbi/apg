"""
APG Enhanced Authentication & Authorization Capability

Revolutionary authentication system combining traditional RBAC with cutting-edge
technologies including behavioral authentication, quantum-resistant cryptography,
zero-knowledge proofs, biometric fusion, neuromorphic processing, and privacy-preserving analytics.

Features:
- AI-Powered Behavioral Authentication with ML pattern analysis
- Quantum-Resistant Cryptography (CRYSTALS-Kyber/Dilithium)
- Zero-Knowledge Proof Authentication for privacy preservation
- Biometric Fusion Engine with liveness detection
- Identity Graph Intelligence for fraud detection
- Neuromorphic Authentication Processor for sub-millisecond decisions
- Privacy-Preserving Analytics with differential privacy
- Federated Identity Mesh for decentralized authentication
- Enhanced Session Management with adaptive timeouts

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from typing import Dict, Any, Optional, List, Set
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import hashlib
import secrets
import jwt
from pathlib import Path
from uuid_extensions import uuid7str
from pydantic import BaseModel, Field, ConfigDict, EmailStr
import threading
from contextlib import asynccontextmanager

# Import revolutionary authentication components
from .behavioral_auth import BehavioralAuthenticator, BehavioralBaseline, AuthScore
from .contextual_risk import ContextualRiskEngine, RiskFactor, RiskAssessment
from .enhanced_models import EnhancedUser, BiometricTemplate, QuantumKey, PrivacyPreferences
from .quantum_auth import QuantumResistantAuth, CRYSTALSKyber, CRYSTALSDilithium
from .zk_proof import ZKProofAuthenticator, ZKChallenge, ZKProof, SchnorrProof
from .biometric_fusion import BiometricFusionEngine, FusionMethod, LivenessStatus
from .adaptive_policies import AdaptivePolicyEngine, PolicyOutcome, LearningMode
from .identity_graph import IdentityGraphEngine
from .neuromorphic_processor import NeuromorphicProcessor, AuthenticationContext as NeuroContext
from .capability_contract import (
	get_capability_contract,
	evaluate_capability_rules,
)

try:
	from .federated_mesh import FederatedIdentityMesh, MeshNode, IdentityAssertion, TrustLevel
	_FEDERATED_MESH_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
	FederatedIdentityMesh = None
	MeshNode = Any
	IdentityAssertion = Any
	TrustLevel = Any
	_FEDERATED_MESH_IMPORT_ERROR = exc

try:
	from .privacy_analytics import PrivacyAnalyticsEngine, PrivacyPreservingQuery, AnalyticsQuery
	_PRIVACY_ANALYTICS_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
	PrivacyAnalyticsEngine = None
	PrivacyPreservingQuery = Any
	AnalyticsQuery = Any
	_PRIVACY_ANALYTICS_IMPORT_ERROR = exc

try:
	from .session_manager import EnhancedSessionManager, EnhancedSession, SessionType, RiskLevel as SessionRiskLevel
	_SESSION_MANAGER_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
	EnhancedSessionManager = None
	EnhancedSession = Any
	SessionType = Any
	SessionRiskLevel = Any
	_SESSION_MANAGER_IMPORT_ERROR = exc

class UserStatus(str, Enum):
	"""User account status options"""
	ACTIVE = "active"
	INACTIVE = "inactive"
	SUSPENDED = "suspended"
	PENDING_VERIFICATION = "pending_verification"
	LOCKED = "locked"
	ARCHIVED = "archived"

class SessionStatus(str, Enum):
	"""Session status options"""
	ACTIVE = "active"
	EXPIRED = "expired"
	REVOKED = "revoked"

class PermissionLevel(str, Enum):
	"""Permission levels"""
	NONE = "none"
	READ = "read"
	WRITE = "write"
	DELETE = "delete"
	ADMIN = "admin"
	OWNER = "owner"

class AccessControlModel(str, Enum):
	"""Access control model types"""
	RBAC = "rbac"  # Role-Based Access Control
	ABAC = "abac"  # Attribute-Based Access Control  
	CBAC = "cbac"  # Capability-Based Access Control
	HYBRID = "hybrid"  # Combination of models

class User(BaseModel):
	"""User account model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Unique user identifier")
	email: EmailStr = Field(..., description="User email address")
	username: Optional[str] = Field(default=None, description="Username (optional)")
	password_hash: str = Field(..., description="Hashed password")
	salt: str = Field(default_factory=lambda: secrets.token_hex(32), description="Password salt")
	
	# Profile information
	first_name: Optional[str] = Field(default=None, description="First name")
	last_name: Optional[str] = Field(default=None, description="Last name")
	display_name: Optional[str] = Field(default=None, description="Display name")
	avatar_url: Optional[str] = Field(default=None, description="Avatar image URL")
	
	# Account metadata
	status: UserStatus = Field(default=UserStatus.ACTIVE, description="Account status")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
	last_login_at: Optional[datetime] = Field(default=None, description="Last login timestamp")
	email_verified_at: Optional[datetime] = Field(default=None, description="Email verification timestamp")
	
	# Multi-tenant support
	tenant_id: Optional[str] = Field(default=None, description="Primary tenant ID")
	tenant_memberships: Set[str] = Field(default_factory=set, description="All tenant memberships")
	
	# Security settings
	mfa_enabled: bool = Field(default=False, description="Multi-factor authentication enabled")
	mfa_secret: Optional[str] = Field(default=None, description="MFA secret key")
	failed_login_attempts: int = Field(default=0, description="Failed login attempt count")
	locked_until: Optional[datetime] = Field(default=None, description="Account lock expiration")
	
	# Preferences
	timezone: str = Field(default="UTC", description="User timezone")
	language: str = Field(default="en", description="Preferred language")
	preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences")

	def update_timestamp(self):
		"""Update the modified timestamp"""
		self.updated_at = datetime.utcnow()

	def is_active(self) -> bool:
		"""Check if user is active"""
		return self.status == UserStatus.ACTIVE

	def is_locked(self) -> bool:
		"""Check if user account is locked"""
		if self.status == UserStatus.LOCKED:
			return True
		if self.locked_until and self.locked_until > datetime.utcnow():
			return True
		return False

	def is_email_verified(self) -> bool:
		"""Check if email is verified"""
		return self.email_verified_at is not None

	def has_tenant_access(self, tenant_id: str) -> bool:
		"""Check if user has access to tenant"""
		return tenant_id in self.tenant_memberships or self.tenant_id == tenant_id

	def get_display_name(self) -> str:
		"""Get user's display name"""
		if self.display_name:
			return self.display_name
		if self.first_name and self.last_name:
			return f"{self.first_name} {self.last_name}"
		if self.first_name:
			return self.first_name
		if self.username:
			return self.username
		return str(self.email)

class Role(BaseModel):
	"""Role definition model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Unique role identifier")
	name: str = Field(..., description="Role name")
	slug: str = Field(..., description="URL-safe role identifier")
	description: Optional[str] = Field(default=None, description="Role description")
	
	# Role metadata
	tenant_id: Optional[str] = Field(default=None, description="Tenant scope (None for global)")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
	created_by: Optional[str] = Field(default=None, description="Creator user ID")
	
	# Permissions
	permissions: Set[str] = Field(default_factory=set, description="Role permissions")
	is_system_role: bool = Field(default=False, description="System-defined role")
	
	def update_timestamp(self):
		"""Update the modified timestamp"""
		self.updated_at = datetime.utcnow()

	def has_permission(self, permission: str) -> bool:
		"""Check if role has specific permission"""
		return permission in self.permissions

	def add_permission(self, permission: str):
		"""Add permission to role"""
		self.permissions.add(permission)
		self.update_timestamp()

	def remove_permission(self, permission: str):
		"""Remove permission from role"""
		self.permissions.discard(permission)
		self.update_timestamp()

class UserRole(BaseModel):
	"""User-Role assignment model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Assignment identifier")
	user_id: str = Field(..., description="User identifier")
	role_id: str = Field(..., description="Role identifier")
	tenant_id: Optional[str] = Field(default=None, description="Tenant scope")
	
	# Assignment metadata
	assigned_at: datetime = Field(default_factory=datetime.utcnow, description="Assignment timestamp")
	assigned_by: Optional[str] = Field(default=None, description="Assigner user ID")
	expires_at: Optional[datetime] = Field(default=None, description="Assignment expiration")
	
	def is_expired(self) -> bool:
		"""Check if assignment is expired"""
		return self.expires_at and self.expires_at < datetime.utcnow()

class Session(BaseModel):
	"""User session model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Session identifier")
	user_id: str = Field(..., description="User identifier")
	tenant_id: Optional[str] = Field(default=None, description="Session tenant context")
	
	# Session data
	status: SessionStatus = Field(default=SessionStatus.ACTIVE, description="Session status")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Session creation")
	expires_at: datetime = Field(..., description="Session expiration")
	last_activity_at: datetime = Field(default_factory=datetime.utcnow, description="Last activity")
	
	# Client information
	ip_address: Optional[str] = Field(default=None, description="Client IP address")
	user_agent: Optional[str] = Field(default=None, description="Client user agent")
	
	# Session tokens
	access_token: str = Field(..., description="JWT access token")
	refresh_token: str = Field(default_factory=lambda: secrets.token_urlsafe(32), description="Refresh token")
	
	def is_active(self) -> bool:
		"""Check if session is active"""
		return (self.status == SessionStatus.ACTIVE and 
				self.expires_at > datetime.utcnow())

	def is_expired(self) -> bool:
		"""Check if session is expired"""
		return self.expires_at <= datetime.utcnow()

	def refresh_activity(self):
		"""Update last activity timestamp"""
		self.last_activity_at = datetime.utcnow()

class Attribute(BaseModel):
	"""Attribute model for ABAC"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	name: str = Field(..., description="Attribute name")
	value: Any = Field(..., description="Attribute value")
	data_type: str = Field(..., description="Attribute data type")
	category: str = Field(..., description="Attribute category (subject/resource/action/environment)")
	
	def matches(self, other_value: Any) -> bool:
		"""Check if attribute matches given value"""
		if self.data_type == "string":
			return str(self.value) == str(other_value)
		elif self.data_type == "number":
			return float(self.value) == float(other_value)
		elif self.data_type == "boolean":
			return bool(self.value) == bool(other_value)
		elif self.data_type == "list":
			return other_value in self.value if isinstance(self.value, list) else False
		elif self.data_type == "range":
			if isinstance(self.value, dict) and "min" in self.value and "max" in self.value:
				return self.value["min"] <= other_value <= self.value["max"]
		return False

class Policy(BaseModel):
	"""ABAC Policy model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Policy identifier")
	name: str = Field(..., description="Policy name")
	description: Optional[str] = Field(default=None, description="Policy description")
	
	# Policy rules
	subject_conditions: List[Dict[str, Any]] = Field(default_factory=list, description="Subject attribute conditions")
	resource_conditions: List[Dict[str, Any]] = Field(default_factory=list, description="Resource attribute conditions")
	action_conditions: List[Dict[str, Any]] = Field(default_factory=list, description="Action conditions")
	environment_conditions: List[Dict[str, Any]] = Field(default_factory=list, description="Environment conditions")
	
	# Policy metadata
	tenant_id: Optional[str] = Field(default=None, description="Tenant scope")
	priority: int = Field(default=100, description="Policy priority (lower = higher priority)")
	effect: str = Field(default="allow", description="Policy effect (allow/deny)")
	enabled: bool = Field(default=True, description="Policy enabled status")
	
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	created_by: Optional[str] = Field(default=None, description="Creator user ID")

class Capability(BaseModel):
	"""Capability model for CBAC"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Capability identifier")
	name: str = Field(..., description="Capability name")
	description: Optional[str] = Field(default=None, description="Capability description")
	
	# Capability permissions
	permissions: Set[str] = Field(default_factory=set, description="Specific permissions granted")
	resources: Set[str] = Field(default_factory=set, description="Resources accessible")
	actions: Set[str] = Field(default_factory=set, description="Actions permitted")
	
	# Capability constraints
	constraints: Dict[str, Any] = Field(default_factory=dict, description="Usage constraints")
	expires_at: Optional[datetime] = Field(default=None, description="Capability expiration")
	max_uses: Optional[int] = Field(default=None, description="Maximum number of uses")
	current_uses: int = Field(default=0, description="Current usage count")
	
	# Delegation
	delegatable: bool = Field(default=False, description="Can be delegated to others")
	delegated_by: Optional[str] = Field(default=None, description="Original capability holder")
	delegation_depth: int = Field(default=0, description="Delegation chain depth")
	
	# Metadata
	tenant_id: Optional[str] = Field(default=None, description="Tenant scope")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	granted_to: str = Field(..., description="User/entity granted this capability")
	granted_by: Optional[str] = Field(default=None, description="Grantor user ID")
	
	def is_valid(self) -> bool:
		"""Check if capability is still valid"""
		if self.expires_at and self.expires_at <= datetime.utcnow():
			return False
		if self.max_uses and self.current_uses >= self.max_uses:
			return False
		return True
	
	def can_delegate(self) -> bool:
		"""Check if capability can be delegated"""
		return self.delegatable and self.is_valid()
	
	def use_capability(self) -> bool:
		"""Use the capability (increment usage count)"""
		if not self.is_valid():
			return False
		self.current_uses += 1
		return True

class UserCapability(BaseModel):
	"""User-Capability assignment model"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str, description="Assignment identifier")
	user_id: str = Field(..., description="User identifier")
	capability_id: str = Field(..., description="Capability identifier")
	tenant_id: Optional[str] = Field(default=None, description="Tenant scope")
	
	# Assignment metadata
	assigned_at: datetime = Field(default_factory=datetime.utcnow, description="Assignment timestamp")
	assigned_by: Optional[str] = Field(default=None, description="Assigner user ID")
	expires_at: Optional[datetime] = Field(default=None, description="Assignment expiration")
	
	def is_expired(self) -> bool:
		"""Check if assignment is expired"""
		return self.expires_at and self.expires_at < datetime.utcnow()

class AccessRequest(BaseModel):
	"""Access request for evaluation"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	subject_id: str = Field(..., description="Subject (user) requesting access")
	resource: str = Field(..., description="Resource being accessed")
	action: str = Field(..., description="Action being performed")
	tenant_id: Optional[str] = Field(default=None, description="Tenant context")
	
	# Attributes for ABAC evaluation
	subject_attributes: Dict[str, Any] = Field(default_factory=dict, description="Subject attributes")
	resource_attributes: Dict[str, Any] = Field(default_factory=dict, description="Resource attributes")
	environment_attributes: Dict[str, Any] = Field(default_factory=dict, description="Environment attributes")
	
	# Request metadata
	timestamp: datetime = Field(default_factory=datetime.utcnow, description="Request timestamp")
	ip_address: Optional[str] = Field(default=None, description="Client IP")
	user_agent: Optional[str] = Field(default=None, description="Client user agent")

class AccessDecision(BaseModel):
	"""Access control decision"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	decision: str = Field(..., description="Access decision (allow/deny)")
	reason: str = Field(..., description="Reason for decision")
	model_used: AccessControlModel = Field(..., description="Access control model used")
	policies_evaluated: List[str] = Field(default_factory=list, description="Policies evaluated")
	capabilities_used: List[str] = Field(default_factory=list, description="Capabilities used")
	evaluation_time_ms: float = Field(..., description="Evaluation time in milliseconds")

class AuthContext:
	"""Thread-local authentication context"""
	
	def __init__(self):
		self._local = threading.local()

	def set_user(self, user_id: Optional[str], tenant_id: Optional[str] = None):
		"""Set current user context"""
		self._local.user_id = user_id
		self._local.tenant_id = tenant_id

	def get_user(self) -> Optional[str]:
		"""Get current user context"""
		return getattr(self._local, 'user_id', None)

	def get_tenant(self) -> Optional[str]:
		"""Get current tenant context"""
		return getattr(self._local, 'tenant_id', None)

	def clear(self):
		"""Clear authentication context"""
		self._local.user_id = None
		self._local.tenant_id = None

	@asynccontextmanager
	async def user_scope(self, user_id: str, tenant_id: Optional[str] = None):
		"""Async context manager for user operations"""
		old_user = self.get_user()
		old_tenant = self.get_tenant()
		self.set_user(user_id, tenant_id)
		try:
			yield user_id
		finally:
			self.set_user(old_user, old_tenant)

class PasswordManager:
	"""Password hashing and validation manager"""
	
	@staticmethod
	def hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
		"""Hash password with salt"""
		if salt is None:
			salt = secrets.token_hex(32)
		
		# Use PBKDF2 with SHA-256
		password_hash = hashlib.pbkdf2_hmac(
			'sha256',
			password.encode('utf-8'),
			salt.encode('utf-8'),
			100000  # iterations
		)
		
		return password_hash.hex(), salt

	@staticmethod
	def verify_password(password: str, password_hash: str, salt: str) -> bool:
		"""Verify password against hash"""
		computed_hash, _ = PasswordManager.hash_password(password, salt)
		return secrets.compare_digest(computed_hash, password_hash)

class JWTManager:
	"""JWT token management"""
	
	def __init__(self, secret_key: Optional[str] = None):
		self.secret_key = secret_key or secrets.token_urlsafe(32)

	def create_access_token(self, user_id: str, tenant_id: Optional[str] = None,
						    expires_delta: Optional[timedelta] = None) -> str:
		"""Create JWT access token"""
		if expires_delta is None:
			expires_delta = timedelta(hours=1)
		
		expire = datetime.utcnow() + expires_delta
		payload = {
			"sub": user_id,
			"exp": expire,
			"iat": datetime.utcnow(),
			"type": "access"
		}
		
		if tenant_id:
			payload["tenant_id"] = tenant_id
		
		return jwt.encode(payload, self.secret_key, algorithm="HS256")

	def verify_token(self, token: str) -> Dict[str, Any]:
		"""Verify and decode JWT token"""
		try:
			payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
			return payload
		except jwt.ExpiredSignatureError:
			raise ValueError("Token has expired")
		except jwt.InvalidTokenError:
			raise ValueError("Invalid token")

class ABACEngine:
	"""Attribute-Based Access Control Engine"""
	
	def __init__(self):
		self._policies: Dict[str, Policy] = {}
		self._attributes: Dict[str, Dict[str, Any]] = {}  # entity_id -> attributes

	def add_policy(self, policy: Policy):
		"""Add ABAC policy"""
		self._policies[policy.id] = policy

	def set_attributes(self, entity_id: str, attributes: Dict[str, Any]):
		"""Set attributes for an entity"""
		self._attributes[entity_id] = attributes

	def get_attributes(self, entity_id: str) -> Dict[str, Any]:
		"""Get attributes for an entity"""
		return self._attributes.get(entity_id, {})

	async def evaluate_access(self, request: AccessRequest) -> AccessDecision:
		"""Evaluate access request using ABAC policies"""
		start_time = datetime.utcnow()
		
		# Get subject attributes
		subject_attrs = {
			**self.get_attributes(request.subject_id),
			**request.subject_attributes
		}
		
		# Evaluate policies
		applicable_policies = []
		for policy in self._policies.values():
			if policy.enabled and (not policy.tenant_id or policy.tenant_id == request.tenant_id):
				if self._policy_applies(policy, request, subject_attrs):
					applicable_policies.append(policy)
		
		# Sort by priority
		applicable_policies.sort(key=lambda p: p.priority)
		
		# Evaluate policies (first match wins)
		decision = "deny"  # Default deny
		reason = "No applicable policies found"
		policies_evaluated = []
		
		for policy in applicable_policies:
			policies_evaluated.append(policy.id)
			if self._evaluate_policy_conditions(policy, request, subject_attrs):
				decision = policy.effect
				reason = f"Policy {policy.name} evaluated to {policy.effect}"
				break
		
		evaluation_time = (datetime.utcnow() - start_time).total_seconds() * 1000
		
		return AccessDecision(
			decision=decision,
			reason=reason,
			model_used=AccessControlModel.ABAC,
			policies_evaluated=policies_evaluated,
			evaluation_time_ms=evaluation_time
		)

	def _policy_applies(self, policy: Policy, request: AccessRequest, 
						subject_attrs: Dict[str, Any]) -> bool:
		"""Check if policy applies to the request"""
		# Basic checks - could be more sophisticated
		return True  # For now, assume all policies are potentially applicable

	def _evaluate_policy_conditions(self, policy: Policy, request: AccessRequest,
									subject_attrs: Dict[str, Any]) -> bool:
		"""Evaluate policy conditions"""
		# Evaluate subject conditions
		if policy.subject_conditions:
			if not self._evaluate_conditions(policy.subject_conditions, subject_attrs):
				return False
		
		# Evaluate resource conditions  
		if policy.resource_conditions:
			if not self._evaluate_conditions(policy.resource_conditions, request.resource_attributes):
				return False
				
		# Evaluate action conditions
		if policy.action_conditions:
			action_attrs = {"action": request.action}
			if not self._evaluate_conditions(policy.action_conditions, action_attrs):
				return False
				
		# Evaluate environment conditions
		if policy.environment_conditions:
			if not self._evaluate_conditions(policy.environment_conditions, request.environment_attributes):
				return False
		
		return True

	def _evaluate_conditions(self, conditions: List[Dict[str, Any]], 
							 attributes: Dict[str, Any]) -> bool:
		"""Evaluate attribute conditions"""
		for condition in conditions:
			attr_name = condition.get("attribute")
			operator = condition.get("operator", "equals")
			expected_value = condition.get("value")
			
			actual_value = attributes.get(attr_name)
			
			if not self._evaluate_condition(actual_value, operator, expected_value):
				return False
		
		return True

	def _evaluate_condition(self, actual: Any, operator: str, expected: Any) -> bool:
		"""Evaluate a single condition"""
		if operator == "equals":
			return actual == expected
		elif operator == "not_equals":
			return actual != expected
		elif operator == "in":
			return actual in expected if isinstance(expected, (list, set)) else False
		elif operator == "not_in":
			return actual not in expected if isinstance(expected, (list, set)) else True
		elif operator == "greater_than":
			return actual > expected
		elif operator == "less_than":
			return actual < expected
		elif operator == "greater_equal":
			return actual >= expected
		elif operator == "less_equal":
			return actual <= expected
		elif operator == "contains":
			return expected in str(actual)
		elif operator == "starts_with":
			return str(actual).startswith(str(expected))
		elif operator == "ends_with":
			return str(actual).endswith(str(expected))
		
		return False

class CBACEngine:
	"""Capability-Based Access Control Engine"""
	
	def __init__(self):
		self._capabilities: Dict[str, Capability] = {}
		self._user_capabilities: List[UserCapability] = []

	def create_capability(self, capability: Capability) -> Capability:
		"""Create new capability"""
		self._capabilities[capability.id] = capability
		return capability

	def grant_capability(self, user_id: str, capability_id: str, 
						 tenant_id: Optional[str] = None,
						 granted_by: Optional[str] = None) -> UserCapability:
		"""Grant capability to user"""
		assignment = UserCapability(
			user_id=user_id,
			capability_id=capability_id,
			tenant_id=tenant_id,
			assigned_by=granted_by
		)
		self._user_capabilities.append(assignment)
		return assignment

	def delegate_capability(self, from_user: str, to_user: str, capability_id: str,
							tenant_id: Optional[str] = None) -> Optional[Capability]:
		"""Delegate capability from one user to another"""
		# Find original capability
		original_cap = self._capabilities.get(capability_id)
		if not original_cap or not original_cap.can_delegate():
			return None
		
		# Create delegated capability
		delegated_cap = Capability(
			name=f"Delegated: {original_cap.name}",
			description=f"Delegated from {from_user}",
			permissions=original_cap.permissions.copy(),
			resources=original_cap.resources.copy(),
			actions=original_cap.actions.copy(),
			constraints=original_cap.constraints.copy(),
			expires_at=original_cap.expires_at,
			delegatable=original_cap.delegatable,
			delegated_by=from_user,
			delegation_depth=original_cap.delegation_depth + 1,
			tenant_id=tenant_id,
			granted_to=to_user,
			granted_by=from_user
		)
		
		# Store delegated capability
		self._capabilities[delegated_cap.id] = delegated_cap
		
		# Grant to target user
		self.grant_capability(to_user, delegated_cap.id, tenant_id, from_user)
		
		return delegated_cap

	async def evaluate_access(self, request: AccessRequest) -> AccessDecision:
		"""Evaluate access request using capabilities"""
		start_time = datetime.utcnow()
		
		# Get user capabilities
		user_caps = [
			assignment for assignment in self._user_capabilities
			if (assignment.user_id == request.subject_id and
				assignment.tenant_id == request.tenant_id and
				not assignment.is_expired())
		]
		
		capabilities_used = []
		decision = "deny"
		reason = "No valid capabilities found"
		
		# Check each capability
		for assignment in user_caps:
			capability = self._capabilities.get(assignment.capability_id)
			if not capability or not capability.is_valid():
				continue
				
			capabilities_used.append(capability.id)
			
			# Check if capability grants access to resource and action
			if self._capability_grants_access(capability, request):
				# Use the capability
				capability.use_capability()
				decision = "allow"
				reason = f"Capability {capability.name} grants access"
				break
		
		evaluation_time = (datetime.utcnow() - start_time).total_seconds() * 1000
		
		return AccessDecision(
			decision=decision,
			reason=reason,
			model_used=AccessControlModel.CBAC,
			capabilities_used=capabilities_used,
			evaluation_time_ms=evaluation_time
		)

	def _capability_grants_access(self, capability: Capability, request: AccessRequest) -> bool:
		"""Check if capability grants access to the requested resource/action"""
		# Check resource access
		if capability.resources and request.resource not in capability.resources:
			# Check for wildcard patterns
			resource_granted = False
			for resource_pattern in capability.resources:
				if resource_pattern.endswith("*") and request.resource.startswith(resource_pattern[:-1]):
					resource_granted = True
					break
			if not resource_granted:
				return False
		
		# Check action permission
		if capability.actions and request.action not in capability.actions:
			# Check for wildcard patterns
			action_granted = False
			for action_pattern in capability.actions:
				if action_pattern.endswith("*") and request.action.startswith(action_pattern[:-1]):
					action_granted = True
					break
			if not action_granted:
				return False
		
		# Check permission strings
		permission_string = f"{request.resource}.{request.action}"
		if capability.permissions:
			if permission_string not in capability.permissions:
				# Check for wildcard patterns
				permission_granted = False
				for perm_pattern in capability.permissions:
					if perm_pattern.endswith("*") and permission_string.startswith(perm_pattern[:-1]):
						permission_granted = True
						break
				if not permission_granted:
					return False
		
		return True

class RevolutionaryAuthenticationManager:
	"""Revolutionary authentication management system integrating all cutting-edge technologies"""
	
	def __init__(self, jwt_secret: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
		# Traditional components
		self._users: Dict[str, User] = {}
		self._users_by_email: Dict[str, str] = {}
		self._roles: Dict[str, Role] = {}
		self._user_roles: List[UserRole] = []
		self._sessions: Dict[str, Session] = {}  # Legacy sessions
		self._context = AuthContext()
		self.password_manager = PasswordManager()
		self.jwt_manager = JWTManager(jwt_secret)
		
		# Access control engines
		self.abac_engine = ABACEngine()
		self.cbac_engine = CBACEngine()
		self.access_control_model = AccessControlModel.HYBRID
		
		# Revolutionary components
		self.config = config or {}
		self.behavioral_authenticator = BehavioralAuthenticator()
		self.contextual_risk_engine = ContextualRiskEngine()
		self.quantum_authenticator = QuantumAuthenticator()
		self.zk_proof_authenticator = ZKProofAuthenticator()
		self.biometric_fusion_engine = BiometricFusionEngine()
		self.adaptive_policy_engine = AdaptivePolicyEngine()
		self.identity_graph_engine = IdentityGraphEngine()
		if FederatedIdentityMesh is None:
			raise ModuleNotFoundError(
				"capabilities.common.auth.federated_mesh requires optional cryptography dependencies"
			) from _FEDERATED_MESH_IMPORT_ERROR
		self.federated_mesh = FederatedIdentityMesh()
		self.neuromorphic_processor = NeuromorphicProcessor()
		if PrivacyAnalyticsEngine is None:
			raise ModuleNotFoundError(
				"capabilities.common.auth.privacy_analytics requires optional cryptography dependencies"
			) from _PRIVACY_ANALYTICS_IMPORT_ERROR
		self.privacy_analytics_engine = PrivacyAnalyticsEngine()
		if EnhancedSessionManager is None:
			raise ModuleNotFoundError(
				"capabilities.common.auth.session_manager requires optional cryptography dependencies"
			) from _SESSION_MANAGER_IMPORT_ERROR
		self.enhanced_session_manager = EnhancedSessionManager(config)
		
		# Enhanced user storage
		self._enhanced_users: Dict[str, EnhancedUser] = {}
		
		# Performance metrics
		self.auth_metrics = {
			"traditional_auth_count": 0,
			"behavioral_auth_count": 0,
			"quantum_auth_count": 0,
			"zk_proof_auth_count": 0,
			"biometric_auth_count": 0,
			"neuromorphic_auth_count": 0,
			"total_authentications": 0,
			"average_auth_time_ms": 0.0,
			"fraud_prevented": 0,
			"privacy_queries": 0
		}
		
		self._initialized = False

	async def initialize(self):
		"""Initialize revolutionary authentication system"""
		if self._initialized:
			return
		
		# Initialize traditional components
		await self._create_system_roles()
		
		# Initialize revolutionary components
		await self._initialize_revolutionary_components()
		
		self._initialized = True
	
	async def _initialize_revolutionary_components(self):
		"""Initialize all revolutionary authentication components"""
		try:
			# Initialize behavioral authentication baselines
			await self.behavioral_authenticator.initialize_baseline_models()
			
			# Initialize quantum cryptography
			await self.quantum_authenticator.initialize_quantum_keys()
			
			# Initialize biometric fusion models
			await self.biometric_fusion_engine.initialize_fusion_models()
			
			# Initialize neuromorphic processor
			# (Already initialized in constructor)
			
			# Initialize identity graph
			await self.identity_graph_engine.initialize_graph_database()
			
			# Initialize privacy analytics
			# (Already initialized in constructor)
			
			print("Revolutionary authentication components initialized successfully")
			
		except Exception as e:
			print(f"Error initializing revolutionary components: {e}")
	
	async def revolutionary_authenticate(
		self,
		email: str,
		auth_data: Dict[str, Any],
		device_info: Dict[str, Any],
		context: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Revolutionary multi-modal authentication using all advanced techniques"""
		start_time = asyncio.get_event_loop().time()
		
		try:
			# Get user
			user = await self.get_user_by_email(email)
			if not user:
				return {"success": False, "reason": "user_not_found"}
			
			enhanced_user = self._enhanced_users.get(user.id)
			if not enhanced_user:
				enhanced_user = await self._create_enhanced_user(user)
			
			auth_results = {"methods": [], "overall_score": 0.0, "decision": "deny"}
			
			# 1. Traditional password check (if provided)
			if "password" in auth_data:
				password_valid = self.password_manager.verify_password(
					auth_data["password"], user.password_hash, user.salt
				)
				if password_valid:
					auth_results["methods"].append({
						"method": "password",
						"score": 0.8,
						"confidence": 0.9
					})
					self.auth_metrics["traditional_auth_count"] += 1
			
			# 2. Behavioral authentication
			if "behavioral_data" in auth_data:
				behavioral_score = await self.behavioral_authenticator.analyze_user_patterns(
					user.id, auth_data["behavioral_data"]
				)
				auth_results["methods"].append({
					"method": "behavioral",
					"score": behavioral_score.confidence,
					"confidence": behavioral_score.confidence,
					"anomalies": behavioral_score.anomalies
				})
				self.auth_metrics["behavioral_auth_count"] += 1
			
			# 3. Biometric fusion (if available)
			if "biometric_data" in auth_data:
				fusion_result = await self.biometric_fusion_engine.authenticate_user(
					user.id, auth_data["biometric_data"]
				)
				auth_results["methods"].append({
					"method": "biometric_fusion",
					"score": fusion_result.confidence_score,
					"confidence": fusion_result.confidence_score,
					"liveness_verified": fusion_result.liveness_verified,
					"modalities_used": [m.value for m in fusion_result.modalities_used]
				})
				self.auth_metrics["biometric_auth_count"] += 1
			
			# 4. Quantum authentication (if quantum keys available)
			if enhanced_user and enhanced_user.quantum_keys:
				if "quantum_challenge_response" in auth_data:
					quantum_result = await self.quantum_authenticator.verify_quantum_signature(
						user.id, auth_data["quantum_challenge_response"]
					)
					auth_results["methods"].append({
						"method": "quantum",
						"score": 1.0 if quantum_result else 0.0,
						"confidence": 1.0 if quantum_result else 0.0
					})
					self.auth_metrics["quantum_auth_count"] += 1
			
			# 5. Zero-knowledge proof authentication
			if "zk_proof" in auth_data:
				zk_result = await self.zk_proof_authenticator.verify_proof(
					user.id, auth_data["zk_proof"]
				)
				auth_results["methods"].append({
					"method": "zero_knowledge",
					"score": 1.0 if zk_result else 0.0,
					"confidence": 1.0 if zk_result else 0.0,
					"privacy_preserved": True
				})
				self.auth_metrics["zk_proof_auth_count"] += 1
			
			# 6. Contextual risk assessment
			risk_assessment = await self.contextual_risk_engine.assess_authentication_risk(
				user.id, device_info, context
			)
			
			# 7. Neuromorphic processing for final decision
			neuro_context = NeuroContext(
				user_id=user.id,
				session_id=context.get("session_id", "unknown"),
				behavioral_features=auth_data.get("behavioral_features", [0.5] * 20),
				biometric_features=auth_data.get("biometric_features", [0.5] * 15),
				contextual_features=auth_data.get("contextual_features", [0.5] * 10),
				risk_indicators=risk_assessment.risk_factors
			)
			
			neuro_decision, confidence, neuro_metadata = await self.neuromorphic_processor.process_authentication(neuro_context)
			
			auth_results["neuromorphic_decision"] = {
				"decision": neuro_decision.value,
				"confidence": confidence,
				"processing_time_ms": neuro_metadata.get("processing_time_ms", 0)
			}
			self.auth_metrics["neuromorphic_auth_count"] += 1
			
			# Calculate overall score
			method_scores = [m["score"] for m in auth_results["methods"]]
			if method_scores:
				auth_results["overall_score"] = sum(method_scores) / len(method_scores)
				
				# Apply neuromorphic confidence weighting
				auth_results["overall_score"] = (
					auth_results["overall_score"] * 0.7 + 
					confidence * 0.3
				)
			
			# Final decision logic
			if neuro_decision.value == "allow" and auth_results["overall_score"] > 0.6:
				auth_results["decision"] = "allow"
			elif neuro_decision.value == "challenge" or auth_results["overall_score"] > 0.4:
				auth_results["decision"] = "challenge"
			else:
				auth_results["decision"] = "deny"
			
			# Update metrics
			auth_time = (asyncio.get_event_loop().time() - start_time) * 1000
			self.auth_metrics["total_authentications"] += 1
			self.auth_metrics["average_auth_time_ms"] = (
				(self.auth_metrics["average_auth_time_ms"] * (self.auth_metrics["total_authentications"] - 1) + 
				 auth_time) / self.auth_metrics["total_authentications"]
			)
			
			# Privacy-preserving analytics (async)
			asyncio.create_task(self._record_authentication_analytics(user.id, auth_results, context))
			
			# Update user context
			if auth_results["decision"] == "allow":
				user.last_login_at = datetime.utcnow()
				user.failed_login_attempts = 0
				
				# Update behavioral baseline
				if "behavioral_data" in auth_data:
					await self.behavioral_authenticator.update_user_baseline(
						user.id, auth_data["behavioral_data"]
					)
				
				# Update identity graph
				await self.identity_graph_engine.record_authentication_event(
					user.id, device_info, context, True
				)
			else:
				user.failed_login_attempts += 1
				await self.identity_graph_engine.record_authentication_event(
					user.id, device_info, context, False
				)
			
			auth_results["processing_time_ms"] = auth_time
			auth_results["user_id"] = user.id
			
			return {"success": True, "auth_results": auth_results}
			
		except Exception as e:
			return {"success": False, "reason": "authentication_error", "error": str(e)}
	
	async def create_enhanced_session(
		self,
		user_id: str,
		auth_results: Dict[str, Any],
		device_info: Dict[str, Any],
		security_context: Optional[Dict[str, Any]] = None
	) -> EnhancedSession:
		"""Create enhanced session with revolutionary features"""
		
		# Determine session type based on device
		session_type = SessionType.WEB
		if "mobile" in device_info.get("user_agent", "").lower():
			session_type = SessionType.MOBILE
		elif "api" in device_info.get("source", "").lower():
			session_type = SessionType.API
		
		# Create enhanced session
		enhanced_session = await self.enhanced_session_manager.create_session(
			user_id=user_id,
			session_type=session_type,
			device_info=device_info,
			security_context=security_context
		)
		
		# Add authentication method metadata
		enhanced_session.metadata["auth_methods"] = [
			m["method"] for m in auth_results.get("methods", [])
		]
		enhanced_session.metadata["auth_score"] = auth_results.get("overall_score", 0.0)
		enhanced_session.metadata["neuromorphic_confidence"] = auth_results.get(
			"neuromorphic_decision", {}
		).get("confidence", 0.0)
		
		return enhanced_session
	
	async def _create_enhanced_user(self, user: User) -> EnhancedUser:
		"""Create enhanced user with revolutionary features"""
		enhanced_user = EnhancedUser(
			# Copy basic user data
			id=user.id,
			email=user.email,
			username=user.username,
			password_hash=user.password_hash,
			salt=user.salt,
			first_name=user.first_name,
			last_name=user.last_name,
			display_name=user.display_name,
			avatar_url=user.avatar_url,
			status=user.status,
			created_at=user.created_at,
			updated_at=user.updated_at,
			last_login_at=user.last_login_at,
			email_verified_at=user.email_verified_at,
			tenant_id=user.tenant_id,
			tenant_memberships=user.tenant_memberships,
			mfa_enabled=user.mfa_enabled,
			mfa_secret=user.mfa_secret,
			failed_login_attempts=user.failed_login_attempts,
			locked_until=user.locked_until,
			timezone=user.timezone,
			language=user.language,
			preferences=user.preferences,
			
			# Initialize revolutionary features
			behavioral_baseline=None,
			biometric_templates=[],
			quantum_keys=[],
			privacy_preferences=None,
			identity_graph_score=0.5,
			trust_score=0.5,
			neuromorphic_profile={},
			authentication_history=[],
			enhanced_metadata={}
		)
		
		self._enhanced_users[user.id] = enhanced_user
		return enhanced_user
	
	async def _record_authentication_analytics(
		self, 
		user_id: str, 
		auth_results: Dict[str, Any], 
		context: Dict[str, Any]
	) -> None:
		"""Record authentication data for privacy-preserving analytics"""
		try:
			analytics_data = {
				"timestamp": datetime.utcnow().isoformat(),
				"auth_methods": [m["method"] for m in auth_results.get("methods", [])],
				"overall_score": auth_results.get("overall_score", 0.0),
				"decision": auth_results.get("decision", "deny"),
				"device_type": context.get("device_type", "unknown"),
				"location": context.get("location"),
				"risk_factors": auth_results.get("risk_factors", []),
				"processing_time": auth_results.get("processing_time_ms", 0)
			}
			
			# Ingest data with privacy preservation
			await self.privacy_analytics_engine.ingest_authentication_data(
				user_id=user_id,
				authentication_data=analytics_data,
				privacy_level=3  # High privacy level
			)
			
			self.auth_metrics["privacy_queries"] += 1
			
		except Exception as e:
			print(f"Error recording authentication analytics: {e}")
	
	async def get_authentication_insights(
		self, 
		time_window_hours: int = 24,
		privacy_budget: float = 0.2
	) -> Dict[str, Any]:
		"""Get privacy-preserving authentication insights"""
		try:
			# Analyze patterns
			patterns = await self.privacy_analytics_engine.analyze_authentication_patterns(
				time_window_hours=time_window_hours,
				privacy_budget=privacy_budget * 0.6
			)
			
			# Detect anomalies
			anomalies = await self.privacy_analytics_engine.detect_authentication_anomalies(
				sensitivity_threshold=2.0,
				privacy_budget=privacy_budget * 0.4
			)
			
			return {
				"patterns": patterns,
				"anomalies": anomalies,
				"privacy_cost": privacy_budget,
				"timestamp": datetime.utcnow().isoformat()
			}
			
		except Exception as e:
			return {"error": str(e)}
	
	async def federated_authenticate(
		self,
		identity_assertion: IdentityAssertion,
		source_node_id: str,
		context: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Authenticate using federated identity mesh"""
		try:
			# Verify assertion through mesh
			verification_result = await self.federated_mesh.verify_identity_assertion(
				assertion=identity_assertion,
				verifying_node_id="local_node",  # This node
				context=context
			)
			
			if verification_result:
				# Create or update local user based on federated identity
				user = await self._handle_federated_user(identity_assertion)
				
				return {
					"success": True,
					"user_id": user.id,
					"trust_level": identity_assertion.trust_level.value,
					"federated": True,
					"source_node": source_node_id
				}
			else:
				return {
					"success": False, 
					"reason": "assertion_verification_failed"
				}
				
		except Exception as e:
			return {
				"success": False, 
				"reason": "federated_auth_error", 
				"error": str(e)
			}
	
	async def _handle_federated_user(self, assertion: IdentityAssertion) -> User:
		"""Handle federated user creation or update"""
		# Check if user exists locally
		subject_attributes = assertion.attributes
		email = subject_attributes.get("email")
		
		if email:
			existing_user = await self.get_user_by_email(email)
			if existing_user:
				# Update federated attributes
				existing_user.metadata = existing_user.metadata or {}
				existing_user.metadata["federated_identities"] = existing_user.metadata.get("federated_identities", [])
				existing_user.metadata["federated_identities"].append({
					"issuer": assertion.issuer_node_id,
					"subject_id": assertion.subject_id,
					"trust_level": assertion.trust_level.value,
					"last_assertion": datetime.utcnow().isoformat()
				})
				return existing_user
		
		# Create new federated user
		federated_user = User(
			email=email or f"federated_{assertion.subject_id}@federated.local",
			username=subject_attributes.get("username", f"fed_{assertion.subject_id}"),
			password_hash="federated_user",  # No local password
			salt="federated",
			first_name=subject_attributes.get("first_name"),
			last_name=subject_attributes.get("last_name"),
			status=UserStatus.ACTIVE,
			metadata={
				"federated": True,
				"federated_identities": [{
					"issuer": assertion.issuer_node_id,
					"subject_id": assertion.subject_id,
					"trust_level": assertion.trust_level.value,
					"first_assertion": datetime.utcnow().isoformat()
				}]
			}
		)
		
		self._users[federated_user.id] = federated_user
		if email:
			self._users_by_email[email] = federated_user.id
		
		return federated_user
	
	async def get_revolutionary_metrics(self) -> Dict[str, Any]:
		"""Get comprehensive metrics for revolutionary authentication system"""
		try:
			# Authentication metrics
			auth_metrics = self.auth_metrics.copy()
			
			# Session metrics
			session_analytics = await self.enhanced_session_manager.get_session_analytics()
			
			# Neuromorphic processor metrics
			neuro_metrics = await self.neuromorphic_processor.get_performance_metrics()
			
			# Privacy analytics report
			privacy_report = await self.privacy_analytics_engine.get_privacy_report()
			
			# Identity graph metrics
			graph_metrics = await self.identity_graph_engine.get_graph_metrics()
			
			# Federated mesh status
			mesh_status = {
				"nodes": len(self.federated_mesh.nodes),
				"trust_relationships": self.federated_mesh.graph.number_of_edges(),
				"successful_authentications": self.federated_mesh.mesh_metrics.get("successful_authentications", 0)
			}
			
			return {
				"authentication_metrics": auth_metrics,
				"session_analytics": session_analytics,
				"neuromorphic_metrics": neuro_metrics,
				"privacy_report": privacy_report,
				"identity_graph": graph_metrics,
				"federated_mesh": mesh_status,
				"timestamp": datetime.utcnow().isoformat(),
				"system_health": "optimal"
			}
			
		except Exception as e:
			return {
				"error": str(e),
				"timestamp": datetime.utcnow().isoformat(),
				"system_health": "degraded"
			}

	# Maintain backward compatibility with existing AuthenticationManager methods
	async def authenticate_user(self, email: str, password: str, tenant_id: Optional[str] = None) -> Optional[User]:
		"""Traditional authentication (backward compatibility)"""
		return await super().authenticate_user(email, password, tenant_id)

	async def _create_system_roles(self):
		"""Create default system roles"""
		system_roles = [
			Role(
				id="super_admin",
				name="Super Administrator",
				slug="super-admin",
				description="Full system access",
				is_system_role=True,
				permissions={"*"}  # All permissions
			),
			Role(
				id="admin",
				name="Administrator",
				slug="admin",
				description="Administrative access",
				is_system_role=True,
				permissions={
					"users.read", "users.create", "users.update", "users.delete",
					"roles.read", "roles.create", "roles.update", "roles.delete",
					"tenants.read", "tenants.create", "tenants.update",
					"system.read", "system.configure"
				}
			),
			Role(
				id="user",
				name="User",
				slug="user",
				description="Standard user access",
				is_system_role=True,
				permissions={
					"profile.read", "profile.update",
					"data.read", "data.create", "data.update"
				}
			),
			Role(
				id="viewer",
				name="Viewer",
				slug="viewer",
				description="Read-only access",
				is_system_role=True,
				permissions={
					"profile.read",
					"data.read"
				}
			)
		]

		for role in system_roles:
			self._roles[role.id] = role

	async def create_user(self, email: str, password: str, tenant_id: Optional[str] = None,
						  **kwargs) -> User:
		"""Create new user account"""
		# Check if user already exists
		if email in self._users_by_email:
			raise ValueError(f"User with email {email} already exists")

		# Hash password
		password_hash, salt = self.password_manager.hash_password(password)

		# Create user
		user = User(
			email=email,
			password_hash=password_hash,
			salt=salt,
			tenant_id=tenant_id,
			**kwargs
		)

		if tenant_id:
			user.tenant_memberships.add(tenant_id)

		# Store user
		self._users[user.id] = user
		self._users_by_email[email] = user.id

		# Assign default role
		default_role = self._roles.get("user")
		if default_role:
			await self.assign_role(user.id, default_role.id, tenant_id)

		# Log user creation
		try:
			from ..audl import audit_log, AuditLevel, AuditEventType
			await audit_log(
				level=AuditLevel.INFO,
				event_type=AuditEventType.USER_CREATED,
				component="authentication",
				action="user_created",
				user_id=user.id,
				tenant_id=tenant_id,
				details={
					"email": email,
					"tenant_id": tenant_id
				}
			)
		except ImportError:
			pass

		return user

	async def authenticate_user(self, email: str, password: str,
								tenant_id: Optional[str] = None) -> Optional[User]:
		"""Authenticate user credentials"""
		user_id = self._users_by_email.get(email)
		if not user_id:
			# Log failed attempt
			try:
				from ..audl import audit_log, AuditLevel, AuditEventType
				await audit_log(
					level=AuditLevel.WARNING,
					event_type=AuditEventType.USER_FAILED_LOGIN,
					component="authentication",
					action="login_failed",
					details={"email": email, "reason": "user_not_found"},
					tenant_id=tenant_id
				)
			except ImportError:
				pass
			return None

		user = self._users[user_id]

		# Check if account is locked
		if user.is_locked():
			try:
				from ..audl import audit_log, AuditLevel, AuditEventType
				await audit_log(
					level=AuditLevel.WARNING,
					event_type=AuditEventType.USER_FAILED_LOGIN,
					component="authentication",
					action="login_failed",
					user_id=user.id,
					details={"reason": "account_locked"},
					tenant_id=tenant_id
				)
			except ImportError:
				pass
			return None

		# Check if user has access to tenant
		if tenant_id and not user.has_tenant_access(tenant_id):
			return None

		# Verify password
		if not self.password_manager.verify_password(password, user.password_hash, user.salt):
			# Increment failed login attempts
			user.failed_login_attempts += 1
			
			# Lock account after 5 failed attempts
			if user.failed_login_attempts >= 5:
				user.status = UserStatus.LOCKED
				user.locked_until = datetime.utcnow() + timedelta(minutes=30)

			# Log failed attempt
			try:
				from ..audl import audit_log, AuditLevel, AuditEventType
				await audit_log(
					level=AuditLevel.WARNING,
					event_type=AuditEventType.USER_FAILED_LOGIN,
					component="authentication",
					action="login_failed",
					user_id=user.id,
					details={"reason": "invalid_password", "attempts": user.failed_login_attempts},
					tenant_id=tenant_id
				)
			except ImportError:
				pass
			return None

		# Reset failed login attempts on successful authentication
		user.failed_login_attempts = 0
		user.last_login_at = datetime.utcnow()
		user.update_timestamp()

		# Log successful login
		try:
			from ..audl import audit_log, AuditLevel, AuditEventType
			await audit_log(
				level=AuditLevel.INFO,
				event_type=AuditEventType.USER_LOGIN,
				component="authentication",
				action="login_successful",
				user_id=user.id,
				tenant_id=tenant_id
			)
		except ImportError:
			pass

		return user

	async def create_session(self, user_id: str, tenant_id: Optional[str] = None,
							 ip_address: Optional[str] = None,
							 user_agent: Optional[str] = None) -> Session:
		"""Create new user session"""
		user = await self.get_user(user_id)
		if not user or not user.is_active():
			raise ValueError("Invalid or inactive user")

		# Create access token
		access_token = self.jwt_manager.create_access_token(user_id, tenant_id)

		# Create session
		session = Session(
			user_id=user_id,
			tenant_id=tenant_id,
			expires_at=datetime.utcnow() + timedelta(hours=8),  # 8 hour session
			ip_address=ip_address,
			user_agent=user_agent,
			access_token=access_token
		)

		# Store session
		self._sessions[session.id] = session

		return session

	async def get_user(self, user_id: str) -> Optional[User]:
		"""Get user by ID"""
		return self._users.get(user_id)

	async def get_user_by_email(self, email: str) -> Optional[User]:
		"""Get user by email"""
		user_id = self._users_by_email.get(email)
		if user_id:
			return await self.get_user(user_id)
		return None

	async def assign_role(self, user_id: str, role_id: str, tenant_id: Optional[str] = None,
						  assigned_by: Optional[str] = None):
		"""Assign role to user"""
		user = await self.get_user(user_id)
		role = self._roles.get(role_id)
		
		if not user or not role:
			raise ValueError("Invalid user or role")

		# Check if assignment already exists
		for assignment in self._user_roles:
			if (assignment.user_id == user_id and 
				assignment.role_id == role_id and 
				assignment.tenant_id == tenant_id):
				return assignment

		# Create assignment
		assignment = UserRole(
			user_id=user_id,
			role_id=role_id,
			tenant_id=tenant_id,
			assigned_by=assigned_by
		)

		self._user_roles.append(assignment)

		# Log role assignment
		try:
			from ..audl import audit_log, AuditLevel, AuditEventType
			await audit_log(
				level=AuditLevel.INFO,
				event_type=AuditEventType.PERMISSION_GRANTED,
				component="authentication",
				action="role_assigned",
				user_id=user_id,
				tenant_id=tenant_id,
				details={
					"role_id": role_id,
					"role_name": role.name,
					"assigned_by": assigned_by
				}
			)
		except ImportError:
			pass

		return assignment

	async def check_permission(self, user_id: str, permission: str,
							   tenant_id: Optional[str] = None) -> bool:
		"""Check if user has specific permission using RBAC"""
		user = await self.get_user(user_id)
		if not user or not user.is_active():
			return False

		# Get user roles for tenant
		user_roles = [
			assignment for assignment in self._user_roles
			if (assignment.user_id == user_id and
				assignment.tenant_id == tenant_id and
				not assignment.is_expired())
		]

		# Check permissions in roles
		for assignment in user_roles:
			role = self._roles.get(assignment.role_id)
			if role:
				# Super admin has all permissions
				if "*" in role.permissions:
					return True
				# Check specific permission
				if permission in role.permissions:
					return True
				# Check wildcard permissions
				for perm in role.permissions:
					if perm.endswith("*") and permission.startswith(perm[:-1]):
						return True

		return False

	async def evaluate_access(self, user_id: str, resource: str, action: str,
							  tenant_id: Optional[str] = None,
							  attributes: Optional[Dict[str, Any]] = None) -> AccessDecision:
		"""Comprehensive access evaluation using configured access control model"""
		request = AccessRequest(
			subject_id=user_id,
			resource=resource,
			action=action,
			tenant_id=tenant_id,
			subject_attributes=attributes or {},
			resource_attributes={},
			environment_attributes={}
		)

		if self.access_control_model == AccessControlModel.RBAC:
			return await self._evaluate_rbac(request)
		elif self.access_control_model == AccessControlModel.ABAC:
			return await self.abac_engine.evaluate_access(request)
		elif self.access_control_model == AccessControlModel.CBAC:
			return await self.cbac_engine.evaluate_access(request)
		elif self.access_control_model == AccessControlModel.HYBRID:
			return await self._evaluate_hybrid(request)
		
		return AccessDecision(
			decision="deny",
			reason="Unknown access control model",
			model_used=self.access_control_model,
			evaluation_time_ms=0.0
		)

	async def _evaluate_rbac(self, request: AccessRequest) -> AccessDecision:
		"""Evaluate access using RBAC"""
		start_time = datetime.utcnow()
		
		permission = f"{request.resource}.{request.action}"
		has_permission = await self.check_permission(
			request.subject_id, permission, request.tenant_id
		)
		
		evaluation_time = (datetime.utcnow() - start_time).total_seconds() * 1000
		
		return AccessDecision(
			decision="allow" if has_permission else "deny",
			reason="RBAC permission check",
			model_used=AccessControlModel.RBAC,
			evaluation_time_ms=evaluation_time
		)

	async def _evaluate_hybrid(self, request: AccessRequest) -> AccessDecision:
		"""Evaluate access using hybrid approach (RBAC + ABAC + CBAC)"""
		start_time = datetime.utcnow()
		
		# Try RBAC first (fastest)
		rbac_decision = await self._evaluate_rbac(request)
		if rbac_decision.decision == "allow":
			rbac_decision.model_used = AccessControlModel.HYBRID
			rbac_decision.reason = "RBAC (in hybrid mode) - " + rbac_decision.reason
			return rbac_decision
		
		# Try CBAC next (capabilities)
		cbac_decision = await self.cbac_engine.evaluate_access(request)
		if cbac_decision.decision == "allow":
			cbac_decision.model_used = AccessControlModel.HYBRID
			cbac_decision.reason = "CBAC (in hybrid mode) - " + cbac_decision.reason
			return cbac_decision
		
		# Finally try ABAC (most flexible but slowest)
		abac_decision = await self.abac_engine.evaluate_access(request)
		abac_decision.model_used = AccessControlModel.HYBRID
		abac_decision.reason = "ABAC (in hybrid mode) - " + abac_decision.reason
		
		evaluation_time = (datetime.utcnow() - start_time).total_seconds() * 1000
		abac_decision.evaluation_time_ms = evaluation_time
		
		return abac_decision

	# Flask-AppBuilder Integration Methods
	def create_fab_role_mapping(self, fab_role_name: str, apg_role_id: str):
		"""Create mapping between Flask-AppBuilder role and APG role"""
		if not hasattr(self, '_fab_role_mappings'):
			self._fab_role_mappings = {}
		self._fab_role_mappings[fab_role_name] = apg_role_id

	def get_fab_role_mapping(self, fab_role_name: str) -> Optional[str]:
		"""Get APG role ID for Flask-AppBuilder role"""
		if not hasattr(self, '_fab_role_mappings'):
			return None
		return self._fab_role_mappings.get(fab_role_name)

	async def sync_with_fab_security_manager(self, fab_security_manager):
		"""Synchronize with Flask-AppBuilder Security Manager"""
		try:
			# Import Flask-AppBuilder types
			from flask_appbuilder.security.sqla.models import User as FabUser, Role as FabRole
			
			# Sync roles from FAB to APG
			fab_roles = fab_security_manager.get_all_roles()
			for fab_role in fab_roles:
				# Create corresponding APG role if not exists
				apg_role_id = self.get_fab_role_mapping(fab_role.name)
				if not apg_role_id:
					apg_role = Role(
						name=fab_role.name,
						slug=fab_role.name.lower().replace(" ", "-"),
						description=f"Synced from Flask-AppBuilder: {fab_role.name}",
						tenant_id=None,  # Global role
						is_system_role=True
					)
					
					# Map FAB permissions to APG permissions
					fab_permissions = fab_role.permissions or []
					for fab_perm in fab_permissions:
						permission_name = f"{fab_perm.view_menu.name}.{fab_perm.permission.name}"
						apg_role.add_permission(permission_name)
					
					self._roles[apg_role.id] = apg_role
					self.create_fab_role_mapping(fab_role.name, apg_role.id)
			
			# Sync users from FAB to APG
			fab_users = fab_security_manager.get_all_users()
			for fab_user in fab_users:
				# Check if user exists in APG
				existing_user = await self.get_user_by_email(fab_user.email)
				if not existing_user:
					# Create APG user (without password since FAB manages it)
					apg_user = User(
						email=fab_user.email,
						username=fab_user.username,
						password_hash="managed_by_fab",  # Placeholder
						salt="managed_by_fab",
						first_name=fab_user.first_name,
						last_name=fab_user.last_name,
						status=UserStatus.ACTIVE if fab_user.active else UserStatus.INACTIVE
					)
					self._users[apg_user.id] = apg_user
					self._users_by_email[fab_user.email] = apg_user.id
					
					# Sync user roles
					for fab_role in fab_user.roles:
						apg_role_id = self.get_fab_role_mapping(fab_role.name)
						if apg_role_id:
							await self.assign_role(apg_user.id, apg_role_id)
		
		except ImportError:
			# Flask-AppBuilder not available
			pass
		except Exception as e:
			# Log error but don't fail initialization
			print(f"Error syncing with Flask-AppBuilder: {e}")

	def has_fab_permission(self, user_id: str, permission_name: str, view_name: str) -> bool:
		"""Check Flask-AppBuilder style permission"""
		fab_permission = f"{view_name}.{permission_name}"
		return asyncio.run(self.check_permission(user_id, fab_permission))

	def fab_permission_decorator(self, permission_name: str, view_name: str):
		"""Decorator for Flask-AppBuilder style permission checking"""
		def decorator(f):
			async def wrapper(*args, **kwargs):
				user_id = self.get_current_user()
				if not user_id:
					from flask import abort
					abort(401)  # Unauthorized
				
				if not self.has_fab_permission(user_id, permission_name, view_name):
					from flask import abort
					abort(403)  # Forbidden
				
				return await f(*args, **kwargs)
			return wrapper
		return decorator

	# Enhanced ABAC methods
	def set_user_attributes(self, user_id: str, attributes: Dict[str, Any]):
		"""Set attributes for user for ABAC evaluation"""
		self.abac_engine.set_attributes(user_id, attributes)

	def add_abac_policy(self, policy: Policy):
		"""Add ABAC policy"""
		self.abac_engine.add_policy(policy)

	def create_time_based_policy(self, name: str, start_time: str, end_time: str,
								 allowed_actions: List[str], tenant_id: Optional[str] = None) -> Policy:
		"""Create time-based access policy"""
		policy = Policy(
			name=name,
			description=f"Time-based access from {start_time} to {end_time}",
			tenant_id=tenant_id,
			environment_conditions=[
				{
					"attribute": "current_time",
					"operator": "greater_equal", 
					"value": start_time
				},
				{
					"attribute": "current_time",
					"operator": "less_equal",
					"value": end_time
				}
			],
			action_conditions=[
				{
					"attribute": "action",
					"operator": "in",
					"value": allowed_actions
				}
			],
			effect="allow"
		)
		
		self.add_abac_policy(policy)
		return policy

	def create_location_based_policy(self, name: str, allowed_locations: List[str],
									 tenant_id: Optional[str] = None) -> Policy:
		"""Create location-based access policy"""
		policy = Policy(
			name=name,
			description=f"Location-based access for: {', '.join(allowed_locations)}",
			tenant_id=tenant_id,
			environment_conditions=[
				{
					"attribute": "location",
					"operator": "in",
					"value": allowed_locations
				}
			],
			effect="allow"
		)
		
		self.add_abac_policy(policy)
		return policy

	# Enhanced CBAC methods
	def create_capability(self, name: str, permissions: Set[str], resources: Set[str],
						  actions: Set[str], expires_at: Optional[datetime] = None,
						  max_uses: Optional[int] = None, delegatable: bool = False,
						  tenant_id: Optional[str] = None) -> Capability:
		"""Create new capability"""
		capability = Capability(
			name=name,
			description=f"Capability for {name}",
			permissions=permissions,
			resources=resources,
			actions=actions,
			expires_at=expires_at,
			max_uses=max_uses,
			delegatable=delegatable,
			tenant_id=tenant_id,
			granted_to="system",  # Will be updated when granted
			granted_by=self.get_current_user()
		)
		
		return self.cbac_engine.create_capability(capability)

	def grant_capability_to_user(self, user_id: str, capability_id: str,
								 tenant_id: Optional[str] = None) -> UserCapability:
		"""Grant capability to user"""
		return self.cbac_engine.grant_capability(
			user_id, capability_id, tenant_id, self.get_current_user()
		)

	def delegate_capability_between_users(self, from_user: str, to_user: str, 
										  capability_id: str, tenant_id: Optional[str] = None) -> Optional[Capability]:
		"""Delegate capability from one user to another"""
		return self.cbac_engine.delegate_capability(from_user, to_user, capability_id, tenant_id)

	def get_current_user(self) -> Optional[str]:
		"""Get current user from context"""
		return self._context.get_user()

	def set_current_user(self, user_id: Optional[str], tenant_id: Optional[str] = None):
		"""Set current user context"""
		self._context.set_user(user_id, tenant_id)

	@asynccontextmanager
	async def user_scope(self, user_id: str, tenant_id: Optional[str] = None):
		"""Execute operations in specific user context"""
		async with self._context.user_scope(user_id, tenant_id):
			yield user_id

# Global instances
_auth_manager: Optional[RevolutionaryAuthenticationManager] = None

def get_auth_manager() -> RevolutionaryAuthenticationManager:
	"""Get global revolutionary authentication manager instance"""
	global _auth_manager
	if _auth_manager is None:
		_auth_manager = RevolutionaryAuthenticationManager()
	return _auth_manager

async def init_authentication(jwt_secret: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> RevolutionaryAuthenticationManager:
	"""Initialize revolutionary authentication system"""
	global _auth_manager
	_auth_manager = RevolutionaryAuthenticationManager(jwt_secret, config)
	await _auth_manager.initialize()
	return _auth_manager

# Backward compatibility alias
AuthenticationManager = RevolutionaryAuthenticationManager

# Utility functions
def get_current_user() -> Optional[str]:
	"""Get current user ID from context"""
	manager = get_auth_manager()
	return manager.get_current_user()

def set_current_user(user_id: Optional[str], tenant_id: Optional[str] = None):
	"""Set current user context"""
	manager = get_auth_manager()
	manager.set_current_user(user_id, tenant_id)

@asynccontextmanager
async def user_context(user_id: str, tenant_id: Optional[str] = None):
	"""Context manager for user operations"""
	manager = get_auth_manager()
	async with manager.user_scope(user_id, tenant_id):
		yield user_id

async def authenticate(email: str, password: str, tenant_id: Optional[str] = None) -> Optional[User]:
	"""Convenience function for user authentication"""
	manager = get_auth_manager()
	return await manager.authenticate_user(email, password, tenant_id)

async def check_permission(permission: str, user_id: Optional[str] = None,
						   tenant_id: Optional[str] = None) -> bool:
	"""Convenience function for permission checking"""
	manager = get_auth_manager()
	if user_id is None:
		user_id = manager.get_current_user()
	if user_id is None:
		return False
	return await manager.check_permission(user_id, permission, tenant_id)


def get_capability_info(tenant_id: str = "default", overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
	"""Return executable AUTH capability metadata and contract details."""
	contract = get_capability_contract(tenant_id, overrides)
	return {
		"metadata": {
			"capability_id": "common/auth",
			"name": "auth",
			"display_name": "Authentication & RBAC",
			"version": "1.0.0",
			"aliases": ["auth_rbac"],
			"description": "Tenant-aware identity, authentication, and authorization capability"
		},
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
	}


def register_capability() -> Dict[str, Any]:
	"""Register the authentication capability with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "auth",
		"aliases": ["auth_rbac"],
		"display_name": "Authentication & RBAC",
		"description": "Tenant-aware identity, session, and authorization control plane",
		"version": "1.0.0",
		"dependencies": [
			"audl",
			"mten",
			"ntfy",
			"keym"
		],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"identity": "Manage tenant-scoped users and identity context",
			"rbac": "Assign and evaluate roles and permissions",
			"sessions": "Issue, monitor, and revoke enriched sessions",
			"behavioral_auth": "Continuously score behavioral trust signals",
			"biometrics": "Manage biometric fusion enrollment and verification",
			"federation": "Coordinate trusted federated identity mesh flows",
			"capability_rules": "Evaluate deterministic capability-specific controls",
			"visual_theming": "Apply tenant-aware trust workspace theming"
		},
		"endpoints": {
			"login": "/api/auth/login",
			"users": "/api/users",
			"sessions": "/api/sessions",
			"biometrics": "/api/biometrics/register",
			"behavioral": "/api/behavioral/analyze",
			"quantum": "/api/quantum/keys",
			"privacy": "/api/analytics/privacy-query",
			"federation": "/api/federated/authenticate"
		},
		"ui_components": {
			route["name"]: route["path"]
			for route in contract["ui"]["routes"]
		},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"permissions": [
			"auth:view",
			"auth:login",
			"auth:manage_roles",
			"auth:manage_sessions",
			"auth:manage_biometrics",
			"auth:manage_keys",
			"auth:view_risk",
			"auth:view_privacy",
			"auth:manage_privacy",
			"auth:manage_federation",
			"auth:admin"
		]
	}
