"""
Pydantic v2 domain models for APG Authentication & RBAC.

Entities: User, Role, Permission, Policy (ABAC), Session, APIKey,
OAuthClient, OAuthToken, MFADevice, LoginAttempt, PasswordPolicy,
IPAllowlist, Group, Delegation, ServiceAccount, AuditEvent.

© 2025 Datacraft - www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from uuid6 import uuid7


def uuid7str() -> str:
	return str(uuid7())


def _utcnow() -> datetime:
	return datetime.now(timezone.utc)


# ── Status & Type Enums ────────────────────────────────────────────────────────

class UserStatus(str, Enum):
	ACTIVE = "active"
	INACTIVE = "inactive"
	LOCKED = "locked"
	SUSPENDED = "suspended"
	PENDING_VERIFICATION = "pending_verification"
	PASSWORD_RESET_REQUIRED = "password_reset_required"


class SessionStatus(str, Enum):
	ACTIVE = "active"
	EXPIRED = "expired"
	REVOKED = "revoked"
	INVALIDATED = "invalidated"


class SessionType(str, Enum):
	INTERACTIVE = "interactive"
	SERVICE = "service"
	DELEGATED = "delegated"
	IMPERSONATED = "impersonated"


class RoleStatus(str, Enum):
	ACTIVE = "active"
	INACTIVE = "inactive"
	DEPRECATED = "deprecated"


class RoleTier(str, Enum):
	STANDARD = "standard"
	ELEVATED = "elevated"
	PRIVILEGED = "privileged"
	ADMIN = "admin"
	SUPER_ADMIN = "super_admin"


class PermissionEffect(str, Enum):
	ALLOW = "allow"
	DENY = "deny"


class PolicyEffect(str, Enum):
	ALLOW = "allow"
	DENY = "deny"


class MFAMethod(str, Enum):
	TOTP = "totp"
	SMS = "sms"
	EMAIL = "email"
	HARDWARE_KEY = "hardware_key"
	PASSKEY = "passkey"
	BACKUP_CODE = "backup_code"


class MFADeviceStatus(str, Enum):
	ACTIVE = "active"
	PENDING_VERIFICATION = "pending_verification"
	REVOKED = "revoked"


class OAuthGrantType(str, Enum):
	AUTHORIZATION_CODE = "authorization_code"
	CLIENT_CREDENTIALS = "client_credentials"
	REFRESH_TOKEN = "refresh_token"
	DEVICE_CODE = "device_code"


class OAuthTokenStatus(str, Enum):
	ACTIVE = "active"
	EXPIRED = "expired"
	REVOKED = "revoked"


class APIKeyStatus(str, Enum):
	ACTIVE = "active"
	INACTIVE = "inactive"
	EXPIRED = "expired"
	REVOKED = "revoked"


class DelegationStatus(str, Enum):
	ACTIVE = "active"
	EXPIRED = "expired"
	REVOKED = "revoked"


class ServiceAccountStatus(str, Enum):
	ACTIVE = "active"
	INACTIVE = "inactive"
	ROTATING = "rotating"
	DECOMMISSIONED = "decommissioned"


class LoginAttemptOutcome(str, Enum):
	SUCCESS = "success"
	FAILED_CREDENTIALS = "failed_credentials"
	FAILED_MFA = "failed_mfa"
	BLOCKED_LOCKOUT = "blocked_lockout"
	BLOCKED_IP = "blocked_ip"
	BLOCKED_SUSPICIOUS = "blocked_suspicious"


class AuditEventType(str, Enum):
	# Identity
	USER_CREATED = "user.created"
	USER_UPDATED = "user.updated"
	USER_DELETED = "user.deleted"
	USER_LOCKED = "user.locked"
	USER_UNLOCKED = "user.unlocked"
	USER_SUSPENDED = "user.suspended"
	PASSWORD_CHANGED = "user.password_changed"
	PASSWORD_RESET = "user.password_reset"
	# Sessions
	SESSION_STARTED = "session.started"
	SESSION_REVOKED = "session.revoked"
	SESSION_EXPIRED = "session.expired"
	# Auth
	LOGIN_SUCCESS = "auth.login_success"
	LOGIN_FAILURE = "auth.login_failure"
	LOGOUT = "auth.logout"
	MFA_ENROLLED = "auth.mfa_enrolled"
	MFA_VERIFIED = "auth.mfa_verified"
	MFA_FAILED = "auth.mfa_failed"
	TOKEN_ISSUED = "auth.token_issued"
	TOKEN_REVOKED = "auth.token_revoked"
	TOKEN_REFRESHED = "auth.token_refreshed"
	# RBAC
	ROLE_CREATED = "rbac.role_created"
	ROLE_ASSIGNED = "rbac.role_assigned"
	ROLE_REVOKED = "rbac.role_revoked"
	PERMISSION_GRANTED = "rbac.permission_granted"
	PERMISSION_DENIED = "rbac.permission_denied"
	# ABAC
	POLICY_CREATED = "abac.policy_created"
	POLICY_UPDATED = "abac.policy_updated"
	POLICY_DELETED = "abac.policy_deleted"
	ACCESS_EVALUATED = "abac.access_evaluated"
	# API Keys
	API_KEY_CREATED = "apikey.created"
	API_KEY_ROTATED = "apikey.rotated"
	API_KEY_REVOKED = "apikey.revoked"
	API_KEY_VALIDATED = "apikey.validated"
	# Delegation / Impersonation
	DELEGATION_GRANTED = "delegation.granted"
	DELEGATION_REVOKED = "delegation.revoked"
	IMPERSONATION_STARTED = "impersonation.started"
	IMPERSONATION_ENDED = "impersonation.ended"
	# Security
	BRUTE_FORCE_DETECTED = "security.brute_force_detected"
	SUSPICIOUS_LOGIN = "security.suspicious_login"
	IP_BLOCKED = "security.ip_blocked"
	PRIVILEGE_ESCALATION = "security.privilege_escalation"


class RiskLevel(str, Enum):
	LOW = "low"
	MEDIUM = "medium"
	HIGH = "high"
	CRITICAL = "critical"


# ── Base model ─────────────────────────────────────────────────────────────────

class AuthBase(BaseModel):
	"""Shared audit columns for every AUTH entity."""
	model_config = ConfigDict(
		extra="forbid",
		validate_by_name=True,
		validate_by_alias=True,
	)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	created_at: datetime = Field(default_factory=_utcnow)
	updated_at: datetime = Field(default_factory=_utcnow)
	created_by: str = "system"
	is_deleted: bool = False


# ── Users ──────────────────────────────────────────────────────────────────────

class UserCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	email: str
	display_name: str
	password_hash: str | None = None
	status: UserStatus = UserStatus.ACTIVE
	metadata: dict[str, Any] = Field(default_factory=dict)


class UserUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	display_name: str | None = None
	status: UserStatus | None = None
	metadata: dict[str, Any] | None = None


class UserResponse(AuthBase):
	email: str
	display_name: str
	status: UserStatus = UserStatus.ACTIVE
	mfa_enabled: bool = False
	failed_login_count: int = 0
	last_login_at: datetime | None = None
	password_changed_at: datetime | None = None
	metadata: dict[str, Any] = Field(default_factory=dict)


# ── Groups ─────────────────────────────────────────────────────────────────────

class GroupCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	name: str
	description: str = ""
	parent_group_id: str | None = None


class GroupResponse(AuthBase):
	name: str
	description: str = ""
	parent_group_id: str | None = None
	member_count: int = 0


class GroupMembershipResponse(AuthBase):
	group_id: str
	user_id: str
	added_by: str


# ── Permissions ────────────────────────────────────────────────────────────────

class PermissionCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	name: str  # e.g. "auth:manage_users"
	resource: str  # e.g. "users"
	action: str  # e.g. "manage"
	effect: PermissionEffect = PermissionEffect.ALLOW
	description: str = ""


class PermissionResponse(AuthBase):
	name: str
	resource: str
	action: str
	effect: PermissionEffect = PermissionEffect.ALLOW
	description: str = ""


# ── Roles ──────────────────────────────────────────────────────────────────────

class RoleCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	name: str
	tier: RoleTier = RoleTier.STANDARD
	description: str = ""
	permission_ids: list[str] = Field(default_factory=list)


class RoleUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	name: str | None = None
	tier: RoleTier | None = None
	description: str | None = None
	status: RoleStatus | None = None


class RoleResponse(AuthBase):
	name: str
	tier: RoleTier = RoleTier.STANDARD
	status: RoleStatus = RoleStatus.ACTIVE
	description: str = ""
	permission_ids: list[str] = Field(default_factory=list)


class RoleAssignmentCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	user_id: str
	role_id: str
	assigned_by: str
	expires_at: datetime | None = None
	justification: str = ""


class RoleAssignmentResponse(AuthBase):
	user_id: str
	role_id: str
	assigned_by: str
	expires_at: datetime | None = None
	justification: str = ""
	is_active: bool = True


# ── ABAC Policies ──────────────────────────────────────────────────────────────

class PolicyCondition(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	attribute: str  # e.g. "user.department", "resource.classification"
	operator: str   # eq, neq, in, not_in, gt, lt, starts_with, contains
	value: Any


class PolicyCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	name: str
	effect: PolicyEffect = PolicyEffect.ALLOW
	priority: int = 100  # lower = higher priority
	subject_conditions: list[PolicyCondition] = Field(default_factory=list)
	resource_conditions: list[PolicyCondition] = Field(default_factory=list)
	action_conditions: list[PolicyCondition] = Field(default_factory=list)
	environment_conditions: list[PolicyCondition] = Field(default_factory=list)
	description: str = ""


class PolicyUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	name: str | None = None
	effect: PolicyEffect | None = None
	priority: int | None = None
	subject_conditions: list[PolicyCondition] | None = None
	resource_conditions: list[PolicyCondition] | None = None
	action_conditions: list[PolicyCondition] | None = None
	environment_conditions: list[PolicyCondition] | None = None


class PolicyResponse(AuthBase):
	name: str
	effect: PolicyEffect = PolicyEffect.ALLOW
	priority: int = 100
	subject_conditions: list[PolicyCondition] = Field(default_factory=list)
	resource_conditions: list[PolicyCondition] = Field(default_factory=list)
	action_conditions: list[PolicyCondition] = Field(default_factory=list)
	environment_conditions: list[PolicyCondition] = Field(default_factory=list)
	description: str = ""
	is_active: bool = True


class ABACDecisionRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	subject_id: str
	resource: str
	action: str
	environment: dict[str, Any] = Field(default_factory=dict)


class ABACDecisionResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	decision: str  # "allow" | "deny"
	matched_policy: str | None = None
	reasons: list[str] = Field(default_factory=list)
	evaluated_at: datetime = Field(default_factory=_utcnow)


# ── Sessions ───────────────────────────────────────────────────────────────────

class SessionCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	user_id: str
	device_id: str
	ip_address: str
	user_agent: str = ""
	auth_method: str = "password"
	session_type: SessionType = SessionType.INTERACTIVE
	mfa_verified: bool = False
	risk_level: RiskLevel = RiskLevel.LOW
	impersonator_id: str | None = None
	delegation_id: str | None = None


class SessionResponse(AuthBase):
	user_id: str
	device_id: str
	ip_address: str
	user_agent: str = ""
	auth_method: str
	session_type: SessionType = SessionType.INTERACTIVE
	status: SessionStatus = SessionStatus.ACTIVE
	mfa_verified: bool = False
	step_up_completed: bool = False
	risk_level: RiskLevel = RiskLevel.LOW
	trust_score: float = 1.0
	expires_at: datetime | None = None
	last_activity_at: datetime = Field(default_factory=_utcnow)
	impersonator_id: str | None = None
	delegation_id: str | None = None


# ── API Keys ───────────────────────────────────────────────────────────────────

class APIKeyCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	user_id: str | None = None
	service_account_id: str | None = None
	name: str
	scopes: list[str] = Field(default_factory=list)
	expires_at: datetime | None = None


class APIKeyResponse(AuthBase):
	user_id: str | None = None
	service_account_id: str | None = None
	name: str
	key_prefix: str  # first 8 chars of key for display (e.g. "ak_abc123")
	key_hash: str    # stored hash, never raw
	key_salt: str
	scopes: list[str] = Field(default_factory=list)
	status: APIKeyStatus = APIKeyStatus.ACTIVE
	expires_at: datetime | None = None
	last_used_at: datetime | None = None
	use_count: int = 0


# ── OAuth2 Clients ─────────────────────────────────────────────────────────────

class OAuthClientCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	name: str
	redirect_uris: list[str]
	allowed_grant_types: list[OAuthGrantType]
	allowed_scopes: list[str] = Field(default_factory=list)
	is_public: bool = False  # public = PKCE required, no client_secret


class OAuthClientResponse(AuthBase):
	name: str
	client_id: str = Field(default_factory=uuid7str)
	client_secret_hash: str | None = None  # None for public clients
	redirect_uris: list[str]
	allowed_grant_types: list[OAuthGrantType]
	allowed_scopes: list[str] = Field(default_factory=list)
	is_public: bool = False
	is_active: bool = True


class OAuthTokenResponse(AuthBase):
	client_id: str
	user_id: str | None = None
	access_token_hash: str
	refresh_token_hash: str | None = None
	scopes: list[str]
	status: OAuthTokenStatus = OAuthTokenStatus.ACTIVE
	expires_at: datetime
	token_type: str = "Bearer"


# ── MFA Devices ────────────────────────────────────────────────────────────────

class MFADeviceCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	user_id: str
	method: MFAMethod
	device_name: str = ""
	totp_secret: str | None = None   # stored encrypted
	phone_number: str | None = None  # for SMS


class MFADeviceResponse(AuthBase):
	user_id: str
	method: MFAMethod
	device_name: str = ""
	status: MFADeviceStatus = MFADeviceStatus.PENDING_VERIFICATION
	last_used_at: datetime | None = None
	use_count: int = 0


class MFAChallengeResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	challenge_id: str = Field(default_factory=uuid7str)
	user_id: str
	method: MFAMethod
	expires_at: datetime
	# For TOTP: empty (user reads from authenticator app)
	# For SMS/email: masked destination
	destination_hint: str = ""


# ── Login Attempts ─────────────────────────────────────────────────────────────

class LoginAttemptResponse(AuthBase):
	user_id: str | None = None
	email: str
	ip_address: str
	user_agent: str = ""
	outcome: LoginAttemptOutcome
	risk_score: float = 0.0
	risk_factors: list[str] = Field(default_factory=list)
	geo_country: str = ""
	geo_city: str = ""
	device_id: str | None = None


# ── Password Policy ────────────────────────────────────────────────────────────

class PasswordPolicyCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	name: str
	min_length: int = 12
	require_uppercase: bool = True
	require_lowercase: bool = True
	require_digits: bool = True
	require_special: bool = True
	max_age_days: int = 90
	history_count: int = 10  # can't reuse last N passwords
	max_failed_attempts: int = 5
	lockout_duration_minutes: int = 30
	breach_check_enabled: bool = True


class PasswordPolicyResponse(AuthBase):
	name: str
	min_length: int = 12
	require_uppercase: bool = True
	require_lowercase: bool = True
	require_digits: bool = True
	require_special: bool = True
	max_age_days: int = 90
	history_count: int = 10
	max_failed_attempts: int = 5
	lockout_duration_minutes: int = 30
	breach_check_enabled: bool = True
	is_default: bool = False


# ── IP Allowlist ───────────────────────────────────────────────────────────────

class IPAllowlistCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	cidr: str  # e.g. "192.168.1.0/24" or "10.0.0.5/32"
	label: str = ""
	applies_to: str = "all"  # "all" | "admin_only" | role_id


class IPAllowlistResponse(AuthBase):
	cidr: str
	label: str = ""
	applies_to: str = "all"
	is_active: bool = True


# ── Delegations ────────────────────────────────────────────────────────────────

class DelegationCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	delegator_id: str
	delegate_id: str
	permission_ids: list[str]
	expires_at: datetime
	justification: str
	requires_mfa: bool = True


class DelegationResponse(AuthBase):
	delegator_id: str
	delegate_id: str
	permission_ids: list[str]
	status: DelegationStatus = DelegationStatus.ACTIVE
	expires_at: datetime
	justification: str
	requires_mfa: bool = True


# ── Service Accounts ───────────────────────────────────────────────────────────

class ServiceAccountCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	name: str
	description: str = ""
	role_ids: list[str] = Field(default_factory=list)
	key_rotation_days: int = 90


class ServiceAccountResponse(AuthBase):
	name: str
	description: str = ""
	status: ServiceAccountStatus = ServiceAccountStatus.ACTIVE
	role_ids: list[str] = Field(default_factory=list)
	key_rotation_days: int = 90
	last_rotated_at: datetime | None = None
	next_rotation_at: datetime | None = None


# ── Audit Events ───────────────────────────────────────────────────────────────

class AuditEventResponse(AuthBase):
	event_type: AuditEventType
	actor_id: str
	actor_ip: str = ""
	target_id: str = ""
	target_type: str = ""
	outcome: str  # "success" | "failure" | "denied"
	risk_level: RiskLevel = RiskLevel.LOW
	session_id: str | None = None
	details: dict[str, Any] = Field(default_factory=dict)


# ── Report / Aggregation models ────────────────────────────────────────────────

class AuthDashboardReport(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	generated_at: datetime = Field(default_factory=_utcnow)
	total_users: int = 0
	active_users: int = 0
	locked_users: int = 0
	active_sessions: int = 0
	total_roles: int = 0
	total_policies: int = 0
	failed_logins_24h: int = 0
	mfa_adoption_pct: float = 0.0
	avg_risk_score: float = 0.0
	high_risk_sessions: int = 0
	api_keys_active: int = 0
	service_accounts: int = 0
	delegations_active: int = 0
	audit_events_24h: int = 0


class RoleAssignmentReport(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	generated_at: datetime = Field(default_factory=_utcnow)
	role_id: str
	role_name: str
	tier: RoleTier
	assigned_user_count: int = 0
	assignments: list[RoleAssignmentResponse] = Field(default_factory=list)


class LoginAuditReport(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	generated_at: datetime = Field(default_factory=_utcnow)
	period_hours: int = 24
	total_attempts: int = 0
	successful: int = 0
	failed: int = 0
	blocked: int = 0
	unique_ips: int = 0
	high_risk_count: int = 0
	brute_force_events: int = 0
