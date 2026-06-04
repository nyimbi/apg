"""
Business rules for Authentication & RBAC.

Every rule is a pure callable — no I/O, no side-effects.
RuleViolation is raised on any violation; callers catch it or let it propagate.

© 2025 Datacraft - www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from __future__ import annotations

import hashlib
import ipaddress
import re
import string
from datetime import datetime, timezone
from typing import Any


# ── Exception ──────────────────────────────────────────────────────────────────

class RuleViolation(Exception):
	"""Raised when a domain business rule is violated."""

	def __init__(
		self,
		rule_name: str,
		reason: str,
		required_action: str = "",
		http_status: int = 422,
	) -> None:
		self.rule_name = rule_name
		self.reason = reason
		self.required_action = required_action
		self.http_status = http_status
		super().__init__(f"[{rule_name}] {reason}")

	def to_dict(self) -> dict[str, Any]:
		return {
			"rule": self.rule_name,
			"reason": self.reason,
			"required_action": self.required_action,
		}


# ── Tenant ─────────────────────────────────────────────────────────────────────

def assert_tenant_context(tenant_id: str | None) -> None:
	"""Every operation must carry a non-empty tenant_id."""
	if not tenant_id or not tenant_id.strip():
		raise RuleViolation(
			"tenant_context_required",
			"tenant_id is required for all AUTH operations",
			"attach_tenant_context",
			http_status=400,
		)


def assert_no_cross_tenant_access(actor_tenant: str, resource_tenant: str) -> None:
	"""Actors may only access resources in their own tenant."""
	if actor_tenant != resource_tenant:
		raise RuleViolation(
			"cross_tenant_access_denied",
			f"actor tenant '{actor_tenant}' cannot access resources in tenant '{resource_tenant}'",
			"use_own_tenant_resources",
			http_status=403,
		)


# ── Identity ───────────────────────────────────────────────────────────────────

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def assert_valid_email(email: str) -> None:
	if not _EMAIL_RE.match(email):
		raise RuleViolation("invalid_email", f"'{email}' is not a valid email address", "provide_valid_email")


def assert_user_not_locked(status: str) -> None:
	if status in {"locked", "suspended"}:
		raise RuleViolation(
			"user_account_locked",
			"account is locked or suspended — authentication denied",
			"contact_administrator",
			http_status=403,
		)


def assert_user_active(status: str) -> None:
	if status not in {"active", "password_reset_required"}:
		raise RuleViolation(
			"user_not_active",
			f"account status '{status}' does not permit this operation",
			"activate_account",
			http_status=403,
		)


# ── Passwords ──────────────────────────────────────────────────────────────────

def assert_password_strength(
	password: str,
	min_length: int = 12,
	require_uppercase: bool = True,
	require_lowercase: bool = True,
	require_digits: bool = True,
	require_special: bool = True,
) -> None:
	"""Validate password against policy constraints."""
	violations: list[str] = []
	if len(password) < min_length:
		violations.append(f"must be at least {min_length} characters")
	if require_uppercase and not any(c.isupper() for c in password):
		violations.append("must contain at least one uppercase letter")
	if require_lowercase and not any(c.islower() for c in password):
		violations.append("must contain at least one lowercase letter")
	if require_digits and not any(c.isdigit() for c in password):
		violations.append("must contain at least one digit")
	special = set(string.punctuation)
	if require_special and not any(c in special for c in password):
		violations.append("must contain at least one special character")
	if violations:
		raise RuleViolation(
			"password_policy_violation",
			"; ".join(violations),
			"choose_stronger_password",
		)


def assert_password_not_in_history(new_hash: str, history_hashes: list[str]) -> None:
	"""Prevent password reuse against stored history."""
	if new_hash in history_hashes:
		raise RuleViolation(
			"password_reuse_denied",
			"new password was recently used — choose a different password",
			"choose_new_password",
		)


def assert_password_not_breached(breach_count: int) -> None:
	"""Reject passwords that appear in breach databases."""
	if breach_count > 0:
		raise RuleViolation(
			"password_breached",
			f"this password appears {breach_count} times in known breach databases",
			"choose_unbreached_password",
		)


def calculate_password_strength_score(password: str) -> float:
	"""
	Return a 0.0–1.0 heuristic strength score.

	Considers length, character class diversity, and entropy estimate.
	Not a substitute for policy enforcement; use alongside assert_password_strength.
	"""
	score = 0.0
	length = len(password)
	# Length contribution (up to 0.4)
	score += min(length / 30.0, 1.0) * 0.4
	# Character class diversity (up to 0.4)
	classes = sum([
		any(c.isupper() for c in password),
		any(c.islower() for c in password),
		any(c.isdigit() for c in password),
		any(c in string.punctuation for c in password),
	])
	score += (classes / 4.0) * 0.4
	# Unique character ratio (up to 0.2)
	unique_ratio = len(set(password)) / max(length, 1)
	score += unique_ratio * 0.2
	return round(min(score, 1.0), 4)


# ── MFA ────────────────────────────────────────────────────────────────────────

_SUPPORTED_MFA_METHODS = {"totp", "sms", "email", "hardware_key", "passkey", "backup_code"}


def assert_mfa_method_supported(method: str) -> None:
	if method not in _SUPPORTED_MFA_METHODS:
		raise RuleViolation(
			"unsupported_mfa_method",
			f"'{method}' is not a supported MFA method; choose from {sorted(_SUPPORTED_MFA_METHODS)}",
			"choose_supported_mfa_method",
		)


def assert_mfa_required_for_privileged(
	role_tier: str,
	mfa_verified: bool,
) -> None:
	"""Privileged / admin operations require verified MFA."""
	if role_tier in {"privileged", "admin", "super_admin"} and not mfa_verified:
		raise RuleViolation(
			"mfa_required_for_privileged_role",
			f"role tier '{role_tier}' requires MFA verification before assignment",
			"complete_mfa_verification",
			http_status=403,
		)


def assert_totp_window(
	provided_code: str,
	expected_code: str,
	window_codes: list[str] | None = None,
) -> None:
	"""Verify TOTP code within ±1 time-step window."""
	valid = {expected_code} | set(window_codes or [])
	if provided_code not in valid:
		raise RuleViolation(
			"totp_code_invalid",
			"TOTP code does not match — check your authenticator app time sync",
			"retry_totp",
			http_status=401,
		)


# ── Sessions ───────────────────────────────────────────────────────────────────

def assert_session_active(status: str) -> None:
	if status != "active":
		raise RuleViolation(
			"session_not_active",
			f"session status is '{status}' — obtain a new session",
			"reauthenticate",
			http_status=401,
		)


def assert_session_not_expired(expires_at: datetime | None) -> None:
	if expires_at is not None and datetime.now(timezone.utc) > expires_at:
		raise RuleViolation(
			"session_expired",
			"session has expired",
			"reauthenticate",
			http_status=401,
		)


def assert_concurrent_session_limit(active_count: int, max_sessions: int) -> None:
	if active_count >= max_sessions:
		raise RuleViolation(
			"concurrent_session_limit_exceeded",
			f"user already has {active_count} active sessions (limit={max_sessions}); oldest will be revoked",
			"revoke_old_sessions",
		)


def assert_no_session_fixation(
	pre_auth_session_id: str | None,
	post_auth_session_id: str,
) -> None:
	"""Ensure session ID is regenerated after successful authentication."""
	if pre_auth_session_id and pre_auth_session_id == post_auth_session_id:
		raise RuleViolation(
			"session_fixation_detected",
			"session ID must be rotated after authentication",
			"rotate_session_id",
			http_status=500,
		)


def assert_step_up_for_sensitive_operation(
	operation: str,
	step_up_completed: bool,
	sensitive_operations: set[str] | None = None,
) -> None:
	"""Sensitive operations require an in-session step-up authentication."""
	sensitive = sensitive_operations or {
		"delete_user", "assign_admin_role", "view_credentials", "export_data",
		"change_password", "revoke_all_sessions", "impersonate",
	}
	if operation in sensitive and not step_up_completed:
		raise RuleViolation(
			"step_up_required",
			f"operation '{operation}' requires step-up authentication",
			"complete_step_up_auth",
			http_status=403,
		)


# ── Brute-force & Rate Limiting ────────────────────────────────────────────────

def assert_not_brute_force(
	failed_attempts: int,
	max_attempts: int,
	lockout_until: datetime | None,
) -> None:
	"""Block authentication when brute-force thresholds are exceeded."""
	if lockout_until and datetime.now(timezone.utc) < lockout_until:
		remaining = int((lockout_until - datetime.now(timezone.utc)).total_seconds())
		raise RuleViolation(
			"account_locked_brute_force",
			f"account is temporarily locked; retry in {remaining}s",
			"wait_or_contact_admin",
			http_status=429,
		)
	if failed_attempts >= max_attempts:
		raise RuleViolation(
			"max_failed_attempts_exceeded",
			f"failed attempts ({failed_attempts}) reached limit ({max_attempts}) — account will be locked",
			"contact_administrator",
			http_status=429,
		)


def calculate_brute_force_lockout_seconds(failed_attempts: int, base_seconds: int = 30) -> int:
	"""Exponential back-off: base * 2^(attempts-1), capped at 24 hours."""
	if failed_attempts <= 0:
		return 0
	return min(base_seconds * (2 ** (failed_attempts - 1)), 86_400)


# ── Risk Scoring ───────────────────────────────────────────────────────────────

def calculate_login_risk_score(
	new_device: bool = False,
	off_hours: bool = False,
	impossible_travel: bool = False,
	tor_exit_node: bool = False,
	known_bad_ip: bool = False,
	failed_attempts_recently: int = 0,
) -> float:
	"""
	Return a 0.0–1.0 composite risk score for a login event.

	Individual factor weights are tuned for a security-forward posture.
	"""
	score = 0.0
	if impossible_travel:    score += 0.50
	if known_bad_ip:         score += 0.40
	if tor_exit_node:        score += 0.30
	if new_device:           score += 0.20
	if off_hours:            score += 0.10
	if failed_attempts_recently > 0:
		score += min(failed_attempts_recently * 0.05, 0.25)
	return round(min(score, 1.0), 4)


def assert_risk_within_threshold(risk_score: float, threshold: float = 0.75) -> None:
	if risk_score >= threshold:
		raise RuleViolation(
			"login_risk_too_high",
			f"risk score {risk_score:.2f} exceeds threshold {threshold:.2f}",
			"additional_verification_required",
			http_status=403,
		)


def assert_suspicious_login_not_detected(
	risk_score: float,
	step_up_completed: bool,
	step_up_threshold: float = 0.40,
) -> None:
	"""Require step-up when risk is elevated but not high enough to block."""
	if risk_score >= step_up_threshold and not step_up_completed:
		raise RuleViolation(
			"suspicious_login_step_up_required",
			f"login risk {risk_score:.2f} requires step-up authentication",
			"complete_step_up_auth",
			http_status=403,
		)


# ── IP Allowlist ───────────────────────────────────────────────────────────────

def calculate_ip_in_allowlist(ip: str, cidrs: list[str]) -> bool:
	"""Return True if ip falls within any CIDR in cidrs."""
	try:
		addr = ipaddress.ip_address(ip)
		return any(addr in ipaddress.ip_network(cidr, strict=False) for cidr in cidrs)
	except ValueError:
		return False


def assert_ip_allowed(ip: str, cidrs: list[str]) -> None:
	"""Block requests from IPs not in the allowlist (when allowlist is non-empty)."""
	if cidrs and not calculate_ip_in_allowlist(ip, cidrs):
		raise RuleViolation(
			"ip_not_in_allowlist",
			f"IP address {ip} is not in the configured allowlist",
			"use_allowed_ip",
			http_status=403,
		)


# ── RBAC ───────────────────────────────────────────────────────────────────────

def assert_role_not_already_assigned(
	user_id: str,
	role_id: str,
	existing_assignments: list[dict[str, Any]],
) -> None:
	for a in existing_assignments:
		if a.get("user_id") == user_id and a.get("role_id") == role_id and a.get("is_active"):
			raise RuleViolation(
				"role_already_assigned",
				f"role '{role_id}' is already active for user '{user_id}'",
				"no_action_required",
			)


def assert_role_active(status: str) -> None:
	if status != "active":
		raise RuleViolation(
			"role_not_active",
			f"cannot assign inactive role (status='{status}')",
			"activate_role_first",
		)


def assert_admin_role_requires_approval(
	role_tier: str,
	approval_provided: bool,
) -> None:
	"""Admin and super_admin role assignments require explicit approval evidence."""
	if role_tier in {"admin", "super_admin"} and not approval_provided:
		raise RuleViolation(
			"admin_role_requires_approval",
			f"role tier '{role_tier}' requires a prior approval record",
			"obtain_role_assignment_approval",
			http_status=403,
		)


def assert_no_privilege_escalation(
	actor_tier: str,
	target_tier: str,
	tier_order: dict[str, int] | None = None,
) -> None:
	"""An actor may not grant a role with a higher tier than their own."""
	order = tier_order or {
		"standard": 1, "elevated": 2, "privileged": 3, "admin": 4, "super_admin": 5,
	}
	if order.get(target_tier, 0) > order.get(actor_tier, 0):
		raise RuleViolation(
			"privilege_escalation_denied",
			f"actor with tier '{actor_tier}' cannot assign tier '{target_tier}'",
			"request_higher_privilege",
			http_status=403,
		)


def assert_reviewer_not_requester(reviewer_id: str, requester_id: str) -> None:
	"""Four-eyes: approver must be a different person from the requester."""
	if reviewer_id == requester_id:
		raise RuleViolation(
			"reviewer_same_as_requester",
			"approval reviewer cannot be the same person as the requester",
			"assign_different_reviewer",
			http_status=400,
		)


# ── OAuth2 / PKCE ──────────────────────────────────────────────────────────────

def assert_pkce_required_for_public_client(is_public: bool, code_challenge: str | None) -> None:
	if is_public and not code_challenge:
		raise RuleViolation(
			"pkce_required",
			"public OAuth2 clients must use PKCE (code_challenge is required)",
			"add_pkce_challenge",
			http_status=400,
		)


def calculate_pkce_challenge(code_verifier: str) -> str:
	"""Return BASE64URL(SHA-256(ASCII(code_verifier))) per RFC 7636."""
	import base64
	digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
	return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def assert_pkce_verifier_matches(code_verifier: str, stored_challenge: str) -> None:
	derived = calculate_pkce_challenge(code_verifier)
	if derived != stored_challenge:
		raise RuleViolation(
			"pkce_verification_failed",
			"code_verifier does not match the stored code_challenge",
			"provide_correct_code_verifier",
			http_status=401,
		)


def assert_implicit_flow_disabled(response_type: str) -> None:
	"""OAuth2 implicit flow (response_type=token) is deprecated per RFC 9700."""
	if response_type == "token":
		raise RuleViolation(
			"implicit_flow_deprecated",
			"OAuth2 implicit flow is disabled; use authorization_code + PKCE",
			"switch_to_authorization_code_pkce",
			http_status=400,
		)


def assert_redirect_uri_registered(redirect_uri: str, registered_uris: list[str]) -> None:
	if redirect_uri not in registered_uris:
		raise RuleViolation(
			"redirect_uri_not_registered",
			f"redirect_uri '{redirect_uri}' is not registered for this client",
			"register_redirect_uri",
			http_status=400,
		)


def assert_auth_code_not_used(used: bool) -> None:
	if used:
		raise RuleViolation(
			"auth_code_already_used",
			"authorization code has already been exchanged — replay attack possible",
			"restart_oauth2_flow",
			http_status=400,
		)


def assert_auth_code_not_expired(issued_at_epoch: float, ttl_seconds: int = 300) -> None:
	import time
	age = time.time() - issued_at_epoch
	if age > ttl_seconds:
		raise RuleViolation(
			"auth_code_expired",
			f"authorization code expired after {ttl_seconds}s (age={int(age)}s)",
			"restart_oauth2_flow",
			http_status=400,
		)


# ── JWT ────────────────────────────────────────────────────────────────────────

def assert_token_not_blacklisted(token_jti: str, blacklist: set[str]) -> None:
	if token_jti in blacklist:
		raise RuleViolation(
			"token_blacklisted",
			"token has been explicitly revoked",
			"reauthenticate",
			http_status=401,
		)


def assert_token_not_expired(exp: int) -> None:
	import time
	if time.time() > exp:
		raise RuleViolation(
			"token_expired",
			"JWT has expired",
			"refresh_token_or_reauthenticate",
			http_status=401,
		)


def assert_token_tenant_matches(token_tenant: str, expected_tenant: str) -> None:
	if token_tenant != expected_tenant:
		raise RuleViolation(
			"token_tenant_mismatch",
			f"token is for tenant '{token_tenant}' but request is for '{expected_tenant}'",
			"use_correct_tenant_token",
			http_status=401,
		)


# ── API Keys ───────────────────────────────────────────────────────────────────

def calculate_api_key_hash(raw_key: str, salt: str, iterations: int = 100_000) -> str:
	"""PBKDF2-HMAC-SHA256. Returns hex digest."""
	dk = hashlib.pbkdf2_hmac("sha256", raw_key.encode(), salt.encode(), iterations)
	return dk.hex()


def assert_api_key_active(status: str) -> None:
	if status not in {"active"}:
		raise RuleViolation(
			"api_key_not_active",
			f"API key status is '{status}'",
			"rotate_or_create_api_key",
			http_status=401,
		)


def assert_api_key_not_expired(expires_at: datetime | None) -> None:
	if expires_at and datetime.now(timezone.utc) > expires_at:
		raise RuleViolation(
			"api_key_expired",
			"API key has expired",
			"rotate_api_key",
			http_status=401,
		)


def assert_api_key_scope(required_scope: str, key_scopes: list[str]) -> None:
	if key_scopes and required_scope not in key_scopes:
		raise RuleViolation(
			"api_key_scope_insufficient",
			f"API key lacks required scope '{required_scope}'",
			"rotate_key_with_correct_scopes",
			http_status=403,
		)


# ── Delegation ─────────────────────────────────────────────────────────────────

def assert_delegation_not_expired(expires_at: datetime) -> None:
	if datetime.now(timezone.utc) > expires_at:
		raise RuleViolation(
			"delegation_expired",
			"delegation grant has expired",
			"renew_delegation",
			http_status=403,
		)


def assert_delegation_permission_subset(
	delegated_permissions: list[str],
	delegator_permissions: list[str],
) -> None:
	"""A delegator can only grant permissions they themselves hold."""
	excess = set(delegated_permissions) - set(delegator_permissions)
	if excess:
		raise RuleViolation(
			"delegation_exceeds_delegator_permissions",
			f"cannot delegate permissions not held by delegator: {sorted(excess)}",
			"reduce_delegated_permissions",
			http_status=403,
		)


# ── Zero-Trust / Continuous Verification ──────────────────────────────────────

def assert_continuous_verification_passed(
	trust_score: float,
	minimum_trust: float = 0.5,
) -> None:
	"""Zero-trust: re-evaluate trust score on each sensitive request."""
	if trust_score < minimum_trust:
		raise RuleViolation(
			"continuous_verification_failed",
			f"current trust score {trust_score:.2f} is below minimum {minimum_trust:.2f}",
			"reauthenticate_or_step_up",
			http_status=403,
		)


def calculate_session_trust_score(
	behavioral_score: float,
	risk_level: str,
	mfa_verified: bool,
	step_up_completed: bool,
) -> float:
	"""Composite trust score for a live session; range 0.0–1.0."""
	penalty = {"low": 0.0, "medium": 0.15, "high": 0.35, "critical": 0.60}.get(risk_level, 0.20)
	bonus = (0.05 if mfa_verified else 0.0) + (0.05 if step_up_completed else 0.0)
	return round(max(min(behavioral_score - penalty + bonus, 1.0), 0.0), 3)


# ── Service Accounts ───────────────────────────────────────────────────────────

def assert_service_account_rotation_due(
	last_rotated_at: datetime | None,
	rotation_days: int,
) -> bool:
	"""Return True if key rotation is overdue; does NOT raise."""
	if last_rotated_at is None:
		return True
	age = (datetime.now(timezone.utc) - last_rotated_at).days
	return age >= rotation_days


def assert_service_account_active(status: str) -> None:
	if status not in {"active"}:
		raise RuleViolation(
			"service_account_not_active",
			f"service account status is '{status}'",
			"activate_service_account",
			http_status=403,
		)


# ── Token Refresh Race Condition ───────────────────────────────────────────────

def assert_refresh_token_not_rotated(
	token_jti: str,
	rotated_tokens: set[str],
) -> None:
	"""
	Detect refresh token reuse after rotation (RFC 6749 §10.4).

	If a rotated token is presented again, the entire family should be revoked.
	"""
	if token_jti in rotated_tokens:
		raise RuleViolation(
			"refresh_token_reuse_detected",
			"refresh token has already been rotated — possible token theft",
			"revoke_all_sessions_and_reauthenticate",
			http_status=401,
		)
