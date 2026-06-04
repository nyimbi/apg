"""
APG Audit Logging — Domain Rules.

Single source of truth for all governance decisions:
  - Assertions (preconditions / invariants)
  - Calculations (checksums, chain-hashes, risk scores, retention dates)
  - Utility predicates (off-hours, external-IP)

All rule violations raise :class:`RuleViolation` with a named rule, human
reason, and a required_action hint — making them machine-readable for the
APG rule engine.

© 2025 Datacraft  www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>
"""
from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timedelta, timezone
from ipaddress import ip_address, ip_network
from typing import Any

# ---------------------------------------------------------------------------
# RuleViolation
# ---------------------------------------------------------------------------

class RuleViolation(Exception):
	"""Raised when a business rule is violated."""

	def __init__(
		self,
		rule_name: str,
		reason: str,
		required_action: str = "",
	) -> None:
		self.rule_name       = rule_name
		self.reason          = reason
		self.required_action = required_action
		super().__init__(f"Rule '{rule_name}' violated: {reason}")


# ---------------------------------------------------------------------------
# Private RFC-1918 + loopback networks
# ---------------------------------------------------------------------------

_PRIVATE_NETS = [
	ip_network("10.0.0.0/8"),
	ip_network("172.16.0.0/12"),
	ip_network("192.168.0.0/16"),
	ip_network("127.0.0.0/8"),
	ip_network("::1/128"),
	ip_network("fc00::/7"),
]

# Off-hours window (local-naive UTC): 20:00 – 07:00
_OFF_HOUR_START = 20
_OFF_HOUR_END   = 7


# ===========================================================================
# ASSERTIONS
# ===========================================================================

def assert_tenant_context(context: dict[str, Any]) -> None:
	"""All operations require a non-empty tenant_id in context."""
	if not context.get("tenant_id"):
		raise RuleViolation(
			"tenant_context_required",
			"tenant_id is required for every operation",
			"attach_tenant_context",
		)


def assert_actor_present(actor_id: str | None) -> None:
	"""
	Every write operation must carry an identified actor.

	Service accounts and background jobs should pass their service identity.
	"""
	if not actor_id or not str(actor_id).strip():
		raise RuleViolation(
			"actor_required",
			"actor_id must be non-empty for all write operations",
			"provide_actor_id",
		)


def assert_no_cross_tenant_access(actor_tenant: str, resource_tenant: str) -> None:
	"""Cross-tenant data access is unconditionally denied."""
	if actor_tenant != resource_tenant:
		raise RuleViolation(
			"cross_tenant_access_denied",
			f"actor tenant '{actor_tenant}' cannot access resource in tenant '{resource_tenant}'",
			"use_own_tenant_resources",
		)


def assert_write_policy(context: dict[str, Any]) -> None:
	"""Write operations require an attached policy."""
	if context.get("operation_type") == "write" and not context.get("policy_attached"):
		raise RuleViolation(
			"write_requires_policy",
			"write operations require an attached policy",
			"attach_policy",
		)


def assert_event_immutable(is_immutable: bool) -> None:
	"""
	Immutable events must never be mutated.

	Pass ``False`` to assert that a record IS mutable (e.g. audit trails).
	Pass ``True`` to assert that the caller is NOT attempting mutation on an
	immutable record — raises if True.
	"""
	if is_immutable:
		raise RuleViolation(
			"event_immutable",
			"audit events are immutable after creation and cannot be modified",
			"create_new_event_instead",
		)


def assert_no_legal_hold_deletion(legal_hold: bool) -> None:
	"""Records under legal hold cannot be soft- or hard-deleted."""
	if legal_hold:
		raise RuleViolation(
			"legal_hold_prevents_deletion",
			"this record is under legal hold and cannot be deleted or archived",
			"release_legal_hold_first",
		)


def assert_checksum_valid(stored: str | None, expected: str) -> None:
	"""
	Verify a stored checksum matches the freshly-derived expected value.

	Uses constant-time comparison to resist timing attacks.
	"""
	if not stored:
		raise RuleViolation(
			"checksum_missing",
			"event has no stored checksum — integrity cannot be verified",
			"re-ingest_event",
		)
	if not hmac.compare_digest(stored, expected):
		raise RuleViolation(
			"checksum_mismatch",
			f"stored checksum '{stored[:16]}…' does not match expected '{expected[:16]}…'",
			"investigate_tamper",
		)


def assert_retention_not_expired(created_at: datetime, retain_days: int) -> None:
	"""
	Raise if a record has exceeded its retention window.

	Used to prevent operations (e.g. re-index) on events that should have been
	archived or purged.
	"""
	expire_at = created_at + timedelta(days=retain_days)
	if datetime.now(timezone.utc) > expire_at:
		raise RuleViolation(
			"retention_period_expired",
			f"record expired on {expire_at.isoformat()} — retention window of {retain_days}d exceeded",
			"run_retention_enforcement",
		)


def assert_dsr_requester_authorised(
	requester_id: str,
	subject_id: str,
	is_admin: bool,
) -> None:
	"""
	A data-subject request may only be submitted by:
	  (a) the data subject themselves, or
	  (b) an authorised admin.
	"""
	if not is_admin and requester_id != subject_id:
		raise RuleViolation(
			"dsr_requester_not_authorised",
			f"requester '{requester_id}' is not authorised to submit a DSR for subject '{subject_id}'",
			"submit_as_subject_or_escalate_to_admin",
		)


def assert_erasure_allowed(subject_id: str, justification: str) -> None:
	"""
	GDPR Art. 17 erasure requires a non-empty justification.

	Core audit-event fields remain exempt under Art. 17(3)(b); only detail
	blobs are pseudonymised.  This rule enforces the documented justification
	requirement.
	"""
	if not subject_id or not subject_id.strip():
		raise RuleViolation(
			"erasure_missing_subject",
			"subject_id must be provided for erasure requests",
			"provide_subject_id",
		)
	if not justification or not justification.strip():
		raise RuleViolation(
			"erasure_missing_justification",
			"a written justification is required for GDPR Art. 17 erasure",
			"provide_erasure_justification",
		)


def assert_evidence_package_not_sealed(status: str) -> None:
	"""
	Sealed evidence packages are legally immutable and cannot be modified.

	``status`` should be the string value of :class:`EvidencePackageStatus`.
	"""
	if status == "sealed":
		raise RuleViolation(
			"evidence_package_sealed",
			"this evidence package has been legally sealed and cannot be modified",
			"create_new_evidence_package",
		)


def assert_batch_size(count: int, max_count: int = 10_000) -> None:
	"""Batch write operations must not exceed max_count events per call."""
	if count == 0:
		raise RuleViolation(
			"batch_empty",
			"batch must contain at least one event",
			"provide_events",
		)
	if count > max_count:
		raise RuleViolation(
			"batch_too_large",
			f"batch of {count} exceeds maximum of {max_count}",
			"split_into_smaller_batches",
		)


def assert_risk_score_range(score: float) -> None:
	"""Risk and anomaly scores must be in [0, 1]."""
	if not (0.0 <= score <= 1.0):
		raise RuleViolation(
			"risk_score_out_of_range",
			f"risk score {score} is outside valid range [0, 1]",
			"normalise_risk_score",
		)


# ===========================================================================
# CALCULATIONS
# ===========================================================================

def calculate_event_checksum(
	*,
	event_id:     str,
	tenant_id:    str,
	timestamp:    datetime,
	event_type:   str,
	actor_id:     str | None,
	action:       str,
	resource_type: str | None,
	resource_id:  str | None,
	success:      bool,
) -> str:
	"""
	Derive a deterministic SHA-256 checksum for an audit event.

	The payload is a canonicalised JSON object (sorted keys) containing only
	the fields that define event identity.  Mutable fields (details, tags,
	anomaly_score) are intentionally excluded so they can be pseudonymised
	without invalidating the integrity proof.
	"""
	payload = {
		"id":            event_id,
		"tenant_id":     tenant_id,
		"timestamp":     timestamp.isoformat(),
		"event_type":    str(event_type),
		"actor_id":      actor_id or "",
		"action":        action,
		"resource_type": resource_type or "",
		"resource_id":   resource_id or "",
		"success":       success,
	}
	canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
	return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def calculate_chain_hash(previous_hash: str, current_checksum: str) -> str:
	"""
	Derive a Merkle-style chain hash linking consecutive audit events.

	``chain_hash[n] = SHA-256(chain_hash[n-1] || checksum[n])``

	The initial tip (before the first event) is 64 zero characters.
	This creates a tamper-evident sequence: any mutation of an earlier event
	causes all subsequent chain hashes to diverge.
	"""
	data = (previous_hash + current_checksum).encode("utf-8")
	return hashlib.sha256(data).hexdigest()


def calculate_risk_score(
	*,
	is_failed_auth:      bool = False,
	is_privileged_actor: bool = False,
	is_off_hours:        bool = False,
	is_external_ip:      bool = False,
	is_sensitive_data:   bool = False,
	is_error_event:      bool = False,
	anomaly_hint:        float = 0.0,
) -> float:
	"""
	Deterministic risk score in [0.0, 1.0].

	Factors and weights (additive, capped at 1.0):
	  failed auth          +0.35
	  privileged actor     +0.20
	  off-hours access     +0.15
	  external IP          +0.15
	  sensitive data       +0.10
	  error event          +0.05
	  anomaly hint         up to +0.20 (scaled linearly)

	These weights encode the MITRE ATT&CK risk model for credential-based
	threats and are tunable via the domain/calculations module.
	"""
	score = 0.0
	if is_failed_auth:      score += 0.35
	if is_privileged_actor: score += 0.20
	if is_off_hours:        score += 0.15
	if is_external_ip:      score += 0.15
	if is_sensitive_data:   score += 0.10
	if is_error_event:      score += 0.05
	# Anomaly contribution: scale [0,1] → [0, 0.20]
	score += max(0.0, min(1.0, anomaly_hint)) * 0.20
	return round(min(1.0, score), 4)


def calculate_retain_until(created_at: datetime, retain_days: int) -> datetime:
	"""
	Return the UTC datetime at which a record may be archived or purged.

	``retain_until = created_at + retain_days``
	"""
	return created_at + timedelta(days=retain_days)


# ===========================================================================
# PREDICATES
# ===========================================================================

def is_off_hours(ts: datetime | None) -> bool:
	"""
	Return True if ``ts`` falls outside standard business hours (UTC).

	Off-hours window: 20:00 – 07:00 UTC.  Handles None gracefully (False).
	"""
	if ts is None:
		return False
	hour = ts.astimezone(timezone.utc).hour
	return hour >= _OFF_HOUR_START or hour < _OFF_HOUR_END


def is_external_ip(ip: str | None) -> bool:
	"""
	Return True if ``ip`` is NOT in a private / loopback range.

	Unparseable or None values return False (benefit of the doubt).
	"""
	if not ip:
		return False
	try:
		addr = ip_address(ip)
	except ValueError:
		return False
	return not any(addr in net for net in _PRIVATE_NETS)


def is_high_risk(risk_score: float, threshold: float = 0.7) -> bool:
	"""Return True if risk_score >= threshold."""
	return risk_score >= threshold


def is_compliance_sensitive(compliance_tags: list[str], framework: str) -> bool:
	"""Return True if the given framework tag is present in event compliance_tags."""
	return framework.upper() in (t.upper() for t in compliance_tags)


__all__ = [
	# Exception
	"RuleViolation",
	# Assertions
	"assert_tenant_context",
	"assert_actor_present",
	"assert_no_cross_tenant_access",
	"assert_write_policy",
	"assert_event_immutable",
	"assert_no_legal_hold_deletion",
	"assert_checksum_valid",
	"assert_retention_not_expired",
	"assert_dsr_requester_authorised",
	"assert_erasure_allowed",
	"assert_evidence_package_not_sealed",
	"assert_batch_size",
	"assert_risk_score_range",
	# Calculations
	"calculate_event_checksum",
	"calculate_chain_hash",
	"calculate_risk_score",
	"calculate_retain_until",
	# Predicates
	"is_off_hours",
	"is_external_ip",
	"is_high_risk",
	"is_compliance_sensitive",
]
