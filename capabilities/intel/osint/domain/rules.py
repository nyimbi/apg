"""Deterministic domain rules for Open Source Intelligence.

These rules are the single source of truth for all governance decisions
within this capability.  Every function is a pure assertion or calculation —
no I/O, no side effects.

Rule categories:
  - Tenant isolation
  - Write policy
  - Source registration
  - Collection task gating
  - Raw/processed intel ingestion
  - Entity and relationship integrity
  - Dissemination approval
  - Agent privileged-action gating
  - Confidence and credibility calculations
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Core exception
# ---------------------------------------------------------------------------

class RuleViolation(Exception):
	"""Raised when a business rule is violated.

	Attributes:
		rule_name: Machine-readable rule identifier.
		reason: Human-readable explanation.
		required_action: Suggested remediation step.
	"""

	def __init__(self, rule_name: str, reason: str, required_action: str = "") -> None:
		self.rule_name = rule_name
		self.reason = reason
		self.required_action = required_action
		super().__init__(f"Rule '{rule_name}' violated: {reason}")


# ---------------------------------------------------------------------------
# Tenant isolation
# ---------------------------------------------------------------------------

def assert_tenant_context(tenant_id: str) -> None:
	"""All operations require a non-empty tenant_id.

	Args:
		tenant_id: The tenant identifier to validate.

	Raises:
		RuleViolation: If tenant_id is absent or blank.
	"""
	if not str(tenant_id or "").strip():
		raise RuleViolation(
			"tenant_context_required",
			"tenant_id is required for all OSINT operations",
			"attach_tenant_context",
		)


def assert_no_cross_tenant_access(actor_tenant: str, resource_tenant: str) -> None:
	"""Cross-tenant resource access is always denied.

	Args:
		actor_tenant: Tenant ID of the requesting actor.
		resource_tenant: Tenant ID of the target resource.

	Raises:
		RuleViolation: If the tenants do not match.
	"""
	if actor_tenant != resource_tenant:
		raise RuleViolation(
			"cross_tenant_access_denied",
			f"actor tenant '{actor_tenant}' cannot access resources of tenant '{resource_tenant}'",
			"use_own_tenant_resources",
		)


# ---------------------------------------------------------------------------
# Write policy
# ---------------------------------------------------------------------------

def assert_write_policy(policy_attached: bool) -> None:
	"""Write operations require an attached policy.

	Args:
		policy_attached: True if a valid policy is attached.

	Raises:
		RuleViolation: If policy_attached is False.
	"""
	if not policy_attached:
		raise RuleViolation(
			"write_requires_policy",
			"write operations require an attached OSINT policy",
			"attach_osint_policy",
		)


# ---------------------------------------------------------------------------
# Source registration rules
# ---------------------------------------------------------------------------

def assert_source_terms_reviewed(terms_review_reference: str) -> None:
	"""Source registration requires a completed terms-of-service review.

	Args:
		terms_review_reference: Non-empty reference to the completed TOS review.

	Raises:
		RuleViolation: If terms_review_reference is blank.
	"""
	if not str(terms_review_reference or "").strip():
		raise RuleViolation(
			"source_terms_review_required",
			"a terms-of-service review reference is required before registering a source",
			"complete_terms_review",
		)


# ---------------------------------------------------------------------------
# Collection task rules
# ---------------------------------------------------------------------------

def assert_high_risk_source_approval(
	risk_tier: str,
	approval_reference: str | None,
) -> None:
	"""High and critical risk sources require an approval reference before collection.

	Args:
		risk_tier: One of 'low', 'medium', 'high', 'critical'.
		approval_reference: Non-empty approval reference string.

	Raises:
		RuleViolation: If tier is high/critical and approval is absent.
	"""
	if risk_tier in {"high", "critical"}:
		if not str(approval_reference or "").strip():
			raise RuleViolation(
				"high_risk_source_requires_approval",
				f"collection from a '{risk_tier}' risk source requires an explicit approval reference",
				"obtain_collection_approval",
			)


# ---------------------------------------------------------------------------
# Intelligence ingestion rules
# ---------------------------------------------------------------------------

def assert_fingerprint_present(fingerprint: str) -> None:
	"""Raw intelligence must carry a non-empty content fingerprint.

	Args:
		fingerprint: Content hash / fingerprint string.

	Raises:
		RuleViolation: If fingerprint is blank.
	"""
	if not str(fingerprint or "").strip():
		raise RuleViolation(
			"fingerprint_required",
			"a content fingerprint is required for deduplication",
			"compute_content_fingerprint",
		)


def assert_not_duplicate(fingerprint: str, known_fingerprints: set[str]) -> None:
	"""Reject raw intel if an identical fingerprint has already been ingested.

	Args:
		fingerprint: Content fingerprint of the new item.
		known_fingerprints: Set of fingerprints already stored for this tenant.

	Raises:
		RuleViolation: If fingerprint is already present.
	"""
	if fingerprint in known_fingerprints:
		raise RuleViolation(
			"duplicate_fingerprint_rejected",
			f"duplicate content detected — fingerprint '{fingerprint}' already ingested",
			"deduplicate_before_ingestion",
		)


def assert_confidence_bounds(score: float) -> None:
	"""Confidence score must be in [0.0, 1.0].

	Args:
		score: Confidence score to validate.

	Raises:
		RuleViolation: If score is outside [0.0, 1.0].
	"""
	try:
		f = float(score)
	except (TypeError, ValueError):
		raise RuleViolation(
			"confidence_score_invalid",
			f"confidence score must be a float in [0.0, 1.0], got '{score}'",
			"set_confidence_0_to_1",
		)
	if not (0.0 <= f <= 1.0):
		raise RuleViolation(
			"confidence_score_out_of_range",
			f"confidence score {f} is outside the valid range [0.0, 1.0]",
			"set_confidence_0_to_1",
		)


# ---------------------------------------------------------------------------
# Entity integrity rules
# ---------------------------------------------------------------------------

def assert_entity_name_present(name: str) -> None:
	"""Extracted entities must have a non-blank name.

	Args:
		name: Entity name string.

	Raises:
		RuleViolation: If name is blank or whitespace-only.
	"""
	if not str(name or "").strip():
		raise RuleViolation(
			"entity_name_required",
			"entity name must be a non-empty string",
			"provide_entity_name",
		)


def assert_relationship_entities_distinct(
	source_entity_id: str,
	target_entity_id: str,
) -> None:
	"""A relationship cannot have the same entity on both ends (self-loop).

	Args:
		source_entity_id: ID of the source entity.
		target_entity_id: ID of the target entity.

	Raises:
		RuleViolation: If both IDs are equal.
	"""
	if source_entity_id == target_entity_id:
		raise RuleViolation(
			"self_loop_relationship_denied",
			"source and target entity must be distinct — self-referential relationships are not permitted",
			"select_distinct_entities",
		)


# ---------------------------------------------------------------------------
# Dissemination rules
# ---------------------------------------------------------------------------

def assert_dissemination_approval(approval_reference: str) -> None:
	"""Intelligence dissemination requires an explicit human approval reference.

	Args:
		approval_reference: Non-empty approval reference string.

	Raises:
		RuleViolation: If approval_reference is blank.
	"""
	if not str(approval_reference or "").strip():
		raise RuleViolation(
			"dissemination_requires_approval",
			"autonomous dissemination is denied — an explicit approval reference is required",
			"obtain_dissemination_approval",
		)


# ---------------------------------------------------------------------------
# Agent governance rules
# ---------------------------------------------------------------------------

def assert_human_approval_for_privileged(
	privileged_scope: bool,
	human_approval_recorded: bool,
) -> None:
	"""Privileged agent actions require recorded human approval.

	Args:
		privileged_scope: True if the requested action has elevated privileges.
		human_approval_recorded: True if a human has explicitly approved the action.

	Raises:
		RuleViolation: If action is privileged but no approval is recorded.
	"""
	if privileged_scope and not human_approval_recorded:
		raise RuleViolation(
			"privileged_action_requires_human_approval",
			"privileged OSINT agent actions require explicit human approval before execution",
			"record_human_approval",
		)


# ---------------------------------------------------------------------------
# Credibility calculations
# ---------------------------------------------------------------------------

def calculate_intel_credibility(
	source_credibility: float,
	corroboration_count: int,
	analyst_confidence: float,
	timeliness_score: float,
) -> float:
	"""Weighted composite credibility score for a processed intelligence item.

	Weights:
		source_credibility   0.40
		corroboration        0.25  (capped at 5 corroborating sources = 1.0)
		analyst_confidence   0.25
		timeliness           0.10

	Args:
		source_credibility: Baseline credibility of the originating source [0, 1].
		corroboration_count: Number of independent sources corroborating the item.
		analyst_confidence: Analyst-assigned confidence score [0, 1].
		timeliness_score: Timeliness/freshness score [0, 1].

	Returns:
		Composite credibility score in [0.0, 1.0], rounded to 4 decimal places.
	"""
	corroboration_factor = min(corroboration_count / 5.0, 1.0)
	composite = (
		min(max(float(source_credibility), 0.0), 1.0) * 0.40
		+ corroboration_factor * 0.25
		+ min(max(float(analyst_confidence), 0.0), 1.0) * 0.25
		+ min(max(float(timeliness_score), 0.0), 1.0) * 0.10
	)
	return round(min(max(composite, 0.0), 1.0), 4)


def calculate_relationship_strength(
	evidence_count: int,
	avg_confidence: float,
	temporal_consistency: float,
) -> float:
	"""Relationship strength score derived from evidence volume, confidence, and
	temporal consistency.

	Weights:
		avg_confidence       0.50
		evidence_count       0.30  (capped at 10 items = 1.0)
		temporal_consistency 0.20

	Args:
		evidence_count: Number of independent evidence items supporting the relationship.
		avg_confidence: Average confidence score across evidence items [0, 1].
		temporal_consistency: Score reflecting how consistently the relationship
			is observed over time [0, 1].

	Returns:
		Strength score in [0.0, 1.0], rounded to 4 decimal places.
	"""
	evidence_factor = min(evidence_count / 10.0, 1.0)
	strength = (
		min(max(float(avg_confidence), 0.0), 1.0) * 0.50
		+ evidence_factor * 0.30
		+ min(max(float(temporal_consistency), 0.0), 1.0) * 0.20
	)
	return round(min(max(strength, 0.0), 1.0), 4)


# ---------------------------------------------------------------------------
# Convenience: collect all assert_* checks for a context dict
# ---------------------------------------------------------------------------

def assert_context(context: dict[str, Any]) -> None:
	"""Run all applicable assertion functions for a context dict.

	Intended for use by the service layer when the capability contract rule
	engine is not available (e.g. during unit tests with minimal setup).

	Args:
		context: Key/value context dict matching the capability contract schema.

	Raises:
		RuleViolation: On the first rule violation found.
	"""
	if not context.get("tenant_context_present", True):
		raise RuleViolation(
			"tenant_context_required",
			"tenant_id is required",
			"attach_tenant_context",
		)
	if context.get("operation_type") == "write" and not context.get("policy_attached", True):
		raise RuleViolation(
			"write_requires_policy",
			"write operations require an attached policy",
			"attach_osint_policy",
		)
