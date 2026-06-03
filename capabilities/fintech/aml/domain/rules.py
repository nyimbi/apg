"""Deterministic domain rules for Anti-Money Laundering.

All governance decisions in this capability flow through this module.
Rules are callable functions returning void on pass, raising RuleViolation on fail.
"""
from __future__ import annotations

from typing import Any


class RuleViolation(Exception):
	"""Raised when a business rule is violated."""

	def __init__(self, rule_name: str, reason: str, required_action: str = "") -> None:
		self.rule_name = rule_name
		self.reason = reason
		self.required_action = required_action
		super().__init__(f"Rule '{rule_name}' violated: {reason}")


# ---------------------------------------------------------------------------
# Tenant / policy guards
# ---------------------------------------------------------------------------

def assert_tenant_context(context: dict[str, Any]) -> None:
	"""All operations require a tenant context."""
	if not context.get("tenant_id"):
		raise RuleViolation(
			"tenant_context_required",
			"tenant_id is required",
			"attach_tenant_context",
		)


def assert_write_policy(context: dict[str, Any]) -> None:
	"""Write operations require an attached policy."""
	if context.get("operation_type") == "write" and not context.get("policy_attached"):
		raise RuleViolation(
			"write_requires_policy",
			"write operations require an attached policy",
			"attach_policy",
		)


def assert_no_cross_tenant_access(actor_tenant: str, resource_tenant: str) -> None:
	"""Cross-tenant access is always denied."""
	if actor_tenant != resource_tenant:
		raise RuleViolation(
			"cross_tenant_access_denied",
			"cross-tenant access is not permitted",
			"use_own_tenant_resources",
		)


# ---------------------------------------------------------------------------
# Transaction monitoring rules
# ---------------------------------------------------------------------------

def assert_transaction_subject_present(subject_reference: str) -> None:
	"""Monitored transactions require a subject reference."""
	if not subject_reference or not subject_reference.strip():
		raise RuleViolation(
			"transaction_subject_required",
			"monitored transactions require a subject reference",
			"attach_subject_reference",
		)


def assert_positive_amount(amount: float) -> None:
	"""Transaction amount must be positive."""
	if amount <= 0:
		raise RuleViolation(
			"positive_amount_required",
			f"transaction amount must be positive, got {amount}",
			"set_positive_amount",
		)


def assert_currency_present(currency: str) -> None:
	"""Transaction must have a currency code."""
	if not currency or len(currency.strip()) < 3:
		raise RuleViolation(
			"currency_required",
			"transaction currency is required (ISO 4217)",
			"set_currency_code",
		)


def assert_kyc_link_present(kyc_profile_id: str) -> None:
	"""AML monitoring requires a linked KYC profile."""
	if not kyc_profile_id or not kyc_profile_id.strip():
		raise RuleViolation(
			"kyc_link_required",
			"AML monitoring requires a linked KYC profile",
			"attach_kyc_profile",
		)


def assert_source_reference_present(source_reference: str, source_capability: str) -> None:
	"""Monitored transaction requires a source capability and reference."""
	if not source_reference or not source_capability:
		raise RuleViolation(
			"source_reference_required",
			"source capability and reference are required for AML monitoring",
			"attach_source_reference",
		)


# ---------------------------------------------------------------------------
# Alert rules
# ---------------------------------------------------------------------------

SUPPORTED_ALERT_TYPES = {
	"large_transaction", "structuring", "velocity", "round_trip", "layering",
	"sanctions", "pep", "high_risk_kyc", "mule_account", "trade_based",
	"crypto_asset", "nft", "correspondent", "terrorist_financing", "agent_review",
}

SUPPORTED_SEVERITIES = {"low", "medium", "high", "critical"}


def assert_alert_type_supported(alert_type: str) -> None:
	if alert_type not in SUPPORTED_ALERT_TYPES:
		raise RuleViolation(
			"unsupported_alert_type",
			f"alert type '{alert_type}' is not supported",
			f"use one of: {sorted(SUPPORTED_ALERT_TYPES)}",
		)


def assert_severity_supported(severity: str) -> None:
	if severity not in SUPPORTED_SEVERITIES:
		raise RuleViolation(
			"unsupported_severity",
			f"severity '{severity}' is not supported",
			f"use one of: {sorted(SUPPORTED_SEVERITIES)}",
		)


def assert_alert_evidence_present(evidence_references: list[str]) -> None:
	if not evidence_references:
		raise RuleViolation(
			"alert_evidence_required",
			"AML alerts require at least one evidence reference",
			"attach_alert_evidence",
		)


def assert_alert_close_has_disposition(closing: bool, disposition: str) -> None:
	if closing and not disposition:
		raise RuleViolation(
			"alert_disposition_required",
			"closing an AML alert requires a disposition",
			"record_alert_disposition",
		)


def assert_alert_escalation_has_reviewer(escalating: bool, reviewer_id: str) -> None:
	if escalating and not reviewer_id:
		raise RuleViolation(
			"alert_reviewer_required",
			"escalating an AML alert requires a reviewer assignment",
			"assign_aml_reviewer",
		)


# ---------------------------------------------------------------------------
# Case rules
# ---------------------------------------------------------------------------

SUPPORTED_CASE_TYPES = {
	"transaction_monitoring", "sanctions_alert", "structuring_alert",
	"mule_account", "high_risk_customer", "terrorist_financing",
	"trade_based_ml", "crypto_asset", "network_analysis",
	"suspicious_activity_report",
}


def assert_case_type_supported(case_type: str) -> None:
	if case_type not in SUPPORTED_CASE_TYPES:
		raise RuleViolation(
			"unsupported_case_type",
			f"case type '{case_type}' is not supported",
			f"use one of: {sorted(SUPPORTED_CASE_TYPES)}",
		)


def assert_investigator_assigned(investigator_id: str) -> None:
	if not investigator_id or not investigator_id.strip():
		raise RuleViolation(
			"investigator_required",
			"AML cases require an assigned investigator",
			"assign_investigator",
		)


def assert_case_is_open_for_investigation(status: str) -> None:
	closed_statuses = {
		"closed_no_action", "closed_action_taken", "referred_to_lea",
	}
	if status in closed_statuses:
		raise RuleViolation(
			"case_already_closed",
			f"case is in terminal status '{status}' and cannot be modified",
			"open_new_case_if_required",
		)


# ---------------------------------------------------------------------------
# SAR rules
# ---------------------------------------------------------------------------

def assert_sar_narrative_present(narrative: str) -> None:
	if not narrative or len(narrative.strip()) < 50:
		raise RuleViolation(
			"sar_narrative_required",
			"SAR narrative must be present and at least 50 characters",
			"write_sar_narrative",
		)


def assert_sar_human_approval(approved_by: str) -> None:
	if not approved_by or not approved_by.strip():
		raise RuleViolation(
			"sar_human_approval_required",
			"SARs require human approval before filing",
			"obtain_sar_approval",
		)


def assert_sar_jurisdiction_present(jurisdiction: str) -> None:
	if not jurisdiction or not jurisdiction.strip():
		raise RuleViolation(
			"sar_jurisdiction_required",
			"SAR requires a jurisdiction",
			"set_sar_jurisdiction",
		)


def assert_sar_filing_reference_set(status: str, filing_reference: str) -> None:
	if status == "filed" and not filing_reference:
		raise RuleViolation(
			"sar_filing_reference_required",
			"filed SARs must have a filing reference from the regulator",
			"set_filing_reference",
		)


# ---------------------------------------------------------------------------
# CTR rules
# ---------------------------------------------------------------------------

def assert_ctr_amount_triggers_reporting(amount: float, threshold: float) -> None:
	if amount < threshold:
		raise RuleViolation(
			"ctr_threshold_not_met",
			f"CTR requires amount >= {threshold}, got {amount}",
			"verify_ctr_eligibility",
		)


# ---------------------------------------------------------------------------
# Watchlist rules
# ---------------------------------------------------------------------------

def assert_match_score_valid(match_score: float) -> None:
	if not 0.0 <= match_score <= 1.0:
		raise RuleViolation(
			"invalid_match_score",
			f"match_score must be in [0.0, 1.0], got {match_score}",
			"set_valid_match_score",
		)


def assert_high_confidence_match_reviewed(match_score: float, status: str) -> None:
	if match_score >= 0.9 and status == "pending":
		raise RuleViolation(
			"high_confidence_match_requires_review",
			f"match_score {match_score} >= 0.9 requires immediate review",
			"review_watchlist_match",
		)


# ---------------------------------------------------------------------------
# Rule evaluation engine
# ---------------------------------------------------------------------------

RULE_REGISTRY: list[dict[str, Any]] = [
	{
		"name": "tenant_context_required",
		"fn": lambda ctx: assert_tenant_context(ctx),
		"applies_to": "*",
	},
	{
		"name": "write_requires_policy",
		"fn": lambda ctx: assert_write_policy(ctx),
		"applies_to": "write",
	},
]


def evaluate_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Evaluate applicable rules against context dict.

	Returns dict with decision ('allow'/'deny'), and actions list.
	"""
	actions: list[dict[str, Any]] = []
	for rule in RULE_REGISTRY:
		applies = rule["applies_to"] == "*" or context.get("operation_type") == rule["applies_to"]
		if not applies:
			continue
		try:
			rule["fn"](context)
		except RuleViolation as exc:
			actions.append({
				"rule": exc.rule_name,
				"reason": exc.reason,
				"required_action": exc.required_action,
			})

	if actions:
		return {"decision": "deny", "actions": actions}
	return {"decision": "allow", "actions": []}
