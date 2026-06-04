"""Deterministic domain rules for Anti-Money Laundering.

Every business constraint is encoded here as a callable function.
service.py imports and calls these directly — no rule logic lives elsewhere.

Rule naming conventions:
  assert_*   — raises RuleViolation on violation, returns None on pass
  calculate_* — pure function, no side effects
"""
from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Violation exception
# ---------------------------------------------------------------------------

class RuleViolation(Exception):
	"""Raised when a business rule is violated."""

	def __init__(self, rule_name: str, reason: str, required_action: str = "") -> None:
		self.rule_name = rule_name
		self.reason = reason
		self.required_action = required_action
		super().__init__(f"Rule '{rule_name}' violated: {reason}")


# ---------------------------------------------------------------------------
# Tenant / access control
# ---------------------------------------------------------------------------

def assert_tenant_context(context: dict[str, Any]) -> None:
	"""All operations require a non-empty tenant_id."""
	if not context.get("tenant_id"):
		raise RuleViolation(
			"tenant_context_required",
			"tenant_id is required for all AML operations",
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
	"""Cross-tenant data access is always denied."""
	if actor_tenant != resource_tenant:
		raise RuleViolation(
			"cross_tenant_access_denied",
			f"actor tenant '{actor_tenant}' cannot access resources of tenant '{resource_tenant}'",
			"use_own_tenant_resources",
		)


# ---------------------------------------------------------------------------
# Transaction monitoring rules
# ---------------------------------------------------------------------------

def assert_transaction_subject_present(subject_reference: str) -> None:
	"""Every monitored transaction must reference a subject (customer/entity)."""
	if not subject_reference or not subject_reference.strip():
		raise RuleViolation(
			"transaction_subject_required",
			"subject_reference is required on every monitored transaction",
			"attach_subject_reference",
		)


def assert_positive_amount(amount: float) -> None:
	"""Transaction amount must be strictly positive."""
	if amount <= 0:
		raise RuleViolation(
			"positive_amount_required",
			f"transaction amount must be > 0, got {amount}",
			"correct_transaction_amount",
		)


def assert_currency_present(currency: str) -> None:
	"""Currency code must be present and non-empty."""
	if not currency or not currency.strip():
		raise RuleViolation(
			"currency_required",
			"currency is required on every transaction",
			"attach_currency_code",
		)


def assert_source_reference_present(source_reference: str, source_capability: str) -> None:
	"""Source reference and capability must both be present for audit trail."""
	if not source_reference or not source_reference.strip():
		raise RuleViolation(
			"source_reference_required",
			"source_reference is required for AML audit trail",
			"attach_source_reference",
		)
	if not source_capability or not source_capability.strip():
		raise RuleViolation(
			"source_capability_required",
			"source_capability is required for AML audit trail",
			"attach_source_capability",
		)


def assert_kyc_link_present(kyc_profile_id: str) -> None:
	"""Transactions must be linked to a KYC profile for customer due diligence."""
	if not kyc_profile_id or not kyc_profile_id.strip():
		raise RuleViolation(
			"kyc_link_required",
			"kyc_profile_id must be provided — transactions must be linked to a KYC profile",
			"link_kyc_profile",
		)


# ---------------------------------------------------------------------------
# Alert rules
# ---------------------------------------------------------------------------

_SUPPORTED_ALERT_TYPES = {
	"large_transaction", "structuring", "velocity", "round_trip", "layering",
	"sanctions", "pep", "high_risk_kyc", "mule_account", "trade_based",
	"crypto_asset", "nft", "correspondent", "terrorist_financing", "agent_review",
}

_SUPPORTED_SEVERITIES = {"low", "medium", "high", "critical"}


def assert_alert_type_supported(alert_type: str) -> None:
	"""Alert type must be one of the recognised AML typologies."""
	if alert_type not in _SUPPORTED_ALERT_TYPES:
		raise RuleViolation(
			"unsupported_alert_type",
			f"alert_type '{alert_type}' is not a supported AML alert type",
			f"use_one_of: {sorted(_SUPPORTED_ALERT_TYPES)}",
		)


def assert_severity_supported(severity: str) -> None:
	"""Alert severity must be a recognised level."""
	if severity not in _SUPPORTED_SEVERITIES:
		raise RuleViolation(
			"unsupported_severity",
			f"severity '{severity}' is not supported",
			f"use_one_of: {sorted(_SUPPORTED_SEVERITIES)}",
		)


def assert_alert_evidence_present(evidence_references: list[str]) -> None:
	"""At least one evidence reference is required to create an alert.

	Prevents ghost alerts with no supporting evidence from polluting the queue.
	"""
	if not evidence_references:
		raise RuleViolation(
			"alert_evidence_required",
			"at least one evidence_reference is required when creating an AML alert",
			"attach_evidence_references",
		)


def assert_alert_escalation_has_reviewer(
	is_escalating: bool,
	reviewer_id: str,
) -> None:
	"""Escalating an alert requires a named reviewer — prevents anonymous escalations."""
	if is_escalating and (not reviewer_id or not reviewer_id.strip()):
		raise RuleViolation(
			"escalation_reviewer_required",
			"reviewer_id is required when escalating an alert",
			"assign_reviewer_before_escalation",
		)


def assert_alert_close_has_disposition(
	is_closing: bool,
	disposition: str,
) -> None:
	"""Closing an alert requires a disposition — raw closure without explanation is prohibited."""
	if is_closing and (not disposition or not disposition.strip()):
		raise RuleViolation(
			"alert_disposition_required",
			"disposition is required when closing an AML alert",
			"provide_disposition_before_closing",
		)


# ---------------------------------------------------------------------------
# Case management rules
# ---------------------------------------------------------------------------

_SUPPORTED_CASE_TYPES = {
	"transaction_monitoring", "sanctions_alert", "structuring_alert",
	"mule_account", "high_risk_customer", "terrorist_financing",
	"trade_based_ml", "crypto_asset", "network_analysis",
	"suspicious_activity_report",
}

_TERMINAL_CASE_STATUSES = {
	"closed_no_action", "closed_action_taken", "referred_to_lea",
}


def assert_case_type_supported(case_type: str) -> None:
	"""Case type must be one of the recognised AML investigation categories."""
	if case_type not in _SUPPORTED_CASE_TYPES:
		raise RuleViolation(
			"unsupported_case_type",
			f"case_type '{case_type}' is not supported",
			f"use_one_of: {sorted(_SUPPORTED_CASE_TYPES)}",
		)


def assert_investigator_assigned(investigator_id: str) -> None:
	"""Cases must have a named investigator — unassigned cases cannot proceed."""
	if not investigator_id or not investigator_id.strip():
		raise RuleViolation(
			"investigator_required",
			"investigator_id must be assigned before opening a case",
			"assign_investigator",
		)


def assert_case_is_open_for_investigation(status: str) -> None:
	"""Prevents modification of cases that have already reached a terminal status.

	Terminal statuses are: closed_no_action, closed_action_taken, referred_to_lea.
	"""
	if status in _TERMINAL_CASE_STATUSES:
		raise RuleViolation(
			"case_already_closed",
			f"case with status '{status}' cannot be modified — it is in a terminal state",
			"reopen_case_if_new_evidence_available",
		)


# ---------------------------------------------------------------------------
# SAR rules
# ---------------------------------------------------------------------------

_SAR_NARRATIVE_MIN_LEN = 50  # FATF / FinCEN guidance: narratives must be substantive


def assert_sar_narrative_present(narrative: str) -> None:
	"""SAR narrative must be substantive — minimum 50 characters.

	Regulators (FinCEN, FCA, FATF) require a clear description of the suspicious
	activity. One-liners are routinely rejected.
	"""
	if not narrative or len(narrative.strip()) < _SAR_NARRATIVE_MIN_LEN:
		raise RuleViolation(
			"sar_narrative_insufficient",
			f"SAR narrative must be at least {_SAR_NARRATIVE_MIN_LEN} characters; "
			f"got {len(narrative.strip()) if narrative else 0}",
			"expand_sar_narrative",
		)


def assert_sar_jurisdiction_present(jurisdiction: str) -> None:
	"""SAR must specify the filing jurisdiction."""
	if not jurisdiction or not jurisdiction.strip():
		raise RuleViolation(
			"sar_jurisdiction_required",
			"jurisdiction is required on every SAR",
			"specify_sar_jurisdiction",
		)


def assert_sar_human_approval(approved_by: str) -> None:
	"""SARs require explicit human approval before filing.

	Automated/agent-only SAR filing is prohibited — a named compliance officer
	must approve every SAR before it is submitted to the regulator.
	"""
	if not approved_by or not approved_by.strip():
		raise RuleViolation(
			"sar_human_approval_required",
			"SARs must be approved by a named compliance officer before filing",
			"obtain_human_approval_for_sar",
		)


# ---------------------------------------------------------------------------
# CTR rules
# ---------------------------------------------------------------------------

def assert_ctr_amount_triggers_reporting(amount: float, threshold: float) -> None:
	"""CTR can only be filed if the transaction amount meets the jurisdictional threshold.

	Filing a CTR for a sub-threshold amount is a regulatory error.
	"""
	if amount < threshold:
		raise RuleViolation(
			"ctr_threshold_not_met",
			f"CTR requires amount >= {threshold}; got {amount}",
			"verify_transaction_amount_before_ctr_filing",
		)


# ---------------------------------------------------------------------------
# Watchlist rules
# ---------------------------------------------------------------------------

def assert_match_score_valid(score: float) -> None:
	"""Watchlist match score must be in [0.0, 1.0]."""
	if not (0.0 <= score <= 1.0):
		raise RuleViolation(
			"invalid_match_score",
			f"match_score must be between 0.0 and 1.0, got {score}",
			"correct_match_score",
		)


# ---------------------------------------------------------------------------
# Trade-based money laundering (TBML) rules
# ---------------------------------------------------------------------------

def assert_tbml_invoice_variance_acceptable(
	invoice_amount: float,
	market_value: float,
	tolerance_pct: float = 0.15,
) -> None:
	"""TBML detection: invoice amount must not deviate more than tolerance_pct from market value.

	Over/under-invoicing is the primary TBML mechanism — a 15% deviation is
	the FATF-recommended threshold for further investigation.
	"""
	if market_value <= 0:
		raise RuleViolation(
			"tbml_invalid_market_value",
			"market_value must be positive for TBML invoice validation",
			"provide_valid_market_value",
		)
	deviation = abs(invoice_amount - market_value) / market_value
	if deviation > tolerance_pct:
		direction = "over_invoiced" if invoice_amount > market_value else "under_invoiced"
		raise RuleViolation(
			"tbml_invoice_variance_exceeded",
			f"invoice amount is {direction}: deviates {deviation:.1%} from market value "
			f"(threshold: {tolerance_pct:.1%})",
			"escalate_for_tbml_review",
		)


# ---------------------------------------------------------------------------
# Crypto / NFT rules
# ---------------------------------------------------------------------------

def assert_crypto_mixer_not_detected(
	mixer_indicators: list[str],
) -> None:
	"""Transactions routed through known mixing services must be blocked/escalated.

	Crypto mixers (Tornado Cash, Chipmixer patterns) are red-flag indicators
	under FATF Recommendation 15 and most national regimes.
	"""
	if mixer_indicators:
		raise RuleViolation(
			"crypto_mixer_detected",
			f"transaction shows mixer routing indicators: {mixer_indicators}",
			"escalate_crypto_mixer_transaction",
		)


def assert_nft_wash_trade_not_detected(
	wash_trade_score: float,
	threshold: float = 0.7,
) -> None:
	"""NFT wash-trade detection: score above threshold triggers rule violation."""
	if wash_trade_score >= threshold:
		raise RuleViolation(
			"nft_wash_trade_detected",
			f"NFT wash-trade score {wash_trade_score:.2f} >= threshold {threshold:.2f}",
			"escalate_nft_wash_trade",
		)


# ---------------------------------------------------------------------------
# Correspondent banking / nested account rules
# ---------------------------------------------------------------------------

def assert_correspondent_nesting_depth_acceptable(
	nesting_depth: int,
	max_depth: int = 3,
) -> None:
	"""Correspondent banking nesting beyond max_depth is a high-risk indicator.

	FATF typology: layered correspondent relationships obscure beneficial ownership.
	Max allowed nesting depth (direct + nested) is 3 by default.
	"""
	if nesting_depth > max_depth:
		raise RuleViolation(
			"correspondent_nesting_too_deep",
			f"correspondent nesting depth {nesting_depth} exceeds maximum {max_depth}",
			"escalate_correspondent_nesting",
		)


# ---------------------------------------------------------------------------
# Terrorist financing rules
# ---------------------------------------------------------------------------

def assert_no_terrorist_financing_indicators(
	tf_indicators: list[str],
) -> None:
	"""Any detected terrorist financing indicator must trigger immediate escalation.

	Zero-tolerance: TF indicators are never false positives until proven otherwise.
	"""
	if tf_indicators:
		raise RuleViolation(
			"terrorist_financing_indicator_detected",
			f"transaction has terrorist financing indicators: {tf_indicators}",
			"escalate_immediately_to_compliance",
		)


# ---------------------------------------------------------------------------
# Regulatory filing rules
# ---------------------------------------------------------------------------

def assert_filing_period_valid(period_start: Any, period_end: Any) -> None:
	"""Regulatory filing period must have start before end."""
	from datetime import datetime
	if isinstance(period_start, str):
		try:
			period_start = datetime.fromisoformat(period_start)
		except ValueError:
			raise RuleViolation("invalid_period_start", "period_start is not a valid datetime", "fix_period_dates")
	if isinstance(period_end, str):
		try:
			period_end = datetime.fromisoformat(period_end)
		except ValueError:
			raise RuleViolation("invalid_period_end", "period_end is not a valid datetime", "fix_period_dates")
	if period_start >= period_end:
		raise RuleViolation(
			"invalid_filing_period",
			f"period_start ({period_start}) must be before period_end ({period_end})",
			"correct_filing_period",
		)
