"""Deterministic domain rules for Intelligence Fusion.

Every business rule is a pure callable — no I/O, no side-effects.
:class:`RuleViolation` is the single exception type raised on violation.

Methodology references:
  - ICD 203 (Analytic Standards)
  - TLP 2.0 (Traffic Light Protocol)
  - IC Information Sharing Standards (ODNI)
  - Classification dominance principle (MR 32-1)

© 2025 Datacraft — Nyimbi Odero
"""
from __future__ import annotations

from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Core exception
# ─────────────────────────────────────────────────────────────────────────────

class RuleViolation(Exception):
	"""Raised when a business rule is violated."""

	def __init__(self, rule_name: str, reason: str, required_action: str = "") -> None:
		self.rule_name = rule_name
		self.reason = reason
		self.required_action = required_action
		super().__init__(f"Rule '{rule_name}' violated: {reason}")


# ─────────────────────────────────────────────────────────────────────────────
# Enumerated valid values (single source of truth — keep in sync with models.py)
# ─────────────────────────────────────────────────────────────────────────────

_VALID_SOURCE_TYPES: frozenset[str] = frozenset({
	"osint", "sigint", "humint", "geoint", "cybint",
	"finint", "socint", "darkweb", "radio", "monitoring", "partner_report",
})

_VALID_WORKSPACE_TYPES: frozenset[str] = frozenset({
	"case_fusion", "threat_fusion", "fraud_fusion",
	"public_safety", "strategic_assessment", "operational_picture", "incident_fusion",
})

_VALID_CORRELATION_TYPES: frozenset[str] = frozenset({
	"entity_match", "time_sequence", "location_overlap",
	"network_link", "pattern_match", "cross_source_confirmation", "contradiction",
})

_VALID_ASSESSMENT_TYPES: frozenset[str] = frozenset({
	"threat", "fraud", "public_safety", "operational", "strategic", "confidence", "impact",
})

_VALID_RISK_LEVELS: frozenset[str] = frozenset({"low", "medium", "high", "critical"})

_VALID_TLP_LEVELS: frozenset[str] = frozenset({
	"TLP:WHITE", "TLP:GREEN", "TLP:AMBER", "TLP:RED", "TLP:CLEAR",
})

_VALID_EVIDENCE_TYPES: frozenset[str] = frozenset({
	"document", "signal", "image", "video",
	"geospatial", "transaction", "indicator", "entity", "observation",
})

_VALID_SAT_METHODS: frozenset[str] = frozenset({
	"analysis_of_competing_hypotheses",
	"key_assumptions_check",
	"devils_advocacy",
	"red_team",
	"cone_of_plausibility",
	"premortem",
	"quality_of_information_check",
})

_VALID_JUDGEMENT_TYPES: frozenset[str] = frozenset({
	"attribution", "intent", "capability", "risk",
	"relationship", "timeline", "course_of_action",
})

_VALID_PRODUCT_TYPES: frozenset[str] = frozenset({
	"sitrep", "threat_assessment", "intelligence_brief",
	"finished_intelligence", "tactical_report", "strategic_estimate",
})

_TLP_ORDER: dict[str, int] = {
	"TLP:WHITE": 0, "TLP:CLEAR": 0,
	"TLP:GREEN": 1, "TLP:AMBER": 2, "TLP:RED": 3,
}

_CLASSIFICATION_ORDER: dict[str, int] = {
	"unclassified": 0, "confidential": 1, "secret": 2, "top_secret": 3,
}

_CONFIDENCE_LEVELS: list[tuple[float, str]] = [
	(0.93, "almost_certain"),
	(0.80, "highly_likely"),
	(0.55, "likely"),
	(0.45, "roughly_even"),
	(0.20, "unlikely"),
	(0.07, "highly_unlikely"),
	(0.00, "remote"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Tenant / access control
# ─────────────────────────────────────────────────────────────────────────────

def assert_tenant_context(tenant_id: str) -> None:
	"""Every operation requires a non-blank tenant_id."""
	if not tenant_id or not tenant_id.strip():
		raise RuleViolation(
			"tenant_context_required",
			"tenant_id is required",
			"attach_tenant_context",
		)


def assert_no_cross_tenant_access(actor_tenant: str, resource_tenant: str) -> None:
	"""Cross-tenant access is unconditionally denied."""
	if actor_tenant != resource_tenant:
		raise RuleViolation(
			"cross_tenant_access_denied",
			f"actor tenant '{actor_tenant}' may not access resources owned by '{resource_tenant}'",
			"use_own_tenant_resources",
		)


# ─────────────────────────────────────────────────────────────────────────────
# Lawful authority
# ─────────────────────────────────────────────────────────────────────────────

def assert_lawful_authority(authority_id: str, authority_valid: bool) -> None:
	"""Intelligence collection requires a valid, non-blank authority reference."""
	if not authority_id or not authority_id.strip() or not authority_valid:
		raise RuleViolation(
			"lawful_authority_required",
			"a valid legal authority is required for intelligence collection",
			"obtain_lawful_authority",
		)


# ─────────────────────────────────────────────────────────────────────────────
# Source / custodian
# ─────────────────────────────────────────────────────────────────────────────

def assert_source_type_supported(source_type: str) -> None:
	"""Source type must be one of the registered disciplines."""
	if source_type not in _VALID_SOURCE_TYPES:
		raise RuleViolation(
			"source_type_not_supported",
			f"'{source_type}' is not a recognised source discipline",
			f"use_one_of: {sorted(_VALID_SOURCE_TYPES)}",
		)


def assert_content_fingerprint_present(fingerprint: str) -> None:
	"""Every ingested item must carry a non-blank content fingerprint for deduplication."""
	if not fingerprint or not fingerprint.strip():
		raise RuleViolation(
			"content_fingerprint_required",
			"a content fingerprint (hash) is required for deduplication",
			"provide_content_fingerprint",
		)


def assert_custodian_assigned(custodian_id: str) -> None:
	"""Every item must have a custodian for chain-of-custody tracking."""
	if not custodian_id or not custodian_id.strip():
		raise RuleViolation(
			"custodian_required",
			"a custodian must be assigned to every intelligence item",
			"assign_custodian",
		)


# ─────────────────────────────────────────────────────────────────────────────
# Confidence
# ─────────────────────────────────────────────────────────────────────────────

def assert_confidence_in_range(confidence: float) -> None:
	"""Confidence score must be in [0.0, 1.0]."""
	if not (0.0 <= confidence <= 1.0):
		raise RuleViolation(
			"confidence_score_invalid",
			f"confidence score {confidence} is outside [0.0, 1.0]",
			"provide_confidence_in_range",
		)


def calculate_confidence_level(score: float) -> str:
	"""Map a numeric confidence score to the ICD-203 estimative word."""
	score = max(0.0, min(1.0, score))
	for threshold, word in _CONFIDENCE_LEVELS:
		if score >= threshold:
			return word
	return "remote"


# ─────────────────────────────────────────────────────────────────────────────
# Workspace
# ─────────────────────────────────────────────────────────────────────────────

def assert_workspace_type_supported(workspace_type: str) -> None:
	"""Workspace type must be one of the registered fusion types."""
	if workspace_type not in _VALID_WORKSPACE_TYPES:
		raise RuleViolation(
			"workspace_type_not_supported",
			f"'{workspace_type}' is not a supported workspace type",
			f"use_one_of: {sorted(_VALID_WORKSPACE_TYPES)}",
		)


def assert_workspace_active(status: str) -> None:
	"""Operations that mutate workspace content require an active workspace."""
	if status != "active":
		raise RuleViolation(
			"workspace_not_active",
			f"workspace status '{status}' does not permit this operation",
			"reactivate_workspace",
		)


def assert_workspace_not_closed(status: str) -> None:
	"""Closed workspaces cannot be updated."""
	if status == "closed":
		raise RuleViolation(
			"workspace_is_closed",
			"closed workspaces cannot be modified",
			"open_new_workspace",
		)


# ─────────────────────────────────────────────────────────────────────────────
# Classification
# ─────────────────────────────────────────────────────────────────────────────

def assert_classification_dominance(
	item_classification: str,
	workspace_classification: str,
) -> None:
	"""
	An item's classification must not exceed the workspace's classification ceiling.

	The workspace sets the maximum classification level for all contained items.
	Exceeding it would require a workspace upgrade first (domination principle).
	"""
	item_rank = _CLASSIFICATION_ORDER.get(item_classification.lower(), 99)
	ws_rank = _CLASSIFICATION_ORDER.get(workspace_classification.lower(), 0)
	if item_rank > ws_rank:
		raise RuleViolation(
			"classification_exceeds_workspace",
			f"item classification '{item_classification}' exceeds workspace ceiling '{workspace_classification}'",
			"upgrade_workspace_classification_or_downgrade_item",
		)


# ─────────────────────────────────────────────────────────────────────────────
# Correlation
# ─────────────────────────────────────────────────────────────────────────────

def assert_correlation_type_supported(correlation_type: str) -> None:
	"""Correlation type must be one of the registered types."""
	if correlation_type not in _VALID_CORRELATION_TYPES:
		raise RuleViolation(
			"correlation_type_not_supported",
			f"'{correlation_type}' is not a supported correlation type",
			f"use_one_of: {sorted(_VALID_CORRELATION_TYPES)}",
		)


def assert_correlation_has_items(item_ids: list[str]) -> None:
	"""A correlation set requires at least two items to be meaningful."""
	if len(item_ids) < 2:
		raise RuleViolation(
			"correlation_requires_minimum_items",
			f"correlation requires at least 2 items; got {len(item_ids)}",
			"add_more_items_to_correlation",
		)


# ─────────────────────────────────────────────────────────────────────────────
# Assessment
# ─────────────────────────────────────────────────────────────────────────────

def assert_assessment_type_supported(assessment_type: str) -> None:
	"""Assessment type must be one of the registered types."""
	if assessment_type not in _VALID_ASSESSMENT_TYPES:
		raise RuleViolation(
			"assessment_type_not_supported",
			f"'{assessment_type}' is not a supported assessment type",
			f"use_one_of: {sorted(_VALID_ASSESSMENT_TYPES)}",
		)


def assert_risk_level_supported(risk_level: str) -> None:
	"""Risk level must be one of: low, medium, high, critical."""
	if risk_level not in _VALID_RISK_LEVELS:
		raise RuleViolation(
			"risk_level_not_supported",
			f"'{risk_level}' is not a supported risk level",
			"use_one_of: low, medium, high, critical",
		)


def assert_assessment_has_hypotheses(hypothesis_ids: list[str]) -> None:
	"""An assessment picture must reference at least one hypothesis."""
	if not hypothesis_ids:
		raise RuleViolation(
			"assessment_requires_hypotheses",
			"assessment picture must reference at least one hypothesis",
			"create_hypothesis_first",
		)


def assert_assessment_has_correlations(correlation_ids: list[str]) -> None:
	"""An assessment picture must reference at least one correlation set."""
	if not correlation_ids:
		raise RuleViolation(
			"assessment_requires_correlations",
			"assessment picture must reference at least one correlation set",
			"create_correlation_first",
		)


# ─────────────────────────────────────────────────────────────────────────────
# TLP / Dissemination
# ─────────────────────────────────────────────────────────────────────────────

def assert_tlp_valid(tlp: str) -> None:
	"""TLP must be one of the defined Traffic Light Protocol levels."""
	if tlp.upper() not in _VALID_TLP_LEVELS:
		raise RuleViolation(
			"tlp_level_not_supported",
			f"'{tlp}' is not a valid TLP level",
			f"use_one_of: {sorted(_VALID_TLP_LEVELS)}",
		)


def assert_tlp_compatible_with_audience(product_tlp: str, recipient_max_tlp: str) -> None:
	"""Product TLP must not exceed recipient's maximum authorised TLP level."""
	p = _TLP_ORDER.get(product_tlp.upper(), 99)
	r = _TLP_ORDER.get(recipient_max_tlp.upper(), -1)
	if p > r:
		raise RuleViolation(
			"tlp_exceeds_recipient_clearance",
			f"product TLP '{product_tlp}' exceeds recipient maximum '{recipient_max_tlp}'",
			"downgrade_tlp_or_use_cleared_recipient",
		)


def assert_audience_specified(audience: str) -> None:
	"""Dissemination requires a named audience."""
	if not audience or not audience.strip():
		raise RuleViolation(
			"audience_required",
			"a named audience is required for dissemination",
			"specify_audience",
		)


def assert_approval_present(approval_reference: str) -> None:
	"""Release and dissemination require a non-blank approval reference."""
	if not approval_reference or not approval_reference.strip():
		raise RuleViolation(
			"approval_reference_required",
			"an approval reference is required for release/dissemination",
			"obtain_approval_reference",
		)


def assert_no_autonomous_dissemination(autonomous: bool) -> None:
	"""Intelligence products must not be disseminated autonomously by agents."""
	if autonomous:
		raise RuleViolation(
			"autonomous_dissemination_denied",
			"autonomous dissemination of intelligence products is prohibited",
			"obtain_human_approval_before_dissemination",
		)


# ─────────────────────────────────────────────────────────────────────────────
# Product lifecycle
# ─────────────────────────────────────────────────────────────────────────────

def assert_product_has_assessments(assessment_ids: list[str]) -> None:
	"""A product must reference at least one assessment picture."""
	if not assessment_ids:
		raise RuleViolation(
			"product_requires_assessments",
			"intelligence product must reference at least one assessment picture",
			"create_assessment_first",
		)


def assert_product_in_draft_for_submit(status: str) -> None:
	"""Only draft products can be submitted for review."""
	if status != "draft":
		raise RuleViolation(
			"product_must_be_draft_for_review",
			f"product in status '{status}' cannot be submitted for review; must be 'draft'",
			"reset_product_to_draft",
		)


def assert_product_in_review_for_approval(status: str) -> None:
	"""Only products under review can be approved."""
	if status != "review":
		raise RuleViolation(
			"product_must_be_in_review_for_approval",
			f"product in status '{status}' cannot be approved; must be 'review'",
			"submit_for_review_first",
		)


def assert_product_in_approved_state(status: str) -> None:
	"""Only approved products can be released."""
	if status != "approved":
		raise RuleViolation(
			"product_must_be_approved_before_release",
			f"product in status '{status}' cannot be released; must be 'approved'",
			"approve_product_first",
		)


def assert_product_not_recalled(status: str) -> None:
	"""Recalled products cannot be modified."""
	if status == "recalled":
		raise RuleViolation(
			"recalled_product_cannot_be_modified",
			"recalled products are immutable",
			"create_new_product_version",
		)


# ─────────────────────────────────────────────────────────────────────────────
# Evidence
# ─────────────────────────────────────────────────────────────────────────────

def assert_evidence_type_supported(evidence_type: str) -> None:
	"""Evidence type must be one of the registered types."""
	if evidence_type not in _VALID_EVIDENCE_TYPES:
		raise RuleViolation(
			"evidence_type_not_supported",
			f"'{evidence_type}' is not a supported evidence type",
			f"use_one_of: {sorted(_VALID_EVIDENCE_TYPES)}",
		)


def assert_chain_of_custody_present(chain: list[str]) -> None:
	"""Evidence must include at least one chain-of-custody entry."""
	if not chain:
		raise RuleViolation(
			"chain_of_custody_required",
			"at least one chain-of-custody entry is required",
			"provide_custody_entry",
		)


def assert_evidence_not_discredited(status: str) -> None:
	"""Discredited evidence cannot be used in hypotheses or assessments."""
	if status == "discredited":
		raise RuleViolation(
			"discredited_evidence_cannot_be_used",
			"discredited evidence must not be referenced in active analysis",
			"remove_discredited_evidence_reference",
		)


def assert_no_evidence_fabrication(fabricated: bool) -> None:
	"""Evidence fabrication is absolutely prohibited."""
	if fabricated:
		raise RuleViolation(
			"evidence_fabrication_denied",
			"fabricating intelligence evidence is prohibited",
			"use_only_genuine_evidence",
		)


def assert_no_source_tampering(tampered: bool) -> None:
	"""Source records must not be altered after ingestion."""
	if tampered:
		raise RuleViolation(
			"source_tampering_denied",
			"tampering with source intelligence records is prohibited",
			"preserve_original_source_integrity",
		)


# ─────────────────────────────────────────────────────────────────────────────
# Hypothesis / SAT
# ─────────────────────────────────────────────────────────────────────────────

def assert_sat_method_supported(method: str) -> None:
	"""SAT method must be one of the registered structured analytic techniques."""
	if method not in _VALID_SAT_METHODS:
		raise RuleViolation(
			"sat_method_not_supported",
			f"'{method}' is not a recognised structured analytic technique",
			f"use_one_of: {sorted(_VALID_SAT_METHODS)}",
		)


def assert_hypothesis_has_alternatives(alternative_hypotheses: list[str]) -> None:
	"""ACH requires at least one alternative hypothesis for comparison."""
	if not alternative_hypotheses:
		raise RuleViolation(
			"ach_requires_alternatives",
			"Analysis of Competing Hypotheses requires at least one alternative hypothesis",
			"add_alternative_hypotheses",
		)


def assert_hypothesis_open_for_update(status: str) -> None:
	"""Only open or inconclusive hypotheses can be updated."""
	if status in ("supported", "refuted"):
		raise RuleViolation(
			"hypothesis_is_closed",
			f"hypothesis in status '{status}' cannot be updated",
			"create_new_hypothesis",
		)


# ─────────────────────────────────────────────────────────────────────────────
# Analyst / judgement
# ─────────────────────────────────────────────────────────────────────────────

def assert_analyst_assigned(analyst_id: str) -> None:
	"""Every analytical artefact requires a named analyst for accountability."""
	if not analyst_id or not analyst_id.strip():
		raise RuleViolation(
			"analyst_required",
			"a named analyst must be assigned to every analytical artefact",
			"assign_analyst",
		)


def assert_judgement_type_supported(judgement_type: str) -> None:
	"""Judgement type must be one of the registered types."""
	if judgement_type not in _VALID_JUDGEMENT_TYPES:
		raise RuleViolation(
			"judgement_type_not_supported",
			f"'{judgement_type}' is not a supported judgement type",
			f"use_one_of: {sorted(_VALID_JUDGEMENT_TYPES)}",
		)


def assert_no_unapproved_attribution(unapproved: bool) -> None:
	"""Attribution judgements must not be published without senior analyst approval."""
	if unapproved:
		raise RuleViolation(
			"unapproved_attribution_denied",
			"attribution statements require senior analyst approval before publication",
			"obtain_attribution_approval",
		)


# ─────────────────────────────────────────────────────────────────────────────
# Privacy / ethical guardrails
# ─────────────────────────────────────────────────────────────────────────────

def assert_no_privacy_bypass(bypass_attempted: bool) -> None:
	"""Privacy protections cannot be circumvented."""
	if bypass_attempted:
		raise RuleViolation(
			"privacy_bypass_denied",
			"bypassing privacy protections is prohibited",
			"apply_privacy_safeguards",
		)


def assert_privileged_agent_action_has_approval(
	is_privileged: bool,
	has_human_approval: bool,
) -> None:
	"""AI/automated agents must have explicit human approval for privileged actions."""
	if is_privileged and not has_human_approval:
		raise RuleViolation(
			"privileged_agent_action_requires_human_approval",
			"privileged agent actions require explicit human approval",
			"obtain_human_approval",
		)


# ─────────────────────────────────────────────────────────────────────────────
# Fusion-specific
# ─────────────────────────────────────────────────────────────────────────────

def assert_minimum_sources_for_fusion(source_count: int, minimum: int = 2) -> None:
	"""Intelligence fusion requires a minimum number of source items."""
	if source_count < minimum:
		raise RuleViolation(
			"insufficient_sources_for_fusion",
			f"fusion requires at least {minimum} sources; got {source_count}",
			"add_more_intelligence_items",
		)


def assert_time_window_valid(start_ts: float, end_ts: float) -> None:
	"""A time window must have a positive duration."""
	if end_ts <= start_ts:
		raise RuleViolation(
			"invalid_time_window",
			f"time window end ({end_ts}) must be after start ({start_ts})",
			"provide_valid_time_window",
		)
