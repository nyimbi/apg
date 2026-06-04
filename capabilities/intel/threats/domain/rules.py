"""Deterministic domain rules for Threat Intelligence.

Every business rule from the capability contract is implemented here as a
pure callable.  No I/O, no side effects — rules are deterministic functions
over domain state.

RuleViolation is the single exception type; callers may catch and translate it
into HTTP 422 / PermissionError as appropriate.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


# ── Exception ─────────────────────────────────────────────────────────────────

class RuleViolation(Exception):
	"""Raised when a business rule is violated."""

	def __init__(self, rule_name: str, reason: str, required_action: str = "") -> None:
		self.rule_name = rule_name
		self.reason = reason
		self.required_action = required_action
		super().__init__(f"Rule '{rule_name}' violated: {reason}")


# ── Tenant / Access rules ──────────────────────────────────────────────────────

def assert_tenant_context(context: dict[str, Any]) -> None:
	"""All operations require a non-empty tenant_id."""
	if not context.get("tenant_id"):
		raise RuleViolation(
			"tenant_context_required",
			"tenant_id is required for all operations",
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
	"""Cross-tenant access is unconditionally denied."""
	if actor_tenant != resource_tenant:
		raise RuleViolation(
			"cross_tenant_access_denied",
			f"actor tenant '{actor_tenant}' may not access resources owned by '{resource_tenant}'",
			"use_own_tenant_resources",
		)


def assert_tenant_owns_resource(tenant_id: str, resource: dict[str, Any]) -> None:
	"""The resource's tenant_id must match the calling tenant."""
	resource_tenant = resource.get("tenant_id", "")
	if resource_tenant != tenant_id:
		raise RuleViolation(
			"tenant_resource_ownership",
			f"resource belongs to tenant '{resource_tenant}', not '{tenant_id}'",
			"use_own_tenant_resources",
		)


# ── Confidence / scoring rules ─────────────────────────────────────────────────

def assert_confidence_in_range(score: float, field: str = "confidence_score") -> None:
	"""Confidence score must be in [0.0, 1.0]."""
	if not (0.0 <= score <= 1.0):
		raise RuleViolation(
			"confidence_score_out_of_range",
			f"{field} must be in [0.0, 1.0], got {score}",
			"correct_confidence_score",
		)


def assert_tlp_valid(tlp: str) -> None:
	"""TLP marking must be one of the recognised levels."""
	valid = {"white", "clear", "green", "amber", "amber+strict", "red"}
	if tlp.lower() not in valid:
		raise RuleViolation(
			"invalid_tlp_marking",
			f"TLP '{tlp}' is not a recognised marking; use one of {valid}",
			"set_valid_tlp",
		)


def assert_classification_valid(classification: str) -> None:
	"""Classification must be a recognised sensitivity label."""
	valid = {"unclassified", "confidential", "secret", "top_secret"}
	if classification.lower() not in valid:
		raise RuleViolation(
			"invalid_classification",
			f"Classification '{classification}' is not valid; use one of {valid}",
			"set_valid_classification",
		)


# ── ThreatActor rules ──────────────────────────────────────────────────────────

VALID_ACTOR_TYPES = {
	"state_actor", "criminal_group", "insider", "hacktivist",
	"terrorist_network", "competitor", "unknown",
}
VALID_ACTOR_STATUSES = {"active", "dormant", "retired", "attributed", "suspected"}

VALID_SOPHISTICATION = {"minimal", "intermediate", "advanced", "nation-state"}
VALID_MOTIVATION = {"espionage", "financial", "hacktivism", "terrorism", "disruption", "unknown"}


def assert_actor_type_valid(actor_type: str) -> None:
	if actor_type not in VALID_ACTOR_TYPES:
		raise RuleViolation(
			"invalid_actor_type",
			f"actor_type '{actor_type}' not recognised; valid: {VALID_ACTOR_TYPES}",
			"set_valid_actor_type",
		)


def assert_actor_has_evidence(evidence_reference: str) -> None:
	if not evidence_reference or not evidence_reference.strip():
		raise RuleViolation(
			"actor_evidence_required",
			"Every threat actor record must cite an evidence reference",
			"attach_evidence_reference",
		)


def assert_actor_not_retired_for_update(status: str) -> None:
	if status == "retired":
		raise RuleViolation(
			"actor_retired_immutable",
			"Retired actors cannot be modified; create a new record",
			"create_new_actor_record",
		)


def assert_attribution_minimum_evidence(evidence_count: int, min_count: int = 2) -> None:
	"""Attribution requires at minimum *min_count* independent evidence pieces."""
	if evidence_count < min_count:
		raise RuleViolation(
			"attribution_insufficient_evidence",
			f"Attribution requires at least {min_count} evidence items; got {evidence_count}",
			"gather_more_attribution_evidence",
		)


# ── ThreatIndicator rules ──────────────────────────────────────────────────────

VALID_INDICATOR_TYPES = {
	"ip_address", "domain", "url",
	"file_hash_md5", "file_hash_sha1", "file_hash_sha256",
	"email_address", "registry_key", "mutex", "network_signature",
	"certificate", "user_agent", "yara_rule", "financial_signal",
	"behavior", "ioc", "tactic", "technique", "procedure",
	"vulnerability", "infrastructure", "narrative",
}


def assert_indicator_type_valid(indicator_type: str) -> None:
	if indicator_type not in VALID_INDICATOR_TYPES:
		raise RuleViolation(
			"invalid_indicator_type",
			f"indicator_type '{indicator_type}' not recognised",
			"set_valid_indicator_type",
		)


def assert_indicator_value_present(value: str) -> None:
	if not value or not value.strip():
		raise RuleViolation(
			"indicator_value_required",
			"indicator value must not be empty",
			"provide_indicator_value",
		)


def assert_indicator_not_expired(valid_until: datetime | None) -> None:
	"""Prevent ingesting indicators whose validity window has already closed."""
	if valid_until is None:
		return
	vu = valid_until.replace(tzinfo=timezone.utc) if valid_until.tzinfo is None else valid_until
	if vu < datetime.now(timezone.utc):
		raise RuleViolation(
			"indicator_validity_expired",
			f"indicator valid_until {valid_until.isoformat()} is in the past",
			"update_valid_until_or_revoke",
		)


def assert_no_duplicate_indicator(
	indicator_type: str,
	value: str,
	existing_fingerprints: set[str],
	fingerprint: str,
) -> None:
	"""Prevent exact-duplicate indicators (same type + value) within a tenant."""
	if fingerprint in existing_fingerprints:
		raise RuleViolation(
			"duplicate_indicator",
			f"Indicator of type '{indicator_type}' with value '{value}' already exists",
			"update_existing_indicator_instead",
		)


# ── ThreatCampaign rules ───────────────────────────────────────────────────────

VALID_CAMPAIGN_TYPES = {
	"intrusion_campaign", "fraud_campaign", "disinformation_campaign",
	"physical_threat_campaign", "insider_campaign", "supply_chain_campaign",
	"ransomware_campaign", "espionage_campaign",
}
VALID_RISK_LEVELS = {"low", "medium", "high", "critical"}


def assert_campaign_type_valid(campaign_type: str) -> None:
	if campaign_type not in VALID_CAMPAIGN_TYPES:
		raise RuleViolation(
			"invalid_campaign_type",
			f"campaign_type '{campaign_type}' not recognised",
			"set_valid_campaign_type",
		)


def assert_risk_level_valid(risk_level: str) -> None:
	if risk_level not in VALID_RISK_LEVELS:
		raise RuleViolation(
			"invalid_risk_level",
			f"risk_level '{risk_level}' not recognised; valid: {VALID_RISK_LEVELS}",
			"set_valid_risk_level",
		)


def assert_campaign_actor_present(actor_id: str) -> None:
	if not actor_id or not actor_id.strip():
		raise RuleViolation(
			"campaign_actor_required",
			"Campaign must be attributed to a threat actor",
			"provide_actor_id",
		)


def assert_campaign_date_order(
	first_seen: datetime | None,
	last_seen: datetime | None,
) -> None:
	"""last_seen must not precede first_seen."""
	if first_seen is None or last_seen is None:
		return
	fs = first_seen.replace(tzinfo=timezone.utc) if first_seen.tzinfo is None else first_seen
	ls = last_seen.replace(tzinfo=timezone.utc) if last_seen.tzinfo is None else last_seen
	if ls < fs:
		raise RuleViolation(
			"campaign_date_order_invalid",
			f"last_seen {last_seen.isoformat()} precedes first_seen {first_seen.isoformat()}",
			"correct_campaign_dates",
		)


# ── ThreatReport rules ─────────────────────────────────────────────────────────

VALID_REPORT_TYPES = {
	"brief", "advisory", "bulletin", "estimate", "watchlist",
	"situation_report", "flash_report", "strategic_assessment",
}
VALID_REPORT_STATUSES = {"draft", "under_review", "approved", "published", "retracted"}


def assert_report_type_valid(report_type: str) -> None:
	if report_type not in VALID_REPORT_TYPES:
		raise RuleViolation(
			"invalid_report_type",
			f"report_type '{report_type}' not recognised",
			"set_valid_report_type",
		)


def assert_report_approval_for_publish(status: str, approved_by: str | None) -> None:
	"""Publishing a report requires prior approval."""
	if status == "published" and not approved_by:
		raise RuleViolation(
			"report_publish_requires_approval",
			"Report cannot be published without a recorded approval",
			"obtain_report_approval_first",
		)


def assert_report_not_retracted_for_update(status: str) -> None:
	if status == "retracted":
		raise RuleViolation(
			"report_retracted_immutable",
			"Retracted reports cannot be modified",
			"create_replacement_report",
		)


# ── ThreatAssessment rules ─────────────────────────────────────────────────────

VALID_ASSESSMENT_TYPES = {
	"threat_profile", "risk_assessment", "priority_assessment",
	"attribution_assessment", "intent_assessment", "capability_assessment",
}


def assert_assessment_type_valid(assessment_type: str) -> None:
	if assessment_type not in VALID_ASSESSMENT_TYPES:
		raise RuleViolation(
			"invalid_assessment_type",
			f"assessment_type '{assessment_type}' not recognised",
			"set_valid_assessment_type",
		)


def assert_assessment_analyst_present(analyst_id: str) -> None:
	if not analyst_id or not analyst_id.strip():
		raise RuleViolation(
			"assessment_analyst_required",
			"Assessment must be assigned to an analyst",
			"provide_analyst_id",
		)


def assert_assessment_not_approved_for_downgrade(
	current_status_approved: bool,
	new_risk_level: str,
	old_risk_level: str,
) -> None:
	"""Approved assessments may not silently downgrade risk level."""
	if current_status_approved and new_risk_level != old_risk_level:
		raise RuleViolation(
			"approved_assessment_risk_level_locked",
			"Approved assessment risk level cannot be changed; create a new assessment",
			"create_new_assessment",
		)


# ── ThreatFeed rules ───────────────────────────────────────────────────────────

VALID_FEED_TYPES = {
	"stix_taxii", "misp", "csv", "json_api", "osint_scrape",
	"partner_share", "internal",
}
VALID_FEED_STATUSES = {"active", "paused", "error", "deprecated"}


def assert_feed_type_valid(feed_type: str) -> None:
	if feed_type not in VALID_FEED_TYPES:
		raise RuleViolation(
			"invalid_feed_type",
			f"feed_type '{feed_type}' not recognised",
			"set_valid_feed_type",
		)


def assert_feed_poll_interval_sane(poll_interval_seconds: int) -> None:
	"""Poll intervals below 60 seconds are not permitted (rate-limit protection)."""
	if poll_interval_seconds < 60:
		raise RuleViolation(
			"feed_poll_interval_too_short",
			f"poll_interval_seconds {poll_interval_seconds} < 60; minimum is 60",
			"increase_poll_interval",
		)


def assert_feed_url_present_for_external(feed_type: str, url: str | None) -> None:
	"""External feeds require a URL."""
	external_types = {"stix_taxii", "misp", "csv", "json_api", "osint_scrape", "partner_share"}
	if feed_type in external_types and not (url and url.strip()):
		raise RuleViolation(
			"feed_url_required",
			f"feed_type '{feed_type}' requires a URL",
			"provide_feed_url",
		)


# ── AttributionEvidence rules ──────────────────────────────────────────────────

VALID_EVIDENCE_TYPES = {
	"technical_indicator", "behavioural_pattern", "infrastructure_overlap",
	"malware_family", "ttps_match", "victim_profile", "geolocation",
	"language_artefact", "operational_tempo", "sigint", "humint",
}


def assert_evidence_type_valid(evidence_type: str) -> None:
	if evidence_type not in VALID_EVIDENCE_TYPES:
		raise RuleViolation(
			"invalid_evidence_type",
			f"evidence_type '{evidence_type}' not recognised",
			"set_valid_evidence_type",
		)


def assert_evidence_analyst_present(analyst_id: str) -> None:
	if not analyst_id or not analyst_id.strip():
		raise RuleViolation(
			"evidence_analyst_required",
			"Attribution evidence must be assigned to an analyst",
			"provide_analyst_id",
		)


# ── IntelRequirement rules ─────────────────────────────────────────────────────

VALID_REQUIREMENT_STATUSES = {"open", "in_progress", "satisfied", "closed"}


def assert_requirement_status_transition(
	current: str,
	new: str,
) -> None:
	"""Enforce valid status transitions for intelligence requirements."""
	allowed: dict[str, set[str]] = {
		"open": {"in_progress", "closed"},
		"in_progress": {"satisfied", "closed", "open"},
		"satisfied": {"closed"},
		"closed": set(),
	}
	if new not in allowed.get(current, set()):
		raise RuleViolation(
			"invalid_requirement_status_transition",
			f"Cannot transition requirement from '{current}' to '{new}'",
			f"valid_transitions_from_{current}: {allowed.get(current, set())}",
		)


# ── STIX / TAXII rules ─────────────────────────────────────────────────────────

def assert_stix_bundle_valid(bundle: dict[str, Any]) -> None:
	"""STIX bundle must have type='bundle' and spec_version='2.1'."""
	if bundle.get("type") != "bundle":
		raise RuleViolation(
			"invalid_stix_bundle_type",
			f"Expected STIX type 'bundle', got '{bundle.get('type')}'",
			"provide_stix_2_1_bundle",
		)
	spec = bundle.get("spec_version", "")
	if spec not in ("2.0", "2.1"):
		raise RuleViolation(
			"invalid_stix_spec_version",
			f"STIX spec_version '{spec}' not supported; use 2.0 or 2.1",
			"use_stix_2_1",
		)


def assert_taxii_url_present(taxii_server_url: str) -> None:
	if not taxii_server_url or not taxii_server_url.strip():
		raise RuleViolation(
			"taxii_server_url_required",
			"TAXII server URL must be provided",
			"provide_taxii_server_url",
		)


def assert_mitre_technique_format(technique_id: str) -> None:
	"""MITRE ATT&CK technique IDs must follow T####[.###] format."""
	import re
	if not re.match(r"^T\d{4}(\.\d{3})?$", technique_id):
		raise RuleViolation(
			"invalid_mitre_technique_id",
			f"'{technique_id}' does not match ATT&CK format T####[.###]",
			"use_valid_mitre_technique_id",
		)


# ── Composite assertion helpers ────────────────────────────────────────────────

def assert_indicator_create_valid(
	tenant_id: str,
	indicator_type: str,
	value: str,
	confidence_score: float,
	tlp: str,
	valid_until: datetime | None,
) -> None:
	"""Run all create-time assertions for a ThreatIndicator in one call."""
	assert_tenant_context({"tenant_id": tenant_id})
	assert_indicator_type_valid(indicator_type)
	assert_indicator_value_present(value)
	assert_confidence_in_range(confidence_score)
	assert_tlp_valid(tlp)
	assert_indicator_not_expired(valid_until)


def assert_actor_create_valid(
	tenant_id: str,
	actor_type: str,
	confidence_score: float,
	evidence_reference: str,
) -> None:
	"""Run all create-time assertions for a ThreatActor."""
	assert_tenant_context({"tenant_id": tenant_id})
	assert_actor_type_valid(actor_type)
	assert_confidence_in_range(confidence_score)
	assert_actor_has_evidence(evidence_reference)


def assert_campaign_create_valid(
	tenant_id: str,
	campaign_type: str,
	risk_level: str,
	actor_id: str,
	first_seen: datetime | None,
	last_seen: datetime | None,
) -> None:
	"""Run all create-time assertions for a ThreatCampaign."""
	assert_tenant_context({"tenant_id": tenant_id})
	assert_campaign_type_valid(campaign_type)
	assert_risk_level_valid(risk_level)
	assert_campaign_actor_present(actor_id)
	assert_campaign_date_order(first_seen, last_seen)


def assert_report_create_valid(
	tenant_id: str,
	report_type: str,
	classification: str,
	tlp: str,
	approval_reference: str,
) -> None:
	"""Run all create-time assertions for a ThreatReport."""
	assert_tenant_context({"tenant_id": tenant_id})
	assert_report_type_valid(report_type)
	assert_classification_valid(classification)
	assert_tlp_valid(tlp)
	if not approval_reference or not approval_reference.strip():
		raise RuleViolation(
			"report_approval_reference_required",
			"Report must cite an approval reference",
			"provide_approval_reference",
		)
