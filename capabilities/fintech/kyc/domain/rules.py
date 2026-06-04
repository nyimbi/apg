"""Deterministic domain rules for Know Your Customer.

Every governance decision in the KYC lifecycle is encoded here as a pure,
callable function.  The service layer calls these; the rule engine exposes
them to the platform policy framework.

Design principles
-----------------
- No I/O — rules are pure functions over scalar/dict arguments.
- Raise :class:`RuleViolation` on any breach; callers catch selectively.
- ``assert_*`` functions enforce pre/post-conditions.
- ``calculate_*`` functions return computed values without side-effects.
- All public names are importable from ``domain.rules`` directly.
"""
from __future__ import annotations

from datetime import date
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Exception
# ─────────────────────────────────────────────────────────────────────────────

class RuleViolation(Exception):
	"""Raised when a business rule is violated."""

	def __init__(self, rule_name: str, reason: str, required_action: str = "") -> None:
		self.rule_name = rule_name
		self.reason = reason
		self.required_action = required_action
		super().__init__(f"Rule '{rule_name}' violated: {reason}")

	def to_dict(self) -> dict[str, str]:
		return {
			"rule": self.rule_name,
			"reason": self.reason,
			"required_action": self.required_action,
		}


# ─────────────────────────────────────────────────────────────────────────────
# Cross-cutting rules
# ─────────────────────────────────────────────────────────────────────────────

def assert_tenant_context(tenant_id: str) -> None:
	"""All operations require a non-empty tenant_id."""
	if not tenant_id or not tenant_id.strip():
		raise RuleViolation(
			"tenant_context_required",
			"tenant_id is required for all KYC operations",
			"attach_tenant_context",
		)


def assert_no_cross_tenant_access(actor_tenant: str, resource_tenant: str) -> None:
	"""Cross-tenant access is always denied."""
	if actor_tenant != resource_tenant:
		raise RuleViolation(
			"cross_tenant_access_denied",
			f"actor tenant '{actor_tenant}' cannot access resource owned by '{resource_tenant}'",
			"use_own_tenant_resources",
		)


def assert_consent_recorded(consent_reference: str) -> None:
	"""Customer consent must be captured before KYC data collection begins."""
	if not consent_reference or not consent_reference.strip():
		raise RuleViolation(
			"consent_required",
			"customer consent reference must be recorded before KYC",
			"record_consent",
		)


# ─────────────────────────────────────────────────────────────────────────────
# Document rules
# ─────────────────────────────────────────────────────────────────────────────

def assert_identity_document_present(has_id: bool) -> None:
	"""An approved identity document is mandatory for KYC approval."""
	if not has_id:
		raise RuleViolation(
			"identity_document_required",
			"at least one verified identity document is required",
			"upload_identity_document",
		)


def assert_address_document_present(has_address_doc: bool) -> None:
	"""Proof of address is required for standard and enhanced KYC tiers."""
	if not has_address_doc:
		raise RuleViolation(
			"address_document_required",
			"proof of address is required",
			"upload_address_document",
		)


def assert_document_not_expired(expiry_date: date | None, document_type: str = "") -> None:
	"""Documents used for KYC must not be expired."""
	if expiry_date is not None and expiry_date < date.today():
		label = f" ({document_type})" if document_type else ""
		raise RuleViolation(
			"document_expired",
			f"document{label} expired on {expiry_date.isoformat()}",
			"upload_valid_document",
		)


def assert_no_deceased_id(is_deceased: bool) -> None:
	"""Using an identity document belonging to a deceased person is fraud."""
	if is_deceased:
		raise RuleViolation(
			"deceased_id_fraud",
			"identity document belongs to a deceased person — potential fraud",
			"escalate_to_fraud_team",
		)


def assert_no_synthetic_identity(synthetic_fraud_score: float) -> None:
	"""Synthetic identity fraud score must be below the threshold (0.6)."""
	if synthetic_fraud_score >= 0.6:
		raise RuleViolation(
			"synthetic_identity",
			f"synthetic identity indicators detected (score={synthetic_fraud_score:.3f})",
			"escalate_to_fraud_team",
		)


def assert_document_confidence(confidence: float, minimum: float = 0.75) -> None:
	"""OCR/extraction confidence must meet the minimum threshold."""
	if confidence < minimum:
		raise RuleViolation(
			"document_confidence_too_low",
			f"document confidence {confidence:.3f} below minimum {minimum}",
			"re_upload_clearer_document",
		)


# ─────────────────────────────────────────────────────────────────────────────
# Biometric rules
# ─────────────────────────────────────────────────────────────────────────────

def assert_biometric_match(match_score: float, threshold: float = 0.85) -> None:
	"""Face match / fingerprint match score must meet the minimum threshold."""
	if match_score < threshold:
		raise RuleViolation(
			"biometric_mismatch",
			f"biometric match score {match_score:.3f} below threshold {threshold}",
			"retake_biometrics",
		)


def assert_liveness_check_passed(liveness_score: float, threshold: float = 0.80) -> None:
	"""Anti-spoofing liveness score must clear the minimum threshold."""
	if liveness_score < threshold:
		raise RuleViolation(
			"liveness_check_failed",
			f"liveness score {liveness_score:.3f} below threshold {threshold}",
			"redo_liveness_check",
		)


# ─────────────────────────────────────────────────────────────────────────────
# Screening rules
# ─────────────────────────────────────────────────────────────────────────────

def assert_screening_completed(screening_done: bool) -> None:
	"""PEP and sanctions screening must be completed before approval."""
	if not screening_done:
		raise RuleViolation(
			"screening_required",
			"PEP and sanctions screening must be completed",
			"run_screening",
		)


def assert_no_unresolved_sanction(has_confirmed_hit: bool) -> None:
	"""Confirmed sanctions matches block KYC approval."""
	if has_confirmed_hit:
		raise RuleViolation(
			"active_sanction",
			"customer has a confirmed sanctions match — relationship cannot proceed",
			"resolve_sanction_match",
		)


def assert_edd_for_pep(is_pep: bool, edd_completed: bool) -> None:
	"""Enhanced due diligence is mandatory for all PEPs (FATF R.12)."""
	if is_pep and not edd_completed:
		raise RuleViolation(
			"edd_required_for_pep",
			"enhanced due diligence is required for politically exposed persons (FATF R.12)",
			"complete_edd",
		)


def assert_edd_for_high_risk(risk_score: int, edd_completed: bool) -> None:
	"""Enhanced due diligence is required when risk score ≥ 55 (FATF R.19, R.20)."""
	if risk_score >= 55 and not edd_completed:
		raise RuleViolation(
			"edd_required_high_risk",
			f"enhanced due diligence required for high-risk customer (score={risk_score})",
			"complete_edd",
		)


# ─────────────────────────────────────────────────────────────────────────────
# Risk & scoring rules
# ─────────────────────────────────────────────────────────────────────────────

def assert_risk_score_range(score: int) -> None:
	"""Risk score must be in [0, 100]."""
	if not (0 <= score <= 100):
		raise RuleViolation(
			"invalid_risk_score",
			f"risk score {score} must be between 0 and 100",
			"recalculate_risk_score",
		)


def assert_kyc_not_expired(expiry_date: date | None) -> None:
	"""KYC approval must not be past its expiry date."""
	if expiry_date is not None and expiry_date < date.today():
		raise RuleViolation(
			"kyc_expired",
			f"customer KYC expired on {expiry_date.isoformat()} — re-KYC required",
			"initiate_re_kyc",
		)


# ─────────────────────────────────────────────────────────────────────────────
# Review & approval rules
# ─────────────────────────────────────────────────────────────────────────────

def assert_no_open_reviews(open_count: int) -> None:
	"""All open KYC reviews must be closed before final approval."""
	if open_count > 0:
		raise RuleViolation(
			"open_reviews_exist",
			f"{open_count} open KYC review(s) must be resolved before approval",
			"resolve_open_reviews",
		)


def assert_ubo_declared(customer_type: str, ubo_count: int) -> None:
	"""Corporate/trust/partnership customers must declare at least one UBO."""
	corporate_types = {"business", "nonprofit", "trust", "partnership", "ngo"}
	if customer_type in corporate_types and ubo_count == 0:
		raise RuleViolation(
			"ubo_declaration_required",
			f"ultimate beneficial owners must be declared for {customer_type} customers (FATF R.24)",
			"declare_ubos",
		)


def assert_valid_ubo_ownership(total_pct: float) -> None:
	"""Declared UBO ownership percentages must not exceed 100%."""
	if total_pct > 100.0:
		raise RuleViolation(
			"ubo_ownership_exceeds_100",
			f"total declared ownership {total_pct:.2f}% exceeds 100%",
			"correct_ubo_declaration",
		)


# ─────────────────────────────────────────────────────────────────────────────
# Edge-case rules
# ─────────────────────────────────────────────────────────────────────────────

def assert_dormant_reactivation_review(is_dormant: bool, reactivation_review_done: bool) -> None:
	"""Dormant accounts must undergo re-KYC review before reactivation."""
	if is_dormant and not reactivation_review_done:
		raise RuleViolation(
			"dormant_reactivation_review_required",
			"dormant account must complete re-KYC review before reactivation",
			"initiate_dormant_reactivation_kyc",
		)


def assert_refugee_minimum_docs(is_refugee: bool, has_any_doc: bool) -> None:
	"""Refugee customers require at least one UNHCR-recognised document."""
	if is_refugee and not has_any_doc:
		raise RuleViolation(
			"refugee_minimum_doc_required",
			"refugee customers must provide at least one UNHCR-recognised identity document",
			"upload_refugee_documentation",
		)


def assert_nominee_disclosure(has_nominees: bool, nominees_disclosed: bool) -> None:
	"""Nominee shareholders must be explicitly disclosed (FATF R.24)."""
	if has_nominees and not nominees_disclosed:
		raise RuleViolation(
			"nominee_disclosure_required",
			"nominee shareholders must be disclosed and their beneficial ownership established",
			"disclose_nominee_shareholders",
		)


def assert_complex_structure_resolved(is_complex: bool, structure_verified: bool) -> None:
	"""Complex corporate ownership structures must be fully mapped and verified."""
	if is_complex and not structure_verified:
		raise RuleViolation(
			"complex_structure_unverified",
			"complex ownership structure must be fully mapped and verified before approval",
			"complete_ownership_structure_verification",
		)


def assert_name_transliteration_available(
	name_script: str,
	name_transliterated: str,
) -> None:
	"""Non-Latin names must have a Latin transliteration for screening."""
	non_latin_scripts = {"arabic", "chinese", "cyrillic", "devanagari"}
	if name_script in non_latin_scripts and not name_transliterated.strip():
		raise RuleViolation(
			"transliteration_required",
			f"name in {name_script} script must have a Latin transliteration for sanctions screening",
			"provide_name_transliteration",
		)


# ─────────────────────────────────────────────────────────────────────────────
# Calculate helpers (pure, no side-effects)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_kyc_tier_for_channel(channel: str, customer_type: str) -> str:
	"""Determine the minimum KYC tier required for a given channel and customer type.

	USSD channel (feature phones) → simplified tier (CBK Tier 1 / BoG Basic).
	Mobile / web individual → standard tier.
	Business / institutional → enhanced tier always.
	"""
	if customer_type in ("business", "nonprofit", "trust", "partnership", "ngo", "government"):
		return "enhanced"
	if channel.lower() in ("ussd", "sms"):
		return "simplified"
	return "standard"


def calculate_monitoring_frequency(risk_band: str) -> str:
	"""Return the ongoing monitoring frequency for a given risk band."""
	_MAP = {
		"low": "annual",
		"medium": "semi_annual",
		"high": "quarterly",
		"very_high": "monthly",
		"unacceptable": "blocked",
	}
	return _MAP.get(risk_band, "annual")


def calculate_edd_requirements(
	is_pep: bool,
	risk_band: str,
	customer_type: str,
	has_complex_structure: bool,
) -> list[str]:
	"""Return the list of EDD requirements applicable to a customer profile."""
	reqs: list[str] = [
		"source_of_wealth_declaration",
		"source_of_funds_evidence",
		"purpose_of_relationship_form",
	]
	if is_pep:
		reqs.append("senior_management_approval")
		reqs.append("pep_relationship_map")
	if customer_type in ("business", "trust", "partnership", "ngo"):
		reqs.append("beneficial_owner_declaration")
		reqs.append("corporate_structure_chart")
	if has_complex_structure:
		reqs.append("group_structure_diagram")
		reqs.append("ultimate_controller_declaration")
	if risk_band == "very_high":
		reqs.append("enhanced_source_of_wealth_evidence")
		reqs.append("site_visit_report")
	return reqs


def calculate_required_steps(customer_type: str, channel: str = "web") -> list[str]:
	"""Return the ordered KYC steps required for a customer type and channel."""
	individual_steps = [
		"identity_document",
		"address_document",
		"biometrics",
		"pep_screening",
		"sanctions_screening",
		"risk_assessment",
		"review",
	]
	business_steps = [
		"identity_document",
		"business_registration",
		"ubo_declaration",
		"address_document",
		"pep_screening",
		"sanctions_screening",
		"adverse_media",
		"risk_assessment",
		"review",
	]
	ussd_steps = ["identity_document", "pep_screening", "risk_assessment"]

	if channel.lower() in ("ussd", "sms"):
		return ussd_steps
	if customer_type in ("business", "nonprofit", "trust", "partnership", "ngo", "government"):
		return business_steps
	return individual_steps
