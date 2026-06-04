"""Deterministic domain rules for Electronic Medical Records.

Every rule is a pure function: no I/O, no side-effects.
Callers that need a hard stop raise RuleViolation directly.
The capability contract engine calls evaluate_capability_rules() for
policy-level decisions; individual service methods call assert_* guards.
"""
from __future__ import annotations

from datetime import date
from typing import Any


# ── exception ─────────────────────────────────────────────────────────────────

class RuleViolation(Exception):
	"""Raised when a business rule is violated."""

	def __init__(self, rule_name: str, reason: str, required_action: str = "") -> None:
		self.rule_name = rule_name
		self.reason = reason
		self.required_action = required_action
		super().__init__(f"Rule '{rule_name}' violated: {reason}")


# ── tenant / access ───────────────────────────────────────────────────────────

def assert_tenant_context(context: dict[str, Any]) -> None:
	"""All operations require a non-empty tenant_id."""
	if not context.get("tenant_id") and not context.get("tenant_context_present"):
		raise RuleViolation(
			"tenant_context_required",
			"tenant_id is required for all EMR operations",
			"attach_tenant_context",
		)


def assert_no_cross_tenant_access(actor_tenant: str, resource_tenant: str) -> None:
	"""Cross-tenant PHI access is always denied."""
	if actor_tenant != resource_tenant:
		raise RuleViolation(
			"cross_tenant_access_denied",
			f"actor tenant '{actor_tenant}' may not access resource of tenant '{resource_tenant}'",
			"use_own_tenant_resources",
		)


def assert_write_policy(context: dict[str, Any]) -> None:
	"""Write operations require an attached policy."""
	if context.get("operation_type") == "write" and not context.get("policy_attached"):
		raise RuleViolation(
			"write_requires_policy",
			"write operations require an attached access policy",
			"attach_write_policy",
		)


def assert_phi_consent(operation: str, phi_consent_present: bool) -> None:
	"""FHIR / data export requires explicit PHI consent."""
	if operation == "fhir_export" and not phi_consent_present:
		raise RuleViolation(
			"fhir_export_requires_phi_consent",
			"PHI consent must be present before exporting FHIR data",
			"obtain_phi_consent",
		)


# ── patient ───────────────────────────────────────────────────────────────────

def assert_patient_not_deceased(is_deceased: bool, operation: str = "") -> None:
	"""Deceased patient charts are locked for updates."""
	if is_deceased:
		raise RuleViolation(
			"deceased_record_locked",
			f"Operation '{operation}' cannot modify a deceased patient's chart",
			"use_amendment_workflow",
		)


def assert_minor_has_guardian_consent(
	is_minor: bool,
	guardian_consent_present: bool,
	operation: str = "",
) -> None:
	"""Non-emergency procedures on minors require guardian consent."""
	if is_minor and not guardian_consent_present:
		raise RuleViolation(
			"minor_requires_guardian_consent",
			f"Operation '{operation}' on a minor requires documented guardian consent",
			"record_guardian_consent",
		)


def assert_mental_health_access(
	is_mental_health_record: bool,
	actor_has_mh_clearance: bool,
) -> None:
	"""Mental health records require enhanced access clearance."""
	if is_mental_health_record and not actor_has_mh_clearance:
		raise RuleViolation(
			"mental_health_record_restricted",
			"Access to mental health records requires enhanced confidentiality clearance",
			"request_mental_health_access",
		)


# ── encounter ─────────────────────────────────────────────────────────────────

def assert_encounter_status_supported(status: str, supported: list[str]) -> None:
	if status not in supported:
		raise RuleViolation(
			"encounter_status_not_supported",
			f"Encounter status '{status}' is not in the supported list",
			"select_supported_encounter_status",
		)


def assert_encounter_is_open(status: str) -> None:
	"""Most clinical documentation requires an open (in-progress) encounter."""
	if status not in ("in_progress", "arrived", "triaged"):
		raise RuleViolation(
			"encounter_not_open",
			f"Encounter status '{status}' is not open; clinical documentation requires an open encounter",
			"open_encounter_first",
		)


# ── notes ─────────────────────────────────────────────────────────────────────

def assert_note_type_supported(note_type: str, supported: list[str]) -> None:
	if note_type not in supported:
		raise RuleViolation(
			"note_type_not_supported",
			f"Note type '{note_type}' is not supported",
			"select_supported_note_type",
		)


def assert_note_not_final(status: str, note_id: str) -> None:
	"""Final notes cannot be edited — only addenda are permitted."""
	if status == "final":
		raise RuleViolation(
			"final_note_immutable",
			f"Note '{note_id}' is already finalised; use addendum for corrections",
			"create_addendum",
		)


def assert_note_content_not_empty(content: str) -> None:
	if not content.strip():
		raise RuleViolation(
			"note_content_empty",
			"Clinical note content must not be empty",
			"provide_note_content",
		)


def assert_amendment_has_original(original_present: bool) -> None:
	if not original_present:
		raise RuleViolation(
			"amendment_requires_original",
			"Note amendment requires the original note to exist",
			"reference_original_note",
		)


# ── diagnosis / problem ───────────────────────────────────────────────────────

def assert_icd10_code_present(icd10_code: str) -> None:
	if not icd10_code.strip():
		raise RuleViolation(
			"icd10_code_required",
			"A valid ICD-10 code is required for problem/diagnosis entries",
			"assign_icd10_code",
		)


def calculate_diagnosis_priority(is_primary: bool, certainty: str) -> int:
	"""Lower number = higher priority in problem list sorting."""
	base = 0 if is_primary else 10
	certainty_offset = {"confirmed": 0, "provisional": 1, "differential": 2, "refuted": 5}.get(certainty, 3)
	return base + certainty_offset


# ── medications / prescribing ─────────────────────────────────────────────────

def assert_allergy_check_performed(performed: bool) -> None:
	if not performed:
		raise RuleViolation(
			"allergy_check_required",
			"Allergy check must be performed before prescribing any medication",
			"perform_allergy_check",
		)


def assert_interaction_check_performed(performed: bool) -> None:
	if not performed:
		raise RuleViolation(
			"interaction_check_required",
			"Drug–drug interaction check must be performed before prescribing",
			"perform_interaction_check",
		)


def assert_no_hard_stop_allergy(hard_stop: bool, drug_name: str, patient_id: str) -> None:
	"""Life-threatening allergy is a prescribing hard stop."""
	if hard_stop:
		raise RuleViolation(
			"life_threatening_allergy_hard_stop",
			f"Patient '{patient_id}' has a life-threatening allergy to '{drug_name}'",
			"select_alternative_medication",
		)


def assert_no_contraindicated_ddi(contraindicated: bool, drug_a: str, drug_b: str) -> None:
	"""Contraindicated drug–drug interaction is a prescribing hard stop."""
	if contraindicated:
		raise RuleViolation(
			"contraindicated_ddi_hard_stop",
			f"Contraindicated interaction between '{drug_a}' and '{drug_b}'",
			"select_alternative_medication",
		)


def assert_controlled_substance_has_schedule(is_controlled: bool, dea_schedule: str | None) -> None:
	if is_controlled and not dea_schedule:
		raise RuleViolation(
			"controlled_substance_requires_schedule",
			"Controlled substance prescriptions must specify a DEA schedule",
			"specify_dea_schedule",
		)


def assert_controlled_quantity_within_cap(quantity: int, schedule: str) -> None:
	"""Schedule II: 30-day cap. Schedule III-V: 90-day cap."""
	cap = 30 if schedule == "II" else 90
	if quantity > cap:
		raise RuleViolation(
			"controlled_substance_quantity_exceeded",
			f"Quantity {quantity} exceeds the {cap}-day cap for Schedule {schedule}",
			"reduce_quantity_to_schedule_cap",
		)


def assert_pregnancy_safe(category: str, drug_name: str) -> None:
	"""Category X drugs are a prescribing hard stop in pregnancy."""
	if category == "X":
		raise RuleViolation(
			"pregnancy_category_x_contraindicated",
			f"Drug '{drug_name}' is FDA Category X — contraindicated in pregnancy",
			"select_pregnancy_safe_alternative",
		)


def assert_renal_not_contraindicated(contraindicated: bool, drug_name: str, egfr: float) -> None:
	if contraindicated:
		raise RuleViolation(
			"renal_contraindication",
			f"Drug '{drug_name}' is contraindicated at eGFR {egfr:.1f} mL/min",
			"select_renal_safe_alternative",
		)


def calculate_refills_remaining(refills_allowed: int, refills_used: int) -> int:
	return max(0, refills_allowed - refills_used)


def assert_refills_available(refills_allowed: int, refills_used: int, requested: int) -> None:
	remaining = calculate_refills_remaining(refills_allowed, refills_used)
	if requested > remaining:
		raise RuleViolation(
			"insufficient_refills",
			f"Requested {requested} refill(s) but only {remaining} remain(s)",
			"check_refill_count",
		)


# ── lab / imaging ─────────────────────────────────────────────────────────────

def assert_lab_order_not_cancelled(status: str, order_id: str) -> None:
	if status == "cancelled":
		raise RuleViolation(
			"cancelled_lab_order",
			f"Lab order '{order_id}' is cancelled and cannot be updated",
			"create_new_lab_order",
		)


def assert_critical_lab_notified(critical: bool, notified: bool, result_id: str) -> None:
	"""Critical lab values must be communicated within the episode."""
	if critical and not notified:
		raise RuleViolation(
			"critical_lab_not_notified",
			f"Critical lab result '{result_id}' has not been notified to the responsible clinician",
			"notify_critical_lab_result",
		)


# ── vitals ────────────────────────────────────────────────────────────────────

def assert_vital_value_positive(value: float, vital_type: str) -> None:
	if value < 0:
		raise RuleViolation(
			"negative_vital_value",
			f"Vital '{vital_type}' cannot have a negative value",
			"correct_vital_value",
		)


def assert_vital_type_supported(vital_type: str, supported: list[str]) -> None:
	if vital_type not in supported:
		raise RuleViolation(
			"vital_type_not_supported",
			f"Vital type '{vital_type}' is not in the supported list",
			"select_supported_vital_type",
		)


# ── allergies ─────────────────────────────────────────────────────────────────

def assert_allergy_type_supported(allergy_type: str, supported: list[str]) -> None:
	if allergy_type not in supported:
		raise RuleViolation(
			"allergy_type_not_supported",
			f"Allergy type '{allergy_type}' is not supported",
			"select_supported_allergy_type",
		)


def assert_allergy_severity_supported(severity: str, supported: list[str]) -> None:
	if severity not in supported:
		raise RuleViolation(
			"allergy_severity_not_supported",
			f"Allergy severity '{severity}' is not supported",
			"select_supported_allergy_severity",
		)


# ── consent ───────────────────────────────────────────────────────────────────

def assert_consent_not_expired(valid_until: str, now_iso: str) -> None:
	"""Expired consents must be renewed before relying on them."""
	if valid_until and valid_until != "guardian_discretion" and valid_until != "immediate_episode_only":
		if valid_until < now_iso:
			raise RuleViolation(
				"consent_expired",
				f"Consent expired on {valid_until}; renewal required",
				"renew_consent",
			)


def assert_consent_present(consent_present: bool, consent_type: str) -> None:
	if not consent_present:
		raise RuleViolation(
			"consent_required",
			f"Active consent of type '{consent_type}' is required for this operation",
			"obtain_consent",
		)


# ── care plan ─────────────────────────────────────────────────────────────────

def assert_care_plan_has_goal(goal: str) -> None:
	if not goal.strip():
		raise RuleViolation(
			"care_plan_requires_goal",
			"A care plan must have at least one documented goal",
			"document_care_plan_goal",
		)


def assert_care_plan_not_revoked(status: str, plan_id: str) -> None:
	if status in ("cancelled", "revoked"):
		raise RuleViolation(
			"care_plan_not_active",
			f"Care plan '{plan_id}' is {status} and cannot be modified",
			"create_new_care_plan",
		)


# ── referral ──────────────────────────────────────────────────────────────────

def assert_referral_urgency_valid(urgency: str) -> None:
	valid = ("routine", "urgent", "emergent")
	if urgency not in valid:
		raise RuleViolation(
			"invalid_referral_urgency",
			f"Referral urgency '{urgency}' must be one of {valid}",
			"set_valid_referral_urgency",
		)


# ── immunisation ──────────────────────────────────────────────────────────────

def assert_immunisation_not_expired(expiration_date: date | None, administered_date: date) -> None:
	if expiration_date and administered_date > expiration_date:
		raise RuleViolation(
			"expired_vaccine_administered",
			f"Vaccine expired on {expiration_date} but administered on {administered_date}",
			"use_non_expired_vaccine",
		)


# ── paediatric dosing ─────────────────────────────────────────────────────────

def calculate_adjusted_paediatric_dose(
	drug_dose_per_kg: float,
	weight_kg: float,
	absolute_max_mg: float,
) -> float:
	"""Return the weight-adjusted dose, capped at the absolute maximum."""
	raw = drug_dose_per_kg * weight_kg
	return round(min(raw, absolute_max_mg), 2)


def assert_paediatric_dose_in_range(
	prescribed_mg: float,
	min_mg: float,
	max_mg: float,
	drug: str,
) -> None:
	if prescribed_mg < min_mg:
		raise RuleViolation(
			"paediatric_underdose",
			f"Prescribed {prescribed_mg:.1f} mg of '{drug}' is below the minimum {min_mg:.1f} mg",
			"increase_dose_to_minimum",
		)
	if prescribed_mg > max_mg:
		raise RuleViolation(
			"paediatric_overdose",
			f"Prescribed {prescribed_mg:.1f} mg of '{drug}' exceeds the maximum {max_mg:.1f} mg",
			"reduce_dose_to_maximum",
		)


# ── deduplication ─────────────────────────────────────────────────────────────

def assert_no_certain_duplicate(match_score: float, threshold: float = 0.85) -> None:
	"""A near-certain duplicate match must be reviewed before creating a new record."""
	if match_score >= threshold:
		raise RuleViolation(
			"probable_duplicate_patient",
			f"Probabilistic match score {match_score:.2f} meets or exceeds threshold {threshold:.2f}",
			"review_duplicate_candidates",
		)


def is_probable_duplicate(match_score: float, soft_threshold: float = 0.60) -> bool:
	"""Return True if the score warrants a soft (non-blocking) duplicate warning."""
	return match_score >= soft_threshold


# ── agent governance ──────────────────────────────────────────────────────────

def assert_agent_action_approved(
	is_privileged: bool,
	human_approval_recorded: bool,
) -> None:
	"""Privileged agent actions require explicit human approval."""
	if is_privileged and not human_approval_recorded:
		raise RuleViolation(
			"agent_privileged_action_requires_approval",
			"This agent action is privileged and requires recorded human approval",
			"record_human_approval",
		)
