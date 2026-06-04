"""Deterministic domain rules for APG Patient Management.

Single source of truth for all governance decisions. Every rule is a pure
function — no I/O, no side effects. Raise RuleViolation for violations;
return normally for passes.
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any


class RuleViolation(Exception):
	"""Raised when a business rule is violated."""

	def __init__(self, rule_name: str, reason: str, required_action: str = "") -> None:
		self.rule_name = rule_name
		self.reason = reason
		self.required_action = required_action
		super().__init__(f"Rule '{rule_name}' violated: {reason}")


# ── tenant & access ────────────────────────────────────────────────────────────

def assert_tenant_context(tenant_id: str | None) -> None:
	"""All operations require a non-empty tenant_id."""
	if not tenant_id or not str(tenant_id).strip():
		raise RuleViolation(
			"tenant_context_required",
			"tenant_id is required for all operations",
			"attach_tenant_context",
		)


def assert_no_cross_tenant_access(actor_tenant: str, resource_tenant: str) -> None:
	"""Cross-tenant patient data access is never permitted."""
	if actor_tenant != resource_tenant:
		raise RuleViolation(
			"cross_tenant_access_denied",
			f"actor tenant '{actor_tenant}' cannot access resources of tenant '{resource_tenant}'",
			"use_own_tenant_resources",
		)


def assert_write_policy(operation_type: str, policy_attached: bool) -> None:
	"""All write operations must have an attached policy token."""
	if operation_type == "write" and not policy_attached:
		raise RuleViolation(
			"write_requires_policy",
			"write operations require an attached policy",
			"attach_write_policy",
		)


# ── patient registration ───────────────────────────────────────────────────────

def assert_no_duplicate_mrn(mrn_exists: bool) -> None:
	if mrn_exists:
		raise RuleViolation(
			"duplicate_mrn_denied",
			"an active record with this MRN already exists",
			"merge_or_use_existing_patient",
		)


def assert_gender_code_supported(gender_code: str, supported: list[str]) -> None:
	if gender_code not in supported:
		raise RuleViolation(
			"gender_code_not_supported",
			f"gender code '{gender_code}' is not in supported set {supported}",
			"select_supported_gender_code",
		)


def assert_patient_not_deceased_for_update(status: str) -> None:
	if status == "deceased":
		raise RuleViolation(
			"deceased_patient_modification_denied",
			"deceased patient records are locked; use the amendment workflow",
			"use_amendment_workflow",
		)


def assert_patient_active_for_admission(status: str) -> None:
	if status == "inactive":
		raise RuleViolation(
			"inactive_patient_adt_denied",
			"inactive patients cannot be admitted; reactivate first",
			"reactivate_patient",
		)
	if status == "deceased":
		raise RuleViolation(
			"deceased_patient_adt_denied",
			"deceased patients cannot be admitted",
			"verify_patient_identity",
		)
	if status == "merged":
		raise RuleViolation(
			"merged_patient_adt_denied",
			"merged patient records cannot be admitted; use the target patient record",
			"use_target_patient_record",
		)


def assert_merge_approval_present(approved_by: str) -> None:
	if not approved_by or not approved_by.strip():
		raise RuleViolation(
			"patient_merge_requires_approval",
			"patient merge requires an approving clinician or supervisor",
			"obtain_merge_approval",
		)


def assert_duplicate_score_below_threshold(score: float, threshold: float = 0.85) -> None:
	if score >= threshold:
		raise RuleViolation(
			"duplicate_patient_risk_high",
			f"duplicate match score {score:.2f} exceeds threshold {threshold}; verify identity",
			"verify_or_link_existing_patient",
		)


# ── admission / ADT ────────────────────────────────────────────────────────────

def assert_admission_type_supported(admission_type: str, supported: list[str]) -> None:
	if admission_type not in supported:
		raise RuleViolation(
			"admission_type_not_supported",
			f"admission type '{admission_type}' is not supported",
			"select_supported_admission_type",
		)


def assert_physician_discharge_order(physician_order_present: bool) -> None:
	if not physician_order_present:
		raise RuleViolation(
			"discharge_requires_physician_order",
			"patient discharge requires a signed physician order",
			"obtain_physician_discharge_order",
		)


def assert_discharge_disposition_supported(disposition: str, supported: list[str]) -> None:
	if disposition not in supported:
		raise RuleViolation(
			"discharge_disposition_not_supported",
			f"discharge disposition '{disposition}' is not supported",
			"select_supported_disposition",
		)


def assert_emergency_bypass_only_for_emergency(
	bypass: bool, admission_type: str
) -> None:
	if bypass and admission_type not in ("emergency", "trauma"):
		raise RuleViolation(
			"emergency_bypass_invalid_type",
			"emergency_bypass_registration is only valid for emergency and trauma admissions",
			"select_emergency_or_trauma_admission_type",
		)


def assert_transfer_receiving_unit_present(receiving_unit: str | None) -> None:
	if not receiving_unit or not receiving_unit.strip():
		raise RuleViolation(
			"transfer_requires_receiving_unit",
			"patient transfer requires a specified receiving unit",
			"specify_receiving_unit",
		)


def assert_adt_event_type_supported(event_type: str, supported: list[str]) -> None:
	if event_type not in supported:
		raise RuleViolation(
			"adt_event_type_not_supported",
			f"ADT event type '{event_type}' is not supported",
			"select_supported_adt_event_type",
		)


# ── bed management ─────────────────────────────────────────────────────────────

def assert_bed_available_for_assignment(bed_status: str) -> None:
	if bed_status not in ("available",):
		raise RuleViolation(
			"bed_not_available_for_assignment",
			f"bed in status '{bed_status}' cannot be assigned; only 'available' beds may be assigned",
			"select_available_bed",
		)


def assert_bed_status_supported(status: str, supported: list[str]) -> None:
	if status not in supported:
		raise RuleViolation(
			"bed_status_not_supported",
			f"bed status '{status}' is not in supported set {supported}",
			"select_supported_bed_status",
		)


def assert_isolation_bed_for_isolation_patient(
	isolation_required: bool, bed_isolation_capable: bool
) -> None:
	if isolation_required and not bed_isolation_capable:
		raise RuleViolation(
			"isolation_bed_required",
			"patient requires isolation; selected bed is not isolation-capable",
			"select_isolation_capable_bed",
		)


def assert_paediatric_age_limit(
	patient_age_months: int, bed_max_age_months: int | None
) -> None:
	if bed_max_age_months is not None and patient_age_months > bed_max_age_months:
		raise RuleViolation(
			"paediatric_age_limit_exceeded",
			(
				f"patient age {patient_age_months} months exceeds paediatric bed"
				f" maximum of {bed_max_age_months} months"
			),
			"assign_adult_bed",
		)


def assert_ward_not_in_overflow(available: int, total: int, threshold_pct: float = 5.0) -> None:
	if total > 0 and (available / total * 100) < threshold_pct:
		raise RuleViolation(
			"ward_overflow_risk",
			f"ward occupancy critical: only {available}/{total} beds available ({available/total*100:.1f}%)",
			"activate_overflow_protocol",
		)


# ── appointments ───────────────────────────────────────────────────────────────

def assert_appointment_type_supported(apt_type: str, supported: list[str]) -> None:
	if apt_type not in supported:
		raise RuleViolation(
			"appointment_type_not_supported",
			f"appointment type '{apt_type}' is not supported",
			"select_supported_appointment_type",
		)


def assert_appointment_slot_available(slot_available: bool) -> None:
	if not slot_available:
		raise RuleViolation(
			"appointment_slot_not_available",
			"the requested appointment slot is not available",
			"select_available_slot",
		)


def assert_cancellation_reason_present(reason: str | None) -> None:
	if not reason or not reason.strip():
		raise RuleViolation(
			"cancellation_reason_required",
			"appointment cancellation requires a stated reason",
			"provide_cancellation_reason",
		)


def assert_telemedicine_consent_obtained(telemedicine: bool, consent: bool) -> None:
	if telemedicine and not consent:
		raise RuleViolation(
			"telemedicine_consent_required",
			"telemedicine booking requires patient consent to be explicitly recorded",
			"obtain_telemedicine_consent",
		)


def assert_appointment_in_future(scheduled_at: datetime) -> None:
	if scheduled_at <= datetime.utcnow():
		raise RuleViolation(
			"appointment_must_be_future",
			"appointments must be scheduled for a future date and time",
			"select_future_datetime",
		)


# ── insurance & billing ────────────────────────────────────────────────────────

def assert_insurance_type_supported(ins_type: str, supported: list[str]) -> None:
	if ins_type not in supported:
		raise RuleViolation(
			"insurance_type_not_supported",
			f"insurance type '{ins_type}' is not supported",
			"select_supported_insurance_type",
		)


def assert_claim_amount_positive(amount: float | Decimal) -> None:
	if float(amount) <= 0:
		raise RuleViolation(
			"claim_amount_must_be_positive",
			f"claim amount {amount} must be positive",
			"provide_positive_claim_amount",
		)


def assert_diagnosis_codes_present(codes: list[str]) -> None:
	if not codes:
		raise RuleViolation(
			"diagnosis_codes_required",
			"at least one ICD-10 diagnosis code is required for claim submission",
			"provide_diagnosis_codes",
		)


def assert_procedure_codes_present(codes: list[str]) -> None:
	if not codes:
		raise RuleViolation(
			"procedure_codes_required",
			"at least one CPT/procedure code is required for claim submission",
			"provide_procedure_codes",
		)


def assert_preauth_not_expired(expires_at: datetime) -> None:
	if datetime.utcnow() > expires_at:
		raise RuleViolation(
			"preauth_expired",
			f"pre-authorisation expired on {expires_at.isoformat()}",
			"resubmit_preauth_request",
		)


def assert_payment_method_supported(method: str, supported: list[str]) -> None:
	if method not in supported:
		raise RuleViolation(
			"payment_method_not_supported",
			f"payment method '{method}' is not supported",
			"select_supported_payment_method",
		)


def assert_uninsured_payment_plan_eligible(
	is_uninsured: bool, payment_plan_eligible: bool
) -> None:
	"""Uninsured patients must be offered a payment plan before billing is finalised."""
	if is_uninsured and not payment_plan_eligible:
		raise RuleViolation(
			"uninsured_patient_must_have_payment_plan",
			"uninsured patients must be offered a payment plan before bill finalisation",
			"create_payment_plan_for_uninsured_patient",
		)


def assert_installments_minimum(installments: int) -> None:
	if installments < 2:
		raise RuleViolation(
			"payment_plan_minimum_installments",
			"payment plans must have at least 2 installments",
			"increase_installment_count",
		)


# ── VIP privacy ────────────────────────────────────────────────────────────────

def assert_vip_access_authorised(patient_vip: bool, actor_has_vip_clearance: bool) -> None:
	if patient_vip and not actor_has_vip_clearance:
		raise RuleViolation(
			"vip_patient_privacy_restriction",
			"access to VIP patient records requires VIP clearance",
			"request_vip_access_clearance",
		)


# ── clinical safety ────────────────────────────────────────────────────────────

def assert_pain_score_in_range(score: int | None) -> None:
	if score is not None and not (0 <= score <= 10):
		raise RuleViolation(
			"pain_score_out_of_range",
			f"pain score {score} must be in range 0–10",
			"correct_pain_score",
		)


def assert_allergy_severity_supported(severity: str) -> None:
	valid = {"mild", "moderate", "severe", "life_threatening"}
	if severity not in valid:
		raise RuleViolation(
			"allergy_severity_not_supported",
			f"allergy severity '{severity}' is not in supported set {sorted(valid)}",
			"select_supported_severity",
		)


def assert_note_type_supported(note_type: str) -> None:
	valid = {"soap", "progress", "discharge", "referral", "nursing", "procedure"}
	if note_type not in valid:
		raise RuleViolation(
			"note_type_not_supported",
			f"clinical note type '{note_type}' is not in supported set {sorted(valid)}",
			"select_supported_note_type",
		)


# ── agents ─────────────────────────────────────────────────────────────────────

def assert_agent_privileged_action_approved(
	agent_action: bool, privileged: bool, human_approval: bool
) -> None:
	if agent_action and privileged and not human_approval:
		raise RuleViolation(
			"privileged_agent_action_requires_human_approval",
			"privileged agent actions require recorded human approval",
			"record_human_approval",
		)


def assert_deposit_adequate_or_waived(adequate: bool, waived: bool) -> None:
	"""Admission deposit must be adequate or explicitly waived by finance."""
	if not adequate and not waived:
		raise RuleViolation(
			"deposit_required_before_admission",
			"admission deposit is insufficient; collect deposit or record a finance waiver",
			"collect_deposit_or_record_waiver",
		)
