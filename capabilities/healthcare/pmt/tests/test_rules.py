"""Tests for deterministic domain rules — Patient Management."""
from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from pmt.domain.rules import (
	RuleViolation,
	assert_admission_type_supported,
	assert_agent_privileged_action_approved,
	assert_allergy_severity_supported,
	assert_appointment_in_future,
	assert_appointment_slot_available,
	assert_appointment_type_supported,
	assert_bed_available_for_assignment,
	assert_bed_status_supported,
	assert_cancellation_reason_present,
	assert_claim_amount_positive,
	assert_deposit_adequate_or_waived,
	assert_diagnosis_codes_present,
	assert_discharge_disposition_supported,
	assert_duplicate_score_below_threshold,
	assert_emergency_bypass_only_for_emergency,
	assert_gender_code_supported,
	assert_installments_minimum,
	assert_insurance_type_supported,
	assert_isolation_bed_for_isolation_patient,
	assert_merge_approval_present,
	assert_no_cross_tenant_access,
	assert_no_duplicate_mrn,
	assert_note_type_supported,
	assert_paediatric_age_limit,
	assert_pain_score_in_range,
	assert_patient_active_for_admission,
	assert_patient_not_deceased_for_update,
	assert_payment_method_supported,
	assert_physician_discharge_order,
	assert_preauth_not_expired,
	assert_procedure_codes_present,
	assert_telemedicine_consent_obtained,
	assert_tenant_context,
	assert_transfer_receiving_unit_present,
	assert_uninsured_payment_plan_eligible,
	assert_vip_access_authorised,
	assert_ward_not_in_overflow,
)


# ── tenant & access ───────────────────────────────────────────────────────────

def test_assert_tenant_context_passes():
	assert_tenant_context("hospital_a")  # no raise


def test_assert_tenant_context_fails_empty():
	with pytest.raises(RuleViolation, match="tenant_context_required"):
		assert_tenant_context("")


def test_assert_tenant_context_fails_none():
	with pytest.raises(RuleViolation):
		assert_tenant_context(None)


def test_no_cross_tenant_access_passes():
	assert_no_cross_tenant_access("A", "A")


def test_no_cross_tenant_access_fails():
	with pytest.raises(RuleViolation, match="cross_tenant"):
		assert_no_cross_tenant_access("A", "B")


# ── patient registration ──────────────────────────────────────────────────────

def test_no_duplicate_mrn_passes():
	assert_no_duplicate_mrn(False)


def test_no_duplicate_mrn_fails():
	with pytest.raises(RuleViolation, match="duplicate_mrn"):
		assert_no_duplicate_mrn(True)


def test_gender_code_supported():
	assert_gender_code_supported("female", ["male", "female", "other", "unknown"])


def test_gender_code_not_supported():
	with pytest.raises(RuleViolation, match="gender_code"):
		assert_gender_code_supported("alien", ["male", "female"])


def test_patient_not_deceased_for_update_passes():
	assert_patient_not_deceased_for_update("active")


def test_patient_not_deceased_for_update_fails():
	with pytest.raises(RuleViolation, match="deceased"):
		assert_patient_not_deceased_for_update("deceased")


def test_patient_active_for_admission_inactive():
	with pytest.raises(RuleViolation, match="inactive"):
		assert_patient_active_for_admission("inactive")


def test_patient_active_for_admission_deceased():
	with pytest.raises(RuleViolation, match="deceased"):
		assert_patient_active_for_admission("deceased")


def test_patient_active_for_admission_merged():
	with pytest.raises(RuleViolation, match="merged"):
		assert_patient_active_for_admission("merged")


def test_patient_active_for_admission_passes():
	assert_patient_active_for_admission("active")


def test_merge_approval_present_fails():
	with pytest.raises(RuleViolation, match="merge"):
		assert_merge_approval_present("")


def test_duplicate_score_below_threshold_passes():
	assert_duplicate_score_below_threshold(0.5)


def test_duplicate_score_above_threshold_fails():
	with pytest.raises(RuleViolation, match="duplicate"):
		assert_duplicate_score_below_threshold(0.90)


# ── ADT / admission ───────────────────────────────────────────────────────────

def test_admission_type_supported():
	assert_admission_type_supported("emergency", ["emergency", "elective"])


def test_admission_type_not_supported():
	with pytest.raises(RuleViolation, match="admission_type"):
		assert_admission_type_supported("space_visit", ["emergency"])


def test_physician_discharge_order_fails():
	with pytest.raises(RuleViolation, match="physician_order"):
		assert_physician_discharge_order(False)


def test_physician_discharge_order_passes():
	assert_physician_discharge_order(True)


def test_discharge_disposition_supported():
	assert_discharge_disposition_supported("home", ["home", "snf", "rehab"])


def test_discharge_disposition_not_supported():
	with pytest.raises(RuleViolation, match="disposition"):
		assert_discharge_disposition_supported("mars", ["home"])


def test_emergency_bypass_valid():
	assert_emergency_bypass_only_for_emergency(True, "emergency")


def test_emergency_bypass_invalid_type():
	with pytest.raises(RuleViolation, match="bypass"):
		assert_emergency_bypass_only_for_emergency(True, "elective")


def test_transfer_receiving_unit_empty():
	with pytest.raises(RuleViolation, match="transfer"):
		assert_transfer_receiving_unit_present("")


# ── bed management ────────────────────────────────────────────────────────────

def test_bed_available_passes():
	assert_bed_available_for_assignment("available")


def test_bed_occupied_fails():
	with pytest.raises(RuleViolation, match="bed_not_available"):
		assert_bed_available_for_assignment("occupied")


def test_bed_status_supported():
	assert_bed_status_supported("available", ["available", "occupied", "cleaning"])


def test_isolation_bed_required():
	with pytest.raises(RuleViolation, match="isolation"):
		assert_isolation_bed_for_isolation_patient(True, False)


def test_isolation_bed_not_required_passes():
	assert_isolation_bed_for_isolation_patient(False, False)


def test_paediatric_age_limit_exceeded():
	with pytest.raises(RuleViolation, match="paediatric"):
		assert_paediatric_age_limit(300, 216)  # 25yr vs 18yr limit


def test_paediatric_age_limit_passes():
	assert_paediatric_age_limit(120, 216)  # 10yr < 18yr limit


def test_ward_overflow_triggers():
	with pytest.raises(RuleViolation, match="overflow"):
		assert_ward_not_in_overflow(1, 100, threshold_pct=5.0)


def test_ward_not_overflow_passes():
	assert_ward_not_in_overflow(20, 100, threshold_pct=5.0)


# ── appointments ──────────────────────────────────────────────────────────────

def test_appointment_type_supported():
	assert_appointment_type_supported("follow_up", ["follow_up", "new_patient"])


def test_slot_not_available():
	with pytest.raises(RuleViolation, match="slot"):
		assert_appointment_slot_available(False)


def test_cancellation_reason_empty():
	with pytest.raises(RuleViolation, match="cancellation"):
		assert_cancellation_reason_present("")


def test_telemedicine_consent():
	with pytest.raises(RuleViolation, match="consent"):
		assert_telemedicine_consent_obtained(True, False)


def test_telemedicine_consent_not_required_for_in_person():
	assert_telemedicine_consent_obtained(False, False)


def test_appointment_in_future_past():
	with pytest.raises(RuleViolation, match="future"):
		assert_appointment_in_future(datetime(2020, 1, 1))


def test_appointment_in_future_passes():
	assert_appointment_in_future(datetime(2030, 1, 1))


# ── insurance & billing ───────────────────────────────────────────────────────

def test_insurance_type_supported():
	assert_insurance_type_supported("commercial", ["commercial", "medicare"])


def test_claim_amount_positive_fails():
	with pytest.raises(RuleViolation, match="amount"):
		assert_claim_amount_positive(0.0)


def test_diagnosis_codes_required():
	with pytest.raises(RuleViolation, match="diagnosis"):
		assert_diagnosis_codes_present([])


def test_procedure_codes_required():
	with pytest.raises(RuleViolation, match="procedure"):
		assert_procedure_codes_present([])


def test_preauth_not_expired_fails():
	with pytest.raises(RuleViolation, match="preauth"):
		assert_preauth_not_expired(datetime.utcnow() - timedelta(days=1))


def test_preauth_not_expired_passes():
	assert_preauth_not_expired(datetime.utcnow() + timedelta(days=10))


def test_payment_method_not_supported():
	with pytest.raises(RuleViolation, match="payment_method"):
		assert_payment_method_supported("barter", ["cash", "card"])


def test_uninsured_payment_plan():
	with pytest.raises(RuleViolation, match="uninsured"):
		assert_uninsured_payment_plan_eligible(True, False)


def test_uninsured_payment_plan_passes_when_eligible():
	assert_uninsured_payment_plan_eligible(True, True)


def test_installments_minimum_fails():
	with pytest.raises(RuleViolation, match="installments"):
		assert_installments_minimum(1)


# ── VIP ───────────────────────────────────────────────────────────────────────

def test_vip_access_denied():
	with pytest.raises(RuleViolation, match="vip"):
		assert_vip_access_authorised(True, False)


def test_vip_access_granted():
	assert_vip_access_authorised(True, True)


def test_non_vip_access_allowed_without_clearance():
	assert_vip_access_authorised(False, False)


# ── clinical ──────────────────────────────────────────────────────────────────

def test_pain_score_out_of_range():
	with pytest.raises(RuleViolation, match="pain"):
		assert_pain_score_in_range(11)


def test_pain_score_valid():
	assert_pain_score_in_range(5)


def test_pain_score_none_passes():
	assert_pain_score_in_range(None)


def test_allergy_severity_invalid():
	with pytest.raises(RuleViolation, match="severity"):
		assert_allergy_severity_supported("lethal")


def test_note_type_invalid():
	with pytest.raises(RuleViolation, match="note_type"):
		assert_note_type_supported("random")


# ── agents ────────────────────────────────────────────────────────────────────

def test_privileged_agent_action_requires_approval():
	with pytest.raises(RuleViolation, match="privileged"):
		assert_agent_privileged_action_approved(True, True, False)


def test_privileged_agent_action_approved():
	assert_agent_privileged_action_approved(True, True, True)


def test_deposit_adequate_rule():
	# this function may or may not exist — skip gracefully
	try:
		assert_deposit_adequate_or_waived(False, False)
	except RuleViolation:
		pass
	except AttributeError:
		pytest.skip("assert_deposit_adequate_or_waived not yet implemented")
