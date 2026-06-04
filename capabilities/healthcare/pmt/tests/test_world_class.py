"""Tests for world-class improvement methods — Patient Management."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import pytest

from pmt.models import AdmissionCreate, BedCreate, PatientCreate
from pmt.service import PatientManagementService, PolicyViolationError


def run(coro):
	return asyncio.get_event_loop().run_until_complete(coro)


def svc():
	return PatientManagementService()


def make_patient(s, tid="t", first="Jane", last="Smith") -> any:
	return run(s.register_patient(PatientCreate(
		tenant_id=tid, first_name=first, last_name=last,
		date_of_birth=datetime(1975, 8, 20), gender_code="female",
		phone="0700000001", created_by="test",
	)))


def make_bed(s, tid="t", unit="WARD_A", iso=False, paed=False) -> any:
	return run(s.register_bed(BedCreate(
		tenant_id=tid, unit_id=unit, bed_number=f"B{unit[-1]}01",
		bed_type="medical_surgical", location="Ground floor",
		isolation_capable=iso, paediatric_only=paed, created_by="admin",
	)))


def make_admission(s, patient, bed, tid="t") -> any:
	return run(s.admit_patient(AdmissionCreate(
		tenant_id=tid, patient_id=patient.id, admission_type="elective",
		admitting_provider_id="dr1", attending_provider_id="dr1",
		unit_id=bed.unit_id, bed_id=bed.id,
		chief_complaint="Scheduled procedure", created_by="admissions",
	)))


# ── triage_patient ─────────────────────────────────────────────────────────────

def test_triage_patient_returns_record():
	s = svc()
	p = make_patient(s)
	result = run(s.triage_patient(
		"t", p.id,
		triage_level="level_3_urgent",
		chief_complaint="Severe headache",
		vital_signs={"bp_systolic": 145, "heart_rate": 90, "spo2": 97},
		triaged_by="nurse1",
		pain_score=6,
	))
	assert result["triage_level"] == "level_3_urgent"
	assert result["pain_score"] == 6
	assert "ews_score" in result


def test_triage_invalid_level_rejected():
	s = svc()
	p = make_patient(s)
	with pytest.raises(AssertionError):
		run(s.triage_patient("t", p.id, "level_6_invalid", "X", {}, "nurse1"))


def test_triage_pain_score_out_of_range():
	s = svc()
	p = make_patient(s)
	with pytest.raises(AssertionError):
		run(s.triage_patient("t", p.id, "level_3_urgent", "X", {}, "nurse1", pain_score=11))


# ── continuous_acuity_watch ───────────────────────────────────────────────────

def test_continuous_acuity_normal_not_escalated():
	s = svc()
	p = make_patient(s)
	result = run(s.continuous_acuity_watch(
		"t", p.id, "enc1",
		vitals={"bp_systolic": 120, "respiratory_rate": 16, "spo2": 98,
				"heart_rate": 75, "temperature_c": 37.0, "avpu_score": 1.0},
		recorded_by="nurse1",
	))
	assert result["escalated"] is False
	assert result["ews_level"] == "low"


def test_continuous_acuity_critical_escalated():
	s = svc()
	p = make_patient(s)
	result = run(s.continuous_acuity_watch(
		"t", p.id, "enc1",
		vitals={"bp_systolic": 80, "respiratory_rate": 30, "spo2": 88,
				"heart_rate": 140, "temperature_c": 39.5, "avpu_score": 0.0},
		recorded_by="nurse1",
	))
	assert result["escalated"] is True
	assert result["ews_level"] in ("high", "critical")


# ── portal_self_triage ────────────────────────────────────────────────────────

def test_self_triage_red_flag_to_ed():
	s = svc()
	p = make_patient(s)
	result = run(s.portal_self_triage("t", p.id, {
		"chest_pain": True, "fever": False,
	}))
	assert result["care_level"] == "emergency_department"
	assert result["urgency"] == "go_now"
	assert "chest_pain" in result["red_flags"]


def test_self_triage_multiple_symptoms_urgent_care():
	s = svc()
	p = make_patient(s)
	result = run(s.portal_self_triage("t", p.id, {
		"fever": True, "fatigue": True, "nausea": True,
	}))
	assert result["care_level"] == "urgent_care"


def test_self_triage_minor_primary_care():
	s = svc()
	p = make_patient(s)
	result = run(s.portal_self_triage("t", p.id, {"mild_cough": True}))
	assert result["care_level"] == "primary_care"


def test_self_triage_empty_symptoms_raises():
	s = svc()
	p = make_patient(s)
	with pytest.raises(AssertionError):
		run(s.portal_self_triage("t", p.id, {}))


# ── waitlist operations ───────────────────────────────────────────────────────

def test_add_to_waitlist():
	s = svc()
	p = make_patient(s)
	entry = run(s.add_to_waitlist("t", p.id, "WARD_A", priority="urgent"))
	assert entry["status"] == "waiting"
	assert entry["priority"] == "urgent"


def test_waitlist_emergency_scores_higher():
	s = svc()
	p1 = make_patient(s, first="Alice", last="Alpha")
	p2 = make_patient(s, first="Bob", last="Beta")
	run(s.add_to_waitlist("t", p1.id, "WARD_A", priority="routine"))
	run(s.add_to_waitlist("t", p2.id, "WARD_A", priority="emergency"))
	entries = run(s.list_waitlist("t", unit_id="WARD_A"))
	assert entries[0]["priority"] == "emergency"


def test_manage_waitlist_offer():
	s = svc()
	p = make_patient(s)
	b = make_bed(s)
	entry = run(s.add_to_waitlist("t", p.id, "WARD_A"))
	updated = run(s.manage_waitlist("t", entry["id"], "offer", offered_bed_id=b.id))
	assert updated["status"] == "offered"
	assert updated["offered_bed_id"] == b.id


def test_manage_waitlist_admit():
	s = svc()
	p = make_patient(s)
	entry = run(s.add_to_waitlist("t", p.id, "WARD_A"))
	updated = run(s.manage_waitlist("t", entry["id"], "admit"))
	assert updated["status"] == "admitted"
	assert updated["admitted_at"] is not None


# ── auto_match_waitlist_to_beds ───────────────────────────────────────────────

def test_auto_match_finds_match():
	s = svc()
	p = make_patient(s)
	make_bed(s, unit="WARD_A")
	run(s.add_to_waitlist("t", p.id, "WARD_A"))
	matches = run(s.auto_match_waitlist_to_beds("t"))
	assert len(matches) == 1
	assert matches[0]["patient_id"] == p.id


def test_auto_match_isolation_constraint():
	s = svc()
	p = make_patient(s)
	make_bed(s, unit="WARD_A", iso=False)  # non-isolation bed
	run(s.add_to_waitlist("t", p.id, "WARD_A", isolation_required=True))
	matches = run(s.auto_match_waitlist_to_beds("t"))
	assert len(matches) == 0  # no isolation-capable bed available


def test_auto_match_isolation_satisfied():
	s = svc()
	p = make_patient(s)
	make_bed(s, unit="WARD_A", iso=True)  # isolation-capable
	run(s.add_to_waitlist("t", p.id, "WARD_A", isolation_required=True))
	matches = run(s.auto_match_waitlist_to_beds("t"))
	assert len(matches) == 1


def test_auto_match_no_beds_returns_empty():
	s = svc()
	p = make_patient(s)
	run(s.add_to_waitlist("t", p.id, "WARD_A"))
	matches = run(s.auto_match_waitlist_to_beds("t"))
	assert matches == []


# ── pre_screen_claim ──────────────────────────────────────────────────────────

def test_pre_screen_missing_codes():
	s = svc()
	p = make_patient(s)
	b = make_bed(s)
	adm = make_admission(s, p, b)
	from pmt.models import InsuranceCreate
	ins = run(s.add_insurance(InsuranceCreate(
		tenant_id="t", patient_id=p.id, insurance_type="commercial",
		payer_name="NHIF", member_id="NH001", effective_date=datetime(2026, 1, 1),
		created_by="reg",
	)))
	result = run(s.pre_screen_claim("t", p.id, adm.id, [], [], ins.id))
	assert result["clean"] is False
	assert "missing_diagnosis_codes" in result["issues"]
	assert "missing_procedure_codes" in result["issues"]


def test_pre_screen_unverified_insurance():
	s = svc()
	p = make_patient(s)
	b = make_bed(s)
	adm = make_admission(s, p, b)
	from pmt.models import InsuranceCreate
	ins = run(s.add_insurance(InsuranceCreate(
		tenant_id="t", patient_id=p.id, insurance_type="commercial",
		payer_name="BlueCross", member_id="BC123", effective_date=datetime(2026, 1, 1),
		created_by="reg",
	)))
	result = run(s.pre_screen_claim("t", p.id, adm.id, ["I21.9"], ["99213"], ins.id))
	assert "insurance_not_verified" in result["issues"]


# ── evaluate_clinical_alerts ──────────────────────────────────────────────────

def test_clinical_alerts_normal_vitals():
	s = svc()
	p = make_patient(s)
	alerts = run(s.evaluate_clinical_alerts(
		"t", p.id, "enc1",
		vitals={"bp_systolic": 120, "heart_rate": 72, "spo2": 99, "respiratory_rate": 14, "temperature_c": 37.0, "avpu_score": 1.0},
		allergies=[], chief_complaint="Follow-up",
	))
	assert alerts == []


def test_clinical_alerts_shock_screen():
	s = svc()
	p = make_patient(s)
	alerts = run(s.evaluate_clinical_alerts(
		"t", p.id, "enc1",
		vitals={"bp_systolic": 85, "heart_rate": 120, "spo2": 94, "respiratory_rate": 18, "temperature_c": 37.0, "avpu_score": 1.0},
		allergies=[], chief_complaint="Dizziness",
	))
	types = [a["type"] for a in alerts]
	assert "shock_screen" in types


def test_clinical_alerts_known_allergy():
	s = svc()
	p = make_patient(s)
	alerts = run(s.evaluate_clinical_alerts(
		"t", p.id, "enc1",
		vitals={"bp_systolic": 120, "heart_rate": 80, "spo2": 99, "respiratory_rate": 14, "temperature_c": 37.0, "avpu_score": 1.0},
		allergies=["penicillin"], chief_complaint="Sore throat",
	))
	types = [a["type"] for a in alerts]
	assert "known_allergy" in types


# ── telemedicine_booking ──────────────────────────────────────────────────────

def test_telemedicine_booking_requires_consent():
	s = svc()
	p = make_patient(s)
	with pytest.raises(PolicyViolationError):
		run(s.telemedicine_booking("t", p.id, "dr1", datetime(2027, 3, 1, 10), "Cough", consent_obtained=False))


def test_telemedicine_booking_creates_join_url():
	s = svc()
	p = make_patient(s)
	result = run(s.telemedicine_booking(
		"t", p.id, "dr1", datetime(2027, 3, 1, 10), "Cough",
		consent_obtained=True, created_by="sched",
	))
	assert result["status"] == "scheduled"
	assert result["join_url"].startswith("https://")


# ── patient_portal_registration ───────────────────────────────────────────────

def test_portal_registration():
	s = svc()
	p = make_patient(s)
	portal = run(s.patient_portal_registration("t", p.id, "amina@example.com", created_by="admin"))
	assert portal["activated"] is False
	assert portal["email"] == "amina@example.com"


def test_portal_registration_email_required():
	s = svc()
	p = make_patient(s)
	with pytest.raises(AssertionError):
		run(s.patient_portal_registration("t", p.id, ""))


# ── deposit and payment plan ──────────────────────────────────────────────────

def test_record_deposit():
	s = svc()
	p = make_patient(s)
	dep = run(s.record_deposit("t", p.id, 5000.0, deposit_type="admission", payment_method="cash"))
	assert dep["amount"] == 5000.0
	assert dep["receipt_reference"].startswith("RCT-DEP-")


def test_create_payment_plan():
	s = svc()
	p = make_patient(s)
	plan = run(s.create_payment_plan("t", p.id, "bill1", 12000.0, 6, datetime(2027, 1, 1)))
	assert plan["installments"] == 6
	assert plan["installment_amount"] == 2000.0
	assert plan["status"] == "active"


def test_create_payment_plan_min_installments():
	s = svc()
	p = make_patient(s)
	with pytest.raises(AssertionError):
		run(s.create_payment_plan("t", p.id, "bill1", 1000.0, 1, datetime(2027, 1, 1)))
