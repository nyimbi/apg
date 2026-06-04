"""Tests for Pydantic v2 models — Patient Management."""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import pytest

from pmt.models import (
	AdmissionCreate, AdmissionType, AppointmentCreate, AppointmentType,
	BedCreate, BedType, BillLineItem, CoPayCreate, DepositCreate,
	DischargeCreate, DischargeDisposition, GenderCode, InsuranceClaimCreate,
	InsuranceCreate, InsuranceType, IsolationReason, PatientBillCreate,
	PatientCreate, PatientPortalCreate, PaymentPlanCreate, TelemedicineBookingCreate,
	TriageCreate, TriageLevel, WaitlistCreate, WaitlistPriority, uuid7str,
)


# ── uuid7str ──────────────────────────────────────────────────────────────────

def test_uuid7str_returns_string():
	v = uuid7str()
	assert isinstance(v, str) and len(v) == 36


def test_uuid7str_unique():
	assert uuid7str() != uuid7str()


# ── PatientCreate ─────────────────────────────────────────────────────────────

def test_patient_create_valid():
	p = PatientCreate(
		tenant_id="t1", first_name="Alice", last_name="Kimani",
		date_of_birth=datetime(1985, 6, 1), gender_code="female",
		phone="0712345678", created_by="reg",
	)
	assert p.first_name == "Alice"
	assert p.vip is False
	assert p.language_preference == "en"


def test_patient_create_strips_whitespace():
	p = PatientCreate(
		tenant_id="  t1  ", first_name="  Bob  ", last_name="  Doe  ",
		date_of_birth=datetime(1990, 1, 1), gender_code="male",
		phone="0700000000", created_by="reg",
	)
	assert p.tenant_id == "t1"
	assert p.first_name == "Bob"


def test_patient_create_rejects_empty_name():
	with pytest.raises(Exception):
		PatientCreate(
			tenant_id="t1", first_name="", last_name="Doe",
			date_of_birth=datetime(1990, 1, 1), gender_code="male",
			phone="0700000000", created_by="reg",
		)


def test_patient_create_forbids_extra():
	with pytest.raises(Exception):
		PatientCreate(
			tenant_id="t1", first_name="X", last_name="Y",
			date_of_birth=datetime(1990, 1, 1), gender_code="male",
			phone="0700000000", created_by="reg",
			unknown_field="bad",  # extra
		)


# ── GenderCode enum ───────────────────────────────────────────────────────────

def test_gender_code_values():
	assert set(GenderCode) == {GenderCode.male, GenderCode.female, GenderCode.other, GenderCode.unknown}


# ── AdmissionCreate validation ────────────────────────────────────────────────

def test_admission_emergency_bypass_valid():
	a = AdmissionCreate(
		tenant_id="t", patient_id="p1", admission_type="emergency",
		admitting_provider_id="dr1", attending_provider_id="dr1",
		unit_id="ED", bed_id="b1", chief_complaint="Chest pain",
		emergency_bypass_registration=True, created_by="nurse",
	)
	assert a.emergency_bypass_registration is True


def test_admission_bypass_invalid_for_elective():
	with pytest.raises(Exception):
		AdmissionCreate(
			tenant_id="t", patient_id="p1", admission_type="elective",
			admitting_provider_id="dr1", attending_provider_id="dr1",
			unit_id="WARD", bed_id="b1", chief_complaint="Knee",
			emergency_bypass_registration=True, created_by="nurse",
		)


# ── TriageCreate validators ───────────────────────────────────────────────────

def test_triage_pain_score_valid():
	t = TriageCreate(
		tenant_id="t", patient_id="p1",
		triage_level="level_3_urgent", chief_complaint="Headache",
		pain_score=7, triaged_by="nurse1", created_by="nurse1",
	)
	assert t.pain_score == 7


def test_triage_pain_score_out_of_range():
	with pytest.raises(Exception):
		TriageCreate(
			tenant_id="t", patient_id="p1",
			triage_level="level_3_urgent", chief_complaint="Headache",
			pain_score=11, triaged_by="nurse1", created_by="nurse1",
		)


# ── AppointmentCreate ─────────────────────────────────────────────────────────

def test_appointment_duration_positive():
	with pytest.raises(Exception):
		AppointmentCreate(
			tenant_id="t", patient_id="p1", provider_id="dr1",
			appointment_type="follow_up",
			scheduled_at=datetime(2027, 1, 1, 10, 0),
			duration_minutes=0,
			location_id="clinic1", reason="Follow-up",
			created_by="sched",
		)


def test_appointment_telemedicine_defaults():
	a = AppointmentCreate(
		tenant_id="t", patient_id="p1", provider_id="dr1",
		appointment_type="telehealth",
		scheduled_at=datetime(2027, 3, 1, 9, 0),
		duration_minutes=20,
		location_id="virtual", reason="Consult",
		created_by="sched",
	)
	assert a.telemedicine is False  # must set explicitly


# ── BillLineItem ──────────────────────────────────────────────────────────────

def test_bill_line_item_qty_positive():
	with pytest.raises(Exception):
		BillLineItem(
			description="Ward charge", quantity=0,
			unit_price=Decimal("3500"), total=Decimal("0"),
		)


def test_bill_line_item_negative_price():
	with pytest.raises(Exception):
		BillLineItem(
			description="Credit", quantity=1,
			unit_price=Decimal("-100"), total=Decimal("-100"),
		)


# ── PaymentPlanCreate ─────────────────────────────────────────────────────────

def test_payment_plan_min_installments():
	with pytest.raises(Exception):
		PaymentPlanCreate(
			tenant_id="t", patient_id="p1", bill_id="b1",
			total_amount=Decimal("10000"), installments=1,
			installment_amount=Decimal("10000"),
			start_date=datetime(2027, 1, 1), created_by="finance",
		)


def test_payment_plan_valid():
	pp = PaymentPlanCreate(
		tenant_id="t", patient_id="p1", bill_id="b1",
		total_amount=Decimal("12000"), installments=6,
		installment_amount=Decimal("2000"),
		start_date=datetime(2027, 1, 1), created_by="finance",
	)
	assert pp.installments == 6


# ── TelemedicineBookingCreate ─────────────────────────────────────────────────

def test_telemedicine_min_duration():
	with pytest.raises(Exception):
		TelemedicineBookingCreate(
			tenant_id="t", patient_id="p1", provider_id="dr1",
			scheduled_at=datetime(2027, 6, 1, 10, 0),
			duration_minutes=3, chief_complaint="Cold",
			created_by="sched",
		)


def test_telemedicine_valid():
	tb = TelemedicineBookingCreate(
		tenant_id="t", patient_id="p1", provider_id="dr1",
		scheduled_at=datetime(2027, 6, 1, 10, 0),
		duration_minutes=20, chief_complaint="Fever",
		consent_obtained=True, created_by="sched",
	)
	assert tb.consent_obtained is True


# ── WaitlistCreate ────────────────────────────────────────────────────────────

def test_waitlist_default_priority():
	w = WaitlistCreate(
		tenant_id="t", patient_id="p1", unit_id="WARD_A",
		created_by="nurse",
	)
	assert w.priority == WaitlistPriority.routine


def test_waitlist_emergency_priority():
	w = WaitlistCreate(
		tenant_id="t", patient_id="p1", unit_id="ICU",
		priority="emergency", isolation_required=True,
		isolation_reason="infectious", created_by="nurse",
	)
	assert w.isolation_required is True


# ── InsuranceClaimCreate ──────────────────────────────────────────────────────

def test_claim_positive_amount():
	with pytest.raises(Exception):
		InsuranceClaimCreate(
			tenant_id="t", patient_id="p1", admission_id="a1",
			insurance_id="i1", bill_id="b1",
			total_billed=Decimal("-100"),
			created_by="billing",
		)


# ── PatientPortalCreate ───────────────────────────────────────────────────────

def test_portal_requires_email():
	with pytest.raises(Exception):
		PatientPortalCreate(
			tenant_id="t", patient_id="p1",
			email="", created_by="admin",
		)
