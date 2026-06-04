"""Tests for PatientManagementService."""

from __future__ import annotations
import asyncio, sys, os
from datetime import datetime, timedelta
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pmt.models import AdmissionCreate, AppointmentCreate, BedCreate, InsuranceCreate, PatientCreate
from pmt.service import PatientManagementService, PolicyViolationError


def run(coro):
	return asyncio.get_event_loop().run_until_complete(coro)


def svc():
	return PatientManagementService()


def make_patient(s, tid="t") -> any:
	return run(s.register_patient(PatientCreate(
		tenant_id=tid, first_name="Jane", last_name="Smith",
		date_of_birth=datetime(1970, 5, 15), gender_code="female",
		phone="555-1234", created_by="registrar",
	)))


def make_bed(s, tid="t", unit="ICU") -> any:
	return run(s.register_bed(BedCreate(
		tenant_id=tid, unit_id=unit, bed_number="101A",
		bed_type="medical_surgical", location="2nd Floor", created_by="admin",
	)))


def test_register_patient_generates_mrn():
	s = svc()
	patient = make_patient(s)
	assert patient.mrn.startswith("MRN")
	assert patient.status == "active"


def test_search_by_last_name():
	s = svc()
	make_patient(s)
	results = run(s.search_patients("t", last_name="Smith"))
	assert len(results) == 1 and results[0].last_name == "Smith"


def test_search_by_mrn():
	s = svc()
	patient = make_patient(s)
	results = run(s.search_patients("t", mrn=patient.mrn))
	assert len(results) == 1 and results[0].id == patient.id


def test_register_bed_available():
	s = svc()
	bed = make_bed(s)
	assert bed.status == "available"


def test_admit_patient_occupies_bed():
	s = svc()
	patient = make_patient(s)
	bed = make_bed(s)
	admission = run(s.admit_patient(AdmissionCreate(
		tenant_id="t", patient_id=patient.id, admission_type="elective",
		admitting_provider_id="dr1", attending_provider_id="dr1",
		unit_id="ICU", bed_id=bed.id, chief_complaint="Knee replacement",
		physician_order_present=True, created_by="admissions",
	)))
	assert admission.status == "admitted"
	updated_bed = run(s.list_beds("t", unit_id="ICU"))[0]
	assert updated_bed.status == "occupied" and updated_bed.patient_id == patient.id


def test_admit_inactive_patient_denied():
	s = svc()
	patient = make_patient(s)
	run(s.update_patient_status("t", patient.id, "inactive"))
	bed = make_bed(s)
	try:
		run(s.admit_patient(AdmissionCreate(
			tenant_id="t", patient_id=patient.id, admission_type="elective",
			admitting_provider_id="dr1", attending_provider_id="dr1",
			unit_id="ICU", bed_id=bed.id, chief_complaint="X",
			physician_order_present=True, created_by="admissions",
		)))
		assert False
	except PolicyViolationError:
		pass


def test_discharge_patient_frees_bed():
	s = svc()
	patient = make_patient(s)
	bed = make_bed(s)
	admission = run(s.admit_patient(AdmissionCreate(
		tenant_id="t", patient_id=patient.id, admission_type="emergency",
		admitting_provider_id="dr1", attending_provider_id="dr1",
		unit_id="ICU", bed_id=bed.id, chief_complaint="Chest pain",
		physician_order_present=True, created_by="admissions",
	)))
	discharged = run(s.discharge_patient("t", admission.id, "home", physician_order_present=True))
	assert discharged.status == "discharged" and discharged.discharge_disposition == "home"
	freed_bed = run(s.list_beds("t", unit_id="ICU"))[0]
	assert freed_bed.status == "cleaning"


def test_discharge_without_physician_order_denied():
	s = svc()
	patient = make_patient(s)
	bed = make_bed(s)
	admission = run(s.admit_patient(AdmissionCreate(
		tenant_id="t", patient_id=patient.id, admission_type="emergency",
		admitting_provider_id="dr1", attending_provider_id="dr1",
		unit_id="ICU", bed_id=bed.id, chief_complaint="X",
		physician_order_present=True, created_by="admissions",
	)))
	try:
		run(s.discharge_patient("t", admission.id, "home", physician_order_present=False))
		assert False
	except PolicyViolationError:
		pass


def test_schedule_appointment():
	s = svc()
	patient = make_patient(s)
	appt = run(s.schedule_appointment(AppointmentCreate(
		tenant_id="t", patient_id=patient.id, provider_id="dr1",
		appointment_type="follow_up", scheduled_at=datetime.utcnow() + timedelta(days=7),
		duration_minutes=30, location_id="clinic_1", reason="BP follow-up",
		slot_available=True, created_by="scheduler",
	)))
	assert appt.status == "scheduled"


def test_schedule_appointment_no_slot_denied():
	s = svc()
	patient = make_patient(s)
	try:
		run(s.schedule_appointment(AppointmentCreate(
			tenant_id="t", patient_id=patient.id, provider_id="dr1",
			appointment_type="follow_up", scheduled_at=datetime.utcnow() + timedelta(days=7),
			duration_minutes=30, location_id="clinic_1", reason="BP",
			slot_available=False, created_by="scheduler",
		)))
		assert False
	except PolicyViolationError:
		pass


def test_cancel_appointment_requires_reason():
	s = svc()
	patient = make_patient(s)
	appt = run(s.schedule_appointment(AppointmentCreate(
		tenant_id="t", patient_id=patient.id, provider_id="dr1",
		appointment_type="follow_up", scheduled_at=datetime.utcnow() + timedelta(days=7),
		duration_minutes=30, location_id="clinic_1", reason="BP",
		slot_available=True, created_by="scheduler",
	)))
	try:
		run(s.cancel_appointment("t", appt.id, ""))
		assert False
	except PolicyViolationError:
		pass


def test_check_in_appointment():
	s = svc()
	patient = make_patient(s)
	appt = run(s.schedule_appointment(AppointmentCreate(
		tenant_id="t", patient_id=patient.id, provider_id="dr1",
		appointment_type="new_patient", scheduled_at=datetime.utcnow() + timedelta(minutes=30),
		duration_minutes=60, location_id="clinic_1", reason="New patient",
		slot_available=True, created_by="scheduler",
	)))
	checked_in = run(s.check_in_appointment("t", appt.id))
	assert checked_in.status == "checked_in" and checked_in.checked_in_at is not None


def test_add_insurance():
	s = svc()
	patient = make_patient(s)
	ins = run(s.add_insurance(InsuranceCreate(
		tenant_id="t", patient_id=patient.id, insurance_type="commercial",
		payer_name="BlueCross", member_id="XYZ123456",
		effective_date=datetime(2026, 1, 1), primary=True, created_by="registrar",
	)))
	assert ins.insurance_type == "commercial" and ins.verification_status == "pending"


def test_merge_patients_requires_approval():
	s = svc()
	p1 = make_patient(s)
	p2 = run(s.register_patient(PatientCreate(
		tenant_id="t", first_name="John", last_name="Jones",
		date_of_birth=datetime(1985, 3, 10), gender_code="male",
		phone="555-9999", created_by="registrar",
	)))
	try:
		run(s.merge_patients("t", p1.id, p2.id, ""))
		assert False
	except PolicyViolationError:
		pass


def test_dashboard_summary():
	s = svc()
	make_patient(s)
	summary = run(s.dashboard_summary("t"))
	assert summary["patients"]["total"] == 1
	assert "admissions" in summary and "beds" in summary
