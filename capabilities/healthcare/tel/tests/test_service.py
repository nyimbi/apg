"""Tests for TelemedicineService."""

from __future__ import annotations
import asyncio, sys, os
from datetime import datetime, timedelta
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tel.models import ConsultationCreate, PrescriptionTransmitCreate, RemoteMonitoringEnrollmentCreate, TeleBillingCreate, TeleSessionCreate
from tel.service import TelemedicineService, PolicyViolationError


def run(coro):
	return asyncio.get_event_loop().run_until_complete(coro)


def svc():
	return TelemedicineService()


def book(s, tid="t", consented=True, e911=True) -> any:
	return run(s.book_consultation(ConsultationCreate(
		tenant_id=tid, patient_id="p1", provider_id="dr1",
		consultation_type="video", scheduled_at=datetime.utcnow() + timedelta(hours=1),
		duration_minutes=30, chief_complaint="Hypertension follow-up",
		platform="webrtc", patient_consent_obtained=consented,
		e911_disclosure_provided=e911, created_by="scheduler",
	)))


def test_book_consultation():
	s = svc()
	consult = book(s)
	assert consult.status == "scheduled" and consult.consultation_type == "video"


def test_cancel_consultation():
	s = svc()
	consult = book(s)
	cancelled = run(s.cancel_consultation("t", consult.id))
	assert cancelled.status == "cancelled"


def test_create_session_no_consent_denied():
	s = svc()
	consult = book(s, consented=False)
	try:
		run(s.create_session(TeleSessionCreate(
			tenant_id="t", consultation_id=consult.id, patient_id="p1",
			provider_id="dr1", platform="webrtc",
			patient_consent_obtained=False, e911_disclosure_provided=True,
			created_by="dr1",
		)))
		assert False
	except PolicyViolationError:
		pass


def test_create_session_no_e911_denied():
	s = svc()
	consult = book(s)
	try:
		run(s.create_session(TeleSessionCreate(
			tenant_id="t", consultation_id=consult.id, patient_id="p1",
			provider_id="dr1", platform="webrtc",
			patient_consent_obtained=True, e911_disclosure_provided=False,
			created_by="dr1",
		)))
		assert False
	except PolicyViolationError:
		pass


def test_create_session_cancelled_consult_denied():
	s = svc()
	consult = book(s)
	run(s.cancel_consultation("t", consult.id))
	try:
		run(s.create_session(TeleSessionCreate(
			tenant_id="t", consultation_id=consult.id, patient_id="p1",
			provider_id="dr1", platform="webrtc",
			patient_consent_obtained=True, e911_disclosure_provided=True,
			created_by="dr1",
		)))
		assert False
	except PolicyViolationError:
		pass


def test_create_and_complete_session():
	s = svc()
	consult = book(s)
	session = run(s.create_session(TeleSessionCreate(
		tenant_id="t", consultation_id=consult.id, patient_id="p1",
		provider_id="dr1", platform="webrtc",
		patient_consent_obtained=True, e911_disclosure_provided=True,
		created_by="dr1",
	)))
	assert session.status == "waiting" and session.join_url.startswith("https://")
	completed = run(s.complete_session("t", session.id))
	assert completed.status == "completed" and completed.ended_at is not None


def test_enroll_monitoring_without_threshold_denied():
	s = svc()
	try:
		run(s.enroll_monitoring(RemoteMonitoringEnrollmentCreate(
			tenant_id="t", patient_id="p1", device_type="glucometer",
			device_id="dev_001", alert_thresholds={},
			provider_id="dr1", alert_threshold_configured=False, created_by="dr1",
		)))
		assert False
	except PolicyViolationError:
		pass


def test_enroll_monitoring_success():
	s = svc()
	enrollment = run(s.enroll_monitoring(RemoteMonitoringEnrollmentCreate(
		tenant_id="t", patient_id="p1", device_type="glucometer",
		device_id="dev_001", alert_thresholds={"glucose_high": 250, "glucose_low": 70},
		provider_id="dr1", alert_threshold_configured=True, created_by="dr1",
	)))
	assert enrollment.status == "active"


def test_transmit_prescription_schedule_ii_without_in_person_denied():
	s = svc()
	try:
		run(s.transmit_prescription(PrescriptionTransmitCreate(
			tenant_id="t", patient_id="p1", consultation_id="c1",
			drug_name="Oxycodone", drug_schedule="schedule_ii",
			dose="5mg", route="oral", frequency="q6h", quantity=30, refills=0,
			prescriber_id="dr1", pharmacy_id="rx1",
			transmission_method="epcs", in_person_visit_completed=False, created_by="dr1",
		)))
		assert False
	except PolicyViolationError:
		pass


def test_transmit_prescription_non_controlled():
	s = svc()
	rx = run(s.transmit_prescription(PrescriptionTransmitCreate(
		tenant_id="t", patient_id="p1", consultation_id="c1",
		drug_name="Metformin", drug_schedule="non_controlled",
		dose="500mg", route="oral", frequency="BID", quantity=60, refills=5,
		prescriber_id="dr1", pharmacy_id="rx1",
		transmission_method="surescripts", in_person_visit_completed=True, created_by="dr1",
	)))
	assert rx.status == "transmitted" and rx.confirmation_number is not None


def test_create_billing_record():
	s = svc()
	bill = run(s.create_billing_record(TeleBillingCreate(
		tenant_id="t", consultation_id="c1", patient_id="p1", provider_id="dr1",
		billing_code="99213", place_of_service="02",
		diagnosis_codes=["I10"], units=1, created_by="biller",
	)))
	assert bill.billing_code == "99213" and bill.status == "pending"


def test_create_billing_unsupported_code_denied():
	s = svc()
	try:
		run(s.create_billing_record(TeleBillingCreate(
			tenant_id="t", consultation_id="c1", patient_id="p1", provider_id="dr1",
			billing_code="99999", place_of_service="02",
			diagnosis_codes=[], units=1, created_by="biller",
		)))
		assert False
	except PolicyViolationError:
		pass


def test_dashboard_summary():
	s = svc()
	book(s)
	summary = run(s.dashboard_summary("t"))
	assert summary["consultations"]["total"] == 1
	assert "sessions" in summary and "monitoring" in summary
