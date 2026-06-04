"""Tests for HealthcareRegulatoryService."""

from __future__ import annotations
import asyncio, sys, os
from datetime import datetime, timedelta
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from reg.models import AccreditationCreate, CorrectiveActionCreate, IncidentCreate, LicenseCreate, RegulatorySubmissionCreate
from reg.service import HealthcareRegulatoryService, PolicyViolationError


def run(coro):
	return asyncio.get_event_loop().run_until_complete(coro)


def svc():
	return HealthcareRegulatoryService()


def make_license(s, tid="t") -> any:
	return run(s.add_license(LicenseCreate(
		tenant_id=tid, license_type="facility_operating", license_number="FL-2026-001",
		issuing_authority="State Health Dept", issued_date=datetime(2024, 1, 1),
		expiry_date=datetime.utcnow() + timedelta(days=200),
		holder_name="General Hospital", scope="Acute care 250 beds", created_by="admin",
	)))


def test_add_license():
	s = svc()
	lic = make_license(s)
	assert lic.id and lic.license_type == "facility_operating"
	assert lic.days_to_expiry > 0


def test_license_expiring_within_90_days():
	s = svc()
	run(s.add_license(LicenseCreate(
		tenant_id="t", license_type="physician", license_number="MD-001",
		issuing_authority="Medical Board", issued_date=datetime(2020, 1, 1),
		expiry_date=datetime.utcnow() + timedelta(days=60),
		holder_name="Dr Smith", scope="General practice", created_by="admin",
	)))
	expiring = run(s.get_expiring_licenses("t", days=90))
	assert len(expiring) == 1


def test_add_accreditation():
	s = svc()
	acc = run(s.add_accreditation(AccreditationCreate(
		tenant_id="t", accreditation_body="joint_commission", program="Hospital Accreditation",
		award_date=datetime(2024, 1, 1), expiry_date=datetime(2027, 1, 1),
		certificate_reference="TJC-2024-001", scope="Full hospital", created_by="admin",
	)))
	assert acc.accreditation_body == "joint_commission" and acc.status == "accredited"


def test_report_incident():
	s = svc()
	incident = run(s.report_incident(IncidentCreate(
		tenant_id="t", incident_type="patient_fall", severity="moderate",
		description="Patient fell in hallway", department="Medical-Surgical",
		occurred_at=datetime.utcnow(), reported_by="nurse1",
		immediate_actions="Patient assessed, no injury", created_by="nurse1",
	)))
	assert incident.status == "open" and incident.incident_type == "patient_fall"


def test_report_sentinel_event():
	s = svc()
	incident = run(s.report_incident(IncidentCreate(
		tenant_id="t", incident_type="sentinel_event", severity="catastrophic",
		description="Wrong-site surgery", department="OR",
		occurred_at=datetime.utcnow(), reported_by="surgeon",
		immediate_actions="Surgery halted", created_by="surgeon",
	)))
	assert incident.incident_type == "sentinel_event"


def test_close_sentinel_event_without_rca_denied():
	s = svc()
	incident = run(s.report_incident(IncidentCreate(
		tenant_id="t", incident_type="sentinel_event", severity="catastrophic",
		description="Wrong-site surgery", department="OR",
		occurred_at=datetime.utcnow(), reported_by="surgeon",
		immediate_actions="Surgery halted", created_by="surgeon",
	)))
	try:
		run(s.close_incident("t", incident.id, "", []))
		assert False
	except PolicyViolationError:
		pass


def test_close_sentinel_event_with_rca():
	s = svc()
	incident = run(s.report_incident(IncidentCreate(
		tenant_id="t", incident_type="sentinel_event", severity="catastrophic",
		description="Wrong-site surgery", department="OR",
		occurred_at=datetime.utcnow(), reported_by="surgeon",
		immediate_actions="Surgery halted", created_by="surgeon",
	)))
	closed = run(s.close_incident("t", incident.id, "RCA-2026-001", ["Surgical checklist mandatory", "Timeout procedure updated"]))
	assert closed.status == "closed" and closed.rca_completed


def test_file_and_submit_submission():
	s = svc()
	sub = run(s.file_submission(RegulatorySubmissionCreate(
		tenant_id="t", report_type="cms_iqr", title="Q1 2026 IQR",
		reporting_period_start=datetime(2026, 1, 1), reporting_period_end=datetime(2026, 3, 31),
		submitted_to="CMS", prepared_by="quality_mgr", created_by="quality_mgr",
	)))
	assert sub.status == "draft"
	submitted = run(s.submit_submission("t", sub.id))
	assert submitted.status == "submitted" and submitted.submission_reference is not None


def test_file_unsupported_report_type_denied():
	s = svc()
	try:
		run(s.file_submission(RegulatorySubmissionCreate(
			tenant_id="t", report_type="unknown_type", title="X",
			reporting_period_start=datetime(2026, 1, 1), reporting_period_end=datetime(2026, 3, 31),
			submitted_to="Agency", prepared_by="mgr", created_by="mgr",
		)))
		assert False
	except PolicyViolationError:
		pass


def test_corrective_action_workflow():
	s = svc()
	ca = run(s.create_corrective_action(CorrectiveActionCreate(
		tenant_id="t", source="Patient fall incident", description="Install bed alarms on all Med-Surg beds",
		assigned_to="facilities_mgr", due_date=datetime.utcnow() + timedelta(days=30), created_by="quality_mgr",
	)))
	assert ca.status == "open"
	completed = run(s.complete_corrective_action("t", ca.id, "quality_director"))
	assert completed.status == "completed" and completed.verified_by == "quality_director"


def test_dashboard_summary():
	s = svc()
	make_license(s)
	summary = run(s.dashboard_summary("t"))
	assert summary["licenses"]["total"] == 1
	assert "incidents" in summary and "submissions" in summary
