"""Service-layer tests for education_sch_mgmt."""

from __future__ import annotations

import asyncio
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from service import SchoolManagementService


def run(coro):
	loop = asyncio.get_event_loop()
	return loop.run_until_complete(coro)


T = "test_school"


# ---------------------------------------------------------------------------
# students
# ---------------------------------------------------------------------------

def test_create_and_get_student():
	svc = SchoolManagementService()
	student = run(svc.create_student(T, "Alice", "Mwangi", "2010-05-01", "STU001", "grade_5", "admin"))
	assert student["first_name"] == "Alice"
	assert student["status"] == "active"
	fetched = run(svc.get_student(T, student["id"]))
	assert fetched["id"] == student["id"]


def test_list_students_by_grade():
	svc = SchoolManagementService()
	run(svc.create_student(T, "A", "A", "2010-01-01", "S001", "grade_5", "admin"))
	run(svc.create_student(T, "B", "B", "2011-01-01", "S002", "grade_5", "admin"))
	run(svc.create_student(T, "C", "C", "2012-01-01", "S003", "grade_6", "admin"))
	grade5 = run(svc.list_students(T, grade_level="grade_5"))
	assert len(grade5) == 2


def test_unsupported_grade_level_denied():
	svc = SchoolManagementService()
	with pytest.raises(ValueError):
		run(svc.create_student(T, "X", "X", "2010-01-01", "S999", "year_99", "admin"))


def test_update_student_status_normal():
	svc = SchoolManagementService()
	student = run(svc.create_student(T, "A", "B", "2010-01-01", "S001", "grade_5", "admin"))
	updated = run(svc.update_student_status(T, student["id"], "suspended"))
	assert updated["status"] == "suspended"


def test_expulsion_without_approval_denied():
	svc = SchoolManagementService()
	student = run(svc.create_student(T, "A", "B", "2010-01-01", "S001", "grade_5", "admin"))
	with pytest.raises(ValueError, match="expulsion_requires_approval"):
		run(svc.update_student_status(T, student["id"], "expelled"))


def test_expulsion_with_approval():
	svc = SchoolManagementService()
	student = run(svc.create_student(T, "A", "B", "2010-01-01", "S001", "grade_5", "admin"))
	updated = run(svc.update_student_status(T, student["id"], "expelled", approval_reference="APPR-EXP-001"))
	assert updated["status"] == "expelled"


# ---------------------------------------------------------------------------
# admissions
# ---------------------------------------------------------------------------

def test_submit_application():
	svc = SchoolManagementService()
	app = run(svc.submit_application(T, "John", "Doe", "2012-03-15", "grade_1", "Jane Doe", "0700000001", "admin"))
	assert app["status"] == "submitted"
	assert app["applicant_first_name"] == "John"


def test_update_admission_to_offered():
	svc = SchoolManagementService()
	app = run(svc.submit_application(T, "John", "Doe", "2012-03-15", "grade_1", "Jane Doe", "0700000001", "admin"))
	offered = run(svc.update_admission_status(T, app["id"], "offered", reviewer_id="rev_1", capacity_available=True))
	assert offered["status"] == "offered"


def test_offer_without_capacity_denied():
	svc = SchoolManagementService()
	app = run(svc.submit_application(T, "John", "Doe", "2012-03-15", "grade_1", "Jane Doe", "0700000001", "admin"))
	with pytest.raises(ValueError, match="admission_offer_requires_capacity_check"):
		run(svc.update_admission_status(T, app["id"], "offered", capacity_available=False))


def test_list_admissions_filtered():
	svc = SchoolManagementService()
	run(svc.submit_application(T, "A", "A", "2012-01-01", "grade_1", "G1", "0700000001", "admin"))
	run(svc.submit_application(T, "B", "B", "2012-01-01", "grade_1", "G2", "0700000002", "admin"))
	submitted = run(svc.list_admissions(T, status="submitted"))
	assert len(submitted) == 2


# ---------------------------------------------------------------------------
# fees
# ---------------------------------------------------------------------------

def test_generate_fee_invoice():
	svc = SchoolManagementService()
	student = run(svc.create_student(T, "A", "B", "2010-01-01", "S001", "grade_5", "admin"))
	invoice = run(svc.generate_fee_invoice(T, student["id"], "tuition", 25000.0, "2025-2026", "term_1", "2025-09-30", "admin"))
	assert invoice["status"] == "pending"
	assert invoice["amount"] == 25000.0


def test_record_fee_payment():
	svc = SchoolManagementService()
	student = run(svc.create_student(T, "A", "B", "2010-01-01", "S001", "grade_5", "admin"))
	invoice = run(svc.generate_fee_invoice(T, student["id"], "tuition", 25000.0, "2025-2026", "term_1", "2025-09-30", "admin"))
	paid = run(svc.record_fee_payment(T, invoice["id"], "MPESA-001"))
	assert paid["status"] == "paid"
	assert paid["payment_reference"] == "MPESA-001"


def test_waive_fee_requires_approval():
	svc = SchoolManagementService()
	student = run(svc.create_student(T, "A", "B", "2010-01-01", "S001", "grade_5", "admin"))
	invoice = run(svc.generate_fee_invoice(T, student["id"], "tuition", 5000.0, "2025-2026", "term_1", "2025-09-30", "admin"))
	with pytest.raises(ValueError, match="fee_waiver_requires_approval"):
		run(svc.waive_fee(T, invoice["id"], ""))


def test_waive_fee_with_approval():
	svc = SchoolManagementService()
	student = run(svc.create_student(T, "A", "B", "2010-01-01", "S001", "grade_5", "admin"))
	invoice = run(svc.generate_fee_invoice(T, student["id"], "tuition", 5000.0, "2025-2026", "term_1", "2025-09-30", "admin"))
	waived = run(svc.waive_fee(T, invoice["id"], "APPR-WAIVE-001"))
	assert waived["status"] == "waived"


def test_unsupported_fee_type_denied():
	svc = SchoolManagementService()
	student = run(svc.create_student(T, "A", "B", "2010-01-01", "S001", "grade_5", "admin"))
	with pytest.raises(ValueError):
		run(svc.generate_fee_invoice(T, student["id"], "bribe", 1000.0, "2025-2026", "term_1", "2025-09-30", "admin"))


# ---------------------------------------------------------------------------
# staff
# ---------------------------------------------------------------------------

def test_create_staff_record():
	svc = SchoolManagementService()
	staff = run(svc.create_staff_record(T, "Bob", "Kamau", "TCH001", "teacher", "bob@school.ke", "2023-01-15", "admin"))
	assert staff["role"] == "teacher"


def test_unsupported_staff_role_denied():
	svc = SchoolManagementService()
	with pytest.raises(ValueError):
		run(svc.create_staff_record(T, "X", "X", "X001", "wizard", "x@x.ke", "2023-01-01", "admin"))


def test_list_staff_by_role():
	svc = SchoolManagementService()
	run(svc.create_staff_record(T, "A", "A", "T001", "teacher", "a@s.ke", "2023-01-01", "admin"))
	run(svc.create_staff_record(T, "B", "B", "T002", "teacher", "b@s.ke", "2023-01-01", "admin"))
	run(svc.create_staff_record(T, "C", "C", "P001", "principal", "c@s.ke", "2023-01-01", "admin"))
	teachers = run(svc.list_staff(T, role="teacher"))
	assert len(teachers) == 2


# ---------------------------------------------------------------------------
# calendar
# ---------------------------------------------------------------------------

def test_create_calendar_event():
	svc = SchoolManagementService()
	event = run(svc.create_calendar_event(T, "Term 1 Opening", "academic", "2025-09-01", "2025-09-01", "2025-2026", "term_1", "admin"))
	assert event["event_type"] == "academic"


def test_unsupported_event_type_denied():
	svc = SchoolManagementService()
	with pytest.raises(ValueError):
		run(svc.create_calendar_event(T, "X", "parade", "2025-01-01", "2025-01-01", "2025-2026", "term_1", "admin"))


# ---------------------------------------------------------------------------
# documents
# ---------------------------------------------------------------------------

def test_upload_document():
	svc = SchoolManagementService()
	student = run(svc.create_student(T, "A", "B", "2010-01-01", "S001", "grade_5", "admin"))
	doc = run(svc.upload_document(T, student["id"], "student", "birth_certificate", "Birth Cert", "files/bc001.pdf", "admin"))
	assert doc["document_type"] == "birth_certificate"


def test_share_document_requires_consent():
	svc = SchoolManagementService()
	student = run(svc.create_student(T, "A", "B", "2010-01-01", "S001", "grade_5", "admin"))
	doc = run(svc.upload_document(T, student["id"], "student", "birth_certificate", "BC", "files/bc.pdf", "admin"))
	with pytest.raises(ValueError, match="document_sharing_requires_consent"):
		run(svc.share_document(T, doc["id"], consent_recorded=False))


# ---------------------------------------------------------------------------
# communications
# ---------------------------------------------------------------------------

def test_dispatch_communication():
	svc = SchoolManagementService()
	comm = run(svc.dispatch_communication(T, "Fee Reminder", "Please pay fees.", "sms", "principal_1", "admin", recipient_groups=["grade_5"]))
	assert comm["channel"] == "sms"
	assert comm["is_draft"] is False


def test_unsupported_channel_denied():
	svc = SchoolManagementService()
	with pytest.raises(ValueError):
		run(svc.dispatch_communication(T, "X", "body", "carrier_pigeon", "sender", "admin"))


# ---------------------------------------------------------------------------
# dashboard
# ---------------------------------------------------------------------------

def test_dashboard_summary():
	svc = SchoolManagementService()
	run(svc.create_student(T, "A", "A", "2010-01-01", "S001", "grade_5", "admin"))
	summary = run(svc.dashboard_summary(T))
	assert summary["students"] >= 1
	assert summary["tenant_id"] == T


# ---------------------------------------------------------------------------
# agents
# ---------------------------------------------------------------------------

def test_register_agent():
	svc = SchoolManagementService()
	agent = run(svc.register_agent(T, "FeeBot", "codex", "fee_processor", "admin"))
	assert agent["role"] == "fee_processor"


def test_invalid_agent_role_rejected():
	svc = SchoolManagementService()
	with pytest.raises(AssertionError):
		run(svc.register_agent(T, "Bot", "codex", "nonexistent_role", "admin"))
