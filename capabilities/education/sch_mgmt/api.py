"""Flask Blueprint REST API for APG School Management."""

from __future__ import annotations

import asyncio
from typing import Any

from flask import Blueprint, jsonify, request

try:
	from .service import SchoolManagementService
	from .capability_contract import get_capability_contract, evaluate_capability_rules
except ImportError:
	from service import SchoolManagementService  # type: ignore
	from capability_contract import get_capability_contract, evaluate_capability_rules  # type: ignore


blueprint = Blueprint("education_sch_mgmt", __name__, url_prefix="/api/sch-mgmt")
_service = SchoolManagementService()


def _loop() -> asyncio.AbstractEventLoop:
	try:
		return asyncio.get_event_loop()
	except RuntimeError:
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
		return loop


def _run(coro: Any) -> Any:
	return _loop().run_until_complete(coro)


def _ok(data: Any, status: int = 200):
	return jsonify({"status": "ok", "data": data}), status


def _err(message: str, status: int = 400):
	return jsonify({"status": "error", "message": message}), status


def _tenant() -> str:
	return request.headers.get("X-Tenant-Id", request.args.get("tenant_id", "default"))


# ---------------------------------------------------------------------------
# Contract / meta
# ---------------------------------------------------------------------------

@blueprint.get("/contract")
def get_contract():
	"""
	GET /api/sch-mgmt/contract
	Returns the capability contract.
	Permission: education_sch_mgmt:view
	"""
	return _ok(get_capability_contract(_tenant()))


@blueprint.post("/evaluate")
def evaluate_rules():
	"""
	POST /api/sch-mgmt/evaluate
	Evaluate business rules against context.
	Permission: education_sch_mgmt:admin
	"""
	body = request.get_json(force=True) or {}
	return _ok(evaluate_capability_rules(body))


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@blueprint.get("/dashboard")
def dashboard():
	"""
	GET /api/sch-mgmt/dashboard
	Dashboard summary.
	Permission: education_sch_mgmt:view
	"""
	return _ok(_run(_service.dashboard_summary(_tenant())))


# ---------------------------------------------------------------------------
# Students
# ---------------------------------------------------------------------------

@blueprint.get("/students")
def list_students():
	"""
	GET /api/sch-mgmt/students[?grade_level=...&status=...]
	List students.
	Permission: education_sch_mgmt:view_students
	"""
	grade = request.args.get("grade_level")
	status = request.args.get("status")
	return _ok(_run(_service.list_students(_tenant(), grade, status)))


@blueprint.post("/students")
def create_student():
	"""
	POST /api/sch-mgmt/students
	Register a new student.
	Permission: education_sch_mgmt:manage_students
	Body: {first_name, last_name, date_of_birth, student_number, grade_level, created_by, ...}
	"""
	body = request.get_json(force=True) or {}
	try:
		result = _run(_service.create_student(
			tenant_id=_tenant(),
			first_name=body["first_name"],
			last_name=body["last_name"],
			date_of_birth=body["date_of_birth"],
			student_number=body["student_number"],
			grade_level=body["grade_level"],
			created_by=body["created_by"],
			gender=body.get("gender"),
			national_id=body.get("national_id"),
			guardian_ids=body.get("guardian_ids", []),
			address=body.get("address", {}),
			contact_info=body.get("contact_info", {}),
			medical_notes=body.get("medical_notes", ""),
			special_needs=body.get("special_needs", ""),
		))
		return _ok(result, 201)
	except (KeyError, AssertionError, ValueError) as e:
		return _err(str(e))


@blueprint.get("/students/<student_id>")
def get_student(student_id: str):
	"""
	GET /api/sch-mgmt/students/<student_id>
	Retrieve a student profile.
	Permission: education_sch_mgmt:view_students
	"""
	result = _run(_service.get_student(_tenant(), student_id))
	if result is None:
		return _err("student not found", 404)
	return _ok(result)


@blueprint.put("/students/<student_id>/status")
def update_student_status(student_id: str):
	"""
	PUT /api/sch-mgmt/students/<student_id>/status
	Update student status.
	Permission: education_sch_mgmt:manage_students
	Body: {status, approval_reference?}
	"""
	body = request.get_json(force=True) or {}
	try:
		return _ok(_run(_service.update_student_status(
			tenant_id=_tenant(),
			student_id=student_id,
			new_status=body["status"],
			approval_reference=body.get("approval_reference"),
		)))
	except (KeyError, AssertionError, ValueError) as e:
		return _err(str(e))


# ---------------------------------------------------------------------------
# Admissions
# ---------------------------------------------------------------------------

@blueprint.get("/admissions")
def list_admissions():
	"""
	GET /api/sch-mgmt/admissions[?status=...]
	List admission applications.
	Permission: education_sch_mgmt:manage_admissions
	"""
	status = request.args.get("status")
	return _ok(_run(_service.list_admissions(_tenant(), status)))


@blueprint.post("/admissions")
def submit_application():
	"""
	POST /api/sch-mgmt/admissions
	Submit an admission application.
	Permission: education_sch_mgmt:manage_admissions
	"""
	body = request.get_json(force=True) or {}
	try:
		result = _run(_service.submit_application(
			tenant_id=_tenant(),
			applicant_first_name=body["applicant_first_name"],
			applicant_last_name=body["applicant_last_name"],
			date_of_birth=body["date_of_birth"],
			grade_level_applying=body["grade_level_applying"],
			guardian_name=body["guardian_name"],
			guardian_contact=body["guardian_contact"],
			created_by=body["created_by"],
			previous_school=body.get("previous_school", ""),
			documents=body.get("documents", []),
			notes=body.get("notes", ""),
		))
		return _ok(result, 201)
	except (KeyError, AssertionError, ValueError) as e:
		return _err(str(e))


@blueprint.put("/admissions/<application_id>/status")
def update_admission_status(application_id: str):
	"""
	PUT /api/sch-mgmt/admissions/<application_id>/status
	Update admission application status.
	Permission: education_sch_mgmt:manage_admissions
	Body: {status, reviewer_id?, offer_reference?, capacity_available?}
	"""
	body = request.get_json(force=True) or {}
	try:
		return _ok(_run(_service.update_admission_status(
			tenant_id=_tenant(),
			application_id=application_id,
			new_status=body["status"],
			reviewer_id=body.get("reviewer_id"),
			offer_reference=body.get("offer_reference"),
			capacity_available=body.get("capacity_available", True),
		)))
	except (KeyError, AssertionError, ValueError) as e:
		return _err(str(e))


# ---------------------------------------------------------------------------
# Fees
# ---------------------------------------------------------------------------

@blueprint.get("/fees")
def list_fees():
	"""
	GET /api/sch-mgmt/fees[?student_id=...&status=...]
	List fee invoices.
	Permission: education_sch_mgmt:manage_fees
	"""
	student_id = request.args.get("student_id")
	status = request.args.get("status")
	return _ok(_run(_service.list_fee_invoices(_tenant(), student_id, status)))


@blueprint.post("/fees")
def generate_invoice():
	"""
	POST /api/sch-mgmt/fees
	Generate a fee invoice.
	Permission: education_sch_mgmt:manage_fees
	"""
	body = request.get_json(force=True) or {}
	try:
		result = _run(_service.generate_fee_invoice(
			tenant_id=_tenant(),
			student_id=body["student_id"],
			fee_type=body["fee_type"],
			amount=body["amount"],
			academic_year=body["academic_year"],
			term=body["term"],
			due_date=body["due_date"],
			created_by=body["created_by"],
			currency=body.get("currency", "KES"),
			description=body.get("description", ""),
		))
		return _ok(result, 201)
	except (KeyError, AssertionError, ValueError) as e:
		return _err(str(e))


@blueprint.put("/fees/<invoice_id>/pay")
def record_payment(invoice_id: str):
	"""
	PUT /api/sch-mgmt/fees/<invoice_id>/pay
	Record a fee payment.
	Permission: education_sch_mgmt:manage_fees
	Body: {payment_reference}
	"""
	body = request.get_json(force=True) or {}
	try:
		return _ok(_run(_service.record_fee_payment(_tenant(), invoice_id, body["payment_reference"])))
	except (KeyError, AssertionError, ValueError) as e:
		return _err(str(e))


@blueprint.put("/fees/<invoice_id>/waive")
def waive_fee(invoice_id: str):
	"""
	PUT /api/sch-mgmt/fees/<invoice_id>/waive
	Waive a fee invoice.
	Permission: education_sch_mgmt:manage_fees
	Body: {approval_reference}
	"""
	body = request.get_json(force=True) or {}
	try:
		return _ok(_run(_service.waive_fee(_tenant(), invoice_id, body["approval_reference"])))
	except (KeyError, AssertionError, ValueError) as e:
		return _err(str(e))


# ---------------------------------------------------------------------------
# Staff
# ---------------------------------------------------------------------------

@blueprint.get("/staff")
def list_staff():
	"""
	GET /api/sch-mgmt/staff[?role=...]
	List staff records.
	Permission: education_sch_mgmt:manage_staff
	"""
	role = request.args.get("role")
	return _ok(_run(_service.list_staff(_tenant(), role)))


@blueprint.post("/staff")
def create_staff():
	"""
	POST /api/sch-mgmt/staff
	Create a staff record.
	Permission: education_sch_mgmt:manage_staff
	"""
	body = request.get_json(force=True) or {}
	try:
		result = _run(_service.create_staff_record(
			tenant_id=_tenant(),
			first_name=body["first_name"],
			last_name=body["last_name"],
			staff_number=body["staff_number"],
			role=body["role"],
			email=body["email"],
			join_date=body["join_date"],
			created_by=body["created_by"],
			phone=body.get("phone"),
			subjects=body.get("subjects", []),
			qualifications=body.get("qualifications", []),
			department=body.get("department"),
		))
		return _ok(result, 201)
	except (KeyError, AssertionError, ValueError) as e:
		return _err(str(e))


# ---------------------------------------------------------------------------
# Academic calendar
# ---------------------------------------------------------------------------

@blueprint.get("/calendar")
def list_events():
	"""
	GET /api/sch-mgmt/calendar[?academic_year=...&term=...]
	List academic calendar events.
	Permission: education_sch_mgmt:manage_calendar
	"""
	ay = request.args.get("academic_year")
	term = request.args.get("term")
	return _ok(_run(_service.list_calendar_events(_tenant(), ay, term)))


@blueprint.post("/calendar")
def create_event():
	"""
	POST /api/sch-mgmt/calendar
	Create an academic calendar event.
	Permission: education_sch_mgmt:manage_calendar
	"""
	body = request.get_json(force=True) or {}
	try:
		result = _run(_service.create_calendar_event(
			tenant_id=_tenant(),
			title=body["title"],
			event_type=body["event_type"],
			start_date=body["start_date"],
			end_date=body["end_date"],
			academic_year=body["academic_year"],
			term=body["term"],
			created_by=body["created_by"],
			description=body.get("description", ""),
			is_public=body.get("is_public", True),
			affected_grade_levels=body.get("affected_grade_levels", []),
			location=body.get("location"),
		))
		return _ok(result, 201)
	except (KeyError, AssertionError, ValueError) as e:
		return _err(str(e))


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

@blueprint.post("/documents")
def upload_document():
	"""
	POST /api/sch-mgmt/documents
	Upload a document.
	Permission: education_sch_mgmt:manage_documents
	"""
	body = request.get_json(force=True) or {}
	try:
		result = _run(_service.upload_document(
			tenant_id=_tenant(),
			owner_id=body["owner_id"],
			owner_type=body["owner_type"],
			document_type=body["document_type"],
			title=body["title"],
			file_reference=body["file_reference"],
			created_by=body["created_by"],
			is_confidential=body.get("is_confidential", False),
			expiry_date=body.get("expiry_date"),
		))
		return _ok(result, 201)
	except (KeyError, AssertionError, ValueError) as e:
		return _err(str(e))


# ---------------------------------------------------------------------------
# Communications
# ---------------------------------------------------------------------------

@blueprint.post("/communications")
def dispatch_communication():
	"""
	POST /api/sch-mgmt/communications
	Dispatch a communication.
	Permission: education_sch_mgmt:send_communications
	"""
	body = request.get_json(force=True) or {}
	try:
		result = _run(_service.dispatch_communication(
			tenant_id=_tenant(),
			subject=body["subject"],
			body=body["body"],
			channel=body["channel"],
			sender_id=body["sender_id"],
			created_by=body["created_by"],
			recipient_ids=body.get("recipient_ids", []),
			recipient_groups=body.get("recipient_groups", []),
		))
		return _ok(result, 201)
	except (KeyError, AssertionError, ValueError) as e:
		return _err(str(e))


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

@blueprint.post("/agents")
def register_agent():
	"""
	POST /api/sch-mgmt/agents
	Register a school management AI agent.
	Permission: education_sch_mgmt:admin
	"""
	body = request.get_json(force=True) or {}
	try:
		result = _run(_service.register_agent(
			tenant_id=_tenant(),
			name=body["name"],
			runtime=body["runtime"],
			role=body["role"],
			created_by=body["created_by"],
			scope=body.get("scope", "school management operations"),
		))
		return _ok(result, 201)
	except (KeyError, AssertionError, ValueError) as e:
		return _err(str(e))
