"""Flask Blueprint REST API for APG Patient Management."""

from __future__ import annotations
import asyncio
from datetime import datetime
from typing import Any
from flask import Blueprint, jsonify, request
from .models import AdmissionCreate, AppointmentCreate, BedCreate, InsuranceCreate, PatientCreate
from .service import PatientManagementService, PolicyViolationError

bp = Blueprint("healthcare_pmt", __name__, url_prefix="/api/healthcare/pmt")
_svc = PatientManagementService()


def _run(coro: Any) -> Any:
	loop = asyncio.new_event_loop()
	try:
		return asyncio.run(coro)
	finally:
		loop.close()


def _err(msg: str, status: int = 400) -> Any:
	return jsonify({"error": msg}), status


def _tenant() -> str:
	return request.headers.get("X-Tenant-ID", request.args.get("tenant_id", "default"))


def _dt(data: dict, field: str) -> None:
	if field in data and isinstance(data[field], str):
		data[field] = datetime.fromisoformat(data[field])


@bp.get("/contract")
def get_contract():
	return jsonify(_run(_svc.describe(_tenant())))


@bp.get("/dashboard")
def dashboard():
	return jsonify(_run(_svc.dashboard_summary(_tenant())))


@bp.get("/patients")
def search_patients():
	patients = _run(_svc.search_patients(_tenant(), last_name=request.args.get("last_name"), mrn=request.args.get("mrn")))
	return jsonify({"items": [p.model_dump(mode="json") for p in patients], "count": len(patients)})


@bp.post("/patients")
def register_patient():
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	_dt(data, "date_of_birth")
	try:
		from .models import PatientCreate as PC
		patient = _run(_svc.register_patient(PC(**data)))
		return jsonify(patient.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.get("/patients/<patient_id>")
def get_patient(patient_id: str):
	patient = _run(_svc.get_patient(_tenant(), patient_id))
	if patient is None:
		return _err("patient_not_found", 404)
	return jsonify(patient.model_dump(mode="json"))


@bp.post("/patients/<patient_id>/merge")
def merge_patients(patient_id: str):
	data = request.get_json(silent=True) or {}
	try:
		patient = _run(_svc.merge_patients(_tenant(), patient_id, data.get("target_id", ""), data.get("approved_by", "")))
		if patient is None:
			return _err("patient_not_found", 404)
		return jsonify(patient.model_dump(mode="json"))
	except PolicyViolationError as e:
		return _err(str(e), 403)


@bp.get("/admissions")
def list_admissions():
	items = _run(_svc.list_admissions(_tenant(), patient_id=request.args.get("patient_id"), status=request.args.get("status")))
	return jsonify({"items": [a.model_dump(mode="json") for a in items], "count": len(items)})


@bp.post("/admissions")
def admit_patient():
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	try:
		admission = _run(_svc.admit_patient(AdmissionCreate(**data)))
		return jsonify(admission.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.post("/admissions/<admission_id>/discharge")
def discharge_patient(admission_id: str):
	data = request.get_json(silent=True) or {}
	try:
		admission = _run(_svc.discharge_patient(
			_tenant(), admission_id,
			data.get("disposition", "home"),
			physician_order_present=data.get("physician_order_present", False),
			discharge_type=data.get("discharge_type", "planned"),
			condition_on_discharge=data.get("condition_on_discharge", "improved"),
		))
		if admission is None:
			return _err("admission_not_found", 404)
		return jsonify(admission.model_dump(mode="json"))
	except PolicyViolationError as e:
		return _err(str(e), 403)


@bp.get("/beds")
def list_beds():
	beds = _run(_svc.list_beds(_tenant(), unit_id=request.args.get("unit_id"), status=request.args.get("status")))
	return jsonify({"items": [b.model_dump(mode="json") for b in beds], "count": len(beds)})


@bp.post("/beds")
def register_bed():
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	try:
		bed = _run(_svc.register_bed(BedCreate(**data)))
		return jsonify(bed.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.put("/beds/<bed_id>/status")
def update_bed_status(bed_id: str):
	data = request.get_json(silent=True) or {}
	try:
		bed = _run(_svc.update_bed_status(_tenant(), bed_id, data.get("status", "")))
		if bed is None:
			return _err("bed_not_found", 404)
		return jsonify(bed.model_dump(mode="json"))
	except PolicyViolationError as e:
		return _err(str(e), 403)


@bp.get("/appointments")
def list_appointments():
	items = _run(_svc.list_appointments(_tenant(), patient_id=request.args.get("patient_id"), provider_id=request.args.get("provider_id"), status=request.args.get("status")))
	return jsonify({"items": [a.model_dump(mode="json") for a in items], "count": len(items)})


@bp.post("/appointments")
def schedule_appointment():
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	_dt(data, "scheduled_at")
	try:
		appt = _run(_svc.schedule_appointment(AppointmentCreate(**data)))
		return jsonify(appt.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.post("/appointments/<appt_id>/cancel")
def cancel_appointment(appt_id: str):
	data = request.get_json(silent=True) or {}
	try:
		appt = _run(_svc.cancel_appointment(_tenant(), appt_id, data.get("reason", "")))
		if appt is None:
			return _err("appointment_not_found", 404)
		return jsonify(appt.model_dump(mode="json"))
	except PolicyViolationError as e:
		return _err(str(e), 403)


@bp.post("/appointments/<appt_id>/check-in")
def check_in_appointment(appt_id: str):
	appt = _run(_svc.check_in_appointment(_tenant(), appt_id))
	if appt is None:
		return _err("appointment_not_found", 404)
	return jsonify(appt.model_dump(mode="json"))


@bp.get("/patients/<patient_id>/insurance")
def list_insurance(patient_id: str):
	items = _run(_svc.list_insurance(_tenant(), patient_id))
	return jsonify({"items": [i.model_dump(mode="json") for i in items], "count": len(items)})


@bp.post("/insurance")
def add_insurance():
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	for f in ("effective_date", "termination_date"):
		_dt(data, f)
	try:
		ins = _run(_svc.add_insurance(InsuranceCreate(**data)))
		return jsonify(ins.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)
