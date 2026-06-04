"""Flask Blueprint REST API for APG Healthcare Regulatory."""

from __future__ import annotations
import asyncio
from datetime import datetime
from typing import Any
from flask import Blueprint, jsonify, request
from .models import AccreditationCreate, CorrectiveActionCreate, IncidentCreate, LicenseCreate, RegulatorySubmissionCreate
from .service import HealthcareRegulatoryService, PolicyViolationError

bp = Blueprint("healthcare_reg", __name__, url_prefix="/api/healthcare/reg")
_svc = HealthcareRegulatoryService()


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


@bp.get("/licenses")
def list_licenses():
	items = _run(_svc.list_licenses(_tenant(), license_type=request.args.get("license_type")))
	return jsonify({"items": [l.model_dump(mode="json") for l in items], "count": len(items)})


@bp.post("/licenses")
def add_license():
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	for f in ("issued_date", "expiry_date"):
		_dt(data, f)
	try:
		lic = _run(_svc.add_license(LicenseCreate(**data)))
		return jsonify(lic.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.get("/licenses/<lic_id>")
def get_license(lic_id: str):
	lic = _run(_svc.get_license(_tenant(), lic_id))
	if lic is None:
		return _err("license_not_found", 404)
	return jsonify(lic.model_dump(mode="json"))


@bp.get("/accreditation")
def list_accreditations():
	items = _run(_svc.list_accreditations(_tenant()))
	return jsonify({"items": [a.model_dump(mode="json") for a in items], "count": len(items)})


@bp.post("/accreditation")
def add_accreditation():
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	for f in ("award_date", "expiry_date"):
		_dt(data, f)
	try:
		acc = _run(_svc.add_accreditation(AccreditationCreate(**data)))
		return jsonify(acc.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.put("/accreditation/<acc_id>/status")
def update_accreditation_status(acc_id: str):
	data = request.get_json(silent=True) or {}
	try:
		acc = _run(_svc.update_accreditation_status(_tenant(), acc_id, data.get("status", "")))
		if acc is None:
			return _err("accreditation_not_found", 404)
		return jsonify(acc.model_dump(mode="json"))
	except PolicyViolationError as e:
		return _err(str(e), 403)


@bp.get("/incidents")
def list_incidents():
	items = _run(_svc.list_incidents(_tenant(), incident_type=request.args.get("incident_type"), severity=request.args.get("severity"), status=request.args.get("status")))
	return jsonify({"items": [i.model_dump(mode="json") for i in items], "count": len(items)})


@bp.post("/incidents")
def report_incident():
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	_dt(data, "occurred_at")
	try:
		incident = _run(_svc.report_incident(IncidentCreate(**data)))
		return jsonify(incident.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.get("/incidents/<incident_id>")
def get_incident(incident_id: str):
	incident = _run(_svc.get_incident(_tenant(), incident_id))
	if incident is None:
		return _err("incident_not_found", 404)
	return jsonify(incident.model_dump(mode="json"))


@bp.post("/incidents/<incident_id>/close")
def close_incident(incident_id: str):
	data = request.get_json(silent=True) or {}
	try:
		incident = _run(_svc.close_incident(_tenant(), incident_id, data.get("rca_reference", ""), data.get("corrective_actions", [])))
		if incident is None:
			return _err("incident_not_found", 404)
		return jsonify(incident.model_dump(mode="json"))
	except PolicyViolationError as e:
		return _err(str(e), 403)


@bp.get("/submissions")
def list_submissions():
	items = _run(_svc.list_submissions(_tenant(), report_type=request.args.get("report_type"), status=request.args.get("status")))
	return jsonify({"items": [s.model_dump(mode="json") for s in items], "count": len(items)})


@bp.post("/submissions")
def file_submission():
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	for f in ("reporting_period_start", "reporting_period_end"):
		_dt(data, f)
	try:
		sub = _run(_svc.file_submission(RegulatorySubmissionCreate(**data)))
		return jsonify(sub.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.post("/submissions/<sub_id>/submit")
def submit_submission(sub_id: str):
	sub = _run(_svc.submit_submission(_tenant(), sub_id))
	if sub is None:
		return _err("submission_not_found", 404)
	return jsonify(sub.model_dump(mode="json"))


@bp.get("/corrective-actions")
def list_corrective_actions():
	items = _run(_svc.list_corrective_actions(_tenant(), status=request.args.get("status")))
	return jsonify({"items": [ca.model_dump(mode="json") for ca in items], "count": len(items)})


@bp.post("/corrective-actions")
def create_corrective_action():
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	_dt(data, "due_date")
	try:
		ca = _run(_svc.create_corrective_action(CorrectiveActionCreate(**data)))
		return jsonify(ca.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.post("/corrective-actions/<ca_id>/complete")
def complete_corrective_action(ca_id: str):
	data = request.get_json(silent=True) or {}
	ca = _run(_svc.complete_corrective_action(_tenant(), ca_id, data.get("verified_by", "")))
	if ca is None:
		return _err("corrective_action_not_found", 404)
	return jsonify(ca.model_dump(mode="json"))
