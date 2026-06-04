"""Flask Blueprint REST API for APG Telemedicine."""

from __future__ import annotations
import asyncio
from datetime import datetime
from typing import Any
from flask import Blueprint, jsonify, request
from .models import ConsultationCreate, PrescriptionTransmitCreate, RemoteMonitoringEnrollmentCreate, TeleBillingCreate, TeleSessionCreate
from .service import TelemedicineService, PolicyViolationError

bp = Blueprint("healthcare_tel", __name__, url_prefix="/api/healthcare/tel")
_svc = TelemedicineService()


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


@bp.get("/schedule")
def list_consultations():
	items = _run(_svc.list_consultations(_tenant(), patient_id=request.args.get("patient_id"), status=request.args.get("status")))
	return jsonify({"items": [c.model_dump(mode="json") for c in items], "count": len(items)})


@bp.post("/schedule")
def book_consultation():
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	_dt(data, "scheduled_at")
	try:
		consult = _run(_svc.book_consultation(ConsultationCreate(**data)))
		return jsonify(consult.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.get("/schedule/<consult_id>")
def get_consultation(consult_id: str):
	consult = _run(_svc.get_consultation(_tenant(), consult_id))
	if consult is None:
		return _err("consultation_not_found", 404)
	return jsonify(consult.model_dump(mode="json"))


@bp.post("/schedule/<consult_id>/cancel")
def cancel_consultation(consult_id: str):
	consult = _run(_svc.cancel_consultation(_tenant(), consult_id))
	if consult is None:
		return _err("consultation_not_found", 404)
	return jsonify(consult.model_dump(mode="json"))


@bp.get("/sessions")
def list_sessions():
	items = _run(_svc.list_sessions(_tenant(), patient_id=request.args.get("patient_id")))
	return jsonify({"items": [s.model_dump(mode="json") for s in items], "count": len(items)})


@bp.post("/sessions")
def create_session():
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	try:
		session = _run(_svc.create_session(TeleSessionCreate(**data)))
		return jsonify(session.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.get("/sessions/<session_id>")
def get_session(session_id: str):
	session = _run(_svc.get_session(_tenant(), session_id))
	if session is None:
		return _err("session_not_found", 404)
	return jsonify(session.model_dump(mode="json"))


@bp.post("/sessions/<session_id>/complete")
def complete_session(session_id: str):
	session = _run(_svc.complete_session(_tenant(), session_id))
	if session is None:
		return _err("session_not_found", 404)
	return jsonify(session.model_dump(mode="json"))


@bp.get("/monitoring")
def list_monitoring():
	items = _run(_svc.list_monitoring(_tenant(), patient_id=request.args.get("patient_id")))
	return jsonify({"items": [m.model_dump(mode="json") for m in items], "count": len(items)})


@bp.post("/monitoring")
def enroll_monitoring():
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	try:
		enrollment = _run(_svc.enroll_monitoring(RemoteMonitoringEnrollmentCreate(**data)))
		return jsonify(enrollment.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.get("/prescriptions")
def list_prescriptions():
	items = _run(_svc.list_prescriptions(_tenant(), patient_id=request.args.get("patient_id")))
	return jsonify({"items": [p.model_dump(mode="json") for p in items], "count": len(items)})


@bp.post("/prescriptions")
def transmit_prescription():
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	try:
		rx = _run(_svc.transmit_prescription(PrescriptionTransmitCreate(**data)))
		return jsonify(rx.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.get("/billing")
def list_billing():
	items = _run(_svc.list_billing(_tenant(), patient_id=request.args.get("patient_id")))
	return jsonify({"items": [b.model_dump(mode="json") for b in items], "count": len(items)})


@bp.post("/billing")
def create_billing():
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	try:
		bill = _run(_svc.create_billing_record(TeleBillingCreate(**data)))
		return jsonify(bill.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)
