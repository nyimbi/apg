"""Flask Blueprint REST API for APG Medical Device Management."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from flask import Blueprint, jsonify, request

from .models import AdverseEventCreate, CalibrationRecordCreate, DeviceCreate, MaintenanceScheduleCreate
from .service import MedicalDeviceManagementService, PolicyViolationError

bp = Blueprint("healthcare_dev", __name__, url_prefix="/api/healthcare/dev")
_svc = MedicalDeviceManagementService()


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


@bp.get("/inventory")
def list_devices():
	devices = _run(_svc.list_devices(_tenant(), device_type=request.args.get("device_type"), status=request.args.get("status")))
	return jsonify({"items": [d.model_dump(mode="json") for d in devices], "count": len(devices)})


@bp.post("/inventory")
def register_device():
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	for f in ("purchase_date", "warranty_expiry"):
		_dt(data, f)
	try:
		device = _run(_svc.register_device(DeviceCreate(**data)))
		return jsonify(device.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.get("/inventory/<device_id>")
def get_device(device_id: str):
	device = _run(_svc.get_device(_tenant(), device_id))
	if device is None:
		return _err("device_not_found", 404)
	return jsonify(device.model_dump(mode="json"))


@bp.put("/inventory/<device_id>/status")
def update_device_status(device_id: str):
	data = request.get_json(silent=True) or {}
	try:
		device = _run(_svc.update_device_status(_tenant(), device_id, data.get("status", "")))
		if device is None:
			return _err("device_not_found", 404)
		return jsonify(device.model_dump(mode="json"))
	except PolicyViolationError as e:
		return _err(str(e), 403)


@bp.get("/maintenance")
def list_maintenance():
	items = _run(_svc.list_maintenance(_tenant(), device_id=request.args.get("device_id"), status=request.args.get("status")))
	return jsonify({"items": [m.model_dump(mode="json") for m in items], "count": len(items)})


@bp.post("/maintenance")
def schedule_maintenance():
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	_dt(data, "scheduled_date")
	try:
		sched = _run(_svc.schedule_maintenance(MaintenanceScheduleCreate(**data)))
		return jsonify(sched.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.post("/maintenance/<sched_id>/complete")
def complete_maintenance(sched_id: str):
	data = request.get_json(silent=True) or {}
	sched = _run(_svc.complete_maintenance(_tenant(), sched_id, data.get("notes", "")))
	if sched is None:
		return _err("maintenance_schedule_not_found", 404)
	return jsonify(sched.model_dump(mode="json"))


@bp.get("/calibration")
def list_calibrations():
	cals = _run(_svc.list_calibrations(_tenant(), device_id=request.args.get("device_id")))
	return jsonify({"items": [c.model_dump(mode="json") for c in cals], "count": len(cals)})


@bp.post("/calibration")
def record_calibration():
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	for f in ("calibration_date", "next_due_date"):
		_dt(data, f)
	try:
		cal = _run(_svc.record_calibration(CalibrationRecordCreate(**data)))
		return jsonify(cal.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.get("/adverse-events")
def list_adverse_events():
	events = _run(_svc.list_adverse_events(_tenant(), device_id=request.args.get("device_id"), severity=request.args.get("severity")))
	return jsonify({"items": [e.model_dump(mode="json") for e in events], "count": len(events)})


@bp.post("/adverse-events")
def report_adverse_event():
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	_dt(data, "occurred_at")
	try:
		event = _run(_svc.report_adverse_event(AdverseEventCreate(**data)))
		return jsonify(event.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.post("/adverse-events/<event_id>/close")
def close_adverse_event(event_id: str):
	data = request.get_json(silent=True) or {}
	event = _run(_svc.close_adverse_event(_tenant(), event_id, data.get("root_cause", ""), data.get("corrective_action", "")))
	if event is None:
		return _err("adverse_event_not_found", 404)
	return jsonify(event.model_dump(mode="json"))
