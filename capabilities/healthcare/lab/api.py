"""Flask Blueprint REST API for APG Laboratory Information System."""

from __future__ import annotations

import asyncio
from typing import Any

from flask import Blueprint, jsonify, request

from .models import InstrumentCreate, LabOrderCreate, LabResultCreate, QCRunCreate, SpecimenCreate
from .service import LaboratoryInformationService, PolicyViolationError

bp = Blueprint("healthcare_lab", __name__, url_prefix="/api/healthcare/lab")
_svc = LaboratoryInformationService()


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


@bp.get("/contract")
def get_contract():
	return jsonify(_run(_svc.describe(_tenant())))


@bp.get("/dashboard")
def dashboard():
	return jsonify(_run(_svc.dashboard_summary(_tenant())))


# ── orders ────────────────────────────────────────────────────────────────────

@bp.get("/orders")
def list_orders():
	orders = _run(_svc.list_orders(_tenant(), patient_id=request.args.get("patient_id"), status=request.args.get("status")))
	return jsonify({"items": [o.model_dump(mode="json") for o in orders], "count": len(orders)})


@bp.post("/orders")
def create_order():
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	try:
		order = _run(_svc.create_order(LabOrderCreate(**data)))
		return jsonify(order.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.get("/orders/<order_id>")
def get_order(order_id: str):
	order = _run(_svc.get_order(_tenant(), order_id))
	if order is None:
		return _err("order_not_found", 404)
	return jsonify(order.model_dump(mode="json"))


@bp.post("/orders/<order_id>/cancel")
def cancel_order(order_id: str):
	data = request.get_json(silent=True) or {}
	order = _run(_svc.cancel_order(_tenant(), order_id, data.get("reason", "")))
	if order is None:
		return _err("order_not_found", 404)
	return jsonify(order.model_dump(mode="json"))


# ── specimens ─────────────────────────────────────────────────────────────────

@bp.get("/specimens")
def list_specimens():
	specimens = _run(_svc.list_specimens(_tenant(), order_id=request.args.get("order_id"), status=request.args.get("status")))
	return jsonify({"items": [s.model_dump(mode="json") for s in specimens], "count": len(specimens)})


@bp.post("/specimens")
def collect_specimen():
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	try:
		spec = _run(_svc.collect_specimen(SpecimenCreate(**data)))
		return jsonify(spec.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.get("/specimens/<specimen_id>")
def get_specimen(specimen_id: str):
	spec = _run(_svc.get_specimen(_tenant(), specimen_id))
	if spec is None:
		return _err("specimen_not_found", 404)
	return jsonify(spec.model_dump(mode="json"))


@bp.post("/specimens/<specimen_id>/reject")
def reject_specimen(specimen_id: str):
	data = request.get_json(silent=True) or {}
	try:
		spec = _run(_svc.reject_specimen(_tenant(), specimen_id, data.get("rejection_reason", "")))
		if spec is None:
			return _err("specimen_not_found", 404)
		return jsonify(spec.model_dump(mode="json"))
	except PolicyViolationError as e:
		return _err(str(e), 403)


@bp.post("/specimens/<specimen_id>/receive")
def receive_specimen(specimen_id: str):
	spec = _run(_svc.receive_specimen(_tenant(), specimen_id))
	if spec is None:
		return _err("specimen_not_found", 404)
	return jsonify(spec.model_dump(mode="json"))


# ── results ───────────────────────────────────────────────────────────────────

@bp.get("/results")
def list_results():
	critical_only = request.args.get("critical_only", "false").lower() == "true"
	results = _run(_svc.list_results(_tenant(), order_id=request.args.get("order_id"), critical_only=critical_only))
	return jsonify({"items": [r.model_dump(mode="json") for r in results], "count": len(results)})


@bp.post("/results")
def enter_result():
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	try:
		result = _run(_svc.enter_result(LabResultCreate(**data)))
		return jsonify(result.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.get("/results/<result_id>")
def get_result(result_id: str):
	result = _run(_svc.get_result(_tenant(), result_id))
	if result is None:
		return _err("result_not_found", 404)
	return jsonify(result.model_dump(mode="json"))


@bp.post("/results/<result_id>/verify")
def verify_result(result_id: str):
	data = request.get_json(silent=True) or {}
	try:
		result = _run(_svc.verify_result(_tenant(), result_id, data.get("verifier_id", ""), data.get("notification_sent", False)))
		if result is None:
			return _err("result_not_found", 404)
		return jsonify(result.model_dump(mode="json"))
	except PolicyViolationError as e:
		return _err(str(e), 403)


# ── critical values ───────────────────────────────────────────────────────────

@bp.get("/critical-values")
def list_critical_values():
	unack_only = request.args.get("unacknowledged_only", "false").lower() == "true"
	notifs = _run(_svc.list_critical_values(_tenant(), unacknowledged_only=unack_only))
	return jsonify({"items": [n.model_dump(mode="json") for n in notifs], "count": len(notifs)})


@bp.post("/critical-values")
def notify_critical_value():
	data = request.get_json(silent=True) or {}
	try:
		notif = _run(_svc.notify_critical_value(
			_tenant(), data["result_id"], data["patient_id"], data["analyte"],
			data["value"], data["unit"], data.get("severity", "critical_high"),
			data["notified_to"], data["notified_by"],
		))
		return jsonify(notif.model_dump(mode="json")), 201
	except KeyError as e:
		return _err(f"missing field: {e}")


@bp.post("/critical-values/<notif_id>/acknowledge")
def acknowledge_critical_value(notif_id: str):
	data = request.get_json(silent=True) or {}
	try:
		notif = _run(_svc.acknowledge_critical_value(_tenant(), notif_id, data.get("acknowledged_by", "")))
		if notif is None:
			return _err("notification_not_found", 404)
		return jsonify(notif.model_dump(mode="json"))
	except PolicyViolationError as e:
		return _err(str(e), 403)


# ── QC ────────────────────────────────────────────────────────────────────────

@bp.get("/qc")
def list_qc_runs():
	qc_runs = _run(_svc.list_qc_runs(_tenant(), instrument_id=request.args.get("instrument_id")))
	return jsonify({"items": [q.model_dump(mode="json") for q in qc_runs], "count": len(qc_runs)})


@bp.post("/qc")
def run_qc():
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	try:
		qc = _run(_svc.run_qc(QCRunCreate(**data)))
		return jsonify(qc.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


# ── instruments ───────────────────────────────────────────────────────────────

@bp.get("/instruments")
def list_instruments():
	instruments = _run(_svc.list_instruments(_tenant()))
	return jsonify({"items": [i.model_dump(mode="json") for i in instruments], "count": len(instruments)})


@bp.post("/instruments")
def register_instrument():
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	try:
		inst = _run(_svc.register_instrument(InstrumentCreate(**data)))
		return jsonify(inst.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.put("/instruments/<instrument_id>/status")
def update_instrument_status(instrument_id: str):
	data = request.get_json(silent=True) or {}
	try:
		inst = _run(_svc.update_instrument_status(_tenant(), instrument_id, data.get("status", "")))
		if inst is None:
			return _err("instrument_not_found", 404)
		return jsonify(inst.model_dump(mode="json"))
	except PolicyViolationError as e:
		return _err(str(e), 403)
