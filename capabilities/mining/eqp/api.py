"""REST API Blueprint for APG Equipment & Plant Management."""

from __future__ import annotations

import asyncio
from typing import Any

from flask import Blueprint, g, jsonify, request

from .models import (
	EquipmentCreate,
	EquipmentFaultCreate,
	EquipmentUpdate,
	FuelDocketCreate,
	InspectionCreate,
	MaintenanceWorkOrderCreate,
	MaintenanceWorkOrderUpdate,
)
from .service import EqpService

api_bp = Blueprint("mining_eqp_api", __name__, url_prefix="/api/mining-eqp")


def _svc() -> EqpService:
	return EqpService(tenant_id=getattr(g, "tenant_id", "default"))


def _loop() -> asyncio.AbstractEventLoop:
	return asyncio.get_event_loop()


def _err(msg: str, code: int = 400) -> tuple[Any, int]:
	return jsonify({"error": msg}), code


# ── Fleet ──────────────────────────────────────────────────────────────────────

@api_bp.get("/fleet")
def list_fleet():
	"""List all equipment in the fleet."""
	svc = _svc()
	results = _loop().run_until_complete(
		svc.list_equipment(
			equipment_class=request.args.get("equipment_class"),
			lifecycle_status=request.args.get("lifecycle_status"),
			dispatch_status=request.args.get("dispatch_status"),
			mine_area=request.args.get("mine_area"),
			limit=int(request.args.get("limit", 200)),
			offset=int(request.args.get("offset", 0)),
		)
	)
	return jsonify({"count": len(results), "items": [r.model_dump() for r in results]})


@api_bp.post("/fleet")
def register_equipment():
	"""Register new equipment."""
	svc = _svc()
	data = request.get_json(force=True) or {}
	data["tenant_id"] = getattr(g, "tenant_id", "default")
	try:
		payload = EquipmentCreate(**data)
		result = _loop().run_until_complete(
			svc.register_equipment(payload, created_by=getattr(g, "user_id", "system"))
		)
		return jsonify(result.model_dump()), 201
	except (ValueError, AssertionError) as exc:
		return _err(str(exc))


@api_bp.get("/fleet/<string:id>")
def get_equipment(id: str):
	"""Get equipment by id."""
	svc = _svc()
	result = _loop().run_until_complete(svc.get_equipment(id))
	if result is None:
		return _err("Not found", 404)
	return jsonify(result.model_dump())


@api_bp.put("/fleet/<string:id>")
def update_equipment(id: str):
	"""Update equipment status or assignment."""
	svc = _svc()
	data = request.get_json(force=True) or {}
	try:
		payload = EquipmentUpdate(**data)
		result = _loop().run_until_complete(svc.update_equipment(id, payload))
		return jsonify(result.model_dump())
	except (KeyError, ValueError) as exc:
		return _err(str(exc), 404 if isinstance(exc, KeyError) else 400)


@api_bp.post("/fleet/<string:id>/decommission")
def decommission_equipment(id: str):
	"""Decommission equipment."""
	svc = _svc()
	data = request.get_json(force=True) or {}
	approved_by = data.get("approved_by", getattr(g, "user_id", "system"))
	try:
		result = _loop().run_until_complete(svc.decommission_equipment(id, approved_by))
		return jsonify(result.model_dump())
	except (KeyError, ValueError) as exc:
		return _err(str(exc), 404 if isinstance(exc, KeyError) else 400)


@api_bp.post("/fleet/<string:id>/dispatch")
def dispatch_equipment(id: str):
	"""Dispatch equipment to a mine area."""
	svc = _svc()
	data = request.get_json(force=True) or {}
	operator_id = data.get("operator_id", getattr(g, "user_id", "system"))
	operator_licensed = bool(data.get("operator_licensed", False))
	destination_area = data.get("destination_area", "")
	if not destination_area:
		return _err("destination_area required")
	try:
		result = _loop().run_until_complete(
			svc.dispatch_equipment(id, operator_id, operator_licensed, destination_area)
		)
		return jsonify(result.model_dump())
	except (KeyError, PermissionError) as exc:
		return _err(str(exc), 404 if isinstance(exc, KeyError) else 403)


# ── Maintenance Work Orders ────────────────────────────────────────────────────

@api_bp.get("/maintenance")
def list_work_orders():
	"""List maintenance work orders."""
	svc = _svc()
	results = _loop().run_until_complete(
		svc.list_work_orders(
			equipment_id=request.args.get("equipment_id"),
			status=request.args.get("status"),
		)
	)
	return jsonify({"count": len(results), "items": [r.model_dump() for r in results]})


@api_bp.post("/maintenance")
def create_work_order():
	"""Create a maintenance work order."""
	svc = _svc()
	data = request.get_json(force=True) or {}
	try:
		payload = MaintenanceWorkOrderCreate(**data)
		result = _loop().run_until_complete(
			svc.create_work_order(payload, created_by=getattr(g, "user_id", "system"))
		)
		return jsonify(result.model_dump()), 201
	except (KeyError, ValueError, AssertionError) as exc:
		return _err(str(exc), 404 if isinstance(exc, KeyError) else 400)


@api_bp.post("/maintenance/<string:id>/approve")
def approve_work_order(id: str):
	"""Approve a work order."""
	svc = _svc()
	data = request.get_json(force=True) or {}
	approver_id = data.get("approver_id", getattr(g, "user_id", "system"))
	try:
		result = _loop().run_until_complete(svc.approve_work_order(id, approver_id))
		return jsonify(result.model_dump())
	except KeyError as exc:
		return _err(str(exc), 404)


@api_bp.put("/maintenance/<string:id>/complete")
def complete_work_order(id: str):
	"""Complete a work order."""
	svc = _svc()
	data = request.get_json(force=True) or {}
	try:
		payload = MaintenanceWorkOrderUpdate(**data)
		result = _loop().run_until_complete(svc.complete_work_order(id, payload))
		return jsonify(result.model_dump())
	except (KeyError, PermissionError, ValueError) as exc:
		code = 404 if isinstance(exc, KeyError) else 403 if isinstance(exc, PermissionError) else 400
		return _err(str(exc), code)


# ── Inspections ────────────────────────────────────────────────────────────────

@api_bp.post("/inspections")
def submit_inspection():
	"""Submit an equipment inspection."""
	svc = _svc()
	data = request.get_json(force=True) or {}
	try:
		payload = InspectionCreate(**data)
		result = _loop().run_until_complete(
			svc.submit_inspection(payload, created_by=getattr(g, "user_id", "system"))
		)
		return jsonify(result.model_dump()), 201
	except (KeyError, ValueError, AssertionError) as exc:
		return _err(str(exc), 404 if isinstance(exc, KeyError) else 400)


@api_bp.get("/inspections")
def list_inspections():
	"""List inspections, optionally filtered by equipment_id."""
	svc = _svc()
	equipment_id = request.args.get("equipment_id")
	if equipment_id:
		results = _loop().run_until_complete(svc.list_inspections_for_equipment(equipment_id))
	else:
		results = list(svc._inspections.values())
		return jsonify({"count": len(results), "items": results})
	return jsonify({"count": len(results), "items": [r.model_dump() for r in results]})


# ── Fuel Dockets ───────────────────────────────────────────────────────────────

@api_bp.post("/fuel")
def record_fuel():
	"""Record a fuel docket."""
	svc = _svc()
	data = request.get_json(force=True) or {}
	try:
		payload = FuelDocketCreate(**data)
		result = _loop().run_until_complete(
			svc.record_fuel_docket(payload, created_by=getattr(g, "user_id", "system"))
		)
		return jsonify(result.model_dump()), 201
	except (KeyError, ValueError, AssertionError) as exc:
		return _err(str(exc), 404 if isinstance(exc, KeyError) else 400)


@api_bp.get("/fuel")
def list_fuel():
	"""List fuel dockets, optionally filtered by equipment_id."""
	svc = _svc()
	equipment_id = request.args.get("equipment_id")
	results = [
		r for r in svc._fuel_dockets.values()
		if not equipment_id or r["equipment_id"] == equipment_id
	]
	return jsonify({"count": len(results), "items": results})


# ── Equipment Faults ───────────────────────────────────────────────────────────

@api_bp.post("/faults")
def report_fault():
	"""Report an equipment fault."""
	svc = _svc()
	data = request.get_json(force=True) or {}
	try:
		payload = EquipmentFaultCreate(**data)
		result = _loop().run_until_complete(
			svc.report_fault(payload, created_by=getattr(g, "user_id", "system"))
		)
		return jsonify(result.model_dump()), 201
	except (KeyError, ValueError, AssertionError) as exc:
		return _err(str(exc), 404 if isinstance(exc, KeyError) else 400)


@api_bp.get("/faults")
def list_faults():
	"""List equipment faults."""
	svc = _svc()
	equipment_id = request.args.get("equipment_id")
	open_only = request.args.get("open_only", "true").lower() == "true"
	results = [
		r for r in svc._faults.values()
		if (not equipment_id or r["equipment_id"] == equipment_id)
		and (not open_only or not r.get("resolved"))
	]
	return jsonify({"count": len(results), "items": results})


@api_bp.post("/faults/<string:id>/resolve")
def resolve_fault(id: str):
	"""Resolve an equipment fault."""
	svc = _svc()
	data = request.get_json(force=True) or {}
	try:
		result = _loop().run_until_complete(
			svc.resolve_fault(id, work_order_id=data.get("work_order_id"))
		)
		return jsonify(result.model_dump())
	except KeyError as exc:
		return _err(str(exc), 404)


# ── KPI Summary ────────────────────────────────────────────────────────────────

@api_bp.get("/kpis")
def fleet_kpis():
	"""Fleet KPI summary."""
	svc = _svc()
	kpis = _loop().run_until_complete(svc.get_fleet_kpi_summary())
	return jsonify(kpis)
