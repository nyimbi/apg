"""Farm Management System Flask Blueprint — agr_fms."""
from __future__ import annotations

import logging
from flask import Blueprint, jsonify, request
from .service import FarmManagementService

_log = logging.getLogger(__name__)
bp = Blueprint("agr_fms", __name__, url_prefix="/api/agriculture/fms")
_svc: dict[str, FarmManagementService] = {}


def _get_svc(tenant_id: str = "default") -> FarmManagementService:
	if tenant_id not in _svc:
		_svc[tenant_id] = FarmManagementService(tenant_id=tenant_id)
	return _svc[tenant_id]


def _tenant() -> str:
	return request.headers.get("X-Tenant-ID", "default")


@bp.get("/health")
async def health():
	return jsonify(await _get_svc(_tenant()).health_check()), 200


# --- parcels ---

@bp.get("/parcels")
async def list_parcels():
	svc = _get_svc(_tenant())
	items = await svc.list_parcels(
		status=request.args.get("status"),
		owner_id=request.args.get("owner_id"),
		limit=int(request.args.get("limit", 50)),
		offset=int(request.args.get("offset", 0)),
	)
	return jsonify({"items": items, "count": len(items)}), 200


@bp.get("/parcels/<parcel_id>")
async def get_parcel(parcel_id: str):
	try:
		return jsonify(await _get_svc(_tenant()).get_parcel(parcel_id)), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/parcels")
async def create_parcel():
	try:
		return jsonify(await _get_svc(_tenant()).create_parcel(request.get_json(force=True) or {})), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.put("/parcels/<parcel_id>")
async def update_parcel(parcel_id: str):
	try:
		return jsonify(await _get_svc(_tenant()).update_parcel(parcel_id, request.get_json(force=True) or {})), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.delete("/parcels/<parcel_id>")
async def delete_parcel(parcel_id: str):
	try:
		return jsonify(await _get_svc(_tenant()).delete_parcel(parcel_id)), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/parcels/<parcel_id>/summary")
async def parcel_summary(parcel_id: str):
	try:
		return jsonify(await _get_svc(_tenant()).get_parcel_summary(parcel_id)), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


# --- inputs ---

@bp.get("/inputs")
async def list_inputs():
	svc = _get_svc(_tenant())
	items = await svc.list_inputs(
		farm_parcel_id=request.args.get("farm_parcel_id"),
		category=request.args.get("category"),
		limit=int(request.args.get("limit", 50)),
	)
	return jsonify({"items": items, "count": len(items)}), 200


@bp.post("/inputs")
async def create_input():
	try:
		return jsonify(await _get_svc(_tenant()).create_input(request.get_json(force=True) or {})), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.delete("/inputs/<input_id>")
async def delete_input(input_id: str):
	try:
		return jsonify(await _get_svc(_tenant()).delete_input(input_id)), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


# --- labour ---

@bp.get("/labour")
async def list_labour():
	svc = _get_svc(_tenant())
	completed_str = request.args.get("completed")
	completed = None if completed_str is None else completed_str.lower() == "true"
	items = await svc.list_labour_schedules(
		farm_parcel_id=request.args.get("farm_parcel_id"),
		task_type=request.args.get("task_type"),
		completed=completed,
	)
	return jsonify({"items": items, "count": len(items)}), 200


@bp.post("/labour")
async def create_labour():
	try:
		return jsonify(await _get_svc(_tenant()).create_labour_schedule(request.get_json(force=True) or {})), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.put("/labour/<schedule_id>")
async def update_labour(schedule_id: str):
	try:
		return jsonify(await _get_svc(_tenant()).update_labour_schedule(schedule_id, request.get_json(force=True) or {})), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.delete("/labour/<schedule_id>")
async def delete_labour(schedule_id: str):
	try:
		return jsonify(await _get_svc(_tenant()).delete_labour_schedule(schedule_id)), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


# --- diary ---

@bp.get("/diary")
async def list_diary():
	svc = _get_svc(_tenant())
	items = await svc.list_diary_entries(
		farm_parcel_id=request.args.get("farm_parcel_id"),
		tag=request.args.get("tag"),
	)
	return jsonify({"items": items, "count": len(items)}), 200


@bp.post("/diary")
async def create_diary():
	try:
		return jsonify(await _get_svc(_tenant()).create_diary_entry(request.get_json(force=True) or {})), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.put("/diary/<entry_id>")
async def update_diary(entry_id: str):
	try:
		return jsonify(await _get_svc(_tenant()).update_diary_entry(entry_id, request.get_json(force=True) or {})), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.delete("/diary/<entry_id>")
async def delete_diary(entry_id: str):
	try:
		return jsonify(await _get_svc(_tenant()).delete_diary_entry(entry_id)), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


# --- costs ---

@bp.get("/costs")
async def get_costs():
	svc = _get_svc(_tenant())
	result = await svc.get_farm_cost_summary(
		farm_parcel_id=request.args.get("farm_parcel_id"),
		from_date=request.args.get("from_date"),
		to_date=request.args.get("to_date"),
	)
	return jsonify(result), 200


@bp.get("/audit")
async def get_audit():
	events = await _get_svc(_tenant()).get_audit_events(int(request.args.get("limit", 100)))
	return jsonify({"events": events, "count": len(events)}), 200
