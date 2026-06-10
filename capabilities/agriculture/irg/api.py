"""Irrigation Management Flask Blueprint — agr_irg."""
from __future__ import annotations
import logging
from flask import Blueprint, jsonify, request
from .service import IrrigationManagementService

_log = logging.getLogger(__name__)
bp = Blueprint("agr_irg", __name__, url_prefix="/api/agriculture/irg")
_svc: dict[str, IrrigationManagementService] = {}


def _get_svc(t: str = "default") -> IrrigationManagementService:
	if t not in _svc:
		_svc[t] = IrrigationManagementService(tenant_id=t)
	return _svc[t]


def _t() -> str:
	return request.headers.get("X-Tenant-ID", "default")


@bp.get("/health")
async def health():
	return jsonify(await _get_svc(_t()).health_check()), 200


# --- sensors ---

@bp.get("/sensors")
async def list_sensors():
	svc = _get_svc(_t())
	active_str = request.args.get("active")
	active = None if active_str is None else active_str.lower() == "true"
	items = await svc.list_sensors(farm_parcel_id=request.args.get("farm_parcel_id"), active=active)
	return jsonify({"items": items, "count": len(items)}), 200


@bp.get("/sensors/<sensor_id>")
async def get_sensor(sensor_id: str):
	try:
		return jsonify(await _get_svc(_t()).get_sensor(sensor_id)), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/sensors")
async def create_sensor():
	try:
		return jsonify(await _get_svc(_t()).create_sensor(request.get_json(force=True) or {})), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.put("/sensors/<sensor_id>")
async def update_sensor(sensor_id: str):
	try:
		return jsonify(await _get_svc(_t()).update_sensor(sensor_id, request.get_json(force=True) or {})), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.delete("/sensors/<sensor_id>")
async def delete_sensor(sensor_id: str):
	try:
		return jsonify(await _get_svc(_t()).delete_sensor(sensor_id)), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


# --- readings ---

@bp.post("/readings")
async def ingest_reading():
	try:
		return jsonify(await _get_svc(_t()).ingest_reading(request.get_json(force=True) or {})), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/sensors/<sensor_id>/readings")
async def list_readings(sensor_id: str):
	items = await _get_svc(_t()).list_readings(sensor_id, limit=int(request.args.get("limit", 100)))
	return jsonify({"items": items, "count": len(items)}), 200


@bp.get("/alerts")
async def get_alerts():
	items = await _get_svc(_t()).get_sensor_alerts(farm_parcel_id=request.args.get("farm_parcel_id"))
	return jsonify({"items": items, "count": len(items)}), 200


# --- schedules ---

@bp.get("/schedules")
async def list_schedules():
	svc = _get_svc(_t())
	items = await svc.list_schedules(
		farm_parcel_id=request.args.get("farm_parcel_id"),
		status=request.args.get("status"),
	)
	return jsonify({"items": items, "count": len(items)}), 200


@bp.post("/schedules")
async def create_schedule():
	try:
		return jsonify(await _get_svc(_t()).create_schedule(request.get_json(force=True) or {})), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.put("/schedules/<schedule_id>")
async def update_schedule(schedule_id: str):
	try:
		return jsonify(await _get_svc(_t()).update_schedule(schedule_id, request.get_json(force=True) or {})), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.delete("/schedules/<schedule_id>")
async def delete_schedule(schedule_id: str):
	try:
		return jsonify(await _get_svc(_t()).delete_schedule(schedule_id)), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/optimise")
async def optimise():
	svc = _get_svc(_t())
	result = await svc.optimise_schedule(
		farm_parcel_id=request.args.get("farm_parcel_id", ""),
		crop_type=request.args.get("crop_type", ""),
		soil_moisture_pct=float(request.args.get("soil_moisture_pct", 50)),
	)
	return jsonify(result), 200


# --- water accounts ---

@bp.get("/water-accounts")
async def list_water_accounts():
	svc = _get_svc(_t())
	items = await svc.list_water_accounts(
		farm_parcel_id=request.args.get("farm_parcel_id"),
		period=request.args.get("period"),
	)
	return jsonify({"items": items, "count": len(items)}), 200


@bp.post("/water-accounts/allocate")
async def allocate_water():
	try:
		body = request.get_json(force=True) or {}
		result = await _get_svc(_t()).set_water_allocation(
			body["farm_parcel_id"], body["period"], float(body["allocated_m3"])
		)
		return jsonify(result), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


# --- canals ---

@bp.get("/canals")
async def list_canals():
	items = await _get_svc(_t()).list_canals()
	return jsonify({"items": items, "count": len(items)}), 200


@bp.post("/canals")
async def create_canal():
	try:
		return jsonify(await _get_svc(_t()).create_canal(request.get_json(force=True) or {})), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.put("/canals/<canal_id>")
async def update_canal(canal_id: str):
	try:
		return jsonify(await _get_svc(_t()).update_canal(canal_id, request.get_json(force=True) or {})), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.delete("/canals/<canal_id>")
async def delete_canal(canal_id: str):
	try:
		return jsonify(await _get_svc(_t()).delete_canal(canal_id)), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/audit")
async def get_audit():
	events = await _get_svc(_t()).get_audit_events(int(request.args.get("limit", 100)))
	return jsonify({"events": events, "count": len(events)}), 200
