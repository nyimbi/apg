"""AgriIoT & Precision Farming Flask Blueprint — agr_iot."""
from __future__ import annotations
import logging
from flask import Blueprint, jsonify, request
from .service import AgriIoTService

_log = logging.getLogger(__name__)
bp = Blueprint("agr_iot", __name__, url_prefix="/api/agriculture/iot")
_svc: dict[str, AgriIoTService] = {}


def _get_svc(t: str = "default") -> AgriIoTService:
	if t not in _svc:
		_svc[t] = AgriIoTService(tenant_id=t)
	return _svc[t]


def _t() -> str:
	return request.headers.get("X-Tenant-ID", "default")


@bp.get("/health")
async def health():
	return jsonify(await _get_svc(_t()).health_check()), 200


# --- devices ---

@bp.get("/devices")
async def list_devices():
	svc = _get_svc(_t())
	active_str = request.args.get("active")
	active = None if active_str is None else active_str.lower() == "true"
	items = await svc.list_devices(
		farm_parcel_id=request.args.get("farm_parcel_id"),
		device_type=request.args.get("device_type"),
		active=active,
	)
	return jsonify({"items": items, "count": len(items)}), 200


@bp.post("/devices")
async def register_device():
	try:
		return jsonify(await _get_svc(_t()).register_device(request.get_json(force=True) or {})), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/devices/<device_id>")
async def get_device(device_id: str):
	try:
		return jsonify(await _get_svc(_t()).get_device(device_id)), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.put("/devices/<device_id>")
async def update_device(device_id: str):
	try:
		return jsonify(await _get_svc(_t()).update_device(device_id, request.get_json(force=True) or {})), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.delete("/devices/<device_id>")
async def delete_device(device_id: str):
	try:
		return jsonify(await _get_svc(_t()).delete_device(device_id)), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


# --- telemetry ---

@bp.post("/telemetry")
async def ingest_telemetry():
	try:
		return jsonify(await _get_svc(_t()).ingest_telemetry(request.get_json(force=True) or {})), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/devices/<device_id>/telemetry")
async def list_telemetry(device_id: str):
	items = await _get_svc(_t()).list_telemetry(device_id, limit=int(request.args.get("limit", 100)))
	return jsonify({"items": items, "count": len(items)}), 200


@bp.get("/field-health/<farm_parcel_id>")
async def field_health(farm_parcel_id: str):
	return jsonify(await _get_svc(_t()).get_field_health_snapshot(farm_parcel_id)), 200


# --- drone imagery ---

@bp.get("/imagery")
async def list_imagery():
	svc = _get_svc(_t())
	items = await svc.list_imagery(
		farm_parcel_id=request.args.get("farm_parcel_id"),
		imagery_type=request.args.get("imagery_type"),
	)
	return jsonify({"items": items, "count": len(items)}), 200


@bp.post("/imagery")
async def upload_imagery():
	try:
		return jsonify(await _get_svc(_t()).upload_imagery(request.get_json(force=True) or {})), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/imagery/<image_id>")
async def get_imagery(image_id: str):
	try:
		return jsonify(await _get_svc(_t()).get_imagery(image_id)), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.delete("/imagery/<image_id>")
async def delete_imagery(image_id: str):
	try:
		return jsonify(await _get_svc(_t()).delete_imagery(image_id)), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/ndvi-trend/<farm_parcel_id>")
async def ndvi_trend(farm_parcel_id: str):
	return jsonify(await _get_svc(_t()).analyse_ndvi_trend(farm_parcel_id)), 200


# --- yield maps ---

@bp.get("/yield-maps")
async def list_yield_maps():
	svc = _get_svc(_t())
	items = await svc.list_yield_maps(
		farm_parcel_id=request.args.get("farm_parcel_id"),
		season=request.args.get("season"),
	)
	return jsonify({"items": items, "count": len(items)}), 200


@bp.post("/yield-maps")
async def create_yield_map():
	try:
		return jsonify(await _get_svc(_t()).create_yield_map(request.get_json(force=True) or {})), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.delete("/yield-maps/<map_id>")
async def delete_yield_map(map_id: str):
	try:
		return jsonify(await _get_svc(_t()).delete_yield_map(map_id)), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


# --- prescriptions ---

@bp.get("/prescriptions")
async def list_prescriptions():
	svc = _get_svc(_t())
	applied_str = request.args.get("applied")
	applied = None if applied_str is None else applied_str.lower() == "true"
	items = await svc.list_prescriptions(farm_parcel_id=request.args.get("farm_parcel_id"), applied=applied)
	return jsonify({"items": items, "count": len(items)}), 200


@bp.post("/prescriptions")
async def create_prescription():
	try:
		return jsonify(await _get_svc(_t()).create_prescription(request.get_json(force=True) or {})), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/prescriptions/<prescription_id>")
async def get_prescription(prescription_id: str):
	try:
		return jsonify(await _get_svc(_t()).get_prescription(prescription_id)), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/prescriptions/<prescription_id>/apply")
async def apply_prescription(prescription_id: str):
	try:
		return jsonify(await _get_svc(_t()).mark_prescription_applied(prescription_id)), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/prescriptions/generate-from-ndvi")
async def generate_from_ndvi():
	try:
		body = request.get_json(force=True) or {}
		result = await _get_svc(_t()).generate_prescription_from_ndvi(
			farm_parcel_id=body["farm_parcel_id"],
			application_type=body["application_type"],
			base_rate=float(body["base_rate"]),
			unit=body["unit"],
		)
		return jsonify(result), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.delete("/prescriptions/<prescription_id>")
async def delete_prescription(prescription_id: str):
	try:
		return jsonify(await _get_svc(_t()).delete_prescription(prescription_id)), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/summary/<farm_parcel_id>")
async def precision_summary(farm_parcel_id: str):
	return jsonify(await _get_svc(_t()).get_precision_farming_summary(farm_parcel_id)), 200


@bp.get("/audit")
async def get_audit():
	events = await _get_svc(_t()).get_audit_events(int(request.args.get("limit", 100)))
	return jsonify({"events": events, "count": len(events)}), 200
