"""Crop Management Flask Blueprint — agr_crp."""
from __future__ import annotations

import logging
from typing import Any

from flask import Blueprint, jsonify, request

from .service import CropManagementService

_log = logging.getLogger(__name__)

bp = Blueprint("agr_crp", __name__, url_prefix="/api/agriculture/crp")
_svc: dict[str, CropManagementService] = {}


def _get_svc(tenant_id: str = "default") -> CropManagementService:
	if tenant_id not in _svc:
		_svc[tenant_id] = CropManagementService(tenant_id=tenant_id)
	return _svc[tenant_id]


def _tenant() -> str:
	return request.headers.get("X-Tenant-ID", "default")


# ------------------------------------------------------------------ health

@bp.get("/health")
async def health():
	svc = _get_svc(_tenant())
	result = await svc.health_check()
	return jsonify(result), 200


# ------------------------------------------------------------------ varieties

@bp.get("/varieties")
async def list_varieties():
	svc = _get_svc(_tenant())
	crop_type = request.args.get("crop_type")
	limit = int(request.args.get("limit", 50))
	offset = int(request.args.get("offset", 0))
	items = await svc.list_varieties(crop_type=crop_type, limit=limit, offset=offset)
	return jsonify({"items": items, "count": len(items)}), 200


@bp.get("/varieties/<variety_id>")
async def get_variety(variety_id: str):
	try:
		svc = _get_svc(_tenant())
		item = await svc.get_variety(variety_id)
		return jsonify(item), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		_log.error("get_variety error: %s", exc)
		return jsonify({"error": "internal_error"}), 500


@bp.post("/varieties")
async def create_variety():
	try:
		svc = _get_svc(_tenant())
		payload = request.get_json(force=True) or {}
		item = await svc.create_variety(payload)
		return jsonify(item), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400
	except Exception as exc:
		_log.error("create_variety error: %s", exc)
		return jsonify({"error": "internal_error"}), 500


@bp.put("/varieties/<variety_id>")
async def update_variety(variety_id: str):
	try:
		svc = _get_svc(_tenant())
		payload = request.get_json(force=True) or {}
		item = await svc.update_variety(variety_id, payload)
		return jsonify(item), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		_log.error("update_variety error: %s", exc)
		return jsonify({"error": "internal_error"}), 500


@bp.delete("/varieties/<variety_id>")
async def delete_variety(variety_id: str):
	try:
		svc = _get_svc(_tenant())
		result = await svc.delete_variety(variety_id)
		return jsonify(result), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		_log.error("delete_variety error: %s", exc)
		return jsonify({"error": "internal_error"}), 500


# ------------------------------------------------------------------ planting calendars

@bp.get("/calendars")
async def list_calendars():
	svc = _get_svc(_tenant())
	region = request.args.get("region")
	crop_type = request.args.get("crop_type")
	items = await svc.list_planting_calendars(region=region, crop_type=crop_type)
	return jsonify({"items": items, "count": len(items)}), 200


@bp.get("/calendars/<calendar_id>")
async def get_calendar(calendar_id: str):
	try:
		svc = _get_svc(_tenant())
		item = await svc.get_planting_calendar(calendar_id)
		return jsonify(item), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/calendars")
async def create_calendar():
	try:
		svc = _get_svc(_tenant())
		payload = request.get_json(force=True) or {}
		item = await svc.create_planting_calendar(payload)
		return jsonify(item), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.put("/calendars/<calendar_id>")
async def update_calendar(calendar_id: str):
	try:
		svc = _get_svc(_tenant())
		payload = request.get_json(force=True) or {}
		item = await svc.update_planting_calendar(calendar_id, payload)
		return jsonify(item), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.delete("/calendars/<calendar_id>")
async def delete_calendar(calendar_id: str):
	try:
		svc = _get_svc(_tenant())
		result = await svc.delete_planting_calendar(calendar_id)
		return jsonify(result), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/calendars/recommend")
async def recommend_window():
	svc = _get_svc(_tenant())
	crop_type = request.args.get("crop_type", "")
	region = request.args.get("region", "")
	result = await svc.recommend_planting_window(crop_type, region)
	return jsonify(result), 200


# ------------------------------------------------------------------ crops

@bp.get("/crops")
async def list_crops():
	svc = _get_svc(_tenant())
	items = await svc.list_crops(
		farm_parcel_id=request.args.get("farm_parcel_id"),
		season=request.args.get("season"),
		status=request.args.get("status"),
		limit=int(request.args.get("limit", 50)),
		offset=int(request.args.get("offset", 0)),
	)
	return jsonify({"items": items, "count": len(items)}), 200


@bp.get("/crops/<crop_id>")
async def get_crop(crop_id: str):
	try:
		svc = _get_svc(_tenant())
		item = await svc.get_crop(crop_id)
		return jsonify(item), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/crops")
async def create_crop():
	try:
		svc = _get_svc(_tenant())
		payload = request.get_json(force=True) or {}
		item = await svc.create_crop(payload)
		return jsonify(item), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.put("/crops/<crop_id>")
async def update_crop(crop_id: str):
	try:
		svc = _get_svc(_tenant())
		payload = request.get_json(force=True) or {}
		item = await svc.update_crop(crop_id, payload)
		return jsonify(item), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.delete("/crops/<crop_id>")
async def delete_crop(crop_id: str):
	try:
		svc = _get_svc(_tenant())
		result = await svc.delete_crop(crop_id)
		return jsonify(result), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


# ------------------------------------------------------------------ phenology

@bp.get("/crops/<crop_id>/phenology")
async def list_phenology(crop_id: str):
	svc = _get_svc(_tenant())
	items = await svc.list_phenology(crop_id)
	return jsonify({"items": items, "count": len(items)}), 200


@bp.post("/phenology")
async def record_phenology():
	try:
		svc = _get_svc(_tenant())
		payload = request.get_json(force=True) or {}
		item = await svc.record_phenology(payload)
		return jsonify(item), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


# ------------------------------------------------------------------ rotation plans

@bp.get("/rotation-plans")
async def list_rotation_plans():
	svc = _get_svc(_tenant())
	items = await svc.list_rotation_plans(
		farm_parcel_id=request.args.get("farm_parcel_id")
	)
	return jsonify({"items": items, "count": len(items)}), 200


@bp.post("/rotation-plans")
async def create_rotation_plan():
	try:
		svc = _get_svc(_tenant())
		payload = request.get_json(force=True) or {}
		item = await svc.create_rotation_plan(payload)
		return jsonify(item), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


# ------------------------------------------------------------------ yield records

@bp.get("/yields")
async def list_yields():
	svc = _get_svc(_tenant())
	items = await svc.list_yield_records(
		crop_id=request.args.get("crop_id"),
		limit=int(request.args.get("limit", 50)),
	)
	return jsonify({"items": items, "count": len(items)}), 200


@bp.post("/yields")
async def create_yield():
	try:
		svc = _get_svc(_tenant())
		payload = request.get_json(force=True) or {}
		item = await svc.create_yield_record(payload)
		return jsonify(item), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


# ------------------------------------------------------------------ audit

@bp.get("/audit")
async def get_audit():
	svc = _get_svc(_tenant())
	events = await svc.get_audit_events(limit=int(request.args.get("limit", 100)))
	return jsonify({"events": events, "count": len(events)}), 200
