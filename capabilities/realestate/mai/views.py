"""Flask Blueprint views for Facilities Maintenance (mai)."""

from __future__ import annotations

import asyncio
from functools import wraps
from typing import Any, Callable

from flask import Blueprint, request, jsonify

from .service import MaiService
from .models import (
	AssetCreate, AssetUpdate,
	PpmScheduleCreate,
	WorkOrderCreate, WorkOrderUpdate,
	MaintenanceContractorCreate,
	SlaCreate,
	InspectionCreate,
	DefectCreate,
)

bp = Blueprint("mai_views", __name__, url_prefix="/realestate/mai")
_svc = MaiService()


def _run(coro):
	return asyncio.get_event_loop().run_until_complete(coro)


def has_access(permission: str):
	def decorator(fn: Callable) -> Callable:
		@wraps(fn)
		def wrapper(*args, **kwargs):
			return fn(*args, **kwargs)
		return wrapper
	return decorator


def _tenant() -> str:
	return request.headers.get("X-Tenant-ID", "default")


def _ok(data: Any, status: int = 200):
	return jsonify({"status": "ok", "data": data}), status


def _err(msg: str, status: int = 400):
	return jsonify({"status": "error", "message": msg}), status


# ── Dashboard ─────────────────────────────────────────────────────────────────

@bp.get("/dashboard")
@has_access("realestate_mai:view")
def dashboard():
	return _ok(_run(_svc.get_sla_dashboard(_tenant())))


# ── Assets ────────────────────────────────────────────────────────────────────

@bp.get("/assets")
@has_access("realestate_mai:assets")
def list_assets():
	return _ok([a.model_dump() for a in _run(_svc.list_assets(_tenant(), request.args.get("property_id"), request.args.get("category"), request.args.get("status")))])


@bp.post("/assets")
@has_access("realestate_mai:assets")
def create_asset():
	try:
		return _ok(_run(_svc.register_asset(AssetCreate(**request.json, tenant_id=_tenant()))).model_dump(), 201)
	except Exception as e:
		return _err(str(e))


@bp.get("/assets/<asset_id>")
@has_access("realestate_mai:assets")
def get_asset(asset_id: str):
	r = _run(_svc.get_asset(asset_id, _tenant()))
	return _ok(r.model_dump()) if r else _err("not found", 404)


@bp.put("/assets/<asset_id>")
@has_access("realestate_mai:assets")
def update_asset(asset_id: str):
	try:
		r = _run(_svc.update_asset(asset_id, _tenant(), AssetUpdate(**request.json)))
		return _ok(r.model_dump()) if r else _err("not found", 404)
	except Exception as e:
		return _err(str(e))


@bp.get("/assets/end-of-life")
@has_access("realestate_mai:assets")
def end_of_life_assets():
	return _ok([a.model_dump() for a in _run(_svc.get_end_of_life_assets(_tenant(), request.args.get("property_id")))])


# ── PPM ───────────────────────────────────────────────────────────────────────

@bp.get("/ppm")
@has_access("realestate_mai:ppm")
def list_ppm():
	return _ok([p.model_dump() for p in _run(_svc.list_ppm_schedules(_tenant(), request.args.get("asset_id"), request.args.get("status")))])


@bp.post("/ppm")
@has_access("realestate_mai:ppm")
def create_ppm():
	try:
		return _ok(_run(_svc.create_ppm_schedule(PpmScheduleCreate(**request.json, tenant_id=_tenant()))).model_dump(), 201)
	except Exception as e:
		return _err(str(e))


@bp.post("/ppm/<ppm_id>/complete")
@has_access("realestate_mai:ppm")
def complete_ppm(ppm_id: str):
	completed_by = request.json.get("completed_by", "unknown") if request.json else "unknown"
	r = _run(_svc.complete_ppm(ppm_id, _tenant(), completed_by))
	return _ok(r.model_dump()) if r else _err("not found", 404)


@bp.get("/ppm/overdue")
@has_access("realestate_mai:ppm")
def overdue_ppm():
	return _ok([p.model_dump() for p in _run(_svc.get_overdue_ppms(_tenant()))])


# ── Work Orders ───────────────────────────────────────────────────────────────

@bp.get("/work-orders")
@has_access("realestate_mai:work_orders")
def list_work_orders():
	return _ok([w.model_dump() for w in _run(_svc.list_work_orders(_tenant(), request.args.get("property_id"), request.args.get("status"), request.args.get("priority")))])


@bp.post("/work-orders")
@has_access("realestate_mai:work_orders")
def create_work_order():
	try:
		return _ok(_run(_svc.raise_work_order(WorkOrderCreate(**request.json, tenant_id=_tenant()))).model_dump(), 201)
	except Exception as e:
		return _err(str(e))


@bp.post("/work-orders/<wo_id>/assign")
@has_access("realestate_mai:work_orders")
def assign_work_order(wo_id: str):
	try:
		contractor_id = request.json["contractor_id"]
		r = _run(_svc.assign_work_order(wo_id, _tenant(), contractor_id))
		return _ok(r.model_dump()) if r else _err("not found", 404)
	except Exception as e:
		return _err(str(e))


@bp.put("/work-orders/<wo_id>")
@has_access("realestate_mai:work_orders")
def update_work_order(wo_id: str):
	try:
		r = _run(_svc.update_work_order(wo_id, _tenant(), WorkOrderUpdate(**request.json)))
		return _ok(r.model_dump()) if r else _err("not found", 404)
	except Exception as e:
		return _err(str(e))


@bp.post("/work-orders/<wo_id>/close")
@has_access("realestate_mai:work_orders")
def close_work_order(wo_id: str):
	try:
		verified_by = request.json.get("verified_by", "unknown") if request.json else "unknown"
		r = _run(_svc.close_work_order(wo_id, _tenant(), verified_by))
		return _ok(r.model_dump()) if r else _err("not found", 404)
	except Exception as e:
		return _err(str(e))


# ── Contractors ───────────────────────────────────────────────────────────────

@bp.get("/contractors")
@has_access("realestate_mai:contractors")
def list_contractors():
	return _ok([c.model_dump() for c in _run(_svc.list_contractors(_tenant(), request.args.get("contractor_type")))])


@bp.post("/contractors")
@has_access("realestate_mai:contractors")
def create_contractor():
	try:
		return _ok(_run(_svc.register_contractor(MaintenanceContractorCreate(**request.json, tenant_id=_tenant()))).model_dump(), 201)
	except Exception as e:
		return _err(str(e))


# ── Inspections ───────────────────────────────────────────────────────────────

@bp.post("/inspections")
@has_access("realestate_mai:inspections")
def create_inspection():
	try:
		return _ok(_run(_svc.create_inspection(InspectionCreate(**request.json, tenant_id=_tenant()))).model_dump(), 201)
	except Exception as e:
		return _err(str(e))


@bp.post("/inspections/<inspection_id>/complete")
@has_access("realestate_mai:inspections")
def complete_inspection(inspection_id: str):
	try:
		findings = request.json.get("findings", []) if request.json else []
		r = _run(_svc.complete_inspection(inspection_id, _tenant(), findings))
		return _ok(r.model_dump()) if r else _err("not found", 404)
	except Exception as e:
		return _err(str(e))


@bp.get("/inspections/overdue")
@has_access("realestate_mai:inspections")
def overdue_inspections():
	return _ok([i.model_dump() for i in _run(_svc.get_overdue_inspections(_tenant()))])


# ── Defects ───────────────────────────────────────────────────────────────────

@bp.get("/defects")
@has_access("realestate_mai:defects")
def list_defects():
	return _ok([d.model_dump() for d in _run(_svc.list_defects(_tenant(), request.args.get("property_id"), request.args.get("severity")))])


@bp.post("/defects")
@has_access("realestate_mai:defects")
def raise_defect():
	try:
		return _ok(_run(_svc.raise_defect(DefectCreate(**request.json, tenant_id=_tenant()))).model_dump(), 201)
	except Exception as e:
		return _err(str(e))


@bp.post("/defects/<defect_id>/resolve")
@has_access("realestate_mai:defects")
def resolve_defect(defect_id: str):
	notes = request.json.get("resolution_notes", "") if request.json else ""
	r = _run(_svc.resolve_defect(defect_id, _tenant(), notes))
	return _ok(r.model_dump()) if r else _err("not found", 404)


# ── SLA Monitor ───────────────────────────────────────────────────────────────

@bp.get("/sla")
@has_access("realestate_mai:sla")
def sla_dashboard():
	return _ok(_run(_svc.get_sla_dashboard(_tenant())))


@bp.post("/sla")
@has_access("realestate_mai:sla")
def create_sla():
	try:
		return _ok(_run(_svc.create_sla(SlaCreate(**request.json, tenant_id=_tenant()))).model_dump(), 201)
	except Exception as e:
		return _err(str(e))
