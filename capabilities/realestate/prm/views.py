"""Flask Blueprint views for Property Management (prm)."""

from __future__ import annotations

import asyncio
from functools import wraps
from typing import Any, Callable

from flask import Blueprint, request, jsonify

from .service import PrmService
from .models import (
	OwnerCreate, OwnerUpdate,
	PropertyCreate, PropertyUpdate,
	UnitCreate, UnitUpdate,
	KpiCalculationRequest,
	DistributionCreate,
	HandoverCreate,
)

bp = Blueprint("prm_views", __name__, url_prefix="/realestate/prm")
_svc = PrmService()


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
@has_access("realestate_prm:view")
def dashboard():
	return _ok(_run(_svc.get_portfolio_summary(_tenant())))


# ── Owners ────────────────────────────────────────────────────────────────────

@bp.get("/owners")
@has_access("realestate_prm:owners")
def list_owners():
	return _ok([o.model_dump() for o in _run(_svc.list_owners(_tenant()))])


@bp.post("/owners")
@has_access("realestate_prm:owners")
def create_owner():
	try:
		payload = OwnerCreate(**request.json, tenant_id=_tenant())
		return _ok(_run(_svc.register_owner(payload)).model_dump(), 201)
	except Exception as e:
		return _err(str(e))


@bp.get("/owners/<owner_id>")
@has_access("realestate_prm:owners")
def get_owner(owner_id: str):
	r = _run(_svc.get_owner(owner_id, _tenant()))
	return _ok(r.model_dump()) if r else _err("not found", 404)


@bp.put("/owners/<owner_id>")
@has_access("realestate_prm:owners")
def update_owner(owner_id: str):
	try:
		r = _run(_svc.update_owner(owner_id, _tenant(), OwnerUpdate(**request.json)))
		return _ok(r.model_dump()) if r else _err("not found", 404)
	except Exception as e:
		return _err(str(e))


# ── Properties ────────────────────────────────────────────────────────────────

@bp.get("/properties")
@has_access("realestate_prm:properties")
def list_properties():
	return _ok([p.model_dump() for p in _run(_svc.list_properties(_tenant(), request.args.get("portfolio_tier"), request.args.get("status")))])


@bp.post("/properties")
@has_access("realestate_prm:properties")
def create_property():
	try:
		return _ok(_run(_svc.register_property(PropertyCreate(**request.json, tenant_id=_tenant()))).model_dump(), 201)
	except Exception as e:
		return _err(str(e))


@bp.get("/properties/<property_id>")
@has_access("realestate_prm:properties")
def get_property(property_id: str):
	r = _run(_svc.get_property(property_id, _tenant()))
	return _ok(r.model_dump()) if r else _err("not found", 404)


@bp.put("/properties/<property_id>")
@has_access("realestate_prm:properties")
def update_property(property_id: str):
	try:
		r = _run(_svc.update_property(property_id, _tenant(), PropertyUpdate(**request.json)))
		return _ok(r.model_dump()) if r else _err("not found", 404)
	except Exception as e:
		return _err(str(e))


@bp.delete("/properties/<property_id>")
@has_access("realestate_prm:properties")
def delete_property(property_id: str):
	try:
		board_approved = request.json.get("board_approved", False) if request.json else False
		deleted = _run(_svc.delete_property(property_id, _tenant(), board_approved))
		return _ok({"deleted": deleted}) if deleted else _err("not found or not authorised", 404)
	except Exception as e:
		return _err(str(e))


@bp.get("/properties/search")
@has_access("realestate_prm:properties")
def search_properties():
	q = request.args.get("q", "")
	return _ok([p.model_dump() for p in _run(_svc.search_properties(_tenant(), q))])


# ── Units ─────────────────────────────────────────────────────────────────────

@bp.get("/units")
@has_access("realestate_prm:units")
def list_units():
	return _ok([u.model_dump() for u in _run(_svc.list_units(_tenant(), request.args.get("property_id"), request.args.get("status")))])


@bp.post("/units")
@has_access("realestate_prm:units")
def create_unit():
	try:
		return _ok(_run(_svc.create_unit(UnitCreate(**request.json, tenant_id=_tenant()))).model_dump(), 201)
	except Exception as e:
		return _err(str(e))


@bp.get("/units/<unit_id>")
@has_access("realestate_prm:units")
def get_unit(unit_id: str):
	r = _run(_svc.get_unit(unit_id, _tenant()))
	return _ok(r.model_dump()) if r else _err("not found", 404)


@bp.put("/units/<unit_id>")
@has_access("realestate_prm:units")
def update_unit(unit_id: str):
	try:
		r = _run(_svc.update_unit(unit_id, _tenant(), UnitUpdate(**request.json)))
		return _ok(r.model_dump()) if r else _err("not found", 404)
	except Exception as e:
		return _err(str(e))


@bp.get("/units/void")
@has_access("realestate_prm:units")
def void_units():
	return _ok([u.model_dump() for u in _run(_svc.get_void_units(_tenant(), request.args.get("property_id")))])


# ── KPIs ──────────────────────────────────────────────────────────────────────

@bp.post("/kpis")
@has_access("realestate_prm:kpis")
def calculate_kpis():
	try:
		req = KpiCalculationRequest(**request.json, tenant_id=_tenant())
		return _ok(_run(_svc.calculate_kpis(req)).model_dump(), 201)
	except Exception as e:
		return _err(str(e))


# ── Distributions ─────────────────────────────────────────────────────────────

@bp.get("/distributions")
@has_access("realestate_prm:distributions")
def list_distributions():
	return _ok([d.model_dump() for d in _run(_svc.list_distributions(_tenant(), request.args.get("owner_id")))])


@bp.post("/distributions")
@has_access("realestate_prm:distributions")
def create_distribution():
	try:
		return _ok(_run(_svc.create_distribution(DistributionCreate(**request.json, tenant_id=_tenant()))).model_dump(), 201)
	except Exception as e:
		return _err(str(e))


@bp.post("/distributions/<dist_id>/approve")
@has_access("realestate_prm:distributions")
def approve_distribution(dist_id: str):
	try:
		data = request.json
		r = _run(_svc.approve_distribution(dist_id, _tenant(), data["approver"], data["second_approver"]))
		return _ok(r.model_dump()) if r else _err("not found", 404)
	except Exception as e:
		return _err(str(e))


# ── Handovers ─────────────────────────────────────────────────────────────────

@bp.post("/handovers")
@has_access("realestate_prm:handovers")
def create_handover():
	try:
		return _ok(_run(_svc.create_handover(HandoverCreate(**request.json, tenant_id=_tenant()))).model_dump(), 201)
	except Exception as e:
		return _err(str(e))


@bp.post("/handovers/<handover_id>/complete")
@has_access("realestate_prm:handovers")
def complete_handover(handover_id: str):
	r = _run(_svc.complete_handover(handover_id, _tenant()))
	return _ok(r.model_dump()) if r else _err("not found", 404)
