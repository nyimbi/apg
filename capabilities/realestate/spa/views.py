"""Flask Blueprint views for Space Planning & Management (spa)."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from functools import wraps
from typing import Any, Callable

from flask import Blueprint, request, jsonify

from .service import SpaService
from .models import (
	FloorPlanCreate,
	SpaceCreate, SpaceUpdate,
	SpaceAllocationCreate,
	MoveCreate,
	BookingCreate,
	OccupancyDataCreate,
	DensityPlanCreate,
)

bp = Blueprint("spa_views", __name__, url_prefix="/realestate/spa")
_svc = SpaService()


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
@has_access("realestate_spa:view")
def dashboard():
	spaces = _run(_svc.list_spaces(_tenant()))
	available = [s for s in spaces if s.status.value == "available"]
	return _ok({"total_spaces": len(spaces), "available_spaces": len(available)})


# ── Floor Plans ───────────────────────────────────────────────────────────────

@bp.get("/floor-plans")
@has_access("realestate_spa:floor_plans")
def list_floor_plans():
	return _ok([fp.model_dump() for fp in _run(_svc.list_floor_plans(_tenant(), request.args.get("property_id")))])


@bp.post("/floor-plans")
@has_access("realestate_spa:floor_plans")
def upload_floor_plan():
	try:
		return _ok(_run(_svc.upload_floor_plan(FloorPlanCreate(**request.json, tenant_id=_tenant()))).model_dump(), 201)
	except Exception as e:
		return _err(str(e))


@bp.get("/floor-plans/<floor_plan_id>")
@has_access("realestate_spa:floor_plans")
def get_floor_plan(floor_plan_id: str):
	r = _run(_svc.get_floor_plan(floor_plan_id, _tenant()))
	return _ok(r.model_dump()) if r else _err("not found", 404)


# ── Spaces ────────────────────────────────────────────────────────────────────

@bp.get("/spaces")
@has_access("realestate_spa:spaces")
def list_spaces():
	return _ok([s.model_dump() for s in _run(_svc.list_spaces(_tenant(), request.args.get("property_id"), request.args.get("space_type"), request.args.get("status")))])


@bp.post("/spaces")
@has_access("realestate_spa:spaces")
def create_space():
	try:
		return _ok(_run(_svc.create_space(SpaceCreate(**request.json, tenant_id=_tenant()))).model_dump(), 201)
	except Exception as e:
		return _err(str(e))


@bp.get("/spaces/<space_id>")
@has_access("realestate_spa:spaces")
def get_space(space_id: str):
	r = _run(_svc.get_space(space_id, _tenant()))
	return _ok(r.model_dump()) if r else _err("not found", 404)


@bp.put("/spaces/<space_id>")
@has_access("realestate_spa:spaces")
def update_space(space_id: str):
	try:
		r = _run(_svc.update_space(space_id, _tenant(), SpaceUpdate(**request.json)))
		return _ok(r.model_dump()) if r else _err("not found", 404)
	except Exception as e:
		return _err(str(e))


@bp.get("/spaces/available")
@has_access("realestate_spa:spaces")
def available_spaces():
	min_cap = int(request.args.get("min_capacity", 1))
	return _ok([s.model_dump() for s in _run(_svc.get_available_spaces(_tenant(), request.args.get("property_id"), request.args.get("space_type"), min_cap))])


# ── Allocations ───────────────────────────────────────────────────────────────

@bp.get("/allocations")
@has_access("realestate_spa:allocations")
def list_allocations():
	return _ok([a.model_dump() for a in _run(_svc.list_allocations(_tenant(), request.args.get("space_id")))])


@bp.post("/allocations")
@has_access("realestate_spa:allocations")
def allocate_space():
	try:
		return _ok(_run(_svc.allocate_space(SpaceAllocationCreate(**request.json, tenant_id=_tenant()))).model_dump(), 201)
	except Exception as e:
		return _err(str(e))


@bp.delete("/allocations/<allocation_id>")
@has_access("realestate_spa:allocations")
def deallocate_space(allocation_id: str):
	r = _run(_svc.deallocate_space(allocation_id, _tenant()))
	return _ok(r.model_dump()) if r else _err("not found", 404)


# ── Moves ─────────────────────────────────────────────────────────────────────

@bp.get("/moves")
@has_access("realestate_spa:moves")
def list_moves():
	return _ok([m.model_dump() for m in _run(_svc.list_moves(_tenant(), request.args.get("status")))])


@bp.post("/moves")
@has_access("realestate_spa:moves")
def create_move():
	try:
		return _ok(_run(_svc.create_move(MoveCreate(**request.json, tenant_id=_tenant()))).model_dump(), 201)
	except Exception as e:
		return _err(str(e))


@bp.post("/moves/<move_id>/approve")
@has_access("realestate_spa:moves")
def approve_move(move_id: str):
	approved_by = request.json.get("approved_by", "unknown") if request.json else "unknown"
	r = _run(_svc.approve_move(move_id, _tenant(), approved_by))
	return _ok(r.model_dump()) if r else _err("not found", 404)


@bp.post("/moves/<move_id>/complete")
@has_access("realestate_spa:moves")
def complete_move(move_id: str):
	r = _run(_svc.complete_move(move_id, _tenant()))
	return _ok(r.model_dump()) if r else _err("not found", 404)


# ── Bookings ──────────────────────────────────────────────────────────────────

@bp.get("/bookings")
@has_access("realestate_spa:bookings")
def list_bookings():
	return _ok([b.model_dump() for b in _run(_svc.list_bookings(_tenant(), request.args.get("space_id"), request.args.get("booking_type")))])


@bp.post("/bookings")
@has_access("realestate_spa:bookings")
def create_booking():
	try:
		return _ok(_run(_svc.create_booking(BookingCreate(**request.json, tenant_id=_tenant()))).model_dump(), 201)
	except Exception as e:
		return _err(str(e))


@bp.delete("/bookings/<booking_id>")
@has_access("realestate_spa:bookings")
def cancel_booking(booking_id: str):
	r = _run(_svc.cancel_booking(booking_id, _tenant()))
	return _ok(r.model_dump()) if r else _err("not found", 404)


# ── Occupancy ─────────────────────────────────────────────────────────────────

@bp.post("/occupancy")
@has_access("realestate_spa:occupancy")
def ingest_occupancy():
	try:
		return _ok(_run(_svc.ingest_occupancy_data(OccupancyDataCreate(**request.json, tenant_id=_tenant()))).model_dump(), 201)
	except Exception as e:
		return _err(str(e))


@bp.get("/occupancy/<property_id>")
@has_access("realestate_spa:occupancy")
def occupancy_metrics(property_id: str):
	from datetime import date
	from_date = date.today().replace(day=1)
	to_date = date.today()
	return _ok(_run(_svc.calculate_occupancy_metrics(_tenant(), property_id, from_date, to_date)))


# ── Density ───────────────────────────────────────────────────────────────────

@bp.post("/density")
@has_access("realestate_spa:density")
def create_density_plan():
	try:
		return _ok(_run(_svc.create_density_plan(DensityPlanCreate(**request.json, tenant_id=_tenant()))).model_dump(), 201)
	except Exception as e:
		return _err(str(e))


@bp.get("/density/<property_id>")
@has_access("realestate_spa:density")
def density_analysis(property_id: str):
	return _ok(_run(_svc.get_density_analysis(_tenant(), property_id)))


# ── Chargeback ────────────────────────────────────────────────────────────────

@bp.get("/chargeback/<property_id>")
@has_access("realestate_spa:chargeback")
def chargeback(property_id: str):
	try:
		period = request.args.get("period", "")
		rate = Decimal(request.args.get("rate_per_sqm", "0"))
		verified = request.args.get("verified", "false").lower() == "true"
		return _ok(_run(_svc.calculate_chargeback(_tenant(), property_id, period, rate, verified)))
	except Exception as e:
		return _err(str(e))
