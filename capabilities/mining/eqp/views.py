"""Flask Blueprint views for APG Equipment & Plant Management."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

from flask import Blueprint, abort, g, jsonify, request

from .service import EqpService

views_bp = Blueprint("mining_eqp_views", __name__, url_prefix="/mining-eqp")


def has_access(permission: str) -> Callable:
	def decorator(fn: Callable) -> Callable:
		@wraps(fn)
		def wrapper(*args: Any, **kwargs: Any) -> Any:
			user = getattr(g, "current_user", None)
			if user is None:
				abort(401)
			perms = getattr(user, "permissions", [])
			if permission not in perms and "mining_eqp:admin" not in perms:
				abort(403)
			return fn(*args, **kwargs)
		return wrapper
	return decorator


def _get_service() -> EqpService:
	return EqpService(tenant_id=getattr(g, "tenant_id", "default"))


# ── Dashboard ──────────────────────────────────────────────────────────────────

@views_bp.get("/dashboard")
@has_access("mining_eqp:view")
def dashboard():
	"""Fleet KPI dashboard — availability, utilisation, breakdowns."""
	import asyncio
	svc = _get_service()
	kpis = asyncio.get_event_loop().run_until_complete(svc.get_fleet_kpi_summary())
	return jsonify({"view": "equipment_dashboard", "data": kpis})


# ── Fleet Register ─────────────────────────────────────────────────────────────

@views_bp.get("/fleet")
@has_access("mining_eqp:view")
def fleet_register():
	"""Fleet register with optional filters."""
	import asyncio
	svc = _get_service()
	equipment_class = request.args.get("equipment_class")
	lifecycle_status = request.args.get("lifecycle_status")
	dispatch_status = request.args.get("dispatch_status")
	mine_area = request.args.get("mine_area")
	results = asyncio.get_event_loop().run_until_complete(
		svc.list_equipment(
			equipment_class=equipment_class,
			lifecycle_status=lifecycle_status,
			dispatch_status=dispatch_status,
			mine_area=mine_area,
		)
	)
	return jsonify({"view": "fleet_register", "count": len(results), "items": [r.model_dump() for r in results]})


@views_bp.get("/fleet/<string:id>")
@has_access("mining_eqp:view")
def equipment_detail(id: str):
	"""Equipment detail view."""
	import asyncio
	svc = _get_service()
	loop = asyncio.get_event_loop()
	eqp = loop.run_until_complete(svc.get_equipment(id))
	if eqp is None:
		abort(404)
	inspections = loop.run_until_complete(svc.list_inspections_for_equipment(id))
	work_orders = loop.run_until_complete(svc.list_work_orders(equipment_id=id))
	return jsonify({
		"view": "equipment_detail",
		"equipment": eqp.model_dump(),
		"recent_inspections": [i.model_dump() for i in inspections[:5]],
		"open_work_orders": [w.model_dump() for w in work_orders if w.status not in ("completed", "cancelled")],
	})


# ── Maintenance ────────────────────────────────────────────────────────────────

@views_bp.get("/maintenance")
@has_access("mining_eqp:view")
def maintenance_list():
	"""Maintenance work order list."""
	import asyncio
	svc = _get_service()
	equipment_id = request.args.get("equipment_id")
	status = request.args.get("status")
	results = asyncio.get_event_loop().run_until_complete(
		svc.list_work_orders(equipment_id=equipment_id, status=status)
	)
	return jsonify({"view": "maintenance", "count": len(results), "items": [r.model_dump() for r in results]})


# ── Dispatch Board ─────────────────────────────────────────────────────────────

@views_bp.get("/dispatch")
@has_access("mining_eqp:dispatch")
def dispatch_board():
	"""Real-time dispatch board — available and operating equipment."""
	import asyncio
	svc = _get_service()
	loop = asyncio.get_event_loop()
	available = loop.run_until_complete(svc.list_equipment(dispatch_status="available"))
	operating = loop.run_until_complete(svc.list_equipment(dispatch_status="operating"))
	breakdown = loop.run_until_complete(svc.list_equipment(dispatch_status="breakdown"))
	return jsonify({
		"view": "dispatch_board",
		"available": [e.model_dump() for e in available],
		"operating": [e.model_dump() for e in operating],
		"breakdown": [e.model_dump() for e in breakdown],
	})


# ── Inspections ────────────────────────────────────────────────────────────────

@views_bp.get("/inspections")
@has_access("mining_eqp:view")
def inspection_list():
	"""Recent inspection results."""
	return jsonify({"view": "inspections", "message": "Query /api/mining-eqp/inspections with equipment_id"})


# ── Fuel Ledger ────────────────────────────────────────────────────────────────

@views_bp.get("/fuel")
@has_access("mining_eqp:view")
def fuel_ledger():
	"""Fuel consumption ledger."""
	return jsonify({"view": "fuel_ledger", "message": "Fuel data via /api/mining-eqp/fuel"})


# ── KPI Dashboard ──────────────────────────────────────────────────────────────

@views_bp.get("/kpis")
@has_access("mining_eqp:reports")
def kpi_dashboard():
	"""Equipment KPI dashboard."""
	import asyncio
	svc = _get_service()
	kpis = asyncio.get_event_loop().run_until_complete(svc.get_fleet_kpi_summary())
	return jsonify({"view": "kpi_dashboard", "data": kpis})


# ── Fault Register ─────────────────────────────────────────────────────────────

@views_bp.get("/faults")
@has_access("mining_eqp:view")
def fault_register():
	"""Open equipment faults."""
	import asyncio
	svc = _get_service()
	open_faults = [f for f in svc._faults.values() if not f.get("resolved")]
	return jsonify({
		"view": "fault_register",
		"count": len(open_faults),
		"items": open_faults,
	})
