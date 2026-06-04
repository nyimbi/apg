"""Flask Blueprint views for APG Mine Production Operations."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

from flask import Blueprint, abort, g, jsonify, request

from .service import ProService

views_bp = Blueprint("mining_pro_views", __name__, url_prefix="/mining-pro")


def has_access(permission: str) -> Callable:
	def decorator(fn: Callable) -> Callable:
		@wraps(fn)
		def wrapper(*args: Any, **kwargs: Any) -> Any:
			user = getattr(g, "current_user", None)
			if user is None:
				abort(401)
			perms = getattr(user, "permissions", [])
			if permission not in perms and "mining_pro:admin" not in perms:
				abort(403)
			return fn(*args, **kwargs)
		return wrapper
	return decorator


def _get_service() -> ProService:
	return ProService(tenant_id=getattr(g, "tenant_id", "default"))


# ── Dashboard ──────────────────────────────────────────────────────────────────

@views_bp.get("/dashboard")
@has_access("mining_pro:view")
def dashboard():
	"""Production overview KPI dashboard."""
	import asyncio
	svc = _get_service()
	summary = asyncio.get_event_loop().run_until_complete(svc.get_production_summary())
	return jsonify({"view": "dashboard", "data": summary})


# ── Shift Reports ──────────────────────────────────────────────────────────────

@views_bp.get("/shifts")
@has_access("mining_pro:view")
def list_shifts():
	"""List shift reports with optional filters."""
	import asyncio
	svc = _get_service()
	mine_area = request.args.get("mine_area")
	shift_type = request.args.get("shift_type")
	status = request.args.get("status")
	limit = int(request.args.get("limit", 100))
	offset = int(request.args.get("offset", 0))
	results = asyncio.get_event_loop().run_until_complete(
		svc.list_shift_reports(mine_area=mine_area, shift_type=shift_type, status=status, limit=limit, offset=offset)
	)
	return jsonify({"view": "shift_reports", "count": len(results), "items": [r.model_dump() for r in results]})


@views_bp.get("/shifts/<string:id>")
@has_access("mining_pro:view")
def shift_detail(id: str):
	"""Shift report detail view."""
	import asyncio
	svc = _get_service()
	report = asyncio.get_event_loop().run_until_complete(svc.get_shift_report(id))
	if report is None:
		abort(404)
	return jsonify({"view": "shift_detail", "report": report.model_dump()})


# ── Blast Views ────────────────────────────────────────────────────────────────

@views_bp.get("/blasts")
@has_access("mining_pro:view")
def list_blasts():
	"""List blasts with optional mine_area/status filter."""
	import asyncio
	svc = _get_service()
	mine_area = request.args.get("mine_area")
	status = request.args.get("status")
	results = asyncio.get_event_loop().run_until_complete(
		svc.list_blasts(mine_area=mine_area, status=status)
	)
	return jsonify({"view": "blasts", "count": len(results), "items": [r.model_dump() for r in results]})


@views_bp.get("/blasts/<string:id>")
@has_access("mining_pro:view")
def blast_detail(id: str):
	"""Blast detail view including hole data."""
	import asyncio
	svc = _get_service()
	blast = asyncio.get_event_loop().run_until_complete(svc.get_blast(id))
	if blast is None:
		abort(404)
	return jsonify({"view": "blast_detail", "blast": blast.model_dump()})


# ── Grade Control ──────────────────────────────────────────────────────────────

@views_bp.get("/grade-control")
@has_access("mining_pro:grade_control")
def grade_control():
	"""Active grade control boundaries."""
	import asyncio
	svc = _get_service()
	mine_area = request.args.get("mine_area", "")
	commodity = request.args.get("commodity", "gold")
	boundary = asyncio.get_event_loop().run_until_complete(
		svc.get_active_grade_boundary(mine_area=mine_area, commodity=commodity)
	)
	return jsonify({
		"view": "grade_control",
		"mine_area": mine_area,
		"commodity": commodity,
		"active_boundary": boundary.model_dump() if boundary else None,
	})


# ── Stockpiles ─────────────────────────────────────────────────────────────────

@views_bp.get("/stockpiles")
@has_access("mining_pro:view")
def list_stockpiles():
	"""List stockpile inventory."""
	import asyncio
	svc = _get_service()
	results = asyncio.get_event_loop().run_until_complete(svc.list_stockpiles())
	total_tonnes = sum(r.current_tonnes for r in results)
	return jsonify({
		"view": "stockpiles",
		"count": len(results),
		"total_tonnes": total_tonnes,
		"items": [r.model_dump() for r in results],
	})


# ── Production Schedule ────────────────────────────────────────────────────────

@views_bp.get("/schedule")
@has_access("mining_pro:schedule")
def production_schedule():
	"""Published production schedules."""
	import asyncio
	svc = _get_service()
	# List all schedules (in-memory service doesn't expose list_schedules yet; return placeholder)
	return jsonify({"view": "schedule", "message": "Use /api/mining-pro/schedules for full schedule list"})


# ── Reports ────────────────────────────────────────────────────────────────────

@views_bp.get("/reports")
@has_access("mining_pro:reports")
def production_reports():
	"""Production KPI report view."""
	import asyncio
	svc = _get_service()
	summary = asyncio.get_event_loop().run_until_complete(svc.get_production_summary())
	return jsonify({"view": "production_reports", "summary": summary})
