"""Flask Blueprint views for APG Mine Safety & Compliance."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

from flask import Blueprint, abort, g, jsonify, request

from .service import SafService

views_bp = Blueprint("mining_saf_views", __name__, url_prefix="/mining-saf")


def has_access(permission: str) -> Callable:
	def decorator(fn: Callable) -> Callable:
		@wraps(fn)
		def wrapper(*args: Any, **kwargs: Any) -> Any:
			user = getattr(g, "current_user", None)
			if user is None:
				abort(401)
			perms = getattr(user, "permissions", [])
			if permission not in perms and "mining_saf:admin" not in perms:
				abort(403)
			return fn(*args, **kwargs)
		return wrapper
	return decorator


def _get_service() -> SafService:
	return SafService(tenant_id=getattr(g, "tenant_id", "default"))


# ── Dashboard ──────────────────────────────────────────────────────────────────

@views_bp.get("/dashboard")
@has_access("mining_saf:view")
def dashboard():
	"""Safety KPI dashboard — LTIFR, open corrective actions, extreme hazards."""
	import asyncio
	svc = _get_service()
	stats = asyncio.get_event_loop().run_until_complete(svc.get_safety_statistics())
	return jsonify({"view": "safety_dashboard", "data": stats})


# ── Incidents ──────────────────────────────────────────────────────────────────

@views_bp.get("/incidents")
@has_access("mining_saf:view")
def list_incidents():
	"""List safety incidents with optional filters."""
	import asyncio
	svc = _get_service()
	incident_type = request.args.get("incident_type")
	status = request.args.get("status")
	limit = int(request.args.get("limit", 100))
	offset = int(request.args.get("offset", 0))
	results = asyncio.get_event_loop().run_until_complete(
		svc.list_incidents(incident_type=incident_type, status=status, limit=limit, offset=offset)
	)
	return jsonify({"view": "incidents", "count": len(results), "items": [r.model_dump() for r in results]})


@views_bp.get("/incidents/<string:id>")
@has_access("mining_saf:view")
def incident_detail(id: str):
	"""Incident detail view."""
	import asyncio
	svc = _get_service()
	incident = asyncio.get_event_loop().run_until_complete(svc.get_incident(id))
	if incident is None:
		abort(404)
	return jsonify({"view": "incident_detail", "incident": incident.model_dump()})


# ── Hazards ────────────────────────────────────────────────────────────────────

@views_bp.get("/hazards")
@has_access("mining_saf:view")
def hazard_register():
	"""Hazard register view."""
	import asyncio
	svc = _get_service()
	risk_rating = request.args.get("risk_rating")
	mine_area = request.args.get("mine_area")
	open_only = request.args.get("open_only", "true").lower() == "true"
	results = asyncio.get_event_loop().run_until_complete(
		svc.list_hazards(risk_rating=risk_rating, mine_area=mine_area, open_only=open_only)
	)
	return jsonify({"view": "hazard_register", "count": len(results), "items": [r.model_dump() for r in results]})


# ── Risk Register ──────────────────────────────────────────────────────────────

@views_bp.get("/risk-register")
@has_access("mining_saf:view")
def risk_register():
	"""Risk register view."""
	import asyncio
	svc = _get_service()
	results = asyncio.get_event_loop().run_until_complete(svc.list_risk_register())
	return jsonify({"view": "risk_register", "count": len(results), "items": [r.model_dump() for r in results]})


# ── Permits to Work ────────────────────────────────────────────────────────────

@views_bp.get("/permits")
@has_access("mining_saf:view")
def list_permits():
	"""List active permits to work."""
	import asyncio
	svc = _get_service()
	mine_area = request.args.get("mine_area")
	results = asyncio.get_event_loop().run_until_complete(svc.list_active_permits(mine_area=mine_area))
	return jsonify({"view": "permits", "count": len(results), "items": [r.model_dump() for r in results]})


# ── Compliance ─────────────────────────────────────────────────────────────────

@views_bp.get("/compliance")
@has_access("mining_saf:compliance")
def compliance_register():
	"""Compliance obligations register."""
	return jsonify({"view": "compliance_register", "message": "Compliance data via /api/mining-saf/corrective-actions"})


# ── Audits ─────────────────────────────────────────────────────────────────────

@views_bp.get("/audits")
@has_access("mining_saf:audit")
def audit_list():
	"""Safety audit list."""
	return jsonify({"view": "audits", "message": "Audit management via API"})


# ── Training Matrix ────────────────────────────────────────────────────────────

@views_bp.get("/training")
@has_access("mining_saf:view")
def training_matrix():
	"""Safety training matrix."""
	return jsonify({"view": "training_matrix", "message": "Training records via API"})


# ── Safety Statistics ──────────────────────────────────────────────────────────

@views_bp.get("/statistics")
@has_access("mining_saf:reports")
def safety_statistics():
	"""Safety statistics and KPI report."""
	import asyncio
	svc = _get_service()
	stats = asyncio.get_event_loop().run_until_complete(svc.get_safety_statistics())
	return jsonify({"view": "safety_statistics", "data": stats})


# ── Corrective Actions ─────────────────────────────────────────────────────────

@views_bp.get("/corrective-actions")
@has_access("mining_saf:view")
def corrective_actions():
	"""List corrective actions."""
	import asyncio
	svc = _get_service()
	status = request.args.get("status")
	results = asyncio.get_event_loop().run_until_complete(svc.list_corrective_actions(status=status))
	return jsonify({"view": "corrective_actions", "count": len(results), "items": [r.model_dump() for r in results]})
