"""Flask Blueprint views for APG Exploration Data Management."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

from flask import Blueprint, abort, g, jsonify, request

from .service import ExpService

views_bp = Blueprint("mining_exp_views", __name__, url_prefix="/mining-exp")


# ── Auth stub (replace with real has_access decorator) ────────────────────────

def has_access(permission: str) -> Callable:
	"""Decorator: verify the current user holds the required permission."""
	def decorator(fn: Callable) -> Callable:
		@wraps(fn)
		def wrapper(*args: Any, **kwargs: Any) -> Any:
			user = getattr(g, "current_user", None)
			if user is None:
				abort(401)
			perms = getattr(user, "permissions", [])
			if permission not in perms and "mining_exp:admin" not in perms:
				abort(403)
			return fn(*args, **kwargs)
		return wrapper
	return decorator


def _get_service() -> ExpService:
	tenant_id = getattr(g, "tenant_id", "default")
	return ExpService(tenant_id=tenant_id)


def _enforce_tenant(data: dict[str, Any]) -> dict[str, Any]:
	"""Inject tenant_id from session context."""
	data["tenant_id"] = getattr(g, "tenant_id", "default")
	return data


# ── Dashboard ──────────────────────────────────────────────────────────────────

@views_bp.get("/dashboard")
@has_access("mining_exp:view")
def dashboard():
	"""Exploration overview dashboard."""
	import asyncio
	svc = _get_service()
	summary = asyncio.get_event_loop().run_until_complete(svc.get_exploration_summary())
	return jsonify({"view": "dashboard", "data": summary})


# ── Drillhole Collar Views ─────────────────────────────────────────────────────

@views_bp.get("/drillholes")
@has_access("mining_exp:view")
def list_drillholes():
	"""List drillhole collars with optional prospect/hole_type filter."""
	import asyncio
	svc = _get_service()
	prospect = request.args.get("prospect")
	hole_type = request.args.get("hole_type")
	limit = int(request.args.get("limit", 100))
	offset = int(request.args.get("offset", 0))
	results = asyncio.get_event_loop().run_until_complete(
		svc.list_drillhole_collars(prospect=prospect, hole_type=hole_type, limit=limit, offset=offset)
	)
	return jsonify({"view": "drillholes", "count": len(results), "items": [r.model_dump() for r in results]})


@views_bp.get("/drillholes/<string:id>")
@has_access("mining_exp:view")
def drillhole_detail(id: str):
	"""Drillhole collar detail with associated assays and geology."""
	import asyncio
	svc = _get_service()
	loop = asyncio.get_event_loop()
	collar = loop.run_until_complete(svc.get_drillhole_collar(id))
	if collar is None:
		abort(404)
	assays = loop.run_until_complete(svc.get_assay_results_for_hole(collar.hole_id))
	geology = loop.run_until_complete(svc.get_geology_for_hole(collar.hole_id))
	return jsonify({
		"view": "drillhole_detail",
		"collar": collar.model_dump(),
		"assay_count": len(assays),
		"geology_intervals": len(geology),
	})


# ── Assay Views ────────────────────────────────────────────────────────────────

@views_bp.get("/assays")
@has_access("mining_exp:view")
def list_assays():
	"""List assay results with optional commodity/grade filter."""
	import asyncio
	svc = _get_service()
	commodity = request.args.get("commodity")
	min_grade = request.args.get("min_grade", type=float)
	limit = int(request.args.get("limit", 200))
	offset = int(request.args.get("offset", 0))
	results = asyncio.get_event_loop().run_until_complete(
		svc.list_assays(commodity=commodity, min_grade=min_grade, limit=limit, offset=offset)
	)
	return jsonify({"view": "assays", "count": len(results), "items": [r.model_dump() for r in results]})


# ── QAQC View ──────────────────────────────────────────────────────────────────

@views_bp.get("/qaqc")
@has_access("mining_exp:view")
def qaqc_dashboard():
	"""QAQC summary across all assay batches."""
	import asyncio
	svc = _get_service()
	all_assays = asyncio.get_event_loop().run_until_complete(svc.list_assays(limit=10000))
	flagged = [a for a in all_assays if a.qaqc_flag]
	return jsonify({
		"view": "qaqc_dashboard",
		"total_samples": len(all_assays),
		"flagged_count": len(flagged),
		"flags": [{"id": a.id, "hole_id": a.hole_id, "flag": a.qaqc_flag} for a in flagged],
	})


# ── Resource Estimate Views ────────────────────────────────────────────────────

@views_bp.get("/resources")
@has_access("mining_exp:resources")
def list_resources():
	"""List resource estimates."""
	import asyncio
	svc = _get_service()
	classification = request.args.get("classification")
	commodity = request.args.get("commodity")
	published_only = request.args.get("published_only", "false").lower() == "true"
	results = asyncio.get_event_loop().run_until_complete(
		svc.list_resource_estimates(classification=classification, commodity=commodity, published_only=published_only)
	)
	return jsonify({"view": "resources", "count": len(results), "items": [r.model_dump() for r in results]})


@views_bp.get("/resources/<string:id>")
@has_access("mining_exp:resources")
def resource_detail(id: str):
	"""Resource estimate detail view."""
	import asyncio
	svc = _get_service()
	resource = asyncio.get_event_loop().run_until_complete(svc.get_resource_estimate(id))
	if resource is None:
		abort(404)
	return jsonify({"view": "resource_detail", "resource": resource.model_dump()})


# ── Compliance Report Views ────────────────────────────────────────────────────

@views_bp.get("/reports")
@has_access("mining_exp:reports")
def list_reports():
	"""List compliance reports."""
	import asyncio
	svc = _get_service()
	published_only = request.args.get("published_only", "false").lower() == "true"
	results = asyncio.get_event_loop().run_until_complete(svc.list_compliance_reports(published_only=published_only))
	return jsonify({"view": "reports", "count": len(results), "items": [r.model_dump() for r in results]})


# ── Map / Spatial View ─────────────────────────────────────────────────────────

@views_bp.get("/maps")
@has_access("mining_exp:view")
def geological_map():
	"""Return collar spatial data for map rendering."""
	import asyncio
	svc = _get_service()
	collars = asyncio.get_event_loop().run_until_complete(svc.list_drillhole_collars(limit=5000))
	features = [
		{
			"type": "Feature",
			"geometry": {"type": "Point", "coordinates": [c.easting, c.northing, c.elevation_m]},
			"properties": {"hole_id": c.hole_id, "hole_type": c.hole_type, "depth_m": c.actual_depth_m or c.planned_depth_m},
		}
		for c in collars
	]
	return jsonify({"view": "geological_map", "type": "FeatureCollection", "features": features})
