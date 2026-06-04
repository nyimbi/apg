"""REST API Blueprint for APG Exploration Data Management."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from flask import Blueprint, g, jsonify, request

from .models import (
	AssayResultCreate,
	ComplianceReportCreate,
	DrillholeCollarCreate,
	GeologyIntervalCreate,
	ResourceEstimateCreate,
	ResourceEstimateUpdate,
)
from .service import ExpService

api_bp = Blueprint("mining_exp_api", __name__, url_prefix="/api/mining-exp")


def _svc() -> ExpService:
	return ExpService(tenant_id=getattr(g, "tenant_id", "default"))


def _loop() -> asyncio.AbstractEventLoop:
	return asyncio.get_event_loop()


def _err(msg: str, code: int = 400) -> tuple[Any, int]:
	return jsonify({"error": msg}), code


# ── Drillhole Collars ──────────────────────────────────────────────────────────

@api_bp.get("/drillholes")
def list_drillholes():
	"""
	List drillhole collars.
	---
	parameters:
	  - name: prospect
	    in: query
	  - name: hole_type
	    in: query
	  - name: limit
	    in: query
	    default: 100
	  - name: offset
	    in: query
	    default: 0
	responses:
	  200:
	    description: List of drillhole collars
	"""
	svc = _svc()
	results = _loop().run_until_complete(
		svc.list_drillhole_collars(
			prospect=request.args.get("prospect"),
			hole_type=request.args.get("hole_type"),
			limit=int(request.args.get("limit", 100)),
			offset=int(request.args.get("offset", 0)),
		)
	)
	return jsonify({"count": len(results), "items": [r.model_dump() for r in results]})


@api_bp.post("/drillholes")
def create_drillhole():
	"""
	Create a drillhole collar.
	---
	requestBody:
	  required: true
	  content:
	    application/json:
	      schema: DrillholeCollarCreate
	responses:
	  201:
	    description: Collar created
	  400:
	    description: Validation error
	"""
	svc = _svc()
	data = request.get_json(force=True) or {}
	data["tenant_id"] = getattr(g, "tenant_id", "default")
	try:
		payload = DrillholeCollarCreate(**data)
		result = _loop().run_until_complete(
			svc.create_drillhole_collar(payload, created_by=getattr(g, "user_id", "system"))
		)
		return jsonify(result.model_dump()), 201
	except (ValueError, AssertionError) as exc:
		return _err(str(exc))


@api_bp.get("/drillholes/<string:id>")
def get_drillhole(id: str):
	"""Get a drillhole collar by id."""
	svc = _svc()
	result = _loop().run_until_complete(svc.get_drillhole_collar(id))
	if result is None:
		return _err("Not found", 404)
	return jsonify(result.model_dump())


@api_bp.patch("/drillholes/<string:id>/depth")
def update_depth(id: str):
	"""Update actual drilled depth."""
	svc = _svc()
	data = request.get_json(force=True) or {}
	actual_depth_m = data.get("actual_depth_m")
	if actual_depth_m is None:
		return _err("actual_depth_m required")
	try:
		result = _loop().run_until_complete(svc.update_drillhole_actual_depth(id, float(actual_depth_m)))
		return jsonify(result.model_dump())
	except (KeyError, AssertionError, ValueError) as exc:
		return _err(str(exc), 404 if isinstance(exc, KeyError) else 400)


# ── Assay Results ──────────────────────────────────────────────────────────────

@api_bp.get("/assays")
def list_assays():
	"""List assay results."""
	svc = _svc()
	results = _loop().run_until_complete(
		svc.list_assays(
			commodity=request.args.get("commodity"),
			min_grade=request.args.get("min_grade", type=float),
			limit=int(request.args.get("limit", 200)),
			offset=int(request.args.get("offset", 0)),
		)
	)
	return jsonify({"count": len(results), "items": [r.model_dump() for r in results]})


@api_bp.post("/assays/import")
def import_assays():
	"""
	Bulk import assay results.
	---
	requestBody:
	  required: true
	  content:
	    application/json:
	      schema:
	        type: array
	        items: AssayResultCreate
	responses:
	  201:
	    description: Assays imported
	  400:
	    description: Validation or business rule error
	"""
	svc = _svc()
	data = request.get_json(force=True) or []
	if not isinstance(data, list):
		return _err("Expected a JSON array of assay records")
	tenant_id = getattr(g, "tenant_id", "default")
	try:
		payloads = [AssayResultCreate(**{**item, "tenant_id": tenant_id}) for item in data]
		results = _loop().run_until_complete(
			svc.import_assay_results(payloads, created_by=getattr(g, "user_id", "system"))
		)
		return jsonify({"imported": len(results), "items": [r.model_dump() for r in results]}), 201
	except (ValueError, AssertionError) as exc:
		return _err(str(exc))


@api_bp.get("/assays/hole/<string:hole_id>")
def get_assays_for_hole(hole_id: str):
	"""Get all assay results for a drillhole."""
	svc = _svc()
	results = _loop().run_until_complete(svc.get_assay_results_for_hole(hole_id))
	return jsonify({"hole_id": hole_id, "count": len(results), "items": [r.model_dump() for r in results]})


@api_bp.post("/assays/<string:id>/qaqc-flag")
def flag_assay_qaqc(id: str):
	"""Attach a QAQC flag to an assay result."""
	svc = _svc()
	data = request.get_json(force=True) or {}
	flag = data.get("flag")
	if not flag:
		return _err("flag field required")
	try:
		result = _loop().run_until_complete(svc.flag_qaqc_result(id, flag))
		return jsonify(result.model_dump())
	except KeyError as exc:
		return _err(str(exc), 404)


# ── Geology ────────────────────────────────────────────────────────────────────

@api_bp.post("/geology")
def log_geology():
	"""Log a geology interval."""
	svc = _svc()
	data = request.get_json(force=True) or {}
	data["tenant_id"] = getattr(g, "tenant_id", "default")
	try:
		payload = GeologyIntervalCreate(**data)
		result = _loop().run_until_complete(
			svc.log_geology_interval(payload, created_by=getattr(g, "user_id", "system"))
		)
		return jsonify(result.model_dump()), 201
	except (ValueError, AssertionError) as exc:
		return _err(str(exc))


@api_bp.get("/geology/hole/<string:hole_id>")
def get_geology_for_hole(hole_id: str):
	"""Get geology intervals for a hole."""
	svc = _svc()
	results = _loop().run_until_complete(svc.get_geology_for_hole(hole_id))
	return jsonify({"hole_id": hole_id, "count": len(results), "items": [r.model_dump() for r in results]})


# ── Resource Estimates ─────────────────────────────────────────────────────────

@api_bp.get("/resources")
def list_resources():
	"""List resource estimates."""
	svc = _svc()
	results = _loop().run_until_complete(
		svc.list_resource_estimates(
			classification=request.args.get("classification"),
			commodity=request.args.get("commodity"),
			published_only=request.args.get("published_only", "false").lower() == "true",
		)
	)
	return jsonify({"count": len(results), "items": [r.model_dump() for r in results]})


@api_bp.post("/resources")
def create_resource():
	"""Create a resource estimate."""
	svc = _svc()
	data = request.get_json(force=True) or {}
	data["tenant_id"] = getattr(g, "tenant_id", "default")
	try:
		payload = ResourceEstimateCreate(**data)
		result = _loop().run_until_complete(
			svc.create_resource_estimate(payload, created_by=getattr(g, "user_id", "system"))
		)
		return jsonify(result.model_dump()), 201
	except (ValueError, AssertionError) as exc:
		return _err(str(exc))


@api_bp.get("/resources/<string:id>")
def get_resource(id: str):
	"""Get a resource estimate."""
	svc = _svc()
	result = _loop().run_until_complete(svc.get_resource_estimate(id))
	if result is None:
		return _err("Not found", 404)
	return jsonify(result.model_dump())


@api_bp.put("/resources/<string:id>")
def update_resource(id: str):
	"""Update a resource estimate."""
	svc = _svc()
	data = request.get_json(force=True) or {}
	try:
		payload = ResourceEstimateUpdate(**data)
		result = _loop().run_until_complete(svc.update_resource_estimate(id, payload))
		return jsonify(result.model_dump())
	except (ValueError, PermissionError) as exc:
		return _err(str(exc))
	except KeyError as exc:
		return _err(str(exc), 404)


@api_bp.post("/resources/<string:id>/approve")
def approve_resource(id: str):
	"""Approve a resource estimate."""
	svc = _svc()
	data = request.get_json(force=True) or {}
	reviewer_id = data.get("reviewer_id", getattr(g, "user_id", "system"))
	try:
		result = _loop().run_until_complete(
			svc.approve_resource_estimate(id, reviewer_id, data.get("notes"))
		)
		return jsonify(result.model_dump())
	except KeyError as exc:
		return _err(str(exc), 404)


@api_bp.post("/resources/<string:id>/publish")
def publish_resource(id: str):
	"""Publish an approved resource estimate."""
	svc = _svc()
	try:
		result = _loop().run_until_complete(svc.publish_resource_estimate(id))
		return jsonify(result.model_dump())
	except (KeyError, PermissionError) as exc:
		return _err(str(exc), 403 if isinstance(exc, PermissionError) else 404)


# ── Compliance Reports ─────────────────────────────────────────────────────────

@api_bp.get("/reports")
def list_reports():
	"""List compliance reports."""
	svc = _svc()
	results = _loop().run_until_complete(
		svc.list_compliance_reports(
			published_only=request.args.get("published_only", "false").lower() == "true"
		)
	)
	return jsonify({"count": len(results), "items": [r.model_dump() for r in results]})


@api_bp.post("/reports")
def create_report():
	"""Create a compliance report."""
	svc = _svc()
	data = request.get_json(force=True) or {}
	data["tenant_id"] = getattr(g, "tenant_id", "default")
	try:
		payload = ComplianceReportCreate(**data)
		result = _loop().run_until_complete(
			svc.create_compliance_report(payload, created_by=getattr(g, "user_id", "system"))
		)
		return jsonify(result.model_dump()), 201
	except (ValueError, AssertionError) as exc:
		return _err(str(exc))


@api_bp.post("/reports/<string:id>/sign-off")
def sign_off_report(id: str):
	"""Competent person sign-off on a compliance report."""
	svc = _svc()
	data = request.get_json(force=True) or {}
	cp_id = data.get("competent_person_id", getattr(g, "user_id", "system"))
	try:
		result = _loop().run_until_complete(svc.sign_off_compliance_report(id, cp_id))
		return jsonify(result.model_dump())
	except (KeyError, PermissionError) as exc:
		return _err(str(exc), 403 if isinstance(exc, PermissionError) else 404)


@api_bp.post("/reports/<string:id>/publish")
def publish_report(id: str):
	"""Publish a signed compliance report."""
	svc = _svc()
	try:
		result = _loop().run_until_complete(svc.publish_compliance_report(id))
		return jsonify(result.model_dump())
	except (KeyError, PermissionError) as exc:
		return _err(str(exc), 403 if isinstance(exc, PermissionError) else 404)


# ── Summary ────────────────────────────────────────────────────────────────────

@api_bp.get("/summary")
def get_summary():
	"""Exploration KPI summary."""
	svc = _svc()
	summary = _loop().run_until_complete(svc.get_exploration_summary())
	return jsonify(summary)
