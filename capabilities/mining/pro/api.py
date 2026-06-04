"""REST API Blueprint for APG Mine Production Operations."""

from __future__ import annotations

import asyncio
from typing import Any

from flask import Blueprint, g, jsonify, request

from .models import (
	BlastCreate,
	BlastUpdate,
	GradeBoundaryCreate,
	ProductionScheduleCreate,
	ShiftReportCreate,
	ShiftReportUpdate,
	StockpileCreate,
	StockpileMovementCreate,
)
from .service import ProService

api_bp = Blueprint("mining_pro_api", __name__, url_prefix="/api/mining-pro")


def _svc() -> ProService:
	return ProService(tenant_id=getattr(g, "tenant_id", "default"))


def _loop() -> asyncio.AbstractEventLoop:
	return asyncio.get_event_loop()


def _err(msg: str, code: int = 400) -> tuple[Any, int]:
	return jsonify({"error": msg}), code


# ── Shift Reports ──────────────────────────────────────────────────────────────

@api_bp.get("/shifts")
def list_shifts():
	"""List shift reports."""
	svc = _svc()
	results = _loop().run_until_complete(
		svc.list_shift_reports(
			mine_area=request.args.get("mine_area"),
			shift_type=request.args.get("shift_type"),
			status=request.args.get("status"),
			limit=int(request.args.get("limit", 100)),
			offset=int(request.args.get("offset", 0)),
		)
	)
	return jsonify({"count": len(results), "items": [r.model_dump() for r in results]})


@api_bp.post("/shifts")
def create_shift():
	"""Create a shift report."""
	svc = _svc()
	data = request.get_json(force=True) or {}
	data["tenant_id"] = getattr(g, "tenant_id", "default")
	try:
		payload = ShiftReportCreate(**data)
		result = _loop().run_until_complete(
			svc.create_shift_report(payload, created_by=getattr(g, "user_id", "system"))
		)
		return jsonify(result.model_dump()), 201
	except (ValueError, AssertionError) as exc:
		return _err(str(exc))


@api_bp.get("/shifts/<string:id>")
def get_shift(id: str):
	"""Get a shift report."""
	svc = _svc()
	result = _loop().run_until_complete(svc.get_shift_report(id))
	if result is None:
		return _err("Not found", 404)
	return jsonify(result.model_dump())


@api_bp.put("/shifts/<string:id>")
def update_shift(id: str):
	"""Update a draft shift report."""
	svc = _svc()
	data = request.get_json(force=True) or {}
	try:
		payload = ShiftReportUpdate(**data)
		result = _loop().run_until_complete(svc.update_shift_report(id, payload))
		return jsonify(result.model_dump())
	except (ValueError, AssertionError) as exc:
		return _err(str(exc))
	except KeyError as exc:
		return _err(str(exc), 404)


@api_bp.post("/shifts/<string:id>/submit")
def submit_shift(id: str):
	"""Submit a shift report for approval."""
	svc = _svc()
	data = request.get_json(force=True) or {}
	supervisor_id = data.get("supervisor_id", getattr(g, "user_id", "system"))
	try:
		result = _loop().run_until_complete(svc.submit_shift_report(id, supervisor_id))
		return jsonify(result.model_dump())
	except (KeyError, ValueError) as exc:
		return _err(str(exc), 404 if isinstance(exc, KeyError) else 400)


@api_bp.post("/shifts/<string:id>/approve")
def approve_shift(id: str):
	"""Approve a submitted shift report."""
	svc = _svc()
	data = request.get_json(force=True) or {}
	reviewer_id = data.get("reviewer_id", getattr(g, "user_id", "system"))
	try:
		result = _loop().run_until_complete(svc.approve_shift_report(id, reviewer_id))
		return jsonify(result.model_dump())
	except (KeyError, ValueError) as exc:
		return _err(str(exc), 404 if isinstance(exc, KeyError) else 400)


# ── Blasts ─────────────────────────────────────────────────────────────────────

@api_bp.get("/blasts")
def list_blasts():
	"""List blast records."""
	svc = _svc()
	results = _loop().run_until_complete(
		svc.list_blasts(
			mine_area=request.args.get("mine_area"),
			status=request.args.get("status"),
			limit=int(request.args.get("limit", 100)),
		)
	)
	return jsonify({"count": len(results), "items": [r.model_dump() for r in results]})


@api_bp.post("/blasts")
def create_blast():
	"""Create a blast record."""
	svc = _svc()
	data = request.get_json(force=True) or {}
	data["tenant_id"] = getattr(g, "tenant_id", "default")
	try:
		payload = BlastCreate(**data)
		result = _loop().run_until_complete(
			svc.create_blast(payload, created_by=getattr(g, "user_id", "system"))
		)
		return jsonify(result.model_dump()), 201
	except (ValueError, AssertionError) as exc:
		return _err(str(exc))


@api_bp.get("/blasts/<string:id>")
def get_blast(id: str):
	"""Get a blast record."""
	svc = _svc()
	result = _loop().run_until_complete(svc.get_blast(id))
	if result is None:
		return _err("Not found", 404)
	return jsonify(result.model_dump())


@api_bp.put("/blasts/<string:id>")
def update_blast(id: str):
	"""Update blast status."""
	svc = _svc()
	data = request.get_json(force=True) or {}
	try:
		payload = BlastUpdate(**data)
		result = _loop().run_until_complete(svc.update_blast(id, payload))
		return jsonify(result.model_dump())
	except (KeyError, ValueError) as exc:
		return _err(str(exc), 404 if isinstance(exc, KeyError) else 400)


@api_bp.post("/blasts/<string:id>/approve-design")
def approve_blast_design(id: str):
	"""Approve a blast design."""
	svc = _svc()
	data = request.get_json(force=True) or {}
	approver_id = data.get("approver_id", getattr(g, "user_id", "system"))
	try:
		result = _loop().run_until_complete(svc.approve_blast_design(id, approver_id))
		return jsonify(result.model_dump())
	except KeyError as exc:
		return _err(str(exc), 404)


@api_bp.post("/blasts/<string:id>/fire")
def fire_blast(id: str):
	"""Record blast firing."""
	svc = _svc()
	data = request.get_json(force=True) or {}
	fire_authority_id = data.get("fire_authority_id", getattr(g, "user_id", "system"))
	try:
		result = _loop().run_until_complete(svc.fire_blast(id, fire_authority_id))
		return jsonify(result.model_dump())
	except (KeyError, PermissionError, ValueError) as exc:
		code = 404 if isinstance(exc, KeyError) else 403 if isinstance(exc, PermissionError) else 400
		return _err(str(exc), code)


# ── Grade Control ──────────────────────────────────────────────────────────────

@api_bp.post("/grade-boundaries")
def create_grade_boundary():
	"""Create a grade control boundary."""
	svc = _svc()
	data = request.get_json(force=True) or {}
	data["tenant_id"] = getattr(g, "tenant_id", "default")
	try:
		payload = GradeBoundaryCreate(**data)
		result = _loop().run_until_complete(
			svc.create_grade_boundary(payload, created_by=getattr(g, "user_id", "system"))
		)
		return jsonify(result.model_dump()), 201
	except (ValueError, AssertionError) as exc:
		return _err(str(exc))


@api_bp.post("/grade-boundaries/<string:id>/approve")
def approve_grade_boundary(id: str):
	"""Approve a grade boundary."""
	svc = _svc()
	data = request.get_json(force=True) or {}
	approver_id = data.get("approver_id", getattr(g, "user_id", "system"))
	try:
		result = _loop().run_until_complete(svc.approve_grade_boundary(id, approver_id))
		return jsonify(result.model_dump())
	except KeyError as exc:
		return _err(str(exc), 404)


# ── Stockpiles ─────────────────────────────────────────────────────────────────

@api_bp.get("/stockpiles")
def list_stockpiles():
	"""List stockpiles."""
	svc = _svc()
	results = _loop().run_until_complete(svc.list_stockpiles())
	return jsonify({"count": len(results), "items": [r.model_dump() for r in results]})


@api_bp.post("/stockpiles")
def create_stockpile():
	"""Create a stockpile."""
	svc = _svc()
	data = request.get_json(force=True) or {}
	data["tenant_id"] = getattr(g, "tenant_id", "default")
	try:
		payload = StockpileCreate(**data)
		result = _loop().run_until_complete(
			svc.create_stockpile(payload, created_by=getattr(g, "user_id", "system"))
		)
		return jsonify(result.model_dump()), 201
	except (ValueError, AssertionError) as exc:
		return _err(str(exc))


@api_bp.post("/stockpiles/movements")
def record_stockpile_movement():
	"""Record a stockpile movement (add/reclaim)."""
	svc = _svc()
	data = request.get_json(force=True) or {}
	try:
		payload = StockpileMovementCreate(**data)
		result = _loop().run_until_complete(
			svc.record_stockpile_movement(payload, created_by=getattr(g, "user_id", "system"))
		)
		return jsonify(result.model_dump()), 201
	except (KeyError, ValueError, AssertionError) as exc:
		return _err(str(exc), 404 if isinstance(exc, KeyError) else 400)


# ── Production Schedules ───────────────────────────────────────────────────────

@api_bp.post("/schedules")
def create_schedule():
	"""Create a production schedule."""
	svc = _svc()
	data = request.get_json(force=True) or {}
	data["tenant_id"] = getattr(g, "tenant_id", "default")
	try:
		payload = ProductionScheduleCreate(**data)
		result = _loop().run_until_complete(
			svc.create_production_schedule(payload, created_by=getattr(g, "user_id", "system"))
		)
		return jsonify(result.model_dump()), 201
	except (ValueError, AssertionError) as exc:
		return _err(str(exc))


@api_bp.post("/schedules/<string:id>/publish")
def publish_schedule(id: str):
	"""Approve and publish a production schedule."""
	svc = _svc()
	data = request.get_json(force=True) or {}
	approver_id = data.get("approver_id", getattr(g, "user_id", "system"))
	try:
		result = _loop().run_until_complete(svc.approve_and_publish_schedule(id, approver_id))
		return jsonify(result.model_dump())
	except KeyError as exc:
		return _err(str(exc), 404)


# ── Summary ────────────────────────────────────────────────────────────────────

@api_bp.get("/summary")
def production_summary():
	"""Production KPI summary."""
	svc = _svc()
	summary = _loop().run_until_complete(svc.get_production_summary())
	return jsonify(summary)
