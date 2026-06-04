"""REST API Blueprint for APG Mine Safety & Compliance."""

from __future__ import annotations

import asyncio
from typing import Any

from flask import Blueprint, g, jsonify, request

from .models import (
	CorrectiveActionCreate,
	HazardCreate,
	IncidentCreate,
	PermitToWorkCreate,
	RiskRegisterEntryCreate,
	RiskRating,
)
from .service import SafService

api_bp = Blueprint("mining_saf_api", __name__, url_prefix="/api/mining-saf")


def _svc() -> SafService:
	return SafService(tenant_id=getattr(g, "tenant_id", "default"))


def _loop() -> asyncio.AbstractEventLoop:
	return asyncio.get_event_loop()


def _err(msg: str, code: int = 400) -> tuple[Any, int]:
	return jsonify({"error": msg}), code


# ── Incidents ──────────────────────────────────────────────────────────────────

@api_bp.get("/incidents")
def list_incidents():
	"""List safety incidents."""
	svc = _svc()
	results = _loop().run_until_complete(
		svc.list_incidents(
			incident_type=request.args.get("incident_type"),
			status=request.args.get("status"),
			limit=int(request.args.get("limit", 100)),
			offset=int(request.args.get("offset", 0)),
		)
	)
	return jsonify({"count": len(results), "items": [r.model_dump() for r in results]})


@api_bp.post("/incidents")
def report_incident():
	"""Report a safety incident."""
	svc = _svc()
	data = request.get_json(force=True) or {}
	data["tenant_id"] = getattr(g, "tenant_id", "default")
	try:
		payload = IncidentCreate(**data)
		result = _loop().run_until_complete(
			svc.report_incident(payload, created_by=getattr(g, "user_id", "system"))
		)
		return jsonify(result.model_dump()), 201
	except (ValueError, AssertionError) as exc:
		return _err(str(exc))


@api_bp.get("/incidents/<string:id>")
def get_incident(id: str):
	"""Get an incident."""
	svc = _svc()
	result = _loop().run_until_complete(svc.get_incident(id))
	if result is None:
		return _err("Not found", 404)
	return jsonify(result.model_dump())


@api_bp.post("/incidents/<string:id>/investigate")
def open_investigation(id: str):
	"""Open an investigation for an incident."""
	svc = _svc()
	data = request.get_json(force=True) or {}
	investigation_id = data.get("investigation_id")
	if not investigation_id:
		return _err("investigation_id required")
	try:
		result = _loop().run_until_complete(svc.open_investigation(id, investigation_id))
		return jsonify(result.model_dump())
	except KeyError as exc:
		return _err(str(exc), 404)


@api_bp.post("/incidents/<string:id>/close")
def close_incident(id: str):
	"""Close a resolved incident."""
	svc = _svc()
	data = request.get_json(force=True) or {}
	close_notes = data.get("close_notes", "")
	closed_by = data.get("closed_by", getattr(g, "user_id", "system"))
	try:
		result = _loop().run_until_complete(svc.close_incident(id, close_notes, closed_by))
		return jsonify(result.model_dump())
	except (KeyError, PermissionError) as exc:
		return _err(str(exc), 404 if isinstance(exc, KeyError) else 403)


@api_bp.post("/incidents/<string:id>/notify-regulatory")
def notify_regulatory(id: str):
	"""Send regulatory notification for a reportable incident."""
	svc = _svc()
	data = request.get_json(force=True) or {}
	sent_by = data.get("sent_by", getattr(g, "user_id", "system"))
	try:
		result = _loop().run_until_complete(svc.send_regulatory_notification(id, sent_by))
		return jsonify(result.model_dump())
	except KeyError as exc:
		return _err(str(exc), 404)


# ── Hazards ────────────────────────────────────────────────────────────────────

@api_bp.get("/hazards")
def list_hazards():
	"""List hazards."""
	svc = _svc()
	results = _loop().run_until_complete(
		svc.list_hazards(
			risk_rating=request.args.get("risk_rating"),
			mine_area=request.args.get("mine_area"),
			open_only=request.args.get("open_only", "true").lower() == "true",
		)
	)
	return jsonify({"count": len(results), "items": [r.model_dump() for r in results]})


@api_bp.post("/hazards")
def identify_hazard():
	"""Identify and record a hazard."""
	svc = _svc()
	data = request.get_json(force=True) or {}
	data["tenant_id"] = getattr(g, "tenant_id", "default")
	try:
		payload = HazardCreate(**data)
		result = _loop().run_until_complete(
			svc.identify_hazard(payload, created_by=getattr(g, "user_id", "system"))
		)
		return jsonify(result.model_dump()), 201
	except (ValueError, AssertionError, PermissionError) as exc:
		return _err(str(exc), 403 if isinstance(exc, PermissionError) else 400)


@api_bp.get("/hazards/<string:id>")
def get_hazard(id: str):
	"""Get a hazard."""
	svc = _svc()
	result = _loop().run_until_complete(svc.get_hazard(id))
	if result is None:
		return _err("Not found", 404)
	return jsonify(result.model_dump())


@api_bp.post("/hazards/<string:id>/close")
def close_hazard(id: str):
	"""Close a resolved hazard."""
	svc = _svc()
	data = request.get_json(force=True) or {}
	try:
		result = _loop().run_until_complete(svc.close_hazard(id, data.get("close_notes", "")))
		return jsonify(result.model_dump())
	except KeyError as exc:
		return _err(str(exc), 404)


# ── Risk Register ──────────────────────────────────────────────────────────────

@api_bp.get("/risk-register")
def list_risk_register():
	"""List risk register entries."""
	svc = _svc()
	min_rating_str = request.args.get("min_rating")
	min_rating = RiskRating(min_rating_str) if min_rating_str else None
	results = _loop().run_until_complete(svc.list_risk_register(min_rating=min_rating))
	return jsonify({"count": len(results), "items": [r.model_dump() for r in results]})


@api_bp.post("/risk-register")
def add_risk_register_entry():
	"""Add a risk register entry."""
	svc = _svc()
	data = request.get_json(force=True) or {}
	data["tenant_id"] = getattr(g, "tenant_id", "default")
	try:
		payload = RiskRegisterEntryCreate(**data)
		result = _loop().run_until_complete(
			svc.add_risk_register_entry(payload, created_by=getattr(g, "user_id", "system"))
		)
		return jsonify(result.model_dump()), 201
	except (ValueError, AssertionError) as exc:
		return _err(str(exc))


@api_bp.get("/risk-register/<string:id>")
def get_risk_register_entry(id: str):
	"""Get a risk register entry."""
	svc = _svc()
	result = _loop().run_until_complete(svc.get_risk_register_entry(id))
	if result is None:
		return _err("Not found", 404)
	return jsonify(result.model_dump())


# ── Permits to Work ────────────────────────────────────────────────────────────

@api_bp.get("/permits")
def list_permits():
	"""List active permits to work."""
	svc = _svc()
	results = _loop().run_until_complete(
		svc.list_active_permits(mine_area=request.args.get("mine_area"))
	)
	return jsonify({"count": len(results), "items": [r.model_dump() for r in results]})


@api_bp.post("/permits")
def issue_permit():
	"""Issue a permit to work."""
	svc = _svc()
	data = request.get_json(force=True) or {}
	data["tenant_id"] = getattr(g, "tenant_id", "default")
	try:
		payload = PermitToWorkCreate(**data)
		result = _loop().run_until_complete(
			svc.issue_permit(payload, created_by=getattr(g, "user_id", "system"))
		)
		return jsonify(result.model_dump()), 201
	except (ValueError, AssertionError) as exc:
		return _err(str(exc))


@api_bp.get("/permits/<string:id>")
def get_permit(id: str):
	"""Get a permit to work."""
	svc = _svc()
	result = _loop().run_until_complete(svc.get_permit(id))
	if result is None:
		return _err("Not found", 404)
	return jsonify(result.model_dump())


@api_bp.post("/permits/<string:id>/close")
def close_permit(id: str):
	"""Close a permit to work."""
	svc = _svc()
	data = request.get_json(force=True) or {}
	closed_by = data.get("closed_by", getattr(g, "user_id", "system"))
	try:
		result = _loop().run_until_complete(svc.close_permit(id, closed_by))
		return jsonify(result.model_dump())
	except (KeyError, ValueError) as exc:
		return _err(str(exc), 404 if isinstance(exc, KeyError) else 400)


@api_bp.get("/permits/<string:id>/valid")
def check_permit_valid(id: str):
	"""Check if a permit is currently valid."""
	svc = _svc()
	valid = _loop().run_until_complete(svc.check_permit_valid(id))
	return jsonify({"permit_id": id, "valid": valid})


# ── Corrective Actions ─────────────────────────────────────────────────────────

@api_bp.get("/corrective-actions")
def list_corrective_actions():
	"""List corrective actions."""
	svc = _svc()
	results = _loop().run_until_complete(
		svc.list_corrective_actions(
			status=request.args.get("status"),
			source_type=request.args.get("source_type"),
		)
	)
	return jsonify({"count": len(results), "items": [r.model_dump() for r in results]})


@api_bp.post("/corrective-actions")
def create_corrective_action():
	"""Create a corrective action."""
	svc = _svc()
	data = request.get_json(force=True) or {}
	data["tenant_id"] = getattr(g, "tenant_id", "default")
	try:
		payload = CorrectiveActionCreate(**data)
		result = _loop().run_until_complete(
			svc.create_corrective_action(payload, created_by=getattr(g, "user_id", "system"))
		)
		return jsonify(result.model_dump()), 201
	except (ValueError, AssertionError) as exc:
		return _err(str(exc))


@api_bp.post("/corrective-actions/<string:id>/close")
def close_corrective_action(id: str):
	"""Close a corrective action."""
	svc = _svc()
	data = request.get_json(force=True) or {}
	closed_by = data.get("closed_by", getattr(g, "user_id", "system"))
	try:
		result = _loop().run_until_complete(svc.close_corrective_action(id, closed_by, data.get("notes")))
		return jsonify(result.model_dump())
	except KeyError as exc:
		return _err(str(exc), 404)


@api_bp.post("/corrective-actions/flag-overdue")
def flag_overdue():
	"""Scan and flag overdue corrective actions."""
	svc = _svc()
	overdue = _loop().run_until_complete(svc.flag_overdue_corrective_actions())
	return jsonify({"flagged_count": len(overdue), "items": [r.model_dump() for r in overdue]})


# ── Statistics ─────────────────────────────────────────────────────────────────

@api_bp.get("/statistics")
def safety_statistics():
	"""Safety KPI statistics."""
	svc = _svc()
	stats = _loop().run_until_complete(svc.get_safety_statistics())
	return jsonify(stats)
