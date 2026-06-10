"""Programme & Project Monitoring — Flask Blueprint with async REST endpoints."""
from __future__ import annotations

import logging
from decimal import Decimal

from flask import Blueprint, jsonify, request

from .service import ProgrammeMonitoringService

_log = logging.getLogger(__name__)

bp = Blueprint("ngo_prg", __name__, url_prefix="/api/ngo/prg")

_svc: ProgrammeMonitoringService | None = None


def _get_service() -> ProgrammeMonitoringService:
	global _svc
	if _svc is None:
		_svc = ProgrammeMonitoringService()
	return _svc


def _run(coro):
	import asyncio
	try:
		loop = asyncio.get_event_loop()
		if loop.is_running():
			import concurrent.futures
			with concurrent.futures.ThreadPoolExecutor() as pool:
				return pool.submit(asyncio.run, coro).result()
		return loop.run_until_complete(coro)
	except Exception as exc:
		_log.error("async execution error: %s", exc)
		raise


@bp.get("/health")
def health():
	return jsonify(_run(_get_service().health_check())), 200


@bp.get("/")
def list_programmes():
	svc = _get_service()
	result = _run(svc.list_programmes(
		status=request.args.get("status"),
		sector=request.args.get("sector"),
	))
	return jsonify({"programmes": result, "count": len(result)}), 200


@bp.get("/<programme_id>")
def get_programme(programme_id: str):
	try:
		return jsonify(_run(_get_service().get_programme(programme_id))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/")
def create_programme():
	data = request.get_json(force=True) or {}
	try:
		result = _run(_get_service().create_programme(
			name=data["name"],
			code=data["code"],
			start_date=data["start_date"],
			end_date=data["end_date"],
			description=data.get("description", ""),
			sector=data.get("sector", ""),
			budget=Decimal(str(data.get("budget", 0))),
			currency=data.get("currency", "KES"),
			lead_staff=data.get("lead_staff", ""),
			geographic_focus=data.get("geographic_focus", ""),
		))
		return jsonify(result), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.put("/<programme_id>")
def update_programme(programme_id: str):
	data = request.get_json(force=True) or {}
	try:
		return jsonify(_run(_get_service().update_programme(programme_id, **data))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except ValueError as exc:
		return jsonify({"error": str(exc)}), 400


@bp.delete("/<programme_id>")
def delete_programme(programme_id: str):
	try:
		return jsonify(_run(_get_service().delete_programme(programme_id))), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/<programme_id>/activate")
def activate_programme(programme_id: str):
	try:
		return jsonify(_run(_get_service().activate_programme(programme_id))), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/<programme_id>/logframes")
def list_logframes(programme_id: str):
	result = _run(_get_service().list_logframes(programme_id=programme_id))
	return jsonify({"logframes": result, "count": len(result)}), 200


@bp.post("/<programme_id>/logframes")
def create_logframe(programme_id: str):
	data = request.get_json(force=True) or {}
	try:
		result = _run(_get_service().create_logframe(
			programme_id=programme_id,
			goal=data["goal"],
			purpose=data["purpose"],
			outputs=data.get("outputs", []),
			activities=data.get("activities", []),
			assumptions=data.get("assumptions", []),
			version=data.get("version", "1.0"),
		))
		return jsonify(result), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/<programme_id>/activities")
def list_activities(programme_id: str):
	result = _run(_get_service().list_activities(
		programme_id=programme_id,
		status=request.args.get("status"),
	))
	return jsonify({"activities": result, "count": len(result)}), 200


@bp.post("/<programme_id>/activities")
def create_activity(programme_id: str):
	data = request.get_json(force=True) or {}
	try:
		result = _run(_get_service().create_activity(
			programme_id=programme_id,
			name=data["name"],
			planned_start=data["planned_start"],
			planned_end=data["planned_end"],
			description=data.get("description", ""),
			responsible_person=data.get("responsible_person", ""),
			budget=Decimal(str(data.get("budget", 0))),
			currency=data.get("currency", "KES"),
			logframe_id=data.get("logframe_id"),
		))
		return jsonify(result), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/<programme_id>/outputs")
def list_outputs(programme_id: str):
	result = _run(_get_service().list_outputs(programme_id=programme_id))
	return jsonify({"outputs": result, "count": len(result)}), 200


@bp.post("/<programme_id>/field-data")
def submit_field_data(programme_id: str):
	data = request.get_json(force=True) or {}
	try:
		result = _run(_get_service().submit_field_data(
			programme_id=programme_id,
			collector=data["collector"],
			collection_date=data["collection_date"],
			data=data.get("data", {}),
			activity_id=data.get("activity_id"),
			location=data.get("location", ""),
			data_type=data.get("data_type", "observation"),
			notes=data.get("notes", ""),
		))
		return jsonify(result), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/<programme_id>/field-data")
def list_field_data(programme_id: str):
	result = _run(_get_service().list_field_data(programme_id=programme_id))
	return jsonify({"field_data": result, "count": len(result)}), 200


@bp.get("/<programme_id>/progress")
def programme_progress(programme_id: str):
	try:
		return jsonify(_run(_get_service().programme_progress_report(programme_id))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/<programme_id>/gantt")
def activity_gantt(programme_id: str):
	try:
		result = _run(_get_service().activity_gantt_data(programme_id))
		return jsonify({"activities": result}), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/portfolio/overview")
def portfolio_overview():
	return jsonify(_run(_get_service().portfolio_overview())), 200


@bp.get("/audit-events")
def get_audit_events():
	limit = int(request.args.get("limit", 100))
	result = _run(_get_service().get_audit_events(limit=limit))
	return jsonify({"events": result, "count": len(result)}), 200
