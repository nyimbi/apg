"""M&E — Flask Blueprint with async REST endpoints."""
from __future__ import annotations

import logging

from flask import Blueprint, jsonify, request

from .service import MEService

_log = logging.getLogger(__name__)

bp = Blueprint("ngo_me", __name__, url_prefix="/api/ngo/me")

_svc: MEService | None = None


def _get_service() -> MEService:
	global _svc
	if _svc is None:
		_svc = MEService()
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


@bp.get("/indicators")
def list_indicators():
	svc = _get_service()
	result = _run(svc.list_indicators(
		programme_id=request.args.get("programme_id"),
		indicator_type=request.args.get("indicator_type"),
		status=request.args.get("status"),
	))
	return jsonify({"indicators": result, "count": len(result)}), 200


@bp.get("/indicators/<indicator_id>")
def get_indicator(indicator_id: str):
	try:
		return jsonify(_run(_get_service().get_indicator(indicator_id))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/indicators")
def create_indicator():
	data = request.get_json(force=True) or {}
	try:
		result = _run(_get_service().create_indicator(
			programme_id=data["programme_id"],
			name=data["name"],
			code=data["code"],
			target_value=float(data["target_value"]),
			target_date=data["target_date"],
			indicator_type=data.get("indicator_type", "output"),
			description=data.get("description", ""),
			unit=data.get("unit", ""),
			baseline_value=float(data.get("baseline_value", 0)),
			baseline_date=data.get("baseline_date", ""),
			disaggregation=data.get("disaggregation", []),
			data_source=data.get("data_source", ""),
			collection_method=data.get("collection_method", ""),
		))
		return jsonify(result), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.put("/indicators/<indicator_id>")
def update_indicator(indicator_id: str):
	data = request.get_json(force=True) or {}
	try:
		return jsonify(_run(_get_service().update_indicator(indicator_id, **data))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.delete("/indicators/<indicator_id>")
def delete_indicator(indicator_id: str):
	try:
		return jsonify(_run(_get_service().delete_indicator(indicator_id))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/indicators/<indicator_id>/trend")
def indicator_trend(indicator_id: str):
	try:
		return jsonify(_run(_get_service().trend_analysis(indicator_id))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/data-collections")
def collect_data():
	data = request.get_json(force=True) or {}
	try:
		result = _run(_get_service().collect_data(
			indicator_id=data["indicator_id"],
			programme_id=data["programme_id"],
			value=float(data["value"]),
			collection_date=data["collection_date"],
			collected_by=data["collected_by"],
			period=data.get("period", ""),
			disaggregation_values=data.get("disaggregation_values", {}),
			notes=data.get("notes", ""),
		))
		return jsonify(result), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/data-collections/bulk")
def bulk_collect_data():
	data = request.get_json(force=True) or {}
	result = _run(_get_service().bulk_collect_data(data.get("data_points", [])))
	return jsonify(result), 200


@bp.get("/data-collections")
def list_data_collections():
	result = _run(_get_service().list_data_collections(
		indicator_id=request.args.get("indicator_id"),
		programme_id=request.args.get("programme_id"),
	))
	return jsonify({"data_collections": result, "count": len(result)}), 200


@bp.get("/progress-reports")
def list_progress_reports():
	result = _run(_get_service().list_progress_reports(
		programme_id=request.args.get("programme_id")
	))
	return jsonify({"reports": result, "count": len(result)}), 200


@bp.post("/progress-reports")
def create_progress_report():
	data = request.get_json(force=True) or {}
	try:
		result = _run(_get_service().create_progress_report(
			programme_id=data["programme_id"],
			report_period=data["report_period"],
			period_start=data["period_start"],
			period_end=data["period_end"],
			prepared_by=data["prepared_by"],
			narrative=data.get("narrative", ""),
			key_achievements=data.get("key_achievements", []),
			challenges=data.get("challenges", []),
			lessons_learned=data.get("lessons_learned", []),
		))
		return jsonify(result), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/progress-reports/<report_id>/submit")
def submit_progress_report(report_id: str):
	try:
		return jsonify(_run(_get_service().submit_progress_report(report_id))), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/evaluations")
def list_evaluations():
	result = _run(_get_service().list_evaluations(
		programme_id=request.args.get("programme_id")
	))
	return jsonify({"evaluations": result, "count": len(result)}), 200


@bp.post("/evaluations")
def create_evaluation():
	data = request.get_json(force=True) or {}
	try:
		result = _run(_get_service().create_evaluation(
			programme_id=data["programme_id"],
			evaluator=data["evaluator"],
			evaluation_date=data["evaluation_date"],
			evaluation_type=data.get("evaluation_type", "mid_term"),
			scope=data.get("scope", ""),
			methodology=data.get("methodology", ""),
			findings=data.get("findings", ""),
			recommendations=data.get("recommendations", ""),
			rating=data.get("rating", "satisfactory"),
		))
		return jsonify(result), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/learning-cycles")
def list_learning_cycles():
	result = _run(_get_service().list_learning_cycles(
		programme_id=request.args.get("programme_id")
	))
	return jsonify({"learning_cycles": result, "count": len(result)}), 200


@bp.post("/learning-cycles")
def create_learning_cycle():
	data = request.get_json(force=True) or {}
	try:
		result = _run(_get_service().create_learning_cycle(
			programme_id=data["programme_id"],
			cycle_name=data["cycle_name"],
			start_date=data["start_date"],
			end_date=data["end_date"],
			facilitator=data["facilitator"],
			learning_questions=data.get("learning_questions", []),
			notes=data.get("notes", ""),
		))
		return jsonify(result), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/learning-cycles/<cycle_id>/findings")
def add_learning_findings(cycle_id: str):
	data = request.get_json(force=True) or {}
	try:
		result = _run(_get_service().add_learning_findings(
			cycle_id=cycle_id,
			findings=data.get("findings", []),
			action_points=data.get("action_points", []),
		))
		return jsonify(result), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/dashboard/<programme_id>")
def indicator_dashboard(programme_id: str):
	return jsonify(_run(_get_service().indicator_performance_dashboard(programme_id))), 200


@bp.get("/impact/<programme_id>")
def impact_summary(programme_id: str):
	return jsonify(_run(_get_service().impact_assessment_summary(programme_id))), 200


@bp.get("/audit-events")
def get_audit_events():
	limit = int(request.args.get("limit", 100))
	result = _run(_get_service().get_audit_events(limit=limit))
	return jsonify({"events": result, "count": len(result)}), 200
