"""Process Mining — Flask Blueprint REST API."""
from __future__ import annotations

import logging
from typing import Any

from flask import Blueprint, jsonify, request

from .service import ProcessMiningService

_log = logging.getLogger(__name__)

bp = Blueprint("pmin", __name__, url_prefix="/api/pmin")
_svc: ProcessMiningService | None = None


def _get_service() -> ProcessMiningService:
	global _svc
	if _svc is None:
		_svc = ProcessMiningService()
	return _svc


def _run(coro: Any) -> Any:
	import asyncio
	try:
		loop = asyncio.get_event_loop()
		if loop.is_running():
			import concurrent.futures
			with concurrent.futures.ThreadPoolExecutor() as pool:
				return pool.submit(asyncio.run, coro).result()
		return loop.run_until_complete(coro)
	except RuntimeError:
		return asyncio.run(coro)


@bp.get("/health")
def health():
	return jsonify(_run(_get_service().health_check())), 200


# ── Event logs ────────────────────────────────────────────────────

@bp.get("/logs")
def list_event_logs():
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	result = _run(svc.list_event_logs(tenant_id))
	return jsonify({"items": result, "total": len(result)}), 200


@bp.post("/logs")
def create_event_log():
	svc = _get_service()
	body = request.get_json(force=True) or {}
	tenant_id = body.pop("tenant_id", "default")
	try:
		return jsonify(_run(svc.create_event_log(tenant_id=tenant_id, **body))), 201
	except (ValueError, KeyError) as exc:
		return jsonify({"error": str(exc)}), 400
	except Exception as exc:
		_log.error("create_event_log error: %s", exc)
		return jsonify({"error": "internal_error"}), 500


@bp.get("/logs/<log_id>")
def get_event_log(log_id: str):
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(svc.get_event_log(tenant_id, log_id))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.put("/logs/<log_id>")
def update_event_log(log_id: str):
	svc = _get_service()
	body = request.get_json(force=True) or {}
	tenant_id = body.pop("tenant_id", "default")
	try:
		return jsonify(_run(svc.update_event_log(tenant_id, log_id, **body))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.delete("/logs/<log_id>")
def delete_event_log(log_id: str):
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(svc.delete_event_log(tenant_id, log_id))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


# ── Event ingestion ───────────────────────────────────────────────

@bp.post("/logs/<log_id>/events")
def ingest_events(log_id: str):
	svc = _get_service()
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	events = body.get("events", [])
	try:
		return jsonify(_run(svc.ingest_events(tenant_id, log_id, events))), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/logs/<log_id>/events/nats")
def ingest_nats_events(log_id: str):
	svc = _get_service()
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	messages = body.get("messages", [])
	try:
		return jsonify(_run(svc.ingest_nats_events(tenant_id, log_id, messages))), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/logs/<log_id>/events")
def get_events(log_id: str):
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	case_id = request.args.get("case_id")
	activity = request.args.get("activity")
	limit = int(request.args.get("limit", 1000))
	try:
		result = _run(svc.get_events(tenant_id, log_id, case_id=case_id, activity=activity, limit=limit))
		return jsonify({"items": result, "total": len(result)}), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/logs/<log_id>/cases/<case_id>")
def get_case_trace(log_id: str, case_id: str):
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(svc.get_case_trace(tenant_id, log_id, case_id))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


# ── Process discovery ─────────────────────────────────────────────

@bp.post("/logs/<log_id>/discover")
def discover_process_model(log_id: str):
	svc = _get_service()
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	algorithm = body.get("algorithm", "alpha_miner")
	noise_threshold = float(body.get("noise_threshold", 0.2))
	try:
		return jsonify(_run(svc.discover_process_model(tenant_id, log_id, algorithm, noise_threshold))), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/models")
def list_bpmn_models():
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	log_id = request.args.get("log_id")
	result = _run(svc.list_bpmn_models(tenant_id, log_id))
	return jsonify({"items": result, "total": len(result)}), 200


@bp.get("/models/<model_id>")
def get_bpmn_model(model_id: str):
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(svc.get_bpmn_model(tenant_id, model_id))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.delete("/models/<model_id>")
def delete_bpmn_model(model_id: str):
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(svc.delete_bpmn_model(tenant_id, model_id))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/models/<model_id>/xml")
def export_bpmn_xml(model_id: str):
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(svc.export_bpmn_xml(tenant_id, model_id))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


# ── Conformance ───────────────────────────────────────────────────

@bp.post("/conformance")
def check_conformance():
	svc = _get_service()
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	log_id = body.get("log_id", "")
	model_id = body.get("model_id", "")
	try:
		return jsonify(_run(svc.check_conformance(tenant_id, log_id, model_id))), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/conformance")
def list_conformance():
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	log_id = request.args.get("log_id")
	result = _run(svc.list_conformance_results(tenant_id, log_id))
	return jsonify({"items": result, "total": len(result)}), 200


@bp.post("/conformance/deviating-cases")
def filter_deviating_cases():
	svc = _get_service()
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	log_id = body.get("log_id", "")
	model_id = body.get("model_id", "")
	try:
		result = _run(svc.filter_deviating_cases(tenant_id, log_id, model_id))
		return jsonify({"items": result, "total": len(result)}), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


# ── Bottleneck analysis ───────────────────────────────────────────

@bp.post("/logs/<log_id>/bottlenecks")
def analyze_bottlenecks(log_id: str):
	svc = _get_service()
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	top_n = int(body.get("top_n", 10))
	try:
		return jsonify(_run(svc.analyze_bottlenecks(tenant_id, log_id, top_n))), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/bottlenecks")
def list_bottleneck_reports():
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	log_id = request.args.get("log_id")
	result = _run(svc.list_bottleneck_reports(tenant_id, log_id))
	return jsonify({"items": result, "total": len(result)}), 200


@bp.get("/bottlenecks/<report_id>")
def get_bottleneck_report(report_id: str):
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(svc.get_bottleneck_report(tenant_id, report_id))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


# ── Variants ──────────────────────────────────────────────────────

@bp.post("/logs/<log_id>/variants")
def discover_variants(log_id: str):
	svc = _get_service()
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	top_n = int(body.get("top_n", 20))
	try:
		return jsonify(_run(svc.discover_variants(tenant_id, log_id, top_n))), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/variants")
def list_variant_analyses():
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	log_id = request.args.get("log_id")
	result = _run(svc.list_variant_analyses(tenant_id, log_id))
	return jsonify({"items": result, "total": len(result)}), 200


@bp.get("/variants/<analysis_id>")
def get_variant_analysis(analysis_id: str):
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(svc.get_variant_analysis(tenant_id, analysis_id))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


# ── Simulation ────────────────────────────────────────────────────

@bp.post("/models/<model_id>/simulate")
def simulate_process(model_id: str):
	svc = _get_service()
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	simulation_cases = int(body.get("simulation_cases", 100))
	try:
		return jsonify(_run(svc.simulate_process(tenant_id, model_id, simulation_cases))), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


# ── Performance ───────────────────────────────────────────────────

@bp.get("/logs/<log_id>/performance")
def performance_metrics(log_id: str):
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(svc.get_performance_metrics(tenant_id, log_id))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


# ── Dashboard + audit ─────────────────────────────────────────────

@bp.get("/dashboard")
def dashboard():
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	return jsonify(_run(svc.process_mining_dashboard(tenant_id))), 200


@bp.get("/audit")
def audit_events():
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	result = _run(svc.get_audit_events(tenant_id))
	return jsonify({"items": result, "total": len(result)}), 200
