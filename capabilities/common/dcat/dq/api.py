"""Data Quality — Flask Blueprint REST API."""
from __future__ import annotations

import logging
from typing import Any

from flask import Blueprint, jsonify, request

from .service import DataQualityService

_log = logging.getLogger(__name__)

bp = Blueprint("dcat_dq", __name__, url_prefix="/api/dcat/dq")
_svc: DataQualityService | None = None


def _get_service() -> DataQualityService:
	global _svc
	if _svc is None:
		_svc = DataQualityService()
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


@bp.get("/rules")
def list_rules():
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	dataset_id = request.args.get("dataset_id")
	rule_type = request.args.get("rule_type")
	result = _run(svc.list_rules(tenant_id, dataset_id=dataset_id, rule_type=rule_type))
	return jsonify({"items": result, "total": len(result)}), 200


@bp.post("/rules")
def create_rule():
	svc = _get_service()
	body = request.get_json(force=True) or {}
	tenant_id = body.pop("tenant_id", "default")
	try:
		result = _run(svc.create_rule(tenant_id=tenant_id, **body))
		return jsonify(result), 201
	except (ValueError, KeyError) as exc:
		return jsonify({"error": str(exc)}), 400
	except Exception as exc:
		_log.error("create_rule error: %s", exc)
		return jsonify({"error": "internal_error"}), 500


@bp.get("/rules/<rule_id>")
def get_rule(rule_id: str):
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(svc.get_rule(tenant_id, rule_id))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.put("/rules/<rule_id>")
def update_rule(rule_id: str):
	svc = _get_service()
	body = request.get_json(force=True) or {}
	tenant_id = body.pop("tenant_id", "default")
	try:
		return jsonify(_run(svc.update_rule(tenant_id, rule_id, **body))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.delete("/rules/<rule_id>")
def delete_rule(rule_id: str):
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(svc.delete_rule(tenant_id, rule_id))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/profiles")
def profile_dataset():
	svc = _get_service()
	body = request.get_json(force=True) or {}
	tenant_id = body.pop("tenant_id", "default")
	try:
		result = _run(svc.profile_dataset(tenant_id=tenant_id, **body))
		return jsonify(result), 201
	except (ValueError, KeyError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/profiles/<dataset_id>")
def get_profile(dataset_id: str):
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(svc.get_profile(tenant_id, dataset_id))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/runs")
def run_checks():
	svc = _get_service()
	body = request.get_json(force=True) or {}
	tenant_id = body.pop("tenant_id", "default")
	dataset_id = body.pop("dataset_id", "")
	data_sample = body.pop("data_sample", None)
	try:
		result = _run(svc.run_quality_checks(tenant_id, dataset_id, data_sample))
		return jsonify(result), 201
	except (ValueError, KeyError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/runs")
def list_runs():
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	dataset_id = request.args.get("dataset_id")
	result = _run(svc.list_runs(tenant_id, dataset_id))
	return jsonify({"items": result, "total": len(result)}), 200


@bp.get("/runs/<run_id>")
def get_run(run_id: str):
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(svc.get_run(tenant_id, run_id))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/anomalies")
def list_anomalies():
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	dataset_id = request.args.get("dataset_id")
	result = _run(svc.list_anomalies(tenant_id, dataset_id))
	return jsonify({"items": result, "total": len(result)}), 200


@bp.post("/anomalies/<anomaly_id>/acknowledge")
def acknowledge_anomaly(anomaly_id: str):
	svc = _get_service()
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	acknowledged_by = body.get("acknowledged_by", "unknown")
	try:
		return jsonify(_run(svc.acknowledge_anomaly(tenant_id, anomaly_id, acknowledged_by))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/scorecard/<dataset_id>")
def get_scorecard(dataset_id: str):
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(svc.get_scorecard(tenant_id, dataset_id))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/reports/<dataset_id>")
def dq_report(dataset_id: str):
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	period_start = request.args.get("period_start", "2000-01-01")
	period_end = request.args.get("period_end", "2099-12-31")
	result = _run(svc.generate_dq_report(tenant_id, dataset_id, period_start, period_end))
	return jsonify(result), 200


@bp.get("/dashboard")
def dashboard():
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	return jsonify(_run(svc.dq_dashboard(tenant_id))), 200


@bp.get("/audit")
def audit_events():
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	result = _run(svc.get_audit_events(tenant_id))
	return jsonify({"items": result, "total": len(result)}), 200
