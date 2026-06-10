"""Feature Flags — Flask Blueprint REST API."""
from __future__ import annotations

import logging
from typing import Any

from flask import Blueprint, jsonify, request

from .service import FeatureFlagService

_log = logging.getLogger(__name__)

bp = Blueprint("fflag", __name__, url_prefix="/api/fflag")
_svc: FeatureFlagService | None = None


def _get_service() -> FeatureFlagService:
	global _svc
	if _svc is None:
		_svc = FeatureFlagService()
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


@bp.get("/flags")
def list_flags():
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	enabled_raw = request.args.get("enabled")
	enabled = None if enabled_raw is None else (enabled_raw.lower() == "true")
	result = _run(svc.list_flags(tenant_id, enabled=enabled))
	return jsonify({"items": result, "total": len(result)}), 200


@bp.post("/flags")
def create_flag():
	svc = _get_service()
	body = request.get_json(force=True) or {}
	tenant_id = body.pop("tenant_id", "default")
	try:
		result = _run(svc.create_flag(tenant_id=tenant_id, **body))
		return jsonify(result), 201
	except (ValueError, KeyError) as exc:
		return jsonify({"error": str(exc)}), 400
	except Exception as exc:
		_log.error("create_flag error: %s", exc)
		return jsonify({"error": "internal_error"}), 500


@bp.get("/flags/<key>")
def get_flag(key: str):
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(svc.get_flag(tenant_id, key))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.put("/flags/<key>")
def update_flag(key: str):
	svc = _get_service()
	body = request.get_json(force=True) or {}
	tenant_id = body.pop("tenant_id", "default")
	actor = body.pop("actor", "system")
	try:
		return jsonify(_run(svc.update_flag(tenant_id, key, actor=actor, **body))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except ValueError as exc:
		return jsonify({"error": str(exc)}), 400


@bp.delete("/flags/<key>")
def delete_flag(key: str):
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	actor = request.args.get("actor", "system")
	try:
		return jsonify(_run(svc.delete_flag(tenant_id, key, actor=actor))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/flags/<key>/enable")
def enable_flag(key: str):
	svc = _get_service()
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	actor = body.get("actor", "system")
	try:
		return jsonify(_run(svc.enable_flag(tenant_id, key, actor=actor))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/flags/<key>/disable")
def disable_flag(key: str):
	svc = _get_service()
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	actor = body.get("actor", "system")
	try:
		return jsonify(_run(svc.disable_flag(tenant_id, key, actor=actor))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/flags/<key>/rollout")
def set_rollout(key: str):
	svc = _get_service()
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	percentage = body.get("percentage", 0.0)
	actor = body.get("actor", "system")
	try:
		return jsonify(_run(svc.set_rollout(tenant_id, key, float(percentage), actor=actor))), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/evaluate/<key>")
def evaluate_flag(key: str):
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	user_id = request.args.get("user_id", "anonymous")
	return jsonify(_run(svc.evaluate_flag(tenant_id, key, user_id))), 200


@bp.post("/evaluate/batch")
def evaluate_many():
	svc = _get_service()
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	flag_keys = body.get("flag_keys", [])
	user_id = body.get("user_id", "anonymous")
	user_attributes = body.get("user_attributes")
	return jsonify(_run(svc.evaluate_many(tenant_id, flag_keys, user_id, user_attributes))), 200


@bp.post("/evaluate/all")
def bulk_evaluate():
	svc = _get_service()
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	user_id = body.get("user_id", "anonymous")
	user_attributes = body.get("user_attributes")
	return jsonify(_run(svc.bulk_evaluate(tenant_id, user_id, user_attributes))), 200


@bp.post("/overrides")
def set_override():
	svc = _get_service()
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		result = _run(svc.set_override(
			tenant_id,
			body.get("flag_key", ""),
			body.get("user_id", ""),
			body.get("enabled", False),
			body.get("variant"),
			body.get("reason", ""),
			body.get("actor", "system"),
		))
		return jsonify(result), 201
	except (ValueError, KeyError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.delete("/overrides")
def clear_override():
	svc = _get_service()
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		result = _run(svc.clear_override(tenant_id, body.get("flag_key", ""), body.get("user_id", "")))
		return jsonify(result), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/overrides")
def list_overrides():
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	key = request.args.get("flag_key")
	result = _run(svc.list_overrides(tenant_id, key))
	return jsonify({"items": result, "total": len(result)}), 200


@bp.post("/experiments")
def create_experiment():
	svc = _get_service()
	body = request.get_json(force=True) or {}
	tenant_id = body.pop("tenant_id", "default")
	try:
		return jsonify(_run(svc.create_experiment(tenant_id=tenant_id, **body))), 201
	except (ValueError, KeyError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/experiments")
def list_experiments():
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	flag_key = request.args.get("flag_key")
	result = _run(svc.list_experiments(tenant_id, flag_key))
	return jsonify({"items": result, "total": len(result)}), 200


@bp.get("/experiments/<experiment_id>")
def get_experiment(experiment_id: str):
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(svc.get_experiment(tenant_id, experiment_id))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/experiments/<experiment_id>/start")
def start_experiment(experiment_id: str):
	svc = _get_service()
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		return jsonify(_run(svc.start_experiment(tenant_id, experiment_id))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/experiments/<experiment_id>/stop")
def stop_experiment(experiment_id: str):
	svc = _get_service()
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	winner = body.get("winner")
	try:
		return jsonify(_run(svc.stop_experiment(tenant_id, experiment_id, winner=winner))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/experiments/<experiment_id>/results")
def experiment_results(experiment_id: str):
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(svc.get_experiment_results(tenant_id, experiment_id))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/experiments/<experiment_id>/assign")
def assign_variant(experiment_id: str):
	svc = _get_service()
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	user_id = body.get("user_id", "anonymous")
	user_attributes = body.get("user_attributes")
	try:
		return jsonify(_run(svc.assign_experiment_variant(tenant_id, experiment_id, user_id, user_attributes))), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/statistics")
def statistics():
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	return jsonify(_run(svc.flag_statistics(tenant_id))), 200


@bp.get("/audit")
def audit_events():
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	result = _run(svc.get_audit_events(tenant_id))
	return jsonify({"items": result, "total": len(result)}), 200


@bp.get("/flags/<key>/history")
def flag_history(key: str):
	svc = _get_service()
	tenant_id = request.args.get("tenant_id", "default")
	result = _run(svc.get_flag_history(tenant_id, key))
	return jsonify({"items": result, "total": len(result)}), 200
