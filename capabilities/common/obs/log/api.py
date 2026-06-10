"""Flask Blueprint REST API for Log Aggregation (obs_log)."""
from __future__ import annotations

import logging
from typing import Any

from flask import Blueprint, request, jsonify

from .service import LogAggregationService

_log = logging.getLogger(__name__)

log_bp = Blueprint("obs_log", __name__, url_prefix="/api/obs/log")

_services: dict[str, LogAggregationService] = {}


def _svc(tenant_id: str = "default") -> LogAggregationService:
	if tenant_id not in _services:
		_services[tenant_id] = LogAggregationService(tenant_id=tenant_id)
	return _services[tenant_id]


def _tenant() -> str:
	return request.headers.get("X-Tenant-ID", "default")


def _ok(data: Any, status: int = 200):
	return jsonify(data), status


def _err(message: str, status: int = 400):
	return jsonify({"error": message}), status


# ------------------------------------------------------------------ health

@log_bp.route("/health", methods=["GET"])
async def health():
	try:
		result = await _svc(_tenant()).health_check()
		return _ok(result)
	except Exception as exc:
		_log.error("health check error: %s", exc)
		return _err(str(exc), 500)


@log_bp.route("/describe", methods=["GET"])
async def describe():
	try:
		result = await _svc(_tenant()).describe()
		return _ok(result)
	except Exception as exc:
		_log.error("describe error: %s", exc)
		return _err(str(exc), 500)


# ------------------------------------------------------------------ log entries

@log_bp.route("/entries", methods=["GET"])
async def list_log_entries():
	try:
		params = request.args
		result = await _svc(_tenant()).list_log_entries(
			service_name=params.get("service_name"),
			level=params.get("level"),
			min_level=params.get("min_level"),
			correlation_id=params.get("correlation_id"),
			trace_id=params.get("trace_id"),
			start_time=params.get("start_time"),
			end_time=params.get("end_time"),
			message_contains=params.get("message_contains"),
			page=int(params.get("page", 1)),
			page_size=int(params.get("page_size", 100)),
		)
		return _ok(result)
	except Exception as exc:
		_log.error("list_log_entries error: %s", exc)
		return _err(str(exc), 400)


@log_bp.route("/entries/<entry_id>", methods=["GET"])
async def get_log_entry(entry_id: str):
	try:
		result = await _svc(_tenant()).get_log_entry(entry_id)
		return _ok(result)
	except KeyError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		_log.error("get_log_entry error: %s", exc)
		return _err(str(exc), 400)


@log_bp.route("/entries", methods=["POST"])
async def ingest_log():
	try:
		body = request.get_json(force=True)
		result = await _svc(_tenant()).ingest_log(
			service_name=body["service_name"],
			level=body.get("level", "INFO"),
			message=body["message"],
			timestamp=body.get("timestamp"),
			correlation_id=body.get("correlation_id"),
			trace_id=body.get("trace_id"),
			span_id=body.get("span_id"),
			fields=body.get("fields"),
			source_file=body.get("source_file"),
			source_line=body.get("source_line"),
			logger_name=body.get("logger_name"),
		)
		status = 200 if result.get("suppressed") else 201
		return _ok(result, status)
	except KeyError as exc:
		return _err(f"Missing field: {exc}", 400)
	except ValueError as exc:
		return _err(str(exc), 400)
	except Exception as exc:
		_log.error("ingest_log error: %s", exc)
		return _err(str(exc), 400)


@log_bp.route("/entries/bulk", methods=["POST"])
async def bulk_ingest_logs():
	try:
		body = request.get_json(force=True)
		entries = body if isinstance(body, list) else body.get("entries", [])
		result = await _svc(_tenant()).bulk_ingest_logs(entries)
		return _ok(result, 207)
	except Exception as exc:
		_log.error("bulk_ingest_logs error: %s", exc)
		return _err(str(exc), 400)


@log_bp.route("/entries/<entry_id>", methods=["DELETE"])
async def delete_log_entry(entry_id: str):
	try:
		result = await _svc(_tenant()).delete_log_entry(entry_id)
		return _ok(result)
	except KeyError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		_log.error("delete_log_entry error: %s", exc)
		return _err(str(exc), 400)


@log_bp.route("/entries/purge", methods=["DELETE"])
async def purge_log_entries():
	try:
		params = request.args
		result = await _svc(_tenant()).purge_log_entries(
			service_name=params.get("service_name"),
			before_timestamp=params.get("before_timestamp"),
		)
		return _ok(result)
	except Exception as exc:
		_log.error("purge_log_entries error: %s", exc)
		return _err(str(exc), 400)


# ------------------------------------------------------------------ search

@log_bp.route("/search", methods=["GET"])
async def search_logs():
	try:
		params = request.args
		result = await _svc(_tenant()).search_logs(
			query=params["query"],
			service_name=params.get("service_name"),
			page=int(params.get("page", 1)),
			page_size=int(params.get("page_size", 50)),
		)
		return _ok(result)
	except KeyError as exc:
		return _err(f"Missing field: {exc}", 400)
	except Exception as exc:
		_log.error("search_logs error: %s", exc)
		return _err(str(exc), 400)


# ------------------------------------------------------------------ correlation

@log_bp.route("/correlation", methods=["POST"])
async def create_correlation_context():
	try:
		body = request.get_json(force=True)
		result = await _svc(_tenant()).create_correlation_context(
			service_name=body["service_name"],
			correlation_id=body.get("correlation_id"),
			trace_id=body.get("trace_id"),
			request_id=body.get("request_id"),
			user_id=body.get("user_id"),
			session_id=body.get("session_id"),
			extra=body.get("extra"),
		)
		return _ok(result, 201)
	except KeyError as exc:
		return _err(f"Missing field: {exc}", 400)
	except Exception as exc:
		_log.error("create_correlation_context error: %s", exc)
		return _err(str(exc), 400)


@log_bp.route("/correlation/<ctx_id>", methods=["GET"])
async def get_correlation_context(ctx_id: str):
	try:
		result = await _svc(_tenant()).get_correlation_context(ctx_id)
		return _ok(result)
	except KeyError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		_log.error("get_correlation_context error: %s", exc)
		return _err(str(exc), 400)


@log_bp.route("/correlation", methods=["GET"])
async def list_correlation_contexts():
	try:
		params = request.args
		result = await _svc(_tenant()).list_correlation_contexts(
			service_name=params.get("service_name"),
			page=int(params.get("page", 1)),
			page_size=int(params.get("page_size", 50)),
		)
		return _ok(result)
	except Exception as exc:
		_log.error("list_correlation_contexts error: %s", exc)
		return _err(str(exc), 400)


@log_bp.route("/correlation/<ctx_id>", methods=["DELETE"])
async def delete_correlation_context(ctx_id: str):
	try:
		result = await _svc(_tenant()).delete_correlation_context(ctx_id)
		return _ok(result)
	except KeyError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		_log.error("delete_correlation_context error: %s", exc)
		return _err(str(exc), 400)


@log_bp.route("/by-correlation/<correlation_id>", methods=["GET"])
async def get_logs_by_correlation(correlation_id: str):
	try:
		result = await _svc(_tenant()).get_logs_by_correlation_id(
			correlation_id,
			page=int(request.args.get("page", 1)),
			page_size=int(request.args.get("page_size", 100)),
		)
		return _ok(result)
	except Exception as exc:
		_log.error("get_logs_by_correlation error: %s", exc)
		return _err(str(exc), 400)


@log_bp.route("/by-trace/<trace_id>", methods=["GET"])
async def get_logs_by_trace(trace_id: str):
	try:
		result = await _svc(_tenant()).get_logs_by_trace_id(
			trace_id,
			page=int(request.args.get("page", 1)),
			page_size=int(request.args.get("page_size", 100)),
		)
		return _ok(result)
	except Exception as exc:
		_log.error("get_logs_by_trace error: %s", exc)
		return _err(str(exc), 400)


# ------------------------------------------------------------------ retention policies

@log_bp.route("/retention", methods=["GET"])
async def list_retention_policies():
	try:
		enabled_only = request.args.get("enabled_only", "false").lower() == "true"
		result = await _svc(_tenant()).list_retention_policies(enabled_only=enabled_only)
		return _ok(result)
	except Exception as exc:
		_log.error("list_retention_policies error: %s", exc)
		return _err(str(exc), 400)


@log_bp.route("/retention", methods=["POST"])
async def create_retention_policy():
	try:
		body = request.get_json(force=True)
		result = await _svc(_tenant()).create_retention_policy(
			name=body["name"],
			retention_days=int(body.get("retention_days", 30)),
			min_level=body.get("min_level", "DEBUG"),
			service_name=body.get("service_name"),
			archive_after_days=body.get("archive_after_days"),
			delete_after_days=body.get("delete_after_days"),
			compress_after_days=body.get("compress_after_days"),
			enabled=bool(body.get("enabled", True)),
		)
		return _ok(result, 201)
	except KeyError as exc:
		return _err(f"Missing field: {exc}", 400)
	except ValueError as exc:
		return _err(str(exc), 400)
	except Exception as exc:
		_log.error("create_retention_policy error: %s", exc)
		return _err(str(exc), 400)


@log_bp.route("/retention/<policy_id>", methods=["GET"])
async def get_retention_policy(policy_id: str):
	try:
		result = await _svc(_tenant()).get_retention_policy(policy_id)
		return _ok(result)
	except KeyError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		_log.error("get_retention_policy error: %s", exc)
		return _err(str(exc), 400)


@log_bp.route("/retention/<policy_id>", methods=["PUT"])
async def update_retention_policy(policy_id: str):
	try:
		body = request.get_json(force=True) or {}
		result = await _svc(_tenant()).update_retention_policy(
			policy_id=policy_id,
			min_level=body.get("min_level"),
			retention_days=body.get("retention_days"),
			archive_after_days=body.get("archive_after_days"),
			delete_after_days=body.get("delete_after_days"),
			compress_after_days=body.get("compress_after_days"),
			enabled=body.get("enabled"),
		)
		return _ok(result)
	except KeyError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		_log.error("update_retention_policy error: %s", exc)
		return _err(str(exc), 400)


@log_bp.route("/retention/<policy_id>", methods=["DELETE"])
async def delete_retention_policy(policy_id: str):
	try:
		result = await _svc(_tenant()).delete_retention_policy(policy_id)
		return _ok(result)
	except KeyError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		_log.error("delete_retention_policy error: %s", exc)
		return _err(str(exc), 400)


@log_bp.route("/retention/apply", methods=["POST"])
async def apply_retention_policies():
	try:
		result = await _svc(_tenant()).apply_retention_policies()
		return _ok(result)
	except Exception as exc:
		_log.error("apply_retention_policies error: %s", exc)
		return _err(str(exc), 500)


# ------------------------------------------------------------------ log level overrides

@log_bp.route("/levels", methods=["GET"])
async def list_level_overrides():
	try:
		service_name = request.args.get("service_name")
		active_only = request.args.get("active_only", "true").lower() == "true"
		result = await _svc(_tenant()).list_level_overrides(service_name=service_name, active_only=active_only)
		return _ok(result)
	except Exception as exc:
		_log.error("list_level_overrides error: %s", exc)
		return _err(str(exc), 400)


@log_bp.route("/levels", methods=["POST"])
async def create_level_override():
	try:
		body = request.get_json(force=True)
		result = await _svc(_tenant()).create_level_override(
			service_name=body["service_name"],
			level=body["level"],
			logger_name=body.get("logger_name"),
			duration_minutes=body.get("duration_minutes"),
			reason=body.get("reason", ""),
		)
		return _ok(result, 201)
	except KeyError as exc:
		return _err(f"Missing field: {exc}", 400)
	except ValueError as exc:
		return _err(str(exc), 400)
	except Exception as exc:
		_log.error("create_level_override error: %s", exc)
		return _err(str(exc), 400)


@log_bp.route("/levels/<override_id>", methods=["GET"])
async def get_level_override(override_id: str):
	try:
		result = await _svc(_tenant()).get_level_override(override_id)
		return _ok(result)
	except KeyError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		_log.error("get_level_override error: %s", exc)
		return _err(str(exc), 400)


@log_bp.route("/levels/<override_id>", methods=["PUT"])
async def update_level_override(override_id: str):
	try:
		body = request.get_json(force=True) or {}
		result = await _svc(_tenant()).update_level_override(
			override_id=override_id,
			level=body.get("level"),
			active=body.get("active"),
		)
		return _ok(result)
	except KeyError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		_log.error("update_level_override error: %s", exc)
		return _err(str(exc), 400)


@log_bp.route("/levels/<override_id>", methods=["DELETE"])
async def delete_level_override(override_id: str):
	try:
		result = await _svc(_tenant()).delete_level_override(override_id)
		return _ok(result)
	except KeyError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		_log.error("delete_level_override error: %s", exc)
		return _err(str(exc), 400)


@log_bp.route("/levels/effective", methods=["GET"])
async def get_effective_log_level():
	try:
		service_name = request.args["service_name"]
		logger_name = request.args.get("logger_name")
		result = await _svc(_tenant()).get_effective_log_level(service_name, logger_name)
		return _ok(result)
	except KeyError as exc:
		return _err(f"Missing field: {exc}", 400)
	except Exception as exc:
		_log.error("get_effective_log_level error: %s", exc)
		return _err(str(exc), 400)


# ------------------------------------------------------------------ Loki configs

@log_bp.route("/loki", methods=["GET"])
async def list_loki_configs():
	try:
		enabled_only = request.args.get("enabled_only", "false").lower() == "true"
		result = await _svc(_tenant()).list_loki_configs(enabled_only=enabled_only)
		return _ok(result)
	except Exception as exc:
		_log.error("list_loki_configs error: %s", exc)
		return _err(str(exc), 400)


@log_bp.route("/loki", methods=["POST"])
async def create_loki_config():
	try:
		body = request.get_json(force=True)
		result = await _svc(_tenant()).create_loki_config(
			name=body["name"],
			endpoint=body["endpoint"],
			tenant_header=body.get("tenant_header"),
			extra_labels=body.get("extra_labels"),
			batch_size=int(body.get("batch_size", 1000)),
			flush_interval_ms=int(body.get("flush_interval_ms", 1000)),
			max_retries=int(body.get("max_retries", 3)),
			enabled=bool(body.get("enabled", True)),
		)
		return _ok(result, 201)
	except KeyError as exc:
		return _err(f"Missing field: {exc}", 400)
	except Exception as exc:
		_log.error("create_loki_config error: %s", exc)
		return _err(str(exc), 400)


@log_bp.route("/loki/<config_id>", methods=["GET"])
async def get_loki_config(config_id: str):
	try:
		result = await _svc(_tenant()).get_loki_config(config_id)
		return _ok(result)
	except KeyError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		_log.error("get_loki_config error: %s", exc)
		return _err(str(exc), 400)


@log_bp.route("/loki/<config_id>", methods=["PUT"])
async def update_loki_config(config_id: str):
	try:
		body = request.get_json(force=True) or {}
		result = await _svc(_tenant()).update_loki_config(
			config_id=config_id,
			enabled=body.get("enabled"),
			batch_size=body.get("batch_size"),
			flush_interval_ms=body.get("flush_interval_ms"),
			extra_labels=body.get("extra_labels"),
		)
		return _ok(result)
	except KeyError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		_log.error("update_loki_config error: %s", exc)
		return _err(str(exc), 400)


@log_bp.route("/loki/<config_id>", methods=["DELETE"])
async def delete_loki_config(config_id: str):
	try:
		result = await _svc(_tenant()).delete_loki_config(config_id)
		return _ok(result)
	except KeyError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		_log.error("delete_loki_config error: %s", exc)
		return _err(str(exc), 400)


@log_bp.route("/loki/export", methods=["GET"])
async def render_loki_payload():
	try:
		service_name = request.args.get("service_name")
		limit = int(request.args.get("limit", 1000))
		result = await _svc(_tenant()).render_loki_push_payload(service_name=service_name, limit=limit)
		return _ok(result)
	except Exception as exc:
		_log.error("render_loki_payload error: %s", exc)
		return _err(str(exc), 500)


# ------------------------------------------------------------------ analytics

@log_bp.route("/stats", methods=["GET"])
async def get_log_statistics():
	try:
		service_name = request.args.get("service_name")
		result = await _svc(_tenant()).get_log_statistics(service_name=service_name)
		return _ok(result)
	except Exception as exc:
		_log.error("get_log_statistics error: %s", exc)
		return _err(str(exc), 500)


@log_bp.route("/errors", methods=["GET"])
async def get_error_summary():
	try:
		service_name = request.args.get("service_name")
		window_minutes = int(request.args.get("window_minutes", 60))
		result = await _svc(_tenant()).get_error_summary(service_name=service_name, window_minutes=window_minutes)
		return _ok(result)
	except Exception as exc:
		_log.error("get_error_summary error: %s", exc)
		return _err(str(exc), 500)


# ------------------------------------------------------------------ audit

@log_bp.route("/audit", methods=["GET"])
async def get_audit_events():
	try:
		limit = int(request.args.get("limit", 100))
		event_type = request.args.get("event_type")
		result = await _svc(_tenant()).get_audit_events(limit=limit, event_type=event_type)
		return _ok(result)
	except Exception as exc:
		_log.error("get_audit_events error: %s", exc)
		return _err(str(exc), 500)
