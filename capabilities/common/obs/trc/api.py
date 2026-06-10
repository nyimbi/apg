"""Flask Blueprint REST API for Distributed Tracing (obs_trc)."""
from __future__ import annotations

import logging
from typing import Any

from flask import Blueprint, request, jsonify

from .service import DistributedTracingService

_log = logging.getLogger(__name__)

trc_bp = Blueprint("obs_trc", __name__, url_prefix="/api/obs/trc")

# Per-tenant service registry — instantiated lazily
_services: dict[str, DistributedTracingService] = {}


def _svc(tenant_id: str = "default") -> DistributedTracingService:
	if tenant_id not in _services:
		_services[tenant_id] = DistributedTracingService(tenant_id=tenant_id)
	return _services[tenant_id]


def _tenant() -> str:
	return request.headers.get("X-Tenant-ID", "default")


def _ok(data: Any, status: int = 200):
	return jsonify(data), status


def _err(message: str, status: int = 400):
	return jsonify({"error": message}), status


# ------------------------------------------------------------------ health

@trc_bp.route("/health", methods=["GET"])
async def health():
	try:
		result = await _svc(_tenant()).health_check()
		return _ok(result)
	except Exception as exc:
		_log.error("health check error: %s", exc)
		return _err(str(exc), 500)


# ------------------------------------------------------------------ describe

@trc_bp.route("/describe", methods=["GET"])
async def describe():
	try:
		result = await _svc(_tenant()).describe()
		return _ok(result)
	except Exception as exc:
		_log.error("describe error: %s", exc)
		return _err(str(exc), 500)


# ------------------------------------------------------------------ spans

@trc_bp.route("/spans", methods=["GET"])
async def list_spans():
	try:
		svc = _svc(_tenant())
		params = request.args
		result = await svc.list_spans(
			trace_id=params.get("trace_id"),
			service_name=params.get("service_name"),
			operation_name=params.get("operation_name"),
			status=params.get("status"),
			error_only=params.get("error_only", "false").lower() == "true",
			min_duration_ms=float(params["min_duration_ms"]) if "min_duration_ms" in params else None,
			page=int(params.get("page", 1)),
			page_size=int(params.get("page_size", 50)),
		)
		return _ok(result)
	except Exception as exc:
		_log.error("list_spans error: %s", exc)
		return _err(str(exc), 400)


@trc_bp.route("/spans/<span_id>", methods=["GET"])
async def get_span(span_id: str):
	try:
		result = await _svc(_tenant()).get_span(span_id)
		return _ok(result)
	except KeyError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		_log.error("get_span error: %s", exc)
		return _err(str(exc), 400)


@trc_bp.route("/spans", methods=["POST"])
async def create_span():
	try:
		body = request.get_json(force=True)
		result = await _svc(_tenant()).create_span(
			operation_name=body["operation_name"],
			service_name=body["service_name"],
			trace_id=body.get("trace_id"),
			parent_span_id=body.get("parent_span_id"),
			start_time=body.get("start_time"),
			tags=body.get("tags"),
			baggage=body.get("baggage"),
			kind=body.get("kind", "internal"),
			sampled=body.get("sampled"),
		)
		return _ok(result, 201)
	except KeyError as exc:
		return _err(f"Missing field: {exc}", 400)
	except Exception as exc:
		_log.error("create_span error: %s", exc)
		return _err(str(exc), 400)


@trc_bp.route("/spans/<span_id>/finish", methods=["PUT"])
async def finish_span(span_id: str):
	try:
		body = request.get_json(force=True) or {}
		result = await _svc(_tenant()).finish_span(
			span_id=span_id,
			end_time=body.get("end_time"),
			status=body.get("status", "ok"),
			status_message=body.get("status_message"),
			tags=body.get("tags"),
			logs=body.get("logs"),
			error=body.get("error", False),
		)
		return _ok(result)
	except KeyError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		_log.error("finish_span error: %s", exc)
		return _err(str(exc), 400)


@trc_bp.route("/spans/<span_id>", methods=["DELETE"])
async def delete_span(span_id: str):
	try:
		result = await _svc(_tenant()).delete_span(span_id)
		return _ok(result)
	except KeyError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		_log.error("delete_span error: %s", exc)
		return _err(str(exc), 400)


@trc_bp.route("/spans/bulk", methods=["POST"])
async def bulk_ingest_spans():
	try:
		body = request.get_json(force=True)
		spans = body if isinstance(body, list) else body.get("spans", [])
		result = await _svc(_tenant()).bulk_ingest_spans(spans)
		return _ok(result, 207)
	except Exception as exc:
		_log.error("bulk_ingest_spans error: %s", exc)
		return _err(str(exc), 400)


# ------------------------------------------------------------------ traces

@trc_bp.route("/traces", methods=["GET"])
async def list_traces():
	try:
		svc = _svc(_tenant())
		params = request.args
		result = await svc.list_traces(
			service_name=params.get("service_name"),
			operation_name=params.get("operation_name"),
			status=params.get("status"),
			error_only=params.get("error_only", "false").lower() == "true",
			page=int(params.get("page", 1)),
			page_size=int(params.get("page_size", 50)),
		)
		return _ok(result)
	except Exception as exc:
		_log.error("list_traces error: %s", exc)
		return _err(str(exc), 400)


@trc_bp.route("/traces/<trace_id>", methods=["GET"])
async def get_trace(trace_id: str):
	try:
		result = await _svc(_tenant()).get_trace(trace_id)
		return _ok(result)
	except KeyError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		_log.error("get_trace error: %s", exc)
		return _err(str(exc), 400)


@trc_bp.route("/traces/<trace_id>", methods=["DELETE"])
async def delete_trace(trace_id: str):
	try:
		result = await _svc(_tenant()).delete_trace(trace_id)
		return _ok(result)
	except KeyError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		_log.error("delete_trace error: %s", exc)
		return _err(str(exc), 400)


@trc_bp.route("/traces/<trace_id>/otlp", methods=["GET"])
async def export_trace_otlp(trace_id: str):
	try:
		result = await _svc(_tenant()).export_trace_otlp(trace_id)
		return _ok(result)
	except KeyError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		_log.error("export_trace_otlp error: %s", exc)
		return _err(str(exc), 400)


# ------------------------------------------------------------------ service map

@trc_bp.route("/service-map", methods=["GET"])
async def get_service_map():
	try:
		result = await _svc(_tenant()).get_service_map()
		return _ok(result)
	except Exception as exc:
		_log.error("get_service_map error: %s", exc)
		return _err(str(exc), 500)


@trc_bp.route("/services/<service_name>/dependencies", methods=["GET"])
async def get_service_dependencies(service_name: str):
	try:
		result = await _svc(_tenant()).get_service_dependencies(service_name)
		return _ok(result)
	except Exception as exc:
		_log.error("get_service_dependencies error: %s", exc)
		return _err(str(exc), 400)


# ------------------------------------------------------------------ sampling rules

@trc_bp.route("/sampling-rules", methods=["GET"])
async def list_sampling_rules():
	try:
		enabled_only = request.args.get("enabled_only", "false").lower() == "true"
		result = await _svc(_tenant()).list_sampling_rules(enabled_only=enabled_only)
		return _ok(result)
	except Exception as exc:
		_log.error("list_sampling_rules error: %s", exc)
		return _err(str(exc), 400)


@trc_bp.route("/sampling-rules", methods=["POST"])
async def create_sampling_rule():
	try:
		body = request.get_json(force=True)
		result = await _svc(_tenant()).create_sampling_rule(
			name=body["name"],
			sample_rate=float(body.get("sample_rate", 1.0)),
			service_name=body.get("service_name"),
			operation_pattern=body.get("operation_pattern"),
			priority=int(body.get("priority", 100)),
			strategy=body.get("strategy", "probabilistic"),
		)
		return _ok(result, 201)
	except KeyError as exc:
		return _err(f"Missing field: {exc}", 400)
	except Exception as exc:
		_log.error("create_sampling_rule error: %s", exc)
		return _err(str(exc), 400)


@trc_bp.route("/sampling-rules/<rule_id>", methods=["GET"])
async def get_sampling_rule(rule_id: str):
	try:
		result = await _svc(_tenant()).get_sampling_rule(rule_id)
		return _ok(result)
	except KeyError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		_log.error("get_sampling_rule error: %s", exc)
		return _err(str(exc), 400)


@trc_bp.route("/sampling-rules/<rule_id>", methods=["PUT"])
async def update_sampling_rule(rule_id: str):
	try:
		body = request.get_json(force=True) or {}
		result = await _svc(_tenant()).update_sampling_rule(
			rule_id=rule_id,
			sample_rate=body.get("sample_rate"),
			priority=body.get("priority"),
			enabled=body.get("enabled"),
			strategy=body.get("strategy"),
		)
		return _ok(result)
	except KeyError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		_log.error("update_sampling_rule error: %s", exc)
		return _err(str(exc), 400)


@trc_bp.route("/sampling-rules/<rule_id>", methods=["DELETE"])
async def delete_sampling_rule(rule_id: str):
	try:
		result = await _svc(_tenant()).delete_sampling_rule(rule_id)
		return _ok(result)
	except KeyError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		_log.error("delete_sampling_rule error: %s", exc)
		return _err(str(exc), 400)


# ------------------------------------------------------------------ export configs

@trc_bp.route("/export-configs", methods=["GET"])
async def list_export_configs():
	try:
		enabled_only = request.args.get("enabled_only", "false").lower() == "true"
		result = await _svc(_tenant()).list_export_configs(enabled_only=enabled_only)
		return _ok(result)
	except Exception as exc:
		_log.error("list_export_configs error: %s", exc)
		return _err(str(exc), 400)


@trc_bp.route("/export-configs", methods=["POST"])
async def create_export_config():
	try:
		body = request.get_json(force=True)
		result = await _svc(_tenant()).create_export_config(
			name=body["name"],
			exporter_type=body["exporter_type"],
			endpoint=body["endpoint"],
			headers=body.get("headers"),
			batch_size=int(body.get("batch_size", 512)),
			flush_interval_ms=int(body.get("flush_interval_ms", 5000)),
			enabled=bool(body.get("enabled", True)),
		)
		return _ok(result, 201)
	except KeyError as exc:
		return _err(f"Missing field: {exc}", 400)
	except Exception as exc:
		_log.error("create_export_config error: %s", exc)
		return _err(str(exc), 400)


@trc_bp.route("/export-configs/<config_id>", methods=["GET"])
async def get_export_config(config_id: str):
	try:
		result = await _svc(_tenant()).get_export_config(config_id)
		return _ok(result)
	except KeyError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		_log.error("get_export_config error: %s", exc)
		return _err(str(exc), 400)


@trc_bp.route("/export-configs/<config_id>", methods=["PUT"])
async def update_export_config(config_id: str):
	try:
		body = request.get_json(force=True) or {}
		result = await _svc(_tenant()).update_export_config(
			config_id=config_id,
			enabled=body.get("enabled"),
			batch_size=body.get("batch_size"),
			flush_interval_ms=body.get("flush_interval_ms"),
			headers=body.get("headers"),
		)
		return _ok(result)
	except KeyError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		_log.error("update_export_config error: %s", exc)
		return _err(str(exc), 400)


@trc_bp.route("/export-configs/<config_id>", methods=["DELETE"])
async def delete_export_config(config_id: str):
	try:
		result = await _svc(_tenant()).delete_export_config(config_id)
		return _ok(result)
	except KeyError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		_log.error("delete_export_config error: %s", exc)
		return _err(str(exc), 400)


@trc_bp.route("/export-configs/<config_id>/test", methods=["POST"])
async def test_export_config(config_id: str):
	try:
		result = await _svc(_tenant()).test_export_config(config_id)
		return _ok(result)
	except KeyError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		_log.error("test_export_config error: %s", exc)
		return _err(str(exc), 400)


# ------------------------------------------------------------------ analytics

@trc_bp.route("/analytics/statistics", methods=["GET"])
async def trace_statistics():
	try:
		service_name = request.args.get("service_name")
		window_minutes = int(request.args.get("window_minutes", 60))
		result = await _svc(_tenant()).get_trace_statistics(service_name=service_name, window_minutes=window_minutes)
		return _ok(result)
	except Exception as exc:
		_log.error("trace_statistics error: %s", exc)
		return _err(str(exc), 400)


@trc_bp.route("/analytics/slow-spans", methods=["GET"])
async def slow_spans():
	try:
		threshold_ms = float(request.args.get("threshold_ms", 1000))
		limit = int(request.args.get("limit", 20))
		result = await _svc(_tenant()).find_slow_spans(threshold_ms=threshold_ms, limit=limit)
		return _ok(result)
	except Exception as exc:
		_log.error("slow_spans error: %s", exc)
		return _err(str(exc), 400)


@trc_bp.route("/audit", methods=["GET"])
async def get_audit_events():
	try:
		limit = int(request.args.get("limit", 100))
		event_type = request.args.get("event_type")
		result = await _svc(_tenant()).get_audit_events(limit=limit, event_type=event_type)
		return _ok(result)
	except Exception as exc:
		_log.error("get_audit_events error: %s", exc)
		return _err(str(exc), 500)
