"""Flask Blueprint REST API for Metrics & SLO (obs_mtx)."""
from __future__ import annotations

import logging
from typing import Any

from flask import Blueprint, request, jsonify, Response

from .service import MetricsSLOService

_log = logging.getLogger(__name__)

mtx_bp = Blueprint("obs_mtx", __name__, url_prefix="/api/obs/mtx")

_services: dict[str, MetricsSLOService] = {}


def _svc(tenant_id: str = "default") -> MetricsSLOService:
	if tenant_id not in _services:
		_services[tenant_id] = MetricsSLOService(tenant_id=tenant_id)
	return _services[tenant_id]


def _tenant() -> str:
	return request.headers.get("X-Tenant-ID", "default")


def _ok(data: Any, status: int = 200):
	return jsonify(data), status


def _err(message: str, status: int = 400):
	return jsonify({"error": message}), status


# ------------------------------------------------------------------ health

@mtx_bp.route("/health", methods=["GET"])
async def health():
	try:
		result = await _svc(_tenant()).health_check()
		return _ok(result)
	except Exception as exc:
		_log.error("health check error: %s", exc)
		return _err(str(exc), 500)


@mtx_bp.route("/describe", methods=["GET"])
async def describe():
	try:
		result = await _svc(_tenant()).describe()
		return _ok(result)
	except Exception as exc:
		_log.error("describe error: %s", exc)
		return _err(str(exc), 500)


# ------------------------------------------------------------------ metric definitions

@mtx_bp.route("/metrics", methods=["GET"])
async def list_metric_definitions():
	try:
		params = request.args
		result = await _svc(_tenant()).list_metric_definitions(
			service_name=params.get("service_name"),
			metric_type=params.get("metric_type"),
			enabled_only=params.get("enabled_only", "true").lower() == "true",
			page=int(params.get("page", 1)),
			page_size=int(params.get("page_size", 50)),
		)
		return _ok(result)
	except Exception as exc:
		_log.error("list_metric_definitions error: %s", exc)
		return _err(str(exc), 400)


@mtx_bp.route("/metrics", methods=["POST"])
async def create_metric_definition():
	try:
		body = request.get_json(force=True)
		result = await _svc(_tenant()).create_metric_definition(
			name=body["name"],
			service_name=body["service_name"],
			metric_type=body["metric_type"],
			description=body.get("description", ""),
			unit=body.get("unit", ""),
			labels=body.get("labels"),
			namespace=body.get("namespace", "apg"),
		)
		return _ok(result, 201)
	except KeyError as exc:
		return _err(f"Missing field: {exc}", 400)
	except ValueError as exc:
		return _err(str(exc), 400)
	except Exception as exc:
		_log.error("create_metric_definition error: %s", exc)
		return _err(str(exc), 400)


@mtx_bp.route("/metrics/<def_id>", methods=["GET"])
async def get_metric_definition(def_id: str):
	try:
		result = await _svc(_tenant()).get_metric_definition(def_id)
		return _ok(result)
	except KeyError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		_log.error("get_metric_definition error: %s", exc)
		return _err(str(exc), 400)


@mtx_bp.route("/metrics/<def_id>", methods=["PUT"])
async def update_metric_definition(def_id: str):
	try:
		body = request.get_json(force=True) or {}
		result = await _svc(_tenant()).update_metric_definition(
			def_id=def_id,
			description=body.get("description"),
			labels=body.get("labels"),
			enabled=body.get("enabled"),
		)
		return _ok(result)
	except KeyError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		_log.error("update_metric_definition error: %s", exc)
		return _err(str(exc), 400)


@mtx_bp.route("/metrics/<def_id>", methods=["DELETE"])
async def delete_metric_definition(def_id: str):
	try:
		result = await _svc(_tenant()).delete_metric_definition(def_id)
		return _ok(result)
	except KeyError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		_log.error("delete_metric_definition error: %s", exc)
		return _err(str(exc), 400)


# ------------------------------------------------------------------ data points

@mtx_bp.route("/data-points", methods=["POST"])
async def record_metric():
	try:
		body = request.get_json(force=True)
		result = await _svc(_tenant()).record_metric(
			metric_name=body["metric_name"],
			value=float(body["value"]),
			service_name=body["service_name"],
			labels=body.get("labels"),
			timestamp=body.get("timestamp"),
		)
		return _ok(result, 201)
	except KeyError as exc:
		return _err(f"Missing field: {exc}", 400)
	except Exception as exc:
		_log.error("record_metric error: %s", exc)
		return _err(str(exc), 400)


@mtx_bp.route("/data-points/bulk", methods=["POST"])
async def bulk_record_metrics():
	try:
		body = request.get_json(force=True)
		points = body if isinstance(body, list) else body.get("points", [])
		result = await _svc(_tenant()).bulk_record_metrics(points)
		return _ok(result, 207)
	except Exception as exc:
		_log.error("bulk_record_metrics error: %s", exc)
		return _err(str(exc), 400)


@mtx_bp.route("/data-points/query", methods=["GET"])
async def query_metric():
	try:
		params = request.args
		result = await _svc(_tenant()).query_metric(
			metric_name=params["metric_name"],
			service_name=params.get("service_name"),
			start_time=params.get("start_time"),
			end_time=params.get("end_time"),
			limit=int(params.get("limit", 1000)),
		)
		return _ok(result)
	except KeyError as exc:
		return _err(f"Missing field: {exc}", 400)
	except Exception as exc:
		_log.error("query_metric error: %s", exc)
		return _err(str(exc), 400)


# ------------------------------------------------------------------ RED metrics

@mtx_bp.route("/red/<service_name>", methods=["GET"])
async def compute_red_metrics(service_name: str):
	try:
		window_minutes = int(request.args.get("window_minutes", 5))
		result = await _svc(_tenant()).compute_red_metrics(service_name, window_minutes)
		return _ok(result)
	except Exception as exc:
		_log.error("compute_red_metrics error: %s", exc)
		return _err(str(exc), 400)


@mtx_bp.route("/red", methods=["GET"])
async def compute_all_red_metrics():
	try:
		window_minutes = int(request.args.get("window_minutes", 5))
		result = await _svc(_tenant()).compute_red_metrics_all_services(window_minutes)
		return _ok(result)
	except Exception as exc:
		_log.error("compute_all_red_metrics error: %s", exc)
		return _err(str(exc), 400)


# ------------------------------------------------------------------ SLOs

@mtx_bp.route("/slos", methods=["GET"])
async def list_slos():
	try:
		params = request.args
		result = await _svc(_tenant()).list_slos(
			service_name=params.get("service_name"),
			slo_type=params.get("slo_type"),
			enabled_only=params.get("enabled_only", "true").lower() == "true",
			page=int(params.get("page", 1)),
			page_size=int(params.get("page_size", 50)),
		)
		return _ok(result)
	except Exception as exc:
		_log.error("list_slos error: %s", exc)
		return _err(str(exc), 400)


@mtx_bp.route("/slos", methods=["POST"])
async def create_slo():
	try:
		body = request.get_json(force=True)
		result = await _svc(_tenant()).create_slo(
			name=body["name"],
			service_name=body["service_name"],
			slo_type=body["slo_type"],
			target=float(body["target"]),
			window_days=int(body.get("window_days", 30)),
			description=body.get("description", ""),
			good_query=body.get("good_query", ""),
			total_query=body.get("total_query", ""),
			latency_threshold_ms=body.get("latency_threshold_ms"),
		)
		return _ok(result, 201)
	except KeyError as exc:
		return _err(f"Missing field: {exc}", 400)
	except ValueError as exc:
		return _err(str(exc), 400)
	except Exception as exc:
		_log.error("create_slo error: %s", exc)
		return _err(str(exc), 400)


@mtx_bp.route("/slos/<slo_id>", methods=["GET"])
async def get_slo(slo_id: str):
	try:
		result = await _svc(_tenant()).get_slo(slo_id)
		return _ok(result)
	except KeyError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		_log.error("get_slo error: %s", exc)
		return _err(str(exc), 400)


@mtx_bp.route("/slos/<slo_id>", methods=["PUT"])
async def update_slo(slo_id: str):
	try:
		body = request.get_json(force=True) or {}
		result = await _svc(_tenant()).update_slo(
			slo_id=slo_id,
			description=body.get("description"),
			target=body.get("target"),
			window_days=body.get("window_days"),
			good_query=body.get("good_query"),
			total_query=body.get("total_query"),
			latency_threshold_ms=body.get("latency_threshold_ms"),
			enabled=body.get("enabled"),
		)
		return _ok(result)
	except KeyError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		_log.error("update_slo error: %s", exc)
		return _err(str(exc), 400)


@mtx_bp.route("/slos/<slo_id>", methods=["DELETE"])
async def delete_slo(slo_id: str):
	try:
		result = await _svc(_tenant()).delete_slo(slo_id)
		return _ok(result)
	except KeyError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		_log.error("delete_slo error: %s", exc)
		return _err(str(exc), 400)


@mtx_bp.route("/slos/<slo_id>/evaluate", methods=["GET"])
async def evaluate_slo(slo_id: str):
	try:
		result = await _svc(_tenant()).evaluate_slo_compliance(slo_id)
		return _ok(result)
	except KeyError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		_log.error("evaluate_slo error: %s", exc)
		return _err(str(exc), 400)


@mtx_bp.route("/slos/evaluate-all", methods=["GET"])
async def evaluate_all_slos():
	try:
		result = await _svc(_tenant()).evaluate_all_slos()
		return _ok(result)
	except Exception as exc:
		_log.error("evaluate_all_slos error: %s", exc)
		return _err(str(exc), 500)


# ------------------------------------------------------------------ burn rate alerts

@mtx_bp.route("/burn-rate-alerts", methods=["GET"])
async def list_burn_rate_alerts():
	try:
		slo_id = request.args.get("slo_id")
		firing_only = request.args.get("firing_only", "false").lower() == "true"
		result = await _svc(_tenant()).list_burn_rate_alerts(slo_id=slo_id, firing_only=firing_only)
		return _ok(result)
	except Exception as exc:
		_log.error("list_burn_rate_alerts error: %s", exc)
		return _err(str(exc), 400)


@mtx_bp.route("/burn-rate-alerts", methods=["POST"])
async def create_burn_rate_alert():
	try:
		body = request.get_json(force=True)
		result = await _svc(_tenant()).create_burn_rate_alert(
			slo_id=body["slo_id"],
			name=body["name"],
			short_window_minutes=int(body.get("short_window_minutes", 60)),
			long_window_minutes=int(body.get("long_window_minutes", 360)),
			burn_rate_threshold=float(body.get("burn_rate_threshold", 14.4)),
			severity=body.get("severity", "critical"),
			notification_channels=body.get("notification_channels"),
		)
		return _ok(result, 201)
	except KeyError as exc:
		return _err(f"Missing field: {exc}", 400)
	except Exception as exc:
		_log.error("create_burn_rate_alert error: %s", exc)
		return _err(str(exc), 400)


@mtx_bp.route("/burn-rate-alerts/<alert_id>", methods=["GET"])
async def get_burn_rate_alert(alert_id: str):
	try:
		result = await _svc(_tenant()).get_burn_rate_alert(alert_id)
		return _ok(result)
	except KeyError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		_log.error("get_burn_rate_alert error: %s", exc)
		return _err(str(exc), 400)


@mtx_bp.route("/burn-rate-alerts/<alert_id>", methods=["PUT"])
async def update_burn_rate_alert(alert_id: str):
	try:
		body = request.get_json(force=True) or {}
		result = await _svc(_tenant()).update_burn_rate_alert(
			alert_id=alert_id,
			enabled=body.get("enabled"),
			burn_rate_threshold=body.get("burn_rate_threshold"),
			severity=body.get("severity"),
			notification_channels=body.get("notification_channels"),
		)
		return _ok(result)
	except KeyError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		_log.error("update_burn_rate_alert error: %s", exc)
		return _err(str(exc), 400)


@mtx_bp.route("/burn-rate-alerts/<alert_id>", methods=["DELETE"])
async def delete_burn_rate_alert(alert_id: str):
	try:
		result = await _svc(_tenant()).delete_burn_rate_alert(alert_id)
		return _ok(result)
	except KeyError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		_log.error("delete_burn_rate_alert error: %s", exc)
		return _err(str(exc), 400)


@mtx_bp.route("/burn-rate-alerts/<alert_id>/evaluate", methods=["GET"])
async def evaluate_burn_rate(alert_id: str):
	try:
		result = await _svc(_tenant()).evaluate_burn_rate(alert_id)
		return _ok(result)
	except KeyError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		_log.error("evaluate_burn_rate error: %s", exc)
		return _err(str(exc), 400)


# ------------------------------------------------------------------ Prometheus

@mtx_bp.route("/prometheus/config", methods=["POST"])
async def configure_prometheus():
	try:
		body = request.get_json(force=True) or {}
		result = await _svc(_tenant()).configure_prometheus_export(
			endpoint=body.get("endpoint", "/metrics"),
			port=int(body.get("port", 9090)),
			scrape_interval_seconds=int(body.get("scrape_interval_seconds", 15)),
			include_namespaces=body.get("include_namespaces"),
			exclude_labels=body.get("exclude_labels"),
		)
		return _ok(result, 201)
	except Exception as exc:
		_log.error("configure_prometheus error: %s", exc)
		return _err(str(exc), 400)


@mtx_bp.route("/prometheus/metrics", methods=["GET"])
async def prometheus_metrics():
	try:
		text = await _svc(_tenant()).render_prometheus_metrics()
		return Response(text, mimetype="text/plain; version=0.0.4; charset=utf-8")
	except Exception as exc:
		_log.error("prometheus_metrics error: %s", exc)
		return _err(str(exc), 500)


# ------------------------------------------------------------------ dashboards

@mtx_bp.route("/dashboards", methods=["GET"])
async def list_dashboards():
	try:
		result = await _svc(_tenant()).list_dashboards(
			service_name=request.args.get("service_name"),
			page=int(request.args.get("page", 1)),
			page_size=int(request.args.get("page_size", 50)),
		)
		return _ok(result)
	except Exception as exc:
		_log.error("list_dashboards error: %s", exc)
		return _err(str(exc), 400)


@mtx_bp.route("/dashboards", methods=["POST"])
async def create_dashboard():
	try:
		body = request.get_json(force=True)
		result = await _svc(_tenant()).create_dashboard(
			name=body["name"],
			description=body.get("description", ""),
			service_name=body.get("service_name"),
			panels=body.get("panels"),
			variables=body.get("variables"),
			refresh_interval_seconds=int(body.get("refresh_interval_seconds", 30)),
			tags=body.get("tags"),
		)
		return _ok(result, 201)
	except KeyError as exc:
		return _err(f"Missing field: {exc}", 400)
	except Exception as exc:
		_log.error("create_dashboard error: %s", exc)
		return _err(str(exc), 400)


@mtx_bp.route("/dashboards/<dash_id>", methods=["GET"])
async def get_dashboard(dash_id: str):
	try:
		result = await _svc(_tenant()).get_dashboard(dash_id)
		return _ok(result)
	except KeyError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		_log.error("get_dashboard error: %s", exc)
		return _err(str(exc), 400)


@mtx_bp.route("/dashboards/<dash_id>", methods=["PUT"])
async def update_dashboard(dash_id: str):
	try:
		body = request.get_json(force=True) or {}
		result = await _svc(_tenant()).update_dashboard(
			dash_id=dash_id,
			name=body.get("name"),
			description=body.get("description"),
			panels=body.get("panels"),
			variables=body.get("variables"),
			refresh_interval_seconds=body.get("refresh_interval_seconds"),
			tags=body.get("tags"),
		)
		return _ok(result)
	except KeyError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		_log.error("update_dashboard error: %s", exc)
		return _err(str(exc), 400)


@mtx_bp.route("/dashboards/<dash_id>", methods=["DELETE"])
async def delete_dashboard(dash_id: str):
	try:
		result = await _svc(_tenant()).delete_dashboard(dash_id)
		return _ok(result)
	except KeyError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		_log.error("delete_dashboard error: %s", exc)
		return _err(str(exc), 400)


@mtx_bp.route("/dashboards/generate/red/<service_name>", methods=["POST"])
async def generate_red_dashboard(service_name: str):
	try:
		result = await _svc(_tenant()).generate_red_dashboard(service_name)
		return _ok(result, 201)
	except Exception as exc:
		_log.error("generate_red_dashboard error: %s", exc)
		return _err(str(exc), 400)


# ------------------------------------------------------------------ audit

@mtx_bp.route("/audit", methods=["GET"])
async def get_audit_events():
	try:
		limit = int(request.args.get("limit", 100))
		event_type = request.args.get("event_type")
		result = await _svc(_tenant()).get_audit_events(limit=limit, event_type=event_type)
		return _ok(result)
	except Exception as exc:
		_log.error("get_audit_events error: %s", exc)
		return _err(str(exc), 500)
