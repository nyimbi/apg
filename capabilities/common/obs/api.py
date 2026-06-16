# Author: Nyimbi Odero
# Company: Datacraft
# Copyright: © 2025
"""Flask Blueprint REST API for Observability (obs) — top-level umbrella.

URL prefix: /api/common/obs

Endpoints:
  POST  /spans          — ingest a trace span
  POST  /metrics        — ingest a metric data point
  POST  /logs           — ingest a structured log entry
  GET   /health/<svc>   — composite health status for a service
  GET   /slo/<id>       — SLO compliance status
  POST  /slo            — create an SLO
  GET   /slo            — list SLOs
  POST  /alerts         — create an alert rule
  GET   /alerts/fire    — evaluate and return currently firing alerts
  GET   /describe       — capability metadata
"""
from __future__ import annotations

import logging
from typing import Any

from flask import Blueprint, jsonify, request

from .models import AlertRule, LogEntry, Metric, SLOConfig, TraceSpan
from .service import ObservabilityService

_log = logging.getLogger(__name__)

bp = Blueprint("obs", __name__, url_prefix="/api/common/obs")

# Singleton service — multi-tenant state is keyed internally by tenant_id
_svc = ObservabilityService()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tenant() -> str:
	return request.headers.get("X-Tenant-ID", "default")


def _ok(data: Any, status: int = 200):
	return jsonify(data), status


def _err(message: str, status: int = 400):
	return jsonify({"error": message}), status


# ---------------------------------------------------------------------------
# Spans
# ---------------------------------------------------------------------------

@bp.route("/spans", methods=["POST"])
async def post_span():
	"""Ingest a trace span.

	Body (JSON): TraceSpan fields (operation_name, service_name, trace_id required).
	Returns: {"span_id": "<id>"}
	"""
	try:
		body = request.get_json(force=True) or {}
		span = TraceSpan(**body)
		span_id = await _svc.record_span(span, tenant_id=_tenant())
		return _ok({"span_id": span_id}, 201)
	except (TypeError, ValueError) as exc:
		return _err(str(exc), 400)
	except Exception as exc:
		_log.error("post_span error: %s", exc)
		return _err(str(exc), 500)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@bp.route("/metrics", methods=["POST"])
async def post_metric():
	"""Ingest a metric data point.

	Body (JSON): Metric fields (name, value, service_name required).
	Returns: 204 No Content on success.
	"""
	try:
		body = request.get_json(force=True) or {}
		metric = Metric(**body)
		await _svc.record_metric(metric, tenant_id=_tenant())
		return "", 204
	except (TypeError, ValueError) as exc:
		return _err(str(exc), 400)
	except Exception as exc:
		_log.error("post_metric error: %s", exc)
		return _err(str(exc), 500)


# ---------------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------------

@bp.route("/logs", methods=["POST"])
async def post_log():
	"""Ingest a structured log entry.

	Body (JSON): LogEntry fields (message, service_name required).
	Returns: 204 No Content on success.
	"""
	try:
		body = request.get_json(force=True) or {}
		entry = LogEntry(**body)
		await _svc.log_event(entry, tenant_id=_tenant())
		return "", 204
	except (TypeError, ValueError) as exc:
		return _err(str(exc), 400)
	except Exception as exc:
		_log.error("post_log error: %s", exc)
		return _err(str(exc), 500)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@bp.route("/health/<service_name>", methods=["GET"])
async def get_health(service_name: str):
	"""Return composite HealthStatus for service_name.

	Headers: X-Tenant-ID (optional, default "default")
	"""
	try:
		status = await _svc.get_health_status(service_name, tenant_id=_tenant())
		return _ok(status.model_dump(mode="json"))
	except ValueError as exc:
		return _err(str(exc), 400)
	except Exception as exc:
		_log.error("get_health error: %s", exc)
		return _err(str(exc), 500)


# ---------------------------------------------------------------------------
# SLOs
# ---------------------------------------------------------------------------

@bp.route("/slo", methods=["POST"])
async def create_slo():
	"""Register a new SLO.

	Body (JSON): SLOConfig fields (name, service_name, target required).
	Returns: stored SLO record.
	"""
	try:
		body = request.get_json(force=True) or {}
		config = SLOConfig(**body)
		result = await _svc.create_slo(config, tenant_id=_tenant())
		return _ok(result, 201)
	except (TypeError, ValueError) as exc:
		return _err(str(exc), 400)
	except Exception as exc:
		_log.error("create_slo error: %s", exc)
		return _err(str(exc), 500)


@bp.route("/slo", methods=["GET"])
async def list_slos():
	"""List all SLOs for the tenant."""
	try:
		result = await _svc.list_slos(tenant_id=_tenant())
		return _ok({"items": result, "total": len(result)})
	except Exception as exc:
		_log.error("list_slos error: %s", exc)
		return _err(str(exc), 500)


@bp.route("/slo/<slo_id>", methods=["GET"])
async def get_slo_status(slo_id: str):
	"""Return SLO compliance status for slo_id."""
	try:
		result = await _svc.get_slo_status(slo_id, tenant_id=_tenant())
		return _ok(result)
	except KeyError as exc:
		return _err(str(exc), 404)
	except ValueError as exc:
		return _err(str(exc), 400)
	except Exception as exc:
		_log.error("get_slo_status error: %s", exc)
		return _err(str(exc), 500)


# ---------------------------------------------------------------------------
# Alert rules
# ---------------------------------------------------------------------------

@bp.route("/alerts", methods=["POST"])
async def create_alert():
	"""Register an alert rule.

	Body (JSON): AlertRule fields (name, service_name, condition, threshold required).
	Returns: stored alert rule record.
	"""
	try:
		body = request.get_json(force=True) or {}
		rule = AlertRule(**body)
		result = await _svc.create_alert_rule(rule, tenant_id=_tenant())
		return _ok(result, 201)
	except (TypeError, ValueError) as exc:
		return _err(str(exc), 400)
	except Exception as exc:
		_log.error("create_alert error: %s", exc)
		return _err(str(exc), 500)


@bp.route("/alerts/fire", methods=["GET"])
async def firing_alerts():
	"""Evaluate all alert rules and return those currently firing."""
	try:
		result = await _svc.evaluate_alert_rules(tenant_id=_tenant())
		return _ok({"firing": result, "count": len(result)})
	except Exception as exc:
		_log.error("firing_alerts error: %s", exc)
		return _err(str(exc), 500)


# ---------------------------------------------------------------------------
# Describe / health
# ---------------------------------------------------------------------------

@bp.route("/describe", methods=["GET"])
async def describe():
	"""Return capability metadata."""
	try:
		result = await _svc.describe()
		return _ok(result)
	except Exception as exc:
		_log.error("describe error: %s", exc)
		return _err(str(exc), 500)


@bp.route("/health", methods=["GET"])
async def capability_health():
	"""Return capability-level health (all subcapabilities)."""
	try:
		result = await _svc.health_check(tenant_id=_tenant())
		return _ok(result)
	except Exception as exc:
		_log.error("capability_health error: %s", exc)
		return _err(str(exc), 500)
