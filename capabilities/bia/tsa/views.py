"""Flask Blueprint views for APG Time Series Analytics (bia_tsa)."""

from __future__ import annotations
import asyncio
from functools import wraps
from flask import Blueprint, abort, g, jsonify, request

try:
	from .capability_contract import CAPABILITY_ID
	from .service import TimeSeriesService
	from .context import get_current_user_id, get_tenant_id_from_request
except ImportError:
	from capability_contract import CAPABILITY_ID
	from service import TimeSeriesService
	from context import get_current_user_id, get_tenant_id_from_request

tsa_bp = Blueprint("bia_tsa", __name__, url_prefix="/bia/tsa")
_svc = TimeSeriesService()

def _require_permission(perm: str):
	def decorator(fn):
		@wraps(fn)
		def wrapper(*args, **kwargs):
			g.tenant_id = get_tenant_id_from_request()
			g.user_id = get_current_user_id() or "anonymous"
			if perm not in request.headers.get("X-Permissions","") and "bia_tsa:admin" not in request.headers.get("X-Permissions",""):
				abort(403)
			return fn(*args, **kwargs)
		return wrapper
	return decorator

def _run(coro): return asyncio.get_event_loop().run_until_complete(coro)

@tsa_bp.get("/dashboard")
@_require_permission("bia_tsa:view")
def dashboard():
	return jsonify({"view": "time_series_dashboard", "stats": _run(_svc.get_stats(g.tenant_id))})

@tsa_bp.get("/streams")
@_require_permission("bia_tsa:streams")
def streams():
	return jsonify({"view": "stream_manager", "streams": _run(_svc.list_streams(g.tenant_id))})

@tsa_bp.get("/streams/<stream_id>")
@_require_permission("bia_tsa:streams")
def stream_detail(stream_id: str):
	s = _run(_svc.get_stream(g.tenant_id, stream_id))
	if not s: abort(404)
	return jsonify({"view": "stream_detail", "stream": s, "anomaly_configs": _run(_svc.list_anomaly_configs(g.tenant_id, stream_id)), "forecasts": _run(_svc.list_forecasts(g.tenant_id, stream_id))})

@tsa_bp.get("/streams/<stream_id>/explore")
@_require_permission("bia_tsa:streams")
def stream_explorer(stream_id: str):
	s = _run(_svc.get_stream(g.tenant_id, stream_id))
	if not s: abort(404)
	decomps = _run(_svc.list_decompositions(g.tenant_id, stream_id))
	windows = _run(_svc.list_windows(g.tenant_id, stream_id))
	return jsonify({"view": "stream_explorer", "stream": s, "decompositions": decomps, "windows": windows})

@tsa_bp.get("/anomalies")
@_require_permission("bia_tsa:anomalies")
def anomaly_detection():
	return jsonify({"view": "anomaly_detection_console", "configs": _run(_svc.list_anomaly_configs(g.tenant_id)), "events": _run(_svc.list_anomaly_events(g.tenant_id))})

@tsa_bp.get("/anomalies/<event_id>")
@_require_permission("bia_tsa:anomalies")
def anomaly_detail(event_id: str):
	events = _run(_svc.list_anomaly_events(g.tenant_id))
	ev = next((e for e in events if e["id"] == event_id), None)
	if not ev: abort(404)
	return jsonify({"view": "anomaly_detail", "event": ev})

@tsa_bp.get("/decomposition")
@_require_permission("bia_tsa:decompose")
def decomposition():
	return jsonify({"view": "decomposition_analyser", "decompositions": _run(_svc.list_decompositions(g.tenant_id))})

@tsa_bp.get("/forecasts")
@_require_permission("bia_tsa:forecast")
def forecasts():
	return jsonify({"view": "forecast_manager", "forecasts": _run(_svc.list_forecasts(g.tenant_id))})

@tsa_bp.get("/forecasts/<forecast_id>")
@_require_permission("bia_tsa:forecast")
def forecast_detail(forecast_id: str):
	f = _run(_svc.get_forecast(g.tenant_id, forecast_id))
	if not f: abort(404)
	return jsonify({"view": "forecast_detail", "forecast": f})

@tsa_bp.get("/windows")
@_require_permission("bia_tsa:streams")
def windows():
	return jsonify({"view": "window_manager", "windows": _run(_svc.list_windows(g.tenant_id))})

@tsa_bp.get("/audit")
@_require_permission("bia_tsa:admin")
def audit_log():
	tenant_id = get_tenant_id_from_request()
	user_id = get_current_user_id()
	return jsonify({"view": "audit_log", "tenant_id": tenant_id, "user_id": user_id, "events": _run(_svc.get_audit_events(tenant_id))})

@tsa_bp.get("/settings")
@_require_permission("bia_tsa:admin")
def settings():
	from .capability_contract import get_capability_contract
	tenant_id = get_tenant_id_from_request()
	return jsonify({"view": "settings", "config": get_capability_contract(tenant_id)["configuration"]})
