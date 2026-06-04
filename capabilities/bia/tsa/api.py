"""Flask Blueprint REST API for APG Time Series Analytics (bia_tsa)."""

from __future__ import annotations
import asyncio
from typing import Any
from flask import Blueprint, jsonify, request

try:
	from .service import TimeSeriesService
	from .capability_contract import get_capability_contract
except ImportError:
	from service import TimeSeriesService
	from capability_contract import get_capability_contract

api_bp = Blueprint("bia_tsa_api", __name__, url_prefix="/api/bia/tsa")
_svc = TimeSeriesService()

def _run(coro): return asyncio.run(coro)
def _tenant(): return request.headers.get("X-Tenant-ID", "default")
def _user(): return request.headers.get("X-User-ID", "anonymous")
def _ok(data: Any, status: int=200): return jsonify({"status": "ok", "data": data}), status
def _err(msg: str, status: int=400): return jsonify({"status": "error", "message": msg}), status

@api_bp.get("/contract")
def get_contract(): return _ok(get_capability_contract(_tenant()))

@api_bp.get("/streams")
def list_streams(): return _ok(_run(_svc.list_streams(_tenant())))

@api_bp.post("/streams")
def register_stream():
	b = request.get_json(silent=True) or {}
	missing = [f for f in ["name","protocol","frequency","source_identifier"] if f not in b]
	if missing: return _err(f"Missing: {missing}", 400)
	try: s = _run(_svc.register_stream(_tenant(), b["name"], b["protocol"], b["frequency"], b.get("owner_id",_user()), b["source_identifier"], b.get("data_type","numeric"), b.get("unit_of_measure"), b.get("description"), b.get("tags",[])))
	except ValueError as e: return _err(str(e), 400)
	return _ok(s, 201)

@api_bp.get("/streams/<stream_id>")
def get_stream(stream_id: str):
	s = _run(_svc.get_stream(_tenant(), stream_id))
	return _ok(s) if s else _err("Not found", 404)

@api_bp.post("/streams/<stream_id>/pause")
def pause_stream(stream_id: str):
	try: s = _run(_svc.pause_stream(_tenant(), stream_id))
	except ValueError as e: return _err(str(e), 404)
	return _ok(s)

@api_bp.post("/streams/<stream_id>/resume")
def resume_stream(stream_id: str):
	try: s = _run(_svc.resume_stream(_tenant(), stream_id))
	except ValueError as e: return _err(str(e), 404)
	return _ok(s)

@api_bp.post("/streams/<stream_id>/archive")
def archive_stream(stream_id: str):
	try: s = _run(_svc.archive_stream(_tenant(), stream_id))
	except ValueError as e: return _err(str(e), 404)
	return _ok(s)

@api_bp.post("/streams/<stream_id>/ingest")
def ingest_data(stream_id: str):
	b = request.get_json(silent=True) or {}
	if "data_points" not in b: return _err("Missing: data_points", 400)
	try: r = _run(_svc.ingest_data(_tenant(), stream_id, b["data_points"]))
	except ValueError as e: return _err(str(e), 400)
	return _ok(r)

@api_bp.get("/anomaly-configs")
def list_anomaly_configs(): return _ok(_run(_svc.list_anomaly_configs(_tenant(), request.args.get("stream_id"))))

@api_bp.post("/anomaly-configs")
def configure_anomaly():
	b = request.get_json(silent=True) or {}
	missing = [f for f in ["stream_id","name","method"] if f not in b]
	if missing: return _err(f"Missing: {missing}", 400)
	try: ac = _run(_svc.configure_anomaly_detection(_tenant(), b["stream_id"], b["name"], b["method"], b.get("owner_id",_user()), b.get("sensitivity",0.95), b.get("config",{})))
	except ValueError as e: return _err(str(e), 400)
	return _ok(ac, 201)

@api_bp.get("/anomaly-events")
def list_anomaly_events(): return _ok(_run(_svc.list_anomaly_events(_tenant(), request.args.get("stream_id"))))

@api_bp.get("/decompositions")
def list_decompositions(): return _ok(_run(_svc.list_decompositions(_tenant(), request.args.get("stream_id"))))

@api_bp.post("/decompositions")
def run_decomposition():
	b = request.get_json(silent=True) or {}
	if "stream_id" not in b: return _err("Missing: stream_id", 400)
	try: r = _run(_svc.run_decomposition(_tenant(), b["stream_id"], b.get("components",["trend","seasonality","residual"]), b.get("model_type","additive")))
	except ValueError as e: return _err(str(e), 400)
	return _ok(r, 201)

@api_bp.get("/forecasts")
def list_forecasts(): return _ok(_run(_svc.list_forecasts(_tenant(), request.args.get("stream_id"))))

@api_bp.post("/forecasts")
def create_forecast():
	b = request.get_json(silent=True) or {}
	missing = [f for f in ["stream_id","model","horizon_periods"] if f not in b]
	if missing: return _err(f"Missing: {missing}", 400)
	try: f = _run(_svc.create_forecast(_tenant(), b["stream_id"], b["model"], b["horizon_periods"], b.get("owner_id",_user()), b.get("confidence_interval",0.95)))
	except ValueError as e: return _err(str(e), 400)
	return _ok(f, 201)

@api_bp.get("/forecasts/<forecast_id>")
def get_forecast(forecast_id: str):
	f = _run(_svc.get_forecast(_tenant(), forecast_id))
	return _ok(f) if f else _err("Not found", 404)

@api_bp.get("/windows")
def list_windows(): return _ok(_run(_svc.list_windows(_tenant(), request.args.get("stream_id"))))

@api_bp.post("/windows")
def create_window():
	b = request.get_json(silent=True) or {}
	missing = [f for f in ["stream_id","name","window_type","size_seconds","aggregation_function"] if f not in b]
	if missing: return _err(f"Missing: {missing}", 400)
	try: w = _run(_svc.create_window(_tenant(), b["stream_id"], b["name"], b["window_type"], b["size_seconds"], b["aggregation_function"], b.get("owner_id",_user())))
	except ValueError as e: return _err(str(e), 400)
	return _ok(w, 201)

@api_bp.delete("/windows/<window_id>")
def delete_window(window_id: str):
	ok = _run(_svc.delete_window(_tenant(), window_id))
	return _ok({"deleted": window_id}) if ok else _err("Not found", 404)

@api_bp.post("/streams/<stream_id>/fill-gaps")
def fill_gaps(stream_id: str):
	b = request.get_json(silent=True) or {}
	try: r = _run(_svc.fill_gaps(_tenant(), stream_id, b.get("method","forward_fill")))
	except ValueError as e: return _err(str(e), 400)
	return _ok(r)

@api_bp.get("/stats")
def get_stats(): return _ok(_run(_svc.get_stats(_tenant())))

@api_bp.get("/audit")
def get_audit(): return _ok(_run(_svc.get_audit_events(_tenant())))
