"""Flask Blueprint REST API for APG Predictive Analytics (bia_pda)."""

from __future__ import annotations
import asyncio
from typing import Any
from flask import Blueprint, jsonify, request

try:
	from .service import PredictiveAnalyticsService
	from .capability_contract import get_capability_contract
except ImportError:
	from service import PredictiveAnalyticsService
	from capability_contract import get_capability_contract

api_bp = Blueprint("bia_pda_api", __name__, url_prefix="/api/bia/pda")
_svc = PredictiveAnalyticsService()

def _run(coro): return asyncio.run(coro)
def _tenant(): return request.headers.get("X-Tenant-ID", "default")
def _user(): return request.headers.get("X-User-ID", "anonymous")
def _ok(data: Any, status: int=200): return jsonify({"status": "ok", "data": data}), status
def _err(msg: str, status: int=400): return jsonify({"status": "error", "message": msg}), status

@api_bp.get("/contract")
def get_contract(): return _ok(get_capability_contract(_tenant()))

@api_bp.get("/models")
def list_models(): return _ok(_run(_svc.list_models(_tenant())))

@api_bp.post("/models")
def train_model():
	b = request.get_json(silent=True) or {}
	missing = [f for f in ["name","model_type","training_datasource_id"] if f not in b]
	if missing: return _err(f"Missing: {missing}", 400)
	try: m = _run(_svc.train_model(_tenant(), b["name"], b["model_type"], b.get("owner_id",_user()), b["training_datasource_id"], b.get("feature_ids",[]), b.get("target_column"), b.get("description"), b.get("tags",[])))
	except ValueError as e: return _err(str(e), 400)
	return _ok(m, 201)

@api_bp.get("/models/<model_id>")
def get_model(model_id: str):
	m = _run(_svc.get_model(_tenant(), model_id))
	return _ok(m) if m else _err("Not found", 404)

@api_bp.post("/models/<model_id>/deploy")
def deploy_model(model_id: str):
	try: m = _run(_svc.deploy_model(_tenant(), model_id))
	except ValueError as e: return _err(str(e), 400)
	return _ok(m)

@api_bp.post("/models/<model_id>/deprecate")
def deprecate_model(model_id: str):
	try: m = _run(_svc.deprecate_model(_tenant(), model_id))
	except ValueError as e: return _err(str(e), 404)
	return _ok(m)

@api_bp.delete("/models/<model_id>")
def delete_model(model_id: str):
	ok = _run(_svc.delete_model(_tenant(), model_id))
	return _ok({"deleted": model_id}) if ok else _err("Not found", 404)

@api_bp.get("/forecasts")
def list_forecasts(): return _ok(_run(_svc.list_forecasts(_tenant(), request.args.get("model_id"))))

@api_bp.post("/forecasts")
def generate_forecast():
	b = request.get_json(silent=True) or {}
	missing = [f for f in ["model_id","horizon"] if f not in b]
	if missing: return _err(f"Missing: {missing}", 400)
	try: f = _run(_svc.generate_forecast(_tenant(), b["model_id"], b["horizon"], b.get("owner_id",_user()), b.get("output_type","point_forecast"), b.get("confidence_interval",0.95), b.get("parameters",{})))
	except ValueError as e: return _err(str(e), 400)
	return _ok(f, 201)

@api_bp.get("/forecasts/<forecast_id>")
def get_forecast(forecast_id: str):
	f = _run(_svc.get_forecast(_tenant(), forecast_id))
	return _ok(f) if f else _err("Not found", 404)

@api_bp.get("/scenarios")
def list_scenarios(): return _ok(_run(_svc.list_scenarios(_tenant(), request.args.get("model_id"))))

@api_bp.post("/scenarios")
def simulate_scenario():
	b = request.get_json(silent=True) or {}
	missing = [f for f in ["model_id","name","scenario_type","parameters"] if f not in b]
	if missing: return _err(f"Missing: {missing}", 400)
	try: sc = _run(_svc.simulate_scenario(_tenant(), b["model_id"], b["name"], b["scenario_type"], b["parameters"], b.get("owner_id",_user()), b.get("description")))
	except ValueError as e: return _err(str(e), 400)
	return _ok(sc, 201)

@api_bp.get("/scenarios/<scenario_id>")
def get_scenario(scenario_id: str):
	s = _run(_svc.get_scenario(_tenant(), scenario_id))
	return _ok(s) if s else _err("Not found", 404)

@api_bp.delete("/scenarios/<scenario_id>")
def delete_scenario(scenario_id: str):
	ok = _run(_svc.delete_scenario(_tenant(), scenario_id))
	return _ok({"deleted": scenario_id}) if ok else _err("Not found", 404)

@api_bp.get("/features")
def list_features(): return _ok(_run(_svc.list_features(_tenant())))

@api_bp.post("/features")
def register_feature():
	b = request.get_json(silent=True) or {}
	missing = [f for f in ["name","feature_type","source_column","datasource_id"] if f not in b]
	if missing: return _err(f"Missing: {missing}", 400)
	try: feat = _run(_svc.register_feature(_tenant(), b["name"], b["feature_type"], b["source_column"], b["datasource_id"], b.get("owner_id",_user()), b.get("description")))
	except ValueError as e: return _err(str(e), 400)
	return _ok(feat, 201)

@api_bp.post("/predict")
def serve_prediction():
	b = request.get_json(silent=True) or {}
	missing = [f for f in ["model_id","input_data"] if f not in b]
	if missing: return _err(f"Missing: {missing}", 400)
	try: pred = _run(_svc.serve_prediction(_tenant(), b["model_id"], b["input_data"]))
	except ValueError as e: return _err(str(e), 400)
	return _ok(pred, 201)

@api_bp.get("/predictions")
def list_predictions(): return _ok(_run(_svc.list_predictions(_tenant())))

@api_bp.get("/stats")
def get_stats(): return _ok(_run(_svc.get_stats(_tenant())))

@api_bp.get("/audit")
def get_audit(): return _ok(_run(_svc.get_audit_events(_tenant())))
