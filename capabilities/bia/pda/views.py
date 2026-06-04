"""Flask Blueprint views for APG Predictive Analytics (bia_pda)."""

from __future__ import annotations
import asyncio
from functools import wraps
from flask import Blueprint, abort, g, jsonify, request

try:
	from .capability_contract import CAPABILITY_ID
	from .service import PredictiveAnalyticsService
except ImportError:
	from capability_contract import CAPABILITY_ID
	from service import PredictiveAnalyticsService

pda_bp = Blueprint("bia_pda", __name__, url_prefix="/bia/pda")
_svc = PredictiveAnalyticsService()

def _require_permission(perm: str):
	def decorator(fn):
		@wraps(fn)
		def wrapper(*args, **kwargs):
			g.tenant_id = request.headers.get("X-Tenant-ID", "default")
			g.user_id = request.headers.get("X-User-ID", "anonymous")
			perms = request.headers.get("X-Permissions", "")
			if perm not in perms and "bia_pda:admin" not in perms:
				abort(403, description=f"Permission required: {perm}")
			return fn(*args, **kwargs)
		return wrapper
	return decorator

def _run(coro): return asyncio.get_event_loop().run_until_complete(coro)

@pda_bp.get("/dashboard")
@_require_permission("bia_pda:view")
def dashboard():
	return jsonify({"view": "predictive_dashboard", "stats": _run(_svc.get_stats(g.tenant_id))})

@pda_bp.get("/models")
@_require_permission("bia_pda:models")
def models():
	return jsonify({"view": "model_library", "models": _run(_svc.list_models(g.tenant_id))})

@pda_bp.get("/models/<model_id>")
@_require_permission("bia_pda:models")
def model_detail(model_id: str):
	m = _run(_svc.get_model(g.tenant_id, model_id))
	if not m: abort(404)
	forecasts = _run(_svc.list_forecasts(g.tenant_id, model_id))
	scenarios = _run(_svc.list_scenarios(g.tenant_id, model_id))
	return jsonify({"view": "model_detail", "model": m, "forecasts": forecasts, "scenarios": scenarios})

@pda_bp.get("/forecasts")
@_require_permission("bia_pda:forecasts")
def forecasts():
	return jsonify({"view": "forecast_explorer", "forecasts": _run(_svc.list_forecasts(g.tenant_id))})

@pda_bp.get("/forecasts/<forecast_id>")
@_require_permission("bia_pda:forecasts")
def forecast_detail(forecast_id: str):
	f = _run(_svc.get_forecast(g.tenant_id, forecast_id))
	if not f: abort(404)
	return jsonify({"view": "forecast_detail", "forecast": f})

@pda_bp.get("/scenarios")
@_require_permission("bia_pda:scenarios")
def scenarios():
	return jsonify({"view": "scenario_builder", "scenarios": _run(_svc.list_scenarios(g.tenant_id))})

@pda_bp.get("/scenarios/<scenario_id>")
@_require_permission("bia_pda:scenarios")
def scenario_detail(scenario_id: str):
	s = _run(_svc.get_scenario(g.tenant_id, scenario_id))
	if not s: abort(404)
	return jsonify({"view": "scenario_detail", "scenario": s})

@pda_bp.get("/features")
@_require_permission("bia_pda:features")
def features():
	return jsonify({"view": "feature_store", "features": _run(_svc.list_features(g.tenant_id))})

@pda_bp.get("/predictions")
@_require_permission("bia_pda:view")
def predictions():
	return jsonify({"view": "prediction_log", "predictions": _run(_svc.list_predictions(g.tenant_id))})

@pda_bp.get("/audit")
@_require_permission("bia_pda:admin")
def audit_log():
	return jsonify({"view": "audit_log", "events": _run(_svc.get_audit_events(g.tenant_id))})

@pda_bp.get("/settings")
@_require_permission("bia_pda:admin")
def settings():
	from .capability_contract import get_capability_contract
	return jsonify({"view": "settings", "config": get_capability_contract(g.tenant_id)["configuration"]})
