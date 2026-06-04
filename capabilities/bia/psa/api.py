"""Flask Blueprint REST API for APG Prescriptive Analytics (bia_psa)."""

from __future__ import annotations
import asyncio
from typing import Any
from flask import Blueprint, jsonify, request

try:
	from .service import PrescriptiveAnalyticsService
	from .capability_contract import get_capability_contract
except ImportError:
	from service import PrescriptiveAnalyticsService
	from capability_contract import get_capability_contract

api_bp = Blueprint("bia_psa_api", __name__, url_prefix="/api/bia/psa")
_svc = PrescriptiveAnalyticsService()

def _run(coro): return asyncio.run(coro)
def _tenant(): return request.headers.get("X-Tenant-ID", "default")
def _user(): return request.headers.get("X-User-ID", "anonymous")
def _ok(data: Any, status: int=200): return jsonify({"status": "ok", "data": data}), status
def _err(msg: str, status: int=400): return jsonify({"status": "error", "message": msg}), status

@api_bp.get("/contract")
def get_contract(): return _ok(get_capability_contract(_tenant()))

@api_bp.get("/optimisations")
def list_optimisations(): return _ok(_run(_svc.list_optimisations(_tenant())))

@api_bp.post("/optimisations")
def create_optimisation():
	b = request.get_json(silent=True) or {}
	missing = [f for f in ["name","optimisation_type","objective_type","objective_description"] if f not in b]
	if missing: return _err(f"Missing: {missing}", 400)
	try: o = _run(_svc.create_optimisation(_tenant(), b["name"], b["optimisation_type"], b["objective_type"], b["objective_description"], b.get("owner_id",_user()), b.get("constraints",[]), b.get("variables",[]), b.get("description")))
	except ValueError as e: return _err(str(e), 400)
	return _ok(o, 201)

@api_bp.get("/optimisations/<opt_id>")
def get_optimisation(opt_id: str):
	o = _run(_svc.get_optimisation(_tenant(), opt_id))
	return _ok(o) if o else _err("Not found", 404)

@api_bp.post("/optimisations/<opt_id>/run")
def run_optimisation(opt_id: str):
	try: o = _run(_svc.run_optimisation(_tenant(), opt_id))
	except ValueError as e: return _err(str(e), 400)
	return _ok(o)

@api_bp.post("/optimisations/<opt_id>/archive")
def archive_optimisation(opt_id: str):
	try: o = _run(_svc.archive_optimisation(_tenant(), opt_id))
	except ValueError as e: return _err(str(e), 404)
	return _ok(o)

@api_bp.delete("/optimisations/<opt_id>")
def delete_optimisation(opt_id: str):
	ok = _run(_svc.delete_optimisation(_tenant(), opt_id))
	return _ok({"deleted": opt_id}) if ok else _err("Not found", 404)

@api_bp.get("/recommendations")
def list_recommendations(): return _ok(_run(_svc.list_recommendations(_tenant(), request.args.get("optimisation_id"))))

@api_bp.post("/recommendations")
def generate_recommendation():
	b = request.get_json(silent=True) or {}
	missing = [f for f in ["optimisation_id","name","recommendation_type","description"] if f not in b]
	if missing: return _err(f"Missing: {missing}", 400)
	try: rec = _run(_svc.generate_recommendation(_tenant(), b["optimisation_id"], b["name"], b["recommendation_type"], b["description"], b.get("owner_id",_user()), b.get("actions",[]), b.get("impact_estimate",{})))
	except ValueError as e: return _err(str(e), 400)
	return _ok(rec, 201)

@api_bp.get("/recommendations/<rec_id>")
def get_recommendation(rec_id: str):
	r = _run(_svc.get_recommendation(_tenant(), rec_id))
	return _ok(r) if r else _err("Not found", 404)

@api_bp.post("/recommendations/<rec_id>/approve")
def approve_recommendation(rec_id: str):
	b = request.get_json(silent=True) or {}
	try: r = _run(_svc.approve_recommendation(_tenant(), rec_id, b.get("approver_id",_user())))
	except ValueError as e: return _err(str(e), 404)
	return _ok(r)

@api_bp.post("/recommendations/<rec_id>/reject")
def reject_recommendation(rec_id: str):
	b = request.get_json(silent=True) or {}
	try: r = _run(_svc.reject_recommendation(_tenant(), rec_id, b.get("approver_id",_user())))
	except ValueError as e: return _err(str(e), 404)
	return _ok(r)

@api_bp.post("/recommendations/<rec_id>/act")
def act_on_recommendation(rec_id: str):
	b = request.get_json(silent=True) or {}
	try: r = _run(_svc.act_on_recommendation(_tenant(), rec_id, b.get("actor_id",_user())))
	except ValueError as e: return _err(str(e), 400)
	return _ok(r)

@api_bp.get("/whatif")
def list_whatifs(): return _ok(_run(_svc.list_whatifs(_tenant())))

@api_bp.post("/whatif")
def create_whatif():
	b = request.get_json(silent=True) or {}
	missing = [f for f in ["name","baseline_model_id","parameters"] if f not in b]
	if missing: return _err(f"Missing: {missing}", 400)
	try: w = _run(_svc.create_whatif(_tenant(), b["name"], b["baseline_model_id"], b["parameters"], b.get("owner_id",_user()), b.get("description")))
	except ValueError as e: return _err(str(e), 400)
	return _ok(w, 201)

@api_bp.get("/whatif/<whatif_id>")
def get_whatif(whatif_id: str):
	w = _run(_svc.get_whatif(_tenant(), whatif_id))
	return _ok(w) if w else _err("Not found", 404)

@api_bp.post("/whatif/<whatif_id>/run")
def run_whatif(whatif_id: str):
	try: w = _run(_svc.run_whatif(_tenant(), whatif_id))
	except ValueError as e: return _err(str(e), 404)
	return _ok(w)

@api_bp.delete("/whatif/<whatif_id>")
def delete_whatif(whatif_id: str):
	ok = _run(_svc.delete_whatif(_tenant(), whatif_id))
	return _ok({"deleted": whatif_id}) if ok else _err("Not found", 404)

@api_bp.post("/decisions")
def record_decision():
	b = request.get_json(silent=True) or {}
	missing = [f for f in ["decision_type","rationale"] if f not in b]
	if missing: return _err(f"Missing: {missing}", 400)
	try: d = _run(_svc.record_decision(_tenant(), b["decision_type"], b["rationale"], b.get("decided_by",_user()), b.get("recommendation_id"), b.get("outcome")))
	except ValueError as e: return _err(str(e), 400)
	return _ok(d, 201)

@api_bp.get("/decisions")
def list_decisions(): return _ok(_run(_svc.list_decisions(_tenant())))

@api_bp.get("/stats")
def get_stats(): return _ok(_run(_svc.get_stats(_tenant())))

@api_bp.get("/audit")
def get_audit(): return _ok(_run(_svc.get_audit_events(_tenant())))
