"""Flask Blueprint views for APG Prescriptive Analytics (bia_psa)."""

from __future__ import annotations
import asyncio
from functools import wraps
from flask import Blueprint, abort, g, jsonify, request

try:
	from .capability_contract import CAPABILITY_ID
	from .service import PrescriptiveAnalyticsService
except ImportError:
	from capability_contract import CAPABILITY_ID
	from service import PrescriptiveAnalyticsService

psa_bp = Blueprint("bia_psa", __name__, url_prefix="/bia/psa")
_svc = PrescriptiveAnalyticsService()

def _require_permission(perm: str):
	def decorator(fn):
		@wraps(fn)
		def wrapper(*args, **kwargs):
			g.tenant_id = request.headers.get("X-Tenant-ID", "default")
			g.user_id = request.headers.get("X-User-ID", "anonymous")
			if perm not in request.headers.get("X-Permissions","") and "bia_psa:admin" not in request.headers.get("X-Permissions",""):
				abort(403)
			return fn(*args, **kwargs)
		return wrapper
	return decorator

def _run(coro): return asyncio.get_event_loop().run_until_complete(coro)

@psa_bp.get("/dashboard")
@_require_permission("bia_psa:view")
def dashboard():
	return jsonify({"view": "prescriptive_dashboard", "stats": _run(_svc.get_stats(g.tenant_id))})

@psa_bp.get("/optimisations")
@_require_permission("bia_psa:optimise")
def optimisations():
	return jsonify({"view": "optimisation_manager", "optimisations": _run(_svc.list_optimisations(g.tenant_id))})

@psa_bp.get("/optimisations/<opt_id>")
@_require_permission("bia_psa:optimise")
def optimisation_detail(opt_id: str):
	o = _run(_svc.get_optimisation(g.tenant_id, opt_id))
	if not o: abort(404)
	recs = _run(_svc.list_recommendations(g.tenant_id, opt_id))
	return jsonify({"view": "optimisation_detail", "optimisation": o, "recommendations": recs})

@psa_bp.get("/recommendations")
@_require_permission("bia_psa:recommendations")
def recommendations():
	return jsonify({"view": "recommendation_queue", "recommendations": _run(_svc.list_recommendations(g.tenant_id))})

@psa_bp.get("/recommendations/<rec_id>")
@_require_permission("bia_psa:recommendations")
def recommendation_detail(rec_id: str):
	r = _run(_svc.get_recommendation(g.tenant_id, rec_id))
	if not r: abort(404)
	return jsonify({"view": "recommendation_detail", "recommendation": r})

@psa_bp.get("/whatif")
@_require_permission("bia_psa:whatif")
def whatif():
	return jsonify({"view": "whatif_builder", "whatifs": _run(_svc.list_whatifs(g.tenant_id))})

@psa_bp.get("/whatif/<whatif_id>")
@_require_permission("bia_psa:whatif")
def whatif_detail(whatif_id: str):
	w = _run(_svc.get_whatif(g.tenant_id, whatif_id))
	if not w: abort(404)
	return jsonify({"view": "whatif_detail", "whatif": w})

@psa_bp.get("/approvals")
@_require_permission("bia_psa:approve")
def approvals():
	recs = _run(_svc.list_recommendations(g.tenant_id))
	pending = [r for r in recs if r["approval_state"] == "pending"]
	return jsonify({"view": "approval_queue", "pending": pending})

@psa_bp.get("/decisions")
@_require_permission("bia_psa:decisions")
def decisions():
	return jsonify({"view": "decision_log", "decisions": _run(_svc.list_decisions(g.tenant_id))})

@psa_bp.get("/audit")
@_require_permission("bia_psa:admin")
def audit_log():
	return jsonify({"view": "audit_log", "events": _run(_svc.get_audit_events(g.tenant_id))})

@psa_bp.get("/settings")
@_require_permission("bia_psa:admin")
def settings():
	from .capability_contract import get_capability_contract
	return jsonify({"view": "settings", "config": get_capability_contract(g.tenant_id)["configuration"]})
