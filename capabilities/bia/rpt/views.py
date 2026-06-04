"""Flask Blueprint views for APG Report Builder (bia_rpt)."""

from __future__ import annotations
import asyncio
from functools import wraps
from flask import Blueprint, abort, g, jsonify, request

try:
	from .capability_contract import CAPABILITY_ID
	from .service import ReportBuilderService
except ImportError:
	from capability_contract import CAPABILITY_ID
	from service import ReportBuilderService

rpt_bp = Blueprint("bia_rpt", __name__, url_prefix="/bia/rpt")
_svc = ReportBuilderService()

def _require_permission(perm: str):
	def decorator(fn):
		@wraps(fn)
		def wrapper(*args, **kwargs):
			g.tenant_id = request.headers.get("X-Tenant-ID", "default")
			g.user_id = request.headers.get("X-User-ID", "anonymous")
			if perm not in request.headers.get("X-Permissions","") and "bia_rpt:admin" not in request.headers.get("X-Permissions",""):
				abort(403)
			return fn(*args, **kwargs)
		return wrapper
	return decorator

def _run(coro): return asyncio.get_event_loop().run_until_complete(coro)

@rpt_bp.get("/dashboard")
@_require_permission("bia_rpt:view")
def dashboard():
	return jsonify({"view": "report_dashboard", "stats": _run(_svc.get_stats(g.tenant_id))})

@rpt_bp.get("/reports")
@_require_permission("bia_rpt:view")
def report_library():
	return jsonify({"view": "report_library", "reports": _run(_svc.list_reports(g.tenant_id))})

@rpt_bp.get("/reports/<report_id>")
@_require_permission("bia_rpt:view")
def report_detail(report_id: str):
	r = _run(_svc.get_report(g.tenant_id, report_id))
	if not r: abort(404)
	return jsonify({"view": "report_detail", "report": r, "schedules": _run(_svc.list_schedules(g.tenant_id, report_id)), "runs": _run(_svc.list_runs(g.tenant_id, report_id))})

@rpt_bp.get("/reports/<report_id>/build")
@_require_permission("bia_rpt:edit")
def report_builder(report_id: str):
	r = _run(_svc.get_report(g.tenant_id, report_id))
	if not r: abort(404)
	return jsonify({"view": "report_builder", "report": r})

@rpt_bp.get("/schedules")
@_require_permission("bia_rpt:schedule")
def schedules():
	return jsonify({"view": "schedule_manager", "schedules": _run(_svc.list_schedules(g.tenant_id))})

@rpt_bp.get("/distribution")
@_require_permission("bia_rpt:distribute")
def distribution():
	return jsonify({"view": "distribution_manager", "distributions": _run(_svc.list_distributions(g.tenant_id))})

@rpt_bp.get("/history")
@_require_permission("bia_rpt:view")
def run_history():
	return jsonify({"view": "run_history", "runs": _run(_svc.list_runs(g.tenant_id))})

@rpt_bp.get("/audit")
@_require_permission("bia_rpt:admin")
def audit_log():
	return jsonify({"view": "audit_log", "events": _run(_svc.get_audit_events(g.tenant_id))})

@rpt_bp.get("/settings")
@_require_permission("bia_rpt:admin")
def settings():
	from .capability_contract import get_capability_contract
	return jsonify({"view": "settings", "config": get_capability_contract(g.tenant_id)["configuration"]})
