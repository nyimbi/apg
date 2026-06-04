"""Flask Blueprint views for APG Self-Service BI (bia_sbi)."""

from __future__ import annotations
import asyncio
from functools import wraps
from flask import Blueprint, abort, g, jsonify, request

try:
	from .capability_contract import CAPABILITY_ID
	from .service import SelfServiceBIService
except ImportError:
	from capability_contract import CAPABILITY_ID
	from service import SelfServiceBIService

sbi_bp = Blueprint("bia_sbi", __name__, url_prefix="/bia/sbi")
_svc = SelfServiceBIService()

def _require_permission(perm: str):
	def decorator(fn):
		@wraps(fn)
		def wrapper(*args, **kwargs):
			g.tenant_id = request.headers.get("X-Tenant-ID", "default")
			g.user_id = request.headers.get("X-User-ID", "anonymous")
			if perm not in request.headers.get("X-Permissions","") and "bia_sbi:admin" not in request.headers.get("X-Permissions",""):
				abort(403)
			return fn(*args, **kwargs)
		return wrapper
	return decorator

def _run(coro): return asyncio.get_event_loop().run_until_complete(coro)

@sbi_bp.get("/")
@_require_permission("bia_sbi:view")
def home():
	return jsonify({"view": "sbi_home", "stats": _run(_svc.get_stats(g.tenant_id))})

@sbi_bp.get("/builder")
@_require_permission("bia_sbi:build")
def builder():
	workspaces = _run(_svc.list_workspaces(g.tenant_id))
	return jsonify({"view": "visual_builder", "workspaces": workspaces})

@sbi_bp.get("/workspaces/<workspace_id>")
@_require_permission("bia_sbi:build")
def workspace(workspace_id: str):
	w = _run(_svc.get_workspace(g.tenant_id, workspace_id))
	if not w: abort(404)
	charts = _run(_svc.list_charts(g.tenant_id, workspace_id))
	return jsonify({"view": "workspace_view", "workspace": w, "charts": charts})

@sbi_bp.get("/ask")
@_require_permission("bia_sbi:query")
def nlq():
	history = _run(_svc.list_nlq_history(g.tenant_id))
	return jsonify({"view": "nlq", "recent_queries": history[-10:]})

@sbi_bp.get("/catalogue")
@_require_permission("bia_sbi:catalogue")
def catalogue():
	return jsonify({"view": "data_catalogue", "entries": _run(_svc.list_catalogue(g.tenant_id))})

@sbi_bp.get("/catalogue/<entry_id>")
@_require_permission("bia_sbi:catalogue")
def catalogue_detail(entry_id: str):
	e = _run(_svc.get_catalogue_entry(g.tenant_id, entry_id))
	if not e: abort(404)
	return jsonify({"view": "catalogue_entry", "entry": e})

@sbi_bp.get("/sandboxes")
@_require_permission("bia_sbi:sandbox")
def sandboxes():
	return jsonify({"view": "sandbox_manager", "sandboxes": _run(_svc.list_sandboxes(g.tenant_id))})

@sbi_bp.get("/sandboxes/<sandbox_id>")
@_require_permission("bia_sbi:sandbox")
def sandbox_detail(sandbox_id: str):
	sb = _run(_svc.get_sandbox(g.tenant_id, sandbox_id))
	if not sb: abort(404)
	return jsonify({"view": "sandbox_detail", "sandbox": sb})

@sbi_bp.get("/catalogue/approvals")
@_require_permission("bia_sbi:admin")
def catalogue_approvals():
	entries = _run(_svc.list_catalogue(g.tenant_id))
	pending = [e for e in entries if e["state"] == "draft"]
	return jsonify({"view": "catalogue_approvals", "pending": pending})

@sbi_bp.get("/audit")
@_require_permission("bia_sbi:admin")
def audit_log():
	return jsonify({"view": "audit_log", "events": _run(_svc.get_audit_events(g.tenant_id))})

@sbi_bp.get("/settings")
@_require_permission("bia_sbi:admin")
def settings():
	from .capability_contract import get_capability_contract
	return jsonify({"view": "settings", "config": get_capability_contract(g.tenant_id)["configuration"]})
