"""Flask Blueprint REST API for APG Self-Service BI (bia_sbi)."""

from __future__ import annotations
import asyncio
from typing import Any
from flask import Blueprint, jsonify, request

try:
	from .service import SelfServiceBIService
	from .capability_contract import get_capability_contract
except ImportError:
	from service import SelfServiceBIService
	from capability_contract import get_capability_contract

api_bp = Blueprint("bia_sbi_api", __name__, url_prefix="/api/bia/sbi")
_svc = SelfServiceBIService()

def _run(coro): return asyncio.run(coro)
def _tenant(): return request.headers.get("X-Tenant-ID", "default")
def _user(): return request.headers.get("X-User-ID", "anonymous")
def _ok(data: Any, status: int=200): return jsonify({"status": "ok", "data": data}), status
def _err(msg: str, status: int=400): return jsonify({"status": "error", "message": msg}), status

@api_bp.get("/contract")
def get_contract(): return _ok(get_capability_contract(_tenant()))

@api_bp.get("/workspaces")
def list_workspaces(): return _ok(_run(_svc.list_workspaces(_tenant())))

@api_bp.post("/workspaces")
def create_workspace():
	b = request.get_json(silent=True) or {}
	if "name" not in b: return _err("Missing: name", 400)
	try: w = _run(_svc.create_workspace(_tenant(), b["name"], b.get("owner_id",_user()), b.get("access_level","personal"), b.get("description"), b.get("tags",[])))
	except ValueError as e: return _err(str(e), 400)
	return _ok(w, 201)

@api_bp.get("/workspaces/<workspace_id>")
def get_workspace(workspace_id: str):
	w = _run(_svc.get_workspace(_tenant(), workspace_id))
	return _ok(w) if w else _err("Not found", 404)

@api_bp.delete("/workspaces/<workspace_id>")
def delete_workspace(workspace_id: str):
	ok = _run(_svc.delete_workspace(_tenant(), workspace_id))
	return _ok({"deleted": workspace_id}) if ok else _err("Not found", 404)

@api_bp.get("/workspaces/<workspace_id>/charts")
def list_charts(workspace_id: str): return _ok(_run(_svc.list_charts(_tenant(), workspace_id)))

@api_bp.post("/workspaces/<workspace_id>/charts")
def create_chart(workspace_id: str):
	b = request.get_json(silent=True) or {}
	missing = [f for f in ["name","chart_type","datasource_id"] if f not in b]
	if missing: return _err(f"Missing: {missing}", 400)
	try: c = _run(_svc.create_chart(_tenant(), workspace_id, b["name"], b["chart_type"], b["datasource_id"], b.get("owner_id",_user()), b.get("config",{})))
	except ValueError as e: return _err(str(e), 400)
	return _ok(c, 201)

@api_bp.delete("/charts/<chart_id>")
def delete_chart(chart_id: str):
	ok = _run(_svc.delete_chart(_tenant(), chart_id))
	return _ok({"deleted": chart_id}) if ok else _err("Not found", 404)

@api_bp.get("/catalogue")
def list_catalogue(): return _ok(_run(_svc.list_catalogue(_tenant())))

@api_bp.post("/catalogue")
def create_catalogue_entry():
	b = request.get_json(silent=True) or {}
	missing = [f for f in ["name","datasource_id","description"] if f not in b]
	if missing: return _err(f"Missing: {missing}", 400)
	try: e = _run(_svc.create_catalogue_entry(_tenant(), b["name"], b["datasource_id"], b.get("owner_id",_user()), b["description"], b.get("governance_tier","governed"), b.get("tags",[]), b.get("schema_ref")))
	except ValueError as e_: return _err(str(e_), 400)
	return _ok(e, 201)

@api_bp.get("/catalogue/<entry_id>")
def get_catalogue_entry(entry_id: str):
	e = _run(_svc.get_catalogue_entry(_tenant(), entry_id))
	return _ok(e) if e else _err("Not found", 404)

@api_bp.post("/catalogue/<entry_id>/approve")
def approve_catalogue_entry(entry_id: str):
	b = request.get_json(silent=True) or {}
	try: e = _run(_svc.approve_catalogue_entry(_tenant(), entry_id, b.get("approver_id",_user())))
	except ValueError as ex: return _err(str(ex), 404)
	return _ok(e)

@api_bp.get("/sandboxes")
def list_sandboxes(): return _ok(_run(_svc.list_sandboxes(_tenant())))

@api_bp.post("/sandboxes")
def create_sandbox():
	b = request.get_json(silent=True) or {}
	if "name" not in b: return _err("Missing: name", 400)
	try: sb = _run(_svc.create_sandbox(_tenant(), b["name"], b.get("owner_id",_user()), b.get("datasource_ids",[]), b.get("description")))
	except ValueError as e: return _err(str(e), 400)
	return _ok(sb, 201)

@api_bp.get("/sandboxes/<sandbox_id>")
def get_sandbox(sandbox_id: str):
	sb = _run(_svc.get_sandbox(_tenant(), sandbox_id))
	return _ok(sb) if sb else _err("Not found", 404)

@api_bp.delete("/sandboxes/<sandbox_id>")
def delete_sandbox(sandbox_id: str):
	ok = _run(_svc.delete_sandbox(_tenant(), sandbox_id))
	return _ok({"deleted": sandbox_id}) if ok else _err("Not found", 404)

@api_bp.post("/ask")
def submit_nlq():
	b = request.get_json(silent=True) or {}
	if "query_text" not in b: return _err("Missing: query_text", 400)
	try: r = _run(_svc.submit_nlq(_tenant(), b["query_text"], b.get("submitted_by",_user()), b.get("workspace_id"), b.get("nlq_engine","hybrid")))
	except ValueError as e: return _err(str(e), 400)
	return _ok(r, 201)

@api_bp.get("/stats")
def get_stats(): return _ok(_run(_svc.get_stats(_tenant())))

@api_bp.get("/audit")
def get_audit(): return _ok(_run(_svc.get_audit_events(_tenant())))
