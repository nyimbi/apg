"""Flask Blueprint views for APG Data Warehouse (bia_dwh)."""

from __future__ import annotations
import asyncio
from functools import wraps
from flask import Blueprint, abort, g, jsonify, request

try:
	from .capability_contract import CAPABILITY_ID
	from .service import DataWarehouseService
except ImportError:
	from capability_contract import CAPABILITY_ID
	from service import DataWarehouseService

dwh_bp = Blueprint("bia_dwh", __name__, url_prefix="/bia/dwh")
_svc = DataWarehouseService()

def _require_permission(perm: str):
	def decorator(fn):
		@wraps(fn)
		def wrapper(*args, **kwargs):
			g.tenant_id = request.headers.get("X-Tenant-ID", "default")
			g.user_id = request.headers.get("X-User-ID", "anonymous")
			perms = request.headers.get("X-Permissions", "")
			if perm not in perms and "bia_dwh:admin" not in perms:
				abort(403, description=f"Permission required: {perm}")
			return fn(*args, **kwargs)
		return wrapper
	return decorator

def _run(coro): return asyncio.get_event_loop().run_until_complete(coro)

@dwh_bp.get("/dashboard")
@_require_permission("bia_dwh:view")
def dashboard():
	return jsonify({"view": "warehouse_dashboard", "stats": _run(_svc.get_stats(g.tenant_id))})

@dwh_bp.get("/schemas")
@_require_permission("bia_dwh:schemas")
def schemas():
	return jsonify({"view": "schema_explorer", "schemas": _run(_svc.list_schemas(g.tenant_id))})

@dwh_bp.get("/schemas/<schema_id>")
@_require_permission("bia_dwh:schemas")
def schema_detail(schema_id: str):
	s = _run(_svc.get_schema(g.tenant_id, schema_id))
	if not s: abort(404)
	tables = _run(_svc.list_tables(g.tenant_id, schema_id))
	return jsonify({"view": "schema_detail", "schema": s, "tables": tables})

@dwh_bp.get("/tables")
@_require_permission("bia_dwh:tables")
def tables():
	return jsonify({"view": "table_catalogue", "tables": _run(_svc.list_tables(g.tenant_id))})

@dwh_bp.get("/tables/<table_id>")
@_require_permission("bia_dwh:tables")
def table_detail(table_id: str):
	t = _run(_svc.get_table(g.tenant_id, table_id))
	if not t: abort(404)
	rules = _run(_svc.list_quality_rules(g.tenant_id, table_id))
	lineage = _run(_svc.get_lineage(g.tenant_id, table_id))
	return jsonify({"view": "table_detail", "table": t, "quality_rules": rules, "lineage": lineage})

@dwh_bp.get("/etl")
@_require_permission("bia_dwh:etl")
def etl_jobs():
	return jsonify({"view": "etl_job_manager", "jobs": _run(_svc.list_etl_jobs(g.tenant_id))})

@dwh_bp.get("/etl/<job_id>")
@_require_permission("bia_dwh:etl")
def etl_job_detail(job_id: str):
	j = _run(_svc.get_etl_job(g.tenant_id, job_id))
	if not j: abort(404)
	return jsonify({"view": "etl_job_detail", "job": j})

@dwh_bp.get("/quality")
@_require_permission("bia_dwh:quality")
def quality():
	return jsonify({"view": "data_quality_console", "rules": _run(_svc.list_quality_rules(g.tenant_id))})

@dwh_bp.get("/lineage")
@_require_permission("bia_dwh:lineage")
def lineage():
	return jsonify({"view": "lineage_viewer", "lineage": _run(_svc.get_lineage(g.tenant_id))})

@dwh_bp.get("/audit")
@_require_permission("bia_dwh:admin")
def audit_log():
	return jsonify({"view": "audit_log", "events": _run(_svc.get_audit_events(g.tenant_id))})

@dwh_bp.get("/settings")
@_require_permission("bia_dwh:admin")
def settings():
	from .capability_contract import get_capability_contract
	return jsonify({"view": "settings", "config": get_capability_contract(g.tenant_id)["configuration"]})
