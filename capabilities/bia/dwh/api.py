"""Flask Blueprint REST API for APG Data Warehouse (bia_dwh)."""

from __future__ import annotations
import asyncio
from typing import Any
from flask import Blueprint, jsonify, request

try:
	from .service import DataWarehouseService
	from .capability_contract import get_capability_contract
except ImportError:
	from service import DataWarehouseService
	from capability_contract import get_capability_contract

api_bp = Blueprint("bia_dwh_api", __name__, url_prefix="/api/bia/dwh")
_svc = DataWarehouseService()

def _run(coro): return asyncio.run(coro)
def _tenant(): return request.headers.get("X-Tenant-ID", "default")
def _user(): return request.headers.get("X-User-ID", "anonymous")
def _ok(data: Any, status: int=200): return jsonify({"status": "ok", "data": data}), status
def _err(msg: str, status: int=400): return jsonify({"status": "error", "message": msg}), status

@api_bp.get("/contract")
def get_contract(): return _ok(get_capability_contract(_tenant()))

@api_bp.get("/schemas")
def list_schemas(): return _ok(_run(_svc.list_schemas(_tenant())))

@api_bp.post("/schemas")
def create_schema():
	b = request.get_json(silent=True) or {}
	missing = [f for f in ["name","schema_type","grain","owner_id"] if f not in b]
	if missing: return _err(f"Missing: {missing}", 400)
	try: s = _run(_svc.create_schema(_tenant(), b["name"], b["schema_type"], b["grain"], b.get("owner_id",_user()), b.get("description"), b.get("tags",[])))
	except ValueError as e: return _err(str(e), 400)
	return _ok(s, 201)

@api_bp.get("/schemas/<schema_id>")
def get_schema(schema_id: str):
	s = _run(_svc.get_schema(_tenant(), schema_id))
	return _ok(s) if s else _err("Not found", 404)

@api_bp.put("/schemas/<schema_id>")
def update_schema(schema_id: str):
	try: s = _run(_svc.update_schema(_tenant(), schema_id, request.get_json(silent=True) or {}))
	except ValueError as e: return _err(str(e), 404)
	return _ok(s)

@api_bp.delete("/schemas/<schema_id>")
def delete_schema(schema_id: str):
	ok = _run(_svc.delete_schema(_tenant(), schema_id))
	return _ok({"deleted": schema_id}) if ok else _err("Not found", 404)

@api_bp.get("/tables")
def list_tables(): return _ok(_run(_svc.list_tables(_tenant(), request.args.get("schema_id"))))

@api_bp.post("/tables")
def register_table():
	b = request.get_json(silent=True) or {}
	missing = [f for f in ["schema_id","name","table_type","columns"] if f not in b]
	if missing: return _err(f"Missing: {missing}", 400)
	try: t = _run(_svc.register_table(_tenant(), b["schema_id"], b["name"], b["table_type"], b["columns"], b.get("owner_id",_user()), b.get("partition_strategy","none"), b.get("storage_tier","hot"), b.get("lineage_ref"), b.get("description")))
	except ValueError as e: return _err(str(e), 400)
	return _ok(t, 201)

@api_bp.get("/tables/<table_id>")
def get_table(table_id: str):
	t = _run(_svc.get_table(_tenant(), table_id))
	return _ok(t) if t else _err("Not found", 404)

@api_bp.put("/tables/<table_id>")
def update_table(table_id: str):
	try: t = _run(_svc.update_table(_tenant(), table_id, request.get_json(silent=True) or {}))
	except ValueError as e: return _err(str(e), 404)
	return _ok(t)

@api_bp.delete("/tables/<table_id>")
def delete_table(table_id: str):
	ok = _run(_svc.delete_table(_tenant(), table_id))
	return _ok({"deleted": table_id}) if ok else _err("Not found", 404)

@api_bp.get("/etl")
def list_etl_jobs(): return _ok(_run(_svc.list_etl_jobs(_tenant())))

@api_bp.post("/etl")
def create_etl_job():
	b = request.get_json(silent=True) or {}
	missing = [f for f in ["name","source_ref","target_table_id","load_strategy"] if f not in b]
	if missing: return _err(f"Missing: {missing}", 400)
	try: j = _run(_svc.create_etl_job(_tenant(), b["name"], b["source_ref"], b["target_table_id"], b["load_strategy"], b.get("owner_id",_user()), b.get("transform_sql"), b.get("schedule_cron"), b.get("description")))
	except ValueError as e: return _err(str(e), 400)
	return _ok(j, 201)

@api_bp.get("/etl/<job_id>")
def get_etl_job(job_id: str):
	j = _run(_svc.get_etl_job(_tenant(), job_id))
	return _ok(j) if j else _err("Not found", 404)

@api_bp.post("/etl/<job_id>/run")
def run_etl_job(job_id: str):
	try: j = _run(_svc.run_etl_job(_tenant(), job_id))
	except ValueError as e: return _err(str(e), 400)
	return _ok(j)

@api_bp.delete("/etl/<job_id>")
def delete_etl_job(job_id: str):
	ok = _run(_svc.delete_etl_job(_tenant(), job_id))
	return _ok({"deleted": job_id}) if ok else _err("Not found", 404)

@api_bp.get("/quality")
def list_quality_rules(): return _ok(_run(_svc.list_quality_rules(_tenant(), request.args.get("table_id"))))

@api_bp.post("/quality")
def add_quality_rule():
	b = request.get_json(silent=True) or {}
	missing = [f for f in ["table_id","name","rule_type"] if f not in b]
	if missing: return _err(f"Missing: {missing}", 400)
	try: r = _run(_svc.add_quality_rule(_tenant(), b["table_id"], b["name"], b["rule_type"], b.get("owner_id",_user()), b.get("column"), b.get("config",{})))
	except ValueError as e: return _err(str(e), 400)
	return _ok(r, 201)

@api_bp.delete("/quality/<rule_id>")
def delete_quality_rule(rule_id: str):
	ok = _run(_svc.delete_quality_rule(_tenant(), rule_id))
	return _ok({"deleted": rule_id}) if ok else _err("Not found", 404)

@api_bp.get("/lineage")
def get_lineage(): return _ok(_run(_svc.get_lineage(_tenant(), request.args.get("table_id"))))

@api_bp.post("/lineage")
def record_lineage():
	b = request.get_json(silent=True) or {}
	missing = [f for f in ["source_table_id","target_table_id"] if f not in b]
	if missing: return _err(f"Missing: {missing}", 400)
	rec = _run(_svc.record_lineage(_tenant(), b["source_table_id"], b["target_table_id"], b.get("etl_job_id"), b.get("transformation_description")))
	return _ok(rec, 201)

@api_bp.get("/stats")
def get_stats(): return _ok(_run(_svc.get_stats(_tenant())))

@api_bp.get("/audit")
def get_audit(): return _ok(_run(_svc.get_audit_events(_tenant())))
