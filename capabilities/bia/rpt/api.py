"""Flask Blueprint REST API for APG Report Builder (bia_rpt)."""

from __future__ import annotations
import asyncio
from typing import Any
from flask import Blueprint, jsonify, request

try:
	from .service import ReportBuilderService
	from .capability_contract import get_capability_contract
except ImportError:
	from service import ReportBuilderService
	from capability_contract import get_capability_contract

api_bp = Blueprint("bia_rpt_api", __name__, url_prefix="/api/bia/rpt")
_svc = ReportBuilderService()

def _run(coro): return asyncio.run(coro)
def _tenant(): return request.headers.get("X-Tenant-ID", "default")
def _user(): return request.headers.get("X-User-ID", "anonymous")
def _ok(data: Any, status: int=200): return jsonify({"status": "ok", "data": data}), status
def _err(msg: str, status: int=400): return jsonify({"status": "error", "message": msg}), status

@api_bp.get("/contract")
def get_contract(): return _ok(get_capability_contract(_tenant()))

@api_bp.get("/reports")
def list_reports(): return _ok(_run(_svc.list_reports(_tenant())))

@api_bp.post("/reports")
def create_report():
	b = request.get_json(silent=True) or {}
	missing = [f for f in ["name","report_type","datasource_id"] if f not in b]
	if missing: return _err(f"Missing: {missing}", 400)
	try: r = _run(_svc.create_report(_tenant(), b["name"], b["report_type"], b.get("owner_id",_user()), b["datasource_id"], b.get("sections",[]), b.get("parameters",[]), b.get("default_format","pdf"), b.get("description"), b.get("tags",[])))
	except ValueError as e: return _err(str(e), 400)
	return _ok(r, 201)

@api_bp.get("/reports/<report_id>")
def get_report(report_id: str):
	r = _run(_svc.get_report(_tenant(), report_id))
	return _ok(r) if r else _err("Not found", 404)

@api_bp.put("/reports/<report_id>")
def update_report(report_id: str):
	try: r = _run(_svc.update_report(_tenant(), report_id, request.get_json(silent=True) or {}))
	except ValueError as e: return _err(str(e), 400)
	return _ok(r)

@api_bp.delete("/reports/<report_id>")
def delete_report(report_id: str):
	try: ok = _run(_svc.delete_report(_tenant(), report_id))
	except ValueError as e: return _err(str(e), 400)
	return _ok({"deleted": report_id}) if ok else _err("Not found", 404)

@api_bp.post("/reports/<report_id>/publish")
def publish_report(report_id: str):
	try: r = _run(_svc.publish_report(_tenant(), report_id))
	except ValueError as e: return _err(str(e), 404)
	return _ok(r)

@api_bp.post("/reports/<report_id>/archive")
def archive_report(report_id: str):
	try: r = _run(_svc.archive_report(_tenant(), report_id))
	except ValueError as e: return _err(str(e), 404)
	return _ok(r)

@api_bp.post("/reports/<report_id>/run")
def run_report(report_id: str):
	b = request.get_json(silent=True) or {}
	try: run = _run(_svc.run_report(_tenant(), report_id, b.get("output_format","pdf"), b.get("parameters",{}), b.get("triggered_by","manual")))
	except ValueError as e: return _err(str(e), 400)
	return _ok(run, 201)

@api_bp.get("/reports/<report_id>/runs")
def list_runs(report_id: str): return _ok(_run(_svc.list_runs(_tenant(), report_id)))

@api_bp.get("/schedules")
def list_schedules(): return _ok(_run(_svc.list_schedules(_tenant(), request.args.get("report_id"))))

@api_bp.post("/schedules")
def create_schedule():
	b = request.get_json(silent=True) or {}
	missing = [f for f in ["report_id","frequency"] if f not in b]
	if missing: return _err(f"Missing: {missing}", 400)
	try: s = _run(_svc.create_schedule(_tenant(), b["report_id"], b["frequency"], b.get("owner_id",_user()), b.get("cron_expression"), b.get("output_format","pdf"), b.get("notification_targets",[])))
	except ValueError as e: return _err(str(e), 400)
	return _ok(s, 201)

@api_bp.delete("/schedules/<schedule_id>")
def delete_schedule(schedule_id: str):
	ok = _run(_svc.delete_schedule(_tenant(), schedule_id))
	return _ok({"deleted": schedule_id}) if ok else _err("Not found", 404)

@api_bp.get("/distributions")
def list_distributions(): return _ok(_run(_svc.list_distributions(_tenant(), request.args.get("report_id"))))

@api_bp.post("/distributions")
def create_distribution():
	b = request.get_json(silent=True) or {}
	missing = [f for f in ["report_id","channel","recipient"] if f not in b]
	if missing: return _err(f"Missing: {missing}", 400)
	try: d = _run(_svc.create_distribution(_tenant(), b["report_id"], b["channel"], b["recipient"], b.get("owner_id",_user()), b.get("output_format","pdf"), b.get("config",{}), b.get("is_external",False)))
	except ValueError as e: return _err(str(e), 400)
	return _ok(d, 201)

@api_bp.post("/distributions/<dist_id>/approve")
def approve_distribution(dist_id: str):
	b = request.get_json(silent=True) or {}
	try: d = _run(_svc.approve_distribution(_tenant(), dist_id, b.get("approver_id",_user())))
	except ValueError as e: return _err(str(e), 404)
	return _ok(d)

@api_bp.get("/stats")
def get_stats(): return _ok(_run(_svc.get_stats(_tenant())))

@api_bp.get("/audit")
def get_audit(): return _ok(_run(_svc.get_audit_events(_tenant())))
