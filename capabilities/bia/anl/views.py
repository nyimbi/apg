"""Flask Blueprint views for APG Analytics Engine (bia_anl)."""

from __future__ import annotations

import asyncio
from datetime import datetime
from functools import wraps
from typing import Any

from flask import Blueprint, abort, g, jsonify, redirect, render_template_string, request, url_for

try:
	from .capability_contract import CAPABILITY_ID, CAPABILITY_NAME, evaluate_capability_rules
	from .service import AnalyticsEngineService
except ImportError:
	from capability_contract import CAPABILITY_ID, CAPABILITY_NAME, evaluate_capability_rules
	from service import AnalyticsEngineService

anl_bp = Blueprint("bia_anl", __name__, url_prefix="/bia/anl")
_svc = AnalyticsEngineService()


def _require_permission(perm: str):
	def decorator(fn):
		@wraps(fn)
		def wrapper(*args, **kwargs):
			tenant_id = request.headers.get("X-Tenant-ID", "default")
			user_id = request.headers.get("X-User-ID", "anonymous")
			perms = request.headers.get("X-Permissions", "")
			if perm not in perms and "bia_anl:admin" not in perms:
				abort(403, description=f"Permission required: {perm}")
			g.tenant_id = tenant_id
			g.user_id = user_id
			return fn(*args, **kwargs)
		return wrapper
	return decorator


def _run(coro):
	loop = asyncio.get_event_loop()
	return loop.run_until_complete(coro)


@anl_bp.route("/dashboard")
@_require_permission("bia_anl:view")
def dashboard():
	tenant_id = g.tenant_id
	stats = _run(_svc.get_dashboard_stats(tenant_id))
	return jsonify({"view": "dashboard", "capability": CAPABILITY_ID, "stats": stats})


@anl_bp.route("/query-builder")
@_require_permission("bia_anl:query")
def query_builder():
	tenant_id = g.tenant_id
	datasources = _run(_svc.list_datasources(tenant_id))
	return jsonify({"view": "query_builder", "datasources": datasources})


@anl_bp.route("/saved-queries")
@_require_permission("bia_anl:query")
def saved_queries():
	tenant_id = g.tenant_id
	queries = _run(_svc.list_queries(tenant_id))
	return jsonify({"view": "saved_queries", "queries": queries, "total": len(queries)})


@anl_bp.route("/saved-queries/<query_id>")
@_require_permission("bia_anl:query")
def query_detail(query_id: str):
	tenant_id = g.tenant_id
	query = _run(_svc.get_query(tenant_id, query_id))
	if not query:
		abort(404, description="Query not found")
	return jsonify({"view": "query_detail", "query": query})


@anl_bp.route("/saved-queries/<query_id>/execute", methods=["POST"])
@_require_permission("bia_anl:query")
def execute_saved_query(query_id: str):
	tenant_id = g.tenant_id
	params = request.get_json(silent=True) or {}
	result = _run(_svc.execute_query(tenant_id, query_id, params))
	return jsonify({"view": "query_result", "result": result})


@anl_bp.route("/cubes")
@_require_permission("bia_anl:cubes")
def cube_explorer():
	tenant_id = g.tenant_id
	cubes = _run(_svc.list_cubes(tenant_id))
	return jsonify({"view": "cube_explorer", "cubes": cubes, "total": len(cubes)})


@anl_bp.route("/cubes/<cube_id>")
@_require_permission("bia_anl:cubes")
def cube_detail(cube_id: str):
	tenant_id = g.tenant_id
	cube = _run(_svc.get_cube(tenant_id, cube_id))
	if not cube:
		abort(404, description="Cube not found")
	return jsonify({"view": "cube_detail", "cube": cube})


@anl_bp.route("/cubes/<cube_id>/refresh", methods=["POST"])
@_require_permission("bia_anl:cubes")
def refresh_cube(cube_id: str):
	tenant_id = g.tenant_id
	result = _run(_svc.refresh_cube(tenant_id, cube_id))
	return jsonify({"view": "cube_refresh", "result": result})


@anl_bp.route("/metrics")
@_require_permission("bia_anl:metrics")
def metric_library():
	tenant_id = g.tenant_id
	metrics = _run(_svc.list_metrics(tenant_id))
	return jsonify({"view": "metric_library", "metrics": metrics, "total": len(metrics)})


@anl_bp.route("/metrics/<metric_id>")
@_require_permission("bia_anl:metrics")
def metric_detail(metric_id: str):
	tenant_id = g.tenant_id
	metric = _run(_svc.get_metric(tenant_id, metric_id))
	if not metric:
		abort(404, description="Metric not found")
	return jsonify({"view": "metric_detail", "metric": metric})


@anl_bp.route("/datasources")
@_require_permission("bia_anl:admin")
def datasource_manager():
	tenant_id = g.tenant_id
	datasources = _run(_svc.list_datasources(tenant_id))
	return jsonify({"view": "datasource_manager", "datasources": datasources})


@anl_bp.route("/datasources/<ds_id>")
@_require_permission("bia_anl:admin")
def datasource_detail(ds_id: str):
	tenant_id = g.tenant_id
	ds = _run(_svc.get_datasource(tenant_id, ds_id))
	if not ds:
		abort(404, description="Datasource not found")
	return jsonify({"view": "datasource_detail", "datasource": ds})


@anl_bp.route("/schedules")
@_require_permission("bia_anl:schedule")
def schedule_manager():
	tenant_id = g.tenant_id
	schedules = _run(_svc.list_schedules(tenant_id))
	return jsonify({"view": "schedule_manager", "schedules": schedules})


@anl_bp.route("/audit")
@_require_permission("bia_anl:admin")
def audit_log():
	tenant_id = g.tenant_id
	events = _run(_svc.get_audit_events(tenant_id))
	return jsonify({"view": "audit_log", "events": events})


@anl_bp.route("/settings")
@_require_permission("bia_anl:admin")
def settings():
	tenant_id = g.tenant_id
	from .capability_contract import get_capability_contract
	contract = get_capability_contract(tenant_id)
	return jsonify({"view": "settings", "config": contract["configuration"]})
