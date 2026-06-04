"""Flask Blueprint REST API for APG Analytics Engine (bia_anl)."""

from __future__ import annotations

import asyncio
from typing import Any

from flask import Blueprint, jsonify, request, abort

try:
	from .service import AnalyticsEngineService
	from .capability_contract import CAPABILITY_ID, get_capability_contract
except ImportError:
	from service import AnalyticsEngineService
	from capability_contract import CAPABILITY_ID, get_capability_contract

api_bp = Blueprint("bia_anl_api", __name__, url_prefix="/api/bia/anl")
_svc = AnalyticsEngineService()


def _run(coro):
	return asyncio.run(coro)


def _tenant() -> str:
	return request.headers.get("X-Tenant-ID", "default")


def _user() -> str:
	return request.headers.get("X-User-ID", "anonymous")


def _ok(data: Any, status: int = 200):
	return jsonify({"status": "ok", "data": data}), status


def _err(msg: str, status: int = 400):
	return jsonify({"status": "error", "message": msg}), status


# ── Contract ──────────────────────────────────────────────────────────────────

@api_bp.get("/contract")
def get_contract():
	"""Return capability contract for this tenant."""
	return _ok(get_capability_contract(_tenant()))


# ── Datasources ───────────────────────────────────────────────────────────────

@api_bp.get("/datasources")
def list_datasources():
	"""
	GET /api/bia/anl/datasources
	List all datasources for the tenant.
	Permission: bia_anl:admin
	"""
	return _ok(_run(_svc.list_datasources(_tenant())))


@api_bp.post("/datasources")
def create_datasource():
	"""
	POST /api/bia/anl/datasources
	Register a new datasource.
	Permission: bia_anl:admin
	Body: {name, datasource_type, connection_config, credentials_vault_ref, owner_id}
	"""
	body = request.get_json(silent=True) or {}
	required = ["name", "datasource_type", "connection_config", "credentials_vault_ref", "owner_id"]
	missing = [f for f in required if f not in body]
	if missing:
		return _err(f"Missing fields: {missing}", 400)
	try:
		ds = _run(_svc.register_datasource(
			tenant_id=_tenant(),
			name=body["name"],
			datasource_type=body["datasource_type"],
			connection_config=body["connection_config"],
			credentials_vault_ref=body["credentials_vault_ref"],
			owner_id=body.get("owner_id", _user()),
			description=body.get("description"),
			tags=body.get("tags", []),
		))
	except ValueError as e:
		return _err(str(e), 400)
	return _ok(ds, 201)


@api_bp.get("/datasources/<ds_id>")
def get_datasource(ds_id: str):
	"""
	GET /api/bia/anl/datasources/<id>
	Get datasource detail.
	Permission: bia_anl:admin
	"""
	ds = _run(_svc.get_datasource(_tenant(), ds_id))
	if not ds:
		return _err("Datasource not found", 404)
	return _ok(ds)


@api_bp.post("/datasources/<ds_id>/test")
def test_datasource(ds_id: str):
	"""
	POST /api/bia/anl/datasources/<id>/test
	Test datasource connection.
	Permission: bia_anl:admin
	"""
	try:
		result = _run(_svc.test_datasource(_tenant(), ds_id))
	except ValueError as e:
		return _err(str(e), 404)
	return _ok(result)


@api_bp.delete("/datasources/<ds_id>")
def delete_datasource(ds_id: str):
	"""
	DELETE /api/bia/anl/datasources/<id>
	Delete a datasource.
	Permission: bia_anl:admin
	"""
	ok = _run(_svc.delete_datasource(_tenant(), ds_id))
	if not ok:
		return _err("Datasource not found", 404)
	return _ok({"deleted": ds_id})


# ── Queries ───────────────────────────────────────────────────────────────────

@api_bp.get("/queries")
def list_queries():
	"""
	GET /api/bia/anl/queries
	List saved queries.
	Permission: bia_anl:query
	"""
	return _ok(_run(_svc.list_queries(_tenant())))


@api_bp.post("/queries")
def create_query():
	"""
	POST /api/bia/anl/queries
	Save a new analytical query.
	Permission: bia_anl:query
	Body: {name, query_type, sql_text, datasource_id, owner_id, ...}
	"""
	body = request.get_json(silent=True) or {}
	required = ["name", "query_type", "sql_text", "datasource_id"]
	missing = [f for f in required if f not in body]
	if missing:
		return _err(f"Missing fields: {missing}", 400)
	try:
		q = _run(_svc.save_query(
			tenant_id=_tenant(),
			name=body["name"],
			query_type=body["query_type"],
			sql_text=body["sql_text"],
			datasource_id=body["datasource_id"],
			owner_id=body.get("owner_id", _user()),
			parameters=body.get("parameters", {}),
			access_level=body.get("access_level", "private"),
			cache_policy=body.get("cache_policy", "session"),
			tags=body.get("tags", []),
			description=body.get("description"),
		))
	except ValueError as e:
		return _err(str(e), 400)
	return _ok(q, 201)


@api_bp.get("/queries/<query_id>")
def get_query(query_id: str):
	"""
	GET /api/bia/anl/queries/<id>
	Get saved query detail.
	Permission: bia_anl:query
	"""
	q = _run(_svc.get_query(_tenant(), query_id))
	if not q:
		return _err("Query not found", 404)
	return _ok(q)


@api_bp.put("/queries/<query_id>")
def update_query(query_id: str):
	"""
	PUT /api/bia/anl/queries/<id>
	Update a saved query.
	Permission: bia_anl:query
	"""
	body = request.get_json(silent=True) or {}
	try:
		q = _run(_svc.update_query(_tenant(), query_id, body))
	except ValueError as e:
		return _err(str(e), 404)
	return _ok(q)


@api_bp.delete("/queries/<query_id>")
def delete_query(query_id: str):
	"""
	DELETE /api/bia/anl/queries/<id>
	Delete a saved query.
	Permission: bia_anl:query
	"""
	ok = _run(_svc.delete_query(_tenant(), query_id))
	if not ok:
		return _err("Query not found", 404)
	return _ok({"deleted": query_id})


@api_bp.post("/queries/<query_id>/execute")
def execute_query(query_id: str):
	"""
	POST /api/bia/anl/queries/<id>/execute
	Execute a saved query with optional parameter overrides.
	Permission: bia_anl:query
	Body: {parameters: {...}}
	"""
	body = request.get_json(silent=True) or {}
	try:
		result = _run(_svc.execute_query(_tenant(), query_id, body.get("parameters", {})))
	except ValueError as e:
		return _err(str(e), 400)
	return _ok(result)


# ── OLAP Cubes ────────────────────────────────────────────────────────────────

@api_bp.get("/cubes")
def list_cubes():
	"""
	GET /api/bia/anl/cubes
	List OLAP cubes.
	Permission: bia_anl:cubes
	"""
	return _ok(_run(_svc.list_cubes(_tenant())))


@api_bp.post("/cubes")
def create_cube():
	"""
	POST /api/bia/anl/cubes
	Create an OLAP cube.
	Permission: bia_anl:cubes
	Body: {name, datasource_id, dimensions, measures, grain_sql, owner_id}
	"""
	body = request.get_json(silent=True) or {}
	required = ["name", "datasource_id", "dimensions", "measures", "grain_sql"]
	missing = [f for f in required if f not in body]
	if missing:
		return _err(f"Missing fields: {missing}", 400)
	try:
		cube = _run(_svc.create_cube(
			tenant_id=_tenant(),
			name=body["name"],
			datasource_id=body["datasource_id"],
			dimensions=body["dimensions"],
			measures=body["measures"],
			grain_sql=body["grain_sql"],
			owner_id=body.get("owner_id", _user()),
			description=body.get("description"),
			tags=body.get("tags", []),
		))
	except ValueError as e:
		return _err(str(e), 400)
	return _ok(cube, 201)


@api_bp.get("/cubes/<cube_id>")
def get_cube(cube_id: str):
	"""
	GET /api/bia/anl/cubes/<id>
	Get cube detail.
	Permission: bia_anl:cubes
	"""
	cube = _run(_svc.get_cube(_tenant(), cube_id))
	if not cube:
		return _err("Cube not found", 404)
	return _ok(cube)


@api_bp.put("/cubes/<cube_id>")
def update_cube(cube_id: str):
	"""
	PUT /api/bia/anl/cubes/<id>
	Update cube metadata.
	Permission: bia_anl:cubes
	"""
	body = request.get_json(silent=True) or {}
	try:
		cube = _run(_svc.update_cube(_tenant(), cube_id, body))
	except ValueError as e:
		return _err(str(e), 404)
	return _ok(cube)


@api_bp.post("/cubes/<cube_id>/refresh")
def refresh_cube(cube_id: str):
	"""
	POST /api/bia/anl/cubes/<id>/refresh
	Trigger cube refresh.
	Permission: bia_anl:cubes
	"""
	try:
		cube = _run(_svc.refresh_cube(_tenant(), cube_id))
	except ValueError as e:
		return _err(str(e), 400)
	return _ok(cube)


@api_bp.post("/cubes/<cube_id>/archive")
def archive_cube(cube_id: str):
	"""
	POST /api/bia/anl/cubes/<id>/archive
	Archive a cube.
	Permission: bia_anl:cubes
	"""
	try:
		cube = _run(_svc.archive_cube(_tenant(), cube_id))
	except ValueError as e:
		return _err(str(e), 404)
	return _ok(cube)


# ── Metrics ───────────────────────────────────────────────────────────────────

@api_bp.get("/metrics")
def list_metrics():
	"""
	GET /api/bia/anl/metrics
	List defined metrics.
	Permission: bia_anl:metrics
	"""
	return _ok(_run(_svc.list_metrics(_tenant())))


@api_bp.post("/metrics")
def create_metric():
	"""
	POST /api/bia/anl/metrics
	Define a calculated metric.
	Permission: bia_anl:metrics
	Body: {name, metric_type, formula, cube_id, owner_id}
	"""
	body = request.get_json(silent=True) or {}
	required = ["name", "metric_type", "formula", "cube_id"]
	missing = [f for f in required if f not in body]
	if missing:
		return _err(f"Missing fields: {missing}", 400)
	try:
		metric = _run(_svc.define_metric(
			tenant_id=_tenant(),
			name=body["name"],
			metric_type=body["metric_type"],
			formula=body["formula"],
			cube_id=body["cube_id"],
			owner_id=body.get("owner_id", _user()),
			unit=body.get("unit"),
			description=body.get("description"),
			tags=body.get("tags", []),
		))
	except ValueError as e:
		return _err(str(e), 400)
	return _ok(metric, 201)


@api_bp.get("/metrics/<metric_id>")
def get_metric(metric_id: str):
	"""
	GET /api/bia/anl/metrics/<id>
	Get metric detail.
	Permission: bia_anl:metrics
	"""
	metric = _run(_svc.get_metric(_tenant(), metric_id))
	if not metric:
		return _err("Metric not found", 404)
	return _ok(metric)


@api_bp.put("/metrics/<metric_id>")
def update_metric(metric_id: str):
	"""
	PUT /api/bia/anl/metrics/<id>
	Update metric definition.
	Permission: bia_anl:metrics
	"""
	body = request.get_json(silent=True) or {}
	try:
		metric = _run(_svc.update_metric(_tenant(), metric_id, body))
	except ValueError as e:
		return _err(str(e), 404)
	return _ok(metric)


@api_bp.delete("/metrics/<metric_id>")
def delete_metric(metric_id: str):
	"""
	DELETE /api/bia/anl/metrics/<id>
	Delete a metric.
	Permission: bia_anl:metrics
	"""
	ok = _run(_svc.delete_metric(_tenant(), metric_id))
	if not ok:
		return _err("Metric not found", 404)
	return _ok({"deleted": metric_id})


# ── Schedules ─────────────────────────────────────────────────────────────────

@api_bp.get("/schedules")
def list_schedules():
	"""
	GET /api/bia/anl/schedules
	List query schedules.
	Permission: bia_anl:schedule
	"""
	return _ok(_run(_svc.list_schedules(_tenant())))


@api_bp.post("/schedules")
def create_schedule():
	"""
	POST /api/bia/anl/schedules
	Schedule a query for recurring execution.
	Permission: bia_anl:schedule
	Body: {query_id, cron_expression, owner_id}
	"""
	body = request.get_json(silent=True) or {}
	required = ["query_id", "cron_expression"]
	missing = [f for f in required if f not in body]
	if missing:
		return _err(f"Missing fields: {missing}", 400)
	try:
		sched = _run(_svc.schedule_query(
			tenant_id=_tenant(),
			query_id=body["query_id"],
			cron_expression=body["cron_expression"],
			owner_id=body.get("owner_id", _user()),
			notification_targets=body.get("notification_targets", []),
		))
	except ValueError as e:
		return _err(str(e), 400)
	return _ok(sched, 201)


# ── Audit & Stats ─────────────────────────────────────────────────────────────

@api_bp.get("/audit")
def get_audit():
	"""
	GET /api/bia/anl/audit
	Get audit events.
	Permission: bia_anl:admin
	"""
	return _ok(_run(_svc.get_audit_events(_tenant())))


@api_bp.get("/stats")
def get_stats():
	"""
	GET /api/bia/anl/stats
	Get dashboard stats.
	Permission: bia_anl:view
	"""
	return _ok(_run(_svc.get_dashboard_stats(_tenant())))
