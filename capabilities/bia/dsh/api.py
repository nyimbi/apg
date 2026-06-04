"""Flask Blueprint REST API for APG Dashboard Management (bia_dsh)."""

from __future__ import annotations

import asyncio
from typing import Any

from flask import Blueprint, jsonify, request

try:
	from .service import DashboardService
	from .capability_contract import CAPABILITY_ID, get_capability_contract
except ImportError:
	from service import DashboardService
	from capability_contract import CAPABILITY_ID, get_capability_contract

api_bp = Blueprint("bia_dsh_api", __name__, url_prefix="/api/bia/dsh")
_svc = DashboardService()


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


@api_bp.get("/contract")
def get_contract():
	return _ok(get_capability_contract(_tenant()))


# ── Dashboards ────────────────────────────────────────────────────────────────

@api_bp.get("/dashboards")
def list_dashboards():
	"""GET /api/bia/dsh/dashboards — list dashboards. Permission: bia_dsh:view"""
	return _ok(_run(_svc.list_dashboards(_tenant())))


@api_bp.post("/dashboards")
def create_dashboard():
	"""POST /api/bia/dsh/dashboards — create dashboard. Permission: bia_dsh:create"""
	body = request.get_json(silent=True) or {}
	missing = [f for f in ["name", "owner_id"] if f not in body]
	if missing:
		return _err(f"Missing fields: {missing}", 400)
	try:
		d = _run(_svc.create_dashboard(
			tenant_id=_tenant(), name=body["name"],
			owner_id=body.get("owner_id", _user()),
			layout_type=body.get("layout_type", "responsive_grid"),
			access_level=body.get("access_level", "private"),
			description=body.get("description"),
			tags=body.get("tags", []),
		))
	except ValueError as e:
		return _err(str(e), 400)
	return _ok(d, 201)


@api_bp.get("/dashboards/<dashboard_id>")
def get_dashboard(dashboard_id: str):
	"""GET /api/bia/dsh/dashboards/<id> — get dashboard. Permission: bia_dsh:view"""
	d = _run(_svc.get_dashboard(_tenant(), dashboard_id))
	if not d:
		return _err("Dashboard not found", 404)
	return _ok(d)


@api_bp.put("/dashboards/<dashboard_id>")
def update_dashboard(dashboard_id: str):
	"""PUT /api/bia/dsh/dashboards/<id> — update dashboard. Permission: bia_dsh:edit"""
	body = request.get_json(silent=True) or {}
	try:
		d = _run(_svc.update_dashboard(_tenant(), dashboard_id, body))
	except ValueError as e:
		return _err(str(e), 404)
	return _ok(d)


@api_bp.delete("/dashboards/<dashboard_id>")
def delete_dashboard(dashboard_id: str):
	"""DELETE /api/bia/dsh/dashboards/<id> — delete dashboard. Permission: bia_dsh:edit"""
	ok = _run(_svc.delete_dashboard(_tenant(), dashboard_id))
	if not ok:
		return _err("Dashboard not found", 404)
	return _ok({"deleted": dashboard_id})


@api_bp.post("/dashboards/<dashboard_id>/publish")
def publish_dashboard(dashboard_id: str):
	"""POST /api/bia/dsh/dashboards/<id>/publish — publish. Permission: bia_dsh:edit"""
	try:
		d = _run(_svc.publish_dashboard(_tenant(), dashboard_id))
	except ValueError as e:
		return _err(str(e), 400)
	return _ok(d)


@api_bp.post("/dashboards/<dashboard_id>/archive")
def archive_dashboard(dashboard_id: str):
	"""POST /api/bia/dsh/dashboards/<id>/archive — archive. Permission: bia_dsh:edit"""
	try:
		d = _run(_svc.archive_dashboard(_tenant(), dashboard_id))
	except ValueError as e:
		return _err(str(e), 404)
	return _ok(d)


# ── Widgets ───────────────────────────────────────────────────────────────────

@api_bp.get("/dashboards/<dashboard_id>/widgets")
def list_widgets(dashboard_id: str):
	"""GET /api/bia/dsh/dashboards/<id>/widgets — list widgets. Permission: bia_dsh:view"""
	return _ok(_run(_svc.list_widgets(_tenant(), dashboard_id)))


@api_bp.post("/dashboards/<dashboard_id>/widgets")
def add_widget(dashboard_id: str):
	"""POST /api/bia/dsh/dashboards/<id>/widgets — add widget. Permission: bia_dsh:edit"""
	body = request.get_json(silent=True) or {}
	missing = [f for f in ["name", "widget_type", "datasource_type", "datasource_id"] if f not in body]
	if missing:
		return _err(f"Missing fields: {missing}", 400)
	try:
		w = _run(_svc.add_widget(
			tenant_id=_tenant(), dashboard_id=dashboard_id,
			name=body["name"], widget_type=body["widget_type"],
			datasource_type=body["datasource_type"], datasource_id=body["datasource_id"],
			owner_id=body.get("owner_id", _user()),
			config=body.get("config", {}), position=body.get("position", {}),
			size=body.get("size", {}), refresh_interval=body.get("refresh_interval", "manual"),
		))
	except ValueError as e:
		return _err(str(e), 400)
	return _ok(w, 201)


@api_bp.get("/widgets/<widget_id>")
def get_widget(widget_id: str):
	"""GET /api/bia/dsh/widgets/<id> — get widget. Permission: bia_dsh:view"""
	w = _run(_svc.get_widget(_tenant(), widget_id))
	if not w:
		return _err("Widget not found", 404)
	return _ok(w)


@api_bp.put("/widgets/<widget_id>")
def update_widget(widget_id: str):
	"""PUT /api/bia/dsh/widgets/<id> — update widget. Permission: bia_dsh:edit"""
	body = request.get_json(silent=True) or {}
	try:
		w = _run(_svc.update_widget(_tenant(), widget_id, body))
	except ValueError as e:
		return _err(str(e), 404)
	return _ok(w)


@api_bp.delete("/widgets/<widget_id>")
def remove_widget(widget_id: str):
	"""DELETE /api/bia/dsh/widgets/<id> — remove widget. Permission: bia_dsh:edit"""
	ok = _run(_svc.remove_widget(_tenant(), widget_id))
	if not ok:
		return _err("Widget not found", 404)
	return _ok({"deleted": widget_id})


# ── Snapshots ─────────────────────────────────────────────────────────────────

@api_bp.get("/dashboards/<dashboard_id>/snapshots")
def list_snapshots(dashboard_id: str):
	"""GET /api/bia/dsh/dashboards/<id>/snapshots. Permission: bia_dsh:snapshots"""
	return _ok(_run(_svc.list_snapshots(_tenant(), dashboard_id)))


@api_bp.post("/dashboards/<dashboard_id>/snapshots")
def take_snapshot(dashboard_id: str):
	"""POST /api/bia/dsh/dashboards/<id>/snapshots. Permission: bia_dsh:snapshots"""
	body = request.get_json(silent=True) or {}
	try:
		snap = _run(_svc.take_snapshot(
			tenant_id=_tenant(), dashboard_id=dashboard_id,
			format=body.get("format", "png"),
			requested_by=body.get("requested_by", _user()),
			label=body.get("label"),
		))
	except ValueError as e:
		return _err(str(e), 400)
	return _ok(snap, 201)


@api_bp.post("/dashboards/<dashboard_id>/snapshots/schedule")
def schedule_snapshot(dashboard_id: str):
	"""POST /api/bia/dsh/dashboards/<id>/snapshots/schedule. Permission: bia_dsh:snapshots"""
	body = request.get_json(silent=True) or {}
	missing = [f for f in ["cron_expression"] if f not in body]
	if missing:
		return _err(f"Missing fields: {missing}", 400)
	try:
		sched = _run(_svc.schedule_snapshot(
			tenant_id=_tenant(), dashboard_id=dashboard_id,
			cron_expression=body["cron_expression"],
			format=body.get("format", "png"),
			owner_id=body.get("owner_id", _user()),
		))
	except ValueError as e:
		return _err(str(e), 400)
	return _ok(sched, 201)


# ── Filters ───────────────────────────────────────────────────────────────────

@api_bp.get("/dashboards/<dashboard_id>/filters")
def list_filters(dashboard_id: str):
	"""GET /api/bia/dsh/dashboards/<id>/filters. Permission: bia_dsh:view"""
	return _ok(_run(_svc.list_filters(_tenant(), dashboard_id)))


@api_bp.post("/dashboards/<dashboard_id>/filters")
def add_filter(dashboard_id: str):
	"""POST /api/bia/dsh/dashboards/<id>/filters. Permission: bia_dsh:edit"""
	body = request.get_json(silent=True) or {}
	missing = [f for f in ["name", "filter_type", "target_field"] if f not in body]
	if missing:
		return _err(f"Missing fields: {missing}", 400)
	try:
		f = _run(_svc.add_filter(
			tenant_id=_tenant(), dashboard_id=dashboard_id,
			name=body["name"], filter_type=body["filter_type"],
			target_field=body["target_field"],
			owner_id=body.get("owner_id", _user()),
			config=body.get("config", {}),
		))
	except ValueError as e:
		return _err(str(e), 400)
	return _ok(f, 201)


@api_bp.delete("/filters/<filter_id>")
def remove_filter(filter_id: str):
	"""DELETE /api/bia/dsh/filters/<id>. Permission: bia_dsh:edit"""
	ok = _run(_svc.remove_filter(_tenant(), filter_id))
	if not ok:
		return _err("Filter not found", 404)
	return _ok({"deleted": filter_id})


# ── Stats ─────────────────────────────────────────────────────────────────────

@api_bp.get("/stats")
def get_stats():
	return _ok(_run(_svc.get_dashboard_stats(_tenant())))


@api_bp.get("/audit")
def get_audit():
	return _ok(_run(_svc.get_audit_events(_tenant())))
