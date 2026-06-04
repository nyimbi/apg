"""Flask Blueprint views for APG Dashboard Management (bia_dsh)."""

from __future__ import annotations

import asyncio
from functools import wraps
from flask import Blueprint, abort, g, jsonify, request

try:
	from .capability_contract import CAPABILITY_ID, evaluate_capability_rules
	from .service import DashboardService
except ImportError:
	from capability_contract import CAPABILITY_ID, evaluate_capability_rules
	from service import DashboardService

dsh_bp = Blueprint("bia_dsh", __name__, url_prefix="/bia/dsh")
_svc = DashboardService()


def _require_permission(perm: str):
	def decorator(fn):
		@wraps(fn)
		def wrapper(*args, **kwargs):
			tenant_id = request.headers.get("X-Tenant-ID", "default")
			user_id = request.headers.get("X-User-ID", "anonymous")
			perms = request.headers.get("X-Permissions", "")
			if perm not in perms and "bia_dsh:admin" not in perms:
				abort(403, description=f"Permission required: {perm}")
			g.tenant_id = tenant_id
			g.user_id = user_id
			return fn(*args, **kwargs)
		return wrapper
	return decorator


def _run(coro):
	loop = asyncio.get_event_loop()
	return loop.run_until_complete(coro)


@dsh_bp.get("/")
@_require_permission("bia_dsh:view")
def home():
	stats = _run(_svc.get_dashboard_stats(g.tenant_id))
	return jsonify({"view": "dashboard_home", "capability": CAPABILITY_ID, "stats": stats})


@dsh_bp.get("/gallery")
@_require_permission("bia_dsh:view")
def gallery():
	dashboards = _run(_svc.list_dashboards(g.tenant_id))
	return jsonify({"view": "dashboard_gallery", "dashboards": dashboards, "total": len(dashboards)})


@dsh_bp.get("/<dashboard_id>/view")
@_require_permission("bia_dsh:view")
def view_dashboard(dashboard_id: str):
	d = _run(_svc.get_dashboard(g.tenant_id, dashboard_id))
	if not d:
		abort(404, description="Dashboard not found")
	widgets = _run(_svc.list_widgets(g.tenant_id, dashboard_id))
	filters = _run(_svc.list_filters(g.tenant_id, dashboard_id))
	return jsonify({"view": "dashboard_view", "dashboard": d, "widgets": widgets, "filters": filters})


@dsh_bp.get("/<dashboard_id>/build")
@_require_permission("bia_dsh:edit")
def builder(dashboard_id: str):
	d = _run(_svc.get_dashboard(g.tenant_id, dashboard_id))
	if not d:
		abort(404, description="Dashboard not found")
	return jsonify({"view": "dashboard_builder", "dashboard": d})


@dsh_bp.get("/widgets")
@_require_permission("bia_dsh:view")
def widget_library():
	from .capability_contract import SUPPORTED_WIDGET_TYPES
	return jsonify({"view": "widget_library", "widget_types": SUPPORTED_WIDGET_TYPES})


@dsh_bp.get("/widgets/<widget_id>")
@_require_permission("bia_dsh:view")
def widget_detail(widget_id: str):
	w = _run(_svc.get_widget(g.tenant_id, widget_id))
	if not w:
		abort(404, description="Widget not found")
	return jsonify({"view": "widget_detail", "widget": w})


@dsh_bp.get("/snapshots")
@_require_permission("bia_dsh:snapshots")
def snapshots():
	snaps = _run(_svc.list_snapshots(g.tenant_id))
	return jsonify({"view": "snapshots", "snapshots": snaps, "total": len(snaps)})


@dsh_bp.get("/<dashboard_id>/filters")
@_require_permission("bia_dsh:edit")
def filter_manager(dashboard_id: str):
	filters = _run(_svc.list_filters(g.tenant_id, dashboard_id))
	return jsonify({"view": "filter_manager", "dashboard_id": dashboard_id, "filters": filters})


@dsh_bp.get("/audit")
@_require_permission("bia_dsh:admin")
def audit_log():
	events = _run(_svc.get_audit_events(g.tenant_id))
	return jsonify({"view": "audit_log", "events": events})


@dsh_bp.get("/settings")
@_require_permission("bia_dsh:admin")
def settings():
	from .capability_contract import get_capability_contract
	return jsonify({"view": "settings", "config": get_capability_contract(g.tenant_id)["configuration"]})
