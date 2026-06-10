"""Flask Blueprint REST API for Organizational Management."""
from __future__ import annotations

import logging
from typing import Any

from flask import Blueprint, jsonify, request

from .service import ORGService

_log = logging.getLogger(__name__)

bp = Blueprint("hcm_org", __name__, url_prefix="/api/hcm/org")
_svc = ORGService()


def _run(coro: Any) -> Any:
	import asyncio
	try:
		loop = asyncio.get_event_loop()
	except RuntimeError:
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
	return loop.run_until_complete(coro)


@bp.get("/health")
def health():
	return jsonify(_run(_svc.health_check()))


@bp.get("/describe")
def describe():
	return jsonify(_run(_svc.describe()))


# ── Org Units ─────────────────────────────────────────────────────────────────

@bp.get("/units")
def list_org_units():
	tenant_id = request.args.get("tenant_id", "default")
	try:
		items = _run(_svc.list_org_units(
			tenant_id,
			unit_type=request.args.get("unit_type"),
			parent_unit_id=request.args.get("parent_unit_id"),
			status=request.args.get("status"),
		))
		return jsonify({"items": items, "total": len(items)})
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/units/<unit_id>")
def get_org_unit(unit_id: str):
	tenant_id = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.get_org_unit(tenant_id, unit_id)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/units")
def create_org_unit():
	data = request.get_json(force=True) or {}
	try:
		record = _run(_svc.create_org_unit(
			tenant_id=data.get("tenant_id", "default"),
			name=data["name"],
			code=data["code"],
			unit_type=data["unit_type"],
			parent_unit_id=data.get("parent_unit_id"),
			manager_employee_id=data.get("manager_employee_id"),
			cost_centre=data.get("cost_centre"),
			location=data.get("location"),
			description=data.get("description"),
		))
		return jsonify(record), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.put("/units/<unit_id>")
def update_org_unit(unit_id: str):
	data = request.get_json(force=True) or {}
	tenant_id = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.update_org_unit(tenant_id, unit_id, **data)))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.put("/units/<unit_id>/move")
def move_org_unit(unit_id: str):
	data = request.get_json(force=True) or {}
	try:
		return jsonify(_run(_svc.move_org_unit(data.get("tenant_id", "default"), unit_id, data.get("new_parent_id"))))
	except (KeyError, ValueError, PermissionError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.delete("/units/<unit_id>")
def delete_org_unit(unit_id: str):
	tenant_id = request.args.get("tenant_id", "default")
	try:
		_run(_svc.delete_org_unit(tenant_id, unit_id))
		return jsonify({"deleted": True})
	except (KeyError, PermissionError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/chart")
def org_chart():
	tenant_id = request.args.get("tenant_id", "default")
	root = request.args.get("root_unit_id")
	try:
		items = _run(_svc.get_org_chart(tenant_id, root_unit_id=root))
		return jsonify({"items": items, "total": len(items)})
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


# ── Positions ─────────────────────────────────────────────────────────────────

@bp.get("/positions")
def list_positions():
	tenant_id = request.args.get("tenant_id", "default")
	try:
		items = _run(_svc.list_positions(
			tenant_id,
			org_unit_id=request.args.get("org_unit_id"),
			status=request.args.get("status"),
		))
		return jsonify({"items": items, "total": len(items)})
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/positions/<position_id>")
def get_position(position_id: str):
	tenant_id = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.get_position(tenant_id, position_id)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/positions")
def create_position():
	data = request.get_json(force=True) or {}
	try:
		record = _run(_svc.create_position(
			tenant_id=data.get("tenant_id", "default"),
			title=data["title"],
			code=data["code"],
			org_unit_id=data["org_unit_id"],
			job_grade=data.get("job_grade"),
			reports_to_position_id=data.get("reports_to_position_id"),
			fte_count=float(data.get("fte_count", 1.0)),
			is_critical=bool(data.get("is_critical", False)),
			location=data.get("location"),
			description=data.get("description"),
		))
		return jsonify(record), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.put("/positions/<position_id>")
def update_position(position_id: str):
	data = request.get_json(force=True) or {}
	tenant_id = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.update_position(tenant_id, position_id, **data)))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.put("/positions/<position_id>/assign")
def assign_employee(position_id: str):
	data = request.get_json(force=True) or {}
	try:
		return jsonify(_run(_svc.assign_employee_to_position(data.get("tenant_id", "default"), position_id, data["employee_id"])))
	except (KeyError, PermissionError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.delete("/positions/<position_id>")
def delete_position(position_id: str):
	tenant_id = request.args.get("tenant_id", "default")
	try:
		_run(_svc.delete_position(tenant_id, position_id))
		return jsonify({"deleted": True})
	except (KeyError, PermissionError) as exc:
		return jsonify({"error": str(exc)}), 400


# ── Reporting Lines ───────────────────────────────────────────────────────────

@bp.get("/reporting-lines")
def list_reporting_lines():
	tenant_id = request.args.get("tenant_id", "default")
	try:
		items = _run(_svc.list_reporting_lines(
			tenant_id,
			employee_id=request.args.get("employee_id"),
			manager_employee_id=request.args.get("manager_employee_id"),
		))
		return jsonify({"items": items, "total": len(items)})
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/reporting-lines")
def create_reporting_line():
	data = request.get_json(force=True) or {}
	try:
		record = _run(_svc.create_reporting_line(
			tenant_id=data.get("tenant_id", "default"),
			employee_id=data["employee_id"],
			manager_employee_id=data["manager_employee_id"],
			effective_date=data["effective_date"],
			line_type=data.get("line_type", "direct"),
			end_date=data.get("end_date"),
		))
		return jsonify(record), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


# ── Restructuring ─────────────────────────────────────────────────────────────

@bp.get("/restructurings")
def list_restructurings():
	tenant_id = request.args.get("tenant_id", "default")
	try:
		items = _run(_svc.list_restructurings(tenant_id, status=request.args.get("status")))
		return jsonify({"items": items, "total": len(items)})
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/restructurings")
def create_restructuring():
	data = request.get_json(force=True) or {}
	try:
		record = _run(_svc.create_restructuring(
			tenant_id=data.get("tenant_id", "default"),
			name=data["name"],
			description=data["description"],
			effective_date=data["effective_date"],
			initiated_by=data["initiated_by"],
			units_affected=data.get("units_affected"),
			positions_affected=data.get("positions_affected"),
		))
		return jsonify(record), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.put("/restructurings/<restructuring_id>")
def update_restructuring(restructuring_id: str):
	data = request.get_json(force=True) or {}
	tenant_id = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.update_restructuring(tenant_id, restructuring_id, **data)))
	except (KeyError, PermissionError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.delete("/restructurings/<restructuring_id>")
def delete_restructuring(restructuring_id: str):
	tenant_id = request.args.get("tenant_id", "default")
	try:
		_run(_svc.delete_restructuring(tenant_id, restructuring_id))
		return jsonify({"deleted": True})
	except (KeyError, PermissionError) as exc:
		return jsonify({"error": str(exc)}), 400


# ── Analytics ─────────────────────────────────────────────────────────────────

@bp.get("/analytics")
def org_analytics():
	tenant_id = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.org_analytics(tenant_id)))
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/dashboard")
def dashboard():
	tenant_id = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.dashboard_summary(tenant_id)))
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/audit-events")
def audit_events():
	tenant_id = request.args.get("tenant_id", "default")
	try:
		events = _run(_svc.get_audit_events(tenant_id))
		return jsonify({"items": events, "total": len(events)})
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400
