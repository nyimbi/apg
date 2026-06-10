"""Matter Management — Flask Blueprint REST API."""
from __future__ import annotations

import logging
from typing import Any

from flask import Blueprint, jsonify, request

from .service import MatterManagementService

_log = logging.getLogger(__name__)

bp = Blueprint("leg_mat", __name__, url_prefix="/api/legal/mat")
_svc: MatterManagementService | None = None


def get_service() -> MatterManagementService:
	global _svc
	if _svc is None:
		_svc = MatterManagementService()
	return _svc


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
	return jsonify(_run(get_service().health_check()))


@bp.get("/describe")
def describe():
	return jsonify(_run(get_service().describe()))


# ── Matters ──────────────────────────────────────────────────────────────────

@bp.get("/matters")
def list_matters():
	svc = get_service()
	tenant = request.args.get("tenant_id", "default")
	try:
		items = _run(svc.list_matters(
			tenant_id=tenant,
			status=request.args.get("status"),
			matter_type=request.args.get("matter_type"),
			client_id=request.args.get("client_id"),
			lead_attorney_id=request.args.get("lead_attorney_id"),
			priority=request.args.get("priority"),
		))
		return jsonify({"items": items, "total": len(items)})
	except Exception as exc:
		_log.error("list_matters error: %s", exc)
		return jsonify({"error": str(exc)}), 400


@bp.get("/matters/<matter_id>")
def get_matter(matter_id: str):
	svc = get_service()
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(svc.get_matter(tenant, matter_id)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		_log.error("get_matter error: %s", exc)
		return jsonify({"error": str(exc)}), 400


@bp.post("/matters")
def create_matter():
	svc = get_service()
	body = request.get_json(force=True) or {}
	try:
		result = _run(svc.create_matter(**body))
		return jsonify(result), 201
	except Exception as exc:
		_log.error("create_matter error: %s", exc)
		return jsonify({"error": str(exc)}), 400


@bp.put("/matters/<matter_id>")
def update_matter(matter_id: str):
	svc = get_service()
	body = request.get_json(force=True) or {}
	tenant = body.pop("tenant_id", "default")
	try:
		return jsonify(_run(svc.update_matter(tenant, matter_id, **body)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		_log.error("update_matter error: %s", exc)
		return jsonify({"error": str(exc)}), 400


@bp.delete("/matters/<matter_id>")
def delete_matter(matter_id: str):
	svc = get_service()
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(svc.delete_matter(tenant, matter_id)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		_log.error("delete_matter error: %s", exc)
		return jsonify({"error": str(exc)}), 400


@bp.post("/matters/<matter_id>/close")
def close_matter(matter_id: str):
	body = request.get_json(force=True) or {}
	tenant = body.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().close_matter(tenant, matter_id, body.get("closed_by", ""), body.get("notes", ""))))
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


# ── Tasks ────────────────────────────────────────────────────────────────────

@bp.get("/tasks")
def list_tasks():
	svc = get_service()
	tenant = request.args.get("tenant_id", "default")
	try:
		items = _run(svc.list_tasks(
			tenant_id=tenant,
			matter_id=request.args.get("matter_id"),
			assigned_to_id=request.args.get("assigned_to_id"),
			status=request.args.get("status"),
		))
		return jsonify({"items": items, "total": len(items)})
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/tasks/<task_id>")
def get_task(task_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().get_task(tenant, task_id)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/tasks")
def create_task():
	body = request.get_json(force=True) or {}
	try:
		return jsonify(_run(get_service().create_task(**body))), 201
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.put("/tasks/<task_id>")
def update_task(task_id: str):
	body = request.get_json(force=True) or {}
	tenant = body.pop("tenant_id", "default")
	try:
		return jsonify(_run(get_service().update_task(tenant, task_id, **body)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.delete("/tasks/<task_id>")
def delete_task(task_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().delete_task(tenant, task_id)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


# ── Deadlines ────────────────────────────────────────────────────────────────

@bp.get("/deadlines")
def list_deadlines():
	tenant = request.args.get("tenant_id", "default")
	try:
		items = _run(get_service().list_deadlines(
			tenant_id=tenant,
			matter_id=request.args.get("matter_id"),
			overdue_only=request.args.get("overdue_only", "").lower() == "true",
		))
		return jsonify({"items": items, "total": len(items)})
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/deadlines/<deadline_id>")
def get_deadline(deadline_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().get_deadline(tenant, deadline_id)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/deadlines")
def create_deadline():
	body = request.get_json(force=True) or {}
	try:
		return jsonify(_run(get_service().create_deadline(**body))), 201
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.put("/deadlines/<deadline_id>")
def update_deadline(deadline_id: str):
	body = request.get_json(force=True) or {}
	tenant = body.pop("tenant_id", "default")
	try:
		return jsonify(_run(get_service().update_deadline(tenant, deadline_id, **body)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.delete("/deadlines/<deadline_id>")
def delete_deadline(deadline_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().delete_deadline(tenant, deadline_id)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


# ── Docket ───────────────────────────────────────────────────────────────────

@bp.get("/docket")
def list_docket_entries():
	tenant = request.args.get("tenant_id", "default")
	try:
		items = _run(get_service().list_docket_entries(
			tenant_id=tenant,
			matter_id=request.args.get("matter_id"),
		))
		return jsonify({"items": items, "total": len(items)})
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/docket")
def create_docket_entry():
	body = request.get_json(force=True) or {}
	try:
		return jsonify(_run(get_service().create_docket_entry(**body))), 201
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


# ── Analytics ────────────────────────────────────────────────────────────────

@bp.get("/dashboard")
def dashboard():
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().matter_dashboard(tenant)))
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/audit")
def audit_events():
	tenant = request.args.get("tenant_id", "default")
	limit = int(request.args.get("limit", 100))
	try:
		return jsonify(_run(get_service().get_audit_events(tenant, limit)))
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400
