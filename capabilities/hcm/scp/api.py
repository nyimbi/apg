"""Flask Blueprint REST API for Succession Planning."""
from __future__ import annotations

import logging
from typing import Any

from flask import Blueprint, jsonify, request

from .service import SCPService

_log = logging.getLogger(__name__)

bp = Blueprint("hcm_scp", __name__, url_prefix="/api/hcm/scp")
_svc = SCPService()


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


# ── Talent Pools ──────────────────────────────────────────────────────────────

@bp.get("/talent-pools")
def list_talent_pools():
	tenant_id = request.args.get("tenant_id", "default")
	try:
		items = _run(_svc.list_talent_pools(tenant_id, status=request.args.get("status")))
		return jsonify({"items": items, "total": len(items)})
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/talent-pools/<pool_id>")
def get_talent_pool(pool_id: str):
	tenant_id = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.get_talent_pool(tenant_id, pool_id)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/talent-pools")
def create_talent_pool():
	data = request.get_json(force=True) or {}
	try:
		record = _run(_svc.create_talent_pool(
			tenant_id=data.get("tenant_id", "default"),
			name=data["name"],
			description=data.get("description"),
			target_roles=data.get("target_roles"),
			min_readiness_level=data.get("min_readiness_level", "developing"),
		))
		return jsonify(record), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.put("/talent-pools/<pool_id>")
def update_talent_pool(pool_id: str):
	data = request.get_json(force=True) or {}
	tenant_id = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.update_talent_pool(tenant_id, pool_id, **data)))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.delete("/talent-pools/<pool_id>")
def delete_talent_pool(pool_id: str):
	tenant_id = request.args.get("tenant_id", "default")
	try:
		_run(_svc.delete_talent_pool(tenant_id, pool_id))
		return jsonify({"deleted": True})
	except (KeyError, PermissionError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/talent-pools/<pool_id>/members")
def list_talent_pool_members(pool_id: str):
	tenant_id = request.args.get("tenant_id", "default")
	try:
		items = _run(_svc.list_talent_pool_members(tenant_id, pool_id))
		return jsonify({"items": items, "total": len(items)})
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/talent-pools/<pool_id>/members")
def add_to_talent_pool(pool_id: str):
	data = request.get_json(force=True) or {}
	try:
		record = _run(_svc.add_to_talent_pool(
			tenant_id=data.get("tenant_id", "default"),
			pool_id=pool_id,
			employee_id=data["employee_id"],
			readiness_level=data["readiness_level"],
			added_by=data["added_by"],
			notes=data.get("notes"),
		))
		return jsonify(record), 201
	except (KeyError, ValueError, PermissionError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.delete("/talent-pools/<pool_id>/members/<employee_id>")
def remove_from_talent_pool(pool_id: str, employee_id: str):
	tenant_id = request.args.get("tenant_id", "default")
	try:
		_run(_svc.remove_from_talent_pool(tenant_id, pool_id, employee_id))
		return jsonify({"removed": True})
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


# ── Readiness Assessments ─────────────────────────────────────────────────────

@bp.get("/readiness-assessments")
def list_readiness_assessments():
	tenant_id = request.args.get("tenant_id", "default")
	try:
		items = _run(_svc.list_readiness_assessments(
			tenant_id,
			employee_id=request.args.get("employee_id"),
			target_role_id=request.args.get("target_role_id"),
			readiness_level=request.args.get("readiness_level"),
		))
		return jsonify({"items": items, "total": len(items)})
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/readiness-assessments/<assessment_id>")
def get_readiness_assessment(assessment_id: str):
	tenant_id = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.get_readiness_assessment(tenant_id, assessment_id)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/readiness-assessments")
def create_readiness_assessment():
	data = request.get_json(force=True) or {}
	try:
		record = _run(_svc.create_readiness_assessment(
			tenant_id=data.get("tenant_id", "default"),
			employee_id=data["employee_id"],
			target_role_id=data["target_role_id"],
			readiness_level=data["readiness_level"],
			performance_rating=float(data["performance_rating"]),
			potential_rating=float(data["potential_rating"]),
			assessed_by=data["assessed_by"],
			development_needs=data.get("development_needs"),
			risks=data.get("risks"),
			notes=data.get("notes"),
		))
		return jsonify(record), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.put("/readiness-assessments/<assessment_id>")
def update_readiness_assessment(assessment_id: str):
	data = request.get_json(force=True) or {}
	tenant_id = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.update_readiness_assessment(tenant_id, assessment_id, **data)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 400


@bp.delete("/readiness-assessments/<assessment_id>")
def delete_readiness_assessment(assessment_id: str):
	tenant_id = request.args.get("tenant_id", "default")
	try:
		_run(_svc.delete_readiness_assessment(tenant_id, assessment_id))
		return jsonify({"deleted": True})
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


# ── Nine-Box Grid ─────────────────────────────────────────────────────────────

@bp.get("/nine-box")
def list_nine_box_entries():
	tenant_id = request.args.get("tenant_id", "default")
	try:
		items = _run(_svc.list_nine_box_entries(
			tenant_id,
			review_cycle=request.args.get("review_cycle"),
			employee_id=request.args.get("employee_id"),
		))
		return jsonify({"items": items, "total": len(items)})
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/nine-box/grid")
def nine_box_grid():
	tenant_id = request.args.get("tenant_id", "default")
	review_cycle = request.args.get("review_cycle", "")
	if not review_cycle:
		return jsonify({"error": "review_cycle query param required"}), 400
	try:
		return jsonify(_run(_svc.get_nine_box_grid(tenant_id, review_cycle)))
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/nine-box")
def place_on_nine_box():
	data = request.get_json(force=True) or {}
	try:
		record = _run(_svc.place_on_nine_box(
			tenant_id=data.get("tenant_id", "default"),
			employee_id=data["employee_id"],
			performance_axis=float(data["performance_axis"]),
			potential_axis=float(data["potential_axis"]),
			review_cycle=data["review_cycle"],
			reviewer_id=data["reviewer_id"],
			label=data.get("label"),
			notes=data.get("notes"),
		))
		return jsonify(record), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


# ── Succession Scenarios ──────────────────────────────────────────────────────

@bp.get("/scenarios")
def list_succession_scenarios():
	tenant_id = request.args.get("tenant_id", "default")
	try:
		items = _run(_svc.list_succession_scenarios(
			tenant_id,
			role_id=request.args.get("role_id"),
			status=request.args.get("status"),
		))
		return jsonify({"items": items, "total": len(items)})
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/scenarios/<scenario_id>")
def get_succession_scenario(scenario_id: str):
	tenant_id = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.get_succession_scenario(tenant_id, scenario_id)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/scenarios")
def create_succession_scenario():
	data = request.get_json(force=True) or {}
	try:
		record = _run(_svc.create_succession_scenario(
			tenant_id=data.get("tenant_id", "default"),
			role_id=data["role_id"],
			role_title=data["role_title"],
			incumbent_employee_id=data.get("incumbent_employee_id"),
			scenario_type=data.get("scenario_type", "planned"),
			successors=data.get("successors"),
			notes=data.get("notes"),
		))
		return jsonify(record), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.put("/scenarios/<scenario_id>")
def update_succession_scenario(scenario_id: str):
	data = request.get_json(force=True) or {}
	tenant_id = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.update_succession_scenario(tenant_id, scenario_id, **data)))
	except (KeyError, PermissionError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.put("/scenarios/<scenario_id>/activate")
def activate_succession_scenario(scenario_id: str):
	data = request.get_json(force=True) or {}
	try:
		return jsonify(_run(_svc.activate_succession_scenario(data.get("tenant_id", "default"), scenario_id, data["approved_by"])))
	except (KeyError, PermissionError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.delete("/scenarios/<scenario_id>")
def delete_succession_scenario(scenario_id: str):
	tenant_id = request.args.get("tenant_id", "default")
	try:
		_run(_svc.delete_succession_scenario(tenant_id, scenario_id))
		return jsonify({"deleted": True})
	except (KeyError, PermissionError) as exc:
		return jsonify({"error": str(exc)}), 400


# ── Critical Roles ────────────────────────────────────────────────────────────

@bp.get("/critical-roles")
def list_critical_roles():
	tenant_id = request.args.get("tenant_id", "default")
	try:
		items = _run(_svc.list_critical_roles(
			tenant_id,
			impact_if_vacant=request.args.get("impact_if_vacant"),
			status=request.args.get("status"),
		))
		return jsonify({"items": items, "total": len(items)})
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/critical-roles/<role_entry_id>")
def get_critical_role(role_entry_id: str):
	tenant_id = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.get_critical_role(tenant_id, role_entry_id)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/critical-roles")
def identify_critical_role():
	data = request.get_json(force=True) or {}
	try:
		record = _run(_svc.identify_critical_role(
			tenant_id=data.get("tenant_id", "default"),
			role_id=data["role_id"],
			role_title=data["role_title"],
			rationale=data["rationale"],
			impact_if_vacant=data["impact_if_vacant"],
			identified_by=data["identified_by"],
			time_to_fill_estimate_days=int(data.get("time_to_fill_estimate_days", 90)),
		))
		return jsonify(record), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.put("/critical-roles/<role_entry_id>")
def update_critical_role(role_entry_id: str):
	data = request.get_json(force=True) or {}
	tenant_id = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.update_critical_role(tenant_id, role_entry_id, **data)))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.delete("/critical-roles/<role_entry_id>")
def delete_critical_role(role_entry_id: str):
	tenant_id = request.args.get("tenant_id", "default")
	try:
		_run(_svc.delete_critical_role(tenant_id, role_entry_id))
		return jsonify({"deleted": True})
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


# ── Reports & Dashboard ───────────────────────────────────────────────────────

@bp.get("/coverage-report")
def succession_coverage_report():
	tenant_id = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.succession_coverage_report(tenant_id)))
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/readiness-report")
def talent_pool_readiness_report():
	tenant_id = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.talent_pool_readiness_report(tenant_id)))
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
