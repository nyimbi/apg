"""Flask Blueprint REST API for Professional Development."""
from __future__ import annotations

import logging
from typing import Any

from flask import Blueprint, jsonify, request

from .service import PROService

_log = logging.getLogger(__name__)

bp = Blueprint("hcm_pro", __name__, url_prefix="/api/hcm/pro")
_svc = PROService()


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


# ── Development Plans ─────────────────────────────────────────────────────────

@bp.get("/development-plans")
def list_development_plans():
	tenant_id = request.args.get("tenant_id", "default")
	try:
		items = _run(_svc.list_development_plans(
			tenant_id,
			employee_id=request.args.get("employee_id"),
			status=request.args.get("status"),
			plan_year=int(request.args["plan_year"]) if request.args.get("plan_year") else None,
		))
		return jsonify({"items": items, "total": len(items)})
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/development-plans/<plan_id>")
def get_development_plan(plan_id: str):
	tenant_id = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.get_development_plan(tenant_id, plan_id)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/development-plans")
def create_development_plan():
	data = request.get_json(force=True) or {}
	try:
		record = _run(_svc.create_development_plan(
			tenant_id=data.get("tenant_id", "default"),
			employee_id=data["employee_id"],
			plan_year=int(data["plan_year"]),
			objectives=data.get("objectives"),
			focus_areas=data.get("focus_areas"),
			target_role_id=data.get("target_role_id"),
			reviewed_by=data.get("reviewed_by"),
		))
		return jsonify(record), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.put("/development-plans/<plan_id>")
def update_development_plan(plan_id: str):
	data = request.get_json(force=True) or {}
	tenant_id = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.update_development_plan(tenant_id, plan_id, **data)))
	except (KeyError, PermissionError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.put("/development-plans/<plan_id>/activate")
def activate_development_plan(plan_id: str):
	data = request.get_json(force=True) or {}
	try:
		return jsonify(_run(_svc.activate_development_plan(data.get("tenant_id", "default"), plan_id, data["reviewed_by"])))
	except (KeyError, PermissionError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.put("/development-plans/<plan_id>/progress")
def update_plan_progress(plan_id: str):
	data = request.get_json(force=True) or {}
	try:
		return jsonify(_run(_svc.update_plan_progress(data.get("tenant_id", "default"), plan_id, float(data["progress_pct"]))))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.delete("/development-plans/<plan_id>")
def delete_development_plan(plan_id: str):
	tenant_id = request.args.get("tenant_id", "default")
	try:
		_run(_svc.delete_development_plan(tenant_id, plan_id))
		return jsonify({"deleted": True})
	except (KeyError, PermissionError) as exc:
		return jsonify({"error": str(exc)}), 400


# ── Skills ────────────────────────────────────────────────────────────────────

@bp.get("/skills")
def list_skills():
	tenant_id = request.args.get("tenant_id", "default")
	try:
		items = _run(_svc.list_skills(tenant_id, category=request.args.get("category")))
		return jsonify({"items": items, "total": len(items)})
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/skills")
def create_skill():
	data = request.get_json(force=True) or {}
	try:
		record = _run(_svc.create_skill(
			tenant_id=data.get("tenant_id", "default"),
			name=data["name"],
			category=data["category"],
			description=data.get("description"),
		))
		return jsonify(record), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.delete("/skills/<skill_id>")
def delete_skill(skill_id: str):
	tenant_id = request.args.get("tenant_id", "default")
	try:
		_run(_svc.delete_skill(tenant_id, skill_id))
		return jsonify({"deleted": True})
	except (KeyError, PermissionError) as exc:
		return jsonify({"error": str(exc)}), 400


# ── Skill Assessments ─────────────────────────────────────────────────────────

@bp.get("/skill-assessments")
def list_skill_assessments():
	tenant_id = request.args.get("tenant_id", "default")
	try:
		items = _run(_svc.list_skill_assessments(
			tenant_id,
			employee_id=request.args.get("employee_id"),
			gap_only=request.args.get("gap_only", "false").lower() == "true",
		))
		return jsonify({"items": items, "total": len(items)})
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/skill-assessments")
def assess_skill():
	data = request.get_json(force=True) or {}
	try:
		record = _run(_svc.assess_skill(
			tenant_id=data.get("tenant_id", "default"),
			employee_id=data["employee_id"],
			skill_id=data["skill_id"],
			current_level=data["current_level"],
			target_level=data["target_level"],
			assessed_by=data.get("assessed_by"),
			evidence=data.get("evidence"),
		))
		return jsonify(record), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/skill-gap-report/<employee_id>")
def skill_gap_report(employee_id: str):
	tenant_id = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.get_skill_gap_report(tenant_id, employee_id)))
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


# ── Mentoring ─────────────────────────────────────────────────────────────────

@bp.get("/mentoring-programmes")
def list_mentoring_programmes():
	tenant_id = request.args.get("tenant_id", "default")
	try:
		items = _run(_svc.list_mentoring_programmes(
			tenant_id,
			employee_id=request.args.get("employee_id"),
			role=request.args.get("role"),
			status=request.args.get("status"),
		))
		return jsonify({"items": items, "total": len(items)})
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/mentoring-programmes")
def create_mentoring_programme():
	data = request.get_json(force=True) or {}
	try:
		record = _run(_svc.create_mentoring_programme(
			tenant_id=data.get("tenant_id", "default"),
			mentee_employee_id=data["mentee_employee_id"],
			mentor_employee_id=data["mentor_employee_id"],
			programme_name=data["programme_name"],
			start_date=data["start_date"],
			objectives=data.get("objectives"),
			end_date=data.get("end_date"),
			meeting_frequency=data.get("meeting_frequency", "monthly"),
		))
		return jsonify(record), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/mentoring-programmes/<programme_id>/sessions")
def log_mentoring_session(programme_id: str):
	data = request.get_json(force=True) or {}
	try:
		record = _run(_svc.log_mentoring_session(
			tenant_id=data.get("tenant_id", "default"),
			programme_id=programme_id,
			session_date=data["session_date"],
			topics_covered=data["topics_covered"],
			action_items=data.get("action_items"),
			notes=data.get("notes"),
		))
		return jsonify(record), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


# ── Certifications ────────────────────────────────────────────────────────────

@bp.get("/certifications")
def list_certifications():
	tenant_id = request.args.get("tenant_id", "default")
	try:
		items = _run(_svc.list_certifications(
			tenant_id,
			employee_id=request.args.get("employee_id"),
			expiring_within_days=int(request.args["expiring_within_days"]) if request.args.get("expiring_within_days") else None,
		))
		return jsonify({"items": items, "total": len(items)})
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/certifications/<cert_id>")
def get_certification(cert_id: str):
	tenant_id = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.get_certification(tenant_id, cert_id)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/certifications")
def add_certification():
	data = request.get_json(force=True) or {}
	try:
		record = _run(_svc.add_certification(
			tenant_id=data.get("tenant_id", "default"),
			employee_id=data["employee_id"],
			certification_name=data["certification_name"],
			issuing_body=data["issuing_body"],
			issue_date=data["issue_date"],
			expiry_date=data.get("expiry_date"),
			credential_id=data.get("credential_id"),
			certificate_url=data.get("certificate_url"),
		))
		return jsonify(record), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.put("/certifications/<cert_id>")
def update_certification(cert_id: str):
	data = request.get_json(force=True) or {}
	tenant_id = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.update_certification(tenant_id, cert_id, **data)))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.delete("/certifications/<cert_id>")
def delete_certification(cert_id: str):
	tenant_id = request.args.get("tenant_id", "default")
	try:
		_run(_svc.delete_certification(tenant_id, cert_id))
		return jsonify({"deleted": True})
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


# ── Career Paths ──────────────────────────────────────────────────────────────

@bp.get("/career-paths")
def list_career_paths():
	tenant_id = request.args.get("tenant_id", "default")
	try:
		items = _run(_svc.list_career_paths(
			tenant_id,
			employee_id=request.args.get("employee_id"),
			status=request.args.get("status"),
		))
		return jsonify({"items": items, "total": len(items)})
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/career-paths")
def create_career_path():
	data = request.get_json(force=True) or {}
	try:
		record = _run(_svc.create_career_path(
			tenant_id=data.get("tenant_id", "default"),
			employee_id=data["employee_id"],
			current_role=data["current_role"],
			target_role=data["target_role"],
			target_timeline_months=int(data.get("target_timeline_months", 24)),
			milestones=data.get("milestones"),
			advisor_employee_id=data.get("advisor_employee_id"),
		))
		return jsonify(record), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.put("/career-paths/<path_id>")
def update_career_path(path_id: str):
	data = request.get_json(force=True) or {}
	tenant_id = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.update_career_path(tenant_id, path_id, **data)))
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.delete("/career-paths/<path_id>")
def delete_career_path(path_id: str):
	tenant_id = request.args.get("tenant_id", "default")
	try:
		_run(_svc.delete_career_path(tenant_id, path_id))
		return jsonify({"deleted": True})
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


# ── Reports & Dashboard ───────────────────────────────────────────────────────

@bp.get("/report/<employee_id>")
def professional_development_report(employee_id: str):
	tenant_id = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.professional_development_report(tenant_id, employee_id)))
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
