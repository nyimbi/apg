"""Legal Compliance Management — Flask Blueprint REST API."""
from __future__ import annotations

import logging
from typing import Any

from flask import Blueprint, jsonify, request

from .service import LegalComplianceService

_log = logging.getLogger(__name__)

bp = Blueprint("leg_cpl", __name__, url_prefix="/api/legal/cpl")
_svc: LegalComplianceService | None = None


def get_service() -> LegalComplianceService:
	global _svc
	if _svc is None:
		_svc = LegalComplianceService()
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


@bp.get("/requirements")
def list_requirements():
	tenant = request.args.get("tenant_id", "default")
	try:
		items = _run(get_service().list_requirements(
			tenant_id=tenant,
			regulation=request.args.get("regulation"),
			jurisdiction=request.args.get("jurisdiction"),
			category=request.args.get("category"),
			status=request.args.get("status"),
			risk_level=request.args.get("risk_level"),
			owner_id=request.args.get("owner_id"),
		))
		return jsonify({"items": items, "total": len(items)})
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/requirements/<requirement_id>")
def get_requirement(requirement_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().get_requirement(tenant, requirement_id)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/requirements")
def create_requirement():
	body = request.get_json(force=True) or {}
	try:
		return jsonify(_run(get_service().create_requirement(**body))), 201
	except Exception as exc:
		_log.error("create_requirement: %s", exc)
		return jsonify({"error": str(exc)}), 400


@bp.put("/requirements/<requirement_id>")
def update_requirement(requirement_id: str):
	body = request.get_json(force=True) or {}
	tenant = body.pop("tenant_id", "default")
	try:
		return jsonify(_run(get_service().update_requirement(tenant, requirement_id, **body)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.delete("/requirements/<requirement_id>")
def delete_requirement(requirement_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().delete_requirement(tenant, requirement_id)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/requirements/<requirement_id>/compliant")
def mark_compliant(requirement_id: str):
	body = request.get_json(force=True) or {}
	tenant = body.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().mark_compliant(tenant, requirement_id, body.get("assessed_by", ""))))
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/requirements/<requirement_id>/non-compliant")
def flag_non_compliant(requirement_id: str):
	body = request.get_json(force=True) or {}
	tenant = body.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().flag_non_compliant(tenant, requirement_id, body.get("reason", ""))))
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/calendar")
def list_calendar():
	tenant = request.args.get("tenant_id", "default")
	try:
		items = _run(get_service().list_calendar_entries(
			tenant_id=tenant,
			requirement_id=request.args.get("requirement_id"),
			status=request.args.get("status"),
		))
		return jsonify({"items": items, "total": len(items)})
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/calendar")
def create_calendar_entry():
	body = request.get_json(force=True) or {}
	try:
		return jsonify(_run(get_service().create_calendar_entry(**body))), 201
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.put("/calendar/<entry_id>")
def update_calendar_entry(entry_id: str):
	body = request.get_json(force=True) or {}
	tenant = body.pop("tenant_id", "default")
	try:
		return jsonify(_run(get_service().update_calendar_entry(tenant, entry_id, **body)))
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/calendar/<entry_id>/complete")
def complete_calendar(entry_id: str):
	body = request.get_json(force=True) or {}
	tenant = body.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().complete_calendar_entry(tenant, entry_id, body.get("completed_by", ""))))
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.delete("/calendar/<entry_id>")
def delete_calendar(entry_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().delete_calendar_entry(tenant, entry_id)))
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/evidence")
def list_evidence():
	tenant = request.args.get("tenant_id", "default")
	req_id = request.args.get("requirement_id", "")
	try:
		items = _run(get_service().list_evidence(tenant, req_id))
		return jsonify({"items": items, "total": len(items)})
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/evidence")
def create_evidence():
	body = request.get_json(force=True) or {}
	try:
		return jsonify(_run(get_service().create_evidence(**body))), 201
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.put("/evidence/<evidence_id>")
def update_evidence(evidence_id: str):
	body = request.get_json(force=True) or {}
	tenant = body.pop("tenant_id", "default")
	try:
		return jsonify(_run(get_service().update_evidence(tenant, evidence_id, **body)))
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.delete("/evidence/<evidence_id>")
def delete_evidence(evidence_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().delete_evidence(tenant, evidence_id)))
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/breaches")
def list_breaches():
	tenant = request.args.get("tenant_id", "default")
	try:
		items = _run(get_service().list_breaches(
			tenant_id=tenant,
			requirement_id=request.args.get("requirement_id"),
			severity=request.args.get("severity"),
			status=request.args.get("status"),
		))
		return jsonify({"items": items, "total": len(items)})
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/breaches/<breach_id>")
def get_breach(breach_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().get_breach(tenant, breach_id)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/breaches")
def create_breach():
	body = request.get_json(force=True) or {}
	try:
		return jsonify(_run(get_service().create_breach(**body))), 201
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.put("/breaches/<breach_id>")
def update_breach(breach_id: str):
	body = request.get_json(force=True) or {}
	tenant = body.pop("tenant_id", "default")
	try:
		return jsonify(_run(get_service().update_breach(tenant, breach_id, **body)))
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.delete("/breaches/<breach_id>")
def delete_breach(breach_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().delete_breach(tenant, breach_id)))
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/breaches/<breach_id>/remediate")
def remediate_breach(breach_id: str):
	body = request.get_json(force=True) or {}
	tenant = body.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().remediate_breach(tenant, breach_id, body.get("remediation_notes", ""))))
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/breaches/<breach_id>/report")
def report_breach(breach_id: str):
	body = request.get_json(force=True) or {}
	tenant = body.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().report_breach_to_regulator(
			tenant, breach_id,
			body.get("regulator", ""), body.get("reference_number", ""), body.get("reported_by", ""),
		)))
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/dashboard")
def dashboard():
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(get_service().compliance_dashboard(tenant)))
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/risk-register")
def risk_register():
	tenant = request.args.get("tenant_id", "default")
	try:
		items = _run(get_service().risk_register(tenant))
		return jsonify({"items": items, "total": len(items)})
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
