"""Flask Blueprint REST API for grc_aud capability.

© 2025 Datacraft  |  Author: Nyimbi Odero
"""
from __future__ import annotations

from flask import Blueprint, jsonify, request

blueprint = Blueprint("grc_aud_api", __name__, url_prefix="/api/v1/grc/aud")


def _svc():
	from .service import AuditManagementService
	return AuditManagementService()


# ── Audit plans ───────────────────────────────────────────────────────────────

@blueprint.post("/plans")
def create_plan():
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.create_audit_plan(
			entity_id=data["entity_id"],
			year=int(data["year"]),
			risk_based_areas=data["risk_based_areas"],
			approved_by=data["approved_by"],
			plan_type=data.get("plan_type", "annual"),
			methodology=data.get("methodology", "risk_based"),
		)
	)
	return jsonify(result), 201


@blueprint.put("/plans/<plan_id>")
def update_plan(plan_id: str):
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.risk_based_plan_update(
			plan_id=plan_id,
			risk_reassessment_data=data["risk_reassessment_data"],
		)
	)
	return jsonify(result)


# ── Engagements CRUD ──────────────────────────────────────────────────────────

@blueprint.get("/engagements")
def list_engagements():
	return jsonify({"engagements": []})


@blueprint.post("/engagements")
def create_engagement():
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.create_audit_engagement(
			plan_id=data["plan_id"],
			area=data["area"],
			objectives=data["objectives"],
			start_date=data["start_date"],
			end_date=data["end_date"],
			lead_auditor_id=data["lead_auditor_id"],
			audit_type=data.get("audit_type", "internal"),
			scope=data.get("scope", "process"),
			auditee_id=data.get("auditee_id"),
			planned_hours=int(data.get("planned_hours", 80)),
		)
	)
	return jsonify(result), 201


@blueprint.get("/engagements/<engagement_id>")
def get_engagement(engagement_id: str):
	import asyncio
	svc = _svc()
	result = asyncio.run(svc._get_engagement(engagement_id))
	return jsonify(result)


@blueprint.put("/engagements/<engagement_id>")
def update_engagement(engagement_id: str):
	return jsonify({"message": "use domain-specific endpoints for engagement updates"})


@blueprint.delete("/engagements/<engagement_id>")
def cancel_engagement(engagement_id: str):
	return jsonify({"message": "engagement cancellation via status update not yet implemented"})


# ── Fieldwork and findings ────────────────────────────────────────────────────

@blueprint.post("/engagements/<engagement_id>/fieldwork")
def fieldwork(engagement_id: str):
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.fieldwork_record(
			engagement_id=engagement_id,
			area_tested=data["area_tested"],
			finding_type=data["finding_type"],
			observation=data["observation"],
			criteria=data["criteria"],
			evidence=data.get("evidence", []),
			risk_rating=data["risk_rating"],
			auditor_id=data.get("auditor_id"),
		)
	)
	return jsonify(result), 201


@blueprint.post("/engagements/<engagement_id>/report")
def draft_report(engagement_id: str):
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.draft_audit_report(
			engagement_id=engagement_id,
			findings=data.get("findings", []),
			recommendations=data.get("recommendations", []),
			auditor_id=data["auditor_id"],
		)
	)
	return jsonify(result), 201


@blueprint.post("/engagements/<engagement_id>/finalise")
def finalise(engagement_id: str):
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.finalise_report(
			engagement_id=engagement_id,
			chief_audit_executive_id=data["chief_audit_executive_id"],
			sign_off_date=data["sign_off_date"],
		)
	)
	return jsonify(result)


@blueprint.post("/engagements/<engagement_id>/qa")
def qa_review(engagement_id: str):
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.quality_assurance_review(
			engagement_id=engagement_id,
			reviewer_id=data["reviewer_id"],
			rating=data["rating"],
			observations=data["observations"],
		)
	)
	return jsonify(result), 201


# ── Findings CRUD ─────────────────────────────────────────────────────────────

@blueprint.get("/findings")
def list_findings():
	return jsonify({"findings": []})


@blueprint.get("/findings/<finding_id>")
def get_finding(finding_id: str):
	import asyncio
	svc = _svc()
	result = asyncio.run(svc._get_finding(finding_id))
	return jsonify(result)


@blueprint.put("/findings/<finding_id>")
def track_issue(finding_id: str):
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.issue_tracking(
			finding_id=finding_id,
			status=data["status"],
			progress_notes=data["progress_notes"],
			updated_by=data["updated_by"],
		)
	)
	return jsonify(result)


@blueprint.delete("/findings/<finding_id>")
def close_finding(finding_id: str):
	import asyncio
	data = request.get_json(force=True) or {}
	svc = _svc()
	result = asyncio.run(
		svc.close_finding(
			finding_id=finding_id,
			close_date=data.get("close_date", ""),
			verified_by=data.get("verified_by", "admin"),
		)
	)
	return jsonify(result)


@blueprint.post("/findings/<finding_id>/management-response")
def management_response(finding_id: str):
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.management_response(
			finding_id=finding_id,
			response_text=data["response_text"],
			action_plan=data["action_plan"],
			owner_id=data["owner_id"],
			deadline=data["deadline"],
		)
	)
	return jsonify(result)


@blueprint.post("/findings/<finding_id>/follow-up")
def follow_up(finding_id: str):
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.follow_up_audit(
			finding_id=finding_id,
			follow_up_date=data["follow_up_date"],
			status=data["status"],
			evidence=data.get("evidence", []),
		)
	)
	return jsonify(result), 201


# ── Reporting ─────────────────────────────────────────────────────────────────

@blueprint.get("/committee-report")
def committee_report():
	import asyncio
	entity_id = request.args["entity_id"]
	period = request.args.get("period", "2026")
	svc = _svc()
	result = asyncio.run(
		svc.audit_committee_report(entity_id=entity_id, period=period)
	)
	return jsonify(result)


@blueprint.get("/kpi")
def kpi():
	import asyncio
	entity_id = request.args["entity_id"]
	period = request.args.get("period", "2026")
	svc = _svc()
	result = asyncio.run(
		svc.kpi_report(entity_id=entity_id, period=period)
	)
	return jsonify(result)


@blueprint.get("/universe")
def universe():
	import asyncio
	entity_id = request.args["entity_id"]
	svc = _svc()
	result = asyncio.run(svc.audit_universe(entity_id))
	return jsonify(result)


@blueprint.get("/analytics")
def analytics():
	import asyncio
	entity_id = request.args["entity_id"]
	period = request.args.get("period", "2026")
	svc = _svc()
	result = asyncio.run(
		svc.audit_analytics(entity_id=entity_id, period=period)
	)
	return jsonify(result)


@blueprint.post("/continuous-audit")
def continuous_audit():
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.continuous_auditing(
			entity_id=data["entity_id"],
			data_analytics_type=data["data_analytics_type"],
			frequency=data["frequency"],
		)
	)
	return jsonify(result), 201


@blueprint.post("/fraud-investigation")
def fraud_investigation():
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.fraud_investigation(
			suspicion_id=data["suspicion_id"],
			investigator_id=data["investigator_id"],
			scope=data["scope"],
			methodology=data["methodology"],
		)
	)
	return jsonify(result), 201


@blueprint.post("/whistleblower")
def whistleblower():
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.whistleblower_case(
			case_id=data.get("case_id", ""),
			category=data["category"],
			description=data["description"],
			received_date=data["received_date"],
		)
	)
	return jsonify(result), 201
