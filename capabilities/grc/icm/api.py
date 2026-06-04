"""Flask Blueprint REST API for grc_icm capability.

© 2025 Datacraft  |  Author: Nyimbi Odero
"""
from __future__ import annotations

from flask import Blueprint, jsonify, request

blueprint = Blueprint("grc_icm_api", __name__, url_prefix="/api/v1/grc/icm")


def _svc():
	from .service import IncidentComplianceService
	return IncidentComplianceService()


# ── Incidents CRUD ────────────────────────────────────────────────────────────

@blueprint.get("/incidents")
def list_incidents():
	return jsonify({"incidents": []})


@blueprint.post("/incidents")
def report_incident():
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.report_incident(
			entity_id=data["entity_id"],
			incident_type=data["incident_type"],
			description=data["description"],
			severity=data["severity"],
			affected_systems=data.get("affected_systems", []),
			reported_by=data["reported_by"],
			title=data.get("title"),
			detection_time=data.get("detection_time"),
		)
	)
	return jsonify(result), 201


@blueprint.get("/incidents/<incident_id>")
def get_incident(incident_id: str):
	import asyncio
	svc = _svc()
	result = asyncio.run(svc._get_incident(incident_id))
	return jsonify(result)


@blueprint.put("/incidents/<incident_id>")
def triage_incident(incident_id: str):
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.incident_triage(
			incident_id=incident_id,
			incident_commander_id=data["incident_commander_id"],
			priority=data["priority"],
			initial_response=data["initial_response"],
		)
	)
	return jsonify(result)


@blueprint.delete("/incidents/<incident_id>")
def close_incident(incident_id: str):
	import asyncio
	data = request.get_json(force=True) or {}
	svc = _svc()
	result = asyncio.run(
		svc.close_incident(
			incident_id=incident_id,
			resolution=data.get("resolution", ""),
			lessons_learned=data.get("lessons_learned", ""),
			closed_by=data.get("closed_by", "admin"),
		)
	)
	return jsonify(result)


# ── Incident lifecycle ────────────────────────────────────────────────────────

@blueprint.post("/incidents/<incident_id>/triage")
def post_triage(incident_id: str):
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.incident_triage(
			incident_id=incident_id,
			incident_commander_id=data["incident_commander_id"],
			priority=data["priority"],
			initial_response=data["initial_response"],
		)
	)
	return jsonify(result)


@blueprint.post("/incidents/<incident_id>/investigate")
def investigate(incident_id: str):
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.incident_investigation(
			incident_id=incident_id,
			findings=data["findings"],
			evidence=data.get("evidence", []),
			investigator_id=data["investigator_id"],
		)
	)
	return jsonify(result)


@blueprint.post("/incidents/<incident_id>/rca")
def root_cause(incident_id: str):
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.root_cause_analysis(
			incident_id=incident_id,
			rca_method=data["rca_method"],
			root_causes=data["root_causes"],
			contributing_factors=data.get("contributing_factors", []),
		)
	)
	return jsonify(result), 201


@blueprint.post("/incidents/<incident_id>/close")
def post_close(incident_id: str):
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.close_incident(
			incident_id=incident_id,
			resolution=data["resolution"],
			lessons_learned=data["lessons_learned"],
			closed_by=data["closed_by"],
		)
	)
	return jsonify(result)


@blueprint.post("/incidents/<incident_id>/regulatory-notification")
def regulatory_notification(incident_id: str):
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.regulatory_notification(
			incident_id=incident_id,
			regulator=data["regulator"],
			notification_type=data["notification_type"],
			deadline=data["deadline"],
		)
	)
	return jsonify(result), 201


@blueprint.post("/incidents/<incident_id>/pir")
def post_incident_review(incident_id: str):
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.post_incident_review(
			incident_id=incident_id,
			review_date=data["review_date"],
			reviewers=data["reviewers"],
			actions=data["actions"],
		)
	)
	return jsonify(result), 201


# ── Corrective actions ────────────────────────────────────────────────────────

@blueprint.get("/corrective-actions")
def list_actions():
	return jsonify({"actions": []})


@blueprint.post("/incidents/<incident_id>/corrective-actions")
def create_action(incident_id: str):
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.corrective_action(
			incident_id=incident_id,
			action_type=data["action_type"],
			description=data["description"],
			owner_id=data["owner_id"],
			deadline=data["deadline"],
		)
	)
	return jsonify(result), 201


@blueprint.put("/corrective-actions/<action_id>")
def update_action(action_id: str):
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.corrective_action_update(
			action_id=action_id,
			progress_pct=float(data["progress_pct"]),
			notes=data["notes"],
			updated_by=data["updated_by"],
		)
	)
	return jsonify(result)


# ── Compliance tests ──────────────────────────────────────────────────────────

@blueprint.get("/compliance-tests")
def list_tests():
	return jsonify({"tests": []})


@blueprint.post("/compliance-tests")
def create_test():
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.compliance_test(
			entity_id=data["entity_id"],
			control_id=data["control_id"],
			test_type=data["test_type"],
			test_date=data["test_date"],
			result=data["result"],
			tester_id=data["tester_id"],
		)
	)
	return jsonify(result), 201


@blueprint.get("/compliance-tests/<test_id>")
def get_test(test_id: str):
	import asyncio
	svc = _svc()
	result = asyncio.run(
		svc._store.get("compliance_tests", test_id)
	)
	return jsonify(result or {})


# ── Deficiencies ──────────────────────────────────────────────────────────────

@blueprint.get("/deficiencies")
def list_deficiencies():
	return jsonify({"deficiencies": []})


@blueprint.post("/deficiencies")
def create_deficiency():
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.compliance_deficiency(
			control_id=data["control_id"],
			deficiency_type=data["deficiency_type"],
			severity=data["severity"],
			identified_by=data["identified_by"],
		)
	)
	return jsonify(result), 201


@blueprint.put("/deficiencies/<deficiency_id>")
def remediation_plan(deficiency_id: str):
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.remediation_plan(
			deficiency_id=deficiency_id,
			remediation_actions=data["remediation_actions"],
			deadline=data["deadline"],
			owner_id=data["owner_id"],
		)
	)
	return jsonify(result)


# ── Reporting ─────────────────────────────────────────────────────────────────

@blueprint.get("/compliance-score")
def compliance_score():
	import asyncio
	entity_id = request.args["entity_id"]
	framework = request.args.get("framework", "iso_27001")
	period = request.args.get("period", "2026-06")
	svc = _svc()
	result = asyncio.run(
		svc.compliance_score(entity_id=entity_id, framework=framework, period=period)
	)
	return jsonify(result)


@blueprint.get("/dashboard")
def dashboard():
	import asyncio
	entity_id = request.args.get("entity_id", "default")
	svc = _svc()
	result = asyncio.run(svc.compliance_dashboard(entity_id))
	return jsonify(result)


@blueprint.get("/analytics")
def analytics():
	import asyncio
	entity_id = request.args["entity_id"]
	period = request.args.get("period", "2026-06")
	svc = _svc()
	result = asyncio.run(
		svc.incident_analytics(entity_id=entity_id, period=period)
	)
	return jsonify(result)
