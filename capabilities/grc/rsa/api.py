"""Flask Blueprint REST API for grc_rsa capability.

© 2025 Datacraft  |  Author: Nyimbi Odero
"""
from __future__ import annotations

from flask import Blueprint, jsonify, request

blueprint = Blueprint("grc_rsa_api", __name__, url_prefix="/api/v1/grc/rsa")


def _svc():
	from .service import RiskAssessmentService
	return RiskAssessmentService()


# ── Risks CRUD ────────────────────────────────────────────────────────────────

@blueprint.get("/risks")
def list_risks():
	return jsonify({"risks": []})


@blueprint.post("/risks")
def create_risk():
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.risk_register_entry(
			entity_id=data["entity_id"],
			risk_name=data["risk_name"],
			category=data["category"],
			description=data["description"],
			owner_id=data["owner_id"],
			risk_id=data.get("risk_id"),
		)
	)
	return jsonify(result), 201


@blueprint.get("/risks/<risk_id>")
def get_risk(risk_id: str):
	import asyncio
	svc = _svc()
	result = asyncio.run(svc._get_risk(risk_id))
	return jsonify(result)


@blueprint.put("/risks/<risk_id>")
def assess_risk(risk_id: str):
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.risk_assessment(
			risk_id=risk_id,
			likelihood_1_5=int(data["likelihood_1_5"]),
			impact_1_5=int(data["impact_1_5"]),
			velocity=data["velocity"],
			assessor_id=data["assessor_id"],
		)
	)
	return jsonify(result)


@blueprint.delete("/risks/<risk_id>")
def close_risk(risk_id: str):
	return jsonify({"message": "risks are closed via treatment completion; use PUT /risks/<id>"})


# ── Assessments ───────────────────────────────────────────────────────────────

@blueprint.post("/risks/<risk_id>/assess")
def post_assessment(risk_id: str):
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.risk_assessment(
			risk_id=risk_id,
			likelihood_1_5=int(data["likelihood_1_5"]),
			impact_1_5=int(data["impact_1_5"]),
			velocity=data["velocity"],
			assessor_id=data["assessor_id"],
		)
	)
	return jsonify(result), 201


@blueprint.post("/risks/<risk_id>/residual")
def update_residual(risk_id: str):
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.update_residual_score(
			risk_id=risk_id,
			control_effectiveness_pct=float(data["control_effectiveness_pct"]),
		)
	)
	return jsonify(result)


# ── Controls ──────────────────────────────────────────────────────────────────

@blueprint.get("/controls")
def list_controls():
	return jsonify({"controls": []})


@blueprint.post("/controls/<control_id>/assess")
def assess_control(control_id: str):
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.control_assessment(
			control_id=control_id,
			effectiveness_rating=data["effectiveness_rating"],
			evidence=data["evidence"],
			assessed_by=data["assessed_by"],
		)
	)
	return jsonify(result), 201


@blueprint.get("/controls/<control_id>/gap")
def control_gap(control_id: str):
	import asyncio
	svc = _svc()
	result = asyncio.run(svc.control_gap(control_id))
	return jsonify(result)


# ── Treatment plans ───────────────────────────────────────────────────────────

@blueprint.post("/risks/<risk_id>/treatment")
def create_treatment(risk_id: str):
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.risk_treatment_plan(
			risk_id=risk_id,
			treatment_type=data["treatment_type"],
			actions=data["actions"],
			owner_id=data["owner_id"],
			deadline=data["deadline"],
		)
	)
	return jsonify(result), 201


@blueprint.put("/treatments/<treatment_id>")
def update_treatment(treatment_id: str):
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.risk_treatment_update(
			treatment_id=treatment_id,
			progress_pct=float(data["progress_pct"]),
			notes=data["notes"],
			updated_by=data["updated_by"],
		)
	)
	return jsonify(result)


# ── KRI ───────────────────────────────────────────────────────────────────────

@blueprint.post("/kri")
def record_kri():
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.key_risk_indicator(
			kri_name=data["kri_name"],
			threshold_amber=float(data["threshold_amber"]),
			threshold_red=float(data["threshold_red"]),
			current_value=float(data["current_value"]),
			period=data["period"],
			entity_id=data.get("entity_id"),
			unit=data.get("unit", ""),
		)
	)
	return jsonify(result), 201


@blueprint.post("/risk-appetite")
def risk_appetite():
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.risk_appetite_statement(
			entity_id=data["entity_id"],
			risk_category=data["risk_category"],
			tolerance_level=data["tolerance_level"],
		)
	)
	return jsonify(result)


# ── Reporting ─────────────────────────────────────────────────────────────────

@blueprint.get("/heat-map")
def heat_map():
	import asyncio
	entity_id = request.args["entity_id"]
	as_of_date = request.args.get("as_of_date", "")
	svc = _svc()
	result = asyncio.run(
		svc.risk_heat_map(entity_id=entity_id, as_of_date=as_of_date)
	)
	return jsonify(result)


@blueprint.get("/dashboard")
def dashboard():
	import asyncio
	entity_id = request.args.get("entity_id", "default")
	svc = _svc()
	result = asyncio.run(svc.risk_dashboard(entity_id))
	return jsonify(result)


@blueprint.get("/board-report")
def board_report():
	import asyncio
	entity_id = request.args["entity_id"]
	period = request.args.get("period", "2026")
	svc = _svc()
	result = asyncio.run(
		svc.board_risk_report(entity_id=entity_id, period=period)
	)
	return jsonify(result)


@blueprint.post("/scenario")
def scenario():
	import asyncio
	data = request.get_json(force=True)
	svc = _svc()
	result = asyncio.run(
		svc.risk_scenario_analysis(
			entity_id=data["entity_id"],
			scenario_type=data["scenario_type"],
			parameters=data.get("parameters", {}),
		)
	)
	return jsonify(result)


@blueprint.get("/analytics")
def analytics():
	import asyncio
	entity_id = request.args["entity_id"]
	period = request.args.get("period", "2026-06")
	svc = _svc()
	result = asyncio.run(
		svc.risk_analytics(entity_id=entity_id, period=period)
	)
	return jsonify(result)
