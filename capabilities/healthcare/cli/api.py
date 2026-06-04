"""Flask Blueprint REST API for APG Clinical Management."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from flask import Blueprint, jsonify, request

from .models import (
	CarePlanCreate, CDSAlertCreate, ClinicalWorkflowCreate, HandoffCreate, ProtocolCreate,
)
from .service import ClinicalManagementService, PolicyViolationError

bp = Blueprint("healthcare_cli", __name__, url_prefix="/api/healthcare/cli")
_svc = ClinicalManagementService()


def _run(coro: Any) -> Any:
	loop = asyncio.new_event_loop()
	try:
		return asyncio.run(coro)
	finally:
		loop.close()


def _err(msg: str, status: int = 400) -> Any:
	return jsonify({"error": msg}), status


def _tenant() -> str:
	return request.headers.get("X-Tenant-ID", request.args.get("tenant_id", "default"))


@bp.get("/contract")
def get_contract():
	return jsonify(_run(_svc.describe(_tenant())))


@bp.get("/dashboard")
def dashboard():
	return jsonify(_run(_svc.dashboard_summary(_tenant())))


# ── care plans ────────────────────────────────────────────────────────────────

@bp.get("/care-plans")
def list_care_plans():
	plans = _run(_svc.list_care_plans(_tenant(), patient_id=request.args.get("patient_id"), status=request.args.get("status")))
	return jsonify({"items": [p.model_dump(mode="json") for p in plans], "count": len(plans)})


@bp.post("/care-plans")
def create_care_plan():
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	try:
		cp = _run(_svc.create_care_plan(CarePlanCreate(**data)))
		return jsonify(cp.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.get("/care-plans/<cp_id>")
def get_care_plan(cp_id: str):
	cp = _run(_svc.get_care_plan(_tenant(), cp_id))
	if cp is None:
		return _err("care_plan_not_found", 404)
	return jsonify(cp.model_dump(mode="json"))


@bp.post("/care-plans/<cp_id>/activate")
def activate_care_plan(cp_id: str):
	try:
		cp = _run(_svc.activate_care_plan(_tenant(), cp_id))
		if cp is None:
			return _err("care_plan_not_found", 404)
		return jsonify(cp.model_dump(mode="json"))
	except PolicyViolationError as e:
		return _err(str(e), 403)


@bp.post("/care-plans/<cp_id>/complete")
def complete_care_plan(cp_id: str):
	cp = _run(_svc.complete_care_plan(_tenant(), cp_id))
	if cp is None:
		return _err("care_plan_not_found", 404)
	return jsonify(cp.model_dump(mode="json"))


@bp.post("/care-plans/<cp_id>/interventions")
def add_intervention(cp_id: str):
	data = request.get_json(silent=True) or {}
	try:
		cp = _run(_svc.add_intervention(_tenant(), cp_id, data.get("intervention_type", ""), data.get("description", "")))
		if cp is None:
			return _err("care_plan_not_found", 404)
		return jsonify(cp.model_dump(mode="json"))
	except PolicyViolationError as e:
		return _err(str(e), 403)


# ── protocols ─────────────────────────────────────────────────────────────────

@bp.get("/protocols")
def list_protocols():
	protocols = _run(_svc.list_protocols(_tenant(), protocol_type=request.args.get("protocol_type")))
	return jsonify({"items": [p.model_dump(mode="json") for p in protocols], "count": len(protocols)})


@bp.post("/protocols")
def create_protocol():
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	try:
		proto = _run(_svc.create_protocol(ProtocolCreate(**data)))
		return jsonify(proto.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.post("/protocols/<proto_id>/complete")
def complete_protocol(proto_id: str):
	proto = _run(_svc.complete_protocol(_tenant(), proto_id))
	if proto is None:
		return _err("protocol_not_found", 404)
	return jsonify(proto.model_dump(mode="json"))


# ── workflows ─────────────────────────────────────────────────────────────────

@bp.get("/workflows")
def list_workflows():
	wfs = _run(_svc.list_workflows(_tenant(), patient_id=request.args.get("patient_id"), state=request.args.get("state")))
	return jsonify({"items": [w.model_dump(mode="json") for w in wfs], "count": len(wfs)})


@bp.post("/workflows")
def create_workflow():
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	if "due_at" in data and isinstance(data["due_at"], str):
		data["due_at"] = datetime.fromisoformat(data["due_at"])
	try:
		wf = _run(_svc.create_workflow(ClinicalWorkflowCreate(**data)))
		return jsonify(wf.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.post("/workflows/<wf_id>/transition")
def transition_workflow(wf_id: str):
	data = request.get_json(silent=True) or {}
	try:
		wf = _run(_svc.transition_workflow(_tenant(), wf_id, data.get("state", "")))
		if wf is None:
			return _err("workflow_not_found", 404)
		return jsonify(wf.model_dump(mode="json"))
	except PolicyViolationError as e:
		return _err(str(e), 403)


# ── CDS alerts ────────────────────────────────────────────────────────────────

@bp.get("/cds-alerts")
def list_cds_alerts():
	alerts = _run(_svc.list_cds_alerts(_tenant(), patient_id=request.args.get("patient_id"), priority=request.args.get("priority")))
	return jsonify({"items": [a.model_dump(mode="json") for a in alerts], "count": len(alerts)})


@bp.post("/cds-alerts")
def create_cds_alert():
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	try:
		alert = _run(_svc.create_cds_alert(CDSAlertCreate(**data)))
		return jsonify(alert.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.post("/cds-alerts/<alert_id>/acknowledge")
def acknowledge_cds_alert(alert_id: str):
	data = request.get_json(silent=True) or {}
	alert = _run(_svc.acknowledge_cds_alert(_tenant(), alert_id, data.get("acknowledged_by", "")))
	if alert is None:
		return _err("alert_not_found", 404)
	return jsonify(alert.model_dump(mode="json"))


# ── handoffs ──────────────────────────────────────────────────────────────────

@bp.get("/handoffs")
def list_handoffs():
	handoffs = _run(_svc.list_handoffs(_tenant(), patient_id=request.args.get("patient_id")))
	return jsonify({"items": [h.model_dump(mode="json") for h in handoffs], "count": len(handoffs)})


@bp.post("/handoffs")
def record_handoff():
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	try:
		handoff = _run(_svc.record_handoff(HandoffCreate(**data)))
		return jsonify(handoff.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.post("/handoffs/<handoff_id>/acknowledge")
def acknowledge_handoff(handoff_id: str):
	data = request.get_json(silent=True) or {}
	handoff = _run(_svc.acknowledge_handoff(_tenant(), handoff_id, data.get("acknowledged_by", "")))
	if handoff is None:
		return _err("handoff_not_found", 404)
	return jsonify(handoff.model_dump(mode="json"))
