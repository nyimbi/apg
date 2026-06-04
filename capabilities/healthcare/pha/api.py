"""Flask Blueprint REST API for APG Pharmacy Management."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from flask import Blueprint, jsonify, request

from .models import (
	ControlledSubstanceLogCreate, DispenseOrderCreate, DrugCreate,
	DrugInteractionCreate, InventoryItemCreate, PriorAuthCreate,
)
from .service import PharmacyManagementService, PolicyViolationError

bp = Blueprint("healthcare_pha", __name__, url_prefix="/api/healthcare/pha")
_svc = PharmacyManagementService()


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


# ── formulary ────────────────────────────────────────────────────────────────

@bp.get("/formulary")
def list_drugs():
	drugs = _run(_svc.list_drugs(_tenant(), formulary_status=request.args.get("formulary_status"), drug_schedule=request.args.get("drug_schedule")))
	return jsonify({"items": [d.model_dump(mode="json") for d in drugs], "count": len(drugs)})


@bp.post("/formulary")
def add_drug():
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	try:
		drug = _run(_svc.add_drug_to_formulary(DrugCreate(**data)))
		return jsonify(drug.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.get("/formulary/<drug_id>")
def get_drug(drug_id: str):
	drug = _run(_svc.get_drug(_tenant(), drug_id))
	if drug is None:
		return _err("drug_not_found", 404)
	return jsonify(drug.model_dump(mode="json"))


@bp.post("/formulary/<drug_id>/lasa")
def mark_lasa(drug_id: str):
	data = request.get_json(silent=True) or {}
	try:
		drug = _run(_svc.mark_drug_lasa(_tenant(), drug_id, data.get("lasa_pair", ""), data.get("alert_type", "")))
		if drug is None:
			return _err("drug_not_found", 404)
		return jsonify(drug.model_dump(mode="json"))
	except PolicyViolationError as e:
		return _err(str(e), 403)


# ── dispensing ────────────────────────────────────────────────────────────────

@bp.get("/dispense")
def list_dispense_orders():
	orders = _run(_svc.list_dispense_orders(_tenant(), patient_id=request.args.get("patient_id"), status=request.args.get("status")))
	return jsonify({"items": [o.model_dump(mode="json") for o in orders], "count": len(orders)})


@bp.post("/dispense")
def create_dispense_order():
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	try:
		order = _run(_svc.create_dispense_order(DispenseOrderCreate(**data)))
		return jsonify(order.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.get("/dispense/<order_id>")
def get_dispense_order(order_id: str):
	order = _run(_svc.get_dispense_order(_tenant(), order_id))
	if order is None:
		return _err("dispense_order_not_found", 404)
	return jsonify(order.model_dump(mode="json"))


@bp.post("/dispense/<order_id>/verify")
def verify_dispense(order_id: str):
	data = request.get_json(silent=True) or {}
	order = _run(_svc.verify_dispense(_tenant(), order_id, data.get("pharmacist_id", "")))
	if order is None:
		return _err("dispense_order_not_found", 404)
	return jsonify(order.model_dump(mode="json"))


@bp.post("/dispense/<order_id>/dispense")
def dispense(order_id: str):
	try:
		order = _run(_svc.dispense(_tenant(), order_id))
		if order is None:
			return _err("dispense_order_not_found", 404)
		return jsonify(order.model_dump(mode="json"))
	except PolicyViolationError as e:
		return _err(str(e), 403)


# ── interactions ──────────────────────────────────────────────────────────────

@bp.get("/interactions")
def list_interactions():
	interactions = _run(_svc.list_interactions(_tenant(), severity=request.args.get("severity")))
	return jsonify({"items": [i.model_dump(mode="json") for i in interactions], "count": len(interactions)})


@bp.post("/interactions")
def record_interaction():
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	try:
		interaction = _run(_svc.record_interaction(DrugInteractionCreate(**data)))
		return jsonify(interaction.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.post("/interactions/check")
def check_interactions():
	data = request.get_json(silent=True) or {}
	drug_ids = data.get("drug_ids", [])
	interactions = _run(_svc.check_interactions(_tenant(), drug_ids))
	return jsonify({"items": [i.model_dump(mode="json") for i in interactions], "count": len(interactions)})


# ── controlled substances ─────────────────────────────────────────────────────

@bp.get("/controlled")
def list_controlled_logs():
	logs = _run(_svc.list_controlled_logs(_tenant(), drug_id=request.args.get("drug_id"), action=request.args.get("action")))
	return jsonify({"items": [l.model_dump(mode="json") for l in logs], "count": len(logs)})


@bp.post("/controlled")
def log_controlled():
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	try:
		log = _run(_svc.log_controlled_substance(ControlledSubstanceLogCreate(**data)))
		return jsonify(log.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


# ── inventory ─────────────────────────────────────────────────────────────────

@bp.get("/inventory")
def list_inventory():
	items = _run(_svc.list_inventory(_tenant(), drug_id=request.args.get("drug_id"), status=request.args.get("status")))
	return jsonify({"items": [i.model_dump(mode="json") for i in items], "count": len(items)})


@bp.post("/inventory")
def add_inventory():
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	if "expiry_date" in data and isinstance(data["expiry_date"], str):
		data["expiry_date"] = datetime.fromisoformat(data["expiry_date"])
	try:
		item = _run(_svc.add_inventory_item(InventoryItemCreate(**data)))
		return jsonify(item.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.put("/inventory/<item_id>/status")
def update_inventory_status(item_id: str):
	data = request.get_json(silent=True) or {}
	try:
		item = _run(_svc.update_inventory_status(_tenant(), item_id, data.get("status", "")))
		if item is None:
			return _err("inventory_item_not_found", 404)
		return jsonify(item.model_dump(mode="json"))
	except PolicyViolationError as e:
		return _err(str(e), 403)


# ── prior auth ────────────────────────────────────────────────────────────────

@bp.get("/prior-auth")
def list_prior_auths():
	pas = _run(_svc.list_prior_auths(_tenant(), patient_id=request.args.get("patient_id"), status=request.args.get("status")))
	return jsonify({"items": [p.model_dump(mode="json") for p in pas], "count": len(pas)})


@bp.post("/prior-auth")
def request_prior_auth():
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	try:
		pa = _run(_svc.request_prior_auth(PriorAuthCreate(**data)))
		return jsonify(pa.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.post("/prior-auth/<pa_id>/approve")
def approve_prior_auth(pa_id: str):
	data = request.get_json(silent=True) or {}
	pa = _run(_svc.approve_prior_auth(_tenant(), pa_id, data.get("decision_by", ""), data.get("expires_in_days", 365)))
	if pa is None:
		return _err("prior_auth_not_found", 404)
	return jsonify(pa.model_dump(mode="json"))


@bp.post("/prior-auth/<pa_id>/deny")
def deny_prior_auth(pa_id: str):
	data = request.get_json(silent=True) or {}
	pa = _run(_svc.deny_prior_auth(_tenant(), pa_id, data.get("decision_by", ""), data.get("denial_reason", "")))
	if pa is None:
		return _err("prior_auth_not_found", 404)
	return jsonify(pa.model_dump(mode="json"))
