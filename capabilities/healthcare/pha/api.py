"""Flask Blueprint REST API for APG Pharmacy Management.

Every entity has full CRUD. Business operation endpoints follow the pattern
POST /<entity>/<id>/<action>. Report endpoints live under GET /reports/<type>.
Tenant isolation is enforced via X-Tenant-ID header or tenant_id query param.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from flask import Blueprint, jsonify, request

from .models import (
	ControlledSubstanceLogCreate,
	CounsellingChecklistCreate,
	DispenseOrderCreate,
	DrugCreate,
	DrugInteractionCreate,
	InventoryItemCreate,
	NarcoticsRegisterEntryCreate,
	PriorAuthCreate,
	PrescriptionCreate,
	ReorderRequestCreate,
	ReturnedMedicationCreate,
	ColdChainRecordCreate,
)
from .service import PharmacyManagementService, PolicyViolationError

bp = Blueprint("healthcare_pha", __name__, url_prefix="/api/healthcare/pha")
_svc = PharmacyManagementService()


def _run(coro: Any) -> Any:
	return asyncio.run(coro)


def _err(msg: str, status: int = 400) -> Any:
	return jsonify({"error": msg}), status


def _tenant() -> str:
	return request.headers.get("X-Tenant-ID", request.args.get("tenant_id", "default"))


def _actor() -> str:
	return request.headers.get("X-Actor-ID", request.args.get("actor_id", "system"))


def _paginate(items: list[Any]) -> dict[str, Any]:
	page = int(request.args.get("page", 1))
	page_size = min(int(request.args.get("page_size", 50)), 200)
	start = (page - 1) * page_size
	end = start + page_size
	return {
		"items": items[start:end],
		"count": len(items[start:end]),
		"total": len(items),
		"page": page,
		"page_size": page_size,
	}


# ── Meta ──────────────────────────────────────────────────────────────────────

@bp.get("/contract")
def get_contract():
	"""Return the capability contract for this tenant."""
	return jsonify(_run(_svc.describe(_tenant())))


@bp.get("/dashboard")
def dashboard():
	"""Pharmacy KPI dashboard — all entity counts plus active alerts."""
	return jsonify(_run(_svc.dashboard_summary(_tenant())))


# ── Formulary ─────────────────────────────────────────────────────────────────

@bp.get("/formulary")
def list_drugs():
	"""List drugs with optional filters: formulary_status, drug_schedule, drug_type."""
	drugs = _run(_svc.list_drugs(
		_tenant(),
		formulary_status=request.args.get("formulary_status"),
		drug_schedule=request.args.get("drug_schedule"),
		drug_type=request.args.get("drug_type"),
	))
	page = _paginate([d.model_dump(mode="json") for d in drugs])
	return jsonify(page)


@bp.post("/formulary")
def add_drug():
	"""Add a drug to the tenant formulary."""
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	data.setdefault("created_by", _actor())
	try:
		drug = _run(_svc.add_drug_to_formulary(DrugCreate(**data)))
		return jsonify(drug.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.get("/formulary/<drug_id>")
def get_drug(drug_id: str):
	"""Get a single drug by ID."""
	drug = _run(_svc.get_drug(_tenant(), drug_id))
	if drug is None:
		return _err("drug_not_found", 404)
	return jsonify(drug.model_dump(mode="json"))


@bp.put("/formulary/<drug_id>")
def update_drug(drug_id: str):
	"""Update formulary status or LASA flags for a drug."""
	data = request.get_json(silent=True) or {}
	try:
		status = data.get("formulary_status")
		if status:
			drug = _run(_svc.update_formulary_status(_tenant(), drug_id, status))
		else:
			drug = _run(_svc.get_drug(_tenant(), drug_id))
		if drug is None:
			return _err("drug_not_found", 404)
		return jsonify(drug.model_dump(mode="json"))
	except PolicyViolationError as e:
		return _err(str(e), 403)


@bp.delete("/formulary/<drug_id>")
def delete_drug(drug_id: str):
	"""Soft-delete a drug from the formulary."""
	try:
		result = _run(_svc.soft_delete_drug(_tenant(), drug_id, _actor()))
		if not result:
			return _err("drug_not_found", 404)
		return jsonify({"deleted": True, "id": drug_id})
	except PolicyViolationError as e:
		return _err(str(e), 403)


@bp.post("/formulary/<drug_id>/lasa")
def mark_lasa(drug_id: str):
	"""Flag a drug as LASA with its pair and alert type."""
	data = request.get_json(silent=True) or {}
	try:
		drug = _run(_svc.mark_drug_lasa(
			_tenant(), drug_id,
			data.get("lasa_pair", ""),
			data.get("alert_type", ""),
		))
		if drug is None:
			return _err("drug_not_found", 404)
		return jsonify(drug.model_dump(mode="json"))
	except PolicyViolationError as e:
		return _err(str(e), 403)


@bp.post("/formulary/<drug_id>/review")
def formulary_review(drug_id: str):
	"""Initiate a P&T committee formulary review."""
	data = request.get_json(silent=True) or {}
	try:
		record = _run(_svc.formulary_review(
			_tenant(), drug_id,
			review_type=data.get("review_type", "annual_review"),
			recommendation=data.get("recommendation", "maintain"),
			reviewed_by=data.get("reviewed_by", _actor()),
			clinical_rationale=data.get("clinical_rationale", ""),
			cost_data=data.get("cost_data"),
		))
		return jsonify(record), 201
	except (PolicyViolationError, KeyError, AssertionError) as e:
		status = 403 if isinstance(e, PolicyViolationError) else 400
		return _err(str(e), status)


# ── Prescriptions ─────────────────────────────────────────────────────────────

@bp.get("/prescriptions")
def list_prescriptions():
	"""List prescriptions filtered by patient_id, status, or drug_id."""
	prescriptions = _run(_svc.list_prescriptions(
		_tenant(),
		patient_id=request.args.get("patient_id"),
		status=request.args.get("status"),
		drug_id=request.args.get("drug_id"),
	))
	return jsonify(_paginate([p.model_dump(mode="json") for p in prescriptions]))


@bp.post("/prescriptions")
def create_prescription():
	"""Receive a new prescription."""
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	data.setdefault("created_by", _actor())
	if "dosage_form" not in data:
		data["dosage_form"] = "tablet"
	try:
		rx = _run(_svc.create_prescription(PrescriptionCreate(**data)))
		return jsonify(rx.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.get("/prescriptions/<rx_id>")
def get_prescription(rx_id: str):
	"""Get a prescription by ID."""
	rx = _run(_svc.get_prescription(_tenant(), rx_id))
	if rx is None:
		return _err("prescription_not_found", 404)
	return jsonify(rx.model_dump(mode="json"))


@bp.put("/prescriptions/<rx_id>")
def update_prescription(rx_id: str):
	"""Update prescription status or refills remaining."""
	data = request.get_json(silent=True) or {}
	data.setdefault("updated_by", _actor())
	try:
		from .models import PrescriptionUpdate
		rx = _run(_svc.update_prescription(_tenant(), rx_id, PrescriptionUpdate(**data)))
		if rx is None:
			return _err("prescription_not_found", 404)
		return jsonify(rx.model_dump(mode="json"))
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.delete("/prescriptions/<rx_id>")
def cancel_prescription(rx_id: str):
	"""Cancel a prescription (soft delete)."""
	try:
		result = _run(_svc.cancel_prescription(_tenant(), rx_id, _actor()))
		if not result:
			return _err("prescription_not_found", 404)
		return jsonify({"cancelled": True, "id": rx_id})
	except PolicyViolationError as e:
		return _err(str(e), 403)


@bp.post("/prescriptions/<rx_id>/verify")
def pharmacist_verify_prescription(rx_id: str):
	"""Pharmacist clinical verification of a prescription."""
	data = request.get_json(silent=True) or {}
	try:
		record = _run(_svc.verify_prescription(
			_tenant(), rx_id,
			pharmacist_id=data.get("pharmacist_id", _actor()),
			clinical_notes=data.get("clinical_notes", ""),
		))
		return jsonify(record)
	except PolicyViolationError as e:
		return _err(str(e), 403)


# ── Dispensing ────────────────────────────────────────────────────────────────

@bp.get("/dispense")
def list_dispense_orders():
	"""List dispense orders filtered by patient_id or status."""
	orders = _run(_svc.list_dispense_orders(
		_tenant(),
		patient_id=request.args.get("patient_id"),
		status=request.args.get("status"),
	))
	return jsonify(_paginate([o.model_dump(mode="json") for o in orders]))


@bp.post("/dispense")
def create_dispense_order():
	"""Create a dispense order with formulary and interaction gate checks."""
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	data.setdefault("created_by", _actor())
	try:
		order = _run(_svc.create_dispense_order(DispenseOrderCreate(**data)))
		return jsonify(order.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.get("/dispense/<order_id>")
def get_dispense_order(order_id: str):
	"""Get a dispense order by ID."""
	order = _run(_svc.get_dispense_order(_tenant(), order_id))
	if order is None:
		return _err("dispense_order_not_found", 404)
	return jsonify(order.model_dump(mode="json"))


@bp.put("/dispense/<order_id>")
def update_dispense_order(order_id: str):
	"""Update a dispense order (status, counselling, label, barcode flags)."""
	data = request.get_json(silent=True) or {}
	data.setdefault("updated_by", _actor())
	try:
		from .models import DispenseOrderUpdate
		order = _run(_svc.update_dispense_order(_tenant(), order_id, DispenseOrderUpdate(**data)))
		if order is None:
			return _err("dispense_order_not_found", 404)
		return jsonify(order.model_dump(mode="json"))
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.delete("/dispense/<order_id>")
def cancel_dispense_order(order_id: str):
	"""Cancel a dispense order."""
	try:
		result = _run(_svc.cancel_dispense_order(_tenant(), order_id, _actor()))
		if not result:
			return _err("dispense_order_not_found", 404)
		return jsonify({"cancelled": True, "id": order_id})
	except PolicyViolationError as e:
		return _err(str(e), 403)


@bp.post("/dispense/<order_id>/verify")
def verify_dispense(order_id: str):
	"""Pharmacist verification step — gates actual dispense."""
	data = request.get_json(silent=True) or {}
	order = _run(_svc.verify_dispense(_tenant(), order_id, data.get("pharmacist_id", _actor())))
	if order is None:
		return _err("dispense_order_not_found", 404)
	return jsonify(order.model_dump(mode="json"))


@bp.post("/dispense/<order_id>/dispense")
def dispense(order_id: str):
	"""Execute dispense for a pharmacist-verified order."""
	try:
		order = _run(_svc.dispense(_tenant(), order_id))
		if order is None:
			return _err("dispense_order_not_found", 404)
		return jsonify(order.model_dump(mode="json"))
	except PolicyViolationError as e:
		return _err(str(e), 403)


@bp.post("/dispense/<order_id>/pickup")
def mark_picked_up(order_id: str):
	"""Record patient pickup of a dispensed order."""
	try:
		order = _run(_svc.mark_picked_up(_tenant(), order_id))
		if order is None:
			return _err("dispense_order_not_found", 404)
		return jsonify(order.model_dump(mode="json"))
	except PolicyViolationError as e:
		return _err(str(e), 403)


@bp.post("/dispense/medication")
def dispense_medication():
	"""Full dispense_medication() workflow — verify + lot-level dispense in one call."""
	data = request.get_json(silent=True) or {}
	expiry_raw = data.get("expiry_date")
	if isinstance(expiry_raw, str):
		data["expiry_date"] = datetime.fromisoformat(expiry_raw)
	try:
		record = _run(_svc.dispense_medication(
			tenant_id=_tenant(),
			prescription_id=data.get("prescription_id", ""),
			lot_number=data.get("lot_number", ""),
			expiry_date=data.get("expiry_date", datetime.utcnow()),
			quantity=float(data.get("quantity", 0)),
			dispensed_by=data.get("dispensed_by", _actor()),
			patient_id=data.get("patient_id", ""),
			drug_id=data.get("drug_id", ""),
		))
		return jsonify(record), 201
	except (PolicyViolationError, AssertionError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.post("/dispense/interactions/check")
def check_interactions_at_dispense():
	"""Point-of-dispense drug interaction safety check for a prescription."""
	data = request.get_json(silent=True) or {}
	try:
		result = _run(_svc.check_drug_interactions_at_dispense(
			_tenant(),
			prescription_id=data.get("prescription_id", ""),
			patient_current_drugs=data.get("patient_current_drugs", []),
		))
		return jsonify(result)
	except (PolicyViolationError, AssertionError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


# ── Drug Interactions ─────────────────────────────────────────────────────────

@bp.get("/interactions")
def list_interactions():
	"""List known drug-drug interactions, optionally filtered by severity."""
	interactions = _run(_svc.list_interactions(
		_tenant(),
		severity=request.args.get("severity"),
	))
	return jsonify(_paginate([i.model_dump(mode="json") for i in interactions]))


@bp.post("/interactions")
def record_interaction():
	"""Record a new known drug-drug interaction pair."""
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	data.setdefault("created_by", _actor())
	try:
		interaction = _run(_svc.record_interaction(DrugInteractionCreate(**data)))
		return jsonify(interaction.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.get("/interactions/<interaction_id>")
def get_interaction(interaction_id: str):
	"""Get a specific interaction record."""
	interaction = _run(_svc.get_interaction(_tenant(), interaction_id))
	if interaction is None:
		return _err("interaction_not_found", 404)
	return jsonify(interaction.model_dump(mode="json"))


@bp.delete("/interactions/<interaction_id>")
def delete_interaction(interaction_id: str):
	"""Soft-delete an interaction record."""
	result = _run(_svc.soft_delete_interaction(_tenant(), interaction_id, _actor()))
	if not result:
		return _err("interaction_not_found", 404)
	return jsonify({"deleted": True, "id": interaction_id})


@bp.post("/interactions/check")
def check_interactions():
	"""Return all known interactions among a list of drug IDs."""
	data = request.get_json(silent=True) or {}
	drug_ids = data.get("drug_ids", [])
	interactions = _run(_svc.check_interactions(_tenant(), drug_ids))
	return jsonify({"items": [i.model_dump(mode="json") for i in interactions], "count": len(interactions)})


@bp.post("/interactions/check-drug")
def check_drug_interactions():
	"""Check interactions for a specific drug against a patient's current drug list."""
	data = request.get_json(silent=True) or {}
	try:
		result = _run(_svc.check_drug_interactions(
			_tenant(),
			drug_id=data.get("drug_id", ""),
			patient_drug_ids=data.get("patient_drug_ids", []),
		))
		return jsonify(result)
	except AssertionError as e:
		return _err(str(e), 400)


# ── Controlled Substances ─────────────────────────────────────────────────────

@bp.get("/controlled")
def list_controlled_logs():
	"""List controlled substance logs, optionally filtered by drug_id or action."""
	logs = _run(_svc.list_controlled_logs(
		_tenant(),
		drug_id=request.args.get("drug_id"),
		action=request.args.get("action"),
	))
	return jsonify(_paginate([l.model_dump(mode="json") for l in logs]))


@bp.post("/controlled")
def log_controlled():
	"""Record a controlled substance action (dispense/waste/destroy/count/transfer/receive)."""
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	data.setdefault("created_by", _actor())
	try:
		log = _run(_svc.log_controlled_substance(ControlledSubstanceLogCreate(**data)))
		return jsonify(log.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.post("/controlled/dispense")
def controlled_substance_dispense():
	"""Dispense a controlled substance with mandatory register entry and witness."""
	data = request.get_json(silent=True) or {}
	try:
		record = _run(_svc.controlled_substance_dispense(
			tenant_id=_tenant(),
			prescription_id=data.get("prescription_id", ""),
			schedule=data.get("schedule", ""),
			register_entry=data.get("register_entry", {}),
			dispensed_by=data.get("dispensed_by", _actor()),
			witness_id=data.get("witness_id", ""),
		))
		return jsonify(record), 201
	except (PolicyViolationError, AssertionError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.post("/controlled/reconcile")
def narcotics_reconcile():
	"""Reconcile narcotics register vs physical count."""
	data = request.get_json(silent=True) or {}
	try:
		record = _run(_svc.narcotics_register_reconciliation(
			_tenant(),
			period=data.get("period", ""),
			reconciled_by=data.get("reconciled_by", _actor()),
			witness_id=data.get("witness_id", ""),
		))
		return jsonify(record), 201
	except (PolicyViolationError, AssertionError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


# ── Narcotics Register ────────────────────────────────────────────────────────

@bp.get("/narcotics")
def list_narcotics_register():
	"""List narcotics register entries filtered by drug_id or action."""
	entries = _run(_svc.list_narcotics_register(
		_tenant(),
		drug_id=request.args.get("drug_id"),
		action=request.args.get("action"),
	))
	return jsonify(_paginate([e.model_dump(mode="json") for e in entries]))


@bp.post("/narcotics")
def narcotics_register_entry():
	"""Create a narcotics register entry (receipt/dispense/waste/destroy/transfer/audit)."""
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	data.setdefault("created_by", _actor())
	try:
		entry = _run(_svc.narcotics_register_entry(NarcoticsRegisterEntryCreate(**data)))
		return jsonify(entry.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.get("/narcotics/<entry_id>")
def get_narcotics_entry(entry_id: str):
	"""Get a narcotics register entry by ID."""
	entry = _run(_svc.get_narcotics_entry(_tenant(), entry_id))
	if entry is None:
		return _err("narcotics_entry_not_found", 404)
	return jsonify(entry.model_dump(mode="json"))


# ── Cold Chain ────────────────────────────────────────────────────────────────

@bp.get("/cold-chain")
def list_cold_chain():
	"""List cold chain records, optionally filtered by drug_id or status."""
	records = _run(_svc.list_cold_chain_records(
		_tenant(),
		drug_id=request.args.get("drug_id"),
		status=request.args.get("status"),
	))
	return jsonify(_paginate([r.model_dump(mode="json") for r in records]))


@bp.post("/cold-chain")
def cold_chain_record():
	"""Record cold chain temperature readings for a refrigerated drug lot."""
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	data.setdefault("created_by", _actor())
	# Support both the structured ColdChainRecordCreate and the richer temperature_log variant
	if "temperature_log" in data:
		try:
			record = _run(_svc.cold_chain_record(
				tenant_id=_tenant(),
				drug_id=data.get("drug_id", ""),
				temperature_log=data["temperature_log"],
				recorded_by=data.get("recorded_by", _actor()),
				storage_requirement=data.get("storage_requirement", "2-8C"),
			))
			return jsonify(record), 201
		except (AssertionError, ValueError) as e:
			return _err(str(e), 400)
	try:
		record = _run(_svc.create_cold_chain_record(ColdChainRecordCreate(**data)))
		return jsonify(record.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.get("/cold-chain/<record_id>")
def get_cold_chain_record(record_id: str):
	"""Get a cold chain record by ID."""
	record = _run(_svc.get_cold_chain_record(_tenant(), record_id))
	if record is None:
		return _err("cold_chain_record_not_found", 404)
	return jsonify(record.model_dump(mode="json"))


@bp.post("/cold-chain/monitor")
def cold_chain_monitoring():
	"""Submit a temperature log for a drug and receive excursion analysis."""
	data = request.get_json(silent=True) or {}
	try:
		result = _run(_svc.cold_chain_monitoring(
			tenant_id=_tenant(),
			drug_id=data.get("drug_id", ""),
			temperature_readings=data.get("temperature_readings", []),
			recorded_by=data.get("recorded_by", _actor()),
			storage_requirement=data.get("storage_requirement", "2-8C"),
		))
		return jsonify(result)
	except (AssertionError, ValueError) as e:
		return _err(str(e), 400)


# ── Inventory ─────────────────────────────────────────────────────────────────

@bp.get("/inventory")
def list_inventory():
	"""List inventory items filtered by drug_id or status."""
	items = _run(_svc.list_inventory(
		_tenant(),
		drug_id=request.args.get("drug_id"),
		status=request.args.get("status"),
	))
	return jsonify(_paginate([i.model_dump(mode="json") for i in items]))


@bp.post("/inventory")
def add_inventory():
	"""Add a new drug inventory lot."""
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	data.setdefault("created_by", _actor())
	if "expiry_date" in data and isinstance(data["expiry_date"], str):
		data["expiry_date"] = datetime.fromisoformat(data["expiry_date"])
	try:
		item = _run(_svc.add_inventory_item(InventoryItemCreate(**data)))
		return jsonify(item.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.get("/inventory/<item_id>")
def get_inventory_item(item_id: str):
	"""Get an inventory item by ID."""
	item = _run(_svc.get_inventory_item(_tenant(), item_id))
	if item is None:
		return _err("inventory_item_not_found", 404)
	return jsonify(item.model_dump(mode="json"))


@bp.put("/inventory/<item_id>")
def update_inventory_item(item_id: str):
	"""Update inventory quantity, status, or location."""
	data = request.get_json(silent=True) or {}
	data.setdefault("updated_by", _actor())
	try:
		from .models import InventoryItemUpdate
		item = _run(_svc.update_inventory_item(_tenant(), item_id, InventoryItemUpdate(**data)))
		if item is None:
			return _err("inventory_item_not_found", 404)
		return jsonify(item.model_dump(mode="json"))
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.put("/inventory/<item_id>/status")
def update_inventory_status(item_id: str):
	"""Update inventory status (in_stock/low_stock/recalled/expired/quarantined)."""
	data = request.get_json(silent=True) or {}
	try:
		item = _run(_svc.update_inventory_status(_tenant(), item_id, data.get("status", "")))
		if item is None:
			return _err("inventory_item_not_found", 404)
		return jsonify(item.model_dump(mode="json"))
	except PolicyViolationError as e:
		return _err(str(e), 403)


@bp.delete("/inventory/<item_id>")
def delete_inventory_item(item_id: str):
	"""Soft-delete an inventory lot."""
	result = _run(_svc.soft_delete_inventory(_tenant(), item_id, _actor()))
	if not result:
		return _err("inventory_item_not_found", 404)
	return jsonify({"deleted": True, "id": item_id})


@bp.post("/inventory/count")
def inventory_count():
	"""Record a physical inventory count and reconcile against system quantities."""
	data = request.get_json(silent=True) or {}
	try:
		result = _run(_svc.inventory_count(
			_tenant(),
			location=data.get("location", ""),
			count_data=data.get("count_data", []),
			counted_by=data.get("counted_by", _actor()),
		))
		return jsonify(result)
	except (PolicyViolationError, AssertionError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


# ── Expiry Tracking ───────────────────────────────────────────────────────────

@bp.get("/expiry")
def check_expiry():
	"""Return all lots expiring within threshold_days (default 30)."""
	threshold = int(request.args.get("threshold_days", 30))
	try:
		alerts = _run(_svc.check_expiry_dates(_tenant(), threshold_days=threshold))
		return jsonify({"alerts": alerts, "count": len(alerts), "threshold_days": threshold})
	except AssertionError as e:
		return _err(str(e), 400)


@bp.post("/expiry/scan")
def scan_expiry():
	"""Full expiry scan: identify, classify, and auto-flag expired lots."""
	threshold = int((request.get_json(silent=True) or {}).get("threshold_days", 30))
	alerts = _run(_svc.check_expiry_dates(_tenant(), threshold_days=threshold))
	return jsonify({"alerts": alerts, "count": len(alerts)})


# ── Returned Medications ──────────────────────────────────────────────────────

@bp.get("/returns")
def list_returns():
	"""List returned medications filtered by patient_id or processed status."""
	returns = _run(_svc.list_returned_medications(
		_tenant(),
		patient_id=request.args.get("patient_id"),
		processed=request.args.get("processed"),
	))
	return jsonify(_paginate([r.model_dump(mode="json") for r in returns]))


@bp.post("/returns")
def create_return():
	"""Process an incoming medication return."""
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	data.setdefault("created_by", _actor())
	try:
		ret = _run(_svc.create_returned_medication(ReturnedMedicationCreate(**data)))
		return jsonify(ret.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.get("/returns/<return_id>")
def get_return(return_id: str):
	"""Get a returned medication record by ID."""
	ret = _run(_svc.get_returned_medication(_tenant(), return_id))
	if ret is None:
		return _err("return_not_found", 404)
	return jsonify(ret.model_dump(mode="json"))


@bp.put("/returns/<return_id>")
def update_return(return_id: str):
	"""Update return disposition or processing status."""
	data = request.get_json(silent=True) or {}
	data.setdefault("updated_by", _actor())
	try:
		from .models import ReturnedMedicationUpdate
		ret = _run(_svc.update_returned_medication(_tenant(), return_id, ReturnedMedicationUpdate(**data)))
		if ret is None:
			return _err("return_not_found", 404)
		return jsonify(ret.model_dump(mode="json"))
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.post("/returns/<return_id>/process")
def process_return(return_id: str):
	"""Mark a medication return as processed with disposition."""
	data = request.get_json(silent=True) or {}
	try:
		ret = _run(_svc.process_returned_medication(
			_tenant(), return_id,
			processed_by=data.get("processed_by", _actor()),
			disposition=data.get("disposition", "destroy"),
		))
		if ret is None:
			return _err("return_not_found", 404)
		return jsonify(ret.model_dump(mode="json"))
	except PolicyViolationError as e:
		return _err(str(e), 403)


# ── Reorder ───────────────────────────────────────────────────────────────────

@bp.get("/reorders")
def list_reorders():
	"""List reorder requests filtered by drug_id or status."""
	reorders = _run(_svc.list_reorder_requests(
		_tenant(),
		drug_id=request.args.get("drug_id"),
		status=request.args.get("status"),
	))
	return jsonify(_paginate([r.model_dump(mode="json") for r in reorders]))


@bp.post("/reorders")
def create_reorder():
	"""Create a reorder request for an inventory item."""
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	data.setdefault("created_by", _actor())
	try:
		reorder = _run(_svc.create_reorder_request(ReorderRequestCreate(**data)))
		return jsonify(reorder.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.get("/reorders/<reorder_id>")
def get_reorder(reorder_id: str):
	"""Get a reorder request by ID."""
	reorder = _run(_svc.get_reorder_request(_tenant(), reorder_id))
	if reorder is None:
		return _err("reorder_not_found", 404)
	return jsonify(reorder.model_dump(mode="json"))


@bp.put("/reorders/<reorder_id>")
def update_reorder(reorder_id: str):
	"""Update a reorder request status or quantity received."""
	data = request.get_json(silent=True) or {}
	data.setdefault("updated_by", _actor())
	try:
		from .models import ReorderRequestUpdate
		reorder = _run(_svc.update_reorder_request(_tenant(), reorder_id, ReorderRequestUpdate(**data)))
		if reorder is None:
			return _err("reorder_not_found", 404)
		return jsonify(reorder.model_dump(mode="json"))
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.post("/reorders/auto")
def automated_reorder():
	"""Trigger automated reorder scan across all drugs for this tenant."""
	threshold_multiplier = float((request.get_json(silent=True) or {}).get("threshold_multiplier", 1.0))
	try:
		result = _run(_svc.automated_reorder(_tenant(), threshold_multiplier=threshold_multiplier))
		return jsonify(result)
	except PolicyViolationError as e:
		return _err(str(e), 403)


@bp.post("/reorders/<reorder_id>/submit")
def submit_reorder(reorder_id: str):
	"""Submit a pending reorder to supplier."""
	try:
		reorder = _run(_svc.submit_reorder(_tenant(), reorder_id, _actor()))
		if reorder is None:
			return _err("reorder_not_found", 404)
		return jsonify(reorder.model_dump(mode="json"))
	except PolicyViolationError as e:
		return _err(str(e), 403)


@bp.post("/reorders/<reorder_id>/receive")
def receive_reorder(reorder_id: str):
	"""Record receipt of an ordered stock delivery."""
	data = request.get_json(silent=True) or {}
	try:
		reorder = _run(_svc.receive_reorder(
			_tenant(), reorder_id,
			quantity_received=float(data.get("quantity_received", 0)),
			received_by=data.get("received_by", _actor()),
		))
		if reorder is None:
			return _err("reorder_not_found", 404)
		return jsonify(reorder.model_dump(mode="json"))
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


# ── Prior Auth ────────────────────────────────────────────────────────────────

@bp.get("/prior-auth")
def list_prior_auths():
	"""List prior authorizations filtered by patient_id or status."""
	pas = _run(_svc.list_prior_auths(
		_tenant(),
		patient_id=request.args.get("patient_id"),
		status=request.args.get("status"),
	))
	return jsonify(_paginate([p.model_dump(mode="json") for p in pas]))


@bp.post("/prior-auth")
def request_prior_auth():
	"""Submit a prior authorization request."""
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	data.setdefault("created_by", _actor())
	try:
		pa = _run(_svc.request_prior_auth(PriorAuthCreate(**data)))
		return jsonify(pa.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.get("/prior-auth/<pa_id>")
def get_prior_auth(pa_id: str):
	"""Get a prior authorization by ID."""
	pa = _run(_svc.get_prior_auth(_tenant(), pa_id))
	if pa is None:
		return _err("prior_auth_not_found", 404)
	return jsonify(pa.model_dump(mode="json"))


@bp.delete("/prior-auth/<pa_id>")
def withdraw_prior_auth(pa_id: str):
	"""Withdraw a pending prior authorization."""
	try:
		result = _run(_svc.withdraw_prior_auth(_tenant(), pa_id, _actor()))
		if not result:
			return _err("prior_auth_not_found", 404)
		return jsonify({"withdrawn": True, "id": pa_id})
	except PolicyViolationError as e:
		return _err(str(e), 403)


@bp.post("/prior-auth/<pa_id>/approve")
def approve_prior_auth(pa_id: str):
	"""Approve a prior authorization."""
	data = request.get_json(silent=True) or {}
	pa = _run(_svc.approve_prior_auth(
		_tenant(), pa_id,
		data.get("decision_by", _actor()),
		data.get("expires_in_days", 365),
	))
	if pa is None:
		return _err("prior_auth_not_found", 404)
	return jsonify(pa.model_dump(mode="json"))


@bp.post("/prior-auth/<pa_id>/deny")
def deny_prior_auth(pa_id: str):
	"""Deny a prior authorization with a reason."""
	data = request.get_json(silent=True) or {}
	pa = _run(_svc.deny_prior_auth(
		_tenant(), pa_id,
		data.get("decision_by", _actor()),
		data.get("denial_reason", ""),
	))
	if pa is None:
		return _err("prior_auth_not_found", 404)
	return jsonify(pa.model_dump(mode="json"))


# ── Counselling ───────────────────────────────────────────────────────────────

@bp.get("/counselling")
def list_counselling():
	"""List counselling checklists filtered by patient_id or dispense_order_id."""
	records = _run(_svc.list_counselling_checklists(
		_tenant(),
		patient_id=request.args.get("patient_id"),
		dispense_order_id=request.args.get("dispense_order_id"),
	))
	return jsonify(_paginate([r.model_dump(mode="json") for r in records]))


@bp.post("/counselling")
def create_counselling():
	"""Record a patient counselling checklist for a dispensed prescription."""
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	data.setdefault("created_by", _actor())
	try:
		checklist = _run(_svc.create_counselling_checklist(CounsellingChecklistCreate(**data)))
		return jsonify(checklist.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.get("/counselling/<checklist_id>")
def get_counselling(checklist_id: str):
	"""Get a counselling checklist by ID."""
	record = _run(_svc.get_counselling_checklist(_tenant(), checklist_id))
	if record is None:
		return _err("counselling_checklist_not_found", 404)
	return jsonify(record.model_dump(mode="json"))


@bp.put("/counselling/<checklist_id>")
def update_counselling(checklist_id: str):
	"""Update counselling checklist items."""
	data = request.get_json(silent=True) or {}
	data.setdefault("updated_by", _actor())
	try:
		from .models import CounsellingChecklistUpdate
		record = _run(_svc.update_counselling_checklist(
			_tenant(), checklist_id, CounsellingChecklistUpdate(**data),
		))
		if record is None:
			return _err("counselling_checklist_not_found", 404)
		return jsonify(record.model_dump(mode="json"))
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.post("/counselling/session")
def counselling_session():
	"""Record a full pharmacist counselling session for a prescription."""
	data = request.get_json(silent=True) or {}
	try:
		record = _run(_svc.patient_counselling_checklist(
			_tenant(),
			prescription_id=data.get("prescription_id", ""),
			counselling_points_covered=data.get("counselling_points_covered", []),
			counselled_by=data.get("counselled_by", _actor()),
			patient_understood=data.get("patient_understood", True),
			language=data.get("language", "english"),
		))
		return jsonify(record), 201
	except (AssertionError, ValueError) as e:
		return _err(str(e), 400)


# ── Drug Substitution ─────────────────────────────────────────────────────────

@bp.post("/substitution")
def drug_substitution():
	"""Approve a generic/therapeutic drug substitution."""
	data = request.get_json(silent=True) or {}
	try:
		record = _run(_svc.drug_substitution(
			_tenant(),
			original_drug=data.get("original_drug", ""),
			generic_equivalent=data.get("generic_equivalent", ""),
			pharmacist_approval=data.get("pharmacist_approval", _actor()),
			patient_id=data.get("patient_id", ""),
			therapeutic_equivalence_code=data.get("therapeutic_equivalence_code", "AB"),
		))
		return jsonify(record), 201
	except (PolicyViolationError, AssertionError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.post("/substitution/find-generic")
def find_generic_substitute():
	"""Find a generic substitute for a brand drug from the formulary."""
	data = request.get_json(silent=True) or {}
	try:
		result = _run(_svc.substitute_drug(
			_tenant(),
			drug_id=data.get("drug_id", ""),
			generic=data.get("generic", True),
		))
		return jsonify(result.model_dump(mode="json"))
	except AssertionError as e:
		return _err(str(e), 400)


# ── Pharmacist Verification ───────────────────────────────────────────────────

@bp.post("/verification")
def pharmacist_verification():
	"""Pharmacist clinical verification workflow for a prescription."""
	data = request.get_json(silent=True) or {}
	try:
		record = _run(_svc.pharmacist_verification(
			_tenant(),
			prescription_id=data.get("prescription_id", ""),
			pharmacist_id=data.get("pharmacist_id", _actor()),
			clinical_notes=data.get("clinical_notes", ""),
			override_reason=data.get("override_reason"),
		))
		return jsonify(record), 201
	except (PolicyViolationError, AssertionError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.get("/verification/<verification_id>")
def get_verification(verification_id: str):
	"""Get a pharmacist verification record by ID."""
	record = _run(_svc.get_verification(_tenant(), verification_id))
	if record is None:
		return _err("verification_not_found", 404)
	return jsonify(record)


# ── Clinical Interventions ────────────────────────────────────────────────────

@bp.post("/interventions")
def clinical_intervention():
	"""Record a pharmacist clinical intervention on a prescription."""
	data = request.get_json(silent=True) or {}
	try:
		record = _run(_svc.pharmacist_clinical_intervention(
			_tenant(),
			prescription_id=data.get("prescription_id", ""),
			intervention_type=data.get("intervention_type", ""),
			outcome=data.get("outcome", ""),
			pharmacist_id=data.get("pharmacist_id", _actor()),
			clinical_notes=data.get("clinical_notes", ""),
			prescriber_contacted=data.get("prescriber_contacted", False),
		))
		return jsonify(record), 201
	except (PolicyViolationError, AssertionError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


# ── Supplier Orders ───────────────────────────────────────────────────────────

@bp.post("/supplier-orders")
def supplier_order():
	"""Place a purchase order with a drug supplier."""
	data = request.get_json(silent=True) or {}
	delivery_raw = data.get("delivery_date")
	if isinstance(delivery_raw, str):
		data["delivery_date"] = datetime.fromisoformat(delivery_raw)
	try:
		record = _run(_svc.supplier_order(
			_tenant(),
			drug_ids=data.get("drug_ids", []),
			quantities=data.get("quantities", []),
			supplier_id=data.get("supplier_id", ""),
			delivery_date=data.get("delivery_date", datetime.utcnow()),
			ordered_by=data.get("ordered_by", _actor()),
		))
		return jsonify(record), 201
	except (PolicyViolationError, AssertionError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


# ── Reports ───────────────────────────────────────────────────────────────────

def _parse_report_dates() -> tuple[datetime, datetime]:
	start_raw = request.args.get("period_start")
	end_raw = request.args.get("period_end")
	period_start = datetime.fromisoformat(start_raw) if start_raw else datetime(datetime.utcnow().year, 1, 1)
	period_end = datetime.fromisoformat(end_raw) if end_raw else datetime.utcnow()
	return period_start, period_end


@bp.get("/reports/dispensing-summary")
def report_dispensing_summary():
	"""Dispensing volume, status breakdown, and counselling rate for a period."""
	period_start, period_end = _parse_report_dates()
	report = _run(_svc.dispensing_summary_report(_tenant(), period_start, period_end))
	return jsonify(report.model_dump(mode="json"))


@bp.get("/reports/inventory-valuation")
def report_inventory_valuation():
	"""Inventory valuation, expiry risk, and reorder status."""
	report = _run(_svc.inventory_valuation_report(_tenant()))
	return jsonify(report.model_dump(mode="json"))


@bp.get("/reports/narcotics-audit")
def report_narcotics_audit():
	"""Narcotics register audit — discrepancies, witness compliance, action breakdown."""
	period_start, period_end = _parse_report_dates()
	report = _run(_svc.narcotics_audit_report(_tenant(), period_start, period_end))
	return jsonify(report.model_dump(mode="json"))


@bp.get("/reports/cold-chain")
def report_cold_chain():
	"""Cold chain compliance — excursion rate, affected drugs."""
	period_start, period_end = _parse_report_dates()
	report = _run(_svc.cold_chain_report(_tenant(), period_start, period_end))
	return jsonify(report.model_dump(mode="json"))


@bp.get("/reports/expiry")
def report_expiry():
	"""Full expiry risk report across all inventory lots."""
	threshold = int(request.args.get("threshold_days", 90))
	alerts = _run(_svc.check_expiry_dates(_tenant(), threshold_days=threshold))
	return jsonify({"threshold_days": threshold, "alerts": alerts, "total": len(alerts)})
