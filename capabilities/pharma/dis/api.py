"""Flask Blueprint REST API for APG Pharma Distribution."""

from __future__ import annotations

from datetime import datetime

from flask import Blueprint, jsonify, request

from .models import ShipmentCreate
from .service import PharmaceuticalDistributionService

blueprint = Blueprint("pharma_dis", __name__, url_prefix="/pharma-dis/api/v1")
_svc = PharmaceuticalDistributionService()


def _svc_for(tenant_id: str) -> PharmaceuticalDistributionService:
	return _svc


def _err(msg: str, status: int = 400) -> tuple:
	return jsonify({"error": msg}), status


def _parse_dt(s: str | None) -> datetime | None:
	if not s:
		return None
	try:
		return datetime.fromisoformat(s)
	except ValueError:
		return None


@blueprint.get("/contract")
def get_contract():
	tenant_id = request.args.get("tenant_id", "default")
	return jsonify(_svc_for(tenant_id).describe(tenant_id))


@blueprint.get("/dashboard")
def dashboard():
	tenant_id = request.args.get("tenant_id", "default")
	return jsonify(_svc_for(tenant_id).dashboard_summary(tenant_id))


# --- shipments ---

@blueprint.get("/shipments")
def list_shipments():
	tenant_id = request.args.get("tenant_id", "default")
	status = request.args.get("status")
	return jsonify([s.model_dump() for s in _svc_for(tenant_id).list_shipments(tenant_id, status=status)])


@blueprint.post("/shipments")
def create_shipment():
	body = request.get_json(force=True) or {}
	try:
		payload = ShipmentCreate(**body)
		result = _svc_for(payload.tenant_id).create_shipment(payload)
		return jsonify(result.model_dump()), 201
	except (PermissionError, ValueError) as e:
		return _err(str(e))


@blueprint.get("/shipments/<shipment_id>")
def get_shipment(shipment_id: str):
	tenant_id = request.args.get("tenant_id", "default")
	try:
		return jsonify(_svc_for(tenant_id).get_shipment(shipment_id, tenant_id).model_dump())
	except KeyError as e:
		return _err(str(e), 404)


@blueprint.post("/shipments/<shipment_id>/dispatch")
def dispatch_shipment(shipment_id: str):
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		result = _svc_for(tenant_id).dispatch_shipment(
			shipment_id, tenant_id,
			body["packing_list_reference"], body["coa_reference"],
			body.get("wda_reference"),
		)
		return jsonify(result.model_dump())
	except (KeyError, PermissionError) as e:
		return _err(str(e), 404 if isinstance(e, KeyError) else 403)


@blueprint.post("/shipments/<shipment_id>/deliver")
def deliver_shipment(shipment_id: str):
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		result = _svc_for(tenant_id).deliver_shipment(
			shipment_id, tenant_id, body.get("serialisation_verified", False)
		)
		return jsonify(result.model_dump())
	except (KeyError, PermissionError) as e:
		return _err(str(e), 404 if isinstance(e, KeyError) else 403)


# --- cold chain ---

@blueprint.post("/cold-chain")
def create_cold_chain():
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		result = _svc_for(tenant_id).create_cold_chain_record(
			tenant_id, body["shipment_id"], body["product_id"],
			body["cold_chain_classification"], body["min_temp_celsius"],
			body["max_temp_celsius"], body["logger_device_id"],
			body["qualification_reference"], body.get("created_by", "system"),
		)
		return jsonify(result.model_dump()), 201
	except (KeyError, PermissionError, ValueError) as e:
		return _err(str(e))


@blueprint.post("/cold-chain/excursions")
def report_excursion():
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		result = _svc_for(tenant_id).report_excursion(
			tenant_id, body["cold_chain_record_id"], body["shipment_id"],
			_parse_dt(body["excursion_start"]) or datetime.utcnow(),
			body["min_recorded"], body["max_recorded"],
			body["severity"], body.get("created_by", "system"),
		)
		return jsonify(result.model_dump()), 201
	except (KeyError, PermissionError, ValueError) as e:
		return _err(str(e))


@blueprint.get("/cold-chain/excursions")
def list_excursions():
	tenant_id = request.args.get("tenant_id", "default")
	shipment_id = request.args.get("shipment_id")
	return jsonify([e.model_dump() for e in _svc_for(tenant_id).list_excursions(tenant_id, shipment_id=shipment_id)])


# --- serialisation ---

@blueprint.post("/serialisation/verify")
def verify_serialisation():
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	serial_number = body.get("serial_number", "")
	return jsonify(_svc_for(tenant_id).verify_serialisation(tenant_id, serial_number))


@blueprint.post("/serialisation")
def serialise_product():
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		result = _svc_for(tenant_id).serialise_product(
			tenant_id, body["product_id"], body["serial_number"],
			body["batch_number"], body["standard"], body["aggregation_level"],
			body.get("gtin"), body.get("created_by", "system"),
		)
		return jsonify(result.model_dump()), 201
	except (KeyError, PermissionError, ValueError) as e:
		return _err(str(e))


# --- recalls ---

@blueprint.get("/recalls")
def list_recalls():
	tenant_id = request.args.get("tenant_id", "default")
	status = request.args.get("status")
	return jsonify([r.model_dump() for r in _svc_for(tenant_id).list_recalls(tenant_id, status=status)])


@blueprint.post("/recalls")
def initiate_recall():
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		result = _svc_for(tenant_id).initiate_recall(
			tenant_id, body["recall_number"], body["recall_class"],
			body["product_id"], body["batch_numbers"], body["reason"],
			body["recall_scope"], body.get("created_by", "system"),
		)
		return jsonify(result.model_dump()), 201
	except (KeyError, PermissionError, ValueError) as e:
		return _err(str(e))


@blueprint.get("/recalls/<recall_id>")
def get_recall(recall_id: str):
	tenant_id = request.args.get("tenant_id", "default")
	recalls = _svc_for(tenant_id).list_recalls(tenant_id)
	recall = next((r for r in recalls if r.id == recall_id), None)
	if recall is None:
		return _err("recall not found", 404)
	return jsonify(recall.model_dump())


@blueprint.post("/recalls/<recall_id>/complete")
def complete_recall(recall_id: str):
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		result = _svc_for(tenant_id).complete_recall(
			recall_id, tenant_id, body.get("units_recalled", 0),
			body.get("units_returned", 0), body.get("effectiveness_check_completed", False),
		)
		return jsonify(result.model_dump())
	except (KeyError, PermissionError) as e:
		return _err(str(e), 404 if isinstance(e, KeyError) else 403)


# --- WDA ---

@blueprint.get("/wda")
def list_wda():
	tenant_id = request.args.get("tenant_id", "default")
	return jsonify([w.model_dump() for w in _svc_for(tenant_id).list_wda(tenant_id)])


@blueprint.post("/wda")
def register_wda():
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		result = _svc_for(tenant_id).register_wda(
			tenant_id, body["wda_number"], body["market"], body["holder_name"],
			body["site_address"], body.get("scope", []),
			body["issuing_authority"], body.get("created_by", "system"),
		)
		return jsonify(result.model_dump()), 201
	except (KeyError, ValueError) as e:
		return _err(str(e))


@blueprint.get("/wda/expiry-alerts")
def wda_expiry_alerts():
	tenant_id = request.args.get("tenant_id", "default")
	return jsonify(_svc_for(tenant_id).check_wda_expiry(tenant_id))


# --- GDP ---

@blueprint.get("/gdp")
def list_gdp_deviations():
	tenant_id = request.args.get("tenant_id", "default")
	return jsonify([d.model_dump() for d in _svc_for(tenant_id).list_gdp_deviations(tenant_id)])
