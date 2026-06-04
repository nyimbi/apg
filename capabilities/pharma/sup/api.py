"""Flask Blueprint REST API for APG Pharma Supply Chain."""

from __future__ import annotations

from datetime import datetime

from flask import Blueprint, jsonify, request

from .models import SupplierCreate
from .service import PharmaceuticalSupplyChainService

blueprint = Blueprint("pharma_sup", __name__, url_prefix="/pharma-sup/api/v1")
_svc = PharmaceuticalSupplyChainService()


def _svc_for(tenant_id: str) -> PharmaceuticalSupplyChainService:
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


# --- suppliers ---

@blueprint.get("/suppliers")
def list_suppliers():
	tenant_id = request.args.get("tenant_id", "default")
	qualified_only = request.args.get("qualified_only", "false").lower() == "true"
	return jsonify([s.model_dump() for s in _svc_for(tenant_id).list_suppliers(tenant_id, qualified_only=qualified_only)])


@blueprint.post("/suppliers")
def create_supplier():
	body = request.get_json(force=True) or {}
	try:
		payload = SupplierCreate(**body)
		result = _svc_for(payload.tenant_id).create_supplier(payload)
		return jsonify(result.model_dump()), 201
	except (PermissionError, ValueError) as e:
		return _err(str(e))


@blueprint.get("/suppliers/<supplier_id>")
def get_supplier(supplier_id: str):
	tenant_id = request.args.get("tenant_id", "default")
	try:
		return jsonify(_svc_for(tenant_id).get_supplier(supplier_id, tenant_id).model_dump())
	except KeyError as e:
		return _err(str(e), 404)


@blueprint.post("/suppliers/<supplier_id>/qualify")
def qualify_supplier(supplier_id: str):
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		result = _svc_for(tenant_id).qualify_supplier(
			supplier_id, tenant_id, body["quality_agreement_reference"],
			_parse_dt(body.get("audit_date")) or datetime.utcnow(),
			body.get("approved_materials", []),
		)
		return jsonify(result.model_dump())
	except (KeyError, PermissionError) as e:
		return _err(str(e), 404 if isinstance(e, KeyError) else 403)


@blueprint.post("/suppliers/<supplier_id>/suspend")
def suspend_supplier(supplier_id: str):
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		result = _svc_for(tenant_id).suspend_supplier(supplier_id, tenant_id, body.get("reason", ""))
		return jsonify(result.model_dump())
	except KeyError as e:
		return _err(str(e), 404)


@blueprint.get("/asl")
def approved_supplier_list():
	tenant_id = request.args.get("tenant_id", "default")
	return jsonify([s.model_dump() for s in _svc_for(tenant_id).list_suppliers(tenant_id, qualified_only=True)])


# --- CMO ---

@blueprint.get("/cmo")
def list_cmos():
	tenant_id = request.args.get("tenant_id", "default")
	active_only = request.args.get("active_only", "true").lower() == "true"
	return jsonify([c.model_dump() for c in _svc_for(tenant_id).list_cmos(tenant_id, active_only=active_only)])


@blueprint.post("/cmo")
def activate_cmo():
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		result = _svc_for(tenant_id).activate_cmo(
			tenant_id, body["cmo_code"], body["name"], body["cmo_type"],
			body["supplier_id"], body["technical_agreement_reference"],
			body["quality_agreement_reference"], body.get("created_by", "system"),
		)
		return jsonify(result.model_dump()), 201
	except (KeyError, PermissionError, ValueError) as e:
		return _err(str(e))


# --- demand planning ---

@blueprint.get("/demand")
def list_forecasts():
	tenant_id = request.args.get("tenant_id", "default")
	product_id = request.args.get("product_id")
	return jsonify([f.model_dump() for f in _svc_for(tenant_id).list_forecasts(tenant_id, product_id=product_id)])


@blueprint.post("/demand")
def create_forecast():
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		result = _svc_for(tenant_id).create_forecast(
			tenant_id, body["forecast_number"], body["product_id"],
			body["method"], body["period"], body.get("forecast_horizon_months", 12),
			body.get("forecasted_demand", {}), body.get("safety_stock", 0.0),
			body.get("created_by", "system"),
		)
		return jsonify(result.model_dump()), 201
	except (KeyError, PermissionError, ValueError) as e:
		return _err(str(e))


# --- import licenses ---

@blueprint.get("/import-licenses")
def list_import_licenses():
	tenant_id = request.args.get("tenant_id", "default")
	return jsonify([l.model_dump() for l in _svc_for(tenant_id).list_import_licenses(tenant_id)])


@blueprint.post("/import-licenses")
def apply_import_license():
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		result = _svc_for(tenant_id).apply_import_license(
			tenant_id, body["license_number"], body["license_type"],
			body["region"], body.get("product_ids", []),
			body["authority_reference"], body["issuing_authority"],
			body["scope"], body.get("created_by", "system"),
		)
		return jsonify(result.model_dump()), 201
	except (KeyError, PermissionError, ValueError) as e:
		return _err(str(e))


@blueprint.post("/import-licenses/<license_id>/grant")
def grant_import_license(license_id: str):
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		result = _svc_for(tenant_id).grant_import_license(
			license_id, tenant_id,
			_parse_dt(body.get("granted_date")) or datetime.utcnow(),
			_parse_dt(body.get("expiry_date")) or datetime.utcnow(),
		)
		return jsonify(result.model_dump())
	except KeyError as e:
		return _err(str(e), 404)


@blueprint.get("/import-licenses/expiry-alerts")
def import_license_expiry_alerts():
	tenant_id = request.args.get("tenant_id", "default")
	return jsonify(_svc_for(tenant_id).check_import_license_expiry(tenant_id))


# --- supply security ---

@blueprint.get("/security")
def list_supply_security():
	tenant_id = request.args.get("tenant_id", "default")
	at_risk_only = request.args.get("at_risk_only", "false").lower() == "true"
	return jsonify([r.model_dump() for r in _svc_for(tenant_id).list_supply_security(tenant_id, at_risk_only=at_risk_only)])


@blueprint.post("/security")
def update_supply_security():
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		result = _svc_for(tenant_id).update_supply_security(
			tenant_id, body["product_id"], body["supply_status"], body["risk_level"],
			body.get("primary_supplier_id"), body.get("created_by", "system"),
			body.get("dual_sourced", False), body.get("inventory_days"),
		)
		return jsonify(result.model_dump())
	except (KeyError, ValueError) as e:
		return _err(str(e))


# --- orders ---

@blueprint.get("/orders")
def list_orders():
	tenant_id = request.args.get("tenant_id", "default")
	supplier_id = request.args.get("supplier_id")
	return jsonify([o.model_dump() for o in _svc_for(tenant_id).list_orders(tenant_id, supplier_id=supplier_id)])


@blueprint.post("/orders")
def place_order():
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		result = _svc_for(tenant_id).place_order(
			tenant_id, body["po_number"], body["order_type"], body["supplier_id"],
			body["product_id"], body["quantity"], body["unit_of_measure"],
			body.get("created_by", "system"), _parse_dt(body.get("expected_delivery")),
			body.get("transport_condition"),
		)
		return jsonify(result.model_dump()), 201
	except (KeyError, PermissionError, ValueError) as e:
		return _err(str(e))


@blueprint.post("/orders/<order_id>/receive")
def receive_order(order_id: str):
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		result = _svc_for(tenant_id).receive_order(order_id, tenant_id, body["coa_reference"])
		return jsonify(result.model_dump())
	except (KeyError, PermissionError) as e:
		return _err(str(e), 404 if isinstance(e, KeyError) else 403)


# --- contracts ---

@blueprint.get("/contracts")
def list_contracts():
	tenant_id = request.args.get("tenant_id", "default")
	supplier_id = request.args.get("supplier_id")
	return jsonify([c.model_dump() for c in _svc_for(tenant_id).list_contracts(tenant_id, supplier_id=supplier_id)])


@blueprint.post("/contracts")
def create_contract():
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		result = _svc_for(tenant_id).create_contract(
			tenant_id, body["contract_number"], body["contract_type"],
			body["supplier_id"], body["title"], body.get("created_by", "system"),
		)
		return jsonify(result.model_dump()), 201
	except (KeyError, ValueError) as e:
		return _err(str(e))


@blueprint.post("/contracts/<contract_id>/approve")
def approve_contract(contract_id: str):
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		result = _svc_for(tenant_id).approve_contract(
			contract_id, tenant_id, body["approval_reference"],
			_parse_dt(body.get("effective_date")) or datetime.utcnow(),
			_parse_dt(body.get("expiry_date")),
		)
		return jsonify(result.model_dump())
	except (KeyError, PermissionError) as e:
		return _err(str(e), 404 if isinstance(e, KeyError) else 403)


@blueprint.get("/contracts/expiry-alerts")
def contract_expiry_alerts():
	tenant_id = request.args.get("tenant_id", "default")
	return jsonify(_svc_for(tenant_id).check_contract_expiry(tenant_id))
