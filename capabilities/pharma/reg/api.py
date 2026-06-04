"""Flask Blueprint REST API for APG Pharma Product Registration."""

from __future__ import annotations

from datetime import datetime

from flask import Blueprint, jsonify, request

from .models import ProductRegistrationCreate
from .service import ProductRegistrationService

blueprint = Blueprint("pharma_reg", __name__, url_prefix="/pharma-reg/api/v1")
_svc = ProductRegistrationService()


def _svc_for(tenant_id: str) -> ProductRegistrationService:
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


# --- registrations ---

@blueprint.get("/registrations")
def list_registrations():
	tenant_id = request.args.get("tenant_id", "default")
	region = request.args.get("region")
	status = request.args.get("status")
	return jsonify([r.model_dump() for r in _svc_for(tenant_id).list_registrations(tenant_id, region=region, status=status)])


@blueprint.post("/registrations")
def create_registration():
	body = request.get_json(force=True) or {}
	try:
		payload = ProductRegistrationCreate(**body)
		result = _svc_for(payload.tenant_id).create_registration(payload)
		return jsonify(result.model_dump()), 201
	except (PermissionError, ValueError) as e:
		return _err(str(e))


@blueprint.get("/registrations/<reg_id>")
def get_registration(reg_id: str):
	tenant_id = request.args.get("tenant_id", "default")
	try:
		return jsonify(_svc_for(tenant_id).get_registration(reg_id, tenant_id).model_dump())
	except KeyError as e:
		return _err(str(e), 404)


@blueprint.put("/registrations/<reg_id>")
def submit_registration(reg_id: str):
	"""Submit a registration (advances to submitted state)."""
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		result = _svc_for(tenant_id).submit_registration(
			reg_id, tenant_id, body["dossier_id"],
			body["local_representative_id"],
			body.get("qp_signed_off", False), body.get("ectd_validated", False),
		)
		return jsonify(result.model_dump())
	except (KeyError, PermissionError) as e:
		return _err(str(e), 404 if isinstance(e, KeyError) else 403)


@blueprint.post("/registrations/<reg_id>/approve")
def approve_registration(reg_id: str):
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		result = _svc_for(tenant_id).approve_registration(
			reg_id, tenant_id, body["registration_number"],
			_parse_dt(body.get("approval_date")) or datetime.utcnow(),
			_parse_dt(body.get("expiry_date")),
			body.get("conditions"),
		)
		return jsonify(result.model_dump())
	except KeyError as e:
		return _err(str(e), 404)


@blueprint.get("/registrations/renewal-alerts")
def renewal_alerts():
	tenant_id = request.args.get("tenant_id", "default")
	return jsonify(_svc_for(tenant_id).check_renewal_alerts(tenant_id))


# --- dossiers ---

@blueprint.get("/dossiers")
def list_dossiers():
	tenant_id = request.args.get("tenant_id", "default")
	product_id = request.args.get("product_id")
	return jsonify([d.model_dump() for d in _svc_for(tenant_id).list_dossiers(tenant_id, product_id=product_id)])


@blueprint.post("/dossiers")
def compile_dossier():
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		result = _svc_for(tenant_id).compile_dossier(
			tenant_id, body["dossier_number"], body["product_id"],
			body["format"], body["version"], body.get("modules_present", []),
			body.get("created_by", "system"),
		)
		return jsonify(result.model_dump()), 201
	except (KeyError, PermissionError, ValueError) as e:
		return _err(str(e))


@blueprint.post("/dossiers/<dossier_id>/validate-ectd")
def validate_ectd(dossier_id: str):
	tenant_id = request.args.get("tenant_id", "default")
	try:
		result = _svc_for(tenant_id).validate_ectd(dossier_id, tenant_id)
		return jsonify(result.model_dump())
	except KeyError as e:
		return _err(str(e), 404)


# --- authority interactions ---

@blueprint.get("/interactions")
def list_interactions():
	tenant_id = request.args.get("tenant_id", "default")
	registration_id = request.args.get("registration_id")
	return jsonify([i.model_dump() for i in _svc_for(tenant_id).list_interactions(tenant_id, registration_id=registration_id)])


@blueprint.post("/interactions")
def record_interaction():
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		result = _svc_for(tenant_id).record_interaction(
			tenant_id, body["registration_id"], body["interaction_type"],
			body["authority"], _parse_dt(body.get("interaction_date")) or datetime.utcnow(),
			body.get("created_by", "system"), body.get("minutes_reference"),
			body.get("participants"),
		)
		return jsonify(result.model_dump()), 201
	except (KeyError, PermissionError, ValueError) as e:
		return _err(str(e))


# --- variations ---

@blueprint.get("/variations")
def list_variations():
	tenant_id = request.args.get("tenant_id", "default")
	registration_id = request.args.get("registration_id")
	return jsonify([v.model_dump() for v in _svc_for(tenant_id).list_variations(tenant_id, registration_id=registration_id)])


@blueprint.post("/variations")
def file_variation():
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		result = _svc_for(tenant_id).file_variation(
			tenant_id, body["variation_number"], body["registration_id"],
			body["variation_type"], body["description"],
			body.get("impact_assessed", False), body.get("created_by", "system"),
		)
		return jsonify(result.model_dump()), 201
	except (KeyError, PermissionError, ValueError) as e:
		return _err(str(e))


# --- certificates ---

@blueprint.get("/certificates")
def list_certificates():
	tenant_id = request.args.get("tenant_id", "default")
	product_id = request.args.get("product_id")
	return jsonify([c.model_dump() for c in _svc_for(tenant_id).list_certificates(tenant_id, product_id=product_id)])


@blueprint.post("/certificates")
def store_certificate():
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		result = _svc_for(tenant_id).store_certificate(
			tenant_id, body["certificate_number"], body["registration_id"],
			body["product_id"], body["region"], body["authority"],
			_parse_dt(body.get("issued_date")) or datetime.utcnow(),
			body["storage_reference"], body.get("created_by", "system"),
			_parse_dt(body.get("expiry_date")), body.get("conditions"),
		)
		return jsonify(result.model_dump()), 201
	except (KeyError, ValueError) as e:
		return _err(str(e))


# --- procedures ---

@blueprint.get("/procedures")
def list_procedures():
	tenant_id = request.args.get("tenant_id", "default")
	registration_id = request.args.get("registration_id")
	return jsonify([p.model_dump() for p in _svc_for(tenant_id).list_procedures(tenant_id, registration_id=registration_id)])


@blueprint.post("/procedures")
def initiate_procedure():
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		result = _svc_for(tenant_id).initiate_procedure(
			tenant_id, body["procedure_number"], body["registration_id"],
			body["procedure_type"], body.get("created_by", "system"),
			body.get("reference_member_state"), body.get("concerned_member_states"),
		)
		return jsonify(result.model_dump()), 201
	except (KeyError, PermissionError, ValueError) as e:
		return _err(str(e))
