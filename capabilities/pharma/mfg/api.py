"""Flask Blueprint REST API for APG Pharma Manufacturing."""

from __future__ import annotations

from datetime import datetime

from flask import Blueprint, jsonify, request

from .models import BatchRecordCreate
from .service import PharmaceuticalManufacturingService

blueprint = Blueprint("pharma_mfg", __name__, url_prefix="/pharma-mfg/api/v1")
_svc = PharmaceuticalManufacturingService()


def _svc_for(tenant_id: str) -> PharmaceuticalManufacturingService:
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


# --- batches ---

@blueprint.get("/batches")
def list_batches():
	tenant_id = request.args.get("tenant_id", "default")
	status = request.args.get("status")
	return jsonify([b.model_dump() for b in _svc_for(tenant_id).list_batches(tenant_id, status=status)])


@blueprint.post("/batches")
def create_batch():
	body = request.get_json(force=True) or {}
	try:
		payload = BatchRecordCreate(**body)
		result = _svc_for(payload.tenant_id).create_batch(payload)
		return jsonify(result.model_dump()), 201
	except (PermissionError, ValueError) as e:
		return _err(str(e))


@blueprint.get("/batches/<batch_id>")
def get_batch(batch_id: str):
	tenant_id = request.args.get("tenant_id", "default")
	try:
		return jsonify(_svc_for(tenant_id).get_batch(batch_id, tenant_id).model_dump())
	except KeyError as e:
		return _err(str(e), 404)


@blueprint.post("/batches/<batch_id>/start")
def start_batch(batch_id: str):
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		result = _svc_for(tenant_id).start_batch(batch_id, tenant_id, body["line_id"])
		return jsonify(result.model_dump())
	except (KeyError, PermissionError) as e:
		return _err(str(e), 404 if isinstance(e, KeyError) else 403)


@blueprint.post("/batches/<batch_id>/release")
def release_batch(batch_id: str):
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		result = _svc_for(tenant_id).release_batch(
			batch_id, tenant_id,
			body["qp_release_reference"], body["electronic_signature_reference"],
		)
		return jsonify(result.model_dump())
	except (KeyError, PermissionError) as e:
		return _err(str(e), 404 if isinstance(e, KeyError) else 403)


@blueprint.post("/batches/<batch_id>/reject")
def reject_batch(batch_id: str):
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		result = _svc_for(tenant_id).reject_batch(batch_id, tenant_id, body.get("rejection_reason", ""))
		return jsonify(result.model_dump())
	except KeyError as e:
		return _err(str(e), 404)


# --- equipment ---

@blueprint.get("/equipment")
def list_equipment():
	tenant_id = request.args.get("tenant_id", "default")
	status = request.args.get("status")
	return jsonify([e.model_dump() for e in _svc_for(tenant_id).list_equipment(tenant_id, status=status)])


@blueprint.post("/equipment")
def register_equipment():
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		result = _svc_for(tenant_id).register_equipment(
			tenant_id, body["equipment_id"], body["name"],
			body["equipment_type"], body["location"],
			body.get("created_by", "system"), body.get("model"), body.get("serial_number"),
		)
		return jsonify(result.model_dump()), 201
	except (KeyError, ValueError) as e:
		return _err(str(e))


@blueprint.post("/equipment/<equipment_id>/qualify")
def qualify_equipment(equipment_id: str):
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		result = _svc_for(tenant_id).qualify_equipment(
			equipment_id, tenant_id, body["qualification_type"],
			body["protocol_reference"], body["report_reference"],
			body.get("performed_by", "system"),
		)
		return jsonify(result.model_dump()), 201
	except (KeyError, PermissionError) as e:
		return _err(str(e), 404 if isinstance(e, KeyError) else 403)


# --- deviations ---

@blueprint.get("/deviations")
def list_deviations():
	tenant_id = request.args.get("tenant_id", "default")
	batch_id = request.args.get("batch_id")
	return jsonify([d.model_dump() for d in _svc_for(tenant_id).list_deviations(tenant_id, batch_id=batch_id)])


@blueprint.post("/deviations")
def raise_deviation():
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		result = _svc_for(tenant_id).raise_deviation(
			tenant_id, body["deviation_number"], body["deviation_type"],
			body["severity"], body["description"], body.get("raised_by", "system"),
			body.get("batch_id"), body.get("equipment_id"),
		)
		return jsonify(result.model_dump()), 201
	except (KeyError, PermissionError, ValueError) as e:
		return _err(str(e))


@blueprint.post("/deviations/<deviation_id>/close")
def close_deviation(deviation_id: str):
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		result = _svc_for(tenant_id).close_deviation(
			deviation_id, tenant_id, body["root_cause"], body.get("capa_reference"),
		)
		return jsonify(result.model_dump())
	except (KeyError, PermissionError) as e:
		return _err(str(e), 404 if isinstance(e, KeyError) else 403)


# --- yield ---

@blueprint.get("/yield")
def list_yields():
	tenant_id = request.args.get("tenant_id", "default")
	batch_id = request.args.get("batch_id")
	return jsonify([y.model_dump() for y in _svc_for(tenant_id).list_yields(tenant_id, batch_id=batch_id)])


@blueprint.post("/yield")
def record_yield():
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		result = _svc_for(tenant_id).record_yield(
			tenant_id, body["batch_id"], body["yield_type"], body["step_name"],
			body["theoretical_quantity"], body["actual_quantity"],
			body.get("created_by", "system"),
		)
		return jsonify(result.model_dump()), 201
	except (KeyError, PermissionError, ValueError) as e:
		return _err(str(e))


@blueprint.post("/yield/reconcile/<batch_id>")
def reconcile_yield(batch_id: str):
	tenant_id = request.args.get("tenant_id", "default")
	return jsonify(_svc_for(tenant_id).reconcile_batch_yield(batch_id, tenant_id))


# --- lines ---

@blueprint.get("/lines")
def list_lines():
	tenant_id = request.args.get("tenant_id", "default")
	return jsonify([l.model_dump() for l in _svc_for(tenant_id).list_lines(tenant_id)])


@blueprint.post("/lines/<line_id>/clear")
def clear_line(line_id: str):
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		result = _svc_for(tenant_id).clear_line(line_id, tenant_id, body.get("cleared_by", "system"))
		return jsonify(result.model_dump())
	except KeyError as e:
		return _err(str(e), 404)


# --- materials ---

@blueprint.get("/materials")
def list_materials():
	tenant_id = request.args.get("tenant_id", "default")
	status = request.args.get("status")
	return jsonify([m.model_dump() for m in _svc_for(tenant_id).list_materials(tenant_id, status=status)])


@blueprint.post("/materials/<material_id>/release")
def release_material(material_id: str):
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		result = _svc_for(tenant_id).release_material(material_id, tenant_id, body["qc_reference"])
		return jsonify(result.model_dump())
	except (KeyError, PermissionError) as e:
		return _err(str(e), 404 if isinstance(e, KeyError) else 403)
