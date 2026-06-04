"""Flask Blueprint REST API for APG Pharma Quality Management System."""

from __future__ import annotations

from datetime import datetime

from flask import Blueprint, jsonify, request

from .models import CapaCreate, ChangeControlCreate
from .service import QualityManagementService

blueprint = Blueprint("pharma_qms", __name__, url_prefix="/pharma-qms/api/v1")
_svc = QualityManagementService()


def _svc_for(tenant_id: str) -> QualityManagementService:
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


# --- change control ---

@blueprint.get("/change-control")
def list_changes():
	tenant_id = request.args.get("tenant_id", "default")
	status = request.args.get("status")
	return jsonify([c.model_dump() for c in _svc_for(tenant_id).list_changes(tenant_id, status=status)])


@blueprint.post("/change-control")
def initiate_change():
	body = request.get_json(force=True) or {}
	try:
		payload = ChangeControlCreate(**body)
		result = _svc_for(payload.tenant_id).initiate_change(payload)
		return jsonify(result.model_dump()), 201
	except (PermissionError, ValueError) as e:
		return _err(str(e))


@blueprint.get("/change-control/<change_id>")
def get_change(change_id: str):
	tenant_id = request.args.get("tenant_id", "default")
	changes = _svc_for(tenant_id).list_changes(tenant_id)
	change = next((c for c in changes if c.id == change_id), None)
	if change is None:
		return _err("change not found", 404)
	return jsonify(change.model_dump())


@blueprint.post("/change-control/<change_id>/approve")
def approve_change(change_id: str):
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		result = _svc_for(tenant_id).approve_change(
			change_id, tenant_id, body["approval_reference"],
			body.get("impact_assessed", False), body.get("risk_assessed", False),
		)
		return jsonify(result.model_dump())
	except (KeyError, PermissionError) as e:
		return _err(str(e), 404 if isinstance(e, KeyError) else 403)


@blueprint.post("/change-control/<change_id>/implement")
def implement_change(change_id: str):
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		impl_date = _parse_dt(body.get("implementation_date")) or datetime.utcnow()
		result = _svc_for(tenant_id).implement_change(change_id, tenant_id, impl_date)
		return jsonify(result.model_dump())
	except (KeyError, PermissionError) as e:
		return _err(str(e), 404 if isinstance(e, KeyError) else 403)


@blueprint.post("/change-control/<change_id>/close")
def close_change(change_id: str):
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		result = _svc_for(tenant_id).close_change(
			change_id, tenant_id, body.get("effectiveness_checked", False),
			body.get("effectiveness_reference", ""),
		)
		return jsonify(result.model_dump())
	except (KeyError, PermissionError) as e:
		return _err(str(e), 404 if isinstance(e, KeyError) else 403)


# --- CAPA ---

@blueprint.get("/capa")
def list_capas():
	tenant_id = request.args.get("tenant_id", "default")
	status = request.args.get("status")
	return jsonify([c.model_dump() for c in _svc_for(tenant_id).list_capas(tenant_id, status=status)])


@blueprint.post("/capa")
def create_capa():
	body = request.get_json(force=True) or {}
	try:
		payload = CapaCreate(**body)
		result = _svc_for(payload.tenant_id).create_capa(payload)
		return jsonify(result.model_dump()), 201
	except (PermissionError, ValueError) as e:
		return _err(str(e))


@blueprint.get("/capa/<capa_id>")
def get_capa(capa_id: str):
	tenant_id = request.args.get("tenant_id", "default")
	capas = _svc_for(tenant_id).list_capas(tenant_id)
	capa = next((c for c in capas if c.id == capa_id), None)
	if capa is None:
		return _err("capa not found", 404)
	return jsonify(capa.model_dump())


@blueprint.post("/capa/<capa_id>/close")
def close_capa(capa_id: str):
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		result = _svc_for(tenant_id).close_capa(
			capa_id, tenant_id, body["root_cause"],
			body.get("root_cause_method", "5_why"),
			body.get("effectiveness_checked", False),
			body.get("effectiveness_result", "effective"),
		)
		return jsonify(result.model_dump())
	except (KeyError, PermissionError) as e:
		return _err(str(e), 404 if isinstance(e, KeyError) else 403)


# --- deviations ---

@blueprint.get("/deviations")
def list_deviations():
	tenant_id = request.args.get("tenant_id", "default")
	status = request.args.get("status")
	return jsonify([d.model_dump() for d in _svc_for(tenant_id).list_deviations(tenant_id, status=status)])


@blueprint.post("/deviations")
def raise_deviation():
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		result = _svc_for(tenant_id).raise_deviation(
			tenant_id, body["deviation_number"], body["deviation_type"],
			body["severity"], body["description"], body.get("raised_by", "system"),
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


# --- documents ---

@blueprint.get("/documents")
def list_documents():
	tenant_id = request.args.get("tenant_id", "default")
	document_type = request.args.get("document_type")
	status = request.args.get("status")
	return jsonify([d.model_dump() for d in _svc_for(tenant_id).list_documents(tenant_id, document_type=document_type, status=status)])


@blueprint.post("/documents")
def create_document():
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		result = _svc_for(tenant_id).create_document(
			tenant_id, body["document_number"], body["title"],
			body["document_type"], body["version"], body["department"],
			body["owner_id"], body.get("created_by", "system"),
		)
		return jsonify(result.model_dump()), 201
	except (KeyError, PermissionError, ValueError) as e:
		return _err(str(e))


@blueprint.get("/documents/<doc_id>")
def get_document(doc_id: str):
	tenant_id = request.args.get("tenant_id", "default")
	docs = _svc_for(tenant_id).list_documents(tenant_id)
	doc = next((d for d in docs if d.id == doc_id), None)
	if doc is None:
		return _err("document not found", 404)
	return jsonify(doc.model_dump())


@blueprint.post("/documents/<doc_id>/approve")
def approve_document(doc_id: str):
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		result = _svc_for(tenant_id).approve_document(doc_id, tenant_id, body["approver_id"])
		return jsonify(result.model_dump())
	except (KeyError, PermissionError) as e:
		return _err(str(e), 404 if isinstance(e, KeyError) else 403)


# --- audits ---

@blueprint.get("/audits")
def list_audits():
	tenant_id = request.args.get("tenant_id", "default")
	audit_type = request.args.get("audit_type")
	return jsonify([a.model_dump() for a in _svc_for(tenant_id).list_audits(tenant_id, audit_type=audit_type)])


@blueprint.post("/audits")
def create_audit():
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		result = _svc_for(tenant_id).create_audit(
			tenant_id, body["audit_number"], body["audit_type"],
			body["auditee"], body.get("auditor_ids", []), body["scope"],
			body.get("created_by", "system"),
			_parse_dt(body.get("planned_date")),
		)
		return jsonify(result.model_dump()), 201
	except (KeyError, PermissionError, ValueError) as e:
		return _err(str(e))


@blueprint.post("/audits/<audit_id>/close")
def close_audit(audit_id: str):
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		result = _svc_for(tenant_id).close_audit(
			audit_id, tenant_id, body["report_reference"],
			body.get("findings_count", 0), body.get("capa_references", []),
		)
		return jsonify(result.model_dump())
	except (KeyError, PermissionError) as e:
		return _err(str(e), 404 if isinstance(e, KeyError) else 403)


# --- validation ---

@blueprint.get("/validation")
def list_validations():
	tenant_id = request.args.get("tenant_id", "default")
	return jsonify([v.model_dump() for v in _svc_for(tenant_id).list_validations(tenant_id)])


@blueprint.post("/validation")
def create_validation():
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		result = _svc_for(tenant_id).create_validation(
			tenant_id, body["validation_number"], body["validation_type"],
			body["subject"], body.get("created_by", "system"),
		)
		return jsonify(result.model_dump()), 201
	except (KeyError, PermissionError, ValueError) as e:
		return _err(str(e))


# --- risk ---

@blueprint.get("/risk")
def list_risks():
	tenant_id = request.args.get("tenant_id", "default")
	risk_level = request.args.get("risk_level")
	return jsonify([r.model_dump() for r in _svc_for(tenant_id).list_risks(tenant_id, risk_level=risk_level)])
