"""Flask Blueprint REST API for APG Pharma Pharmacovigilance."""

from __future__ import annotations

from datetime import datetime

from flask import Blueprint, jsonify, request

from .models import AdvEventCaseCreate
from .service import PharmacovigilanceService

blueprint = Blueprint("pharma_pvi", __name__, url_prefix="/pharma-pvi/api/v1")
_svc = PharmacovigilanceService()


def _svc_for(tenant_id: str) -> PharmacovigilanceService:
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


# --- cases ---

@blueprint.get("/cases")
def list_cases():
	tenant_id = request.args.get("tenant_id", "default")
	status = request.args.get("status")
	serious_only = request.args.get("serious_only", "false").lower() == "true"
	return jsonify([c.model_dump() for c in _svc_for(tenant_id).list_cases(tenant_id, status=status, serious_only=serious_only)])


@blueprint.post("/cases")
def create_case():
	body = request.get_json(force=True) or {}
	try:
		payload = AdvEventCaseCreate(**body)
		result = _svc_for(payload.tenant_id).create_case(payload)
		return jsonify(result.model_dump()), 201
	except (PermissionError, ValueError) as e:
		return _err(str(e))


@blueprint.get("/cases/<case_id>")
def get_case(case_id: str):
	tenant_id = request.args.get("tenant_id", "default")
	try:
		return jsonify(_svc_for(tenant_id).get_case(case_id, tenant_id).model_dump())
	except KeyError as e:
		return _err(str(e), 404)


@blueprint.post("/cases/<case_id>/process")
def process_case(case_id: str):
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		result = _svc_for(tenant_id).process_case(
			case_id, tenant_id, body["narrative"], body["causality"],
			body["meddra_pt"], body["meddra_soc"],
			body.get("processed_by", "system"), body.get("duplicate_check_done", True),
		)
		return jsonify(result.model_dump())
	except (KeyError, PermissionError) as e:
		return _err(str(e), 404 if isinstance(e, KeyError) else 403)


@blueprint.post("/cases/<case_id>/close")
def close_case(case_id: str):
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		result = _svc_for(tenant_id).close_case(
			case_id, tenant_id, body.get("resolution", ""), body.get("medical_reviewed", False),
		)
		return jsonify(result.model_dump())
	except (KeyError, PermissionError) as e:
		return _err(str(e), 404 if isinstance(e, KeyError) else 403)


@blueprint.post("/cases/<case_id>/duplicate")
def mark_duplicate(case_id: str):
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		result = _svc_for(tenant_id).mark_duplicate(case_id, tenant_id, body["duplicate_of"])
		return jsonify(result.model_dump())
	except KeyError as e:
		return _err(str(e), 404)


# --- ICSR submissions ---

@blueprint.get("/submissions")
def list_icsr_submissions():
	tenant_id = request.args.get("tenant_id", "default")
	case_id = request.args.get("case_id")
	return jsonify([s.model_dump() for s in _svc_for(tenant_id).list_icsr_submissions(tenant_id, case_id=case_id)])


@blueprint.post("/submissions")
def submit_icsr():
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		result = _svc_for(tenant_id).submit_icsr(
			tenant_id, body["case_id"], body["regulatory_database"],
			body["submission_type"], _parse_dt(body.get("due_date")) or datetime.utcnow(),
			body.get("e2b_r3_formatted", True), body.get("created_by", "system"),
		)
		return jsonify(result.model_dump()), 201
	except (KeyError, PermissionError, ValueError) as e:
		return _err(str(e))


# --- signals ---

@blueprint.get("/signals")
def list_signals():
	tenant_id = request.args.get("tenant_id", "default")
	product_id = request.args.get("product_id")
	return jsonify([s.model_dump() for s in _svc_for(tenant_id).list_signals(tenant_id, product_id=product_id)])


@blueprint.post("/signals")
def create_signal():
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		result = _svc_for(tenant_id).create_signal(
			tenant_id, body["signal_number"], body["product_id"],
			body["signal_type"], body["meddra_pt"], body["description"],
			body.get("detected_by", "system"), body.get("detection_method", "disproportionality"),
			body.get("created_by", "system"),
		)
		return jsonify(result.model_dump()), 201
	except (KeyError, PermissionError, ValueError) as e:
		return _err(str(e))


@blueprint.get("/signals/<signal_id>")
def get_signal(signal_id: str):
	tenant_id = request.args.get("tenant_id", "default")
	signals = _svc_for(tenant_id).list_signals(tenant_id)
	signal = next((s for s in signals if s.id == signal_id), None)
	if signal is None:
		return _err("signal not found", 404)
	return jsonify(signal.model_dump())


@blueprint.post("/signals/<signal_id>/close")
def close_signal(signal_id: str):
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		result = _svc_for(tenant_id).close_signal(
			signal_id, tenant_id, body.get("clinical_reviewed", False), body.get("closure_reason", "")
		)
		return jsonify(result.model_dump())
	except (KeyError, PermissionError) as e:
		return _err(str(e), 404 if isinstance(e, KeyError) else 403)


# --- PSUR ---

@blueprint.get("/psur")
def list_psur():
	tenant_id = request.args.get("tenant_id", "default")
	product_id = request.args.get("product_id")
	return jsonify([p.model_dump() for p in _svc_for(tenant_id).list_psur_reports(tenant_id, product_id=product_id)])


@blueprint.post("/psur")
def create_psur():
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		result = _svc_for(tenant_id).create_psur(
			tenant_id, body["report_number"], body["product_id"], body["report_type"],
			_parse_dt(body["data_lock_point"]) or datetime.utcnow(),
			_parse_dt(body["international_birth_date"]) or datetime.utcnow(),
			_parse_dt(body["period_start"]) or datetime.utcnow(),
			_parse_dt(body["period_end"]) or datetime.utcnow(),
			body["ibrd_reference"], body.get("created_by", "system"),
		)
		return jsonify(result.model_dump()), 201
	except (KeyError, PermissionError, ValueError) as e:
		return _err(str(e))


@blueprint.post("/psur/<psur_id>/submit")
def submit_psur(psur_id: str):
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		result = _svc_for(tenant_id).submit_psur(psur_id, tenant_id, body.get("benefit_risk_assessed", False))
		return jsonify(result.model_dump())
	except (KeyError, PermissionError) as e:
		return _err(str(e), 404 if isinstance(e, KeyError) else 403)


# --- follow-ups ---

@blueprint.get("/follow-ups")
def list_follow_ups():
	tenant_id = request.args.get("tenant_id", "default")
	case_id = request.args.get("case_id")
	pending_only = request.args.get("pending_only", "false").lower() == "true"
	return jsonify([f.model_dump() for f in _svc_for(tenant_id).list_follow_ups(tenant_id, case_id=case_id, pending_only=pending_only)])


@blueprint.post("/follow-ups")
def request_follow_up():
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		result = _svc_for(tenant_id).request_follow_up(
			tenant_id, body["case_id"], body["follow_up_type"],
			body["requested_from"], _parse_dt(body["due_date"]) or datetime.utcnow(),
			body.get("created_by", "system"),
		)
		return jsonify(result.model_dump()), 201
	except (KeyError, PermissionError, ValueError) as e:
		return _err(str(e))


# --- literature ---

@blueprint.get("/literature")
def list_literature():
	tenant_id = request.args.get("tenant_id", "default")
	relevant_only = request.args.get("relevant_only", "false").lower() == "true"
	return jsonify([l.model_dump() for l in _svc_for(tenant_id).list_literature(tenant_id, relevant_only=relevant_only)])
