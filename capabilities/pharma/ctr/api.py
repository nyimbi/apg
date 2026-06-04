"""Flask Blueprint REST API for APG Pharma Clinical Trials Management."""

from __future__ import annotations

from datetime import datetime

from flask import Blueprint, jsonify, request

from .models import AdverseEventCreate, ClinicalTrialCreate, TrialPatientCreate, TrialSiteCreate
from .service import ClinicalTrialsService

blueprint = Blueprint("pharma_ctr", __name__, url_prefix="/pharma-ctr/api/v1")
_svc = ClinicalTrialsService()


def _svc_for(tenant_id: str) -> ClinicalTrialsService:
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


# --- contract ---

@blueprint.get("/contract")
def get_contract():
	tenant_id = request.args.get("tenant_id", "default")
	return jsonify(_svc_for(tenant_id).describe(tenant_id))


@blueprint.get("/dashboard")
def dashboard():
	tenant_id = request.args.get("tenant_id", "default")
	return jsonify(_svc_for(tenant_id).dashboard_summary(tenant_id))


# --- trials ---

@blueprint.get("/trials")
def list_trials():
	"""List clinical trials."""
	tenant_id = request.args.get("tenant_id", "default")
	phase = request.args.get("phase")
	return jsonify([t.model_dump() for t in _svc_for(tenant_id).list_trials(tenant_id, phase=phase)])


@blueprint.post("/trials")
def create_trial():
	"""Create a new clinical trial."""
	body = request.get_json(force=True) or {}
	try:
		payload = ClinicalTrialCreate(**body)
		result = _svc_for(payload.tenant_id).create_trial(payload)
		return jsonify(result.model_dump()), 201
	except (PermissionError, ValueError) as e:
		return _err(str(e))


@blueprint.get("/trials/<trial_id>")
def get_trial(trial_id: str):
	"""Get a trial by ID."""
	tenant_id = request.args.get("tenant_id", "default")
	try:
		return jsonify(_svc_for(tenant_id).get_trial(trial_id, tenant_id).model_dump())
	except KeyError as e:
		return _err(str(e), 404)


@blueprint.post("/trials/<trial_id>/activate")
def activate_trial(trial_id: str):
	"""Activate a trial."""
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	irb_ref = body.get("irb_approval_reference", "")
	try:
		result = _svc_for(tenant_id).activate_trial(trial_id, tenant_id, irb_ref)
		return jsonify(result.model_dump())
	except (KeyError, PermissionError) as e:
		return _err(str(e), 404 if isinstance(e, KeyError) else 403)


# --- protocols ---

@blueprint.get("/protocols")
def list_protocols():
	tenant_id = request.args.get("tenant_id", "default")
	trial_id = request.args.get("trial_id")
	return jsonify([p.model_dump() for p in _svc_for(tenant_id).list_protocols(tenant_id, trial_id=trial_id)])


@blueprint.post("/protocols")
def create_protocol():
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		result = _svc_for(tenant_id).create_protocol(
			tenant_id, body["trial_id"], body["version"], body.get("created_by", "system")
		)
		return jsonify(result.model_dump()), 201
	except (KeyError, PermissionError, ValueError) as e:
		return _err(str(e))


@blueprint.post("/protocols/<protocol_id>/approve")
def approve_protocol(protocol_id: str):
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		result = _svc_for(tenant_id).approve_protocol(
			protocol_id, tenant_id, body.get("irb_approval_reference", ""),
			body.get("approved_by", "system")
		)
		return jsonify(result.model_dump())
	except (KeyError, PermissionError) as e:
		return _err(str(e), 404 if isinstance(e, KeyError) else 403)


# --- sites ---

@blueprint.get("/sites")
def list_sites():
	tenant_id = request.args.get("tenant_id", "default")
	trial_id = request.args.get("trial_id")
	return jsonify([s.model_dump() for s in _svc_for(tenant_id).list_sites(tenant_id, trial_id=trial_id)])


@blueprint.post("/sites")
def select_site():
	body = request.get_json(force=True) or {}
	try:
		payload = TrialSiteCreate(**body)
		result = _svc_for(payload.tenant_id).select_site(payload)
		return jsonify(result.model_dump()), 201
	except (PermissionError, ValueError) as e:
		return _err(str(e))


@blueprint.post("/sites/<site_id>/initiate")
def initiate_site(site_id: str):
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	initiation_date = _parse_dt(body.get("initiation_visit_date")) or datetime.utcnow()
	try:
		result = _svc_for(tenant_id).initiate_site(site_id, tenant_id, initiation_date, body.get("initiated_by", "system"))
		return jsonify(result.model_dump())
	except (KeyError, PermissionError) as e:
		return _err(str(e), 404 if isinstance(e, KeyError) else 403)


# --- patients ---

@blueprint.get("/patients")
def list_patients():
	tenant_id = request.args.get("tenant_id", "default")
	trial_id = request.args.get("trial_id")
	site_id = request.args.get("site_id")
	return jsonify([p.model_dump() for p in _svc_for(tenant_id).list_patients(tenant_id, trial_id=trial_id, site_id=site_id)])


@blueprint.post("/patients/enrol")
def enrol_patient():
	body = request.get_json(force=True) or {}
	try:
		payload = TrialPatientCreate(**{k: v for k, v in body.items() if k != "informed_consent_date"})
		ic_date = _parse_dt(body.get("informed_consent_date")) or datetime.utcnow()
		result = _svc_for(payload.tenant_id).enrol_patient(payload, ic_date)
		return jsonify(result.model_dump()), 201
	except (PermissionError, ValueError) as e:
		return _err(str(e))


@blueprint.post("/patients/<patient_id>/randomise")
def randomise_patient(patient_id: str):
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		result = _svc_for(tenant_id).randomise_patient(
			patient_id, tenant_id, body["trial_id"],
			body["randomisation_method"], body["treatment_arm"],
			body["randomisation_code"], body.get("randomised_by", "system"),
			body.get("stratification_factors"),
		)
		return jsonify(result.model_dump()), 201
	except (KeyError, PermissionError, ValueError) as e:
		return _err(str(e))


# --- adverse events ---

@blueprint.get("/adverse-events")
def list_adverse_events():
	tenant_id = request.args.get("tenant_id", "default")
	trial_id = request.args.get("trial_id")
	serious_only = request.args.get("serious_only", "false").lower() == "true"
	return jsonify([ae.model_dump() for ae in _svc_for(tenant_id).list_adverse_events(tenant_id, trial_id=trial_id, serious_only=serious_only)])


@blueprint.post("/adverse-events")
def report_ae():
	body = request.get_json(force=True) or {}
	try:
		payload = AdverseEventCreate(**body)
		result = _svc_for(payload.tenant_id).report_ae(payload)
		return jsonify(result.model_dump()), 201
	except (PermissionError, ValueError) as e:
		return _err(str(e))


# --- submissions ---

@blueprint.get("/submissions")
def list_submissions():
	tenant_id = request.args.get("tenant_id", "default")
	trial_id = request.args.get("trial_id")
	return jsonify([s.model_dump() for s in _svc_for(tenant_id).list_submissions(tenant_id, trial_id=trial_id)])


@blueprint.post("/submissions")
def file_submission():
	body = request.get_json(force=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		result = _svc_for(tenant_id).file_submission(
			tenant_id, body["trial_id"], body["submission_type"], body["authority"],
			body["cover_letter_reference"], body["dossier_reference"],
			body.get("created_by", "system"),
		)
		return jsonify(result.model_dump()), 201
	except (KeyError, PermissionError, ValueError) as e:
		return _err(str(e))
