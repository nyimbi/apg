"""Flask Blueprint REST API for APG Electronic Medical Records."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from flask import Blueprint, jsonify, request

from .models import (
	AllergyCreate, ClinicalNoteCreate, EncounterCreate,
	MedicationCreate, ProblemCreate, VitalSignCreate,
)
from .service import ElectronicMedicalRecordsService, PolicyViolationError

bp = Blueprint("healthcare_emr", __name__, url_prefix="/api/healthcare/emr")
_svc = ElectronicMedicalRecordsService()


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


# ── encounters ────────────────────────────────────────────────────────────────

@bp.get("/encounters")
def list_encounters():
	patient_id = request.args.get("patient_id")
	encs = _run(_svc.list_encounters(_tenant(), patient_id=patient_id))
	return jsonify({"items": [e.model_dump(mode="json") for e in encs], "count": len(encs)})


@bp.post("/encounters")
def create_encounter():
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	try:
		enc = _run(_svc.create_encounter(EncounterCreate(**data)))
		return jsonify(enc.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.get("/encounters/<enc_id>")
def get_encounter(enc_id: str):
	enc = _run(_svc.get_encounter(_tenant(), enc_id))
	if enc is None:
		return _err("encounter_not_found", 404)
	return jsonify(enc.model_dump(mode="json"))


@bp.post("/encounters/<enc_id>/close")
def close_encounter(enc_id: str):
	data = request.get_json(silent=True) or {}
	enc = _run(_svc.close_encounter(_tenant(), enc_id, data.get("icd10_codes")))
	if enc is None:
		return _err("encounter_not_found", 404)
	return jsonify(enc.model_dump(mode="json"))


# ── notes ─────────────────────────────────────────────────────────────────────

@bp.get("/notes")
def list_notes():
	patient_id = request.args.get("patient_id")
	note_type = request.args.get("note_type")
	notes = _run(_svc.list_notes(_tenant(), patient_id=patient_id, note_type=note_type))
	return jsonify({"items": [n.model_dump(mode="json") for n in notes], "count": len(notes)})


@bp.post("/notes")
def create_note():
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	try:
		note = _run(_svc.create_note(ClinicalNoteCreate(**data)))
		return jsonify(note.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.get("/notes/<note_id>")
def get_note(note_id: str):
	note = _run(_svc.get_note(_tenant(), note_id))
	if note is None:
		return _err("note_not_found", 404)
	return jsonify(note.model_dump(mode="json"))


@bp.post("/notes/<note_id>/amend")
def amend_note(note_id: str):
	data = request.get_json(silent=True) or {}
	try:
		note = _run(_svc.amend_note(_tenant(), note_id, data.get("author_id", ""), data.get("content", "")))
		if note is None:
			return _err("note_not_found", 404)
		return jsonify(note.model_dump(mode="json")), 201
	except PolicyViolationError as e:
		return _err(str(e), 403)


@bp.post("/notes/<note_id>/finalize")
def finalize_note(note_id: str):
	data = request.get_json(silent=True) or {}
	note = _run(_svc.finalize_note(_tenant(), note_id, data.get("cosigned_by")))
	if note is None:
		return _err("note_not_found", 404)
	return jsonify(note.model_dump(mode="json"))


# ── problems ──────────────────────────────────────────────────────────────────

@bp.get("/patients/<patient_id>/problems")
def list_problems(patient_id: str):
	status = request.args.get("status")
	probs = _run(_svc.list_problems(_tenant(), patient_id, status=status))
	return jsonify({"items": [p.model_dump(mode="json") for p in probs], "count": len(probs)})


@bp.post("/problems")
def add_problem():
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	try:
		prob = _run(_svc.add_problem(ProblemCreate(**data)))
		return jsonify(prob.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.post("/problems/<problem_id>/resolve")
def resolve_problem(problem_id: str):
	prob = _run(_svc.resolve_problem(_tenant(), problem_id))
	if prob is None:
		return _err("problem_not_found", 404)
	return jsonify(prob.model_dump(mode="json"))


# ── medications ───────────────────────────────────────────────────────────────

@bp.get("/patients/<patient_id>/medications")
def list_medications(patient_id: str):
	status = request.args.get("status")
	meds = _run(_svc.list_medications(_tenant(), patient_id, status=status))
	return jsonify({"items": [m.model_dump(mode="json") for m in meds], "count": len(meds)})


@bp.post("/medications")
def prescribe_medication():
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	try:
		med = _run(_svc.prescribe_medication(MedicationCreate(**data)))
		return jsonify(med.model_dump(mode="json")), 201
	except PolicyViolationError as e:
		return _err(str(e), 403)
	except Exception as e:
		return _err(str(e))


@bp.post("/medications/<med_id>/discontinue")
def discontinue_medication(med_id: str):
	med = _run(_svc.discontinue_medication(_tenant(), med_id))
	if med is None:
		return _err("medication_not_found", 404)
	return jsonify(med.model_dump(mode="json"))


# ── allergies ─────────────────────────────────────────────────────────────────

@bp.get("/patients/<patient_id>/allergies")
def list_allergies(patient_id: str):
	allergies = _run(_svc.list_allergies(_tenant(), patient_id))
	return jsonify({"items": [a.model_dump(mode="json") for a in allergies], "count": len(allergies)})


@bp.post("/allergies")
def record_allergy():
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	try:
		allergy = _run(_svc.record_allergy(AllergyCreate(**data)))
		return jsonify(allergy.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.get("/patients/<patient_id>/allergy-check/<drug_name>")
def check_drug_allergy(patient_id: str, drug_name: str):
	result = _run(_svc.check_drug_allergy(_tenant(), patient_id, drug_name))
	return jsonify(result)


# ── vitals ────────────────────────────────────────────────────────────────────

@bp.get("/patients/<patient_id>/vitals")
def list_vitals(patient_id: str):
	vital_type = request.args.get("vital_type")
	vitals = _run(_svc.list_vitals(_tenant(), patient_id, vital_type=vital_type))
	return jsonify({"items": [v.model_dump(mode="json") for v in vitals], "count": len(vitals)})


@bp.post("/vitals")
def record_vital():
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	if "recorded_at" in data and isinstance(data["recorded_at"], str):
		data["recorded_at"] = datetime.fromisoformat(data["recorded_at"])
	try:
		vital = _run(_svc.record_vital(VitalSignCreate(**data)))
		return jsonify(vital.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


# ── FHIR export ───────────────────────────────────────────────────────────────

@bp.post("/fhir-export")
def fhir_export():
	data = request.get_json(silent=True) or {}
	patient_id = data.get("patient_id", "")
	resource_types = data.get("resource_types", ["Condition", "MedicationRequest", "AllergyIntolerance"])
	phi_consent = data.get("phi_consent_present", False)
	try:
		bundle = _run(_svc.fhir_export(_tenant(), patient_id, resource_types, phi_consent))
		return jsonify(bundle)
	except PolicyViolationError as e:
		return _err(str(e), 403)


@bp.get("/dashboard")
def dashboard():
	return jsonify(_run(_svc.dashboard_summary(_tenant())))
