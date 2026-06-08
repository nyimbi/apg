"""Flask Blueprint REST API for APG Electronic Medical Records.

All endpoints are tenant-scoped via X-Tenant-ID header.
Async service methods are driven via asyncio.run().
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from flask import Blueprint, jsonify, request

from .models import (
	AllergyCreate,
	CarePlanCreate, CarePlanUpdate,
	ClinicalNoteCreate, ClinicalNoteUpdate,
	ConsentCreate,
	DiagnosisCreate,
	EncounterCreate, EncounterUpdate,
	FamilyHistoryCreate,
	ImagingOrderCreate,
	ImmunisationCreate,
	LabOrderCreate, LabResultCreate,
	MedicationCreate,
	PatientCreate, PatientUpdate,
	PrescriptionCreate, PrescriptionUpdate,
	ProblemCreate,
	ReferralCreate, ReferralUpdate,
	VitalSignCreate,
)
from .service import EMRService, PolicyViolationError, DrugSafetyError

bp = Blueprint("healthcare_emr", __name__, url_prefix="/api/healthcare/emr")


# ── helpers ───────────────────────────────────────────────────────────────────

def _run(coro: Any) -> Any:
	return asyncio.run(coro)


def _err(msg: str, status: int = 400) -> Any:
	return jsonify({"error": msg}), status


def _tenant() -> str:
	return request.headers.get("X-Tenant-ID", request.args.get("tenant_id", "default"))


def _actor() -> str:
	return request.headers.get("X-Actor-ID", request.args.get("actor_id", "api"))


def _svc() -> EMRService:
	return EMRService(tenant_id=_tenant(), actor_id=_actor())


def _page() -> tuple[int, int]:
	"""Return (page, page_size) from query params."""
	try:
		page = max(1, int(request.args.get("page", 1)))
		size = min(500, max(1, int(request.args.get("page_size", 50))))
	except (TypeError, ValueError):
		page, size = 1, 50
	return page, size


def _paginate(items: list[Any], page: int, size: int) -> dict[str, Any]:
	start = (page - 1) * size
	slice_ = items[start : start + size]
	return {
		"items": slice_,
		"total": len(items),
		"page": page,
		"page_size": size,
		"pages": max(1, (len(items) + size - 1) // size),
	}


def _handle(coro: Any) -> Any:
	"""Run a coroutine and map common exceptions to HTTP responses."""
	try:
		return coro
	except PolicyViolationError as e:
		return _err(str(e), 403)
	except DrugSafetyError as e:
		return _err(str(e), 422)
	except ValueError as e:
		return _err(str(e), 400)


# ── capability contract ───────────────────────────────────────────────────────

@bp.get("/contract")
def get_contract():
	"""Return the capability contract for the current tenant."""
	return jsonify(_run(_svc().describe(_tenant())))


# ── patients ──────────────────────────────────────────────────────────────────

@bp.get("/patients")
def list_patients():
	"""List patients with optional filters and pagination."""
	svc = _svc()
	page, size = _page()
	status = request.args.get("status")
	search = request.args.get("search")
	try:
		patients = _run(svc.list_patients(
			_tenant(),
			status=status,
			search=search,
		))
		items = [p.model_dump(mode="json") for p in patients]
		return jsonify(_paginate(items, page, size))
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.post("/patients")
def register_patient():
	"""Register a new patient with biometric dedup check."""
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	data.setdefault("created_by", _actor())
	try:
		patient = _run(_svc().register_patient(PatientCreate(**data)))
		return jsonify(patient.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.get("/patients/<patient_id>")
def get_patient(patient_id: str):
	p = _run(_svc().get_patient(_tenant(), patient_id))
	if p is None:
		return _err("patient_not_found", 404)
	return jsonify(p.model_dump(mode="json"))


@bp.put("/patients/<patient_id>")
def update_patient(patient_id: str):
	data = request.get_json(silent=True) or {}
	try:
		p = _run(_svc().update_patient(_tenant(), patient_id, PatientUpdate(**data)))
		if p is None:
			return _err("patient_not_found", 404)
		return jsonify(p.model_dump(mode="json"))
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.delete("/patients/<patient_id>")
def delete_patient(patient_id: str):
	"""Soft delete (mark is_deleted=True)."""
	ok = _run(_svc().delete_patient(_tenant(), patient_id))
	if not ok:
		return _err("patient_not_found", 404)
	return jsonify({"deleted": True, "id": patient_id})


@bp.post("/patients/<patient_id>/merge")
def merge_patients(patient_id: str):
	"""Merge a duplicate patient into the canonical record."""
	data = request.get_json(silent=True) or {}
	surviving_id = data.get("surviving_id")
	if not surviving_id:
		return _err("surviving_id required", 400)
	try:
		result = _run(_svc().merge_patients(_tenant(), duplicate_id=patient_id, surviving_id=surviving_id))
		return jsonify(result)
	except ValueError as e:
		return _err(str(e), 400)


@bp.post("/patients/dedup-check")
def dedup_check():
	"""Run probabilistic dedup check against incoming patient data."""
	data = request.get_json(silent=True) or {}
	try:
		candidates = _run(_svc().patient_deduplication_check(data))
		return jsonify({"candidates": [c.model_dump() for c in candidates]})
	except ValueError as e:
		return _err(str(e), 400)


@bp.get("/patients/<patient_id>/summary")
def patient_summary(patient_id: str):
	"""Return clinical summary report for a patient."""
	try:
		summary = _run(_svc().generate_clinical_summary(patient_id))
		return jsonify(summary)
	except ValueError as e:
		return _err(str(e), 404)


# ── encounters ────────────────────────────────────────────────────────────────

@bp.get("/encounters")
def list_encounters():
	patient_id = request.args.get("patient_id")
	page, size = _page()
	encs = _run(_svc().list_encounters(_tenant(), patient_id=patient_id))
	items = [e.model_dump(mode="json") for e in encs]
	return jsonify(_paginate(items, page, size))


@bp.post("/encounters")
def create_encounter():
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	data.setdefault("created_by", _actor())
	try:
		enc = _run(_svc().create_encounter(EncounterCreate(**data)))
		return jsonify(enc.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.get("/encounters/<enc_id>")
def get_encounter(enc_id: str):
	enc = _run(_svc().get_encounter(_tenant(), enc_id))
	if enc is None:
		return _err("encounter_not_found", 404)
	return jsonify(enc.model_dump(mode="json"))


@bp.put("/encounters/<enc_id>")
def update_encounter(enc_id: str):
	data = request.get_json(silent=True) or {}
	try:
		enc = _run(_svc().update_encounter(_tenant(), enc_id, EncounterUpdate(**data)))
		if enc is None:
			return _err("encounter_not_found", 404)
		return jsonify(enc.model_dump(mode="json"))
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.post("/encounters/<enc_id>/admit")
def admit_patient(enc_id: str):
	data = request.get_json(silent=True) or {}
	try:
		enc = _run(_svc().admit_patient(_tenant(), enc_id, data))
		if enc is None:
			return _err("encounter_not_found", 404)
		return jsonify(enc.model_dump(mode="json"))
	except ValueError as e:
		return _err(str(e), 400)


@bp.post("/encounters/<enc_id>/discharge")
def discharge_patient(enc_id: str):
	data = request.get_json(silent=True) or {}
	try:
		result = _run(_svc().discharge_patient(
			encounter_id=enc_id,
			discharge_diagnosis=data.get("discharge_diagnosis", ""),
			treatment_summary=data.get("treatment_summary", ""),
			follow_up=data.get("follow_up", ""),
			discharge_medications=data.get("discharge_medications", []),
		))
		return jsonify(result)
	except ValueError as e:
		return _err(str(e), 400)


@bp.post("/encounters/<enc_id>/transfer")
def transfer_patient(enc_id: str):
	data = request.get_json(silent=True) or {}
	try:
		result = _run(_svc().transfer_patient(
			encounter_id=enc_id,
			to_location_id=data.get("to_location_id", ""),
			to_provider_id=data.get("to_provider_id"),
			reason=data.get("reason", ""),
		))
		return jsonify(result)
	except ValueError as e:
		return _err(str(e), 400)


@bp.post("/encounters/<enc_id>/close")
def close_encounter(enc_id: str):
	data = request.get_json(silent=True) or {}
	enc = _run(_svc().close_encounter(_tenant(), enc_id, data.get("icd10_codes")))
	if enc is None:
		return _err("encounter_not_found", 404)
	return jsonify(enc.model_dump(mode="json"))


# ── clinical notes ────────────────────────────────────────────────────────────

@bp.get("/notes")
def list_notes():
	patient_id = request.args.get("patient_id")
	note_type = request.args.get("note_type")
	page, size = _page()
	notes = _run(_svc().list_notes(_tenant(), patient_id=patient_id, note_type=note_type))
	items = [n.model_dump(mode="json") for n in notes]
	return jsonify(_paginate(items, page, size))


@bp.post("/notes")
def create_note():
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	try:
		note = _run(_svc().create_note(ClinicalNoteCreate(**data)))
		return jsonify(note.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.get("/notes/<note_id>")
def get_note(note_id: str):
	note = _run(_svc().get_note(_tenant(), note_id))
	if note is None:
		return _err("note_not_found", 404)
	return jsonify(note.model_dump(mode="json"))


@bp.put("/notes/<note_id>")
def update_note(note_id: str):
	data = request.get_json(silent=True) or {}
	try:
		note = _run(_svc().update_note(_tenant(), note_id, ClinicalNoteUpdate(**data)))
		if note is None:
			return _err("note_not_found", 404)
		return jsonify(note.model_dump(mode="json"))
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.post("/notes/<note_id>/amend")
def amend_note(note_id: str):
	data = request.get_json(silent=True) or {}
	try:
		note = _run(_svc().amend_note(
			_tenant(), note_id,
			data.get("author_id", _actor()),
			data.get("content", ""),
		))
		if note is None:
			return _err("note_not_found", 404)
		return jsonify(note.model_dump(mode="json")), 201
	except PolicyViolationError as e:
		return _err(str(e), 403)


@bp.post("/notes/<note_id>/finalize")
def finalize_note(note_id: str):
	data = request.get_json(silent=True) or {}
	note = _run(_svc().finalize_note(_tenant(), note_id, data.get("cosigned_by")))
	if note is None:
		return _err("note_not_found", 404)
	return jsonify(note.model_dump(mode="json"))


@bp.post("/notes/<note_id>/sign")
def sign_note(note_id: str):
	data = request.get_json(silent=True) or {}
	try:
		result = _run(_svc().sign_clinical_note(note_id, data.get("clinician_id", _actor())))
		return jsonify(result)
	except ValueError as e:
		return _err(str(e), 400)


@bp.post("/notes/<note_id>/addendum")
def add_addendum(note_id: str):
	data = request.get_json(silent=True) or {}
	try:
		result = _run(_svc().addendum_to_note(
			note_id,
			addendum_text=data.get("addendum_text", ""),
			added_by=data.get("added_by", _actor()),
		))
		return jsonify(result), 201
	except ValueError as e:
		return _err(str(e), 400)


# ── problems / diagnoses ──────────────────────────────────────────────────────

@bp.get("/patients/<patient_id>/problems")
def list_problems(patient_id: str):
	status = request.args.get("status")
	page, size = _page()
	probs = _run(_svc().list_problems(_tenant(), patient_id, status=status))
	items = [p.model_dump(mode="json") for p in probs]
	return jsonify(_paginate(items, page, size))


@bp.post("/problems")
def add_problem():
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	data.setdefault("created_by", _actor())
	try:
		prob = _run(_svc().add_problem(ProblemCreate(**data)))
		return jsonify(prob.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.post("/problems/<problem_id>/resolve")
def resolve_problem(problem_id: str):
	prob = _run(_svc().resolve_problem(_tenant(), problem_id))
	if prob is None:
		return _err("problem_not_found", 404)
	return jsonify(prob.model_dump(mode="json"))


@bp.post("/encounters/<enc_id>/diagnoses")
def assign_diagnosis(enc_id: str):
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	data.setdefault("encounter_id", enc_id)
	data.setdefault("created_by", _actor())
	try:
		dx = _run(_svc().assign_icd10_diagnosis(
			encounter_id=enc_id,
			icd10_code=data.get("icd10_code", ""),
			description=data.get("description", ""),
			certainty=data.get("certainty", "confirmed"),
			is_primary=data.get("is_primary", False),
		))
		return jsonify(dx), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.get("/suggest-diagnoses")
def suggest_diagnoses():
	symptoms = request.args.get("symptoms", "")
	if not symptoms:
		return _err("symptoms query param required", 400)
	suggestions = _run(_svc().suggest_diagnoses(symptoms))
	return jsonify({"suggestions": suggestions})


# ── medications ───────────────────────────────────────────────────────────────

@bp.get("/patients/<patient_id>/medications")
def list_medications(patient_id: str):
	status = request.args.get("status")
	page, size = _page()
	meds = _run(_svc().list_medications(_tenant(), patient_id, status=status))
	items = [m.model_dump(mode="json") for m in meds]
	return jsonify(_paginate(items, page, size))


@bp.post("/medications")
def prescribe_medication():
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	data.setdefault("created_by", _actor())
	try:
		med = _run(_svc().prescribe_medication(MedicationCreate(**data)))
		return jsonify(med.model_dump(mode="json")), 201
	except PolicyViolationError as e:
		return _err(str(e), 403)
	except Exception as e:
		return _err(str(e), 400)


@bp.post("/medications/<med_id>/discontinue")
def discontinue_medication(med_id: str):
	med = _run(_svc().discontinue_medication(_tenant(), med_id))
	if med is None:
		return _err("medication_not_found", 404)
	return jsonify(med.model_dump(mode="json"))


@bp.post("/patients/<patient_id>/medication-stop")
def stop_medication(patient_id: str):
	data = request.get_json(silent=True) or {}
	try:
		result = _run(_svc().stop_medication(
			patient_id=patient_id,
			medication_id=data.get("medication_id", ""),
			reason=data.get("reason", ""),
			stopped_by=data.get("stopped_by", _actor()),
		))
		return jsonify(result)
	except ValueError as e:
		return _err(str(e), 400)


@bp.get("/patients/<patient_id>/medication-reconciliation")
def get_mar(patient_id: str):
	enc_id = request.args.get("encounter_id", "")
	period_from = request.args.get("from", "")
	period_to = request.args.get("to", "")
	result = _run(_svc().medication_administration_record(
		patient_id=patient_id,
		encounter_id=enc_id,
		period_from=period_from,
		period_to=period_to,
	))
	return jsonify(result)


@bp.post("/patients/<patient_id>/reconcile-medications")
def reconcile_medications(patient_id: str):
	data = request.get_json(silent=True) or {}
	try:
		result = _run(_svc().medication_reconciliation(
			patient_id=patient_id,
			encounter_id=data.get("encounter_id", ""),
			home_medications=data.get("home_medications", []),
		))
		return jsonify(result)
	except ValueError as e:
		return _err(str(e), 400)


# ── prescriptions ─────────────────────────────────────────────────────────────

@bp.get("/patients/<patient_id>/prescriptions")
def list_prescriptions(patient_id: str):
	page, size = _page()
	rxs = _run(_svc().generate_prescription_list(patient_id))
	return jsonify(_paginate(rxs, page, size))


@bp.post("/prescriptions")
def create_prescription():
	data = request.get_json(silent=True) or {}
	try:
		rx = _run(_svc().create_prescription(
			patient_id=data.get("patient_id", ""),
			drug=data.get("drug", ""),
			dose=float(data.get("dose", 0)),
			frequency=data.get("frequency", ""),
			duration_days=int(data.get("duration_days", 1)),
			route=data.get("route", "oral"),
			prescriber_id=data.get("prescriber_id", _actor()),
			encounter_id=data.get("encounter_id", ""),
		))
		return jsonify(rx), 201
	except DrugSafetyError as e:
		return _err(str(e), 422)
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.post("/prescriptions/<rx_id>/verify")
def verify_prescription(rx_id: str):
	data = request.get_json(silent=True) or {}
	try:
		rx = _run(_svc().verify_prescription(rx_id, data.get("pharmacist_id", _actor())))
		return jsonify(rx)
	except ValueError as e:
		return _err(str(e), 400)


@bp.post("/prescriptions/<rx_id>/dispense")
def dispense_medication(rx_id: str):
	data = request.get_json(silent=True) or {}
	try:
		rx = _run(_svc().dispense_medication(
			prescription_id=rx_id,
			lot_number=data.get("lot_number", ""),
			expiry_date=data.get("expiry_date", ""),
			quantity=float(data.get("quantity", 0)),
			dispensed_by=data.get("dispensed_by", _actor()),
		))
		return jsonify(rx)
	except ValueError as e:
		return _err(str(e), 400)


@bp.post("/prescriptions/<rx_id>/refill")
def refill_prescription(rx_id: str):
	data = request.get_json(silent=True) or {}
	try:
		rx = _run(_svc().refill_prescription(
			prescription_id=rx_id,
			refill_count=int(data.get("refill_count", 1)),
			dispensed_by=data.get("dispensed_by", _actor()),
		))
		return jsonify(rx)
	except ValueError as e:
		return _err(str(e), 400)


# ── allergies ─────────────────────────────────────────────────────────────────

@bp.get("/patients/<patient_id>/allergies")
def list_allergies(patient_id: str):
	page, size = _page()
	allergies = _run(_svc().list_allergies(_tenant(), patient_id))
	items = [a.model_dump(mode="json") for a in allergies]
	return jsonify(_paginate(items, page, size))


@bp.post("/allergies")
def record_allergy():
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	data.setdefault("created_by", _actor())
	try:
		allergy = _run(_svc().record_allergy(AllergyCreate(**data)))
		return jsonify(allergy.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.get("/patients/<patient_id>/allergy-check/<drug_name>")
def check_drug_allergy(patient_id: str, drug_name: str):
	result = _run(_svc().check_drug_allergy(_tenant(), patient_id, drug_name))
	return jsonify(result)


# ── vitals ────────────────────────────────────────────────────────────────────

@bp.get("/patients/<patient_id>/vitals")
def list_vitals(patient_id: str):
	vital_type = request.args.get("vital_type")
	page, size = _page()
	vitals = _run(_svc().list_vitals(_tenant(), patient_id, vital_type=vital_type))
	items = [v.model_dump(mode="json") for v in vitals]
	return jsonify(_paginate(items, page, size))


@bp.post("/vitals")
def record_vital():
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	data.setdefault("recorded_by", _actor())
	if "recorded_at" in data and isinstance(data["recorded_at"], str):
		data["recorded_at"] = datetime.fromisoformat(data["recorded_at"])
	try:
		vital = _run(_svc().record_vital(VitalSignCreate(**data)))
		return jsonify(vital.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


# ── lab orders & results ──────────────────────────────────────────────────────

@bp.get("/patients/<patient_id>/lab-orders")
def list_lab_orders(patient_id: str):
	page, size = _page()
	orders = _run(_svc().list_lab_orders(_tenant(), patient_id))
	items = [o.model_dump(mode="json") for o in orders]
	return jsonify(_paginate(items, page, size))


@bp.post("/lab-orders")
def create_lab_order():
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	data.setdefault("created_by", _actor())
	try:
		order = _run(_svc().order_lab_test(LabOrderCreate(**data)))
		return jsonify(order.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.get("/lab-orders/<order_id>")
def get_lab_order(order_id: str):
	order = _run(_svc().get_lab_order(_tenant(), order_id))
	if order is None:
		return _err("lab_order_not_found", 404)
	return jsonify(order.model_dump(mode="json"))


@bp.delete("/lab-orders/<order_id>")
def cancel_lab_order(order_id: str):
	result = _run(_svc().cancel_lab_order(_tenant(), order_id))
	if result is None:
		return _err("lab_order_not_found", 404)
	return jsonify(result.model_dump(mode="json"))


@bp.post("/lab-results")
def receive_lab_result():
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	data.setdefault("created_by", _actor())
	try:
		result = _run(_svc().receive_lab_result(LabResultCreate(**data)))
		return jsonify(result.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.get("/patients/<patient_id>/lab-results")
def list_lab_results(patient_id: str):
	page, size = _page()
	results = _run(_svc().list_lab_results(_tenant(), patient_id))
	items = [r.model_dump(mode="json") for r in results]
	return jsonify(_paginate(items, page, size))


@bp.post("/lab-results/<result_id>/notify-critical")
def notify_critical_lab(result_id: str):
	data = request.get_json(silent=True) or {}
	try:
		result = _run(_svc().flag_critical_lab_result(
			result_id=result_id,
			notified_to=data.get("notified_to", ""),
		))
		return jsonify(result.model_dump(mode="json"))
	except ValueError as e:
		return _err(str(e), 400)


# ── imaging ───────────────────────────────────────────────────────────────────

@bp.get("/patients/<patient_id>/imaging-orders")
def list_imaging_orders(patient_id: str):
	page, size = _page()
	orders = _run(_svc().list_imaging_orders(_tenant(), patient_id))
	items = [o.model_dump(mode="json") for o in orders]
	return jsonify(_paginate(items, page, size))


@bp.post("/imaging-orders")
def create_imaging_order():
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	data.setdefault("created_by", _actor())
	try:
		order = _run(_svc().order_imaging(ImagingOrderCreate(**data)))
		return jsonify(order.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.get("/imaging-orders/<order_id>")
def get_imaging_order(order_id: str):
	order = _run(_svc().get_imaging_order(_tenant(), order_id))
	if order is None:
		return _err("imaging_order_not_found", 404)
	return jsonify(order.model_dump(mode="json"))


@bp.post("/imaging-orders/<order_id>/report")
def add_imaging_report(order_id: str):
	data = request.get_json(silent=True) or {}
	try:
		order = _run(_svc().add_imaging_report(
			order_id=order_id,
			radiologist_id=data.get("radiologist_id", _actor()),
			impression=data.get("impression", ""),
		))
		if order is None:
			return _err("imaging_order_not_found", 404)
		return jsonify(order.model_dump(mode="json"))
	except ValueError as e:
		return _err(str(e), 400)


# ── care plans ────────────────────────────────────────────────────────────────

@bp.get("/patients/<patient_id>/care-plans")
def list_care_plans(patient_id: str):
	page, size = _page()
	plans = _run(_svc().list_care_plans(_tenant(), patient_id))
	items = [p.model_dump(mode="json") for p in plans]
	return jsonify(_paginate(items, page, size))


@bp.post("/care-plans")
def create_care_plan():
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	data.setdefault("created_by", _actor())
	try:
		plan = _run(_svc().create_care_plan(CarePlanCreate(**data)))
		return jsonify(plan.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.get("/care-plans/<plan_id>")
def get_care_plan(plan_id: str):
	plan = _run(_svc().get_care_plan(_tenant(), plan_id))
	if plan is None:
		return _err("care_plan_not_found", 404)
	return jsonify(plan.model_dump(mode="json"))


@bp.put("/care-plans/<plan_id>")
def update_care_plan(plan_id: str):
	data = request.get_json(silent=True) or {}
	try:
		plan = _run(_svc().update_care_plan(_tenant(), plan_id, CarePlanUpdate(**data)))
		if plan is None:
			return _err("care_plan_not_found", 404)
		return jsonify(plan.model_dump(mode="json"))
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.post("/care-plans/<plan_id>/activate")
def activate_care_plan(plan_id: str):
	plan = _run(_svc().activate_care_plan(_tenant(), plan_id))
	if plan is None:
		return _err("care_plan_not_found", 404)
	return jsonify(plan.model_dump(mode="json"))


@bp.post("/care-plans/<plan_id>/complete")
def complete_care_plan(plan_id: str):
	plan = _run(_svc().complete_care_plan(_tenant(), plan_id))
	if plan is None:
		return _err("care_plan_not_found", 404)
	return jsonify(plan.model_dump(mode="json"))


# ── referrals ─────────────────────────────────────────────────────────────────

@bp.get("/patients/<patient_id>/referrals")
def list_referrals(patient_id: str):
	page, size = _page()
	refs = _run(_svc().list_referrals(_tenant(), patient_id))
	return jsonify(_paginate(refs, page, size))


@bp.post("/referrals")
def create_referral():
	data = request.get_json(silent=True) or {}
	try:
		ref = _run(_svc().create_referral(
			patient_id=data.get("patient_id", ""),
			from_provider_id=data.get("from_provider_id", _actor()),
			to_specialty=data.get("to_specialty", ""),
			reason=data.get("reason", ""),
			urgency=data.get("urgency", "routine"),
		))
		return jsonify(ref), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


@bp.post("/referrals/<ref_id>/accept")
def accept_referral(ref_id: str):
	data = request.get_json(silent=True) or {}
	try:
		ref = _run(_svc().accept_referral(
			referral_id=ref_id,
			accepting_provider=data.get("accepting_provider", _actor()),
			appointment_date=data.get("appointment_date", ""),
		))
		return jsonify(ref)
	except ValueError as e:
		return _err(str(e), 400)


@bp.post("/referrals/<ref_id>/cancel")
def cancel_referral(ref_id: str):
	data = request.get_json(silent=True) or {}
	try:
		ref = _run(_svc().cancel_referral(ref_id, reason=data.get("reason", "")))
		return jsonify(ref)
	except ValueError as e:
		return _err(str(e), 400)


# ── consents ──────────────────────────────────────────────────────────────────

@bp.get("/patients/<patient_id>/consents")
def list_consents(patient_id: str):
	scope = request.args.get("scope")
	consents = _run(_svc().list_consents(_tenant(), patient_id, scope=scope))
	return jsonify({"items": consents, "count": len(consents)})


@bp.post("/consents")
def record_consent():
	data = request.get_json(silent=True) or {}
	try:
		consent = _run(_svc().record_consent(
			patient_id=data.get("patient_id", ""),
			consent_type=data.get("consent_type", data.get("scope", "")),
			obtained_by=data.get("obtained_by", _actor()),
			valid_until=data.get("valid_until", ""),
		))
		return jsonify(consent), 201
	except ValueError as e:
		return _err(str(e), 400)


@bp.get("/patients/<patient_id>/consent-check")
def check_consent(patient_id: str):
	scope = request.args.get("scope", "")
	result = _run(_svc().check_consent(patient_id, scope))
	return jsonify(result)


@bp.post("/consents/emergency-override")
def emergency_consent_override():
	data = request.get_json(silent=True) or {}
	try:
		result = _run(_svc().emergency_consent_override(
			patient_id=data.get("patient_id", ""),
			reason=data.get("reason", ""),
			authorised_by=data.get("authorised_by", _actor()),
		))
		return jsonify(result), 201
	except ValueError as e:
		return _err(str(e), 400)


@bp.post("/consents/minor-consent")
def minor_consent():
	data = request.get_json(silent=True) or {}
	try:
		result = _run(_svc().minor_consent(
			patient_id=data.get("patient_id", ""),
			guardian_id=data.get("guardian_id", ""),
			relationship=data.get("relationship", ""),
			consent_type=data.get("consent_type", ""),
		))
		return jsonify(result), 201
	except ValueError as e:
		return _err(str(e), 400)


# ── immunisations ─────────────────────────────────────────────────────────────

@bp.get("/patients/<patient_id>/immunisations")
def list_immunisations(patient_id: str):
	page, size = _page()
	imms = _run(_svc().list_immunisations(_tenant(), patient_id))
	items = [i.model_dump(mode="json") for i in imms]
	return jsonify(_paginate(items, page, size))


@bp.post("/immunisations")
def record_immunisation():
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	data.setdefault("created_by", _actor())
	if "administered_date" in data and isinstance(data["administered_date"], str):
		from datetime import date
		data["administered_date"] = date.fromisoformat(data["administered_date"])
	try:
		imm = _run(_svc().record_immunisation(ImmunisationCreate(**data)))
		return jsonify(imm.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


# ── family history ────────────────────────────────────────────────────────────

@bp.get("/patients/<patient_id>/family-history")
def list_family_history(patient_id: str):
	fhx = _run(_svc().list_family_history(_tenant(), patient_id))
	return jsonify({"items": [f.model_dump(mode="json") for f in fhx], "count": len(fhx)})


@bp.post("/family-history")
def add_family_history():
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	data.setdefault("created_by", _actor())
	try:
		fhx = _run(_svc().add_family_history(FamilyHistoryCreate(**data)))
		return jsonify(fhx.model_dump(mode="json")), 201
	except (PolicyViolationError, ValueError) as e:
		return _err(str(e), 403 if isinstance(e, PolicyViolationError) else 400)


# ── drug safety checks ────────────────────────────────────────────────────────

@bp.post("/drug-safety/interaction-check")
def drug_interaction_check():
	data = request.get_json(silent=True) or {}
	drug_list = data.get("drugs", [])
	if not drug_list:
		return _err("drugs list required", 400)
	results = _run(_svc().check_drug_drug_interactions(drug_list))
	return jsonify({"interactions": results, "count": len(results)})


@bp.post("/drug-safety/allergy-check")
def drug_allergy_check():
	data = request.get_json(silent=True) or {}
	try:
		result = _run(_svc().check_drug_allergy_alert(
			patient_id=data.get("patient_id", ""),
			drug_name=data.get("drug_name", ""),
			drug_class=data.get("drug_class", ""),
		))
		return jsonify(result)
	except ValueError as e:
		return _err(str(e), 400)


@bp.post("/drug-safety/paediatric-dose-check")
def paediatric_dose_check():
	data = request.get_json(silent=True) or {}
	try:
		result = _run(_svc().paediatric_dose_check(
			drug=data.get("drug", ""),
			weight_kg=float(data.get("weight_kg", 0)),
			age_months=int(data.get("age_months", 0)),
			prescribed_dose=float(data.get("prescribed_dose", 0)),
			route=data.get("route", "oral"),
		))
		return jsonify(result)
	except (AssertionError, ValueError) as e:
		return _err(str(e), 400)


@bp.post("/drug-safety/pregnancy-check")
def pregnancy_check():
	data = request.get_json(silent=True) or {}
	try:
		result = _run(_svc().pregnancy_safety_check(
			drug_name=data.get("drug_name", ""),
			trimester=int(data.get("trimester", 1)),
		))
		return jsonify(result)
	except (AssertionError, ValueError) as e:
		return _err(str(e), 400)


@bp.post("/drug-safety/renal-adjustment")
def renal_adjustment():
	data = request.get_json(silent=True) or {}
	try:
		result = _run(_svc().renal_dose_adjustment(
			drug_name=data.get("drug_name", ""),
			egfr_ml_per_min=float(data.get("egfr_ml_per_min", 0)),
		))
		return jsonify(result)
	except (AssertionError, ValueError) as e:
		return _err(str(e), 400)


@bp.post("/drug-safety/controlled-substance-check")
def controlled_substance_check():
	data = request.get_json(silent=True) or {}
	try:
		result = _run(_svc().controlled_substance_check(
			drug=data.get("drug", ""),
			schedule=data.get("schedule", ""),
			quantity=int(data.get("quantity", 0)),
			prescriber_id=data.get("prescriber_id", _actor()),
		))
		return jsonify(result)
	except (AssertionError, ValueError) as e:
		return _err(str(e), 400)


# ── clinical decision support ─────────────────────────────────────────────────

@bp.get("/cds/<patient_id>/alerts")
def cds_alerts(patient_id: str):
	"""Run full clinical decision support and return all active alerts."""
	alerts = _run(_svc().clinical_decision_support(patient_id))
	return jsonify({"patient_id": patient_id, "alerts": [a.model_dump() for a in alerts]})


@bp.get("/cds/<patient_id>/reminders")
def cds_reminders(patient_id: str):
	reminders = _run(_svc().clinical_reminder_check(patient_id))
	return jsonify({"reminders": reminders, "count": len(reminders)})


@bp.get("/cds/<patient_id>/chads2-vasc")
def chads2_vasc(patient_id: str):
	result = _run(_svc().CHADS2_VASc_score(patient_id))
	return jsonify(result)


@bp.post("/cds/<patient_id>/wells-pe")
def wells_pe(patient_id: str):
	data = request.get_json(silent=True) or {}
	result = _run(_svc().WELLS_score_PE(patient_id, data))
	return jsonify(result)


@bp.post("/cds/<patient_id>/qsofa")
def qsofa(patient_id: str):
	data = request.get_json(silent=True) or {}
	try:
		result = _run(_svc().QSOFA_score(
			patient_id=patient_id,
			respiratory_rate=int(data.get("respiratory_rate", 0)),
			mentation_altered=bool(data.get("mentation_altered", False)),
			sbp=int(data.get("sbp", 120)),
		))
		return jsonify(result)
	except (AssertionError, ValueError) as e:
		return _err(str(e), 400)


@bp.post("/cds/<patient_id>/news2")
def news2(patient_id: str):
	data = request.get_json(silent=True) or {}
	result = _run(_svc().NEWS2_score(patient_id, data))
	return jsonify(result)


@bp.get("/cds/guideline-alert")
def guideline_alert():
	patient_id = request.args.get("patient_id", "")
	dx_code = request.args.get("diagnosis_code", "")
	alerts = _run(_svc().clinical_guideline_alert(patient_id, dx_code))
	return jsonify({"alerts": alerts})


# ── FHIR export ───────────────────────────────────────────────────────────────

@bp.post("/fhir/export")
def fhir_export():
	data = request.get_json(silent=True) or {}
	patient_id = data.get("patient_id", "")
	resource_types = data.get("resource_types", [
		"Patient", "Condition", "MedicationRequest", "AllergyIntolerance",
		"Observation", "DocumentReference",
	])
	phi_consent = data.get("phi_consent_present", False)
	try:
		bundle = _run(_svc().fhir_export(_tenant(), patient_id, resource_types, phi_consent))
		return jsonify(bundle)
	except PolicyViolationError as e:
		return _err(str(e), 403)


@bp.post("/fhir/bundle")
def fhir_bundle():
	data = request.get_json(silent=True) or {}
	patient_id = data.get("patient_id", "")
	resource_types = data.get("resource_types", ["Patient", "Encounter", "Condition"])
	try:
		bundle = _run(_svc().fhir_bundle_export(patient_id, resource_types))
		return jsonify(bundle)
	except ValueError as e:
		return _err(str(e), 400)


@bp.get("/fhir/patient/<patient_id>")
def fhir_patient_resource(patient_id: str):
	resource = _run(_svc().fhir_patient_resource(patient_id))
	return jsonify(resource)


@bp.get("/fhir/encounter/<enc_id>")
def fhir_encounter_resource(enc_id: str):
	try:
		resource = _run(_svc().fhir_encounter_resource(enc_id))
		return jsonify(resource)
	except ValueError as e:
		return _err(str(e), 404)


# ── HL7 v2 message processing ─────────────────────────────────────────────────

@bp.post("/hl7/inbound")
def hl7_inbound():
	"""Accept an HL7 v2 message (text/plain or JSON-wrapped) and process it."""
	content_type = request.content_type or ""
	if "json" in content_type:
		data = request.get_json(silent=True) or {}
		message = data.get("message", "")
	else:
		message = request.get_data(as_text=True)
	if not message:
		return _err("HL7 message body required", 400)
	try:
		result = _run(_svc().hl7_message_processing(message))
		return jsonify(result)
	except ValueError as e:
		return _err(str(e), 400)


# ── reports ───────────────────────────────────────────────────────────────────

@bp.get("/reports/dashboard")
def report_dashboard():
	return jsonify(_run(_svc().dashboard_summary(_tenant())))


@bp.get("/reports/patient-summary/<patient_id>")
def report_patient_summary(patient_id: str):
	result = _run(_svc().generate_clinical_summary(patient_id))
	return jsonify(result)


@bp.get("/reports/critical-labs")
def report_critical_labs():
	"""Return unnotified critical lab results for the tenant."""
	results = _run(_svc().list_unnotified_critical_labs(_tenant()))
	return jsonify({"items": [r.model_dump(mode="json") for r in results], "count": len(results)})


@bp.get("/reports/controlled-substances")
def report_controlled_substances():
	"""Controlled substance prescriptions today."""
	results = _run(_svc().controlled_substance_report(_tenant()))
	return jsonify(results)


# ── FHIR R4 endpoints ─────────────────────────────────────────────────────────
# Exposes APG EMR data as HL7 FHIR R4 resources for interoperability with
# Epic, Cerner, OpenMRS, Apple Health Records, and national health exchanges.

@bp.get("/fhir/r4/metadata")
def fhir_capability_statement():
	"""FHIR R4 CapabilityStatement — server capabilities declaration."""
	from .fhir import FHIRAdapter
	base_url = request.host_url.rstrip("/") + bp.url_prefix
	adapter = FHIRAdapter(base_url=f"{base_url}/fhir/r4")
	return jsonify(adapter.capability_statement()), 200, {
		"Content-Type": "application/fhir+json",
		"Cache-Control": "max-age=86400",
	}


@bp.get("/fhir/r4/Patient/<patient_id>")
def fhir_get_patient(patient_id: str):
	"""Return FHIR R4 Patient resource for a given patient ID."""
	from .fhir import FHIRAdapter
	try:
		patient = _run(_svc().get_patient(patient_id))
		if patient is None:
			return jsonify({"resourceType": "OperationOutcome", "issue": [{"severity": "error", "code": "not-found"}]}), 404
		base_url = request.host_url.rstrip("/") + bp.url_prefix
		adapter = FHIRAdapter(base_url=f"{base_url}/fhir/r4")
		return jsonify(adapter.patient_to_fhir(patient)), 200, {"Content-Type": "application/fhir+json"}
	except Exception as exc:
		return _err(str(exc), 500)


@bp.get("/fhir/r4/Patient")
def fhir_search_patients():
	"""FHIR R4 Patient search (supports ?_count= and simple text search)."""
	from .fhir import FHIRAdapter
	limit = int(request.args.get("_count", 20))
	patients = _run(_svc().list_patients(_tenant()))[:limit]
	base_url = request.host_url.rstrip("/") + bp.url_prefix
	adapter = FHIRAdapter(base_url=f"{base_url}/fhir/r4")
	bundle = {
		"resourceType": "Bundle",
		"type": "searchset",
		"total": len(patients),
		"entry": [
			{"resource": adapter.patient_to_fhir(p), "fullUrl": f"{base_url}/fhir/r4/Patient/{p.get('id', '')}"}
			for p in patients
		],
	}
	return jsonify(bundle), 200, {"Content-Type": "application/fhir+json"}


@bp.get("/fhir/r4/Encounter/<encounter_id>")
def fhir_get_encounter(encounter_id: str):
	"""Return FHIR R4 Encounter resource."""
	from .fhir import FHIRAdapter
	try:
		encounter = _run(_svc().get_encounter(encounter_id))
		if encounter is None:
			return jsonify({"resourceType": "OperationOutcome", "issue": [{"severity": "error", "code": "not-found"}]}), 404
		base_url = request.host_url.rstrip("/") + bp.url_prefix
		adapter = FHIRAdapter(base_url=f"{base_url}/fhir/r4")
		return jsonify(adapter.encounter_to_fhir(encounter)), 200, {"Content-Type": "application/fhir+json"}
	except Exception as exc:
		return _err(str(exc), 500)
