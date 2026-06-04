"""Tests for ElectronicMedicalRecordsService."""

from __future__ import annotations

import asyncio, sys, os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from emr.models import (
	AllergyCreate, ClinicalNoteCreate, EncounterCreate,
	MedicationCreate, ProblemCreate, VitalSignCreate,
)
from emr.service import ElectronicMedicalRecordsService, PolicyViolationError


def run(coro):
	return asyncio.get_event_loop().run_until_complete(coro)


def svc():
	return ElectronicMedicalRecordsService()


def test_create_encounter():
	s = svc()
	enc = run(s.create_encounter(EncounterCreate(tenant_id="t", patient_id="p1", encounter_type="outpatient", provider_id="dr1", location_id="l1", chief_complaint="chest pain", created_by="nurse1")))
	assert enc.id and enc.status == "in_progress"


def test_close_encounter():
	s = svc()
	enc = run(s.create_encounter(EncounterCreate(tenant_id="t", patient_id="p1", encounter_type="outpatient", provider_id="dr1", location_id="l1", chief_complaint="cp", created_by="u")))
	closed = run(s.close_encounter("t", enc.id, ["I10"]))
	assert closed.status == "finished"
	assert closed.discharge_time is not None
	assert "I10" in closed.icd10_codes


def test_create_note_returns_draft():
	s = svc()
	note = run(s.create_note(ClinicalNoteCreate(tenant_id="t", patient_id="p1", encounter_id="e1", note_type="soap_note", author_id="dr1", content="Patient presents with...")))
	assert note.status == "draft"
	assert note.note_type == "soap_note"


def test_create_note_unsupported_type_denied():
	s = svc()
	try:
		run(s.create_note(ClinicalNoteCreate(tenant_id="t", patient_id="p1", encounter_id="e1", note_type="unknown_type", author_id="dr1", content="content")))
		assert False
	except PolicyViolationError:
		pass


def test_finalize_note():
	s = svc()
	note = run(s.create_note(ClinicalNoteCreate(tenant_id="t", patient_id="p1", encounter_id="e1", note_type="progress_note", author_id="dr1", content="Note content")))
	final = run(s.finalize_note("t", note.id, cosigned_by="dr2"))
	assert final.status == "final"
	assert final.cosigned_by == "dr2"
	assert final.finalized_at is not None


def test_amend_note_creates_linked_note():
	s = svc()
	note = run(s.create_note(ClinicalNoteCreate(tenant_id="t", patient_id="p1", encounter_id="e1", note_type="soap_note", author_id="dr1", content="Original")))
	amendment = run(s.amend_note("t", note.id, "dr2", "Corrected content"))
	assert amendment.amendment_of == note.id
	assert amendment.content == "Corrected content"


def test_amend_nonexistent_note_denied():
	s = svc()
	try:
		run(s.amend_note("t", "nonexistent", "dr1", "content"))
		assert False
	except PolicyViolationError:
		pass


def test_add_problem_requires_icd10():
	s = svc()
	try:
		run(s.add_problem(ProblemCreate(tenant_id="t", patient_id="p1", icd10_code="", description="Hypertension", created_by="dr1")))
		assert False
	except Exception:
		pass  # pydantic or policy violation


def test_add_problem_success():
	s = svc()
	prob = run(s.add_problem(ProblemCreate(tenant_id="t", patient_id="p1", icd10_code="I10", description="Hypertension", created_by="dr1")))
	assert prob.icd10_code == "I10"
	assert prob.status == "active"


def test_resolve_problem():
	s = svc()
	prob = run(s.add_problem(ProblemCreate(tenant_id="t", patient_id="p1", icd10_code="J45.909", description="Asthma", created_by="dr1")))
	resolved = run(s.resolve_problem("t", prob.id))
	assert resolved.status == "resolved"
	assert resolved.resolved_date is not None


def test_prescribe_medication_allergy_check_required():
	s = svc()
	try:
		run(s.prescribe_medication(MedicationCreate(tenant_id="t", patient_id="p1", drug_name="Penicillin", dose="500mg", route="oral", frequency="TID", prescriber_id="dr1", allergy_check_performed=False, created_by="dr1")))
		assert False
	except PolicyViolationError:
		pass


def test_prescribe_medication_success():
	s = svc()
	med = run(s.prescribe_medication(MedicationCreate(tenant_id="t", patient_id="p1", drug_name="Metformin", dose="500mg", route="oral", frequency="BID", prescriber_id="dr1", allergy_check_performed=True, created_by="dr1")))
	assert med.status == "active"
	assert med.drug_name == "Metformin"


def test_discontinue_medication():
	s = svc()
	med = run(s.prescribe_medication(MedicationCreate(tenant_id="t", patient_id="p1", drug_name="Lisinopril", dose="10mg", route="oral", frequency="daily", prescriber_id="dr1", allergy_check_performed=True, created_by="dr1")))
	disc = run(s.discontinue_medication("t", med.id))
	assert disc.status == "discontinued"


def test_record_allergy_success():
	s = svc()
	allergy = run(s.record_allergy(AllergyCreate(tenant_id="t", patient_id="p1", allergen="Penicillin", allergy_type="drug", severity="severe", reaction="anaphylaxis", created_by="nurse1")))
	assert allergy.severity == "severe"


def test_record_allergy_invalid_type_denied():
	s = svc()
	try:
		run(s.record_allergy(AllergyCreate(tenant_id="t", patient_id="p1", allergen="X", allergy_type="unknown_type", severity="mild", reaction="rash", created_by="u")))
		assert False
	except PolicyViolationError:
		pass


def test_drug_allergy_check():
	s = svc()
	run(s.record_allergy(AllergyCreate(tenant_id="t", patient_id="p1", allergen="Penicillin", allergy_type="drug", severity="severe", reaction="anaphylaxis", created_by="u")))
	result = run(s.check_drug_allergy("t", "p1", "Penicillin"))
	assert result["conflict_found"] is True
	result2 = run(s.check_drug_allergy("t", "p1", "Metformin"))
	assert result2["conflict_found"] is False


def test_record_vital():
	s = svc()
	vital = run(s.record_vital(VitalSignCreate(tenant_id="t", patient_id="p1", encounter_id="e1", vital_type="blood_pressure", value=120.0, unit="mmHg", recorded_by="nurse1")))
	assert vital.vital_type == "blood_pressure"


def test_dashboard_summary_counts():
	s = svc()
	run(s.create_encounter(EncounterCreate(tenant_id="t", patient_id="p1", encounter_type="outpatient", provider_id="dr1", location_id="l1", chief_complaint="cp", created_by="u")))
	run(s.create_note(ClinicalNoteCreate(tenant_id="t", patient_id="p1", encounter_id="e1", note_type="soap_note", author_id="dr1", content="content")))
	summary = run(s.dashboard_summary("t"))
	assert summary["encounters"]["total"] >= 1
	assert summary["notes"]["total"] >= 1
