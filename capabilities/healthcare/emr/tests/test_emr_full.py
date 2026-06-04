"""Comprehensive test suite for APG Electronic Medical Records.

Covers: patient CRUD, dedup, encounter lifecycle, notes, problems, medications,
prescriptions, drug safety, lab orders/results, imaging, care plans, referrals,
consents, immunisations, family history, CDS, FHIR export, HL7 processing,
calculations, and domain rules.
"""
from __future__ import annotations

import asyncio
import sys
import os
from datetime import date, datetime
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from emr.models import (
	AllergyCreate,
	CarePlanCreate,
	ClinicalNoteCreate, ClinicalNoteUpdate,
	EncounterCreate,
	FamilyHistoryCreate,
	ImagingOrderCreate,
	ImmunisationCreate,
	LabOrderCreate, LabResultCreate, LabResultFlag,
	MedicationCreate,
	PatientCreate, PatientName, PatientUpdate,
	ProblemCreate,
	VitalSignCreate,
)
from emr.service import EMRService, PolicyViolationError, DrugSafetyError


# ── helpers ───────────────────────────────────────────────────────────────────

def run(coro: Any) -> Any:
	return asyncio.get_event_loop().run_until_complete(coro)


def make_svc(tenant: str = "t1") -> EMRService:
	return EMRService(tenant_id=tenant, actor_id="test_actor")


def make_patient_payload(tenant: str = "t1", family: str = "Otieno") -> PatientCreate:
	return PatientCreate(
		tenant_id=tenant,
		name=PatientName(family=family, given=["John"]),
		birth_date=date(1985, 6, 15),
		gender="male",
		created_by="test",
	)


def make_encounter(svc: EMRService, patient_id: str, tenant: str = "t1") -> Any:
	return run(svc.create_encounter(EncounterCreate(
		tenant_id=tenant,
		patient_id=patient_id,
		encounter_type="outpatient",
		provider_id="dr1",
		location_id="ward_a",
		chief_complaint="fever and cough",
		created_by="test",
	)))


# ═══════════════════════════════════════════════════════════════════════════════
# PATIENT
# ═══════════════════════════════════════════════════════════════════════════════

def test_register_patient_creates_record():
	svc = make_svc()
	p = run(svc.register_patient(make_patient_payload()))
	assert p.id
	assert p.name.family == "Otieno"
	assert p.birth_date == date(1985, 6, 15)
	assert p.status == "active"
	assert not p.is_deleted


def test_register_patient_stores_in_tenant():
	svc = make_svc("hospital_a")
	p = run(svc.register_patient(make_patient_payload(tenant="hospital_a")))
	retrieved = run(svc.get_patient("hospital_a", p.id))
	assert retrieved is not None
	assert retrieved.id == p.id


def test_get_patient_not_found_returns_none():
	svc = make_svc()
	assert run(svc.get_patient("t1", "nonexistent")) is None


def test_get_patient_cross_tenant_isolation():
	svc_a = make_svc("tenant_a")
	svc_b = make_svc("tenant_b")
	p = run(svc_a.register_patient(make_patient_payload(tenant="tenant_a")))
	# tenant_b cannot see tenant_a patient
	assert run(svc_b.get_patient("tenant_b", p.id)) is None


def test_list_patients_search_by_name():
	svc = make_svc()
	run(svc.register_patient(make_patient_payload(family="Kamau")))
	run(svc.register_patient(make_patient_payload(family="Wanjiku")))
	results = run(svc.list_patients("t1", search="kamau"))
	assert any(p.name.family == "Kamau" for p in results)
	assert all(p.name.family != "Wanjiku" for p in results)


def test_update_patient():
	svc = make_svc()
	p = run(svc.register_patient(make_patient_payload()))
	updated = run(svc.update_patient("t1", p.id, PatientUpdate(marital_status="married")))
	assert updated.marital_status == "married"


def test_delete_patient_soft_delete():
	svc = make_svc()
	p = run(svc.register_patient(make_patient_payload()))
	ok = run(svc.delete_patient("t1", p.id))
	assert ok
	assert run(svc.get_patient("t1", p.id)) is None


def test_dedup_check_returns_candidates():
	svc = make_svc()
	p = run(svc.register_patient(PatientCreate(
		tenant_id="t1",
		name=PatientName(family="Njoroge", given=["Peter"]),
		birth_date=date(1990, 3, 20),
		gender="male",
		created_by="test",
	)))
	candidates = run(svc.patient_deduplication_check({
		"family": "Njoroge",
		"given_0": "Peter",
		"birth_date": "1990-03-20",
		"gender": "male",
	}))
	assert any(c.candidate_patient_id == p.id for c in candidates)


def test_merge_patients_marks_duplicate():
	svc = make_svc()
	dup = run(svc.register_patient(make_patient_payload(family="Mwangi")))
	surviving = run(svc.register_patient(make_patient_payload(family="Mwangi")))
	result = run(svc.merge_patients("t1", dup.id, surviving.id))
	assert result["status"] == "merged"
	merged = run(svc.get_patient("t1", dup.id))
	# soft-deleted or merged — either way get_patient returns None or status=merged
	if merged:
		assert merged.status == "merged"
		assert merged.merged_into == surviving.id


# ═══════════════════════════════════════════════════════════════════════════════
# ENCOUNTERS
# ═══════════════════════════════════════════════════════════════════════════════

def test_create_and_get_encounter():
	svc = make_svc()
	p = run(svc.register_patient(make_patient_payload()))
	enc = make_encounter(svc, p.id)
	assert enc.id
	assert enc.status == "in_progress"
	fetched = run(svc.get_encounter("t1", enc.id))
	assert fetched.id == enc.id


def test_close_encounter_sets_discharge_time():
	svc = make_svc()
	p = run(svc.register_patient(make_patient_payload()))
	enc = make_encounter(svc, p.id)
	closed = run(svc.close_encounter("t1", enc.id, ["J18.9"]))
	assert closed.status == "finished"
	assert closed.discharge_time is not None
	assert "J18.9" in closed.icd10_codes


def test_admit_patient_updates_status():
	svc = make_svc()
	p = run(svc.register_patient(make_patient_payload()))
	enc = make_encounter(svc, p.id)
	admitted = run(svc.admit_patient("t1", enc.id, {}))
	assert admitted.status == "in_progress"


def test_transfer_patient_changes_location():
	svc = make_svc()
	p = run(svc.register_patient(make_patient_payload()))
	enc = make_encounter(svc, p.id)
	result = run(svc.transfer_patient(enc.id, "ward_b", "dr2", "ICU transfer"))
	assert result["to_location_id"] == "ward_b"
	assert result["from_location_id"] == "ward_a"


def test_discharge_patient_creates_summary():
	svc = make_svc()
	p = run(svc.register_patient(make_patient_payload()))
	enc = make_encounter(svc, p.id)
	result = run(svc.discharge_patient(
		encounter_id=enc.id,
		discharge_diagnosis="Pneumonia",
		treatment_summary="IV antibiotics for 5 days",
		follow_up="GP in 1 week",
		discharge_medications=[{"drug_name": "Amoxicillin", "dose": "500mg", "frequency": "TID"}],
	))
	assert result["discharge_summary"] is not None
	assert result["encounter"]["status"] == "finished"


def test_assign_icd10_to_encounter():
	svc = make_svc()
	p = run(svc.register_patient(make_patient_payload()))
	enc = make_encounter(svc, p.id)
	dx = run(svc.assign_icd10_diagnosis(enc.id, "E11.9", "Type 2 diabetes mellitus", "confirmed", True))
	assert dx["icd10_code"] == "E11.9"
	assert dx["is_primary"] is True
	# should also update the encounter's icd10_codes list
	updated_enc = run(svc.get_encounter("t1", enc.id))
	assert "E11.9" in updated_enc.icd10_codes


# ═══════════════════════════════════════════════════════════════════════════════
# CLINICAL NOTES
# ═══════════════════════════════════════════════════════════════════════════════

def test_create_soap_note():
	svc = make_svc()
	note = run(svc.create_note(ClinicalNoteCreate(
		tenant_id="t1", patient_id="p1", encounter_id="e1",
		note_type="soap_note", author_id="dr1",
		content="S: fever. O: temp 38.5. A: URTI. P: paracetamol.",
		subjective="fever", objective="temp 38.5", assessment="URTI", plan="paracetamol",
	)))
	assert note.status == "draft"
	assert note.subjective == "fever"


def test_finalize_note_cosign():
	svc = make_svc()
	note = run(svc.create_note(ClinicalNoteCreate(
		tenant_id="t1", patient_id="p1", encounter_id="e1",
		note_type="progress_note", author_id="dr1", content="Follow-up note",
	)))
	final = run(svc.finalize_note("t1", note.id, cosigned_by="dr2"))
	assert final.status == "final"
	assert final.cosigned_by == "dr2"
	assert final.finalized_at is not None


def test_sign_clinical_note():
	svc = make_svc()
	note = run(svc.create_note(ClinicalNoteCreate(
		tenant_id="t1", patient_id="p1", encounter_id="e1",
		note_type="soap_note", author_id="dr1", content="Content",
	)))
	result = run(svc.sign_clinical_note(note.id, "dr_senior"))
	assert result["status"] == "final"
	assert result["signed_by"] == "dr_senior"


def test_sign_already_final_note_raises():
	svc = make_svc()
	note = run(svc.create_note(ClinicalNoteCreate(
		tenant_id="t1", patient_id="p1", encounter_id="e1",
		note_type="soap_note", author_id="dr1", content="Content",
	)))
	run(svc.sign_clinical_note(note.id, "dr1"))
	try:
		run(svc.sign_clinical_note(note.id, "dr2"))
		assert False, "Should raise"
	except ValueError:
		pass


def test_addendum_to_note():
	svc = make_svc()
	note = run(svc.create_note(ClinicalNoteCreate(
		tenant_id="t1", patient_id="p1", encounter_id="e1",
		note_type="soap_note", author_id="dr1", content="Original content",
	)))
	run(svc.sign_clinical_note(note.id, "dr1"))
	addendum = run(svc.addendum_to_note(note.id, "Clarification: temp was axillary", "dr1"))
	assert addendum["original_note_id"] == note.id
	assert "Clarification" in addendum["content_preview"]


def test_update_draft_note():
	svc = make_svc()
	note = run(svc.create_note(ClinicalNoteCreate(
		tenant_id="t1", patient_id="p1", encounter_id="e1",
		note_type="soap_note", author_id="dr1", content="Draft",
	)))
	updated = run(svc.update_note("t1", note.id, ClinicalNoteUpdate(content="Updated draft")))
	assert updated.content == "Updated draft"


def test_update_final_note_raises():
	svc = make_svc()
	note = run(svc.create_note(ClinicalNoteCreate(
		tenant_id="t1", patient_id="p1", encounter_id="e1",
		note_type="soap_note", author_id="dr1", content="Content",
	)))
	run(svc.finalize_note("t1", note.id))
	try:
		run(svc.update_note("t1", note.id, ClinicalNoteUpdate(content="Cannot edit")))
		assert False, "Should raise PolicyViolationError"
	except PolicyViolationError:
		pass


def test_amend_note_links_to_original():
	svc = make_svc()
	note = run(svc.create_note(ClinicalNoteCreate(
		tenant_id="t1", patient_id="p1", encounter_id="e1",
		note_type="soap_note", author_id="dr1", content="Original",
	)))
	amendment = run(svc.amend_note("t1", note.id, "dr2", "Corrected content"))
	assert amendment.amendment_of == note.id


# ═══════════════════════════════════════════════════════════════════════════════
# PROBLEMS
# ═══════════════════════════════════════════════════════════════════════════════

def test_add_problem_normalises_icd10():
	svc = make_svc()
	prob = run(svc.add_problem(ProblemCreate(
		tenant_id="t1", patient_id="p1", icd10_code="i10", description="Hypertension", created_by="dr1"
	)))
	assert prob.icd10_code == "I10"


def test_list_problems_by_status():
	svc = make_svc()
	run(svc.add_problem(ProblemCreate(tenant_id="t1", patient_id="p1", icd10_code="I10", description="HTN", created_by="dr1")))
	run(svc.add_problem(ProblemCreate(tenant_id="t1", patient_id="p1", icd10_code="E11.9", description="T2DM", created_by="dr1")))
	active = run(svc.list_problems("t1", "p1", status="active"))
	assert len(active) == 2


def test_resolve_problem_sets_date():
	svc = make_svc()
	prob = run(svc.add_problem(ProblemCreate(
		tenant_id="t1", patient_id="p1", icd10_code="J45.909", description="Asthma", created_by="dr1"
	)))
	resolved = run(svc.resolve_problem("t1", prob.id))
	assert resolved.status == "resolved"
	assert resolved.resolved_date is not None


def test_suggest_diagnoses_chest_pain():
	svc = make_svc()
	suggestions = run(svc.suggest_diagnoses("patient presents with chest pain and shortness of breath"))
	assert len(suggestions) > 0
	codes = [s["icd10_code"] for s in suggestions]
	# chest pain and SOB should both contribute
	assert any("I" in c or "R" in c or "J" in c for c in codes)


# ═══════════════════════════════════════════════════════════════════════════════
# MEDICATIONS & PRESCRIBING
# ═══════════════════════════════════════════════════════════════════════════════

def test_prescribe_medication_allergy_check_enforced():
	svc = make_svc()
	try:
		run(svc.prescribe_medication(MedicationCreate(
			tenant_id="t1", patient_id="p1", drug_name="Amoxicillin",
			dose="500mg", route="oral", frequency="TID",
			prescriber_id="dr1", allergy_check_performed=False, created_by="dr1",
		)))
		assert False, "Should have raised PolicyViolationError"
	except PolicyViolationError:
		pass


def test_prescribe_and_discontinue_medication():
	svc = make_svc()
	med = run(svc.prescribe_medication(MedicationCreate(
		tenant_id="t1", patient_id="p1", drug_name="Atorvastatin",
		dose="20mg", route="oral", frequency="daily",
		prescriber_id="dr1", allergy_check_performed=True, created_by="dr1",
	)))
	assert med.status == "active"
	disc = run(svc.discontinue_medication("t1", med.id))
	assert disc.status == "discontinued"
	assert disc.end_date is not None


def test_stop_medication_with_reason():
	svc = make_svc()
	med = run(svc.prescribe_medication(MedicationCreate(
		tenant_id="t1", patient_id="p1", drug_name="Metformin",
		dose="500mg", route="oral", frequency="BID",
		prescriber_id="dr1", allergy_check_performed=True, created_by="dr1",
	)))
	result = run(svc.stop_medication("p1", med.id, "Patient reports GI intolerance", "dr1"))
	assert result["reason"] == "Patient reports GI intolerance"


def test_create_prescription_with_safety_checks():
	svc = make_svc()
	p = run(svc.register_patient(make_patient_payload()))
	enc = make_encounter(svc, p.id)
	rx = run(svc.create_prescription(
		patient_id=p.id, drug="paracetamol", dose=500.0,
		frequency="QID", duration_days=5, route="oral",
		prescriber_id="dr1", encounter_id=enc.id,
	))
	assert rx["status"] == "active"
	assert "safety_summary" in rx
	assert rx["allergy_checked"] is True
	assert rx["interaction_checked"] is True


def test_create_prescription_life_threatening_allergy_hard_stop():
	svc = make_svc()
	p = run(svc.register_patient(make_patient_payload()))
	enc = make_encounter(svc, p.id)
	# record life-threatening allergy to paracetamol
	run(svc.record_allergy(AllergyCreate(
		tenant_id="t1", patient_id=p.id, allergen="paracetamol",
		allergy_type="drug", severity="life_threatening",
		reaction="anaphylaxis", created_by="dr1",
	)))
	try:
		run(svc.create_prescription(
			patient_id=p.id, drug="paracetamol", dose=500.0,
			frequency="QID", duration_days=5, route="oral",
			prescriber_id="dr1", encounter_id=enc.id,
		))
		assert False, "Should raise DrugSafetyError"
	except DrugSafetyError:
		pass


def test_verify_and_dispense_prescription():
	svc = make_svc()
	p = run(svc.register_patient(make_patient_payload()))
	enc = make_encounter(svc, p.id)
	rx = run(svc.create_prescription(
		patient_id=p.id, drug="ibuprofen", dose=400.0,
		frequency="TID", duration_days=7, route="oral",
		prescriber_id="dr1", encounter_id=enc.id,
	))
	verified = run(svc.verify_prescription(rx["id"], "pharm1"))
	assert verified["pharmacist_verified"] is True
	dispensed = run(svc.dispense_medication(rx["id"], "LOT001", "2026-12-31", 21.0, "pharm1"))
	assert dispensed["lot_number"] == "LOT001"
	assert dispensed["dispensed_by"] == "pharm1"


def test_medication_reconciliation_detects_omission():
	svc = make_svc()
	p = run(svc.register_patient(make_patient_payload()))
	enc = make_encounter(svc, p.id)
	# patient reports taking warfarin at home but it's not in EMR
	result = run(svc.medication_reconciliation(
		patient_id=p.id, encounter_id=enc.id,
		home_medications=[{"drug_name": "Warfarin", "dose": "5mg"}],
	))
	assert result["discrepancy_count"] >= 1
	omissions = [d for d in result["discrepancies"] if d["type"] == "omission"]
	assert any("Warfarin" in d["description"] for d in omissions)


def test_refill_prescription():
	svc = make_svc()
	p = run(svc.register_patient(make_patient_payload()))
	enc = make_encounter(svc, p.id)
	rx = run(svc.create_prescription(
		patient_id=p.id, drug="lisinopril", dose=10.0,
		frequency="daily", duration_days=30, route="oral",
		prescriber_id="dr1", encounter_id=enc.id,
	))
	# grant 2 refills
	rx["refills_allowed"] = 2
	svc._prescriptions[("t1", rx["id"])] = rx
	run(svc.verify_prescription(rx["id"], "pharm1"))
	run(svc.dispense_medication(rx["id"], "LOT002", "2027-01-01", 30.0, "pharm1"))
	refilled = run(svc.refill_prescription(rx["id"], 1, "pharm2"))
	assert refilled["refills_used"] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# DRUG SAFETY
# ═══════════════════════════════════════════════════════════════════════════════

def test_ddi_warfarin_aspirin():
	svc = make_svc()
	ddis = run(svc.check_drug_drug_interactions(["warfarin", "aspirin"]))
	assert len(ddis) == 1
	assert ddis[0]["severity"] == "major"


def test_ddi_simvastatin_clarithromycin_contraindicated():
	svc = make_svc()
	ddis = run(svc.check_drug_drug_interactions(["simvastatin", "clarithromycin"]))
	assert any(d["severity"] == "contraindicated" for d in ddis)


def test_ddi_no_interaction():
	svc = make_svc()
	ddis = run(svc.check_drug_drug_interactions(["paracetamol", "amoxicillin"]))
	assert ddis == []


def test_paediatric_dose_check_within_range():
	svc = make_svc()
	result = run(svc.paediatric_dose_check("paracetamol", 20.0, 60, 250.0, "oral"))
	assert result["status"] == "within_range"


def test_paediatric_dose_check_overdose():
	svc = make_svc()
	result = run(svc.paediatric_dose_check("paracetamol", 20.0, 60, 10000.0, "oral"))
	assert result["status"] == "overdose"


def test_paediatric_dose_check_underdose():
	svc = make_svc()
	result = run(svc.paediatric_dose_check("paracetamol", 20.0, 60, 1.0, "oral"))
	assert result["status"] == "underdose"


def test_pregnancy_category_x_hard_stop():
	svc = make_svc()
	result = run(svc.pregnancy_safety_check("warfarin", 1))
	assert result["category"] == "X"
	assert result["hard_stop"] is True


def test_pregnancy_category_b_safe():
	svc = make_svc()
	result = run(svc.pregnancy_safety_check("paracetamol", 2))
	assert result["category"] == "B"
	assert result["hard_stop"] is False


def test_renal_dose_adjustment_metformin():
	svc = make_svc()
	result = run(svc.renal_dose_adjustment("metformin", 25.0))
	assert result["contraindicated"] is True


def test_renal_dose_no_adjustment_needed():
	svc = make_svc()
	result = run(svc.renal_dose_adjustment("metformin", 75.0))
	assert result["adjustment_required"] is False


def test_controlled_substance_check_schedule_ii_cap():
	svc = make_svc()
	result = run(svc.controlled_substance_check("morphine", "II", 45, "dr1"))
	assert result["exceeds_cap"] is True
	assert result["approved"] is False


def test_controlled_substance_within_cap():
	svc = make_svc()
	result = run(svc.controlled_substance_check("morphine", "II", 28, "dr1"))
	assert result["exceeds_cap"] is False
	assert result["approved"] is True


def test_duplicate_therapy_detection():
	svc = make_svc()
	p = run(svc.register_patient(make_patient_payload()))
	# prescribe atorvastatin (statin class)
	run(svc.prescribe_medication(MedicationCreate(
		tenant_id="t1", patient_id=p.id, drug_name="atorvastatin",
		dose="20mg", route="oral", frequency="daily",
		prescriber_id="dr1", allergy_check_performed=True, created_by="dr1",
	)))
	result = run(svc.check_duplicate_therapy(p.id, "simvastatin", "statin"))
	assert result["duplicate_found"] is True
	assert result["class_duplicates"]


# ═══════════════════════════════════════════════════════════════════════════════
# VITALS
# ═══════════════════════════════════════════════════════════════════════════════

def test_record_vital_blood_pressure():
	svc = make_svc()
	vital = run(svc.record_vital(VitalSignCreate(
		tenant_id="t1", patient_id="p1", encounter_id="e1",
		vital_type="blood_pressure", value=140.0, value2=90.0, unit="mmHg",
		recorded_by="nurse1",
	)))
	assert vital.vital_type == "blood_pressure"
	assert vital.value2 == 90.0


def test_list_vitals_filtered_by_type():
	svc = make_svc()
	run(svc.record_vital(VitalSignCreate(tenant_id="t1", patient_id="p1", encounter_id="e1", vital_type="blood_pressure", value=120.0, unit="mmHg", recorded_by="n1")))
	run(svc.record_vital(VitalSignCreate(tenant_id="t1", patient_id="p1", encounter_id="e1", vital_type="heart_rate", value=72.0, unit="bpm", recorded_by="n1")))
	bps = run(svc.list_vitals("t1", "p1", vital_type="blood_pressure"))
	assert all(v.vital_type == "blood_pressure" for v in bps)
	assert len(bps) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# LAB ORDERS & RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_order_and_receive_lab():
	svc = make_svc()
	p = run(svc.register_patient(make_patient_payload()))
	enc = make_encounter(svc, p.id)
	order = run(svc.order_lab_test(LabOrderCreate(
		tenant_id="t1", patient_id=p.id, encounter_id=enc.id,
		ordering_provider_id="dr1", test_code="2093-3",
		test_name="Total Cholesterol", priority="routine", created_by="dr1",
	)))
	assert order.status == "requested"

	result = run(svc.receive_lab_result(LabResultCreate(
		tenant_id="t1", order_id=order.id, patient_id=p.id,
		test_code="2093-3", test_name="Total Cholesterol",
		value="5.8", value_numeric=5.8, unit="mmol/L",
		reference_range="<5.0", flag=LabResultFlag.high,
		created_by="lab_tech",
	)))
	assert result.flag == LabResultFlag.high
	# order should now be completed
	updated_order = run(svc.get_lab_order("t1", order.id))
	assert updated_order.status == "completed"


def test_critical_lab_notification():
	svc = make_svc()
	p = run(svc.register_patient(make_patient_payload()))
	enc = make_encounter(svc, p.id)
	order = run(svc.order_lab_test(LabOrderCreate(
		tenant_id="t1", patient_id=p.id, encounter_id=enc.id,
		ordering_provider_id="dr1", test_code="2339-0",
		test_name="Blood Glucose", priority="stat", created_by="dr1",
	)))
	result = run(svc.receive_lab_result(LabResultCreate(
		tenant_id="t1", order_id=order.id, patient_id=p.id,
		test_code="2339-0", test_name="Blood Glucose",
		value="1.9", value_numeric=1.9, unit="mmol/L",
		flag=LabResultFlag.critical_low, created_by="lab_tech",
	)))
	assert not result.critical_notified
	notified = run(svc.flag_critical_lab_result(result.id, "dr_oncall"))
	assert notified.critical_notified is True
	assert notified.critical_notified_to == "dr_oncall"


def test_list_unnotified_critical_labs():
	svc = make_svc()
	p = run(svc.register_patient(make_patient_payload()))
	enc = make_encounter(svc, p.id)
	order = run(svc.order_lab_test(LabOrderCreate(
		tenant_id="t1", patient_id=p.id, encounter_id=enc.id,
		ordering_provider_id="dr1", test_code="718-7",
		test_name="Haemoglobin", priority="stat", created_by="dr1",
	)))
	run(svc.receive_lab_result(LabResultCreate(
		tenant_id="t1", order_id=order.id, patient_id=p.id,
		test_code="718-7", test_name="Haemoglobin",
		value="4.5", value_numeric=4.5, unit="g/dL",
		flag=LabResultFlag.critical_low, created_by="lab_tech",
	)))
	unnotified = run(svc.list_unnotified_critical_labs("t1"))
	assert len(unnotified) >= 1


def test_cancel_lab_order():
	svc = make_svc()
	p = run(svc.register_patient(make_patient_payload()))
	enc = make_encounter(svc, p.id)
	order = run(svc.order_lab_test(LabOrderCreate(
		tenant_id="t1", patient_id=p.id, encounter_id=enc.id,
		ordering_provider_id="dr1", test_code="1234-5",
		test_name="CBC", created_by="dr1",
	)))
	cancelled = run(svc.cancel_lab_order("t1", order.id))
	assert cancelled.status == "cancelled"


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGING
# ═══════════════════════════════════════════════════════════════════════════════

def test_order_imaging_creates_accession():
	svc = make_svc()
	p = run(svc.register_patient(make_patient_payload()))
	enc = make_encounter(svc, p.id)
	order = run(svc.order_imaging(ImagingOrderCreate(
		tenant_id="t1", patient_id=p.id, encounter_id=enc.id,
		ordering_provider_id="dr1", modality="CXR",
		body_part="chest", clinical_indication="pneumonia",
		created_by="dr1",
	)))
	assert order.accession_number is not None
	assert order.status == "requested"


def test_add_imaging_report():
	svc = make_svc()
	p = run(svc.register_patient(make_patient_payload()))
	enc = make_encounter(svc, p.id)
	order = run(svc.order_imaging(ImagingOrderCreate(
		tenant_id="t1", patient_id=p.id, encounter_id=enc.id,
		ordering_provider_id="dr1", modality="CT",
		body_part="chest", created_by="dr1",
	)))
	reported = run(svc.add_imaging_report(order.id, "rad1", "No acute cardiopulmonary pathology"))
	assert reported.status == "completed"
	assert "pathology" in reported.impression


# ═══════════════════════════════════════════════════════════════════════════════
# CARE PLANS
# ═══════════════════════════════════════════════════════════════════════════════

def test_create_and_activate_care_plan():
	svc = make_svc()
	p = run(svc.register_patient(make_patient_payload()))
	plan = run(svc.create_care_plan(CarePlanCreate(
		tenant_id="t1", patient_id=p.id,
		title="Diabetes Management Plan",
		goal="HbA1c < 7% within 6 months",
		icd10_codes=["E11.9"],
		created_by="dr1",
	)))
	assert plan.status == "draft"
	activated = run(svc.activate_care_plan("t1", plan.id))
	assert activated.status == "active"


def test_complete_care_plan():
	svc = make_svc()
	p = run(svc.register_patient(make_patient_payload()))
	plan = run(svc.create_care_plan(CarePlanCreate(
		tenant_id="t1", patient_id=p.id,
		title="Post-op rehab", goal="Full mobility", created_by="dr1",
	)))
	run(svc.activate_care_plan("t1", plan.id))
	completed = run(svc.complete_care_plan("t1", plan.id))
	assert completed.status == "completed"


# ═══════════════════════════════════════════════════════════════════════════════
# REFERRALS
# ═══════════════════════════════════════════════════════════════════════════════

def test_create_and_accept_referral():
	svc = make_svc()
	p = run(svc.register_patient(make_patient_payload()))
	ref = run(svc.create_referral(p.id, "dr1", "cardiology", "Chest pain workup", "urgent"))
	assert ref["status"] == "active"
	accepted = run(svc.accept_referral(ref["id"], "cardiologist1", "2026-07-15"))
	assert accepted["status"] == "completed"
	assert accepted["appointment_date"] == "2026-07-15"


def test_cancel_referral():
	svc = make_svc()
	p = run(svc.register_patient(make_patient_payload()))
	ref = run(svc.create_referral(p.id, "dr1", "nephrology", "CKD follow-up", "routine"))
	cancelled = run(svc.cancel_referral(ref["id"], "Patient declined"))
	assert cancelled["status"] == "cancelled"


# ═══════════════════════════════════════════════════════════════════════════════
# CONSENTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_record_and_check_consent():
	svc = make_svc()
	p = run(svc.register_patient(make_patient_payload()))
	run(svc.record_consent(p.id, "treatment", "nurse1", "2099-12-31"))
	result = run(svc.check_consent(p.id, "treatment"))
	assert result["consent_present"] is True


def test_emergency_consent_override():
	svc = make_svc()
	p = run(svc.register_patient(make_patient_payload()))
	override = run(svc.emergency_consent_override(p.id, "Unconscious patient, life-threatening", "dr_emergency"))
	assert override["override"] is True
	assert override["authorised_by"] == "dr_emergency"


def test_minor_consent():
	svc = make_svc()
	p = run(svc.register_patient(PatientCreate(
		tenant_id="t1",
		name=PatientName(family="Kamau", given=["Junior"]),
		birth_date=date(2015, 1, 1),
		gender="male",
		created_by="test",
	)))
	consent = run(svc.minor_consent(p.id, "guardian_001", "mother", "treatment"))
	assert consent["minor_consent"] is True
	assert consent["guardian_id"] == "guardian_001"


# ═══════════════════════════════════════════════════════════════════════════════
# IMMUNISATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def test_record_immunisation():
	svc = make_svc()
	p = run(svc.register_patient(make_patient_payload()))
	imm = run(svc.record_immunisation(ImmunisationCreate(
		tenant_id="t1", patient_id=p.id,
		vaccine_code="140", vaccine_name="Influenza, seasonal",
		administered_date=date(2026, 5, 15),
		administered_by="nurse2",
		created_by="nurse2",
	)))
	assert imm.vaccine_code == "140"
	assert imm.status == "completed"


def test_list_immunisations():
	svc = make_svc()
	p = run(svc.register_patient(make_patient_payload()))
	run(svc.record_immunisation(ImmunisationCreate(
		tenant_id="t1", patient_id=p.id,
		vaccine_code="140", vaccine_name="Influenza",
		administered_date=date(2026, 5, 1), administered_by="n1", created_by="n1",
	)))
	run(svc.record_immunisation(ImmunisationCreate(
		tenant_id="t1", patient_id=p.id,
		vaccine_code="115", vaccine_name="Td (tetanus/diphtheria)",
		administered_date=date(2026, 4, 1), administered_by="n1", created_by="n1",
	)))
	imms = run(svc.list_immunisations("t1", p.id))
	assert len(imms) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# FAMILY HISTORY
# ═══════════════════════════════════════════════════════════════════════════════

def test_add_and_list_family_history():
	svc = make_svc()
	p = run(svc.register_patient(make_patient_payload()))
	fhx = run(svc.add_family_history(FamilyHistoryCreate(
		tenant_id="t1", patient_id=p.id,
		relationship="father",
		deceased=True,
		age_at_death=65,
		conditions=["I25.10", "E11.9"],
		notes="Father had CAD and T2DM",
		created_by="dr1",
	)))
	assert fhx.relationship == "father"
	all_fhx = run(svc.list_family_history("t1", p.id))
	assert len(all_fhx) == 1
	assert "I25.10" in all_fhx[0].conditions


# ═══════════════════════════════════════════════════════════════════════════════
# CLINICAL DECISION SUPPORT
# ═══════════════════════════════════════════════════════════════════════════════

def test_cds_drug_allergy_alert_raised():
	svc = make_svc()
	p = run(svc.register_patient(make_patient_payload()))
	run(svc.record_allergy(AllergyCreate(
		tenant_id="t1", patient_id=p.id,
		allergen="penicillin", allergy_type="drug",
		severity="severe", reaction="anaphylaxis", created_by="dr1",
	)))
	run(svc.prescribe_medication(MedicationCreate(
		tenant_id="t1", patient_id=p.id, drug_name="penicillin",
		dose="500mg", route="oral", frequency="QID",
		prescriber_id="dr1", allergy_check_performed=True, created_by="dr1",
	)))
	alerts = run(svc.clinical_decision_support(p.id))
	allergy_alerts = [a for a in alerts if a.alert_type == "drug_allergy"]
	assert len(allergy_alerts) >= 1
	assert allergy_alerts[0].severity in ("warning", "critical")


def test_cds_ddi_alert_raised():
	svc = make_svc()
	p = run(svc.register_patient(make_patient_payload()))
	run(svc.prescribe_medication(MedicationCreate(
		tenant_id="t1", patient_id=p.id, drug_name="warfarin",
		dose="5mg", route="oral", frequency="daily",
		prescriber_id="dr1", allergy_check_performed=True, created_by="dr1",
	)))
	run(svc.prescribe_medication(MedicationCreate(
		tenant_id="t1", patient_id=p.id, drug_name="aspirin",
		dose="75mg", route="oral", frequency="daily",
		prescriber_id="dr1", allergy_check_performed=True, created_by="dr1",
	)))
	alerts = run(svc.clinical_decision_support(p.id))
	ddi_alerts = [a for a in alerts if a.alert_type == "drug_interaction"]
	assert len(ddi_alerts) >= 1


def test_chads2_vasc_score():
	svc = make_svc()
	p = run(svc.register_patient(make_patient_payload()))
	# add CHF and hypertension to problem list
	run(svc.add_problem(ProblemCreate(tenant_id="t1", patient_id=p.id, icd10_code="I50.9", description="Heart failure", created_by="dr1")))
	run(svc.add_problem(ProblemCreate(tenant_id="t1", patient_id=p.id, icd10_code="I10", description="Hypertension", created_by="dr1")))
	result = run(svc.CHADS2_VASc_score(p.id))
	assert result["score"] >= 2  # CHF=1 + HTN=1


def test_qsofa_score_sepsis_positive():
	svc = make_svc()
	result = run(svc.QSOFA_score("p1", respiratory_rate=24, mentation_altered=True, sbp=95))
	assert result["score"] == 3
	assert result["sepsis_screen_positive"] is True


def test_news2_high_risk():
	svc = make_svc()
	result = run(svc.NEWS2_score("p1", {
		"respiratory_rate": 28, "spo2": 88, "supplemental_oxygen": True,
		"systolic_bp": 88, "heart_rate": 125, "temperature": 39.5, "consciousness": "V",
	}))
	assert result["risk_level"] in ("medium", "high")
	assert result["total_score"] >= 5


def test_wells_pe_high_probability():
	svc = make_svc()
	result = run(svc.WELLS_score_PE("p1", {
		"dvt_signs": True, "pe_most_likely_diagnosis": True,
		"heart_rate_gt_100": True, "haemoptysis": False,
		"malignancy": False, "immobilisation_or_surgery": False,
		"prior_dvt_or_pe": False,
	}))
	assert result["probability"] == "high"
	assert result["score"] >= 7


def test_clinical_reminder_check_diabetes():
	svc = make_svc()
	p = run(svc.register_patient(make_patient_payload()))
	run(svc.add_problem(ProblemCreate(tenant_id="t1", patient_id=p.id, icd10_code="E11.9", description="T2DM", created_by="dr1")))
	reminders = run(svc.clinical_reminder_check(p.id))
	# should include HbA1c reminder for E11 prefix
	assert any("HbA1c" in r["description"] or "hba1c" in r["reminder_key"] for r in reminders)


def test_guideline_alert_diabetes():
	svc = make_svc()
	alerts = run(svc.clinical_guideline_alert("p1", "E11.9"))
	assert len(alerts) >= 1
	assert any("metformin" in a["title"].lower() or "diabetes" in a["title"].lower() for a in alerts)


# ═══════════════════════════════════════════════════════════════════════════════
# FHIR EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

def test_fhir_export_requires_phi_consent():
	svc = make_svc()
	try:
		run(svc.fhir_export("t1", "p1", ["Condition"], phi_consent_present=False))
		assert False, "Should raise PolicyViolationError"
	except PolicyViolationError:
		pass


def test_fhir_export_bundle_structure():
	svc = make_svc()
	p = run(svc.register_patient(make_patient_payload()))
	run(svc.add_problem(ProblemCreate(tenant_id="t1", patient_id=p.id, icd10_code="I10", description="HTN", created_by="dr1")))
	bundle = run(svc.fhir_export("t1", p.id, ["Condition"], phi_consent_present=True))
	assert bundle["resourceType"] == "Bundle"
	assert bundle["type"] == "collection"
	entries = bundle["entry"]
	assert any(e["resource"]["resourceType"] == "Condition" for e in entries)


def test_fhir_bundle_export_multi_resource():
	svc = make_svc()
	p = run(svc.register_patient(make_patient_payload()))
	run(svc.add_problem(ProblemCreate(tenant_id="t1", patient_id=p.id, icd10_code="I10", description="HTN", created_by="dr1")))
	run(svc.prescribe_medication(MedicationCreate(
		tenant_id="t1", patient_id=p.id, drug_name="amlodipine",
		dose="5mg", route="oral", frequency="daily",
		prescriber_id="dr1", allergy_check_performed=True, created_by="dr1",
	)))
	bundle = run(svc.fhir_bundle_export(p.id, ["Patient", "Condition", "MedicationRequest"]))
	resource_types = {e["resource"]["resourceType"] for e in bundle["entry"]}
	assert "Patient" in resource_types
	assert "Condition" in resource_types
	assert "MedicationRequest" in resource_types


# ═══════════════════════════════════════════════════════════════════════════════
# HL7 MESSAGE PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def test_hl7_adt_a01_processed():
	svc = make_svc()
	msg = (
		"MSH|^~\\&|EMR|HOSPITAL|LAB|HOSPITAL|20260601120000||ADT^A01|MSG001|P|2.5\r"
		"EVN|A01|20260601120000\r"
		"PID|1||P123^^^HOSP^MR||Kamau^John||19850615|M\r"
	)
	result = run(svc.hl7_message_processing(msg))
	assert result["ack_code"] == "AA"
	assert "patient_admit_notification_received" in result["actions_taken"]


def test_hl7_oru_r01_result():
	svc = make_svc()
	msg = (
		"MSH|^~\\&|LAB|HOSPITAL|EMR|HOSPITAL|20260601130000||ORU^R01|MSG002|P|2.5\r"
		"OBR|1||ACC001|HBA1C^HbA1c\r"
		"OBX|1|NM|4548-4^HbA1c|7.2|%|4.0-6.0|H|||F\r"
	)
	result = run(svc.hl7_message_processing(msg))
	assert result["ack_code"] == "AA"
	assert "observation_result_received" in result["actions_taken"]
	assert result["obx_count"] == 1


def test_hl7_empty_message_raises():
	svc = make_svc()
	try:
		run(svc.hl7_message_processing(""))
		assert False
	except ValueError:
		pass


# ═══════════════════════════════════════════════════════════════════════════════
# CALCULATIONS (domain/calculations.py)
# ═══════════════════════════════════════════════════════════════════════════════

def test_bmi_normal():
	from emr.domain.calculations import calculate_bmi, bmi_category
	bmi = calculate_bmi(70, 175)
	assert 22 < bmi < 23
	assert bmi_category(bmi) == "normal"


def test_bmi_obese():
	from emr.domain.calculations import calculate_bmi, bmi_category
	bmi = calculate_bmi(100, 160)
	assert bmi_category(bmi) in ("obese_class_i", "obese_class_ii", "obese_class_iii")


def test_cockroft_gault_egfr():
	from emr.domain.calculations import cockroft_gault_egfr, ckd_stage
	egfr = cockroft_gault_egfr(age_years=65, weight_kg=70, serum_creatinine_umol_L=120, is_female=False)
	assert 50 < egfr < 80
	assert "G2" in ckd_stage(egfr) or "G3" in ckd_stage(egfr)


def test_child_pugh_class_a():
	from emr.domain.calculations import child_pugh_score
	score, cls = child_pugh_score(20, 38, 1.2, "none", "none")
	assert cls == "A"
	assert score == 5


def test_child_pugh_class_c():
	from emr.domain.calculations import child_pugh_score
	score, cls = child_pugh_score(60, 25, 2.5, "moderate_severe", "grade_3_4")
	assert cls == "C"


def test_patient_match_score_exact():
	from emr.domain.calculations import patient_match_score
	a = {"family": "Njoroge", "given_0": "Peter", "birth_date": "1990-03-20", "gender": "male"}
	b = {"family": "Njoroge", "given_0": "Peter", "birth_date": "1990-03-20", "gender": "male"}
	score, fields = patient_match_score(a, b)
	assert score >= 0.45
	assert "birth_date" in fields
	assert "family_name_exact" in fields


def test_patient_match_score_no_match():
	from emr.domain.calculations import patient_match_score
	a = {"family": "Smith", "given_0": "Alice", "birth_date": "1970-01-01", "gender": "female"}
	b = {"family": "Jones", "given_0": "Bob", "birth_date": "1985-06-15", "gender": "male"}
	score, _ = patient_match_score(a, b)
	assert score < 0.30


def test_interpret_blood_pressure_hypertensive_crisis():
	from emr.domain.calculations import interpret_blood_pressure
	assert interpret_blood_pressure(185, 125) == "hypertensive_crisis"


def test_interpret_temperature_hypothermia():
	from emr.domain.calculations import interpret_temperature_celsius
	assert interpret_temperature_celsius(34.5) == "hypothermia"


def test_is_critical_vital_spo2():
	from emr.domain.calculations import is_critical_vital
	assert is_critical_vital("oxygen_saturation", 88.0) is True
	assert is_critical_vital("oxygen_saturation", 97.0) is False


def test_news2_subscore():
	from emr.domain.calculations import news2_subscore
	assert news2_subscore("respiratory_rate", 28) == 3
	assert news2_subscore("respiratory_rate", 16) == 0
	assert news2_subscore("spo2", 88) == 3


def test_estimate_weight_by_age():
	from emr.domain.calculations import estimate_weight_by_age_kg
	# neonate
	assert estimate_weight_by_age_kg(0) == 3.5
	# 12-month infant
	w = estimate_weight_by_age_kg(12)
	assert 9 < w < 11
	# 5-year-old: Broselow ≈ 2*5+8=18
	w5 = estimate_weight_by_age_kg(60)
	assert 17 < w5 < 19


def test_maintenance_fluid_rate():
	from emr.domain.calculations import maintenance_fluid_rate_ml_per_hour
	# 10 kg child → 100 mL/h ÷ 24 ≈ 4.2 mL/h
	rate = maintenance_fluid_rate_ml_per_hour(10)
	assert abs(rate - 41.7) < 1


def test_anion_gap_normal():
	from emr.domain.calculations import anion_gap
	ag = anion_gap(140, 104, 24)
	assert ag == 12.0


# ═══════════════════════════════════════════════════════════════════════════════
# DOMAIN RULES (domain/rules.py)
# ═══════════════════════════════════════════════════════════════════════════════

def test_assert_tenant_context_raises():
	from emr.domain.rules import assert_tenant_context, RuleViolation
	try:
		assert_tenant_context({})
		assert False
	except RuleViolation as e:
		assert e.rule_name == "tenant_context_required"


def test_assert_no_cross_tenant_raises():
	from emr.domain.rules import assert_no_cross_tenant_access, RuleViolation
	try:
		assert_no_cross_tenant_access("tenant_a", "tenant_b")
		assert False
	except RuleViolation as e:
		assert "cross_tenant" in e.rule_name


def test_assert_allergy_check_performed_raises():
	from emr.domain.rules import assert_allergy_check_performed, RuleViolation
	try:
		assert_allergy_check_performed(False)
		assert False
	except RuleViolation as e:
		assert e.rule_name == "allergy_check_required"


def test_assert_controlled_quantity_cap():
	from emr.domain.rules import assert_controlled_quantity_within_cap, RuleViolation
	try:
		assert_controlled_quantity_within_cap(45, "II")
		assert False
	except RuleViolation as e:
		assert "controlled_substance" in e.rule_name


def test_assert_paediatric_dose_overdose():
	from emr.domain.rules import assert_paediatric_dose_in_range, RuleViolation
	try:
		assert_paediatric_dose_in_range(500.0, 10.0, 100.0, "paracetamol")
		assert False
	except RuleViolation as e:
		assert e.rule_name == "paediatric_overdose"


def test_assert_pregnancy_category_x():
	from emr.domain.rules import assert_pregnancy_safe, RuleViolation
	try:
		assert_pregnancy_safe("X", "warfarin")
		assert False
	except RuleViolation as e:
		assert "pregnancy_category_x" in e.rule_name


def test_is_probable_duplicate():
	from emr.domain.rules import is_probable_duplicate
	assert is_probable_duplicate(0.75) is True
	assert is_probable_duplicate(0.30) is False


def test_calculate_refills_remaining():
	from emr.domain.rules import calculate_refills_remaining
	assert calculate_refills_remaining(5, 3) == 2
	assert calculate_refills_remaining(2, 2) == 0
	assert calculate_refills_remaining(1, 3) == 0  # used > allowed → clamp to 0


# ═══════════════════════════════════════════════════════════════════════════════
# REPORTS & DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

def test_dashboard_summary_structure():
	svc = make_svc()
	p = run(svc.register_patient(make_patient_payload()))
	enc = make_encounter(svc, p.id)
	run(svc.create_note(ClinicalNoteCreate(
		tenant_id="t1", patient_id=p.id, encounter_id=enc.id,
		note_type="soap_note", author_id="dr1", content="Note",
	)))
	summary = run(svc.dashboard_summary("t1"))
	assert "encounters" in summary
	assert "notes" in summary
	assert summary["encounters"]["total"] >= 1
	assert summary["notes"]["total"] >= 1


def test_generate_clinical_summary():
	svc = make_svc()
	p = run(svc.register_patient(make_patient_payload()))
	run(svc.add_problem(ProblemCreate(tenant_id="t1", patient_id=p.id, icd10_code="I10", description="HTN", created_by="dr1")))
	run(svc.prescribe_medication(MedicationCreate(
		tenant_id="t1", patient_id=p.id, drug_name="amlodipine",
		dose="5mg", route="oral", frequency="daily",
		prescriber_id="dr1", allergy_check_performed=True, created_by="dr1",
	)))
	summary = run(svc.generate_clinical_summary(p.id))
	assert summary["problem_count"] >= 1
	assert summary["medication_count"] >= 1


def test_controlled_substance_report():
	svc = make_svc()
	p = run(svc.register_patient(make_patient_payload()))
	enc = make_encounter(svc, p.id)
	# morphine triggers Schedule II check which will fail at 45 days — use 28
	try:
		run(svc.create_prescription(
			patient_id=p.id, drug="morphine", dose=10.0,
			frequency="Q4H", duration_days=5, route="iv",
			prescriber_id="dr1", encounter_id=enc.id,
		))
	except DrugSafetyError:
		pass
	report = run(svc.controlled_substance_report("t1"))
	assert "total_controlled_prescriptions" in report
	assert report["date"] is not None
