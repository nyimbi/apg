"""Healthcare capability integration tests: EMRService.

All tests are sync; async service methods called via asyncio.run().
Uses the _NullStore in-memory backend — zero config.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio
from datetime import date, datetime


# ── helpers ───────────────────────────────────────────────────────────────────

def _svc(tenant_id: str = "test-tenant"):
	from capabilities.healthcare.emr.service import EMRService
	return EMRService(tenant_id=tenant_id, actor_id="test-actor")


def _register_patient(svc, tenant_id: str = "test-tenant", family: str = "Smith", given: str = "John"):
	"""Helper: register a patient and return the PatientResponse."""
	from capabilities.healthcare.emr.models import PatientCreate, PatientName, Gender
	payload = PatientCreate(
		tenant_id=tenant_id,
		name=PatientName(family=family, given=[given]),
		birth_date=date(1985, 6, 15),
		gender=Gender.male,
		created_by="test-actor",
	)
	return asyncio.run(svc.register_patient(payload))


def _add_problem(svc, patient_id: str, icd10: str = "E11", description: str = "Type 2 diabetes"):
	from capabilities.healthcare.emr.models import ProblemCreate
	payload = ProblemCreate(
		tenant_id=svc.tenant_id,
		patient_id=patient_id,
		icd10_code=icd10,
		description=description,
		status="active",
		created_by="test-actor",
	)
	return asyncio.run(svc.add_problem(payload))


# ── 1. patient registration ───────────────────────────────────────────────────

def test_emr_patient_registration():
	"""EMRService.register_patient creates a patient with the correct fields."""
	svc = _svc()
	patient = _register_patient(svc)
	assert patient.id
	assert patient.name.family == "Smith"
	assert patient.name.given == ["John"]
	assert patient.tenant_id == "test-tenant"
	assert str(patient.gender) in ("male", "Gender.male")


# ── 2. drug–drug interaction check ───────────────────────────────────────────

def test_emr_drug_interactions():
	"""check_drug_drug_interactions returns a list (possibly empty)."""
	svc = _svc()
	# warfarin + aspirin is a known major interaction
	result = asyncio.run(svc.check_drug_drug_interactions(["warfarin", "aspirin"]))
	assert isinstance(result, list)
	assert len(result) >= 1
	interaction = result[0]
	assert "drug_a" in interaction
	assert "severity" in interaction
	assert interaction["severity"] in ("major", "contraindicated")


# ── 3. clinical reminder check ────────────────────────────────────────────────

def test_emr_clinical_reminder():
	"""clinical_reminder_check returns a list (reminders fire for known ICD-10 triggers)."""
	svc = _svc()
	patient = _register_patient(svc)
	# E11 (diabetes) triggers HbA1c, foot exam, eye exam reminders
	_add_problem(svc, patient.id, icd10="E11", description="Type 2 diabetes mellitus")

	reminders = asyncio.run(svc.clinical_reminder_check(patient.id))
	assert isinstance(reminders, list)
	# at least one reminder should fire for a diabetic patient
	assert len(reminders) >= 1
	keys = {r["reminder_key"] for r in reminders}
	# hba1c, foot_exam or eye_exam should appear
	assert keys & {"hba1c", "foot_exam", "eye_exam"}


# ── 4. FHIR patient resource ──────────────────────────────────────────────────

def test_emr_fhir_export():
	"""fhir_patient_resource returns a dict with resourceType == 'Patient'."""
	svc = _svc()
	patient = _register_patient(svc)
	resource = asyncio.run(svc.fhir_patient_resource(patient.id))
	assert isinstance(resource, dict)
	assert resource["resourceType"] == "Patient"
	assert resource["id"] == patient.id


# ── 5. NEWS2 score ────────────────────────────────────────────────────────────

def test_emr_news2_score():
	"""NEWS2_score returns a dict with a 'total_score' key (int 0–20)."""
	svc = _svc()
	patient = _register_patient(svc)
	vitals = {
		"respiratory_rate": 18,
		"spo2": 97.0,
		"supplemental_oxygen": False,
		"systolic_bp": 125,
		"heart_rate": 80,
		"temperature": 37.2,
		"consciousness": "A",
	}
	result = asyncio.run(svc.NEWS2_score(patient.id, vitals))
	assert isinstance(result, dict)
	assert "total_score" in result
	assert isinstance(result["total_score"], int)
	assert 0 <= result["total_score"] <= 20


# ── 6. rule evaluation ────────────────────────────────────────────────────────

def test_emr_rule_evaluation():
	"""EMR rule engine allows when tenant context is present."""
	from capabilities.healthcare.emr.capability_contract import evaluate_capability_rules

	result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation_type": "read",
		"policy_attached": True,
	})
	assert result["decision"] == "allow"

	# Missing tenant should deny
	deny_result = evaluate_capability_rules({"tenant_context_present": False})
	assert deny_result["decision"] == "deny"


# ── 7. healthcare manifest ────────────────────────────────────────────────────

def test_healthcare_manifest():
	"""Healthcare domain contains exactly 9 capabilities in the manifest."""
	from capabilities.manifest import get_domain
	caps = get_domain("healthcare")
	assert len(caps) == 9, f"expected 9 healthcare capabilities, got {len(caps)}"
	ids = {c.get("capability_id") or c.get("id") or c.get("code") for c in caps}
	assert any("emr" in str(cid) for cid in ids), f"healthcare_emr not found in: {ids}"


# ── 8. clinical decision support — generate_clinical_summary ─────────────────

def test_clinical_decision_support():
	"""generate_clinical_summary returns a dict with a 'patient_id' key."""
	svc = _svc()
	patient = _register_patient(svc, family="Jones", given="Mary")
	_add_problem(svc, patient.id, icd10="I10", description="Hypertension")

	summary = asyncio.run(svc.generate_clinical_summary(patient.id))
	assert isinstance(summary, dict)
	assert "patient_id" in summary
	assert summary["patient_id"] == patient.id
	assert "active_problems" in summary
	assert "active_medications" in summary
	assert "allergies" in summary
