# EMR Capability — User Guide

## Overview

The APG Electronic Medical Records capability (`healthcare_emr`) provides a full clinical record system built on FHIR R4 standards. It handles the entire patient encounter lifecycle from registration through discharge, including medication safety, clinical decision support, lab management, and population health analytics.

All service methods are async and follow the tenant-scoped adapter pattern. Instantiate `EMRService` with a `tenant_id` and optional adapters for auth, audit, and notifications.

---

## Quick Start

```python
from capabilities.healthcare.emr.service import EMRService
from capabilities.healthcare.emr.models import (
    PatientCreate, HumanName, EncounterCreate, ProblemCreate,
    MedicationCreate, AllergyCreate, VitalSignCreate,
)

svc = EMRService(tenant_id="clinic_a", actor_id="dr_smith")

# Register a patient
patient = await svc.register_patient(PatientCreate(
    tenant_id="clinic_a",
    name=HumanName(family="Wanjiru", given=["Grace"]),
    birth_date=date(1985, 4, 12),
    gender="female",
    created_by="dr_smith",
))

# Open an encounter
enc = await svc.create_encounter(EncounterCreate(
    tenant_id="clinic_a",
    patient_id=patient.id,
    encounter_type="outpatient",
    provider_id="dr_smith",
    chief_complaint="Cough and fever for 3 days",
    created_by="dr_smith",
))
```

---

## Patient Management

### Registration with Deduplication

`register_patient()` runs probabilistic matching against all existing patients before creating a new record. A match score ≥ 0.85 raises `PolicyViolationError`; scores 0.40–0.84 are returned as candidates for review without blocking registration.

```python
patient = await svc.register_patient(payload)
# Raises PolicyViolationError if certain duplicate found
```

### Patient Search

```python
patients = await svc.list_patients(
    tenant_id="clinic_a",
    search="wanjiru",        # family or given name substring
    status="active",
)
```

### Merging Duplicate Records

```python
result = await svc.merge_patients(
    tenant_id="clinic_a",
    duplicate_id="pid_old",
    surviving_id="pid_keep",
)
# All encounters, notes, problems, medications, allergies re-keyed to surviving record
```

---

## Encounter Lifecycle

```
create_encounter → admit_patient → [transfer_patient] → discharge_patient
                                                       └→ create_discharge_summary
```

```python
# Create and admit
enc = await svc.create_encounter(payload)
enc = await svc.admit_patient(tenant_id, enc.id, admit_data={})

# Transfer to ICU
transfer = await svc.transfer_patient(
    encounter_id=enc.id,
    to_location_id="icu_ward_3",
    to_provider_id="dr_otieno",
    reason="Deteriorating haemodynamics, NEWS2=8",
)

# Discharge
result = await svc.discharge_patient(
    encounter_id=enc.id,
    discharge_diagnosis="Community-acquired pneumonia (J18.9)",
    treatment_summary="IV amoxicillin 5 days, O2 therapy, fluid replacement.",
    follow_up="GP review in 1 week. Repeat CXR in 6 weeks.",
    discharge_medications=[{"drug_name": "Amoxicillin", "dose": "500mg", "frequency": "TDS"}],
)
```

---

## Clinical Notes

### Creating and Finalizing

```python
from capabilities.healthcare.emr.models import ClinicalNoteCreate

note = await svc.create_note(ClinicalNoteCreate(
    tenant_id="clinic_a",
    patient_id=patient.id,
    encounter_id=enc.id,
    note_type="soap",
    author_id="dr_smith",
    subjective="Patient reports productive cough, fever 38.8°C for 3 days.",
    objective="RR 22, SpO2 94% on air, crackles right base.",
    assessment="Community-acquired pneumonia.",
    plan="IV amoxicillin, O2 2L/min, admission.",
    icd10_codes=["J18.9"],
    content="",  # auto-assembled from SOAP fields
    created_by="dr_smith",
))

# Sign the note (makes it immutable)
signed = await svc.sign_clinical_note(note.id, clinician_id="dr_smith")

# Add addendum to signed note
addendum = await svc.addendum_to_note(
    note_id=note.id,
    addendum_text="CXR confirmed right lower lobe consolidation.",
    added_by="dr_smith",
)
```

### Note Types
`soap` | `progress` | `discharge_summary` | `operative` | `consult` | `procedure` | `nursing` | `history_physical` | `emergency` | `addendum`

---

## Medication Safety

The prescribing workflow runs a 4-stage safety gate before creating a prescription.

### Full Safety Gate (`create_prescription`)

```python
rx = await svc.create_prescription(
    patient_id=patient.id,
    drug="amoxicillin",
    dose=500.0,
    frequency="TDS",
    duration_days=7,
    route="oral",
    prescriber_id="dr_smith",
    encounter_id=enc.id,
)
# rx["safety_summary"] contains: allergy_conflicts, ddi_interactions,
# duplicate_therapy, controlled_substance_flags
```

Hard stops raise `DrugSafetyError`:
- Life-threatening allergy on record
- Controlled substance quantity cap exceeded
- PGx contraindication (when PGx results are on file)

### Individual Safety Checks

```python
# Drug-drug interactions
ddis = await svc.check_drug_drug_interactions(["warfarin", "aspirin", "ibuprofen"])

# Pregnancy safety
result = await svc.pregnancy_safety_check("warfarin", trimester=1)
# result["hard_stop"] == True → Category X

# Renal dose adjustment
adj = await svc.renal_dose_adjustment("metformin", egfr_ml_per_min=28.0)
# adj["contraindicated"] == True → eGFR < 30

# Paediatric dosing
check = await svc.paediatric_dose_check(
    drug="amoxicillin", weight_kg=15.0, age_months=36,
    prescribed_dose=200.0, route="oral",
)
# check["status"] == "within_range" | "underdose" | "overdose"
```

### Medication Reconciliation

```python
reconciliation = await svc.medication_reconciliation(
    patient_id=patient.id,
    encounter_id=enc.id,
    home_medications=[
        {"drug_name": "metformin", "dose": "500mg"},
        {"drug_name": "lisinopril", "dose": "10mg"},
    ],
)
# reconciliation["discrepancies"] lists omissions, commissions, dose discrepancies
```

---

## Vital Signs and Trend Analysis

### Recording

```python
from capabilities.healthcare.emr.models import VitalSignCreate

vital = await svc.record_vital(VitalSignCreate(
    tenant_id="clinic_a",
    patient_id=patient.id,
    encounter_id=enc.id,
    vital_type="blood_pressure_systolic",
    value=158.0,
    unit="mmHg",
    recorded_by="nurse_auma",
    recorded_at=datetime.utcnow(),
    created_by="nurse_auma",
))
```

### Trend Analysis

```python
trend = await svc.analyse_vital_trend(
    patient_id=patient.id,
    vital_type="blood_pressure_systolic",
    window_hours=12,
    alert_threshold_slope=5.0,  # mmHg/hour is significant
)
# trend["trend_direction"] = "worsening" | "stable" | "improving" | "insufficient_data"
# trend["alert"] = True if threshold exceeded or worsening + anomalies present
# trend["anomalous_readings"] = list of readings > 2 SD from mean
```

---

## Clinical Decision Support

### NEWS2 (National Early Warning Score 2)

```python
news2 = await svc.NEWS2_score(patient.id, vitals={
    "respiratory_rate": 22,
    "spo2": 94.0,
    "supplemental_oxygen": True,
    "systolic_bp": 105,
    "heart_rate": 112,
    "temperature": 38.4,
    "consciousness": "A",
})
# news2["total_score"], news2["risk_level"], news2["response_recommendation"]
```

| Score | Risk | Response |
|-------|------|----------|
| 0 | low_stable | Routine monitoring |
| 1–4 | low | Reassess per protocol |
| 5–6 | medium | Clinical review within 30 minutes |
| ≥7 | high | Urgent review, consider HDU/ICU |

### Charlson Comorbidity Index

```python
cci = await svc.compute_charlson_comorbidity_index(
    patient_id=patient.id,
    age_years=68,
)
# cci["cci_score"], cci["age_adjusted_cci"], cci["estimated_10yr_survival_pct"]
# cci["conditions_contributing"] lists each ICD-10 contributing to the score
```

### LACE+ Readmission Risk

```python
lace = await svc.compute_lace_plus_score(
    patient_id=patient.id,
    encounter_id=enc.id,
    age_years=68,
)
# lace["lace_plus_score"], lace["risk_tier"], lace["estimated_30day_readmission_pct"]
# lace["recommendation"] provides discharge planning guidance
```

### Sepsis Bundle Tracking

Triggered when qSOFA ≥ 2 or NEWS2 ≥ 7 is detected:

```python
tracker = await svc.track_sepsis_bundle(
    patient_id=patient.id,
    encounter_id=enc.id,
    recognition_time=datetime.utcnow(),
)
# tracker["compliance_pct"] — percentage of Hour-1 bundle completed
# tracker["incomplete_elements"] — list of outstanding bundle items
# tracker["alert"] — True if >45 minutes elapsed with incomplete elements
```

Bundle elements tracked: blood cultures, serum lactate, IV antibiotics, IV fluids, supplemental oxygen, urine output measurement.

---

## Lab Management

```python
from capabilities.healthcare.emr.models import LabOrderCreate, LabResultCreate, LabResultFlag

# Order
order = await svc.order_lab_test(LabOrderCreate(
    tenant_id="clinic_a",
    patient_id=patient.id,
    encounter_id=enc.id,
    ordering_provider_id="dr_smith",
    test_code="58410-2",
    test_name="FBC",
    specimen_type="blood",
    priority="routine",
    clinical_indication="Admission workup",
    created_by="dr_smith",
))

# Receive result (critical value auto-logged)
result = await svc.receive_lab_result(LabResultCreate(
    tenant_id="clinic_a",
    order_id=order.id,
    patient_id=patient.id,
    test_code="58410-2",
    test_name="Haemoglobin",
    value="6.2",
    value_numeric=6.2,
    unit="g/dL",
    reference_range="12.0–16.0",
    flag=LabResultFlag.critical_low,
    result_status="final",
    result_time=datetime.utcnow(),
    created_by="lab_system",
))

# Acknowledge critical result
await svc.flag_critical_lab_result(result.id, notified_to="dr_smith")
```

---

## Electronic Medication Administration Record (eMAR)

Record each individual dose administration — linking Five Rights verification to every event.

```python
# Record a dose given
event = await svc.record_dose_administration(
    patient_id=patient.id,
    encounter_id=enc.id,
    prescription_id=rx["id"],
    drug_name="amoxicillin",
    dose="500mg",
    route="oral",
    administered_by="nurse_auma",
    variance_code="none",
)

# Get eMAR report for the encounter
emar = await svc.get_emar_report(
    patient_id=patient.id,
    encounter_id=enc.id,
)
# emar["adherence_pct"], emar["total_variances"], emar["drug_summaries"]
```

Variance codes: `none` | `held` | `refused` | `partial` | `modified`

---

## Advance Directives

```python
directive = await svc.record_advance_directive(
    patient_id=patient.id,
    directive_type="dnr",
    provisions=[
        "No cardiopulmonary resuscitation",
        "Comfort measures only",
        "No mechanical ventilation",
    ],
    signed_date="2026-01-15",
    agent_name="John Wanjiru",
    agent_contact="+254 722 000 001",
)

# Retrieve all active directives (surfaces in CDS and clinical summaries)
directives = await svc.get_active_directives(patient_id=patient.id)
```

Directive types: `living_will` | `dnr` | `dnar` | `polst` | `molst` | `healthcare_proxy` | `organ_donation`

---

## Pharmacogenomics (PGx)

Store genetic test results and get CPIC-grade prescribing recommendations.

```python
# Record a PGx test result
pgx = await svc.record_pgx_result(
    patient_id=patient.id,
    gene="CYP2C19",
    diplotype="*4/*4",
    phenotype="poor metaboliser",
    tested_by="lab_genomics",
    test_date="2026-05-01",
    panel_name="CardioGenomics Panel",
)
# pgx["prescribing_alerts"] — immediate alerts for active medications

# Check before prescribing
check = await svc.check_pgx_prescribing(patient_id=patient.id, drug_name="clopidogrel")
# check["has_contraindication"] == True → DrugSafetyError raised in create_prescription
# check["alerts"][0]["recommendation"] → CPIC guidance text
```

---

## Population Health Cohort Builder

```python
# Identify diabetic patients lost to follow-up (no encounter in 6+ months)
cohort = await svc.build_patient_cohort(
    diagnosis_prefixes=["E11", "E10"],
    last_encounter_before_days=180,
    missing_screening_keys=["hba1c", "eye_exam", "foot_exam"],
)
# cohort["cohort_size"], cohort["members"]
# Each member: patient_id, family_name, last_encounter_days_ago, open_care_gaps

# Hypertensive patients not on first-line agents
cohort2 = await svc.build_patient_cohort(
    diagnosis_prefixes=["I10"],
    age_range=(40, 75),
)
```

---

## Disease Trajectory Tracking

Track longitudinal biomarker trends for chronic conditions:

```python
trajectory = await svc.get_disease_trajectory(
    patient_id=patient.id,
    icd10_prefix="E11",  # Type 2 Diabetes
)
# trajectory["trajectory"] — chronological list of HbA1c, eGFR, urine albumin, blood glucose values
# trajectory["biomarker_trends"] — per-biomarker trend direction
# trajectory["overall_trend"] — "stable" | "changing"
```

Supported prefixes: `E11`, `E10`, `I10`, `I50`, `N18`, `J44`

---

## FHIR R4 Export

```python
bundle = await svc.fhir_bundle_export(
    patient_id=patient.id,
    resource_types=[
        "Patient", "Encounter", "Condition", "MedicationRequest",
        "AllergyIntolerance", "Observation", "DocumentReference",
    ],
)
# Returns FHIR R4 transaction Bundle with all specified resources
```

Supported resource types: `Patient` | `Encounter` | `Condition` | `MedicationRequest` | `AllergyIntolerance` | `Observation` | `DocumentReference`

PGx results export as `MolecularSequence` resources. Advance directives export as `Consent` resources (included automatically in full bundles).

---

## HL7 v2 Message Processing

```python
hl7_msg = (
    "MSH|^~\\&|LAB|HOSP|EMR|CLINIC|20260601120000||ORU^R01|MSG001|P|2.5\r"
    "PID|1||P12345|||Wanjiru^Grace||19850412|F\r"
    "OBX|1|NM|718-7^Hemoglobin||6.2|g/dL|12.0-16.0|LL|||F\r"
)
ack = await svc.hl7_message_processing(hl7_msg)
# ack["ack_code"] == "AA", ack["actions_taken"]
```

Supported message types: `ADT^A01` (admit), `ADT^A03` (discharge), `ADT^A08` (update), `ORM^O01` (order), `ORU^R01` (observation result).

---

## Audit Trail

Every mutating operation appends an event to `svc._audit_events`:

```python
[{
    "tenant_id": "clinic_a",
    "actor_id": "dr_smith",
    "event": "medication_prescribed",
    "entity_id": "01920abc...",
    "timestamp": "2026-06-11T09:42:00.123456",
}]
```

The audit list is ordered by insertion sequence. In production, persist to an append-only PostgreSQL table with row-level security. Use `verify_audit_chain()` (improvement I11) to validate hash chain integrity for medicolegal compliance.

---

## Testing

```bash
# Run CI tests
uv run pytest -vxs capabilities/healthcare/emr/tests/

# Run full test suite including integration tests
uv run pytest -vxs capabilities/healthcare/emr/tests/test_emr_full.py
```

All tests use real `EMRService` instances with the null in-memory store — no mocks required. See `capabilities/healthcare/emr/tests/conftest.py` for fixtures.

---

## Error Reference

| Exception | When | Resolution |
|-----------|------|------------|
| `PolicyViolationError` | Capability rule denied | Check rule context; see `evaluate_capability_rules()` |
| `DrugSafetyError` | Hard stop in prescribing gate | Review safety_summary; allergy or CS cap exceeded |
| `ValueError` | Entity not found or invalid state | Check tenant_id, entity existence, and status |
| `AssertionError` | Missing required field | Pass all required parameters |

---

## Composability Keywords

The following keywords in a NATS message body or API payload trigger cross-capability integration:

- `fhir_export_requested` → triggers `healthcare_emr.fhir_bundle_export` + `healthcare_ana.ingest_bundle`
- `encounter_opened` → triggers `healthcare_pmt.register_visit_event`
- `lab_critical_result` → triggers `healthcare_ntfy.send_critical_alert`
- `medication_prescribed` → triggers `healthcare_pha.validate_formulary`
- `patient_discharged` → triggers `healthcare_ana.update_cohort_membership`
- `sepsis_bundle_incomplete` → triggers `healthcare_ntfy.escalate_deterioration`
- `pgx_contraindication` → triggers `healthcare_gen.flag_prescribing_barrier`
