# Electronic Medical Records

## Overview
Full-featured EMR capability providing patient chart management, SOAP and structured clinical note authoring, problem list maintenance with ICD-10 coding, medication reconciliation with allergy-check enforcement, vital signs recording, and HL7 FHIR R4 export. Designed for HIPAA compliance with cross-tenant PHI isolation enforced at the rule layer.

## Capability ID
`healthcare_emr`

## Provides
- patient_chart_management: Unified patient chart view aggregating notes, problems, medications, allergies, vitals, and encounters
- clinical_note_authoring: SOAP, progress, discharge, operative, and 7 other structured note types with co-signature and amendment workflows
- problem_list_management: ICD-10-coded active/chronic/resolved problem tracking per patient
- medication_reconciliation: Admission and discharge medication reconciliation with discrepancy flagging
- allergy_tracking: Drug, food, environmental, and contrast allergy recording with severity levels
- vital_signs_recording: 9 vital sign types with trend data per encounter
- fhir_r4_export: FHIR R4 Bundle export for Patient, Condition, MedicationRequest, AllergyIntolerance, and more
- icd10_coding: ICD-10 code assignment on problems, encounters, and notes
- encounter_management: Encounter lifecycle from open to closed with discharge summary linkage

## Requires
- auth: PHI access authorization
- audl: Audit trail for all chart modifications
- mten: Multi-tenant isolation
- conf: Tenant-specific configuration
- ntfy: Allergy conflict and critical value alerts
- nlpc: Clinical note search and NLP-assisted coding suggestions
- wflo: Note co-signature and amendment approval workflows
- mqeb: Event emission for downstream analytics and lab systems

## Configuration

| Key | Description |
|-----|-------------|
| notes.co_signature_required | Require co-signature for note finalization |
| notes.addendum_allowed | Allow addendum notes after finalization |
| medications.reconciliation_on_admission | Warn if reconciliation skipped on admission |
| fhir.version | FHIR version (currently R4) |
| governance.medication_allergy_check_required | Block prescription if allergy check not performed |

## API Routes

| Method | Path | Description | Permission |
|--------|------|-------------|------------|
| GET | /api/healthcare/emr/encounters | List encounters | healthcare_emr:encounters |
| POST | /api/healthcare/emr/encounters | Create encounter | healthcare_emr:encounters |
| GET | /api/healthcare/emr/encounters/<id> | Encounter detail | healthcare_emr:encounters |
| POST | /api/healthcare/emr/encounters/<id>/close | Close encounter | healthcare_emr:encounters |
| GET | /api/healthcare/emr/notes | List notes | healthcare_emr:notes |
| POST | /api/healthcare/emr/notes | Create note | healthcare_emr:notes_write |
| GET | /api/healthcare/emr/notes/<id> | Note detail | healthcare_emr:notes |
| POST | /api/healthcare/emr/notes/<id>/amend | Amend note | healthcare_emr:notes_write |
| POST | /api/healthcare/emr/notes/<id>/finalize | Finalize note | healthcare_emr:notes_write |
| GET | /api/healthcare/emr/patients/<id>/problems | Problem list | healthcare_emr:problems |
| POST | /api/healthcare/emr/problems | Add problem | healthcare_emr:problems |
| POST | /api/healthcare/emr/problems/<id>/resolve | Resolve problem | healthcare_emr:problems |
| GET | /api/healthcare/emr/patients/<id>/medications | Medication list | healthcare_emr:medications |
| POST | /api/healthcare/emr/medications | Prescribe medication | healthcare_emr:medications |
| POST | /api/healthcare/emr/medications/<id>/discontinue | Discontinue medication | healthcare_emr:medications |
| GET | /api/healthcare/emr/patients/<id>/allergies | Allergy list | healthcare_emr:allergies |
| POST | /api/healthcare/emr/allergies | Record allergy | healthcare_emr:allergies |
| GET | /api/healthcare/emr/patients/<id>/allergy-check/<drug> | Drug allergy check | healthcare_emr:allergies |
| GET | /api/healthcare/emr/patients/<id>/vitals | Vital signs | healthcare_emr:vitals |
| POST | /api/healthcare/emr/vitals | Record vital | healthcare_emr:vitals |
| POST | /api/healthcare/emr/fhir-export | FHIR R4 export | healthcare_emr:fhir |

## Business Rules

| Rule | Condition | Effect |
|------|-----------|--------|
| tenant_context_required | tenant_context_present=False | deny |
| cross_tenant_record_access_denied | cross_tenant_access=True | deny |
| note_type_supported | operation=create_note, note_type_supported=False | deny |
| note_amendment_requires_original | operation=amend_note, original_note_present=False | deny |
| problem_requires_icd10 | operation=add_problem, icd10_code_present=False | deny |
| medication_allergy_check_required | operation=prescribe_medication, allergy_check_performed=False | deny |
| deceased_record_locked | operation=update_chart, patient_deceased=True | deny |
| fhir_export_requires_phi_consent | operation=fhir_export, phi_consent_present=False | deny |
| note_cosignature_required | operation=finalize_note, cosignature_required=True, cosignature_present=False | deny |
| medication_reconciliation_on_admission | operation=admit_patient, med_reconciliation_performed=False | warn |

## Data Models
- ClinicalNoteCreate/Response: note_type, SOAP fields, ICD-10 codes, amendment_of, cosigned_by, finalized_at
- ProblemCreate/Response: icd10_code, description, status, onset_date, resolved_date
- MedicationCreate/Response: drug_name, NDC/RxNorm codes, dose, route, frequency, allergy_check_performed
- AllergyCreate/Response: allergen, allergy_type, severity, reaction, status
- VitalSignCreate/Response: vital_type, value, unit, recorded_at
- EncounterCreate/Response: encounter_type, provider_id, admit_time, discharge_time, icd10_codes

## Streaming Events
- note_created, note_amended, encounter_opened, encounter_closed
- problem_added, problem_resolved
- medication_prescribed, medication_discontinued
- allergy_recorded, vital_recorded, fhir_export_generated

## Edge Cases Handled
- Allergy check must be explicitly flagged true before any medication can be prescribed
- Amending a note creates a new linked document; the original is preserved read-only
- Deceased patient charts are locked at the rule layer — modifications require amendment workflow
- FHIR export requires explicit PHI consent flag; the bundle includes only requested resource types
- Drug allergy check is case-insensitive substring match against allergen field

## Composability Notes
EMR feeds patient history to `healthcare_ana` for cohort membership and outcomes measurement. Medication data flows to `healthcare_pha` for formulary and dispensing validation. Lab orders originate from encounters and feed back as DiagnosticReport FHIR resources. `healthcare_pmt` ADT events trigger encounter creation in EMR.
