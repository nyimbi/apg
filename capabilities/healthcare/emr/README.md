# Electronic Medical Records

## Overview

Full-featured FHIR R4 EMR capability providing the complete clinical record lifecycle: patient registration with probabilistic deduplication, SOAP and structured clinical note authoring, problem list management with ICD-10 coding, medication safety (DDI, allergy, paediatric dosing, pregnancy, renal adjustment, duplicate therapy, controlled substances), vital signs recording with trend analysis, lab orders and critical result management, imaging orders, care plans, immunisations, family history, referrals, discharge summaries, advance directives, and HL7 FHIR R4 bundle export. Designed for HIPAA compliance with cross-tenant PHI isolation enforced at the rule layer.

## Capability ID
`healthcare_emr`

## Provides

| Feature | Description |
|---------|-------------|
| patient_registration | Full patient CRUD with probabilistic dedup, merge, and soft-delete |
| clinical_note_authoring | SOAP, progress, discharge, operative, addendum, and 7 other note types with co-signature, amendment, and sign workflows |
| problem_list_management | ICD-10-coded active/chronic/resolved problem tracking |
| medication_safety | DDI, allergy conflict, duplicate therapy, paediatric dosing, pregnancy safety, renal adjustment, controlled substance checks |
| prescribing_workflow | Full Rx lifecycle: create → verify → dispense → refill → reconcile → stop |
| vital_signs_recording | 9 vital types with trend analysis and anomaly detection |
| vital_trend_analysis | Rolling-window trend analysis with slope-based deterioration alerts |
| fhir_r4_export | FHIR R4 Bundle export (Patient, Encounter, Condition, MedicationRequest, AllergyIntolerance, Observation, DocumentReference, Consent, MolecularSequence) |
| clinical_decision_support | CHA₂DS₂-VASc, Wells PE, qSOFA, NEWS2, CCI, LACE+, Sepsis Bundle tracking |
| icd10_coding | ICD-10 assignment on problems, encounters, notes; CPT procedure coding |
| encounter_management | Encounter lifecycle: open → admit → transfer → discharge |
| lab_management | Lab order + result lifecycle with critical value flagging and eMAR |
| imaging_management | Imaging orders with radiology report attachment |
| care_plans | Care plan lifecycle with team and activity tracking |
| immunisations | Vaccine administration recording |
| family_history | Family history with condition tracking |
| consent_management | Treatment, research, emergency override, and minor consent |
| advance_directives | Living will, DNR/DNAR, POLST, MOLST, healthcare proxy recording |
| referral_management | Outbound referral create/accept/cancel |
| discharge_workflow | Discharge summary generation with medication reconciliation |
| population_health | Cohort builder with multi-criteria filtering |
| disease_trajectory | Longitudinal biomarker trend tracking per condition |
| pharmacogenomics | PGx result storage with CPIC-grade prescribing recommendations |
| emar | Electronic Medication Administration Record with dose-level tracking |
| hl7_processing | HL7 v2 ADT/ORM/ORU message parsing and acknowledgement |
| readmission_risk | LACE+ 30-day unplanned readmission risk scoring |
| comorbidity_index | Charlson Comorbidity Index with 10-year survival estimate |

## Requires

| Cap ID | Purpose |
|--------|---------|
| auth | PHI access authorization |
| audt | Audit trail for all chart modifications |
| mten | Multi-tenant isolation |
| conf | Tenant-specific configuration |
| ntfy | Allergy conflict and critical value notifications |
| nlpc | Clinical note NLP-assisted coding suggestions |
| wflo | Note co-signature and amendment approval workflows |
| nats | Real-time event streaming for lab results, alerts, population health |

## Configuration

| Key | Description | Default |
|-----|-------------|---------|
| notes.co_signature_required | Require co-signature for note finalization | false |
| notes.addendum_allowed | Allow addendum notes after finalization | true |
| medications.reconciliation_on_admission | Warn if reconciliation skipped on admission | true |
| fhir.version | FHIR version | R4 |
| governance.medication_allergy_check_required | Block prescription if allergy check not performed | true |
| sepsis.bundle_alert_minutes | Minutes after qSOFA≥2 before sepsis bundle alert fires | 45 |
| cds.news2_alert_threshold | NEWS2 score threshold for urgent escalation alert | 7 |
| pgx.cpic_minimum_level | Minimum CPIC evidence level to surface PGx alert | B |

## API Routes

| Method | Path | Description | Permission |
|--------|------|-------------|------------|
| GET | /api/healthcare/emr/patients | List patients | healthcare_emr:patients |
| POST | /api/healthcare/emr/patients | Register patient | healthcare_emr:patients_write |
| GET | /api/healthcare/emr/patients/\<id\> | Get patient | healthcare_emr:patients |
| PUT | /api/healthcare/emr/patients/\<id\> | Update patient | healthcare_emr:patients_write |
| POST | /api/healthcare/emr/patients/\<id\>/merge | Merge duplicate patient | healthcare_emr:patients_admin |
| GET | /api/healthcare/emr/encounters | List encounters | healthcare_emr:encounters |
| POST | /api/healthcare/emr/encounters | Create encounter | healthcare_emr:encounters |
| GET | /api/healthcare/emr/encounters/\<id\> | Encounter detail | healthcare_emr:encounters |
| POST | /api/healthcare/emr/encounters/\<id\>/close | Close encounter | healthcare_emr:encounters |
| POST | /api/healthcare/emr/encounters/\<id\>/admit | Admit patient | healthcare_emr:encounters |
| POST | /api/healthcare/emr/encounters/\<id\>/discharge | Discharge patient | healthcare_emr:encounters |
| POST | /api/healthcare/emr/encounters/\<id\>/transfer | Transfer patient | healthcare_emr:encounters |
| GET | /api/healthcare/emr/notes | List notes | healthcare_emr:notes |
| POST | /api/healthcare/emr/notes | Create note | healthcare_emr:notes_write |
| GET | /api/healthcare/emr/notes/\<id\> | Note detail | healthcare_emr:notes |
| POST | /api/healthcare/emr/notes/\<id\>/amend | Amend note | healthcare_emr:notes_write |
| POST | /api/healthcare/emr/notes/\<id\>/finalize | Finalize note | healthcare_emr:notes_write |
| POST | /api/healthcare/emr/notes/\<id\>/sign | Sign note | healthcare_emr:notes_write |
| POST | /api/healthcare/emr/notes/\<id\>/addendum | Add addendum | healthcare_emr:notes_write |
| GET | /api/healthcare/emr/patients/\<id\>/problems | Problem list | healthcare_emr:problems |
| POST | /api/healthcare/emr/problems | Add problem | healthcare_emr:problems |
| POST | /api/healthcare/emr/problems/\<id\>/resolve | Resolve problem | healthcare_emr:problems |
| GET | /api/healthcare/emr/patients/\<id\>/medications | Medication list | healthcare_emr:medications |
| POST | /api/healthcare/emr/medications | Prescribe medication | healthcare_emr:medications |
| POST | /api/healthcare/emr/medications/\<id\>/discontinue | Discontinue medication | healthcare_emr:medications |
| POST | /api/healthcare/emr/medications/reconcile | Medication reconciliation | healthcare_emr:medications |
| GET | /api/healthcare/emr/patients/\<id\>/allergies | Allergy list | healthcare_emr:allergies |
| POST | /api/healthcare/emr/allergies | Record allergy | healthcare_emr:allergies |
| GET | /api/healthcare/emr/patients/\<id\>/allergy-check/\<drug\> | Drug allergy check | healthcare_emr:allergies |
| GET | /api/healthcare/emr/patients/\<id\>/vitals | Vital signs | healthcare_emr:vitals |
| POST | /api/healthcare/emr/vitals | Record vital | healthcare_emr:vitals |
| GET | /api/healthcare/emr/patients/\<id\>/vitals/\<type\>/trend | Vital trend analysis | healthcare_emr:vitals |
| GET | /api/healthcare/emr/patients/\<id\>/cds | Clinical decision support | healthcare_emr:cds |
| GET | /api/healthcare/emr/patients/\<id\>/cds/chads2vasc | CHA₂DS₂-VASc score | healthcare_emr:cds |
| GET | /api/healthcare/emr/patients/\<id\>/cds/news2 | NEWS2 score | healthcare_emr:cds |
| GET | /api/healthcare/emr/patients/\<id\>/cds/qsofa | qSOFA score | healthcare_emr:cds |
| GET | /api/healthcare/emr/patients/\<id\>/cds/cci | Charlson Comorbidity Index | healthcare_emr:cds |
| GET | /api/healthcare/emr/patients/\<id\>/cds/lace-plus | LACE+ readmission risk | healthcare_emr:cds |
| POST | /api/healthcare/emr/encounters/\<id\>/sepsis-bundle | Track sepsis bundle | healthcare_emr:cds |
| GET | /api/healthcare/emr/patients/\<id\>/lab-orders | Lab orders | healthcare_emr:labs |
| POST | /api/healthcare/emr/lab-orders | Create lab order | healthcare_emr:labs |
| POST | /api/healthcare/emr/lab-results | Receive lab result | healthcare_emr:labs |
| POST | /api/healthcare/emr/lab-results/\<id\>/notify-critical | Notify critical lab | healthcare_emr:labs |
| POST | /api/healthcare/emr/emar/administer | Record dose administration | healthcare_emr:medications |
| GET | /api/healthcare/emr/encounters/\<id\>/emar | eMAR report | healthcare_emr:medications |
| GET | /api/healthcare/emr/patients/\<id\>/imaging-orders | Imaging orders | healthcare_emr:imaging |
| POST | /api/healthcare/emr/imaging-orders | Create imaging order | healthcare_emr:imaging |
| POST | /api/healthcare/emr/imaging-orders/\<id\>/report | Attach radiology report | healthcare_emr:imaging |
| GET | /api/healthcare/emr/patients/\<id\>/care-plans | Care plans | healthcare_emr:care_plans |
| POST | /api/healthcare/emr/care-plans | Create care plan | healthcare_emr:care_plans |
| POST | /api/healthcare/emr/patients/\<id\>/advance-directives | Record advance directive | healthcare_emr:consent |
| GET | /api/healthcare/emr/patients/\<id\>/advance-directives | Get active directives | healthcare_emr:consent |
| GET | /api/healthcare/emr/patients/\<id\>/trajectory/\<prefix\> | Disease trajectory | healthcare_emr:cds |
| POST | /api/healthcare/emr/patients/\<id\>/pgx | Record PGx result | healthcare_emr:medications |
| GET | /api/healthcare/emr/patients/\<id\>/pgx/check/\<drug\> | PGx prescribing check | healthcare_emr:medications |
| POST | /api/healthcare/emr/cohort | Build population health cohort | healthcare_emr:population |
| POST | /api/healthcare/emr/fhir-export | FHIR R4 bundle export | healthcare_emr:fhir |

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
| pgx_contraindication_hard_stop | pgx_check=contraindicated | raise DrugSafetyError |
| advance_directive_dnr_visible | encounter_created, active_dnr_present | surface in CDS alerts |
| final_note_immutable | operation=update_note, note_status=final | deny → use addendum |

## Data Models

- `PatientCreate/Response`: name (HumanName), birth_date, gender, address, telecom, identifiers, biometric_hash, next_of_kin, blood_type
- `ClinicalNoteCreate/Response`: note_type, SOAP fields, ICD-10 codes, amendment_of, cosigned_by, finalized_at
- `ProblemCreate/Response`: icd10_code, description, status (active/chronic/resolved/inactive), onset_date, resolved_date
- `MedicationCreate/Response`: drug_name, NDC/RxNorm codes, dose, route, frequency, allergy_check_performed
- `AllergyCreate/Response`: allergen, allergy_type, severity (mild/moderate/severe/life_threatening), reaction, status
- `VitalSignCreate/Response`: vital_type, value, unit, recorded_at, method, position
- `EncounterCreate/Response`: encounter_type, provider_id, admit_time, discharge_time, icd10_codes, discharge_summary_id
- `LabOrderCreate/Response`: test_code, test_name, specimen_type, priority, clinical_indication
- `LabResultCreate/Response`: value, value_numeric, unit, reference_range, flag, critical_notified
- `ImagingOrderCreate/Response`: modality, body_part, laterality, cpt_code, contrast_required, accession_number
- `CarePlanCreate/Response`: title, goal, activities, icd10_codes, care_team, status
- `ImmunisationCreate/Response`: vaccine_code, dose_quantity, route, site, lot_number, expiration_date
- `PrescriptionCreate/Response`: includes safety_summary (allergy_conflicts, ddi_interactions, duplicate_therapy)

## Streaming Events (NATS subjects)

All events are published to `emr.{tenant_id}.{resource}.{action}`:

- `emr.{tenant}.encounter.opened`, `emr.{tenant}.encounter.closed`
- `emr.{tenant}.note.created`, `emr.{tenant}.note.finalized`, `emr.{tenant}.note.amended`
- `emr.{tenant}.problem.added`, `emr.{tenant}.problem.resolved`
- `emr.{tenant}.medication.prescribed`, `emr.{tenant}.medication.discontinued`
- `emr.{tenant}.allergy.recorded`
- `emr.{tenant}.vital.recorded`
- `emr.{tenant}.labs.critical` — critical lab result requiring immediate notification
- `emr.{tenant}.vitals.trend` — vital trend deterioration alert
- `emr.{tenant}.sepsis.bundle` — sepsis bundle incomplete at 45-minute mark
- `emr.{tenant}.fhir.exported`

Bytewax pipelines consume these streams for real-time dashboards, risk scoring models, and downstream integrations.

## Clinical Decision Support Scores

| Score | Method | Clinical Use |
|-------|--------|-------------|
| CHA₂DS₂-VASc | `CHADS2_VASc_score()` | AF anticoagulation decision |
| Wells PE | `WELLS_score_PE()` | PE pre-test probability |
| qSOFA | `QSOFA_score()` | Sepsis screening |
| NEWS2 | `NEWS2_score()` | General deterioration alerting |
| Charlson CCI | `compute_charlson_comorbidity_index()` | Risk adjustment, survival estimate |
| LACE+ | `compute_lace_plus_score()` | 30-day readmission risk |
| Sepsis Bundle | `track_sepsis_bundle()` | Hour-1 bundle compliance |

## Edge Cases Handled

- Allergy check must be explicitly flagged true before any medication can be prescribed
- Amending a note creates a new linked document; the original is preserved read-only
- Final notes are immutable — corrections must go through the addendum workflow
- Deceased patient charts are locked — modifications require amendment workflow
- FHIR export requires explicit PHI consent flag
- Drug allergy check is case-insensitive substring match covering drug name and drug class
- Prescription safety gate runs allergy + DDI + duplicate therapy + controlled substance checks in sequence; hard stops raise `DrugSafetyError`
- Patient merge re-keys all encounters, notes, problems, medications, and allergies to the surviving record
- Sepsis bundle tracker fires NATS alert if any bundle element is incomplete beyond 45 minutes of qSOFA recognition
- Advance directives surface automatically in CDS alerts and clinical summaries
- PGx contraindications raise `DrugSafetyError` in the prescribing workflow

## Composability Notes

EMR feeds patient history to `healthcare_ana` for cohort membership and outcomes measurement. Medication data flows to `healthcare_pha` for formulary and dispensing validation. Lab orders originate from encounters and feed back as FHIR DiagnosticReport resources. `healthcare_pmt` ADT events trigger encounter creation in EMR. Population health cohorts flow to `healthcare_pop` for outreach campaign management. PGx results are visible in `healthcare_gen` genomics capability.

---

## World-Class Enhancements (v2.0)

Fifteen targeted improvements over baseline implementation:

- **I1. Real-Time Clinical Event Streaming via NATS** [Integration / Streaming]
- **I2. Longitudinal Vital Trend Analysis with Statistical Anomaly Detection** [Clinical Decision Support]
- **I3. SMART on FHIR OAuth2 App Launch Integration** [Interoperability / Security]
- **I4. AI-Assisted Clinical Note Structuring via Local LLM (Ollama)** [AI / Clinical Productivity]
- **I5. Predictive Readmission Risk Scoring (LACE+ Index)** [Population Health / Risk Stratification]
- **I6. Charlson Comorbidity Index (CCI) Calculation** [Clinical Decision Support / Risk Adjustment]
- **I7. Structured Medication Administration Record (eMAR) with Dose Tracking** [Patient Safety / Medication Management]
- **I8. Sepsis Bundle Compliance Tracking (Sepsis-6/Hour-1)** [Quality / Patient Safety]
- **I9. Genomics/Pharmacogenomics Integration Layer** [Precision Medicine]
- **I10. Patient-Generated Health Data (PGHD) Ingestion** [Connected Health / Patient Engagement]
- **I11. Audit Trail Immutability with Cryptographic Hash Chaining** [Compliance / Security]
- **I12. Advance Directive and POLST/MOLST Management** [Patient Safety / End-of-Life Care]
- **I13. Outpatient Population Health Cohort Builder** [Population Health / Analytics]
- **I14. Multi-Factor Diagnostic Confidence Scoring** [Clinical Decision Support / AI]
- **I15. Cross-Encounter Longitudinal Disease Trajectory Tracking** [Chronic Disease Management]

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
