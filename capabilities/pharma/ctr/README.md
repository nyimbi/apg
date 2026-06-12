# Clinical Trials Management

## Overview
Manages the complete clinical trial lifecycle from protocol development through site initiation, patient enrolment, randomisation, adverse event reporting, data management, and regulatory submissions. Enforces GCP compliance, informed consent requirements, IRB approvals, and ICH E6 expedited reporting timelines at every boundary.

## Capability ID
`pharma_ctr`

## Provides
- trial_protocol_workflow: Version-controlled protocol management with IRB review tracking
- site_selection_workflow: Site qualification, initiation, and monitoring visit lifecycle
- patient_randomisation_workflow: Multi-method randomisation including response-adaptive (Thompson sampling) with IVRS integration and stratification
- adverse_event_workflow: MedDRA-coded AE capture with CTCAE grading, ICH E2B(R3) narrative generation, and expedited reporting
- clinical_data_management_workflow: EDC-integrated query management and data lock coordination
- regulatory_submission_workflow: IND/CTA/MAA filing with eCTD module assembly and authority tracking
- informed_consent_workflow: IC version tracking with protocol-amendment re-consent triggers
- monitoring_visit_workflow: SIV/SCOV documentation with GCP checklist and IMP accountability
- safety_reporting_workflow: SADIE 24h and SUSAR 15-day expedited reporting with PRR signal detection
- trial_closure_workflow: Site close-out, database lock, and CSR generation
- inspection_readiness_workflow: GCP inspection readiness scoring per TransCelerate RBM metrics
- supply_chain_workflow: IMP demand forecasting and reorder trigger generation

## Requires
| Capability | Reason |
|------------|--------|
| auth | Role-based access for investigators, monitors, and sponsors |
| audl | 21 CFR Part 11 / ICH E6 audit trail |
| mten | Sponsor-level trial isolation |
| conf | Protocol-specific configuration |
| ntfy | Safety reporting timeline breach notifications |
| wflo | Protocol amendment and submission approval |
| comp | GCP compliance enforcement |
| nlpc | Protocol text analysis and narrative generation |
| mqeb | Safety event streaming |

## Configuration
| Key | Description | Default |
|-----|-------------|---------|
| adverse_events.reporting_timeline_hours.sadie | SADIE reporting deadline | 24 |
| adverse_events.reporting_timeline_hours.susar | SUSAR reporting deadline (days) | 15 |
| patients.ic_required | Informed consent mandatory | true |
| randomisation.ivrs_integration | Use IVRS for randomisation | true |
| inspection_readiness.tmf_minimum_docs | Minimum expected TMF documents | 10 |

## API Routes
| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| /pharma-ctr/api/v1/trials | POST | Create trial | pharma_ctr:trials |
| /pharma-ctr/api/v1/trials/<id>/activate | POST | Activate after IRB | pharma_ctr:trials |
| /pharma-ctr/api/v1/patients/enrol | POST | Enrol patient | pharma_ctr:patients |
| /pharma-ctr/api/v1/patients/<id>/randomise | POST | Randomise patient | pharma_ctr:randomisation |
| /pharma-ctr/api/v1/patients/<id>/adaptive-randomise | POST | Response-adaptive randomisation | pharma_ctr:randomisation |
| /pharma-ctr/api/v1/adverse-events | POST | Report AE | pharma_ctr:ae |
| /pharma-ctr/api/v1/adverse-events/<id>/narrative | POST | Generate SUSAR narrative | pharma_ctr:ae |
| /pharma-ctr/api/v1/submissions | POST | File regulatory submission | pharma_ctr:submissions |
| /pharma-ctr/api/v1/submissions/ectd | POST | Assemble eCTD package | pharma_ctr:submissions |
| /pharma-ctr/api/v1/trials/<id>/signals | GET | Safety signal detection | pharma_ctr:safety |
| /pharma-ctr/api/v1/trials/<id>/inspection-readiness | GET | Inspection readiness score | pharma_ctr:quality |
| /pharma-ctr/api/v1/trials/<id>/imp-forecast | GET | IMP supply forecast | pharma_ctr:supply |
| /pharma-ctr/api/v1/trials/<id>/ssr | POST | Blinded sample size re-estimation | pharma_ctr:statistics |
| /pharma-ctr/api/v1/protocols/amendment-impact | POST | Protocol amendment impact analysis | pharma_ctr:protocols |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| trial_irb_approval_required | Trial activated without IRB approval | Deny — obtain IRB approval |
| patient_ic_required | Patient enrolled without informed consent | Deny — obtain informed consent |
| site_qualification_required | Site initiated without qualification visit | Deny — complete qualification |
| sadie_24h_reporting | Serious AE not reported within 24h | Deny — expedite report |
| susar_15d_reporting | SUSAR not reported within 15 days | Deny — expedite SUSAR |
| data_lock_requires_query_resolution | Data locked with open queries | Deny — resolve queries |
| ectd_module_completeness | Submission missing required eCTD modules | Warn — flag missing modules |
| blinded_ssr_cap | SSR adjusted enrollment capped at 2x original | Enforce — per ICH E9(R1) |
| adaptive_rand_requires_sap | RAR used without SAP pre-specification | Require documented SAP reference |

## Key Service Methods

### Synchronous
- `describe()` / `evaluate()` — contract and rule evaluation
- `create_trial()` / `register_trial()` / `activate_trial()` / `get_trial()` / `list_trials()`
- `create_protocol()` / `approve_protocol()` / `list_protocols()`
- `select_site()` / `initiate_site()` / `list_sites()`
- `site_initiation_visit()` / `site_close_out()`
- `enrol_patient()` / `randomise_patient()` / `randomise_subject()` / `list_patients()`
- `informed_consent_tracking()`
- `collect_crf_data()` / `validate_crf()` / `query_management()`
- `report_ae()` / `report_adverse_event()` / `classify_ae_causality()` / `report_sar()` / `list_adverse_events()`
- `safety_monitoring_committee_report()` / `interim_analysis()`
- `database_lock()` / `protocol_deviation()` / `tmf_document_upload()`
- `generate_clinical_study_report()` / `regulatory_submission()` / `file_submission()` / `list_submissions()`
- `dashboard_summary()`

### Async
- `adaptive_randomisation()` — Thompson sampling response-adaptive randomisation
- `detect_safety_signals()` — PRR-based signal detection across AE data
- `compute_inspection_readiness_score()` — GCP inspection readiness scoring (0–100)
- `protocol_amendment_impact()` — Amendment impact assessment with re-consent triggering
- `generate_susar_narrative()` — ICH E2B(R3) compliant narrative generation (LLM-enhanced)
- `blinded_sample_size_reestimation()` — Cui-Hung-Wang blinded SSR
- `imp_supply_forecast()` — Per-site IMP demand forecasting with reorder triggers
- `ectd_submission_package()` — eCTD module assembly from TMF documents
- `export_records()` — Bulk data export (json/csv)
- `health_check()` — Service health status
- `compliance_report()` — GxP compliance summary
- `bulk_create_records()` — Batch record creation

## Data Models
- ClinicalTrial: trial_number, phase, trial_type, sponsor_id, blinding, indication
- TrialProtocol: version, status, irb_approval_reference, amendment_reason
- TrialSite: site_number, country, principal_investigator_id, target_enrollment, status
- TrialPatient: patient_code, status, informed_consent_date, randomisation_code, treatment_arm
- AdverseEvent: ae_type, severity_grade, meddra_pt, meddra_soc, causality, narrative
- RegulatorySubmission: submission_type, authority, cover_letter_reference, dossier_reference
- RandomisationRecord: randomisation_method, treatment_arm, stratification_factors

## Streaming Events
- trial_created, protocol_approved, site_initiated, patient_enrolled
- patient_randomised, adverse_event_reported, susar_reported
- data_query_raised, data_locked, submission_filed
- monitoring_visit_completed, trial_closed
- safety_signal_detected, inspection_readiness_computed
- ectd_package_assembled, imp_reorder_triggered

## Edge Cases Handled
- Site must be in initiated status before any patient can be enrolled
- Protocol IRB approval must pre-date site initiation to be valid
- SUSAR reports trigger both expedited reporting and signal detection simultaneously
- Randomisation code generation is audit-logged independently of patient status updates
- Screen failure patients remain in database with reason for future screening rate analysis
- Blinded SSR capped at 2× original enrollment to maintain study feasibility
- eCTD package assembly flags missing modules as gaps rather than hard-blocking submission

## Composability Notes
Safety data flows from `pharma_ctr` to `pharma_pvi` for post-market safety surveillance. Protocol amendments link to `pharma_qms` change control. Regulatory submissions feed `pharma_reg` approval tracking. IMP forecasts integrate with `pharma_mfg` supply chain scheduling.

---

## World-Class Enhancements (v2.0)

- **I1.** Clinical Trials Management — World-Class Improvements
- **I2.** Adaptive Trial Design Engine
- **I3.** eCTD-Structured Regulatory Submission Assembly
- **I4.** Real-Time Safety Signal Detection via BCPNN
- **I5.** CTMS ↔ EDC Bidirectional Sync
- **I6.** AI-Assisted Protocol Deviation Triage
- **I7.** Stratified Patient Matching for Screen Failure Analysis
- **I8.** TMF Completeness Automation via Document Intelligence
- **I9.** Site Performance Predictive Scoring
- **I10.** Automated SUSAR Narratives via NLP
- **I11.** Protocol Amendment Impact Analysis
- **I12.** Continuous Audit Trail Streaming to SIEM
- **I13.** Blinded Sample Size Re-estimation
- **I14.** Supply Chain IMP Forecasting
- **I15.** GCP Inspection Readiness Scoring

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
