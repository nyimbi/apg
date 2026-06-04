# Clinical Trials Management

## Overview
Manages the complete clinical trial lifecycle from protocol development through site initiation, patient enrolment, randomisation, adverse event reporting, data management, and regulatory submissions. Enforces GCP compliance, informed consent requirements, IRB approvals, and ICH E6 expedited reporting timelines at every boundary.

## Capability ID
`pharma_ctr`

## Provides
- trial_protocol_workflow: Version-controlled protocol management with IRB review tracking
- site_selection_workflow: Site qualification, initiation, and monitoring visit lifecycle
- patient_randomisation_workflow: Multi-method randomisation with IVRS integration and stratification
- adverse_event_workflow: MedDRA-coded AE capture with CTCAE grading and expedited reporting
- clinical_data_management_workflow: EDC-integrated query management and data lock coordination
- regulatory_submission_workflow: IND/CTA/MAA filing with authority tracking
- informed_consent_workflow: IC version tracking and patient status management
- monitoring_visit_workflow: On-site and remote monitoring visit documentation
- safety_reporting_workflow: SADIE 24h and SUSAR 15-day expedited reporting
- trial_closure_workflow: Site closure and final clinical study report preparation

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

## API Routes
| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| /pharma-ctr/api/v1/trials | POST | Create trial | pharma_ctr:trials |
| /pharma-ctr/api/v1/trials/<id>/activate | POST | Activate after IRB | pharma_ctr:trials |
| /pharma-ctr/api/v1/patients/enrol | POST | Enrol patient | pharma_ctr:patients |
| /pharma-ctr/api/v1/patients/<id>/randomise | POST | Randomise patient | pharma_ctr:randomisation |
| /pharma-ctr/api/v1/adverse-events | POST | Report AE | pharma_ctr:ae |
| /pharma-ctr/api/v1/submissions | POST | File regulatory submission | pharma_ctr:submissions |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| trial_irb_approval_required | Trial activated without IRB approval | Deny — obtain IRB approval |
| patient_ic_required | Patient enrolled without informed consent | Deny — obtain informed consent |
| site_qualification_required | Site initiated without qualification visit | Deny — complete qualification |
| sadie_24h_reporting | Serious AE not reported within 24h | Deny — expedite report |
| susar_15d_reporting | SUSAR not reported within 15 days | Deny — expedite SUSAR |
| data_lock_requires_query_resolution | Data locked with open queries | Deny — resolve queries |

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

## Edge Cases Handled
- Site must be in initiated status before any patient can be enrolled
- Protocol IRB approval must pre-date site initiation to be valid
- SUSAR reports trigger both expedited reporting and signal detection simultaneously
- Randomisation code generation is audit-logged independently of patient status updates
- Screen failure patients remain in database with reason for future screening rate analysis

## Composability Notes
Safety data flows from `pharma_ctr` to `pharma_pvi` for post-market safety surveillance. Protocol amendments link to `pharma_qms` change control. Regulatory submissions feed `pharma_reg` approval tracking.
