# Pharmacovigilance

## Overview
Manages the complete pharmacovigilance lifecycle from adverse event intake through ICSR submission, signal detection, PSUR/PBRER generation, and regulatory database reporting. Enforces ICH E2B(R3) formatting, 7-day/15-day expedited reporting timelines, MedDRA coding, duplicate detection, and benefit-risk assessment requirements.

## Capability ID
`pharma_pvi`

## Provides
- adverse_event_collection_workflow: Multi-source AE intake with source classification
- case_processing_workflow: MedDRA coding, narrative writing, causality assessment, duplicate check
- signal_detection_workflow: Disproportionality analysis, literature signal capture, clinical review
- psur_generation_workflow: PSUR/PBRER compilation with IBRD reference and benefit-risk assessment
- regulatory_reporting_workflow: E2B(R3) ICSR submission to EudraVigilance, FDA FAERS, WHO VigiBase
- literature_screening_workflow: Periodic database screening with relevance assessment
- benefit_risk_assessment_workflow: Signal-triggered benefit-risk evaluation
- follow_up_management_workflow: Structured follow-up request and response tracking
- duplicate_detection_workflow: Case deduplication with master case linkage
- meddra_coding_workflow: Mandatory MedDRA PT/SOC coding with hierarchy traversal

## Requires
| Capability | Reason |
|------------|--------|
| auth | Identity and access control |
| audl | Audit trail for regulatory submissions |
| mten | Multi-tenant data isolation |
| conf | Configuration for reporting timelines |
| ntfy | Notifications for timeline breaches |
| wflo | Case processing and PSUR approval workflow |
| comp | Regulatory compliance enforcement |
| nlpc | Narrative generation assistance and literature screening |
| mqeb | Event streaming for case and signal events |

## Configuration
| Key | Description | Default |
|-----|-------------|---------|
| case_processing.reporting_timelines.7day_expedited | SUSAR reporting deadline (days) | 7 |
| case_processing.reporting_timelines.15day_expedited | Serious AE expedited deadline (days) | 15 |
| literature.screening_frequency_days | Literature database screening interval | 7 |
| psur.submission_timeline_days | PSUR submission deadline after DLP | 60 |

## API Routes
| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| /pharma-pvi/api/v1/cases | POST | Create AE case | pharma_pvi:cases |
| /pharma-pvi/api/v1/cases/<id>/process | POST | Process case with MedDRA | pharma_pvi:cases |
| /pharma-pvi/api/v1/submissions | POST | Submit ICSR | pharma_pvi:submissions |
| /pharma-pvi/api/v1/signals | POST | Create safety signal | pharma_pvi:signals |
| /pharma-pvi/api/v1/psur | POST | Create PSUR report | pharma_pvi:psur |
| /pharma-pvi/api/v1/follow-ups | POST | Request follow-up | pharma_pvi:follow_up |
| /pharma-pvi/api/v1/literature | GET | List literature records | pharma_pvi:literature |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| meddra_coding_required | Case processed without MedDRA coding | Deny — apply MedDRA coding |
| 7day_expedited_reporting | SUSAR not submitted within 7 days | Deny — expedite submission |
| 15day_expedited_reporting | Serious case not submitted within 15 days | Deny — expedite submission |
| medical_review_required | Serious case closed without medical review | Deny — obtain medical review |
| psur_benefit_risk_required | PSUR submitted without benefit-risk assessment | Deny — complete assessment |
| e2b_r3_format_required | ICSR not in E2B(R3) format | Deny — format as E2B(R3) |

## Data Models
- AdvEventCase: case_number, source, case_type, product_id, serious, seriousness_criteria, meddra_pt, meddra_soc, narrative, causality
- IcsrSubmission: case_id, regulatory_database, e2b_r3_message_id, due_date, follow_up_number
- SafetySignal: signal_number, product_id, signal_type, meddra_pt, detection_method, strength_of_evidence
- PsurReport: report_number, report_type, data_lock_point, international_birth_date, ibrd_reference, benefit_risk_assessed
- LiteratureRecord: database_source, article_reference, relevant, product_id, case_created
- FollowUpRequest: case_id, follow_up_type, requested_from, due_date, status

## Streaming Events
- ae_received, case_created, case_processed, case_closed
- duplicate_detected, follow_up_requested, follow_up_received
- signal_detected, signal_evaluated, signal_closed
- literature_match_found, psur_submitted, icsr_submitted
- 7day_report_filed, 15day_report_filed

## Edge Cases Handled
- Duplicate detection must be performed before case can be processed
- SUSAR cases trigger both 7-day and 15-day timeline enforcement simultaneously
- Literature records marked as relevant must be linked to a product before case creation
- PSUR benefit-risk assessment must be affirmative before submission regardless of DLP deadline
- Closed signals require clinical review even when disproportionality signals are refuted

## Composability Notes
Receives adverse event signals from `pharma_ctr` for clinical trial SAEs. Feeds signal data to `pharma_rec` for post-market surveillance reports. PSUR submissions link to `pharma_reg` product registration lifecycle. Integrates with `nlpc` for automated narrative drafting assistance.
