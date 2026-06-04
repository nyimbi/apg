# Quality Management System

## Overview
End-to-end pharmaceutical QMS covering change control, CAPA management, deviation handling, controlled document management, audit management, validation lifecycle, and risk assessment. All workflows enforce GMP compliance, electronic signature requirements, and effectiveness check obligations before closure.

## Capability ID
`pharma_qms`

## Provides
- change_control_workflow: Impact-assessed change initiation through implementation and effectiveness check
- capa_management_workflow: Root-cause-driven CAPA with overdue escalation and effectiveness verification
- deviation_management_workflow: GMP deviation capture, investigation, and CAPA linkage
- document_control_workflow: Version-controlled SOP/WI management with periodic review enforcement
- audit_management_workflow: Internal and supplier audit lifecycle with findings-CAPA linkage
- validation_lifecycle_workflow: Protocol-approval-gated validation execution and report sign-off
- risk_management_workflow: ICH Q9-aligned risk assessment with mitigation tracking
- quality_metrics_workflow: KPI dashboard for open items, overdue counts, and trend analysis
- supplier_quality_workflow: Supplier deviation and CAPA linkage
- qms_review_workflow: Management review cycle coordination

## Requires
| Capability | Reason |
|------------|--------|
| auth | Role-based access for QP, QA, and management |
| audl | 21 CFR Part 11 compliant audit trail |
| mten | Tenant-level QMS isolation |
| conf | Configurable review cycles and thresholds |
| ntfy | Overdue CAPA and document review notifications |
| wflo | Multi-level approval workflow |
| comp | GMP compliance enforcement |
| schd | Periodic review and audit scheduling |
| mqeb | Event streaming for QMS lifecycle events |

## Configuration
| Key | Description | Default |
|-----|-------------|---------|
| capa.overdue_escalation_days | Days before overdue escalation | 30 |
| documents.periodic_review_months | Document review cycle | 24 |
| deviations.capa_threshold_severity | Severity requiring mandatory CAPA | major |

## API Routes
| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| /pharma-qms/api/v1/change-control | POST | Initiate change | pharma_qms:change_control |
| /pharma-qms/api/v1/change-control/<id>/approve | POST | Approve change | pharma_qms:change_control |
| /pharma-qms/api/v1/capa | POST | Create CAPA | pharma_qms:capa |
| /pharma-qms/api/v1/capa/<id>/close | POST | Close CAPA | pharma_qms:capa |
| /pharma-qms/api/v1/deviations | POST | Raise deviation | pharma_qms:deviations |
| /pharma-qms/api/v1/documents | POST | Create document | pharma_qms:documents |
| /pharma-qms/api/v1/audits | POST | Create audit | pharma_qms:audits |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| change_impact_assessment_required | Change approved without impact assessment | Deny — complete impact assessment |
| capa_root_cause_required | CAPA closed without root cause | Deny — identify root cause |
| document_approval_required | Document made effective without approval | Deny — obtain approval |
| audit_finding_capa_required | Audit closed with unlinked findings | Deny — raise CAPA for findings |
| critical_deviation_24h_reporting | Critical deviation not reported within 24h | Deny — expedite report |
| validation_protocol_approval_required | Validation executed without approved protocol | Deny — approve protocol |

## Data Models
- ChangeControl: change_number, change_type, gmp_impact, impact_assessment_reference, risk_assessment_reference, effectiveness_check_reference
- CapaRecord: capa_number, capa_type, root_cause, root_cause_method, effectiveness_result, overdue
- QmsDeviation: deviation_number, deviation_type, severity, gmp_impact, capa_reference
- ControlledDocument: document_number, document_type, version, status, next_review_date
- QualityAudit: audit_number, audit_type, findings_count, capa_references
- ValidationRecord: validation_number, validation_type, protocol_reference, revalidation_due
- RiskAssessment: assessment_number, risk_level, mitigation_required, residual_risk_level

## Streaming Events
- change_initiated, change_approved, change_implemented
- capa_raised, capa_closed, capa_overdue
- deviation_raised, deviation_closed
- document_approved, document_superseded, document_periodic_review_due
- audit_completed, audit_finding_raised
- validation_approved, validation_revalidation_required

## Edge Cases Handled
- CAPA effectiveness check must be affirmative before status can be set to closed_effective
- Audit findings with no CAPA references block audit closure regardless of findings_count
- Documents with open periodic review requests cannot be superseded without completing the review
- Critical deviations trigger a 24-hour reporting clock independent of severity reassessment
- Change control with regulatory impact requires separate regulatory notification workflow

## Composability Notes
Change control integrates with `pharma_reg` for variations requiring regulatory submission. Deviations from `pharma_mfg` feed into QMS CAPA workflow. Audit findings from `pharma_rec` inspections link to QMS CAPA. Validation records from `pharma_mfg` equipment qualification are referenced here.
