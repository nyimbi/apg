# Regulatory Compliance

## Overview
Manages pharmaceutical regulatory compliance obligations across multiple frameworks (FDA, EMA, GMP, ICH), including gap assessments, inspection readiness, label change management, post-market surveillance, regulatory intelligence dissemination, and regulatory commitment tracking. Enforces inspection response timelines, label QP approval, and overdue commitment escalation.

## Capability ID
`pharma_rec`

## Provides
- regulatory_compliance_monitoring_workflow: Framework gap assessment and implementation tracking
- inspection_readiness_workflow: Inspection planning, preparation checklist, and response management
- label_management_workflow: Version-controlled label changes with QP approval and market adaptation
- post_market_surveillance_workflow: PMS protocol management and report submission
- regulatory_intelligence_workflow: Guidance document capture, impact assessment, and dissemination
- commitment_tracking_workflow: Milestone-tracked regulatory commitment fulfillment with overdue escalation
- compliance_gap_assessment_workflow: Structured gap assessment with critical/major/minor classification
- inspection_response_workflow: Warning letter and OAI response with deadline enforcement
- regulatory_change_impact_workflow: Change impact assessment and product-level gap linkage
- compliance_audit_workflow: Full compliance audit trail and evidence management

## Requires
| Capability | Reason |
|------------|--------|
| auth | Role-based access for regulatory affairs and compliance |
| audl | Compliance audit trail |
| mten | Company-level regulatory data isolation |
| conf | Framework and deadline configuration |
| ntfy | Inspection notifications and commitment overdue alerts |
| wflo | Label approval and commitment workflow |
| comp | Regulatory framework compliance enforcement |
| nlpc | Guidance document analysis and impact assessment |
| mqeb | Event streaming for inspection and commitment events |

## Configuration
| Key | Description | Default |
|-----|-------------|---------|
| audits_inspections.response_timeline_days.warning_letter | Warning letter response deadline | 30 |
| commitments.overdue_escalation_days | Days before overdue escalation | 14 |
| compliance_frameworks.periodic_review_months | Framework review cycle | 12 |

## API Routes
| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| /pharma-rec/api/v1/compliance | POST | Register compliance framework | pharma_rec:compliance |
| /pharma-rec/api/v1/inspections | POST | Record inspection | pharma_rec:inspections |
| /pharma-rec/api/v1/inspections/<id>/outcome | POST | Record inspection outcome | pharma_rec:inspections |
| /pharma-rec/api/v1/labeling | POST | Create label record | pharma_rec:labeling |
| /pharma-rec/api/v1/labeling/<id>/approve | POST | QP approve label | pharma_rec:labeling |
| /pharma-rec/api/v1/commitments | POST | Create regulatory commitment | pharma_rec:commitments |
| /pharma-rec/api/v1/commitments/overdue | GET | List overdue commitments | pharma_rec:commitments |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| warning_letter_30d_response | Warning letter response not filed within 30 days | Deny — expedite response |
| inspection_capa_required | Inspection closed with unlinked findings | Deny — raise CAPA |
| label_qp_approval_required | Label made effective without QP approval | Deny — obtain QP approval |
| commitment_overdue_escalation | Overdue commitment not escalated | Deny — escalate commitment |
| regulatory_intel_impact_assessment | Intel recorded without impact assessment | Deny — complete impact assessment |
| inspection_readiness_required | Readiness confirmed without assessment | Deny — complete readiness assessment |

## Data Models
- ComplianceFrameworkRecord: framework, applicable_sites, gap_assessment_reference, implementation_plan_reference
- InspectionRecord: inspection_number, inspection_type, authority, outcome, response_deadline
- LabelRecord: label_number, product_id, market, language, version, change_type, qp_approved
- PostMarketSurveillanceRecord: pms_number, pms_type, protocol_reference, status, signals_identified
- RegulatoryIntelligenceRecord: intel_type, region, title, impact_assessed, products_affected
- RegulatoryCommitment: commitment_number, authority, milestones, due_date, overdue
- GapAssessment: framework, site, critical_gaps, major_gaps, minor_gaps, implementation_plan_reference

## Streaming Events
- compliance_gap_identified, inspection_announced, inspection_completed
- warning_letter_received, inspection_response_submitted
- label_change_approved, label_updated
- pms_report_submitted, commitment_fulfilled, commitment_overdue
- regulatory_change_detected, impact_assessment_required

## Edge Cases Handled
- Warning letter response deadline is 30 calendar days from receipt date, not from inspection completion
- Official Action Indicated triggers a 15-day response deadline, shorter than warning letters
- Label QP approval must be separate from artwork approval; both are required before label becomes effective
- Commitment overdue escalation fires at 14 days past due, not at the due date itself
- Gap assessments must capture separate counts for critical/major/minor gaps for risk stratification

## Composability Notes
Feeds inspection findings to `pharma_qms` CAPA. Label changes trigger variation filings in `pharma_reg`. PMS reports integrate with `pharma_pvi` signal detection. Regulatory intelligence impacts product registration strategy in `pharma_reg`.
