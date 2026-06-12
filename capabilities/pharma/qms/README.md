# Quality Management System

## Overview
End-to-end pharmaceutical QMS covering GxP batch records, CAPA management, deviation handling, OOS investigations, controlled document management, audit management, validation lifecycle, quality risk assessment, supplier qualification, and product complaint management. All workflows enforce cGMP/GDP compliance, 21 CFR Part 11 electronic signature requirements, and effectiveness check obligations before closure. Integrates with the APG NATS/Bytewax event mesh for real-time quality signal detection.

## Capability ID
`pharma_qms`

## Provides
- `change_control_workflow`: Impact-assessed change initiation through implementation and effectiveness check
- `capa_management_workflow`: Root-cause-driven CAPA with overdue escalation, effectiveness verification, and ML-assisted recurrence prediction
- `deviation_management_workflow`: GMP deviation capture, investigation, CAPA linkage, and semantic clustering for systemic pattern detection
- `oos_investigation_workflow`: Structured OOS/OOT Phase 1/Phase 2 investigation per FDA OOS Guidance with SLA enforcement
- `document_control_workflow`: Version-controlled SOP/WI management with periodic review enforcement and multi-level e-signature
- `audit_management_workflow`: Internal and supplier audit lifecycle with findings-CAPA linkage
- `validation_lifecycle_workflow`: Protocol-approval-gated validation execution and report sign-off
- `risk_management_workflow`: ICH Q9-aligned FMEA/HACCP risk assessment with mitigation tracking
- `quality_metrics_workflow`: KPI dashboard for open items, overdue counts, and trend analysis
- `supplier_quality_workflow`: Supplier qualification, risk profiling, and automatic requalification triggers
- `batch_release_workflow`: AI-assisted batch risk scoring and release recommendation
- `regulatory_impact_workflow`: ICH Q10 change classification and jurisdiction-specific notification timeline tracking
- `inspection_readiness_workflow`: Continuous 0–100 inspection readiness scoring with gap analysis
- `spc_trend_workflow`: SPC/Nelson-rule trend analysis for proactive process signal detection
- `periodic_review_workflow`: NATS-driven scheduling and escalation for document and validation review cycles

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
| mqeb | NATS event streaming for QMS lifecycle events |

## Configuration
| Key | Description | Default |
|-----|-------------|---------|
| capa.overdue_escalation_days | Days before overdue escalation | 30 |
| documents.periodic_review_months | Document review cycle | 24 |
| deviations.capa_threshold_severity | Severity requiring mandatory CAPA | major |
| oos.phase1_sla_hours | Phase 1 OOS investigation SLA | 120 |
| inspection.readiness_lead_time_days | Review window for readiness score | 60 |
| spc.default_chart_type | Default control chart type | xbar_r |

## API Routes
| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| /pharma-qms/api/v1/change-control | POST | Initiate change | pharma_qms:change_control |
| /pharma-qms/api/v1/change-control/\<id\>/approve | POST | Approve change | pharma_qms:change_control |
| /pharma-qms/api/v1/change-control/\<id\>/regulatory-impact | POST | Classify regulatory impact | pharma_qms:change_control |
| /pharma-qms/api/v1/capa | POST | Create CAPA | pharma_qms:capa |
| /pharma-qms/api/v1/capa/\<id\>/close | POST | Close CAPA | pharma_qms:capa |
| /pharma-qms/api/v1/capa/\<id\>/predict-effectiveness | GET | Predict CAPA effectiveness | pharma_qms:capa |
| /pharma-qms/api/v1/deviations | POST | Raise deviation | pharma_qms:deviations |
| /pharma-qms/api/v1/deviations/cluster | POST | Cluster similar deviations | pharma_qms:deviations |
| /pharma-qms/api/v1/oos | POST | Initiate OOS investigation | pharma_qms:deviations |
| /pharma-qms/api/v1/documents | POST | Create document | pharma_qms:documents |
| /pharma-qms/api/v1/documents/\<id\>/sign-and-approve | POST | E-sign and approve document | pharma_qms:documents |
| /pharma-qms/api/v1/documents/periodic-reviews | GET | List documents due for review | pharma_qms:documents |
| /pharma-qms/api/v1/audits | POST | Create audit | pharma_qms:audits |
| /pharma-qms/api/v1/batches/\<id\>/risk-score | GET | Compute batch risk score | pharma_qms:batch_release |
| /pharma-qms/api/v1/spc/trend-analysis | POST | Run SPC trend analysis | pharma_qms:metrics |
| /pharma-qms/api/v1/inspection-readiness | GET | Get inspection readiness score | pharma_qms:metrics |
| /pharma-qms/api/v1/schedule/periodic-reviews | POST | Schedule periodic reviews | pharma_qms:admin |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| change_impact_assessment_required | Change approved without impact assessment | Deny — complete impact assessment |
| capa_root_cause_required | CAPA closed without root cause | Deny — identify root cause |
| document_approval_required | Document made effective without approval | Deny — obtain approval |
| audit_finding_capa_required | Audit closed with unlinked findings | Deny — raise CAPA for findings |
| critical_deviation_24h_reporting | Critical deviation not reported within 24 h | Deny — expedite report |
| validation_protocol_approval_required | Validation executed without approved protocol | Deny — approve protocol |
| oos_phase_gate | OOS Phase 2 requires documented Phase 1 conclusion | Deny — complete Phase 1 |
| regulatory_notification_deadline | Change notification deadline exceeded | Alert — submit prior approval supplement |

## Data Models
- `ChangeControl`: change_number, change_type, gmp_impact, regulatory_impact, impact_assessment_reference, risk_assessment_reference, effectiveness_check_reference
- `CapaRecord`: capa_number, capa_type, root_cause, root_cause_method, effectiveness_result, overdue
- `QmsDeviation`: deviation_number, deviation_type, severity, gmp_impact, capa_reference, batch_id
- `ControlledDocument`: document_number, document_type, version, status, next_review_date
- `QualityAudit`: audit_number, audit_type, findings_count, capa_references
- `ValidationRecord`: validation_number, validation_type, protocol_reference, revalidation_due
- `RiskAssessment`: assessment_number, risk_level, mitigation_required, residual_risk_level

## Streaming Events (NATS / Bytewax)
All events are published to `apg.pharma.qms.lifecycle` unless otherwise noted.

- `change_initiated`, `change_approved`, `change_implemented`, `change_closed`
- `regulatory_notification_classified`, `regulatory_immediate_action_required`
- `capa_raised`, `capa_closed`, `capa_overdue`, `capa_effectiveness_predicted`
- `deviation_raised`, `deviation_closed`, `recurring_deviation_signal`
- `oos_investigation_initiated`, `oos_phase2_escalated`
- `document_approved`, `document_superseded`, `document_periodic_review_due`
- `audit_completed`, `audit_finding_raised`
- `validation_approved`, `validation_revalidation_required`
- `spc_signal_detected`, `spc_analysis_completed`
- `batch_risk_score_computed`
- `inspection_readiness_scored`
- `review_due`, `revalidation_due` (on `apg.pharma.qms.scheduling`)

## New Async Methods (v1.1)
| Method | Description |
|--------|-------------|
| `initiate_oos_investigation` | Phase-gated OOS/OOT investigation with SLA enforcement |
| `run_spc_trend_analysis` | Nelson/WE rule SPC analysis with Cpk and out-of-control signal detection |
| `classify_regulatory_impact` | ICH Q10 change classification with jurisdiction-specific submission deadlines |
| `generate_inspection_readiness_score` | Weighted 0–100 readiness score from live QMS state |
| `predict_capa_effectiveness` | Ollama-assisted CAPA recurrence risk scoring |
| `compute_batch_risk_score` | Weighted RPN batch quality risk aggregation |
| `schedule_periodic_reviews` | NATS-aware review scheduler for documents and validations |
| `cluster_similar_deviations` | Semantic deviation clustering for systemic pattern detection |

## Edge Cases Handled
- CAPA effectiveness check must be affirmative before status can be set to `closed_effective`
- Audit findings with no CAPA references block audit closure regardless of `findings_count`
- Documents with open periodic review requests cannot be superseded without completing the review
- Critical deviations trigger a 24-hour reporting clock independent of severity reassessment
- Change control with regulatory impact requires separate regulatory notification workflow
- OOS investigations enforce Phase 1 SLA; breach events escalate to QP via NATS
- SPC signals with out-of-control points auto-recommend preventive CAPA creation
- Deviation clusters of 3+ similar records trigger systemic CAPA drafts

## Composability Notes
Change control integrates with `pharma_reg` for variations requiring regulatory submission. Deviations from `pharma_mfg` feed into QMS CAPA workflow. Audit findings from `pharma_rec` inspections link to QMS CAPA. Validation records from `pharma_mfg` equipment qualification are referenced here. Batch risk scores integrate with `pharma_lims` for analytical data and `pharma_mfg` for equipment qualification status. Inspection readiness scores feed `pharma_mgmt` management review dashboards.

---

## World-Class Enhancements (v2.0)

Fifteen targeted improvements over baseline implementation:

- **I1. AI-Powered Batch Release Decision Engine** [Intelligent Automation]
- **I2. Real-Time OOS/OOT Investigation Workflow** [Regulatory Compliance]
- **I3. Intelligent Trend Analysis and SPC Integration** [Proactive Quality]
- **I4. Electronic Batch Record (EBR) Integration and Release** [GxP Data Integrity]
- **I5. Configurable Multi-Level Electronic Signature Workflows** [Regulatory Compliance (21 CFR Part 11 / Annex 11)]
- **I6. Regulatory Submission Change Impact Tracker** [Regulatory Affairs Integration]
- **I7. NATS-Driven QMS Event Mesh with Bytewax Stream Processing** [Architecture / Observability]
- **I8. Automated GMP Training Record Verification** [Personnel Compliance]
- **I9. Intelligent CAPA Effectiveness Prediction** [AI / Predictive Quality]
- **I10. Batch-Level Quality Risk Scoring Dashboard** [Analytics]
- **I11. Cross-Capability Deviation Clustering and Pattern Detection** [Intelligence / Analytics]
- **I12. Supplier Quality Risk Profiling with Automatic Requalification Triggers** [Supply Chain Quality]
- **I13. Regulatory Inspection Readiness Scorecard** [Audit Readiness]
- **I14. Version-Controlled Change Package with Linked Evidence** [Data Integrity / Traceability]
- **I15. Automated Periodic Review and Revalidation Scheduling via NATS** [Lifecycle Management]

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
