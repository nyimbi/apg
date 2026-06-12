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

---

## World-Class Enhancements (v2.0)

**I1. Electronic Signature Enforcement** — HMAC-bound e-signature records (signer, timestamp, IP, reason) replacing boolean QP approval flags; rejects incomplete or expired credentials [21 CFR Part 11 / EU Annex 11]

**I2. Document Version Graph with Immutable Audit Spine** — Append-only DAG in PostgreSQL via recursive CTE; content-hashed nodes, mutations produce new nodes never overwrite, satisfies 21 CFR §11.10(e) [Audit Trail]

**I3. Training Records Module with Competency Matrix** — `TrainingRecord` + `TrainingCurriculumService` maps roles to required SOPs, tracks completions/scores, blocks regulated activities until training is current [GxP Compliance]

**I4. CAPA Bidirectional Linkage** — `raise_capa()` calls `pharma_qms` contract, stores returned CAPA ID on the finding, and tracks closure status for complete inspection remediation state [Quality Integration]

**I5. Structured Inspection Readiness Checklist Engine** — `ReadinessChecklist` model with per-type templates (GMP/GCP/GDP/ISO 13485), owner/due-date/evidence tracking, gates `record_inspection_outcome()` on configurable completion threshold [Inspection Readiness]

**I6. Variation Filing Integration with `pharma_reg`** — `async file_variation()` invokes `pharma_reg` contract, attaches submission number to `LabelRecord`, listens for approval/rejection events via MQ [Regulatory Submission]

**I7. Real-Time Commitment Risk Scoring** — Continuous score `days_remaining/total_duration` weighted by criticality and authority tier (FDA > EMA > NCA); emits `commitment_risk_changed` events at configurable thresholds [Risk Management]

**I8. Regulatory Intelligence NLP Pipeline** — `async classify_intel()` via local Ollama/`nlpc`: extracts affected product classes, dossier sections, impact tags; auto-routes to product owners [AI/NLP]

**I9. Authority Interaction Minutes & Action Item Tracker** — `MeetingMinutes` model with structured action items, owners, due dates; emits `action_item_created` events and integrates with `wflo` for reminders [Regulatory Affairs]

**I10. Import Licence Automated Renewal Workflow** — State-machine (applied → issued → active → near_expiry → renewal_submitted → renewed); `async check_licence_renewals()` cron blocks shipment releases on lapsed licences [Import/Export]

**I11. Regulatory Submission Dossier Completeness Check** — `async validate_dossier(dossier_id, submission_type)` applies eCTD v3.2.2/NeeS rule engine, returns per-section pass/fail report, blocks submission on critical gaps [Submission Quality]

**I12. Multi-Jurisdiction Parallel Compliance Tracking** — `JurisdictionProfile` model normalises FDA/EMA/HC/PMDA/TGA per-product, aggregates cross-border compliance calendars, surfaces conflicting reporting cycles [Global Compliance]

**I13. Batch Record Integration for Recall Readiness** — `async trace_batch(batch_id)` joins batch release (`pharma_qms`), label (`pharma_rec`), and distribution (`pharma_sup`) into a traceability bundle within a 10-minute SLA [Supply Chain]

**I14. Configurable Escalation Matrix via Event Bus** — `async dispatch_escalation(event_type, reference_id, tenant_id)` backed by `ntfy`; tenant-scoped, hot-reloadable from `conf`, maps event types to roles/channels/delays [Notifications]

**I15. Compliance Health Score with Benchmarking** — Weighted KPI score 0–100 per tenant-product (inspection 30%, commitment risk 25%, label 15%, PMS 15%, intel 10%, training 5%); historical trend storage and anonymised peer benchmarking [Analytics]

---

## New Methods

The following async methods (added in v2.0) are the highest-impact additions to `RecordsManagementService`.

### `compliance_report()`

Generate a structured GxP compliance report for a tenant, aggregating framework status, open gaps, overdue commitments, and inspection outcomes.

```python
svc = RecordsManagementService()

report = await svc.compliance_report(tenant_id="acme", standard="GxP")
# Returns dict with keys: standard, tenant_id, generated_at,
#   frameworks_registered, open_gaps, overdue_commitments,
#   recent_inspections, compliance_score
print(report["compliance_score"])  # e.g. 87.4
```

### `analytics_summary()`

Compute aggregated analytics over a configurable period — useful for management dashboards and trend detection.

```python
summary = await svc.analytics_summary(tenant_id="acme", period="monthly")
# Returns dict with keys: period, tenant_id, total_records, events_by_type,
#   compliance_rate, top_risks
# period accepts: "daily" | "weekly" | "monthly" | "quarterly"
for risk in summary["top_risks"]:
    print(risk["commitment_id"], risk["days_overdue"])
```

### `export_records()`

Export all tenant regulatory records in a portable format for audit submission or cross-system migration.

```python
export = await svc.export_records(tenant_id="acme", format="json")
# Returns dict with keys: tenant_id, format, exported_at, record_count, data
# format accepts: "json" | "csv"
import json
with open("audit_export.json", "w") as f:
    json.dump(export["data"], f, indent=2)
```
