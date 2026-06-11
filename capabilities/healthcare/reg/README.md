# Healthcare Regulatory

## Overview

Regulatory compliance management for healthcare facilities. Covers facility and professional licensing with expiry risk scoring, accreditation management (TJC, DNV, CAP, etc.), incident reporting with sentinel event workflow enforcement, AI-assisted ICD-10 coding, HIPAA Security Rule gap analysis, cross-framework compliance matrix, survey readiness scoring, breach notification timelines, state-specific rule evaluation, regulatory intelligence feeds, and RCA workflow management. Sentinel event closure requires a completed root cause analysis reference.

## Capability ID

`healthcare_reg`

## Provides

- `facility_licensing_management` — Track facility and professional licenses with predictive expiry risk scoring
- `accreditation_management` — Manage accreditation cycles for TJC, DNV, CAP, AABB, CMS, and other bodies
- `incident_reporting` — Report patient safety incidents from near-miss to sentinel events with RCA workflow engine
- `hipaa_compliance_tracking` — HIPAA Security Rule gap analysis, risk assessments, and breach notification timelines
- `regulatory_submission_management` — Manage CMS IQR/OQR, state, DEA, and FDA MDR submission lifecycle
- `audit_management` — Internal, external, and mock survey audit tracking with survey readiness scorecard
- `corrective_action_tracking` — Open, assign, complete, and verify corrective actions linked to incidents or findings
- `compliance_dashboard` — Cross-framework compliance status dashboard with real-time KPI cards
- `icd_code_suggestion` — AI-assisted ICD-10/CPT code suggestion via locally-hosted Ollama model
- `regulatory_intelligence` — Real-time feed of CMS, FDA MedWatch, and OIG regulatory updates mapped to capability areas
- `state_rules_engine` — State-specific regulatory obligation evaluation (CA, TX, NY, and extensible)
- `compliance_matrix` — Multi-framework control heat map across HIPAA, CMS, TJC, DEA, and state health codes

## Requires

| Capability | Purpose |
|-----------|---------|
| `auth` | Role-based access for quality, compliance, and regulatory staff |
| `audl` | Immutable audit trail for all regulatory records |
| `mten` | Multi-tenant isolation |
| `conf` | Tenant-specific regulatory framework configuration |
| `ntfy` | License expiry alerts and sentinel event notifications |
| `wflo` | Incident investigation and corrective action approval workflows |
| `comp` | Regulatory compliance framework tracking |
| `moni` | Submission deadline monitoring |
| `mqeb` | Event emission via NATS JetStream for downstream quality analytics |

## Configuration

| Key | Description | Default |
|-----|-------------|---------|
| `licensing.expiry_warning_days` | Days before license expiry to trigger alert | 90 |
| `licensing.risk_score_enabled` | Enable probabilistic lapse risk scoring | true |
| `incidents.sentinel_event_notification_hours` | Hours to notify after sentinel event | 72 |
| `incidents.root_cause_analysis_required_for_sentinel` | Block sentinel close without RCA reference | true |
| `submissions.supported_types` | Allowed regulatory report types | see contract |
| `hipaa.gap_analysis_enabled` | Enable automated HIPAA Security Rule gap analysis | true |
| `intelligence.sources` | Regulatory intelligence feed sources | cms, fda_medwatch, oig_work_plan |
| `state_rules.enabled_states` | US states with active rule sets loaded | CA, TX, NY |
| `icd_suggestion.model` | Ollama model name for ICD-10 suggestions | llama3-medical |

## API Routes

| Method | Path | Description | Permission |
|--------|------|-------------|------------|
| GET | `/api/healthcare/reg/licenses` | List licenses | `healthcare_reg:licenses` |
| POST | `/api/healthcare/reg/licenses` | Add license | `healthcare_reg:licenses` |
| GET | `/api/healthcare/reg/licenses/<id>` | License detail | `healthcare_reg:licenses` |
| GET | `/api/healthcare/reg/licenses/<id>/risk-score` | Predictive expiry risk score | `healthcare_reg:licenses` |
| POST | `/api/healthcare/reg/licenses/<id>/renew` | Initiate renewal | `healthcare_reg:licenses` |
| GET | `/api/healthcare/reg/accreditation` | List accreditations | `healthcare_reg:accreditation` |
| POST | `/api/healthcare/reg/accreditation` | Add accreditation | `healthcare_reg:accreditation` |
| PUT | `/api/healthcare/reg/accreditation/<id>/status` | Update status | `healthcare_reg:accreditation` |
| GET | `/api/healthcare/reg/accreditation/readiness/<body>` | Survey readiness scorecard | `healthcare_reg:accreditation` |
| GET | `/api/healthcare/reg/incidents` | List incidents | `healthcare_reg:incidents` |
| POST | `/api/healthcare/reg/incidents` | Report incident | `healthcare_reg:incidents_write` |
| GET | `/api/healthcare/reg/incidents/<id>` | Incident detail | `healthcare_reg:incidents` |
| POST | `/api/healthcare/reg/incidents/<id>/close` | Close incident | `healthcare_reg:incidents_write` |
| POST | `/api/healthcare/reg/incidents/<id>/rca` | Create RCA workflow | `healthcare_reg:incidents_write` |
| POST | `/api/healthcare/reg/incidents/<id>/rca/<wid>/advance` | Advance RCA stage | `healthcare_reg:incidents_write` |
| GET | `/api/healthcare/reg/submissions` | List submissions | `healthcare_reg:submissions` |
| POST | `/api/healthcare/reg/submissions` | File submission | `healthcare_reg:submissions` |
| POST | `/api/healthcare/reg/submissions/<id>/submit` | Submit to agency | `healthcare_reg:submissions` |
| GET | `/api/healthcare/reg/corrective-actions` | List CAs | `healthcare_reg:corrective_actions` |
| POST | `/api/healthcare/reg/corrective-actions` | Create CA | `healthcare_reg:corrective_actions` |
| POST | `/api/healthcare/reg/corrective-actions/<id>/complete` | Complete CA | `healthcare_reg:corrective_actions` |
| POST | `/api/healthcare/reg/hipaa/risk-assessment` | HIPAA risk assessment | `healthcare_reg:hipaa` |
| POST | `/api/healthcare/reg/hipaa/gap-analysis` | HIPAA Security Rule gap analysis | `healthcare_reg:hipaa` |
| POST | `/api/healthcare/reg/hipaa/breach` | Data breach notification | `healthcare_reg:hipaa` |
| GET | `/api/healthcare/reg/hipaa/breach/<id>/timeline` | Breach notification timeline | `healthcare_reg:hipaa` |
| GET | `/api/healthcare/reg/compliance/matrix` | Cross-framework compliance matrix | `healthcare_reg:compliance` |
| GET | `/api/healthcare/reg/compliance/dashboard` | Full compliance dashboard | `healthcare_reg:view` |
| GET | `/api/healthcare/reg/compliance/calendar` | Regulatory calendar | `healthcare_reg:view` |
| POST | `/api/healthcare/reg/icd/suggest` | AI ICD-10 code suggestions | `healthcare_reg:icd` |
| GET | `/api/healthcare/reg/intelligence` | Regulatory intelligence feed | `healthcare_reg:intelligence` |
| POST | `/api/healthcare/reg/state-rules/evaluate` | State-specific rule evaluation | `healthcare_reg:state_rules` |

## Key Service Methods (New in v1.1)

| Method | Description |
|--------|-------------|
| `suggest_icd_codes(clinical_text, max_suggestions)` | AI-assisted ICD-10 suggestions via Ollama |
| `hipaa_gap_analysis(tenant_id, config_snapshot)` | Automated HIPAA Security Rule gap analysis |
| `compliance_matrix_status(tenant_id, frameworks)` | Cross-framework compliance control heat map |
| `license_expiry_risk_score(tenant_id, lic_id)` | Probabilistic license lapse risk score |
| `rca_workflow_create(incident_id, rca_type)` | Structured TJC RCA2 workflow for sentinel events |
| `rca_workflow_advance(incident_id, workflow_id, stage, stage_data)` | Advance RCA workflow stage |
| `survey_readiness_scorecard(tenant_id, accreditation_body)` | Continuous survey readiness scoring |
| `breach_notification_timeline(tenant_id, breach_id, ...)` | Multi-jurisdiction breach notification obligations |
| `regulatory_intelligence_fetch(tenant_id, sources, since_days)` | Live regulatory update feed from CMS/FDA/OIG |
| `state_rules_evaluate(tenant_id, state_code, operation, context)` | State-specific regulatory rule evaluation |

## Business Rules

| Rule | Condition | Effect |
|------|-----------|--------|
| `sentinel_event_requires_rca` | close_incident + sentinel_event + rca_completed=False | deny |
| `closed_submission_not_modifiable` | update_submission + status=closed | deny |
| `sentinel_event_notification_required` | incident_type=sentinel_event + notification_sent=False | warn |
| `hipaa_breach_requires_notification` | incident_type=hipaa_breach + breach_notification_sent=False | warn |
| `license_expiry_alert_required` | days_to_expiry=90 + alert_sent=False | warn |
| `large_breach_media_notice` | records_affected>=500 + media_notice_sent=False | warn |
| `rca_45_day_deadline` | sentinel_event + rca_days_elapsed>45 | critical_alert |
| `gdpr_72h_dpa_notification` | jurisdiction=gdpr + breach_hours_elapsed>72 | deny |

## Data Models

- `LicenseCreate/Response` — license_type, license_number, expiry_date, days_to_expiry, renewal_initiated
- `AccreditationCreate/Response` — accreditation_body, program, award_date, expiry_date, status
- `IncidentCreate/Response` — incident_type, severity, rca_completed, rca_reference, corrective_actions
- `RegulatorySubmissionCreate/Response` — report_type, submission_reference, status, decision_at
- `CorrectiveActionCreate/Response` — source, assigned_to, due_date, status, verified_by

## Streaming Events (NATS JetStream)

| Event | Subject | Description |
|-------|---------|-------------|
| `license_added` | `apg.healthcare.reg.{tenant}.license` | New license registered |
| `license_expiring` | `apg.healthcare.reg.{tenant}.alerts` | License within expiry window |
| `accreditation_status_changed` | `apg.healthcare.reg.{tenant}.accreditation` | Status update |
| `incident_reported` | `apg.healthcare.reg.{tenant}.incident` | New incident filed |
| `sentinel_event_reported` | `apg.healthcare.reg.{tenant}.alerts.critical` | Sentinel event — 72h clock |
| `rca_workflow_created` | `apg.healthcare.reg.{tenant}.rca` | RCA workflow initiated |
| `submission_filed` | `apg.healthcare.reg.{tenant}.submission` | Submission created |
| `submission_accepted` | `apg.healthcare.reg.{tenant}.submission` | Agency acceptance |
| `corrective_action_completed` | `apg.healthcare.reg.{tenant}.cap` | CAR closed |
| `hipaa_gap_analysis_completed` | `apg.healthcare.reg.{tenant}.hipaa` | Gap analysis result |
| `breach_notification_timeline_generated` | `apg.healthcare.reg.{tenant}.breach` | Timeline obligations |
| `regulatory_intelligence_fetched` | `apg.healthcare.reg.intelligence.{tenant}` | New regulatory updates |

## Edge Cases Handled

- Sentinel event incidents cannot be closed without a non-empty RCA reference — hard deny at service layer
- License `days_to_expiry` is computed at creation; refresh via `list_licenses` for current value
- Closed submissions cannot be modified; file an amendment submission
- Serious adverse device events from `healthcare_dev` map directly to FDA MDR incidents
- Breaches with 500+ records automatically require media notice (45 CFR 164.406)
- GDPR DPA notification deadline is computed in hours, not days, and sorts first in breach timelines
- RCA workflows older than 45 days from incident occurrence emit a critical alert to NATS

## Composability Notes

Quality indicators from `healthcare_ana` feed into CMS IQR/OQR submission auto-population. Device adverse events from `healthcare_dev` map to FDA MDR incidents via `mdr_submission_pipeline`. Controlled substance logs from `healthcare_pha` underpin DEA Schedule II submissions. HIPAA breach incidents trigger the breach notification workflow through `ntfy`. RCA workflows emit state changes consumed by `wflo` for approval routing. Regulatory intelligence items are mapped to affected capability areas and forwarded to relevant capability owners via `ntfy`.

## Streaming Platform

Event streaming uses **NATS JetStream** + **Bytewax** for real-time compliance KPI aggregation. Subjects follow the pattern `apg.healthcare.reg.{event_category}.{tenant_id}`. Consumer groups allow multiple downstream capabilities to subscribe independently without coordination.
