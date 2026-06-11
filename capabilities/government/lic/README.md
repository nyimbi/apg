# Licensing and Permits

**Capability ID**: `government_lic` | **Domain**: `government` | **Version**: `1.1.0`
**Company**: Datacraft | **Copyright**: © 2025

## Overview

Business and professional licence applications, renewals, inspections, revocations, and fee collection with full compliance monitoring. Enforces that licences cannot be renewed if the last inspection failed, prevents duplicate licences, and requires formal notice before revocation.

v1.1.0 adds: risk-based compliance scoring, SLA tracking, late-fee auto-assessment, W3C VC digital credentials, revocation appeals, offline mobile inspector sync, scored inspection checklists, policy impact analysis, and a ranked compliance scorecard.

## Capability ID
`government_lic`

## Provides

| Workflow | Description |
|---|---|
| `licence_application_workflow` | Application intake and document verification |
| `licence_issuance_workflow` | Issue licence after approved application |
| `inspection_scheduling_workflow` | Schedule and record licence inspections |
| `licence_renewal_workflow` | Renewal with inspection pre-requisite check |
| `fee_collection_workflow` | Application, renewal, and penalty fee collection |
| `licence_revocation_workflow` | Revocation with notice period enforcement |
| `licensing_review_workflow` | Governance review of licensing decisions |
| `licensing_agent_workflow` | Automated renewal notification and compliance agents |
| `licence_status_tracking_workflow` | Real-time licence status monitoring |
| `compliance_monitoring_workflow` | Ongoing compliance against licence conditions |
| `risk_based_inspection_workflow` | Risk-scored compliance inspection targeting |
| `sla_tracking_workflow` | Application processing SLA monitoring and escalation |
| `digital_credential_workflow` | W3C VC digital licence credential issuance |
| `appeal_workflow` | Revocation appeal filing and tribunal tracking |
| `offline_inspection_workflow` | Mobile inspector offline sync and reconciliation |

## Requires

| Capability | Reason |
|---|---|
| auth | Applicant and officer RBAC |
| audl | Licensing decision audit trail |
| mten | Tenant-scoped licence registry |
| conf | Licence type configuration and fee schedules |
| ntfy | Renewal reminders and inspection notifications |
| wflo | Application and approval workflow |
| schd | Inspection scheduling |
| comp | Regulatory compliance checks |
| moni | Expiry and compliance monitoring |
| mqeb | Event streaming via bytewax |

## Configuration

| Key | Description |
|---|---|
| `governance.licence_without_payment_denied` | Application fee must be paid before processing |
| `governance.expired_licence_operation_denied` | Operations under expired licence blocked |
| `governance.inspection_fail_blocks_renewal` | Failed inspection prevents renewal |
| `governance.duplicate_licence_denied` | One active licence per holder per type |
| `governance.late_fee_rate_per_day` | Late renewal penalty rate in KES (default: 500) |
| `governance.sla_days.business` | SLA target for business licence applications (default: 21) |
| `governance.sla_days.professional` | SLA target for professional licence applications (default: 14) |
| `governance.sla_days.temporary` | SLA target for temporary permit applications (default: 5) |
| `governance.inspection_pass_threshold_pct` | Minimum checklist score to pass inspection (default: 80) |
| `governance.appeal_window_days` | Days after revocation within which appeal may be filed (default: 30) |

## API Routes

| Path | Method | Description | Permission |
|---|---|---|---|
| `/government-lic/applications` | GET/POST | Licence applications | `government_lic:apply` |
| `/government-lic/licences` | GET | Licence register | `government_lic:licences` |
| `/government-lic/inspections` | GET/POST | Inspection schedule | `government_lic:inspect` |
| `/government-lic/renewals` | GET/POST | Licence renewals | `government_lic:renew` |
| `/government-lic/fees` | GET/POST | Fee collection | `government_lic:fees` |
| `/government-lic/revocations` | GET/POST | Revocations | `government_lic:revoke` |
| `/government-lic/compliance` | GET | Compliance dashboard | `government_lic:compliance` |
| `/government-lic/risk-scores` | GET | Risk-based compliance scores | `government_lic:compliance` |
| `/government-lic/sla` | GET | SLA status report | `government_lic:compliance` |
| `/government-lic/credentials/<id>` | GET | Digital licence credential (W3C VC) | `government_lic:licences` |
| `/government-lic/appeals` | POST | File revocation appeal | `government_lic:appeal` |
| `/government-lic/inspections/sync` | GET | Offline inspection sync payload | `government_lic:inspect` |
| `/government-lic/scorecard` | GET | Compliance scorecard | `government_lic:compliance` |

## Key Service Methods

### Core (sync)

| Method | Description |
|---|---|
| `describe()` | Return capability contract |
| `evaluate(context)` | Evaluate policy rules |
| `submit_application(...)` | Submit a licence application (full params) |
| `apply_licence(...)` | Submit via simplified interface |
| `background_check(application_id)` | Run background check on applicant |
| `premises_inspection(...)` | Schedule premises inspection |
| `issue_licence(...)` | Issue licence after approval |
| `renew_licence(...)` | Renew with full params |
| `licence_renewal(licence_id, docs)` | Renew via simplified interface |
| `suspend_licence(...)` | Suspend licence for period |
| `revoke_licence(...)` | Revoke with full params |
| `licence_revoke(licence_id, reason)` | Revoke via simplified interface |
| `licence_register(filters)` | Query public licence register |
| `fee_collection(...)` | Collect a fee payment |
| `collect_fee(...)` | Collect fee with full params |
| `schedule_inspection(...)` | Schedule inspection |
| `record_inspection_outcome(...)` | Record inspection result |
| `record_review(...)` | Record governance review |
| `register_agent(...)` | Register licensing agent |
| `validate_batch(...)` | Validate batch via bytewax |
| `dashboard_summary(tenant_id)` | Return dashboard counts |

### Async (v1.0.0)

| Method | Description |
|---|---|
| `citizen_licence_lookup(citizen_id)` | All licences for a citizen (portal self-service) |
| `bulk_licence_renewal(licence_ids)` | Bulk renew up to 200 licences |
| `compliance_audit()` | Tenant compliance audit report |
| `expiry_notifications(days_ahead)` | Licences expiring within N days |
| `online_application(...)` | Accept online application via citizen portal |
| `fee_reconciliation()` | Reconcile fees vs applications |
| `background_check_status(application_id)` | Background check status query |
| `regulatory_reporting(period)` | Regulatory activity report |
| `performance_kpi_report()` | Licensing KPI card |
| `export_licences(fmt)` | Export licence registry (csv/json) |
| `health_check()` | Service health metrics |
| `audit_trail(from_date, to_date)` | Audit events in date range |
| `random_compliance_inspection()` | Select licences for random inspection |
| `bulk_status_update(updates)` | Bulk licence status updates |
| `inter_jurisdiction_check(...)` | Validate licence in target jurisdiction |
| `licence_renewal_pipeline(days_ahead)` | Licences due for renewal sorted by urgency |
| `licence_kpi_summary(period)` | Concise KPI card for dashboard |
| `licence_analytics_detail(period)` | Analytics by type, status, jurisdiction |

### Async (v1.1.0 — world-class enhancements)

| Method | Description |
|---|---|
| `risk_score_licence(licence_id)` | Compute risk-based compliance score (0–100) |
| `sla_status_report()` | SLA compliance report for pending applications |
| `late_fee_assessment(licence_id)` | Assess and record late renewal penalty |
| `appeal_revocation(...)` | File appeal against revocation (30-day window) |
| `inspection_checklist_evaluate(...)` | Score completed inspection checklist |
| `impact_analysis(proposed_change)` | Dry-run policy/fee change impact analysis |
| `digital_licence_credential(licence_id)` | Issue W3C VC digital licence credential |
| `inspection_sync_payload(inspector_id)` | Package inspections for offline mobile sync |
| `compliance_scorecard()` | Ranked compliance scorecard for all active licences |

## Business Rules

| Rule | Condition | Effect |
|---|---|---|
| `application_fee_required` | `fee_paid=False` | deny |
| `duplicate_licence_denied` | `duplicate_detected=True` | deny |
| `renewal_inspection_fail_blocks` | `last_inspection_failed=True` | deny |
| `revocation_notice_required` | `notice_served=False` | deny |
| `revocation_approval_required` | `approval_present=False` | deny |
| `late_fee_blocks_renewal` | `late_fee.paid=False` | deny |
| `appeal_window_enforced` | `days_since_revocation > 30` | deny |
| `checklist_pass_threshold` | `score_pct < 80` | outcome=fail |

## Data Models

| Model | Key Fields |
|---|---|
| `LicenceApplication` | id, tenant_id, licence_type, applicant_id, status, fee_paid |
| `Licence` | id, tenant_id, licence_type, licence_number, holder_id, expiry_date, status |
| `LicenceInspection` | id, licence_id, inspection_type, inspector_id, scheduled_date, outcome, findings |
| `LicenceRenewal` | id, licence_id, renewal_type, new_expiry_date, renewal_fee_paid |
| `FeeRecord` | id, application_id, fee_type, amount, currency, receipt_number, paid |
| `LicenceRevocation` | id, licence_id, reason, notice_served, approval_reference, revoked_at |
| `LicensingReview` | id, reference_id, reviewer_id, status, evidence_reference |
| `LicensingAgent` | id, name, runtime, role, scope |

## Streaming Events

```
licence_application_submitted    licence_issued             inspection_scheduled
inspection_outcome_recorded      licence_renewed            fee_collected
licence_suspended                licence_revoked            licence_expired
lic_risk_scored                  lic_sla_reported           lic_late_fee_assessed
lic_revocation_appeal_filed      lic_checklist_evaluated    lic_impact_analysed
lic_digital_credential_issued    lic_inspection_sync_packaged  lic_scorecard_generated
```

## Edge Cases Handled

- Renewal after failed inspection — blocked until re-inspection passes
- Duplicate licence for same holder and type — blocked even if previous expired
- Revocation without notice period — denied regardless of breach severity
- Late renewal automatically assessed a `late_renewal_penalty` fee blocking issuance until paid
- Multiple inspection types active simultaneously for complex facilities
- Appeal filed after 30-day window — rejected with clear error
- Checklist score below 80% — outcome set to `fail` regardless of individual item count
- Offline inspector sync — idempotent payload with 48-hour TTL

## Composability Notes

Composes with:
- `government_csr` — licence applications submitted through citizen portal
- `government_bud` — licence fees credited to AIA vote accounts
- `government_cas` — licence complaints create cases; appeals trigger case records
- `government_con` — contractor registration uses professional licence
- `government_per` — building permits require valid contractor licence
- `government_pay` — payment gateway integration for online fee collection
