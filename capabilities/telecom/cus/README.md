# Customer Management

**Capability ID**: `telecom_cus` | **Domain**: `telecom` | **Version**: `1.1.0`
**Company**: Datacraft | **Copyright**: © 2025

## Overview

End-to-end customer lifecycle management covering onboarding, KYC verification,
plan activation, SIM and device management, customer service case tracking, churn
risk scoring, SLA breach monitoring, dunning workflow, and GDPR/POPIA erasure.
Enforces KYC requirements, credit checks for postpaid plans, IMEI blacklist checks,
and tenant-scoped PII access controls.

## Capability ID
`telecom_cus`

## Provides
- `customer_lifecycle_workflow`: Prospect → active → churn lifecycle
- `kyc_workflow`: Document submission, verification, rejection, and erasure
- `plan_management_workflow`: Prepaid, postpaid, and hybrid plan activation
- `sim_management_workflow`: SIM provisioning, swap (with fraud safeguard), and block
- `device_management_workflow`: IMEI registration with blacklist check
- `case_tracking_workflow`: Multi-type case management with SLA tracking and escalation
- `customer_360_view`: Unified customer profile across all sub-entities
- `churn_management_workflow`: Deterministic risk scoring and retention interventions
- `dunning_workflow`: Payment failure → reminder → suspension → deactivation pipeline
- `segmentation_api`: Criteria-based paginated customer filtering for campaign targeting
- `cus_agent_workflow`: Customer management automation agents

## Requires
| Capability | Reason |
|------------|--------|
| auth | Authentication |
| audl | Customer event audit trail |
| mten | Tenant isolation |
| conf | Configuration |
| ntfy | Customer notifications |
| wflo | Case and plan workflow states |
| nlpc | Customer search and case NLP |
| mqeb | Event streaming |
| comp | KYC and regulatory compliance |

## Configuration
| Key | Description |
|-----|-------------|
| customers.kyc_required | KYC is mandatory at creation |
| plans.credit_check_for_postpaid | Postpaid requires credit check |
| sims.max_sims_per_customer | Hard limit of 10 |
| devices.imei_check / blacklist_check | Both mandatory |
| cases.sla_hours | Type-specific SLA hours |

## API Routes
| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| /telecom-cus/customers | GET/POST | Customer console | telecom_cus:customers |
| /telecom-cus/customers/\<id\> | GET | Customer 360 view | telecom_cus:customers |
| /telecom-cus/kyc | GET/POST | KYC document console | telecom_cus:kyc |
| /telecom-cus/plans | GET/POST | Plan activation | telecom_cus:plans |
| /telecom-cus/sims | GET/POST | SIM management | telecom_cus:sims |
| /telecom-cus/devices | GET/POST | Device registration | telecom_cus:devices |
| /telecom-cus/cases | GET/POST | Case queue | telecom_cus:cases |
| /telecom-cus/sla-breaches | GET | SLA breach dashboard | telecom_cus:cases |
| /telecom-cus/segments | POST | Customer segmentation | telecom_cus:customers |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| customer_kyc_required | KYC not initiated | deny |
| kyc_bypass_denied | agent bypass scope | deny |
| postpaid_credit_check_required | postpaid + no credit check | deny |
| device_blacklist_check_required | blacklist not checked | deny |
| pii_access_requires_approval | PII access without approval | deny |
| cross_tenant_access_denied | cross-tenant agent scope | deny |

## Data Models
- `CusCustomer`: id, tenant_id, customer_type, msisdn, name, status, kyc_status
- `CusKycDocument`: id, tenant_id, customer_id, document_type, document_reference, status, verified_by, expires_at
- `CusPlan`: id, tenant_id, customer_id, plan_type, plan_name, plan_reference, activated_at, status
- `CusSim`: id, tenant_id, customer_id, iccid, imsi, msisdn, status, provisioned_at
- `CusDevice`: id, tenant_id, customer_id, device_type, imei, model, blacklist_checked
- `CusCase`: id, tenant_id, customer_id, case_type, status, description, assigned_to, opened_at, resolved_at
- `CusLifecycleEvent`: id, tenant_id, customer_id, event_type, event_reference, occurred_at
- `CusAgent`: id, tenant_id, name, runtime, role, scope

## Service Methods

### Core (sync)
| Method | Description |
|--------|-------------|
| `create_customer` | Onboard a new subscriber |
| `update_customer_status` | Lifecycle status transitions |
| `submit_kyc_document` | Submit identity document |
| `verify_kyc` / `reject_kyc` | Document verification |
| `activate_plan` | Activate a service plan |
| `provision_sim` | Provision a SIM card |
| `update_sim_status` | Block / swap SIM status |
| `register_device` | IMEI registration with blacklist check |
| `open_case` / `update_case_status` | Case lifecycle |
| `record_lifecycle_event` | Append lifecycle audit entry |
| `register_agent` | Register automation agent |

### Extended (async)
| Method | Description |
|--------|-------------|
| `get_customer_360` | Unified profile in one call |
| `sim_swap` | SIM swap with 30-day fraud cooling-off |
| `get_sla_breaches` | Scan for SLA breaches and at-risk cases |
| `score_churn_risk` | Deterministic churn score (0.0–1.0) |
| `segment_customers` | Criteria-based paginated segmentation |
| `trigger_dunning` | Multi-step dunning pipeline |
| `escalate_case` | Tier-based case escalation with SLA reset |
| `request_data_erasure` | GDPR/POPIA right-to-erasure |
| `bulk_import_customers` | Idempotent batch import with dry-run |
| `create_account` | Full account creation with contact and address |
| `kyc_check` | Batch KYC document submission and auto-verify |
| `activate_service` | Activate a value-added service |
| `suspend_service` / `restore_service` | Service suspension lifecycle |
| `complaint_log` / `complaint_resolution` | Complaint with SLA tracking |
| `churn_risk_intervention` | Execute retention offer |
| `nps_survey_result` / `record_nps` | Record NPS responses |
| `nps_analytics` | NPS score computation |
| `customer_lifecycle_report` | Period analytics report |
| `customer_analytics` | KPI dashboard data |
| `kyc_compliance_report` | Regulatory KYC compliance report |
| `export_customers` | JSON / CSV customer export |
| `health_check` | Service health status |

## Streaming Events
- `customer_onboarded`, `kyc_verified`, `kyc_rejected`, `plan_activated`, `plan_changed`
- `sim_provisioned`, `sim_blocked`, `sim_swapped`
- `case_opened`, `case_resolved`, `case_escalated`, `sla_breach_detected`
- `customer_churned`, `churn_risk_flagged`, `churn_intervention_executed`
- `dunning_reminder_triggered`, `dunning_soft_suspension_triggered`, `dunning_deactivation_triggered`
- `data_erasure_requested`, `nps_detractor_flagged`, `cus_agent_registered`

## Edge Cases Handled
- Prepaid plans do not require credit check even if `credit_check_completed=False`
- KYC verification updates parent customer's `kyc_status` automatically
- SIM status `stolen_blocked` fires a distinct audit event vs generic status update
- SIM swap within 30-day cooling-off window auto-creates a `fraud_report` case
- Churn risk scoring emits `churn_risk_flagged` lifecycle event when score ≥ 0.65
- Dunning suspends service automatically at > 14 days overdue
- Data erasure pseudonymises PII while preserving audit trail
- Bulk import deduplicates by MSISDN and supports dry-run validation
- `get_sla_breaches` emits per-breach audit events enabling automated escalation pipelines

## Composability Notes
Feeds customer identity to `telecom_bil` (invoicing), `telecom_ord` (order validation),
and `telecom_ana` (churn analysis). KYC data feeds `comp` for regulatory reporting.
SIM and device data feeds `telecom_pro` for provisioning workflows.
Segmentation output drives campaign targeting in `telecom_ana`.
Dunning workflow emits events consumed by `telecom_bil`.

## Further Reading
- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `docs/user_guide.md` — Detailed usage guide
- `WORLD_CLASS_IMPROVEMENTS.md` — Improvement roadmap (15 items, 9 implemented)
- `SPECIFICATION.md` — Full capability specification
