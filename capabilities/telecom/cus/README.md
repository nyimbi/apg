# Customer Management

## Overview
End-to-end customer lifecycle management covering onboarding, KYC verification, plan activation, SIM and device management, and customer service case tracking. Enforces KYC requirements, credit checks for postpaid plans, IMEI blacklist checks, and tenant-scoped PII access controls.

## Capability ID
`telecom_cus`

## Provides
- customer_lifecycle_workflow: Prospect → active → churn lifecycle
- kyc_workflow: Document submission, verification, and rejection
- plan_management_workflow: Prepaid, postpaid, and hybrid plan activation
- sim_management_workflow: SIM provisioning, swap, and block
- device_management_workflow: IMEI registration with blacklist check
- case_tracking_workflow: Multi-type case management with SLA tracking
- customer_360_view: Unified customer profile across all sub-entities
- churn_management_workflow: Churn risk flagging and retention actions
- cus_agent_workflow: Customer management automation agents

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
| /telecom-cus/customers/<id> | GET | Customer 360 view | telecom_cus:customers |
| /telecom-cus/kyc | GET/POST | KYC document console | telecom_cus:kyc |
| /telecom-cus/plans | GET/POST | Plan activation | telecom_cus:plans |
| /telecom-cus/sims | GET/POST | SIM management | telecom_cus:sims |
| /telecom-cus/devices | GET/POST | Device registration | telecom_cus:devices |
| /telecom-cus/cases | GET/POST | Case queue | telecom_cus:cases |

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
- CusCustomer: id, tenant_id, customer_type, msisdn, name, status, kyc_status
- CusKycDocument: id, tenant_id, customer_id, document_type, document_reference, status, verified_by, expires_at
- CusPlan: id, tenant_id, customer_id, plan_type, plan_name, plan_reference, activated_at, status
- CusSim: id, tenant_id, customer_id, iccid, imsi, msisdn, status, provisioned_at
- CusDevice: id, tenant_id, customer_id, device_type, imei, model, blacklist_checked
- CusCase: id, tenant_id, customer_id, case_type, status, description, assigned_to
- CusLifecycleEvent: id, tenant_id, customer_id, event_type, event_reference, occurred_at
- CusAgent: id, tenant_id, name, runtime, role, scope

## Streaming Events
- customer_onboarded, kyc_verified, kyc_rejected, plan_activated, plan_changed
- sim_provisioned, sim_blocked, case_opened, case_resolved, customer_churned, cus_agent_registered

## Edge Cases Handled
- Prepaid plans do not require credit check even if credit_check_completed=False
- KYC verification updates parent customer's kyc_status automatically
- SIM status stolen_blocked fires a distinct audit event vs generic status update
- Customer 360 view assembles cross-entity data without extra service calls
- Case SLA tracking hours differ by case type (complaint=24h, technical_fault=4h)

## Composability Notes
Feeds customer identity to telecom_bil (invoicing), telecom_ord (order validation), and telecom_ana (churn analysis). KYC data feeds comp for regulatory reporting. SIM and device data feeds telecom_pro for provisioning workflows.
