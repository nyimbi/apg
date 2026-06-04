# Telecom Billing

## Overview
Convergent billing capability covering the full billing stack: CDR mediation and normalisation, real-time rating and charging, bill cycle management, invoice generation and approval, dunning workflow, payment reconciliation, discount management, and convergent billing for households and corporate groups.

## Capability ID
`telecom_bil`

## Provides
- mediation_workflow: CDR normalisation and duplicate detection
- rating_workflow: Multi-scheme rating (flat, tiered, volume, time-of-day)
- charging_workflow: Usage, recurring, and one-time charge generation
- invoice_workflow: Draft → approval → dispatch lifecycle
- bill_cycle_management: Monthly, quarterly, event-based cycle management
- dunning_workflow: 6-step dunning escalation through to collections
- payment_reconciliation_workflow: Multi-method payment matching
- discount_workflow: Loyalty, promotional, and corporate discounts (max 50%)
- convergent_billing_workflow: Single-bill and household account grouping
- billing_agent_workflow: Automated billing agent management

## Requires
| Capability | Reason |
|------------|--------|
| auth | Authentication |
| audl | Audit trail |
| mten | Tenant context |
| conf | Configuration |
| ntfy | Payment and invoice notifications |
| wflo | Dunning and approval workflows |
| mqeb | Event stream |
| schd | Bill cycle scheduling |
| comp | Regulatory billing compliance |

## Configuration
| Key | Description |
|-----|-------------|
| rating.supported_rating_types | 8 rating schemes |
| bill_cycles.grace_period_days | Default 5-day grace period |
| discounts.max_discount_pct | Hard cap at 50% |
| dunning.escalation_days | [7,14,21,30,45,60] day steps |
| governance.bill_suppression_denied | Agents cannot suppress bills |

## API Routes
| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| /telecom-bil/mediation | GET/POST | CDR mediation console | telecom_bil:mediation |
| /telecom-bil/rating | GET/POST | Charge rating engine | telecom_bil:rating |
| /telecom-bil/bill-cycles | GET/POST | Bill cycle manager | telecom_bil:bill_cycles |
| /telecom-bil/invoices | GET/POST | Invoice console | telecom_bil:invoices |
| /telecom-bil/dunning | GET/POST | Dunning management | telecom_bil:dunning |
| /telecom-bil/payments | GET/POST | Payment ledger | telecom_bil:payments |
| /telecom-bil/discounts | GET/POST | Discount workbench | telecom_bil:discounts |
| /telecom-bil/convergent | GET/POST | Convergent accounts | telecom_bil:convergent |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| tenant_context_required | missing context | deny |
| charge_amount_must_be_positive | amount ≤ 0 | deny |
| invoice_approval_required | approve without reference | deny |
| discount_exceeds_max_allowed | discount_pct > 50 | deny |
| write_off_requires_approval | no approval reference | deny |
| bill_suppression_denied | agent suppresses bill | deny |
| cross_tenant_billing_denied | cross-tenant agent scope | deny |

## Data Models
- BilCdr: id, tenant_id, source, mediation_status, msisdn, duration_seconds, data_volume_bytes
- BilCharge: id, tenant_id, customer_id, charge_type, rating_type, amount, currency, tax_amount
- BilCycle: id, tenant_id, cycle_type, cutoff_date, start_date, end_date, status
- BilInvoice: id, tenant_id, customer_id, cycle_id, total_amount, currency, status, due_date
- BilDunningStep: id, tenant_id, invoice_id, step, triggered_at, next_step_date
- BilPayment: id, tenant_id, invoice_id, payment_method, amount, reference, paid_at
- BilDiscount: id, tenant_id, customer_id, discount_type, discount_pct, approval_reference
- BilConvergentAccount: id, tenant_id, convergent_mode, master_account_id, member_account_ids
- BilAgent: id, tenant_id, name, runtime, role, scope

## Streaming Events
- cdr_mediated, charge_rated, invoice_generated, invoice_approved, invoice_sent
- payment_received, dunning_step_triggered, discount_applied, write_off_recorded, bil_agent_registered

## Edge Cases Handled
- Write-off operations require separate approval from invoice approval
- Discount cap enforced at 50% regardless of discount type or corporate status
- Dunning steps are stateful — skipping steps requires override approval
- CDR duplicate detection is enabled by default; duplicates are held not rejected
- Convergent account members can be on different plan types (prepaid + postpaid)

## Composability Notes
Receives CDRs from telecom_net mediation layer. Consumes customer data from telecom_cus for invoice addressing. Feeds revenue data to telecom_ana for ARPU and revenue assurance analysis. Integrates with comp for tax and regulatory levy calculation.
