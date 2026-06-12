# USSD Government Services (gov_usd)

USSD-based government services: permit status, tax balance inquiry, ID verification, certificate requests, citizen portfolio, SLA compliance, fraud detection, and cryptographic receipts.

## Overview

Provides a complete USSD gateway for citizen-facing government services, accessible from any mobile phone without internet. Supports session management, OTP authentication, payment references, SMS confirmations, permit expiry alerts, workflow orchestration, and telemetry.

## Capability ID

`gov_usd` | Domain: `government` | Version: `1.1.0`

## Features

| Feature | Description |
|---------|-------------|
| USSD Session Management | Full lifecycle: create, advance, close, resume, delete |
| Permit Status Enquiry | Check validity for 6 permit types |
| Tax Balance Enquiry | KRA PIN lookup for 7 tax types |
| ID Verification | IPRS/immigration lookup for national ID, passport, alien card, military ID |
| Certificate Requests | Apply, track, and bulk-update 7 certificate types |
| OTP Authentication | 6-digit USSD OTP with attempt locking |
| Payment References | Government fee payment codes with M-Pesa confirmation |
| Citizen Portfolio | Consolidated service history per MSISDN |
| Rate Limiting | Rolling-window per-MSISDN throttle |
| Fraud Risk Scoring | Composite 0-1 risk signal from ID failures, OTP failures, burst rate |
| Permit Expiry Alerts | Proactive SMS at 90/30/7 days before permit expiry |
| SLA Compliance | Breach detection for pending certificate requests |
| Bulk Certificate Updates | Atomic batch status reconciliation |
| Cryptographic Receipts | HMAC-SHA256 tamper-evident payment receipts |
| Receipt Verification | USSD-verifiable receipt code lookup |
| Permit Workflow Orchestration | Concurrent ID + tax + permit fan-out via asyncio.gather |
| Telemetry Snapshot | Structured metrics for bytewax/NATS aggregation |
| USSD Menu Management | Localisation-ready multi-level menu definitions |
| SMS Notifications | Post-USSD confirmation delivery |
| Audit Trail | Append-only event log per tenant |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/government/usd/health | Service health check |
| GET | /api/government/usd/dashboard | Dashboard metrics |
| GET | /api/government/usd/sessions | List USSD sessions |
| GET | /api/government/usd/sessions/{id} | Get session |
| POST | /api/government/usd/sessions | Create session |
| PUT | /api/government/usd/sessions/{id} | Advance session |
| DELETE | /api/government/usd/sessions/{id} | Delete session |
| POST | /api/government/usd/permits/enquiries | Enquire permit status |
| GET | /api/government/usd/permits/enquiries | List permit enquiries |
| POST | /api/government/usd/tax/enquiries | Enquire tax balance |
| GET | /api/government/usd/tax/enquiries | List tax enquiries |
| POST | /api/government/usd/id-verifications | Verify ID |
| GET | /api/government/usd/id-verifications | List verifications |
| GET | /api/government/usd/certificates | List certificate requests |
| GET | /api/government/usd/certificates/{id} | Get certificate request |
| POST | /api/government/usd/certificates | Submit certificate request |
| PUT | /api/government/usd/certificates/{id} | Update certificate request |
| DELETE | /api/government/usd/certificates/{id} | Delete certificate request |
| GET | /api/government/usd/citizen/{msisdn}/portfolio | Citizen portfolio |
| POST | /api/government/usd/rate-limit/check | Check rate limit |
| POST | /api/government/usd/fraud/score | Fraud risk score |
| POST | /api/government/usd/permits/expiry-alerts | Schedule expiry alerts |
| GET | /api/government/usd/sla/compliance | SLA compliance report |
| POST | /api/government/usd/certificates/bulk-update | Bulk update certificates |
| POST | /api/government/usd/payments/{id}/receipt | Generate signed receipt |
| POST | /api/government/usd/receipts/verify | Verify receipt code |
| POST | /api/government/usd/workflows/permit | Orchestrate permit workflow |
| GET | /api/government/usd/telemetry | Telemetry snapshot |
| GET | /api/government/usd/audit-events | List audit events |

## New Methods (v1.1.0)

| Method | Description |
|--------|-------------|
| `get_citizen_portfolio(msisdn, tenant_id)` | Aggregate full service history for a MSISDN |
| `check_rate_limit(msisdn, operation, ...)` | Enforce rolling-window per-MSISDN rate limits |
| `score_fraud_risk(msisdn, tenant_id)` | Compute composite fraud risk score |
| `schedule_permit_expiry_alerts(tenant_id, warning_days)` | Proactive SMS for permits nearing expiry |
| `check_sla_compliance(tenant_id, sla_windows_days)` | Detect SLA-breached certificate requests |
| `bulk_update_certificate_requests(updates, tenant_id)` | Atomic batch status reconciliation |
| `generate_signed_receipt(reference_id, tenant_id)` | HMAC-SHA256 tamper-evident receipt |
| `verify_receipt_code(receipt_code, tenant_id)` | Validate a previously issued receipt code |
| `orchestrate_permit_workflow(...)` | Concurrent ID + tax + permit compliance fan-out |
| `emit_telemetry_snapshot(tenant_id)` | Structured telemetry for bytewax/NATS pipelines |

## Composability

This capability integrates with:

- **fintech/payments** — M-Pesa STK push for payment references
- **intel/alerts** — SLA breach and fraud risk event subscriptions via NATS `gov.usd.*`
- **intel/correlation** — Cross-session fraud pattern correlation on MSISDN
- **common/notifications** — SMS and push delivery for expiry alerts

NATS subjects published:
- `gov.usd.session.created` / `gov.usd.session.closed`
- `gov.usd.payment.confirmed`
- `gov.usd.sla_breach_detected`
- `gov.usd.fraud_risk_scored`
- `gov.usd.metrics.<tenant_id>`

## Quick Start

```python
from capabilities.government.usd.service import USSDGovService

svc = USSDGovService(tenant_id="nairobi_county")

# Start a session
session = await svc.create_session("254700000000", "*384#")

# Orchestrate a full permit workflow
result = await svc.orchestrate_permit_workflow(
    msisdn="254700000000",
    id_number="12345678",
    id_type="national_id",
    tax_pin="A000000000A",
    permit_number="BP-2024-001234",
    permit_type="business_permit",
)
print(result["outcome"])  # "approved" | "rejected"
```

## Supported Types

**Permit types**: `business_permit`, `building_permit`, `health_certificate`, `liquor_licence`, `food_hygiene`, `environmental_clearance`

**Tax types**: `income_tax`, `vat`, `paye`, `corporation_tax`, `withholding_tax`, `excise_duty`, `turnover_tax`

**ID types**: `national_id`, `passport`, `alien_card`, `military_id`

**Certificate types**: `good_conduct`, `tax_compliance`, `business_registration`, `birth_certificate`, `death_certificate`, `marriage_certificate`, `clearance_certificate`

---

## World-Class Enhancements (v2.0)

- **I1.** USSD Government Services — World-Class Improvements
- **I2.** Adaptive Multi-Modal Session Continuity
- **I3.** Biometric Liveness-Anchored ID Verification
- **I4.** NATS-Backed Real-Time Event Streaming
- **I5.** Permit Expiry Early-Warning Push
- **I6.** Composable Payment Gateway Orchestration
- **I7.** Citizen Service History & Portfolio
- **I8.** Dynamic Menu Localisation (Swahili / Regional Languages)
- **I9.** Rate Limiting & Fraud Pattern Detection
- **I10.** Bulk Certificate Status Reconciliation
- **I11.** Service-Level Agreement (SLA) Tracking
- **I12.** Offline-Capable Agent (Sub-dealer) USSD Proxy
- **I13.** Tax Payment Instalment Plan Management
- **I14.** Cryptographic Receipt Generation
- **I15.** Cross-Capability Workflow Orchestration via NATS

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
