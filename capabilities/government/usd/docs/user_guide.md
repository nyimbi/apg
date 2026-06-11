# USSD Government Services — User Guide

## Overview

The USSD Government Services capability (`gov_usd`) enables citizens to access government services from any mobile phone by dialling a USSD code. No internet connection or smartphone required. Version 1.1.0 adds fraud protection, SLA compliance monitoring, permit expiry alerts, cryptographic receipts, and citizen portfolio views.

---

## Supported Services

| Service | USSD Code | Description |
|---------|-----------|-------------|
| Permit Status | `*384#` | Check business, building, health, liquor, food hygiene, environmental permits |
| Tax Balance | `*385#` | KRA PIN tax balance for income tax, VAT, PAYE, and more |
| ID Verification | `*386#` | Verify national ID, passport, alien card against IPRS |
| Certificate Requests | `*400#` | Apply for good conduct, tax compliance, birth/death/marriage certificates |
| Receipt Verification | `*500*VERIFY*{CODE}#` | Validate a cryptographic payment receipt |

---

## Citizen Flows

### Permit Status Enquiry
1. Dial `*384#`
2. Select `1. Check Permit`
3. Enter permit number (e.g. `BP-2024-001234`)
4. Select permit type
5. View status on screen; receive SMS confirmation

### Tax Balance Enquiry
1. Dial `*385#`
2. Enter KRA PIN (e.g. `A000000000A`)
3. Select tax type
4. View outstanding balance and due date

### ID Verification
1. Dial `*386#`
2. Enter ID number
3. Confirm with OTP sent via SMS
4. Receive verification result on screen

### Certificate Request
1. Dial `*400#`
2. Select certificate type
3. Enter national ID and full name
4. Pay via M-Pesa paybill (reference provided)
5. Receive reference number via SMS
6. Track status by reference number: re-dial `*400#` → `2. Track Request`

### Receipt Verification
1. Locate your 16-character receipt code from your confirmation SMS
2. Dial `*500*VERIFY*{CODE}#` (replace `{CODE}` with your code)
3. Screen displays payment amount, date, and service type

---

## API Reference

### Sessions

```
POST /api/government/usd/sessions
{
  "msisdn": "254700000000",
  "service_code": "*384#",
  "tenant_id": "nairobi_county"
}
```

```
PUT /api/government/usd/sessions/{session_id}?tenant_id=nairobi_county
{
  "input_text": "1",
  "session_data": {"selected_service": "permit"}
}
```

### Permit Enquiry

```
POST /api/government/usd/permits/enquiries
{
  "msisdn": "254700000000",
  "permit_number": "BP-2024-001234",
  "permit_type": "business_permit",
  "tenant_id": "nairobi_county"
}
```

### Tax Balance

```
POST /api/government/usd/tax/enquiries
{
  "msisdn": "254700000000",
  "tax_pin": "A000000000A",
  "tax_type": "income_tax",
  "tenant_id": "kra"
}
```

Response includes `outstanding_balance`, `currency`, `due_date`, and `compliance_status`.

### ID Verification

```
POST /api/government/usd/id-verifications
{
  "msisdn": "254700000000",
  "id_number": "12345678",
  "id_type": "national_id",
  "full_name": "John Doe"
}
```

### Certificate Request

```
POST /api/government/usd/certificates
{
  "msisdn": "254700000000",
  "certificate_type": "good_conduct",
  "applicant_id": "12345678",
  "applicant_name": "John Doe",
  "tenant_id": "dci_kenya"
}
```

```
PUT /api/government/usd/certificates/{request_id}?tenant_id=dci_kenya
{
  "status": "issued",
  "certificate_number": "GC-2024-789456",
  "issued_by": "DCI Nairobi"
}
```

### Bulk Certificate Update

```
POST /api/government/usd/certificates/bulk-update
{
  "updates": [
    {"request_id": "cert-abc123", "status": "issued", "certificate_number": "GC-001"},
    {"request_id": "cert-def456", "status": "rejected", "notes": "Incomplete application"}
  ],
  "tenant_id": "dci_kenya"
}
```

All-or-nothing validation: if any `request_id` is not found, zero updates are applied.

---

## New Features (v1.1.0)

### Citizen Portfolio

Retrieve a consolidated view of all government interactions for a MSISDN:

```
GET /api/government/usd/citizen/254700000000/portfolio?tenant_id=nairobi_county
```

Response:
```json
{
  "msisdn": "254700000000",
  "active_permits": 2,
  "pending_certificate_requests": 1,
  "pending_certificate_types": ["good_conduct"],
  "unpaid_references": 1,
  "unpaid_total": 5000.0
}
```

Useful for pre-populating USSD sessions with citizen context, reducing average session depth.

---

### Rate Limiting

Enforce per-MSISDN rolling-window limits to prevent USSD flooding:

```
POST /api/government/usd/rate-limit/check
{
  "msisdn": "254700000000",
  "operation": "permit_enquiry",
  "window_seconds": 60,
  "max_calls": 5,
  "tenant_id": "nairobi_county"
}
```

Returns `allowed: true/false`. Raises HTTP 429 when limit exceeded.

---

### Fraud Risk Scoring

Score a MSISDN's behavioural risk profile (0.0 = clean, 1.0 = high risk):

```
POST /api/government/usd/fraud/score
{
  "msisdn": "254700000000",
  "tenant_id": "nairobi_county"
}
```

Response:
```json
{
  "risk_score": 0.12,
  "risk_level": "low",
  "signals": {
    "id_failure_rate": 0.0,
    "otp_failure_rate": 0.0,
    "burst_score": 0.15,
    "id_enum_score": 0.0
  }
}
```

---

### Permit Expiry Alerts

Trigger proactive SMS alerts for permits expiring within threshold windows:

```
POST /api/government/usd/permits/expiry-alerts
{
  "tenant_id": "nairobi_county",
  "warning_days": [90, 30, 7]
}
```

Sends one SMS per permit at the most urgent applicable threshold. SMS text:
> "PERMIT ALERT: Your Business Permit (BP-2024-001234) expires in 7 day(s) on 2024-12-31. Dial *384# to renew."

---

### SLA Compliance

Check which pending certificate requests are in breach of service level targets:

```
GET /api/government/usd/sla/compliance?tenant_id=dci_kenya
```

Response:
```json
{
  "total_pending": 45,
  "compliant": 40,
  "breached": 5,
  "compliance_rate": 0.8889,
  "breached_requests": [
    {
      "request_id": "cert-xyz",
      "certificate_type": "good_conduct",
      "elapsed_days": 9.5,
      "target_days": 7,
      "is_breached": true
    }
  ]
}
```

Default SLA windows: `good_conduct=7d`, `tax_compliance=3d`, `birth_certificate=5d`, `business_registration=10d`.

---

### Payment Receipts

Generate a cryptographic receipt after a payment is confirmed:

```
POST /api/government/usd/payments/{reference_id}/receipt?tenant_id=nairobi_county
```

Response:
```json
{
  "receipt_code": "A3FK9ZX2MN7P",
  "amount": 5000.0,
  "currency": "KES",
  "service_type": "business_permit_renewal",
  "verify_ussd": "*500*VERIFY*A3FK9ZX2MN7P#"
}
```

Verify a code:

```
POST /api/government/usd/receipts/verify
{
  "receipt_code": "A3FK9ZX2MN7P",
  "tenant_id": "nairobi_county"
}
```

---

### Permit Workflow Orchestration

Run all prerequisite checks for a permit application concurrently:

```
POST /api/government/usd/workflows/permit
{
  "msisdn": "254700000000",
  "id_number": "12345678",
  "id_type": "national_id",
  "tax_pin": "A000000000A",
  "permit_number": "BP-2024-001234",
  "permit_type": "business_permit",
  "tenant_id": "nairobi_county"
}
```

All three checks (ID verification, tax compliance, permit validity) run in parallel via `asyncio.gather`. Response:
```json
{
  "workflow_id": "wf-abc123",
  "outcome": "approved",
  "all_steps_passed": true,
  "steps": {
    "id_verification": {"passed": true, "result_id": "idv-xyz"},
    "tax_compliance":  {"passed": true, "result_id": "taxenq-xyz"},
    "permit_status":   {"passed": true, "result_id": "penq-xyz"}
  }
}
```

---

### Telemetry Snapshot

Emit a structured metrics snapshot for bytewax/NATS aggregation:

```
GET /api/government/usd/telemetry?tenant_id=nairobi_county
```

Published to NATS subject `gov.usd.metrics.<tenant_id>`. Includes session funnel drop-offs by menu level, error counts by event type, resource counts, and service-code breakdown.

---

## Authentication

All endpoints require `tenant_id` (query parameter or request body). Production deployments should add JWT or API-key middleware at the blueprint level.

OTP flow for sensitive operations:
1. Call `POST /api/government/usd/otp` — receive `otp_id` (code delivered via SMS)
2. Include `otp_id` + `otp_code` in subsequent request
3. OTP locked after 3 failed attempts

---

## Error Reference

| HTTP Code | Meaning |
|-----------|---------|
| 400 | Bad request — missing required field |
| 404 | Resource not found |
| 422 | Validation error — invalid enum value or constraint |
| 429 | Rate limit exceeded |
| 500 | Internal service error |

Service-level errors (`PermissionError`, `ValueError`) are translated to appropriate HTTP codes by the API layer.

---

## NATS Integration

Events published to NATS (subject pattern `gov.usd.<event_type>`):

| Subject | Trigger |
|---------|---------|
| `gov.usd.session.created` | New USSD session started |
| `gov.usd.payment.confirmed` | Payment reference confirmed |
| `gov.usd.sla_breach_detected` | Certificate request exceeds SLA |
| `gov.usd.fraud_risk_scored` | Risk score computed |
| `gov.usd.permit_workflow_approved` | All workflow steps pass |
| `gov.usd.metrics.<tenant_id>` | Telemetry snapshot emitted |

Subscribe with a bytewax pipeline to build real-time dashboards or trigger downstream capabilities.

---

## Copyright

© 2025 Datacraft | www.datacraft.co.ke
