# APG Digital Payments — User Guide

**Version**: 2.0.0 | **Platform**: APG | **Capability ID**: `fintech_payments`

---

## Overview

APG Digital Payments is an Africa-first, globally capable payment processing platform. It unifies M-Pesa (Daraja), MTN MoMo, Airtel Money, Tigo Pesa, card (Visa/Mastercard), SWIFT, EFT/RTGS, and PesaLink into a single, consistent API with full lifecycle management, FX conversion, dispute resolution, and regulatory reporting.

---

## Getting Started

### Quick payment (M-Pesa STK Push)

```bash
curl -X POST https://your-apg-host/api/v1/payments/mpesa/stk-push \
  -H "X-Tenant-ID: my-org" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"phone": "254712345678", "amount": "1000", "reference": "INV-001"}'
```

Response:
```json
{"status": "ok", "data": {"id": "...", "amount": "1000", "currency": "KES", "status": "initiated"}}
```

### Check status

```bash
curl /api/v1/payments/transactions/<id> -H "X-Tenant-ID: my-org"
```

---

## Payment Methods

| Method | Endpoint | Countries | Limits |
|--------|----------|-----------|--------|
| M-Pesa STK Push | `POST /mpesa/stk-push` | KE | KES 1–300,000 |
| M-Pesa B2C | `POST /mpesa/b2c` | KE | KES 1–300,000 |
| M-Pesa B2B | `POST /mpesa/b2b` | KE | KES 1–300,000 |
| MTN MoMo | `POST /mtn-momo/request-to-pay` | UG, GH, CI | UGX 500–2,000,000 |
| Airtel Money | `POST /airtel-money/push` | KE, TZ, UG | KES 1–70,000 |
| Card (Visa/MC) | `POST /card/authorise` | Global | Per 3DS rules |
| SWIFT | `POST /swift/transfer` | Global | Per correspondent |
| EFT/RTGS | `POST /bank/eft` | KE | KES 1–unlimited |
| Batch | `POST /batch` | Multi | Up to 10,000/batch |

---

## Common Workflows

### 1. Collect from a customer (M-Pesa)

1. `POST /mpesa/stk-push` → customer sees prompt on phone
2. Customer enters PIN
3. Daraja sends callback to your `callback_url`
4. `POST /mpesa/callback` (your server forwards Daraja payload)
5. Transaction moves to `completed`

### 2. Pay out to a customer (B2C salary/refund)

1. `POST /mpesa/b2c` with phone and amount
2. Poll `GET /transactions/<id>` or receive webhook
3. Status moves to `completed` when Safaricom confirms

### 3. International wire (SWIFT)

1. `POST /fx/rate?from_currency=USD&to_currency=KES` — confirm rate
2. `POST /fx/convert` — lock in conversion
3. `POST /swift/transfer` — initiate SWIFT with IBAN + BIC
4. Monitor via `GET /transactions/<id>`

### 4. Bulk payroll

1. `POST /batch` with recipients[], amounts[], references[]
2. `POST /batch/<id>/validate` — pre-flight check
3. `POST /batch/<id>/process` — execute
4. `GET /batch/<id>` — monitor progress

### 5. Dispute / chargeback

1. `POST /transactions/<id>/dispute` with reason + evidence
2. Ops team calls `POST /disputes/<id>/investigate`
3. Decision: `POST /disputes/<id>/resolve` with `accept|reject|partial`
4. Accepted disputes auto-trigger refund

---

## Transaction Limits (Kenya CBK)

| KYC Tier | Per Transaction | Daily | Monthly |
|----------|----------------|-------|---------|
| Basic | KES 150,000 | KES 300,000 | KES 3,000,000 |
| Standard | KES 500,000 | KES 1,000,000 | KES 10,000,000 |
| Full KYC | KES 1,000,000 | KES 5,000,000 | KES 50,000,000 |
| Enhanced | Unlimited | Unlimited | Unlimited |

Pass `kyc_tier` in your merchant/customer profile or check via `POST /limits/check`.

---

## Fee Structure

All fees include Kenya 20% excise duty where applicable.

| Method | Base Fee | Excise | Total (example: KES 5,000) |
|--------|----------|--------|---------------------------|
| M-Pesa STK | KES 57 | KES 11.40 | KES 68.40 |
| M-Pesa B2C | KES 57 | KES 11.40 | KES 68.40 |
| Bank EFT | KES 55 | KES 11 | KES 74.16 (incl. VAT) |
| SWIFT (SHA) | USD 15 | — | USD 15 |

Calculate exact fees before payment: `POST /fees/calculate`.

---

## Webhooks

Register an endpoint to receive real-time events:

```bash
curl -X POST /api/v1/payments/webhooks \
  -d '{"event_types": ["payment.completed", "payment.failed"], "url": "https://your-server.com/webhook"}'
```

Events: `payment.initiated`, `payment.completed`, `payment.failed`, `payment.refunded`, `payment.reversed`, `settlement.complete`, `dispute.opened`, `dispute.resolved`

All webhook payloads are HMAC-SHA256 signed with your `secret`. Verify before processing.

---

## Refunds

```bash
# Full refund
curl -X POST /api/v1/payments/transactions/<id>/refund \
  -d '{"reason": "customer_request"}'

# Partial refund
curl -X POST /api/v1/payments/transactions/<id>/refund \
  -d '{"amount": "500", "reason": "partial_return"}'
```

Constraints:
- Refund amount ≤ original amount
- Cumulative refunds ≤ original amount
- Transaction must be in `completed` / `settled` status

---

## Reports

| Report | Endpoint | Use Case |
|--------|----------|----------|
| Volume | `GET /reports/volume` | Daily transaction counts by channel |
| Revenue | `GET /reports/revenue` | Fee income by payment method |
| Failures | `GET /reports/failures` | Failure rate analysis for ops |
| Regulatory | `GET /reports/regulatory` | CBK/CBN/BoU CTR/STR filing |
| Customer Patterns | `GET /reports/customer-patterns` | AML velocity analysis |
| Dispute Analytics | `GET /disputes/analytics` | Chargeback ratio monitoring |

All reports accept `period_from` and `period_to` (YYYY-MM-DD).

---

## Error Codes

| HTTP | Code | Meaning |
|------|------|---------|
| 400 | `bad_request` | Malformed JSON or missing required field |
| 403 | `permission_denied` | Cross-tenant access or policy violation |
| 422 | `validation_error` | Business rule violation (e.g. limit exceeded) |
| 422 | `missing_field` | Required field absent from body |
| 500 | `internal_error` | Unexpected server error — retry with backoff |

---

## Idempotency

Always include `idempotency_key` for payment initiation:

```json
{"phone": "254712345678", "amount": "1000", "reference": "INV-001", "idempotency_key": "inv-001-attempt-1"}
```

Duplicate requests with the same key return the original result without re-processing. Key TTL is 24 hours.
