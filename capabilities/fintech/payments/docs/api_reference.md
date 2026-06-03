# APG Digital Payments — API Reference

Base URL: `/api/v1/payments`
Auth: `Authorization: Bearer <token>` or `X-Tenant-ID: <tenant>`

All responses: `{"status": "ok"|"error", "data": {...}}` or `{"status": "error", "error": {"code": "...", "message": "..."}}`

---

## Health

### GET /health
Liveness probe. No auth required.
**200**: `{"data": {"capability": "fintech_payments", "status": "healthy"}}`

---

## Payment Initiation

### POST /initiate
Initiate a payment using any method.

**Body**:
```json
{
  "method": "mpesa_stk",
  "amount": "1000",
  "currency": "KES",
  "recipient": "254712345678",
  "reference": "INV-001",
  "narration": "Invoice payment",
  "idempotency_key": "inv-001-v1",
  "metadata": {}
}
```
**201**: Transaction record. **422**: Rule violation.

---

## M-Pesa

### POST /mpesa/stk-push
STK Push (Lipa na M-Pesa). Customer receives PIN prompt.

**Body**: `phone` (E.164 254...), `amount`, `reference` (≤12 chars), `description`, `callback_url`

### POST /mpesa/b2c
Business-to-Customer payout (salary, refund, winnings).

**Body**: `phone`, `amount`, `occasion`, `remarks`

### POST /mpesa/b2b
Business-to-Business transfer (paybill to paybill).

**Body**: `business_short_code`, `amount`, `account_reference`, `remarks`

### POST /mpesa/callback
Receive and process Daraja callback. Forward raw Daraja payload.

---

## MTN MoMo

### POST /mtn-momo/request-to-pay
Request payment from MTN MoMo customer.

**Body**: `phone`, `amount`, `currency` (UGX/GHS), `external_id`, `narration`

---

## Airtel Money

### POST /airtel-money/push
Push payment request to Airtel Money customer.

**Body**: `phone`, `amount`, `currency`, `reference`, `narration`

---

## Card

### POST /card/authorise
Authorise card payment. **Token only — never raw PAN** (PCI-DSS).

**Body**: `card_token`, `amount`, `currency`, `merchant_id`, `three_ds_result` (required >KES 10k), `cvv_result`, `avs_result`

### POST /card/{txn_id}/capture
Capture authorised amount. Body: `amount` (optional, defaults to full auth amount).

### POST /card/{txn_id}/void
Void authorisation before capture.

---

## SWIFT / Bank Transfers

### POST /swift/transfer
Cross-border SWIFT wire.

**Body**: `sender_bic` (8/11 chars), `receiver_bic`, `iban`, `amount`, `currency`, `purpose_code` (3 chars), `charges` (SHA/OUR/BEN)

### POST /bank/eft
Domestic EFT / RTGS / PesaLink.

**Body**: `from_account`, `to_account`, `bank_code`, `amount`, `currency`, `reference`, `narration`, `clearing_type` (eft/rtgs/pesalink)

---

## Batch Payments

### POST /batch
Create bulk payment batch.

**Body**: `payment_date` (YYYY-MM-DD), `method`, `currency`, `recipients[]`, `amounts[]`, `references[]`

**201**: `{"id": "...", "status": "queued", "total_amount": "..."}`

### POST /batch/{id}/validate
Pre-flight validation. Returns validation errors before processing.

### POST /batch/{id}/process
Execute validated batch. Processes async; poll `GET /batch/{id}` for status.

### GET /batch/{id}
Get batch status: `queued → processing → completed/partial_failure`

---

## FX

### GET /fx/rate?from_currency=USD&to_currency=KES
Current indicative FX rate.

### POST /fx/convert
**Body**: `from_currency`, `to_currency`, `amount`
**201**: `{"from_amount": "100", "to_amount": "12788", "rate": "127.88", "spread_bps": 150}`

### GET /fx/report?period_from=&period_to=
FX gain/loss report for period.

---

## Transactions

### GET /transactions?status=&method=&date_from=&date_to=&offset=0&limit=50
List with filters. Max limit 1000.

### GET /transactions/{id}
Single transaction detail.

### POST /transactions/{id}/confirm
Confirm pending transaction. Body: `provider_ref`

### POST /transactions/{id}/cancel
Cancel/expire pending transaction.

### POST /transactions/{id}/refund
**Body**: `amount` (optional — full refund if omitted), `reason`

### POST /transactions/{id}/reverse
Wrong-number reversal. Body: `reason`. 24-hour window from creation.

### POST /transactions/{id}/dispute
**Body**: `reason`, `evidence_description`

---

## Refunds

### GET /refunds/{id}
Refund status.

### POST /refunds/{id}/approve
Approve pending refund (ops team).

---

## Disputes & Chargebacks

### POST /disputes/{id}/investigate
**Body**: `investigation_notes`

### POST /disputes/{id}/resolve
**Body**: `decision` (accept|reject|partial), `chargeback_amount`, `decision_reason`

### GET /disputes/analytics?period_from=&period_to=
Dispute rate, resolution time, chargeback ratio.

---

## Settlement

### POST /settlement/run
**Body**: `settlement_date` (YYYY-MM-DD), `bank_account`

### POST /settlement/{id}/reconcile
**Body**: `actual_amounts[]` (optional)

---

## Merchants

### POST /merchants
**Body**: `name`, `category_code` (4-digit MCC), `settlement_account`, `paybill_number`, `till_number`

### GET /merchants/{id}/report?period_from=&period_to=
Merchant settlement and volume report.

---

## Virtual Accounts

### POST /virtual-accounts
**Body**: `owner_id`, `currency`

### POST /virtual-accounts/{id}/credit
**Body**: `amount`, `reference`

---

## Webhooks

### POST /webhooks
**Body**: `event_types[]`, `url` (HTTPS required), `secret`

### POST /webhooks/{id}/test
Fire test event to webhook endpoint.

---

## Fees

### POST /fees/calculate
No state change — returns fee breakdown.
**Body**: `method`, `amount`, `currency`
**200**: `{"fee_amount": "57", "excise_tax": "11.40", "total_charge": "68.40"}`

---

## Limits

### POST /limits/check
**Body**: `customer_id`, `amount`, `currency`, `kyc_tier`

---

## Reports

### GET /reports/volume?period_from=&period_to=
Transaction counts and volumes by channel and day.

### GET /reports/revenue?period_from=&period_to=
Fee revenue by payment method.

### GET /reports/failures?period_from=&period_to=
Failure rate analysis by method.

### GET /reports/regulatory?period_from=&period_to=&regulator=cbk
CBK/CBN/BoU CTR/STR report. Regulator: `cbk` (Kenya), `cbn` (Nigeria), `bou` (Uganda).

### GET /reports/customer-patterns?customer_id=
Customer payment pattern and AML velocity analysis.

---

## Dashboard

### GET /dashboard
Payment operations KPIs: volume, revenue, failures, disputes, settlement status.
