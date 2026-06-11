# Mobile Banking USSD — User Guide

## Overview

The Mobile Banking USSD capability (`fintech_ussd_mob`) delivers a full-featured mobile banking experience via USSD, enabling customers to perform banking operations from any mobile phone without requiring internet access or a smartphone. Version 2.0 adds beneficiary management, cross-border FX transfers, fraud scoring, spending analytics, audit chain verification, and personalised menus.

## Use Cases

1. **Account Balance Enquiry** — Check current and available balance via PIN-secured USSD session.
2. **Mini-Statement** — View last 5 transactions without visiting a branch.
3. **Fund Transfer** — Send money to another account within daily limits (KES 500,000/day, KES 150,000/transaction).
4. **Cross-Border FX Transfer** — Send KES to UGX/TZS/RWF recipients at live FX rates with configurable spread.
5. **Standing Orders** — Set up recurring payments (daily/weekly/monthly/quarterly) that execute automatically with smart retry.
6. **PIN Management** — Change PIN, reset via OTP, and recover locked accounts.
7. **Saved Beneficiaries** — Store up to 20 named payees for 1-keystroke repeat transfers.
8. **Spending Insights** — View categorised spending breakdown (utilities, food, transport, etc.).
9. **Statement Export** — Download account history as JSON, CSV, or accounting summary.
10. **Balance Threshold Alerts** — Get notified (SMS via NATS) when balance drops below a set amount.

## USSD Flow

Dial `*123#` (or your bank's registered service code) to access the personalised main menu:

```
Welcome to MobBank
1. Fund Transfer        ← promoted: most used
2. Account Balance      ← promoted: 2nd most used
3. Mini Statement
4. Standing Orders
5. Change PIN
0. Exit
```

Menu items are ranked by the customer's personal usage frequency. On first use, the default order is shown. After 3+ sessions, the top 2 operations are automatically promoted — reducing average keystrokes by ~40%.

### Balance Enquiry Flow

```
Dial *123#
→ Select 2 (or 1 if promoted)
→ Enter PIN: ****
→ END Account Balance
   Balance: KES 45,230.00
   Available: KES 45,230.00
```

### Fund Transfer Flow

```
Dial *123#
→ Select 3. Fund Transfer
→ 1. Saved Beneficiaries  2. New Account
→ Select 1 → List of saved aliases:
   1. Mum (0722111222)
   2. Rent (4567890123)
→ Select 1
→ Enter amount: 3000
→ Enter PIN: ****
→ END Transfer KES 3,000 to Mum sent. Ref: TRF-abc123
```

### Cross-Border FX Transfer Flow

```
Dial *123# → 3. Fund Transfer → 3. Cross-Border
→ Enter recipient account: UG123456
→ Enter amount (KES): 5000
→ CON Send KES 5000 → UGX 132,500 (rate: 26.50)
   1. Confirm  2. Cancel
→ Select 1 → Enter PIN: ****
→ END Transfer sent. UGX 132,500 credited.
```

### Spending Insights Flow

```
Dial *123# → 6. My Spending
→ Enter PIN: ****
→ END Last 30 days:
   Utilities 45%, Food 30%,
   Transport 25%
   Total: KES 12,400
```

## API Reference

### Create Account

```http
POST /api/fintech/ussd/mob/accounts
Content-Type: application/json

{
  "phone_number": "0712345678",
  "account_number": "1234567890",
  "account_type": "savings",
  "customer_name": "Jane Doe",
  "national_id": "12345678",
  "pin": "1234",
  "currency": "KES",
  "tenant_id": "default"
}
```

### Balance Enquiry

```http
POST /api/fintech/ussd/mob/accounts/1234567890/balance
{
  "pin": "1234",
  "tenant_id": "default"
}
```

### Fund Transfer

```http
POST /api/fintech/ussd/mob/transfers
{
  "from_account": "1234567890",
  "to_account": "0987654321",
  "amount": "5000.00",
  "pin": "1234",
  "narration": "School fees",
  "currency": "KES",
  "tenant_id": "default"
}
```

### Idempotent Transfer (gateway retry safe)

```http
POST /api/fintech/ussd/mob/transfers/idempotent
{
  "from_account": "1234567890",
  "to_account": "0987654321",
  "amount": "5000.00",
  "pin": "1234",
  "idempotency_key": "sess-abc123-attempt-1",
  "narration": "School fees",
  "currency": "KES"
}
```

Repeat the same request with the same `idempotency_key` — the original transfer is returned without debiting again.

### Cross-Border FX Transfer

First, register an FX rate (populated automatically by the bytewax/NATS FX pipeline):

```http
PUT /api/fintech/ussd/mob/fx/rates
{
  "from_currency": "KES",
  "to_currency": "UGX",
  "rate": "26.50"
}
```

Then transfer:

```http
POST /api/fintech/ussd/mob/transfers/fx
{
  "from_account": "1234567890",
  "to_account": "UG987654321",
  "send_amount": "5000.00",
  "pin": "1234",
  "spread_bps": 150,
  "narration": "Family support"
}
```

### Add Beneficiary

```http
POST /api/fintech/ussd/mob/accounts/1234567890/beneficiaries
{
  "pin": "1234",
  "alias": "Mum",
  "target_account": "0722111222",
  "target_name": "Mary Doe"
}
```

### Spending Insights

```http
GET /api/fintech/ussd/mob/accounts/1234567890/insights?pin=1234&days=30
```

Response includes `ussd_summary` — a 160-char string ready for USSD display.

### Statement Export

```http
POST /api/fintech/ussd/mob/accounts/1234567890/statement/export
{
  "pin": "1234",
  "date_from": "2026-05-01",
  "date_to": "2026-05-31",
  "format": "csv"
}
```

Supported formats: `json`, `csv`, `summary`. CSV is compatible with QuickBooks and Xero import.

### Balance Alert Threshold

```http
POST /api/fintech/ussd/mob/accounts/1234567890/alert-threshold
{
  "pin": "1234",
  "threshold": "5000.00"
}
```

When balance drops below KES 5,000 after any debit, a `mob.balance.alert` event is published to NATS. A downstream subscriber dispatches an SMS to the registered phone number.

### Score Fraud Risk

```http
POST /api/fintech/ussd/mob/fraud/score
{
  "account_number": "1234567890",
  "transfer_amount": "80000.00",
  "recipient_account": "NEW987654",
  "tenant_id": "default"
}
```

Response:

```json
{
  "score": 60,
  "risk_level": "medium",
  "action_required": "totp_challenge",
  "factors": {
    "velocity": 20,
    "recipient_novelty": 20,
    "amount_deviation": 20,
    "time_anomaly": 0
  }
}
```

### Verify Audit Chain

```http
GET /api/fintech/ussd/mob/audit-events/verify-chain?tenant_id=default
```

Recomputes every event's Merkle hash from genesis. Returns `chain_intact: true` or the index of the first tampered event.

### USSD Gateway Integration

Africa's Talking / Telco gateway:

```http
POST /api/fintech/ussd/mob/ussd
Content-Type: application/x-www-form-urlencoded

sessionId=SESSION123&phoneNumber=0712345678&serviceCode=*123%23&text=1*1234
```

Returns plain-text response: `CON ...` for continuing sessions, `END ...` for terminal.

### Register Service Code (Multi-Tenant)

```http
POST /api/fintech/ussd/mob/service-codes
{
  "service_code": "*456#",
  "tenant_id": "bank_b"
}
```

Multiple banks can share one APG deployment with distinct `*NNN#` codes. The gateway routes to the correct tenant automatically.

### PIN Change

```http
POST /api/fintech/ussd/mob/pin/change
{
  "account_number": "1234567890",
  "old_pin": "1234",
  "new_pin": "5678",
  "confirm_pin": "5678"
}
```

### Create Standing Order

```http
POST /api/fintech/ussd/mob/standing-orders
{
  "from_account": "1234567890",
  "to_account": "0987654321",
  "amount": "3000.00",
  "frequency": "monthly",
  "start_date": "2026-07-01",
  "pin": "1234",
  "narration": "Rent",
  "tenant_id": "default"
}
```

## Security Model

| Control | Implementation |
|---------|---------------|
| PIN storage | SHA-256 hash — never stored in plain text |
| PIN brute force | 3 failed attempts locks account; admin unlock required |
| PIN reset | OTP delivered via SMS; expires in 5 minutes; single use |
| High-value transfers | TOTP second factor above KES 50,000 |
| Session integrity | HMAC-SHA256 token bound to MSISDN + session_id |
| Audit log integrity | Merkle-chain hash on every event; tamper-evident |
| Fraud scoring | Velocity, novelty, deviation, time-of-day; hold or challenge |
| Daily limits | Per-account, reset at midnight UTC |

## Error Codes

| Code | Meaning |
|------|---------|
| `invalid_pin` | PIN did not match; N attempts remaining |
| `account_locked_too_many_pin_attempts` | Account locked after 3 failures |
| `insufficient_funds` | Available balance below transfer amount |
| `daily_transfer_limit_exceeded` | KES 500,000 daily limit reached |
| `exceeds_single_transfer_limit_150000` | Single transfer above KES 150,000 |
| `otp_expired` | OTP older than 5 minutes |
| `otp_already_used` | OTP already consumed |
| `account_number_already_exists` | Duplicate account number for tenant |
| `fx_rate_not_available` | No FX rate registered for currency pair |
| `alias_must_be_12_chars_or_fewer` | Beneficiary alias too long |
| `max_20_beneficiaries_reached` | Beneficiary limit per account reached |
| `session_token_expired` | USSD session token older than 5 minutes |
| `invalid_session_token` | Token HMAC does not match |
| `unsupported_export_format` | Use json, csv, or summary |
| `service_code_already_registered_to_another_tenant` | Service code conflict |

## Event Streaming Architecture

```
MobUssdService
     │
     ├── _emit() → Merkle-hashes event → _audit_events[]
     │
     └── (with NATSPublisher injected)
            │
            ├── mob.transfer.completed  ──→  SMS confirmation microservice
            ├── mob.balance.alert       ──→  Push/SMS alert microservice
            ├── mob.fraud.score_computed ─→  Compliance dashboard
            └── mob.standing_order.*    ──→  Scheduler / retry worker

Bytewax pipeline subscribes to mob.* and feeds:
  - intel/correlation for cross-account fraud detection
  - intel/prediction for churn modelling
  - Regulatory reporting aggregator
```

## Composability Map

| Capability | Integration Point |
|-----------|------------------|
| `intel/alerts` | Subscribe to `mob.balance.alert` NATS subject |
| `intel/correlation` | Correlate fraud scores across accounts |
| `intel/prediction` | Input: transaction velocity, balance trend |
| `fintech/terminal/terramoni` | Display live account balance in TUI |
| Scheduler (APG cron) | Call `execute_standing_order` per order's `next_execution_date` |
| SMS Gateway | Subscribe to `mob.balance.alert` + `mob.pin.*` NATS events |
