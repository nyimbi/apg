# Payment USSD App (fintech_ussd_pay)

USSD payments: bill pay, merchant payment, airtime top-up, utility pay, send money confirmation.

## Overview

Provides a comprehensive USSD-based payment interface covering East African payment use cases. Supports 9 pre-loaded billers (KPLC, Nairobi Water, KRA, NHIF, NSSF, DStv, Zuku, Safaricom Postpaid) and populates with tenant-specific billers.

Large send-money transactions (>= KES 10,000) require a second-factor confirmation step, matching telco USSD UX conventions.

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/fintech/ussd/pay/health` | Service health check |
| GET | `/api/fintech/ussd/pay/describe` | Capability metadata |
| GET | `/api/fintech/ussd/pay/billers` | List billers |
| POST | `/api/fintech/ussd/pay/billers` | Register biller |
| GET | `/api/fintech/ussd/pay/billers/<id>` | Get biller |
| PUT | `/api/fintech/ussd/pay/billers/<id>` | Update biller |
| DELETE | `/api/fintech/ussd/pay/billers/<id>` | Deactivate biller |
| GET | `/api/fintech/ussd/pay/bills` | List bill payments |
| POST | `/api/fintech/ussd/pay/bills` | Pay a bill |
| GET | `/api/fintech/ussd/pay/bills/<id>` | Get bill payment |
| POST | `/api/fintech/ussd/pay/bills/<id>/reverse` | Reverse bill payment |
| GET | `/api/fintech/ussd/pay/merchants` | List merchant payments |
| POST | `/api/fintech/ussd/pay/merchants` | Pay merchant |
| GET | `/api/fintech/ussd/pay/merchants/<id>` | Get merchant payment |
| GET | `/api/fintech/ussd/pay/airtime` | List airtime top-ups |
| POST | `/api/fintech/ussd/pay/airtime` | Buy airtime |
| GET | `/api/fintech/ussd/pay/airtime/<id>` | Get airtime top-up |
| GET | `/api/fintech/ussd/pay/utilities` | List utility payments |
| POST | `/api/fintech/ussd/pay/utilities` | Pay utility |
| GET | `/api/fintech/ussd/pay/utilities/<id>` | Get utility payment |
| GET | `/api/fintech/ussd/pay/send-money` | List send money transactions |
| POST | `/api/fintech/ussd/pay/send-money` | Initiate send money |
| GET | `/api/fintech/ussd/pay/send-money/<id>` | Get transaction |
| POST | `/api/fintech/ussd/pay/send-money/<id>/confirm` | Confirm transaction |
| POST | `/api/fintech/ussd/pay/send-money/<id>/cancel` | Cancel transaction |
| POST | `/api/fintech/ussd/pay/ussd` | Process USSD request |
| GET | `/api/fintech/ussd/pay/history` | Payment history for phone |
| GET | `/api/fintech/ussd/pay/statistics` | Tenant statistics |
| GET | `/api/fintech/ussd/pay/volume/daily` | Daily volume report |
| GET | `/api/fintech/ussd/pay/search` | Search payments |
| GET | `/api/fintech/ussd/pay/audit-events` | Audit event log |

## Business Rules

- Send money >= KES 10,000 requires explicit confirmation step
- Airtime: KES 5 minimum, KES 10,000 maximum per transaction
- KPLC Prepaid payments generate an electricity token (5-group numeric)
- Bill amounts validated against per-biller min/max limits
- Telcos: safaricom, airtel, telkom, faiba
- Utility codes: kplc_prepaid, kplc_postpaid, nairobi_water, mombasa_water, kisumu_water, nwsc

## World-Class Enhancements (v2.0)

**I1. Scheduled / Recurring Payments** — Cron-driven payment schedules with count/date expiry for standing orders [Feature]

**I2. Favourite / Speed-Dial Payments** — Aliased payment templates reduce 6-step dial to 2 steps for repeat payers [UX]

**I3. Payment Limits & Velocity Controls** — Per-tenant/phone daily, monthly, per-txn, and hourly transaction caps (CBK-compliant) [Risk / Compliance]

**I4. Two-Factor OTP Confirmation** — HMAC-SHA1 TOTP step-up authentication for high-value transactions above a configurable threshold [Security]

**I5. Bulk Payment Disbursement** — Fan-out up to 200 payout records via `asyncio.gather` with per-item status and aggregate receipt [Feature]

**I6. Merchant QR Code Payment** — Generates USSD deep-link QR (Base64 PNG) for pre-filled merchant payment sessions [Feature]

**I7. Transaction Dispute & Chargeback Workflow** — Structured dispute lifecycle (raised → under_review → resolved) with 72-hour CBK compliance [Operations]

**I8. Cashback & Rewards Engine** — Configurable per-tenant cashback rate applied post-payment with unclaimed rewards ledger [Engagement]

**I9. FX / Multi-Currency Bill Pay** — Rate-store driven currency conversion at point of payment for diaspora remittance corridors [Feature]

**I10. USSD Session Timeout & Resume** — Stale session expiry with context restoration for network-dropped sessions [UX / Reliability]

**I11. Payment Notifications via SMS/WhatsApp** — Protocol-based `NotificationAdapter` with pluggable SMS/WhatsApp backends and default logging adapter [UX]

**I12. Biller Account Validation (Pre-payment Lookup)** — Protocol-based pre-validation returning account name and outstanding balance before payment commits [Risk]

**I13. Paybill Split Payment** — Splits a single bill across multiple participants with per-contributor `pay_bill` execution and partial completion tracking [Feature]

**I14. Offline Voucher / Float Management** — Per-agent float accounts with low-water-mark alerts and pre-deduction guards for rural agent networks [Feature]

**I15. Audit Trail Export (CSV / JSON)** — Cross-payment-type chronological export enriched with biller/merchant names for CBK AML/CFT compliance [Compliance]

## New Methods

The three highest-impact async methods added in v2.0:

### `initiate_bulk_disbursement`

Disburse to up to 200 recipients in a single call. All entries are validated before fan-out; failed items do not abort the batch.

```python
svc = PaymentUSSDService(tenant_id="acme")
result = await svc.initiate_bulk_disbursement(
    from_phone="+254700000001",
    pin="1234",
    recipients=[
        {"to_phone": "+254711000001", "amount": 5000, "narration": "Jan stipend"},
        {"to_phone": "+254722000002", "amount": 7500, "narration": "Jan stipend"},
    ],
    tenant_id="acme",
)
# result: {bulk_id, total_amount, success_count, fail_count, items: [...]}
```

### `export_audit_trail`

Produces a CBK-compliant transaction report across all payment types for a date range.

```python
report = await svc.export_audit_trail(
    tenant_id="acme",
    date_from="2026-01-01",
    date_to="2026-01-31",
    fmt="csv",   # or "json"
)
# report: {content: "<csv string>", record_count: 412, exported_at: "2026-02-01T08:00:00Z"}
with open("jan_audit.csv", "w") as f:
    f.write(report["content"])
```

### `expire_stale_sessions` / `resume_ussd_session`

Clean up sessions abandoned mid-flow and restore context for the same phone on re-dial.

```python
# Periodic cleanup (call from a scheduler every 60 seconds)
expired = await svc.expire_stale_sessions(max_age_seconds=180)
# expired: {expired_count: 3, session_ids: [...]}

# On new dial-in, check for resumable context
resumed = await svc.resume_ussd_session(
    phone_number="+254700000001",
    tenant_id="acme",
)
if resumed["found"]:
    # Restore menu state from resumed["context"]
    menu_level = resumed["context"]["level"]
```
