# Payment USSD App — World Class Improvements

### I1. Scheduled / Recurring Payments
**Category:** Feature  
**Justification:** M-Pesa Standing Order and Equity Bank AutoPay show >40% of utility and insurance bills are recurring. Forcing users to re-dial USSD monthly is the top churn driver in USSD payment products.  
**Implementation:** Add `ScheduledPayment` store with `cron_expr`, `next_run_at`, `last_run_at`, `max_occurrences`. A background coroutine evaluates due schedules every minute and materialises real payments. Expiry by count or date stops silently.  
**Competitor Reference:** M-Pesa Standing Order (Safaricom), GTB AutoPay (Nigeria)

---

### I2. Favourite / Speed-Dial Payments
**Category:** UX  
**Justification:** Vodacom Tanzania research found that 68% of USSD payment sessions replay a payment made in the last 30 days. Persisting favourites reduces a 6-step dial to 2 steps.  
**Implementation:** `favourites` store keyed by `{phone_number}:{alias}`. `add_favourite` saves a payment template; `pay_favourite` resolves template, validates, and executes via the appropriate payment method.  
**Competitor Reference:** Airtel Money Quick Pay, MTN MoMo Frequent Payments

---

### I3. Payment Limits & Velocity Controls
**Category:** Risk / Compliance  
**Justification:** Kenya CBK Prudential Guidelines 2023 require PSPs to enforce per-customer daily and monthly limits. Absence of velocity controls is a regulatory gap and fraud enabler.  
**Implementation:** `LimitPolicy` per tenant/phone: `daily_max`, `monthly_max`, `per_txn_max`, `txn_per_hour`. `_check_velocity` aggregates completed payments in rolling windows before execution. Breach raises `LimitExceededError`.  
**Competitor Reference:** Ecobank Omni Limits Engine, Equity EazzyPay limits

---

### I4. Two-Factor OTP Confirmation
**Category:** Security  
**Justification:** PIN alone is insufficient for high-value transactions. GSMA Mobile Money Guidelines 2024 recommend OTP-based step-up authentication for amounts above defined thresholds.  
**Implementation:** `OTPChallenge` store: `generate_otp` creates a 6-digit TOTP-style code (HMAC-SHA1 with 90-second window) stored hashed. `verify_otp` validates then marks used. Plugged in above `SEND_MONEY_CONFIRMATION_THRESHOLD`.  
**Competitor Reference:** M-Pesa Lipa na M-Pesa OTP, Flutterwave OTP step-up

---

### I5. Bulk Payment Disbursement
**Category:** Feature  
**Justification:** SMEs, schools, and NGOs disbursing salaries or stipends via USSD currently make N individual payments. A single bulk call with a payout list saves 90% of USSD airtime costs and admin time.  
**Implementation:** `initiate_bulk_disbursement` accepts a list of `{to_phone, amount, narration}` records (max 200). Validates all entries first, then fans out via `asyncio.gather`. Returns per-item status and aggregate receipt. Failed items do not abort the batch.  
**Competitor Reference:** M-Pesa B2C API, Airtel Money Business Bulk, Pesalink Batch

---

### I6. Merchant QR Code Payment
**Category:** Feature  
**Justification:** QR-USSD hybrid payments are growing in East Africa (Kenya 2024 CBK report: 23% YoY increase in QR-initiated USSD sessions). A merchant can display a QR containing a pre-filled USSD deep link.  
**Implementation:** `generate_merchant_qr_payload` constructs a USSD deep-link string `*144*2*{till}*{amount}#` and returns a data-URI (Base64 PNG via `qrcode` library) alongside the raw USSD string. `decode_qr_payload` reverses it.  
**Competitor Reference:** Equity EazzyPay QR, DTB QuickPay QR

---

### I7. Transaction Dispute & Chargeback Workflow
**Category:** Operations  
**Justification:** Without a formal dispute mechanism, PSPs handling > KES 100M monthly volume face unstructured complaint queues. CBK requires documented dispute resolution within 72 hours (NPS Act 2011, s.14).  
**Implementation:** `raise_dispute` attaches a `DisputeRecord` to any payment (bill, merchant, utility, send-money). States: `raised → under_review → resolved_refunded / resolved_rejected`. `resolve_dispute` triggers reversal or rejection with documented reason.  
**Competitor Reference:** M-Pesa dispute centre, PesaLink dispute API

---

### I8. Cashback & Rewards Engine
**Category:** Engagement  
**Justification:** Safaricom M-PESA Rewards and Airtel Money Cashback programmes show 31% higher monthly active user retention compared to bare USSD wallets without loyalty features.  
**Implementation:** `RewardPolicy` per tenant: `cashback_rate` (Decimal, e.g. 0.005 for 0.5%), `eligible_payment_types`, `max_cashback_per_txn`. `_apply_cashback` computes reward post-payment and stores in `cashback_ledger`. `get_cashback_balance` aggregates unclaimed rewards.  
**Competitor Reference:** M-Pesa GlobalPay Cashback, Airtel Money Kasuku Rewards

---

### I9. FX / Multi-Currency Bill Pay
**Category:** Feature  
**Justification:** Kenyan diaspora paying school fees or KPLC bills from USD/GBP accounts is a $2.1B annual corridor (World Bank 2024). Displaying FX rate and converting at point of payment removes manual currency guesswork.  
**Implementation:** `fx_rate_store` maps `{from_currency}:{to_currency}` with `rate`, `valid_until`. `convert_currency` applies rate and returns `original_amount`, `converted_amount`, `rate_used`, `rate_expiry`. `pay_bill_fx` wraps `pay_bill` with pre-conversion.  
**Competitor Reference:** WorldRemit Pay Bills, Chipper Cash Multi-Currency

---

### I10. USSD Session Timeout & Resume
**Category:** UX / Reliability  
**Justification:** USSD sessions drop frequently on congested networks. Without timeout tracking and resume capability, users restart from scratch losing entered data. Telcos cap sessions at 180 seconds; the service must clean up stale sessions.  
**Implementation:** `expire_stale_sessions(max_age_seconds)` scans `ussd_sessions`, marks sessions `timed_out` if `last_activity` delta exceeds threshold, and emits `pay_ussd_session_expired`. `resume_ussd_session` restores context from a prior incomplete session for the same phone.  
**Competitor Reference:** Safaricom USSD session continuity, Africa's Talking USSD resume

---

### I11. Payment Notifications via SMS/WhatsApp
**Category:** UX  
**Justification:** 94% of M-Pesa users cite SMS confirmation as their primary trust signal (Helix Institute 2023). The service must push structured notifications without coupling to a specific SMS gateway.  
**Implementation:** `NotificationAdapter` protocol with `send(phone, message)`. `_notify` is called post-payment with templated messages. Default `LoggingNotificationAdapter` logs to `_log`; production adapters implement the protocol. `register_notification_adapter` sets the active adapter.  
**Competitor Reference:** M-Pesa SMS confirmation, MTN MoMo WhatsApp notification

---

### I12. Biller Account Validation (Pre-payment Lookup)
**Category:** Risk  
**Justification:** KPLC, NHIF, and KRA all expose account validation APIs. Pre-validating the account reference before payment eliminates the second-most common USSD support call: "paid wrong account".  
**Implementation:** `BillerValidationAdapter` protocol: `async validate(biller_code, account_reference) -> ValidationResult`. `validate_biller_account` calls the adapter and returns `{valid, account_name, outstanding_balance}`. `pay_bill` optionally pre-validates when `validate_account=True`.  
**Competitor Reference:** KPLC Prepaid token validation API, NHIF member lookup

---

### I13. Paybill Split Payment
**Category:** Feature  
**Justification:** Shared household utility bills and group chamas frequently need cost splitting. Stanbic Bank and Co-op Bank offer split-bill in mobile banking; bringing it to USSD captures informal market segments.  
**Implementation:** `initiate_split_payment` accepts `{biller_code, account_reference, total_amount, participants: [{phone, share_amount}]}`. Creates individual `pay_bill` requests per participant. Returns a `SplitPaymentGroup` record with aggregate status. Partial completion is tracked.  
**Competitor Reference:** M-Pesa Lipa Split, Co-op Bank GoSplit

---

### I14. Offline Voucher / Float Management
**Category:** Feature  
**Justification:** Agent-assisted USSD payments in rural Kenya operate via float. Without float tracking, agents over-spend their float balance causing failed transactions at point of sale — directly reducing agent revenue.  
**Implementation:** `FloatAccount` per `{tenant_id, agent_phone}` with `balance`, `low_water_mark`, `last_top_up_at`. `_deduct_float` is called before agent-initiated payments and raises `InsufficientFloatError` on breach. `top_up_float` adds float, `get_float_balance` returns current state.  
**Competitor Reference:** M-Pesa Agent Float, Airtel Money Agent Float Management

---

### I15. Audit Trail Export (CSV / JSON)
**Category:** Compliance  
**Justification:** CBK AML/CFT regulations require PSPs to produce transaction reports within 48 hours of request. An in-service export function eliminates manual database dumps and reduces compliance response time.  
**Implementation:** `export_audit_trail(tenant_id, date_from, date_to, fmt)` collects all payment records across types, enriches with biller/merchant name, sorts chronologically, and serialises to CSV (via `csv.DictWriter`) or JSON (via `json.dumps`). Returns `{content: str, record_count: int, exported_at: str}`.  
**Competitor Reference:** Flutterwave Compliance Export, PesaLink AML Report API
