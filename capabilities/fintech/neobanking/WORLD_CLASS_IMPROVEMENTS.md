# Digital Neobanking — World-Class Improvements

15 high-impact improvements to elevate this capability from functional to production-grade.

---

## 1. Real-Time Fraud Signal Integration at Transaction Posting

**Problem**: `post_transaction` only validates a `risk_reference` string — it never calls the fraud engine.
**Improvement**: Add async hook to `fintech_fraud` at transaction time; block or flag transactions that breach configurable velocity, geo-anomaly, or device-fingerprint thresholds. Surface a `fraud_signal` field on every transaction response so downstream consumers can gate on it without a separate lookup.

---

## 2. Multi-Currency Account Support with FX Conversion

**Problem**: Transfers require identical currencies; cross-currency moves are silently blocked.
**Improvement**: Add `fx_convert_and_transfer()` that calls a `fintech_fx` adapter, applies the mid-market rate + configurable spread, posts the FX fee as a separate transaction, and settles both legs atomically. Store `fx_rate`, `fx_fee`, and `original_currency` on the credit leg for audit.

---

## 3. Savings Pot Auto-Sweep Rules

**Problem**: Savings pots are purely manual — users must call `savings_pot_deposit` explicitly.
**Improvement**: Add `savings_pot_autosweep_rule()` that attaches a trigger (end-of-day, percentage-of-balance, after-salary-credit) to a pot. A scheduled `execute_autosweep_rules()` method evaluates each rule, executes the deposit, and emits a `savings_autosweep_executed` event. Enables zero-effort savings habits.

---

## 4. Tiered Overdraft with Daily Interest Accrual

**Problem**: Overdraft is a flat limit with a hardcoded 18 % rate and no accrual logic.
**Improvement**: Add `overdraft_interest_accrual()` that calculates the overdrawn balance each EOD, applies the daily rate (annual / 365), posts an `overdraft_fee` transaction, and tracks a running `overdraft_balance` ledger. Add configurable tier thresholds (micro / standard / extended) with different rates.

---

## 5. Programmable Virtual Cards with Per-MCC Controls

**Problem**: Virtual cards have only a `spending_limit`; no category or merchant controls.
**Improvement**: Add `virtual_card_update_controls()` to set per-MCC allow/block lists, per-country allow/block lists, and time-of-day windows. Enforce controls in `post_transaction` when the transaction has a `card_id` attached. Enables "school fees only" or "no gambling" card constraints.

---

## 6. Account-Level Event Webhooks

**Problem**: Notifications fire via an opaque `_notify` adapter with no tenant-configurable routing.
**Improvement**: Add `register_account_webhook()` that stores a URL + secret + event filter per account. `_maybe_notify` fans out to registered webhooks using HMAC-SHA256 signed payloads. Include a `webhook_delivery_log` and `replay_webhook()` for failed deliveries. Standard pattern for embedded finance partners.

---

## 7. Intelligent Spending Insights with Budget Enforcement

**Problem**: `spending_analytics` is a passive report; it does not advise or gate.
**Improvement**: Add `set_spending_budget()` to record category budgets and `spending_budget_check()` to return remaining budget, burn rate, and a projected over-budget date. Emit `budget_warning` notifications at 75 % and 100 % consumption. Reduces customer churn through proactive engagement.

---

## 8. Regulatory Reporting Pipeline (CBK, RBA, CMA)

**Problem**: `cbk_neobanking_return` is a stub that counts records without any aggregations.
**Improvement**: Build a full `regulatory_report()` with pluggable jurisdiction templates (CBK, RBA, CMA). Each template defines required fields, validation rules, and submission format. Add `submit_regulatory_report()` that archives the report, updates submission status, and records the acknowledgment reference.

---

## 9. Idempotency Keys on All Mutating Operations

**Problem**: No idempotency layer — double-submitted transfers are posted twice.
**Improvement**: Add an `idempotency_store: dict[str, dict]` and decorate all write methods with an idempotency check: if the key has been seen, return the cached response. Keys expire after a configurable TTL (default: 24 h). Prevents duplicate charges in mobile retry scenarios.

---

## 10. Account Verification (Proof of Funds / Balance Attestation)

**Problem**: No way to prove account ownership or balance to third parties without full data export.
**Improvement**: Add `generate_balance_attestation()` that produces a signed JSON structure containing tenant, account, balance, and timestamp, signed with a tenant key from `keym`. Add `verify_balance_attestation()` for counterparty verification. Enables open banking consent flows without full PSD2 overhead.

---

## 11. Transaction Dispute and Chargeback Workflow

**Problem**: Service cases handle disputes generically; no structured chargeback lifecycle.
**Improvement**: Add `open_chargeback()` that creates a structured dispute record with provisional credit to the customer, sets a `chargeback_hold` on the merchant leg, and tracks dispute status through `merchant_response`, `arbitration`, and `final_ruling` states. Automates the Visa/Mastercard chargeback timeline.

---

## 12. Customer Risk Score Aggregation

**Problem**: Fraud and AML references are stored but never aggregated into a usable risk profile.
**Improvement**: Add `compute_customer_risk_score()` that pulls signals from transaction velocity, geographic diversity, savings behaviour, overdraft utilisation, and KYC freshness into a single 0–100 risk score. Score drives account limits, overdraft eligibility, and alert thresholds automatically.

---

## 13. Batch Statement Generation with PDF Export Hook

**Problem**: Statements are generated one at a time with no bulk support or format options.
**Improvement**: Add `bulk_issue_statements()` that generates statements for all accounts in a date range in a single call. Add a `statement_format` parameter (`json`, `pdf`, `csv`) that calls the appropriate export adapter. Queue generation via a background job when `item_count > 500`.

---

## 14. Account Linking (Joint & Subsidiary Accounts)

**Problem**: Joint and subsidiary account types are listed as supported but no linking logic exists.
**Improvement**: Add `link_accounts()` to model parent/child or joint relationships. Joint accounts require consent from all linked customers; subsidiary accounts inherit limits from the parent. Add `get_linked_accounts()` and propagate freeze/close actions to linked accounts.

---

## 15. Passwordless Biometric Consent Capture

**Problem**: `consent_reference` is a free string — there is no structured consent lifecycle.
**Improvement**: Add `record_consent()` to capture consent events with type (`account_opening`, `data_sharing`, `marketing`, `overdraft`), channel (`sms_otp`, `biometric`, `e-signature`), and evidence hash. Add `revoke_consent()` and `list_consent_history()`. Satisfies Kenya Data Protection Act Article 30 requirements.
