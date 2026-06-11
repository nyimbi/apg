# Buy Now Pay Later — World-Class Improvements

**Capability**: `fintech_bnpl` | **Author**: Nyimbi Odero | **Date**: 2026-06-11

---

## 1. Dynamic Risk-Based Pricing Engine

Replace the static 18%/24% annual rate with a real-time pricing model that accepts credit score, plan type, merchant category, and consumer tenure as inputs. Output a personalised APR with tiered pricing bands (prime, near-prime, subprime). This alone drives approval-rate optimisation without loosening underwriting thresholds.

**Impact**: +12–18% revenue per approved plan; fewer subprime write-offs.

---

## 2. Adaptive Credit Limit Reassessment

Currently credit limits are set once via `set_credit_limit`. Add `async reassess_credit_limit(customer_id)` that reruns the scoring model on up-to-date repayment history, adjusts the limit up or down, and emits a `credit_limit_reassessed` event. Schedule this on a 30-day cadence via the cron adapter.

**Impact**: Reduces charge-offs on degraded consumers; unlocks higher limits for good payers automatically.

---

## 3. Instalment Restructuring / Hardship Programme

Add `async restructure_plan(plan_id, new_term_days, hardship_reason)` that extends the repayment term, recalculates instalments, waives accrued late fees up to a policy cap, and records a hardship evidence object. Requires reviewer approval above a principal threshold.

**Impact**: Reduces default-to-write-off pipeline; improves regulatory standing with CBK.

---

## 4. Multi-Currency FX Settlement

`record_merchant_settlement` assumes a single currency. Add a cross-currency settlement path: `async cross_currency_settlement(merchant_id, period, settlement_currency)` that fetches an FX rate from a pluggable rate provider, converts GMV, deducts MDR in the merchant's home currency, and records both gross (consumer currency) and net (settlement currency) figures.

**Impact**: Enables pan-African merchant expansion without per-country currency code hacks.

---

## 5. Webhooks & Event Delivery SLA Tracking

Add `async register_merchant_webhook(merchant_id, url, events)` and `async deliver_webhook_event(event)` backed by an at-least-once delivery queue with exponential backoff. Track per-webhook delivery latency and SLA breach rate. Expose `async webhook_health(merchant_id)` for merchant dashboards.

**Impact**: Tier-1 BNPL providers (Afterpay, Klarna) all offer webhook parity — this closes that gap.

---

## 6. Consumer Behavioural Scoring (Non-Credit Bureau)

Augment `credit_check_bnpl` with a thin behavioural model: purchase frequency, average basket size, days-to-pay distribution, and channel diversity. Score 0–100. Blend 30% into the credit score proxy. Expose `async consumer_behaviour_score(customer_id)` for CRM integrations.

**Impact**: Significantly improves thin-file / no-bureau consumer approvals in emerging markets.

---

## 7. Merchant Category-Aware Fraud Velocity Controls

Add `async check_fraud_velocity(consumer_id, merchant_id, amount)` that counts checkout attempts in rolling 1h/24h windows, flags category anomalies (e.g., a grocery consumer suddenly buying electronics at 2 AM), and returns a velocity risk level (`low`, `medium`, `high`, `block`). Plugs into `create_checkout_session` as a pre-filter.

**Impact**: Reduces synthetic-identity and account-takeover fraud at the checkout entry point.

---

## 8. Instalment Deferral (Skip-a-Payment)

Add `async defer_instalment(plan_id, instalment_sequence, deferral_fee)` allowing consumers to push one instalment to the end of the schedule (subject to a policy-capped fee). Enforces a max-deferrals-per-plan rule, recalculates the trailing schedule, and records the deferral as an evidence object.

**Impact**: High-engagement feature; Paidy and Zip both offer this. Reduces self-cure chargebacks.

---

## 9. Bulk Settlement Reconciliation

Replace the single-plan settlement path with `async bulk_settlement_reconciliation(merchant_id, period)` that aggregates all settled and pending plans, groups by settlement day, runs a three-way match (plan principal vs. instalment collected vs. payment rail confirmation), and flags discrepancies for human review. Returns a structured reconciliation report.

**Impact**: Finance teams currently export CSVs and reconcile manually — this eliminates that entirely.

---

## 10. Portfolio-Level Stress Testing

Add `async stress_test_portfolio(scenario)` where `scenario` is one of `{"rate_hike_200bps", "unemployment_spike_5pct", "fx_depreciation_20pct"}`. Apply scenario shocks to each active plan's probability-of-default and expected loss, and return a portfolio-level expected credit loss (ECL) estimate aligned with IFRS 9 Stage classification.

**Impact**: Required for CBK credit provider licensing; enables proactive provision booking.

---

## 11. Consent & Privacy Rights Automation

Add `async process_data_subject_request(customer_id, request_type)` where `request_type` in `{"erasure", "portability", "restriction"}`. For erasure: pseudonymise PII in all linked records and emit a `data_erased` audit event. For portability: serialise all consumer BNPL records to a machine-readable export. Enforces GDPR / Kenya Data Protection Act timelines (30-day SLA counter).

**Impact**: Mandatory for any BNPL operating in Kenya under the DPA 2019; currently entirely absent.

---

## 12. A/B Testing Framework for Plan Offers

Add `async get_plan_offer(customer_id, merchant_id, amount, experiment_id)` that routes the consumer to one of N configured offer variants (different plan types, rates, down-payment requirements) and records assignment. Add `async record_offer_conversion(experiment_id, assignment_id, converted)` to close the loop. Aggregate results via `async experiment_results(experiment_id)`.

**Impact**: Allows revenue and product teams to iteratively improve approval rates and take rates without engineering sprints.

---

## 13. Real-Time Delinquency Triage Queue

Add `async build_delinquency_queue()` that scans all overdue instalments, scores each for recovery probability (recency + frequency + monetary model), and returns a prioritised triage list with recommended actions (`auto_retry`, `sms_nudge`, `agent_call`, `debt_recovery`). Integrates with `debt_recovery_initiation` for automatic escalation above a configurable days-past-due threshold.

**Impact**: Collections efficiency metric improves 30–40% when agents work a scored queue vs. FIFO.

---

## 14. Regulatory Limit Engine (Per-Consumer Exposure Cap)

Many CBK and BoU BNPL frameworks cap total BNPL exposure per consumer across providers. Add `async check_regulatory_exposure(customer_id, amount, regulator)` that looks up the consumer's declared multi-provider exposure (from a pluggable bureau adapter), adds the requested amount, and rejects if the regulatory cap is breached. Persist the exposure delta on approval.

**Impact**: Direct compliance requirement for CBK Digital Credit Provider framework (2023 regulations).

---

## 15. Embedded BNPL SDK Contract / Merchant Widget Configuration

Add `async generate_merchant_widget_config(merchant_id, theme, plan_types, locale)` that returns a signed, tenant-scoped JSON configuration payload for the embedded JS/mobile SDK. Includes eligible plan types for the merchant's category, instalment messaging copy (localised), APR disclosure copy, and a short-lived signing token. Rotate signing tokens on every call.

**Impact**: Closes the "last mile" between backend BNPL logic and the merchant's storefront integration — the current gap forces each merchant to hand-wire the SDK.
