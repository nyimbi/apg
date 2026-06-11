# Mobile Banking — World-Class Improvement Proposals

**Capability**: `fintech_mobile` | **Version target**: 2.0.0

---

## 1. QR Code Payment Generation & Scanning

**Current gap**: No QR payment support. M-Pesa, card networks, and merchants all use QR-based flows.

**Improvement**: Add `generate_qr_payment(account_id, amount, reference)` and `scan_qr_payment(qr_payload)` async methods. QR payload encodes: `{account_id, amount, currency, expires_at, hmac_signature}`. Short-lived (90-second) HMAC-signed tokens prevent replay attacks. Critical for point-of-sale and peer-to-peer contactless payments in the Kenyan market.

---

## 2. Biometric Re-authentication Step-Up

**Current gap**: Auth factors are registered but never re-challenged at transaction time. High-value or anomalous transactions silently pass with the same session token.

**Improvement**: `step_up_auth(customer_id, device_id, factor_type, challenge_context)` issues a cryptographic challenge. The response is verified against the registered factor's `strength_reference` before the triggering operation is allowed to proceed. Supports FIDO2/WebAuthn assertions and local biometric attestations.

---

## 3. Real-Time Transaction Velocity Checks

**Current gap**: No per-customer velocity window enforcement. A compromised account can fire unlimited transfers in a session.

**Improvement**: `check_velocity(customer_id, amount, window_seconds, max_count, max_volume)` evaluates sliding-window counters persisted in `_velocity_windows`. Returns a `{allowed, current_count, current_volume, remaining_capacity}` dict. Plugs into `initiate_payment` and `funds_transfer` as a pre-flight gate.

---

## 4. Scheduled / Standing-Order Payments

**Current gap**: All payments are ad-hoc. No repeat billing, standing orders, or scheduled transfers exist.

**Improvement**: `create_standing_order(account_id, to_account, amount, frequency, start_date, end_date)` stores a standing-order record. `process_due_standing_orders()` is called by the Bytewax streaming layer to execute any orders due in the current window, idempotently, with a `last_executed_at` guard.

---

## 5. Multi-Currency FX Conversion Pre-Flight

**Current gap**: Cross-currency payments are denied at the policy layer with no conversion path offered. Users are dropped without a resolution path.

**Improvement**: `fx_conversion_quote(from_currency, to_currency, amount)` fetches indicative rates (from a pluggable rate provider or ECB fallback), returns `{rate, converted_amount, fee, quote_expires_at, quote_id}`. `accept_fx_quote(quote_id, account_id)` locks the rate and creates a wallet funding event. Proper async rate-locking prevents slippage.

---

## 6. Device Risk Scoring via Behavioural Signals

**Current gap**: Device `risk_tier` is static (set at bind-time). There is no runtime re-evaluation of device risk based on usage anomalies.

**Improvement**: `score_device_risk(device_id, session_context)` takes `{location, typing_cadence, app_version, rooted_flag, vpn_detected}` and returns a `{risk_score, risk_tier, signals}` dict. Score is stored against the device record; exceeding a threshold triggers automatic step-up auth or device suspension.

---

## 7. Offline USSD Transaction Queue

**Current gap**: USSD sessions are fully synchronous. Network interruptions during a session are fatal — the session is lost, and so is the intent.

**Improvement**: `queue_ussd_transaction(msisdn, transaction_intent, session_id)` serialises the user's intent before any network-bound operation. `drain_ussd_queue(msisdn)` replays queued intents when connectivity is restored. Idempotency keys prevent double-execution. Especially valuable for rural areas with spotty GPRS.

---

## 8. Regulatory KYC Refresh Workflow

**Current gap**: Once enrolled, customer KYC is never re-verified. CBK Mobile Banking Regulations (2021) require periodic refresh for high-tier accounts.

**Improvement**: `kyc_refresh(customer_id, updated_kyc_data, verifier_reference)` computes a KYC diff, records `changed_fields`, updates `kyc_status`, and emits a `mobile_kyc_refreshed` audit event. `kyc_expiry_check(customer_id)` returns days-to-expiry based on the tier-specific refresh cycle (Tier 1: annual; Tier 2: biennial).

---

## 9. Push Notification Delivery Receipts

**Current gap**: `push_notification_settings` stores preferences but there is no delivery confirmation loop. Failed notifications (expired tokens, app uninstall) are silently dropped.

**Improvement**: `record_push_delivery_receipt(notification_id, customer_id, status, failure_reason)` captures FCM/APNs delivery receipts. `get_push_delivery_stats(customer_id)` returns `{sent, delivered, failed, failure_reasons}`. Enables retry logic and channel fallback (SMS if push fails).

---

## 10. Loan Disbursement to Mobile Wallet

**Current gap**: `loan_application_mobile` returns a decision but never disburses funds.

**Improvement**: `disburse_loan(application_id, disbursement_account)` validates `decision == "approved"`, creates a `funds_transfer` to the nominated account, records `disbursed_at`, and updates the application to `disbursed` status. Idempotent — re-calling with the same `application_id` returns the existing disbursement record.

---

## 11. Customer Dispute Management

**Current gap**: `open_service_request` is a generic intake form with no dispute-specific lifecycle.

**Improvement**: `raise_payment_dispute(customer_id, payment_id, dispute_reason, amount_disputed)` creates a typed dispute record with `{dispute_id, status: "raised", sla_deadline, escalation_tier}`. `resolve_dispute(dispute_id, resolution, credited_amount)` closes it and issues a credit if warranted. Fulfils CBK consumer protection requirements.

---

## 12. SIM Swap Detection & Account Lock

**Current gap**: SIM swaps are the most common account takeover vector in mobile banking, but there is no SIM swap detection gate.

**Improvement**: `detect_sim_swap(msisdn, device_id)` queries a carrier API adapter and checks `sim_swap_recency_hours`. If a SIM swap is detected within 48 hours, the account is locked (`lock_account(customer_id, reason="sim_swap_detected")`), all active sessions are invalidated, and a fraud event is auto-recorded at `critical` severity with `human_approval` required for unlock.

---

## 13. Spend Analytics & Categorisation

**Current gap**: Mini-statements return raw transaction records with no categorisation, budgeting insight, or trend data.

**Improvement**: `spend_analytics(account_id, period)` buckets transactions into `{food, transport, utilities, airtime, transfers, loan_repayments, other}` using merchant-code matching and biller-code prefixes. Returns `{period, total_spend, by_category, top_merchants, savings_rate}`. Feeds the in-app personal finance dashboard.

---

## 14. Webhook Event Subscriptions

**Current gap**: Events stream only to Bytewax. Third-party integrators (BNPL providers, insurance, lending) have no outbound event hook.

**Improvement**: `register_webhook(tenant_id, url, events, signing_secret)` stores a webhook subscription. `dispatch_webhook(event_type, payload)` fans out to all matching subscriptions, signs with HMAC-SHA256, retries up to 3 times with exponential backoff, and records delivery status in `_webhook_deliveries`. Solves real-time integration for partner ecosystems.

---

## 15. Zero-Knowledge Balance Proof

**Current gap**: Balance inquiry returns the raw balance, which is privacy-sensitive when used as collateral proof to third parties.

**Improvement**: `prove_balance_threshold(account_id, threshold, verifier_id)` returns a signed assertion `{account_id_hash, threshold_met: bool, verifier_id, signed_at, signature}` without exposing the actual balance. The verifier can confirm "balance >= threshold" without learning the exact figure. Built on simple HMAC-based commitments; upgradeable to ZK-SNARK proofs via an adapter interface.

---

*Last updated: 2026-06-11*
