# WALT - World-Class Improvement Proposals

**Capability**: Digital Wallet (`walt`) | **Domain**: `common`
**Author**: Nyimbi Odero | **Date**: 2026-06-11

---

## 1. Multi-Currency FX Conversion with Locked Rates

**Problem**: `wallet_transfer` hard-blocks cross-currency transfers with a `currency_mismatch` error. Real-world wallets need to move value across currency boundaries.

**Proposal**: Add `wallet_fx_transfer(source_wallet_id, dest_wallet_id, amount, fx_rate, fx_rate_ref, ttl_seconds)`. The caller supplies a pre-negotiated `fx_rate` and a `fx_rate_ref` (e.g. from an FX provider quote). The service validates that the rate is within a configured tolerance band, locks both wallet balances atomically, debits source in source currency, credits destination in destination currency at the locked rate, and emits `wallet_fx_transfer` events on both wallets. Rate-lock TTL prevents stale quotes being replayed.

**Impact**: Enables cross-border payment products, multi-currency e-wallets, and B2B treasury flows without needing an external FX microservice at design time.

---

## 2. Spending Limits Engine (Daily / Monthly / Per-Transaction)

**Problem**: There is no enforcement of per-wallet or per-instrument spending caps. A single rogue transaction can drain a wallet.

**Proposal**: Add a `LimitProfile` model (daily_limit_minor, monthly_limit_minor, per_txn_limit_minor, currency) linked to wallet IDs. Implement `set_wallet_limit(wallet_id, limit_profile)`, `get_wallet_limit(wallet_id)`, and a `_check_spending_limits(wallet, amount_minor)` guard invoked inside `authorize_transaction`. Exceeding a limit raises `PermissionError("spending_limit_exceeded")` with the specific limit name in the payload.

**Impact**: Prerequisite for prepaid card products, parental controls, corporate expense cards, and regulatory compliance (PSD2 strong SCA for high-value transactions).

---

## 3. Idempotent Replay Protection Store

**Problem**: `idempotency_key` is stored on `TransactionRecord` but never checked before processing. Duplicate POST calls create duplicate transactions.

**Proposal**: Maintain a `_idempotency_store: dict[str, str]` mapping `(tenant_id, idempotency_key)` -> `transaction_id`. At the top of `authorize_transaction`, if the key is already present, return the existing transaction record without mutation. Add a TTL mechanism (ISO timestamp + configurable window, default 24 h) to expire stale keys. Emit `transaction_idempotent_replay` audit event on replay.

**Impact**: Eliminates double-charge bugs that arise from network retries, mobile app reconnects, and payment gateway timeouts. Critical for PCI-DSS and financial accuracy.

---

## 4. Dispute and Chargeback Lifecycle

**Problem**: There is no first-class concept of disputes or chargebacks. `transaction_reverse` is a blunt instrument that does not record the dispute chain.

**Proposal**: Add `DisputeRecord` (dispute_id, transaction_id, reason_code, evidence_ref, status, opened_at, resolved_at). Implement `open_dispute(transaction_id, reason_code, evidence_ref)`, `resolve_dispute(dispute_id, resolution, resolution_ref)`, and `list_disputes(wallet_id, status_filter)`. Link dispute records to the original transaction. Block settlement of transactions with open disputes.

**Impact**: Enables issuer-side chargeback handling, consumer protection flows, and compliance with card network rules (Visa/Mastercard reason codes).

---

## 5. Scheduled Recurring Payments

**Problem**: There is no concept of scheduled or recurring debits. Applications must implement this externally and re-invent authorization logic.

**Proposal**: Add `RecurringPaymentSchedule` model (schedule_id, wallet_id, instrument_id, amount, currency, frequency, next_run_at, end_at, status). Implement `create_recurring_payment(...)`, `pause_recurring_payment(schedule_id)`, `cancel_recurring_payment(schedule_id)`, and `list_due_schedules(tenant_id, as_of)` to surface schedules whose `next_run_at <= as_of`. Calling code (a Bytewax worker or cron) calls `execute_recurring_payment(schedule_id)` which internally calls `authorize_transaction` + `capture_transaction` and advances `next_run_at`.

**Impact**: Enables subscription billing, standing orders, loan repayments, and insurance premiums without building a separate scheduler per product.

---

## 6. Wallet Tags and Metadata Indexing

**Problem**: `WalletRecord` has no user-extensible tags or metadata. Finding wallets by product type, campaign, or custom dimension requires full-table scans over `list_wallets`.

**Proposal**: Add `tags: dict[str, str]` to `WalletRecord`. Implement `wallet_tag(wallet_id, tags)` (merge/upsert semantics), `wallet_untag(wallet_id, keys)`, and `wallet_search(tenant_id, tags, currency, status)` with O(n) in-memory filter (adapter layer can map to a PostgreSQL GIN index). Tag keys are restricted to `[a-z0-9_]` with max length 64; values max 256 chars.

**Impact**: Enables product-segmented reporting, A/B test cohort isolation, and marketing attribution without schema changes.

---

## 7. Ledger Double-Entry Journal

**Problem**: The current balance model is a simple integer counter. There is no immutable audit trail of individual ledger movements, making forensic reconstruction and regulatory reporting difficult.

**Proposal**: Add `LedgerEntry` dataclass (entry_id, wallet_id, debit_minor, credit_minor, running_balance_minor, entry_ref, entry_type, created_at). On every balance mutation (topup, withdrawal, transfer, capture, reversal, cashback, loyalty conversion), append an immutable `LedgerEntry`. Add `get_ledger(wallet_id, limit, before_entry_id)` for paginated traversal. Never mutate entries; corrections use compensating entries.

**Impact**: Produces a GAAP/IFRS-compliant audit ledger, enables balance reconciliation without relying on `WalletRecord.balance_minor`, and simplifies forensic investigations.

---

## 8. KYC / Compliance Gate per Operation

**Problem**: `compliance_policy_ref` is stored on wallet creation but never evaluated against incoming transactions. High-value operations proceed regardless of KYC status.

**Proposal**: Add a `ComplianceStatus` enum (pending, verified, limited, blocked) and store it on `WalletRecord`. Implement `set_compliance_status(wallet_id, status, evidence_ref, actor)`. Inside `authorize_transaction`, `wallet_withdraw`, and `create_settlement_batch`, call `_check_compliance(wallet)` which blocks transactions for `blocked` wallets and limits amount for `limited` wallets (configurable cap). Emit `compliance_gate_triggered` audit events.

**Impact**: Directly addresses FATF recommendations, AML regulations, and platform liability for unverified users transacting large amounts.

---

## 9. Real-Time Velocity Controls

**Problem**: The rule engine checks per-transaction risk scores but has no time-window aggregation. A low-risk-score attacker can execute hundreds of small transactions rapidly.

**Proposal**: Add `_velocity_counters: dict[str, list[tuple[str, int]]]` keyed by `(tenant_id, wallet_id)`, storing `(timestamp, amount_minor)` tuples. On each `authorize_transaction`, prune entries older than the velocity window (default 1 h, configurable), sum amounts, and compare against `velocity_limit_minor`. Breach raises `PermissionError("velocity_limit_exceeded")` and emits a `velocity_breach` audit event with severity `high`.

**Impact**: Defeats card-testing attacks, account takeover draining, and rapid-fire fraud patterns that evade per-transaction risk models.

---

## 10. Async Batch Transaction Processing

**Problem**: `authorize_transaction` processes one transaction at a time. Bulk operations (payroll disbursements, mass refunds, loyalty payouts) require calling it in a tight loop, which serializes I/O and inflates latency.

**Proposal**: Add `batch_authorize(tenant_id, items: list[BatchAuthorizeItem])` where `BatchAuthorizeItem` is a Pydantic model containing the per-transaction fields. Use `asyncio.gather(*[self._authorize_one(item) for item in items], return_exceptions=True)` internally. Return a `BatchAuthorizeResult` with `succeeded`, `failed`, and `partial` counts plus per-item results. Failed items include their error code; succeeded items return full transaction records.

**Impact**: Reduces payroll disbursement latency by an order of magnitude, enables high-throughput promotional credit drops, and simplifies Bytewax pipeline fan-out patterns.

---

## 11. Instrument Expiry and Rotation

**Problem**: `PaymentInstrumentRecord` has no expiry date. Expired cards continue to authorize transactions until the underlying payment rail rejects them, causing late failure.

**Proposal**: Add `expires_at: str | None` field to `PaymentInstrumentRecord`. In `register_instrument`, accept optional `expires_at`. Add `_check_instrument_expiry(instrument)` called from `authorize_transaction` that raises `PermissionError("instrument_expired")` when `expires_at < utc_now()`. Implement `rotate_instrument(old_instrument_id, new_instrument_ref, token_ref, expires_at)` which registers the new instrument, migrates authorized-but-not-captured transactions, and marks the old instrument as `expired`.

**Impact**: Prevents surprise declined transactions, enables automatic card-on-file refresh flows, and reduces customer friction on subscription renewals.

---

## 12. Webhook / Event Notification Dispatch Registry

**Problem**: External services have no way to subscribe to wallet events without polling `list_audit_events`. This couples consumers to the service's internal store.

**Proposal**: Add `WebhookSubscription` model (sub_id, tenant_id, event_types: list[str], endpoint_url, signing_secret, active). Implement `register_webhook(tenant_id, event_types, endpoint_url, signing_secret)`, `deactivate_webhook(sub_id)`, and `list_webhooks(tenant_id)`. In `_record_event`, collect matching active subscriptions and append outbound dispatch records to `_pending_webhooks`. A background worker (or Bytewax sink) processes these. HMAC-SHA256 sign payloads with the subscription's signing secret.

**Impact**: Enables real-time event-driven integrations (fraud alerts to Slack, settlement notifications to ERP, balance updates to mobile push), removing the polling anti-pattern.

---

## 13. Wallet Freeze with Partial Unfreeze

**Problem**: `wallet_lock` / `wallet_unlock` are binary. Regulators and compliance teams often need to allow incoming credits while blocking outgoing debits (asset preservation orders, AML holds).

**Proposal**: Add `freeze_mode: str` field to `WalletRecord` with values `none`, `debit_only`, `credit_only`, `full`. Implement `wallet_freeze(wallet_id, freeze_mode, reason, legal_ref, actor)` and `wallet_unfreeze(wallet_id, actor, unfreeze_ref)`. In `authorize_transaction`, `wallet_withdraw`, and `wallet_transfer`, check `freeze_mode` and raise `PermissionError("wallet_frozen:{mode}")` for blocked directions. Emit `wallet_frozen` / `wallet_unfrozen` audit events at severity `high`.

**Impact**: Directly required for asset freezing orders, AML investigations, and insolvency proceedings without needing a separate legal-hold system.

---

## 14. Tiered Cashback and Reward Rules Engine

**Problem**: `cashback_credit` accepts a flat `amount`. Building tiered reward structures (e.g. 2% on first $500/month, 1.5% thereafter) requires the calling application to implement reward logic, leading to drift between products.

**Proposal**: Add `CashbackRule` model (rule_id, wallet_id or tenant_id, tier_breakpoints: list[dict], promotion_ref, active). Implement `register_cashback_rule(...)` and `compute_cashback(wallet_id, transaction_id)` which looks up active rules, evaluates tier breakpoints against the month-to-date cashback total, and returns the computed cashback amount. Calling `apply_cashback(wallet_id, transaction_id)` invokes `compute_cashback` and calls `cashback_credit` with the result.

**Impact**: Centralizes reward logic in the wallet service, making promotions auditable, testable, and consistent across all product surfaces.

---

## 15. Paginated Cursor-Based List APIs

**Problem**: All `list_*` methods return unbounded in-memory lists. For tenants with millions of transactions, this causes memory exhaustion and multi-second response times.

**Proposal**: Add a `Page[T]` generic return type with fields `items: list[T]`, `next_cursor: str | None`, `prev_cursor: str | None`, `total_hint: int`. Implement `list_transactions_paged(tenant_id, wallet_id, cursor, limit, sort_by, direction)`, `list_audit_events_paged(...)`, and `list_wallets_paged(...)`. Cursors encode `(sort_key, id)` tuples as base64 strings. Existing `list_*` methods remain as backward-compatible convenience wrappers calling the paged variants with `limit=None`.

**Impact**: Prerequisite for production deployment. Without pagination, the service is unusable at scale and vulnerable to memory exhaustion DoS. This improvement unlocks enterprise adoption.
