# Digital Wallets — World-Class Improvement Roadmap

**Capability**: `fintech_wallets` | **Version target**: 2.0.0 | **Author**: Nyimbi Odero | **Date**: 2026-06-11

---

## 1. Double-Entry Accounting with Contra Accounts

**Current state**: Credits and debits are single-sided ledger mutations on one wallet.  
**Improvement**: Enforce proper double-entry accounting where every debit has a matching credit on a contra account (e.g., `FLOAT-SUSPENSE`, `FX-GAIN`, `MERCHANT-PAYABLE`). This eliminates phantom money creation, satisfies accounting auditors, and enables accurate trial-balance generation.  
**Impact**: Compliance, auditability, prevents balance-sheet leakage.

---

## 2. Idempotency Store with TTL Eviction

**Current state**: Idempotency keys are checked against the in-memory ledger only; no deduplication window, no persistence.  
**Improvement**: Maintain a Redis-backed idempotency store keyed by `(tenant_id, idempotency_key)` with a 24-hour TTL. Return the original response on replay instead of raising a duplicate error. This is the only correct behavior for distributed payment systems where network retries are normal.  
**Impact**: Eliminates double-charges on network retry storms; required for PSP certifications.

---

## 3. Event Sourcing with Immutable Ledger

**Current state**: Balances are stored as mutable fields on `Wallet` dataclass; ledger entries are advisory.  
**Improvement**: Make the ledger the source of truth. Derive wallet balance by summing all ledger entries (with a periodic snapshot for performance). No balance mutation — only append. This enables point-in-time balance reconstruction, dispute resolution, and regulatory reporting.  
**Impact**: Regulatory compliance (CBK, PCI-DSS), dispute resolution latency from days to seconds.

---

## 4. Multi-Leg FX Transfers with Rate Lock

**Current state**: `currency_conversion_in_wallet` applies a static spot rate with a hard-coded 1.5% spread and only debits the source wallet.  
**Improvement**: Implement proper multi-leg FX: (1) lock rate for configurable TTL (e.g., 30s), (2) debit source wallet in source currency, (3) credit target wallet in target currency, (4) post spread to an FX income account, (5) emit a `fx_rate_locked` event. Expose `lock_fx_rate()`, `execute_locked_conversion()`, and `expire_fx_lock()` methods.  
**Impact**: Real-world FX desk behavior; prevents slippage losses on volatile currencies.

---

## 5. Wallet Velocity Controls and Fraud Scoring

**Current state**: Spend limits are simple daily/monthly counters with no pattern analysis.  
**Improvement**: Track transaction velocity (transactions-per-minute, distinct recipients-per-hour, geo-velocity) and compute a fraud score per transaction. Block or step-up authenticate when score exceeds configurable thresholds. Integrate with `fintech_intel` for ML scoring via a pluggable `FraudScorer` protocol.  
**Impact**: Reduces card-not-present fraud rates; satisfies CBK fraud monitoring requirements.

---

## 6. Wallet Group and Sub-Wallet Hierarchy

**Current state**: Wallets are flat — no parent-child relationship.  
**Improvement**: Support a wallet hierarchy: a parent `treasury` wallet can have child `sub-wallets` that share a pooled balance. Sub-wallets get individual limits but draw from the parent pool. This models fleet cards, family accounts, and corporate expense accounts natively.  
**Impact**: Unlocks corporate banking and fleet management use cases without new capability.

---

## 7. Scheduled Payments and Standing Orders

**Current state**: All operations are on-demand; no time-based execution.  
**Improvement**: Add `schedule_payment()` that persists a standing order (cron expression, amount, target wallet, description) and a scheduler runner that executes due orders with idempotent retry. Expose `list_scheduled_payments()`, `cancel_scheduled_payment()`, and `pause_scheduled_payment()`.  
**Impact**: Enables rent, utility, and subscription billing from wallets without external scheduler dependency.

---

## 8. Reconciliation Engine

**Current state**: No reconciliation against external provider statements.  
**Improvement**: Implement `reconcile_wallet()` that accepts an external statement (list of provider transactions) and matches them against internal ledger entries. Output matched, unmatched-internal, and unmatched-external sets. Flag discrepancies as `RECON_BREAK` evidence records. Emit a `reconciliation_completed` audit event.  
**Impact**: Eliminates manual reconciliation work; catches provider settlement errors before they compound.

---

## 9. Wallet Access Control Lists (ACL)

**Current state**: All tenants within a `tenant_id` have the same access; no wallet-level RBAC.  
**Improvement**: Introduce per-wallet ACLs: `grant_wallet_access(wallet_id, principal, permissions)` and `revoke_wallet_access()`. Permissions: `read`, `credit`, `debit`, `hold`, `admin`. Enforce ACL checks in every write operation before the capability-contract `_enforce()` call.  
**Impact**: Required for joint accounts, delegated payment authority, and corporate approval workflows.

---

## 10. Configurable Fee Engine

**Current state**: No fee charging — top-ups, transfers, and withdrawals are all free.  
**Improvement**: Implement a fee engine with configurable fee schedules: flat, percentage, tiered-volume, and interchange. Fee schedules are tenant-scoped and instrument-scoped. Fees are automatically debited to a `fee_income` wallet and returned in operation responses as `fee_charged`. Expose `get_fee_estimate()` for pre-flight fee display.  
**Impact**: Revenue generation capability; required for product managers to price wallet products.

---

## 11. Regulatory Reporting (CTR / STR)

**Current state**: Audit events are emitted but no regulatory threshold monitoring exists.  
**Improvement**: Add a compliance monitor that detects Currency Transaction Reports (CTR) triggers (single transaction >= KES 1,000,000 or USD 10,000) and Suspicious Transaction Reports (STR) triggers (structuring patterns, rapid round-tripping). Auto-generate draft regulatory reports as evidence records and notify the compliance team.  
**Impact**: Direct CBK AML compliance requirement; avoid fines and license revocation.

---

## 12. Wallet Snapshots for Point-in-Time Queries

**Current state**: `wallet_balance()` only returns current state; no historical balance queries.  
**Improvement**: Persist balance snapshots at configurable intervals (end-of-day, end-of-month) and after large transactions. Expose `balance_at(wallet_id, timestamp)` that returns the closest snapshot and applies subsequent ledger deltas to reconstruct the exact balance at any moment.  
**Impact**: Dispute resolution, regulatory reporting, product analytics, customer support tooling.

---

## 13. Asynchronous Bulk Operations with Progress Tracking

**Current state**: `bulk_create_wallets()` processes synchronously in a loop with no progress reporting.  
**Improvement**: Implement a proper async bulk pipeline: partition into micro-batches of 100, process batches concurrently with `asyncio.gather`, report progress via server-sent events or a `job_id`-based polling endpoint (`get_bulk_job_status(job_id)`). Use back-pressure to rate-limit at configurable TPS.  
**Impact**: Handles onboarding 100k customers at launch without timeout failures.

---

## 14. Cryptographic Receipt Generation

**Current state**: Transaction evidence is an in-memory dict with no integrity guarantee.  
**Improvement**: Sign every completed transaction with an HMAC-SHA256 receipt using a tenant-scoped signing key managed by the `keym` capability. Return the receipt hash in the operation response. Expose `verify_receipt(receipt_hash, transaction_id)` for customer-facing verification. Store receipt hashes in the evidence record.  
**Impact**: Prevents internal tampering claims; provides customer-facing transaction authenticity proof.

---

## 15. Real-Time Balance Streaming via WebSocket / SSE

**Current state**: Balance is a pull operation; clients must poll `wallet_balance()`.  
**Improvement**: Implement a `subscribe_wallet_balance(wallet_id)` async generator that yields balance updates whenever a ledger entry is posted to that wallet. Wire into the Bytewax event stream so that the wallet service emits delta events on `apg.fintech.wallets.balance_updates`. Front-end widgets and downstream capabilities (payments, neobanking) subscribe for real-time fund availability.  
**Impact**: Eliminates polling load; enables real-time payment confirmation UX; required for POS terminal integration.

---

*Improvements are ordered by combined compliance + revenue + engineering impact.*
