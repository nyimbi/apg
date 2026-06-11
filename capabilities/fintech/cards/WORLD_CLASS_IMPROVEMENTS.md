# Digital Cards — World-Class Improvements

**Capability**: `fintech_cards` | **Version target**: 2.0.0
**Author**: Nyimbi Odero | **© 2025 Datacraft**

---

## 1. Real PAN/CVV Cryptographic Generation (EMV Spec Compliance)

Current `_generate_cvv` uses SHA-256 with a static salt — not EMV-compliant and not production-safe. Replace with a proper CVV/CVV2 derivation using 3DES (TDEA) over a service master key, binding to PAN + expiry + service-code. Store only the verification value, never the PAN in plaintext.

**Impact**: Eliminates card data security audit findings; required for PCI-DSS Level 1 compliance.

---

## 2. Hardware Security Module (HSM) Integration Adapter

Card data (PAN, CVV, PIN blocks) must be generated and stored inside an HSM boundary. Add an `HSMAdapter` interface with pluggable backends: `softhsm` (dev), `thales` (prod), `aws_cloudhsm`. All key operations (PIN derivation, token DUKPT, EMV AC generation) route through this adapter.

**Impact**: Satisfies PCI-PIN and PCI-DSS key management requirements; mandatory for live card issuing.

---

## 3. EMV Chip Personalisation Pipeline

Physical card issuance currently produces no personalisation data. Add `personalise_emv_card()` that generates the EMV Application Cryptogram seed, personalisation script, and ARQC verification key set. Integrate with a Bureau Service adapter (e.g. CPI Card Group, Valid) for actual embossing job submission.

**Impact**: Enables physical card programs with real chip data; without this, physical cards cannot authenticate at EMV terminals.

---

## 4. Network Token Lifecycle Management (EMV Payment Tokenisation)

Token provisioning stores a token reference string but performs no cryptographic binding. Implement the full EMV Payment Tokenisation spec: Token Service Provider (TSP) registration, domain-restricted DPAN issuance, token cryptogram (TAVV) generation per transaction, and token suspension/deletion with scheme notification.

**Impact**: Required for Apple Pay, Google Pay, and Samsung Pay live integration; current implementation produces non-functional tokens.

---

## 5. Real-Time Velocity & Spend Control Engine

`process_card_transaction` resets `daily_spend` only via the manual `reset_daily_spend` call — it has no wall-clock reset. Replace the in-memory dict with a velocity ledger backed by Redis `INCRBYFLOAT` with TTL-based windows (per-minute, hourly, daily, monthly). Support sliding-window counters to prevent midnight-boundary gaming.

**Impact**: Prevents fraud patterns exploiting daily-limit reset timing; real issuers enforce sub-second velocity checks.

---

## 6. 3DS 2.x (EMV 3-D Secure) Full Protocol Implementation

Current `card_3ds_challenge` returns an OTP hint in plaintext and accepts any 6-char alphanumeric string as valid. Implement the full 3DS 2.x flow: ACS (Access Control Server) challenge, CReq/CRes message exchange, authentication value (CAVV/AAV) generation, and frictionless-flow risk-based authentication. Use device fingerprinting signals to gate OTP challenges.

**Impact**: Without real 3DS, CNP fraud liability sits with the issuer; 3DS shifts liability to the merchant/acquirer.

---

## 7. Async Event Streaming via Bytewax / Redpanda

`validate_batch` simply returns a static acceptance dict. Replace with a real async producer that writes typed `CloudEvent` records to a `apg.fintech.cards.lifecycle` Redpanda topic using the `bytewax` source/sink pattern. Each card lifecycle event (issued, activated, blocked, authorized, disputed) should be emitted atomically with the database write (transactional outbox pattern).

**Impact**: Enables downstream capabilities (`fintech_payments`, `fintech_wallets`) to react to card state changes in real time.

---

## 8. Idempotency Keys for All Mutating Operations

No mutating method currently checks for duplicate requests. Add an idempotency layer: callers supply an `idempotency_key` (UUID), and the service stores the response for 24 hours. Duplicate keys return the cached response without re-executing business logic. Back with Redis `SET NX EX`.

**Impact**: Prevents double-issuance, duplicate authorizations, and double-disputes in retried HTTP calls — a common production incident vector.

---

## 9. Card Lifecycle State Machine with Explicit Transitions

`_card_status` is a raw string dict with no enforced transition graph. Replace with a proper finite state machine: `inactive → active → frozen → active → blocked → [terminal]`. Block illegal transitions (e.g. `inactive → blocked` without going through `active`). Use `transitions` library or a hand-rolled `StateMachine` base class.

**Impact**: Prevents invalid state combinations caught only in production; simplifies compliance attestation ("card can only be blocked from active state").

---

## 10. PostgreSQL-Backed Persistent Store with Alembic Migrations

All service state (cards, tokens, authorizations, disputes) lives in in-memory Python dicts. The `database/store.py` and `alembic/` skeleton exist but are not wired to the service. Complete the `DatabaseStore` implementation using `asyncpg` connection pools, wire it to `DigitalCardsService` via the `store` constructor parameter, and replace all in-memory dicts with store calls.

**Impact**: Survives process restarts; enables horizontal scaling; unlocks the Alembic migration pipeline already partially defined.

---

## 11. Fraud Score Feedback Loop (Model Retraining Sink)

`ml_card_fraud_score` calls Ollama but discards the result after returning it. Add a feedback sink: store `(transaction_id, features, fraud_score, actual_outcome)` tuples to a `fraud_feedback` PostgreSQL table. Expose a `retrain_fraud_model()` method that ships the feedback dataset to a local Ollama fine-tune job. Closes the model drift loop.

**Impact**: Fraud models degrade within weeks without retraining on live data; this is the minimum feedback infrastructure needed.

---

## 12. Dispute Workflow with SLA Timers and Escalation

`file_dispute` and `resolve_dispute` have no SLA tracking. Add: `dispute_sla_deadline` (Visa/MC mandate 45 days for pre-arbitration), automated escalation events when deadline approaches (T-7d, T-1d), and integration with the `ntfy` adapter to notify the assigned reviewer. Track chargeback win/loss rates per merchant category.

**Impact**: Missing SLA management is the top operational risk in card issuing; chargebacks past deadline result in automatic cardholder wins and scheme fines.

---

## 13. Multi-Currency Settlement and FX Rate Engine

All amounts are stored and processed in a single currency. Add multi-currency support: store original transaction currency + amount + exchange rate + settlement currency + settlement amount. Integrate an FX rate adapter (ECB feed locally, or `fintech_fx` capability if available). Compute card-holder-facing DCC (Dynamic Currency Conversion) markups.

**Impact**: Cross-border transactions require currency conversion at authorization time; without this, international spend controls and reporting are unreliable.

---

## 14. Tenant Isolation with Row-Level Security (RLS)

Tenant isolation is currently enforced only in application code (`_tenant_card_or_none` pattern). Add PostgreSQL Row-Level Security policies on all card tables so that even a miscoded query cannot leak cross-tenant data. The service should connect with a per-tenant role or pass `SET app.current_tenant = $1` before every query.

**Impact**: Application-level tenant checks are one bug away from a data breach; RLS provides defence-in-depth as a database-enforced boundary.

---

## 15. OpenTelemetry Distributed Tracing Integration

No instrumentation exists. Wrap every public async method with an `opentelemetry.trace.get_tracer()` span, recording `card_id`, `tenant_id`, `operation`, and `decision` as span attributes. Export to a local Jaeger or OTLP-compatible collector. Add `trace_id` to all audit events and API responses so operations can correlate a card decline across fraud, AML, and auth services.

**Impact**: Without tracing, diagnosing a declined transaction across fraud + AML + auth + 3DS services requires log-grepping across 4 services; tracing reduces MTTR from hours to minutes.

---

*Priority order: 10 (persistence) → 2 (HSM) → 8 (idempotency) → 5 (velocity engine) → 9 (state machine) → 6 (3DS) → 4 (network tokens) → 7 (streaming) → 14 (RLS) → 15 (tracing) → 1 (EMV PAN) → 3 (EMV chip) → 12 (dispute SLA) → 13 (multi-currency) → 11 (fraud feedback).*
