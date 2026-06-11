# Cross-Border Remittance — World-Class Improvements

**Capability**: `fintech_remittance` | **Version**: 1.2.0  
© 2025 Datacraft — www.datacraft.co.ke

---

## 1. Multi-Hop Corridor Routing with Cost Optimization

**Current state**: `partner_routing` picks a single pre-mapped partner from a static dict. No fallback cost ranking.

**Improvement**: Implement a shortest-path corridor graph (Dijkstra over fee + FX-spread edges) supporting multi-leg routes (e.g., KE → AE → IN) when no direct corridor exists. Each edge carries a composite cost score = fee_pct + fx_spread_pct + sla_penalty. Method: `async def optimal_corridor_path(send_country, receive_country, send_currency, receive_currency) -> dict`.

**Impact**: Opens ~40 indirect corridors with no new partner agreements; reduces remittance cost by an estimated 0.3–0.8% on indirect routes.

---

## 2. Real-Time FX Rate Feed Integration

**Current state**: FX rates are static module-level constants. Quotes use an `indicative` label with no staleness check.

**Improvement**: Add `async def refresh_fx_rates(providers: list[str]) -> dict` that fetches live rates from pluggable provider adapters (ExchangeRate-API, CBK Open Data, Wise sandbox). Rates cached in a `BoundedCache` with TTL=60s. Quote generation checks cache age and refuses to quote if last refresh > 5 minutes (configurable). Drift alert fires if bid/ask spread exceeds 2%.

**Impact**: Eliminates the single largest source of FX settlement risk in the current implementation.

---

## 3. Velocity-Based AML Risk Scoring Engine

**Current state**: `compliance_check` returns a hard-coded `aml_risk_score` of 15 or 95 with no transaction history context.

**Improvement**: `async def compute_velocity_risk(sender_id, window_hours, tenant_id) -> dict` accumulates per-sender transaction volume, frequency, and corridor diversity over configurable rolling windows (1h/24h/7d/30d). Score = weighted sum of: (daily_count / threshold), (daily_value / threshold), (distinct_corridors), (burst_ratio). Triggers automatic STR pre-filing at score ≥ 80.

**Impact**: Reduces false-negative AML events; aligns with FATF Recommendation 16 transaction monitoring requirements.

---

## 4. Idempotent Transfer Submission with Distributed Locks

**Current state**: `create_transfer` raises `ValueError` on duplicate `transfer_id` but provides no idempotency key mechanism. Retries from mobile clients cause double submissions.

**Improvement**: Add `idempotency_key: str` parameter to `initiate_remittance` and `create_transfer`. Deduplication window configurable (default 24h). Backend stores SHA-256(idempotency_key) → transfer_id. Returns existing record on duplicate rather than error.

**Impact**: Eliminates duplicate transfer class of bugs that is the #1 support ticket type in production remittance platforms.

---

## 5. Structured Payout Status Webhook Framework

**Current state**: `recipient_notification` appends to an in-memory log. No outbound webhook delivery, no retry, no HMAC signing.

**Improvement**: `async def register_webhook(tenant_id, url, events, secret) -> dict` and `async def dispatch_webhook(event_type, payload) -> dict`. Webhooks signed with HMAC-SHA256 (X-APG-Signature header). Retry queue with exponential backoff (3 attempts, 5s/30s/300s delays). Delivery receipts stored in `evidence`.

**Impact**: Enables partners to build event-driven integrations without polling the status endpoint every 30 seconds.

---

## 6. Tiered KYC Limits Enforcement at Quote Time

**Current state**: KYC checks are structurally required but limit enforcement (transaction limits per KYC tier) is absent. A Tier-1 KYC sender can initiate a KES 5M transfer unchecked.

**Improvement**: `async def enforce_kyc_tier_limits(sender_id, amount, send_currency, kyc_tier) -> dict`. KYC tier → daily/monthly limit matrix configurable per corridor. Returns `allowed: bool`, `limit_remaining`, `tier_upgrade_required`. Enforced at `initiate_remittance` before FX quote.

**Impact**: Direct regulatory compliance requirement under CBK PSP Guidelines Section 4.3 and KYC Tier framework.

---

## 7. Beneficiary Account Validation (IBAN/NUBAN/Sort Code)

**Current state**: `bank_payout` accepts any dict with `account_number` and `bank_code` with no format validation.

**Improvement**: `async def validate_bank_account(country, bank_code, account_number, account_type) -> dict`. Validates format per country: IBAN checksum (EU/GB), NUBAN algorithm (NG), sort-code + 8-digit (GB), RTGS routing (KE). Returns `valid: bool`, `bank_name`, `branch_name`, `swift_code`. Fails fast before payout dispatch.

**Impact**: Eliminates ~15% of failed bank payouts caused by malformed account numbers.

---

## 8. FX Forward Contract and Rate Lock

**Current state**: Quotes expire after 15 minutes with no option to lock a rate for deferred settlement.

**Improvement**: `async def create_fx_forward(send_currency, receive_currency, amount, settlement_date, tenor_days) -> dict`. Locks the indicative rate for up to 30 days with a forward points adjustment (carry = interest rate differential). Forward ID stored; `create_transfer` can reference `forward_id` in lieu of a spot quote.

**Impact**: Enables corporate clients sending regular payroll remittances to hedge FX exposure; common requirement in trade remittance.

---

## 9. Corridor Risk Heat Map and Exposure Reporting

**Current state**: `corridor_analytics` computes basic aggregates per corridor but no risk exposure or concentration analysis.

**Improvement**: `async def corridor_risk_heatmap(tenant_id, period) -> dict`. For each corridor: open exposure (submitted but unpaid), settlement risk (FX movement since quote), concentration risk (single corridor > 30% of total volume), partner credit risk score. Output is a ranked heat map with RAG (Red/Amber/Green) status.

**Impact**: Enables treasury to monitor and cap corridor exposure in real time — required for PAPSS (Pan-African Payment and Settlement System) participation.

---

## 10. Regulatory Sandbox Mode with Synthetic Data

**Current state**: No test/sandbox isolation. Tests use the same code path as production which risks polluting shared state.

**Improvement**: `RemittanceService(sandbox=True)` mode activates synthetic FX rates (deterministic, seeded), compliance decisions bypass real sanction lists (returning configurable outcomes), and all transfer IDs are prefixed `SANDBOX-`. `async def sandbox_reset() -> dict` purges all sandbox state. Sandbox events do not emit to Bytewax.

**Impact**: Enables partner certification and automated integration tests without production data risk.

---

## 11. Multi-Currency Wallet Sweep for Funding

**Current state**: Funding reference is a free-text string. No integration with wallet balances.

**Improvement**: `async def wallet_sweep_funding(sender_id, amount, preferred_currency, wallet_ids) -> dict`. Queries wallet balances across multiple currencies, selects optimal wallet(s) using greedy cover (minimize FX conversion cost), reserves funds atomically, returns `funding_reference` for use in `create_transfer`. Integrates with `fintech_wallets` adapter.

**Impact**: Enables diaspora senders with multi-currency wallets to fund remittances without manual currency conversion step.

---

## 12. AI-Powered Purpose Code Classification

**Current state**: `purpose_code` is caller-supplied with no validation against transaction description or beneficiary profile.

**Improvement**: `async def classify_purpose_code(transaction_description, sender_profile, beneficiary_profile) -> dict`. Calls local Ollama model (mistral/llama3) to classify transaction purpose from free-text description into supported purpose codes. Returns `predicted_code`, `confidence`, `alternative_codes`. Fallback: rule-based keyword matching. Used to flag mismatches between declared and predicted purpose (AML signal).

**Impact**: Reduces purpose code misclassification (a common technique in structuring) and removes manual step for bulk B2B remittance clients.

---

## 13. ISO 20022 pacs.008 Message Generation

**Current state**: Receipts are JSON dicts. No interoperable payment message format.

**Improvement**: `async def generate_pacs008(transfer_id, instruction_id) -> dict`. Produces a compliant ISO 20022 `pacs.008` (FI-to-FI Customer Credit Transfer) XML message from transfer and quote data. Fields: `GrpHdr`, `CdtTrfTxInf`, `CdtrAgt` SWIFT BIC, `CdtrAcct` IBAN/account. Required for SWIFT GPI, PAPSS, and bilateral correspondent banking integrations.

**Impact**: Unlocks SWIFT GPI corridor connectivity and African central bank RTGS integration; prerequisite for institutional remittance volumes.

---

## 14. Dynamic Fee Negotiation for High-Volume Senders

**Current state**: Fee tiers in `fee_schedule` are static hard-coded bands.

**Improvement**: `async def negotiate_fee(sender_id, monthly_volume_kes, corridor, commitment_months) -> dict`. Calculates volume-based rebate tier using a configurable schedule. Returns `negotiated_fee_pct`, `rebate_amount`, `agreement_id`, `valid_until`. Signed agreements stored in evidence. `get_fx_quote` checks for active fee agreement before applying default corridor fee.

**Impact**: Enables acquisition and retention of high-volume diaspora senders (top 5% of senders typically represent 40%+ of volume).

---

## 15. End-to-End Transfer Simulation (Dry Run)

**Current state**: No way to validate an entire remittance flow (quote → compliance → routing → payout) without creating real records.

**Improvement**: `async def simulate_transfer(sender_id, recipient, amount, send_currency, receive_currency, corridor, payout_method) -> dict`. Runs the full `initiate_remittance` logic in a read-only context: FX quote, compliance check, partner routing, KYC tier check, fee negotiation — all executed against live data but no records written. Returns `simulation_id`, `would_succeed: bool`, `fx_rate`, `total_cost`, `estimated_delivery`, `blocking_reasons`. Useful for pre-flight checks in mobile apps before the sender confirms.

**Impact**: Reduces user abandon rate by surfacing blocking conditions (compliance hold, KYC tier limit, unsupported corridor) before the user commits; standard feature in Wise, Remitly, and Western Union digital flows.
