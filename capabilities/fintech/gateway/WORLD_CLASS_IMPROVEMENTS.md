# World-Class Improvements: Fintech Gateway

## 1. Async-First Service Layer
All service methods are synchronous today; the event loop blocks on I/O (provider HTTP calls, DB writes). Convert the entire public API to `async def`, use `asyncio.gather` for fan-out operations (multi-rail authorization), and add a thin sync shim (`run_sync`) only for legacy callers. This removes the hidden latency cliffs that appear at >100 concurrent payments.

## 2. Streaming Risk Decisions via LLM (Ollama)
Replace the static rule engine `evaluate_capability_rules` with a streaming Ollama inference call for borderline risk scores (0.4–0.7). Low-risk (<0.4) and blocked (>0.9) cases still use the fast path. Mid-range cases get a structured prompt returning `{decision, confidence, rationale}` in <200 ms with a locally-hosted model (e.g., `llama3.2`). This cuts false positives on novel fraud patterns without exposing data to third parties.

## 3. Multi-Rail Payment Routing with Fallback Graph
Replace the single `provider_connection_id` field on a payment intent with a prioritized **rail graph**: `[("mpesa", priority=1), ("pesalink", priority=2), ("rtgs", priority=3)]`. The `authorize_payment` method walks the graph, attempting each rail in priority order, recording the attempt in an `authorization_attempts` list. This makes failover automatic, observable, and replayable rather than a manual `provider_failover` call.

## 4. Idempotency Registry with TTL
Add a bounded LRU idempotency cache (`BoundedCache`) shared across all write operations — not just webhooks. Every mutating method accepts an optional `idempotency_key`; if the key is seen within a configurable TTL (default 24 h), the cached response is returned immediately without re-execution. This makes every API call safely retryable by default.

## 5. Decimal-Safe Amount Arithmetic Throughout
Several methods accept `float | int | Decimal` and immediately coerce with `Decimal(str(amount))`, but comparison operators (`>`, `==`) elsewhere in the codebase mix float and Decimal, producing silent precision errors at scale. Introduce an `Amount` newtype (a `Decimal` subclass with factory `Amount.of(v)`) and make all arithmetic and comparisons go through it. Catches mismatches at construction, not at 2 AM during reconciliation.

## 6. MPESA Callback Reconciliation Loop
`mpesa_stk_push` creates an intent with `status: pending` but there is no mechanism to reconcile Safaricom's async callback. Add `mpesa_confirm_stk_callback(tenant_id, checkout_request_id, result_code, mpesa_receipt_number)` that advances the intent to `captured` or `failed`, updates the audit trail, and emits a `mpesa_stk_confirmed` event. Without this, pending intents accumulate indefinitely.

## 7. Equity Bank & KCB EFT Integration
The `connect_provider` method references `SUPPORTED_PROVIDERS` but the service has no Equity/KCB-specific initiator methods (contrast: dedicated `mpesa_stk_push`, `pesalink_transfer`, `rtgs_payment`). Add `equity_eft_transfer` and `kcb_connect_transfer` with bank-specific validations (account format, daily limit, branch code) so the multi-rail router can dispatch to these rails without generic passthrough.

## 8. Webhook Signature Verification
`ingest_webhook` stores signatures but never verifies them — the `signature_present` rule only checks non-empty. Add a `verify_webhook_signature(provider, raw_body: bytes, signature: str, secret_reference: str) -> bool` method that dispatches to provider-specific HMAC verification (Stripe: `sha256`, MPESA: custom CBK scheme, Flutterwave: `sha512`). Gate `ingest_webhook` on this verification rather than just presence.

## 9. Composite Settlement Reconciliation with Break Detection
`reconcile_settlements` produces a single variance number. Extend it to perform **line-item matching**: for each capture, find the corresponding settlement line by provider reference and flag captures with no settlement (`unmatched_captures`) and settlement lines with no capture (`phantom_credits`). Phantom credits are a high-priority fraud vector in mobile money rails.

## 10. Merchant Risk Scoring Pipeline
`onboard_merchant` accepts a static `risk_level` string from the caller. Add `score_merchant_risk(merchant_id, tenant_id) -> dict` that computes a dynamic risk score from: country risk tier (FATF grey/black list), business category (MCC), monthly volume, dispute rate, and chargebacks. The result feeds back into `onboard_merchant` as an override if the computed score is higher than the submitted one.

## 11. Payment Intent State Machine Enforcement
The `status` field transitions are controlled by individual methods but never validated against a formal state machine. A `void_payment_intent` only guards `draft`; nothing prevents `captured → authorized` regression. Add a `INTENT_TRANSITIONS` graph and an `_assert_valid_transition(current, target)` guard called in every mutating method. Eliminates an entire category of integration bugs.

## 12. Regulatory Capital Reporting (CBK PSP Returns)
The `cbk_return_filing` method generates a summary count. CBK's Payment Service Provider regulations require structured XML/JSON returns with: transaction volumes by payment type, cross-border flows by currency, settlement accounts used, and customer due diligence counts. Upgrade to a `CbkPspReturn` Pydantic model with full field coverage and a `validate_cbk_return` method that checks completeness before marking `filed`.

## 13. Tenant-Level Rate Limiting and Quota Enforcement
There are no per-tenant rate limits. High-volume tenants can starve shared in-memory state stores. Add a `TenantQuota` model with configurable `max_intents_per_minute`, `max_capture_volume_per_day`, and `max_merchants`. Enforce in `_assert_rules` by passing current-window counts in the context dict. Enables SLA tiering without infrastructure changes.

## 14. Async Audit Trail with Structured Schema
`_emit` appends to an in-memory list with minimal fields. Upgrade to a structured `AuditEvent` Pydantic model with: `correlation_id`, `causation_id`, `actor_type` (human/agent/system), `actor_id`, `ip_address`, `request_id`, `before_state`, `after_state`. Use `asyncio.Queue` for async emit so the audit path is never on the critical payment path.

## 15. Payment Analytics Aggregation Engine
`dashboard_summary` does linear scans over all in-memory dicts on every call. Replace with an in-process aggregation layer using a `TenantMetrics` dataclass that maintains running totals updated incrementally by `_emit`. Dashboard calls become O(1) lookups. At >10 k intents per tenant, the current implementation will produce measurable latency spikes on every dashboard refresh.
