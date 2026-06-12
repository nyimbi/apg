# Fintech Gateway

## Overview
Fintech Gateway is the payment orchestration capability responsible for merchant onboarding, payment provider connections, payment method tokenization, payment intent lifecycle, routing decisions, fraud risk review, authorization and capture, refunds, webhook ingestion, settlement reconciliation, and dispute management. It is the operational hub connecting the APG payment layer to external processors (Stripe, Adyen, MPESA, Flutterwave, Pesapal, DPO, PayPal, and others) while enforcing routing, risk, and governance rules on every payment.

`blocked` risk produces hard denies on authorization. Overcapture and overrefund are blocked by the rule engine. Settlement variance requires review. Webhook ingestion requires idempotency keys and signature verification. All gateway events stream to `apg.fintech.gateway.lifecycle` via Bytewax.

**Capability ID**: `fintech_gateway`  **Version**: 2.1.0

## Capability ID
`fintech_gateway`  Version: 2.1.0

## Provides
| Service | Description |
|---------|-------------|
| merchant_onboarding_lifecycle | Onboard merchants with code, legal name, country, KYC, and high-risk review gates |
| provider_connection_lifecycle | Connect payment providers with type, credential reference, and supported methods |
| payment_method_tokenization_workflow | Tokenize payment methods for card, bank, mobile money, wallet, and cash voucher |
| payment_intent_lifecycle | Create and track payment intents with merchant, amount, currency, and method |
| payment_routing_workflow | Route payment intents to providers based on risk and routing decisions |
| fraud_risk_review_workflow | Score and review payment risk; blocked risk denies authorization |
| authorization_capture_workflow | Authorize and capture payments with overcapture protection |
| refund_lifecycle | Process refunds with overrefund protection and large-refund review |
| webhook_ingestion_workflow | Ingest signed provider webhooks with idempotency key enforcement |
| settlement_reconciliation_workflow | Record settlements with variance review |
| payment_dispute_workflow | Open and resolve payment disputes with owner assignment and resolution review |
| gateway_agents | Register AI agents for merchant underwriting, routing, fraud, and settlement review |
| mpesa_rails | STK Push, B2B transfer with phone validation and async callback reconciliation |
| pesalink_rails | CBK-cleared interbank transfers up to KES 999,999 |
| rtgs_rails | High-value RTGS payments (KES 1M+) with mandatory approval |
| cbk_regulatory | CBK PSP return filing with volume, cross-border, and CDD fields |
| merchant_lifecycle | Suspend, reactivate, and bulk-onboard merchants |
| analytics | Fraud risk aggregation, dashboard O(1) summaries, transaction export |

## Requires
| Capability | Purpose |
|------------|---------|
| auth | Authentication |
| audl | Audit trail |
| ntfy | Merchant and operations notifications |
| composition_events | Composition event bus |
| composition_config | Configuration management |
| keym | Key management |
| encr | Encryption for payment credentials |
| cash_management | Cash management integration |
| accounts_receivable | Accounts receivable integration |
| customer_relationship_management | CRM integration |
| business_intelligence | Analytics |

## Quick Start

```python
from capabilities.fintech.gateway.service import FintechGatewayService

svc = FintechGatewayService(tenant_id="acme")

# Onboard a merchant
merchant = svc.onboard_merchant(
    merchant_id="m1", tenant_id="acme",
    merchant_code="ACME001", legal_name="Acme Ltd",
    country="KE", risk_level="low",
)

# Connect MPESA provider
provider = svc.connect_provider(
    connection_id="p1", tenant_id="acme",
    provider="mpesa", provider_type="mobile_money",
    credential_reference="vault://mpesa/prod", priority=1,
)

# Create and authorize a payment intent
intent = svc.create_payment_intent(
    intent_id="i1", tenant_id="acme",
    merchant_id=merchant["id"], payment_method_id=method["id"],
    amount=5000, currency="KES",
)
auth = svc.authorize_payment(
    authorization_id="a1", tenant_id="acme",
    payment_intent_id=intent["id"],
    provider_connection_id=provider["id"],
)
capture = svc.capture_payment("c1", "acme", auth["id"], 5000)
```

## Configuration Reference
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| merchants.supported_providers | list | stripe, adyen, mpesa, dpo, flutterwave, pesapal, paypal, manual | Supported payment providers |
| merchants.supported_provider_types | list | card, bank, mobile_money, wallet, settlement, fraud | Provider categories |
| payment_intents.supported_currencies | list | USD, EUR, GBP, KES, ZAR, NGN, GHS, UGX, TZS | Supported currencies |
| disputes.supported_reasons | list | fraud, duplicate, product_not_received, service_not_provided, authorization, processing_error, other | Dispute reason codes |

## API Routes
| Name | Path | Method | Permission | Group |
|------|------|--------|------------|-------|
| dashboard | /fintech-gateway/dashboard | GET | fintech_gateway:view | Overview |
| merchants | /fintech-gateway/merchants | GET/POST | fintech_gateway:manage_merchants | Merchants |
| providers | /fintech-gateway/providers | GET/POST | fintech_gateway:manage_providers | Providers |
| payment_methods | /fintech-gateway/payment-methods | GET/POST | fintech_gateway:manage_payment_methods | Payments |
| payments | /fintech-gateway/payments | GET/POST | fintech_gateway:process | Payments |
| routing | /fintech-gateway/routing | GET/POST | fintech_gateway:route | Operations |
| risk | /fintech-gateway/risk | GET/POST | fintech_gateway:risk | Risk |
| webhooks | /fintech-gateway/webhooks | GET/POST | fintech_gateway:webhooks | Operations |
| settlements | /fintech-gateway/settlements | GET/POST | fintech_gateway:settle | Finance |
| disputes | /fintech-gateway/disputes | GET/POST | fintech_gateway:disputes | Risk |
| agents | /fintech-gateway/agents | GET/POST | fintech_gateway:admin | Automation |
| settings | /fintech-gateway/settings | GET/POST | fintech_gateway:admin | Administration |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| high_risk_merchant_requires_review | Merchant with `risk_level: high` without review | require_review |
| provider_requires_credentials | Provider connection without credential reference | deny |
| payment_intent_amount_positive | Payment intent amount <= 0 | deny |
| high_risk_payment_requires_review | Payment risk `high` without review | require_review |
| blocked_risk_denies_authorization | Payment risk `blocked` | deny |
| high_value_authorization_requires_approval | High-value authorization without approval | require_review |
| capture_blocks_overcapture | Capture amount exceeds authorized amount | deny |
| refund_blocks_overrefund | Refund amount exceeds captured balance | deny |
| large_refund_requires_review | Large refund without review | require_review |
| webhook_requires_signature | Webhook without signature | deny |
| webhook_requires_idempotency | Webhook without idempotency key | deny |
| settlement_variance_requires_review | Settlement variance without review | require_review |
| dispute_resolution_requires_review | Dispute resolution without review | deny |
| gateway_batch_requires_bytewax | Batch without Bytewax | deny |
| gateway_event_requires_bytewax | Event without Bytewax | deny |

## Data Models
| Model | Key Fields |
|-------|-----------|
| Merchant | id, merchant_code, legal_name, country, kyc_reference, risk_level, status |
| ProviderConnection | id, provider, provider_type, credential_reference, priority, status |
| PaymentMethod | id, merchant_id, customer_reference, method_type, token_reference, status |
| PaymentIntent | id, merchant_id, payment_method_id, amount, currency, risk_level, status |
| Authorization | id, payment_intent_id, provider_id, authorized_amount, status |
| Capture | id, authorization_id, capture_amount, status |
| Refund | id, capture_id, refund_amount, status |
| WebhookEvent | id, provider_id, event_id, signature, idempotency_key, payload |
| Settlement | id, provider_id, settlement_reference, amount, variance, status |
| PaymentDispute | id, payment_id, reason, owner_id, resolution_review_reference, status |

## Streaming Events
| Event | Trigger |
|-------|---------|
| merchant_onboarded | Merchant passes onboarding |
| merchant_suspended | Merchant account suspended |
| merchant_reactivated | Suspended merchant reactivated |
| provider_connected | Provider connection established |
| payment_method_tokenized | Payment method tokenized |
| payment_method_deactivated | Payment method deactivated |
| payment_intent_created | Payment intent created |
| payment_intent_voided | Draft intent voided |
| payment_risk_assessed | Risk assessment recorded |
| payment_authorized | Authorization approved |
| payment_captured | Payment captured |
| payment_refunded | Refund processed |
| webhook_ingested | Provider webhook ingested |
| webhook_retried | Failed webhook re-dispatched |
| settlement_recorded | Settlement reconciled |
| payment_dispute_opened | Dispute opened |
| payment_dispute_resolved | Dispute resolved |
| mpesa_stk_push_initiated | M-Pesa STK Push initiated |
| mpesa_b2b_initiated | M-Pesa B2B transfer initiated |
| pesalink_transfer_initiated | PesaLink interbank transfer initiated |
| rtgs_payment_initiated | RTGS high-value payment initiated |
| provider_failover_activated | Provider failover triggered |
| gateway_agent_registered | AI agent registered |

## New Methods

### M-Pesa Rails

```python
# STK Push — customer-facing mobile checkout
push = svc.mpesa_stk_push(
    tenant_id="acme",
    phone="0712345678",       # accepts 07x, 01x, 254x, +254x
    amount=1500.00,
    reference="INV-2026-001",
    description="School fees",
)
# push["status"] == "pending" — reconcile via mpesa_confirm_stk_callback (v2.0)

# B2B till-to-till transfer
b2b = svc.mpesa_b2b_transfer(
    tenant_id="acme",
    sender_till="123456",
    receiver_till="654321",
    amount=50000,
    reference="PAYROLL-JUN",
)
```

### PesaLink and RTGS

```python
# PesaLink interbank (max KES 999,999)
pl = svc.pesalink_transfer(
    tenant_id="acme",
    account_number="1234567890",
    bank_code="011",           # CBK bank code
    amount=250000,
    reference="SUP-PAY-001",
)

# RTGS high-value (min KES 1,000,000 — requires approver)
rtgs = svc.rtgs_payment(
    tenant_id="acme",
    beneficiary_account="9876543210",
    bank_code="012",
    amount=5_000_000,
    reference="BOND-SETTLE",
    approved_by="treasury@acme.co.ke",
)
```

### Provider Failover

```python
# Automatic rerouting of authorized intents from primary to fallback
result = svc.provider_failover(
    tenant_id="acme",
    primary_provider_id="provider-stripe-001",
    fallback_provider_id="provider-flutterwave-001",
)
# result["rerouted"] == count of intents moved to fallback
```

### Settlement Reconciliation

```python
recon = svc.reconcile_settlements(
    tenant_id="acme",
    period_start="2026-06-01",
    period_end="2026-06-30",
)
# recon["status"] in {"balanced", "variance"}
# recon["variance"] is Decimal-safe string
```

### Merchant Lifecycle

```python
# Bulk onboard (partial failure tolerant)
result = svc.bulk_onboard_merchants("acme", [
    {"merchant_code": "M001", "legal_name": "Foo Ltd", "country": "KE"},
    {"merchant_code": "M002", "legal_name": "Bar Inc", "country": "UG", "risk_level": "high", "reviewed_by": "ops"},
])
# result["failed"] lists errors per input; result["processed"] is success count

# Suspend / reactivate
svc.suspend_merchant("merchant-abc", "acme", reason="KYC_expired")
svc.reactivate_merchant("merchant-abc", "acme", reviewed_by="compliance@acme.co.ke")
```

### Regulatory Filing

```python
cbk = svc.cbk_return_filing(
    tenant_id="acme",
    period="2026-Q2",
    return_type="psp_monthly",
    submitted_by="cfo@acme.co.ke",
)
# cbk["status"] == "filed"; includes captured_volume, refunded_volume counts
```

## World-Class Enhancements (v2.0)

The following improvements are planned / in progress for the v2.0 milestone:

1. **Async-First Service Layer** — convert all public methods to `async def`; use `asyncio.gather` for multi-rail fan-out; add `run_sync` shim for legacy callers. Removes latency cliffs at >100 concurrent payments.

2. **Streaming Risk Decisions via LLM (Ollama)** — replace static rule engine for mid-range risk scores (0.4–0.7) with a local Ollama inference call returning `{decision, confidence, rationale}` in <200 ms. Low/blocked cases stay on the fast path.

3. **Multi-Rail Routing with Fallback Graph** — replace single `provider_connection_id` with a prioritized rail graph `[(mpesa, 1), (pesalink, 2), (rtgs, 3)]`; `authorize_payment` walks the graph automatically with attempt history.

4. **Idempotency Registry with TTL** — bounded LRU cache shared across all write operations; any mutating method accepts `idempotency_key`; cached response returned within 24 h TTL.

5. **Decimal-Safe Amount Arithmetic** — introduce `Amount` newtype (Decimal subclass + `Amount.of(v)` factory); all arithmetic and comparisons go through it; catches float/Decimal mixing at construction.

6. **MPESA Callback Reconciliation Loop** — `mpesa_confirm_stk_callback(tenant_id, checkout_request_id, result_code, mpesa_receipt_number)` advances pending intents to `captured` or `failed` and emits `mpesa_stk_confirmed`.

7. **Equity Bank & KCB EFT Integration** — add `equity_eft_transfer` and `kcb_connect_transfer` with bank-specific validations (account format, daily limit, branch code) so the multi-rail router can dispatch to these rails.

8. **Webhook Signature Verification** — `verify_webhook_signature(provider, raw_body, signature, secret_reference)` with provider-specific HMAC (Stripe: sha256, MPESA: CBK scheme, Flutterwave: sha512); gates `ingest_webhook` on result.

9. **Composite Settlement Reconciliation with Break Detection** — line-item matching per capture; flags `unmatched_captures` and `phantom_credits` (high-priority fraud vector on mobile money rails).

10. **Merchant Risk Scoring Pipeline** — `score_merchant_risk(merchant_id, tenant_id)` computes dynamic score from FATF tier, MCC, monthly volume, dispute rate, chargebacks; overrides static `risk_level` if computed score is higher.

11. **Payment Intent State Machine Enforcement** — `INTENT_TRANSITIONS` graph + `_assert_valid_transition(current, target)` guard on every mutating method; eliminates regression bugs like `captured → authorized`.

12. **Regulatory Capital Reporting (CBK PSP Returns)** — `CbkPspReturn` Pydantic model with full field coverage: volumes by payment type, cross-border flows by currency, settlement accounts, CDD counts; `validate_cbk_return` checks completeness before filing.

13. **Tenant-Level Rate Limiting and Quota Enforcement** — `TenantQuota` model with `max_intents_per_minute`, `max_capture_volume_per_day`, `max_merchants`; enforced in `_assert_rules` via context window counts.

14. **Async Audit Trail with Structured Schema** — `AuditEvent` Pydantic model with `correlation_id`, `causation_id`, `actor_type`, `actor_id`, `ip_address`, `before_state`, `after_state`; emitted via `asyncio.Queue` (off critical path).

15. **Payment Analytics Aggregation Engine** — `TenantMetrics` dataclass updated incrementally by `_emit`; `dashboard_summary` becomes O(1) instead of O(n) linear scan; eliminates latency spikes at >10k intents per tenant.

## Edge Cases Handled
- `blocked` risk level produces a hard deny on authorization; no override path exists
- Capture validates against authorized amount; partial captures allowed; over-captures blocked
- Refund validates against remaining captured balance; over-refunds blocked at rule engine level
- Webhook idempotency keys prevent duplicate event processing; missing key rejected even with valid signature
- Dispute resolution requires a recorded review; no auto-resolution path
- Settlement amounts use `_lt` suffix (`amount < 0`); zero-amount settlements are valid reconciliation entries
- RTGS requires `approved_by`; PesaLink enforces KES 999,999 ceiling; MPESA validates phone prefix

## Composability
- **Upstream**: `fintech_payments` routes payment orders to the gateway; `fintech_wallets` provides wallet payment methods; `fintech_kyc` provides merchant KYC evidence
- **Downstream**: `cash_management` and `accounts_receivable` receive settlement and reconciliation data; `business_intelligence` consumes gateway event data
- **Peer**: Deployed alongside `fintech_payments` (application-facing payment layer) and `fintech_fraud` (risk scoring before authorization)

## Development Notes
- Version 2.1.0 reflects a more mature capability with additional provider types and routing features vs 1.1.0 for other caps
- `composition_events` and `composition_config` are platform-level dependencies (differ from the `auth`/`audl` pattern in other caps)
- `_lte` / `_lt` condition suffixes enable numeric comparisons: `amount_lte: 0` fires when amount <= 0, `settlement_amount_lt: 0` fires when amount < 0
- `max_autonomous_scope` for gateway agents is `recommend_validate_and_prepare` — agents cannot execute payments; they can only prepare and validate
- Aliases `GatewayService` and `PaymentGatewayService` are available for backward compatibility
