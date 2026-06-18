# Digital Cards

## Overview
Digital Cards (`fintech_cards`) provides executable card issuing and operations workflows: program governance, cardholder onboarding, virtual and physical card issuance, token provisioning (wallet, device, merchant, network tokens), authorization decisions with fraud and AML controls, spend control enforcement, 3DS challenge flows, dispute intake, and card analytics. It is the issuing layer that sits between a payment wallet and the card network.

Physical cards require a shipping address; virtual cards require a wallet reference. All card operations are linked to a funding account and cardholder consent. High-impact authorizations trigger a require-review gate. Events stream to `apg.fintech.cards.lifecycle` via Bytewax.

## Capability ID
`fintech_cards`  Version: 2.0.0

## Features
- Virtual and physical card issuance with masked PAN generation
- Token provisioning: wallet, device, merchant, network (Apple Pay, Google Pay, Samsung Pay)
- Real-time authorization decisions with per-card spend controls (daily/monthly limits, blocked categories, allowed countries)
- 3DS challenge initiation and OTP verification
- Card lifecycle management: activate, block, unblock, freeze, replace
- Dispute filing and resolution with evidence tracking
- Card statement and spending insights per period
- M-Pesa card linking for mobile money top-ups
- International usage controls per region
- Contactless (NFC) enable/disable
- Loyalty/rewards points calculation
- Bulk card issuance
- Card analytics and program-level reporting
- AI-powered fraud scoring via Ollama (when `OLLAMA_BASE_URL` is set)
- Audit trail on every mutating operation

## Provides
| Service | Description |
|---------|-------------|
| card_program_governance | Register card programs with BIN range, supported currencies, and settlement account |
| cardholder_card_lifecycle | Onboard cardholders with KYC; issue virtual and physical cards |
| tokenized_card_credentialing | Provision wallet, device, merchant, and network tokens with key domain references |
| card_authorization_control | Evaluate authorizations with fraud decision, AML result, currency, and category checks |
| card_dispute_workflow | File and manage card disputes with evidence and reviewer assignment |
| card_agent_workflow | Register AI agents for authorization review, token governance, and dispute handling |
| card_spend_controls | Set per-card daily/monthly limits, blocked categories, and geo restrictions |
| card_3ds | Initiate and verify 3DS OTP challenges |
| card_analytics | Aggregate spend analytics and approval rates per period |

## Requires
| Capability | Purpose |
|------------|---------|
| auth | Authentication |
| audl | Audit trail |
| ntfy | Cardholder and operations notifications |
| nlpc | NLP processing |
| keym | Key management |
| encr | Encryption for card data |
| fintech_payments | Payment authorization and settlement |
| fintech_wallets | Wallet backing and hold management |
| fintech_kyc | Cardholder identity verification |
| fintech_aml | AML screening |
| fintech_fraud | Fraud signal scoring per authorization |

## Quick Start

```python
import asyncio
from capabilities.fintech.cards.service import DigitalCardsService

svc = DigitalCardsService(tenant_id="acme", actor_id="ops-bot")

async def main():
    # Issue a virtual card
    card = await svc.issue_virtual_card("cust-001", spend_limit=50_000.0, currency="KES")
    card_id = card["card_id"]

    # Activate it
    await svc.activate_card(card_id)

    # Set spend controls
    await svc.set_spend_controls(card_id, {
        "daily_limit": 10_000.0,
        "blocked_categories": ["gambling"],
    })

    # Process a purchase
    txn = await svc.process_card_transaction(card_id, "Naivas Supermarket", 1_500.0, "KES")
    print(txn)

asyncio.run(main())
```

## API Routes
| Name | Path | Method | Permission | Group |
|------|------|--------|------------|-------|
| dashboard | /fintech-cards/dashboard | GET | fintech_cards:view | Overview |
| programs | /fintech-cards/programs | GET/POST | fintech_cards:manage_programs | Programs |
| cardholders | /fintech-cards/cardholders | GET/POST | fintech_cards:manage_cardholders | Cards |
| cards | /fintech-cards/cards | GET/POST | fintech_cards:issue | Cards |
| tokens | /fintech-cards/tokens | GET/POST | fintech_cards:tokenize | Tokens |
| authorizations | /fintech-cards/authorizations | GET/POST | fintech_cards:authorize | Controls |
| disputes | /fintech-cards/disputes | GET/POST | fintech_cards:dispute | Exceptions |
| agents | /fintech-cards/agents | GET/POST | fintech_cards:admin | Automation |
| settings | /fintech-cards/settings | GET/POST | fintech_cards:admin | Administration |

## Configuration Reference
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| cards.supported_types | list | virtual, physical | Card form factors |
| cards.supported_products | list | debit, prepaid, expense, fleet, merchant | Card product types |
| tokens.supported_types | list | wallet, device, merchant, network | Token use cases |
| authorizations.high_value_threshold | number | 100000 | Authorization amount requiring approval |
| authorizations.supported_fraud_decisions | list | clear, review, hold, block | Valid fraud decision values |
| authorizations.supported_aml_results | list | clear, review, blocked | Valid AML result values |

## New Methods

### `issue_virtual_card` — issue and configure in one call
```python
card = await svc.issue_virtual_card(
    customer_id="cust-42",
    spend_limit=25_000.0,
    currency="KES",
)
# Returns: card_id, masked_pan, expiry, status="inactive"
await svc.activate_card(card["card_id"])
```

### `process_card_transaction` — enforces spend controls inline
```python
result = await svc.process_card_transaction(
    card_id="vc-cust-42-a1b2",
    merchant="Shell Petrol Station",
    amount=3_000.0,
    currency="KES",
)
# Returns: {status: "approved"|"declined", decline_reason?, transaction_id}
```

### `card_3ds_challenge` / `verify_3ds_challenge` — 3DS flow
```python
challenge = await svc.card_3ds_challenge(card_id, transaction_id="txn-001")
# In prod: OTP sent via SMS, not returned. In dev: challenge["otp_hint"] available before pop.

result = await svc.verify_3ds_challenge(
    challenge_id=challenge["challenge_id"],
    otp_provided="A3F9C1",
)
# Returns: {verified: True|False, status: "verified"|"failed"|"expired"}
```

### `card_spending_insights` — per-card category breakdown
```python
insights = await svc.card_spending_insights(card_id="vc-cust-42-a1b2", period="2025-06")
# Returns: total_spend, transaction_count, top_merchants list
```

### `mpesa_card_link` — M-Pesa integration
```python
link = await svc.mpesa_card_link(card_id="vc-cust-42-a1b2", mpesa_number="0712345678")
# Returns: {card_id, mpesa_number (last 9 digits), status: "active"}
```

### `bulk_issue_cards` — batch provisioning
```python
result = await svc.bulk_issue_cards([
    {"customer_id": "cust-1", "spend_limit": 10_000, "currency": "KES"},
    {"customer_id": "cust-2", "spend_limit": 50_000, "currency": "USD"},
])
# Returns: {processed: N, failed: M, card_ids: [...]}
```

## World-Class Enhancements (v2.0)

These 15 improvements define the production-readiness roadmap. Priority order matches the list footnote in `WORLD_CLASS_IMPROVEMENTS.md`.

| # | Enhancement | Impact |
|---|------------|--------|
| 1 | **EMV-compliant PAN/CVV generation** — replace SHA-256 stub with 3DES TDEA over a service master key binding PAN + expiry + service code | PCI-DSS Level 1 compliance; eliminates card data audit findings |
| 2 | **HSM Adapter** — pluggable `HSMAdapter` with `softhsm` (dev), `thales`/`aws_cloudhsm` (prod) backends; all key ops route through it | Satisfies PCI-PIN key management; mandatory for live card issuing |
| 3 | **EMV Chip Personalisation Pipeline** — `personalise_emv_card()` generating AC seed, personalisation script, ARQC verification key set + bureau adapter | Enables physical card programs; required for EMV terminal auth |
| 4 | **Network Token Lifecycle (EMV Payment Tokenisation)** — TSP registration, domain-restricted DPAN issuance, TAVV per transaction, suspension/deletion with scheme notification | Required for Apple Pay / Google Pay live integration |
| 5 | **Redis Velocity & Spend Control Engine** — replace in-memory dict with `INCRBYFLOAT` TTL windows (per-minute, hourly, daily, monthly) with sliding counters | Prevents midnight-boundary gaming and velocity fraud patterns |
| 6 | **3DS 2.x Full Protocol** — ACS challenge, CReq/CRes exchange, CAVV/AAV generation, device fingerprinting for frictionless-flow risk gating | Shifts CNP fraud liability to merchant/acquirer |
| 7 | **Async Event Streaming (Bytewax / Bytewax)** — typed `CloudEvent` records on `apg.fintech.cards.lifecycle` with transactional outbox pattern | Enables real-time downstream reaction in `fintech_payments`, `fintech_wallets` |
| 8 | **Idempotency Keys** — `idempotency_key` UUID per mutating call, 24-hour response cache via Redis `SET NX EX` | Prevents double-issuance and duplicate authorizations on retries |
| 9 | **Card Lifecycle State Machine** — `inactive → active → frozen → active → blocked → [terminal]` with enforced transition graph | Prevents invalid state combinations; simplifies compliance attestation |
| 10 | **PostgreSQL-Backed Store + Alembic Migrations** — wire existing `DatabaseStore` skeleton to `asyncpg` pools; replace in-memory dicts | Survives restarts; enables horizontal scaling |
| 11 | **Fraud Score Feedback Loop** — store `(txn_id, features, score, actual_outcome)` to `fraud_feedback` table; expose `retrain_fraud_model()` | Closes model drift loop; fraud models degrade without live retraining |
| 12 | **Dispute SLA Timers + Escalation** — `dispute_sla_deadline` (Visa/MC 45-day), T-7d/T-1d escalation events via `ntfy`, win/loss rate tracking | Prevents automatic cardholder wins and scheme fines |
| 13 | **Multi-Currency Settlement + FX Engine** — store original currency + exchange rate + settlement currency; DCC markup computation; ECB feed or `fintech_fx` adapter | Reliable international spend controls and cross-border reporting |
| 14 | **PostgreSQL Row-Level Security** — RLS policies on all card tables; per-tenant role or `SET app.current_tenant` before every query | Database-enforced tenancy; application-level checks alone are one bug from a data breach |
| 15 | **OpenTelemetry Distributed Tracing** — spans on every public async method with `card_id`, `tenant_id`, `operation`, `decision` attributes; OTLP export; `trace_id` in all audit events | Reduces MTTR from hours to minutes across fraud + AML + auth + 3DS services |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| program_bin_range_required | Program without BIN range | deny |
| cardholder_kyc_required | Cardholder without KYC evidence | deny |
| physical_shipping_required | Physical card without shipping address | deny |
| wallet_reference_required | Card issuance without wallet reference | deny |
| card_consent_required | Card issuance without consent evidence | deny |
| token_key_domain_required | Token without key domain reference | deny |
| fraud_block_denies_authorization | Fraud decision is `block` | deny |
| aml_block_denies_authorization | AML result is `blocked` | deny |
| high_impact_authorization_requires_approval | High-impact authorization without approval | require_review |
| dispute_evidence_required | Dispute without evidence | deny |
| card_batch_requires_bytewax | Batch without Bytewax | deny |

## Data Models
| Model | Key Fields |
|-------|-----------|
| CardProgram | id, name, owner_id, bin_range, currency, settlement_account, status |
| Cardholder | id, customer_reference, kyc_profile_id, country, status |
| Card | id, program_id, cardholder_id, card_type, product, wallet_reference, funding_account, masked_pan, status |
| CardToken | id, card_id, token_type, token_reference, key_domain, device_or_merchant_reference |
| CardAuthorization | id, card_id, amount, currency, merchant_category, fraud_decision, aml_result, status |
| CardDispute | id, transaction_reference, reason, evidence_references, reviewer_id, status |

## Streaming Events
Events emitted to the fintech event stream via Bytewax.
| Event | Trigger |
|-------|---------|
| card_program_registered | New card program created |
| cardholder_onboarded | Cardholder enrolled |
| card_issued | Card issued to cardholder |
| virtual_card_issued | Virtual card issued via async path |
| card_activated | Card activated |
| card_blocked | Card blocked |
| card_unblocked | Card unblocked |
| card_replaced | Card replaced (lost/stolen/damaged) |
| card_token_provisioned | Token provisioned for card |
| card_tokenised | Card tokenised for a digital wallet |
| card_authorization_decided | Authorization evaluated |
| card_3ds_challenge_initiated | 3DS challenge started |
| card_3ds_verified | 3DS OTP verified |
| card_transaction_processed | Purchase processed via spend-control path |
| card_dispute_filed | Dispute filed |
| card_dispute_resolved | Dispute resolved |
| spend_controls_set | Spend controls updated |
| mpesa_card_linked | Card linked to M-Pesa number |
| international_enabled | International usage enabled |
| card_agent_registered | AI agent registered |

## Edge Cases Handled
- Fraud `block` and AML `blocked` decisions both independently deny authorization
- Physical card issuance requires a shipping address; virtual does not
- Token provisioning requires device OR merchant reference (not both)
- Card consent is distinct from cardholder KYC — separate evidence attachment required
- Authorization currency must be in `SUPPORTED_CURRENCIES`; checked before fraud/AML
- Blocked cards cannot be activated or have PIN changed
- 3DS challenges expire after 5 minutes; expired challenges return `verified: false` without error

## Composability
- **Upstream**: `fintech_kyc` provides cardholder identity; `fintech_fraud` and `fintech_aml` provide per-authorization signals
- **Downstream**: `fintech_payments` handles settlement; `fintech_wallets` provides wallet holds; `fintech_mobile` uses card linking for mobile payment flows
- **Peer**: Deployed alongside `fintech_wallets` (wallet-backed funding) and `fintech_neobanking` (debit cards for neobank customers)
- **M-Pesa**: `mpesa_card_link` bridges card issuing to East African mobile money rails

## Development Notes
- The `high_impact` flag in authorization context is caller-computed — set `high_impact: True` when amount > `high_value_threshold`
- BIN range is stored as a string (e.g. "411111"); uniqueness enforcement is the service layer's responsibility
- `encr` (encryption capability) is a required dependency; card data must never traverse the system unencrypted
- Token types `wallet` and `device` require a key domain reference; `merchant` and `network` additionally require device or merchant reference
- `CardService` is an alias for `DigitalCardsService` for backward compatibility
- `ml_card_fraud_score` is a no-op unless `OLLAMA_BASE_URL` is set in the environment

---
© 2025 Datacraft | Author: Nyimbi Odero | www.datacraft.co.ke
