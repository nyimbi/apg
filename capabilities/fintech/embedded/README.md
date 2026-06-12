# Embedded Finance

## Overview
Embedded Finance enables non-financial businesses to offer financial products inside their own applications without owning banking infrastructure. It manages partner program onboarding, host application registration, product placement publishing, customer consent capture, and the end-to-end lifecycle of embedded accounts, payments, card offers, lending offers, settlement batches, and revenue share — all within a consent-scoped access model.

Every embedded financial journey requires an active consent grant scoped to the product and channel. Payment scope mismatch between consent and request is a hard deny. Revenue share percentages are bounded 0–100%. All embedded finance events stream to `apg.fintech.embedded.lifecycle` via Bytewax.

## Capability ID
`fintech_embedded`  Version: 2.0.0

## Provides
| Service | Description |
|---------|-------------|
| partner_program_workflow | Onboard partner programs with KYB, contract, and risk review evidence |
| host_application_workflow | Register host applications with domain, environment, and terms controls |
| embedded_product_placement_workflow | Publish product placements with scope, channel, and risk policy |
| embedded_customer_consent_workflow | Capture scoped customer consent with expiry |
| embedded_account_workflow | Open embedded accounts with KYC and wallet references |
| embedded_payment_workflow | Initiate embedded payments with placement, consent scope, and risk reference |
| embedded_card_workflow | Offer embedded cards with positive limit and risk reference |
| embedded_lending_workflow | Create lending offers with affordability and underwriting evidence |
| embedded_settlement_workflow | Close settlement batches with reconciliation and positive amount controls |
| embedded_revenue_share_workflow | Record revenue share with program reference and bounded percentage |
| embedded_finance_agent_workflow | Register AI agents for partner risk, consent, and settlement review |
| partner_onboarding_workflow | Multi-step KYB onboarding with checklist and document upload |
| white_label_wallet_workflow | Provision white-label wallets with virtual account numbers |
| embedded_lending_origination | Originate embedded loans with affordability scoring and repayment schedule |
| embedded_insurance_workflow | Issue embedded insurance policies (life, health, credit, device, travel, micro) |
| compliance_paas_workflow | KYC/AML/PEP/sanctions check with risk decision and recommended action |
| partner_analytics_workflow | Reconciliation reports, revenue share calculation, and API usage analytics |
| webhook_management_workflow | Register and manage partner webhook subscriptions with signing secrets |

## Requires
| Capability | Purpose |
|------------|---------|
| auth | Authentication |
| audl | Audit trail |
| ntfy | Partner and customer notifications |
| nlpc | NLP processing |
| keym | Key management |
| fintech_apis | API product access layer |
| fintech_payments | Payment execution |
| fintech_wallets | Wallet management for accounts |
| fintech_cards | Card product offering |
| fintech_lending | Lending product offering |
| fintech_bnpl | BNPL product offering |
| fintech_kyc | Customer identity verification |
| fintech_aml | AML screening |
| fintech_fraud | Fraud screening |
| fintech_mobile | Mobile channel integration |

## Quick Start

```python
import asyncio
from capabilities.fintech.embedded.service import EmbeddedFinanceService

svc = EmbeddedFinanceService(tenant_id="acme", actor_id="ops-bot")

async def main():
    # 1. Onboard a partner
    onboarding = await svc.partner_onboarding(
        partner_id="paygo",
        business_details={
            "legal_name": "PayGo Ltd",
            "registration_number": "CPR/2024/001",
            "country": "KE",
            "contact_email": "tech@paygo.co.ke",
        },
        integration_type="sdk",
    )

    # 2. Provision a white-label wallet
    wallet = await svc.white_label_wallet("paygo", "cust_001", "KES")

    # 3. Originate a loan
    loan = await svc.embedded_lending("paygo", "cust_001", 50_000_00)

    print(loan["offer_id"], loan["monthly_payment_minor"])

asyncio.run(main())
```

## Configuration Reference
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| placements.supported_products | list | accounts, wallet, payments, cards, loans, bnpl, remittance, insurance, marketplace_finance | Embeddable products |
| placements.supported_channels | list | checkout, marketplace, mobile_app, web_app, pos, agent, api | Placement channels |
| payments.supported_currencies | list | USD, KES, EUR, GBP, NGN, GHS, ZAR | Supported currencies |
| revenue_share.minimum_percent | number | 0 | Minimum revenue share % |
| revenue_share.maximum_percent | number | 100 | Maximum revenue share % |

## API Routes
| Name | Path | Method | Permission | Group |
|------|------|--------|------------|-------|
| dashboard | /fintech-embedded/dashboard | GET | fintech_embedded:view | Overview |
| programs | /fintech-embedded/programs | GET/POST | fintech_embedded:programs | Partners |
| applications | /fintech-embedded/applications | GET/POST | fintech_embedded:applications | Partners |
| placements | /fintech-embedded/placements | GET/POST | fintech_embedded:placements | Products |
| consents | /fintech-embedded/consents | GET/POST | fintech_embedded:consents | Consent |
| accounts | /fintech-embedded/accounts | GET/POST | fintech_embedded:accounts | Journeys |
| payments | /fintech-embedded/payments | GET/POST | fintech_embedded:payments | Journeys |
| cards | /fintech-embedded/cards | GET/POST | fintech_embedded:cards | Journeys |
| lending | /fintech-embedded/lending | GET/POST | fintech_embedded:lending | Journeys |
| settlements | /fintech-embedded/settlements | GET/POST | fintech_embedded:settlements | Operations |
| revenue_share | /fintech-embedded/revenue-share | GET/POST | fintech_embedded:revenue_share | Operations |
| agents | /fintech-embedded/agents | GET/POST | fintech_embedded:admin | Automation |
| settings | /fintech-embedded/settings | GET/POST | fintech_embedded:admin | Administration |

## New Methods

### `partner_onboarding` — multi-step KYB with checklist

```python
result = await svc.partner_onboarding(
    partner_id="mchanga",
    business_details={
        "legal_name": "Mchanga Finance Ltd",
        "registration_number": "PVT/2025/042",
        "country": "KE",
        "contact_email": "api@mchanga.co.ke",
        "industry": "microfinance",
    },
    integration_type="white_label",
)
# result["onboarding_checklist"] — 6-step KYB workflow with references
# result["kyb_reference"], result["contract_reference"], result["risk_reference"]
```

### `white_label_wallet` — virtual account provisioning

```python
wallet = await svc.white_label_wallet(
    partner_id="mchanga",
    customer_id="cust_ke_00123",
    currency="KES",
)
# wallet["account_number"] — APG-prefixed virtual account number
# wallet["wallet_id"], wallet["kyc_tier"] = "tier_1"
```

### `embedded_lending` — affordability scoring + repayment schedule

```python
loan = await svc.embedded_lending(
    partner_id="mchanga",
    customer_id="cust_ke_00123",
    amount=100_000_00,  # KES 100,000 in minor units
)
# loan["risk_tier"] in ("low", "medium", "high")
# loan["interest_rate_pa"], loan["monthly_payment_minor"]
# loan["repayment_schedule"] — 12-month annuity schedule
```

### `compliance_paas_check` — KYC/AML/sanctions decision

```python
check = await svc.compliance_paas_check(
    partner_id="mchanga",
    event={
        "type": "transaction_monitoring",
        "customer_reference": "cust_ke_00123",
        "amount_minor": 5_000_000,
    },
)
# check["decision"] in ("allow", "monitor", "review", "block")
# check["pep_hit"], check["sanctions_hit"], check["aml_flag"]
# check["recommended_action"]
```

### `embedded_analytics_dashboard` — unified partner analytics

```python
analytics = await svc.embedded_analytics_dashboard(
    partner_id="mchanga",
    period="2026-05",
)
# analytics["reconciliation"] — payment vs settlement gap + discrepancy flag
# analytics["revenue"] — gross share, platform fee, net payable
# analytics["api_usage"] — call counts, success rate, quota utilisation
```

## World-Class Enhancements (v2.0)

1. **ISO 20022 Payment Rails** — Structured pacs.008/camt.054 messages for SWIFT, SEPA, and RTP straight-through processing with end-to-end reconciliation references.
2. **Dynamic Consent Lifecycle** — Hierarchical consent tree with time-bounded, amount-bounded, and counter-bounded grants replacing the flat scope list; per-operation challenge at authorization.
3. **Waterfall Revenue Share** — Configurable DAG-based waterfall engine: platform fee first, tiered partner share by volume, sub-partner splits for marketplace and aggregator models.
4. **Federated Learning Credit Scoring** — Local model training on partner data; only gradient updates shared; risk tiers and interest rates continuously calibrated across the network.
5. **Virtual IBAN / VAN Pooling** — Pooled namespace for BBAN/IBAN/BIC derivation; inbound funds auto-routed to the correct embedded account; eliminates manual reconciliation.
6. **Event-Sourced Ledger with CQRS** — Every state transition is an immutable domain event; read models are projections rebuilt from the append-only log; point-in-time account reconstruction.
7. **PCI-DSS Level 1 Card Data Vault** — PAN/CVV/expiry tokenized in a CDV; service layer handles only vault tokens; supports network tokenization for Apple Pay / Google Pay.
8. **Multi-Party Computation Fraud Detection** — MPC across partner networks for joint fraud signal computation without raw data exposure; detects mule accounts and velocity patterns consortium-wide.
9. **Embedded BNPL Origination** — 3- and 6-installment plans, merchant-funded vs. lender-funded interest, checkout widget, deferred settlement with separate underwriting and chargeback rules.
10. **Regulatory Reporting Automation** — Nightly IFRS 9 ECL staging (Stage 1/2/3); CBK prudential ratios from settlement data; XBRL/CSV output with filing endpoint acknowledgement.
11. **Open Banking API Gateway (PSD2/FAPI 2.0)** — AIS/PIS endpoints with PAR, RAR, TPP client registration, token introspection, and consent dashboard; plug-in for third-party providers.
12. **Adaptive Rate Limiting** — Multi-dimensional token bucket per partner/endpoint/IP/customer; quotas configurable at onboarding; proactive notifications at 50/80/95/100% consumption.
13. **Embedded Savings & Investments** — Fixed-deposit accounts, unit trust / money market subscriptions, goal-based savings with auto round-up; auto-debit authorization via consent model.
14. **Cross-Border Remittance Corridors** — Correspondent bank routing per corridor, FX rate locking, OFAC/UN sanctions pre-checks, delivery SLAs, ISO 20022 pacs.008 instruction generation.
15. **OpenTelemetry Observability + SLO Dashboard** — Spans with tenant/partner/operation attributes, payment throughput and consent grant rate metrics, SLO burn-rate dashboard via Prometheus + Grafana.

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| program_kyb_required | Partner program without KYB | deny |
| program_contract_required | Partner program without contract | deny |
| placement_scopes_required | Placement without scopes | deny |
| placement_risk_policy_required | Placement without risk policy | deny |
| payment_consent_covers_scope | Payment scope not covered by consent | deny |
| payment_placement_matches_application | Payment placement belongs to different application | deny |
| card_positive_limit | Card offer with zero or negative limit | deny |
| lending_affordability_required | Lending offer without affordability evidence | deny |
| settlement_reconciliation_required | Settlement batch without reconciliation | deny |
| revenue_share_percent_bounded | Revenue share outside 0–100% | deny |
| embedded_batch_requires_bytewax | Batch without Bytewax | deny |

## Data Models
| Model | Key Fields |
|-------|-----------|
| PartnerProgram | id, name, kyb_reference, contract_reference, risk_reference, status |
| HostApplication | id, program_id, name, environment, domain, terms_reference, status |
| ProductPlacement | id, application_id, product, channel, scopes, risk_policy_reference, status |
| CustomerConsent | id, application_id, customer_reference, scopes, expiry_date |
| EmbeddedAccount | id, application_id, kyc_reference, wallet_reference, status |
| EmbeddedPayment | id, application_id, placement_id, consent_id, amount, currency, risk_reference, status |
| EmbeddedCardOffer | id, application_id, limit, risk_reference |
| EmbeddedLendingOffer | id, application_id, affordability_reference, underwriting_reference |
| SettlementBatch | id, reconciliation_reference, amount, status |
| RevenueShare | id, program_id, percent, period |

## Streaming Events
Events emitted to the fintech event stream via Bytewax.
| Event | Trigger |
|-------|---------|
| partner_program_registered | Partner program onboarded |
| host_application_registered | Host application registered |
| product_placement_published | Product placement activated |
| customer_consent_captured | Customer consent recorded |
| embedded_account_opened | Account opened in partner app |
| embedded_payment_initiated | Payment initiated via placement |
| embedded_card_offered | Card offer created |
| embedded_lending_offer_created | Lending offer created |
| settlement_batch_closed | Settlement batch closed |
| revenue_share_recorded | Revenue share recorded |
| embedded_agent_registered | AI agent registered |
| partner_onboarding_initiated | Partner KYB onboarding started |
| white_label_wallet_provisioned | White-label wallet issued |
| embedded_lending_originated | Lending offer originated via partner |
| embedded_insurance_issued | Insurance policy issued |
| compliance_paas_check_run | Compliance event evaluated |
| payment_widget_embedded | Payment widget generated |
| webhook_subscriptions_updated | Partner webhook configuration changed |
| revenue_share_calculated | Revenue share computation completed |

## Edge Cases Handled
- Consent scope coverage is checked at payment initiation — a payment requesting `payments.write` scope fails if the active consent only grants `payments.read`; the scope mismatch produces a hard deny
- Placement-application consistency: a payment cannot use a placement that belongs to a different host application than the one in the payment request
- Lending offers require both affordability AND underwriting evidence — one without the other is rejected
- Revenue share of exactly 0% and exactly 100% are both valid; the rule only rejects values outside the closed interval [0, 100]
- Host applications in sandbox environment cannot be promoted to production without a new registration
- Compliance PaaS decisions escalate from allow → monitor → review → block based on composite risk score, PEP hit, and sanctions match
- Revenue share net payable floored at zero: `max(gross_share - platform_fee, 0)`

## Composability
- **Upstream**: `fintech_apis` provides the API product layer; `fintech_kyc` and `fintech_aml` provide identity and compliance evidence; `fintech_fraud` provides checkout-level scoring
- **Downstream**: `fintech_payments`, `fintech_wallets`, `fintech_cards`, `fintech_lending`, and `fintech_bnpl` are the product capabilities surfaced via placements
- **Peer**: Commonly deployed with `fintech_apis` (API access control) and `fintech_mobile` (mobile-first partner channels)

## Development Notes
- Partner program KYB is distinct from end-customer KYC — the program-level KYB verifies the partner entity; customer-level KYC happens when the customer opens an embedded account
- The `payment_consent_covers_scope` rule checks that the active consent includes the specific scope required — the service layer resolves scope coverage before setting the context flag
- Product placements are scoped to a specific application; a placement registered for application A cannot be used by application B even within the same partner program
- Revenue share tracking is informational — the capability records the share percentage and period but does not execute disbursement; revenue share payments are handled by `fintech_payments`
- `EmbeddedService` is an alias for `EmbeddedFinanceService` for backward compatibility
