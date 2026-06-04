# Embedded Finance

## Overview
Embedded Finance enables non-financial businesses to offer financial products inside their own applications without owning banking infrastructure. It manages partner program onboarding, host application registration, product placement publishing, customer consent capture, and the end-to-end lifecycle of embedded accounts, payments, card offers, lending offers, settlement batches, and revenue share — all within a consent-scoped access model.

Every embedded financial journey requires an active consent grant scoped to the product and channel. Payment scope mismatch between consent and request is a hard deny. Revenue share percentages are bounded 0–100%. All embedded finance events stream to `apg.fintech.embedded.lifecycle` via Bytewax.

## Capability ID
`fintech_embedded`  Version: 1.1.0

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

## Edge Cases Handled
- Consent scope coverage is checked at payment initiation — a payment requesting `payments.write` scope fails if the active consent only grants `payments.read`; the scope mismatch produces a hard deny
- Placement-application consistency: a payment cannot use a placement that belongs to a different host application than the one in the payment request
- Lending offers require both affordability AND underwriting evidence — one without the other is rejected; this prevents offers based solely on affordability without underwriting sign-off
- Revenue share of exactly 0% and exactly 100% are both valid edge cases; the rule only rejects values outside the closed interval [0, 100]
- Host applications in sandbox environment cannot be promoted to production without a new registration; environment mismatch at placement time is rejected

## Composability
- **Upstream**: `fintech_apis` provides the API product layer that partner applications consume; `fintech_kyc` and `fintech_aml` provide identity and compliance evidence; `fintech_fraud` provides checkout-level fraud scoring
- **Downstream**: `fintech_payments`, `fintech_wallets`, `fintech_cards`, `fintech_lending`, and `fintech_bnpl` are the product capabilities that placements surface to end users
- **Peer**: Commonly deployed with `fintech_apis` (API access control) and `fintech_mobile` (mobile-first partner channels)

## Development Notes
- Partner program KYB is distinct from end-customer KYC — the program-level KYB verifies the partner entity; customer-level KYC happens when the customer opens an embedded account
- The `payment_consent_covers_scope` rule checks that the active consent includes the specific scope required by the payment operation — this requires the service layer to resolve scope coverage before setting the context flag
- Product placements are scoped to a specific application; a placement registered for application A cannot be used by application B even within the same partner program
- Revenue share tracking is informational — the capability records the share percentage and period but does not execute the actual payment; revenue share disbursement is handled by `fintech_payments`
