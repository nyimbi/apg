# APG Agency Banking Capability Specification

## Purpose

The Agency Banking capability turns third-party agent networks into a
first-class APG fintech package. It lets generated applications compose bank,
wallet, payment, remittance, and lending services through accredited agents,
outlets, float accounts, cash movements, commission settlement, supervision, and
dispute workflows.

The executable core runs locally without provider credentials. Live core
banking ledgers, mobile money operators, POS estates, cash-in-transit partners,
field-force tools, regulator filing, and durable Bytewax workers stay behind
adapter boundaries.

## Functional Scope

The package provides:

- agency program governance;
- outlet onboarding and accreditation;
- individual teller/agent accreditation;
- agent float account control;
- simplified customer onboarding;
- cash-in, cash-out, transfers, bill payments, airtime, loan collection,
  loan disbursement, account opening, balance inquiry, mini-statement, card
  services, insurance, savings, and government-payment transactions;
- cash movement and liquidity rebalancing;
- commission settlement;
- dispute intake and evidence tracking;
- field supervision and remediation;
- provider-neutral AI-agent composition for operations, liquidity, compliance,
  dispute, settlement, fraud, and field-supervision review.

## Composition Contract

Capability id: `fintech_agency`

Provides:

- `agency_program_governance`
- `agency_outlet_lifecycle`
- `agency_agent_accreditation`
- `agency_float_management`
- `agency_customer_workflow`
- `agency_transaction_workflow`
- `agency_cash_movement_workflow`
- `agency_commission_settlement_workflow`
- `agency_dispute_workflow`
- `agency_supervision_workflow`
- `agency_ai_agent_workflow`

Requires:

- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `keym`
- `fintech_payments`
- `fintech_wallets`
- `fintech_cards`
- `fintech_kyc`
- `fintech_aml`
- `fintech_fraud`
- `fintech_remittance`
- `fintech_neobanking`
- `fintech_lending`

## Supported Domains

Supported currencies: USD, EUR, GBP, KES, ZAR, NGN, GHS, UGX, TZS.

Supported countries: KE, UG, TZ, RW, GH, NG, ZA, GB, US, AE.

Supported outlet types: retail shop, pharmacy, supermarket, petrol station,
mobile-money agent, post office, cooperative, microfinance outlet, community
bank, mobile agent.

Supported channels: POS terminal, mobile app, USSD, SMS, web portal, tablet,
feature phone, API.

Supported settlement models: real time, hourly batch, daily batch, bilateral,
central switch.

Supported AI-agent runtimes: Codex, Claude Code, OpenCode, Pi.

## Rule Engine

The deterministic rule engine must run before state changes. It validates tenant
context, policy evidence, supported country/currency/channel/service sets,
business registration, licensing, location, float capacity, KYC/AML/fraud
evidence, transaction limits, float sufficiency, commission evidence, dispute
evidence, supervision evidence, Bytewax lifecycle processing, and privileged
AI-agent actions.

Rules may return `allow`, `deny`, or `require_review`. The service layer raises
`PermissionError` for denied writes and unresolved required reviews.

## UI And Theming

The package publishes APG Python UI metadata for:

- dashboard;
- programs;
- outlets;
- agents;
- float accounts;
- customers;
- transactions;
- cash movements;
- commissions;
- disputes;
- supervision;
- AI agents;
- settings.

Theme metadata uses semantic tokens and component metadata so generated
applications can render dense operational agency-banking screens with consistent
visual theming and tenant overrides.

## Streaming

Lifecycle metadata uses Bytewax:

- processor: `bytewax`;
- stream: `apg.fintech.agency.lifecycle`;
- key: `tenant_id`.

No alternate broker configuration is part of the contract.

## Non-Goals

This slice does not implement live POS device management, live cash vault
posting, mobile-money operator connections, regulator filings, field-force
mobile apps, rendered UI verification, durable worker deployment, or load tests.
Those remain adapter-backed follow-up work.
