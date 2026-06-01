# APG Buy Now Pay Later Capability Specification

## Purpose

The Buy Now Pay Later capability turns BNPL from a placeholder into an
executable APG fintech package that generated applications can compose into
merchant checkout, consumer credit, settlement, servicing, and compliance
workflows.

The capability is deliberately dependency-light. It can run locally for
compiler output, smoke tests, demos, and downstream composition while live
checkout gateways, acquirers, credit bureaus, payment rails, collection
providers, regulator reporting, and durable Bytewax workers remain behind
adapter boundaries.

## Functional Scope

The package provides:

- merchant BNPL program governance;
- consumer onboarding and eligibility lifecycle;
- merchant profile and checkout channel registration;
- checkout-session capture with AML, fraud, payment, consent, and cart evidence;
- affordability decisioning with deterministic approval, decline, and review
  controls;
- BNPL plan creation for pay-in-3, pay-in-4, monthly installments, and invoice
  split products;
- installment schedule management;
- merchant settlement and reconciliation workflow;
- dispute intake and evidence workflow;
- provider-neutral AI-agent composition for BNPL operations, affordability,
  merchant risk, settlement, dispute, and compliance review.

## Composition Contract

Capability id: `fintech_bnpl`

Provides:

- `bnpl_merchant_program_governance`
- `consumer_bnpl_lifecycle`
- `merchant_checkout_workflow`
- `affordability_decisioning`
- `bnpl_plan_workflow`
- `installment_schedule_workflow`
- `merchant_settlement_workflow`
- `bnpl_dispute_workflow`
- `bnpl_agent_workflow`

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
- `fintech_lending`
- `fintech_neobanking`

## Supported Domains

Supported currencies: USD, EUR, GBP, KES, ZAR, NGN, GHS, UGX, TZS.

Supported countries: KE, UG, TZ, RW, GH, NG, ZA, GB, US, AE.

Supported merchant categories: retail, electronics, grocery, travel, education,
medical, services, marketplace.

Supported checkout channels: web, mobile, pos, marketplace, api.

Supported plan types: pay-in-3, pay-in-4, monthly installments, invoice split.

Supported AI-agent runtimes: Codex, Claude Code, OpenCode, Pi.

## Rule Engine

The rule engine is deterministic and must be enforced before state changes. It
guards tenant context, write policies, program terms, supported country and
currency sets, merchant and consumer evidence, checkout evidence, affordability
outcomes, plan terms, installment status, settlement controls, dispute evidence,
Bytewax lifecycle processing, and privileged AI-agent actions.

Rules may return `allow`, `deny`, or `require_review`. Service methods raise
`PermissionError` when a write is denied or requires unresolved review evidence.

## UI And Theming

The package publishes APG Python UI metadata for:

- dashboard;
- merchant programs;
- consumers;
- merchants;
- checkout sessions;
- affordability decisions;
- BNPL plans;
- installments;
- settlements;
- disputes;
- AI agents;
- settings.

Theme metadata uses semantic color, density, icon, and component tokens so a
generated application can render the capability consistently while allowing
tenant-level visual overrides.

## Streaming

Lifecycle metadata is expressed with Bytewax:

- processor: `bytewax`;
- stream: `apg.fintech.bnpl.lifecycle`;
- key: `tenant_id`.

No alternate broker configuration is part of the package contract.

## Non-Goals

This slice does not implement live credit bureau pulls, gateway captures,
acquirer settlement files, card-network disputes, live collection providers,
regulatory filing, durable worker deployment, or rendered UI verification. Those
remain adapter-backed follow-up work.
