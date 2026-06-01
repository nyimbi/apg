# APG Digital Neobanking Specification

## Purpose

Digital Neobanking is the APG capability for building package-backed retail,
SME, merchant, and embedded digital banking experiences. It owns neobank
program governance, customer onboarding, deposit-account lifecycle, payment
rail links, account transaction posting, savings pots, statements, service
cases, and AI-agent-assisted banking operations.

The capability is designed to compose with APG payments, wallets, cards, KYC,
AML, fraud, lending, and remittance packages while leaving live core-banking,
issuer-processor, card-network, payment-rail, notification, audit,
key-management, and durable Bytewax worker integrations behind adapters.

## Users

- Neobank program owners define tenant-scoped banking programs, supported
  countries, currencies, owners, and settlement accounts.
- Operations teams onboard customers with KYC, AML, fraud, country, and
  consent evidence.
- Banking teams open deposit accounts, link payment rails, post transactions,
  issue statements, and maintain savings pots.
- Customer-support and risk teams open service cases with evidence and reviewer
  ownership.
- Automation teams register provider-neutral neobanking agents for Codex,
  Claude Code, OpenCode, Pi, and future runtimes.

## Functional Scope

Digital Neobanking must provide:

- Program governance for owner, country, currency, and settlement evidence.
- Digital customer onboarding with customer, KYC, AML, fraud, country, and
  consent evidence.
- Deposit-account lifecycle for current, savings, joint, business, youth, and
  merchant accounts.
- Payment-rail linking for bank transfer, card, wallet, mobile money, and
  internal transfers.
- Account transaction posting with risk references, direction calculation,
  balance updates, and high-impact transaction review.
- Savings-pot workflow with target and source-account controls.
- Statement workflow with period and transaction-count evidence.
- Customer service cases with reason, reviewer, account, customer, and evidence
  controls.
- Provider-neutral AI-agent registration and privileged-agent approval gates.
- Bytewax lifecycle metadata for APG composition and orchestration.

## Rules

The deterministic rule engine must deny or require review for missing tenant
context, missing write policy, incomplete program setup, incomplete customer
evidence, invalid account setup, unsupported payment rails, incomplete rail
links, invalid transactions, missing risk evidence, high-impact transactions
without human approval, invalid savings pots, incomplete statement periods,
incomplete service cases, unsupported agent runtimes, unsupported agent roles,
privileged agent actions without approval, and non-Bytewax batch routing.

## UI And Theming

The capability publishes framework-neutral route and view metadata for:

- dashboard
- programs
- customers
- accounts
- rails
- transactions
- savings
- statements
- cases
- agents
- settings

The visual theme uses compact operational tokens and status-chip components for
dense account-operations, support, and compliance screens.

## Adapter Boundaries

This package does not directly call live core banking systems, bank ledgers,
issuer processors, card networks, payment rails, mobile-money operators,
customer support systems, audit sinks, notification providers, key-management
services, regulator filing systems, or durable Bytewax workers. Those surfaces
remain adapter contracts for generated applications to bind explicitly.

## Acceptance Criteria

- `get_capability_contract()` validates through the APG contract registry.
- `evaluate_capability_rules()` returns deterministic allow, deny, and review
  decisions with matched rules and actions.
- Service methods enforce guardrails before mutating local state.
- API helpers, view models, and `app.py` are importable without a web framework.
- `self_test()` passes and the semantic model exposes configuration, rules,
  routes, theme, streaming, dependencies, and agent-team metadata.
- Focused tests exercise the account lifecycle, guardrail failures, API
  helpers, view models, app entrypoint, and publishable contract surfaces.
