# APG Digital Lending Specification

## Purpose

Digital Lending is the APG capability for configurable credit products,
borrower onboarding, application intake, underwriting, offers, disbursements,
repayment scheduling, collections, and AI-agent-assisted credit operations.
It is designed to compose with APG fintech payments, wallets, cards, KYC, AML,
fraud, and remittance capabilities while keeping live bureau, statement,
disbursement, servicing, and collections providers behind adapters.

## Users

- Credit product managers define tenant-scoped loan products, term limits,
  rates, amount limits, currencies, and repayment frequencies.
- Borrower operations teams onboard customers with KYC, income, country, and
  consent evidence.
- Credit analysts review applications, affordability evidence, bank-statement
  references, AML/Fraud evidence, and behavior evidence from cards or
  remittance flows.
- Underwriting reviewers record scores, decisions, adverse-action reasons, and
  human approvals.
- Servicing teams issue offers, record disbursement approvals, schedule
  repayments, and open collections cases.
- Automation teams register provider-neutral lending agents running on Codex,
  Claude Code, OpenCode, Pi, or future runtimes.

## Functional Scope

Digital Lending must provide:

- Loan product governance with owner, product type, currency, amount limit,
  term, rate, and repayment-frequency controls.
- Borrower lifecycle management with KYC, income, country, and consent
  evidence.
- Credit application workflow with affordability, bank-statement, AML, fraud,
  remittance, and card behavior evidence.
- Underwriting decisioning with score bounds, supported decisions, evidence,
  adverse-action reasons, and human approval gates.
- Loan offer workflow with amount, APR, term, expiry, status, and borrower
  acceptance evidence.
- Disbursement controls across payment account, wallet, card, and bank-transfer
  rails with funding account and approval evidence.
- Repayment schedule workflow with due amount, due date, repayment frequency,
  and installment count.
- Collections workflow with overdue account, reason, reviewer, and contact
  policy evidence.
- First-class lending-agent registration with provider-neutral runtimes and
  supported roles.
- Bytewax lifecycle stream metadata for generated applications and runtime
  orchestration.

## Rules

The deterministic rule engine must deny or require review for missing tenant
context, missing write policy, invalid product setup, incomplete borrower
evidence, incomplete application evidence, high-amount applications without
review, invalid underwriting decisions, missing adverse-action reasons, final
decisions without approval, invalid offer terms, accepted offers without
borrower acceptance, disbursements without accepted offers or approval, invalid
repayment schedules, incomplete collection cases, unsupported agent runtimes,
unsupported agent roles, and non-Bytewax batch processing.

## UI And Theming

The capability publishes framework-neutral route and view-model metadata for:

- dashboard
- products
- borrowers
- applications
- underwriting
- offers
- disbursements
- repayments
- collections
- agents
- settings

The theme uses compact operational tokens, status-chip components, and route
icons suitable for dense credit operations screens.

## Adapter Boundaries

The package does not call live bureaus, bank-statement analyzers, affordability
providers, disbursement rails, servicing platforms, collections agencies, audit
sinks, notification systems, key-management systems, or durable Bytewax workers
directly. Those integrations are represented as explicit dependency and adapter
contracts so generated APG applications can bind them safely.

## Acceptance Criteria

- `get_capability_contract()` validates through the APG contract registry.
- `evaluate_capability_rules()` returns deterministic allow, deny, and review
  decisions with matched rules and actions.
- Service methods enforce contract guardrails before mutating local state.
- API helpers, view models, and `app.py` are importable without a web framework.
- `self_test()` passes and the semantic model exposes configuration, rules,
  routes, theme, streaming, dependencies, and agent-team metadata.
- Focused tests exercise the happy path, guardrail failures, API helpers, view
  models, app entrypoint, and publishable contract surfaces.
