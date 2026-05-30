# WALT Capability Specification

## Identity

- Capability name: Wallet and Payment Core
- Capability ID: `walt`
- Category: common
- Runtime target: APG Python capability package

## Mission

WALT gives generated APG applications a governed wallet and payment core. It
coordinates tenant wallet records, ledger references, compliance policies,
payment instruments, transaction authorization, MFA, risk scoring, capture,
settlement, reconciliation, payment-agent review, audit events, and Bytewax
lifecycle streaming.

## Functional Scope

WALT owns the executable lifecycle for:

- tenant wallet creation and balance tracking;
- ledger and compliance policy references;
- multi-currency wallet records;
- encrypted and tokenized payment instrument registration;
- transaction authorization for debit and credit flows;
- high-value MFA enforcement;
- high-risk transaction review;
- capture and balance/hold updates;
- settlement batch creation;
- reconciliation evidence and exception status;
- first-class wallet/payment agents for instrument, payment, risk, settlement,
  reconciliation, and chargeback review;
- wallet/payment audit and dashboard evidence.

## Configuration Contract

The configuration schema requires:

- `tenant_id`
- `wallets`
- `payments`
- `settlement`
- `walt_agents`
- `governance`
- `observability`
- `adapters`
- `ui`
- `theme`

WALT must expose these through `get_capability_contract()`, generated semantic
model evidence, and package registration metadata.

## Domain Records

### Wallet

A wallet contains tenant, owner, currency, ledger reference, compliance policy,
balance, hold amount, status, and timestamps.

### Payment Instrument

A payment instrument contains tenant, wallet, instrument reference, instrument
type, token reference, encryption status, verifier, status, and timestamp.

### Transaction

A transaction contains tenant, wallet, instrument, direction, amount, currency,
status, risk score, MFA state, risk-review state, idempotency key, matched
rules, required actions, and lifecycle timestamps.

### Settlement Batch

A settlement batch contains tenant, transaction IDs, settlement account, total,
currency, reconciliation state, status, creator, and timestamp.

### Reconciliation

A reconciliation record contains tenant, settlement batch, evidence reference,
matched count, exception count, status, recorder, and timestamp.

### WALT Agent

A WALT agent is a first-class composition element with tenant, name, runtime,
role, scope, owner, status, and human approval policy.

Supported runtimes:

- `codex`
- `claude_code`
- `opencode`
- `pi`

Supported roles:

- `payment_reviewer`
- `risk_reviewer`
- `settlement_reviewer`
- `reconciliation_reviewer`
- `instrument_reviewer`
- `chargeback_reviewer`

## Lifecycle States

Wallet states:

- `active`
- `disabled`
- `frozen`

Transaction states:

- `authorized`
- `captured`
- `review_required`
- `declined`
- `settled`

Settlement states:

- `ready`
- `reconciled`
- `exception_review`

Lifecycle stream states:

- `active`
- `authorized`
- `captured`
- `review_required`
- `settled`
- `reconciled`
- `exception_review`
- `blocked`

## Rules

The deterministic rule engine must enforce:

- tenant context on all executable operations;
- owner, ledger, and compliance policy on wallet creation;
- encryption, tokenization, and verification for payment instruments;
- MFA for high-value transactions;
- risk score evidence for transaction authorization;
- Bytewax stream metadata for transaction authorization;
- reconciliation evidence for settlement;
- approval for settlement;
- Bytewax stream metadata for settlement;
- durable evidence reference for reconciliation;
- review for high-risk transactions;
- Bytewax stream coordination for batch settlement mutation;
- approved WALT-agent runtimes;
- approved WALT-agent roles;
- human approval for privileged agent payment actions.

## Service Requirements

`WaltService` must provide:

- `describe()`
- `evaluate()`
- `create_wallet()`
- `register_instrument()`
- `authorize_transaction()`
- `capture_transaction()`
- `create_settlement_batch()`
- `record_reconciliation()`
- `register_walt_agent()`
- `validate_agent_payment_action()`
- `validate_batch_settlement()`
- list helpers for every record type;
- `dashboard_summary()`.

## API Requirements

`api.py` must expose payload-oriented helpers for status, wallet creation,
instrument registration, transaction authorization, capture, settlement,
reconciliation, agents, agent-action validation, batch settlement validation,
compatibility record creation, and system listing.

## UI Requirements

WALT exposes APG Python view models for:

- `/walt/dashboard`
- `/walt/wallets`
- `/walt/transactions`
- `/walt/instruments`
- `/walt/settlement`
- `/walt/reconciliation`
- `/walt/risk`
- `/walt/agents`
- `/walt/policy`
- `/walt/settings`

The UI contract must expose rules, summaries, agent policy, Bytewax streaming,
and visual theme tokens.

## Visual Theming

The default visual theme is `walt_wallet_ops`. It defines compact density,
wallet grids, balance pills, ledger bands, transaction lists, risk chips,
settlement timelines, reconciliation chips, tokenized-card lists, encryption
chips, review lanes, and guardrail chips.

## Streaming

WALT lifecycle events use Bytewax:

- processor: `bytewax`
- stream: `apg.walt.lifecycle`
- key: `tenant_id`

Events:

- `wallet_created`
- `instrument_registered`
- `transaction_authorized`
- `transaction_captured`
- `settlement_batch_created`
- `reconciliation_recorded`
- `walt_agent_registered`

## Adapter Boundaries

The in-package runtime must stay dependency-light. Production deployments bind
token vaults, encryption systems, payment processors, banking or mobile money
rails, ledger stores, compliance engines, risk engines, settlement networks,
audit sinks, and Bytewax workers through adapters.

## Acceptance Criteria

- README, specification, plan, and capability summary exist.
- Contract shape validates.
- Generated app evidence is refreshed from the contract.
- Tests cover contract, rules, service, API, views, agent guardrails, and
  Bytewax guardrails.
- Focused package tests pass.
- Implementation audit reports domain-specific behavior with no warnings.
- Publish plan reports side-effect-free output with no warnings.
