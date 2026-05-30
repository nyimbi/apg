# Wallet and Payment Core Capability Packet

- Capability Name: Wallet and Payment Core
- Capability ID: `walt`
- Category: common
- Version: 1.0.0

## Purpose

WALT provides executable APG wallet, payment, settlement, reconciliation,
payment-risk, agent-review, audit, and Bytewax stream behavior. It lets
generated applications compose tenant wallet ledgers, tokenized instruments,
transaction authorization, high-value MFA, risk review, capture, settlement
approval, reconciliation evidence, exception queues, and AI-assisted review
lanes.

## Provides

- `wallet_ledger`
- `payment_instruments`
- `transaction_authorization`
- `settlement`
- `reconciliation`
- `payment_risk_governance`
- `walt_agents`

## Requires

- `encr`
- `auth`
- `comp`
- `audl`
- `wflo`

## Configuration Areas

WALT configuration is defined by `capability_contract.py` and covers:

- tenant context;
- wallet owner, ledger integrity, multi-currency, and balance policy;
- instrument encryption, tokenization, and verification;
- high-value MFA, transaction limits, and risk scoring;
- settlement approval, reconciliation, exception queue, and chargeback policy;
- first-class wallet/payment agent runtimes, roles, and human approval;
- audit and financial state-change governance;
- Bytewax lifecycle-stream observability;
- adapter boundaries for encryption, authorization, compliance, ledger, audit, and event streaming;
- UI route toggles and theme tokens.

## Lifecycle

WALT supports the following lifecycle:

1. Create a tenant wallet with owner, ledger, compliance policy, currency, and balance.
2. Register encrypted and tokenized payment instruments with verification evidence.
3. Authorize debit or credit transactions with MFA, risk score, idempotency, and Bytewax stream metadata.
4. Route high-risk transactions for review.
5. Capture authorized transactions and update wallet holds and balances.
6. Create settlement batches only from captured transactions with reconciliation, approval, and Bytewax stream metadata.
7. Record reconciliation evidence and exception status.
8. Register governed AI agents that review instruments, payments, risk, settlement, reconciliation, and chargebacks.

## Deterministic Rules

- `tenant_context_required`
- `wallet_requires_owner`
- `wallet_requires_ledger`
- `wallet_requires_compliance_policy`
- `instrument_requires_encryption`
- `instrument_requires_token`
- `instrument_requires_verification`
- `high_value_requires_mfa`
- `transaction_requires_risk_score`
- `transaction_requires_bytewax_stream`
- `settlement_requires_reconciliation`
- `settlement_requires_approval`
- `settlement_requires_bytewax_stream`
- `reconciliation_requires_evidence`
- `high_risk_transaction_requires_review`
- `batch_settlement_requires_bytewax`
- `walt_agent_runtime_supported`
- `walt_agent_role_supported`
- `privileged_agent_payment_action_requires_human_approval`

## UI

WALT exposes APG Python view models for dashboard, wallet console, transaction
console, instrument vault, settlement center, reconciliation queue, payment
risk, agent workbench, policy center, and settings.

## Theme

WALT uses the `walt_wallet_ops` theme with compact density, wallet grids,
balance pills, ledger bands, transaction lists, risk chips, settlement
timelines, reconciliation chips, tokenized instrument lists, encryption chips,
review lanes, and guardrail chips.

## Streaming

WALT lifecycle events are described by the Bytewax stream manifest:

- processor: `bytewax`
- stream: `apg.walt.lifecycle`
- key: `tenant_id`
- events: `wallet_created`, `instrument_registered`,
  `transaction_authorized`, `transaction_captured`,
  `settlement_batch_created`, `reconciliation_recorded`,
  `walt_agent_registered`

## Adapter Boundaries

The in-package service is dependency-light and stores records in memory for
generated apps, tests, and publish-plan probes. Production deployments should
bind token vaults, encryption systems, payment processors, banking or mobile
money rails, ledger stores, compliance engines, fraud/risk engines, settlement
networks, audit sinks, and Bytewax workers through APG adapters without
weakening the deterministic contract.

## Focused Verification

- `./.venv/bin/python -m py_compile capabilities/common/walt/__init__.py capabilities/common/walt/models.py capabilities/common/walt/wallet_runtime.py capabilities/common/walt/service.py capabilities/common/walt/api.py capabilities/common/walt/views.py capabilities/common/walt/capability_contract.py capabilities/common/walt/app.py capabilities/common/walt/test_capability_contract.py capabilities/common/walt/tests/test_package_contract.py`
- `./.venv/bin/pytest -q capabilities/common/walt/test_capability_contract.py capabilities/common/walt/tests/test_package_contract.py`
- `./.venv/bin/apg capabilities implementation-audit --root capabilities/common/walt --json`
- `./.venv/bin/apg capabilities publish-plan capabilities/common/walt --json`
