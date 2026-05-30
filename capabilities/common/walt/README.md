# WALT - Wallet and Payment Core

WALT is the APG capability for governed wallet and payment operations. It gives
generated applications a composable runtime for tenant wallets, payment
instruments, transaction authorization, MFA checks, risk review, capture,
settlement, reconciliation, AI-assisted review, and Bytewax lifecycle events.

Use WALT when an application needs a tenant-aware financial core with explicit
guardrails for wallet ledgers, instruments, transactions, settlement, and audit.

## What WALT Provides

- Tenant-scoped wallet ledger records.
- Balance and hold tracking.
- Encrypted and tokenized payment instrument registration.
- Debit and credit transaction authorization.
- High-value MFA enforcement.
- Risk review routing for high-risk transactions.
- Capture and wallet balance updates.
- Settlement batch creation.
- Reconciliation evidence and exception status.
- First-class WALT agents for Codex, Claude Code, OpenCode, and Pi based review
  lanes.
- Bytewax lifecycle stream metadata.
- Dashboard, wallet, transaction, instrument, settlement, reconciliation, risk,
  agent, policy, and settings view models.

## Quick Start

```python
from capabilities.common.walt import WaltService

service = WaltService()

wallet = service.create_wallet(
    tenant_id="tenant-a",
    owner_ref="customer-1",
    currency="USD",
    ledger_ref="ledger://tenant-a/customer-1",
    compliance_policy_ref="policy://wallets/default",
    initial_balance="250.00",
    actor="operator-1",
)

instrument = service.register_instrument(
    tenant_id="tenant-a",
    wallet_id=wallet["id"],
    instrument_ref="card://source",
    instrument_type="card",
    token_ref="tok_123",
    encrypted=True,
    verified_by="vault-service",
)

transaction = service.authorize_transaction(
    tenant_id="tenant-a",
    wallet_id=wallet["id"],
    instrument_id=instrument["id"],
    amount="75.50",
    currency="USD",
    mfa_completed=True,
    risk_score=0.2,
    idempotency_key="txn-1",
    actor="cashier-1",
)

captured = service.capture_transaction(
    tenant_id="tenant-a",
    transaction_id=transaction["id"],
    actor="cashier-1",
)
```

## Settlement And Reconciliation

Settlement requires captured transactions, reconciliation state, approval, and
Bytewax lifecycle metadata.

```python
settlement = service.create_settlement_batch(
    tenant_id="tenant-a",
    transaction_ids=[captured["id"]],
    settlement_account_ref="settlement://merchant/primary",
    reconciliation_completed=True,
    created_by="settlement-ops",
    approval_ref="approval://settlement/1",
    event_stream="bytewax",
)

service.record_reconciliation(
    tenant_id="tenant-a",
    settlement_batch_id=settlement["id"],
    reconciliation_ref="recon://batch/1",
    matched_count=1,
    exception_count=0,
    recorded_by="recon-ops",
)
```

## WALT Agents

WALT treats payment review agents as governed composition elements.

```python
agent = service.register_walt_agent(
    tenant_id="tenant-a",
    name="Risk reviewer",
    runtime="codex",
    role="risk_reviewer",
    scope="review high-risk transactions and settlement evidence",
)

decision = service.validate_agent_payment_action(
    tenant_id="tenant-a",
    agent_id=agent["id"],
    action="settle_batch",
    privileged_scope=True,
)

assert decision["decision"] == "deny"
```

Privileged agent payment actions require human approval:

```python
decision = service.validate_agent_payment_action(
    tenant_id="tenant-a",
    agent_id=agent["id"],
    action="settle_batch",
    privileged_scope=True,
    human_approval_ref="approval://agent/payment",
)

assert decision["decision"] == "allow"
```

## Batch Settlement Guardrail

Batch settlement mutation must use Bytewax stream coordination.

```python
decision = service.validate_batch_settlement(
    tenant_id="tenant-a",
    batch_count=4,
    event_stream="bytewax",
)

assert decision["decision"] == "allow"
```

## Deterministic Rules

WALT enforces:

- tenant context on all executable operations;
- wallet owner, ledger reference, and compliance policy;
- instrument encryption, tokenization, and verifier attribution;
- MFA for high-value transactions;
- risk score evidence for transaction authorization;
- Bytewax lifecycle stream metadata for transactions and settlement;
- reconciliation evidence and settlement approval;
- review for high-risk transactions;
- Bytewax coordination for batch settlement mutation;
- supported WALT-agent runtime and role;
- human approval for privileged agent actions.

## API Helpers

`api.py` provides payload-oriented helpers:

- `capability_status()`
- `create_wallet()`
- `register_instrument()`
- `authorize_transaction()`
- `capture_transaction()`
- `create_settlement_batch()`
- `record_reconciliation()`
- `register_walt_agent()`
- `validate_agent_payment_action()`
- `validate_batch_settlement()`
- `create_record()`
- `list_records()`
- `list_wallet_payments()`

## UI Routes

- dashboard: `/walt/dashboard`
- wallets: `/walt/wallets`
- transactions: `/walt/transactions`
- instruments: `/walt/instruments`
- settlement: `/walt/settlement`
- reconciliation: `/walt/reconciliation`
- risk: `/walt/risk`
- agents: `/walt/agents`
- policy: `/walt/policy`
- settings: `/walt/settings`

## Bytewax Stream

WALT publishes lifecycle metadata for Bytewax:

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

The in-package service stores records in memory so generated applications,
tests, and publish-plan probes can execute without external infrastructure.
Production systems should attach token vaults, encryption systems, payment
processors, banking or mobile money rails, ledger stores, compliance engines,
risk engines, settlement networks, audit sinks, and Bytewax workers through APG
adapters.

## Verification

```bash
./.venv/bin/python -m py_compile capabilities/common/walt/__init__.py capabilities/common/walt/capability_contract.py capabilities/common/walt/models.py capabilities/common/walt/wallet_runtime.py capabilities/common/walt/service.py capabilities/common/walt/api.py capabilities/common/walt/views.py capabilities/common/walt/app.py capabilities/common/walt/test_capability_contract.py capabilities/common/walt/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/walt/test_capability_contract.py capabilities/common/walt/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/walt --json
./.venv/bin/apg capabilities publish-plan capabilities/common/walt --json
```
