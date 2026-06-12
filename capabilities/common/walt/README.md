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
- Debit and credit transaction authorization with idempotent replay protection.
- High-value MFA enforcement and KYC/compliance gating.
- Real-time velocity controls and spending limits engine.
- Risk review routing for high-risk transactions and fraud checks.
- Capture and wallet balance updates.
- Settlement batch creation with reconciliation evidence.
- Dispute and chargeback lifecycle management.
- Scheduled and recurring payment support.
- Multi-currency FX transfers with locked rates.
- Double-entry ledger journal for GAAP/IFRS audit trails.
- Tiered cashback and reward rules engine.
- Webhook/event notification dispatch to external consumers.
- First-class WALT agents for Codex, Claude Code, OpenCode, and Pi review lanes.
- Bytewax lifecycle stream metadata.
- Dashboard, wallet, transaction, instrument, settlement, reconciliation, risk,
  agent, policy, and settings view models.

## Quick Start

```python
from capabilities.common.walt import WaltService
import asyncio

service = WaltService()

# Sync API (v1 - backward-compatible)
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

# Privileged actions require human approval
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

```python
decision = service.validate_batch_settlement(
    tenant_id="tenant-a",
    batch_count=4,
    event_stream="bytewax",
)

assert decision["decision"] == "allow"
```

---

## World-Class Enhancements (v2.0)

15 production-grade improvements planned for the v2.0 roadmap (see
`WORLD_CLASS_IMPROVEMENTS.md` for full specs):

1. **Multi-Currency FX Transfers** — `wallet_fx_transfer` with locked rate
   validation, tolerance bands, and atomic cross-currency debit/credit.
2. **Spending Limits Engine** — `LimitProfile` with daily, monthly, and
   per-transaction caps enforced inside `authorize_transaction`.
3. **Idempotent Replay Protection** — `(tenant_id, idempotency_key)` store
   checked before processing; returns existing record on replay with a 24 h TTL.
4. **Dispute and Chargeback Lifecycle** — `DisputeRecord` with `open_dispute`,
   `resolve_dispute`, and `list_disputes`; blocks settlement on open disputes.
5. **Scheduled Recurring Payments** — `RecurringPaymentSchedule` with
   `create/pause/cancel_recurring_payment` and `execute_recurring_payment`.
6. **Wallet Tags and Metadata Indexing** — `wallet_tag/untag/search` with
   GIN-index-ready tag dictionaries on `WalletRecord`.
7. **Double-Entry Ledger Journal** — Immutable `LedgerEntry` appended on every
   balance mutation; `get_ledger` for paginated forensic traversal.
8. **KYC / Compliance Gate** — `ComplianceStatus` enum stored on wallet;
   `set_compliance_status` + `_check_compliance` guard on high-value ops.
9. **Real-Time Velocity Controls** — Time-window aggregation keyed by wallet;
   `velocity_limit_minor` breach raises `PermissionError("velocity_limit_exceeded")`.
10. **Async Batch Authorization** — `batch_authorize(tenant_id, items)` fans out
    via `asyncio.gather`; returns `BatchAuthorizeResult` with per-item outcomes.
11. **Instrument Expiry and Rotation** — `expires_at` on instruments;
    `rotate_instrument` migrates authorized-but-not-captured transactions.
12. **Webhook / Event Notification Dispatch** — `WebhookSubscription` registry
    with HMAC-SHA256-signed outbound dispatch; `register/deactivate_webhook`.
13. **Wallet Freeze with Partial Unfreeze** — `freeze_mode` field
    (`none/debit_only/credit_only/full`) with `wallet_freeze/unfreeze` and
    direction-aware enforcement.
14. **Tiered Cashback and Reward Rules** — `CashbackRule` with tier breakpoints;
    `register_cashback_rule`, `compute_cashback`, and `apply_cashback`.
15. **Paginated Cursor-Based List APIs** — `Page[T]` return type with
    `list_transactions_paged`, `list_audit_events_paged`, `list_wallets_paged`;
    existing `list_*` methods become backward-compatible wrappers.

---

## New Methods

The async API surface added in v1.x. All methods are `async def` on `WaltService`.

### `wallet_topup` / `wallet_withdraw`

Credit or debit a wallet directly without the authorization-hold-capture cycle.
Suitable for top-ups from external rails and direct withdrawals.

```python
topup = await service.wallet_topup(
    tenant_id="tenant-a",
    wallet_id=wallet["id"],
    amount="100.00",
    instrument_id=instrument["id"],
    actor="mobile-app",
    reference="mobile-topup-001",
)
# topup["new_balance_minor"] reflects updated balance

withdrawal = await service.wallet_withdraw(
    tenant_id="tenant-a",
    wallet_id=wallet["id"],
    amount="20.00",
    instrument_id=instrument["id"],
    actor="atm-node",
)
```

### `wallet_transfer`

Atomic same-currency transfer between two wallets. Raises
`PermissionError("wallet_transfer_currency_mismatch")` on cross-currency
attempts (use `wallet_fx_transfer` in v2.0 for those).

```python
transfer = await service.wallet_transfer(
    tenant_id="tenant-a",
    source_wallet_id=wallet_a["id"],
    destination_wallet_id=wallet_b["id"],
    amount="50.00",
    actor="payment-engine",
    reference="p2p-ref-42",
)
```

### `transaction_reverse`

Reverse a captured transaction; credits/debits the wallet back and sets
transaction status to `reversed`. Requires a non-empty `reason`.

```python
reversal = await service.transaction_reverse(
    tenant_id="tenant-a",
    transaction_id=captured["id"],
    reason="customer_dispute",
    actor="support-agent",
)
```

### `fraud_check` + `fraud_summary`

Run a heuristic (or model-backed) fraud assessment on a transaction, then
aggregate outcomes for reporting.

```python
check = await service.fraud_check(
    tenant_id="tenant-a",
    transaction_id=transaction["id"],
    check_model="heuristic",
)
# check["flagged"] is True for amounts > 10 000 minor units or risk_score > 0.7

summary = await service.fraud_summary(tenant_id="tenant-a")
# {"total_checks": N, "flagged_count": M, "flag_rate": 0.03, ...}
```

### `wallet_analytics`

Aggregate balance, volume, fraud, reversal, and cashback statistics across all
wallets in a tenant or scoped to a single wallet.

```python
stats = await service.wallet_analytics(
    tenant_id="tenant-a",
    wallet_id=wallet["id"],   # omit for tenant-wide rollup
)
# stats["total_balance"], stats["total_volume"], stats["fraud_flagged_count"]
```

---

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
- `wallet_topup`
- `wallet_withdrawal`
- `wallet_transfer`
- `transaction_reversed`
- `wallet_locked` / `wallet_unlocked`
- `wallet_merged`
- `cashback_credited`
- `loyalty_converted`
- `fraud_flagged`

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
