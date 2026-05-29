# Wallet and Payment Core Capability Specification

- **Capability Name**: Wallet and Payment Core
- **Capability ID**: `walt`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package executes the APG contract for `walt` as a deterministic wallet,
payment, settlement, reconciliation, and financial-governance runtime.

WALT gives composed APG applications a tenant-scoped local core for:

- wallet creation with owner, ledger, compliance policy, currency, balance, and
  hold tracking;
- encrypted and tokenized payment instrument registration;
- payment authorization, high-value MFA enforcement, risk review routing,
  capture, balance updates, and ledger audit events;
- settlement batch creation over captured transactions;
- reconciliation records, exception queues, and settlement status updates;
- dashboard, wallet, transaction, instrument, settlement, reconciliation, risk,
  settings, rule, route, and theme surfaces for UI composition.

Live payment processors, custody stores, bank rails, KYC providers, fraud
engines, token vaults, and settlement networks are adapter boundaries. The
checked-in package supplies deterministic local behavior that compiler output,
capacity examples, tests, publish tooling, and APG composition can execute
without those live integrations.

## Provided Services

- `wallet_ledger`
- `payment_instruments`
- `transaction_authorization`
- `settlement`
- `reconciliation`
- `capability_rules`
- `visual_theming`

## Required Services

- `encr` for production instrument encryption or token-vault integration
- `auth` for actor identity, MFA evidence, and wallet permissions
- `comp` for APG composition and capability discovery
- `audl` for durable financial audit trails
- Optional `wflo`, `ntfy`, `conn`, and `anom` adapters for approvals,
  notifications, integrations, and anomaly or fraud analysis

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

Important sections:

- `wallets`: owner requirement, ledger integrity, multi-currency support, and
  negative-balance blocking.
- `payments`: instrument tokenization, transaction limits, high-value MFA, and
  risk scoring.
- `settlement`: settlement approval, reconciliation, exception queues, and
  chargeback support.
- `governance`: tenant context, financial audit, encrypted instruments, and
  compliance policy requirements.
- `ui`: wallet dashboard, transaction console, settlement center, and
  reconciliation queue toggles.
- `theme`: default `walt_wallet_ops` visual theme and tenant override policy.

## Rules

- `tenant_context_required`
- `wallet_requires_owner`
- `instrument_requires_encryption`
- `high_value_requires_mfa`
- `settlement_requires_reconciliation`
- `high_risk_transaction_requires_review`

These rules are enforced in `WaltService` before state-changing operations.
Deny decisions raise `PermissionError` with the rule reason. Review decisions
create review-required records with `required_actions` so APG workflows or
human approval queues can continue the process.

## Runtime Behavior

`service.py` exposes `WaltService`, a dependency-light runtime with:

- `create_wallet()` for tenant-scoped wallet setup, owner checks, ledger
  references, compliance policy references, initial balances, and audit events;
- `register_instrument()` for encrypted/tokenized payment instruments;
- `authorize_transaction()` for debit and credit authorization, MFA, risk
  scoring, review routing, balance-hold creation, idempotency keys, and audit;
- `capture_transaction()` for balance and hold updates;
- `create_settlement_batch()` for captured-transaction settlement with
  reconciliation evidence;
- `record_reconciliation()` for matched or exception reconciliation outcomes;
- list and dashboard helpers for wallets, instruments, transactions,
  settlements, reconciliations, and audit events;
- `create_record()` and `list_records()` compatibility shims backed by wallet
  creation and wallet listing.

`wallet_runtime.py` owns the serializable dataclasses, stable ID generation,
money conversion helpers, currency and instrument normalization, UTC
timestamps, and rule required-action extraction.

`api.py` exposes dependency-light function wrappers over the service for APG
generated runtimes and package smoke tests. `views.py` exposes route-aligned
view models for dashboard, wallet console, transaction console, instrument
vault, settlement center, reconciliation queue, risk review, and settings.

## UI

The package exposes 8 APG Python UI route contracts through `views.py` and the
package semantic model:

- `/walt/dashboard`
- `/walt/wallets`
- `/walt/transactions`
- `/walt/instruments`
- `/walt/settlement`
- `/walt/reconciliation`
- `/walt/risk`
- `/walt/settings`

## Theme

The package uses the `walt_wallet_ops` APG theme contract.

Theme tokens cover wallet operations with compact density, wallet grids,
transaction tables, instrument vaults, settlement timelines, risk chips, and
reconciliation status styling.

## Proof Commands

Focused package proof:

```bash
./.venv/bin/python -m py_compile capabilities/common/walt/__init__.py capabilities/common/walt/models.py capabilities/common/walt/wallet_runtime.py capabilities/common/walt/service.py capabilities/common/walt/api.py capabilities/common/walt/views.py capabilities/common/walt/capability_contract.py capabilities/common/walt/app.py capabilities/common/walt/test_capability_contract.py capabilities/common/walt/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/walt/test_capability_contract.py capabilities/common/walt/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/walt --json
./.venv/bin/apg capabilities publish-plan capabilities/common/walt --json
```

Global package health proof:

```bash
./.venv/bin/apg capabilities implementation-audit --json
./.venv/bin/apg capabilities audit --strict-package-artifacts --json
```

## Known Non-Goals

- No live card network, mobile-money, bank, ledger, custody, token-vault, KYC,
  fraud-model, or settlement-network integration is performed in this package.
- No real cryptographic storage is performed locally; production instrument
  protection belongs behind `encr` and token-vault adapters.
- No external payment side effects are emitted by tests or local package
  methods.
