# Wallet and Payment Core

**Capability ID**: `walt` | **Domain**: `common` | **Version**: `1.0.0`

## Description

WALT is the APG capability for governed wallet and payment operations. It gives generated applications a composable runtime for tenant wallets, payment instruments, transaction authorization, MFA checks, risk review, capture,

## Installation

```bash
pip install apg-common-walt
```

## Provides

- `wallet_ledger`
- `payment_instruments`
- `transaction_authorization`
- `settlement`
- `reconciliation`

## Requires

- `encr`
- `auth`
- `comp`
- `audl`
- `wflo`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/walt/dashboard` | `walt:view` | Overview |
| `/walt/wallets` | `walt:manage_wallets` | Wallets |
| `/walt/transactions` | `walt:authorize` | Transactions |
| `/walt/instruments` | `walt:manage_wallets` | Payments |
| `/walt/settlement` | `walt:settle` | Settlement |
| `/walt/reconciliation` | `walt:settle` | Settlement |
| `/walt/risk` | `walt:view` | Governance |
| `/walt/agents` | `walt:admin` | Automation |

## Key Service Methods

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

_(See `service.py` for complete API.)_

## Interoperability

`walt` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use walt;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `WALT_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
