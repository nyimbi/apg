# Blockchain Ledger Services

**Capability ID**: `bclg` | **Domain**: `common` | **Version**: `1.0.0`

## Description

BCLG provides governed distributed-ledger services for APG applications. It covers tenant ledger registration, key-custody binding, signed transaction submission, high-value transaction review, smart contract deployment approval,

## Installation

```bash
pip install apg-common-bclg
```

## Provides

- `ledger_registry`
- `transaction_governance`
- `smart_contract_governance`
- `key_custody_governance`
- `ledger_audit`

## Requires

- `encr`
- `keym`
- `comp`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/bclg/dashboard` | `bclg:view` | Overview |
| `/bclg/ledgers` | `bclg:manage_ledgers` | Ledgers |
| `/bclg/transactions` | `bclg:transact` | Transactions |
| `/bclg/transactions/reviews` | `bclg:review_transactions` | Transactions |
| `/bclg/contracts` | `bclg:manage_contracts` | Contracts |
| `/bclg/contracts/reviews` | `bclg:review_contracts` | Contracts |
| `/bclg/keys` | `bclg:admin` | Security |
| `/bclg/agents` | `bclg:review_transactions` | Governance |

## Key Service Methods

- `describe()`
- `evaluate()`
- `register_ledger()`
- `list_ledgers()`
- `bind_key_custody()`
- `list_key_custody()`
- `submit_transaction()`
- `request_transaction_review()`
- `decide_transaction_review()`
- `approve_transaction()`

_(See `service.py` for complete API.)_

## Interoperability

`bclg` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use bclg;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `BCLG_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
