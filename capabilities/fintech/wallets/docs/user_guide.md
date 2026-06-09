# Digital Wallets

**Capability ID**: `fintech_wallets` | **Domain**: `fintech` | **Version**: `1.1.0`

## Description

Digital Wallets provides the stored-value ledger layer: wallet lifecycle (consumer, merchant, agent, escrow, treasury), instrument registration with verified token references, double-entry ledger operations (credit, debit, transfer), hold management for reserved funds, and limit governance. It is the balance-holding layer that other capabilities — payments, mobile, agency, neobanking — use to maintain available and held balances for their customers and operational accounts.

## Installation

```bash
pip install apg-fintech-wallets
```

## Provides

- `wallet_lifecycle`
- `stored_value_ledger`
- `wallet_instrument_registry`
- `wallet_transfer_workflow`
- `wallet_hold_workflow`

## Requires

- `auth`
- `audl`
- `ntfy`
- `walt`
- `fintech_payments`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/fintech-wallets/dashboard` | `fintech_wallets:view` | Overview |
| `/fintech-wallets/wallets` | `fintech_wallets:manage_wallets` | Wallets |
| `/fintech-wallets/instruments` | `fintech_wallets:manage_instruments` | Wallets |
| `/fintech-wallets/ledger` | `fintech_wallets:view_ledger` | Ledger |
| `/fintech-wallets/limits` | `fintech_wallets:govern_limits` | Governance |
| `/fintech-wallets/holds` | `fintech_wallets:operate` | Operations |
| `/fintech-wallets/agents` | `fintech_wallets:admin` | Automation |
| `/fintech-wallets/settings` | `fintech_wallets:admin` | Administration |

## Key Service Methods

- `describe()`
- `evaluate()`
- `open_wallet()`
- `register_instrument()`
- `credit_wallet()`
- `debit_wallet()`
- `transfer()`
- `place_hold()`
- `release_hold()`
- `register_wallet_agent()`

_(See `service.py` for complete API.)_

## Interoperability

`fintech_wallets` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use fintech_wallets;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `FINTECH_WALLETS_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
