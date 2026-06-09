# Cryptocurrency Services

**Capability ID**: `fintech_crypto` | **Domain**: `fintech` | **Version**: `1.1.0`

## Description

Cryptocurrency Services provides governed digital asset operations: asset registry, custody account management, balance snapshots, order management, trade execution recording, transfer requests with approval gates, compliance screening (wallet, transaction, sanctions, travel rule), market price snapshots, and governance reviews. It is the regulated operational layer over blockchain infrastructure, providing the audit trail and compliance controls that raw chain operations lack.

## Installation

```bash
pip install apg-fintech-crypto
```

## Provides

- `crypto_asset_workflow`
- `crypto_custody_workflow`
- `crypto_balance_workflow`
- `crypto_order_workflow`
- `crypto_trade_workflow`

## Requires

- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `keym`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/fintech-crypto/dashboard` | `fintech_crypto:view` | Overview |
| `/fintech-crypto/assets` | `fintech_crypto:assets` | Assets |
| `/fintech-crypto/custody` | `fintech_crypto:custody` | Custody |
| `/fintech-crypto/balances` | `fintech_crypto:balances` | Portfolio |
| `/fintech-crypto/orders` | `fintech_crypto:orders` | Trading |
| `/fintech-crypto/trades` | `fintech_crypto:trades` | Trading |
| `/fintech-crypto/transfers` | `fintech_crypto:transfers` | Treasury |
| `/fintech-crypto/screening` | `fintech_crypto:screening` | Compliance |

## Key Service Methods

- `describe()`
- `evaluate()`
- `create_crypto_wallet()`
- `get_wallet_balance()`
- `buy_crypto()`
- `sell_crypto()`
- `crypto_to_crypto_swap()`
- `send_crypto()`
- `receive_crypto()`
- `crypto_price_feed()`

_(See `service.py` for complete API.)_

## Interoperability

`fintech_crypto` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use fintech_crypto;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `FINTECH_CRYPTO_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
