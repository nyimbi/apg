# Multi-Currency Management

**Capability ID**: `loc_mcy` | **Domain**: `loc` | **Version**: `1.0.0`

## Description

Multi-Currency Management (MCY) provides full lifecycle management of currencies, exchange rates, FX revaluation, currency translation, and FX gain/loss reporting for organisations operating across multiple currencies. It enforces positive exchange rates, arms-length approval for manual rates, approval-gated revaluation posting, and tenant-scoped isolation of all currency data.

## Installation

```bash
pip install apg-loc-mcy
```

## Provides

- `currency_configuration`
- `exchange_rate_management`
- `fx_revaluation_workflow`
- `currency_translation_workflow`
- `fx_gain_loss_reporting`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/loc-mcy/dashboard` | `loc_mcy:view` | Overview |
| `/loc-mcy/currencies` | `loc_mcy:currencies` | Setup |
| `/loc-mcy/currencies/create` | `loc_mcy:currencies_write` | Setup |
| `/loc-mcy/exchange-rates` | `loc_mcy:exchange_rates` | Rates |
| `/loc-mcy/exchange-rates/create` | `loc_mcy:exchange_rates_write` | Rates |
| `/loc-mcy/exchange-rates/upload` | `loc_mcy:exchange_rates_write` | Rates |
| `/loc-mcy/revaluation` | `loc_mcy:revaluation` | Processing |
| `/loc-mcy/revaluation/create` | `loc_mcy:revaluation_write` | Processing |

## Key Service Methods

- `uuid7str()`
- `uuid7str()`
- `describe()`
- `evaluate()`
- `configure_currency()`
- `get_currency()`
- `get_currency_by_code()`
- `list_currencies()`
- `update_currency()`
- `record_exchange_rate()`

_(See `service.py` for complete API.)_

## Interoperability

`loc_mcy` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use loc_mcy;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `LOC_MCY_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
