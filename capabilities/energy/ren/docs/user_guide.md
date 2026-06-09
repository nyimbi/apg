# Renewable Energy

**Capability ID**: `energy_ren` | **Domain**: `energy` | **Version**: `1.0.0`

## Description

Renewable Energy manages the full lifecycle of renewable generation assets — solar PV, wind, hydro, biomass, geothermal and others. It tracks curtailment events with revenue loss accounting, issues and retires Renewable Energy Certificates (RECs) with double-issuance prevention, manages carbon credits requiring third-party verification, administers feed-in tariffs, publishes multi-horizon generation forecasts, and computes performance metrics against benchmarks.

## Installation

```bash
pip install apg-energy-ren
```

## Provides

- `renewable_asset_registry`
- `curtailment_tracking`
- `rec_certificate_management`
- `carbon_credit_management`
- `feed_in_tariff_management`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/energy-ren/dashboard` | `energy_ren:view` | Overview |
| `/energy-ren/assets` | `energy_ren:assets` | Assets |
| `/energy-ren/assets/<id>` | `energy_ren:assets` | Assets |
| `/energy-ren/curtailment` | `energy_ren:curtailment` | Operations |
| `/energy-ren/recs` | `energy_ren:recs` | Certificates |
| `/energy-ren/carbon-credits` | `energy_ren:carbon_credits` | Certificates |
| `/energy-ren/feed-in-tariffs` | `energy_ren:feed_in_tariffs` | Finance |
| `/energy-ren/forecasting` | `energy_ren:forecasting` | Analytics |

## Key Service Methods

- `describe()`
- `evaluate()`
- `register_asset()`
- `update_asset_status()`
- `list_assets()`
- `get_asset()`
- `record_curtailment()`
- `approve_curtailment()`
- `list_curtailments()`
- `get_curtailment_summary()`

_(See `service.py` for complete API.)_

## Interoperability

`energy_ren` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use energy_ren;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `ENERGY_REN_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
