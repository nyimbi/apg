# Fuel Management

**Capability ID**: `transport_fue` | **Domain**: `transport` | **Version**: `1.0.0`

## Description

The Fuel Management capability covers fuel procurement, transaction recording with odometer capture, fuel card management and reconciliation, bunker management, carbon footprint calculation across GHG Protocol and ISO 14064 standards, and storage tank monitoring. Built-in phantom fill and theft pattern detection protect against fraud.

## Installation

```bash
pip install apg-transport-fue
```

## Provides

- `fuel_procurement_workflow`
- `fuel_consumption_tracking_workflow`
- `bunker_management_workflow`
- `fuel_card_reconciliation_workflow`
- `carbon_footprint_reporting_workflow`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/transport-fuel/dashboard` | `transport_fue:view` | Overview |
| `/transport-fuel/procurement` | `transport_fue:procurement` | Procurement |
| `/transport-fuel/transactions` | `transport_fue:transactions` | Transactions |
| `/transport-fuel/cards` | `transport_fue:cards` | Cards |
| `/transport-fuel/cards/reconciliation` | `transport_fue:cards` | Cards |
| `/transport-fuel/storage` | `transport_fue:storage` | Storage |
| `/transport-fuel/efficiency` | `transport_fue:efficiency` | Analytics |
| `/transport-fuel/carbon` | `transport_fue:carbon` | Sustainability |

## Key Service Methods

- `describe()`
- `evaluate()`
- `create_procurement()`
- `record_transaction()`
- `register_fuel_card()`
- `reconcile_fuel_card()`
- `record_carbon_emission()`
- `register_storage_tank()`
- `register_fuel_agent()`
- `validate_batch()`

_(See `service.py` for complete API.)_

## Interoperability

`transport_fue` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use transport_fue;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `TRANSPORT_FUE_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
