# Ore Processing & Metallurgy

**Capability ID**: `mining_ore` | **Domain**: `mining` | **Version**: `1.0.0`

## Description

Manages ore processing plant operations including plant feed tracking, process circuit status monitoring, reagent inventory management, metallurgical mass balance preparation and approval, product quality assurance, ore reconciliation, and process deviation alert management. Enforces metallurgical integrity constraints including recovery bounds [0, 100%], cyanide code compliance, approval gating before balance publication, and off-specification product dispatch controls.

## Installation

```bash
pip install apg-mining-ore
```

## Provides

- `plant_feed_tracking`
- `metallurgical_balance_workflow`
- `reagent_management`
- `recovery_optimisation_tracking`
- `product_quality_management`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/mining-ore/dashboard` | `mining_ore:view` | Overview |
| `/mining-ore/plant-feed` | `mining_ore:view` | Plant Feed |
| `/mining-ore/plant-feed/record` | `mining_ore:write` | Plant Feed |
| `/mining-ore/circuits` | `mining_ore:view` | Process |
| `/mining-ore/circuits/:id` | `mining_ore:view` | Process |
| `/mining-ore/reagents` | `mining_ore:view` | Reagents |
| `/mining-ore/reagents/usage` | `mining_ore:write` | Reagents |
| `/mining-ore/met-balance` | `mining_ore:met_balance` | Metallurgy |

## Key Service Methods

- `record_plant_feed()`
- `get_plant_feed()`
- `list_plant_feeds()`
- `get_feed_summary()`
- `update_circuit_status()`
- `get_current_circuit_statuses()`
- `record_reagent_usage()`
- `add_reagent_stock()`
- `get_reagent_inventory()`
- `list_reagent_usage()`

_(See `service.py` for complete API.)_

## Interoperability

`mining_ore` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use mining_ore;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `MINING_ORE_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
