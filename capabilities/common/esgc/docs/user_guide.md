# ESG and Carbon Tracking

**Capability ID**: `esgc` | **Domain**: `common` | **Version**: `1.0.0`

## Description

`esgc` is the APG common ESG and carbon tracking capability. It lets generated applications compose tenant-scoped emissions inventories, factor libraries, activity emissions, sustainability reports, reduction targets, compliance

## Installation

```bash
pip install apg-common-esgc
```

## Provides

- `emissions_inventory`
- `factor_library`
- `activity_emissions`
- `sustainability_reporting`
- `target_tracking`

## Requires

- `auth`
- `conf`
- `audl`
- `geos`
- `pred`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/esgc/dashboard` | `esgc:view` | Overview |
| `/esgc/emissions` | `esgc:manage_data` | Inventory |
| `/esgc/factors` | `esgc:manage_data` | Inventory |
| `/esgc/data-sources` | `esgc:manage_data` | Data |
| `/esgc/reports` | `esgc:report` | Reporting |
| `/esgc/targets` | `esgc:view` | Targets |
| `/esgc/agents` | `esgc:govern` | Governance |
| `/esgc/rules` | `esgc:govern` | Governance |

## Key Service Methods

- `uuid7str()`
- `uuid7str()`
- `put()`
- `get()`
- `list()`
- `delete()`
- `log_event()`
- `send()`
- `create_inventory()`
- `register_factor()`

_(See `service.py` for complete API.)_

## Interoperability

`esgc` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use esgc;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `ESGC_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
