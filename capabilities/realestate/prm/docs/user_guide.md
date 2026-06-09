# Property Management

**Capability ID**: `realestate_prm` | **Domain**: `realestate` | **Version**: `1.0.0`

## Description

Central portfolio management for all real estate assets. Registers properties and units, manages owner entities and their distributions, tracks performance KPIs (occupancy, WAULT, yield), coordinates handovers, and provides an owner portal and searchable data room for each property.

## Installation

```bash
pip install apg-realestate-prm
```

## Provides

- `property_portfolio_management`
- `unit_management`
- `owner_portal_service`
- `property_performance_reporting`
- `portfolio_analytics`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/realestate/prm/dashboard` | `realestate_prm:view` | Overview |
| `/realestate/prm/portfolio` | `realestate_prm:portfolio` | Portfolio |
| `/realestate/prm/properties` | `realestate_prm:properties` | Properties |
| `/realestate/prm/properties/<id>` | `realestate_prm:properties` | Properties |
| `/realestate/prm/units` | `realestate_prm:units` | Units |
| `/realestate/prm/owners` | `realestate_prm:owners` | Owners |
| `/realestate/prm/owner-portal` | `realestate_prm:owner_portal` | Owners |
| `/realestate/prm/performance` | `realestate_prm:performance` | Analytics |

## Key Service Methods

- `register_owner()`
- `get_owner()`
- `list_owners()`
- `update_owner()`
- `register_property()`
- `get_property()`
- `list_properties()`
- `update_property()`
- `delete_property()`
- `create_unit()`

_(See `service.py` for complete API.)_

## Interoperability

`realestate_prm` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use realestate_prm;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `REALESTATE_PRM_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
