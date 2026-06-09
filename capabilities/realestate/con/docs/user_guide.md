# Property Contracts

**Capability ID**: `realestate_con` | **Domain**: `realestate` | **Version**: `1.0.0`

## Description

Full contract lifecycle management for all real estate agreements: sale/purchase, management contracts, construction contracts, service agreements, joint ventures, and development agreements. Covers party management, digital signatures, milestone tracking, variation orders (with board-approval thresholds), dispute resolution, retention management, and a searchable clause library.

## Installation

```bash
pip install apg-realestate-con
```

## Provides

- `contract_lifecycle_management`
- `contractor_registry_management`
- `milestone_tracking_workflow`
- `variation_order_management`
- `dispute_resolution_workflow`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/realestate/con/dashboard` | `realestate_con:view` | Overview |
| `/realestate/con/contracts` | `realestate_con:contracts` | Contracts |
| `/realestate/con/contracts/<id>` | `realestate_con:contracts` | Contracts |
| `/realestate/con/contractors` | `realestate_con:contractors` | Contractors |
| `/realestate/con/milestones` | `realestate_con:milestones` | Execution |
| `/realestate/con/variations` | `realestate_con:variations` | Execution |
| `/realestate/con/disputes` | `realestate_con:disputes` | Disputes |
| `/realestate/con/clauses` | `realestate_con:clauses` | Library |

## Key Service Methods

- `create_contract()`
- `get_contract()`
- `list_contracts()`
- `update_contract()`
- `execute_contract()`
- `terminate_contract()`
- `sign_contract_party()`
- `get_expiry_pipeline()`
- `register_contractor()`
- `get_contractor()`

_(See `service.py` for complete API.)_

## Interoperability

`realestate_con` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use realestate_con;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `REALESTATE_CON_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
