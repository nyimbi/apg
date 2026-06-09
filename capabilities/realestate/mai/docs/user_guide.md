# Facilities Maintenance

**Capability ID**: `realestate_mai` | **Domain**: `realestate` | **Version**: `1.0.0`

## Description

Full CAFM-grade maintenance management: asset register with lifecycle tracking, preventive maintenance (PPM) schedules with automatic next-due calculation, corrective and emergency work orders with SLA deadline enforcement, contractor management with insurance validation, statutory inspection tracking, defect management, and SLA compliance dashboards.

## Installation

```bash
pip install apg-realestate-mai
```

## Provides

- `preventive_maintenance_scheduling`
- `work_order_management`
- `contractor_management`
- `asset_lifecycle_tracking`
- `cafm_integration_bridge`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/realestate/mai/dashboard` | `realestate_mai:view` | Overview |
| `/realestate/mai/work-orders` | `realestate_mai:work_orders` | Operations |
| `/realestate/mai/ppm` | `realestate_mai:ppm` | Planning |
| `/realestate/mai/assets` | `realestate_mai:assets` | Assets |
| `/realestate/mai/assets/<id>` | `realestate_mai:assets` | Assets |
| `/realestate/mai/contractors` | `realestate_mai:contractors` | Contractors |
| `/realestate/mai/inspections` | `realestate_mai:inspections` | Quality |
| `/realestate/mai/defects` | `realestate_mai:defects` | Quality |

## Key Service Methods

- `register_asset()`
- `get_asset()`
- `list_assets()`
- `update_asset()`
- `get_end_of_life_assets()`
- `create_ppm_schedule()`
- `list_ppm_schedules()`
- `complete_ppm()`
- `get_overdue_ppms()`
- `raise_work_order()`

_(See `service.py` for complete API.)_

## Interoperability

`realestate_mai` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use realestate_mai;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `REALESTATE_MAI_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
