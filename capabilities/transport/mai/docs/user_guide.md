# Vehicle Maintenance

**Capability ID**: `transport_mai` | **Domain**: `transport` | **Version**: `1.0.0`

## Description

The Vehicle Maintenance capability manages preventive and corrective maintenance job scheduling, workshop bay allocation, parts inventory and ordering, warranty tracking, vehicle inspections with digital signature capture, and roadworthiness certificate management. It enforces pre-dispatch safety checks and blocks operation of expired-MOT or unsafe vehicles.

## Installation

```bash
pip install apg-transport-mai
```

## Provides

- `preventive_maintenance_schedule_workflow`
- `workshop_management_workflow`
- `parts_inventory_workflow`
- `warranty_tracking_workflow`
- `roadworthiness_compliance_workflow`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/transport-maintenance/dashboard` | `transport_mai:view` | Overview |
| `/transport-maintenance/jobs` | `transport_mai:jobs` | Jobs |
| `/transport-maintenance/jobs/create` | `transport_mai:jobs_write` | Jobs |
| `/transport-maintenance/workshop` | `transport_mai:workshop` | Workshop |
| `/transport-maintenance/parts` | `transport_mai:parts` | Parts |
| `/transport-maintenance/warranty` | `transport_mai:warranty` | Warranty |
| `/transport-maintenance/inspections` | `transport_mai:inspections` | Compliance |
| `/transport-maintenance/roadworthiness` | `transport_mai:compliance` | Compliance |

## Key Service Methods

- `describe()`
- `evaluate()`
- `create_job()`
- `update_job_status()`
- `dispatch_vehicle_check()`
- `allocate_workshop()`
- `order_parts()`
- `record_warranty()`
- `conduct_inspection()`
- `issue_roadworthiness()`

_(See `service.py` for complete API.)_

## Interoperability

`transport_mai` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use transport_mai;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `TRANSPORT_MAI_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
