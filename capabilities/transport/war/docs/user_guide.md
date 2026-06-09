# Warehouse Operations

**Capability ID**: `transport_war` | **Domain**: `transport` | **Version**: `1.0.0`

## Description

The Warehouse Operations capability handles all inbound and outbound warehouse processes: goods receiving (ASN, PO, blind), directed putaway with 7 strategies, multi-method picking, packing with weight verification, cross-docking, cycle counting with approval workflows, dock door management, and inventory adjustment control. Cold-chain temperature checks are enforced at receiving. Unapproved inventory adjustments are blocked.

## Installation

```bash
pip install apg-transport-war
```

## Provides

- `warehouse_receiving_workflow`
- `putaway_workflow`
- `picking_workflow`
- `packing_workflow`
- `cross_docking_workflow`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/transport-warehouse/dashboard` | `transport_war:view` | Overview |
| `/transport-warehouse/receiving` | `transport_war:receiving` | Inbound |
| `/transport-warehouse/putaway` | `transport_war:putaway` | Inbound |
| `/transport-warehouse/inventory` | `transport_war:inventory` | Inventory |
| `/transport-warehouse/picking` | `transport_war:picking` | Outbound |
| `/transport-warehouse/packing` | `transport_war:packing` | Outbound |
| `/transport-warehouse/cross-dock` | `transport_war:cross_dock` | Operations |
| `/transport-warehouse/cycle-count` | `transport_war:cycle_count` | Inventory |

## Key Service Methods

- `describe()`
- `evaluate()`
- `register_warehouse()`
- `receive_goods()`
- `execute_putaway()`
- `create_pick_task()`
- `complete_pick_task()`
- `create_pack_task()`
- `complete_packing()`
- `initiate_cycle_count()`

_(See `service.py` for complete API.)_

## Interoperability

`transport_war` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use transport_war;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `TRANSPORT_WAR_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
