# Warehouse Management System User Guide

## Overview

`scm_wms` provides directed warehouse operations: warehouse and bin configuration, directed put-away with auto-bin suggestion, FIFO/FEFO/wave picking, pack task management, shipment dispatch, cycle counting with automatic variance adjustment, cross-docking, and slotting optimisation.

## Key Use Cases

- **Bin management**: Define warehouse bin grid (aisle/bay/level) with type, capacity, and pick sequence.
- **Directed put-away**: System suggests best bin based on type, capacity, and pick sequence; inventory updated on confirmation.
- **Directed picking**: FIFO/FEFO/wave/batch pick methods; inventory deducted on completion; shortage flags on short-pick.
- **Pack/ship**: Consolidate picked items into cartons; assign carrier and dispatch with tracking number.
- **Cycle counting**: Spot, ABC, zone, or full warehouse counts; automatic inventory adjustment on variance.
- **Cross-docking**: Route inbound goods directly to outbound orders without put-away.
- **Slotting optimisation**: Rank bins by velocity to minimise picker travel distance.

## API Reference

### Create Warehouse and Bins

```
POST /api/scm/wms/warehouses
{"tenant_id": "acme", "name": "Nairobi DC", "code": "NBI-DC1"}

POST /api/scm/wms/bins
{
  "tenant_id": "acme",
  "warehouse_id": "wh-abc",
  "aisle": "A", "bay": "01", "level": "01",
  "bin_code": "A-01-01",
  "capacity_units": 100,
  "pick_sequence": 1
}
```

### Directed Pick

```
POST /api/scm/wms/pick-tasks
{
  "tenant_id": "acme",
  "order_id": "ORD-001",
  "sku": "PROD-A",
  "quantity": 5,
  "bin_id": "bin-xyz",
  "pick_method": "fifo"
}

POST /api/scm/wms/pick-tasks/{id}/complete
{"tenant_id": "acme", "picked_quantity": 5, "completed_by": "picker.01"}
```

### Cycle Count

```
POST /api/scm/wms/cycle-counts
{"tenant_id": "acme", "warehouse_id": "wh-abc", "count_method": "spot"}

POST /api/scm/wms/cycle-counts/{id}/submit
{
  "tenant_id": "acme",
  "results": [{"bin_id": "bin-xyz", "sku": "PROD-A", "counted_quantity": 48}],
  "completed_by": "stock.controller"
}
```

## Task Status Flow

pending → in_progress → completed | exception | cancelled
