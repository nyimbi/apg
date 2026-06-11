# Warehouse Management System User Guide

## Overview

`scm_wms` provides directed warehouse operations across the full inbound-to-outbound flow:

- Warehouse and bin configuration with bulk creation
- Lot/batch registration with expiry dates for genuine FEFO enforcement
- Directed put-away with auto-bin suggestion
- FIFO/FEFO/wave/batch picking with wave planning
- Pack task management and shipment dispatch
- Replenishment task generation (manual and automated threshold-based)
- Dock appointment scheduling with collision detection
- Inbound quality inspection with lot quarantine routing
- Customer returns processing (sellable / quarantine / scrap routing)
- Cycle counting with automatic variance adjustment
- Cross-docking (inbound directly to outbound)
- Slotting optimisation by velocity
- Inventory consolidation planning for fragmented stock

---

## Key Use Cases

### 1. Bin Management

Define the warehouse bin grid with aisle/bay/level coordinates, bin type (`standard`, `bulk`, `cold`, `hazmat`, `quarantine`, `oversize`), capacity, and pick sequence. Bulk-create up to hundreds of bins in a single call.

```
POST /api/scm/wms/bins/bulk
{
  "tenant_id": "acme",
  "warehouse_id": "wh-abc",
  "bins_data": [
    {"aisle": "A", "bay": "01", "level": "01", "bin_code": "A-01-01", "capacity_units": 100, "pick_sequence": 1},
    {"aisle": "A", "bay": "01", "level": "02", "bin_code": "A-01-02", "capacity_units": 100, "pick_sequence": 2}
  ]
}
```

### 2. Lot Registration and FEFO Picking

Register inbound lots with expiry dates before put-away. Use the FEFO suggestion endpoint to get an ordered pick plan that exhausts earliest-expiring stock first.

```
POST /api/scm/wms/lots
{
  "tenant_id": "acme", "sku": "DAIRY-001", "lot_number": "LOT-2026-06",
  "quantity": 500, "expiry_date": "2026-09-01", "bin_id": "bin-cold-01"
}

GET /api/scm/wms/lots/fefo-suggestion?warehouse_id=wh-abc&sku=DAIRY-001&quantity_needed=120
```

Response lists lot/bin/quantity tuples in expiry order until the full quantity is covered.

### 3. Directed Put-Away

System suggests the best available bin based on type, capacity, and pick sequence. Confirming the task updates the inventory ledger.

```
POST /api/scm/wms/putaway-tasks
{"tenant_id": "acme", "receipt_id": "REC-001", "sku": "PROD-A", "quantity": 50}

POST /api/scm/wms/putaway-tasks/{id}/complete
{"tenant_id": "acme", "confirmed_bin_id": "bin-xyz", "completed_by": "receiver.01"}
```

### 4. Quality Inspection Before Put-Away

Capture inspection results on inbound goods. `fail` or `quarantine` outcomes update the lot status and route goods to quarantine bins.

```
POST /api/scm/wms/quality-inspections
{
  "tenant_id": "acme", "receipt_id": "REC-001", "sku": "PROD-A",
  "lot_id": "lot-abc", "quantity_inspected": 50, "sample_size": 5,
  "outcome": "quarantine", "defect_codes": ["MOLD", "DMGD"],
  "inspected_by": "qa.officer"
}
```

### 5. Wave Planning and Directed Picking

Group pick tasks into a wave sorted by bin pick_sequence, then release to pickers as a batch. Reduces empty travel and enables concurrent zone picking.

```
POST /api/scm/wms/wave-plans
{
  "tenant_id": "acme", "warehouse_id": "wh-abc",
  "pick_task_ids": ["pick-001", "pick-002", "pick-003"],
  "wave_name": "WAVE-AM-01", "carrier_cutoff": "2026-06-11T14:00:00Z",
  "assigned_pickers": ["picker.01", "picker.02"]
}

POST /api/scm/wms/wave-plans/{id}/release
{"tenant_id": "acme", "released_by": "supervisor.01"}
```

Individual pick tasks can still be completed independently:

```
POST /api/scm/wms/pick-tasks/{id}/complete
{"tenant_id": "acme", "picked_quantity": 5, "completed_by": "picker.01"}
```

### 6. Replenishment

Manually create a replenishment task to move stock from a bulk reserve bin to a pick-face bin:

```
POST /api/scm/wms/replenishment-tasks
{
  "tenant_id": "acme", "warehouse_id": "wh-abc", "sku": "PROD-A",
  "source_bin_id": "bin-bulk-01", "target_bin_id": "bin-pickface-07",
  "quantity": 50
}
```

Auto-generate replenishments for all SKUs below their configured thresholds:

```
POST /api/scm/wms/replenishment-tasks/auto-generate
{
  "tenant_id": "acme", "warehouse_id": "wh-abc",
  "thresholds": {"PROD-A": 10, "PROD-B": 5},
  "reserve_bin_map": {"PROD-A": "bin-bulk-01", "PROD-B": "bin-bulk-02"},
  "replenish_qty_map": {"PROD-A": 50, "PROD-B": 25}
}
```

### 7. Dock Appointment Scheduling

Reserve dock doors by time-slot to prevent congestion. The system checks for duplicate appointments at the same door + time.

```
POST /api/scm/wms/dock-appointments
{
  "tenant_id": "acme", "warehouse_id": "wh-abc",
  "dock_door": "DOCK-3", "appointment_type": "inbound",
  "scheduled_at": "2026-06-11T08:00:00Z", "carrier_id": "DHL",
  "vehicle_plate": "KBZ 001X"
}

POST /api/scm/wms/dock-appointments/{id}/check-in
{"tenant_id": "acme", "checked_in_by": "gatehouse.01"}
```

### 8. Returns Processing

Receive customer returns, capture condition and reason, and route to the correct bin automatically.

```
POST /api/scm/wms/return-receipts
{
  "tenant_id": "acme", "original_order_id": "ORD-456",
  "sku": "PROD-A", "quantity": 3,
  "return_reason": "wrong_item", "condition": "sellable",
  "customer_id": "CUST-100"
}

POST /api/scm/wms/return-receipts/{id}/process
{"tenant_id": "acme", "processed_by": "returns.clerk"}
```

`sellable` condition adds quantity back to inventory. `quarantine` and `scrap` route to typed bins without restocking.

### 9. Pack / Ship

```
POST /api/scm/wms/pack-tasks
{
  "tenant_id": "acme", "order_id": "ORD-001",
  "pick_task_ids": ["pick-001", "pick-002"],
  "packing_station": "PS-3"
}

POST /api/scm/wms/pack-tasks/{id}/complete
{
  "tenant_id": "acme",
  "cartons": [{"carton_id": "CTN-001", "weight_kg": 2.5, "items": ["PROD-A×5"]}],
  "total_weight_kg": 2.5, "completed_by": "packer.01"
}

POST /api/scm/wms/ship-tasks/{id}/dispatch
{"tenant_id": "acme", "tracking_number": "DHL1234567890", "dispatched_by": "shipper.01"}
```

### 10. Cycle Count

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

Variances are automatically adjusted in the inventory ledger.

### 11. Inventory Consolidation

Identify SKUs fragmented across many low-quantity bins and plan movements to merge them:

```
POST /api/scm/wms/inventory/consolidate
{
  "tenant_id": "acme", "warehouse_id": "wh-abc",
  "sku": "PROD-A", "min_qty_threshold": 5.0
}
```

Response lists source bins, target bin, and movement quantities. Use `create_replenishment_task` to execute each line.

---

## Task Status Flow

```
pending → in_progress → completed | exception | cancelled
```

Dock appointments: `scheduled → checked_in → completed | cancelled`

Quality inspections: always `completed` (terminal) with outcome `pass | fail | quarantine | scrap`

Return receipts: `pending → processed`

---

## Reference Constants

| Constant | Values |
|----------|--------|
| `BIN_TYPES` | standard, bulk, cold, hazmat, quarantine, oversize |
| `PICK_METHODS` | fifo, fefo, lifo, zone, wave, batch |
| `COUNT_METHODS` | spot, abc, full, zone |
| `INSPECTION_OUTCOMES` | pass, fail, quarantine, scrap |
| `RETURN_CONDITIONS` | sellable, quarantine, scrap |
| `DOCK_TYPES` | inbound, outbound |
