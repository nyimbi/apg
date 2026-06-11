# Warehouse Operations — User Guide

**Capability ID**: `transport_war` | **Domain**: `transport` | **Version**: `1.1.0`

---

## Overview

The Warehouse Operations capability (`transport_war`) is the operational backbone for distribution centre management. It covers the full goods lifecycle: inbound receipt via PO, blind, and ASN methods; directed putaway; wave-optimised picking; weight-verified packing; cross-docking; cold-chain telemetry; multi-DC transfers; FEFO batch tracking; reverse logistics; SLA breach risk monitoring; carrier performance scorecarding; and equipment utilisation reporting.

All operations are tenant-scoped, fully audited, and emit streaming events to the `mqeb` event bus.

---

## Installation

```bash
pip install apg-transport-war
```

---

## Quick Start

```python
import asyncio
from apg_transport_war import WarehouseOperationsService

svc = WarehouseOperationsService(tenant_id="acme", actor_id="ops_user")

# Register a warehouse
wh = svc.register_warehouse(
    "WH-001", "acme", "general", "Nairobi Main DC",
    "Industrial Area, Nairobi", "ambient", 5000.0, 12,
)

# Receive a PO
async def run():
    receipt = await svc.receive_goods_async(
        po_id="PO-2026-0001",
        items=[{"sku": "SKU-A", "qty": 100}, {"sku": "SKU-B", "qty": 50}],
        condition="good",
        received_by="operator_01",
        warehouse_id="WH-001",
    )
    print(receipt)

asyncio.run(run())
```

---

## Core Workflows

### 1. Goods Receiving

#### Manual Receipt
```python
receipt = svc.receive_goods(
    "GR-001", "acme", "WH-001", "po_based",
    "SUPP-001", "PO-2026-001", 10, "2026-06-11T08:00:00Z",
    barcode_scanned=True, damage_inspection_completed=True,
)
```

#### ASN Auto-Close
```python
result = await svc.process_asn(
    asn_id="ASN-2026-001",
    asn_payload={
        "supplier_id": "SUPP-001",
        "po_reference": "PO-2026-001",
        "lines": [
            {"sku": "SKU-A", "expected_qty": 100, "actual_qty": 98},
            {"sku": "SKU-B", "expected_qty": 50,  "actual_qty": 50},
        ],
    },
    warehouse_id="WH-001",
    variance_tolerance_pct=3.0,
)
# auto_closed=True because all variances <= 3%
```

#### Cold-Chain Telemetry
```python
telem = await svc.cold_chain_telemetry(
    receipt_id="GR-001",
    readings=[
        {"ts": "2026-06-11T08:05:00Z", "temp_c": 4.2, "humidity_pct": 85.0, "sensor_id": "S01"},
        {"ts": "2026-06-11T08:10:00Z", "temp_c": 3.8, "humidity_pct": 84.5, "sensor_id": "S01"},
        {"ts": "2026-06-11T08:15:00Z", "temp_c": 9.1, "humidity_pct": 84.0, "sensor_id": "S01"},  # breach
    ],
    sla_min_c=2.0,
    sla_max_c=8.0,
)
# telem["breach_count"] == 1
# telem["overall_compliant"] == False
```

---

### 2. Putaway

```python
result = await svc.putaway(
    receipt_id="GR-001",
    locations=[
        {"sku": "SKU-A", "slot_id": "A-01-01"},
        {"sku": "SKU-B", "slot_id": "B-03-07"},
    ],
    operator_id="operator_01",
)
```

---

### 3. Inventory Management

#### Batch/FEFO Registration
```python
batch = await svc.register_batch(
    sku="SKU-A",
    batch_id="BATCH-2026-001",
    expiry_date="2026-12-31",
    qty=98,
    warehouse_id="WH-001",
    approved_by="qa_manager",
)
# batch["fefo_position"] indicates expiry sort order
```

#### Inventory Adjustment
```python
adj = await svc.inventory_adjustment(
    sku="SKU-A",
    quantity=-5,          # remove 5 units (damage write-off)
    reason="damage_writeoff",
    approved_by="warehouse_manager",
    warehouse_id="WH-001",
)
```

#### SKU Lookup
```python
stock = await svc.sku_lookup("SKU-A")
# {"sku": "SKU-A", "qty": 93, "in_stock": True, ...}
```

---

### 4. Multi-DC Transfers

```python
# At source DC
transfer = await svc.create_transfer_order(
    transfer_id="TRF-2026-001",
    source_warehouse_id="WH-001",
    dest_warehouse_id="WH-002",
    items=[{"sku": "SKU-A", "qty": 30}],
    approved_by="logistics_manager",
)

# At destination DC (after physical arrival)
received = await svc.receive_transfer_order(
    transfer_id="TRF-2026-001",
    dest_warehouse_id="WH-002",
    received_by="wh002_receiver",
)
```

---

### 5. Picking

#### Standard Pick
```python
pick = await svc.pick_order(
    order_id="ORD-2026-001",
    picker_id="picker_01",
    warehouse_id="WH-001",
    pick_method="single_order",
    lines=[{"sku": "SKU-A", "qty": 2, "location": "A-01-01"}],
)
```

#### Wave Pick (Batch Optimisation)
```python
wave = await svc.wave_pick(
    wave_id="WAVE-2026-001",
    order_ids=["ORD-001", "ORD-002", "ORD-003", "ORD-004"],
    warehouse_id="WH-001",
    zone_config={"ZONE-A": 0, "ZONE-B": 1, "ZONE-C": 2},
    operator_id="picker_01",
)
# wave["estimated_travel_reduction_pct"] == 35.0
# wave["sequenced_pick_list"] sorted by zone
```

---

### 6. Packing and Shipping

```python
pack = await svc.pack_order(
    pick_id="PICK-ORD-001-ABCD12",
    packer_id="packer_01",
    box_type="standard",
    weight_kg=2.4,
)

shipment = await svc.ship_order(
    pack_id=pack["id"],
    carrier="dhl",
    tracking="1234567890",
    dock_door_id="DOOR-001",
)
```

---

### 7. Cross-Docking

```python
# Receipt must already exist
xdock = await svc.cross_dock(
    inbound_id="GR-001",
    outbound_orders=["ORD-010", "ORD-011"],
    warehouse_id="WH-001",
)
# xdock["transit_storage"] == False
```

---

### 8. Cycle Counting

```python
count = await svc.cycle_count(
    location_id="A-01-01",
    counter_id="counter_01",
    warehouse_id="WH-001",
    count_type="random",
)
# count["auto_approved"] == True if discrepancy < 1%
```

---

### 9. SLA Breach Risk Monitoring

```python
risk = await svc.sla_breach_risk("WH-001")
# risk["at_risk"] == True if estimated clear time > 4 hours
# risk["recommended_action"] == "escalate_to_wave_pick"
```

Integrate with `ntfy` to page supervisors when `at_risk == True`:

```python
if risk["at_risk"]:
    await notify.send(channel="ops", message=f"SLA risk: {risk['open_pick_tasks']} open picks, est {risk['estimated_clear_hours']}h to clear")
```

---

### 10. Analytics and Reporting

#### Warehouse KPIs
```python
kpis = await svc.warehouse_analytics("2026-06")
kpis = await svc.warehouse_kpi_summary()
```

#### Carrier Performance
```python
report = await svc.carrier_performance_report("2026-06", carrier="dhl")
```

#### Equipment Utilisation
```python
util = await svc.equipment_utilisation_report("WH-001", "2026-06")
# util["equipment"] contains forklift and AGV utilisation %
# util["overall_fleet_utilisation_pct"] triggers maintenance_due flag
```

#### Order Accuracy
```python
accuracy = await svc.order_accuracy_report("2026-06")
```

#### Labour Productivity
```python
labour = await svc.labour_productivity("WH-001", "2026-06")
```

#### Space Utilisation
```python
space = await svc.space_utilisation("WH-001")
# status: "normal" | "high" | "critical"
```

---

### 11. Returns Processing

```python
returns = await svc.returns_processing(
    receipt_id="RET-GR-001",
    items=[
        {"sku": "SKU-A", "qty": 2, "condition_grade": "A"},  # restock
        {"sku": "SKU-B", "qty": 1, "condition_grade": "C"},  # dispose
    ],
    reason="customer_return",
    warehouse_id="WH-001",
)
```

---

## Configuration Reference

All keys are tenant-scoped. Override via the `conf` capability or `TRANSPORT_WAR_` environment variables.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `receiving.temperature_check_for_cold_chain` | bool | true | Enforce temp check at receipt |
| `receiving.barcode_required` | bool | true | Require barcode scan |
| `putaway.default_strategy` | str | fixed_slot | Default putaway strategy |
| `picking.fefo_enabled` | bool | true | Use FEFO when batch data present |
| `packing.weight_check_required` | bool | true | Enforce weight check at pack |
| `cycle_counting.discrepancy_threshold_pct` | float | 1.0 | Auto-approve threshold |
| `asn.variance_tolerance_pct` | float | 2.0 | ASN auto-close variance threshold |
| `sla.breach_risk_threshold_hours` | float | 4.0 | Hours before SLA alert triggers |
| `cold_chain.sla_min_c` | float | 2.0 | Minimum SLA temperature (Celsius) |
| `cold_chain.sla_max_c` | float | 8.0 | Maximum SLA temperature (Celsius) |

---

## Composability

```apg
use transport_war;           // warehouse operations
use transport_dis;           // outbound dispatch planning (fed by ship_order)
use transport_car;           // carrier bookings (feeds process_asn)
use intel_aler;              // cold-chain breach and SLA alerts
use schd;                    // dock door appointment scheduling
use comp;                    // regulatory compliance (hazmat, bonded)
use maint;                   // equipment maintenance scheduling
```

---

## Business Rules Summary

| Rule | Trigger | Effect |
|------|---------|--------|
| unapproved_stock_adjustment_denied | Missing approver | PermissionError |
| inventory_manipulation_denied | Manipulation flag set | PermissionError |
| cold_chain_temp_check_required | Cold chain receipt, no temp check | PermissionError |
| receipt_barcode_required | Missing barcode scan | PermissionError |
| cross_tenant_warehouse_denied | Cross-tenant write attempt | PermissionError |
| asn_variance_exceeds_tolerance | Line variance > tolerance | flags for review |
| sla_breach_risk_escalation | Queue depth > threshold | audit event + recommendation |

---

## Streaming Events Reference

| Event | Source Method | Payload |
|-------|--------------|---------|
| `goods_received` | `receive_goods` | receipt_id, warehouse_id |
| `putaway_completed` | `execute_putaway` | task_id, slot_id |
| `pick_task_created` | `create_pick_task` | task_id, order_id |
| `pick_completed` | `complete_pick_task` | task_id |
| `packing_completed` | `complete_packing` | task_id |
| `cross_dock_executed` | `cross_dock` | xdock_id |
| `cycle_count_completed` | `complete_cycle_count` | count_id, discrepancy_pct |
| `inventory_adjusted` | `adjust_inventory` | adjustment_id, sku |
| `cold_chain_telemetry_recorded` | `cold_chain_telemetry` | telemetry_id |
| `cold_chain_breach_detected` | `cold_chain_telemetry` | telemetry_id, breach_count |
| `wave_pick_created` | `wave_pick` | wave_id, order_count |
| `asn_auto_closed` | `process_asn` | receipt_id |
| `asn_variance_requires_review` | `process_asn` | receipt_id |
| `transfer_order_created` | `create_transfer_order` | transfer_id |
| `transfer_order_received` | `receive_transfer_order` | transfer_id |
| `batch_registered` | `register_batch` | batch_id, sku |
| `sla_breach_risk_detected` | `sla_breach_risk` | warehouse_id |
| `carrier_performance_report_generated` | `carrier_performance_report` | report_id |
| `equipment_utilisation_report_generated` | `equipment_utilisation_report` | report_id |
| `return_processed` | `returns_processing` | return_id |
| `order_shipped` | `ship_order` | shipment_id |

---

## Further Reading

- `service.py` — Full business logic implementation
- `models.py` — In-memory data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder view models
- `capability_contract.py` — Policy rules and supported enumerations
- `WORLD_CLASS_IMPROVEMENTS.md` — 15 prioritised improvement roadmap items
- `SPECIFICATION.md` — Detailed capability specification
