# Warehouse Operations

## Overview
The Warehouse Operations capability handles all inbound and outbound warehouse processes: goods receiving (ASN, PO, blind), directed putaway with 7 strategies, multi-method picking including wave optimisation, packing with weight verification, cross-docking, cycle counting with approval workflows, dock door management, cold-chain telemetry, multi-DC transfer orders, batch/FEFO tracking, SLA breach risk monitoring, carrier performance scorecarding, and equipment utilisation reporting.

Cold-chain temperature checks are enforced at receiving. Unapproved inventory adjustments are blocked.

## Capability ID
`transport_war`

## Provides
- `warehouse_receiving_workflow`: Multi-method goods receipt with ASN auto-close and damage inspection
- `cold_chain_workflow`: Per-reading telemetry with SLA compliance and breach alerting
- `putaway_workflow`: Strategy-based putaway with slot verification and rule engine
- `picking_workflow`: Single, batch, zone, wave, and robotic pick methods with FEFO support
- `packing_workflow`: Pack type selection with weight and label verification
- `cross_docking_workflow`: Cross-dock receipt to dispatch without storage
- `cycle_counting_workflow`: ABC, random, and full-count cycles with approval
- `multi_dc_transfer_workflow`: Transfer orders between distribution centres with in-transit tracking
- `reverse_logistics_workflow`: Grade-based returns disposition with compliance records
- `carrier_performance_workflow`: Carrier KPIs and manifest generation
- `equipment_telemetry_workflow`: Forklift/AGV utilisation and maintenance scheduling
- `sla_monitoring_workflow`: Order SLA breach risk detection and auto-escalation
- `wms_integration_workflow`: Bidirectional sync with 8 WMS platforms

## Requires
- auth, audl, mten, conf: Core platform services
- ntfy: Pick priority, stock alert, and SLA escalation notifications
- wflo: Goods receipt and pick-pack state machine
- moni: Throughput and capacity monitoring
- comp: Regulatory compliance for bonded/hazmat warehouses
- mqeb: Event streaming
- schd: Receiving appointment scheduling
- intel_aler: Cold-chain breach and near-expiry alerting
- transport_dis: Outbound load planning
- transport_car: Carrier booking integration

## Configuration

| Key | Description | Default |
|-----|-------------|---------|
| cycle_counting.discrepancy_threshold_pct | Alert threshold | 1.0% |
| cycle_counting.approval_required_for_adjust | Require approval | true |
| receiving.temperature_check_for_cold_chain | Cold chain check | true |
| packing.weight_check_required | Weight verification | true |
| asn.variance_tolerance_pct | ASN auto-close variance | 2.0% |
| sla.breach_risk_threshold_hours | SLA risk horizon | 4.0h |
| fefo.enabled | First-Expired-First-Out picking | true |

## API Routes

| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| /transport-warehouse/receiving | GET | Goods receipts | transport_war:receiving |
| /transport-warehouse/putaway | GET | Putaway tasks | transport_war:putaway |
| /transport-warehouse/picking | GET | Pick tasks | transport_war:picking |
| /transport-warehouse/packing | GET | Pack tasks | transport_war:packing |
| /transport-warehouse/cross-dock | GET | Cross-dock ops | transport_war:cross_dock |
| /transport-warehouse/cycle-count | GET | Cycle counts | transport_war:cycle_count |
| /transport-warehouse/dock-doors | GET | Dock door status | transport_war:dock_doors |
| /transport-warehouse/inventory | GET | Inventory view | transport_war:inventory |
| /transport-warehouse/cold-chain | GET | Telemetry records | transport_war:cold_chain |
| /transport-warehouse/transfers | GET | DC transfer orders | transport_war:transfers |
| /transport-warehouse/batches | GET | Batch/lot register | transport_war:batches |
| /transport-warehouse/sla-risk | GET | SLA breach risk | transport_war:sla |
| /transport-warehouse/carrier-performance | GET | Carrier KPIs | transport_war:carriers |
| /transport-warehouse/equipment | GET | Equipment utilisation | transport_war:equipment |

## Core Service Methods

### Inbound
| Method | Description |
|--------|-------------|
| `register_warehouse()` | Register a warehouse |
| `receive_goods()` | Record manual goods receipt |
| `receive_goods_async()` | Receive a PO with per-item inventory update |
| `process_asn()` | Parse ASN, auto-close if variance within tolerance |
| `execute_putaway()` / `putaway()` | Strategy-based putaway |
| `cold_chain_telemetry()` | Record and validate cold-chain readings |

### Inventory
| Method | Description |
|--------|-------------|
| `adjust_inventory()` / `inventory_adjustment()` | Approved stock adjustment |
| `initiate_cycle_count()` / `cycle_count()` | Cycle count with auto-approval |
| `complete_cycle_count()` | Close cycle count |
| `sku_lookup()` | Current qty and location by SKU |
| `space_utilisation()` | Warehouse capacity utilisation |
| `slotting_optimisation()` | ABC-based slot recommendations |
| `register_batch()` | Batch/lot registration with FEFO ordering |
| `create_transfer_order()` | Inter-DC inventory deduction |
| `receive_transfer_order()` | Credit destination warehouse from transfer |

### Outbound
| Method | Description |
|--------|-------------|
| `create_pick_task()` / `pick_order()` | Create and assign pick task |
| `complete_pick_task()` | Mark pick task done |
| `wave_pick()` | Wave-based optimised pick clustering |
| `create_pack_task()` / `pack_order()` | Pack with weight verification |
| `complete_packing()` | Finalise pack, print label |
| `ship_order()` | Ship via carrier, allocate dock door |
| `cross_dock()` | Direct inbound-to-outbound without storage |
| `returns_processing()` | Grade-based returns disposition |

### Operations & Analytics
| Method | Description |
|--------|-------------|
| `dock_schedule()` | Dock door booking assignments |
| `dock_door_availability()` | Real-time door status |
| `update_dock_door_status()` | Update individual door |
| `warehouse_analytics()` | Aggregate KPIs for a period |
| `warehouse_kpi_summary()` | Dashboard KPI card |
| `order_accuracy_report()` | Pick accuracy rate |
| `labour_productivity()` | Operator UPH vs benchmark |
| `carrier_performance_report()` | Carrier shipment KPIs |
| `equipment_utilisation_report()` | Forklift/AGV utilisation |
| `sla_breach_risk()` | SLA risk assessment for open picks |
| `export_warehouse_data()` | Operations data export |
| `health_check()` | Service health status |

## Business Rules

| Rule | Condition | Effect |
|------|-----------|--------|
| unapproved_stock_adjustment_denied | No approver | deny |
| inventory_manipulation_denied | Manipulation detected | deny |
| cold_chain_temp_check_required | Cold chain, no temp check | deny |
| receipt_barcode_required | No barcode scan | deny |
| cross_tenant_warehouse_denied | Cross-tenant write | deny |
| asn_variance_exceeds_tolerance | ASN line variance > threshold | flag for review |
| sla_breach_risk_escalation | Queue depth exceeds threshold | escalate to wave pick |

## Data Models
- Warehouse: id, warehouse_type, name, location, storage_condition, capacity_sqm, dock_door_count
- GoodsReceipt: id, warehouse_id, receipt_method, supplier_id, line_count, temperature_checked
- PutawayTask: id, receipt_id, strategy, slot_id, confirmed, operator_id
- PickTask: id, order_id, pick_method, warehouse_id, lines_count, priority, completed_at
- PackTask: id, pick_task_id, pack_type, weight_kg, weight_checked, label_printed
- CycleCount: id, warehouse_id, count_type, discrepancy_pct, approved, approved_by
- DockDoor: id, warehouse_id, door_number, status, current_job_ref
- InventoryAdjustment: id, warehouse_id, sku, quantity_before, quantity_after, approved_by
- WarehouseAgent: id, name, runtime, role, scope

## Streaming Events
- goods_received, putaway_completed, pick_task_created, pick_completed
- packing_completed, cross_dock_executed, cycle_count_completed, inventory_adjusted
- dock_door_allocated, cold_chain_telemetry_recorded, cold_chain_breach_detected
- wave_pick_created, asn_auto_closed, asn_variance_requires_review
- transfer_order_created, transfer_order_received, batch_registered
- sla_breach_risk_detected, carrier_performance_report_generated
- equipment_utilisation_report_generated, return_processed, warehouse_data_exported

## Edge Cases Handled
- Cold chain receiving requires temperature to be recorded — cannot be bypassed
- Inventory adjustments require named approver — empty string is rejected
- Manipulation detection flag independently blocks adjustments even with approver
- Packing weight check is enforced at completion, not at creation
- Cycle count cannot be approved with empty approver ID
- ASN auto-close is blocked if any line variance exceeds configured tolerance
- Transfer order receipt is idempotent for lines already credited
- FEFO batch list is kept sorted by expiry ascending after each batch registration

## Composability Notes
Interfaces with `transport_car` for cargo receipt against bookings. Outbound pick-pack feeds `transport_dis` for dispatch planning. WMS integration syncs with external SAP EWM or Manhattan Associates systems. Dock door appointments integrate with `transport_sch` scheduling. Cold-chain breaches and near-expiry alerts feed `intel_aler`. Multi-DC transfers cross-wire two `transport_war` instances. Equipment maintenance events feed the `maint` capability.

---

## World-Class Enhancements (v2.0)

- **I1.** Warehouse & Distribution — World-Class Improvements
- **I2.** Cold Chain Telemetry Integration
- **I3.** Wave-Based Pick Optimisation
- **I4.** ASN-Driven Auto-Receiving
- **I5.** Real-Time Space Utilisation Heatmap Data
- **I6.** Carrier-Integrated Manifesting
- **I7.** Reverse Logistics Disposition Workflow
- **I8.** Labour Demand Forecasting
- **I9.** Bonded Warehouse Customs Tracking
- **I10.** Multi-DC Transfer Orders
- **I11.** Batch Expiry / FEFO Picking
- **I12.** Automated Putaway Rule Engine
- **I13.** Carrier Performance Scorecarding
- **I14.** IoT Equipment Telemetry (Forklift/AGV)
- **I15.** Hazmat Segregation Enforcement

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
