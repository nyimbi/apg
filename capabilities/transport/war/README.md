# Warehouse Operations

## Overview
The Warehouse Operations capability handles all inbound and outbound warehouse processes: goods receiving (ASN, PO, blind), directed putaway with 7 strategies, multi-method picking, packing with weight verification, cross-docking, cycle counting with approval workflows, dock door management, and inventory adjustment control. Cold-chain temperature checks are enforced at receiving. Unapproved inventory adjustments are blocked.

## Capability ID
`transport_war`

## Provides
- warehouse_receiving_workflow: Multi-method goods receipt with damage inspection
- putaway_workflow: Strategy-based putaway with slot verification
- picking_workflow: Single, batch, zone, wave, and robotic pick methods
- packing_workflow: Pack type selection with weight and label verification
- cross_docking_workflow: Cross-dock receipt to dispatch without storage
- cycle_counting_workflow: ABC, random, and full-count cycles with approval
- wms_integration_workflow: Bidirectional sync with 8 WMS platforms

## Requires
- auth, audl, mten, conf: Core platform services
- ntfy: Pick priority and stock alert notifications
- wflo: Goods receipt and pick-pack state machine
- moni: Throughput and capacity monitoring
- comp: Regulatory compliance for bonded/hazmat warehouses
- mqeb: Event streaming
- schd: Receiving appointment scheduling

## Configuration

| Key | Description | Default |
|-----|-------------|---------|
| cycle_counting.discrepancy_threshold_pct | Alert threshold | 1.0% |
| cycle_counting.approval_required_for_adjust | Require approval | true |
| receiving.temperature_check_for_cold_chain | Cold chain check | true |
| packing.weight_check_required | Weight verification | true |

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

## Business Rules

| Rule | Condition | Effect |
|------|-----------|--------|
| unapproved_stock_adjustment_denied | No approver | deny |
| inventory_manipulation_denied | Manipulation detected | deny |
| cold_chain_temp_check_required | Cold chain, no temp check | deny |
| receipt_barcode_required | No barcode scan | deny |
| cross_tenant_warehouse_denied | Cross-tenant write | deny |

## Data Models
- Warehouse: id, warehouse_type, name, location, storage_condition, capacity_sqm, dock_door_count
- GoodsReceipt: id, warehouse_id, receipt_method, supplier_id, line_count, temperature_checked
- PutawayTask: id, receipt_id, strategy, slot_id, confirmed, operator_id
- PickTask: id, order_id, pick_method, warehouse_id, lines_count, priority, completed_at
- PackTask: id, pick_task_id, pack_type, weight_kg, weight_checked, label_printed
- CycleCount: id, warehouse_id, count_type, discrepancy_pct, approved, approved_by
- DockDoor: id, warehouse_id, door_number, status, current_job_ref
- InventoryAdjustment: id, warehouse_id, sku, quantity_before, quantity_after, approved_by

## Streaming Events
- goods_received, putaway_completed, pick_task_created, pick_completed
- packing_completed, cross_dock_completed, cycle_count_completed, inventory_adjusted, dock_door_allocated

## Edge Cases Handled
- Cold chain receiving requires temperature to be recorded — cannot be bypassed
- Inventory adjustments require named approver — empty string is rejected
- Manipulation detection flag independently blocks adjustments even with approver
- Packing weight check is enforced at completion, not at creation
- Cycle count cannot be approved with empty approver ID

## Composability Notes
Interfaces with `transport_car` for cargo receipt against bookings. Outbound pick-pack feeds `transport_dis` for dispatch planning. WMS integration syncs with external SAP EWM or Manhattan Associates systems. Dock door appointments integrate with `transport_sch` scheduling.
