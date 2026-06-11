# Warehouse & Distribution — World-Class Improvements

**Capability**: `transport_war` | **Domain**: `transport`
**Author**: Nyimbi Odero | **Date**: 2026-06-11

---

## 1. Cold Chain Telemetry Integration

**Current gap**: Temperature check is a boolean flag at receipt time.
**Improvement**: Persist a continuous cold-chain telemetry record per receipt/SKU covering the full dwell time — inbound sensor readings at dock, in-transit (if cross-docked), and storage zone. Expose `cold_chain_telemetry(receipt_id)` returning time-series temperature/humidity data. Alert when any reading exceeds SLA thresholds. Composability hook: feeds `intel_aler` for breach notifications.

---

## 2. Wave-Based Pick Optimisation

**Current gap**: Pick tasks are created order-by-order.
**Improvement**: Add `wave_pick(wave_id, order_ids, zone_config)` that clusters orders by SKU proximity and assigns a consolidated pick route (nearest-neighbour heuristic). Returns a sequenced pick list per zone, reducing travel distance by an estimated 30–40%. Composability hook: feeds `transport_dis` for outbound load planning.

---

## 3. ASN-Driven Auto-Receiving

**Current gap**: Goods receipt is always manually triggered.
**Improvement**: Add `process_asn(asn_payload)` that parses an inbound ASN (EDI 856 or JSON schema), pre-creates a draft receipt, validates expected vs. actual quantities on arrival, and auto-closes the receipt if variance is within tolerance. Composability hook: integrates with `transport_car` carrier bookings and `wflo` state machine.

---

## 4. Real-Time Space Utilisation Heatmap Data

**Current gap**: `space_utilisation` returns a single percentage scalar.
**Improvement**: Expose `space_heatmap(warehouse_id)` that returns per-zone and per-aisle utilisation ratios, identifying hot-spots vs. dead stock zones. Data drives automatic slotting suggestions from `slotting_optimisation`. Composability hook: feeds `intel_dash` visualisation layer.

---

## 5. Carrier-Integrated Manifesting

**Current gap**: Shipment records carrier and tracking number as free-text.
**Improvement**: Add `generate_manifest(shipment_ids, carrier)` that aggregates packed orders into a carrier manifest, validates carrier-specific field requirements (DHL vs. FedEx vs. local courier), and produces a manifest document reference. Composability hook: integrates `transport_sch` for cut-off time enforcement and `transport_dis` for load plans.

---

## 6. Reverse Logistics Disposition Workflow

**Current gap**: `returns_processing` grades items as A/B (restock) or C (dispose) with no intermediate dispositions.
**Improvement**: Add `reverse_logistics_disposition(return_id, items)` supporting grade-level routing: A → direct restock, B → refurbishment queue, C → liquidation, D → destruction with compliance record. Each route triggers appropriate audit events and inventory actions. Composability hook: feeds `comp` for regulatory disposal documentation.

---

## 7. Labour Demand Forecasting

**Current gap**: `labour_productivity` is a backward-looking report.
**Improvement**: Add `forecast_labour_demand(period, forecast_horizon_days)` that projects future staff requirements from historical pick/pack/receive throughput trends, seasonality factors, and open order backlog. Returns recommended headcount by activity and shift. Composability hook: feeds `schd` for shift planning and `hr` capability roster management.

---

## 8. Bonded Warehouse Customs Tracking

**Current gap**: No distinction between bonded and standard storage.
**Improvement**: Add `bond_entry(entry_id, items, customs_reference)` and `bond_release(entry_id, release_authority)` for managing goods held under customs bond. Enforces that items cannot leave bonded zone without a valid release authority. Generates a customs dossier. Composability hook: integrates `comp` regulatory and `finance_tax` for duty calculation.

---

## 9. Multi-DC Transfer Orders

**Current gap**: Inventory is silo'd to a single warehouse.
**Improvement**: Add `create_transfer_order(source_wh_id, dest_wh_id, items)` that creates a paired outbound shipment and inbound receipt, deducts inventory at source, and creates a staged pending receipt at destination. In-transit qty tracked separately to prevent double-counting. Composability hook: cross-wires two `transport_war` instances and feeds `transport_dis` for the inter-DC leg.

---

## 10. Batch Expiry / FEFO Picking

**Current gap**: Pick tasks have no expiry awareness.
**Improvement**: Add `register_batch(sku, batch_id, expiry_date, qty, warehouse_id)` and modify pick task creation to route via FEFO (First Expired, First Out) logic when batch data exists. Returns pick list sorted by expiry ascending. Composability hook: integrates `intel_aler` to flag batches expiring within configurable horizon.

---

## 11. Automated Putaway Rule Engine

**Current gap**: Putaway strategy is a flat enum; slot selection is manual.
**Improvement**: Add `configure_putaway_rules(warehouse_id, rules)` allowing tenant-defined rules: velocity-based, product-class-based, hazmat segregation, and height/weight constraints. `execute_putaway` consults the rule engine to auto-propose the optimal slot. Composability hook: shares SKU master data with `scm_inv` inventory capability.

---

## 12. Carrier Performance Scorecarding

**Current gap**: Shipment records are individual with no aggregated carrier metrics.
**Improvement**: Add `carrier_performance_report(period, carrier)` computing on-time delivery rate, average transit days, damage claim rate, and cost per kg from shipment history. Composability hook: feeds procurement/sourcing capability for carrier contract negotiation and `intel_rep` reporting.

---

## 13. IoT Equipment Telemetry (Forklift/AGV)

**Current gap**: No equipment state tracking.
**Improvement**: Add `register_equipment(equipment_id, type, warehouse_id)` and `ingest_equipment_telemetry(equipment_id, readings)` (position, battery/fuel, fault codes). Enables utilisation reporting and predictive maintenance alerts. `equipment_utilisation_report(warehouse_id, period)` returns uptime and idle-time ratios. Composability hook: integrates `maint` for maintenance scheduling.

---

## 14. Hazmat Segregation Enforcement

**Current gap**: No special handling for hazardous materials.
**Improvement**: Add `classify_hazmat(sku, un_number, hazmat_class, packaging_group)` and enforce segregation rules during putaway — incompatible hazmat classes cannot share a zone. Putaway rule engine rejects invalid slot assignments with a regulation reference. Composability hook: integrates `comp` for SDS document management and `transport_dis` to enforce ADR/IATA restrictions.

---

## 15. SLA-Driven Order Fulfilment Monitoring

**Current gap**: Pick task priority is a free-text string with no SLA enforcement.
**Improvement**: Add `register_order_sla(order_id, sla_tier, due_at)` and `check_sla_breach_risk(warehouse_id)` that returns orders at risk of breaching their SLA window given current pick queue depth and labour availability. Auto-escalates at-risk orders to high-priority pick wave. Composability hook: feeds `intel_aler` for customer notification and `ntfy` for floor supervisor paging.
