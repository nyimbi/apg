# World-Class Improvements — scm_wms

© 2025 Datacraft | Author: Nyimbi Odero

---

## 1. Wave Planning Engine

Current directed picking treats each pick task as independent. A wave planning engine groups pick tasks into optimised waves based on carrier cut-off times, zone proximity, and picker capacity. Each wave gets a frozen task list, preventing mid-wave inventory contention and enabling parallel execution across zones. Reduces labour cost by 15–25% in high-volume DCs.

## 2. Lot/Batch and Serial Number Tracking

Inventory records track only SKU + bin. Adding `lot_id`, `serial_numbers[]`, `expiry_date`, and `manufacture_date` fields to inventory lines enables FEFO enforcement based on real expiry data rather than assumed arrival order. Critical for food, pharma, and medical device verticals where regulatory traceability is mandatory.

## 3. FEFO Enforcement with Real Expiry Dates

`pick_method = "fefo"` currently has no expiry data to sort on, making it functionally identical to FIFO. Coupling lot tracking (improvement 2) with a `suggest_pick_bins_fefo()` method that sorts candidate inventory lines by `expiry_date ASC` gives genuine first-expired-first-out behaviour. Avoids write-offs from expired stock.

## 4. Dock Appointment Scheduling

Inbound and outbound docks are unmanaged. A dock door / time-slot reservation system prevents congestion, allows labour pre-planning, and provides data for yard management integration. Model: `DockAppointment(dock_door, appointment_type={inbound|outbound}, scheduled_at, carrier_id, vehicle_plate, status)`.

## 5. Replenishment Task Generation

When a pick-face bin drops below a configurable minimum quantity, the system should automatically generate a `ReplenishmentTask` to top-up from a bulk reserve bin. Replaces ad-hoc supervisor intervention and prevents stockouts at the pick face, a major source of pick-task exceptions in fast-moving operations.

## 6. Returns / Reverse Logistics Processing

No current model for inbound returns. A `ReturnReceipt` flow: capture return reason, inspect condition (`sellable`, `quarantine`, `scrap`), direct to appropriate bin, and update inventory accordingly. Integrates with cycle count quality holds and supplier claims.

## 7. Labour Productivity Tracking

Extend task completion records with `start_time`, `end_time`, and `units_processed` to derive lines-per-hour and cases-per-hour KPIs per worker and per shift. Enables engineered labour standards, gamified performance dashboards, and data-driven staffing models.

## 8. Task Interleaving / Combo Tasking

Assign a single worker a combined trip that performs a put-away on the way out to a pick location and a replenishment drop on the way back. Interleaving reduces empty travel, a major hidden cost in large warehouses. Requires a graph model of bin adjacency or an aisle-sequence travel-distance estimator.

## 9. Hazardous Materials (HAZMAT) Segregation Rules

BIN_TYPE already includes `hazmat` but there are no segregation rules. A `HazmatRule` model that declares incompatible UNNA codes / GHS classes and prevents co-location of incompatible materials in adjacent bins or the same zone. Required for regulatory compliance (ADR, IATA, OSHA).

## 10. Multi-Location Inventory Consolidation

Over time, a single SKU accumulates small quantities across many bins (bin fragmentation). A `consolidate_inventory()` method identifies SKUs with sub-threshold quantities spread across multiple bins and generates movement tasks to merge them into the fewest bins. Frees capacity and improves pick-path efficiency.

## 11. Real-Time Bin Utilisation Heatmap API

Export a spatial utilisation dataset keyed by `(aisle, bay, level)` with `fill_pct`, `last_activity_at`, and `sku_count`. Downstream dashboards can render this as a warehouse floor heatmap, making dead stock and over-utilised zones immediately visible to operations managers without SQL queries.

## 12. Carrier Rate Shopping Integration

`create_ship_task()` accepts a fixed `carrier_id`. Adding a `rate_shop()` async method that queries multiple carrier adapters concurrently with `asyncio.gather` and returns ranked rate options (cost, transit days, service level) lets the system auto-select the cheapest carrier meeting the SLA, directly reducing shipping cost.

## 13. Inbound Quality Inspection Workflow

Receiving currently flows straight to put-away. Inserting a `QualityInspection` step captures sample size, pass/fail results, and defect codes before goods are released to stock. Non-conforming lots are routed to quarantine bins automatically, preventing defective goods from entering the pick-face.

## 14. Persistent Storage Adapter Pattern

All state lives in-process dicts. Introducing a `StorageAdapter` abstract base with a `PostgresStorageAdapter` concrete implementation (using `asyncpg`) lets the service survive restarts and scale horizontally. The adapter pattern (already partially present in `domain/adapters.py`) isolates persistence logic from business logic, preserving testability.

## 15. Event Streaming to Message Broker

`_emit()` appends to an in-process list. Replacing or augmenting this with a `BrokerAdapter` interface (Kafka / Redis Streams / NATS) allows downstream capabilities (scm_oms, scm_tms, intel_alerts) to react to WMS events in real time without polling. Enables event-driven architecture patterns: inventory reservation, shipment status propagation, and alerting on cycle-count variances.

---

*Priority order for implementation: 14 (persistence) → 2 (lot tracking) → 3 (FEFO) → 5 (replenishment) → 1 (wave planning) → 8 (task interleaving) → remaining in business-value order.*
