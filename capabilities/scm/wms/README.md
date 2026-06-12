# Warehouse Management System (scm_wms)

Bin management, put-away rules, directed pick/pack/ship, cycle counting, cross-docking, slotting optimisation, lot/FEFO tracking, replenishment, dock scheduling, quality inspection, and returns processing.

## World-Class Enhancements (v2.0)

1. **Wave Planning Engine** — groups pick tasks into optimised waves by zone/carrier cutoff, reducing labour cost 15–25%.
2. **Lot/Batch & Serial Tracking** — `lot_id`, `serial_numbers[]`, `expiry_date`, `manufacture_date` fields on every inventory line.
3. **FEFO Enforcement** — `suggest_pick_bins_fefo()` sorts candidates by `expiry_date ASC`; genuine first-expired-first-out.
4. **Dock Appointment Scheduling** — time-slot reservations per dock door for inbound/outbound; enables labour pre-planning.
5. **Replenishment Task Generation** — auto-create `ReplenishmentTask` when pick-face qty drops below configurable threshold.
6. **Returns / Reverse Logistics** — `ReturnReceipt` flow with condition grading (sellable/quarantine/scrap) and bin routing.
7. **Labour Productivity Tracking** — `start_time`, `end_time`, `units_processed` per task; derives lines-per-hour KPIs.
8. **Task Interleaving / Combo Tasking** — assign combined put-away + pick trips to reduce empty travel.
9. **HAZMAT Segregation Rules** — `HazmatRule` model declares incompatible UNNA/GHS classes; prevents co-location.
10. **Inventory Consolidation** — `consolidate_inventory()` plans movement tasks to merge fragmented SKU quantities.
11. **Bin Utilisation Heatmap API** — exports `(aisle, bay, level)` dataset with `fill_pct` for floor-map dashboards.
12. **Carrier Rate Shopping** — `rate_shop()` queries carrier adapters concurrently via `asyncio.gather`; auto-selects cheapest.
13. **Inbound Quality Inspection** — `QualityInspection` gate before put-away; non-conforming lots routed to quarantine.
14. **Persistent Storage Adapter** — `StorageAdapter` ABC + `PostgresStorageAdapter` (asyncpg); survives restarts.
15. **Event Streaming to Broker** — `BrokerAdapter` interface (Kafka/Redis Streams/NATS) replaces in-process `_emit()` list.

## New Methods

### `suggest_pick_bins_fefo` — FEFO pick planning

Returns an ordered list of `(lot, bin, pick_quantity)` tuples that exhaust the earliest-expiring stock first.

```python
svc = WMSService(tenant_id="acme")
picks = await svc.suggest_pick_bins_fefo(
    warehouse_id="wh-001",
    sku="MILK-2L",
    quantity_needed=48.0,
)
# [{"lot_id": "lot-abc", "expiry_date": "2026-06-15", "bin_id": "bin-12", "pick_quantity": 24.0}, ...]
```

### `create_wave_plan` — wave planning

Groups pending pick tasks into a travel-optimised wave and flags tasks at risk of missing carrier cutoffs.

```python
wave = await svc.create_wave_plan(
    warehouse_id="wh-001",
    pick_task_ids=["pt-1", "pt-2", "pt-3", "pt-4"],
    wave_name="MORNING-WAVE-01",
    carrier_cutoff="2026-06-12T14:00:00",
    assigned_pickers=["worker-A", "worker-B"],
)
# wave["late_risk_task_ids"] lists any tasks beyond pick position 20

await svc.release_wave_plan(wave_id=wave["id"], released_by="supervisor-1")
# All constituent pick tasks transition to status="in_progress"
```

### `consolidate_inventory` — bin defragmentation

Identifies a SKU spread across many bins with sub-threshold quantities and returns a movement plan to merge them.

```python
plan = await svc.consolidate_inventory(
    warehouse_id="wh-001",
    sku="BOLT-M6",
    min_qty_threshold=5.0,
)
# plan["moves"] = [{"from_bin_id": ..., "to_bin_id": ..., "quantity": ...}, ...]
# plan["bins_freed"] = 7
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/scm/wms/health | Health check |
| GET | /api/scm/wms/describe | Capability contract |
| GET | /api/scm/wms/audit-events | Audit event log |
| GET | /api/scm/wms/analytics | Warehouse KPI analytics |
| **Warehouses** | | |
| GET | /api/scm/wms/warehouses | List warehouses |
| POST | /api/scm/wms/warehouses | Create warehouse |
| GET | /api/scm/wms/warehouses/{id} | Get warehouse |
| **Bins** | | |
| GET | /api/scm/wms/bins | List bins (filter: warehouse_id, bin_type) |
| POST | /api/scm/wms/bins | Create bin |
| POST | /api/scm/wms/bins/bulk | Bulk-create bins |
| GET | /api/scm/wms/bins/{id} | Get bin |
| PUT | /api/scm/wms/bins/{id} | Update bin |
| DELETE | /api/scm/wms/bins/{id} | Deactivate bin |
| GET | /api/scm/wms/bins/suggest-putaway | Suggest best put-away bin |
| **Lots** | | |
| GET | /api/scm/wms/lots | List lots (filter: sku, bin_id, expiring_before) |
| POST | /api/scm/wms/lots | Register lot with expiry/manufacture dates |
| GET | /api/scm/wms/lots/fefo-suggestion | FEFO pick plan for a SKU + quantity |
| **Put-away** | | |
| GET | /api/scm/wms/putaway-tasks | List put-away tasks |
| POST | /api/scm/wms/putaway-tasks | Create put-away task |
| POST | /api/scm/wms/putaway-tasks/{id}/complete | Complete put-away |
| **Pick** | | |
| GET | /api/scm/wms/pick-tasks | List pick tasks |
| POST | /api/scm/wms/pick-tasks | Create pick task |
| POST | /api/scm/wms/pick-tasks/{id}/complete | Complete pick |
| **Wave planning** | | |
| POST | /api/scm/wms/wave-plans | Create wave plan |
| POST | /api/scm/wms/wave-plans/{id}/release | Release wave to pickers |
| **Pack** | | |
| GET | /api/scm/wms/pack-tasks | List pack tasks |
| POST | /api/scm/wms/pack-tasks | Create pack task |
| POST | /api/scm/wms/pack-tasks/{id}/complete | Complete pack |
| **Ship** | | |
| POST | /api/scm/wms/ship-tasks | Create ship task |
| POST | /api/scm/wms/ship-tasks/{id}/dispatch | Dispatch shipment |
| **Replenishment** | | |
| GET | /api/scm/wms/replenishment-tasks | List replenishment tasks |
| POST | /api/scm/wms/replenishment-tasks | Create replenishment task |
| POST | /api/scm/wms/replenishment-tasks/{id}/complete | Complete replenishment |
| POST | /api/scm/wms/replenishment-tasks/auto-generate | Auto-generate from thresholds |
| **Dock appointments** | | |
| GET | /api/scm/wms/dock-appointments | List appointments |
| POST | /api/scm/wms/dock-appointments | Create dock appointment |
| POST | /api/scm/wms/dock-appointments/{id}/check-in | Check in vehicle |
| **Quality inspections** | | |
| GET | /api/scm/wms/quality-inspections | List inspections |
| POST | /api/scm/wms/quality-inspections | Record quality inspection |
| **Returns** | | |
| GET | /api/scm/wms/return-receipts | List return receipts |
| POST | /api/scm/wms/return-receipts | Create return receipt |
| POST | /api/scm/wms/return-receipts/{id}/process | Process return |
| **Cycle counting** | | |
| GET | /api/scm/wms/cycle-counts | List cycle counts |
| POST | /api/scm/wms/cycle-counts | Create cycle count |
| POST | /api/scm/wms/cycle-counts/{id}/submit | Submit results |
| **Cross-docking** | | |
| GET | /api/scm/wms/cross-docks | List cross-docks |
| POST | /api/scm/wms/cross-docks | Create cross-dock |
| POST | /api/scm/wms/cross-docks/{id}/complete | Complete cross-dock |
| **Slotting** | | |
| POST | /api/scm/wms/slotting | Run slotting optimisation |
| **Inventory** | | |
| GET | /api/scm/wms/inventory | Query inventory |
| POST | /api/scm/wms/inventory/consolidate | Generate consolidation plan |
