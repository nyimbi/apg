# Warehouse Management System (scm_wms)

Bin management, put-away rules, directed pick/pack/ship, cycle counting, cross-docking, slotting optimisation, lot/FEFO tracking, replenishment, dock scheduling, quality inspection, and returns processing.

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
