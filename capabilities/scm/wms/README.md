# Warehouse Management System (scm_wms)

Bin management, put-away rules, directed pick/pack/ship, cycle counting, cross-docking, slotting optimisation.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/scm/wms/health | Health check |
| GET | /api/scm/wms/describe | Capability contract |
| GET | /api/scm/wms/warehouses | List warehouses |
| POST | /api/scm/wms/warehouses | Create warehouse |
| GET | /api/scm/wms/warehouses/{id} | Get warehouse |
| GET | /api/scm/wms/bins | List bins |
| POST | /api/scm/wms/bins | Create bin |
| GET | /api/scm/wms/bins/{id} | Get bin |
| PUT | /api/scm/wms/bins/{id} | Update bin |
| DELETE | /api/scm/wms/bins/{id} | Deactivate bin |
| GET | /api/scm/wms/putaway-tasks | List put-away tasks |
| POST | /api/scm/wms/putaway-tasks | Create put-away task |
| POST | /api/scm/wms/putaway-tasks/{id}/complete | Complete put-away |
| GET | /api/scm/wms/pick-tasks | List pick tasks |
| POST | /api/scm/wms/pick-tasks | Create pick task |
| POST | /api/scm/wms/pick-tasks/{id}/complete | Complete pick |
| GET | /api/scm/wms/pack-tasks | List pack tasks |
| POST | /api/scm/wms/pack-tasks | Create pack task |
| POST | /api/scm/wms/pack-tasks/{id}/complete | Complete pack |
| GET | /api/scm/wms/cycle-counts | List cycle counts |
| POST | /api/scm/wms/cycle-counts | Create cycle count |
| POST | /api/scm/wms/cycle-counts/{id}/submit | Submit results |
| GET | /api/scm/wms/cross-docks | List cross-docks |
| POST | /api/scm/wms/cross-docks | Create cross-dock |
| POST | /api/scm/wms/cross-docks/{id}/complete | Complete cross-dock |
| POST | /api/scm/wms/slotting | Run slotting optimisation |
| GET | /api/scm/wms/inventory | Query inventory |
| GET | /api/scm/wms/analytics | Warehouse analytics |
| GET | /api/scm/wms/audit-events | Audit events |
