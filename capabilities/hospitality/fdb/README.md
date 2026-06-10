# F&B Management (hos_fdb)

Restaurant POS, table management, menu engineering, kitchen display, recipe costing, and inventory control.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/hospitality/fdb/health | Health check |
| GET | /api/hospitality/fdb/menu-items | List menu items |
| POST | /api/hospitality/fdb/menu-items | Create menu item |
| GET | /api/hospitality/fdb/menu-items/{id} | Get menu item |
| PUT | /api/hospitality/fdb/menu-items/{id} | Update menu item |
| DELETE | /api/hospitality/fdb/menu-items/{id} | Deactivate item |
| GET | /api/hospitality/fdb/tables | List tables |
| POST | /api/hospitality/fdb/tables | Create table |
| POST | /api/hospitality/fdb/tables/{id}/seat | Seat guests |
| GET | /api/hospitality/fdb/orders | List orders |
| POST | /api/hospitality/fdb/orders | Create order |
| POST | /api/hospitality/fdb/orders/{id}/send-to-kitchen | Send to KDS |
| POST | /api/hospitality/fdb/orders/{id}/settle | Settle bill |
| DELETE | /api/hospitality/fdb/orders/{id} | Void order |
| POST | /api/hospitality/fdb/kitchen-tickets/{id}/complete | Complete ticket |
| POST | /api/hospitality/fdb/recipes | Create recipe |
| GET | /api/hospitality/fdb/inventory | List inventory |
| POST | /api/hospitality/fdb/inventory | Add inventory item |
| POST | /api/hospitality/fdb/inventory/{id}/adjust | Adjust stock |
| GET | /api/hospitality/fdb/menu-engineering | Menu engineering report |
| GET | /api/hospitality/fdb/revenue-report | Daily revenue |
| GET | /api/hospitality/fdb/dashboard | Dashboard |
