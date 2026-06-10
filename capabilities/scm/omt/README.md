# Order Management & Tracking (scm_omt)

Order lifecycle, ATP, backorder management, split shipments, order promising, customer notifications.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/scm/omt/health | Health check |
| GET | /api/scm/omt/describe | Capability contract |
| GET | /api/scm/omt/orders | List orders |
| POST | /api/scm/omt/orders | Create order |
| GET | /api/scm/omt/orders/{id} | Get order |
| PUT | /api/scm/omt/orders/{id} | Update order |
| DELETE | /api/scm/omt/orders/{id} | Delete order |
| POST | /api/scm/omt/orders/{id}/confirm | Confirm order |
| POST | /api/scm/omt/orders/{id}/cancel | Cancel order |
| GET | /api/scm/omt/atp | Check ATP |
| POST | /api/scm/omt/atp | Update ATP |
| GET | /api/scm/omt/backorders | List backorders |
| POST | /api/scm/omt/backorders | Create backorder |
| POST | /api/scm/omt/backorders/{id}/fulfil | Fulfil backorder |
| POST | /api/scm/omt/notifications | Send notification |
| GET | /api/scm/omt/notifications | List notifications |
| GET | /api/scm/omt/analytics | Order analytics |
| GET | /api/scm/omt/analytics/fulfilment-rate | Fulfilment rate |
| GET | /api/scm/omt/audit-events | Audit events |
