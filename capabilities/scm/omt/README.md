# Order Management & Tracking (scm_omt)

Order lifecycle, ATP (available-to-promise), backorder management, split shipments,
order promising, customer notifications, RMA / reverse logistics, SLA breach detection,
and priority-based order routing.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/scm/omt/health | Health check |
| GET | /api/scm/omt/describe | Capability contract |
| GET | /api/scm/omt/orders | List orders |
| POST | /api/scm/omt/orders | Create order (supports idempotency_key) |
| GET | /api/scm/omt/orders/{id} | Get order |
| PUT | /api/scm/omt/orders/{id} | Update order |
| DELETE | /api/scm/omt/orders/{id} | Delete order |
| POST | /api/scm/omt/orders/{id}/confirm | Confirm order |
| POST | /api/scm/omt/orders/{id}/cancel | Cancel order |
| POST | /api/scm/omt/orders/{id}/hold | Place on hold |
| POST | /api/scm/omt/orders/{id}/release | Release hold |
| GET | /api/scm/omt/orders/queue | Priority-scored order queue |
| POST | /api/scm/omt/orders/{id}/route | Route lines to warehouses |
| GET | /api/scm/omt/orders/{id}/delivery-windows | Available delivery windows |
| POST | /api/scm/omt/orders/bulk-confirm | Bulk confirm (bounded concurrency) |
| GET | /api/scm/omt/atp | Point-in-time ATP check |
| POST | /api/scm/omt/atp | Update ATP |
| POST | /api/scm/omt/atp/horizon | Update date-bucketed ATP horizon |
| GET | /api/scm/omt/atp/horizon | ATP-by-date check |
| GET | /api/scm/omt/backorders | List backorders |
| POST | /api/scm/omt/backorders | Create backorder |
| POST | /api/scm/omt/backorders/{id}/fulfil | Fulfil backorder |
| GET | /api/scm/omt/split-shipments | List split shipments |
| POST | /api/scm/omt/split-shipments | Create split shipment |
| POST | /api/scm/omt/promises | Promise order delivery |
| POST | /api/scm/omt/promises/{id}/revoke | Revoke promise |
| POST | /api/scm/omt/promises/re-promise | Re-promise all breached promises |
| POST | /api/scm/omt/notifications | Send notification |
| GET | /api/scm/omt/notifications | List notifications |
| GET | /api/scm/omt/rmas | List RMAs |
| POST | /api/scm/omt/rmas | Create RMA |
| POST | /api/scm/omt/rmas/{id}/approve | Approve RMA |
| POST | /api/scm/omt/rmas/{id}/receive | Record return receipt |
| POST | /api/scm/omt/customers/{id}/tier | Set customer SLA tier |
| GET | /api/scm/omt/analytics | Order analytics |
| GET | /api/scm/omt/analytics/fulfilment-rate | Fulfilment rate |
| GET | /api/scm/omt/analytics/sla-breaches | Detect SLA breaches |
| GET | /api/scm/omt/audit-events | Audit events (causal chain) |

## Order State Machine

```
draft → confirmed → allocated → picking → packed → shipped ──────────────┐
  │          │           │          │        │           │                │
  └──────────┴───────────┴──────────┴────────┴──on_hold──┘          delivered
                                                   │
                                              cancelled
                     packed → partially_shipped → shipped → delivered
```

All transitions are validated against the formal `TRANSITIONS` adjacency map in `service.py`.
Invalid transitions raise `ValueError`.

## Key New Capabilities (v1.1)

- **Idempotent order creation** — pass `idempotency_key` to prevent duplicate orders on retry.
- **Date-bucketed ATP horizon** — supply/demand events projected forward; `check_atp_by_date` answers "will stock be available by date X?"
- **Priority order queue** — `get_order_queue` scores orders by `revenue × priority × customer_tier` for warehouse pick sequencing.
- **Order routing** — `route_order` assigns lines to warehouses using `consolidate` or `fastest` policy.
- **Delivery window negotiation** — `get_available_delivery_windows` filters candidate dates against warehouse calendar + ATP.
- **SLA breach detection** — `detect_sla_breaches` scans all active orders; supports auto-escalation notifications.
- **Re-promising engine** — `re_promise_breached_orders` revokes stale promises and issues system re-promises.
- **RMA / reverse logistics** — full create → approve → receive lifecycle for returns.
- **Bounded bulk operations** — `_bounded_gather` caps concurrency at 10 (configurable) to protect downstream systems.
- **Causal audit trail** — every audit event carries `causation_id` and `correlation_id` for end-to-end trace reconstruction.
