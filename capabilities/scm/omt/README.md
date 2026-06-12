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

## World-Class Enhancements (v2.0)

1. **Order Line-Level Status Tracking** — each line transitions independently (`allocated`, `backordered`, `picked`, `packed`, `shipped`, `cancelled`) enabling partial-fulfilment workflows.
2. **Partial Fulfilment & Overship Guard** — `shipped_quantity` accumulator per line; rejects requests exceeding ordered quantity; auto-promotes order to `partially_shipped`.
3. **ATP Horizon Simulation** — date-bucketed ATP built from supply/demand events; `check_atp_by_date` answers "will stock be available by date X?" (capable-to-promise foundation).
4. **Dynamic Re-promising Engine** — scans active promises against revised ATP profile, flags/revokes stale promises, triggers customer notification pipeline automatically.
5. **Order Scoring & Priority Lanes** — composite score (`revenue × priority_weight × customer_tier_weight`); `get_order_queue` returns a priority-sorted warehouse pick list.
6. **Rule-Based Order Routing** — `route_order` assigns lines to warehouses via `consolidate` or `fastest` policy; integrates with `scm_wms` via event bus.
7. **Idempotency Keys on Order Creation** — `idempotency_key` field on `create_order`; LRU cache with configurable TTL (default 24 h) prevents duplicate orders on retry.
8. **Configurable Order State-Machine** — formal `TRANSITIONS` adjacency map + single `_assert_transition` guard; state machine is auditable from code alone.
9. **Bulk Operations with Concurrency Cap** — `_bounded_gather` semaphore limits concurrency to 10 (configurable); applied to all bulk mutation methods.
10. **Carrier Integration Adapter Interface** — `CarrierAdapter` protocol with `fetch_tracking_events`; `sync_shipment_tracking` polls shipped orders and updates status automatically.
11. **Tax & Duty Calculation Hook** — `TaxEngine` protocol invoked in `confirm_order`; swappable implementations (Avalara, TaxJar, flat-rate) via dependency injection.
12. **Customer Return & Reverse Logistics (RMA)** — full `create_rma` → `approve_rma` → `receive_return` lifecycle; links to origin order, captures condition codes, triggers WMS inventory adjustments.
13. **Delivery Window Negotiation** — `get_available_delivery_windows` reads warehouse calendars and ATP horizon to return feasible date/time windows for customer selection.
14. **SLA Breach Detection & Escalation** — `detect_sla_breaches` emits `sla_breach_detected` audit events and queues account-manager escalation notifications; integrates with APScheduler/Celery beat.
15. **Event-Sourced Audit Trail with Causality Chain** — every audit event carries `causation_id` (triggering event) and `correlation_id` (root workflow) for full causal trace reconstruction.

## New Methods

### `update_atp_horizon` — Build a rolling ATP profile

Projects supply and demand events forward to produce a date-bucketed ATP profile.
Use this before calling `check_atp_by_date` to enable capable-to-promise logic.

```python
svc = OrderManagementService(tenant_id="acme")

await svc.update_atp_horizon(
    sku="SKU-001",
    opening_stock=100.0,
    supply_events=[
        {"date": "2026-06-15", "quantity": 500},
        {"date": "2026-06-30", "quantity": 300},
    ],
    demand_events=[
        {"date": "2026-06-10", "quantity": 120},
        {"date": "2026-06-20", "quantity": 200},
    ],
    warehouse_id="WH-NBI",
)

result = await svc.check_atp_by_date(
    sku="SKU-001",
    requested_quantity=250,
    requested_date="2026-06-25",
    warehouse_id="WH-NBI",
)
# result["available"] → True/False, result["atp_at_date"] → float
```

### `detect_sla_breaches` — Scan for overdue orders and escalate

Finds all active orders past their promised delivery date. Pass `escalate=True`
to fire email notifications to the account manager in the same call.

```python
summary = await svc.detect_sla_breaches(
    tenant_id="acme",
    escalate=True,
    escalation_recipient="ops@acme.com",
)
# summary["total_breached"] → int
# summary["breached_orders"] → list of {order_id, order_number, promised_delivery_date, current_status}
```

### `get_order_queue` — Priority-scored warehouse pick list

Returns confirmed orders sorted by `revenue × priority_weight × customer_tier_weight`.
Feed directly to the warehouse pick system to maximise value delivery.

```python
await svc.set_customer_tier(customer_id="CUST-42", tier="strategic")

queue = await svc.get_order_queue(tenant_id="acme", status_filter="confirmed")
# queue[0] → highest-score order dict, includes "_score" field
# Typical use: pass queue[:20] to the warehouse picking API each shift
```
