# Order Management & Tracking — User Guide

## Overview

`scm_omt` manages the complete B2B/B2C order lifecycle from quote through delivery and returns.
It provides:

- Order CRUD with a formal, validated state machine
- Available-to-Promise (ATP) checks — point-in-time and date-bucketed horizon
- Backorder registration and fulfilment
- Split-shipment creation and tracking
- Delivery promise commitment and automated re-promising
- Priority-scored order queue for warehouse sequencing
- Rule-based order routing to warehouses
- Delivery window negotiation with warehouse calendar awareness
- SLA breach detection with escalation notifications
- Return Merchandise Authorization (RMA) — create, approve, receive
- Customer notifications via email, SMS, push, or webhook
- Bounded-concurrency bulk operations
- Event-sourced audit trail with causation/correlation chain

---

## Order Lifecycle

### State Machine

```
draft → confirmed → allocated → picking → packed ─┬─→ shipped → delivered
  │                                                └─→ partially_shipped ─┐
  │                                                                        └─→ shipped → delivered
  └──── on_hold (from any non-terminal) ─┬─→ confirmed
                                          └─→ cancelled
```

Transitions are enforced by `_assert_transition`. Any call that would jump an order to a
non-adjacent state raises `ValueError`.

### Create Order

```python
order = await svc.create_order(
    customer_id="CUST-001",
    lines=[{"sku": "PROD-A", "quantity": 10, "unit_price": 49.99}],
    shipping_address={"city": "Nairobi", "country": "KE"},
    priority="high",
    idempotency_key="checkout-session-abc123",  # prevents duplicate on retry
    tenant_id="acme",
)
```

### Confirm / Cancel / Hold

```python
await svc.confirm_order(order["id"], confirmed_by="ops@acme.com", tenant_id="acme")
await svc.cancel_order(order["id"], reason="Customer request", cancelled_by="ops@acme.com")
await svc.place_order_on_hold(order["id"], reason="Fraud review", held_by="fraud-bot")
await svc.release_order_hold(order["id"], released_by="ops@acme.com")
```

---

## ATP Checks

### Point-in-time

```python
result = await svc.check_atp(
    sku="PROD-A", requested_quantity=10, warehouse_id="WH-01", tenant_id="acme"
)
# result["can_fulfil"] → True/False, result["shortage_quantity"]
```

### Date-bucketed horizon

```python
# Load supply and demand events to build a rolling ATP profile
await svc.update_atp_horizon(
    sku="PROD-A",
    opening_stock=50.0,
    supply_events=[
        {"date": "2026-06-15", "quantity": 100},
        {"date": "2026-07-01", "quantity": 200},
    ],
    demand_events=[
        {"date": "2026-06-10", "quantity": 30},
        {"date": "2026-06-20", "quantity": 80},
    ],
    warehouse_id="WH-01",
    tenant_id="acme",
)

# Ask whether stock will be available by a specific date
result = await svc.check_atp_by_date(
    sku="PROD-A", requested_quantity=40,
    requested_date="2026-06-20", tenant_id="acme"
)
# result["can_fulfil"], result["atp_at_date"]
```

---

## Backorders

```python
# Register a backorder when stock is short
bo = await svc.create_backorder(
    order_id=order["id"],
    sku="PROD-A",
    backordered_quantity=5,
    reason="Insufficient stock",
    expected_fulfilment_date="2026-07-01",
    tenant_id="acme",
)

# Mark fulfilled when stock arrives
await svc.fulfil_backorder(bo["id"], fulfilled_by="warehouse@acme.com")
```

---

## Split Shipments

```python
split = await svc.create_split_shipment(
    order_id=order["id"],
    split_lines=[
        {"sku": "PROD-A", "quantity": 5, "warehouse_id": "WH-01"},
        {"sku": "PROD-A", "quantity": 5, "warehouse_id": "WH-02"},
    ],
    reason="Stock split across warehouses",
    tenant_id="acme",
)
```

---

## Order Promising & Re-promising

```python
# Commit a delivery date
promise = await svc.promise_order(
    order_id=order["id"],
    promised_date="2026-06-30",
    promised_by="planner@acme.com",
    confidence_pct=95.0,
    tenant_id="acme",
)

# If supply is disrupted, run the re-promising sweep
result = await svc.re_promise_breached_orders(
    tenant_id="acme",
    auto_revoke=True,
    new_promise_offset_days=5,
)
# result["revoked_count"], result["repromised_count"]
```

---

## Priority Order Queue & Routing

```python
# Assign SLA tier to a customer
await svc.set_customer_tier("CUST-001", tier="strategic", tenant_id="acme")

# Get the scored queue for the warehouse to work from top-down
queue = await svc.get_order_queue(tenant_id="acme", status_filter="confirmed")
# queue[0] has the highest composite score (revenue × priority_weight × tier_weight)

# Route lines to warehouses
plan = await svc.route_order(
    order_id=order["id"],
    warehouse_atp_snapshots=[
        {"warehouse_id": "WH-01", "sku": "PROD-A", "available_quantity": 8},
        {"warehouse_id": "WH-02", "sku": "PROD-A", "available_quantity": 15},
    ],
    policy="consolidate",  # or "fastest"
    tenant_id="acme",
)
# plan["assignments"] → [{line_index, sku, warehouse_id, quantity}, ...]
# plan["unroutable_lines"] → lines with no satisfying warehouse
```

---

## Delivery Window Negotiation

```python
windows = await svc.get_available_delivery_windows(
    order_id=order["id"],
    candidate_dates=["2026-06-18", "2026-06-20", "2026-06-25"],
    warehouse_calendar={
        "blackout_dates": ["2026-06-20"],  # public holiday
        "cutoff_time": "14:00",
    },
    tenant_id="acme",
)
# windows["feasible_windows"]   → [{"date": "2026-06-18", ...}, ...]
# windows["infeasible_windows"] → [{"date": "2026-06-20", "reason": "warehouse_blackout"}]
```

---

## SLA Breach Detection

```python
report = await svc.detect_sla_breaches(
    tenant_id="acme",
    escalate=True,
    escalation_recipient="ops-manager@acme.com",
)
# report["total_breached"], report["breached_orders"]
```

---

## RMA / Reverse Logistics

```python
# Customer requests a return after delivery
rma = await svc.create_rma(
    order_id=order["id"],
    lines=[{"sku": "PROD-A", "return_quantity": 2, "condition": "damaged"}],
    reason="Item arrived damaged",
    requested_by="customer@example.com",
    tenant_id="acme",
)

await svc.approve_rma(rma["id"], approved_by="ops@acme.com")
await svc.receive_return(
    rma["id"],
    received_by="warehouse@acme.com",
    condition_notes="Outer packaging crushed; inner item intact",
)
```

---

## Customer Notifications

```python
await svc.send_notification(
    order_id=order["id"],
    channel="email",          # email | sms | push | webhook
    event_type="order_shipped",
    message="Your order ORD-ACME-001001 has shipped!",
    recipient="customer@example.com",
    tenant_id="acme",
)
```

---

## Bulk Operations

`bulk_confirm_orders` applies a bounded semaphore (default concurrency = 10) so large batches
do not saturate DB connections or downstream rate limits:

```python
result = await svc.bulk_confirm_orders(
    order_ids=["ord-aaa", "ord-bbb", "ord-ccc"],
    confirmed_by="ops@acme.com",
    tenant_id="acme",
    concurrency=5,  # override default
)
# result["confirmed"], result["failed"], result["errors"]
```

---

## Audit Events

Every mutation emits an audit event carrying `causation_id` (the triggering event id) and
`correlation_id` (the root workflow id) for full causal trace reconstruction:

```python
events = await svc.get_audit_events(tenant_id="acme")
# Each event: {id, event_type, record_id, causation_id, correlation_id, emitted_at, ...}
```

---

## Order Analytics

```python
metrics = await svc.order_analytics(tenant_id="acme")
# {total_orders, by_status, total_order_value, open_backorders, active_promises, ...}

rate = await svc.fulfilment_rate(tenant_id="acme")
# {total_orders, delivered, fulfilment_rate_pct, open_backorders}
```
