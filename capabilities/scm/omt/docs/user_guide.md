# Order Management & Tracking User Guide

## Overview

`scm_omt` manages the complete order lifecycle from creation through delivery, including available-to-promise (ATP) checks, backorder handling, split shipments, order promising, and automated customer notifications.

## Key Use Cases

- **Order lifecycle**: Create, confirm, allocate, pick, pack, ship, and deliver orders.
- **ATP checks**: Query real-time available inventory before promising to customers.
- **Backorder management**: Track and fulfil shortfalls when stock is unavailable.
- **Split shipments**: Divide an order into multiple partial deliveries.
- **Order promising**: Commit delivery dates with confidence scores.
- **Customer notifications**: Push order status updates via email, SMS, push, or webhook.

## API Reference

### Create Order

```
POST /api/scm/omt/orders
{
  "tenant_id": "acme",
  "customer_id": "CUST-001",
  "lines": [
    {"sku": "PROD-A", "quantity": 5, "unit_price": 99.99}
  ],
  "shipping_address": {"city": "Nairobi"},
  "priority": "high"
}
```

### Check ATP

```
GET /api/scm/omt/atp?tenant_id=acme&sku=PROD-A&quantity=5
```

### Create Backorder

```
POST /api/scm/omt/backorders
{
  "tenant_id": "acme",
  "order_id": "ord-xyz",
  "sku": "PROD-A",
  "backordered_quantity": 2,
  "reason": "Insufficient stock",
  "expected_fulfilment_date": "2026-07-01"
}
```

## Order Status Flow

draft → confirmed → allocated → picking → packed → shipped → delivered

Side transitions: any non-terminal → on_hold → confirmed | cancelled
