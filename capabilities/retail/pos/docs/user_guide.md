# APG Point of Sale — User Guide
© 2025 Datacraft | www.datacraft.co.ke

## Overview

APG POS is a multi-tenant, offline-capable point of sale system. It runs standalone or as a component within the APG platform ecosystem, integrating with inventory, loyalty, and reporting capabilities.

---

## Getting Started

### Opening a Session

1. Register your terminal (admin, once per device):
   ```
   POST /retail-pos/api/v1/terminals
   { "store_id": "store-nbi-01", "terminal_code": "T001", "terminal_type": "fixed_counter" }
   ```

2. Open a cashier session with your opening float:
   ```
   POST /retail-pos/api/v1/sessions
   { "terminal_id": "<id>", "cashier_id": "alice", "opening_float": 1000.00 }
   ```

3. The session is now open. All transactions are recorded against it.

---

## Processing a Sale

### Quick Sale (all-in-one)
```
POST /retail-pos/api/v1/process-sale
{
  "session_id": "<session_id>",
  "cashier_id": "alice",
  "items": [
    {"sku": "MILK-1L", "qty": 2, "description": "Whole Milk 1L"},
    {"sku": "BREAD-WHT", "qty": 1}
  ],
  "payments": [
    {"method": "cash", "amount": 500.00}
  ]
}
```

### Step-by-Step Sale
1. Begin transaction: `POST /transactions`
2. Add items: `POST /transactions/<id>/items`
3. Apply discount (optional): `POST /transactions/<id>/discount`
4. Record payment: `POST /transactions/<id>/pay`
5. Complete: `POST /transactions/<id>/complete`

### Payment Methods
| Method | Key | Notes |
|--------|-----|-------|
| Cash | `cash` | Change computed automatically |
| Card (debit) | `card_debit` | Requires auth_code |
| Card (credit) | `card_credit` | Requires auth_code |
| M-Pesa | `mobile_money` | Requires M-Pesa reference |
| Loyalty Points | `loyalty_points` | Requires customer_id on transaction |
| Gift Card | `gift_card` | Requires gift_card_number |

### Split Payment
Pay across multiple methods in one call:
```json
{
  "payments": [
    {"method": "cash", "amount": 200.00},
    {"method": "mobile_money", "amount": 300.00, "reference": "QXZ123456"}
  ]
}
```

---

## Discounts

| Type | Key | Value |
|------|-----|-------|
| Percentage | `percentage` | 0–100 |
| Fixed amount | `fixed_amount` | KES amount |
| Coupon code | `coupon_code` | `coupon_code` field |
| Loyalty points | `loyalty_points` | Points to redeem |

Manager/staff discounts require `approved_by` (supervisor ID).

---

## Refunds

```
POST /retail-pos/api/v1/refunds
{
  "original_transaction_id": "<txn_id>",
  "session_id": "<session_id>",
  "terminal_id": "<terminal_id>",
  "items": [{"sku": "MILK-1L", "quantity": 1, "unit_price": 120.0, "line_total": 120.0}],
  "reason": "defective"
}
```

Refund reasons: `defective`, `wrong_item`, `customer_change_mind`, `overcharge`, `duplicate`, `not_as_described`, `other`

---

## Voiding a Transaction

Requires supervisor authorisation:
```
POST /retail-pos/api/v1/transactions/<id>/void
{ "reason": "operator error", "supervisor_id": "<supervisor_id>" }
```

Voiding a completed transaction automatically reverses inventory deductions and session totals.

---

## Loyalty Points

- Points are earned automatically on completed sales (1 point per KES 1 by default).
- Check balance: `GET /loyalty/<customer_id>`
- Redeem at checkout: include `{"method": "loyalty_points", "amount": <value>}` in payments.
- Maximum redemption: 50% of transaction total.

---

## Receipts

Generate after completing a transaction:
```
POST /retail-pos/api/v1/receipts
{
  "transaction_id": "<txn_id>",
  "format": "thermal",          // thermal | email | sms | digital
  "recipient_email": "customer@example.com"
}
```

Thermal receipts contain ESC/POS-style formatted text for direct printing.
Email receipts contain HTML with item breakdown and grand total.

---

## Parked Transactions

Park a transaction to serve another customer:
- Park: `POST /transactions/<id>/park`
- Retrieve: `POST /transactions/<id>/retrieve`

---

## Cash Management

| Event | Key |
|-------|-----|
| Opening float | `opening_float` (set at session open) |
| Safe drop | `safe_drop` (requires `authorised_by`) |
| Petty cash out | `petty_cash_out` |
| Till loan | `till_loan` |

End-of-shift cash count:
```
POST /retail-pos/api/v1/sessions/<session_id>/reconcile
{
  "counted_cash": 1250.00,
  "denominations": {"1000": 1, "200": 1, "50": 1},
  "counted_by": "alice"
}
```

---

## Closing a Session

```
POST /retail-pos/api/v1/sessions/<session_id>/close
{ "closing_cash": 1250.00, "notes": "Smooth shift" }
```

Variance = counted_cash − (opening_float + cash_sales).

---

## Offline Mode

When connectivity is lost:
1. Transactions are created with `offline_mode: true`.
2. On reconnect, submit the batch:
   ```
   POST /retail-pos/api/v1/offline/sync
   {
     "terminal_id": "<terminal_id>",
     "session_id": "<session_id>",
     "transactions": [...],
     "sync_sequence": 1
   }
   ```
3. The server validates, deduplicates, and returns `accepted` / `rejected` lists.

`sync_sequence` must be monotonically increasing per terminal to detect gaps.

---

## End of Day

```
POST /retail-pos/api/v1/eod
{ "store_id": "store-nbi-01", "business_date": "2026-06-04", "generated_by": "manager-bob" }
```

The EOD report includes: gross sales, refunds, discounts, tax, payment method breakdown, hourly profile, and top SKUs. EOD is idempotent-guarded: running it twice for the same date raises an error.

Approve: `POST /eod/<report_id>/approve`

---

## Reports

| Report | Endpoint |
|--------|----------|
| Sales summary | `GET /reports/sales-summary?period=today` |
| Till variance | `GET /reports/till-variance?period=today` |
| Hourly breakdown | `GET /reports/hourly?store_id=<id>` |
| Top SKUs | `GET /reports/top-skus?top_n=10` |
| EOD by date | `GET /reports/eod/<store_id>/<YYYY-MM-DD>` |

---

## Tax Exemption

For tax-exempt customers (NGOs, diplomats):
- Set `tax_exempt: true` on the transaction.
- Provide `tax_exempt_ref` (certificate number) — required.
- All line-item VAT is zeroed out.

---

## Supervisor Overrides

Overrides are required for: price overrides, manager discounts, voids, refunds above limit.

Create an override:
```
POST /retail-pos/api/v1/overrides
{
  "session_id": "<id>",
  "terminal_id": "<id>",
  "supervisor_id": "manager-bob",
  "override_type": "price_override",
  "target_id": "<transaction_id>",
  "notes": "Customer requested price match"
}
```

Self-approval (cashier == supervisor) is blocked by the system.

---

## Basket Suggestions (Co-Purchase Intelligence)

For loyalty customers, the system can suggest items the customer almost always buys alongside what is already in the basket:

```
GET /retail-pos/api/v1/transactions/<id>/basket-suggestions?customer_id=<id>&top_n=3
```

Returns up to `top_n` SKUs ranked by co-purchase frequency from the customer's loyalty history. No external ML service required.

---

## Fraud Risk Scoring

Every completed transaction can be scored for fraud risk:

```
GET /retail-pos/api/v1/transactions/<id>/fraud-score
```

Response:
```json
{
  "transaction_id": "...",
  "fraud_risk_score": 45,
  "risk_level": "medium",
  "signals": ["high_discount_rate_23pct"],
  "requires_review": false
}
```

Transactions with `risk_level: high` (score ≥ 60) are flagged in the supervisor queue.

---

## Live Store Dashboard

Real-time trading snapshot for store managers:

```
GET /retail-pos/api/v1/stores/<store_id>/live
```

Returns active sessions, open baskets, rolling transactions-per-minute (last 5 min), hour revenue, and payment method mix. Designed for SSE push every 15–30 seconds.

---

## Session Performance Metrics

Cashier throughput ranking for all open sessions:

```
GET /retail-pos/api/v1/stores/<store_id>/session-metrics
```

Returns per-cashier: transactions/hour, average basket value, void rate %, discount rate %, and an `alert` flag when thresholds are exceeded (void rate > 5% or discount rate > 15%).

---

## Predictive Cash Management

Check how long the current till cash will last at current velocity:

```
GET /retail-pos/api/v1/sessions/<id>/cash-runway?horizon_minutes=30
```

Response:
```json
{
  "current_cash": 4800.00,
  "cash_velocity_per_hour": 1200.00,
  "predicted_shortage_in_minutes": 22.0,
  "alert": true,
  "recommended_action": "request_safe_drop"
}
```

An alert fires when projected shortage is within `horizon_minutes` (default 30).

---

## Inventory Soft-Reserve

When `add_item` is called the system can soft-reserve stock to prevent two cashiers from selling the same last unit:

```
POST /retail-pos/api/v1/inventory/reserve
{
  "transaction_id": "<id>",
  "sku": "MILK-1L",
  "quantity": 2,
  "store_id": "store-nbi-01"
}
```

The hold expires automatically after 15 minutes. To release manually:
```
DELETE /retail-pos/api/v1/inventory/holds/<transaction_id>/<sku>
```

---

## Shift Handover

Enforce dual cash counts between outgoing and incoming cashiers:

1. Initiate handover (locks the session):
   ```
   POST /retail-pos/api/v1/sessions/<outgoing_session_id>/handover
   { "incoming_cashier_id": "bob" }
   ```

2. Both cashiers submit their count independently:
   ```
   POST /retail-pos/api/v1/handovers/<handover_id>/count
   { "cashier_id": "alice", "counted_cash": 1350.00 }

   POST /retail-pos/api/v1/handovers/<handover_id>/count
   { "cashier_id": "bob", "counted_cash": 1345.00 }
   ```

3. When both counts are received:
   - Variance ≤ KES 10: status = `completed` → terminal released for new session
   - Variance > KES 10: status = `disputed` → supervisor review required

---

## Customer Purchase History

View a customer's full purchase history and spending analytics:

```
GET /retail-pos/api/v1/customers/<customer_id>/history?period=2026-06&limit=50
```

Response includes: total spend, average basket, loyalty balance, top SKUs by purchase frequency, payment method preferences, and last visit date.
