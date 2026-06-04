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
