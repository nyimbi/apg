# APG Point of Sale — API Reference
© 2025 Datacraft | www.datacraft.co.ke

Base URL: `/retail-pos/api/v1`
Auth header: `X-Tenant-ID: <tenant_id>`
Content-Type: `application/json`

## Error Format
```json
{"error": "description of error", "status": 400}
```

---

## Terminals

| Method | Path | Description |
|--------|------|-------------|
| GET | `/terminals` | List terminals. Query: `?store_id=` |
| POST | `/terminals` | Register terminal |
| GET | `/terminals/<id>` | Get terminal |
| PUT | `/terminals/<id>` | Update terminal |
| DELETE | `/terminals/<id>` | Soft-delete terminal |
| POST | `/terminals/<id>/heartbeat` | Mark online |
| POST | `/terminals/<id>/offline` | Mark offline |

### Register Terminal
```json
POST /terminals
{
  "store_id": "store-nbi-01",
  "terminal_code": "T001",
  "terminal_type": "fixed_counter",
  "floor_limit": 10000.0,
  "default_currency": "KES"
}
```
Returns: terminal object, `201`

---

## Sessions

| Method | Path | Description |
|--------|------|-------------|
| GET | `/sessions` | List sessions. Query: `?store_id=&status=` |
| POST | `/sessions` | Open session |
| GET | `/sessions/<id>` | Get session |
| GET | `/sessions/<id>/summary` | Full totals summary |
| POST | `/sessions/<id>/close` | Close with cash count |
| POST | `/sessions/<id>/suspend` | Suspend |
| POST | `/sessions/<id>/resume` | Resume |
| POST | `/sessions/<id>/reconcile` | Submit physical cash count |

### Open Session
```json
POST /sessions
{
  "terminal_id": "<id>",
  "cashier_id": "alice",
  "opening_float": 1000.00,
  "store_id": "store-nbi-01"
}
```

### Close Session
```json
POST /sessions/<id>/close
{ "closing_cash": 1050.00, "notes": "Good shift" }
```

### Cash Count Reconciliation
```json
POST /sessions/<id>/reconcile
{
  "counted_cash": 1050.00,
  "denominations": {"1000": 1, "50": 1},
  "counted_by": "alice"
}
```
Returns: `{expected_cash, counted_cash, variance, denominations}`

---

## Transactions

| Method | Path | Description |
|--------|------|-------------|
| GET | `/transactions` | List. Query: `?session_id=&page=&page_size=` |
| POST | `/transactions` | Begin transaction basket |
| GET | `/transactions/<id>` | Get transaction |
| PUT | `/transactions/<id>` | Update notes/customer |
| DELETE | `/transactions/<id>` | Void (supervisor required) |
| POST | `/transactions/<id>/items` | Add item |
| DELETE | `/transactions/<id>/items/<sku>` | Remove item |
| POST | `/transactions/<id>/discount` | Apply discount |
| POST | `/transactions/<id>/pay` | Record split payment |
| POST | `/transactions/<id>/complete` | Complete transaction |
| POST | `/transactions/<id>/park` | Park (suspend) |
| POST | `/transactions/<id>/retrieve` | Retrieve parked |
| POST | `/transactions/<id>/void` | Void (alternate) |
| POST | `/transactions/<id>/payments/cash` | Cash payment |
| POST | `/transactions/<id>/payments/card` | Card payment |
| POST | `/transactions/<id>/payments/mpesa` | M-Pesa payment |

### Add Item
```json
POST /transactions/<id>/items
{
  "sku": "MILK-1L",
  "quantity": 2,
  "description": "Whole Milk 1L",
  "price_override": null
}
```

### Apply Discount
```json
POST /transactions/<id>/discount
{
  "discount_type": "percentage",
  "value": 10.0,
  "approved_by": "supervisor-bob"
}
```
discount_type options: `percentage`, `fixed_amount`, `coupon_code`, `loyalty_points`

### Split Payment
```json
POST /transactions/<id>/pay
{
  "payments": [
    {"method": "cash", "amount": 200.00},
    {"method": "mobile_money", "amount": 100.00, "reference": "QXZ123"}
  ]
}
```

---

## Process Sale (Convenience)

```json
POST /process-sale
{
  "session_id": "<id>",
  "cashier_id": "alice",
  "items": [
    {"sku": "MILK-1L", "qty": 2},
    {"sku": "BREAD", "qty": 1, "price": 80.0}
  ],
  "payments": [{"method": "cash", "amount": 500.0}],
  "customer_id": "cust-001",
  "discount": {"type": "coupon_code", "coupon_code": "SAVE10"}
}
```
Returns: completed transaction object, `201`

---

## Refunds

| Method | Path | Description |
|--------|------|-------------|
| GET | `/refunds` | List. Query: `?session_id=` |
| POST | `/refunds` | Process refund |
| GET | `/refunds/<id>` | Get refund |

```json
POST /refunds
{
  "original_transaction_id": "<txn_id>",
  "session_id": "<session_id>",
  "terminal_id": "<terminal_id>",
  "items": [
    {"sku": "MILK-1L", "quantity": 1, "unit_price": 120.0, "line_total": 120.0, "description": "Milk"}
  ],
  "reason": "defective",
  "manager_auth_id": "supervisor-bob"
}
```

---

## Discounts

| Method | Path | Description |
|--------|------|-------------|
| GET | `/discounts` | List. Query: `?active=true` |
| POST | `/discounts` | Create discount |
| GET | `/discounts/<id>` | Get discount |
| PUT | `/discounts/<id>` | Update discount |
| DELETE | `/discounts/<id>` | Deactivate discount |

---

## Cash Management

| Method | Path | Description |
|--------|------|-------------|
| GET | `/cash` | List events. Query: `?session_id=` |
| POST | `/cash` | Record cash event |

```json
POST /cash
{
  "session_id": "<id>",
  "terminal_id": "<id>",
  "store_id": "store-nbi-01",
  "cashier_id": "alice",
  "event_type": "safe_drop",
  "amount": 500.00,
  "authorised_by": "manager-bob",
  "denominations": {"500": 1}
}
```
event_type options: `opening_float`, `safe_drop`, `safe_pickup`, `petty_cash_out`, `petty_cash_in`, `till_loan`, `correction`

---

## Loyalty

| Method | Path | Description |
|--------|------|-------------|
| GET | `/loyalty/<customer_id>` | Balance + redemption value |
| GET | `/loyalty/<customer_id>/history` | Transaction history |
| POST | `/loyalty/earn-redeem` | Earn/redeem points |

---

## Receipts

| Method | Path | Description |
|--------|------|-------------|
| GET | `/receipts` | List. Query: `?transaction_id=` |
| POST | `/receipts` | Generate receipt |
| GET | `/receipts/<id>` | Get receipt |

```json
POST /receipts
{
  "transaction_id": "<id>",
  "format": "email",
  "recipient_email": "customer@example.com"
}
```
format options: `thermal`, `email`, `sms`, `digital`, `both`

---

## Price & Stock

| Method | Path | Description |
|--------|------|-------------|
| GET | `/price-check` | `?sku=MILK-1L&tier=vip` |
| GET | `/stock-check` | `?sku=MILK-1L&store_id=store-nbi-01` |
| GET | `/inventory/movements` | `?store_id=` |
| GET | `/inventory/low-stock` | `?store_id=&threshold_days=7` |

---

## Supervisor Overrides

| Method | Path | Description |
|--------|------|-------------|
| POST | `/overrides` | Create override |
| GET | `/overrides` | List. `?session_id=` |

override_type: `price_override`, `discount_override`, `void`, `refund`, `close_session`

---

## Offline Sync

```json
POST /offline/sync
{
  "terminal_id": "<id>",
  "session_id": "<id>",
  "transactions": [...],
  "sync_sequence": 1
}
```
Returns: `{accepted: [...], rejected: [...], duplicate_skipped: [...], sync_completed_at}`

---

## End of Day

| Method | Path | Description |
|--------|------|-------------|
| POST | `/eod` | Generate EOD report |
| GET | `/eod` | List reports. `?store_id=` |
| GET | `/eod/<id>` | Get report |
| POST | `/eod/<id>/approve` | Approve report |
| GET | `/reports/eod/<store_id>/<YYYY-MM-DD>` | Get by date |

---

## Reports

| Method | Path | Description |
|--------|------|-------------|
| GET | `/reports/sales-summary` | `?period=today\|week\|month` |
| GET | `/reports/till-variance` | `?period=today` |
| GET | `/reports/hourly` | `?store_id=&date=YYYY-MM-DD` |
| GET | `/reports/top-skus` | `?store_id=&top_n=10` |

---

## Capability Contract

| Method | Path | Description |
|--------|------|-------------|
| GET | `/contract` | Full capability contract |
| POST | `/rules/evaluate` | Evaluate rules against context |
