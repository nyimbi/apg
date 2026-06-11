# Procurement Management User Guide

## Overview

`scm_prc` handles the full source-to-pay cycle.  The service is stateful and
tenant-isolated: every record carries a `tenant_id` and all queries are scoped
to the caller's tenant context.

Core domains:

| Domain | What it covers |
|--------|---------------|
| RFQ | Issue competitive requests, collect vendor responses, weighted scorecard, award |
| Purchase Orders | Full lifecycle from draft through receipt and invoice |
| Three-Way Match | Automated PO/receipt/invoice reconciliation with configurable tolerances |
| Vendor Evaluation | Periodic scored assessments across quality, delivery, price, service |
| Contract Management | Ceiling tracking, spend-down alerts, compliance check |
| Multi-Currency | Dated exchange rates, normalised spend analytics |
| SLA Monitoring | Configurable per-document-type thresholds, breach/warning scan |
| Audit | Append-only tamper-evident chain (SHA-256 linked events) |
| Analytics | Spend, delivery performance, process cycle times, KPI dashboard |

---

## RFQ Process

### 1. Create and issue an RFQ

```
POST /api/scm/prc/rfqs
{
  "tenant_id": "acme",
  "title": "Q3 Office Supplies",
  "lines": [
    {"sku": "PEN-BLK", "quantity": 500, "unit_of_measure": "EA"},
    {"sku": "PAPER-A4", "quantity": 200, "unit_of_measure": "BOX"}
  ],
  "vendor_ids": ["VEND-001", "VEND-002", "VEND-003"],
  "deadline": "2026-07-15"
}

POST /api/scm/prc/rfqs/{id}/issue
{"tenant_id": "acme", "issued_by": "john.doe"}
```

### 2. Record vendor responses

Each response can carry optional scoring fields used by the competitive scorecard:

```
POST /api/scm/prc/rfqs/{id}/responses
{
  "tenant_id": "acme",
  "vendor_id": "VEND-001",
  "quoted_lines": [...],
  "total_quoted_amount": 14250.00,
  "currency": "USD",
  "valid_until": "2026-07-30",
  "lead_time_days": 7,
  "quality_score": 8.5,
  "sustainability_score": 7.0
}
```

### 3. Weighted competitive scorecard

```
POST /api/scm/prc/rfqs/{id}/score
{
  "tenant_id": "acme",
  "weights": {"price": 0.40, "lead_time": 0.30, "quality": 0.20, "sustainability": 0.10}
}
```

Response returns `ranked` array with `composite_score` and `rank` per vendor,
plus a `recommended_vendor` shortcut.  Weights must sum to 1.0.

### 4. Award

```
POST /api/scm/prc/rfqs/{id}/award
{"tenant_id": "acme", "winning_vendor_id": "VEND-001", "awarded_by": "john.doe"}
```

---

## Purchase Order Lifecycle

```
POST /api/scm/prc/purchase-orders
{
  "tenant_id": "acme",
  "vendor_id": "VEND-001",
  "lines": [
    {"sku": "PEN-BLK", "quantity": 500, "unit_price": 0.28, "currency": "USD"}
  ],
  "payment_terms": "NET30",
  "rfq_id": "rfq-abc123"
}
```

Subsequent transitions: `send → acknowledge → receive`.

### Delivery schedule

Attach a structured schedule to enable on-time delivery tracking:

```
PUT /api/scm/prc/purchase-orders/{id}/delivery-schedule
{
  "tenant_id": "acme",
  "schedule": [
    {
      "sku": "PEN-BLK",
      "expected_date": "2026-07-20",
      "expected_quantity": 250.0,
      "actual_date": null,
      "actual_quantity": null
    },
    {
      "sku": "PEN-BLK",
      "expected_date": "2026-07-27",
      "expected_quantity": 250.0,
      "actual_date": null,
      "actual_quantity": null
    }
  ]
}
```

Populate `actual_date` and `actual_quantity` when milestones are met.
`GET /api/scm/prc/analytics/delivery-performance` then computes on-time rates
per vendor.

---

## Three-Way Match

```
POST /api/scm/prc/three-way-matches
{
  "tenant_id": "acme",
  "po_id": "po-abc",
  "receipt_id": "rcpt-xyz",
  "invoice_number": "INV-2026-999",
  "invoiced_amount": 15000.00
}
```

Tolerance bands:

| Variance | Result | Action |
|----------|--------|--------|
| ≤ 1% of PO value | `matched` | Auto-approved, PO → `invoiced` |
| 1%–5% | `partial` | Held for manual review |
| > 5% | `disputed` | Held for resolution, SLA clock starts |

Resolve a disputed match:

```
POST /api/scm/prc/three-way-matches/{id}/resolve
{"tenant_id": "acme", "resolution": "approved", "resolved_by": "ap.manager", "notes": "Vendor credit note accepted"}
```

---

## Contract Management

### Create a contract

```
POST /api/scm/prc/contracts
{
  "tenant_id": "acme",
  "vendor_id": "VEND-001",
  "contract_reference": "MSA-2026-001",
  "start_date": "2026-01-01",
  "end_date": "2026-12-31",
  "value": 500000.00,
  "currency": "USD"
}
```

### Spend-down tracking

```
GET /api/scm/prc/contracts/{id}/spend-status?tenant_id=acme
```

Response:
```json
{
  "contract_id": "...",
  "ceiling": 500000.0,
  "consumed": 412000.0,
  "remaining": 88000.0,
  "utilisation_pct": 82.4,
  "alert_level": "80pct"
}
```

Alert events `contract_nearing_limit_80pct` and `contract_nearing_limit_95pct`
are emitted automatically when thresholds are crossed.

---

## Multi-Currency Spend

Store exchange rates (forward and reverse derived automatically):

```
POST /api/scm/prc/exchange-rates
{"tenant_id": "acme", "from_currency": "EUR", "to_currency": "USD", "rate": 1.082}
```

Normalised spend report:

```
GET /api/scm/prc/analytics/spend/normalised?tenant_id=acme&reporting_currency=USD
```

POs whose currency has no stored rate appear in the `unconverted_pos` list so
nothing is silently dropped.

---

## SLA Monitoring

Configure thresholds (hours):

```
POST /api/scm/prc/sla/configure
{
  "tenant_id": "acme",
  "sla_config": {
    "po_acknowledgement": 24,
    "disputed_invoice_resolution": 72,
    "rfq_response": 120,
    "goods_receipt": 48
  }
}
```

Scan for breaches:

```
GET /api/scm/prc/sla/breaches?tenant_id=acme
```

Returns `breached` (elapsed > SLA) and `warnings` (>75% elapsed) per document.

---

## Audit Chain Verification

Every service event is SHA-256 chained.  Verify integrity at any time:

```
GET /api/scm/prc/audit-events/verify-chain?tenant_id=acme
```

Response:
```json
{
  "valid": true,
  "event_count": 142,
  "broken_at_index": null
}
```

`valid: false` with a `broken_at_index` indicates a tampered or deleted event.

---

## Analytics

| Endpoint | Description |
|----------|-------------|
| `/analytics/spend` | Spend by vendor in transaction currencies |
| `/analytics/spend/normalised` | Spend converted to reporting currency |
| `/analytics/dashboard` | KPI dashboard: open RFQs/POs, match rate, disputed invoices |
| `/analytics/delivery-performance` | On-time delivery rate per vendor |
| `/analytics/cycle-times` | Mean/min/max hours: RFQ→award, PO draft→sent, sent→ack, sent→received |

---

## Status Flows

```
RFQ:  draft → issued → responses_received → awarded | cancelled

PO:   draft → sent → acknowledged → partially_received
                                  → received → invoiced → closed | cancelled

3WM:  pending → approved | rejected
```
