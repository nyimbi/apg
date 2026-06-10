# Procurement Management User Guide

## Overview

`scm_prc` handles the full source-to-pay cycle: RFQ issuance and vendor responses, purchase order lifecycle, goods receipt, three-way invoice matching, periodic vendor evaluations, contract management, and spend analytics.

## Key Use Cases

- **RFQ process**: Issue requests for quotation to multiple vendors, collect responses, and award to the best-value vendor.
- **Purchase orders**: Create, send, acknowledge, and close POs with full line-item detail.
- **Goods receipt**: Record partial or full receipt against a PO, updating line-level quantities.
- **Three-way match**: Automatically compare PO value, received quantity, and vendor invoice — flag disputes when variance exceeds 1% tolerance.
- **Vendor evaluation**: Score vendors quarterly across quality, delivery, price, and service dimensions.
- **Contract compliance**: Verify every PO is backed by an active vendor contract.
- **Spend analytics**: Aggregate spend by vendor, identify top suppliers, track match rates.

## API Reference

### Create and Issue RFQ

```
POST /api/scm/prc/rfqs
{
  "tenant_id": "acme",
  "title": "Q3 Office Supplies",
  "lines": [{"sku": "PEN-BLK", "quantity": 500, "unit_of_measure": "EA"}],
  "vendor_ids": ["VEND-001", "VEND-002"],
  "deadline": "2026-07-15"
}

POST /api/scm/prc/rfqs/{id}/issue
{"tenant_id": "acme", "issued_by": "john.doe"}
```

### Three-Way Match

```
POST /api/scm/prc/three-way-matches
{
  "tenant_id": "acme",
  "po_id": "po-abc",
  "receipt_id": "rcpt-xyz",
  "invoice_number": "INV-2024-999",
  "invoiced_amount": 15000.00
}
```

## Status Flows

- **RFQ**: draft → issued → responses_received → awarded | cancelled
- **PO**: draft → sent → acknowledged → partially_received | received → invoiced → closed | cancelled
- **Three-way match**: pending → approved | rejected
