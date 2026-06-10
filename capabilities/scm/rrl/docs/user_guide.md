# Returns & Reverse Logistics User Guide

## Overview

`scm_rrl` manages the complete reverse supply chain: customer return authorisations, quality inspection, refurbishment work orders, disposal of unsalvageable goods, credit note issuance, and reverse shipment tracking.

## Key Use Cases

- **RMA lifecycle**: Customers request returns; agents approve/reject and track through receipt and resolution.
- **Quality inspection**: Inspect returned items, grade condition, route to refurbishment or disposal.
- **Refurbishment**: Create work orders for repair/refurbishment and track actual vs estimated cost.
- **Disposal management**: Authorise and record disposal of scrap items with method and certificate.
- **Credit notes**: Issue financial credits to customers after return resolution.
- **Reverse shipments**: Book and track inbound carrier collections from customer premises.

## API Reference

### Create RMA

```
POST /api/scm/rrl/rmas
{
  "tenant_id": "acme",
  "order_id": "ORD-001",
  "customer_id": "CUST-001",
  "items": [{"sku": "PROD-A", "quantity": 1}],
  "reason_code": "defective",
  "requested_resolution": "replacement"
}
```

### Issue Credit Note

```
POST /api/scm/rrl/credit-notes
{
  "tenant_id": "acme",
  "rma_id": "rma-xyz",
  "customer_id": "CUST-001",
  "amount": 299.99,
  "reason": "Product defect confirmed",
  "issued_by": "finance.team"
}
```

## RMA Status Flow

pending → approved → received → processing → resolved | closed

Side exit: pending | approved → rejected
