# Agricultural Supply Chain — User Guide

## Overview

agr_sup enables end-to-end supply chain visibility from farm harvest through export clearance,
with immutable traceability events, cold chain monitoring, and procurement management.

## Key Use Cases

- **Traceability**: Register produce batches at harvest, track status changes (farm → collection
  → processing → storage → transport → export → delivered). Every status change is logged.
- **Input Procurement**: Manage supplier orders from request through delivery and invoicing.
  Track fulfilment ratios and on-time delivery rates per supplier.
- **Cold Chain Management**: Log temperature/humidity readings per batch per location.
  Automatic breach detection based on product-specific temperature limits.
- **Export Documentation**: Attach phytosanitary certificates, certificates of origin, and
  commercial invoices to batches. Export readiness check verifies all required docs are present.

## Example Workflows

### Create a Produce Batch
```
POST /api/agriculture/sup/batches
{
  "product_type": "flowers",
  "farm_parcel_id": "par-001",
  "farmer_id": "farmer-001",
  "harvest_date": "2025-04-08",
  "weight_kg": 500,
  "quality_grade": "A"
}
```

### Log Cold Chain Temperature
```
POST /api/agriculture/sup/cold-chain
{"batch_id": "bat-abc", "location": "Nairobi Cold Store", "temperature_c": 5.2, "humidity_pct": 85}
```

### Check Export Readiness
```
GET /api/agriculture/sup/batches/bat-abc/export-readiness
```
