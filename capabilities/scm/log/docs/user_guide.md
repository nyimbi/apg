# Logistics & Transportation User Guide

## Overview

The `scm_log` capability provides end-to-end logistics management including carrier integration, multi-modal shipment tracking, freight invoice auditing, route optimisation, customs documentation generation, and third-party logistics (3PL) provider management.

## Key Use Cases

- **Carrier onboarding**: Register air, sea, road, rail, and courier carriers with service details.
- **Shipment lifecycle**: Create, book, track, and complete shipments across freight modes.
- **Freight audit**: Compare carrier invoices against expected charges and resolve discrepancies.
- **Route optimisation**: Define and optimise routes by cost, time, or CO2 footprint.
- **Customs documentation**: Auto-generate commercial invoices, packing lists, bills of lading, and customs declarations.
- **3PL management**: Register and assign third-party logistics providers for outsourced fulfilment.
- **Delivery exceptions**: Raise and resolve damage, delay, or loss events.

## API Reference

### Carriers

```
POST /api/scm/log/carriers
{
  "tenant_id": "acme",
  "name": "DHL Express",
  "carrier_code": "DHLX",
  "carrier_type": "air",
  "country_of_origin": "DE",
  "services_offered": ["express", "economy"]
}
```

### Shipments

```
POST /api/scm/log/shipments
{
  "tenant_id": "acme",
  "carrier_id": "carrier-abc123",
  "origin_address": {"city": "Nairobi", "country": "KE"},
  "destination_address": {"city": "London", "country": "GB"},
  "weight_kg": 25.5,
  "freight_mode": "air",
  "service_level": "express"
}
```

### Tracking

```
POST /api/scm/log/shipments/{id}/tracking
{
  "tenant_id": "acme",
  "event_type": "in_transit",
  "location": "Dubai Hub",
  "description": "Departed transit hub"
}
```

### Freight Audit

```
POST /api/scm/log/freight-audits
{
  "tenant_id": "acme",
  "shipment_id": "shp-xyz",
  "carrier_id": "carrier-abc",
  "invoice_number": "INV-2024-001",
  "invoiced_amount": 1250.00,
  "expected_amount": 1100.00
}
```

## Status Flows

- **Shipment**: draft → booked → in_transit → delivered | exception | cancelled
- **Freight Audit**: pending → approved | disputed → resolved
- **Customs Document**: draft → submitted → approved | rejected
- **Delivery Exception**: open → resolved
