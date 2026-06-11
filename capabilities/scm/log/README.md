# Logistics & Transportation (scm_log)

Carrier integration, shipment tracking, freight audit, route optimisation,
customs documentation, 3PL management, carbon footprint tracking, SLA monitoring,
shipment consolidation, carrier scorecards, freight insurance, and proof-of-delivery.

## API Endpoints

### Core

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/scm/log/health | Health check |
| GET | /api/scm/log/describe | Capability contract |
| GET | /api/scm/log/audit-events | Tenant audit event log |

### Carriers

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/scm/log/carriers | List carriers |
| POST | /api/scm/log/carriers | Create carrier |
| GET | /api/scm/log/carriers/{id} | Get carrier |
| PUT | /api/scm/log/carriers/{id} | Update carrier |
| DELETE | /api/scm/log/carriers/{id} | Soft-delete carrier |
| POST | /api/scm/log/carriers/{id}/rate | Rate carrier performance |
| GET | /api/scm/log/carriers/{id}/scorecard | Carrier performance scorecard |

### Shipments

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/scm/log/shipments | List shipments |
| POST | /api/scm/log/shipments | Create shipment |
| POST | /api/scm/log/shipments/bulk | Bulk create shipments |
| GET | /api/scm/log/shipments/{id} | Get shipment |
| PUT | /api/scm/log/shipments/{id} | Update shipment |
| POST | /api/scm/log/shipments/{id}/book | Book shipment |
| POST | /api/scm/log/shipments/{id}/cancel | Cancel shipment |
| POST | /api/scm/log/shipments/consolidate/suggest | Suggest consolidation groups |
| POST | /api/scm/log/shipments/consolidate | Create consolidated shipment |

### Tracking

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/scm/log/shipments/{id}/tracking | Get tracking events |
| POST | /api/scm/log/shipments/{id}/tracking | Add tracking event |
| GET | /api/scm/log/shipments/{id}/pod | Get proof-of-delivery records |
| POST | /api/scm/log/shipments/{id}/pod | Attach proof-of-delivery |

### Freight Audit

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/scm/log/freight-audits | List freight audits |
| POST | /api/scm/log/freight-audits | Create freight audit |
| POST | /api/scm/log/freight-audits/{id}/resolve | Resolve audit |

### Routes

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/scm/log/routes | List routes |
| POST | /api/scm/log/routes | Create route |
| POST | /api/scm/log/routes/{id}/optimise | Optimise route |

### Customs Documents

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/scm/log/customs-documents | List customs documents |
| POST | /api/scm/log/customs-documents | Create customs document |
| POST | /api/scm/log/customs-documents/{id}/submit | Submit document |

### 3PL Providers

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/scm/log/3pl-providers | List 3PL providers |
| POST | /api/scm/log/3pl-providers | Register 3PL provider |
| POST | /api/scm/log/shipments/{id}/assign-3pl | Assign shipment to 3PL |

### Delivery Exceptions

| Method | Path | Description |
|--------|------|-------------|
| POST | /api/scm/log/shipments/{id}/exceptions | Raise delivery exception |
| POST | /api/scm/log/exceptions/{id}/resolve | Resolve exception |

### Carbon & Sustainability

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/scm/log/shipments/{id}/co2 | Calculate shipment CO2e |
| GET | /api/scm/log/analytics/emissions | Emissions report (Scope 3) |

### SLA Monitoring

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/scm/log/sla/breaches | Check SLA breaches and at-risk shipments |
| GET | /api/scm/log/sla/performance | SLA performance report by carrier |

### Insurance

| Method | Path | Description |
|--------|------|-------------|
| POST | /api/scm/log/shipments/{id}/insurance/quote | Request insurance quote |
| POST | /api/scm/log/shipments/{id}/insurance/bind | Bind insurance policy |
| POST | /api/scm/log/shipments/{id}/insurance/claim | File insurance claim |

### Analytics

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/scm/log/analytics/shipments | Shipment analytics |
| GET | /api/scm/log/analytics/freight-costs | Freight cost summary |

## Status Flows

```
Shipment:         draft → booked → in_transit → delivered | exception | cancelled
Freight Audit:    pending → approved | disputed → resolved
Customs Document: draft → submitted → approved | rejected
Delivery Exception: open → resolved
Insurance Policy: active → expired | cancelled
Insurance Claim:  filed → under_review → approved | rejected
```

## Supported Values

- **Freight modes**: air, sea, road, rail, multimodal
- **Service levels**: express, standard, economy
- **Carrier types**: air, sea, road, rail, multimodal, courier
- **Document types**: commercial_invoice, packing_list, bill_of_lading, certificate_of_origin, customs_declaration, airway_bill
- **Tracking events**: pickup, in_transit, customs_clearance, out_for_delivery, delivered, exception, returned
