# Supplier Relationship Management User Guide

## Overview

`scm_srm` manages the full supplier lifecycle: onboarding and approval, periodic scorecard measurement, risk assessment, collaboration messaging, formal performance reviews, preferred supplier designation, and certification tracking.

## Key Use Cases

- **Supplier onboarding**: Register suppliers with category, payment terms, and contact details; route through approval workflow.
- **Scorecard**: Quarterly multi-dimension scoring (quality, delivery, responsiveness, cost, sustainability).
- **Risk management**: Record financial, geopolitical, operational, compliance, ESG, and concentration risks; track mitigation plans.
- **Collaboration portal**: Send forecasts, PO updates, complaints, and escalations directly to supplier contacts.
- **Performance reviews**: Formal periodic reviews with action items and next review scheduling.
- **Preferred suppliers**: Grant/revoke preferred status with rationale; filter sourcing to preferred list.
- **Certifications**: Track ISO, halal, organic, fair-trade, and custom certs with expiry dates.

## API Reference

### Create Supplier

```
POST /api/scm/srm/suppliers
{
  "tenant_id": "acme",
  "name": "Acme Packaging Ltd",
  "supplier_code": "PKG-001",
  "country": "KE",
  "category": "packaging",
  "payment_terms": "NET45"
}
```

### Create Scorecard

```
POST /api/scm/srm/scorecards
{
  "tenant_id": "acme",
  "supplier_id": "supp-xyz",
  "period": "2026-Q2",
  "quality_score": 8.5,
  "delivery_score": 9.0,
  "responsiveness_score": 7.5,
  "cost_score": 8.0,
  "reviewed_by": "procurement.manager"
}
```

## Supplier Status Flow

pending_approval → active → probation | suspended | blacklisted | inactive
