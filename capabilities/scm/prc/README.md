# Procurement Management (scm_prc)

RFQ, purchase order, three-way match, vendor evaluation, contract compliance, spend analytics.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/scm/prc/health | Health check |
| GET | /api/scm/prc/describe | Capability contract |
| GET | /api/scm/prc/rfqs | List RFQs |
| POST | /api/scm/prc/rfqs | Create RFQ |
| GET | /api/scm/prc/rfqs/{id} | Get RFQ |
| POST | /api/scm/prc/rfqs/{id}/issue | Issue RFQ to vendors |
| POST | /api/scm/prc/rfqs/{id}/award | Award RFQ |
| GET | /api/scm/prc/purchase-orders | List purchase orders |
| POST | /api/scm/prc/purchase-orders | Create PO |
| GET | /api/scm/prc/purchase-orders/{id} | Get PO |
| PUT | /api/scm/prc/purchase-orders/{id} | Update PO |
| DELETE | /api/scm/prc/purchase-orders/{id} | Cancel PO |
| POST | /api/scm/prc/purchase-orders/{id}/send | Send PO to vendor |
| POST | /api/scm/prc/purchase-orders/{id}/receive | Record goods receipt |
| GET | /api/scm/prc/three-way-matches | List 3-way matches |
| POST | /api/scm/prc/three-way-matches | Create 3-way match |
| GET | /api/scm/prc/vendor-evaluations | List evaluations |
| POST | /api/scm/prc/vendor-evaluations | Create evaluation |
| GET | /api/scm/prc/contracts | List contracts |
| POST | /api/scm/prc/contracts | Create contract |
| GET | /api/scm/prc/analytics/spend | Spend analytics |
| GET | /api/scm/prc/analytics/dashboard | Procurement dashboard |
| GET | /api/scm/prc/audit-events | Audit events |
