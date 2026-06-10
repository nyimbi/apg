# Logistics & Transportation (scm_log)

Carrier integration, shipment tracking, freight audit, route optimisation, customs documentation, 3PL management.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/scm/log/health | Health check |
| GET | /api/scm/log/describe | Capability contract |
| GET | /api/scm/log/carriers | List carriers |
| POST | /api/scm/log/carriers | Create carrier |
| GET | /api/scm/log/carriers/{id} | Get carrier |
| PUT | /api/scm/log/carriers/{id} | Update carrier |
| DELETE | /api/scm/log/carriers/{id} | Delete carrier |
| GET | /api/scm/log/shipments | List shipments |
| POST | /api/scm/log/shipments | Create shipment |
| GET | /api/scm/log/shipments/{id} | Get shipment |
| PUT | /api/scm/log/shipments/{id} | Update shipment |
| POST | /api/scm/log/shipments/{id}/book | Book shipment |
| POST | /api/scm/log/shipments/{id}/cancel | Cancel shipment |
| GET | /api/scm/log/shipments/{id}/tracking | Get tracking events |
| POST | /api/scm/log/shipments/{id}/tracking | Add tracking event |
| GET | /api/scm/log/freight-audits | List freight audits |
| POST | /api/scm/log/freight-audits | Create freight audit |
| POST | /api/scm/log/freight-audits/{id}/resolve | Resolve audit |
| GET | /api/scm/log/routes | List routes |
| POST | /api/scm/log/routes | Create route |
| POST | /api/scm/log/routes/{id}/optimise | Optimise route |
| GET | /api/scm/log/analytics/shipments | Shipment analytics |
| GET | /api/scm/log/analytics/freight-costs | Freight cost summary |
| GET | /api/scm/log/audit-events | Audit events |
