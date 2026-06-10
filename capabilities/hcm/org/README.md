# Organizational Management (hcm_org)

Capability for managing organisational structure: org chart, positions, reporting lines, span of control, headcount planning, and restructuring.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/hcm/org/health | Health check |
| GET | /api/hcm/org/describe | Capability contract |
| GET | /api/hcm/org/units | List org units |
| GET | /api/hcm/org/units/{id} | Get org unit |
| POST | /api/hcm/org/units | Create org unit |
| PUT | /api/hcm/org/units/{id} | Update org unit |
| PUT | /api/hcm/org/units/{id}/move | Move org unit |
| DELETE | /api/hcm/org/units/{id} | Delete org unit |
| GET | /api/hcm/org/chart | Full org chart |
| GET | /api/hcm/org/positions | List positions |
| GET | /api/hcm/org/positions/{id} | Get position |
| POST | /api/hcm/org/positions | Create position |
| PUT | /api/hcm/org/positions/{id} | Update position |
| PUT | /api/hcm/org/positions/{id}/assign | Assign employee |
| DELETE | /api/hcm/org/positions/{id} | Delete position |
| GET | /api/hcm/org/reporting-lines | List reporting lines |
| POST | /api/hcm/org/reporting-lines | Create reporting line |
| GET | /api/hcm/org/restructurings | List restructurings |
| POST | /api/hcm/org/restructurings | Create restructuring |
| PUT | /api/hcm/org/restructurings/{id} | Update restructuring |
| DELETE | /api/hcm/org/restructurings/{id} | Delete restructuring |
| GET | /api/hcm/org/analytics | Org analytics |
| GET | /api/hcm/org/dashboard | Dashboard |
| GET | /api/hcm/org/audit-events | Audit trail |
