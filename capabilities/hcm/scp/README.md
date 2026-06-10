# Succession Planning (hcm_scp)

Capability for talent management and succession planning: talent pools, readiness assessments, nine-box grid placement, succession scenarios, and critical role identification.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/hcm/scp/health | Health check |
| GET | /api/hcm/scp/describe | Capability contract |
| GET | /api/hcm/scp/talent-pools | List talent pools |
| GET | /api/hcm/scp/talent-pools/{id} | Get talent pool |
| POST | /api/hcm/scp/talent-pools | Create talent pool |
| PUT | /api/hcm/scp/talent-pools/{id} | Update talent pool |
| DELETE | /api/hcm/scp/talent-pools/{id} | Delete talent pool |
| GET | /api/hcm/scp/talent-pools/{id}/members | List members |
| POST | /api/hcm/scp/talent-pools/{id}/members | Add member |
| DELETE | /api/hcm/scp/talent-pools/{id}/members/{eid} | Remove member |
| GET | /api/hcm/scp/readiness-assessments | List assessments |
| POST | /api/hcm/scp/readiness-assessments | Create assessment |
| PUT | /api/hcm/scp/readiness-assessments/{id} | Update assessment |
| DELETE | /api/hcm/scp/readiness-assessments/{id} | Delete assessment |
| GET | /api/hcm/scp/nine-box | List nine-box entries |
| GET | /api/hcm/scp/nine-box/grid | Nine-box grid view |
| POST | /api/hcm/scp/nine-box | Place on nine-box |
| GET | /api/hcm/scp/scenarios | List succession scenarios |
| POST | /api/hcm/scp/scenarios | Create scenario |
| PUT | /api/hcm/scp/scenarios/{id} | Update scenario |
| PUT | /api/hcm/scp/scenarios/{id}/activate | Activate scenario |
| DELETE | /api/hcm/scp/scenarios/{id} | Delete scenario |
| GET | /api/hcm/scp/critical-roles | List critical roles |
| POST | /api/hcm/scp/critical-roles | Identify critical role |
| PUT | /api/hcm/scp/critical-roles/{id} | Update critical role |
| DELETE | /api/hcm/scp/critical-roles/{id} | Remove critical role |
| GET | /api/hcm/scp/coverage-report | Succession coverage |
| GET | /api/hcm/scp/readiness-report | Readiness by pool |
| GET | /api/hcm/scp/dashboard | Dashboard |
| GET | /api/hcm/scp/audit-events | Audit trail |
