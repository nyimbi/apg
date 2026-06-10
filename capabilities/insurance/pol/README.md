# Policy Administration (ins_pol)

Policy lifecycle management: issuance, endorsements, renewals, cancellations, reinstatements, and document generation.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/insurance/pol/health | Service health check |
| GET | /api/insurance/pol/describe | Capability description |
| GET | /api/insurance/pol/policies | List policies |
| POST | /api/insurance/pol/policies | Issue new policy |
| GET | /api/insurance/pol/policies/{id} | Get policy detail |
| PUT | /api/insurance/pol/policies/{id} | Update policy |
| DELETE | /api/insurance/pol/policies/{id} | Void draft policy |
| POST | /api/insurance/pol/policies/{id}/endorse | Create endorsement |
| POST | /api/insurance/pol/policies/{id}/renew | Initiate renewal |
| POST | /api/insurance/pol/policies/{id}/cancel | Cancel policy |
| POST | /api/insurance/pol/policies/{id}/reinstate | Reinstate policy |
| POST | /api/insurance/pol/policies/{id}/documents | Generate document |
| GET | /api/insurance/pol/portfolio/summary | Portfolio metrics |
| GET | /api/insurance/pol/audit | Audit trail |
