# County / Devolved Services (gov_cty)

County revenue collection, permit issuance, social welfare, devolved health, public works ticketing.

## Overview

End-to-end county government service management: collect rates and fees, issue business/building permits, manage social welfare programmes, run devolved health facilities, and track public works maintenance tickets. Multi-tenant, designed for Kenya's 47 counties.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/government/cty/health | Service health check |
| GET | /api/government/cty/dashboard | Dashboard metrics |
| GET | /api/government/cty/revenues | List revenues |
| GET | /api/government/cty/revenues/{id} | Get revenue record |
| POST | /api/government/cty/revenues | Collect revenue |
| POST | /api/government/cty/revenues/{id}/confirm | Confirm payment |
| GET | /api/government/cty/revenues/summary | Revenue summary |
| GET | /api/government/cty/permits | List permits |
| GET | /api/government/cty/permits/{id} | Get permit |
| POST | /api/government/cty/permits | Apply for permit |
| PUT | /api/government/cty/permits/{id} | Update permit |
| DELETE | /api/government/cty/permits/{id} | Delete permit |
| GET | /api/government/cty/welfare | List welfare applications |
| GET | /api/government/cty/welfare/{id} | Get application |
| POST | /api/government/cty/welfare | Apply for welfare |
| PUT | /api/government/cty/welfare/{id} | Update application |
| GET | /api/government/cty/health-facilities | List facilities |
| POST | /api/government/cty/health-facilities | Register facility |
| POST | /api/government/cty/patients | Register patient |
| GET | /api/government/cty/tickets | List tickets |
| GET | /api/government/cty/tickets/{id} | Get ticket |
| POST | /api/government/cty/tickets | Create ticket |
| PUT | /api/government/cty/tickets/{id} | Update ticket |
| DELETE | /api/government/cty/tickets/{id} | Delete ticket |
| GET | /api/government/cty/audit-events | List audit events |
