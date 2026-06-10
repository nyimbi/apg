# leg_adr — ADR / Dispute Resolution

Arbitration case management, mediation workflows, settlement tracking, award enforcement.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/legal/adr/health | Health check |
| GET | /api/legal/adr/cases | List cases |
| GET | /api/legal/adr/cases/{id} | Get case |
| POST | /api/legal/adr/cases | File case |
| PUT | /api/legal/adr/cases/{id} | Update case |
| DELETE | /api/legal/adr/cases/{id} | Close case |
| POST | /api/legal/adr/cases/{id}/advance | Advance status |
| GET | /api/legal/adr/cases/{id}/neutrals | List neutrals |
| POST | /api/legal/adr/neutrals | Appoint neutral |
| POST | /api/legal/adr/neutrals/{id}/challenge | Challenge neutral |
| DELETE | /api/legal/adr/neutrals/{id} | Remove neutral |
| GET | /api/legal/adr/cases/{id}/proceedings | List proceedings |
| POST | /api/legal/adr/proceedings | Schedule proceeding |
| POST | /api/legal/adr/proceedings/{id}/conclude | Conclude proceeding |
| DELETE | /api/legal/adr/proceedings/{id} | Cancel proceeding |
| GET | /api/legal/adr/awards | List awards |
| GET | /api/legal/adr/awards/{id} | Get award |
| POST | /api/legal/adr/awards | Render award |
| PUT | /api/legal/adr/awards/{id} | Update award |
| DELETE | /api/legal/adr/awards/{id} | Set aside award |
| POST | /api/legal/adr/awards/{id}/challenge | Challenge award |
| POST | /api/legal/adr/awards/{id}/enforce | File enforcement |
| GET | /api/legal/adr/settlements | List settlements |
| POST | /api/legal/adr/settlements | Record settlement |
| PUT | /api/legal/adr/settlements/{id} | Update settlement |
| DELETE | /api/legal/adr/settlements/{id} | Void settlement |
| GET | /api/legal/adr/dashboard | ADR dashboard |
| GET | /api/legal/adr/audit | Audit events |

## Service Class

`ADRDisputeResolutionService` — supports arbitration, mediation, conciliation, expert determination. Auto-generates case numbers (ARB-YYYY-NNNN), tracks panel constitution, proceedings, awards with set-aside/enforcement lifecycle, and negotiated settlements.
