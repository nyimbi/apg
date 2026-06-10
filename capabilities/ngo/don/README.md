# Donor Relationship Management (ngo_don)

Donor registry, communication history, pledge tracking, receipt generation, stewardship plans.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/ngo/don/health` | Service health check |
| GET | `/api/ngo/don/` | List donors |
| POST | `/api/ngo/don/` | Create donor |
| GET | `/api/ngo/don/<id>` | Get donor |
| PUT | `/api/ngo/don/<id>` | Update donor |
| DELETE | `/api/ngo/don/<id>` | Deactivate donor |
| GET | `/api/ngo/don/search?q=` | Search donors |
| GET | `/api/ngo/don/<id>/communications` | Communication history |
| POST | `/api/ngo/don/<id>/communications` | Log communication |
| GET | `/api/ngo/don/<id>/pledges` | List pledges |
| POST | `/api/ngo/don/<id>/pledges` | Create pledge |
| GET | `/api/ngo/don/<id>/receipts` | List receipts |
| POST | `/api/ngo/don/<id>/receipts` | Generate receipt |
| GET | `/api/ngo/don/<id>/stewardship` | Stewardship plans |
| POST | `/api/ngo/don/<id>/stewardship` | Create stewardship plan |
| GET | `/api/ngo/don/<id>/history` | Giving history |
| GET | `/api/ngo/don/portfolio/summary` | Portfolio summary |
| GET | `/api/ngo/don/portfolio/retention` | Retention analysis |
| GET | `/api/ngo/don/pledges/overdue` | Overdue pledges |
| GET | `/api/ngo/don/audit-events` | Audit log |
