# Grant Management (ngo_grn)

Grant pipeline, proposal management, budget tracking, disbursement, compliance reporting, audits.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/ngo/grn/health` | Service health check |
| GET | `/api/ngo/grn/` | List grants |
| POST | `/api/ngo/grn/` | Create grant |
| GET | `/api/ngo/grn/<id>` | Get grant |
| PUT | `/api/ngo/grn/<id>` | Update grant |
| DELETE | `/api/ngo/grn/<id>` | Delete grant |
| POST | `/api/ngo/grn/<id>/activate` | Activate grant |
| POST | `/api/ngo/grn/<id>/close` | Close grant |
| GET | `/api/ngo/grn/<id>/proposals` | List proposals |
| POST | `/api/ngo/grn/<id>/proposals` | Submit proposal |
| GET | `/api/ngo/grn/<id>/budget-lines` | List budget lines |
| POST | `/api/ngo/grn/<id>/budget-lines` | Create budget line |
| GET | `/api/ngo/grn/<id>/disbursements` | List disbursements |
| POST | `/api/ngo/grn/<id>/disbursements` | Record disbursement |
| GET | `/api/ngo/grn/<id>/compliance-reports` | List compliance reports |
| POST | `/api/ngo/grn/<id>/compliance-reports` | Submit compliance report |
| GET | `/api/ngo/grn/<id>/audit-findings` | List audit findings |
| POST | `/api/ngo/grn/<id>/audit-findings` | Record audit finding |
| GET | `/api/ngo/grn/<id>/summary` | Donor-facing summary |
| GET | `/api/ngo/grn/portfolio/summary` | Portfolio summary |
| GET | `/api/ngo/grn/audit-events` | Audit event log |
