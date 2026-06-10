# Results-Based Financing (ngo_rbf)

Result verification, payment triggers, disbursement-linked indicators, third-party verification.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/ngo/rbf/health` | Health check |
| GET | `/api/ngo/rbf/contracts` | List contracts |
| POST | `/api/ngo/rbf/contracts` | Create contract |
| GET | `/api/ngo/rbf/contracts/<id>` | Get contract |
| PUT | `/api/ngo/rbf/contracts/<id>` | Update contract |
| DELETE | `/api/ngo/rbf/contracts/<id>` | Delete contract |
| POST | `/api/ngo/rbf/contracts/<id>/activate` | Activate contract |
| GET | `/api/ngo/rbf/contracts/<id>/performance` | Performance summary |
| GET | `/api/ngo/rbf/contracts/<id>/dli-achievement` | DLI achievement report |
| GET | `/api/ngo/rbf/dlis` | List DLIs |
| POST | `/api/ngo/rbf/dlis` | Create DLI |
| GET | `/api/ngo/rbf/dlis/<id>` | Get DLI |
| GET | `/api/ngo/rbf/claims` | List claims |
| POST | `/api/ngo/rbf/claims` | Submit result claim |
| GET | `/api/ngo/rbf/claims/<id>` | Get claim |
| GET | `/api/ngo/rbf/verifications` | List verifications |
| POST | `/api/ngo/rbf/verifications` | Create verification |
| GET | `/api/ngo/rbf/payment-triggers` | List payment triggers |
| POST | `/api/ngo/rbf/payment-triggers` | Trigger payment |
| POST | `/api/ngo/rbf/payment-triggers/<id>/confirm` | Confirm payment |
| GET | `/api/ngo/rbf/portfolio/summary` | Portfolio summary |
| GET | `/api/ngo/rbf/audit-events` | Audit log |
