# leg_ctr — Contract Lifecycle Management

Drafting, review, redlining, approval workflow, e-signature, renewal alerts, obligations tracking.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/legal/ctr/health | Health check |
| GET | /api/legal/ctr/contracts | List contracts |
| GET | /api/legal/ctr/contracts/{id} | Get contract |
| POST | /api/legal/ctr/contracts | Create contract |
| PUT | /api/legal/ctr/contracts/{id} | Update contract |
| DELETE | /api/legal/ctr/contracts/{id} | Archive contract |
| POST | /api/legal/ctr/contracts/{id}/submit | Submit for review |
| POST | /api/legal/ctr/contracts/{id}/execute | Execute contract |
| POST | /api/legal/ctr/contracts/{id}/terminate | Terminate contract |
| GET | /api/legal/ctr/contracts/{id}/redlines | List redlines |
| POST | /api/legal/ctr/redlines | Create redline |
| POST | /api/legal/ctr/redlines/{id}/resolve | Resolve redline |
| POST | /api/legal/ctr/obligations | Create obligation |
| POST | /api/legal/ctr/approvals | Request approval |
| POST | /api/legal/ctr/approvals/{id}/decide | Decide approval |
| GET | /api/legal/ctr/expiring | Expiring contracts |
| GET | /api/legal/ctr/dashboard | Contract dashboard |
| GET | /api/legal/ctr/audit | Audit events |

## Service Class

`ContractLifecycleService` — full contract lifecycle from draft to execution, with version history, redlining, multi-level approval, e-signature, renewals, and obligation tracking.
