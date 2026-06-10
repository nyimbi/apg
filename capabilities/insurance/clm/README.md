# Claims Management (ins_clm)

FNOL, claims assessment, reserve management, payment processing, fraud detection, and subrogation.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/insurance/clm/health | Service health check |
| GET | /api/insurance/clm/describe | Capability description |
| GET | /api/insurance/clm/claims | List claims |
| POST | /api/insurance/clm/claims | Register FNOL |
| GET | /api/insurance/clm/claims/{id} | Get claim detail |
| PUT | /api/insurance/clm/claims/{id} | Update claim |
| DELETE | /api/insurance/clm/claims/{id} | Withdraw claim |
| POST | /api/insurance/clm/claims/{id}/reserve | Set reserve |
| POST | /api/insurance/clm/claims/{id}/payment | Process payment |
| POST | /api/insurance/clm/claims/{id}/fraud | Fraud assessment |
| POST | /api/insurance/clm/claims/{id}/approve | Approve claim |
| POST | /api/insurance/clm/claims/{id}/subrogation | Initiate subrogation |
| GET | /api/insurance/clm/summary | Claims summary |
| GET | /api/insurance/clm/audit | Audit trail |
