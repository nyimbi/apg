# Agricultural Credit Scoring (agr_crd)

Yield-based credit scoring, seasonal loan products, group lending, collateral registry.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/agriculture/crd/health | Health check |
| GET | /api/agriculture/crd/profiles | List profiles |
| POST | /api/agriculture/crd/profiles | Create profile |
| GET | /api/agriculture/crd/profiles/{id} | Get profile |
| PUT | /api/agriculture/crd/profiles/{id} | Update profile |
| DELETE | /api/agriculture/crd/profiles/{id} | Delete profile |
| POST | /api/agriculture/crd/score/{farmer_id} | Score farmer |
| GET | /api/agriculture/crd/loans | List loans |
| POST | /api/agriculture/crd/loans | Apply for loan |
| GET | /api/agriculture/crd/loans/{id} | Get loan |
| PUT | /api/agriculture/crd/loans/{id} | Update loan |
| POST | /api/agriculture/crd/loans/{id}/repayment | Record repayment |
| GET | /api/agriculture/crd/collateral | List collateral |
| POST | /api/agriculture/crd/collateral | Register collateral |
| GET | /api/agriculture/crd/groups | Group loans |
| POST | /api/agriculture/crd/groups | Create group loan |
| GET | /api/agriculture/crd/portfolio | Portfolio summary |
| GET | /api/agriculture/crd/audit | Audit log |
