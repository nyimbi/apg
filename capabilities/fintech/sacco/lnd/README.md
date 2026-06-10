# SACCO Lending (fintech_sacco_lnd)

Loan products, credit scoring, guarantor management, repayment schedules, arrears management, and CRB reporting.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/fintech/sacco/lnd/health | Service health |
| GET | /api/fintech/sacco/lnd/products | List loan products |
| POST | /api/fintech/sacco/lnd/products | Create product |
| GET | /api/fintech/sacco/lnd/products/{id} | Get product |
| PUT | /api/fintech/sacco/lnd/products/{id} | Update product |
| DELETE | /api/fintech/sacco/lnd/products/{id} | Deactivate product |
| GET | /api/fintech/sacco/lnd/loans | List loans |
| POST | /api/fintech/sacco/lnd/loans | Apply for loan |
| GET | /api/fintech/sacco/lnd/loans/{id} | Get loan |
| PUT | /api/fintech/sacco/lnd/loans/{id} | Update loan |
| DELETE | /api/fintech/sacco/lnd/loans/{id} | Cancel loan |
| POST | /api/fintech/sacco/lnd/loans/{id}/approve | Approve loan |
| POST | /api/fintech/sacco/lnd/loans/{id}/reject | Reject loan |
| POST | /api/fintech/sacco/lnd/loans/{id}/disburse | Disburse loan |
| GET | /api/fintech/sacco/lnd/loans/{id}/schedule | Repayment schedule |
| POST | /api/fintech/sacco/lnd/repayments | Record repayment |
| GET | /api/fintech/sacco/lnd/repayments | List repayments |
| POST | /api/fintech/sacco/lnd/credit-score | Compute credit score |
| GET | /api/fintech/sacco/lnd/credit-score/{member_id} | Get member credit score |
| POST | /api/fintech/sacco/lnd/arrears/check | Run arrears check |
| GET | /api/fintech/sacco/lnd/arrears | List arrears |
| POST | /api/fintech/sacco/lnd/crb | Submit CRB report |
| GET | /api/fintech/sacco/lnd/crb | List CRB reports |
| GET | /api/fintech/sacco/lnd/summary | Portfolio summary |
| GET | /api/fintech/sacco/lnd/audit | Audit events |
