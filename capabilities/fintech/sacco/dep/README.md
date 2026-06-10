# SACCO Deposits & Savings (fintech_sacco_dep)

Savings products, deposit taking, withdrawal processing, minimum balances, and interest accrual.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/fintech/sacco/dep/health | Service health |
| GET | /api/fintech/sacco/dep/products | List savings products |
| POST | /api/fintech/sacco/dep/products | Create product |
| GET | /api/fintech/sacco/dep/products/{id} | Get product |
| PUT | /api/fintech/sacco/dep/products/{id} | Update product |
| DELETE | /api/fintech/sacco/dep/products/{id} | Deactivate product |
| GET | /api/fintech/sacco/dep/accounts | List accounts |
| POST | /api/fintech/sacco/dep/accounts | Open account |
| GET | /api/fintech/sacco/dep/accounts/{id} | Get account |
| PUT | /api/fintech/sacco/dep/accounts/{id} | Update account |
| DELETE | /api/fintech/sacco/dep/accounts/{id} | Close account |
| GET | /api/fintech/sacco/dep/accounts/{id}/statement | Account statement |
| GET | /api/fintech/sacco/dep/accounts/{id}/min-balance | Check minimum balance |
| POST | /api/fintech/sacco/dep/deposits | Make deposit |
| POST | /api/fintech/sacco/dep/withdrawals | Process withdrawal |
| GET | /api/fintech/sacco/dep/transactions | List transactions |
| POST | /api/fintech/sacco/dep/interest/accrue | Run interest accrual |
| GET | /api/fintech/sacco/dep/interest/postings | List interest postings |
| GET | /api/fintech/sacco/dep/summary | Portfolio summary |
| GET | /api/fintech/sacco/dep/audit | Audit events |
