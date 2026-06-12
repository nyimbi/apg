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

---

## World-Class Enhancements (v2.0)

Fifteen targeted improvements over baseline implementation:

- **I1. Tiered Interest Rate Engine** [Enhancement]
- **I2. Goal-Based Savings Targets** [Enhancement]
- **I3. Standing Order / Recurring Deposit Scheduling** [Enhancement]
- **I4. Fixed Deposit Maturity Processing** [Enhancement]
- **I5. Dividend / Interest Capitalisation** [Enhancement]
- **I6. Withdrawal Notice Period Enforcement** [Enhancement]
- **I7. Dormancy Scoring & Automated Classification** [Enhancement]
- **I8. Deposit Limit & Velocity Controls** [Enhancement]
- **I9. Inter-Account Transfer** [Enhancement]
- **I10. Projected Balance & Interest Calculator** [Enhancement]
- **I11. Regulatory Reporting — SASRA SF01 Export** [Enhancement]
- **I12. Multi-Currency Savings Support** [Enhancement]
- **I13. Savings Group / Chama Account Support** [Enhancement]
- **I14. Penalty & Charge Engine** [Enhancement]
- **I15. Real-Time Balance Notification Hooks** [Enhancement]

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
