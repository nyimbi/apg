# Insurance Regulatory Reporting (ins_reg)

IRA/NAICOM/FSA returns, Solvency II reporting, statistical returns, market conduct filings.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/insurance/reg/health | Health check |
| GET | /api/insurance/reg/describe | Capability description |
| GET | /api/insurance/reg/returns | List returns |
| POST | /api/insurance/reg/returns | Create return |
| GET | /api/insurance/reg/returns/{id} | Get return |
| PUT | /api/insurance/reg/returns/{id} | Update return |
| DELETE | /api/insurance/reg/returns/{id} | Cancel return |
| POST | /api/insurance/reg/returns/{id}/review | Review return |
| POST | /api/insurance/reg/returns/{id}/submit | Submit to regulator |
| POST | /api/insurance/reg/returns/{id}/accept | Record acceptance |
| POST | /api/insurance/reg/solvency | Prepare solvency report |
| GET | /api/insurance/reg/solvency | List solvency reports |
| POST | /api/insurance/reg/statistical | Compile statistical return |
| POST | /api/insurance/reg/market-conduct | File market conduct |
| GET | /api/insurance/reg/market-conduct | List filings |
| GET | /api/insurance/reg/calendar | Compliance calendar |
| GET | /api/insurance/reg/calendar/upcoming | Upcoming deadlines |
| POST | /api/insurance/reg/calendar | Add deadline |
| GET | /api/insurance/reg/summary | Regulatory summary |
| GET | /api/insurance/reg/audit | Audit trail |
