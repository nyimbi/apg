# Actuarial Tools (ins_act)

Mortality tables, loss ratios, reserve calculations, IBNR, pricing models, experience analysis.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/insurance/act/health | Health check |
| GET | /api/insurance/act/describe | Capability description |
| GET | /api/insurance/act/mortality-tables | List mortality tables |
| POST | /api/insurance/act/mortality-tables | Load mortality table |
| GET | /api/insurance/act/mortality-tables/{id} | Get table |
| DELETE | /api/insurance/act/mortality-tables/{id} | Retire table |
| POST | /api/insurance/act/loss-ratio | Calculate loss ratio |
| GET | /api/insurance/act/loss-ratios | List loss ratio reports |
| POST | /api/insurance/act/reserves | Calculate reserve |
| GET | /api/insurance/act/reserves | List reserves |
| POST | /api/insurance/act/ibnr | Estimate IBNR |
| GET | /api/insurance/act/ibnr | List IBNR estimates |
| POST | /api/insurance/act/pricing-models | Create pricing model |
| GET | /api/insurance/act/pricing-models | List pricing models |
| POST | /api/insurance/act/pricing-models/{id}/apply | Apply model |
| POST | /api/insurance/act/experience-analysis | Run analysis |
| GET | /api/insurance/act/experience-analyses | List analyses |
| GET | /api/insurance/act/summary | Actuarial summary |
| GET | /api/insurance/act/audit | Audit trail |
