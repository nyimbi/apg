# Revenue Management & Rates (hos_rms)

Dynamic pricing, demand forecasting, rate parity, yield optimisation, and competitor rate monitoring.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/hospitality/rms/health | Health check |
| GET | /api/hospitality/rms/rate-plans | List rate plans |
| POST | /api/hospitality/rms/rate-plans | Create rate plan |
| GET | /api/hospitality/rms/rate-plans/{id} | Get rate plan |
| PUT | /api/hospitality/rms/rate-plans/{id} | Update rate plan |
| DELETE | /api/hospitality/rms/rate-plans/{id} | Deactivate rate plan |
| GET | /api/hospitality/rms/rate-plans/{id}/effective-rate | Compute effective rate |
| GET | /api/hospitality/rms/forecasts | List demand forecasts |
| POST | /api/hospitality/rms/forecasts | Create forecast |
| GET | /api/hospitality/rms/competitor-rates | List competitor rates |
| POST | /api/hospitality/rms/competitor-rates | Add competitor rate |
| GET | /api/hospitality/rms/parity-alerts | List parity alerts |
| POST | /api/hospitality/rms/parity-alerts/{id}/resolve | Resolve alert |
| POST | /api/hospitality/rms/yield-optimisation | Run yield optimisation |
| GET | /api/hospitality/rms/yield-reports | List yield reports |
| POST | /api/hospitality/rms/seasonal-rules | Create seasonal rule |
| POST | /api/hospitality/rms/revenue-targets | Set revenue target |
| GET | /api/hospitality/rms/parity-report | Rate parity report |
| GET | /api/hospitality/rms/dashboard | Dashboard |
