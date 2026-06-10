# Data Quality (dcat_dq)

Dataset profiling, quality scoring, anomaly detection, completeness/uniqueness/accuracy rules, DQ reports.

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/dcat/dq/health | Service health |
| GET | /api/dcat/dq/rules | List rules |
| POST | /api/dcat/dq/rules | Create rule |
| GET | /api/dcat/dq/rules/{id} | Get rule |
| PUT | /api/dcat/dq/rules/{id} | Update rule |
| DELETE | /api/dcat/dq/rules/{id} | Delete rule |
| POST | /api/dcat/dq/profiles | Profile dataset |
| GET | /api/dcat/dq/profiles/{dataset_id} | Get latest profile |
| POST | /api/dcat/dq/runs | Run quality checks |
| GET | /api/dcat/dq/runs | List runs |
| GET | /api/dcat/dq/runs/{id} | Get run |
| GET | /api/dcat/dq/anomalies | List anomalies |
| POST | /api/dcat/dq/anomalies/{id}/acknowledge | Acknowledge anomaly |
| GET | /api/dcat/dq/scorecard/{dataset_id} | Quality scorecard |
| GET | /api/dcat/dq/reports/{dataset_id} | DQ report |
| GET | /api/dcat/dq/dashboard | Tenant dashboard |
| GET | /api/dcat/dq/audit | Audit trail |
