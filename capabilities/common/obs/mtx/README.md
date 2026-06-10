# obs_mtx — Metrics & SLO

RED metrics (Rate/Error/Duration), SLO definition, burn rate alerts, Prometheus export, dashboard generation.

**Capability ID:** `obs_mtx` | **Domain:** observability | **Version:** 1.0.0

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/obs/mtx/health` | Health check |
| GET | `/api/obs/mtx/describe` | Capability descriptor |
| GET | `/api/obs/mtx/metrics` | List metric definitions |
| POST | `/api/obs/mtx/metrics` | Create metric definition |
| GET | `/api/obs/mtx/metrics/<id>` | Get metric definition |
| PUT | `/api/obs/mtx/metrics/<id>` | Update metric definition |
| DELETE | `/api/obs/mtx/metrics/<id>` | Delete metric definition |
| POST | `/api/obs/mtx/data-points` | Record a metric data point |
| POST | `/api/obs/mtx/data-points/bulk` | Bulk record data points |
| GET | `/api/obs/mtx/data-points/query` | Query data points |
| GET | `/api/obs/mtx/red/<service_name>` | RED metrics for one service |
| GET | `/api/obs/mtx/red` | RED metrics for all services |
| GET | `/api/obs/mtx/slos` | List SLOs |
| POST | `/api/obs/mtx/slos` | Create SLO |
| GET | `/api/obs/mtx/slos/<id>` | Get SLO |
| PUT | `/api/obs/mtx/slos/<id>` | Update SLO |
| DELETE | `/api/obs/mtx/slos/<id>` | Delete SLO |
| GET | `/api/obs/mtx/slos/<id>/evaluate` | Evaluate SLO compliance |
| GET | `/api/obs/mtx/slos/evaluate-all` | Evaluate all SLOs |
| GET | `/api/obs/mtx/burn-rate-alerts` | List burn rate alerts |
| POST | `/api/obs/mtx/burn-rate-alerts` | Create burn rate alert |
| GET | `/api/obs/mtx/burn-rate-alerts/<id>` | Get alert |
| PUT | `/api/obs/mtx/burn-rate-alerts/<id>` | Update alert |
| DELETE | `/api/obs/mtx/burn-rate-alerts/<id>` | Delete alert |
| GET | `/api/obs/mtx/burn-rate-alerts/<id>/evaluate` | Evaluate burn rate |
| POST | `/api/obs/mtx/prometheus/config` | Configure Prometheus export |
| GET | `/api/obs/mtx/prometheus/metrics` | Prometheus text exposition |
| GET | `/api/obs/mtx/dashboards` | List dashboards |
| POST | `/api/obs/mtx/dashboards` | Create dashboard |
| GET | `/api/obs/mtx/dashboards/<id>` | Get dashboard |
| PUT | `/api/obs/mtx/dashboards/<id>` | Update dashboard |
| DELETE | `/api/obs/mtx/dashboards/<id>` | Delete dashboard |
| POST | `/api/obs/mtx/dashboards/generate/red/<service>` | Auto-generate RED dashboard |
| GET | `/api/obs/mtx/audit` | Audit event log |

## Headers

Pass `X-Tenant-ID: <tenant>` on every request for multi-tenant isolation.
