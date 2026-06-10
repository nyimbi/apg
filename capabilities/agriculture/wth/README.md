# Weather & Climate Analytics (agr_wth)

Forecast integration, alert thresholds, historical patterns, climate risk assessment.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/agriculture/wth/health | Health check |
| GET | /api/agriculture/wth/forecasts | List forecasts |
| POST | /api/agriculture/wth/forecasts | Ingest forecast |
| GET | /api/agriculture/wth/forecasts/latest | Latest forecast |
| DELETE | /api/agriculture/wth/forecasts/{id} | Delete forecast |
| GET | /api/agriculture/wth/thresholds | List thresholds |
| POST | /api/agriculture/wth/thresholds | Create threshold |
| PUT | /api/agriculture/wth/thresholds/{id} | Update threshold |
| DELETE | /api/agriculture/wth/thresholds/{id} | Delete threshold |
| GET | /api/agriculture/wth/alerts | List alerts |
| POST | /api/agriculture/wth/alerts/{id}/acknowledge | Acknowledge alert |
| GET | /api/agriculture/wth/history | Historical patterns |
| POST | /api/agriculture/wth/history | Add historical record |
| GET | /api/agriculture/wth/history/normals | Monthly normals |
| POST | /api/agriculture/wth/risk-assessments | Compute risk |
| GET | /api/agriculture/wth/risk-assessments | List assessments |
| GET | /api/agriculture/wth/audit | Audit log |
