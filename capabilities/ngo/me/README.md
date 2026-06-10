# M&E — Monitoring & Evaluation (ngo_me)

Indicator framework, data collection, progress reporting, impact assessment, learning cycles.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/ngo/me/health` | Health check |
| GET | `/api/ngo/me/indicators` | List indicators |
| POST | `/api/ngo/me/indicators` | Create indicator |
| GET | `/api/ngo/me/indicators/<id>` | Get indicator |
| PUT | `/api/ngo/me/indicators/<id>` | Update indicator |
| DELETE | `/api/ngo/me/indicators/<id>` | Delete indicator |
| GET | `/api/ngo/me/indicators/<id>/trend` | Trend analysis |
| POST | `/api/ngo/me/data-collections` | Collect data point |
| POST | `/api/ngo/me/data-collections/bulk` | Bulk data collection |
| GET | `/api/ngo/me/data-collections` | List data collections |
| GET | `/api/ngo/me/progress-reports` | List progress reports |
| POST | `/api/ngo/me/progress-reports` | Create progress report |
| POST | `/api/ngo/me/progress-reports/<id>/submit` | Submit report |
| GET | `/api/ngo/me/evaluations` | List evaluations |
| POST | `/api/ngo/me/evaluations` | Create evaluation |
| GET | `/api/ngo/me/learning-cycles` | List learning cycles |
| POST | `/api/ngo/me/learning-cycles` | Create learning cycle |
| POST | `/api/ngo/me/learning-cycles/<id>/findings` | Add findings |
| GET | `/api/ngo/me/dashboard/<programme_id>` | Indicator dashboard |
| GET | `/api/ngo/me/impact/<programme_id>` | Impact summary |
| GET | `/api/ngo/me/audit-events` | Audit log |
