# Programme & Project Monitoring (ngo_prg)

Logframe management, activity tracking, output/outcome recording, field data collection.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/ngo/prg/health` | Health check |
| GET | `/api/ngo/prg/` | List programmes |
| POST | `/api/ngo/prg/` | Create programme |
| GET | `/api/ngo/prg/<id>` | Get programme |
| PUT | `/api/ngo/prg/<id>` | Update programme |
| DELETE | `/api/ngo/prg/<id>` | Delete programme |
| POST | `/api/ngo/prg/<id>/activate` | Activate programme |
| GET | `/api/ngo/prg/<id>/logframes` | List logframes |
| POST | `/api/ngo/prg/<id>/logframes` | Create logframe |
| GET | `/api/ngo/prg/<id>/activities` | List activities |
| POST | `/api/ngo/prg/<id>/activities` | Create activity |
| GET | `/api/ngo/prg/<id>/outputs` | List outputs |
| POST | `/api/ngo/prg/<id>/field-data` | Submit field data |
| GET | `/api/ngo/prg/<id>/field-data` | List field data |
| GET | `/api/ngo/prg/<id>/progress` | Progress report |
| GET | `/api/ngo/prg/<id>/gantt` | Gantt chart data |
| GET | `/api/ngo/prg/portfolio/overview` | Portfolio overview |
| GET | `/api/ngo/prg/audit-events` | Audit log |
