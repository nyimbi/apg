# Beneficiary Registry (ngo_ben)

Beneficiary profiling, programme enrolment, vulnerability scoring, transfer management, deduplication.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/ngo/ben/health` | Health check |
| GET | `/api/ngo/ben/` | List beneficiaries |
| POST | `/api/ngo/ben/` | Register beneficiary |
| GET | `/api/ngo/ben/<id>` | Get beneficiary |
| PUT | `/api/ngo/ben/<id>` | Update beneficiary |
| DELETE | `/api/ngo/ben/<id>` | Deactivate beneficiary |
| GET | `/api/ngo/ben/<id>/enrolments` | List enrolments |
| POST | `/api/ngo/ben/<id>/enrolments` | Enrol in programme |
| GET | `/api/ngo/ben/<id>/assessments` | List vulnerability assessments |
| POST | `/api/ngo/ben/<id>/assessments` | Create vulnerability assessment |
| GET | `/api/ngo/ben/<id>/transfers` | List transfers |
| POST | `/api/ngo/ben/<id>/transfers` | Create transfer |
| GET | `/api/ngo/ben/<id>/dedup` | Duplicate check |
| GET | `/api/ngo/ben/analytics/vulnerability` | Vulnerability distribution |
| POST | `/api/ngo/ben/analytics/dedup-scan` | Full registry dedup scan |
| GET | `/api/ngo/ben/audit-events` | Audit log |
