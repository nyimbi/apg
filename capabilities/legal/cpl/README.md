# leg_cpl — Legal Compliance Management

Regulatory requirement tracking, compliance calendar, evidence collection, breach reporting.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/legal/cpl/health | Health check |
| GET | /api/legal/cpl/requirements | List requirements |
| GET | /api/legal/cpl/requirements/{id} | Get requirement |
| POST | /api/legal/cpl/requirements | Create requirement |
| PUT | /api/legal/cpl/requirements/{id} | Update requirement |
| DELETE | /api/legal/cpl/requirements/{id} | Archive requirement |
| POST | /api/legal/cpl/requirements/{id}/compliant | Mark compliant |
| POST | /api/legal/cpl/requirements/{id}/non-compliant | Flag non-compliant |
| GET | /api/legal/cpl/calendar | List calendar entries |
| POST | /api/legal/cpl/calendar | Create calendar entry |
| PUT | /api/legal/cpl/calendar/{id} | Update calendar entry |
| POST | /api/legal/cpl/calendar/{id}/complete | Complete calendar entry |
| DELETE | /api/legal/cpl/calendar/{id} | Cancel calendar entry |
| GET | /api/legal/cpl/evidence | List evidence |
| POST | /api/legal/cpl/evidence | Attach evidence |
| PUT | /api/legal/cpl/evidence/{id} | Update evidence |
| DELETE | /api/legal/cpl/evidence/{id} | Archive evidence |
| GET | /api/legal/cpl/breaches | List breaches |
| GET | /api/legal/cpl/breaches/{id} | Get breach |
| POST | /api/legal/cpl/breaches | Report breach |
| PUT | /api/legal/cpl/breaches/{id} | Update breach |
| DELETE | /api/legal/cpl/breaches/{id} | Close breach |
| POST | /api/legal/cpl/breaches/{id}/remediate | Remediate breach |
| POST | /api/legal/cpl/breaches/{id}/report | Report to regulator |
| GET | /api/legal/cpl/dashboard | Compliance dashboard |
| GET | /api/legal/cpl/risk-register | Risk register |
| GET | /api/legal/cpl/audit | Audit events |

## Service Class

`LegalComplianceService` — multi-regulation tracking (GDPR, AML, POCAMLA, Companies Act), compliance calendar with reminders, evidence collection, breach investigation workflow with regulatory reporting.
