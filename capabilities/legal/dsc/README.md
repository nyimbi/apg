# leg_dsc — Document & eDiscovery

Document repository, version control, privilege logging, litigation hold, eDiscovery production.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/legal/dsc/health | Health check |
| GET | /api/legal/dsc/documents | List documents |
| GET | /api/legal/dsc/documents/{id} | Get document |
| POST | /api/legal/dsc/documents | Upload document |
| PUT | /api/legal/dsc/documents/{id} | Update document |
| DELETE | /api/legal/dsc/documents/{id} | Archive document |
| GET | /api/legal/dsc/documents/search | Search documents |
| POST | /api/legal/dsc/privilege-log | Log privilege |
| GET | /api/legal/dsc/privilege-log | List privilege log |
| GET | /api/legal/dsc/holds | List litigation holds |
| GET | /api/legal/dsc/holds/{id} | Get hold |
| POST | /api/legal/dsc/holds | Issue litigation hold |
| POST | /api/legal/dsc/holds/{id}/release | Release hold |
| DELETE | /api/legal/dsc/holds/{id} | Delete hold |
| GET | /api/legal/dsc/productions | List production sets |
| POST | /api/legal/dsc/productions | Create production set |
| POST | /api/legal/dsc/productions/{id}/finalize | Finalize production |
| GET | /api/legal/dsc/stats | Repository statistics |
| GET | /api/legal/dsc/audit | Audit events |

## Service Class

`DocumentEDiscoveryService` — document ingestion, version control, attorney-client privilege logging, auto-litigation hold application, eDiscovery production set creation with Bates numbering.
