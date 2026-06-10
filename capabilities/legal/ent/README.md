# leg_ent — Entity & Corporate Secretary

Company registry, board management, statutory filings, annual returns, share register.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/legal/ent/health | Health check |
| GET | /api/legal/ent/entities | List entities |
| GET | /api/legal/ent/entities/{id} | Get entity |
| POST | /api/legal/ent/entities | Register entity |
| PUT | /api/legal/ent/entities/{id} | Update entity |
| DELETE | /api/legal/ent/entities/{id} | Deactivate entity |
| GET | /api/legal/ent/entities/{id}/directors | List directors |
| POST | /api/legal/ent/directors | Appoint director |
| PUT | /api/legal/ent/directors/{id} | Update director |
| DELETE | /api/legal/ent/directors/{id} | Remove director |
| GET | /api/legal/ent/entities/{id}/shareholders | List shareholders |
| POST | /api/legal/ent/shareholders | Register shareholder |
| POST | /api/legal/ent/shareholders/transfer | Transfer shares |
| GET | /api/legal/ent/filings | List filings |
| POST | /api/legal/ent/filings | Schedule filing |
| POST | /api/legal/ent/filings/{id}/complete | Complete filing |
| DELETE | /api/legal/ent/filings/{id} | Cancel filing |
| POST | /api/legal/ent/resolutions | Create board resolution |
| GET | /api/legal/ent/entities/{id}/resolutions | List resolutions |
| GET | /api/legal/ent/dashboard | Corporate dashboard |
| GET | /api/legal/ent/audit | Audit events |

## Service Class

`EntityCorporateSecretaryService` — entity registration, director appointment/removal, share register management, share transfers, statutory filing tracking, board resolutions.
