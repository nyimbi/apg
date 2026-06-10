# Extension Services (agr_ext)

Agricultural advisory delivery, demo plot management, training records, knowledge base.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/agriculture/ext/health | Health check |
| GET | /api/agriculture/ext/advisories | List advisories |
| POST | /api/agriculture/ext/advisories | Create advisory |
| GET | /api/agriculture/ext/advisories/{id} | Get advisory |
| POST | /api/agriculture/ext/advisories/{id}/follow-up | Mark follow-up done |
| DELETE | /api/agriculture/ext/advisories/{id} | Delete advisory |
| GET | /api/agriculture/ext/demo-plots | List demo plots |
| POST | /api/agriculture/ext/demo-plots | Create demo plot |
| PUT | /api/agriculture/ext/demo-plots/{id} | Update demo plot |
| DELETE | /api/agriculture/ext/demo-plots/{id} | Delete demo plot |
| GET | /api/agriculture/ext/trainings | List trainings |
| POST | /api/agriculture/ext/trainings | Create training |
| PUT | /api/agriculture/ext/trainings/{id} | Update training |
| DELETE | /api/agriculture/ext/trainings/{id} | Delete training |
| GET | /api/agriculture/ext/knowledge | Knowledge articles |
| GET | /api/agriculture/ext/knowledge/search | Search knowledge |
| POST | /api/agriculture/ext/knowledge | Create article |
| GET | /api/agriculture/ext/knowledge/{id} | Get article |
| PUT | /api/agriculture/ext/knowledge/{id} | Update article |
| DELETE | /api/agriculture/ext/knowledge/{id} | Delete article |
| GET | /api/agriculture/ext/summary | Extension summary |
| GET | /api/agriculture/ext/audit | Audit log |
