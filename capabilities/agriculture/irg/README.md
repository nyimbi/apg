# Irrigation Management (agr_irg)

Sensor integration, irrigation schedule optimisation, water accounting, canal management.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/agriculture/irg/health | Health check |
| GET | /api/agriculture/irg/sensors | List sensors |
| POST | /api/agriculture/irg/sensors | Register sensor |
| PUT | /api/agriculture/irg/sensors/{id} | Update sensor |
| DELETE | /api/agriculture/irg/sensors/{id} | Delete sensor |
| POST | /api/agriculture/irg/readings | Ingest reading |
| GET | /api/agriculture/irg/sensors/{id}/readings | Sensor readings |
| GET | /api/agriculture/irg/alerts | Threshold alerts |
| GET | /api/agriculture/irg/schedules | List schedules |
| POST | /api/agriculture/irg/schedules | Create schedule |
| PUT | /api/agriculture/irg/schedules/{id} | Update schedule |
| DELETE | /api/agriculture/irg/schedules/{id} | Delete schedule |
| GET | /api/agriculture/irg/optimise | Optimisation recommendation |
| GET | /api/agriculture/irg/water-accounts | Water accounts |
| POST | /api/agriculture/irg/water-accounts/allocate | Set allocation |
| GET | /api/agriculture/irg/canals | List canals |
| POST | /api/agriculture/irg/canals | Create canal |
| GET | /api/agriculture/irg/audit | Audit log |
