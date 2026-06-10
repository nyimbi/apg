# AgriIoT & Precision Farming (agr_iot)

Soil sensor ingestion, drone imagery analysis, yield mapping, variable rate prescriptions.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/agriculture/iot/health | Health check |
| GET | /api/agriculture/iot/devices | List devices |
| POST | /api/agriculture/iot/devices | Register device |
| GET | /api/agriculture/iot/devices/{id} | Get device |
| PUT | /api/agriculture/iot/devices/{id} | Update device |
| DELETE | /api/agriculture/iot/devices/{id} | Delete device |
| POST | /api/agriculture/iot/telemetry | Ingest telemetry |
| GET | /api/agriculture/iot/devices/{id}/telemetry | Device readings |
| GET | /api/agriculture/iot/field-health/{parcel_id} | Field snapshot |
| GET | /api/agriculture/iot/imagery | Drone imagery list |
| POST | /api/agriculture/iot/imagery | Upload imagery |
| GET | /api/agriculture/iot/imagery/{id} | Get image |
| DELETE | /api/agriculture/iot/imagery/{id} | Delete image |
| GET | /api/agriculture/iot/ndvi-trend/{parcel_id} | NDVI trend |
| GET | /api/agriculture/iot/yield-maps | Yield maps |
| POST | /api/agriculture/iot/yield-maps | Create yield map |
| DELETE | /api/agriculture/iot/yield-maps/{id} | Delete yield map |
| GET | /api/agriculture/iot/prescriptions | Prescriptions |
| POST | /api/agriculture/iot/prescriptions | Create prescription |
| GET | /api/agriculture/iot/prescriptions/{id} | Get prescription |
| POST | /api/agriculture/iot/prescriptions/{id}/apply | Mark applied |
| POST | /api/agriculture/iot/prescriptions/generate-from-ndvi | Auto-generate |
| DELETE | /api/agriculture/iot/prescriptions/{id} | Delete |
| GET | /api/agriculture/iot/summary/{parcel_id} | Precision summary |
| GET | /api/agriculture/iot/audit | Audit log |
