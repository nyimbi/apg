# Crop Management (agr_crp)

Planting calendar, phenology tracking, variety registry, crop rotation planning, yield recording.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/agriculture/crp/health | Health check |
| GET | /api/agriculture/crp/varieties | List varieties |
| POST | /api/agriculture/crp/varieties | Register variety |
| GET | /api/agriculture/crp/varieties/{id} | Get variety |
| PUT | /api/agriculture/crp/varieties/{id} | Update variety |
| DELETE | /api/agriculture/crp/varieties/{id} | Delete variety |
| GET | /api/agriculture/crp/calendars | List planting calendars |
| POST | /api/agriculture/crp/calendars | Create calendar |
| GET | /api/agriculture/crp/calendars/recommend | Recommend planting window |
| GET | /api/agriculture/crp/crops | List crops |
| POST | /api/agriculture/crp/crops | Create crop record |
| GET | /api/agriculture/crp/crops/{id} | Get crop |
| PUT | /api/agriculture/crp/crops/{id} | Update crop |
| DELETE | /api/agriculture/crp/crops/{id} | Delete crop |
| GET | /api/agriculture/crp/crops/{id}/phenology | Phenology observations |
| POST | /api/agriculture/crp/phenology | Record observation |
| GET | /api/agriculture/crp/rotation-plans | List rotation plans |
| POST | /api/agriculture/crp/rotation-plans | Create rotation plan |
| GET | /api/agriculture/crp/yields | List yield records |
| POST | /api/agriculture/crp/yields | Record yield |
| GET | /api/agriculture/crp/audit | Audit log |
