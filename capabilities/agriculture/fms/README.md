# Farm Management System (agr_fms)

Parcel registry, input recording, labour scheduling, cost tracking, farm diary.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/agriculture/fms/health | Health check |
| GET | /api/agriculture/fms/parcels | List parcels |
| POST | /api/agriculture/fms/parcels | Create parcel |
| GET | /api/agriculture/fms/parcels/{id} | Get parcel |
| PUT | /api/agriculture/fms/parcels/{id} | Update parcel |
| DELETE | /api/agriculture/fms/parcels/{id} | Delete parcel |
| GET | /api/agriculture/fms/parcels/{id}/summary | Cost summary |
| GET | /api/agriculture/fms/inputs | List input records |
| POST | /api/agriculture/fms/inputs | Record input usage |
| DELETE | /api/agriculture/fms/inputs/{id} | Delete input |
| GET | /api/agriculture/fms/labour | List labour schedules |
| POST | /api/agriculture/fms/labour | Create schedule |
| PUT | /api/agriculture/fms/labour/{id} | Update schedule |
| DELETE | /api/agriculture/fms/labour/{id} | Delete schedule |
| GET | /api/agriculture/fms/diary | List diary entries |
| POST | /api/agriculture/fms/diary | Create entry |
| PUT | /api/agriculture/fms/diary/{id} | Update entry |
| DELETE | /api/agriculture/fms/diary/{id} | Delete entry |
| GET | /api/agriculture/fms/costs | Cost summary |
| GET | /api/agriculture/fms/audit | Audit log |
