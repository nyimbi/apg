# Fleet Management — API Reference

Base URL: `/api/fle/v1`

**Auth headers** (required on every request):
```
X-Tenant-ID: <tenant_id>
X-Actor-ID:  <user_id>
```

---

## Vehicles

| Method | Path | Description |
|--------|------|-------------|
| GET | `/vehicles` | List vehicles. Query: `status`, `page`, `per_page` |
| POST | `/vehicles` | Register vehicle |
| GET | `/vehicles/<id>` | Get vehicle detail |
| PUT | `/vehicles/<id>` | Partial update |
| DELETE | `/vehicles/<id>` | Soft delete |
| POST | `/vehicles/<id>/status` | Set status. Body: `{"status": "active"}` |

## Drivers

| Method | Path | Description |
|--------|------|-------------|
| GET | `/drivers` | List drivers. Query: `status` |
| POST | `/drivers` | Register driver |
| GET | `/drivers/<id>` | Get driver detail |
| PUT | `/drivers/<id>` | Partial update |
| DELETE | `/drivers/<id>` | Soft delete |
| GET | `/drivers/<id>/score` | Get behaviour score |

## Assignments

| Method | Path | Description |
|--------|------|-------------|
| POST | `/assignments` | Assign driver to vehicle |

## Trips

| Method | Path | Description |
|--------|------|-------------|
| GET | `/trips` | List trips. Query: `status`, `vehicle_id`, `driver_id` |
| POST | `/trips` | Plan trip |
| GET | `/trips/<id>` | Get trip |
| PUT | `/trips/<id>` | Update trip |
| POST | `/trips/<id>/dispatch` | Dispatch |
| POST | `/trips/<id>/start` | Start. Body: `{"odometer_start_km": 50000}` |
| POST | `/trips/<id>/complete` | Complete. Body: `{"odometer_end_km": 50450, "fuel_consumed_l": 80.5}` |
| POST | `/trips/<id>/cancel` | Cancel. Body: `{"reason": "..."}` |
| POST | `/trips/<id>/breakdown` | Record breakdown |
| POST | `/trips/<id>/change-driver` | Change driver. Body: `{"new_driver_id": "...", "reason": "..."}` |

## Fuel

| Method | Path | Description |
|--------|------|-------------|
| GET | `/fuel` | List records. Query: `vehicle_id` |
| POST | `/fuel` | Record fuel purchase |

## Maintenance

| Method | Path | Description |
|--------|------|-------------|
| GET | `/maintenance` | List. Query: `vehicle_id`, `status` |
| POST | `/maintenance` | Schedule |
| POST | `/maintenance/<id>/start` | Start |
| POST | `/maintenance/<id>/complete` | Complete. Body: `{"actual_cost": 9500}` |

## Inspections

| Method | Path | Description |
|--------|------|-------------|
| GET | `/inspections` | List. Query: `vehicle_id` |
| POST | `/inspections` | Record inspection |
| POST | `/inspections/<id>/process-failure` | Process failed inspection |

## COF Inspections

| Method | Path | Description |
|--------|------|-------------|
| GET | `/cof` | List. Query: `vehicle_id` |
| POST | `/cof` | Record COF inspection |

## Incidents

| Method | Path | Description |
|--------|------|-------------|
| GET | `/incidents` | List. Query: `vehicle_id`, `status` |
| POST | `/incidents` | Report incident |
| POST | `/incidents/<id>/close` | Close. Body: `{"resolution": "..."}` |

## Insurance

| Method | Path | Description |
|--------|------|-------------|
| GET | `/insurance` | List. Query: `vehicle_id` |
| POST | `/insurance` | Add policy |

## Registrations

| Method | Path | Description |
|--------|------|-------------|
| GET | `/registrations` | List. Query: `vehicle_id` |
| POST | `/registrations` | Add registration |

## Tachograph

| Method | Path | Description |
|--------|------|-------------|
| GET | `/tachograph` | List. Query: `driver_id` |
| POST | `/tachograph` | Record tachograph data |

## Telematics

| Method | Path | Description |
|--------|------|-------------|
| GET | `/telematics` | List events. Query: `vehicle_id`, `event_type` |
| POST | `/telematics` | Ingest telematics event |
| GET | `/telematics/position/<vehicle_id>` | Get last position |

## Reports

| Method | Path | Description |
|--------|------|-------------|
| GET | `/reports/tco/<vehicle_id>` | TCO breakdown |
| GET | `/reports/utilisation` | Fleet utilisation analytics |
| GET | `/reports/compliance-calendar` | All compliance events |
| GET | `/reports/predictive-maintenance` | Predictive maintenance alerts |
| GET | `/reports/driver-score/<driver_id>` | Driver behaviour score |

## Dashboard & Health

| Method | Path | Description |
|--------|------|-------------|
| GET | `/dashboard` | Fleet KPIs |
| GET | `/health` | Health check |
