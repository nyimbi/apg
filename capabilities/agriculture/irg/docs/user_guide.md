# Irrigation Management — User Guide

## Overview

agr_irg manages the full irrigation stack: IoT sensor ingestion with threshold alerting,
schedule creation and optimisation, water accounting per parcel/period, and canal infrastructure.

## Key Use Cases

- **Sensor Integration**: Register soil moisture, flow meter, pressure, and weather sensors.
  Ingest readings with automatic threshold alerting when values fall outside configured ranges.
- **Schedule Optimisation**: Get data-driven irrigation recommendations based on crop type
  and current soil moisture deficit.
- **Water Accounting**: Track allocated vs used water volumes per parcel per period.
  Balances auto-update when irrigation schedules are marked completed.
- **Canal Management**: Register distribution canals with capacity and served parcel lists.

## Example Workflows

### Register a Soil Moisture Sensor
```
POST /api/agriculture/irg/sensors
{
  "name": "Block A SM-1",
  "sensor_type": "soil_moisture",
  "farm_parcel_id": "par-001",
  "unit": "pct",
  "min_threshold": 30,
  "max_threshold": 80
}
```

### Ingest a Reading
```
POST /api/agriculture/irg/readings
{"sensor_id": "sen-abc", "value": 25.3, "recorded_at": "2025-04-10T08:00:00Z"}
```

### Get Optimised Schedule
```
GET /api/agriculture/irg/optimise?farm_parcel_id=par-001&crop_type=maize&soil_moisture_pct=28
```

### Set Water Allocation
```
POST /api/agriculture/irg/water-accounts/allocate
{"farm_parcel_id": "par-001", "period": "2025-04", "allocated_m3": 5000}
```
