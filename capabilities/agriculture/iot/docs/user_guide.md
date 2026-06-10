# AgriIoT & Precision Farming — User Guide

## Overview

agr_iot is the precision agriculture layer: it ingests IoT telemetry from soil sensors and
weather stations, processes drone imagery for NDVI zone analysis, records spatial yield maps,
and generates variable-rate application prescriptions.

## Key Use Cases

- **IoT Device Management**: Register sensors, drones, yield monitors. Every device has a
  farm parcel association and firmware/calibration tracking.
- **Telemetry Ingestion**: Push raw sensor readings (soil moisture, temperature, EC, pH)
  as key-value maps. Device last-seen timestamps auto-update.
- **Drone Imagery**: Register flight imagery with NDVI statistics. Zone analysis is
  auto-generated from aggregate NDVI values if per-zone data is not supplied.
- **NDVI Trend Analysis**: Track vegetation health across multiple flights — trend classified
  as improving / stable / declining.
- **Yield Mapping**: Store spatial yield data from combine harvesters as zone arrays with
  per-zone yield_kg and area_ha.
- **Variable-Rate Prescriptions**: Auto-generate fertiliser/chemical prescriptions from NDVI:
  stressed zones (low NDVI) receive higher application rates.

## Example Workflows

### Register a Soil Sensor
```
POST /api/agriculture/iot/devices
{
  "name": "Block A Soil Sensor",
  "device_type": "soil_sensor",
  "farm_parcel_id": "par-001",
  "location_lat": -0.4167,
  "location_lng": 36.9500,
  "serial_number": "SS-2025-001"
}
```

### Ingest Telemetry
```
POST /api/agriculture/iot/telemetry
{
  "device_id": "dev-abc",
  "readings": {"soil_moisture_pct": 32.5, "temperature_c": 22.1, "ec_mS_cm": 0.8},
  "recorded_at": "2025-04-15T06:00:00Z"
}
```

### Upload NDVI Imagery
```
POST /api/agriculture/iot/imagery
{
  "farm_parcel_id": "par-001",
  "imagery_type": "ndvi",
  "captured_at": "2025-04-10T09:00:00Z",
  "file_url": "https://storage.example.com/flights/flight-001.tif",
  "ndvi_mean": 0.45,
  "ndvi_min": 0.20,
  "ndvi_max": 0.72,
  "coverage_ha": 5.0
}
```

### Auto-Generate Fertiliser Prescription
```
POST /api/agriculture/iot/prescriptions/generate-from-ndvi
{
  "farm_parcel_id": "par-001",
  "application_type": "nitrogen_fertiliser",
  "base_rate": 100,
  "unit": "kg/ha"
}
```
