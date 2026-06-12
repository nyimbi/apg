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

## World-Class Enhancements (v2.0)

- **I1. Real-Time Anomaly Detection** — Rolling Z-score flagging of sensor readings within minutes of failure or disease onset [AI/ML]
- **I2. Soil Nutrient Deficit Scoring** — Composite NPK/pH/EC deficit index (0–100) normalised against crop-specific target ranges [AI/ML]
- **I3. Variable-Rate Irrigation Scheduling** — Zone-level irrigation prescriptions from soil moisture deficit, ET coefficients, and weather forecast [Feature]
- **I4. Multi-Spectral Vegetation Index Suite** — EVI, SAVI, NDRE alongside NDVI from per-band reflectance arrays (Red, NIR, RedEdge, Blue) [Feature]
- **I5. Geospatial Zone Polygon Storage** — GeoJSON geometry field on zone records with shapely bounding-box validation and export endpoint [Feature]
- **I6. Equipment Telematics Integration** — ISOXML TaskData (ISO 11783-10) export for direct upload to John Deere GreenStar / AGCO terminals [Integration]
- **I7. Yield Variance Attribution Analysis** — Pearson correlation of yield against NDVI history, soil readings, and application rates with ranked factor output [AI/ML]
- **I8. Pest & Disease Risk Heatmap** — Degree-day accumulation models for insect/fungal risk per zone with low/moderate/high/critical classification [AI/ML]
- **I9. Drone Flight Planning & Mission Export** — Grid waypoint generation from parcel GeoJSON with GSD, overlap, flight-time, and KML/CSV output [Feature]
- **I10. Carbon Sequestration Estimation** — tCO₂e/ha change from SOC sensor readings using IPCC Tier 2 methodology for carbon credit audit trails [Compliance]
- **I11. Pesticide Application Compliance Log** — EU-compliant spray diary (Directive 2009/128/EC) assembled from prescription events with PHI validation [Compliance]
- **I12. Sensor Calibration Drift Detection** — Cross-sensor divergence monitoring over 7-day rolling windows with `device.calibration_alert` event emission [Performance]
- **I13. Satellite Imagery Fallback** — Sentinel-2/PlanetScope NDVI composites fetched when drone flights are weather-blocked [Integration]
- **I14. Multi-Tenant Data Isolation Audit** — Query-time tenant_id guard on every list/get method with cross-tenant access logging [Security]
- **I15. Agronomic Advisory NL Report** — Plain-language 200-word field advisory with priority actions generated via locally-hosted Ollama LLM [UX]

## New Methods

### `ingest_telemetry` — stream sensor readings with anomaly tagging (I1)

```python
svc = AgriIoTService(tenant_id="farm-001")

# Register a soil moisture sensor first
device = await svc.register_device({
    "name": "SM-North-Field",
    "device_type": "soil_moisture",
    "farm_parcel_id": "parcel-42",
    "location": "North paddock, row 3",
})

# Ingest a reading — anomaly detection fires automatically on the readings dict
record = await svc.ingest_telemetry({
    "device_id": device["id"],
    "readings": {"soil_moisture_pct": 8.2, "soil_temp_c": 24.1},
    "gps_lat": -1.2921,
    "gps_lng": 36.8219,
})
# record["id"] usable for downstream prescription generation
```

### `generate_prescription_from_ndvi` — auto variable-rate prescription from latest imagery (I3)

```python
# Upload NDVI imagery first (zone_analysis populated automatically)
img = await svc.upload_imagery({
    "farm_parcel_id": "parcel-42",
    "imagery_type": "ndvi",
    "ndvi_mean": 0.61,
    "ndvi_min": 0.32,
    "ndvi_max": 0.87,
    "captured_at": "2026-06-12T08:00:00Z",
})

# Generate zone-level fertiliser prescription — stressed zones receive up to 1.5× base rate
rx = await svc.generate_prescription_from_ndvi(
    farm_parcel_id="parcel-42",
    application_type="fertiliser",
    base_rate=150.0,
    unit="kg/ha",
)
# rx["zones"] → list of {zone_id, area_ha, ndvi_mean, application_rate, unit}
```

### `get_precision_farming_summary` — full parcel status rollup (I7)

```python
summary = await svc.get_precision_farming_summary("parcel-42")
# {
#   "farm_parcel_id": "parcel-42",
#   "active_devices": 4,
#   "drone_flights": 7,
#   "latest_ndvi_mean": 0.61,
#   "yield_maps": 2,
#   "prescriptions_created": 5,
#   "prescriptions_applied": 3,
# }
```
