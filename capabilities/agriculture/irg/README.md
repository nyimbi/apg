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

---

## World-Class Enhancements (v2.0)

Fifteen improvements targeting competitive parity with Trimble Ag, Lindsay Zimmatic, Netafim CONNECT, and Hortau.

**I1. ET₀-Based Evapotranspiration Scheduling (FAO-56)** — FAO-56 Penman-Monteith + crop coefficient (Kc) replaces rule-of-thumb thresholds, reducing water spend 20–35% [AI/ML]

**I2. Multi-Zone Valve Control State Machine** — `ZoneController` entity with idle→opening→running→closing→fault states; mutual-exclusion interlock prevents water hammer on shared laterals [Feature]

**I3. Water Cost Accounting with Tariff Bands** — Progressive volumetric tariff bands (Kenya NIA, DWS) applied to `used_m3`; `compute_water_cost()` returns itemised cost breakdown with currency [Compliance]

**I4. Soil Texture and Water-Holding Capacity Profiles** — `SoilProfile` entity per parcel (texture class, bulk density, field capacity, PWP) used by `optimise_schedule` to derive AWC and translate mm targets to m³ [Feature]

**I5. Weather Forecast Integration for Predictive Rain-Skip** — `schedule_next_event` skips events when forecast precipitation ≥ threshold; skip reason persisted in audit trail [Integration]

**I6. Pressure and Flow Anomaly Detection (Leak/Blockage)** — Z-score control chart on flow_rate vs historical baseline per zone-event pair; flags `leak | blockage | pressure_drop` in `get_sensor_alerts` [AI/ML]

**I7. Fertigation Injection Scheduling** — `FertigationEvent` linked to `IrrigationSchedule`; stores nutrient type, target ppm, injector flow rate, and logs actual nutrient delivered against nutrient budget [Feature]

**I8. Irrigation Uniformity Coefficient (CU) Analysis** — Christiansen CU from catch-can test vectors; classifies excellent/acceptable/poor per FAO-56 benchmarks; stored against zone entity [Feature]

**I9. Regulatory Water Use Reporting (WUA/DWAF Compliance)** — `generate_water_use_declaration()` aggregates `used_m3` by source type and emits a JSON/PDF-ready compliance payload with certification timestamp [Compliance]

**I10. Geospatial Parcel Boundary and Raster Overlay Support** — GeoJSON polygon on sensors and schedules; `get_spatial_coverage()` returns a GIS-ready FeatureCollection with latest readings as properties [Feature]

**I11. Irrigation Programme Library (Crop × Season Templates)** — Built-in `PROGRAMME_LIBRARY` with 12+ crop × season combinations; `apply_programme_template()` generates a full schedule set from agronomic stage tables [UX]

**I12. Real-Time Telemetry Streaming via Async Generator/SSE** — `subscribe_sensor_telemetry()` async generator yields `SensorReadingEvent` objects; API layer wraps in SSE with bounded back-pressure queue [Performance]

**I13. Multi-Tenant Water Rights and Quota Enforcement** — `guard_tenant_id` at every write; `water_rights_register` maps parcels to licenced volumes and expiry; rejects schedules exceeding remaining rights [Security]

**I14. Energy Cost Optimisation for Pump Scheduling** — `TariffPeriod` schedule (peak/off-peak windows, `Decimal` unit cost); `optimise_schedule` aligns `suggested_start` to lowest-cost window and returns `estimated_pump_cost_saved` [AI/ML]

**I15. Carbon and Water Footprint Tracking (ESG Reporting)** — `compute_footprint()` derives water-use intensity (m³/tonne), estimates pump CO₂e from kW × runtime × grid factor; persists `FootprintRecord` for CSV export [Compliance]

---

## New Methods

Three high-impact async methods added in v2.0:

### `optimise_schedule` — Agronomic Schedule Optimisation

```python
svc = IrrigationService(tenant_id="farm-001")

result = await svc.optimise_schedule(
    farm_parcel_id="parcel-42",
    crop_type="maize",
    soil_moisture_pct=38.5,
)
# {"farm_parcel_id": "parcel-42", "recommendation": "irrigate",
#  "suggested_duration_minutes": 78, "deficit_pct": 26.5,
#  "target_moisture_pct": 65}
```

Returns `{"recommendation": "no_irrigation_needed"}` when moisture already meets target — safe to call on every sensor poll cycle.

---

### `get_irrigation_efficiency_report` — Scheduled vs Actual Volume Audit

```python
report = await svc.get_irrigation_efficiency_report(farm_parcel_id="parcel-42")
# {"farm_parcel_id": "parcel-42", "completed_irrigations": 14,
#  "planned_volume_m3": 420.0, "actual_volume_m3": 397.6,
#  "efficiency_pct": 94.7}
```

Aggregates all `status=completed` schedules. `efficiency_pct` is `None` when no completed records exist — guard before display.

---

### `set_water_allocation` — Volumetric Budget Enforcement

```python
account = await svc.set_water_allocation(
    farm_parcel_id="parcel-42",
    period="2026-06",          # YYYY-MM
    allocated_m3=1200.0,
)
# {"id": "wac-...", "allocated_m3": 1200.0, "used_m3": 83.4,
#  "balance_m3": 1116.6, "period": "2026-06", ...}
```

Upserts the account for the period; recalculates `balance_m3` from existing `used_m3`. Emits a `water_account.allocated` audit event on every call.
