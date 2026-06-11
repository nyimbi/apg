# Irrigation Management — World-Class Improvements

Fifteen targeted improvements that move agr_irg from functional baseline to competitive parity
with Trimble Ag, Lindsay Zimmatic, Netafim CONNECT, and Hortau.

---

### I1. ET₀-Based Evapotranspiration Scheduling (FAO-56)
**Category**: AI/ML
**Justification**: Rule-of-thumb moisture thresholds waste 20–35% water. Computing reference
evapotranspiration (FAO-56 Penman-Monteith) combined with crop coefficient (Kc) tables delivers
scientifically-grounded timing that reduces water spend while improving yield — the core value
proposition of Lindsay FieldNET Advisor.
**Implementation**: Accept daily weather parameters (Tmax, Tmin, RH, wind speed, solar radiation);
compute ET₀ in mm/day; multiply by stage-specific Kc to derive net irrigation demand; express as
m³ for any parcel given known area and soil water-holding capacity.
**Competitive reference**: Lindsay FieldNET Advisor, Trimble Ag Water Manager

---

### I2. Multi-Zone Valve Control State Machine
**Category**: Feature
**Justification**: Smart controllers (Rachio, Rain Bird IQ) expose per-zone valve state with
interlock logic so two adjacent zones never fire simultaneously on the same lateral. Without this,
water hammer and pressure collapse occur on smaller-bore networks — a common failure on
smallholder schemes.
**Implementation**: Add a `ZoneController` entity with a state machine (idle → opening → running →
closing → fault); enforce mutual-exclusion rules in `execute_schedule`; surface zone fault recovery
workflows with audit events.
**Competitive reference**: Rachio Pro Zone Management, Rain Bird IQ4

---

### I3. Water Cost Accounting with Tariff Bands
**Category**: Compliance
**Justification**: Agricultural water authorities (Kenya NIA, South Africa DWS) apply volumetric
tariff bands — first N m³ at one rate, excess at penalty rate. Operators need real-time cost
forecasts against annual water budgets; Hortau and CropX show cost-per-hectare as a primary KPI.
**Implementation**: Store tariff schedules (band breakpoints + unit cost as `Decimal`) per water
authority; `compute_water_cost(farm_parcel_id, period)` applies progressive bands to `used_m3` and
returns itemised cost breakdown with currency.
**Competitive reference**: Hortau Stress Index Dashboard, CropX WaterIQ

---

### I4. Soil Texture and Water-Holding Capacity Profiles
**Category**: Feature
**Justification**: The same 30-minute event saturates sandy loam and barely wets heavy clay
differently. Without soil texture metadata the schedule optimiser cannot compute infiltration rates
or root-zone storage depth — leading to run-off and leaching losses.
**Implementation**: Add `SoilProfile` entity keyed to `farm_parcel_id`; store texture class, bulk
density, field capacity (% vol), permanent wilting point; use these in `optimise_schedule` to
compute available water capacity (AWC) and translate mm targets to m³.
**Competitive reference**: Trimble Ag Field360, Arable Mark sensor platform

---

### I5. Weather Forecast Integration for Predictive Rain-Skip Scheduling
**Category**: Integration
**Justification**: Rain cancellations are the single largest source of irrigation waste. Netafim
CONNECT and CropX skip scheduled events when forecast precipitation exceeds a user-defined
threshold, reducing water bills and preventing waterlogging.
**Implementation**: Accept a weather forecast payload (or call an internal weather service
capability) with 24-hour precipitation probability and amount; in `schedule_next_event` return
`skip_reason: precipitation_forecast` when forecast rain ≥ threshold; persist skip in audit trail.
**Competitive reference**: Netafim CONNECT Rain Delay, Trimble Crop Health

---

### I6. Pressure and Flow Anomaly Detection (Leak / Blockage)
**Category**: AI/ML
**Justification**: Subsurface drip leaks waste 10–40% of delivered water before physical discovery.
Trimble and Hortau use statistical control charts on pressure and flow sensor pairs to detect leaks
and emitter blockages within hours.
**Implementation**: For each `(zone, irrigation_event)` pair, compute z-score of observed flow_rate
vs historical baseline; flag the event `anomaly_type: leak | blockage | pressure_drop` when
|z| > configurable threshold; surface in `get_sensor_alerts` with severity classification.
**Competitive reference**: Hortau Canopy Pressure Monitoring, Trimble WaterIQ Leak Detection

---

### I7. Fertigation Injection Scheduling
**Category**: Feature
**Justification**: Nutrient delivery through irrigation is the dominant fertiliser application
method for drip/sprinkler systems. Without fertigation scheduling, operators use separate
paper-based records that create compliance gaps and over-application risk.
**Implementation**: Add `FertigationEvent` linked to an `IrrigationSchedule`; store nutrient type,
target concentration (ppm), injector flow rate, and injection window; compute actual nutrient
delivered and log against nutrient budget.
**Competitive reference**: Netafim CONNECT Fertigation Module, Cropwise Nutrients (Syngenta)

---

### I8. Irrigation Uniformity Coefficient (CU) Analysis
**Category**: Feature
**Justification**: Distribution uniformity (DU/CU) is the agronomic audit metric required by
irrigation water management regulators in AU, ZA, and the US; systems that cannot compute CU from
catch-can test data cannot help operators justify water licence renewals.
**Implementation**: Accept a list of catch-can volumes from a uniformity test for a given zone;
compute Christiansen Uniformity Coefficient (CU = 100 × [1 − σ/μ]); classify as
excellent/acceptable/poor against FAO-56 benchmarks; store result against the zone entity.
**Competitive reference**: Irrigear CU Calculator, Lindsay FieldNET Advisor Audits

---

### I9. Regulatory Water Use Reporting (WUA / DWAF Compliance)
**Category**: Compliance
**Justification**: Water User Associations in East Africa and Southern Africa require monthly water
use declarations. Producing these manually from irrigation logs creates compliance risk. Automated
generation is table stakes for enterprise agribusiness buyers.
**Implementation**: `generate_water_use_declaration(farm_parcel_id, period)` aggregates `used_m3`
by source type, applies the authority's reporting template, and emits a structured JSON/PDF-ready
payload with certification timestamp.
**Competitive reference**: Gallagher Water Compliance Suite, Hortau Regulatory Reports

---

### I10. Geospatial Parcel Boundary and Raster Overlay Support
**Category**: Feature
**Justification**: Modern precision irrigation platforms (Climate FieldView, John Deere Ops Center)
overlay irrigation data on parcel maps so agronomists can correlate dry patches with sensor
clusters. Without spatial context, integration with GIS workflows is impossible.
**Implementation**: Accept GeoJSON polygon on `create_sensor` / `create_schedule`; persist
`geometry` field; expose `get_spatial_coverage(farm_parcel_id)` that returns a FeatureCollection
of sensors and latest readings as feature properties.
**Competitive reference**: John Deere Operations Center Field Mapping, Climate FieldView

---

### I11. Irrigation Programme Library (Crop × Season Templates)
**Category**: UX
**Justification**: Agronomists spend 4–6 hours per season manually configuring irrigation
programmes for standard crops. A curated template library (maize × rainy season, tomato × dry
season) reduces onboarding time and encoding errors for extension officers managing multiple
smallholder clients.
**Implementation**: Ship a built-in `PROGRAMME_LIBRARY` dict with 12+ crop × season combinations,
each containing stage-by-stage water requirements, trigger moisture thresholds, and recommended
method; `apply_programme_template` generates the full schedule set.
**Competitive reference**: Netafim IrriWay Crop Library, Trimble Ag Prescription Templates

---

### I12. Real-Time Telemetry Streaming via Async Generator / SSE
**Category**: Performance
**Justification**: Polling-based sensor dashboards introduce 30–120 s latency on cellular
connections. Hortau and Arable push real-time telemetry over WebSocket, enabling operators to
respond to frost or heat events within seconds, not minutes.
**Implementation**: `subscribe_sensor_telemetry(farm_parcel_id)` as an async generator that yields
`SensorReadingEvent` objects; API layer wraps in SSE; back-pressure handled with a bounded async
queue per subscription.
**Competitive reference**: Hortau Live Sensor Feed, Arable Real-Time Dashboard

---

### I13. Multi-Tenant Water Rights and Quota Enforcement
**Category**: Security
**Justification**: Water rights are legally binding; allowing one tenant's schedules to draw against
another tenant's quota is a liability. Hard enforcement at the service layer — not just UI — is
required for ISO 27001 and regulatory audit trails.
**Implementation**: Verify `farm_parcel_id` ownership via `guard_tenant_id` at every write; persist
a `water_rights_register` mapping parcels to licenced volumes, sources, and expiry dates; reject
schedule creation when cumulative planned volume exceeds remaining rights.
**Competitive reference**: Hortau Multi-Tenancy Model, Trimble Ag Enterprise Permissions

---

### I14. Energy Cost Optimisation for Pump Scheduling
**Category**: AI/ML
**Justification**: Pumping accounts for 60–80% of irrigation operating cost in sub-Saharan Africa.
Time-of-use electricity tariffs (Eskom, Kenya Power) create a 3–4× price differential between
peak and off-peak windows; aligning irrigation with off-peak is the fastest ROI improvement
available.
**Implementation**: Accept a `TariffPeriod` schedule (peak/off-peak windows and unit cost as
`Decimal`); `optimise_schedule` returns a `suggested_start` aligned with the lowest-cost window
within the agronomic window; compute and return `estimated_pump_cost_saved`.
**Competitive reference**: Trimble Energy Manager, CropX Smart Pump Scheduler

---

### I15. Carbon and Water Footprint Tracking (ESG Reporting)
**Category**: Compliance
**Justification**: Export-market buyers (EU, UK) increasingly require Scope 3 water and carbon
footprint data. Certifications like SAI Platform FSA require water-use intensity per tonne of
produce; ESG dashboards are an emerging procurement requirement.
**Implementation**: `compute_footprint(farm_parcel_id, harvest_kg)` derives water-use intensity
(m³/tonne), estimates pump energy from motor kW × runtime, and CO₂e from grid emission factor;
persists `FootprintRecord` with vintage, crop, and scope for CSV export.
**Competitive reference**: SAI Platform FSA Module, Agoro Carbon Alliance, Trimble Ag Sustainability
