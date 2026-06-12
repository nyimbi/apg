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

## World-Class Enhancements (v2.0)

15 improvements elevating agr_crp from basic CRUD to a decision-grade precision agriculture platform.

- **I1. GDD Accumulation Engine** — Accumulates Growing Degree Days from daily Tmax/Tmin against crop-specific base temp; predicts days to next phenological stage. [AI/ML]
- **I2. Yield Gap Analysis** — Computes actual vs. attainable yield, returns gap_kg/gap_pct and ranked limiting factors (fertiliser, water, seed density). [AI/ML]
- **I3. Crop Water Stress Index (CWSI) Monitoring** — Records CWSI events per sensor; returns time-series with alert flags at moderate (>0.5) and severe (>0.8) thresholds. [Feature]
- **I4. Input Cost Ledger with ROI** — Tracks per-crop input applications (seed, fertiliser, labour); computes gross_revenue, net_margin, and roi_pct using Decimal precision. [Feature]
- **I5. Seed Lot Traceability Chain** — Registers seed lots with cert hash; walks crop → seed_lot → supplier → certification for full provenance. [Compliance]
- **I6. Multi-Season Yield Trend Analysis** — Aggregates 3–5 season yield/cost/ROI per parcel; returns yoy_delta_pct and regression slope for trajectory analysis. [AI/ML]
- **I7. Pest and Disease Pressure Alerts** — Records pest observations with GPS and severity; aggregates active alerts and emits `pest.alert` CloudEvents on escalation. [Feature]
- **I8. Agronomic Advisory Rules Engine** — Evaluates growth stage + GDD + pest state; emits ranked `{action, urgency, rationale, deadline}` recommendations from a deterministic rules table. [AI/ML]
- **I9. Soil Health Score Integration** — Attaches OC%, pH, and bulk density per parcel; adjusts variety yield potential by a correction factor derived from soil deviation from optimum. [Integration]
- **I10. Carbon Sequestration Estimation** — Applies IPCC Tier 1 factors per crop in a rotation sequence; returns estimated CO2e/ha and creditable practices for VCM applications. [Compliance]
- **I11. Weather-Adjusted Planting Window Scoring** — Scores planting readiness 0–100 from accumulated rainfall vs. calendar threshold; returns go/no-go and days_until_optimal. [Feature]
- **I12. Harvest Logistics Scheduling** — Creates harvest slots with equipment/date constraints; detects equipment conflicts across crops in the same season. [Feature]
- **I13. Variety Performance Leaderboard** — Aggregates yield records by variety per region/season; ranks by mean yield_kg_ha with confidence intervals for evidence-based selection. [Feature]
- **I14. Compliance Export: Season KPI Reporting** — Marshals crops + yields + inputs + rotation into JSON/CSV with SHA-256 checksum; emits `report.generated` audit event. [Compliance]
- **I15. Inter-Capability Event Bus Integration** — Publishes CloudEvents 1.0 to async queue with `source=agr_crp/{tenant_id}`; downstream capabilities subscribe by `agr_crp.*` prefix. [Integration]

## New Methods

Three high-impact async methods added in v2.0:

### `get_gdd_accumulation(crop_id, as_of)`

Returns accumulated Growing Degree Days and predicted days to the next phenological stage.

```python
svc = CropManagementService(tenant_id="ke_001")
result = await svc.get_gdd_accumulation(crop_id="crp_abc123", as_of="2026-06-01")
# {
#   "crop_id": "crp_abc123",
#   "accumulated_gdd": 412.5,
#   "base_temp_c": 10.0,
#   "current_stage": "flowering",
#   "next_stage": "grain_fill",
#   "predicted_days_to_next_stage": 8
# }
```

### `analyze_yield_gap(crop_id)`

Computes attainable vs. actual yield and returns a ranked list of limiting factors.

```python
result = await svc.analyze_yield_gap(crop_id="crp_abc123")
# {
#   "crop_id": "crp_abc123",
#   "attainable_yield_kg": 4800.0,
#   "actual_yield_kg": 3120.0,
#   "gap_kg": 1680.0,
#   "gap_pct": 35.0,
#   "limiting_factors": ["water_deficit", "low_seed_density", "suboptimal_n_application"]
# }
```

### `publish_crop_event(event_type, entity_id, payload)`

Serialises a CloudEvents 1.0 envelope to the async event queue for downstream capabilities.

```python
await svc.publish_crop_event(
    event_type="agr_crp.pest.alert",
    entity_id="crp_abc123",
    payload={"pest_type": "fall_armyworm", "severity": "high", "location_gps": "-1.28,36.82"},
)
# Emits to queue with source="agr_crp/ke_001", datacontenttype="application/json"
# Consumed by: agr_irr, agr_frt, agr_mkt, fin_ins, agr_soil
```
