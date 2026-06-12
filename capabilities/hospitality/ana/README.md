# Hospitality Analytics (hos_ana)

RevPAR, ADR, occupancy, GOP PAR, segment analysis, pace reporting, and guest satisfaction intelligence.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/hospitality/ana/health | Health check |
| GET | /api/hospitality/ana/kpi-snapshots | List KPI snapshots |
| POST | /api/hospitality/ana/kpi-snapshots | Record KPI snapshot |
| GET | /api/hospitality/ana/kpi-snapshots/{id} | Get snapshot |
| PUT | /api/hospitality/ana/kpi-snapshots/{id} | Update snapshot |
| DELETE | /api/hospitality/ana/kpi-snapshots/{id} | Delete snapshot |
| GET | /api/hospitality/ana/kpi-summary | Period KPI summary |
| GET | /api/hospitality/ana/segment-reports | List segment reports |
| POST | /api/hospitality/ana/segment-reports | Record segment data |
| GET | /api/hospitality/ana/segment-mix | Segment mix report |
| GET | /api/hospitality/ana/pace-reports | List pace reports |
| POST | /api/hospitality/ana/pace-reports | Record pace data |
| GET | /api/hospitality/ana/pace-comparison | Pace comparison |
| GET | /api/hospitality/ana/satisfaction-surveys | List surveys |
| POST | /api/hospitality/ana/satisfaction-surveys | Record survey |
| GET | /api/hospitality/ana/satisfaction-summary | NPS & satisfaction KPIs |
| POST | /api/hospitality/ana/benchmarks | Record benchmark |
| GET | /api/hospitality/ana/benchmarks | List benchmarks |
| POST | /api/hospitality/ana/competitive-sets | Create comp set |
| POST | /api/hospitality/ana/channel-revenue | Record channel revenue |
| GET | /api/hospitality/ana/channel-mix | Channel mix report |
| GET | /api/hospitality/ana/executive-dashboard | Executive dashboard |
| GET | /api/hospitality/ana/dashboard | Summary dashboard |

## World-Class Enhancements (v2.0)

15 improvements that elevate `hos_ana` from a KPI store to a full revenue intelligence platform.

**I1. Revenue Strategy Displacement Forecasting** — ranks open segments by marginal revenue contribution to identify which business to displace before compression events [AI/ML]

**I2. Unconstrained Demand Estimation** — deconvolves censored sell-out data to expose true demand, demand capture rate, and lost revenue estimate [AI/ML]

**I3. Rate Parity Monitoring & Leakage Detection** — computes per-channel parity delta, flags violations by severity tier, and tracks a 0–100 parity score per period [Compliance]

**I4. Dynamic Forecast with Pickup Velocity** — computes 7-day rolling pickup velocity, compares against same-date-LY curve, emits `pace_signal` enum and projected final occupancy [Feature]

**I5. GOPPAR Cost-Side Decomposition** — accepts departmental expense records and computes flow-through ratio, departmental margin, and triggers margin compression alerts [Feature]

**I6. Sentiment-Driven Reputation Score** — rule-based NLP over free-text survey comments produces topic-level sentiment scores and a 0–100 `reputation_index` [AI/ML]

**I7. Length-of-Stay Optimisation Reporting** — computes LOS distribution histograms and `shoulder_night_waste_pct`, flagging restriction opportunities above 8% waste [Feature]

**I8. Volatility-Adjusted RevPAR (vRevPAR)** — applies coefficient-of-variation to rolling RevPAR to produce `v_revpar` and a `demand_stability_tier` for asset managers [Feature]

**I9. Booking Window & Lead-Time Analytics** — segments reservations into 0–7d / 8–30d / 31–90d / 90d+ buckets and emits `window_compression_alert` on >5pp MoM shift [Feature]

**I10. Geo-Demand Heatmap (Origin Market Analytics)** — aggregates origin country/region/city mix, ADR, and LOS by feeder market and flags anomalous shifts vs trailing 90 days [Feature]

**I11. Event & Compression Calendar Overlay** — maintains an event registry and annotates pace comparisons with `events[]` and demand-adjusted pace signal baselines [Integration]

**I12. Multi-Property Portfolio Rollup** — fans out KPI and dashboard queries across `property_ids`, returns weighted portfolio summary alongside per-property breakdown [Feature]

**I13. Anomaly Detection & Automated KPI Alerts** — z-scores RevPAR against trailing 28-day distribution and appends `AnomalyAlert` records with severity and suggested action [AI/ML]

**I14. Carbon & Sustainability KPI Tracking (ESG)** — accepts energy/water/waste/carbon per occupied-room-night, benchmarks against AHLA norms, emits `esg_score` [Compliance]

**I15. TRevPAR Cross-Department Attribution** — aggregates F&B, spa, parking, and events revenue into `trevpar`, computes ancillary per occupied room, and models the rooms-discount crossover point [Feature]

## New Methods

Three highest-impact async methods added in v2.0:

### `record_kpi_snapshot` — KPI ingestion with derived metrics

Records a daily snapshot and auto-computes occupancy rate, ADR, RevPAR, TRevPAR, and ancillary per occupied room. Emits a `kpi_snapshot_recorded` audit event.

```python
snap = await svc.record_kpi_snapshot(
    date="2026-06-01",
    total_rooms=200,
    occupied_rooms=172,
    total_revenue=68_400.0,
    room_revenue=58_000.0,
    ancillary_revenue=10_400.0,
    goppar=142.50,
    tenant_id="prop_001",
)
# snap["revpar"]  -> 290.0
# snap["total_revpar"] -> 342.0
# snap["ancillary_per_occupied_room"] -> 60.47
```

### `pace_comparison` — booking pace vs historical baseline

Returns the latest OTB position for a future date alongside the pickup velocity trajectory and `vs_same_time_last_year_pct`.

```python
pace = await svc.pace_comparison(
    future_date="2026-07-04",
    days_out=30,
    tenant_id="prop_001",
)
# pace["latest_booked_rooms"] -> 118
# pace["pickup_last_7_days"]  -> 14
# pace["vs_same_time_last_year_pct"] -> 8.3
```

### `executive_dashboard` — period roll-up with NPS

Aggregates all KPI snapshots in the window into average occupancy, ADR, RevPAR, total revenue, and NPS score computed from promoter/detractor ratio.

```python
dash = await svc.executive_dashboard(
    date_from="2026-06-01",
    date_to="2026-06-30",
    tenant_id="prop_001",
)
# dash["avg_revpar"]      -> 267.40
# dash["nps_score"]       -> 54.0
# dash["total_revenue"]   -> 1_846_200.0
```
