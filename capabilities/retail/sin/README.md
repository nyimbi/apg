# Store Intelligence

## Overview
Provides anonymised in-store analytics: foot traffic counting with multi-sensor support, zone-level dwell time and heatmap generation, AI-assisted planogram compliance auditing, real-time shelf availability alerting with automatic replenishment triggering, shopper conversion funnel tracking, store KPI scorecards with peer-group benchmarking, and a store performance dashboard. All personal data is anonymised at ingest; raw video storage and biometric identification are denied by rule engine.

## Capability ID
`retail_sin`

## Provides
| Service | Description |
|---|---|
| store_foot_traffic_analytics | Anonymised entry/exit counting per zone |
| planogram_compliance_monitoring | AI and manual compliance scoring |
| shelf_availability_alerting | OOS and low-stock alert lifecycle |
| store_conversion_optimisation | Funnel stage conversion rate tracking |
| store_performance_benchmarking | KPI vs peer group, region, and target |
| zone_heatmap_analytics | PII-masked spatial intensity grids |
| store_kpi_reporting | Multi-category KPI snapshot and trend |
| replenishment_triggering | Automated replenishment flag on OOS alert |
| shopper_journey_analytics | Conversion event stitching per session |
| store_format_benchmarking | Cross-format performance comparison |

## Requires
| Capability | Reason |
|---|---|
| auth | Staff and agent authentication |
| audl | Audit trail for data ingest and alerts |
| mten | Tenant isolation per retailer |
| conf | Sensor, zone, and audit configuration |
| ntfy | Critical shelf alert notifications |
| mqeb | Bytewax batch traffic ingest |
| moni | Sensor health monitoring |
| nlpc | Natural language KPI query |
| schd | Scheduled planogram audits and KPI reports |
| geos | Geospatial zone polygon and store location |

## Configuration
| Key | Default | Description |
|---|---|---|
| traffic.counting_interval_seconds | 60 | Sensor aggregation interval |
| traffic.anonymisation_required | true | PII stripped at ingest |
| planogram.audit_frequency_hours | 24 | Default audit schedule |
| planogram.ai_compliance_check_enabled | true | AI image analysis active |
| shelf.oos_replenishment_sla_minutes | 30 | Target replenishment time |
| shelf.low_stock_threshold_pct | 20 | % of par considered low |
| heatmaps.retention_days | 90 | Heatmap data TTL |
| benchmarking.peer_group_min_stores | 5 | Minimum peer group size |

## API Routes
| Path | Method | Description | Permission |
|---|---|---|---|
| /retail-sin/api/v1/stores | GET/POST | List/create stores | retail_sin:view/admin |
| /retail-sin/api/v1/stores/<id> | GET | Store performance summary | retail_sin:view |
| /retail-sin/api/v1/zones | GET/POST | List/create zones | retail_sin:view/admin |
| /retail-sin/api/v1/sensors | GET/POST | List/register sensors | retail_sin:admin |
| /retail-sin/api/v1/sensors/<id>/heartbeat | POST | Sensor heartbeat | retail_sin:write |
| /retail-sin/api/v1/traffic | GET/POST | List/record traffic | retail_sin:view/write |
| /retail-sin/api/v1/traffic/summary | GET | Traffic summary | retail_sin:view |
| /retail-sin/api/v1/planogram | GET/POST | List/record audits | retail_sin:view/write |
| /retail-sin/api/v1/planogram/compliance/<id> | GET | Compliance rate | retail_sin:view |
| /retail-sin/api/v1/shelf-alerts | GET/POST | List/raise alerts | retail_sin:view/write |
| /retail-sin/api/v1/shelf-alerts/<id>/resolve | PUT | Resolve alert | retail_sin:write |
| /retail-sin/api/v1/shelf-alerts/<id>/replenish | POST | Trigger replenishment | retail_sin:write |
| /retail-sin/api/v1/conversion | GET/POST | Funnel/record event | retail_sin:view/write |
| /retail-sin/api/v1/kpis | GET/POST | List/record KPI snapshots | retail_sin:view/write |
| /retail-sin/api/v1/heatmaps | GET/POST | List/create heatmaps | retail_sin:view/write |

## Business Rules
| Rule | Condition | Effect |
|---|---|---|
| pii_anonymisation_required | sensor data not anonymised | deny |
| raw_video_storage_denied | data_type=raw_video | deny |
| biometric_id_denied | biometric_id_present=True | deny |
| heatmap_pii_masking_required | pii_masked=False | deny |
| store_location_required | no location on create | deny |
| store_sqm_required | sqm=0 on create | deny |
| benchmark_min_peer_stores | fewer than 5 peer stores | deny |
| sensor_type_supported | unsupported sensor type | deny |
| batch_traffic_requires_bytewax | batch without Bytewax | deny |

## Data Models
| Model | Key Fields |
|---|---|
| SinStoreResponse | id, store_code, store_format, sqm_total, sqm_selling |
| SinZoneResponse | id, zone_code, zone_type, sqm, polygon_coords |
| SinSensorResponse | id, sensor_type, status, last_heartbeat_at |
| SinTrafficCountResponse | id, entries, exits, occupancy_peak, dwell_avg_seconds |
| SinPlanogramAuditResponse | id, compliance_status, compliance_score_pct |
| SinShelfAlertResponse | id, alert_type, severity, status, replenishment_triggered |
| SinConversionEventResponse | id, conversion_metric, from_stage, to_stage, converted |
| SinKpiSnapshotResponse | id, kpi_category, kpi_values, vs_benchmark_delta |
| SinHeatmapResponse | id, resolution, grid_data, pii_masked |

## Streaming Events
- `traffic_count_recorded`, `zone_dwell_recorded`
- `planogram_audit_completed`, `planogram_deviation_detected`
- `shelf_alert_raised`, `shelf_alert_resolved`
- `oos_replenishment_triggered`
- `conversion_event_recorded`
- `kpi_snapshot_published`, `benchmark_updated`
- `heatmap_generated`

## Edge Cases Handled
- Raw video ingest: hard deny by rule engine, counts only accepted
- Biometric identification in sensor data: hard deny
- Heatmap without PII masking: assertion at service layer
- Peer benchmark with insufficient stores: rule engine denies
- Sensor offline detection: heartbeat gap triggers moni alert
- Planogram audit compliance score mapped to status automatically
- Replenishment flag idempotent: re-triggering safe

## Composability Notes
- **retail_omc** journey stage events feed conversion funnel
- **retail_pos** session close events contribute to conversion analytics
- **retail_prm** promotion placement can be guided by planogram compliance data
- **retail_loy** CLV segments can enrich shopper journey attribution
- Heatmaps and zone dwell data inform planogram and category placement decisions

## World-Class Enhancements (v2.0)

1. **Real-Time Anomaly Detection** — Async streaming z-score anomaly detection on foot-traffic counts; routes `SinTrafficAnomalyEvent` to `ntfy` within seconds.
2. **AI Planogram Deviation Classifier** — Each deviation record carries `deviation_type`, `severity_score`, and `ai_confidence`; compliance score is a weighted sum, not a lookup.
3. **Dwell-Time Cohort Segmentation** — `record_dwell_cohort()` captures histogram buckets (0–30s, 30–120s, 2–5m, 5m+) per zone; unlocks fixture design decisions.
4. **Loss Prevention Incident Lifecycle** — `report_lp_incident()` / `escalate_lp_incident()` / `close_lp_incident()`; OOS + zero sensor activity auto-triggers LP suspicion.
5. **Sensor Network Health Scoring** — `sensor_network_health()` returns overall 0–100 score, heartbeat age, uncovered zones; emits `sensor_network_degraded` via `moni`.
6. **Peer-Group Benchmarking Engine** — `benchmark_peer_group()` selects format+region+sqm peers, returns percentile rank, gap-to-median, gap-to-top-quartile.
7. **Shopper Journey Attribution Graph** — `stitch_shopper_journey()` builds directed path graph from `session_id`; surfaces drop-off rates at each zone transition.
8. **Dynamic Reorder Point Calculation** — Newsvendor-model ROP (`μ_lead × d + Z_α × σ_d × √lead_time`) surfaced as `reorder_point` on `SinShelfAlertResponse`.
9. **Multi-Store Promotional Lift Analysis** — `analyse_promo_lift()` computes traffic/conversion deltas vs. matched pre-period and holdout group; returns p-value + CI.
10. **Temporal KPI Trend Detection** — `detect_kpi_trends()` fits linear regression, returns slope, R², direction, and weeks-to-breach estimate before threshold is crossed.
11. **Occupancy Capacity Compliance** — `check_occupancy_compliance()` validates against `max_capacity × 0.85`; emits `capacity_limit_approaching` at 80% of legal limit.
12. **Heatmap Temporal Diff** — `compute_heatmap_diff()` returns signed intensity delta grid normalised by total traffic; makes layout experiments measurable.
13. **Staff Schedule Demand Forecasting** — `forecast_staffing_demand()` uses 4-week trailing DOW traffic averages and `traffic_to_staff_ratio` to produce a 2-week headcount schedule.
14. **Multi-Sensor Fusion for Entry Counting** — `fuse_sensor_counts()` applies Kalman-style inverse-variance weighting across concurrent sensors; emits fused count with `confidence_interval`.
15. **Privacy-Preserving Export (Differential Privacy)** — `export_records(privacy_budget=ε)` adds Laplace noise to counts/revenue before export; logs epsilon and sensitivity bounds.

## New Methods

### `benchmark_peer_group` — competitive KPI positioning

```python
result = await svc.benchmark_peer_group(
    store_id="store-abc",
    period="2026-05",
    kpi_metric="conversion_rate",
    min_peer_stores=5,
)
# result: {percentile_rank, gap_to_median, gap_to_top_quartile, peer_count}
```

Selects peers by `store_format`. Returns `error: insufficient_peer_stores` if fewer than `min_peer_stores` exist — business rule enforced at the service layer.

### `detect_kpi_trends` — predictive degradation alerting

```python
trend = await svc.detect_kpi_trends(
    store_id="store-abc",
    kpi_metric="footfall",
    n_periods=8,
)
# trend: {slope, r_squared, trend_direction, weeks_to_breach_threshold}
# trend_direction: "improving" | "stable" | "degrading"
```

Fits OLS over the last `n_periods` KPI snapshots. When `trend_direction == "degrading"`, `weeks_to_breach_threshold` gives lead time for area manager intervention.

### `forecast_staffing_demand` — demand-driven scheduling

```python
schedule = await svc.forecast_staffing_demand(
    store_id="store-abc",
    forecast_weeks=2,
    traffic_to_staff_ratio=50.0,
)
# schedule["schedule"]["Mon"]: {avg_daily_traffic, recommended_headcount}
```

Aggregates 4 weeks of historical traffic by ISO weekday. `recommended_headcount = max(1, round(avg_traffic / traffic_to_staff_ratio))`. Replace the hardcoded `staff_count=8` anti-pattern with this.

## Key Service Methods

### Store & Zone Management
| Method | Description |
|---|---|
| `create_store()` / `get_store()` / `list_stores()` | Store CRUD |
| `create_zone()` / `list_zones()` | Zone management |

### Sensors
| Method | Description |
|---|---|
| `register_sensor()` / `sensor_heartbeat()` | Sensor lifecycle |
| `sensor_network_health()` | Network health score (0–100), heartbeat age, uncovered zones |

### Foot Traffic & Occupancy
| Method | Description |
|---|---|
| `foot_traffic_record()` / `record_traffic_count()` | Ingest traffic counts |
| `get_traffic_summary()` / `conversion_rate()` | Traffic aggregation |
| `check_occupancy_compliance()` | Fire-code capacity breach detection and warnings |

### Planogram Compliance
| Method | Description |
|---|---|
| `record_planogram_audit()` / `list_planogram_audits()` | Audit lifecycle |
| `get_store_compliance_rate()` | Weighted average compliance score |
| `planogram_compliance_check()` | Trigger audit for a store/category |

### Shelf Availability & Loss Prevention
| Method | Description |
|---|---|
| `raise_shelf_alert()` / `resolve_shelf_alert()` / `trigger_replenishment()` | Alert lifecycle |
| `shelf_availability_scan()` | Batch OOS scan |
| `report_lp_incident()` / `escalate_lp_incident()` / `close_lp_incident()` | LP incident lifecycle |
| `list_lp_incidents()` | LP incident query |

### Conversion & Shopper Journeys
| Method | Description |
|---|---|
| `record_conversion_event()` / `get_conversion_funnel()` | Funnel tracking |
| `stitch_shopper_journey()` | Ordered zone-path reconstruction per session |

### Analytics, KPIs & Benchmarking
| Method | Description |
|---|---|
| `heat_map_analytics()` / `create_heatmap()` | Heatmap generation |
| `compute_heatmap_diff()` | Signed intensity delta between two heatmap snapshots |
| `record_kpi_snapshot()` / `list_kpi_snapshots()` | KPI time series |
| `detect_kpi_trends()` | Linear trend + R² + weeks-to-breach estimate |
| `benchmark_peer_group()` | Percentile rank vs. format-matched peer group |
| `sales_density()` / `staff_productivity()` / `store_ranking()` | Operational KPIs |
| `forecast_staffing_demand()` | Demand-driven headcount schedule by day-of-week |
| `competitor_price_monitoring()` | Price index tracking |

### Reporting
| Method | Description |
|---|---|
| `store_performance_summary()` | Dashboard aggregate |
| `store_diagnostics_report()` | Full diagnostics: traffic, compliance, alerts, staff, density |
