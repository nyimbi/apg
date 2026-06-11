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
