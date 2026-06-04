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
