# Asset Tracking

## Overview
The Asset Tracking capability provides real-time GPS tracking for vehicles, trailers, containers, pallets, and equipment. It supports geofence creation (circle, polygon, corridor, exclusion zone), cold-chain temperature monitoring with breach detection, container tracking with ISO number and seal management, utilisation analytics, harsh-event detection, journey leg segmentation, fleet benchmarking, location anomaly detection (GPS spoofing), and offline telemetry replay. Tamper detection requires immediate escalation.

## Capability ID
`transport_tra`

## Provides
- realtime_gps_tracking_workflow: Continuous position updates with configurable GPS interval
- geofencing_workflow: Entry/exit alerts for 8 geofence types with dwell analytics
- cold_chain_monitoring_workflow: Temperature breach detection against 7 standards with compliance certificates
- container_tracking_workflow: ISO container lifecycle from loading to empty return with detention alerts
- asset_utilisation_workflow: Idle/active time and distance analytics per period
- journey_analytics_workflow: Multi-leg journey segmentation with stop dwell times
- fleet_safety_workflow: Harsh braking, acceleration and speeding detection with safety scoring
- fleet_benchmarking_workflow: Cross-fleet utilisation percentiles and idle cost reporting
- offline_replay_workflow: Buffered telemetry deduplication and replay

## Requires
- auth, audl, mten, conf: Core platform services
- ntfy: Geofence, temperature breach, tamper, and harsh-event alerts
- wflo: Container status workflow
- moni: Continuous asset health monitoring
- comp: Cold chain regulatory compliance certificates
- mqeb: Event streaming via Bytewax
- nlpc: Location address parsing

## Configuration

| Key | Description | Default |
|-----|-------------|---------|
| monitoring.data_retention_days | Data retention | 365 |
| geofencing.max_geofences_per_tenant | Geofence limit | 500 |
| utilisation.idle_threshold_minutes | Idle detection | 30 |
| cold_chain.continuous_logging_enabled | Continuous log | true |

## API Routes

| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| /transport-tracking/dashboard | GET | KPI dashboard | transport_tra:view |
| /transport-tracking/map | GET | Live map (clustered) | transport_tra:view |
| /transport-tracking/assets | GET | Asset registry | transport_tra:assets |
| /transport-tracking/geofencing | GET | Geofence manager | transport_tra:geofencing |
| /transport-tracking/alerts | GET | Alert console | transport_tra:alerts |
| /transport-tracking/cold-chain | GET | Cold chain monitor | transport_tra:cold_chain |
| /transport-tracking/containers | GET | Container tracking | transport_tra:containers |
| /transport-tracking/utilisation | GET | Utilisation reports | transport_tra:utilisation |
| /transport-tracking/reports | GET | Tracking reports | transport_tra:reports |

## Service Methods

### Core (synchronous)
| Method | Description |
|--------|-------------|
| `register_asset()` | Register a trackable asset |
| `update_asset_location()` | Record a GPS location update |
| `create_geofence()` | Create a geofence boundary |
| `raise_alert()` / `acknowledge_alert()` | Alert lifecycle |
| `record_cold_chain()` | Cold chain temperature reading |
| `register_container()` / `update_container_status()` | Container lifecycle |
| `record_utilisation()` | Asset utilisation metrics |
| `register_tracking_agent()` | AI agent registration |
| `list_assets()` / `list_active_alerts()` / `dashboard_summary()` | Queries |

### Extended (async)
| Method | Description |
|--------|-------------|
| `register_tracked_asset()` | Register asset + device binding |
| `update_location()` | Location ingest with automatic geofence checks |
| `geofence_create()` | Geofence creation with alert mode |
| `geofence_event()` | Record and alert on geofence entry/exit |
| `cold_chain_monitoring()` | Cold chain reading with profile auto-lookup |
| `container_tracking()` | Container location + status update |
| `asset_utilisation()` | Per-asset utilisation from location pings |
| `theft_alert()` | Critical theft alert with asset deactivation |
| `tracking_report()` | Full asset tracking report |
| `fleet_map_view()` | GeoJSON FeatureCollection of all asset positions |
| `journey_analytics()` | Multi-leg journey segmentation with stop dwell |
| `detect_harsh_events()` | Harsh braking, acceleration, speeding from GPS deltas |
| `fleet_utilisation_benchmark()` | Fleet-wide p25/p50/p75/p95 utilisation + idle cost |
| `detect_location_anomaly()` | GPS spoofing / position-jump anomaly detection |
| `cold_chain_compliance_summary()` | Per-asset compliance % + deviation detail |
| `container_dwell_report()` | Depot dwell times + detention risk per container |
| `fleet_map_clusters()` | Geohash-clustered map view for large fleets |
| `replay_buffered_telemetry()` | Offline ping deduplication and replay |
| `speeding_violations()` | Fleet-wide speeding violation league table |
| `eta_calculate()` | ETA to destination from last known position |
| `tracking_kpi_summary()` | KPI card for dashboard |
| `cold_chain_alert()` | Immediate cold chain breach alert |
| `customs_checkpoint()` | Customs clearance event record |
| `geofence_exit_alert()` | Geofence exit detection + alert |
| `tracking_analytics_detail()` | Detailed analytics by period |
| `bulk_register_assets()` | Bulk asset registration |
| `alert_summary()` | Active alert summary by type and severity |
| `cold_chain_compliance_report()` | Fleet-wide cold chain compliance % |
| `export_tracking_data()` | Export metadata and download reference |
| `health_check()` | Service health status |
| `deactivate_asset()` | Deactivate an asset (sold, scrapped, stolen) |
| `resolve_alert()` | Resolve an active alert |
| `asset_location_history()` | Recent location pings for an asset |

## Business Rules

| Rule | Condition | Effect |
|------|-----------|--------|
| tamper_alert_escalation_required | Tamper detected | deny |
| asset_unique_id_required | No unique ID | deny |
| geofence_area_required | No boundary defined | deny |
| container_iso_number_required | No ISO number | deny |
| cross_tenant_tracking_denied | Cross-tenant write | deny |

## Data Models
- TrackedAsset: id, asset_type, unique_id, owner_id, tracking_technology, active
- AssetLocationUpdate: id, asset_id, latitude, longitude, speed_kmh, heading_degrees
- Geofence: id, geofence_type, name, boundary_definition, alert_on_entry, alert_on_exit
- TrackingAlert: id, asset_id, alert_type, severity, raised_at, acknowledged_at, resolved_at
- ColdChainRecord: id, asset_id, standard, min_temp_c, max_temp_c, recorded_temp_c, breached
- Container: id, iso_number, seal_number, status, owner_id, current_location
- AssetUtilisationRecord: id, asset_id, period, idle_time_minutes, active_time_minutes, utilisation_pct
- TrackingAgent: id, name, runtime, role, scope

## Streaming Events
- asset_registered, asset_location_updated, geofence_entered, geofence_exited
- tracking_alert_raised, cold_chain_breach_detected, container_status_changed
- utilisation_report_generated, telemetry_replay_complete, harsh_events_detected

## Edge Cases Handled
- Tamper detection blocks location update and forces escalation
- Cold chain breach is automatically calculated from standard profile without client computation
- Utilisation percentage computed server-side from idle/active minutes
- ISO container numbers required; seal-only registration rejected
- Geofence boundary must be defined; name-only geofences rejected
- Position-jump anomalies (>300 km/h implied) flagged as potential GPS spoofing
- Offline buffered pings are deduplicated before replay
- Fleet map clustering prevents browser overload at national scale (10 000+ assets)

## Composability Notes
Provides live position data to `transport_dis` for dispatch tracking. Vehicle IDs reference `transport_fle` registry. Container tracking integrates with `transport_car` for cargo custody chain. Cold chain compliance summaries feed `comp` for certificate generation. Journey leg data feeds `transport_ord` for delivery SLA monitoring.

---

## World-Class Enhancements (v2.0)

- **I1.** Asset Tracking — World Class Improvements
- **I2.** Streaming Telemetry Ingest Pipeline
- **I3.** Predictive Route Deviation Detection
- **I4.** Multi-Leg Journey Analytics
- **I5.** Dwell Time & Detention Tracking
- **I6.** Harsh Event Detection (Acceleration / Braking)
- **I7.** Multi-Standard Cold Chain Certificate Generation
- **I8.** Real-Time Geofence Dwell Analytics
- **I9.** Asset Clustering for Map Density Control
- **I10.** Offline Telemetry Buffer & Replay
- **I11.** Fleet-Wide Utilisation Benchmarking
- **I12.** Alert Suppression & Deduplication
- **I13.** Anomaly-Based Tamper Detection
- **I14.** Audit Log Streaming to Immutable Store
- **I15.** Multi-Tenant Isolation via Row-Level Security

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
