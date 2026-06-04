# Asset Tracking

## Overview
The Asset Tracking capability provides real-time GPS tracking for vehicles, trailers, containers, pallets, and equipment. It supports geofence creation (circle, polygon, corridor, exclusion zone), cold-chain temperature monitoring with breach detection, container tracking with ISO number and seal management, and utilisation analytics. Tamper detection requires immediate escalation.

## Capability ID
`transport_tra`

## Provides
- realtime_gps_tracking_workflow: Continuous position updates with configurable GPS interval
- geofencing_workflow: Entry/exit alerts for 8 geofence types
- cold_chain_monitoring_workflow: Temperature breach detection against 6 standards
- container_tracking_workflow: ISO container lifecycle from loading to empty return
- asset_utilisation_workflow: Idle/active time and distance analytics per period

## Requires
- auth, audl, mten, conf: Core platform services
- ntfy: Geofence, temperature breach, and tamper alerts
- wflo: Container status workflow
- moni: Continuous asset health monitoring
- comp: Cold chain regulatory compliance
- mqeb: Event streaming
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
| /transport-tracking/map | GET | Live map | transport_tra:view |
| /transport-tracking/assets | GET | Asset registry | transport_tra:assets |
| /transport-tracking/geofencing | GET | Geofence manager | transport_tra:geofencing |
| /transport-tracking/alerts | GET | Alert console | transport_tra:alerts |
| /transport-tracking/cold-chain | GET | Cold chain monitor | transport_tra:cold_chain |
| /transport-tracking/containers | GET | Container tracking | transport_tra:containers |
| /transport-tracking/utilisation | GET | Utilisation reports | transport_tra:utilisation |

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
- TrackingAlert: id, asset_id, alert_type, severity, raised_at, acknowledged_at
- ColdChainRecord: id, asset_id, standard, min_temp_c, max_temp_c, recorded_temp_c, breached
- Container: id, iso_number, seal_number, status, owner_id, current_location
- AssetUtilisationRecord: id, asset_id, period, idle_time_minutes, active_time_minutes, utilisation_pct

## Streaming Events
- asset_registered, asset_location_updated, geofence_entered, geofence_exited
- tracking_alert_raised, cold_chain_breach_detected, container_status_changed, utilisation_report_generated

## Edge Cases Handled
- Tamper detection blocks location update and forces escalation — no silent acceptance
- Cold chain breach is automatically calculated from min/max range without client computation
- Utilisation percentage is computed server-side from idle/active minutes
- ISO container numbers must be provided — seal-only registration is not supported
- Geofence boundary must be defined — name-only geofences are rejected

## Composability Notes
Provides live position data to `transport_dis` for dispatch tracking. Vehicle IDs reference `transport_fle` registry. Container tracking integrates with `transport_car` for cargo custody chain. Cold chain events feed compliance certificates via `comp`.
