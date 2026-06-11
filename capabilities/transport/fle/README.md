# Fleet Management

## Overview
The Fleet Management capability handles the complete vehicle lifecycle from registration to disposal, telematics integration with major providers, driver management with CPC and tachograph tracking, utilisation analytics, and compliance enforcement including DVLA, C-TPAT, and Euro emissions standards.

## Capability ID
`transport_fle`

## Provides
- vehicle_lifecycle_workflow: Vehicle registration, status management, and disposal
- telematics_integration_workflow: Real-time telematics data from 9+ providers
- driver_management_workflow: Driver registration, licence tracking, CPC and tacho management
- fleet_utilisation_analytics_workflow: Distance, hours, idle time, and load factor analytics
- fleet_compliance_workflow: MOT, operator licence, tachograph, and DVLA compliance

## Requires
- auth, audl, mten, conf: Core platform services
- ntfy: Compliance expiry and maintenance alerts
- wflo: Vehicle status state machine
- moni: Fleet health monitoring
- comp: Regulatory compliance framework
- mqeb: Event streaming
- schd: Driver shift scheduling integration

## Configuration

| Key | Description | Default |
|-----|-------------|---------|
| compliance.mot_tracking_enabled | MOT expiry tracking | true |
| compliance.tachograph_enabled | Tachograph compliance | true |
| telematics.driver_behaviour_scoring | Behaviour scoring | true |
| drivers.hours_of_service_enabled | HOS tracking | true |

## API Routes

| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| /transport-fleet/vehicles | GET | Vehicle registry | transport_fle:vehicles |
| /transport-fleet/drivers | GET | Driver registry | transport_fle:drivers |
| /transport-fleet/telematics | GET | Telematics events | transport_fle:telematics |
| /transport-fleet/compliance | GET | Compliance records | transport_fle:compliance |
| /transport-fleet/utilisation | GET | Utilisation analytics | transport_fle:utilisation |
| /api/fle/v1/fuel/<id>/audit | POST | Fuel fraud anomaly audit | transport_fle:audit |
| /api/fle/v1/drivers/<id>/fatigue-risk | GET | Driver fatigue risk score | transport_fle:drivers |
| /api/fle/v1/vehicles/<id>/disposal | GET | Disposal/replacement recommendation | transport_fle:vehicles |
| /api/fle/v1/shifts/optimise | POST | HOS-constrained shift assignment | transport_fle:dispatch |
| /api/fle/v1/reports/budget-variance | POST | MTD budget burn-rate variance | transport_fle:reports |
| /api/fle/v1/incidents/<id>/claim-pack | GET | Insurance claim evidence pack | transport_fle:incidents |
| /api/fle/v1/geofence/event | POST | Geofence workflow trigger | transport_fle:telematics |
| /api/fle/v1/telematics/<id>/coaching | POST | In-cab coaching event | transport_fle:telematics |
| /api/fle/v1/vehicles/<id>/health | GET | 360-degree vehicle health snapshot | transport_fle:vehicles |
| /api/fle/v1/reports/driver-leaderboard | GET | Driver behaviour leaderboard | transport_fle:reports |
| /api/fle/v1/maintenance/<id>/defer | POST | Defer maintenance with audit trail | transport_fle:maintenance |

## Business Rules

| Rule | Condition | Effect |
|------|-----------|--------|
| non_compliant_vehicle_dispatch_denied | Compliance check failed | deny |
| unlicenced_driver_dispatch_denied | Licence invalid | deny |
| vehicle_vin_required | VIN absent | deny |
| fuel_type_supported | Unsupported fuel type | deny |
| cross_tenant_fleet_denied | Cross-tenant write | deny |

## Data Models
- Vehicle: id, vehicle_type, registration, vin, fuel_type, ownership_type, status, make, model, year
- Driver: id, name, licence_number, licence_class, status, tacho_card_number, cpc_expiry
- TelematicsRecord: id, vehicle_id, provider, event_type, latitude, longitude, speed_kmh
- ComplianceRecord: id, vehicle_id, standard, certificate_ref, issued_at, expires_at, passed
- UtilisationRecord: id, vehicle_id, metric, value, period_start, period_end

## Streaming Events
- vehicle_registered, vehicle_status_changed, driver_registered, driver_status_changed
- telematics_event, compliance_check_completed, utilisation_report_generated

## Edge Cases Handled
- Compliance check must explicitly pass before dispatch is cleared — default is not assumed
- Fuel type validation prevents unsupported powertrains from being registered
- Telematics provider must be on the supported list to prevent rogue data ingestion
- CPC expiry tracked independently from licence validity

## Advanced Analytics Methods

| Service Method | Description |
|----------------|-------------|
| `audit_fuel_record(id)` | Statistical fraud detection: price deviation, volume anomaly, duplicate receipt |
| `assess_driver_fatigue_risk(driver_id, days)` | Tachograph-pattern fatigue risk score without hardware |
| `disposal_recommendation(vehicle_id, market_value)` | Data-driven replace/monitor/retain recommendation |
| `optimise_shift_assignments(date)` | Greedy HOS-constrained driver–trip assignment |
| `fleet_budget_variance(fuel_budget, maint_budget)` | MTD burn-rate vs budget with projected month-end |
| `generate_incident_claim_pack(incident_id)` | Automated insurance claim evidence package |
| `process_geofence_event(vehicle_id, fence, type)` | Geofence workflow trigger with step execution |
| `generate_driver_coaching_event(telematics_id)` | Contextual in-cab micro-coaching message |
| `vehicle_health_snapshot(vehicle_id)` | 360-degree health score + all risk signals in one call |
| `driver_leaderboard(top_n)` | Ranked driver behaviour leaderboard for gamification |
| `deferred_maintenance(id, new_date, reason)` | Defer scheduled maintenance with full audit trail |

## Composability Notes
Acts as the master registry for `transport_dis` (driver and vehicle assignment), `transport_mai` (maintenance scheduling), `transport_fue` (fuel transaction attribution), and `transport_tra` (asset tracking linkage via vehicle ID).

Emits domain events consumed by: `ntfy` (compliance and budget alerts), `wflo` (geofence workflows, post-trip inspections), `bia` (ESG/emissions dashboard), `schd` (shift schedule feed), `scm` (parts availability for predictive maintenance).
