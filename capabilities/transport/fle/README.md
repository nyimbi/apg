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

## Composability Notes
Acts as the master registry for `transport_dis` (driver and vehicle assignment), `transport_mai` (maintenance scheduling), `transport_fue` (fuel transaction attribution), and `transport_tra` (asset tracking linkage via vehicle ID).
