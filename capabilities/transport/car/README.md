# Cargo Management

## Overview
The Cargo Management capability provides end-to-end cargo lifecycle management including booking creation, manifest generation, dangerous goods compliance, real-time cargo tracking, and revenue management. It enforces IATA, IMDG, ADR, and C-TPAT compliance standards and integrates with bytewax for streaming cargo lifecycle events.

## Capability ID
`transport_car`

## Provides
- cargo_booking_workflow: Full cargo booking lifecycle from draft to delivery
- cargo_manifest_workflow: Manifest creation, submission, and customs declaration
- dangerous_goods_compliance_workflow: DG class declaration, UN number validation, packing group enforcement
- cargo_tracking_workflow: Real-time event-based cargo tracking with geofencing
- cargo_revenue_workflow: Revenue line recording with rate cards and surcharges
- cargo_compliance_workflow: Multi-standard compliance checking and certificate management

## Requires
- auth: Authentication and authorisation for all operations
- audl: Audit trail for all cargo write operations
- mten: Multi-tenancy context enforcement
- conf: Configuration management
- ntfy: Notifications for cargo status changes
- wflo: Workflow state machine for booking lifecycle
- moni: Operational monitoring and alerting
- comp: Regulatory compliance framework
- mqeb: Event bus for bytewax streaming
- schd: Scheduling for time-window constraints

## Configuration

| Key | Description | Default |
|-----|-------------|---------|
| tenant_id | Tenant identifier | default |
| cargo_types.supported_types | Valid cargo classifications | 12 types |
| dangerous_goods.compliance_standards | DG compliance frameworks | IATA, IMDG, ADR, etc. |
| revenue.approval_required_above_threshold | Require approval for large charges | true |

## API Routes

| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| /transport-cargo/dashboard | GET | Cargo operations overview | transport_car:view |
| /transport-cargo/bookings | GET | List all bookings | transport_car:bookings |
| /transport-cargo/bookings/create | POST | Create new booking | transport_car:bookings_write |
| /transport-cargo/manifests | GET | List manifests | transport_car:manifests |
| /transport-cargo/dangerous-goods | GET | DG declarations console | transport_car:dg_compliance |
| /transport-cargo/tracking | GET | Cargo tracking board | transport_car:tracking |
| /transport-cargo/revenue | GET | Revenue records | transport_car:revenue |
| /transport-cargo/compliance | GET | Compliance checks | transport_car:compliance |
| /transport-cargo/yard | GET | Yard/CFS location management | transport_car:yard |
| /transport-cargo/documents | GET | Transport document generation | transport_car:documents |
| /transport-cargo/consolidation | GET | LCL/FCL consolidation | transport_car:consolidation |
| /transport-cargo/carbon | GET | Carbon footprint reports | transport_car:carbon |
| /transport-cargo/eta | GET | Predictive ETA dashboard | transport_car:tracking |
| /transport-cargo/disputes | GET | Dispute resolution console | transport_car:disputes |
| /transport-cargo/customs | GET | Customs pre-clearance | transport_car:customs |

## Business Rules

| Rule | Condition | Effect |
|------|-----------|--------|
| tenant_context_required | No tenant context | deny |
| booking_shipper_required | Shipper absent | deny |
| unapproved_dg_shipment_denied | Hazmat without DG approval | deny |
| weight_falsification_denied | Weight manipulation detected | deny |
| cross_tenant_cargo_denied | Cross-tenant write attempt | deny |
| phantom_fill_detection | Phantom fill pattern detected | deny |
| dg_segregation_on_consolidation | Incompatible DG classes co-loaded | deny |
| customs_draft_required | Pre-clearance submission on non-draft | deny |

## Data Models
- CargoBooking: id, tenant_id, cargo_type, shipper_id, consignee_id, origin, destination, weight_kg, status
- CargoManifest: id, tenant_id, booking_id, status, customs_declaration_ref
- DangerousGoodsDeclaration: id, tenant_id, booking_id, dg_class, un_number, packing_group, emergency_contact
- CargoTrackingEvent: id, tenant_id, booking_id, event_type, location, timestamp
- CargoRevenueRecord: id, tenant_id, booking_id, revenue_type, amount, currency
- CargoComplianceRecord: id, tenant_id, booking_id, standard, certificate_ref, passed
- CargoAgent: id, tenant_id, name, runtime, role, scope
- YardAssignment: assignment_id, booking_id, yard_id, bay, stack, dwell_started_at, free_storage_days
- TransportDocument: document_ref, document_type (bol/awb/cmr), booking_id, issued_at
- Consolidation: hbl_ref, container_type, booking_ids, fill_rate_weight_pct, fill_rate_volume_pct
- CarbonFootprint: booking_id, mode, distance_km, net_kg_co2e, offset_cost_usd
- DisputeRecord: dispute_id, booking_id, dispute_type, status, insurance_policy_ref
- CustomsDeclaration: declaration_ref, shipment_id, duty_lines, total_estimated_duty, gateway_ref

## Service Methods (async)

### Core Booking
- `book_cargo()` — full booking with volumetric weight and rate calculation
- `create_booking()` — low-level booking creation with rule enforcement
- `get_booking_async()` / `list_bookings_async()` / `cancel_booking_async()`
- `booking_amendment()` — amend weight/volume/incoterm/packaging
- `bulk_create_bookings()` — batch booking creation
- `rate_inquiry()` — freight rate estimate without creating a booking

### Documentation
- `cargo_manifest()` — generate consolidated manifest with DG and revenue lines
- `generate_transport_document()` — produce BoL, AWB, or CMR document record

### DG & Compliance
- `dangerous_goods_check()` — validate DG classification and return requirements
- `compliance_check()` — verify booking meets regulatory requirements
- `record_compliance()` — persist compliance certificate record

### Tracking & ETA
- `track_cargo()` — full tracking chain with milestone progress
- `predict_eta()` — P50/P90 ETA forecast with confidence intervals
- `update_tracking()` — record a tracking event

### Customs & Trade
- `customs_declaration()` — build declaration with HS-code duty estimates
- `submit_customs_pre_clearance()` — submit to ASYCUDA/TradNet/ICEGATE

### Yard & Storage
- `assign_yard_location()` — assign CFS/ICD bay and stack
- `release_from_yard()` — release with storage-charge calculation
- `detention_demurrage()` — compute D&D charges

### Consolidation & Logistics
- `consolidate_bookings()` — LCL → FCL with DG segregation validation
- `calculate_carbon_footprint()` — Scope-3 carbon report with DG surcharge

### Finance & Revenue
- `record_revenue()` — persist revenue line
- `revenue_management()` — route-level yield and contribution margin
- `cost_analysis()` — cost breakdown by period

### Claims & Disputes
- `cargo_insurance()` — attach insurance policy with premium calculation
- `cargo_loss_claim()` — file loss/damage claim against a shipment
- `open_dispute()` — open structured dispute with insurance linkage
- `list_loss_claims()` — list all tenant claims

### Analytics & Reporting
- `cargo_analytics()` — aggregate KPIs for a period
- `analytics_dashboard()` — dashboard metrics
- `reporting_export()` — structured report generation
- `export_cargo_data()` — data export with download reference
- `performance_kpi()` — booking volume, revenue-per-kg, on-time rate
- `dashboard_async()` / `dashboard_summary()` — operational overview

### Operations & Admin
- `exception_handling()` — log and escalate cargo exceptions
- `bulk_operation()` — apply an operation to multiple bookings
- `customer_notification()` — send consignee status notification
- `predictive_maintenance()` — asset maintenance window prediction
- `integration_external()` — push data to external logistics systems
- `validate_batch()` — bytewax batch routing validation
- `health_check()` — service liveness and counters

## Streaming Events
- cargo_booked, cargo_manifest_submitted, cargo_dg_declared, cargo_tracking_updated
- cargo_delivered, cargo_revenue_recorded, cargo_compliance_checked, cargo_agent_registered
- cargo_yard_assigned, cargo_yard_released, transport_document_generated
- cargo_consolidation_created, carbon_footprint_calculated, cargo_dispute_opened
- customs_pre_clearance_submitted, detention_demurrage_calculated

## Edge Cases Handled
- Hazardous cargo requires explicit DG approval before booking confirmation
- Weight falsification detection blocks fraudulent weight declarations
- Cross-tenant cargo access is denied at the rule engine level
- Phantom fill detection guards against fuel/quantity fraud
- Manifest submission requires an existing confirmed booking
- Revenue amounts must be strictly positive
- LCL consolidation blocks incompatible DG class pairs (explosives + flammables)
- Customs pre-clearance submission requires declaration to be in draft status
- Yard release computes actual dwell from timestamp, not a user-supplied value
- ETA prediction degrades gracefully when no tracking events exist (uses default 96h base)

## Composability Notes
Composes with `transport_dis` for dispatch assignment, `transport_rou` for route planning per booking, `transport_tra` for real-time GPS tracking of cargo, and `transport_sch` for load scheduling. The DG compliance workflow integrates with `comp` for regulatory certificate management. Carbon footprint data feeds into the ESG reporting pipeline. Customs pre-clearance integrates with the `comp` capability's document vault.

## Improvement Roadmap
See `WORLD_CLASS_IMPROVEMENTS.md` for 15 planned enhancements covering persistent storage, event sourcing, rate card engine, IoT integration, ML-powered ETA, automated customs, and Pydantic v2 model migration.
