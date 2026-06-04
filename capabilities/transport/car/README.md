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

## Business Rules

| Rule | Condition | Effect |
|------|-----------|--------|
| tenant_context_required | No tenant context | deny |
| booking_shipper_required | Shipper absent | deny |
| unapproved_dg_shipment_denied | Hazmat without DG approval | deny |
| weight_falsification_denied | Weight manipulation detected | deny |
| cross_tenant_cargo_denied | Cross-tenant write attempt | deny |
| phantom_fill_detection | Phantom fill pattern detected | deny |

## Data Models
- CargoBooking: id, tenant_id, cargo_type, shipper_id, consignee_id, origin, destination, weight_kg, status
- CargoManifest: id, tenant_id, booking_id, status, customs_declaration_ref
- DangerousGoodsDeclaration: id, tenant_id, booking_id, dg_class, un_number, packing_group, emergency_contact
- CargoTrackingEvent: id, tenant_id, booking_id, event_type, location, timestamp
- CargoRevenueRecord: id, tenant_id, booking_id, revenue_type, amount, currency
- CargoComplianceRecord: id, tenant_id, booking_id, standard, certificate_ref, passed
- CargoAgent: id, tenant_id, name, runtime, role, scope

## Streaming Events
- cargo_booked, cargo_manifest_submitted, cargo_dg_declared, cargo_tracking_updated
- cargo_delivered, cargo_revenue_recorded, cargo_compliance_checked, cargo_agent_registered

## Edge Cases Handled
- Hazardous cargo requires explicit DG approval before booking confirmation
- Weight falsification detection blocks fraudulent weight declarations
- Cross-tenant cargo access is denied at the rule engine level
- Phantom fill detection guards against fuel/quantity fraud
- Manifest submission requires an existing confirmed booking
- Revenue amounts must be strictly positive

## Composability Notes
Composes with `transport_dis` for dispatch assignment, `transport_rou` for route planning per booking, `transport_tra` for real-time GPS tracking of cargo, and `transport_sch` for load scheduling. The DG compliance workflow integrates with `comp` for regulatory certificate management.
