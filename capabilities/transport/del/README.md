# Delivery Management

## Overview
The Delivery Management capability handles last-mile delivery planning, proof-of-delivery capture, customer notifications, failed delivery handling, rescheduling workflows, SLA tracking, and return management. It enforces geo-stamped POD capture and protects against POD falsification.

## Capability ID
`transport_del`

## Provides
- delivery_planning_workflow: Delivery creation with time-window constraints and SLA tiers
- proof_of_delivery_workflow: Multi-modal POD capture (signature, photo, PIN, biometric, locker)
- customer_notification_workflow: Multi-channel ETA and delivery notifications
- failed_delivery_workflow: Failed attempt recording with auto-reschedule logic
- sla_tracking_workflow: SLA commitment tracking and breach alerting
- delivery_return_workflow: RMA-based return initiation

## Requires
- auth, audl, mten, conf: Core platform services
- ntfy: Customer notification delivery
- wflo: Delivery state machine
- moni: SLA breach monitoring
- comp: Regulatory compliance for deliveries
- mqeb: Event streaming
- schd: Time-window scheduling

## Configuration

| Key | Description | Default |
|-----|-------------|---------|
| failed_deliveries.max_attempts | Max delivery attempts | 3 |
| rescheduling.max_reschedule_count | Max reschedules per delivery | 3 |
| proof_of_delivery.geo_stamp_required | Geo-stamp mandatory | true |
| sla.breach_alert_enabled | Alert on SLA breach | true |

## API Routes

| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| /transport-delivery/deliveries | GET | List deliveries | transport_del:deliveries |
| /transport-delivery/pod | GET | POD records | transport_del:pod |
| /transport-delivery/failed | GET | Failed deliveries | transport_del:failed |
| /transport-delivery/sla | GET | SLA tracking | transport_del:sla |
| /transport-delivery/returns | GET | Return management | transport_del:returns |
| /transport-delivery/notifications | GET | Notification log | transport_del:notifications |

## Business Rules

| Rule | Condition | Effect |
|------|-----------|--------|
| pod_falsification_denied | Falsification detected | deny |
| max_reschedule_exceeded | >3 reschedules | deny |
| pod_geo_stamp_required | No geo-stamp | deny |
| return_rma_required | No RMA number | deny |
| cross_tenant_delivery_denied | Cross-tenant write | deny |

## Data Models
- Delivery: id, delivery_type, recipient_name, delivery_address, time_window, status, sla_tier, attempt_count
- ProofOfDelivery: id, delivery_id, pod_type, geo_stamp, captured_at, signatory_name
- FailedDelivery: id, delivery_id, failure_reason, failed_at
- DeliveryReschedule: id, delivery_id, source, new_time_window, reschedule_count
- SlaRecord: id, delivery_id, sla_tier, committed_at, actual_at, breached
- DeliveryNotification: id, delivery_id, channel, recipient_contact, sent_at
- DeliveryReturn: id, delivery_id, return_reason, rma_number

## Streaming Events
- delivery_created, delivery_assigned, delivery_out_for_delivery, delivery_completed
- delivery_failed, pod_recorded, sla_breached, delivery_notification_sent, delivery_returned

## Edge Cases Handled
- Max 3 reschedules enforced at rule level — 4th attempt initiates return process
- POD falsification detection via geo-stamp cross-validation
- Failed delivery automatically increments attempt_count on the parent Delivery record
- Geo-stamp is mandatory for all POD types (even safe_place and neighbour)
- RMA number required before any return can be created

## Composability Notes
Composes with `transport_dis` for driver dispatch, `transport_tra` for live vehicle tracking during last-mile, `transport_sch` for time-window scheduling, and `transport_rou` for delivery route optimisation.
