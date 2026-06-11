# Delivery Management

## Overview
The Delivery Management capability handles last-mile delivery planning, proof-of-delivery capture, customer notifications, failed delivery handling, rescheduling workflows, SLA tracking, and return management. It enforces geo-stamped POD capture, protects against POD falsification, and exposes advanced analytics, route optimisation, carbon tracking, driver gamification, and insurance claim management.

## Capability ID
`transport_del`

## Provides
- `delivery_planning_workflow` — Delivery creation with time-window constraints and SLA tiers
- `proof_of_delivery_workflow` — Multi-modal POD capture (signature, photo, PIN, biometric, locker)
- `customer_notification_workflow` — Multi-channel ETA and delivery notifications; HMAC-signed webhooks
- `failed_delivery_workflow` — Failed attempt recording with auto-reschedule logic and risk scoring
- `sla_tracking_workflow` — SLA commitment tracking, penalty auto-calculation, and breach alerting
- `delivery_return_workflow` — RMA-based return initiation with insurance claim support
- `route_optimisation_workflow` — Driver route resequencing with per-stop ETA offsets
- `carbon_tracking_workflow` — GHG Protocol Scope 3 CO2e estimation per delivery leg
- `driver_incentive_workflow` — Gamified driver scoring and incentive payout calculation
- `delivery_manifest_workflow` — Multi-parcel manifest creation and batch POD completion

## Requires
- `auth`, `audl`, `mten`, `conf` — Core platform services
- `ntfy` — Customer notification delivery
- `wflo` — Delivery state machine
- `moni` — SLA breach monitoring
- `comp` — Regulatory compliance for deliveries
- `mqeb` — Event streaming
- `schd` — Time-window scheduling

## Configuration

| Key | Description | Default |
|-----|-------------|---------|
| `failed_deliveries.max_attempts` | Max delivery attempts before forced return | 3 |
| `rescheduling.max_reschedule_count` | Max reschedules per delivery | 3 |
| `proof_of_delivery.geo_stamp_required` | Geo-stamp mandatory for all POD types | true |
| `sla.breach_alert_enabled` | Alert on SLA breach | true |
| `risk_scoring.failure_threshold` | Failure probability above which pre-call triggers | 0.55 |
| `carbon.default_vehicle_type` | Vehicle type for carbon estimates when unspecified | van |
| `webhook.signing_algorithm` | HMAC algorithm for outbound webhooks | hmac-sha256 |

## API Routes

| Path | Method | Description | Permission |
|------|--------|-------------|------------|
| `/transport-delivery/deliveries` | GET | List deliveries | `transport_del:deliveries` |
| `/transport-delivery/deliveries` | POST | Create delivery | `transport_del:deliveries_write` |
| `/transport-delivery/pod` | GET | POD records | `transport_del:pod` |
| `/transport-delivery/failed` | GET | Failed deliveries | `transport_del:failed` |
| `/transport-delivery/sla` | GET | SLA tracking | `transport_del:sla` |
| `/transport-delivery/returns` | GET | Return management | `transport_del:returns` |
| `/transport-delivery/notifications` | GET | Notification log | `transport_del:notifications` |
| `/transport-delivery/routes/optimise` | POST | Optimise driver route | `transport_del:routes` |
| `/transport-delivery/carbon` | POST | Compute carbon footprint | `transport_del:carbon` |
| `/transport-delivery/manifests` | POST | Create delivery manifest | `transport_del:manifests` |
| `/transport-delivery/claims` | POST | File insurance claim | `transport_del:claims` |
| `/transport-delivery/eta` | PUT | Update real-time ETA | `transport_del:eta` |
| `/transport-delivery/incentives` | GET | Driver incentive report | `transport_del:incentives` |
| `/transport-delivery/webhooks` | POST | Register webhook | `transport_del:webhooks` |

## Key Service Methods

### Core (synchronous)
| Method | Description |
|--------|-------------|
| `create_delivery()` | Create a delivery record |
| `record_pod()` | Record proof of delivery |
| `record_failed_delivery()` | Record a failed attempt |
| `reschedule_delivery()` | Reschedule a delivery |
| `set_sla()` | Attach SLA commitment |
| `send_notification()` | Fire a notification |
| `create_return()` | Initiate an RMA return |
| `register_delivery_agent()` | Register an AI agent |

### Async — Workflows
| Method | Description |
|--------|-------------|
| `create_delivery_async()` | Create delivery + SLA + notification in one call |
| `assign_driver()` | Assign driver and vehicle; notify customer |
| `proof_of_delivery()` | Multi-modal POD with SLA compliance check |
| `failed_delivery()` | Record failure and trigger next action |
| `reattempt_delivery()` | Schedule reattempt with max-3 guard |
| `customer_notification()` | Event-driven multi-channel notifications |
| `delivery_sla_check()` | Real-time SLA status with penalty exposure |
| `returns_management()` | Initiate return with restocking fee |
| `delivery_rating()` | Record customer rating and update driver average |

### Async — Analytics & Reporting
| Method | Description |
|--------|-------------|
| `last_mile_analytics()` | Aggregate last-mile KPIs for a period |
| `delivery_performance_report()` | KPI report (delegates to analytics) |
| `driver_performance_report()` | Per-driver completion rate and avg rating |
| `sla_breach_report()` | SLA breach count and rate by tier |
| `pod_compliance_check()` | POD coverage rate across delivered shipments |
| `cost_analysis()` | Delivery cost estimate with reschedule overhead |
| `performance_kpi()` | High-level success rate and totals |
| `export_delivery_data()` | Export delivery records metadata |
| `reporting_export()` | Full statistics export |
| `analytics_dashboard()` | Aggregated metrics for ops dashboard |

### Async — Advanced
| Method | Description |
|--------|-------------|
| `optimise_route()` | Resequence deliveries for a driver (VRP heuristic) |
| `register_webhook()` | Register HMAC-signed webhook for lifecycle events |
| `compute_carbon_footprint()` | GHG Protocol Scope 3 CO2e per delivery leg |
| `score_failed_delivery_risk()` | Predict failure probability before dispatch |
| `create_delivery_manifest()` | Group deliveries into a single driver run |
| `complete_manifest()` | Batch POD for all deliveries in a manifest |
| `file_insurance_claim()` | Create damage/loss/theft claim with payout estimate |
| `update_realtime_eta()` | Haversine ETA recompute from live GPS position |
| `compute_driver_incentive()` | Gamified score + payout for driver period |

### Utility
| Method | Description |
|--------|-------------|
| `bulk_create_deliveries()` | Batch create from order list |
| `bulk_operation()` | Apply operation to multiple deliveries |
| `update_delivery_status()` | Status transition with audit trail |
| `compliance_check()` | Verify POD + SLA for a single delivery |
| `health_check()` | Service health and store counts |
| `predictive_maintenance()` | Vehicle next-service prediction |
| `integration_external()` | Push records to 3PL / courier system |
| `exception_handling()` | Log delivery exception (damage, missed, refused) |

## Business Rules

| Rule | Condition | Effect |
|------|-----------|--------|
| `pod_falsification_denied` | Geo-stamp cross-validation fails | deny |
| `max_reschedule_exceeded` | >3 reschedules attempted | deny |
| `pod_geo_stamp_required` | No geo-stamp on POD submission | deny |
| `return_rma_required` | No RMA number on return | deny |
| `cross_tenant_delivery_denied` | Cross-tenant write attempt | deny |
| `driver_already_assigned` | Active assignment exists | ValueError |
| `max_attempts_exceeded` | Reattempt on 3-attempt delivery | ValueError |
| `rating_delivered_only` | Rating on non-delivered delivery | ValueError |
| `invalid_claim_type` | Unsupported insurance claim type | ValueError |

## Data Models
- `Delivery` — id, delivery_type, recipient_name, delivery_address, time_window, status, sla_tier, attempt_count
- `ProofOfDelivery` — id, delivery_id, pod_type, geo_stamp, captured_at, signatory_name
- `FailedDelivery` — id, delivery_id, failure_reason, failed_at, notes, rescheduled
- `DeliveryReschedule` — id, delivery_id, source, new_time_window, reschedule_count
- `SlaRecord` — id, delivery_id, sla_tier, committed_at, achieved_at, met
- `DeliveryNotification` — id, delivery_id, channel, recipient_contact, sent_at, notification_type
- `DeliveryReturn` — id, delivery_id, return_reason, rma_number, initiated_at
- `DeliveryAgent` — id, name, runtime, role, scope

## Streaming Events
```
delivery_created          delivery_assigned         delivery_out_for_delivery
delivery_completed        delivery_failed           pod_recorded
sla_breached              delivery_notification_sent delivery_returned
driver_assigned           route_optimised           carbon_footprint_computed
delivery_manifest_created delivery_manifest_completed insurance_claim_filed
realtime_eta_updated      driver_incentive_computed  webhook_registered
```

## Edge Cases Handled
- Max 3 reschedules enforced at rule level — 4th attempt initiates return process
- POD falsification detection via geo-stamp cross-validation
- Failed delivery automatically increments `attempt_count` on the parent record
- Geo-stamp mandatory for all POD types including `safe_place` and `neighbour`
- RMA number required before any return record can be created
- `assign_driver` is idempotent on the first call; re-assignment raises `ValueError`
- Driver rating only permitted when delivery status is `delivered`
- Manifest `complete_manifest` is safe to call with partially-delivered manifests
- Carbon footprint load correction applied only above 500 kg baseline
- Insurance payout capped at declared value × claim-type rate

## Composability Notes
Composes with:
- `transport_dis` — Driver dispatch and capacity management
- `transport_tra` — Live vehicle tracking, GPS breadcrumb ingest
- `transport_sch` — Time-window scheduling optimisation
- `transport_rou` — Network-level route planning (feeds `optimise_route`)
- `billing` — SLA penalty invoicing and insurance claim settlement
- `ident` — Biometric KYC verification for high-value POD
- `esg` — Carbon report ingestion for Scope 3 GHG disclosure
