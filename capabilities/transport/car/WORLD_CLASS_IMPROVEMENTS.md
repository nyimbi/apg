# Cargo Management — World-Class Improvement Roadmap

**Capability**: `transport_car` | **Domain**: `transport` | **Author**: Nyimbi Odero | **Date**: 2026-06-11

---

## 1. Persistent Storage Layer (PostgreSQL)

The current service uses in-memory dicts. Migrating to async SQLAlchemy + PostgreSQL gives durability, concurrent tenants, and crash recovery. The `database/store.py` stub exists but is not wired into `CargoManagementService`. Wire `_store` into every read/write method, fall back to in-memory for tests.

**Impact**: production readiness, multi-node deployability.

---

## 2. Event Sourcing for Booking Lifecycle

Replace mutable `booking.status` with an append-only event log. Every state transition (`confirmed → in_transit → delivered`) becomes an immutable `BookingStateEvent`. Current state is derived by replaying the log. This enables audit rewind, conflict-free distributed updates, and event-driven projections.

**Impact**: auditability, eventual consistency, replay-based debugging.

---

## 3. Async Webhook / Push Notification Engine

`customer_notification` currently returns a stub dict. Wire it to an outbound webhook dispatcher (HTTP POST to registered URLs) with exponential-backoff retry, dead-letter queue via bytewax, and per-tenant webhook registration. Integrate with `ntfy` capability for SMS/email/push.

**Impact**: real-time consignee visibility, SLA alerting, carrier event push.

---

## 4. Rate Card Engine with Tariff Versioning

Current rate logic is hardcoded scalars (`2.50`, `0.85`). Replace with a versioned tariff table: `(route, cargo_type, weight_band, effective_date) → rate`. Support fuel escalation clauses, seasonal adjustments, and FAK (freight-all-kinds) grouping. Expose `get_tariff`, `publish_tariff`, `supersede_tariff` methods.

**Impact**: accurate invoicing, carrier contract modelling, yield management.

---

## 5. Real-Time IoT Tracking Integration

`update_tracking` accepts manual event pushes. Add an async `ingest_iot_telemetry` method that consumes a stream of GPS/temperature/shock sensor payloads from IoT devices (MQTT or Kafka topic), normalises them into `CargoTrackingEvent` objects, evaluates geofences, and fires alerts on breach. Integrates with `transport_tra`.

**Impact**: end-to-end shipment visibility, cold-chain SLA enforcement.

---

## 6. Multi-Modal Route Optimisation

Add `optimise_route(origin, destination, cargo_type, constraints)` that calls into `transport_rou`, evaluates road/sea/air combinations, applies dangerous-goods mode restrictions (class_7 cannot fly IATA), and returns a ranked list of route options with cost, transit time, and carbon footprint. Cache results per route+day.

**Impact**: cost reduction, DG mode compliance automation, carbon reporting.

---

## 7. Carbon Footprint Calculator

Add `calculate_carbon_footprint(booking_id)` that multiplies weight × distance × modal emission factor (g CO₂/tonne-km: road 62, sea 8, air 602), applies DG risk surcharges from `_DG_RISK_SURCHARGES`, and returns a structured carbon report including offset credit estimate. Support SBTi scope-3 export format.

**Impact**: ESG compliance, carbon offset marketplace integration, shipper sustainability reporting.

---

## 8. Predictive ETA with ML Confidence Intervals

Current `track_cargo` infers progress from milestone index. Add `predict_eta(booking_id)` that feeds historical transit-time distributions (per route/carrier/season) into a lightweight regression model (via local Ollama), returns P50/P90 ETA with confidence interval, and updates as new tracking events arrive.

**Impact**: proactive exception management, consignee SLA commitments, detention risk flagging.

---

## 9. Automated Customs Pre-Clearance

Extend `customs_declaration` to submit electronically to customs APIs (Kenya TradNet, ASYCUDA World). Add `submit_customs_pre_clearance(declaration_ref)` with status polling, document upload (certificates of origin, phytosanitary), and automated HS-code classification via Ollama embedding similarity against HS tariff schedule. Return clearance ETA.

**Impact**: dwell-time reduction, border-crossing predictability, compliance automation.

---

## 10. Cargo Consolidation (LCL/FCL Optimiser)

Add `consolidate_bookings(booking_ids, container_type)` that bins confirmed LCL bookings into FCL containers using a 3D bin-packing algorithm, validates weight limits, segregates incompatible DG classes, and generates a single consolidated manifest. Returns fill-rate metric and savings estimate.

**Impact**: cost savings for LCL shippers, container utilisation, DG segregation enforcement.

---

## 11. Dispute Resolution Workflow

Add `open_dispute(booking_id, dispute_type, evidence_refs)` → `escalate_dispute` → `close_dispute` state machine. Dispute types: `weight_discrepancy`, `damage`, `short_delivery`, `delay_penalty`. Route to cargo surveyor agent, auto-attach insurance claim if applicable, integrate with `comp` for arbitration records.

**Impact**: structured SLA breach handling, reduced manual back-and-forth, legal audit trail.

---

## 12. Cargo Yard / Warehouse Management Integration

Add `assign_yard_location(booking_id, yard_id, bay, stack)` and `release_from_yard(booking_id)` that track physical dwell in CFS/ICD facilities. Compute storage charges automatically after free-storage expiry. Integrate with `transport_sch` for loading window scheduling.

**Impact**: physical asset tracking, storage revenue capture, demurrage accuracy.

---

## 13. Document Generation (Bill of Lading, AWB, CMR)

Add `generate_bill_of_lading(booking_id)`, `generate_air_waybill(booking_id)`, `generate_cmr(booking_id)` that render transport documents as PDF/HTML from Jinja2 templates, embed HS codes, DG declarations, incoterm terms, and shipper/consignee EORI numbers. Sign with tenant key for non-repudiation.

**Impact**: eliminates manual document prep, reduces errors, enables paperless trade.

---

## 14. Tenant-Level Rate Analytics and Benchmarking

Add `benchmark_rates(tenant_id, period)` that compares tenant freight yields against anonymised peer-group medians segmented by route band and cargo type. Flag routes where yield is >15% below median as revenue leakage candidates. Expose as a dashboard panel.

**Impact**: revenue leakage detection, carrier negotiation intelligence, strategic pricing.

---

## 15. Pydantic v2 Model Migration

`models.py` uses plain `@dataclass`. Migrate to Pydantic v2 `BaseModel` with `model_config = ConfigDict(extra='forbid', validate_by_name=True)`, typed validators (weight must be >0, HS codes 4–10 digits, UN numbers 4 digits), and `model_json_schema()` export for OpenAPI generation. Add `Annotated[float, AfterValidator(lambda v: v if v > 0 else ...)]` guards.

**Impact**: runtime type safety, auto-generated API docs, eliminates manual `_positive()` guards.
