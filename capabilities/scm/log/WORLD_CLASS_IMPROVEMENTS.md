# Logistics & Transportation (scm_log) — World-Class Improvements

15 targeted improvements to elevate this capability from functional to production-grade.

---

## 1. Real-Time Multi-Carrier Webhook Integration

**Current gap**: Carrier status updates are manually pushed via `add_tracking_event`. Real carriers (DHL, FedEx, UPS) push webhooks.

**Improvement**: Add `register_carrier_webhook` and `process_carrier_webhook_payload` methods. Normalise disparate carrier payloads (DHL TrackingNotificationRequest, FedEx Event, UPS Quantum View) into canonical `scm_log_tracking_event` records. Include HMAC signature verification per carrier. Emit normalised events onto an internal bus so downstream subscribers (WMS, ERP, customer notification) receive a single schema.

**Impact**: Eliminates polling, cuts tracking latency from hours to seconds, supports 50+ carrier integrations without bespoke code per carrier.

---

## 2. AI-Assisted Route Optimisation (VRP Solver Integration)

**Current gap**: `optimise_route` returns hard-coded scores — no actual solver runs.

**Improvement**: Integrate OR-Tools / Google Optimization Tools VRP solver via `async` subprocess call. Accept a fleet of vehicles, time windows, capacity constraints, and a multi-stop delivery manifest. Return ordered stop sequence, ETA per stop, total distance and CO2 estimate (GLEC framework). Cache solved plans in `BoundedCache` keyed on `(origin, destination, mode, objective_hash)` with a 6-hour TTL.

**Impact**: 15-25% fleet mileage reduction in typical last-mile scenarios; quantifiable CO2 reduction reportable under Scope 3 emissions.

---

## 3. Automated Freight Rate Benchmarking

**Current gap**: `create_freight_audit` accepts expected vs invoiced but has no market-rate source.

**Improvement**: Add `benchmark_freight_rate(origin, destination, weight_kg, mode, service_level)` that queries a configurable rate-card store (XLSX/PostgreSQL) and returns p10/p50/p90 market rates for the lane. Integrate with Freightos API or Xeneta as optional adapters. Store benchmark snapshots so historical drift is auditable. Wire `create_freight_audit` to auto-populate `expected_amount` from the benchmark when not supplied.

**Impact**: Removes reliance on manually maintained rate tables; surfaces overcharges automatically at invoice receipt.

---

## 4. Shipment Consolidation Engine

**Current gap**: Shipments are managed individually; no grouping logic exists.

**Improvement**: Add `suggest_consolidation(shipments: list[str])` that groups candidate shipments by origin/destination corridor, compatible freight mode, and departure window (±N hours). Return a consolidation proposal with estimated savings. Add `create_consolidated_shipment` that merges individual shipments under a master bill of lading, recalculates chargeable weight, and maintains parent-child relationships.

**Impact**: 20-40% cost reduction on LCL sea freight; reduces customs documentation overhead proportionally.

---

## 5. Carbon Footprint Tracking (Scope 3 GHG)

**Current gap**: No emissions data recorded anywhere in the model.

**Improvement**: Add `calculate_shipment_co2(shipment_id)` using GLEC Framework v3 emission factors per freight mode (air: 0.602 kgCO2e/tonne-km, sea FCL: 0.016, road: 0.096, rail: 0.028). Store `co2_kg` on each shipment at booking. Add `emissions_report(date_from, date_to)` aggregating Scope 3 Category 4 (upstream) and Category 9 (downstream) transport emissions by carrier, lane, and mode. Export to GHG Protocol-compliant CSV.

**Impact**: Enables ESG reporting; differentiates on sustainability KPIs for enterprise procurement.

---

## 6. Proof-of-Delivery (POD) Document Management

**Current gap**: `delivered` tracking event carries only text; no document attachment.

**Improvement**: Add `attach_pod(shipment_id, document_bytes, mime_type, captured_by, signature_hash)` that stores POD metadata (not raw bytes — reference to object storage URL) and links it to the shipment. Validate `signature_hash` (SHA-256 of signature image). Add `get_pod(shipment_id)` returning the POD record with a pre-signed URL. Emit `pod_attached` audit event.

**Impact**: Eliminates manual POD reconciliation; provides dispute-proof delivery confirmation for freight audits.

---

## 7. SLA Breach Detection and Alerting

**Current gap**: No monitoring of estimated vs actual delivery against contracted SLAs.

**Improvement**: Add `check_sla_breaches(tenant_id)` that scans in-transit shipments, compares `estimated_delivery` against `datetime.utcnow()`, and flags shipments at-risk (within 4h of SLA) and breached. Include carrier SLA contract data. Add `sla_performance_report(carrier_id, date_from, date_to)` computing on-time percentage, average delay hours, and breach penalty amounts per carrier contract. Emit `sla_breach_detected` events.

**Impact**: Proactive intervention before customer-facing failures; SLA penalty recovery from underperforming carriers.

---

## 8. Dangerous Goods (DG/HAZMAT) Compliance Module

**Current gap**: No hazardous materials handling — a legal liability gap for any shipper.

**Improvement**: Add `classify_dangerous_goods(un_number, proper_shipping_name, hazard_class, packing_group)` validated against IATA/IMDG regulations. Attach DG declaration to shipment via `attach_dg_declaration(shipment_id, dg_records)`. Block booking if carrier `services_offered` does not include the required DG class. Generate DG manifest document. Store UN number cross-reference table for lookup.

**Impact**: Legal compliance; avoids carrier rejection at booking; prevents regulatory fines.

---

## 9. Multi-Currency Freight Cost Normalisation

**Current gap**: Currency is stored as a string; no FX conversion — comparing audits across currencies is broken.

**Improvement**: Add `normalise_freight_costs(currency_to, tenant_id)` that fetches exchange rates (configurable provider: ECB, Open Exchange Rates) and converts all freight audit amounts to the target currency. Store `fx_rate_used` and `rate_date` on each converted record for auditability. Add currency to analytics aggregations. Cache rates with 1-hour TTL.

**Impact**: Accurate global cost reporting without manual spreadsheet FX conversion; required for multinational P&L.

---

## 10. Carrier Scorecard and Tender Management

**Current gap**: `rate_carrier` stores ratings but there is no carrier selection logic for new shipments.

**Improvement**: Add `generate_carrier_scorecard(carrier_id, period_days)` computing: on-time rate, exception rate, average cost per kg, CO2 per tonne-km, audit dispute rate. Add `run_tender(shipment_criteria, candidate_carrier_ids)` that scores each carrier against criteria weights (cost, time, sustainability, reliability) and returns a ranked recommendation. Store tender results for procurement audit trail.

**Impact**: Data-driven carrier selection replacing relationship-based decisions; typically yields 8-12% freight cost reduction over 12 months.

---

## 11. Customs Tariff Classification and Duty Estimation

**Current gap**: `create_customs_document` accepts HS codes but does not validate them or estimate duties.

**Improvement**: Add `validate_hs_code(hs_code, country)` against a local HS code table (updated from WCO database). Add `estimate_duties(shipment_id, document_id)` computing estimated import duty and VAT based on HS code, declared value, country pair, and applicable trade agreement (e.g., EAC, EU-UK TCA). Return breakdown: CIF value, tariff rate, duty amount, VAT base, VAT amount, total landed cost.

**Impact**: Eliminates customs surprises; enables accurate landed cost calculation for procurement and e-commerce pricing.

---

## 12. Shipment Insurance Integration

**Current gap**: `declared_value` is stored but no insurance workflow exists.

**Improvement**: Add `request_insurance_quote(shipment_id, coverage_type)` (all-risk, named-perils) returning premium estimate based on declared value, freight mode, route risk score, and commodity type. Add `bind_insurance_policy(shipment_id, quote_id, insured_by)` that records policy number and coverage details. Add `file_insurance_claim(shipment_id, claim_type, description, claimed_amount)` for exception-triggered claims.

**Impact**: Closes financial risk gap; enables self-serve insurance binding rather than manual broker engagement.

---

## 13. Event-Driven Architecture — Internal CloudEvents Bus

**Current gap**: `_emit` writes to an in-memory list; no external subscribers, no replay, no fan-out.

**Improvement**: Replace `_emit` with an async `_publish(event)` that serialises records as CloudEvents v1.0 JSON (with `source: scm_log`, `type: com.datacraft.scm.log.<event_type>`, `datacontenttype: application/json`). Route to configurable transports: in-process queue (asyncio.Queue), Redis Streams, or Kafka topic `scm.logistics.events`. Provide `subscribe(event_pattern, handler_coro)` for in-process consumers. Maintain in-memory replay buffer of last 1000 events per tenant.

**Impact**: Enables real-time integration with WMS, finance, customer portals without polling; foundation for event sourcing and CQRS.

---

## 14. Predictive ETA Engine

**Current gap**: `estimated_delivery` is set manually; no model-driven prediction.

**Improvement**: Add `predict_eta(shipment_id)` that applies a gradient-boosted model (scikit-learn, persisted as a joblib artifact) trained on historical `(carrier_id, route_id, weight_kg, freight_mode, service_level, origin_region, destination_region, day_of_week)` → `actual_transit_days`. Incorporate live disruption signals: weather API (Open-Meteo), port congestion index (Portcast), carrier delay history. Return `predicted_eta_iso`, `confidence_interval_hours`, and `disruption_flags`.

**Impact**: Reduces customer service contacts by 30-50%; enables proactive rerouting before misses occur.

---

## 15. Distributed Tracing and Observability

**Current gap**: Only an audit event list exists; no performance metrics, no distributed trace context.

**Improvement**: Instrument every service method with OpenTelemetry spans (`tracer.start_as_current_span(method_name)`). Propagate W3C `traceparent` headers through carrier API calls. Export metrics (shipments_created_total, tracking_events_per_carrier, freight_audit_variance_usd, route_optimisation_latency_ms) as Prometheus counters/histograms via a `/metrics` endpoint. Add structured JSON logging with `trace_id` and `span_id` fields. Provide a Grafana dashboard definition JSON.

**Impact**: Mean time to diagnose production incidents drops from hours to minutes; capacity planning becomes data-driven.

---

## Priority Order (effort vs impact)

| # | Improvement | Effort | Impact |
|---|-------------|--------|--------|
| 7 | SLA Breach Detection | Low | High |
| 5 | Carbon Footprint Tracking | Low | High |
| 6 | Proof-of-Delivery Management | Low | High |
| 13 | CloudEvents Bus | Medium | High |
| 2 | VRP Route Optimisation | Medium | High |
| 10 | Carrier Scorecard & Tender | Medium | High |
| 3 | Freight Rate Benchmarking | Medium | Medium |
| 4 | Shipment Consolidation | Medium | Medium |
| 9 | Multi-Currency Normalisation | Medium | Medium |
| 11 | Customs Tariff & Duty | Medium | Medium |
| 1 | Carrier Webhook Integration | High | High |
| 8 | DG/HAZMAT Compliance | High | High |
| 14 | Predictive ETA Engine | High | High |
| 12 | Insurance Integration | High | Medium |
| 15 | Distributed Tracing | High | Medium |
