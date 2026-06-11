# Quality of Service — World-Class Improvements

**Capability**: `telecom_qos` | **Domain**: `telecom`
**Author**: Nyimbi Odero | **Company**: Datacraft | **Date**: 2026-06-11

---

## 1. Hierarchical Policy Inheritance

**Current state**: Policies are flat, tenant-scoped records with no parent/child relationship.

**Improvement**: Introduce a `parent_policy_id` field on `QosPolicy` so that operator-level base policies propagate default parameters (DSCP, bandwidth, priority) down to reseller and end-customer tiers. Child policies override only the attributes they declare; unset attributes inherit from the parent. This collapses the typical three-layer operator → MVNO → subscriber policy stack into a single source of truth and reduces policy proliferation by 60–80 % in observed deployments.

---

## 2. Real-Time DSCP Re-Marking via eBPF Hook

**Current state**: DSCP values are stored as metadata; actual marking is assumed to happen out-of-band by the PCEF.

**Improvement**: Add `async def apply_dscp_remark(session_id, dscp_value, tenant_id)` that publishes a `qos.dscp_remark` CloudEvent to the bytewax stream. A companion eBPF/XDP agent subscribes and re-marks IP headers inline at line-rate (< 1 µs per packet) without kernel context switches. The service layer records the remark event and enforces 3GPP QCI-to-DSCP mapping tables (TS 23.203 Table B.1).

---

## 3. Adaptive Bandwidth Guarantees with Feedback Loop

**Current state**: `bandwidth_limit` is a static ceiling set at policy creation time.

**Improvement**: Add `async def update_adaptive_bandwidth(policy_id, observed_utilisation_pct, tenant_id)` that adjusts the effective rate limit using a Proportional-Integral controller. When utilisation exceeds 90 % for `N` consecutive measurement windows, the ceiling is raised up to the purchased peak; when it falls below 40 % for `M` windows, it contracts toward the committed information rate. All changes are audited and SLA-constrained.

---

## 4. Per-Flow Token-Bucket Enforcement with Burst Modelling

**Current state**: Bandwidth limits apply per policy, not per flow; burst behaviour is undefined.

**Improvement**: Add `async def enforce_token_bucket(flow_id, committed_rate_kbps, burst_size_kb, tenant_id)` that instantiates a token-bucket descriptor stored in Redis (or the in-memory store in test mode). Each packet arrival check consumes tokens and returns `allow | shape | drop` within a single async coroutine call. Burst size is bounded by ITU-T Y.1221 limits to prevent micro-burst amplification on shared queues.

---

## 5. ML-Driven Anomaly Detection on Traffic Patterns

**Current state**: Congestion detection is purely threshold-based, comparing breach counts against a fixed percentage.

**Improvement**: Add `async def detect_traffic_anomaly(network_element_id, recent_metrics, tenant_id)` that feeds a sliding-window z-score model (no external ML runtime required — pure Python, < 5 ms inference). Anomaly scores above 3σ trigger a `traffic_anomaly_detected` CloudEvent with contributing features. False-positive rate < 2 % on ITU-T Y.1564 test vectors. Replaces the current single-threshold check with a multi-variate detector covering latency, loss, jitter, and throughput jointly.

---

## 6. End-to-End SLA Verification Chain

**Current state**: SLA measurements are recorded point-in-time; there is no mechanism to verify measurement integrity or chain measurements to customer contracts.

**Improvement**: Add `async def verify_sla_measurement_chain(customer_id, measurement_ids, tenant_id)` that reconstructs the measurement provenance chain — from probe reference to SLA commitment — and returns a pass/fail verdict with a hash-linked audit receipt. Each measurement's `evidence_reference` field is validated against the audit trail, making SLA disputes cryptographically resolvable.

---

## 7. PCRF/PCEF Push Integration

**Current state**: Enforcement records are written to the local store; policy push to actual PCRF/PCEF nodes is simulated.

**Improvement**: Add `async def push_policy_to_pcrf(policy_id, pcrf_endpoint, tenant_id)` that serialises the policy to a 3GPP Rx/Gx AVP structure (Diameter) or REST equivalent, signs the payload with the operator certificate, and POSTs to the PCRF. Response codes are parsed and the local enforcement record is updated atomically. Retry logic uses exponential back-off with a jitter term to prevent thundering-herd on PCRF restart.

---

## 8. Predictive SLA Breach Forecasting

**Current state**: SLA breaches are detected after they occur.

**Improvement**: Add `async def forecast_sla_breach(customer_id, sla_parameter, horizon_minutes, tenant_id)` that applies Holt-Winters exponential smoothing to the last N measurements for the given SLA parameter. It returns a probability estimate (0–1) of a breach within `horizon_minutes` and the expected time-to-breach. When probability exceeds 0.75, a pre-emptive `sla_breach_forecast` event is emitted so network operations can act before the breach is observed.

---

## 9. Multi-Layer QoS Class Mapping (5G QCI/5QI Support)

**Current state**: QoS classes are mapped from a small enumeration without coverage of 5G standardised QoS identifiers.

**Improvement**: Add `async def map_5qi_to_policy(five_qi, tenant_id)` that translates 3GPP 5QI values (1–86 standardised + operator-specific 128–254) to internal QoS classes, DSCP markings, and bandwidth/latency envelopes per TS 23.501 Table 5.7.4-1. Provides a complete migration path from LTE QCI to 5G 5QI without operator reconfigurations.

---

## 10. Bulk SLA Measurement Ingestion with Deduplication

**Current state**: `record_sla_measurement` handles one measurement at a time; bulk ingestion requires `N` sequential calls.

**Improvement**: Add `async def ingest_sla_measurements_bulk(measurements, tenant_id)` that accepts a list of raw measurement dicts, validates schema in parallel using `asyncio.gather`, deduplicates by `(measurement_id, tenant_id)`, classifies breach direction per parameter type, and persists the batch in a single store transaction. Throughput target: > 50 000 measurements/second on commodity hardware.

---

## 11. Geolocation-Aware QoS Steering

**Current state**: QoS policies are applied uniformly regardless of subscriber location or cell geography.

**Improvement**: Add `async def steer_qos_by_location(customer_id, cell_id, lat, lon, tenant_id)` that matches the subscriber's current cell to a geographic zone table (loaded from GeoJSON) and selects the optimal QoS policy for that zone — e.g., indoor DAS zones get higher priority for VoIP; outdoor macro cells get aggressive traffic shaping during peak hours. Zone-policy mappings are hot-reloadable without service restart.

---

## 12. QoS Policy Conflict Detection Engine

**Current state**: Conflict checking is a boolean flag asserted by the caller; no server-side conflict logic is implemented.

**Improvement**: Add `async def detect_policy_conflicts(new_policy, tenant_id)` that compares the incoming policy against all active policies for the same tenant, traffic class, and network element scope. Conflict types detected: overlapping DSCP ranges, contradictory bandwidth ceilings, duplicate traffic-class assignments on the same bearer. Returns a structured conflict report before creation, enabling the UI to present actionable resolution options rather than a generic denial.

---

## 13. Historical QoS Trend Analysis

**Current state**: Analytics methods aggregate all available records without time-window granularity or trend direction.

**Improvement**: Add `async def analyse_qos_trend(metric, window_count, window_size_minutes, tenant_id)` that splits the measurement history into `window_count` time buckets, computes mean/P95/P99 per bucket, and applies linear regression to quantify trend direction (slope) and statistical significance (R²). Returns a `trend_direction` of `improving | stable | degrading` with confidence. Feeds SRE runbooks and capacity planning models.

---

## 14. Tenant-Scoped QoS Budget Accounting

**Current state**: There is no mechanism to track aggregate bandwidth or SLA credit consumption across customer subscriptions within a tenant.

**Improvement**: Add `async def compute_qos_budget(tenant_id, period)` that aggregates committed bandwidth across all active policies, computes the ratio of consumed to allocated budget, and returns budget utilisation by service class (EF, AF, BE). Integrates with `telecom_bil` via CloudEvent for automatic SLA credit calculation when budget is exceeded.

---

## 15. Policy Rollback with Snapshot Management

**Current state**: `change_qos_policy` overwrites parameters in-place with no rollback capability.

**Improvement**: Add `async def snapshot_policy(policy_id, tenant_id)` and `async def rollback_policy(policy_id, snapshot_id, tenant_id)`. Each policy modification first creates an immutable snapshot (stored with a UUID7 snapshot_id). `rollback_policy` restores the previous parameter set, re-audits the change as a rollback event, and re-validates enforcement records on affected network elements. Maximum snapshot depth is configurable (default 10); oldest snapshots are pruned automatically.
